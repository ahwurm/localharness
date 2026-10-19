"""The browser's checking proxy: a loopback CONNECT-only listener that judges every CONNECT by netguard's rule with the private allowlist off."""
from __future__ import annotations

import asyncio
import base64
import binascii
import re
import secrets
from collections import deque
from dataclasses import dataclass, field
from functools import partial

import httpx

from localharness.tools.builtin import netguard

NAV_TIMEOUT_S = 30
TUNNEL_MAX_BYTES = 32_000_000
TUNNEL_MAX_S = NAV_TIMEOUT_S + 20
_CHUNK = 65536
_AUTHORITY = re.compile(r"(\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9.-]+):([0-9]{1,5})")
_STREAM_ERRORS = (asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, OSError)
_open_connection = asyncio.open_connection  # the tests' transport seam


@dataclass
class Proxy:
    port: int
    username: str
    password: str = field(repr=False)
    refused: deque[str] = field(default_factory=lambda: deque(maxlen=64))
    refusals: int = 0
    upstream: str | None = field(default=None, repr=False)
    _server: asyncio.Server | None = field(default=None, repr=False)
    _writers: set[asyncio.StreamWriter] = field(default_factory=set, repr=False)
    _tasks: set[asyncio.Task] = field(default_factory=set, repr=False)

    async def close(self) -> None:
        self._server.close()
        for t in list(self._tasks):
            t.cancel()
        for w in list(self._writers):
            w.close()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._server.wait_closed()


async def serve(upstream: str | None) -> Proxy:
    proxy = Proxy(0, secrets.token_urlsafe(16), secrets.token_urlsafe(16), upstream=upstream)
    proxy._server = await asyncio.start_server(partial(_handle, proxy), "127.0.0.1", 0)
    proxy.port = proxy._server.sockets[0].getsockname()[1]
    return proxy


async def _reply(writer: asyncio.StreamWriter, status: str, *extra: str, keep: bool = False) -> bool:
    lines = [f"HTTP/1.1 {status}", *extra, "Content-Length: 0", *([] if keep else ["Connection: close"])]
    writer.write(("\r\n".join(lines) + "\r\n\r\n").encode())
    await writer.drain()
    return keep


def _authorised(proxy: Proxy, values: list[str]) -> bool:
    if len(values) != 1:
        return False
    scheme, _, token = values[0].partition(" ")
    try:
        got = base64.b64decode(token.strip(), validate=True)
    except binascii.Error:
        return False
    return scheme.lower() == "basic" and secrets.compare_digest(got, f"{proxy.username}:{proxy.password}".encode())


async def _dial(hops, port: int):
    for hop in hops:
        try:
            return await asyncio.wait_for(_open_connection(hop.host, port), NAV_TIMEOUT_S)
        except OSError:
            continue
    return None


async def _via(upstream: str, authority: str):
    """A tunnel to `authority` through another proxy, or None when it does not answer 200."""
    url = httpx.URL(upstream)
    tls = url.scheme == "https"
    reader, writer = await asyncio.wait_for(
        _open_connection(url.host, url.port or (443 if tls else 80), **({"ssl": True} if tls else {})), NAV_TIMEOUT_S)
    lines = [f"CONNECT {authority} HTTP/1.1", f"Host: {authority}"]
    if url.userinfo:
        lines.append("Proxy-Authorization: Basic " + base64.b64encode(f"{url.username}:{url.password}".encode()).decode())
    writer.write(("\r\n".join(lines) + "\r\n\r\n").encode())
    answer = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), NAV_TIMEOUT_S)
    if answer.split(b" ", 2)[1:2] != [b"200"]:
        writer.close()
        return None
    return reader, writer


async def _pump(src: asyncio.StreamReader, dst: asyncio.StreamWriter, limit: int) -> None:
    total = 0
    while data := await src.read(_CHUNK):
        total += len(data)
        if total > limit:
            return
        dst.write(data)
        await dst.drain()


async def _first_to_end(*coros) -> None:
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _tunnel(reader, writer, up_reader, up_writer) -> None:
    try:
        await asyncio.wait_for(
            _first_to_end(_pump(reader, up_writer, TUNNEL_MAX_BYTES), _pump(up_reader, writer, TUNNEL_MAX_BYTES)),
            TUNNEL_MAX_S)
    except asyncio.TimeoutError:
        pass


async def _one(proxy: Proxy, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> bool:
    """Answer one request on this connection; True when the connection stays open for the next."""
    head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), NAV_TIMEOUT_S)
    request, *lines = head.decode("latin-1").split("\r\n")
    parts = request.split(" ")
    if parts[0] != "CONNECT":
        return await _reply(writer, "405 Method Not Allowed")
    headers = [(k.strip().lower(), v.strip()) for k, _, v in (line.partition(":") for line in lines if line)]
    if not _authorised(proxy, [v for k, v in headers if k == "proxy-authorization"]):
        return await _reply(writer, "407 Proxy Authentication Required", 'Proxy-Authenticate: Basic realm="localharness"')
    match = _AUTHORITY.fullmatch(parts[1]) if len(parts) == 3 else None
    if match is None or not 0 < int(match[2]) < 65536:
        return await _reply(writer, "400 Bad Request")
    authority, port = parts[1], int(match[2])
    try:
        hop = await netguard.check(f"https://{authority}/", allow_private=False)
    except netguard.Refused:
        proxy.refused.append(f"{authority} not public")
        proxy.refusals += 1
        return await _reply(writer, "403 Forbidden", keep=True)
    chain = proxy.upstream or hop.proxy
    try:
        tunnel = await (_via(chain, authority) if chain else _dial(hop.pinned, port))
    except _STREAM_ERRORS:
        tunnel = None
    if tunnel is None:
        return await _reply(writer, "502 Bad Gateway")
    up_reader, up_writer = tunnel
    proxy._writers.add(up_writer)
    try:
        writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await writer.drain()
        await _tunnel(reader, writer, up_reader, up_writer)
    finally:
        proxy._writers.discard(up_writer)
        up_writer.close()
    return False


async def _handle(proxy: Proxy, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    proxy._writers.add(writer)
    proxy._tasks.add(task := asyncio.current_task())
    try:
        while await _one(proxy, reader, writer):
            pass
    except _STREAM_ERRORS:
        pass
    finally:
        proxy._tasks.discard(task)
        proxy._writers.discard(writer)
        writer.close()
