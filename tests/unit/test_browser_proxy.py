"""The browser's proxy, driven with raw streams and httpx: no browser, no network.

Names resolve through a patched netguard._resolve; the proxy reaches a "public" address only through
its `_open_connection` seam, which these tests point at a local echo listener."""
from __future__ import annotations

import asyncio
import base64

import httpx
import pytest

from localharness.tools.builtin import browser_proxy, netguard
from tests.unit.test_web_fetch_guard import _PROXY_VARS, public_web

PUBLIC = "93.184.216.34"


@pytest.fixture(autouse=True)
def _clean_slate(monkeypatch):
    monkeypatch.setattr(netguard, "_ALLOW_PRIVATE", ())
    for name in _PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)


@pytest.fixture
async def echo():
    async def handler(reader, writer):
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
        writer.close()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    yield server.sockets[0].getsockname()[1]
    server.close()
    await server.wait_closed()


@pytest.fixture
def dialled(monkeypatch, echo) -> list[tuple[str, int]]:
    """Every (host, port) the proxy tried to open; each attempt connects to the echo listener."""
    seen: list[tuple[str, int]] = []

    async def open_connection(host, port, **kwargs):
        seen.append((host, port))
        return await asyncio.open_connection("127.0.0.1", echo)

    monkeypatch.setattr(browser_proxy, "_open_connection", open_connection)
    return seen


@pytest.fixture
async def proxy(monkeypatch, dialled):
    public_web(monkeypatch, PUBLIC)
    p = await browser_proxy.serve(None)
    yield p
    await p.close()


@pytest.fixture
async def connect(proxy):
    """Open a raw connection to the proxy; every one is closed at the end of the test."""
    writers = []

    async def open_(request: str = ""):
        reader, writer = await asyncio.open_connection("127.0.0.1", proxy.port)
        writers.append(writer)
        if request:
            writer.write(request.encode())
            await writer.drain()
        return reader, writer

    yield open_
    for w in writers:
        w.close()


def auth(p, password: str | None = None) -> str:
    token = base64.b64encode(f"{p.username}:{password or p.password}".encode()).decode()
    return f"Proxy-Authorization: Basic {token}\r\n"


def connect_request(p, authority: str) -> str:
    return f"CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n{auth(p)}\r\n"


async def head(reader) -> str:
    return (await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)).decode()


async def test_a_connect_to_a_public_name_tunnels_to_the_address_that_was_checked(proxy, connect, dialled):
    reader, writer = await connect(connect_request(proxy, "example.com:443"))
    assert (await head(reader)).startswith("HTTP/1.1 200")
    writer.write(b"hello")
    assert await asyncio.wait_for(reader.readexactly(5), 5) == b"hello"
    assert dialled == [(PUBLIC, 443)]


async def test_a_connect_to_loopback_is_refused_and_recorded_and_nothing_is_dialled(proxy, dialled):
    client = httpx.AsyncClient(proxy=f"http://{proxy.username}:{proxy.password}@127.0.0.1:{proxy.port}")
    async with client:
        with pytest.raises(httpx.ProxyError):
            await client.get("https://127.0.0.1/")
    assert list(proxy.refused) == ["127.0.0.1:443 not public"]
    assert dialled == []
