"""The browser's proxy, driven with raw streams and httpx: no browser, no network.

Names resolve through a patched netguard._resolve; the proxy reaches a "public" address only through
its `_open_connection` seam, which these tests point at a local echo listener."""
from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import httpx
import pytest

from localharness.tools.builtin import browser_proxy, netguard
from tests.unit.test_web_fetch_guard import _PROXY_VARS, _dns, public_web

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
    """Every (host, port) the proxy tried to open; an attempt connects to the echo listener, except one
    to loopback, which is a test's own fake upstream and is dialled for real."""
    seen: list[tuple[str, int]] = []

    async def open_connection(host, port, **kwargs):
        seen.append((host, port))
        return await asyncio.open_connection(*((host, port) if host == "127.0.0.1" else ("127.0.0.1", echo)))

    monkeypatch.setattr(browser_proxy, "_open_connection", open_connection)
    return seen


@pytest.fixture
async def proxy(monkeypatch, dialled):
    public_web(monkeypatch, PUBLIC)
    p = await browser_proxy.serve(None)
    yield p
    await asyncio.wait_for(p.close(), 5)


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


async def refusal(proxy, connect, authority: str) -> str:
    reader, _ = await connect(connect_request(proxy, authority))
    return await head(reader)


@pytest.mark.parametrize("authority", [
    "127.0.0.1:443", "10.0.0.1:443", "192.168.1.1:443", "172.16.0.1:443", "169.254.169.254:443",
    "100.64.0.1:443", "[::1]:443", "localhost:443"])
async def test_a_connect_to_a_non_public_authority_is_refused_recorded_and_not_dialled(
        authority, proxy, connect, dialled, monkeypatch):
    _dns(monkeypatch, {"localhost": ["127.0.0.1"]})
    assert (await refusal(proxy, connect, authority)).startswith("HTTP/1.1 403")
    assert proxy.refused[-1] == f"{authority} not public"
    assert dialled == []


async def test_a_bracketed_ipv6_loopback_is_refused_without_asking_the_resolver(proxy, connect, monkeypatch):
    asked = _dns(monkeypatch, {})
    assert (await refusal(proxy, connect, "[::1]:443")).startswith("HTTP/1.1 403")
    assert asked == []


async def test_an_allowlisted_private_address_is_admitted_by_netguard_and_still_refused_here(
        proxy, connect, dialled, monkeypatch):
    netguard.set_private_allowlist(["10.0.0.0/8"])
    assert (await netguard.check("https://10.0.0.5/")).pinned
    assert (await refusal(proxy, connect, "10.0.0.5:443")).startswith("HTTP/1.1 403")
    assert list(proxy.refused) == ["10.0.0.5:443 not public"]
    assert dialled == []


async def test_a_refusal_keeps_the_connection_and_the_next_connect_is_checked_again(proxy, connect):
    reader, writer = await connect(connect_request(proxy, "10.0.0.1:443"))
    assert (await head(reader)).startswith("HTTP/1.1 403")
    writer.write(connect_request(proxy, "127.0.0.1:443").encode())
    assert (await head(reader)).startswith("HTTP/1.1 403")
    assert list(proxy.refused) == ["10.0.0.1:443 not public", "127.0.0.1:443 not public"]


@pytest.mark.parametrize("request_line", ["GET http://example.com/ HTTP/1.1", "GET / HTTP/1.1"])
async def test_a_request_that_is_not_a_connect_gets_405_and_the_connection_closes_unrecorded(
        request_line, proxy, connect):
    reader, _ = await connect(f"{request_line}\r\nHost: example.com\r\n{auth(proxy)}\r\n")
    assert (await head(reader)).startswith("HTTP/1.1 405")
    assert await asyncio.wait_for(reader.read(), 5) == b""
    assert not proxy.refused


@pytest.mark.parametrize("credential", [
    "", "Proxy-Authorization: Basic d3Jvbmc6d3Jvbmc=\r\n", "Proxy-Authorization: Bearer abc\r\n",
    "Proxy-Authorization: Basic !!!\r\n"])
async def test_missing_or_wrong_credentials_get_407_with_the_realm_and_close_unrecorded(
        credential, proxy, connect):
    reader, _ = await connect(f"CONNECT example.com:443 HTTP/1.1\r\n{credential}\r\n")
    answer = await head(reader)
    assert answer.startswith("HTTP/1.1 407")
    assert 'Proxy-Authenticate: Basic realm="localharness"' in answer
    assert await asyncio.wait_for(reader.read(), 5) == b""
    assert not proxy.refused


async def test_the_right_user_with_a_wrong_password_gets_407(proxy, connect):
    reader, _ = await connect(f"CONNECT example.com:443 HTTP/1.1\r\n{auth(proxy, 'nope')}\r\n")
    assert (await head(reader)).startswith("HTTP/1.1 407")


@pytest.mark.parametrize("target", [
    "example.com", "user@example.com:443", "example.com:0", "example.com:70000", "exa mple.com:443"])
async def test_an_authority_that_is_not_host_and_port_gets_400_and_close(target, proxy, connect):
    reader, _ = await connect(f"CONNECT {target} HTTP/1.1\r\n{auth(proxy)}\r\n")
    assert (await head(reader)).startswith("HTTP/1.1 400")
    assert await asyncio.wait_for(reader.read(), 5) == b""
    assert not proxy.refused


async def test_refused_keeps_the_newest_64_of_70(proxy, connect):
    reader, writer = await connect()
    for i in range(1, 71):
        writer.write(connect_request(proxy, f"10.0.0.{i}:443").encode())
        assert (await head(reader)).startswith("HTTP/1.1 403")
    assert len(proxy.refused) == 64
    assert proxy.refused[-1] == "10.0.0.70:443 not public"


async def test_a_tunnel_forwards_at_most_the_byte_cap_in_a_direction(proxy, connect, monkeypatch):
    monkeypatch.setattr(browser_proxy, "TUNNEL_MAX_BYTES", 1024)
    reader, writer = await connect(connect_request(proxy, "example.com:443"))
    assert (await head(reader)).startswith("HTTP/1.1 200")
    writer.write(b"x" * 4096)
    assert len(await asyncio.wait_for(reader.read(), 5)) <= 1024


async def test_a_tunnel_closes_when_its_time_is_up(proxy, connect, monkeypatch):
    monkeypatch.setattr(browser_proxy, "TUNNEL_MAX_S", 0.2)
    reader, _ = await connect(connect_request(proxy, "example.com:443"))
    assert (await head(reader)).startswith("HTTP/1.1 200")
    assert await asyncio.wait_for(reader.read(), 2) == b""


async def test_the_checked_addresses_are_tried_in_order_and_the_name_is_resolved_once(
        proxy, connect, echo, monkeypatch):
    second = "93.184.216.35"
    asked = _dns(monkeypatch, {"example.com": [PUBLIC, second]})
    tried = []

    async def open_connection(host, port, **kwargs):
        tried.append(host)
        if host == PUBLIC:
            raise ConnectionRefusedError
        return await asyncio.open_connection("127.0.0.1", echo)

    monkeypatch.setattr(browser_proxy, "_open_connection", open_connection)
    reader, writer = await connect(connect_request(proxy, "example.com:443"))
    assert (await head(reader)).startswith("HTTP/1.1 200")
    writer.write(b"hi")
    assert await asyncio.wait_for(reader.readexactly(2), 5) == b"hi"
    assert (tried, asked) == ([PUBLIC, second], ["example.com"])


async def test_no_reachable_address_gets_502(proxy, connect, monkeypatch):
    async def open_connection(host, port, **kwargs):
        raise ConnectionRefusedError

    monkeypatch.setattr(browser_proxy, "_open_connection", open_connection)
    assert (await refusal(proxy, connect, "example.com:443")).startswith("HTTP/1.1 502")


async def test_only_bytes_travel_after_a_connect_succeeds(proxy, connect):
    reader, writer = await connect(
        f"CONNECT example.com:443 HTTP/1.1\r\nX-Probe: secret\r\n{auth(proxy)}\r\n")
    assert (await head(reader)).startswith("HTTP/1.1 200")
    writer.write(b"hello")
    assert await asyncio.wait_for(reader.readexactly(5), 5) == b"hello"


async def test_two_connects_at_once_are_each_checked(proxy, connect, dialled):
    first, second = await asyncio.gather(
        refusal(proxy, connect, "example.com:443"), refusal(proxy, connect, "10.0.0.1:443"))
    assert (first.split()[1], second.split()[1]) == ("200", "403")
    assert dialled == [(PUBLIC, 443)]
    assert list(proxy.refused) == ["10.0.0.1:443 not public"]


async def test_each_serve_has_its_own_random_credentials_and_close_stops_the_listener(monkeypatch):
    a, b = await browser_proxy.serve(None), await browser_proxy.serve(None)
    assert len({a.username, a.password, b.username, b.password}) == 4
    assert a.port != b.port
    await a.close()
    await b.close()
    with pytest.raises(OSError):
        await asyncio.open_connection("127.0.0.1", a.port)


async def test_close_closes_every_open_tunnel(monkeypatch, dialled):
    public_web(monkeypatch, PUBLIC)
    p = await browser_proxy.serve(None)
    reader, writer = await asyncio.open_connection("127.0.0.1", p.port)
    writer.write(connect_request(p, "example.com:443").encode())
    assert (await head(reader)).startswith("HTTP/1.1 200")
    await p.close()
    assert await asyncio.wait_for(reader.read(), 5) == b""
    writer.close()


async def test_close_does_not_wait_for_a_handler_still_dialling(monkeypatch):
    public_web(monkeypatch, PUBLIC)
    dialling, cancelled = asyncio.Event(), []

    async def hang(host, port, **kwargs):
        dialling.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    monkeypatch.setattr(browser_proxy, "_open_connection", hang)
    p = await browser_proxy.serve(None)
    _, writer = await asyncio.open_connection("127.0.0.1", p.port)
    writer.write(connect_request(p, "example.com:443").encode())
    await asyncio.wait_for(dialling.wait(), 5)
    await asyncio.wait_for(p.close(), 5)
    assert cancelled == [True]
    writer.close()


@pytest.fixture
async def start_upstream():
    """A fake upstream proxy on loopback: records each request head, answers `status`, then echoes."""
    servers, heads, writers = [], [], []

    async def start(status: int = 200):
        async def handler(reader, writer):
            writers.append(writer)
            heads.append((await reader.readuntil(b"\r\n\r\n")).decode())
            writer.write(f"HTTP/1.1 {status} Whatever\r\n\r\n".encode())
            await writer.drain()
            while status == 200 and (data := await reader.read(65536)):
                writer.write(data)
                await writer.drain()
            writer.close()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        servers.append(server)
        return SimpleNamespace(port=server.sockets[0].getsockname()[1], heads=heads)

    yield start
    for w in writers:
        w.close()
    for s in servers:
        s.close()
        await s.wait_closed()


async def test_behind_an_environment_proxy_a_public_authority_is_connected_through_it_by_name(
        proxy, connect, dialled, start_upstream, monkeypatch):
    up = await start_upstream()
    monkeypatch.setenv("https_proxy", f"http://user:pw@127.0.0.1:{up.port}")
    reader, writer = await connect(
        f"CONNECT example.com:443 HTTP/1.1\r\nX-Probe: secret\r\n{auth(proxy)}\r\n")
    assert (await head(reader)).startswith("HTTP/1.1 200")
    writer.write(b"hi")
    assert await asyncio.wait_for(reader.readexactly(2), 5) == b"hi"
    token = base64.b64encode(b"user:pw").decode()
    assert up.heads[0] == (f"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n"
                           f"Proxy-Authorization: Basic {token}\r\n\r\n")
    assert dialled == [("127.0.0.1", up.port)]


async def test_behind_an_environment_proxy_a_private_authority_is_refused_without_contacting_it(
        proxy, connect, dialled, start_upstream, monkeypatch):
    up = await start_upstream()
    monkeypatch.setenv("https_proxy", f"http://127.0.0.1:{up.port}")
    assert (await refusal(proxy, connect, "10.0.0.1:443")).startswith("HTTP/1.1 403")
    assert up.heads == [] and dialled == []


async def test_an_upstream_named_to_serve_carries_every_tunnel_with_no_environment(
        start_upstream, dialled, monkeypatch):
    public_web(monkeypatch, PUBLIC)
    up = await start_upstream()
    p = await browser_proxy.serve(f"http://127.0.0.1:{up.port}")
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", p.port)
        writer.write(connect_request(p, "example.com:443").encode())
        assert (await head(reader)).startswith("HTTP/1.1 200")
        writer.write(b"hi")
        assert await asyncio.wait_for(reader.readexactly(2), 5) == b"hi"
        assert up.heads[0].startswith("CONNECT example.com:443 HTTP/1.1\r\n")
        assert "Proxy-Authorization" not in up.heads[0]
        writer.close()
    finally:
        await p.close()


async def test_an_upstream_that_does_not_answer_200_gets_the_client_502(
        proxy, connect, start_upstream, monkeypatch):
    up = await start_upstream(status=403)
    monkeypatch.setenv("https_proxy", f"http://127.0.0.1:{up.port}")
    assert (await refusal(proxy, connect, "example.com:443")).startswith("HTTP/1.1 502")
