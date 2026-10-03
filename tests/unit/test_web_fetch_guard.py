"""A URL the model hands to web_fetch reaches only public addresses (SEC-04).

netguard.check() decides each hop: http and https only, no user name or password, every address the
host resolves to must be public (or allowlisted on purpose in the machine config), and the request is
pinned to the addresses that were checked; web_fetch follows redirects by hand, checking each hop.
No test here touches the network: every host name goes through a patched netguard._resolve; only
numeric spellings and `localhost` use the real resolver, which answers them from this machine
(inet_aton and /etc/hosts); fetches go to an httpx.MockTransport or a stub proxy on 127.0.0.1."""
from __future__ import annotations

import asyncio
import ipaddress
import socket

import httpx
import pytest

from localharness.agent.context import ContentStore
from localharness.config.models import ToolConfig
from localharness.tools.builtin import bind_agent_store_tools, netguard, register_builtin_tools, web_tool
from localharness.tools.builtin.web_tool import WebFetchTool
from localharness.tools.registry import ToolRegistry

PRIVATE = "not a public address — add it to org.web_fetch_allow_private in your machine config to fetch it"
_PROXY_VARS = ("http_proxy", "https_proxy", "all_proxy", "no_proxy")


@pytest.fixture(autouse=True)
def _clean_slate(monkeypatch):
    """No allowlist, and no proxy setting from the developer's own shell can change a result."""
    monkeypatch.setattr(netguard, "_ALLOW_PRIVATE", ())
    for name in _PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)


def _dns(monkeypatch, table: dict[str, list[str]]) -> list[str]:
    """Patch the resolver: answer from `table`, socket.gaierror for any other name. Returns the list
    of names asked, in order, so a test can count lookups."""
    asked: list[str] = []

    async def _resolve(host: str, port: int) -> list[str]:
        asked.append(host)
        if host not in table:
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
        return list(table[host])

    monkeypatch.setattr(netguard, "_resolve", _resolve)
    return asked


def public_web(monkeypatch, address: str = "93.184.216.34") -> list[str]:
    """For tests elsewhere that drive web_fetch through a fake client: every name resolves to one
    public address, and no proxy variable from the developer's shell applies. Returns the names asked."""
    for name in _PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    asked: list[str] = []

    async def _resolve(host: str, port: int) -> list[str]:
        asked.append(host)
        return [address]

    monkeypatch.setattr(netguard, "_resolve", _resolve)
    return asked


def _serve(monkeypatch, handler) -> list[httpx.Request]:
    """Route web_fetch through an httpx.MockTransport (the web tool's _TRANSPORT seam). Returns the
    requests it received, in order, each recorded before the handler answers or raises."""
    seen: list[httpx.Request] = []

    def _record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    monkeypatch.setattr(web_tool, "_TRANSPORT", httpx.MockTransport(_record))
    return seen


def _text(body: str, status: int = 200) -> httpx.Response:
    return httpx.Response(status, text=body, headers={"content-type": "text/plain"})


def _to(location: str) -> httpx.Response:
    return httpx.Response(302, headers={"location": location})


async def _fetch(url: str):
    return await WebFetchTool(ContentStore()).run(url=url)


async def _refusal(raw: str) -> str:
    with pytest.raises(netguard.Refused) as exc:
        await netguard.check(raw)
    text = str(exc.value)
    assert "\n" not in text, "a refusal is one line"
    return text


# --- the public-address rule, version-independent (Python 3.12.3's is_global gaps closed) ---

@pytest.mark.parametrize("addr", [
    "127.0.0.1", "10.0.0.1", "192.168.1.1", "172.16.0.1", "169.254.169.254", "100.64.0.1",
    "100.127.255.255", "0.0.0.0", "224.0.0.1", "::1", "::", "::ffff:127.0.0.1", "::ffff:100.64.0.1",
    "::ffff:224.0.0.1", "::127.0.0.1", "64:ff9b::7f00:1", "fe80::1", "fc00::1", "fec0::1", "ff02::1",
])
def test_not_public(addr):
    assert netguard.is_public(ipaddress.ip_address(addr)) is False


@pytest.mark.parametrize("addr", ["93.184.216.34", "1.1.1.1", "2606:4700::1111", "64:ff9b::5db8:d822"])
def test_public(addr):
    assert netguard.is_public(ipaddress.ip_address(addr)) is True


# --- refused before anything is resolved ---

async def test_a_user_name_or_password_is_refused_before_any_lookup(monkeypatch):
    asked = _dns(monkeypatch, {"example.com": ["93.184.216.34"]})
    for raw in ("http://x@93.184.216.34/", "http://x@127.0.0.1/", "https://:secret@example.com/"):
        assert await _refusal(raw) == "refused: a URL with a user name or password in it is not fetched"
    assert asked == []


async def test_only_http_and_https_are_fetched(monkeypatch):
    asked = _dns(monkeypatch, {"example.com": ["93.184.216.34"]})
    for raw in ("ftp://example.com/", "file:///etc/passwd", "gopher://example.com/", "example.com/page"):
        assert await _refusal(raw) == "refused: only http and https URLs are fetched"
    assert asked == []


async def test_a_url_without_a_host_is_refused(monkeypatch):
    asked = _dns(monkeypatch, {})
    assert await _refusal("http:///nohost") == "refused: the URL names no host"
    assert asked == []


async def test_dotted_octal_never_becomes_a_request():
    """httpx 0.28.1 rejects 0177.0.0.1 as an invalid URL; the system resolver would map it to
    127.0.0.1. Either way it is refused — never fetched."""
    text = await _refusal("http://0177.0.0.1/")
    assert text == "refused: 'http://0177.0.0.1/' is not a valid URL" or PRIVATE in text


# --- every spelling of a private target (the real local resolver, no network) ---

@pytest.mark.parametrize("raw", [
    "http://127.0.0.1/admin", "http://[::ffff:7f00:1]/", "http://2130706433/", "http://0x7f000001/",
    "http://127.1/", "http://localhost/", "http://localhost:8000/x", "http://169.254.169.254/latest/",
    "http://100.64.0.1/", "http://[::1]:8000/", "http://10.1.2.3/x", "http://192.168.0.5/x",
    "http://172.16.0.1/x", "http://[fe80::1%25eth0]/",
])
async def test_every_private_spelling_is_refused_naming_the_setting(raw):
    assert PRIVATE in await _refusal(raw)


async def test_a_name_that_resolves_privately_is_refused_naming_the_setting(monkeypatch):
    _dns(monkeypatch, {"intranet.example": ["10.0.0.5"]})
    assert await _refusal("https://intranet.example/x") == (
        "refused: intranet.example resolves to 10.0.0.5, which is not a public address — add it to "
        "org.web_fetch_allow_private in your machine config to fetch it")


async def test_one_private_address_among_public_ones_is_refused(monkeypatch):
    _dns(monkeypatch, {"intranet.example": ["93.184.216.34", "10.0.0.5"]})
    assert "intranet.example resolves to 10.0.0.5, which is " + PRIVATE in await _refusal(
        "https://intranet.example/x")


async def test_a_name_that_does_not_resolve_is_refused(monkeypatch):
    _dns(monkeypatch, {})
    assert await _refusal("https://intranet.example/x") == "refused: could not resolve intranet.example"


async def test_a_literal_address_is_judged_without_asking_the_resolver(monkeypatch):
    asked = _dns(monkeypatch, {})
    assert await _refusal("http://127.0.0.1/admin") == "refused: 127.0.0.1 is " + PRIVATE
    hop = await netguard.check("http://93.184.216.34/x")
    assert [str(u) for u in hop.pinned] == ["http://93.184.216.34/x"]
    assert asked == []


# --- one checked hop, pinned to the checked addresses ---

async def test_the_request_is_pinned_to_the_checked_address(monkeypatch):
    _dns(monkeypatch, {"example.com": ["93.184.216.34"]})
    hop = await netguard.check("https://example.com:8443/p?q=1")
    assert hop.pinned == (httpx.URL("https://93.184.216.34:8443/p?q=1"),)
    assert hop.host_header == "example.com:8443"
    assert hop.sni == "example.com"
    assert hop.url == httpx.URL("https://example.com:8443/p?q=1")
    assert hop.proxy is None


async def test_an_ipv6_answer_is_pinned_in_brackets(monkeypatch):
    _dns(monkeypatch, {"example.com": ["2606:4700::1111"]})
    hop = await netguard.check("https://example.com:8443/p?q=1")
    assert [str(u) for u in hop.pinned] == ["https://[2606:4700::1111]:8443/p?q=1"]
    assert hop.host_header == "example.com:8443"


async def test_every_checked_address_is_kept_in_resolver_order_without_repeats(monkeypatch):
    _dns(monkeypatch, {"example.com": ["93.184.216.34", "93.184.216.35", "93.184.216.34"]})
    hop = await netguard.check("https://example.com/")
    assert [u.host for u in hop.pinned] == ["93.184.216.34", "93.184.216.35"]


async def test_the_name_checked_is_the_name_on_the_wire(monkeypatch):
    """httpx sends faß.de as xn--fa-hia.de (IDNA 2008); url.host reads 'faß.de', which the system
    resolver would encode as fass.de (IDNA 2003) — another domain. The guard resolves the ASCII name
    the connection, the Host header and a proxy would use."""
    asked = _dns(monkeypatch, {"xn--fa-hia.de": ["93.184.216.34"], "fass.de": ["10.0.0.5"]})
    hop = await netguard.check("https://faß.de/")
    assert asked == ["xn--fa-hia.de"]
    assert [u.host for u in hop.pinned] == ["93.184.216.34"]
    assert (hop.host_header, hop.sni) == ("xn--fa-hia.de", "xn--fa-hia.de")


async def test_a_nat64_address_is_judged_by_the_ipv4_address_it_carries(monkeypatch):
    """On an IPv6-only network with DNS64, every IPv4-only site resolves into 64:ff9b::/96."""
    _dns(monkeypatch, {"v4only.example": ["64:ff9b::5db8:d822"], "rebound.example": ["64:ff9b::7f00:1"]})
    hop = await netguard.check("https://v4only.example/")
    assert [u.host for u in hop.pinned] == ["64:ff9b::5db8:d822"]
    assert PRIVATE in await _refusal("https://rebound.example/")


# --- the machine-level allowlist ---

async def test_the_allowlist_admits_a_private_network_on_purpose(monkeypatch):
    _dns(monkeypatch, {"intranet.example": ["10.0.0.5"]})
    netguard.set_private_allowlist(["10.0.0.0/8"])
    hop = await netguard.check("https://intranet.example/x")
    assert [str(u) for u in hop.pinned] == ["https://10.0.0.5/x"]
    assert hop.host_header == "intranet.example"


async def test_the_allowlist_admits_a_host_by_name(monkeypatch):
    _dns(monkeypatch, {"intranet.example": ["10.0.0.5"]})
    netguard.set_private_allowlist(["Intranet.Example."])
    hop = await netguard.check("https://intranet.example/x")
    assert [u.host for u in hop.pinned] == ["10.0.0.5"]


async def test_an_allowlisted_network_admits_only_itself(monkeypatch):
    _dns(monkeypatch, {"intranet.example": ["10.0.0.5"]})
    netguard.set_private_allowlist(["192.168.0.0/16", "other.example"])
    assert PRIVATE in await _refusal("https://intranet.example/x")


async def test_ipv6_loopback_can_be_allowlisted_like_ipv4_loopback(monkeypatch):
    """::1 sits in ::/8, which Python files under is_reserved; it is loopback, and a user who
    allowlists a local service must reach it on both loopbacks."""
    _dns(monkeypatch, {"nas.example": ["::1", "127.0.0.1"]})
    netguard.set_private_allowlist(["nas.example"])
    hop = await netguard.check("http://nas.example:8080/")
    assert [u.host for u in hop.pinned] == ["::1", "127.0.0.1"]


async def test_multicast_unspecified_and_reserved_are_refused_even_when_allowlisted():
    netguard.set_private_allowlist(["224.0.0.0/4", "0.0.0.0/0", "::/0"])
    assert await _refusal("http://224.0.0.1/") == (
        "refused: 224.0.0.1 is a multicast, unspecified or reserved address")
    for raw in ("http://0.0.0.0/", "http://240.0.0.1/", "http://255.255.255.255/", "http://[ff02::1]/",
                "http://[::]/", "http://[::ffff:224.0.0.1]/"):
        assert (await _refusal(raw)).endswith(" a multicast, unspecified or reserved address")


# --- the proxy decision, per hop (urllib's rules, decided once, handed to httpx) ---

async def test_without_a_proxy_variable_the_hop_has_no_proxy(monkeypatch):
    _dns(monkeypatch, {"example.com": ["93.184.216.34"]})
    for raw in ("https://example.com/", "http://example.com/"):
        assert (await netguard.check(raw)).proxy is None


async def test_https_proxy_applies_to_https_after_the_check(monkeypatch):
    asked = _dns(monkeypatch, {"example.com": ["93.184.216.34"]})
    direct = await netguard.check("https://example.com/")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.test:3128")
    hop = await netguard.check("https://example.com/")
    assert hop.proxy == "http://proxy.test:3128"
    assert hop.pinned == direct.pinned
    assert asked == ["example.com", "example.com"], "the check ran with the proxy set"
    assert (await netguard.check("http://example.com/")).proxy is None


async def test_all_proxy_without_a_scheme_applies_to_both_schemes(monkeypatch):
    _dns(monkeypatch, {"example.com": ["93.184.216.34"]})
    monkeypatch.setenv("ALL_PROXY", "proxy.test:3128")
    for raw in ("https://example.com/", "http://example.com/"):
        assert (await netguard.check(raw)).proxy == "http://proxy.test:3128"


async def test_no_proxy_bypasses_the_proxy(monkeypatch):
    _dns(monkeypatch, {"example.com": ["93.184.216.34"]})
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.test:3128")
    monkeypatch.setenv("ALL_PROXY", "proxy.test:3128")
    monkeypatch.setenv("NO_PROXY", "example.com")
    for raw in ("https://example.com/", "http://example.com/"):
        assert (await netguard.check(raw)).proxy is None


async def test_a_proxy_never_skips_the_local_check(monkeypatch):
    _dns(monkeypatch, {"intranet.example": ["10.0.0.5"]})
    without = await _refusal("https://intranet.example/x")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.test:3128")
    assert await _refusal("https://intranet.example/x") == without


# --- web_fetch: every hop checked, pinned, redirects followed by hand ---

async def test_a_public_fetch_is_unchanged_and_pinned(monkeypatch):
    _dns(monkeypatch, {"example.com": ["93.184.216.34"]})
    seen = _serve(monkeypatch, lambda r: _text("hello"))
    result = await _fetch("https://example.com/page")
    assert result.success, result.error
    assert result.output.startswith(web_tool._UNTRUSTED) and "hello" in result.output
    assert result.metadata["url"] == "https://example.com/page"
    [request] = seen
    assert request.url.host == "93.184.216.34"
    assert request.headers["host"] == "example.com"
    assert request.extensions["sni_hostname"] == "example.com"
    assert request.headers["user-agent"] == web_tool._UA
    assert request.extensions["timeout"] == {"connect": 20.0, "read": 20.0, "write": 20.0, "pool": 20.0}


async def test_a_redirect_to_loopback_is_refused_after_one_request(monkeypatch):
    _dns(monkeypatch, {"example.com": ["93.184.216.34"]})
    seen = _serve(monkeypatch, lambda r: _to("http://127.0.0.1/admin"))
    result = await _fetch("https://example.com/page")
    assert (result.success, result.error_type) == (False, "validation_error")
    assert result.error == "refused: 127.0.0.1 is " + PRIVATE
    assert len(seen) == 1


async def test_a_redirect_to_a_name_that_resolves_privately_is_refused(monkeypatch):
    _dns(monkeypatch, {"example.com": ["93.184.216.34"], "intranet.example": ["10.0.0.5"]})
    seen = _serve(monkeypatch, lambda r: _to("https://intranet.example/secret"))
    result = await _fetch("https://example.com/page")
    assert result.error == "refused: intranet.example resolves to 10.0.0.5, which is " + PRIVATE
    assert result.error_type == "validation_error"
    assert len(seen) == 1


async def test_a_relative_redirect_is_followed_on_the_same_host(monkeypatch):
    asked = _dns(monkeypatch, {"example.com": ["93.184.216.34"]})
    seen = _serve(monkeypatch, lambda r: _to("/next") if r.url.path == "/page" else _text("next page"))
    result = await _fetch("https://example.com/page")
    assert result.success, result.error
    assert "next page" in result.output
    assert result.metadata["url"] == "https://example.com/next"
    assert [str(r.url) for r in seen] == ["https://93.184.216.34/page", "https://93.184.216.34/next"]
    assert [r.headers["host"] for r in seen] == ["example.com", "example.com"]
    assert asked == ["example.com", "example.com"], "each hop is resolved and checked"


async def test_more_than_five_redirects_are_refused(monkeypatch):
    asked = _dns(monkeypatch, {f"h{n}.example": ["93.184.216.34"] for n in range(8)})
    seen = _serve(monkeypatch, lambda r: _to(f"https://h{int(r.headers['host'][1]) + 1}.example/"))
    result = await _fetch("https://h0.example/")
    assert (result.error, result.error_type) == ("refused: more than 5 redirects", "validation_error")
    assert len(seen) == 6
    assert asked == [f"h{n}.example" for n in range(6)]


async def test_a_changed_dns_answer_cannot_redirect_the_request(monkeypatch):
    """DNS rebinding: the first answer is public, every later one loopback. The request goes to the
    address that was checked, and the name is resolved once for the hop."""
    calls: list[str] = []

    async def _resolve(host: str, port: int) -> list[str]:
        calls.append(host)
        return ["93.184.216.34"] if len(calls) == 1 else ["127.0.0.1"]

    monkeypatch.setattr(netguard, "_resolve", _resolve)
    seen = _serve(monkeypatch, lambda r: _text("hello"))
    result = await _fetch("https://example.com/page")
    assert result.success, result.error
    assert [r.url.host for r in seen] == ["93.184.216.34"]
    assert calls == ["example.com"]


async def test_a_name_that_rebinds_between_hops_is_refused_at_the_next_hop(monkeypatch):
    answers = iter([["93.184.216.34"], ["127.0.0.1"]])

    async def _resolve(host: str, port: int) -> list[str]:
        return next(answers)

    monkeypatch.setattr(netguard, "_resolve", _resolve)
    seen = _serve(monkeypatch, lambda r: _to("/again"))
    result = await _fetch("https://example.com/page")
    assert result.error == "refused: example.com resolves to 127.0.0.1, which is " + PRIVATE
    assert len(seen) == 1


async def test_a_dead_address_moves_to_the_next_checked_one(monkeypatch):
    asked = _dns(monkeypatch, {"example.com": ["93.184.216.34", "93.184.216.35"]})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "93.184.216.34":
            raise httpx.ConnectError("connection refused", request=request)
        return _text("hello")

    seen = _serve(monkeypatch, handler)
    result = await _fetch("https://example.com/page")
    assert result.success, result.error
    assert "hello" in result.output
    assert [r.url.host for r in seen] == ["93.184.216.34", "93.184.216.35"]
    assert all(r.headers["host"] == "example.com" for r in seen)
    assert all(r.extensions["sni_hostname"] == "example.com" for r in seen)
    assert asked == ["example.com"], "the next CHECKED address, never a fresh lookup"


@pytest.mark.parametrize("dead", [httpx.ConnectError, httpx.ConnectTimeout])
async def test_when_every_address_is_dead_the_fetch_fails(monkeypatch, dead):
    _dns(monkeypatch, {"example.com": ["93.184.216.34", "93.184.216.35"]})

    def handler(request: httpx.Request) -> httpx.Response:
        raise dead("unreachable", request=request)

    seen = _serve(monkeypatch, handler)
    result = await _fetch("https://example.com/page")
    assert (result.success, result.error_type) == (False, "execution_error")
    assert result.error.startswith("fetch failed:")
    assert len(seen) == 2


async def test_an_http_error_is_not_retried_on_the_next_address(monkeypatch):
    _dns(monkeypatch, {"example.com": ["93.184.216.34", "93.184.216.35"]})
    seen = _serve(monkeypatch, lambda r: _text("boom", status=500))
    result = await _fetch("https://example.com/page")
    assert result.error.startswith("fetch failed:")
    assert len(seen) == 1, "only a connection that never opened moves on"


async def test_a_connection_error_is_fetch_failed_as_today(monkeypatch):
    _dns(monkeypatch, {"example.com": ["93.184.216.34"]})

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 111] Connection refused", request=request)

    _serve(monkeypatch, handler)
    result = await _fetch("https://example.com/page")
    assert result.error == "fetch failed: [Errno 111] Connection refused"


@pytest.mark.parametrize("raw", [
    "http://127.0.0.1/x", "http://[::ffff:7f00:1]/", "http://2130706433/", "http://x@127.0.0.1/",
    "http://169.254.169.254/latest/meta-data/", "http://100.64.0.1/", "ftp://example.com/",
    "http://0177.0.0.1/", "http://localhost:8000/x",
])
async def test_every_refusal_reaches_the_model_as_one_line_and_nothing_is_requested(monkeypatch, raw):
    seen = _serve(monkeypatch, lambda r: _text("must not be reached"))
    result = await _fetch(raw)
    assert (result.success, result.error_type) == (False, "validation_error")
    assert result.error.startswith("refused: ") and "\n" not in result.error
    assert seen == []


async def test_fetch_then_query_through_the_registry_the_loop_uses(monkeypatch):
    """The real path: register_builtin_tools, the agent's own ContentStore bound by
    bind_agent_store_tools, both calls through ToolRegistry.dispatch."""
    _dns(monkeypatch, {"example.com": ["93.184.216.34"]})
    page = "filler text. " * 600 + "The 2025 revenue was 42 million." + " tail." * 10
    seen = _serve(monkeypatch, lambda r: _text(page))
    registry = ToolRegistry()
    await register_builtin_tools(registry)
    store = ContentStore()
    bind_agent_store_tools(registry, store)
    config = ToolConfig(inherit=["global"])
    fetched = await registry.dispatch("web_fetch", {"url": "https://example.com/report"}, "root", "", config)
    assert fetched.success, fetched.error
    assert seen[0].url.host == "93.184.216.34"
    fetch_id = fetched.metadata["fetch_id"]
    assert store.get(fetch_id) == page
    for pattern in ("2025 revenue", r"revenue was \d+ million"):
        hit = await registry.dispatch("web_page_query", {"fetch_id": fetch_id, "pattern": pattern},
                                      "root", "", config)
        assert hit.success, hit.error
        assert "The 2025 revenue was 42 million." in hit.output


# --- behind an environment proxy: the original URL, after the local check (R22) ---

def _sni_of(record: bytes) -> str | None:
    """The server_name a TLS ClientHello record carries (RFC 8446 4.1.2, RFC 6066 3), or None."""
    i = 5 + 4 + 2 + 32                                   # record header, handshake header, version, random
    i += 1 + record[i]                                   # session id
    i += 2 + int.from_bytes(record[i:i + 2], "big")      # cipher suites
    i += 1 + record[i]                                   # compression methods
    end = i + 2 + int.from_bytes(record[i:i + 2], "big")
    i += 2
    while i + 4 <= end:
        kind, size = int.from_bytes(record[i:i + 2], "big"), int.from_bytes(record[i + 2:i + 4], "big")
        if kind == 0:                                    # list length, name type, name length, name
            n = int.from_bytes(record[i + 7:i + 9], "big")
            return record[i + 9:i + 9 + n].decode("ascii")
        i += 4 + size
    return None


async def test_behind_a_connect_proxy_the_tls_name_is_the_host(monkeypatch):
    """The real httpx/httpcore proxy path (no MockTransport): a stub proxy on 127.0.0.1 records what
    it is asked. httpcore's CONNECT tunnel starts TLS with the URL's host and ignores sni_hostname,
    so a pinned IP there would send `CONNECT 93.184.216.34:443` and no server name at all."""
    connections: list[int] = []
    seen: list = []

    async def stub(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        connections.append(1)
        try:
            line = (await reader.readuntil(b"\r\n\r\n")).split(b"\r\n", 1)[0].decode()
            seen.append(line)
            if line.startswith("CONNECT "):
                writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
                await writer.drain()
                record = await reader.readexactly(5)
                record += await reader.readexactly(int.from_bytes(record[3:5], "big"))
                seen.append(("server_name", _sni_of(record)))
            else:
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 5\r\n"
                             b"Connection: close\r\n\r\nhello")
                await writer.drain()
        finally:
            writer.close()

    server = await asyncio.start_server(stub, "127.0.0.1", 0)
    proxy = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    monkeypatch.setenv("HTTPS_PROXY", proxy)
    monkeypatch.setenv("HTTP_PROXY", proxy)
    asked = _dns(monkeypatch, {"example.com": ["93.184.216.34"], "intranet.example": ["10.0.0.5"]})
    try:
        tls = await _fetch("https://example.com/page")
        assert not tls.success and tls.error.startswith("fetch failed:"), "the stub hangs up after the hello"
        assert seen == ["CONNECT example.com:443 HTTP/1.1", ("server_name", "example.com")]
        seen.clear()

        plain = await _fetch("http://example.com/page")
        assert plain.success, plain.error
        assert "hello" in plain.output
        assert seen == ["GET http://example.com/page HTTP/1.1"]
        seen.clear()

        before = len(connections)
        refused = await _fetch("https://intranet.example/x")
        assert refused.error_type == "validation_error" and PRIVATE in refused.error
        assert len(connections) == before and seen == [], "the proxy was never contacted"
        assert asked == ["example.com", "example.com", "intranet.example"]
    finally:
        server.close()
        await server.wait_closed()


async def test_a_host_no_proxy_bypasses_goes_direct_to_the_checked_address(monkeypatch):
    """NO_PROXY is matched against the NAME, once, by netguard; httpx never applies the environment's
    proxies itself. Left to httpx, the pinned request (whose URL carries the IP) would no longer match
    NO_PROXY=example.com and would go to the proxy. Loopback is reached here on purpose, through the
    allowlist, so the direct path is a real connection that never leaves this machine."""
    proxied: list[bytes] = []

    async def proxy_stub(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        proxied.append((await reader.readuntil(b"\r\n\r\n")).split(b"\r\n", 1)[0])
        writer.close()

    async def origin(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        host = [h for h in head.split(b"\r\n") if h.lower().startswith(b"host:")][0]
        body = b"direct hello, " + host
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n"
                     b"Connection: close\r\n\r\n%s" % (len(body), body))
        await writer.drain()
        writer.close()

    proxy = await asyncio.start_server(proxy_stub, "127.0.0.1", 0)
    site = await asyncio.start_server(origin, "127.0.0.1", 0)
    port = site.sockets[0].getsockname()[1]
    monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy.sockets[0].getsockname()[1]}")
    monkeypatch.setenv("NO_PROXY", "example.com")
    _dns(monkeypatch, {"example.com": ["127.0.0.1"]})
    netguard.set_private_allowlist(["127.0.0.1"])
    try:
        result = await _fetch(f"http://example.com:{port}/page")
        assert result.success, result.error
        assert f"direct hello, Host: example.com:{port}" in result.output
        assert proxied == []
    finally:
        for server in (proxy, site):
            server.close()
            await server.wait_closed()


async def test_a_lookup_that_times_out_or_answers_nothing_is_refused(monkeypatch):
    async def slow(host: str, port: int) -> list[str]:
        raise asyncio.TimeoutError

    monkeypatch.setattr(netguard, "_resolve", slow)
    assert await _refusal("https://example.com/") == "refused: could not resolve example.com"
    _dns(monkeypatch, {"example.com": []})
    assert await _refusal("https://example.com/") == "refused: could not resolve example.com"
