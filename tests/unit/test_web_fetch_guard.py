"""A URL the model hands to web_fetch reaches only public addresses (SEC-04).

netguard.check() decides each hop: http and https only, no user name or password, every address the
host resolves to must be public (or allowlisted on purpose in the machine config), and the request is
pinned to the addresses that were checked. No test here touches the network: every host name goes
through a patched netguard._resolve; only numeric spellings and `localhost` use the real resolver,
which answers them from this machine (inet_aton and /etc/hosts)."""
from __future__ import annotations

import ipaddress
import socket

import httpx
import pytest

from localharness.tools.builtin import netguard

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
