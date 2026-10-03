"""Which addresses a URL the model hands to a web tool may reach.

This module holds the session's private-address allowlist: the machine's
`org.web_fetch_allow_private` (machine-level only, so a project value never reaches it), set once per
session by `localharness start`. Module-level because the web tools are rebuilt per agent
(tools/builtin bind_agent_store_tools) and every subagent's copy must read the same list. Empty
before any start — tests and the bench fetch public addresses only.

The rule (`check`): http and https only, no user name or password in the URL, and EVERY address the
host resolves to must be public (`is_public`) or admitted by the allowlist; multicast, unspecified and
reserved addresses never are. The verdict is about addresses, not spellings: `2130706433`,
`0x7f000001`, `127.1` and `localhost` are judged by what the resolver makes of them, so no regex ever
reads the URL. web_fetch calls `check` for every redirect hop.

Why the request is pinned: a name checked once and then handed to the HTTP client is resolved AGAIN
when the connection opens, and a DNS answer that changes in between (DNS rebinding) would reach
whatever it now names. So `check` returns the request already pointed at the addresses it checked:
the URL carries the IP, the Host header and the TLS server name carry the host. When an environment
proxy applies to the hop, the original URL goes to the proxy after the same local check instead (the
proxy resolves the name again — a named residual); see `Hop.proxy`."""
from __future__ import annotations

import asyncio
import ipaddress
import socket
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

import httpx

_ALLOW_PRIVATE: tuple[str, ...] = ()

MAX_REDIRECTS = 5
RESOLVE_TIMEOUT_S = 10.0
_NAT64 = ipaddress.ip_network("64:ff9b::/96")


def set_private_allowlist(entries: Iterable[str]) -> None:
    """Set this session's allowlist (`localharness start`, after the config loads)."""
    global _ALLOW_PRIVATE
    _ALLOW_PRIVATE = tuple(entries)


def private_allowlist() -> tuple[str, ...]:
    """The allowlist the running session set; () when none did."""
    return _ALLOW_PRIVATE


def is_public(ip) -> bool:
    """True when `ip` is a public internet address. `is_global` alone is not enough on Python 3.12.3
    (measured): it reads True for multicast (224.0.0.1, ff02::1), IPv4-mapped CGNAT and multicast
    (::ffff:100.64.0.1, ::ffff:224.0.0.1), IPv4-compatible ::127.0.0.1, NAT64 of loopback
    (64:ff9b::7f00:1) and site-local fec0::1. The forms that carry an IPv4 address are judged by that
    address, the rest are refused by hand, so the answer does not depend on the Python version."""
    if ip.version == 6:
        if ip.ipv4_mapped is not None:
            return is_public(ip.ipv4_mapped)
        if int(ip) < 2**32:                      # ::, ::1 and IPv4-compatible ::a.b.c.d
            return False
        if ip in _NAT64:
            return is_public(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
        if ip.is_site_local:                     # fec0::/10 reads is_global=True on 3.12.3
            return False
    return ip.is_global and not ip.is_multicast  # 224/4 and ff00::/8 read is_global=True


class Refused(Exception):
    """A URL web_fetch will not fetch; str() is the one line the model reads."""


def _never(ip) -> bool:
    """Refused even when allowlisted: multicast, unspecified and reserved addresses. IPv4-mapped and
    NAT64 forms are judged by the IPv4 address they carry, and ::1 is loopback (allowlistable like
    127.0.0.1): Python files all of ::/8 — ::1 and NAT64's 64:ff9b::/96 included — under is_reserved."""
    if ip.version == 6:
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip in _NAT64:
            ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        elif ip.is_loopback:
            return False
    return ip.is_multicast or ip.is_unspecified or ip.is_reserved


def _allowed(host: str, ip) -> bool:
    """Whether the machine's allowlist admits `ip` (an address or network entry) or `host` (a name)."""
    for entry in private_allowlist():
        try:
            if ip in ipaddress.ip_network(entry, strict=False):  # a v6 address in a v4 network: False
                return True
        except ValueError:
            if host.lower().rstrip(".") == entry.lower().rstrip("."):
                return True
    return False


def _literal(host: str):
    """The address `host` spells literally (an IPv6 zone `%…` dropped), else None."""
    try:
        return ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return None


async def _resolve(host: str, port: int) -> list[str]:
    """Every address the system resolver gives `host`, in its order (the tests' seam)."""
    infos = await asyncio.wait_for(
        asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM), RESOLVE_TIMEOUT_S)
    return [i[4][0] for i in infos]


@dataclass(frozen=True)
class Hop:
    """One checked request. `pinned` holds one URL per distinct checked address, in resolver order,
    never empty (orchestrator ruling R15: pin to the checked set, not the first address, so a host
    with one dead address still answers). `host_header` and `sni` carry the name the URL had.
    `proxy` is the environment proxy that applies to this hop's URL, or None (R22)."""
    url: httpx.URL
    pinned: tuple[httpx.URL, ...]
    host_header: str
    sni: str
    proxy: Optional[str]


def _env_proxy(url: httpx.URL) -> Optional[str]:
    """The environment proxy that applies to this URL, or None (orchestrator ruling R22). Decided
    here, once per hop, with urllib's rules — `getproxies()` (the *_proxy variables; the system
    settings on macOS and Windows) and `proxy_bypass()` (NO_PROXY) — and handed to httpx as an
    explicit transport, so httpx's own NO_PROXY matching never runs and the two can never disagree."""
    proxies = urllib.request.getproxies()
    proxy = proxies.get(url.scheme) or proxies.get("all")
    if not proxy or urllib.request.proxy_bypass(url.host):
        return None
    return proxy if "://" in proxy else f"http://{proxy}"  # httpx's own reading of a bare host:port


async def check(raw: "str | httpx.URL") -> Hop:
    """Decide whether web_fetch may fetch `raw` and return the request pinned to what was checked.
    Raises Refused (one line for the model) otherwise; nothing is requested before this returns."""
    try:
        url = httpx.URL(raw)
    except httpx.InvalidURL:
        raise Refused(f"refused: {str(raw)!r} is not a valid URL") from None
    if url.scheme not in ("http", "https"):
        raise Refused("refused: only http and https URLs are fetched")
    if url.userinfo:
        raise Refused("refused: a URL with a user name or password in it is not fetched")
    # The ASCII form is the name that goes on the wire (url.host decodes xn-- labels); resolving it
    # checks exactly the name the connection, the Host header and the proxy would use.
    host = url.raw_host.decode("ascii")
    if not host:
        raise Refused("refused: the URL names no host")
    literal = _literal(host)
    if literal is not None:
        addrs = [host]  # an address spelled literally is judged as written; no lookup
    else:
        try:
            addrs = await _resolve(host, url.port or (443 if url.scheme == "https" else 80))
        except (OSError, asyncio.TimeoutError):
            addrs = []
    if not addrs:
        raise Refused(f"refused: could not resolve {host}")
    checked = []
    for addr in addrs:
        ip = ipaddress.ip_address(addr.split("%", 1)[0])
        who = f"{host} is" if ip == literal else f"{host} resolves to {ip}, which is"
        if _never(ip):
            raise Refused(f"refused: {who} a multicast, unspecified or reserved address")
        if not is_public(ip) and not _allowed(host, ip):
            raise Refused(f"refused: {who} not a public address — add it to org.web_fetch_allow_private "
                          "in your machine config to fetch it")
        checked.append(ip)
    return Hop(url=url, pinned=tuple(url.copy_with(host=str(ip)) for ip in dict.fromkeys(checked)),
               host_header=url.netloc.decode("ascii"), sni=host, proxy=_env_proxy(url))
