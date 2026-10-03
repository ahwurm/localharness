"""A credential set for a plain-http address that is not this machine travels unencrypted: one
warning line, never a refusal (the owner's frictionless rule — a proposer or a model server may be
local or in the cloud, over http or https). Read at a plugin's setup check and in `localharness
start`'s summary. Standard library and pydantic only."""
from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlsplit

from pydantic import SecretStr


def sends_key(value: Any) -> bool:
    """Is a key set? ("none" and empty mean no key.)"""
    raw = value.get_secret_value() if isinstance(value, SecretStr) else value
    return bool(raw) and raw != "none"


def _this_machine(host: str) -> bool:
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def cleartext_warning(key_path: str, url: str, sent: bool, *, verb: str = "travels") -> str | None:
    """One line when `sent` and `url` is http:// to a host that is not this machine, else None."""
    parts = urlsplit(url or "")
    host = parts.hostname or ""
    if not sent or parts.scheme != "http" or not host or _this_machine(host):
        return None
    where = f"[{host}]" if ":" in host else host
    if parts.port:
        where += f":{parts.port}"
    return f"{key_path} {verb} unencrypted to http://{where} — use https"


def cleartext_warnings(harness: Any) -> list[str]:
    """The provider's key and each peer endpoint's key and headers, in that order."""
    out = [cleartext_warning("provider.api_key", harness.provider.base_url,
                             sends_key(harness.provider.api_key))]
    for i, ep in enumerate(harness.extra_endpoints):
        out.append(cleartext_warning(f"extra_endpoints[{i}].api_key", ep.base_url, sends_key(ep.api_key)))
        out.append(cleartext_warning(f"extra_endpoints[{i}].extra_headers", ep.base_url,
                                     bool(ep.extra_headers), verb="travel"))
    return [w for w in out if w]
