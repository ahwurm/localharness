"""A credential set for a plain-http address that is not this machine travels unencrypted (#34): one
warning line naming the setting and the address, never a refusal — the proposer and the model server
may be local or in the cloud, over http or https, and the request still goes."""
from __future__ import annotations

import pytest
from pydantic import SecretStr

from localharness.config.models import EndpointRef, HarnessConfig, ProviderConfig


def _harness(provider_url: str, key: str, endpoints: list[EndpointRef] = ()) -> HarnessConfig:
    return HarnessConfig(
        provider=ProviderConfig(provider_type="vllm", base_url=provider_url, api_key=key, default_model="m"),
        extra_endpoints=list(endpoints))


@pytest.mark.parametrize(("url", "where"), [
    ("http://p.example:8000/v1", "http://p.example:8000"),
    ("http://p.example/v1", "http://p.example"),
    ("http://10.0.0.5:8000/v1", "http://10.0.0.5:8000"),
    ("http://[2001:db8::1]:8000/v1", "http://[2001:db8::1]:8000"),
])
def test_a_key_over_plain_http_to_another_machine_is_one_line(url, where) -> None:
    from localharness.config.cleartext import cleartext_warning

    assert cleartext_warning("proposer.api_key", url, True) == (
        f"proposer.api_key travels unencrypted to {where} — use https")


@pytest.mark.parametrize("url", [
    "https://p.example/v1", "http://127.0.0.1:8000/v1", "http://127.8.9.10/v1", "http://localhost:9/v1",
    "http://[::1]:8000/v1", "http://x.localhost/v1", "", "not a url",
])
def test_https_this_machine_or_no_address_gets_no_line(url) -> None:
    from localharness.config.cleartext import cleartext_warning

    assert cleartext_warning("proposer.api_key", url, True) is None


def test_nothing_sent_gets_no_line() -> None:
    from localharness.config.cleartext import cleartext_warning

    assert cleartext_warning("proposer.api_key", "http://p.example/v1", False) is None


@pytest.mark.parametrize(("value", "sent"), [
    (SecretStr("sk-x"), True), ("sk-x", True), (SecretStr("none"), False), ("none", False),
    (SecretStr(""), False), ("", False), (None, False)])
def test_a_key_is_sent_unless_it_is_none_or_empty(value, sent) -> None:
    from localharness.config.cleartext import sends_key

    assert sends_key(value) is sent


def test_the_provider_then_each_peer_endpoint_key_and_headers() -> None:
    from localharness.config.cleartext import cleartext_warnings

    peer = EndpointRef(name="peer", base_url="http://peer.lan/v1", api_key="sk-p", extra_headers={"X-K": "h"})
    assert cleartext_warnings(_harness("http://10.0.0.5:8000/v1", "sk-x", [peer])) == [
        "provider.api_key travels unencrypted to http://10.0.0.5:8000 — use https",
        "extra_endpoints[0].api_key travels unencrypted to http://peer.lan — use https",
        "extra_endpoints[0].extra_headers travel unencrypted to http://peer.lan — use https",
    ]


def test_no_key_an_https_address_or_this_machine_gives_nothing() -> None:
    from localharness.config.cleartext import cleartext_warnings

    keyless_peer = EndpointRef(name="peer", base_url="http://peer.lan/v1")
    local_peer = EndpointRef(name="local", base_url="http://127.0.0.1:11434/v1", api_key="sk-l",
                             extra_headers={"X-K": "h"})
    assert cleartext_warnings(_harness("http://10.0.0.5:8000/v1", "none", [keyless_peer, local_peer])) == []
    assert cleartext_warnings(_harness("https://api.example/v1", "sk-x")) == []


def test_the_line_never_carries_the_key() -> None:
    from localharness.config.cleartext import cleartext_warnings

    peer = EndpointRef(name="peer", base_url="http://peer.lan/v1", api_key="sk-PEER-SENTINEL",
                       extra_headers={"X-K": "hdr-SENTINEL"})
    lines = cleartext_warnings(_harness("http://10.0.0.5/v1", "sk-PROVIDER-SENTINEL", [peer]))
    assert len(lines) == 3 and "SENTINEL" not in " ".join(lines)
