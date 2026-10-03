"""Every stored credential is a SecretStr (SEC-05): the provider's key, a peer endpoint's key and
headers, the active endpoint's key, and an MCP server's environment and headers. Shown, each is the
mask; sent, each is the raw value (unwrapped once, where it leaves the process); written, each is the
raw value (the machine's 0600 file holds what the user typed)."""
from __future__ import annotations

import asyncio
import stat

import httpx
import pytest
import yaml
from pydantic import SecretStr

from localharness.cli import model_ops
from localharness.config.loader import ConfigLoader
from localharness.config.models import (
    ActiveSelection,
    EndpointRef,
    HarnessConfig,
    MCPServerConfig,
    ProviderConfig,
)
from localharness.provider import client as client_mod
from localharness.provider.client import LLMClient, LLMConfig
from localharness.tools import mcp as mcp_mod
from localharness.tools.mcp import MCPServerClient


def _provider(**kw) -> ProviderConfig:
    return ProviderConfig(provider_type="vllm", base_url="http://127.0.0.1:8000/v1", default_model="m", **kw)


def _hidden(model, value: str) -> None:
    assert value not in repr(model) and value not in str(model)


# --- the types ----------------------------------------------------------------------------------

def test_the_provider_and_endpoint_keys_are_secrets() -> None:
    provider = _provider(api_key="sk-x")
    peer = EndpointRef(name="p", base_url="http://h/v1", api_key="sk-e", extra_headers={"X-K": "h1"})
    active = ActiveSelection(base_url="http://h/v1", model="m", api_key="sk-a")
    for holder, secret, raw in ((provider, provider.api_key, "sk-x"), (peer, peer.api_key, "sk-e"),
                                (peer, peer.extra_headers["X-K"], "h1"), (active, active.api_key, "sk-a")):
        assert isinstance(secret, SecretStr) and secret.get_secret_value() == raw
        _hidden(holder, raw)
    assert _provider().api_key.get_secret_value() == "none"
    assert EndpointRef(name="p", base_url="http://h/v1").api_key.get_secret_value() == "none"
    assert EndpointRef(name="p", base_url="http://h/v1").extra_headers == {}
    assert ActiveSelection(base_url="http://h/v1", model="m").api_key.get_secret_value() == "none"


def test_an_mcp_servers_env_and_headers_are_secrets() -> None:
    server = MCPServerConfig(name="s", transport="stdio", command="srv",
                             env={"TOKEN": "env-t"}, headers={"Authorization": "hdr-a"})
    for secret, raw in ((server.env["TOKEN"], "env-t"), (server.headers["Authorization"], "hdr-a")):
        assert isinstance(secret, SecretStr) and secret.get_secret_value() == raw
        _hidden(server, raw)
    bare = MCPServerConfig(name="s", transport="stdio", command="srv")
    assert (bare.env, bare.headers) == ({}, {})


# --- the wire -----------------------------------------------------------------------------------

def test_the_session_sends_the_raw_key(monkeypatch) -> None:
    """The model server receives the key and headers as typed — wrapped or plain — and so does the
    endpoint a /model switch re-points the client at."""
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"object": "list", "data": []})

    monkeypatch.setattr(client_mod, "_TRANSPORT", httpx.MockTransport(answer))

    async def drive() -> None:
        for key, headers in ((SecretStr("sk-x"), {"X-Key": SecretStr("h1")}), ("sk-x", {"X-Key": "h1"})):
            llm = LLMClient(LLMConfig(base_url="http://h.test/v1", model="m", api_key=key, extra_headers=headers))
            try:
                await llm._client.models.list()
                llm.rebind_endpoint("http://h2.test/v1", api_key=SecretStr("k2"), extra_headers={"X": SecretStr("1")})
                await llm._client.models.list()
            finally:
                await llm.aclose()

    asyncio.run(drive())
    assert len(seen) == 4
    for first, rebound in ((seen[0], seen[1]), (seen[2], seen[3])):
        assert (first.url.host, first.headers["authorization"], first.headers["x-key"]) == (
            "h.test", "Bearer sk-x", "h1")
        assert (rebound.url.host, rebound.headers["authorization"], rebound.headers["x"]) == (
            "h2.test", "Bearer k2", "1")


class _Stop(Exception):
    """Raised where the transport would start: the parameters it was handed are what is checked."""


class _Refuses:
    async def __aenter__(self):
        raise _Stop

    async def __aexit__(self, *exc) -> bool:
        return False


def test_an_mcp_server_receives_its_env_and_headers_raw(monkeypatch) -> None:
    got: dict = {}

    def stdio(params):
        got["params"] = params
        return _Refuses()

    def http(url, headers=None, **_):
        got["headers"] = headers
        return _Refuses()

    monkeypatch.setattr(mcp_mod, "stdio_client", stdio)
    monkeypatch.setattr(mcp_mod, "streamablehttp_client", http)
    local = MCPServerConfig(name="s", transport="stdio", command="srv", env={"TOKEN": "t"})
    remote = MCPServerConfig(name="r", transport="streamable_http", url="http://mcp.test/mcp",
                             headers={"Authorization": "a"})
    assert isinstance(local.env["TOKEN"], SecretStr) and isinstance(remote.headers["Authorization"], SecretStr)
    for config in (local, remote):
        with pytest.raises(_Stop):
            asyncio.run(MCPServerClient(config).connect())
    assert got["params"].env["TOKEN"] == "t" and type(got["params"].env["TOKEN"]) is str
    assert got["headers"] == {"Authorization": "a"} and type(got["headers"]["Authorization"]) is str


# --- the writers --------------------------------------------------------------------------------

def test_init_writes_the_keys_as_typed(tmp_path) -> None:
    """config.yaml holds the raw key (never the mask), loads back with it, and is owner-only."""
    from localharness.cli.init_cmd import _write_harness

    harness = HarnessConfig(
        provider=_provider(api_key="sk-INIT"),
        extra_endpoints=[EndpointRef(name="peer", base_url="http://peer.test/v1", api_key="sk-PEER-INIT",
                                     extra_headers={"X-K": "hdr-INIT"})])
    path = _write_harness(tmp_path, harness, memory_on=True)
    text = path.read_text(encoding="utf-8")
    assert "api_key: sk-INIT" in text and "*****" not in text
    assert "api_key: sk-PEER-INIT" in text and "X-K: hdr-INIT" in text
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    loaded = ConfigLoader(config_dir=tmp_path).load_harness()
    assert loaded.provider.api_key.get_secret_value() == "sk-INIT"
    assert loaded.extra_endpoints[0].api_key.get_secret_value() == "sk-PEER-INIT"
    assert loaded.extra_endpoints[0].extra_headers["X-K"].get_secret_value() == "hdr-INIT"


def test_switching_to_a_peer_keeps_its_key_out_of_overrides(tmp_path) -> None:
    """The key stays with the endpoint it belongs to: the active-endpoint record carries none, and a
    record an earlier version wrote with a key loses it on the next switch."""
    overrides = tmp_path / "overrides.yaml"
    overrides.write_text(yaml.safe_dump({"active_endpoint": {
        "name": "old", "base_url": "http://old.test/v1", "model": "o", "api_key": "sk-OLD"}}), encoding="utf-8")
    harness = HarnessConfig(provider=_provider(), extra_endpoints=[
        EndpointRef(name="peer", base_url="http://peer.test/v1", provider_type="vllm", api_key="sk-PEER")])

    assert asyncio.run(model_ops.persist_active_endpoint(
        harness, harness.extra_endpoints[0], "peer-model", config_dir=tmp_path)) is None

    text = overrides.read_text(encoding="utf-8")
    assert yaml.safe_load(text)["active_endpoint"] == {
        "name": "peer", "base_url": "http://peer.test/v1", "provider_type": "vllm", "model": "peer-model"}
    assert "sk-PEER" not in text and "sk-OLD" not in text
    assert isinstance(harness.active_endpoint.api_key, SecretStr)
    assert harness.active_endpoint.api_key.get_secret_value() == "sk-PEER"


def test_reveal_unwraps_every_secret_under_a_node() -> None:
    from localharness.config.redact import reveal

    node = {"a": SecretStr("x"), "b": [SecretStr("y"), 1], "c": {"d": SecretStr("z")}}
    assert reveal(node) == {"a": "x", "b": ["y", 1], "c": {"d": "z"}}
