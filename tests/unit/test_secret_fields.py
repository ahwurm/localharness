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


# --- the masking edges (redact) -----------------------------------------------------------------

def test_is_secret_reads_containers_and_leaves_a_model_to_its_own_fields() -> None:
    from typing import Optional

    from localharness.config.redact import is_secret

    assert is_secret(Optional[list[SecretStr]]) and is_secret(dict[str, SecretStr])
    assert not is_secret(list[EndpointRef]) and not is_secret(Optional[ActiveSelection])


def test_an_alias_choice_of_a_secret_field_is_a_secret_path() -> None:
    from pydantic import AliasChoices, AliasPath, BaseModel, Field

    from localharness.config.redact import secret_paths, secret_values

    class Keyed(BaseModel):
        key: SecretStr = Field(SecretStr(""), validation_alias=AliasChoices("apiKey", "api_key",
                                                                           AliasPath("auth", "key")))

    paths = secret_paths(Keyed)
    assert {("apiKey",), ("api_key",), ("auth",)} <= set(paths)
    assert "sk-AC" in secret_values(Keyed, {"apiKey": "sk-AC"})


@pytest.mark.parametrize("text", [
    "a: *sk-PLAIN-123abc",                                  # an undefined alias
    "a: !sk-PLAIN-123abc x",                                # an unknown tag
    "a: &sk-PLAIN-123abc 1\nb: &sk-PLAIN-123abc 2\n",       # an anchor defined twice
], ids=["alias", "tag", "anchor"])
def test_a_yaml_alias_or_tag_named_after_a_secret_is_never_quoted(text) -> None:
    from localharness.config.redact import yaml_problem

    with pytest.raises(yaml.YAMLError) as exc:
        yaml.safe_load(text)
    problem = yaml_problem(exc.value)
    assert "'<name>'" in problem and "sk-PLAIN" not in problem, problem


def test_only_real_secret_leaves_inside_a_list_of_models_are_collected() -> None:
    """R14: a peer endpoint's key and header values are secrets; its name, address and type are not,
    so an error that names them still reads as written."""
    from localharness.config.redact import at_secret, secret_paths, secret_values

    paths = secret_paths(HarnessConfig)
    assert ("extra_endpoints", "*", "api_key") in paths and ("extra_endpoints", "*", "extra_headers") in paths
    assert ("extra_endpoints",) not in paths
    data = {"extra_endpoints": [{"name": "peer-1", "base_url": "http://peer.lan/v1", "provider_type": "vllm",
                                 "api_key": "sk-E", "extra_headers": {"X-K": "h-E"}}]}
    found = secret_values(HarnessConfig, data)
    assert {"sk-E", "h-E"} <= found
    assert not found & {"peer-1", "http://peer.lan/v1", "vllm", "X-K"}
    assert at_secret(HarnessConfig, ("extra_endpoints", 0, "api_key"))
    assert at_secret(HarnessConfig, ("extra_endpoints", 0, "extra_headers", "X-K"))
    assert not at_secret(HarnessConfig, ("extra_endpoints", 0, "name"))


def test_an_ordinary_validation_message_is_never_masked(tmp_path) -> None:
    """R14: an agent's MCP servers hold secrets (env, headers) beside ordinary words (names, commands,
    the transport). pydantic's message prints as written; the secrets never print — not in a
    message, and not in the value an error on the whole server shows."""
    from pydantic import ValidationError

    from localharness.config.loader import ConfigValidationError
    from localharness.config.models import AgentConfig
    from localharness.config.redact import secret_paths, secret_values, validation_text

    assert ("tools", "mcp_servers", "*", "env") in secret_paths(AgentConfig)
    data = {"name": "helper", "role": "helps", "tools": {"mcp_servers": [
        {"name": "a", "transport": "bogus", "command": "x", "env": {"K": "sekret-1"}},
        {"name": "b", "transport": "stdio", "command": "y"},
        {"name": "c", "transport": "stdio", "env": {"K": "sekret-2"}, "headers": {"H": "sekret-3"}}]}}
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "helper.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigValidationError) as loaded:
        ConfigLoader(config_dir=tmp_path).load_agent("helper")
    with pytest.raises(ValidationError) as raw:
        AgentConfig.model_validate(data)
    for text in (str(loaded.value), validation_text(raw.value, secret_values(AgentConfig, data))):
        assert "Input should be 'stdio' or 'streamable_http'" in text, text
        assert "'command' is required for stdio transport" in text, text
        for secret in ("sekret-1", "sekret-2", "sekret-3"):
            assert secret not in text, text
    assert "got: 'bogus'" in str(loaded.value)  # an ordinary value is shown as it was written


# --- JSON output, and a key typed without echo --------------------------------------------------

def test_json_output_serializes_models_and_masks_their_secrets() -> None:
    import json

    from localharness.cli.components_cmd import _serialize_value

    ep = EndpointRef(name="p", base_url="http://h/v1", provider_type="vllm", api_key="sk-x",
                     extra_headers={"X": "h-x"})
    got = _serialize_value([ep])
    assert got[0]["api_key"] == "**********" and got[0]["extra_headers"] == {"X": "**********"}
    assert (got[0]["name"], got[0]["base_url"], got[0]["provider_type"]) == ("p", "http://h/v1", "vllm")
    assert "sk-x" not in json.dumps(got) and "h-x" not in json.dumps(got)
    assert _serialize_value({"a": {"b": SecretStr("s")}, "c": (SecretStr("t"), 1)}) == {
        "a": {"b": "**********"}, "c": ["**********", 1]}
    for value in (None, True, 3, 2.5, "x"):
        assert _serialize_value(value) == value


def _machine(tmp_path):
    g = tmp_path / "g"
    g.mkdir()
    (g / "config.yaml").write_text(yaml.safe_dump({
        "version": "1", "org": {"audit_log_path": str(g / "audit.jsonl")},
        "provider": {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1", "default_model": "m"}}),
        encoding="utf-8")
    return g


def _set(g, *args, **kw):
    from typer.testing import CliRunner

    from localharness.cli.app import app
    return CliRunner().invoke(app, ["components", "set", *args, "--config-dir", str(g)], **kw)


def _overrides(g) -> dict:
    return yaml.safe_load((g / "overrides.yaml").read_text(encoding="utf-8"))


def test_a_dash_reads_the_key_from_stdin_and_shows_only_the_mask(tmp_path) -> None:
    g = _machine(tmp_path)
    result = _set(g, "provider.api_key", "-", input="sk-STDIN-0008\n")

    assert result.exit_code == 0, result.output
    assert _overrides(g)["provider"]["api_key"] == "sk-STDIN-0008"
    assert "'**********'" in result.output and "STDIN" not in result.output
    assert "STDIN" not in (g / "audit.jsonl").read_text(encoding="utf-8")


def test_a_dash_on_a_terminal_asks_with_a_hidden_prompt(tmp_path, monkeypatch) -> None:
    from localharness.cli import components_cmd

    asked: list = []

    def prompt(text, **kw):
        asked.append((text, kw))
        return "sk-TTY-0012"
    monkeypatch.setattr(components_cmd, "_stdin_is_a_terminal", lambda: True)
    monkeypatch.setattr(components_cmd.typer, "prompt", prompt)
    g = _machine(tmp_path)
    result = _set(g, "provider.api_key", "-")

    assert result.exit_code == 0, result.output
    assert asked == [("provider.api_key", {"hide_input": True, "default": "", "show_default": False})]
    assert _overrides(g)["provider"]["api_key"] == "sk-TTY-0012" and "TTY" not in result.output


def test_a_dash_with_nothing_read_writes_nothing(tmp_path) -> None:
    g = _machine(tmp_path)
    result = _set(g, "provider.api_key", "-", input="")

    assert result.exit_code == 2, result.output
    assert "No value read for provider.api_key; nothing was written." in result.output
    assert not (g / "overrides.yaml").exists()


def test_a_dash_for_a_setting_that_is_not_secret_is_the_value_itself(tmp_path) -> None:
    g = _machine(tmp_path)
    result = _set(g, "provider.default_model", "-", input="never-read\n")

    assert result.exit_code == 0, result.output
    assert _overrides(g)["provider"]["default_model"] == "-"


def test_a_typed_value_holding_a_key_is_never_echoed_by_a_refusal(tmp_path) -> None:
    """A peer endpoint list typed on the command line cannot be set (it is not a list), and the
    refusal must not print the key inside it."""
    g = _machine(tmp_path)
    typed = '[{"name": "x", "base_url": "http://x/v1", "api_key": "sk-TYPED-0009"}]'
    result = _set(g, "extra_endpoints", typed)

    assert result.exit_code == 2, result.output
    assert "Validation failed for extra_endpoints" in result.output
    assert "sk-TYPED-0009" not in result.output and "TYPED-0009" not in result.output
