"""SEC-05 over the real CLI: a machine whose config holds every kind of stored credential — the
provider's key, a peer endpoint's key and header, the active endpoint's key, and an MCP server's env
and header in a global agent file — run through every command that shows settings. No key, and no
key's last 12 characters (pydantic shortens a long input to its head and tail), reaches stdout,
stderr or the audit log; the files hold the raw value; and what the session sends carries it raw.

A scratch HOME (`_layout`, no workspace), so every command resolves the machine's own config the way
a real install does; the model server address is port 9 (discard), never reached. Plain `def`
tests: `components set` and the client drive call asyncio.run themselves."""
from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli.app import app
from tests.unit.test_doctor_layer_report import _layout

runner = CliRunner()
MASK = "**********"
PROVIDER_KEY = "sk-SENT-PROVIDER-0001"
PEER_KEY = "sk-SENT-PEER-0002"
PEER_HEADER = "hdr-SENT-0003"
ACTIVE_KEY = "sk-SENT-ACTIVE-0004"
MCP_ENV = "env-SENT-0005"
MCP_HEADER = "mcp-SENT-0006"
NEW_KEY = "sk-SENT-NEW-0007"
STDIN_KEY = "sk-SENT-STDIN-0008"
SENTINELS = (PROVIDER_KEY, PEER_KEY, PEER_HEADER, ACTIVE_KEY, MCP_ENV, MCP_HEADER, NEW_KEY, STDIN_KEY)


def _config(**provider) -> dict:
    return {
        "version": "1",
        "provider": {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1",
                     "default_model": "test-model", "available_models": ["test-model"],
                     "api_key": PROVIDER_KEY, **provider},
        "extra_endpoints": [{"name": "peer-1", "base_url": "http://peer.lan/v1", "provider_type": "vllm",
                             "api_key": PEER_KEY, "extra_headers": {"X-Api-Key": PEER_HEADER}}],
        "active_endpoint": {"name": "peer-1", "base_url": "http://peer.lan/v1", "provider_type": "vllm",
                            "model": "peer-model", "api_key": ACTIVE_KEY},
    }


def _agent(**server) -> dict:
    return {"name": "orchestrator", "role": "General-purpose assistant", "model": "inherit",
            "tools": {"mcp_servers": [{"name": "files", "transport": "stdio", "command": "mcp-files",
                                       "env": {"TOKEN": MCP_ENV}, "headers": {"Authorization": MCP_HEADER},
                                       **server}]}}


def _machine(tmp_path: Path, monkeypatch, fake_home, config: dict, agent: dict) -> Path:
    layout = _layout(tmp_path, monkeypatch, fake_home, workspace=False,
                     global_config=yaml.safe_dump(config, sort_keys=False))
    (layout.global_dir / "agents").mkdir()
    (layout.global_dir / "agents" / "orchestrator.yaml").write_text(
        yaml.safe_dump(agent, sort_keys=False), encoding="utf-8")
    return layout.global_dir


@pytest.fixture
def machine(tmp_path, monkeypatch, fake_home) -> Path:
    return _machine(tmp_path, monkeypatch, fake_home, _config(), _agent())


def _run(*args: str, **kw):
    """One real command; no stored key or its tail in anything it printed or raised."""
    result = runner.invoke(app, list(args), **kw)
    for where in (result.stdout, result.stderr, repr(result.exception)):
        for secret in SENTINELS:
            assert secret not in where and secret[-12:] not in where, f"{secret} reached {args}:\n{where}"
    assert result.exception is None or isinstance(result.exception, SystemExit), repr(result.exception)
    return result


def _audit_is_clean(g: Path) -> str:
    text = (g / "audit.jsonl").read_text(encoding="utf-8")
    for secret in SENTINELS:
        assert secret not in text and secret[-12:] not in text, text
    return text


# --- every surface that shows a setting -----------------------------------------------------------

def test_components_get_shows_the_mask_in_text_and_json(machine) -> None:
    text = _run("components", "get", "provider.api_key")
    assert text.exit_code == 0 and "SecretStr('**********')" in text.stdout, text.output

    peers = _run("components", "get", "extra_endpoints")
    assert peers.exit_code == 0 and "peer-1" in peers.stdout and "http://peer.lan/v1" in peers.stdout

    as_json = _run("components", "get", "extra_endpoints", "--json")
    assert as_json.exit_code == 0, as_json.output  # TypeError: EndpointRef is not JSON serializable, in 0.16.0
    [peer] = json.loads(as_json.stdout)["value"]
    assert (peer["name"], peer["api_key"], peer["extra_headers"]) == ("peer-1", MASK, {"X-Api-Key": MASK})

    key = _run("components", "get", "provider.api_key", "--json")
    assert key.exit_code == 0 and json.loads(key.stdout)["value"] == MASK


def test_components_list_masks_text_and_json(machine) -> None:
    assert _run("components", "list").exit_code == 0
    listed = _run("components", "list", "--json")
    assert listed.exit_code == 0, listed.output
    rows = {row["path"]: row["current_value"] for row in json.loads(listed.stdout)}
    assert rows["provider.api_key"] == MASK and rows["active_endpoint.api_key"] == MASK
    assert rows["extra_endpoints"][0]["api_key"] == MASK


def test_config_show_masks_text_and_json(machine) -> None:
    shown = _run("config", "show")
    assert shown.exit_code == 0 and "provider.api_key" in shown.stdout, shown.output
    as_json = _run("config", "show", "--json")
    assert as_json.exit_code == 0, as_json.output  # the same TypeError, in 0.16.0
    assert MASK in as_json.stdout and "peer-1" in as_json.stdout


def test_validate_and_doctor_never_print_a_key(machine) -> None:
    valid = _run("validate")
    assert valid.exit_code == 0 and "orchestrator.yaml" in valid.stdout, valid.output
    doctor = _run("doctor")  # exits 1: nothing serves port 9; only its output is graded
    assert "Config valid" in doctor.stdout, doctor.output


@pytest.mark.parametrize("broken", ["a provider field", "a whole endpoint and a whole server"])
def test_a_refused_config_never_prints_a_key(tmp_path, monkeypatch, fake_home, broken) -> None:
    """A field error, and errors on a whole endpoint and a whole MCP server — whose input is the
    section around them, keys included: validate, doctor and components list print the problem and
    never the key."""
    if broken == "a provider field":
        config, agent = _config(timeout_seconds="x"), _agent()
        expect = "provider.timeout_seconds"
    else:
        config = _config()
        config["extra_endpoints"][0]["lifecycle"] = {"runtime": "vllm", "model": "m", "port": 8001,
                                                     "binary": "/usr/bin/vllm"}
        agent = _agent(command=None)
        expect = "must be gpu=True"
    _machine(tmp_path, monkeypatch, fake_home, config, agent)

    valid = _run("validate")
    flat = " ".join(valid.output.split())
    assert valid.exit_code == 1 and expect in flat, valid.output
    doctor = _run("doctor")
    assert "Config invalid" in doctor.stdout and expect in " ".join(doctor.stdout.split()), doctor.output
    listed = _run("components", "list")
    assert listed.exit_code == 2 and expect in " ".join(listed.output.split()), listed.output
    if broken != "a provider field":
        assert "'command' is required for stdio transport" in flat, valid.output
        # each value shown with its keys masked, and its ordinary words as written (R14)
        assert f"'X-Api-Key': '{MASK}'" in flat and f"'TOKEN': '{MASK}'" in flat, valid.output
        assert "'name': 'peer-1'" in flat and "'base_url': 'http://peer.lan/v1'" in flat, valid.output


# --- writing a key ----------------------------------------------------------------------------------

def test_components_set_prints_and_audits_only_the_mask_and_stores_the_key(machine) -> None:
    result = _run("components", "set", "provider.api_key", NEW_KEY)

    assert result.exit_code == 0, result.output
    assert f"provider.api_key = '{MASK}' (was: '{MASK}')" in " ".join(result.stdout.split())
    overrides = machine / "overrides.yaml"
    assert yaml.safe_load(overrides.read_text(encoding="utf-8"))["provider"]["api_key"] == NEW_KEY
    assert stat.S_IMODE(overrides.stat().st_mode) == 0o600
    [event] = [json.loads(line) for line in _audit_is_clean(machine).splitlines()]
    assert (event["path"], event["before_value"], event["after_value"]) == ("provider.api_key", MASK, MASK)


def test_components_set_dash_reads_the_key_without_echo(machine) -> None:
    result = _run("components", "set", "provider.api_key", "-", input=f"{STDIN_KEY}\n")

    assert result.exit_code == 0, result.output
    assert yaml.safe_load((machine / "overrides.yaml").read_text(encoding="utf-8"))["provider"]["api_key"] == STDIN_KEY
    assert MASK in result.stdout
    _audit_is_clean(machine)


# --- what the session sends -----------------------------------------------------------------------

class _Stop(Exception):
    """Raised where an MCP transport would start: the parameters it was handed are what is checked."""


class _Refuses:
    async def __aenter__(self):
        raise _Stop

    async def __aexit__(self, *exc) -> bool:
        return False


def test_the_loaded_keys_reach_the_wire_raw(machine, monkeypatch) -> None:
    """The machine's config, loaded as `start` loads it: the model server gets the provider's key,
    a /model switch to the peer sends the peer's key and header, and the MCP server is spawned with
    its env and header — each raw, each exactly once unwrapped."""
    from localharness.config.loader import ConfigLoader
    from localharness.provider import client as client_mod
    from localharness.provider.client import LLMClient, LLMConfig
    from localharness.tools import mcp as mcp_mod
    from localharness.tools.mcp import MCPServerClient

    seen: list[httpx.Request] = []
    monkeypatch.setattr(client_mod, "_TRANSPORT", httpx.MockTransport(
        lambda request: seen.append(request) or httpx.Response(200, json={"object": "list", "data": []})))
    loader = ConfigLoader(config_dir=machine)
    harness, agent = loader.load_harness(), loader.load_agent("orchestrator")
    provider, peer = harness.provider, harness.extra_endpoints[0]

    async def drive() -> None:
        # as start_cmd builds the session's client, and as the REPL's /model re-points it
        llm = LLMClient(LLMConfig(base_url=provider.base_url, model=provider.default_model,
                                  api_key=provider.api_key))
        try:
            await llm._client.models.list()
            llm.rebind_endpoint(peer.base_url, api_key=peer.api_key, extra_headers=peer.extra_headers)
            await llm._client.models.list()
        finally:
            await llm.aclose()

    asyncio.run(drive())
    assert [(r.url.host, r.headers["authorization"]) for r in seen] == [
        ("127.0.0.1", f"Bearer {PROVIDER_KEY}"), ("peer.lan", f"Bearer {PEER_KEY}")]
    assert seen[1].headers["x-api-key"] == PEER_HEADER

    got: dict = {}

    def stdio(params):
        got["params"] = params
        return _Refuses()

    def http(url, headers=None, **_):
        got["headers"] = headers
        return _Refuses()

    monkeypatch.setattr(mcp_mod, "stdio_client", stdio)
    monkeypatch.setattr(mcp_mod, "streamablehttp_client", http)
    server = agent.tools.mcp_servers[0]
    for config in (server, server.model_copy(update={"transport": "streamable_http", "url": "http://mcp.test/mcp"})):
        with pytest.raises(_Stop):
            asyncio.run(MCPServerClient(config).connect())
    assert got["params"].env["TOKEN"] == MCP_ENV
    assert got["headers"] == {"Authorization": MCP_HEADER}
