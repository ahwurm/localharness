"""`localharness model` CLI — list + switch parity with the REPL /model (gap #4).

Offline: the live-model probe is faked at model_ops.list_live_models so no runtime is hit.
"""
from __future__ import annotations

import json as _json

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli import model_ops
from localharness.cli.app import app
from localharness.config.overlay import load_overlay

runner = CliRunner()


class _HtmlResp:
    """A reached endpoint whose body isn't an OpenAI model list (e.g. an HTML error page)."""

    def json(self):
        raise _json.JSONDecodeError("Expecting value", "<html></html>", 0)


def test_list_live_models_unreachable_on_network_error(monkeypatch):
    """#38: a transport error (connection refused/DNS/timeout) is unreachable → ([], False)."""
    import httpx

    def _boom(*a, **k):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", _boom)
    assert model_ops.list_live_models("http://localhost:9/v1") == ([], False)


def test_list_live_models_malformed_raises_distinct_signal(monkeypatch):
    """#38: reached-but-malformed (bad JSON / wrong shape) is NOT unreachable — it's a distinct
    typed signal so callers stop saying 'is it running?' at a live-but-wrong endpoint."""
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _HtmlResp())
    with pytest.raises(model_ops.MalformedModelListError):
        model_ops.list_live_models("http://localhost:8081/v1")


def test_list_live_models_empty_body_is_reachable(monkeypatch):
    """Reached-but-empty ({"data": []}) stays distinct from unreachable (a legit empty result)."""
    import httpx

    class _EmptyResp:
        def json(self):
            return {"data": []}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _EmptyResp())
    assert model_ops.list_live_models("http://x/v1") == ([], True)


def test_model_list_malformed_response_distinct_from_unreachable(components_home, monkeypatch):
    """#38: a reached-but-malformed /models body renders as its OWN message, NOT 'Is it running?'
    (which wrongly implies the server is down)."""
    import httpx

    _seed_config(components_home)
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _HtmlResp())
    result = runner.invoke(app, ["model"])
    assert result.exit_code == 2, result.output
    out = result.output.lower()
    assert "is it running" not in out
    assert "wasn't understood" in out or "openai-compatible" in out


def _seed_config(home, *, agents=None) -> None:
    """Write a minimal loadable HarnessConfig at LOCALHARNESS_HOME/config.yaml (audit → tmp)."""
    data = {
        "version": "1",
        "provider": {
            "provider_type": "vllm",
            "base_url": "http://localhost:8081/v1",
            "default_model": "model-a",
            "available_models": ["model-a"],
        },
        "org": {"default_model": "model-a", "audit_log_path": str(home / "audit.jsonl")},
    }
    (home / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    if agents:
        adir = home / "agents"
        adir.mkdir(exist_ok=True)
        for fname, body in agents.items():
            (adir / fname).write_text(body, encoding="utf-8")


def _fake_live(models, reachable=True):
    return lambda *a, **k: (list(models), reachable)


def test_model_list_shows_served_and_active(components_home, monkeypatch):
    _seed_config(components_home)
    monkeypatch.setattr(model_ops, "list_live_models", _fake_live(["model-a", "model-b"]), raising=False)
    result = runner.invoke(app, ["model"])
    assert result.exit_code == 0, result.output
    assert "model-a" in result.output and "[active]" in result.output
    assert "model-b" in result.output


def test_model_switch_persists_default(components_home, monkeypatch):
    _seed_config(components_home)
    monkeypatch.setattr(model_ops, "list_live_models", _fake_live(["model-a", "model-b"]), raising=False)
    result = runner.invoke(app, ["model", "model-b"])
    assert result.exit_code == 0, result.output
    assert "model-b" in result.output
    overlay = load_overlay(components_home / "overrides.yaml")
    assert overlay["provider"]["default_model"] == "model-b"
    assert overlay["org"]["default_model"] == "model-b"


def test_model_switch_by_number(components_home, monkeypatch):
    _seed_config(components_home)
    monkeypatch.setattr(model_ops, "list_live_models", _fake_live(["model-a", "model-b"]), raising=False)
    result = runner.invoke(app, ["model", "2"])
    assert result.exit_code == 0, result.output
    overlay = load_overlay(components_home / "overrides.yaml")
    assert overlay["provider"]["default_model"] == "model-b"


def test_model_switch_unknown_hard_errors_naming_available(components_home, monkeypatch):
    _seed_config(components_home)
    monkeypatch.setattr(model_ops, "list_live_models", _fake_live(["model-a"]), raising=False)
    result = runner.invoke(app, ["model", "ghost"])
    assert result.exit_code == 2
    assert "ghost" in result.output and "model-a" in result.output


def test_model_switch_unreachable_degrades_with_disclosure(components_home, monkeypatch):
    _seed_config(components_home)
    monkeypatch.setattr(model_ops, "list_live_models", _fake_live([], reachable=False), raising=False)
    result = runner.invoke(app, ["model", "future-model"])
    assert result.exit_code == 0, result.output
    assert "future-model" in result.output
    # Degraded but persisted, with an explicit "could not verify" disclosure.
    assert "verif" in result.output.lower() or "unverified" in result.output.lower()
    overlay = load_overlay(components_home / "overrides.yaml")
    assert overlay["provider"]["default_model"] == "future-model"


def test_model_config_dir_isolates_overlay(tmp_path, monkeypatch):
    """#35: `--config-dir` must isolate the overlay — a switch under dirA writes dirA's
    overlay and NEVER touches a sibling config dir. Before the fix the overlay keyed only on
    LOCALHARNESS_HOME/~-default, so `--config-dir` was ignored (isolation was a lie)."""
    dir_a = tmp_path / "a"
    dir_a.mkdir()
    dir_b = tmp_path / "b"
    dir_b.mkdir()
    _seed_config(dir_a)
    _seed_config(dir_b)
    monkeypatch.setattr(
        model_ops, "list_live_models", _fake_live(["model-a", "model-b"]), raising=False
    )
    result = runner.invoke(app, ["model", "model-b", "--config-dir", str(dir_a)])
    assert result.exit_code == 0, result.output
    assert load_overlay(dir_a / "overrides.yaml")["provider"]["default_model"] == "model-b"
    # Fully isolated: the sibling config dir is never written.
    assert not (dir_b / "overrides.yaml").exists()


def test_model_config_dir_env_var_honored(tmp_path, monkeypatch):
    """#35: the LOCALHARNESS_DIR env var (what --config-dir binds) routes the overlay too."""
    cfg_dir = tmp_path / "envdir"
    cfg_dir.mkdir()
    _seed_config(cfg_dir)
    monkeypatch.setenv("LOCALHARNESS_DIR", str(cfg_dir))
    monkeypatch.setattr(
        model_ops, "list_live_models", _fake_live(["model-a", "model-b"]), raising=False
    )
    result = runner.invoke(app, ["model", "model-b"])
    assert result.exit_code == 0, result.output
    assert load_overlay(cfg_dir / "overrides.yaml")["provider"]["default_model"] == "model-b"


def test_model_switch_persists_server_model_survives_reload(tmp_path):
    """#34: a switch on a managed-server harness must persist server.model too — else cold start
    rebuilds `vllm serve <srv.model>` from the STALE config value and relaunches the old model.
    Prove it survives a fresh ConfigLoader (the exact cold-start path)."""
    import asyncio

    from localharness.config.loader import ConfigLoader

    home = tmp_path / ".localharness"
    home.mkdir()
    data = {
        "version": "1",
        "provider": {
            "provider_type": "vllm",
            "base_url": "http://localhost:8081/v1",
            "default_model": "model-a",
            "available_models": ["model-a"],
        },
        "org": {"default_model": "model-a"},
        "server": {"runtime": "vllm", "launch": "binary", "binary": "/x/vllm", "model": "model-a"},
    }
    (home / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    harness = ConfigLoader(config_dir=home).load_harness()
    assert harness.server.model == "model-a"

    asyncio.run(model_ops.persist_default_model(harness, "model-b", config_dir=home))

    fresh = ConfigLoader(config_dir=home).load_harness()
    assert fresh.provider.default_model == "model-b"
    assert fresh.server.model == "model-b"  # THE gap: server.model persisted + survives reload
    assert load_overlay(home / "overrides.yaml")["server"]["model"] == "model-b"


def test_model_switch_no_server_never_invents_server_key(tmp_path):
    """#34 invariant: with NO managed server configured, persist must never write a server.* key
    (a bare server:{model} would fail ManagedServerConfig validation — binary/launch required)."""
    import asyncio

    from localharness.config.loader import ConfigLoader

    home = tmp_path / ".localharness"
    home.mkdir()
    data = {
        "version": "1",
        "provider": {
            "provider_type": "vllm",
            "base_url": "http://localhost:8081/v1",
            "default_model": "model-a",
            "available_models": ["model-a"],
        },
        "org": {"default_model": "model-a"},
    }
    (home / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    harness = ConfigLoader(config_dir=home).load_harness()
    assert harness.server is None

    asyncio.run(model_ops.persist_default_model(harness, "model-b", config_dir=home))

    assert "server" not in load_overlay(home / "overrides.yaml")
    fresh = ConfigLoader(config_dir=home).load_harness()
    assert fresh.provider.default_model == "model-b" and fresh.server is None


def test_model_switch_audit_failure_still_persists_exit_zero(components_home, monkeypatch):
    """#37: an audit-log emit failure must NOT be reported as a persist failure. The durable
    overlay write already succeeded — surface a secondary warning and exit 0, not exit 2."""
    _seed_config(components_home)
    monkeypatch.setattr(
        model_ops, "list_live_models", _fake_live(["model-a", "model-b"]), raising=False
    )

    class _BoomBus:
        def __init__(self, *a, **k):
            pass

        async def publish(self, *a, **k):
            raise RuntimeError("audit disk full")

    monkeypatch.setattr(model_ops, "EventBus", _BoomBus)
    result = runner.invoke(app, ["model", "model-b"])
    assert result.exit_code == 0, result.output  # persist succeeded despite audit failure
    # Overlay durably written.
    assert (
        load_overlay(components_home / "overrides.yaml")["provider"]["default_model"] == "model-b"
    )
    # Secondary warning surfaced, honestly labeled as audit-only.
    assert "audit" in result.output.lower()


def test_pinned_agents_includes_division_pin(tmp_path):
    """#36: a division-pinned model traps an inheriting agent too — start resolves
    agent->division->org, so a persisted org/provider switch never reaches it. pinned_agents
    must list it (annotated 'via division <name>') alongside agent-level pins; a full inheritor
    (agent + division both inherit) is NOT listed."""
    (tmp_path / "agents").mkdir()
    (tmp_path / "divisions").mkdir()
    (tmp_path / "divisions" / "research.yaml").write_text(
        "name: research\nmodel: division-pinned-model\n", encoding="utf-8"
    )
    (tmp_path / "agents" / "writer.yaml").write_text(  # inherits at agent level, division pins
        "name: writer\nrole: x\ndivision: research\n", encoding="utf-8"
    )
    (tmp_path / "agents" / "pinned.yaml").write_text(  # agent-level pin
        "name: pinned-agent\nrole: x\nmodel: agent-pinned-model\n", encoding="utf-8"
    )
    (tmp_path / "agents" / "plain.yaml").write_text(  # inherits all the way to org
        "name: plain\nrole: x\nmodel: inherit\n", encoding="utf-8"
    )

    result = dict(model_ops.pinned_agents(tmp_path))
    assert result["pinned-agent"] == "agent-pinned-model"  # agent-level pin still listed
    assert result["writer (via division research)"] == "division-pinned-model"  # division pin
    assert not any("plain" in name for name in result)  # a full inheritor is never listed


def test_model_switch_rejects_empty_name(components_home, monkeypatch):
    """#39: an empty/whitespace name must be rejected loudly (exit 2), never resolved or
    persisted. Before the fix "" fell through to the unreachable-degrade branch and persisted ""."""
    _seed_config(components_home)
    # Unreachable endpoint = the degrade branch that used to swallow "" as the new default.
    monkeypatch.setattr(
        model_ops, "list_live_models", _fake_live([], reachable=False), raising=False
    )
    for bad in ("", "   "):
        result = runner.invoke(app, ["model", bad])
        assert result.exit_code == 2, (bad, result.output)
        assert not (components_home / "overrides.yaml").exists()  # nothing persisted


def test_model_switch_empty_name_rejected_before_checkpoint_branch(components_home, monkeypatch):
    """#39: Path("").expanduser().exists() is truthy (== cwd), so an empty name could slip
    through the managed-server checkpoint branch and persist "". The guard must reject FIRST."""
    data = {
        "version": "1",
        "provider": {
            "provider_type": "vllm",
            "base_url": "http://localhost:8081/v1",
            "default_model": "model-a",
            "available_models": ["model-a"],
        },
        "org": {"default_model": "model-a", "audit_log_path": str(components_home / "audit.jsonl")},
        "server": {"runtime": "vllm", "launch": "binary", "binary": "/x/vllm", "model": "model-a"},
    }
    (components_home / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr(
        model_ops, "list_live_models", _fake_live([], reachable=True), raising=False
    )
    from localharness.provider import server as managed_server

    monkeypatch.setattr(managed_server, "list_cached_models", lambda: [], raising=False)
    result = runner.invoke(app, ["model", ""])
    assert result.exit_code == 2, result.output
    assert not (components_home / "overrides.yaml").exists()


def test_model_list_warns_when_default_not_served(components_home, monkeypatch):
    """#50: server reachable but the configured default is NOT among the served models —
    the list must state so plainly instead of silently showing no [active] marker anywhere."""
    _seed_config(components_home)  # default_model = model-a
    monkeypatch.setattr(
        model_ops, "list_live_models", _fake_live(["model-x", "model-y"]), raising=False
    )
    result = runner.invoke(app, ["model"])
    assert result.exit_code == 0, result.output
    assert "not among the served models" in result.output
    assert "model-a" in result.output  # names the stale default


def test_model_list_no_warning_when_default_served(components_home, monkeypatch):
    """#50: when the configured default IS served, the mismatch warning must NOT fire."""
    _seed_config(components_home)  # default_model = model-a
    monkeypatch.setattr(
        model_ops, "list_live_models", _fake_live(["model-a", "model-b"]), raising=False
    )
    result = runner.invoke(app, ["model"])
    assert result.exit_code == 0, result.output
    assert "not among the served models" not in result.output


def test_model_cli_warns_on_pinned_agent(components_home, monkeypatch):
    _seed_config(
        components_home,
        agents={"pinned.yaml": "name: pinned-agent\nrole: x\nmodel: some-pinned-model\n"},
    )
    monkeypatch.setattr(model_ops, "list_live_models", _fake_live(["model-a", "model-b"]), raising=False)
    result = runner.invoke(app, ["model", "model-b"])
    assert result.exit_code == 0, result.output
    assert "pinned-agent" in result.output and "some-pinned-model" in result.output
