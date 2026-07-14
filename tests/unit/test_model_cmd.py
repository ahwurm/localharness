"""`localharness model` CLI — list + switch parity with the REPL /model (gap #4).

Offline: the live-model probe is faked at model_ops.list_live_models so no runtime is hit.
"""
from __future__ import annotations

import yaml
from typer.testing import CliRunner

from localharness.cli import model_ops
from localharness.cli.app import app
from localharness.config.overlay import load_overlay

runner = CliRunner()


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


def test_model_cli_warns_on_pinned_agent(components_home, monkeypatch):
    _seed_config(
        components_home,
        agents={"pinned.yaml": "name: pinned-agent\nrole: x\nmodel: some-pinned-model\n"},
    )
    monkeypatch.setattr(model_ops, "list_live_models", _fake_live(["model-a", "model-b"]), raising=False)
    result = runner.invoke(app, ["model", "model-b"])
    assert result.exit_code == 0, result.output
    assert "pinned-agent" in result.output and "some-pinned-model" in result.output
