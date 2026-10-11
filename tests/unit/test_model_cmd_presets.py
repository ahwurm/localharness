"""`localharness model <name>` resolves presets as `/model` does (D-35): a lifecycle peer's name or
served name selects that peer, the primary's served name clears the selection, anything else is
today's path.

Offline: the live-model probe is faked at model_ops.list_live_models, no server is started.
"""
from __future__ import annotations

import yaml
from typer.testing import CliRunner

from localharness.cli import model_cmd, model_ops
from localharness.cli.app import app
from localharness.config.overlay import load_overlay

runner = CliRunner()

PRIMARY_SERVED = "qwen-27b"


def _peer(name="flash-next", *, served_name="flash-next", extra_args=None, port=8082,
          lifecycle=True) -> dict:
    ep = {
        "name": name,
        "base_url": f"http://127.0.0.1:{port}/v1",
        "provider_type": "llamacpp",
    }
    if lifecycle:
        ep["gpu"] = True
        lc = {"gpu": True, "runtime": "llamacpp", "model": "/m/b.gguf", "port": port}
        if served_name:
            lc["served_name"] = served_name
        if extra_args:
            lc["extra_args"] = list(extra_args)
        ep["lifecycle"] = lc
    return ep


def _seed(home, *, peers=None, server=None, provider=None) -> None:
    """A llamacpp primary served as `qwen-27b` plus the given peers, at <home>/config.yaml."""
    data = {
        "version": "1",
        "provider": provider or {
            "provider_type": "llamacpp",
            "base_url": "http://localhost:8081/v1",
            "default_model": PRIMARY_SERVED,
            "available_models": [PRIMARY_SERVED],
        },
        "org": {"default_model": PRIMARY_SERVED, "audit_log_path": str(home / "audit.jsonl")},
        "server": server or {
            "runtime": "llamacpp", "binary": "/x/llama-server", "model": "/m/a.gguf",
            "served_name": PRIMARY_SERVED, "port": 8081,
        },
        "extra_endpoints": peers if peers is not None else [_peer()],
    }
    (home / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def _unreachable(monkeypatch) -> None:
    monkeypatch.setattr(model_ops, "list_live_models", lambda *a, **k: ([], False), raising=False)


def _overlay(home) -> dict:
    return load_overlay(home / "overrides.yaml")


def test_peer_by_name_selects_active_endpoint(components_home, monkeypatch):
    _seed(components_home)
    _unreachable(monkeypatch)
    result = runner.invoke(app, ["model", "flash-next", "--config-dir", str(components_home)])
    assert result.exit_code == 0, result.output
    overlay = _overlay(components_home)
    assert overlay["active_endpoint"] == {
        "name": "flash-next", "base_url": "http://127.0.0.1:8082/v1",
        "provider_type": "llamacpp", "model": "flash-next",
    }
    assert "effort" not in overlay["active_endpoint"]
    assert "server" not in overlay and "provider" not in overlay
    assert "Selected flash-next on flash-next" in result.output.replace("\n", " ")


def test_peer_by_served_name_selects_that_peer(components_home, monkeypatch):
    _seed(components_home, peers=[_peer("flash-next", served_name="fn-q3")])
    _unreachable(monkeypatch)
    for arg in ("fn-q3", "flash-next"):
        result = runner.invoke(app, ["model", arg, "--config-dir", str(components_home)])
        assert result.exit_code == 0, result.output
        active = _overlay(components_home)["active_endpoint"]
        assert active["name"] == "flash-next" and active["model"] == "fn-q3"


def test_peer_served_name_from_extra_args_alias(components_home, monkeypatch):
    _seed(components_home, peers=[_peer("flash-next", served_name=None,
                                        extra_args=["-a", "alias-x"])])
    _unreachable(monkeypatch)
    result = runner.invoke(app, ["model", "alias-x", "--config-dir", str(components_home)])
    assert result.exit_code == 0, result.output
    active = _overlay(components_home)["active_endpoint"]
    assert active["name"] == "flash-next" and active["model"] == "alias-x"


def test_selecting_a_peer_drops_a_previous_effort(components_home, monkeypatch):
    _seed(components_home)
    _unreachable(monkeypatch)
    (components_home / "overrides.yaml").write_text(
        yaml.safe_dump({"active_endpoint": {
            "name": "old", "base_url": "http://127.0.0.1:9/v1", "provider_type": "llamacpp",
            "model": "old", "effort": "high"}}), encoding="utf-8")
    result = runner.invoke(app, ["model", "flash-next", "--config-dir", str(components_home)])
    assert result.exit_code == 0, result.output
    assert "effort" not in _overlay(components_home)["active_endpoint"]


def test_peer_served_name_rule():
    from types import SimpleNamespace as NS

    def ep(name, **lc):
        return NS(name=name, lifecycle=NS(served_name=lc.get("served_name"),
                                          extra_args=lc.get("extra_args", [])))

    assert model_ops.peer_served_name(ep("p", served_name="s")) == "s"
    assert model_ops.peer_served_name(ep("p", extra_args=["-a", "al"])) == "al"
    assert model_ops.peer_served_name(ep("p", extra_args=["--alias", "al2"])) == "al2"
    assert model_ops.peer_served_name(ep("p", extra_args=["--served-model-name", "al3"])) == "al3"
    assert model_ops.peer_served_name(ep("p", extra_args=["--alias=al4"])) == "al4"
    assert model_ops.peer_served_name(ep("p", extra_args=["--served-model-name=al5"])) == "al5"
    assert model_ops.peer_served_name(ep("p", extra_args=["--alias="])) == "p"
    assert model_ops.peer_served_name(ep("p", extra_args=["-a"])) == "p"
    assert model_ops.peer_served_name(ep("p")) == "p"
    assert model_ops.peer_served_name(NS(name="q", lifecycle=None)) == "q"


def test_primary_served_name_rule():
    from types import SimpleNamespace as NS

    prov = NS(default_model="dm")
    assert model_ops.primary_served_name(None, prov) is None
    assert model_ops.primary_served_name(NS(served_name="s", extra_args=[]), prov) == "s"
    assert model_ops.primary_served_name(NS(served_name=None, extra_args=["-a", "al"]), prov) == "al"
    assert model_ops.primary_served_name(NS(served_name=None, extra_args=[]), prov) == "dm"


# --- the primary's served name clears the selection (D-35, D-58) ---

def _select_peer_overlay(home, **extra) -> None:
    body = {"active_endpoint": {
        "name": "flash-next", "base_url": "http://127.0.0.1:8082/v1",
        "provider_type": "llamacpp", "model": "flash-next"}}
    body.update(extra)
    (home / "overrides.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")


def _record_persist_default(monkeypatch) -> list[dict]:
    """Wrap persist_default_model, recording the keywords it is called with."""
    calls: list[dict] = []
    real = model_ops.persist_default_model

    async def spy(harness, model, **kw):
        calls.append({"model": model, **kw})
        return await real(harness, model, **kw)

    monkeypatch.setattr(model_ops, "persist_default_model", spy)
    return calls


def test_primary_served_name_clears_selection_and_sets_default(components_home, monkeypatch):
    _seed(components_home, provider={
        "provider_type": "llamacpp", "base_url": "http://localhost:8081/v1",
        "default_model": "stale-default", "available_models": ["stale-default"]})
    _unreachable(monkeypatch)
    _select_peer_overlay(components_home)
    result = runner.invoke(app, ["model", PRIMARY_SERVED, "--config-dir", str(components_home)])
    assert result.exit_code == 0, result.output
    overlay = _overlay(components_home)
    assert "active_endpoint" not in overlay
    assert overlay["provider"]["default_model"] == PRIMARY_SERVED
    # a llama.cpp primary: the alias would become `-m`, so server.model is never written
    assert "server" not in overlay


def test_primary_switch_back_with_default_already_served_name(components_home, monkeypatch):
    _seed(components_home)
    _unreachable(monkeypatch)
    _select_peer_overlay(components_home)
    result = runner.invoke(app, ["model", PRIMARY_SERVED, "--config-dir", str(components_home)])
    assert result.exit_code == 0, result.output
    overlay = _overlay(components_home)
    assert "active_endpoint" not in overlay
    assert "server" not in overlay


def test_primary_served_name_passes_local_operator_both_ways(components_home, monkeypatch):
    for value in (True, False):
        _seed(components_home, provider={
            "provider_type": "llamacpp", "base_url": "http://localhost:8081/v1",
            "default_model": "stale-default", "available_models": ["stale-default"]})
        (components_home / "overrides.yaml").unlink(missing_ok=True)
        _unreachable(monkeypatch)
        calls = _record_persist_default(monkeypatch)
        monkeypatch.setattr(model_cmd, "_is_local_operator", lambda v=value: v)
        result = runner.invoke(app, ["model", PRIMARY_SERVED, "--config-dir", str(components_home)])
        assert result.exit_code == 0, result.output
        assert [c["local_operator"] for c in calls] == [value]


def test_primary_served_name_already_default_does_not_rewrite(components_home, monkeypatch):
    _seed(components_home)
    _unreachable(monkeypatch)
    calls = _record_persist_default(monkeypatch)
    result = runner.invoke(app, ["model", PRIMARY_SERVED, "--config-dir", str(components_home)])
    assert result.exit_code == 0, result.output
    assert "already the default" in result.output.replace("\n", " ")
    assert not (components_home / "overrides.yaml").exists()
    assert calls == []

    # and with an overlay present its bytes and mtime stay untouched
    path = components_home / "overrides.yaml"
    path.write_text(yaml.safe_dump({"org": {"default_model": PRIMARY_SERVED}}), encoding="utf-8")
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    result = runner.invoke(app, ["model", PRIMARY_SERVED, "--config-dir", str(components_home)])
    assert result.exit_code == 0, result.output
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_clear_active_endpoint_without_the_key_writes_nothing(tmp_path):
    from localharness.config.loader import ConfigLoader

    _seed(tmp_path)
    harness = ConfigLoader(config_dir=str(tmp_path)).load_harness()
    path = tmp_path / "overrides.yaml"
    path.write_text(yaml.safe_dump({"org": {"default_model": "x"}}), encoding="utf-8")
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    model_ops.clear_active_endpoint(harness, config_dir=tmp_path)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_clear_active_endpoint_removes_only_that_key(tmp_path):
    from localharness.config.loader import ConfigLoader

    _seed(tmp_path)
    agent = {"agent": {"researcher": {"model": "m-1"}}}
    _select_peer_overlay(tmp_path, org={"default_model": PRIMARY_SERVED}, **agent)
    harness = ConfigLoader(config_dir=str(tmp_path)).load_harness()
    assert harness.active_endpoint is not None
    model_ops.clear_active_endpoint(harness, config_dir=tmp_path)
    overlay = load_overlay(tmp_path / "overrides.yaml")
    assert overlay == {"org": {"default_model": PRIMARY_SERVED}, **agent}
    assert harness.active_endpoint is None


def _seed_vllm(home) -> None:
    data = {
        "version": "1",
        "provider": {"provider_type": "vllm", "base_url": "http://localhost:8081/v1",
                     "default_model": "org/m", "available_models": ["org/m"]},
        "org": {"default_model": "org/m", "audit_log_path": str(home / "audit.jsonl")},
        "server": {"runtime": "vllm", "binary": "/x/vllm", "model": "org/m", "port": 8081},
        "extra_endpoints": [_peer(lifecycle=False)],
    }
    (home / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def test_vllm_primary_default_name_keeps_todays_path(components_home, monkeypatch):
    _seed_vllm(components_home)
    monkeypatch.setattr(model_ops, "list_live_models",
                        lambda *a, **k: (["org/m"], True), raising=False)
    result = runner.invoke(app, ["model", "org/m", "--config-dir", str(components_home)])
    assert result.exit_code == 0, result.output
    assert "already the default" in result.output.replace("\n", " ")
    assert "active_endpoint" not in _overlay(components_home)


def test_unknown_name_with_reachable_server_still_errors(components_home, monkeypatch):
    _seed_vllm(components_home)
    monkeypatch.setattr(model_ops, "list_live_models",
                        lambda *a, **k: (["org/m"], True), raising=False)
    result = runner.invoke(app, ["model", "ghost", "--config-dir", str(components_home)])
    assert result.exit_code == 2, result.output
    assert "ghost" in result.output and "org/m" in result.output
    assert not (components_home / "overrides.yaml").exists()


def test_lifecycle_less_peer_name_is_not_a_preset(components_home, monkeypatch):
    _seed_vllm(components_home)
    monkeypatch.setattr(model_ops, "list_live_models",
                        lambda *a, **k: (["org/m"], True), raising=False)
    result = runner.invoke(app, ["model", "flash-next", "--config-dir", str(components_home)])
    assert result.exit_code == 2, result.output
    assert "unknown model" in result.output
    assert "active_endpoint" not in _overlay(components_home)


def test_a_vllm_alias_swap_back_never_becomes_the_vllm_model(components_home, monkeypatch):
    """WR-54: `--served-model-name qwen` is the name the primary answers to, not what `vllm serve` loads."""
    _seed_vllm(components_home)
    path = components_home / "config.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["server"].update({"model": "other/m", "extra_args": ["--served-model-name", "qwen"]})
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    _unreachable(monkeypatch)
    calls = _record_persist_default(monkeypatch)

    result = runner.invoke(app, ["model", "qwen", "--config-dir", str(components_home)])

    assert result.exit_code == 0, result.output
    overlay = _overlay(components_home)
    assert overlay["provider"]["default_model"] == "qwen" and overlay["org"]["default_model"] == "qwen"
    assert "server" not in overlay
    assert [c["write_server_model"] for c in calls] == [False]


def test_a_plain_model_switch_on_a_vllm_primary_still_writes_server_model(components_home, monkeypatch):
    _seed_vllm(components_home)
    monkeypatch.setattr(model_ops, "list_live_models",
                        lambda *a, **k: (["org/m", "org/n"], True), raising=False)

    result = runner.invoke(app, ["model", "org/n", "--config-dir", str(components_home)])

    assert result.exit_code == 0, result.output
    assert _overlay(components_home)["server"]["model"] == "org/n"


def test_the_repls_cold_peer_label_is_the_same_rule():
    """IN-55: one copy of the rule, so `--alias=NAME` matches in `/model` as in `localharness model`."""
    from types import SimpleNamespace as NS

    from localharness.cli.repl import OrchestratorREPL

    for args, expected in ((["--alias=al4"], "al4"), (["-a", "al"], "al"),
                           (["--served-model-name", "al3"], "al3"), ([], "p")):
        ep = NS(name="p", lifecycle=NS(served_name=None, extra_args=args))
        assert OrchestratorREPL._cold_served_name(ep) == expected
    assert model_ops.primary_served_name(NS(served_name=None, extra_args=["--alias=al4"]),
                                         NS(default_model="d")) == "al4"
