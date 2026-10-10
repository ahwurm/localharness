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
