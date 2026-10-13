"""`localharness model <name>` decides machine trust as `start` does (D-09, D-10, SEC-02).

A person at a terminal is asked; a pipe, a script or an agent's command is not, so a piped `y`
adopts nothing and an EOF records no No. A withheld target is refused (exit 2) with nothing cleared.
The CLI never launches (D-10): a preset selection records the intent the next start's ladder reads.

Offline: the live-model probe is faked at model_ops.list_live_models; no server is started.
"""
from __future__ import annotations

import yaml
from typer.testing import CliRunner

from localharness.cli import model_cmd, model_ops, workspace
from localharness.cli.app import app
from localharness.cli.workspace import MachineTrust
from localharness.config import trust
from localharness.config.loader import ConfigLoader
from localharness.config.overlay import load_overlay
from tests.unit.test_model_cmd_presets import _overlay, _peer, _seed, _unreachable

runner = CliRunner()


def _spy_decide(monkeypatch) -> list[dict]:
    """Wrap decide_machine_trust as model_cmd names it, recording each call's keywords."""
    calls: list[dict] = []
    real = model_cmd.decide_machine_trust

    def spy(global_dir, **kw):
        calls.append(dict(kw))
        return real(global_dir, **kw)

    monkeypatch.setattr(model_cmd, "decide_machine_trust", spy)
    return calls


def _spy_loader(monkeypatch) -> list[dict]:
    """Wrap the ConfigLoader model_cmd builds, recording its keywords."""
    calls: list[dict] = []

    class Spy(ConfigLoader):
        def __init__(self, *a, **kw):
            calls.append(dict(kw))
            super().__init__(*a, **kw)

    monkeypatch.setattr(model_cmd, "ConfigLoader", Spy)
    return calls


def _tty(monkeypatch, value: bool) -> None:
    import sys

    monkeypatch.setattr(sys.stdin, "isatty", lambda: value, raising=False)


def test_a_switch_off_a_terminal_decides_trust_and_records_the_intent(components_home, monkeypatch):
    _seed(components_home)
    _unreachable(monkeypatch)
    decided = _spy_decide(monkeypatch)
    loaders = _spy_loader(monkeypatch)

    result = runner.invoke(app, ["model", "flash-next", "--config-dir", str(components_home)])

    assert result.exit_code == 0, result.output
    assert [(c["ask"], c["launching"]) for c in decided] == [(False, False)]
    assert loaders and loaders[0]["machine_withheld"] == {}
    overlay = _overlay(components_home)
    assert overlay["pending_preset"] == "flash-next"
    assert overlay["active_endpoint"]["name"] == "flash-next"
    last = [ln for ln in result.output.splitlines() if ln.strip()][-1]
    assert last == "Selected flash-next — the next start serves it"


def test_the_next_ladder_serves_the_selected_record_on_the_pending_rung(components_home, monkeypatch):
    _seed(components_home)
    _unreachable(monkeypatch)
    result = runner.invoke(app, ["model", "flash-next", "--config-dir", str(components_home)])
    assert result.exit_code == 0, result.output

    harness = ConfigLoader(config_dir=str(components_home)).load_harness()
    serving = model_ops.resolve_serving(harness, MachineTrust(), None)

    assert isinstance(serving, model_ops.Serving)
    assert serving.record.name == "flash-next"
    assert serving.rung == model_ops.RUNG_PENDING


# --- the trust matrix (T-13-18, T-13-19, T-13-20) ---

WITHHELD_HINT = "run `localharness start` on a terminal to review them"


def _record_then_change_peer_model(home) -> None:
    """The machine adopted its files, then the peer's launch model changed: an uncovered entry."""
    trust.record_machine(home, trust.machine_snapshot(home))
    cfg = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    cfg["extra_endpoints"][0]["lifecycle"]["model"] = "/m/changed.gguf"
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _overlay_with_selection(home) -> bytes:
    (home / "overrides.yaml").write_text(yaml.safe_dump({
        "pending_preset": "gemma",
        "active_endpoint": {"name": "flash-next", "base_url": "http://127.0.0.1:8082/v1",
                            "provider_type": "llamacpp", "model": "flash-next"},
    }), encoding="utf-8")
    return (home / "overrides.yaml").read_bytes()


def test_a_piped_y_adopts_nothing_and_a_withheld_target_exits_2(components_home, monkeypatch):
    _seed(components_home)
    _unreachable(monkeypatch)
    _record_then_change_peer_model(components_home)
    held = _overlay_with_selection(components_home)
    recorded = trust.recorded_machine(components_home)
    decided = _spy_decide(monkeypatch)

    result = runner.invoke(
        app, ["model", "flash-next", "--config-dir", str(components_home)], input="y\n")

    assert result.exit_code == 2, result.output
    assert [c["ask"] for c in decided] == [False]
    assert trust.recorded_machine(components_home) == recorded
    assert result.output.count(WITHHELD_HINT) == 1
    assert (components_home / "overrides.yaml").read_bytes() == held


def test_an_eof_records_no_no(components_home, monkeypatch):
    _seed(components_home)
    _unreachable(monkeypatch)
    _record_then_change_peer_model(components_home)
    held = _overlay_with_selection(components_home)
    recorded = trust.recorded_machine(components_home)
    declined_before = set(workspace._DECLINED)

    result = runner.invoke(
        app, ["model", "flash-next", "--config-dir", str(components_home)], input="")

    assert result.exit_code == 2, result.output
    assert set(workspace._DECLINED) == declined_before
    assert trust.recorded_machine(components_home) == recorded
    assert (components_home / "overrides.yaml").read_bytes() == held


def test_a_withheld_primary_is_refused_with_nothing_cleared(components_home, monkeypatch):
    _seed(components_home)
    _unreachable(monkeypatch)
    trust.record_machine(components_home, trust.machine_snapshot(components_home))
    cfg = yaml.safe_load((components_home / "config.yaml").read_text(encoding="utf-8"))
    cfg["server"]["model"] = "/m/changed.gguf"
    (components_home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    held = _overlay_with_selection(components_home)

    result = runner.invoke(
        app, ["model", "qwen-27b", "--config-dir", str(components_home)], input="y\n")

    assert result.exit_code == 2, result.output
    assert result.output.count(WITHHELD_HINT) == 1
    assert (components_home / "overrides.yaml").read_bytes() == held


def test_the_question_is_asked_on_a_terminal_only(components_home, monkeypatch):
    """`ask` is `_is_local_operator()`: a tty and no agent-command marker, never an argument."""
    import types

    _seed(components_home)
    _unreachable(monkeypatch)
    seen: list[bool] = []

    def decide(global_dir, **kw):
        seen.append(kw["ask"])
        return MachineTrust()

    monkeypatch.setattr(model_cmd, "decide_machine_trust", decide)

    class Tty:
        def isatty(self):
            return True

    # CliRunner swaps sys.stdin during the call, so the terminal is stood in at the module's `sys`.
    monkeypatch.setattr(model_cmd, "sys", types.SimpleNamespace(stdin=Tty()))
    monkeypatch.delenv("LOCALHARNESS_COMMIT", raising=False)
    assert runner.invoke(
        app, ["model", "flash-next", "--config-dir", str(components_home)]).exit_code == 0
    monkeypatch.setenv("LOCALHARNESS_COMMIT", "abc")
    assert runner.invoke(
        app, ["model", "flash-next", "--config-dir", str(components_home)]).exit_code == 0

    assert seen == [True, False]


def test_list_and_download_decide_nothing(components_home, monkeypatch):
    _seed(components_home)
    monkeypatch.setattr(model_ops, "list_live_models",
                        lambda *a, **k: (["qwen-27b"], True), raising=False)
    monkeypatch.setattr(model_cmd, "_download", lambda *a, **k: None)
    decided = _spy_decide(monkeypatch)

    assert runner.invoke(app, ["model", "--config-dir", str(components_home)]).exit_code == 0
    assert runner.invoke(
        app, ["model", "--download", "org/x", "--config-dir", str(components_home)]).exit_code == 0

    assert decided == []


def test_an_ambiguous_prefix_exits_2_and_names_both(components_home, monkeypatch):
    _seed(components_home, peers=[_peer("qwen-a", port=8082), _peer("qwen-b", port=8083)],
          provider={"provider_type": "llamacpp", "base_url": "http://localhost:8081/v1",
                    "default_model": "gemma", "available_models": ["gemma"]},
          server={"runtime": "llamacpp", "binary": "/x/llama-server", "model": "/m/a.gguf",
                  "served_name": "gemma", "port": 8081})
    _unreachable(monkeypatch)

    result = runner.invoke(app, ["model", "qwen", "--config-dir", str(components_home)])

    assert result.exit_code == 2, result.output
    flat = result.output.replace("\n", " ")
    assert "ambiguous" in flat and "qwen-a" in flat and "qwen-b" in flat
    assert not (components_home / "overrides.yaml").exists()


def test_an_unknown_name_with_a_reachable_server_keeps_todays_error(components_home, monkeypatch):
    _seed(components_home)
    monkeypatch.setattr(model_ops, "list_live_models",
                        lambda *a, **k: (["qwen-27b", "other"], True), raising=False)

    result = runner.invoke(app, ["model", "ghost", "--config-dir", str(components_home)])

    assert result.exit_code == 2, result.output
    assert "unknown model" in result.output.replace("\n", " ")
    assert not (components_home / "overrides.yaml").exists()


def test_a_number_keeps_its_list_meaning(components_home, monkeypatch):
    _seed(components_home)
    monkeypatch.setattr(model_ops, "list_live_models",
                        lambda *a, **k: (["other-served", "qwen-27b"], True), raising=False)

    result = runner.invoke(app, ["model", "1", "--config-dir", str(components_home)])

    assert result.exit_code == 0, result.output
    overlay = _overlay(components_home)
    assert overlay["provider"]["default_model"] == "other-served"
    assert "pending_preset" not in overlay


def test_the_primary_swap_back_clears_the_peer_and_writes_the_intent(components_home, monkeypatch):
    _seed(components_home)
    _unreachable(monkeypatch)
    (components_home / "overrides.yaml").write_text(yaml.safe_dump({"active_endpoint": {
        "name": "flash-next", "base_url": "http://127.0.0.1:8082/v1",
        "provider_type": "llamacpp", "model": "flash-next"}}), encoding="utf-8")

    result = runner.invoke(app, ["model", "qwen-27b", "--config-dir", str(components_home)])

    assert result.exit_code == 0, result.output
    overlay = load_overlay(components_home / "overrides.yaml")
    assert "active_endpoint" not in overlay
    assert overlay["pending_preset"] == "qwen-27b"
    assert "Selected qwen-27b — the next start serves it" in result.output.replace("\n", " ")
