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
