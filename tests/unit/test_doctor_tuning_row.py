"""doctor's `Tuning:` rows (DOCTOR-01, spec 20 §6.5): what each launchable record's next launch derives.

The world is `test_serve_cmd.serve_world` (a recorded machine, a Swift llama.cpp primary, the box faked at the
autosense seams). The rows come from the same sense + plan `activate` runs; nothing is prepared, started,
decided or written. No process starts and no request leaves the box: doctor's httpx is a mock.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from localharness.agent.context import WindowProbe
from localharness.cli import doctor_cmd, workspace
from localharness.cli.app import app
from localharness.config.loader import ConfigLoader
from localharness.provider import lifecycle, server, speed_stats
from tests.unit.gguf_support import write_gguf
from tests.unit.test_doctor_cmd import (  # noqa: F401  (autouse fixtures, imported for their effect)
    _models_resp,
    _route_the_window_probe,
    _stub_token_counter,
)
from tests.unit.test_serve_cmd import attach_peer, llama_peer, serve_world
from tests.unit.test_tuning_derive import FLASH_NEXT

runner = CliRunner()
ROWS = re.compile(r"^i\s+(Tuning: .*)$", re.MULTILINE)


@pytest.fixture
def mock_httpx(monkeypatch):
    mock = MagicMock()
    mock.get.return_value = _models_resp({"data": [{"id": "swift"}]})
    monkeypatch.setattr(doctor_cmd, "httpx", mock)
    return mock


@pytest.fixture
def world(tmp_path, monkeypatch, mock_httpx):
    return serve_world(tmp_path, monkeypatch)


def _doctor(world):
    with patch("localharness.agent.context.probe_window", return_value=WindowProbe(None, None)):
        return runner.invoke(app, ["doctor", "--config-dir", str(world.g)])


def _rows(result) -> list[str]:
    return ROWS.findall(result.output)


def _flash(world, name: str = "flash-next", port: int = 8082) -> dict:
    peer = llama_peer(name, port)
    peer["lifecycle"]["model"] = str(write_gguf(world.tmp / f"{name}.gguf", FLASH_NEXT))
    return peer


def _fitted(world, index: int | None = None) -> int:
    harness = ConfigLoader(config_dir=world.g, local_config_dir=None).load_harness()
    spec = harness.server if index is None else harness.extra_endpoints[index].lifecycle
    plan = lifecycle.plan_launch(
        spec, lifecycle.sense_launch(spec, world.g, harness.autosense), server.launch_decision(spec, False),
        autosense=harness.autosense, api_key_held=False)
    return plan.full_fit


def _files(root: Path) -> dict[str, bytes]:
    return {str(p): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_the_swift_primary_and_a_peer_get_a_row_each_and_an_attach_only_peer_gets_none(world):
    world.edit_config(lambda d: d["extra_endpoints"].extend([_flash(world), attach_peer("attached")]))
    world.rerecord()

    result = _doctor(world)

    assert _rows(result) == [
        f"Tuning: swift — kv q8_0 · mtp (1 layer) · -c {_fitted(world)} (fit)",
        f"Tuning: flash-next — kv q8_0 · -c {_fitted(world, 0)} (fit)"]
    out = result.output
    assert out.index("LLM endpoint reachable") < out.index("Tuning: swift") < out.index("Tuning: flash-next")
    assert out.index("Tuning: flash-next") < out.index("Config directory writable")


def test_the_rows_prepare_nothing_start_nothing_decide_nothing_and_write_nothing(world, monkeypatch):
    def raises(*a, **kw):
        raise AssertionError("a Tuning row must not call this")

    monkeypatch.setattr(server, "prepare_config_ini_home", raises)
    monkeypatch.setattr(speed_stats, "record_retry", raises)
    monkeypatch.setattr(workspace, "decide_machine_trust", raises)
    monkeypatch.setattr(subprocess, "Popen", raises)
    monkeypatch.setattr(doctor_cmd, "_llama_server_linked_backends", lambda binary: None)  # doctor's own ldd probe
    before = _files(world.g)

    result = _doctor(world)

    assert len(_rows(result)) == 1
    assert _files(world.g) == before
    assert not (world.g / "xdg").exists()
