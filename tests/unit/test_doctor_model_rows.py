"""doctor's `Sampling:`, `Autostart:` and `Models:` rows (DOCTOR-02, DOCTOR-03, DOCTOR-04, spec 20 §7, §9.2-9.3, §11.4).

The world is the autostart status world (a global dir with a llama.cpp record, a fake home, a fake service
manager), with doctor's endpoint faked as `test_doctor_window_row` does. The rows read the config, the library
file and the manager's state `show`; nothing is started, prepared or written, and no request leaves the box.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from localharness.agent.context import WindowProbe
from localharness.cli import doctor_cmd
from localharness.cli.app import app
from localharness.provider import model_library
from localharness.agent.gguf_header import FileKind
from tests.unit.test_autostart_on_off import SERVED, on_world, world  # noqa: F401 — `world` is the fixture
from tests.unit.test_autostart_status import EMPTY, OFF_LINE, installed, put_marker, says, tree
from tests.unit.test_doctor_cmd import (  # noqa: F401  (autouse fixtures, imported for their effect)
    _models_resp,
    _route_the_window_probe,
    _stub_token_counter,
)
from tests.unit.test_serve_cmd import llama_peer
from tests.unit.units_support import fake_manager  # noqa: F401 — the fixture

runner = CliRunner()
ROWS = re.compile(r"^i\s+((?:Sampling|Autostart|Models): .*)$", re.MULTILINE)
SETUP = "localharness model --setup lists them"


@pytest.fixture
def mock_httpx(monkeypatch):
    mock = MagicMock()
    mock.get.return_value = _models_resp({"data": [{"id": SERVED}]})
    monkeypatch.setattr(doctor_cmd, "httpx", mock)
    return mock


@pytest.fixture
def endpoint_down(mock_httpx):
    mock_httpx.ConnectError, mock_httpx.TimeoutException = httpx.ConnectError, httpx.TimeoutException
    mock_httpx.get.side_effect = httpx.ConnectError("down")


def doctor(world, *, own: float | None = None):
    with patch("localharness.agent.context.probe_window", return_value=WindowProbe(None, None, own)):
        return runner.invoke(app, ["doctor", "--config-dir", str(world.g)])


def rows(result) -> dict[str, str]:
    found = ROWS.findall(result.output)
    return {line.split(":", 1)[0]: line for line in found}


def configure(world, **org) -> None:
    path = world.g / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["org"].update(org)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")


def library_of(world, *, seen: int = 0, declined: int = 0) -> list[Path]:
    """A library file holding `seen` new and `declined` declined models, each file on disk."""
    made, entries = [], {}
    for i in range(seen + declined):
        path = world.tmp / f"found-{i}.gguf"
        path.write_bytes(b"x")
        made.append(path)
        entries[os.path.realpath(path)] = model_library.Entry("2026-01-01", 1, FileKind.MODEL, i >= seen)
    model_library.Library(entries).save(model_library.library_path(world.g))
    return made


# ===================================================================== the tracer


def test_doctor_prints_sampling_autostart_and_models_after_the_tuning_rows(
        tmp_path, monkeypatch, fake_manager, mock_httpx):
    made = on_world(tmp_path, monkeypatch, fake_manager, peers=[llama_peer("flash-next", 8081)])
    configure(made, default_temperature=0.6)
    library_of(made, seen=1, declined=1)
    says(made, fragment=EMPTY, active="inactive", file_state="", pid=0, invocation="")

    result = doctor(made, own=1.0)

    assert result.exit_code == 0, (result.output, result.exception)
    assert list(rows(result).values()) == [
        "Sampling: temperature 0.6 from org.default_temperature "
        "(the model's own is 1.0 — unset org.default_temperature to use it)",
        OFF_LINE,
        f"Models: 2 set up · 1 found, not set up · 1 declined — {SETUP}"]
    out = result.output
    assert out.index("Tuning: ") < out.index("Sampling: ") < out.index("Autostart: ") < out.index(
        "Models: ") < out.index("Config directory writable")


def test_doctor_runs_no_manager_verb_and_writes_nothing(tmp_path, monkeypatch, fake_manager, mock_httpx):
    made = on_world(tmp_path, monkeypatch, fake_manager, peers=[llama_peer("flash-next", 8081)])
    library_of(made, seen=1, declined=1)
    installed(made)
    put_marker(made)
    before = tree(made.g), tree(made.home)

    result = doctor(made, own=1.0)

    assert result.exit_code == 0, (result.output, result.exception)
    assert made.verbs == []
    assert not [c for c in made.manager.calls if c[0] == "journalctl"]
    assert (tree(made.g), tree(made.home)) == before
