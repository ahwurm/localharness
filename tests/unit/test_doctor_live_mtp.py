"""doctor says whether a record's verified live process drafts from its own argv; /props is never read for it."""
from __future__ import annotations

import re
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from localharness.agent.context import WindowProbe
from localharness.cli import doctor_cmd
from localharness.cli.app import app
from localharness.provider import lifecycle, procfs
from tests.unit.test_doctor_cmd import (  # noqa: F401  (autouse fixtures, imported for their effect)
    _models_resp,
    _route_the_window_probe,
    _stub_token_counter,
)
from tests.unit.test_doctor_tuning_row import _flash
from tests.unit.test_serve_cmd import serve_world

runner = CliRunner()
ROWS = re.compile(r"^i\s+(Tuning: .*)$", re.MULTILINE)
ARGV_ON = ("llama-server", "-m", "m.gguf", "--spec-type", "draft-mtp")
ARGV_OFF = ("llama-server", "-m", "m.gguf")


@pytest.fixture
def mock_httpx(monkeypatch):
    from unittest.mock import MagicMock

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


def _live(monkeypatch, *, holder: str, outcome):
    """A verified launch of the record whose model path ends `holder`, read as `outcome`."""
    def own_launch(config_dir, specs):
        hit = any(s is not None and str(s.model).endswith(holder) for s in specs)
        return procfs.Launch(4242, 1, 8080, "spawned", True) if hit else None

    monkeypatch.setattr(lifecycle, "own_launch", own_launch)
    monkeypatch.setattr(procfs, "read_process", lambda pid: outcome)


def _record(argv):
    return procfs.ProcRecord(4242, 1000, argv, "/x/llama-server", None, 1, None, None, ())


def _swift_row(result) -> str:
    return next(r for r in ROWS.findall(result.output) if r.startswith("Tuning: swift"))


def test_a_serving_process_whose_argv_carries_draft_mtp_is_serving_mtp_on_whatever_props_reports(
        world, monkeypatch, mock_httpx):
    mock_httpx.get.return_value = _models_resp({"data": [{"id": "swift"}], "speculative": {"types": "none"}})
    _live(monkeypatch, holder="model.gguf", outcome=_record(ARGV_ON))

    assert _swift_row(_doctor(world)).endswith(" · serving now: mtp on")


def test_a_serving_process_whose_argv_has_no_spec_type_is_serving_mtp_off(world, monkeypatch):
    _live(monkeypatch, holder="model.gguf", outcome=_record(ARGV_OFF))

    assert _swift_row(_doctor(world)).endswith(" · serving now: mtp off")


def test_no_verified_launch_of_the_record_says_nothing_and_another_records_process_lends_none(world, monkeypatch):
    world.edit_config(lambda d: d["extra_endpoints"].append(_flash(world)))
    world.rerecord()
    _live(monkeypatch, holder="flash-next.gguf", outcome=_record(ARGV_ON))

    rows = ROWS.findall(_doctor(world).output)

    assert "serving now" not in next(r for r in rows if r.startswith("Tuning: swift"))
    assert next(r for r in rows if r.startswith("Tuning: flash-next")).endswith(" · serving now: mtp on")


def test_a_refused_process_read_says_nothing(world, monkeypatch):
    _live(monkeypatch, holder="model.gguf", outcome=procfs.ProcRefusal("is gone"))

    assert "serving now" not in _swift_row(_doctor(world))
