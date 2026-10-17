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
from localharness.cli import autostart, doctor_cmd, library
from localharness.cli.app import app
from localharness.provider import model_library, units
from localharness.provider.gguf_header import FileKind
from localharness.provider.server import server_dir
from tests.unit.procfs_support import fake_proc  # noqa: F401 — the fixture
from tests.unit.test_autostart_on_off import SERVED, on_world, world  # noqa: F401 — `world` is the fixture
from tests.unit.test_autostart_status import BASE, EMPTY, OFF_LINE, installed, put_marker, says, tree
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


# ===================================================================== Sampling


def test_a_rung_equal_to_the_servers_own_is_the_plain_form(world, mock_httpx):
    configure(world, default_temperature=0.6)

    assert rows(doctor(world, own=0.6))["Sampling"] == "Sampling: temperature 0.6 from org.default_temperature"


def test_a_rung_with_the_endpoint_down_is_the_plain_form_and_the_row_still_prints(world, endpoint_down):
    configure(world, default_temperature=0.6)

    result = doctor(world)

    assert rows(result)["Sampling"] == "Sampling: temperature 0.6 from org.default_temperature"
    assert "LLM endpoint unreachable" in result.output


def test_no_rung_and_a_server_that_said_is_the_models_own(world, mock_httpx):
    assert rows(doctor(world, own=1.0))["Sampling"] == "Sampling: the model's own (temperature 1.0)"


def test_no_rung_and_the_endpoint_down_sends_no_temperature(world, endpoint_down):
    assert rows(doctor(world))["Sampling"] == "Sampling: the model's own — no temperature is sent"


def test_an_agent_file_names_its_own_rung(world, mock_httpx):
    agents = world.g / "agents"
    agents.mkdir()
    (agents / "orchestrator.yaml").write_text(
        yaml.safe_dump({"name": "orchestrator", "role": "root", "temperature": 0.2}), encoding="utf-8")

    assert rows(doctor(world, own=1.0))["Sampling"] == (
        "Sampling: temperature 0.2 from temperature in agents/orchestrator.yaml "
        "(the model's own is 1.0 — unset temperature in agents/orchestrator.yaml to use it)")


# ===================================================================== Autostart


def test_the_row_joins_every_status_line_with_the_live_marker(world, mock_httpx):
    installed(world)
    put_marker(world)

    row = rows(doctor(world))["Autostart"]

    assert row == "; ".join([*BASE, f"serving {SERVED} at the retry shape since boot — mtp off, -c halved"])


def test_a_marker_of_another_run_is_no_clause(world, mock_httpx):
    installed(world)
    put_marker(world, invocation="old-inv")

    assert rows(doctor(world))["Autostart"] == "; ".join(BASE)


def test_a_stopped_unit_ends_the_row_with_its_exit(world, mock_httpx):
    installed(world)
    says(world, active="failed", exit_status=75, pid=0, invocation="")

    row = rows(doctor(world))["Autostart"]

    lines = autostart.status_lines(world.g)
    assert lines[-1].startswith("stopped: ") and row.endswith("; " + lines[-1])
    assert row == "; ".join(lines)


def test_an_in_flight_request_outranking_a_pending_preset_takes_its_place(tmp_path, monkeypatch, fake_manager,
                                                                         fake_proc, mock_httpx):  # noqa: F811
    from localharness.cli import model_ops
    made = on_world(tmp_path, monkeypatch, fake_manager,
                    peers=[llama_peer("flash-next", 8081), llama_peer("qwen-27b", 8082)])
    fake_proc.process(os.getpid(), argv=("python",), start_time=111, comm="python")
    installed(made)
    model_ops.write_pending_preset(made.harness(), "qwen-27b", config_dir=made.g)
    units.write_target(made.g, "flash-next", in_flight=True)

    row = rows(doctor(made))["Autostart"]

    assert f"in flight: flash-next (requested by pid {os.getpid()}) — outranks pending_preset qwen-27b" in row
    assert row == "; ".join(autostart.status_lines(made.g))


def test_a_second_launcher_on_a_records_port_has_its_line(world, mock_httpx):
    from tests.unit.test_units_manager import answer_scan, show_block, unit_file
    installed(world)
    llm = unit_file(world.tmp, "llm.service", "/opt/llama-server -m /m.gguf --port 8080")
    answer_scan(world.manager, ["llm.service enabled"], [show_block("llm.service", llm)])

    row = rows(doctor(world))["Autostart"]

    assert "llm.service (enabled) also launches a server on port 8080" in row


@pytest.mark.parametrize("make", ["off", "installed", "stopped"])
def test_the_doctor_row_is_the_joined_status_lines(world, make):
    if make == "off":
        says(world, fragment=EMPTY, active="inactive", file_state="", pid=0, invocation="")
    else:
        installed(world)
        if make == "stopped":
            says(world, active="failed", exit_status=75, pid=0, invocation="")

    assert autostart.doctor_row(world.g) == "; ".join(autostart.status_lines(world.g))


# ===================================================================== Models


@pytest.mark.parametrize("seen, declined, row", [
    (3, 0, f"Models: 1 set up · 3 found, not set up — {SETUP}"),
    (0, 0, f"Models: 1 set up — {SETUP}"),
    (0, 2, f"Models: 1 set up · 2 declined — {SETUP}"),
])
def test_zero_clauses_are_dropped_and_the_order_holds(world, mock_httpx, seen, declined, row):
    library_of(world, seen=seen, declined=declined)

    assert rows(doctor(world))["Models"] == row


def test_a_box_with_nothing_set_up_or_found_says_so(world, mock_httpx):
    configure(world)
    raw = yaml.safe_load((world.g / "config.yaml").read_text(encoding="utf-8"))
    del raw["server"]
    (world.g / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")

    assert library.counts(world.g) == library.Counts(0, 0, 0)
    assert rows(doctor(world))["Models"] == doctor_cmd.MODELS_NONE_ROW


def test_a_model_reached_through_a_symlink_counts_once(world, mock_httpx):
    (path,) = library_of(world, seen=1)
    root = world.tmp / "models"
    root.mkdir()
    (root / "alias.gguf").symlink_to(path)
    (root / "real.gguf").write_bytes(b"x")

    held = library.counts(world.g)

    assert held.found == 1
    assert rows(doctor(world))["Models"] == f"Models: 1 set up · 1 found, not set up — {SETUP}"


def test_doctor_without_a_config_stops_before_any_row(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALHARNESS_DIR", str(tmp_path / "empty"))
    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path / "empty")])

    assert rows(result) == {}


def test_the_boot_marker_and_the_unit_target_are_left_alone(world, mock_httpx):
    installed(world)
    kept = put_marker(world)
    units.write_target(world.g, "x", in_flight=False)
    target = server_dir(world.g) / units.TARGET_NAME
    target_bytes = target.read_bytes()

    doctor(world)

    assert (server_dir(world.g) / units.BOOT_MARKER_NAME).read_bytes() == kept
    assert target.read_bytes() == target_bytes
