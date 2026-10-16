"""`localharness model --autostart status` and the doctor row — what autostart is doing, read and never written
(spec 20 §9.2 `status`, §9.3; D-01, D-02, D-11, D-13, D-18; AUTO-01, AUTO-05, AUTO-08).

The world is plan `on`'s (`tests/unit/test_autostart_on_off.py`): a global dir with a llama.cpp record, a fake
home, the unit file under a redirected `XDG_CONFIG_HOME`, a linger directory, and a fake service manager. A case
turns autostart on for real (so the unit file, its `# env` block and the machine record are the ones `on` writes),
then answers the manager's `show` with the state it wants `status` to read.

`status` runs anywhere: every case calls it with the REAL local-operator test, no terminal and the agent-command
marker set. It reads the manager, the unit file, the global config, linger and the stores once each, and writes
nothing; a manager verb or a journal read from it fails the case.

Nothing here runs a service manager, a login manager or the journal program, starts a unit or a server, or dials
a port.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from localharness.cli import autostart, model_cmd, workspace
from localharness.cli.app import app
from localharness.provider import units
from tests.unit.test_autostart_on_off import (  # noqa: F401 — `world` is the fixture
    DISPLAY, REAL_OPERATOR_TEST, SERVED, World, stdout_lines, world,
)
from tests.unit.units_support import fake_manager  # noqa: F401 — the fixture

FIRST = f"Autostart: on — {DISPLAY} (enabled, active, NRestarts 0, MainPID 4242)"
OFF_LINE = "Autostart: off — localharness model --autostart on serves the last choice at login"
SHELL_PATH = "/usr/local/bin:/usr/bin:/bin"
EMPTY = object()


def says(world: World, *, active: str = "active", file_state: str = "enabled", restarts: int = 0,
         exit_status: int = 0, pid: int = 4242, invocation: str = "abc", result: str = "success",
         fragment: object = None, drop_ins: str = "") -> None:
    """Answer the manager's state `show` with these fields (the unit loaded from its own file by default)."""
    path = units.harness_unit_path() if fragment is None else ("" if fragment is EMPTY else fragment)
    text = "\n".join([
        f"FragmentPath={path}", f"DropInPaths={drop_ins}", f"ActiveState={active}",
        f"UnitFileState={file_state}", f"NRestarts={restarts}", f"ExecMainStatus={exit_status}",
        f"MainPID={pid}", f"InvocationID={invocation}", f"Result={result}", ""])
    world.manager.answer(lambda argv: argv[:3] == ("systemctl", "--user", "show") and "NRestarts" in argv[4],
                         out=text)


def installed(world: World, **state) -> World:
    """Turn autostart on for real, then let the manager say `state`; forget the calls `on` made."""
    first = world.run()
    assert first.exit_code == 0, (first.output, first.exception)
    says(world, **state)
    world.manager.calls.clear()
    return world


def anywhere(world: World) -> None:
    """The REAL local-operator test with no terminal and the agent-command marker set: a gate would refuse."""
    world.mp.setattr(model_cmd, "_is_local_operator", REAL_OPERATOR_TEST)
    world.mp.setattr(model_cmd, "sys", SimpleNamespace(stdin=SimpleNamespace(isatty=lambda: False)))
    world.mp.setenv("LOCALHARNESS_COMMIT", "1")


def status(world: World):
    return world.run(mode="status")


def tree(root: Path) -> dict[str, tuple[bytes, int]]:
    """Every file under ``root`` as bytes and mtime, for the 'nothing was written' comparisons."""
    found = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            found[str(path)] = (path.read_bytes(), path.stat().st_mtime_ns)
    return found


# ===================================================================== the tracer


def test_status_off_a_terminal_prints_state_target_linger_and_the_captured_path(world):
    installed(world)
    anywhere(world)

    result = status(world)

    assert result.exit_code == 0, (result.output, result.exception)
    assert stdout_lines(result) == [
        FIRST, f"serves: {SERVED}", "linger: on", f"PATH captured: {SHELL_PATH}"]
    assert result.stderr == ""


def test_status_writes_nothing_and_decides_no_trust(world):
    installed(world)
    anywhere(world)

    def decided(*a, **kw):
        raise AssertionError("status decided machine trust")

    world.mp.setattr(model_cmd, "decide_machine_trust", decided)
    world.mp.setattr(workspace, "decide_machine_trust", decided)
    before_g, before_home = tree(world.g), tree(world.home)

    result = status(world)

    assert result.exit_code == 0, (result.output, result.exception)
    assert tree(world.g) == before_g and tree(world.home) == before_home
    assert world.unit_path.read_text(encoding="utf-8") == world.body()


def test_status_asks_the_manager_only_to_read(world):
    installed(world)
    anywhere(world)

    status(world)

    assert world.manager.calls and {c[:2] for c in world.manager.calls} == {("systemctl", "--user")}
    assert {c[2] for c in world.manager.calls} <= {"show", "list-unit-files"}
    assert world.verbs == []


def test_status_is_read_only_in_source_too():
    source = Path(autostart.__file__).read_text(encoding="utf-8")
    body = source[source.index("def _gather("):]

    for forbidden in ("decide_machine_trust", "adopt_entries", "forget_entries", "write_target", "remove_target",
                      "remove_marker", "write_marker", "write_pending_preset", "clear_gone", "systemctl_user(",
                      "journal_tail", "read_journal"):
        assert forbidden not in body, forbidden


def test_status_with_no_unit_and_no_loaded_fragment_says_off_and_nothing_else(world):
    says(world, fragment=EMPTY, active="inactive", file_state="", pid=0, invocation="")
    anywhere(world)

    result = status(world)

    assert result.exit_code == 0, (result.output, result.exception)
    assert stdout_lines(result) == [OFF_LINE]
    assert not world.unit_path.exists()


def test_status_never_asks_for_a_terminal(world):
    installed(world)
    anywhere(world)

    result = status(world)

    assert result.exit_code == 0
    assert "terminal" not in result.output


def test_doctor_row_is_the_first_status_line(world):
    installed(world)

    assert autostart.doctor_row(world.g) == FIRST == autostart.status_lines(world.g)[0]


def test_doctor_row_is_the_off_line_with_no_unit(world):
    says(world, fragment=EMPTY, active="inactive", file_state="", pid=0, invocation="")

    assert autostart.doctor_row(world.g) == OFF_LINE


def test_main_pid_shows_a_dash_when_no_process(world):
    installed(world, active="inactive", file_state="enabled", pid=0, invocation="")

    assert autostart.doctor_row(world.g) == (
        f"Autostart: on — {DISPLAY} (enabled, inactive, NRestarts 0, MainPID -)")


def test_an_unreadable_config_still_answers_with_the_unit_line(world):
    installed(world)
    (world.g / "config.yaml").write_text("provider: [not, a, mapping", encoding="utf-8")
    anywhere(world)

    result = status(world)

    assert result.exit_code == 0, (result.output, result.exception)
    lines = stdout_lines(result)
    assert lines[0] == FIRST and lines[1].startswith("serves: (the config could not be loaded: ")
    assert "linger: on" in lines


def test_the_option_help_lists_all_three_modes():
    result = CliRunner().invoke(app, ["model", "--help"])

    assert "on | off | status" in " ".join(result.stdout.split())

