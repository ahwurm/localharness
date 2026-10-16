"""`localharness model --autostart off` — the gate, the identity re-check, disable, remove, reload, prune (spec 20
§9.1, §9.2 `off`; D-01, D-11, D-13; AUTO-01, AUTO-04..AUTO-06).

The world is plan `on`'s (`tests/unit/test_autostart_on_off.py`): a global dir with a llama.cpp record, a fake home,
the unit file's directory under a redirected `XDG_CONFIG_HOME`, a terminal operator and a fake service manager.
Each case turns autostart on first, so `off` has the unit `on` really wrote to undo, then forgets the calls `on`
made. An argv nobody answered is the runner's None, so a verb `off` did not mean to issue fails the case.

Nothing here runs a service manager or a login manager, starts a unit or a server, or dials a port.
"""
from __future__ import annotations

import fcntl
import os

import pytest
from typer.testing import CliRunner

from localharness.cli import autostart, model_cmd, workspace
from localharness.cli.app import app
from localharness.cli.workspace import decide_machine_trust
from localharness.config import trust
from localharness.provider import server, units
from tests.unit.test_autostart_on_off import (  # noqa: F401 — `world` is the fixture
    DISPLAY, REAL_OPERATOR_TEST, RELOAD, STATE_SHOW, UNIT, World, lock_is_free, on_world, stdout_lines, world,
)
from tests.unit.units_support import fake_manager  # noqa: F401 — the fixture

RESET_FAILED = ("systemctl", "--user", "reset-failed", UNIT)
DISABLE_NOW = ("systemctl", "--user", "disable", "--now", UNIT)
OFF_TERMINAL_LINE = "autostart is set from a terminal: localharness model --autostart off"
EXPECTED_VERBS = [STATE_SHOW, STATE_SHOW, RESET_FAILED, DISABLE_NOW, RELOAD]
OTHER_FRAGMENT = "/etc/xdg/systemd/user/localharness-llm.service"


def turned_on(world: World) -> World:
    """Run `on` for real in the world, then answer the verbs `off` issues and forget `on`'s calls."""
    first = world.run()
    assert first.exit_code == 0, (first.output, first.exception)
    world.manager.answer_prefix("systemctl", "--user", "reset-failed", out="")
    world.manager.answer_prefix("systemctl", "--user", "disable", out="")
    world.manager.calls.clear()
    return world


def state_calls(world: World) -> list[tuple[str, ...]]:
    return [c for c in world.manager.calls if c == STATE_SHOW]


# ===================================================================== the tracer


def test_off_after_on_rechecks_disables_removes_reloads_prunes_and_says_so(world):
    turned_on(world)
    target = units.read_target(world.g)
    assert trust.recorded_autostart_entry(world.g) is not None

    def decided(*a, **kw):
        raise AssertionError("off asked or decided machine trust")

    world.mp.setattr(model_cmd, "decide_machine_trust", decided)

    result = world.run(mode="off")

    assert result.exit_code == 0, (result.output, result.exception)
    assert world.manager.calls == EXPECTED_VERBS
    assert not world.unit_path.exists()
    assert trust.recorded_autostart_entry(world.g) is None
    assert stdout_lines(result) == [f"autostart off: removed {DISPLAY}; nothing starts at login"]
    assert result.stderr == ""
    assert units.read_target(world.g) == target  # R-14-26: the last unit target stays


def test_off_holds_the_launch_lock_for_the_verbs_and_frees_it(world):
    turned_on(world)
    seen = []
    real = world.manager

    def watching(argv, **kw):
        seen.append((tuple(argv), lock_is_free(world.g)))
        return real(argv, **kw)

    world.mp.setattr(units, "_run_manager", watching)

    result = world.run(mode="off")

    assert result.exit_code == 0
    held = {argv[2]: free for argv, free in seen if argv[2] != "show"}
    assert held == {"reset-failed": False, "disable": False, "daemon-reload": False}
    assert lock_is_free(world.g)


def test_off_prunes_only_the_autostart_entry(world):
    turned_on(world)
    before = trust.recorded_machine(world.g)
    others = [e for e in before["entries"] if e["name"] != trust.AUTOSTART_ENTRY_NAME]

    world.run(mode="off")

    after = trust.recorded_machine(world.g)
    assert after is not None and after["entries"] == others


def test_off_never_issues_a_verb_outside_the_closed_set(world):
    turned_on(world)

    world.run(mode="off")

    verbs = {c[2] for c in world.manager.calls}
    assert verbs <= {"show", "reset-failed", "disable", "daemon-reload"}


# ===================================================================== a foreign unit is left alone (D-11, D-13)


def _drop_in(world):
    world.manager_says(drop_ins="/home/u/.config/systemd/user/localharness-llm.service.d/x.conf")
    return "has drop-ins"


def _other_directory(world):
    world.manager_says(fragment=OTHER_FRAGMENT)
    return "FragmentPath"


def _edited_byte(world):
    path = world.unit_path
    path.write_text(path.read_text(encoding="utf-8").replace("Restart=", "Restart =", 1), encoding="utf-8")
    return "differs from what localharness writes"


@pytest.mark.parametrize("make_foreign", [_drop_in, _other_directory, _edited_byte],
                         ids=["a_drop_in", "another_directory", "an_edited_byte"])
def test_foreign_unit_is_left_as_it_is_and_named(world, make_foreign):
    turned_on(world)
    phrase = make_foreign(world)
    kept = world.unit_path.read_text(encoding="utf-8")
    entry = trust.recorded_autostart_entry(world.g)

    result = world.run(mode="off")

    assert result.exit_code == 1
    line = result.stderr.strip()
    assert line.startswith(f"autostart off: {DISPLAY} is not the unit localharness wrote (")
    assert phrase in line and line.endswith(") — left as it is")
    assert result.stdout == ""
    assert world.manager.calls == [STATE_SHOW]  # the state read, then no verb
    assert world.unit_path.read_text(encoding="utf-8") == kept
    assert trust.recorded_autostart_entry(world.g) == entry


def test_foreign_only_on_the_recheck_under_the_lock_stops_everything(world):
    turned_on(world)
    shows = []
    real = world.manager

    def drifting(argv, **kw):
        if tuple(argv) == STATE_SHOW:
            shows.append(1)
            if len(shows) == 2:  # the fresh read, taken under the lock, names a drop-in
                world.manager_says(drop_ins="/x/y.d/z.conf")
        return real(argv, **kw)

    world.mp.setattr(units, "_run_manager", drifting)

    result = world.run(mode="off")

    assert result.exit_code == 1
    assert "has drop-ins (/x/y.d/z.conf)" in result.stderr and "left as it is" in result.stderr
    assert world.manager.calls == [STATE_SHOW, STATE_SHOW]
    assert world.unit_path.exists() and trust.recorded_autostart_entry(world.g) is not None
    assert lock_is_free(world.g)


def test_foreign_a_manager_that_cannot_be_asked_removes_nothing(world):
    turned_on(world)
    world.manager.answer(lambda argv: argv[:3] == ("systemctl", "--user", "show"), out=None)

    result = world.run(mode="off")

    assert result.exit_code == 1
    assert "left as it is" in result.stderr
    assert world.unit_path.exists()
    assert world.manager.calls == [STATE_SHOW]


def test_foreign_a_file_that_is_a_link_is_not_followed_or_removed(world, tmp_path):
    turned_on(world)
    elsewhere = tmp_path / "elsewhere.service"
    elsewhere.write_text("[Unit]\n", encoding="utf-8")
    world.unit_path.unlink()
    world.unit_path.symlink_to(elsewhere)

    result = world.run(mode="off")

    assert result.exit_code == 1 and "left as it is" in result.stderr
    assert elsewhere.read_text(encoding="utf-8") == "[Unit]\n" and world.unit_path.is_symlink()
    assert world.manager.calls == [STATE_SHOW]


# ===================================================================== a file the manager never loaded


def _on_that_failed_at_the_reload(world: World) -> None:
    """`on` wrote the file and adopted its entry, then the reload failed: the manager has nothing loaded."""
    world.manager.answer(RELOAD, code=1, err="no user bus")
    first = world.run()
    assert first.exit_code == 1 and world.unit_path.exists()
    world.manager.answer(RELOAD, out="")
    world.manager.answer_prefix("systemctl", "--user", "disable", out="")
    world.manager.answer_prefix("systemctl", "--user", "reset-failed", out="")
    world.manager_says(fragment="")
    world.manager.calls.clear()


DISABLE = ("systemctl", "--user", "disable", UNIT)


def test_a_file_on_wrote_and_the_manager_never_loaded_is_disabled_removed_reloaded_and_pruned_without_a_stop(world):
    decide_machine_trust(world.g, ask=False)
    _on_that_failed_at_the_reload(world)
    assert trust.recorded_autostart_entry(world.g) is not None

    result = world.run(mode="off")

    assert result.exit_code == 0, (result.output, result.exception)
    # no reset-failed and no `--now` (nothing runs); a plain disable clears a boot link an earlier `on` left,
    # while the file it reads is still there
    assert world.manager.calls == [STATE_SHOW, STATE_SHOW, DISABLE, RELOAD]
    assert not world.unit_path.exists()
    assert trust.recorded_autostart_entry(world.g) is None
    assert stdout_lines(result) == [
        f"autostart off: removed {DISPLAY}; nothing starts at login"
        " (the service manager had not loaded it; its boot link was cleared)"]
    assert lock_is_free(world.g)


def test_an_unloaded_file_whose_boot_link_the_manager_will_not_clear_is_still_removed_and_says_a_link_may_remain(world):
    decide_machine_trust(world.g, ask=False)
    _on_that_failed_at_the_reload(world)
    world.manager.answer_prefix("systemctl", "--user", "disable", out="", code=1, err="Failed to disable unit")

    result = world.run(mode="off")

    assert result.exit_code == 0, (result.output, result.exception)
    assert not world.unit_path.exists() and trust.recorded_autostart_entry(world.g) is None
    assert world.manager.calls == [STATE_SHOW, STATE_SHOW, DISABLE, RELOAD]
    (line,) = stdout_lines(result)
    assert line.startswith(f"autostart off: removed {DISPLAY}; nothing starts at login")
    assert "a dangling link may remain" in line and "boot link was cleared" not in line


def test_an_unloaded_file_with_a_manager_that_cannot_run_the_disable_is_still_removed(world):
    decide_machine_trust(world.g, ask=False)
    _on_that_failed_at_the_reload(world)
    world.manager.answer_prefix("systemctl", "--user", "disable", out=None)

    result = world.run(mode="off")

    assert result.exit_code == 0, (result.output, result.exception)
    assert not world.unit_path.exists()
    assert "a dangling link may remain" in stdout_lines(result)[0]


def test_an_unloaded_file_that_is_not_byte_equal_to_the_render_stays_refused_as_foreign(world):
    decide_machine_trust(world.g, ask=False)
    _on_that_failed_at_the_reload(world)
    path = world.unit_path
    path.write_text(path.read_text(encoding="utf-8").replace("Restart=", "Restart =", 1), encoding="utf-8")
    kept = path.read_text(encoding="utf-8")
    entry = trust.recorded_autostart_entry(world.g)

    result = world.run(mode="off")

    assert result.exit_code == 1
    assert "FragmentPath is empty" in result.stderr and result.stderr.strip().endswith("— left as it is")
    assert world.manager.calls == [STATE_SHOW]
    assert path.read_text(encoding="utf-8") == kept and trust.recorded_autostart_entry(world.g) == entry


def test_an_unloaded_file_beside_a_drop_in_stays_refused_as_foreign(world):
    decide_machine_trust(world.g, ask=False)
    _on_that_failed_at_the_reload(world)
    world.manager_says(fragment="", drop_ins="/x/y.d/z.conf")

    result = world.run(mode="off")

    assert result.exit_code == 1 and world.unit_path.exists()
    assert world.manager.calls == [STATE_SHOW]


# ===================================================================== nothing to do


def test_nothing_to_do_when_there_is_no_file_and_nothing_loaded(world):
    world.manager_says(fragment="")
    world.manager.calls.clear()

    result = world.run(mode="off")

    assert result.exit_code == 0
    assert stdout_lines(result) == [f"autostart is off — {DISPLAY} does not exist"]
    assert result.stderr == ""
    assert world.manager.calls == [STATE_SHOW]


def test_off_twice_the_second_says_it_is_already_off(world):
    turned_on(world)
    assert world.run(mode="off").exit_code == 0
    world.manager_says(fragment="")
    world.manager.calls.clear()

    again = world.run(mode="off")

    assert again.exit_code == 0 and stdout_lines(again) == [f"autostart is off — {DISPLAY} does not exist"]
    assert world.manager.calls == [STATE_SHOW]


# ===================================================================== the gate (D-01, AUTO-01)


def test_gate_off_without_a_local_operator_is_refused_and_nothing_runs(world):
    turned_on(world)
    world.mp.setattr(model_cmd, "local_operator", lambda no_input: False)

    result = world.run(mode="off")

    assert result.exit_code == 2
    assert result.stderr.strip() == OFF_TERMINAL_LINE and result.stdout == ""
    assert world.manager.calls == []
    assert world.unit_path.exists() and trust.recorded_autostart_entry(world.g) is not None


def test_gate_the_agent_command_marker_refuses_off_even_on_a_tty(world):
    turned_on(world)
    world.mp.setattr(model_cmd, "local_operator", REAL_OPERATOR_TEST)
    world.mp.setattr(workspace, "_stdin_is_a_terminal", lambda: True)
    world.mp.setattr(workspace, "_stdout_is_a_terminal", lambda: True)
    world.mp.setenv("LOCALHARNESS_COMMIT", "1")

    result = world.run(mode="off")

    assert result.exit_code == 2 and result.stderr.strip() == OFF_TERMINAL_LINE
    assert world.manager.calls == [] and world.unit_path.exists()


def test_gate_off_takes_no_model_name_beside_it(world):
    result = CliRunner().invoke(app, ["model", "qwen", "--autostart", "off", "--config-dir", str(world.g)])

    assert result.exit_code == 2 and result.stderr.strip() == autostart.AUTOSTART_ALONE_LINE
    assert world.manager.calls == []


# ===================================================================== a failing verb (T-14-44)


def test_a_failing_disable_removes_nothing_prunes_nothing_and_hides_the_key(tmp_path, monkeypatch, fake_manager):
    world = on_world(tmp_path, monkeypatch, fake_manager, provider_key="sk-planted")
    turned_on(world)
    world.manager.answer_prefix("systemctl", "--user", "disable", code=1,
                                err="Failed to disable sk-planted\nbye")
    entry = trust.recorded_autostart_entry(world.g)

    result = world.run(mode="off")

    assert result.exit_code == 1
    assert "autostart off: disable failed (exit 1)" in result.stderr
    assert "sk-planted" not in result.stderr and "sk-planted" not in result.stdout
    assert world.unit_path.exists() and trust.recorded_autostart_entry(world.g) == entry
    assert RELOAD not in world.manager.calls
    assert lock_is_free(world.g)


def test_a_disable_the_manager_does_not_answer_removes_nothing(world):
    turned_on(world)
    world.manager.answer_prefix("systemctl", "--user", "disable", out=None)

    result = world.run(mode="off")

    assert result.exit_code == 1
    assert result.stderr.strip() == "autostart off: disable failed: the service manager did not answer"
    assert world.unit_path.exists() and trust.recorded_autostart_entry(world.g) is not None


def test_a_failing_reset_failed_does_not_stop_the_removal(world):
    turned_on(world)
    world.manager.answer_prefix("systemctl", "--user", "reset-failed", code=1, err="not failed")

    result = world.run(mode="off")

    assert result.exit_code == 0
    assert not world.unit_path.exists() and trust.recorded_autostart_entry(world.g) is None


def test_a_failing_reload_after_the_removal_still_prunes_and_says_so(world):
    turned_on(world)
    world.manager.answer(RELOAD, code=1, err="boom")

    result = world.run(mode="off")

    assert result.exit_code == 1
    assert "autostart off: daemon-reload failed (exit 1): boom" in result.stderr
    assert not world.unit_path.exists() and trust.recorded_autostart_entry(world.g) is None


def test_no_service_manager_program_removes_nothing(world):
    turned_on(world)
    world.mp.setattr(units, "systemctl_path", lambda: None)

    result = world.run(mode="off")

    assert result.exit_code == 1
    assert result.stderr.strip() == autostart.NO_MANAGER_LINE
    assert world.manager.calls == [] and world.unit_path.exists()


def test_a_busy_launch_lock_is_one_line_and_removes_nothing(world):
    turned_on(world)
    world.mp.setattr(autostart, "LOCK_WAIT_SECONDS", 0.05)
    fd = os.open(server.lock_path(world.g), os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        result = world.run(mode="off")
    finally:
        os.close(fd)

    assert result.exit_code == 1 and result.stderr.strip()
    assert world.unit_path.exists() and trust.recorded_autostart_entry(world.g) is not None
    assert RESET_FAILED not in world.manager.calls and DISABLE_NOW not in world.manager.calls
