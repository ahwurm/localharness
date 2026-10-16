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

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli import autostart, model_cmd, model_ops, workspace
from localharness.cli.app import app
from localharness.config import trust
from localharness.provider import lifecycle, units
from localharness.provider.server import server_dir
from tests.unit.procfs_support import fake_proc  # noqa: F401 — the fixture
from tests.unit.test_autostart_on_off import (  # noqa: F401 — `world` is the fixture
    DISPLAY, REAL_OPERATOR_TEST, SERVED, World, on_world, stdout_lines, world,
)
from tests.unit.test_serve_cmd import attach_peer, docker_peer, llama_peer, ollama_peer
from tests.unit.test_units_manager import answer_scan, show_block, unit_file
from tests.unit.units_support import fake_manager  # noqa: F401 — the fixture

FIRST = f"Autostart: on — {DISPLAY} (enabled, active, NRestarts 0, MainPID 4242)"
OFF_LINE = "Autostart: off — localharness model --autostart on serves the last choice at login"
SHELL_PATH = "/usr/local/bin:/usr/bin:/bin"
EMPTY = object()
OWN_START = 111  # the start time the in-flight cases register for the test process
SERVES = f"serves: {SERVED}"
PATH_LINE = f"PATH captured: {SHELL_PATH}"
BASE = [FIRST, SERVES, "linger: on", PATH_LINE]


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


# ===================================================================== a unit that is not the harness's own (D-11, D-13)


def _drop_in(world):
    says(world, drop_ins="/home/u/.config/systemd/user/localharness-llm.service.d/x.conf")
    return "has drop-ins (/home/u/.config/systemd/user/localharness-llm.service.d/x.conf)"


def _other_directory(world):
    other = "/etc/xdg/systemd/user/localharness-llm.service"
    says(world, fragment=other)
    return f"is loaded from {other}, not {world.unit_path} (FragmentPath)"


def _edited_byte(world):
    world.unit_path.write_text(
        world.unit_path.read_text(encoding="utf-8").replace("Restart=", "Restart =", 1), encoding="utf-8")
    return "differs from what localharness writes"


@pytest.mark.parametrize("make_foreign", [_drop_in, _other_directory, _edited_byte],
                         ids=["a_drop_in", "another_directory", "an_edited_byte"])
def test_foreign_unit_is_named_with_the_first_failed_test_and_still_answers(world, make_foreign):
    installed(world)
    phrase = make_foreign(world)
    anywhere(world)
    kept = tree(world.g), world.unit_path.read_bytes()

    result = status(world)

    assert result.exit_code == 0, (result.output, result.exception)
    lines = stdout_lines(result)
    assert lines[0] == f"Autostart: foreign — {DISPLAY} {phrase}"
    assert lines[1] == SERVES and "linger: on" in lines
    assert (tree(world.g), world.unit_path.read_bytes()) == kept


def test_foreign_a_manager_that_does_not_answer_says_so_in_the_foreign_line(world):
    installed(world)
    world.manager.answer(lambda argv: argv[:3] == ("systemctl", "--user", "show") and "NRestarts" in argv[4],
                         out=None)

    assert autostart.doctor_row(world.g) == (
        f"Autostart: foreign — {DISPLAY} could not be asked: the service manager did not answer")


# ===================================================================== linger and a second launcher (D-10, D-11)


def test_linger_off_names_the_command_that_turns_it_on(tmp_path, monkeypatch, fake_manager):
    made = on_world(tmp_path, monkeypatch, fake_manager, linger=False)
    installed(made)

    lines = autostart.status_lines(made.g)

    assert lines == [FIRST, SERVES, f"linger: off — at boot too after: loginctl enable-linger {units.current_user()}",
                     PATH_LINE]


def test_a_second_launcher_on_the_port_is_named_after_the_captured_path(world):
    installed(world)
    llm = unit_file(world.tmp, "llm.service", "/opt/llama-server -m /m.gguf --port 8080")
    answer_scan(world.manager, ["llm.service enabled"], [show_block("llm.service", llm)])

    lines = autostart.status_lines(world.g)

    assert lines == [*BASE, "llm.service (enabled) also launches a server on port 8080 — two launchers race at login"]


def test_no_second_launcher_means_no_line(world):
    installed(world)

    assert autostart.status_lines(world.g) == BASE


# ===================================================================== the boot marker nobody confirmed (D-18, AUTO-08)


def put_marker(world: World, *, invocation: str | None = "abc", shape: str = "retry",
               dropped: tuple[str, ...] = ("draft-mtp",), planned: int | None = 57344, full: int | None = 114688,
               record: str | None = None) -> bytes:
    key = lifecycle.record_key(world.harness().server) if record is None else record
    units.write_marker(world.g, units.Marker(
        record=key, binary="/x/llama-server", invocation=invocation, cause="none", shape=shape,
        planned_context=planned, full_fit=full, free_bytes=None, dropped=dropped,
        at="2026-10-15T00:00:00+00:00"))
    return (server_dir(world.g) / units.BOOT_MARKER_NAME).read_bytes()


def test_marker_of_the_live_run_at_the_retry_shape_says_what_was_dropped_and_halved(world):
    installed(world)
    put_marker(world)

    assert autostart.status_lines(world.g) == [
        *BASE, f"serving {SERVED} at the retry shape since boot — mtp off, -c halved"]


def test_marker_of_the_live_run_at_the_full_shape_has_no_parts(world):
    installed(world)
    put_marker(world, shape="full", dropped=(), planned=114688)

    assert autostart.status_lines(world.g)[-1] == f"serving {SERVED} at the full shape since boot"


def test_marker_parsers_dropped_is_named(world):
    installed(world)
    put_marker(world, shape="retry", dropped=("draft-mtp", "parsers"), planned=None, full=None)

    assert autostart.status_lines(world.g)[-1] == (
        f"serving {SERVED} at the retry shape since boot — mtp off, parsers off")


def test_marker_naming_no_record_shows_its_raw_key(world):
    installed(world)
    put_marker(world, record="llamacpp:gone-model", shape="full", dropped=(), planned=114688)

    assert autostart.status_lines(world.g)[-1] == "serving llamacpp:gone-model at the full shape since boot"


def test_marker_of_another_run_is_not_the_current_boot_and_is_left_alone(world):
    installed(world)
    kept = put_marker(world, invocation="old-inv")

    lines = autostart.status_lines(world.g)

    assert lines == BASE
    assert (server_dir(world.g) / units.BOOT_MARKER_NAME).read_bytes() == kept


def test_marker_with_no_invocation_is_never_a_live_run(world):
    installed(world)
    put_marker(world, invocation=None)

    assert autostart.status_lines(world.g) == BASE


def test_marker_while_the_unit_is_inactive_gets_no_line(world):
    installed(world, active="inactive", pid=0, invocation="")
    kept = put_marker(world, invocation="abc")

    lines = autostart.status_lines(world.g)

    assert not any(line.startswith("serving ") for line in lines)
    assert (server_dir(world.g) / units.BOOT_MARKER_NAME).read_bytes() == kept


def test_marker_absent_means_no_such_line(world):
    installed(world)

    assert not any(line.startswith("serving ") for line in autostart.status_lines(world.g))


def test_marker_line_is_in_the_doctor_row_only_through_status(world):
    installed(world)
    put_marker(world)

    assert autostart.doctor_row(world.g) == FIRST


# ===================================================================== why it stopped, in words (D-02, D-11, D-18)


def test_stopped_78_after_the_units_captured_path_changed_names_the_unit_file(world):
    installed(world)
    units.write_harness_unit(world.body({"PATH": "/other/bin"}))  # the unit's PATH edited after adoption
    says(world, active="failed", exit_status=78, pid=0, invocation="")

    lines = autostart.status_lines(world.g)

    assert lines[-1] == (f"stopped: the machine record withholds autostart.env ({DISPLAY}) — "
                         "run localharness start in a terminal")
    assert lines[0].startswith("Autostart: on — ") and "failed" in lines[0]


def test_stopped_78_after_the_primarys_binary_changed_names_the_record(world):
    installed(world)
    config = yaml.safe_load((world.g / "config.yaml").read_text(encoding="utf-8"))
    config["server"]["binary"] = "/x/other-llama-server"
    (world.g / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    says(world, active="failed", exit_status=78, pid=0, invocation="")

    assert autostart.status_lines(world.g)[-1] == (
        f"stopped: the machine record withholds {SERVED} — run localharness start in a terminal")


def test_stopped_78_with_no_machine_record_names_the_unit_file(world):
    installed(world)
    trust.forget_entries(world.g, [trust.AUTOSTART_KEY])
    says(world, active="failed", exit_status=78, pid=0, invocation="")

    assert f"withholds autostart.env ({DISPLAY})" in autostart.status_lines(world.g)[-1]


def test_stopped_75_is_the_live_pidfile_line(world):
    installed(world, active="failed", exit_status=75, pid=0, invocation="")

    assert autostart.status_lines(world.g)[-1] == units.STOPPED_LIVE_PIDFILE_LINE


def test_stopped_after_the_start_limit_names_the_command_and_never_runs_it(world):
    installed(world, active="failed", exit_status=1, result="start-limit-hit", pid=0, invocation="", restarts=3)
    anywhere(world)

    result = status(world)

    assert stdout_lines(result)[-1] == "stopped after 3 failed loads — journalctl --user -u localharness-llm.service"
    assert [c for c in world.manager.calls if c and c[0] != "systemctl"] == []
    assert not any("journal" in " ".join(c) for c in world.manager.calls)


@pytest.mark.parametrize("peer, name, kind", [
    (docker_peer("dock", 8082), "dock", "docker"),
    (ollama_peer("oll", 11434), "oll", "Ollama"),
    (attach_peer("attached", 1234), "attached", "attach-only"),
], ids=["docker", "ollama", "attach_only"])
def test_not_served_by_the_unit_when_it_is_inactive_and_the_target_is_not_its_kind(
        tmp_path, monkeypatch, fake_manager, peer, name, kind):
    made = on_world(tmp_path, monkeypatch, fake_manager, peers=[peer])
    installed(made, active="inactive", pid=0, invocation="")
    (made.g / "overrides.yaml").write_text(yaml.safe_dump({"active_endpoint": {
        "name": name, "base_url": peer["base_url"], "provider_type": peer["provider_type"],
        "model": (peer.get("lifecycle") or {}).get("model", "m")}}), encoding="utf-8")

    lines = autostart.status_lines(made.g)

    assert lines[1] == f"serves: {name}"
    assert lines[-1] == f"{name} is {kind} — not served by the unit"


def test_an_active_healthy_unit_has_no_translated_line(world):
    installed(world)

    assert autostart.status_lines(world.g) == BASE


def test_an_inactive_unit_serving_its_own_kind_has_no_translated_line(world):
    installed(world, active="inactive", pid=0, invocation="")

    assert autostart.status_lines(world.g)[-1] == PATH_LINE


# ===================================================================== a live in-flight target (R-14-45, R-14-46)


@pytest.fixture
def peered(tmp_path, monkeypatch, fake_manager, fake_proc):  # noqa: F811
    made = on_world(tmp_path, monkeypatch, fake_manager,
                    peers=[llama_peer("flash-next", 8081), llama_peer("qwen-27b", 8082)])
    fake_proc.process(os.getpid(), argv=("python",), start_time=OWN_START, comm="python")
    return installed(made)


def set_pending(world: World, name: str) -> None:
    model_ops.write_pending_preset(world.harness(), name, config_dir=world.g)


def plant_target(world: World, body: str) -> None:
    directory = server_dir(world.g)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / units.TARGET_NAME).write_text(body, encoding="utf-8")


OUTRANKS = ("in flight: flash-next (requested by pid {pid}) — "
            "outranks pending_preset qwen-27b while that process runs")


def test_in_flight_outranking_a_pending_preset_is_named_with_the_process_and_the_choice(peered):
    set_pending(peered, "qwen-27b")
    units.write_target(peered.g, "flash-next", in_flight=True)
    before = tree(peered.g)

    lines = autostart.status_lines(peered.g)

    outranks = OUTRANKS.format(pid=os.getpid())
    assert lines == [FIRST, "serves: flash-next", outranks, "linger: on", PATH_LINE]
    assert autostart.doctor_row(peered.g) == f"{FIRST}; {outranks}"
    assert tree(peered.g) == before  # nothing cleared: `status` is not a second writer of unit-target


def test_in_flight_without_a_pending_preset_is_the_short_line_and_the_doctor_row_stays_one_line(peered):
    units.write_target(peered.g, "flash-next", in_flight=True)

    lines = autostart.status_lines(peered.g)

    assert lines == [FIRST, "serves: flash-next", f"in flight: flash-next (requested by pid {os.getpid()})",
                     "linger: on", PATH_LINE]
    assert autostart.doctor_row(peered.g) == FIRST


def test_in_flight_through_the_command_prints_the_same_lines_off_a_terminal(peered):
    set_pending(peered, "qwen-27b")
    units.write_target(peered.g, "flash-next", in_flight=True)
    anywhere(peered)

    result = status(peered)

    assert result.exit_code == 0, (result.output, result.exception)
    assert OUTRANKS.format(pid=os.getpid()) in stdout_lines(result)


@pytest.mark.parametrize("body", [f"flash-next {os.getpid()} {OWN_START + 1}\n", "flash-next\n"],
                         ids=["a_gone_requester", "a_plain_entry"])
def test_in_flight_no_line_for_a_gone_requester_or_a_plain_entry(peered, body):
    set_pending(peered, "qwen-27b")
    plant_target(peered, body)

    lines = autostart.status_lines(peered.g)

    assert lines == [FIRST, "serves: qwen-27b", "linger: on", PATH_LINE]
    assert autostart.doctor_row(peered.g) == FIRST


def test_in_flight_naming_no_record_is_the_short_line_with_its_raw_name(peered):
    set_pending(peered, "qwen-27b")
    plant_target(peered, f"nonesuch {os.getpid()} {OWN_START}\n")

    lines = autostart.status_lines(peered.g)

    assert lines[1:3] == ["serves: qwen-27b", f"in flight: nonesuch (requested by pid {os.getpid()})"]
    assert autostart.doctor_row(peered.g) == FIRST

