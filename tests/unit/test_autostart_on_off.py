"""`localharness model --autostart on` — gate, scope, capture, write, adopt, reload, re-check, enable (spec 20
§9.1, §9.2, §12; D-01..D-10, AUTO-01, AUTO-03..AUTO-06).

(`off` has its own module, test_autostart_off.py; `status` is tested with its own plan.)

`on_world` builds the one world every case shares: a global dir (conftest's hermetic `LOCALHARNESS_HOME`
renamed per case) with a llama.cpp `server:` record, a fake home, the unit file's directory under a redirected
`XDG_CONFIG_HOME`, a linger directory holding this user, a terminal operator, and a fake service manager
(`units_support.FakeManager`) that answers `daemon-reload`, the state `show` of the harness unit (loaded from
its own file, no drop-ins), `enable` and an empty `list-unit-files`. An argv nobody answered is the runner's
None, so a verb the code did not mean to issue fails the case.

Nothing here runs a service manager or a login manager, starts a unit or a server, or dials a port; the unit
file is written under tmp and linger is a stat under tmp.
"""
from __future__ import annotations

import errno
import fcntl
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli import autostart, model_cmd
from localharness.cli.app import app
from localharness.cli.workspace import decide_machine_trust
from localharness.config import trust
from localharness.config.loader import ConfigLoader
from localharness.provider import lifecycle, server, units
from tests.unit.procfs_support import fake_proc  # noqa: F401 — the fixture
from tests.unit.test_serve_cmd import docker_peer, ollama_peer
from tests.unit.test_units_manager import answer_scan, show_block, unit_file
from tests.unit.trust_support import answers, never_asked
from tests.unit.unit_strategy_support import FakeServe
from tests.unit.units_support import FakeManager, fake_manager  # noqa: F401 — the fixture

REAL_OPERATOR_TEST = model_cmd._is_local_operator  # the real gate, kept before any case replaces it
SERVED = "qwen-27b-swift-1.5"
INTERPRETER = "/opt/lh/bin/python"
UNIT = units.HARNESS_UNIT
SHELL_PATH = "/usr/local/bin:/usr/bin:/bin"
RELOAD = ("systemctl", "--user", "daemon-reload")
STATE_SHOW = ("systemctl", "--user", "show", "-p", units._STATE_PROPS, "--", UNIT)
ENABLE_NOW = ("systemctl", "--user", "enable", "--now", UNIT)
ENABLE = ("systemctl", "--user", "enable", UNIT)
LIST_UNITS = ("systemctl", "--user", "list-unit-files")
TERMINAL_LINE = "autostart is set from a terminal: localharness model --autostart on"
DISPLAY = "~/.config/systemd/user/localharness-llm.service"


@dataclass
class World:
    tmp: Path
    g: Path
    home: Path
    linger: Path
    manager: FakeManager
    mp: pytest.MonkeyPatch

    def run(self, *args: str, mode: str = "on", config_dir: Path | None = None, before: tuple = ()):
        return CliRunner().invoke(
            app, [*before, "model", "--autostart", mode, "--config-dir", str(config_dir or self.g), *args])

    def manager_says(self, *, drop_ins: str = "", fragment: str | None = None, active: str = "inactive") -> None:
        text = "\n".join([
            f"FragmentPath={units.harness_unit_path() if fragment is None else fragment}",
            f"DropInPaths={drop_ins}", f"ActiveState={active}", "UnitFileState=disabled", "NRestarts=0",
            "ExecMainStatus=0", "MainPID=0", "InvocationID=", "Result=success", ""])
        self.manager.answer(lambda argv: argv[:3] == ("systemctl", "--user", "show") and "NRestarts" in argv[4],
                            out=text)

    @property
    def unit_path(self) -> Path:
        return units.harness_unit_path()

    @property
    def verbs(self) -> list[tuple[str, ...]]:
        return [c for c in self.manager.calls if c[2] not in ("show", "list-unit-files")]

    def body(self, captured: dict | None = None) -> str:
        return units.render_harness_unit(
            INTERPRETER, self.g, {"PATH": SHELL_PATH} if captured is None else captured,
            home=units.harness_home())

    def harness(self):
        return ConfigLoader(config_dir=self.g, local_config_dir=None).load_harness()


def write_config(g: Path, *, server_record: dict | None = None, peers=(), provider_key: str | None = None) -> None:
    record = server_record or {"runtime": "llamacpp", "binary": "/x/llama-server", "model": "/m/a.gguf",
                               "served_name": SERVED, "port": 8080}
    provider = {"provider_type": "llamacpp", "base_url": "http://127.0.0.1:8080/v1", "default_model": SERVED,
                "available_models": [SERVED]}
    if provider_key is not None:
        provider["api_key"] = provider_key
    (g / "config.yaml").write_text(yaml.safe_dump({
        "version": "1", "provider": provider,
        "org": {"default_model": SERVED, "audit_log_path": str(g / "audit.jsonl")},
        "extra_endpoints": list(peers), "server": record}), encoding="utf-8")


def on_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manager: FakeManager, *,
             dirname: str = ".localharness", peers=(), provider_key: str | None = None,
             linger: bool = True) -> World:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    g = tmp_path / dirname
    g.mkdir()
    monkeypatch.setenv("LOCALHARNESS_HOME", str(g))
    monkeypatch.setattr(sys, "executable", INTERPRETER)
    monkeypatch.setenv("PATH", SHELL_PATH)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("LOCALHARNESS_COMMIT", raising=False)
    write_config(g, peers=peers, provider_key=provider_key)

    lingering = tmp_path / "linger"
    lingering.mkdir()
    if linger:
        (lingering / units.current_user()).write_text("", encoding="utf-8")
    monkeypatch.setattr(units, "LINGER_DIR", str(lingering))
    monkeypatch.setattr(units, "systemctl_path", lambda: "/usr/bin/systemctl")
    monkeypatch.setattr(model_cmd, "_is_local_operator", lambda: True)
    never_asked(monkeypatch)

    world = World(tmp_path, g, home, lingering, manager, monkeypatch)
    manager.answer(RELOAD, out="")
    manager.answer_prefix("systemctl", "--user", "enable", out="")
    manager.answer(LIST_UNITS + ("--type=service", "--plain", "--no-legend", "--no-pager"), out="")
    world.manager_says()
    return world


@pytest.fixture
def world(tmp_path, monkeypatch, fake_manager) -> World:
    return on_world(tmp_path, monkeypatch, fake_manager)


def receipt(captured: str = autostart.CAPTURED_TEXT, *, boot: bool = True, takes_over: bool = False) -> str:
    line = (autostart.RECEIPT_BOOT.format(path=DISPLAY, name=SERVED, captured=captured) if boot else
            autostart.RECEIPT_LOGIN.format(path=DISPLAY, name=SERVED, user=units.current_user(),
                                           captured=captured))
    return line + (autostart.TAKES_OVER_CLAUSE if takes_over else "")


def lock_is_free(g: Path) -> bool:
    fd = os.open(server.lock_path(g), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
            return False
        raise
    finally:
        os.close(fd)
    return True


def stdout_lines(result) -> list[str]:
    return [line for line in result.stdout.splitlines() if line.strip()]


# ===================================================================== the tracer


def test_on_from_a_terminal_writes_the_unit_adopts_its_entry_reloads_and_enables_it_now(world):
    result = world.run()

    assert result.exit_code == 0, (result.output, result.exception)
    assert world.unit_path.read_text(encoding="utf-8") == world.body()
    entry = trust.recorded_autostart_entry(world.g)
    assert entry is not None and entry == trust.autostart_entries(world.g)[0]
    assert world.manager.calls[:3] == [RELOAD, STATE_SHOW, ENABLE_NOW]
    assert world.manager.calls[3][:3] == LIST_UNITS  # the scan for a second launcher, after the receipt
    assert stdout_lines(result) == [
        f"autostart on: {DISPLAY} serves {SERVED} at login and boot (linger on; PATH captured from this shell)"]
    assert stdout_lines(result) == [receipt()]


def test_the_unit_file_is_the_harness_own_unit_by_all_three_tests(world):
    world.run()

    state = units.unit_runtime_state()

    assert state is not None
    assert units.harness_unit_failure(state, world.g) is None


def test_only_the_user_manager_is_asked_and_the_login_managers_program_is_never_run(world):
    world.run()

    assert world.manager.calls and {call[:2] for call in world.manager.calls} == {("systemctl", "--user")}
    source = Path(autostart.__file__).read_text(encoding="utf-8")
    assert not any(word in source for word in ("subprocess", "os.system", "Popen", "execv"))


# ===================================================================== the gate (D-01, D-03)


def test_gate_not_a_terminal_refuses_before_a_decision_a_write_or_a_verb(world):
    world.mp.setattr(model_cmd, "_is_local_operator", lambda: False)

    def decided(*a, **kw):
        raise AssertionError("trust was decided without a local operator")

    world.mp.setattr(model_cmd, "decide_machine_trust", decided)

    result = world.run()

    assert result.exit_code == 2
    assert result.stderr.strip() == TERMINAL_LINE and result.stdout == ""
    assert not world.unit_path.exists() and world.manager.calls == []
    assert trust.recorded_machine(world.g) is None


def _real_gate_on_a_tty(world, tty: bool) -> None:
    world.mp.setattr(model_cmd, "_is_local_operator", REAL_OPERATOR_TEST)
    world.mp.setattr(model_cmd, "sys", SimpleNamespace(stdin=SimpleNamespace(isatty=lambda: tty)))


def test_gate_the_agent_command_marker_refuses_even_on_a_tty(world):
    _real_gate_on_a_tty(world, True)
    world.mp.setenv("LOCALHARNESS_COMMIT", "1")

    result = world.run()

    assert result.exit_code == 2 and result.stderr.strip() == TERMINAL_LINE
    assert not world.unit_path.exists() and world.manager.calls == []


def test_gate_a_pipe_has_no_terminal_and_is_refused(world):
    _real_gate_on_a_tty(world, False)

    result = world.run()

    assert result.exit_code == 2 and result.stderr.strip() == TERMINAL_LINE
    assert world.manager.calls == []


def test_gate_a_real_terminal_without_the_marker_is_let_through(world):
    _real_gate_on_a_tty(world, True)

    result = world.run()

    assert result.exit_code == 0, (result.output, result.exception)
    assert world.unit_path.exists()


def test_the_trust_question_is_asked_once_with_ask_true_after_the_gate(world):
    calls = []
    real = model_cmd.decide_machine_trust

    def spy(global_dir, **kw):
        calls.append(dict(kw))
        return real(global_dir, **kw)

    world.mp.setattr(model_cmd, "decide_machine_trust", spy)

    result = world.run()

    assert result.exit_code == 0
    assert [c["ask"] for c in calls] == [True]


@pytest.mark.parametrize("mode", ["maybe", "status", "ON", ""])
def test_a_value_on_does_not_take_exits_2_with_the_modes_line(world, mode):
    result = world.run(mode=mode)

    assert result.exit_code == 2
    assert result.stderr.strip() == "--autostart takes on, off or status"
    assert world.manager.calls == [] and not world.unit_path.exists()


@pytest.mark.parametrize("extra", [("qwen",), ("--download", "org/model"), ("--file", "x.gguf")])
def test_a_model_name_or_download_beside_autostart_exits_2(world, extra):
    args = ["model", *extra, "--autostart", "on", "--config-dir", str(world.g)]

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 2
    assert result.stderr.strip() == autostart.AUTOSTART_ALONE_LINE
    assert world.manager.calls == [] and not world.unit_path.exists()


# ===================================================================== the scope (D-02)


@pytest.mark.parametrize("peer, name, kind", [
    (docker_peer("dock", 8082), "dock", "docker"),
    (ollama_peer("oll", 11434), "oll", "Ollama"),
])
def test_scope_a_target_the_unit_cannot_serve_is_refused_with_the_fix(tmp_path, monkeypatch, fake_manager,
                                                                      peer, name, kind):
    world = on_world(tmp_path, monkeypatch, fake_manager, peers=[peer])
    units.write_target(world.g, name)

    result = world.run()

    assert result.exit_code == 2
    assert result.stderr.strip() == (
        f"autostart serves llama.cpp and a vLLM binary; {name} is {kind} — "
        "localharness model <a llama.cpp preset> first")
    assert not world.unit_path.exists() and world.manager.calls == []


def test_scope_the_ladder_without_a_preset_or_a_workspace_layer_is_what_decides(world):
    # the unit has neither a workspace layer nor a preset: a project config next to the terminal cannot
    # change what it will serve. The ladder lands on the global primary.
    project = world.tmp / "proj"
    (project / ".localharness").mkdir(parents=True)
    (project / ".localharness" / "config.yaml").write_text(
        yaml.safe_dump({"server": {"runtime": "ollama", "model": "x"}}), encoding="utf-8")
    world.mp.chdir(project)

    result = world.run()

    assert result.exit_code == 0, (result.output, result.exception)
    assert stdout_lines(result) == [receipt()]


def test_scope_a_record_the_person_declined_is_not_turned_on(tmp_path, monkeypatch, fake_manager):
    world = on_world(tmp_path, monkeypatch, fake_manager)
    decide_machine_trust(world.g, ask=False)
    cfg = yaml.safe_load((world.g / "config.yaml").read_text(encoding="utf-8"))
    cfg["server"]["model"] = "/m/changed.gguf"
    (world.g / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    answers(monkeypatch, False)

    result = world.run()

    assert result.exit_code == 2
    assert not world.unit_path.exists() and world.manager.calls == []


# ===================================================================== the capture (D-07)


def test_capture_relative_elements_are_dropped_and_named_and_hf_home_is_made_absolute(world):
    models = world.tmp / "models"
    world.mp.chdir(world.tmp)
    world.mp.setenv("PATH", "./bin:/usr/bin::/bin")
    world.mp.setenv("LD_LIBRARY_PATH", "./lib:/opt/cuda/lib")
    world.mp.setenv("HF_HOME", "models/hf")

    result = world.run()

    assert result.exit_code == 0, (result.output, result.exception)
    env_lines = [ln for ln in world.unit_path.read_text(encoding="utf-8").splitlines() if ln.startswith("# env ")]
    assert env_lines == ["# env PATH=/usr/bin:/bin", f"# env HF_HOME={models / 'hf'}",
                         "# env LD_LIBRARY_PATH=/opt/cuda/lib"]
    assert stdout_lines(result) == [receipt(
        "PATH captured from this shell, dropped relative: PATH ./bin, PATH (empty), LD_LIBRARY_PATH ./lib, "
        f"HF_HOME made absolute: models/hf → {models / 'hf'}")]


def test_capture_an_absolute_hf_home_and_ld_library_path_are_kept_as_they_are(world):
    world.mp.setenv("HF_HOME", "/data/hf")
    world.mp.setenv("LD_LIBRARY_PATH", "/opt/cuda/lib:/usr/lib")

    result = world.run()

    assert result.exit_code == 0
    text = world.unit_path.read_text(encoding="utf-8")
    assert "# env HF_HOME=/data/hf\n# env LD_LIBRARY_PATH=/opt/cuda/lib:/usr/lib\n[Unit]" in text
    assert stdout_lines(result) == [receipt()]  # nothing dropped, nothing made absolute


def test_charset_a_path_element_with_a_space_is_refused_naming_it_and_nothing_is_written(world):
    world.mp.setenv("PATH", "/usr/bin:/opt/my tools")

    result = world.run()

    assert result.exit_code == 2
    assert "/opt/my tools" in result.stderr
    assert not world.unit_path.exists() and world.manager.calls == []


def test_charset_a_config_dir_with_a_space_is_refused_naming_it(tmp_path, monkeypatch, fake_manager):
    world = on_world(tmp_path, monkeypatch, fake_manager, dirname="my cfg")

    result = world.run()

    assert result.exit_code == 2
    assert "my cfg" in result.stderr
    assert not world.unit_path.exists() and world.manager.calls == []


def test_charset_a_path_with_no_absolute_element_is_refused(world):
    world.mp.setenv("PATH", "./bin:bin")

    result = world.run()

    assert result.exit_code == 2 and "PATH" in result.stderr
    assert not world.unit_path.exists()


# ===================================================================== the receipt (D-09)


def test_the_receipt_without_linger_names_the_command_and_runs_nothing_for_it(tmp_path, monkeypatch, fake_manager):
    world = on_world(tmp_path, monkeypatch, fake_manager, linger=False)

    result = world.run()

    assert result.exit_code == 0
    assert stdout_lines(result) == [receipt(boot=False)]
    assert f"after: loginctl enable-linger {units.current_user()};" in result.stdout
    assert [c[0] for c in world.manager.calls] == ["systemctl"] * len(world.manager.calls)


def test_linger_is_a_file_under_the_linger_directory(world):
    (world.linger / units.current_user()).unlink()

    result = world.run()

    assert stdout_lines(result) == [receipt(boot=False)]


# ===================================================================== a running server is kept (D-10)


def test_kept_a_unit_that_is_already_active_is_enabled_without_now(world):
    world.manager_says(active="active")

    result = world.run()

    assert result.exit_code == 0
    assert world.verbs == [RELOAD, ENABLE]
    assert stdout_lines(result) == [receipt(takes_over=True)]


def test_kept_a_live_launch_of_a_record_is_not_restarted_signalled_or_stopped(world, fake_proc, monkeypatch):
    spec = world.harness().server
    fake_proc.process(4242, argv=FakeServe.record_argv(spec), exe=str(spec.binary), start_time=5000,
                      cgroup=None)
    server.server_dir(world.g).mkdir(parents=True, exist_ok=True)
    server.pid_path(world.g).write_text("4242 5000 8080", encoding="utf-8")
    killed = []
    monkeypatch.setattr(os, "kill", lambda *a: killed.append(a))

    result = world.run()

    assert result.exit_code == 0, (result.output, result.exception)
    assert world.verbs == [RELOAD, ENABLE]
    assert not [v for v in world.verbs if v[2] in ("restart", "stop", "start")] and killed == []
    assert stdout_lines(result) == [receipt(takes_over=True)]


def test_live_nothing_serving_enables_now(world):
    result = world.run()

    assert world.verbs == [RELOAD, ENABLE_NOW]
    assert autostart.TAKES_OVER_CLAUSE not in result.stdout


def test_live_a_pidfile_whose_process_is_not_the_record_does_not_count_as_serving(world, fake_proc):
    fake_proc.process(4242, argv=("sleep", "1000"), exe="/usr/bin/sleep", start_time=5000, cgroup=None,
                      comm="sleep")
    server.server_dir(world.g).mkdir(parents=True, exist_ok=True)
    server.pid_path(world.g).write_text("4242 5000 8080", encoding="utf-8")

    world.run()

    assert world.verbs == [RELOAD, ENABLE_NOW]


# ===================================================================== a second launcher (D-10)


def test_second_launcher_is_named_once_after_the_receipt(world):
    llm = unit_file(world.tmp, "llm.service", "/opt/llama-server -m /m.gguf --port 8080")
    answer_scan(world.manager, ["llm.service enabled"], [show_block("llm.service", llm)])

    result = world.run()

    assert result.exit_code == 0
    assert stdout_lines(result) == [receipt()]
    assert result.stderr.strip().splitlines() == [
        "llm.service (enabled) also launches a server on port 8080 — two launchers race at login"]


def test_second_launcher_none_means_no_line(world):
    result = world.run()

    assert result.stderr == ""


# ===================================================================== a rewrite


def test_rewrite_an_older_body_is_shown_before_the_receipt(world):
    units.write_harness_unit(world.body({"PATH": "/old/bin"}))

    result = world.run()

    assert result.exit_code == 0
    body = world.body()
    assert stdout_lines(result) == [
        f"rewriting {DISPLAY}; it now reads:", *[ln for ln in body.splitlines() if ln.strip()], receipt()]
    assert world.unit_path.read_text(encoding="utf-8") == body


def test_rewrite_an_identical_body_prints_neither_line(world):
    world.run()
    world.manager.calls.clear()

    result = world.run()

    assert result.exit_code == 0
    assert "rewriting" not in result.stdout
    assert stdout_lines(result) == [receipt()]


# ===================================================================== failures (D-10)


def test_fail_no_service_manager_program_writes_nothing(world):
    world.mp.setattr(units, "systemctl_path", lambda: None)

    result = world.run()

    assert result.exit_code == 1
    assert result.stderr.strip() == (
        "autostart needs the user service manager (systemctl in /usr/bin or /bin) — not found")
    assert not world.unit_path.exists() and world.manager.calls == []


def test_fail_a_reload_that_fails_names_the_verb_the_code_and_not_the_key(tmp_path, monkeypatch, fake_manager):
    world = on_world(tmp_path, monkeypatch, fake_manager, provider_key="sk-planted")
    world.manager.answer(RELOAD, code=1, err="Failed to reload with sk-planted\nbye")

    result = world.run()

    assert result.exit_code == 1
    assert "daemon-reload failed (exit 1)" in result.stderr
    assert "sk-planted" not in result.stderr and "sk-planted" not in result.stdout
    assert ENABLE_NOW not in world.manager.calls and ENABLE not in world.manager.calls


def test_fail_a_manager_that_does_not_answer_the_reload_is_one_line(world):
    world.manager.answer(RELOAD, out=None)

    result = world.run()

    assert result.exit_code == 1
    assert result.stderr.strip() == "autostart on: daemon-reload failed: the service manager did not answer"


def test_fail_an_enable_that_fails_names_the_verb_and_the_code(world):
    world.manager.answer_prefix("systemctl", "--user", "enable", code=1, err="Unit is masked")

    result = world.run()

    assert result.exit_code == 1
    assert result.stderr.strip() == "autostart on: enable failed (exit 1): Unit is masked"
    assert stdout_lines(result) == []  # no receipt for what did not happen


def test_fail_a_drop_in_after_the_reload_enables_nothing_and_names_it(world):
    world.manager_says(drop_ins="/home/u/.config/systemd/user/localharness-llm.service.d/x.conf")

    result = world.run()

    assert result.exit_code == 1
    assert "has drop-ins (/home/u/.config/systemd/user/localharness-llm.service.d/x.conf)" in result.stderr
    assert "nothing was enabled" in result.stderr
    assert ENABLE_NOW not in world.manager.calls and ENABLE not in world.manager.calls


def test_fail_a_unit_loaded_from_another_directory_enables_nothing(world):
    world.manager_says(fragment="/etc/systemd/user/localharness-llm.service")

    result = world.run()

    assert result.exit_code == 1
    assert "is loaded from /etc/systemd/user/localharness-llm.service" in result.stderr
    assert world.verbs == [RELOAD]


def test_fail_a_manager_that_cannot_be_asked_after_the_reload_enables_nothing(world):
    world.manager.answer(lambda argv: argv[:3] == ("systemctl", "--user", "show") and "NRestarts" in argv[4],
                         out=None)

    result = world.run()

    assert result.exit_code == 1 and "nothing was enabled" in result.stderr
    assert world.verbs == [RELOAD]


def test_fail_the_lock_busy_beyond_its_wait_exits_1(world):
    @contextmanager
    def busy(*args, **kw):
        raise TimeoutError("another start holds the launch lock")
        yield  # pragma: no cover

    world.mp.setattr(lifecycle, "launch_lock_sync", busy)

    result = world.run()

    assert result.exit_code == 1 and "another start holds the launch lock" in result.stderr
    assert not world.unit_path.exists() and world.manager.calls == []


def test_fail_an_unwritable_unit_directory_exits_1_with_the_path(world):
    def refuse(body):
        raise PermissionError(13, "Permission denied")

    world.mp.setattr(units, "write_harness_unit", refuse)

    result = world.run()

    assert result.exit_code == 1
    assert "could not write" in result.stderr and "Permission denied" in result.stderr
    assert world.manager.calls == []


# ===================================================================== interrupted (AUTO-04)


def test_interrupted_between_the_write_and_the_adoption_leaves_an_unadopted_entry(world):
    decide_machine_trust(world.g, ask=False)  # the machine record, as a first start makes it
    harness = world.harness()
    record = autostart.model_ops.serving_records(harness)[0]

    def interrupted(*args, **kw):
        raise KeyboardInterrupt

    world.mp.setattr(trust, "adopt_entries", interrupted)
    lines: list[str] = []

    with pytest.raises(KeyboardInterrupt):
        autostart.turn_on(harness, record, world.g, environ={"PATH": SHELL_PATH}, cwd=str(world.tmp),
                          out=lines.append, err=lines.append)

    assert world.unit_path.read_text(encoding="utf-8") == world.body()
    assert world.manager.calls == []  # not reloaded, not enabled
    withheld = decide_machine_trust(world.g, ask=False).withheld  # what `serve` would see: exit 78
    assert (trust.LAUNCH_KIND, trust.AUTOSTART_ENTRY_NAME) in withheld.get(trust.AUTOSTART_FILE, frozenset())
    assert lock_is_free(world.g)  # the interrupt released the launch lock
