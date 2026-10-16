"""The service-manager runner and the live state of the harness's unit (spec 20 §9.2, D-10, D-12, D-13).

Three layers, none of which reaches a real service manager. The state reader and `own_unit` run against
`fake_manager` (a canned `ManagerResult` per argv). The real runner `_run_manager` is exercised only by
`TestTheRealRunner`, which carries `real_manager_seam` and fakes `subprocess.run` itself, with
`MANAGER_DIRS` pointed at a tmp directory holding a file that is never executed. Everywhere else the
conftest stub answers None and a spawn would fail the test."""
from __future__ import annotations

import os
import subprocess
import types
from pathlib import Path

import pytest

from localharness.provider import units as u
from tests.unit.units_support import FakeManager, fake_manager  # noqa: F401

PROPS = ("FragmentPath,DropInPaths,ActiveState,UnitFileState,NRestarts,ExecMainStatus,MainPID,"
         "InvocationID,Result")
SHOW_ARGV = ("systemctl", "--user", "show", "-p", PROPS, "--", "localharness-llm.service")
INTERPRETER = "/opt/lh/bin/python"
CONFIG_DIR = "/home/u/.localharness"
HOME = "/home/u"
CAPTURED = {"PATH": "/usr/bin:/bin"}


@pytest.fixture(autouse=True)
def _redirected(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def show(**over: str) -> str:
    """The nine-property `show` text of a healthy, enabled, running unit, with `over` applied."""
    fields = {
        "FragmentPath": str(u.harness_unit_path()), "DropInPaths": "", "ActiveState": "active",
        "UnitFileState": "enabled", "NRestarts": "1", "ExecMainStatus": "0", "MainPID": "4242",
        "InvocationID": "abc", "Result": "success",
    }
    fields.update(over)
    return "".join(f"{k}={v}\n" for k, v in fields.items() if v is not None)


def rendered() -> str:
    return u.render_harness_unit(INTERPRETER, CONFIG_DIR, CAPTURED, home=HOME, version="0.19.0")


# -------------------------------------------------------------------------- unit_runtime_state


def test_the_nine_properties_are_asked_for_by_one_exact_argv(fake_manager):
    fake_manager.answer(SHOW_ARGV, show())
    state = u.unit_runtime_state()
    assert state == u.UnitRuntimeState(
        "abc", 1, main_pid=4242, active_state="active", unit_file_state="enabled",
        fragment_path=str(u.harness_unit_path()), drop_in_paths="", exec_main_status=0, result="success")
    assert fake_manager.calls == [SHOW_ARGV]
    assert fake_manager.limits == [(u.SYSTEMCTL_TIMEOUT_S, u.SYSTEMCTL_OUTPUT_CAP)]
    assert state.own is False


def test_a_main_pid_of_zero_is_no_process(fake_manager):
    fake_manager.answer(SHOW_ARGV, show(MainPID="0"))
    assert u.unit_runtime_state().main_pid is None


def test_an_empty_invocation_id_is_none(fake_manager):
    fake_manager.answer(SHOW_ARGV, show(InvocationID=""))
    assert u.unit_runtime_state().invocation_id is None


def test_a_unit_that_is_not_loaded_is_still_a_state(fake_manager):
    fake_manager.answer(SHOW_ARGV, show(FragmentPath="", ActiveState="inactive", UnitFileState="",
                                        NRestarts="0", MainPID="0", InvocationID=""))
    state = u.unit_runtime_state()
    assert state is not None and state.fragment_path == "" and state.active_state == "inactive"


@pytest.mark.parametrize("key", ["NRestarts", "ExecMainStatus", "MainPID"])
@pytest.mark.parametrize("value", ["x", "", "-1", "1.5", "٣"])
def test_a_number_that_is_not_a_whole_number_voids_the_state(fake_manager, key, value):
    fake_manager.answer(SHOW_ARGV, show(**{key: value}))
    assert u.unit_runtime_state() is None


@pytest.mark.parametrize("key", ["NRestarts", "ExecMainStatus", "MainPID"])
def test_a_missing_number_voids_the_state(fake_manager, key):
    fake_manager.answer(SHOW_ARGV, show(**{key: None}))
    assert u.unit_runtime_state() is None


def test_the_first_of_a_repeated_key_wins(fake_manager):
    fake_manager.answer(SHOW_ARGV, show() + "MainPID=1\n")
    assert u.unit_runtime_state().main_pid == 4242


def test_a_non_zero_exit_is_not_a_state(fake_manager):
    fake_manager.answer(SHOW_ARGV, show(), code=1)
    assert u.unit_runtime_state() is None


def test_a_runner_that_could_not_ask_is_not_a_state(fake_manager):
    fake_manager.answer(SHOW_ARGV, None)
    assert u.unit_runtime_state() is None


def test_an_answer_over_the_cap_is_not_a_state(fake_manager):
    fake_manager.answer(SHOW_ARGV, show() + "X=" + "y" * u.SYSTEMCTL_OUTPUT_CAP)
    assert u.unit_runtime_state() is None


def test_under_the_suite_stub_nothing_runs_and_the_state_is_none(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test reached subprocess.run")
    monkeypatch.setattr(subprocess, "run", refuse)
    assert u.unit_runtime_state() is None
    assert u._run_manager(["systemctl", "--user", "show"], timeout=1, cap=1) is None


# --------------------------------------------------------------------- own_unit / harness_unit_state


def test_the_unit_the_harness_wrote_is_an_own_state(fake_manager):
    u.write_harness_unit(rendered())
    fake_manager.answer(SHOW_ARGV, show())
    state = u.harness_unit_state(CONFIG_DIR, interpreter=INTERPRETER, home=HOME)
    assert state is not None and state.own is True
    assert state.main_pid == 4242 and state.invocation_id == "abc"


def test_a_drop_in_makes_the_unit_foreign(fake_manager):
    u.write_harness_unit(rendered())
    fake_manager.answer(SHOW_ARGV, show(DropInPaths="/x.conf"))
    assert u.harness_unit_state(CONFIG_DIR, interpreter=INTERPRETER, home=HOME) is None


def test_another_fragment_path_makes_the_unit_foreign(fake_manager):
    u.write_harness_unit(rendered())
    fake_manager.answer(SHOW_ARGV, show(FragmentPath="/etc/systemd/user/localharness-llm.service"))
    assert u.harness_unit_state(CONFIG_DIR, interpreter=INTERPRETER, home=HOME) is None


def test_an_edited_file_makes_the_unit_foreign(fake_manager):
    u.write_harness_unit(rendered().replace("RestartSec=5", "RestartSec=1"))
    fake_manager.answer(SHOW_ARGV, show())
    assert u.harness_unit_state(CONFIG_DIR, interpreter=INTERPRETER, home=HOME) is None


def test_no_file_means_not_own(fake_manager):
    fake_manager.answer(SHOW_ARGV, show())
    assert u.harness_unit_state(CONFIG_DIR, interpreter=INTERPRETER, home=HOME) is None


def test_a_manager_that_did_not_answer_is_not_own():
    u.write_harness_unit(rendered())
    assert u.harness_unit_state(CONFIG_DIR, interpreter=INTERPRETER, home=HOME) is None


def test_own_unit_refuses_anything_that_is_not_a_runtime_state():
    u.write_harness_unit(rendered())
    assert u.own_unit(object(), CONFIG_DIR, interpreter=INTERPRETER, home=HOME) is None
    assert u.own_unit(None, CONFIG_DIR, interpreter=INTERPRETER, home=HOME) is None
    shown = u.UnitState("localharness-llm.service", "loaded", str(u.harness_unit_path()), "enabled", "")
    assert u.own_unit(shown, CONFIG_DIR, interpreter=INTERPRETER, home=HOME) is None


def test_own_unit_keeps_every_field_and_sets_own():
    u.write_harness_unit(rendered())
    plain = u.UnitRuntimeState("abc", 2, main_pid=7, active_state="active", exec_main_status=0,
                               fragment_path=str(u.harness_unit_path()))
    own = u.own_unit(plain, CONFIG_DIR, interpreter=INTERPRETER, home=HOME)
    assert own is not None and own.own is True
    assert own == u.UnitRuntimeState("abc", 2, main_pid=7, active_state="active", exec_main_status=0,
                                     fragment_path=str(u.harness_unit_path()), own=True)
    assert plain.own is False


# ------------------------------------------------------------------------------ the real runner


@pytest.mark.real_manager_seam
class TestTheRealRunner:
    """`_run_manager` for real, with a fake `subprocess.run` and a tmp `MANAGER_DIRS`. The file named for
    the program is an empty executable that the fake never runs."""

    @pytest.fixture
    def bin_dir(self, tmp_path, monkeypatch) -> Path:
        directory = tmp_path / "sysbin"
        directory.mkdir()
        monkeypatch.setattr(u, "MANAGER_DIRS", (str(directory),))
        return directory

    @pytest.fixture
    def env_root(self, tmp_path, monkeypatch) -> Path:
        root = tmp_path / "run-user"
        (root / str(os.getuid())).mkdir(parents=True)
        monkeypatch.setattr(u, "RUNTIME_DIR_ROOT", str(root))
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        return root

    @staticmethod
    def program(directory: Path, name: str = "systemctl", *, mode: int = 0o755) -> Path:
        path = directory / name
        path.write_bytes(b"")
        path.chmod(mode)
        return path

    @staticmethod
    def faked_run(monkeypatch, *, code=0, out=b"", err=b"", raises=None):
        calls: list[tuple[list[str], dict]] = []

        def fake_run(cmd, **kw):
            calls.append((list(cmd), kw))
            if raises is not None:
                raise raises
            return subprocess.CompletedProcess(cmd, code, stdout=out, stderr=err)

        monkeypatch.setattr(u.subprocess, "run", fake_run)
        return calls

    def test_the_program_is_the_system_dir_one_with_a_closed_stdin_and_the_runtime_dir(
            self, bin_dir, env_root, monkeypatch):
        exe = self.program(bin_dir)
        calls = self.faked_run(monkeypatch, out=b"ok\n")
        result = u._run_manager(["systemctl", "--user", "daemon-reload"], timeout=5, cap=100)
        assert result == u.ManagerResult(0, "ok\n", "")
        (cmd, kw), = calls
        assert cmd == [str(exe), "--user", "daemon-reload"]
        assert kw["stdin"] is subprocess.DEVNULL and kw["timeout"] == 5 and kw["check"] is False
        assert kw["capture_output"] is True and "shell" not in kw
        assert kw["env"]["XDG_RUNTIME_DIR"] == str(env_root / str(os.getuid()))

    def test_a_missing_program_is_none_and_nothing_runs(self, bin_dir, monkeypatch):
        calls = self.faked_run(monkeypatch)
        assert u._run_manager(["systemctl", "--user", "show"], timeout=5, cap=100) is None
        assert calls == []

    def test_a_program_that_is_not_executable_is_missing(self, bin_dir, monkeypatch):
        self.program(bin_dir, mode=0o644)
        calls = self.faked_run(monkeypatch)
        assert u._run_manager(["systemctl", "--user", "show"], timeout=5, cap=100) is None
        assert calls == []

    def test_a_directory_named_for_the_program_is_missing(self, bin_dir, monkeypatch):
        (bin_dir / "systemctl").mkdir()
        calls = self.faked_run(monkeypatch)
        assert u._run_manager(["systemctl", "--user", "show"], timeout=5, cap=100) is None
        assert calls == []

    def test_a_program_on_path_or_in_cwd_is_never_found(self, bin_dir, tmp_path, monkeypatch):
        planted = tmp_path / "planted"
        planted.mkdir()
        self.program(planted)
        monkeypatch.setenv("PATH", str(planted))
        monkeypatch.chdir(planted)
        calls = self.faked_run(monkeypatch)
        assert u.systemctl_path() is None
        assert u._run_manager(["systemctl", "--user", "show"], timeout=5, cap=100) is None
        assert calls == []

    @pytest.mark.parametrize("name", ["/bin/systemctl", "../systemctl", "./systemctl", "", "sub/systemctl"])
    def test_a_name_with_a_separator_is_not_a_program(self, bin_dir, name, monkeypatch):
        self.program(bin_dir)
        calls = self.faked_run(monkeypatch)
        assert u._run_manager([name, "--user", "show"], timeout=5, cap=100) is None
        assert u._run_manager([], timeout=5, cap=100) is None
        assert calls == []

    def test_the_first_system_dir_with_the_program_wins(self, tmp_path, monkeypatch):
        first, second = tmp_path / "a", tmp_path / "b"
        first.mkdir()
        second.mkdir()
        monkeypatch.setattr(u, "MANAGER_DIRS", (str(first), str(second)))
        later = self.program(second)
        assert u.systemctl_path() == str(later)
        earlier = self.program(first)
        assert u.systemctl_path() == str(earlier)

    def test_the_journal_program_is_found_the_same_way(self, bin_dir):
        assert u.journalctl_path() is None
        journal = self.program(bin_dir, "journalctl")
        assert u.journalctl_path() == str(journal)

    def test_the_default_directories_are_the_two_system_ones(self):
        assert u.MANAGER_DIRS == ("/usr/bin", "/bin")

    def test_a_timeout_is_none(self, bin_dir, monkeypatch):
        self.program(bin_dir)
        self.faked_run(monkeypatch, raises=subprocess.TimeoutExpired("systemctl", 5))
        assert u._run_manager(["systemctl", "--user", "show"], timeout=5, cap=100) is None

    @pytest.mark.parametrize("exc", [OSError("no exec"), ValueError("embedded null")])
    def test_a_program_that_cannot_start_is_none(self, bin_dir, monkeypatch, exc):
        self.program(bin_dir)
        self.faked_run(monkeypatch, raises=exc)
        assert u._run_manager(["systemctl", "--user", "show"], timeout=5, cap=100) is None

    def test_stdout_over_the_cap_is_none(self, bin_dir, monkeypatch):
        self.program(bin_dir)
        self.faked_run(monkeypatch, out=b"x" * 101)
        assert u._run_manager(["systemctl", "--user", "show"], timeout=5, cap=100) is None

    def test_stdout_at_the_cap_is_kept(self, bin_dir, monkeypatch):
        self.program(bin_dir)
        self.faked_run(monkeypatch, out=b"x" * 100)
        assert u._run_manager(["systemctl", "--user", "show"], timeout=5, cap=100).out == "x" * 100

    def test_a_non_zero_exit_is_an_answer(self, bin_dir, monkeypatch):
        self.program(bin_dir)
        self.faked_run(monkeypatch, code=5, out=b"o", err=b"Unit not found.\n")
        assert u._run_manager(["systemctl", "--user", "stop", "x"], timeout=5, cap=100) == \
            u.ManagerResult(5, "o", "Unit not found.\n")

    def test_stderr_is_kept_to_its_own_small_cap(self, bin_dir, monkeypatch):
        self.program(bin_dir)
        self.faked_run(monkeypatch, err=b"e" * (u.MANAGER_STDERR_CAP + 50))
        assert len(u._run_manager(["systemctl", "--user", "show"], timeout=5, cap=100).err) == \
            u.MANAGER_STDERR_CAP

    def test_undecodable_bytes_are_replaced(self, bin_dir, monkeypatch):
        self.program(bin_dir)
        self.faked_run(monkeypatch, out=b"a\xffb")
        assert u._run_manager(["systemctl", "--user", "show"], timeout=5, cap=100).out == "a�b"

    def test_the_state_reader_goes_through_the_real_runner(self, bin_dir, env_root, monkeypatch):
        exe = self.program(bin_dir)
        calls = self.faked_run(monkeypatch, out=show().encode())
        assert u.unit_runtime_state().main_pid == 4242
        (cmd, kw), = calls
        assert cmd == [str(exe), *SHOW_ARGV[1:]]


@pytest.mark.real_manager_seam
class TestTheEnvironment:
    @pytest.fixture
    def root(self, tmp_path, monkeypatch) -> Path:
        root = tmp_path / "run-user"
        root.mkdir()
        monkeypatch.setattr(u, "RUNTIME_DIR_ROOT", str(root))
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        return root

    def test_an_unset_runtime_dir_is_filled_in_from_the_owned_directory(self, root):
        (root / str(os.getuid())).mkdir()
        assert u.manager_env()["XDG_RUNTIME_DIR"] == str(root / str(os.getuid()))

    def test_an_empty_runtime_dir_counts_as_unset(self, root, monkeypatch):
        (root / str(os.getuid())).mkdir()
        monkeypatch.setenv("XDG_RUNTIME_DIR", "")
        assert u.manager_env()["XDG_RUNTIME_DIR"] == str(root / str(os.getuid()))

    def test_a_set_value_is_kept_as_it_is(self, root, monkeypatch):
        (root / str(os.getuid())).mkdir()
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/somewhere/else")
        assert u.manager_env()["XDG_RUNTIME_DIR"] == "/somewhere/else"

    def test_an_absent_directory_is_not_added(self, root):
        assert "XDG_RUNTIME_DIR" not in u.manager_env()

    def test_a_directory_owned_by_another_uid_is_not_added(self, root, monkeypatch):
        directory = root / str(os.getuid())
        directory.mkdir()
        real = os.stat(directory)
        other = types.SimpleNamespace(st_mode=real.st_mode, st_uid=os.getuid() + 1)
        monkeypatch.setattr(u.os, "stat", lambda path, *a, **k: other)
        assert "XDG_RUNTIME_DIR" not in u.manager_env()

    def test_a_file_in_place_of_the_directory_is_not_added(self, root):
        (root / str(os.getuid())).write_text("x")
        assert "XDG_RUNTIME_DIR" not in u.manager_env()

    def test_the_rest_of_the_environment_is_carried(self, root, monkeypatch):
        monkeypatch.setenv("LH_PROBE_VAR", "kept")
        assert u.manager_env()["LH_PROBE_VAR"] == "kept"


def test_fake_manager_is_the_runner_shape(fake_manager):
    assert isinstance(fake_manager, FakeManager)
    assert u._run_manager(["systemctl", "--user", "x"], timeout=1, cap=1) is None
    fake_manager.answer_prefix("systemctl", "--user", out="fine", code=3, err="bad")
    assert u._run_manager(["systemctl", "--user", "x"], timeout=1, cap=100, extra=True) == \
        u.ManagerResult(3, "fine", "bad")
