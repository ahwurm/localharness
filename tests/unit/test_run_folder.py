"""The run folder: lifecycle, sweep, fallback, end, and who begins and ends it."""
import asyncio
import errno
import fcntl
import os
import re
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import pytest

from localharness.core import run_folder
from localharness.core.run_folder import begin, end, fallback_cause, foreign_count, marker, scratch_of, sweep

pytestmark = pytest.mark.skipif(os.name != "posix", reason="descriptor-relative run folders")

HEX = re.compile(r"[0-9a-f]{12}")
OTHER = "f" * 32


@pytest.fixture
def cfg(tmp_path):
    d = tmp_path / "cfg"
    d.mkdir()
    (d / "config.yaml").write_text("{}")
    return d


@pytest.fixture
def temp(tmp_path, monkeypatch):
    d = tmp_path / "tmp"
    d.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(d))
    return d


def _ids(cfg):
    return sorted(p.name for p in (cfg / "run").iterdir() if p.name != ".gitignore")


def test_begin_makes_the_folder_and_end_removes_it(cfg):
    run = begin(cfg)
    assert HEX.fullmatch(run.path.name) and run.path.parent == (cfg / "run").resolve()
    assert sorted(p.name for p in run.path.iterdir()) == [".marker", "scratch", "t"]
    assert (run.path / ".marker").read_text() == marker()
    assert (cfg / "run" / ".gitignore").read_text().strip() == "*"
    assert oct(run.path.stat().st_mode & 0o777) == "0o700"
    end(run)
    assert _ids(cfg) == []


def test_begin_private_makes_state_and_current_and_a_normal_begin_does_not(cfg):
    run = begin(cfg, private=True)
    assert run.private and run.state.is_dir() and oct(run.state.stat().st_mode & 0o777) == "0o700"
    assert run_folder.current() is run
    normal = begin(cfg)
    assert not normal.private and not normal.state.exists() and run_folder.current() is normal
    end(run)
    end(normal)


def test_end_clears_current_only_while_it_is_that_run(cfg):
    first, second = begin(cfg), begin(cfg)
    end(first)
    assert run_folder.current() is second
    end(second)
    assert run_folder.current() is None


def test_a_private_fallback_run_has_state_too(cfg, temp, monkeypatch):
    monkeypatch.setattr(run_folder, "_enforced", lambda name, parent: False)
    run = begin(cfg, private=True)
    assert run.parent is None and run.private and run.state.is_dir()
    end(run)
    assert not run.path.exists()


@pytest.mark.parametrize("private,waits", [(True, True), (False, False)])
def test_a_private_end_waits_longer_for_a_slow_removal(cfg, monkeypatch, private, waits):
    run = begin(cfg, private=private)
    monkeypatch.setattr(run_folder, "END_WAIT_S", 1.2)
    monkeypatch.setattr(run_folder, "PRIVATE_END_WAIT_S", 2.4)
    monkeypatch.setattr(run_folder, "_finish", lambda r: (time.sleep(1.8), ())[1])
    t0 = time.monotonic()
    end(run)
    assert (time.monotonic() - t0 > 1.7) is waits


def test_no_config_no_run_folder(tmp_path):
    assert begin(tmp_path) is None
    assert not (tmp_path / "run").exists()


def test_gitignore_survives_two_begins(cfg):
    a = begin(cfg)
    first = (cfg / "run" / ".gitignore").stat()
    b = begin(cfg)
    assert (cfg / "run" / ".gitignore").stat().st_ino == first.st_ino
    end(a)
    end(b)


def test_scratch_of_and_command_env(cfg):
    run = begin(cfg)
    try:
        assert scratch_of(run, None) == run.scratch
        assert scratch_of(run, "x") is None and scratch_of(None, None) is None
        assert run.command_env() == {"LOCALHARNESS_SCRATCH": str(run.path / "scratch"), "TMPDIR": str(run.path / "t")}
        assert run.command_env("x") == {}
    finally:
        end(run)


def _make(cfg, name, *, mark=None, scratch=False, deleting=False):
    d = cfg / "run" / (f".deleting-{name}" if deleting else name)
    d.mkdir(parents=True)
    if mark:
        (d / ".marker").write_text(mark)
    if scratch:
        (d / "scratch").mkdir()
        (d / "scratch" / "f").write_text("f")
    return d


@pytest.mark.parametrize("kind,gone", [
    ("ours", True), ("empty", True), ("unmarked-with-scratch", False), ("another-machine", False),
    ("deleting", True),
])
def test_sweep_table(cfg, kind, gone):
    name = "0123456789ab"
    d = {
        "ours": lambda: _make(cfg, name, mark=marker(), scratch=True),
        "empty": lambda: _make(cfg, name),
        "unmarked-with-scratch": lambda: _make(cfg, name, scratch=True),
        "another-machine": lambda: _make(cfg, name, mark=OTHER, scratch=True),
        "deleting": lambda: _make(cfg, name, deleting=True, scratch=True),
    }[kind]()
    sweep(cfg)
    assert d.exists() is not gone


def test_sweep_unlinks_files_with_the_names_and_leaves_the_rest(cfg):
    run = cfg / "run"
    run.mkdir()
    (run / "0123456789ab").write_text("x")
    (run / ".deleting-0123456789ab").write_text("x")
    (run / "keep-me").mkdir()
    (run / ".gitignore").write_text("*\n")
    (run / "0123456789AB").mkdir()
    sweep(cfg)
    assert sorted(p.name for p in run.iterdir()) == [".gitignore", "0123456789AB", "keep-me"]


def test_a_live_folder_survives_even_renamed(cfg):
    run = begin(cfg)
    try:
        gone = run.path.parent / f".deleting-{run.path.name}"
        os.rename(run.path, gone)
        sweep(cfg)
        assert gone.exists() and (gone / "scratch").exists()
    finally:
        os.rename(gone, run.path)
        end(run)


def test_a_lock_failing_otherwise_leaves_the_folder(cfg, monkeypatch):
    d = _make(cfg, "0123456789ab", mark=marker(), scratch=True)

    def refuse(fd):
        raise OSError(errno.ENOLCK, "no locks")

    monkeypatch.setattr(run_folder, "_lock", refuse)
    sweep(cfg)
    assert d.exists()


def test_begin_sweeps_what_a_dead_session_left(cfg):
    dead = _make(cfg, "0123456789ab", mark=marker(), scratch=True)
    run = begin(cfg)
    try:
        assert not dead.exists()
    finally:
        end(run)


_made: list[str] = []


def _audit(event, args):
    if event == "os.mkdir" and isinstance(args[0], str) and HEX.fullmatch(args[0]):
        _made.append(args[0])


sys.addaudithook(_audit)


def test_only_begin_makes_a_run_folder_name(cfg):
    _make(cfg, "0123456789ab", mark=marker(), scratch=True)
    _made.clear()
    run = begin(cfg)
    sweep(cfg)
    end(run)
    assert _made == [run.path.name]


def test_a_lock_that_is_not_enforced_uses_the_temp_folder(cfg, temp, monkeypatch):
    monkeypatch.setattr(fcntl, "flock", lambda fd, op: None)
    run = begin(cfg)
    assert run.path.parent == temp.resolve() or run.path.parent == temp
    assert run.path.name.startswith("localharness-run-") and run.parent is None
    assert [p for p in (cfg / "run").iterdir() if HEX.fullmatch(p.name)] == []
    assert (run.path / "scratch").is_dir() and (run.path / "t").is_dir()
    end(run)
    assert not run.path.exists()


def test_a_linked_run_dir_uses_the_temp_folder(cfg, temp, tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    (cfg / "run").symlink_to(other)
    run = begin(cfg)
    assert run.path.name.startswith("localharness-run-")
    assert list(other.iterdir()) == []
    end(run)
    assert not run.path.exists()


def test_a_run_dir_of_another_account_uses_the_temp_folder(cfg, temp, monkeypatch):
    (cfg / "run").mkdir()
    real = os.geteuid()
    calls = iter([real + 1])
    monkeypatch.setattr(os, "geteuid", lambda: next(calls, real))
    run = begin(cfg)
    assert run.path.name.startswith("localharness-run-")
    end(run)


def test_end_is_latched_and_removes_once(cfg, monkeypatch):
    run = begin(cfg)
    real, calls = run_folder.empty_dir, []
    gate = threading.Event()

    def slow(fd, label="."):
        calls.append(label)
        gate.wait(5)
        return real(fd, label)

    monkeypatch.setattr(run_folder, "empty_dir", slow)
    first = threading.Thread(target=end, args=(run,))
    first.start()
    time.sleep(0.2)
    threading.Timer(0.3, gate.set).start()
    end(run)
    first.join(5)
    assert len(calls) == 1 and not run.path.exists()


def test_end_gives_up_after_two_seconds_and_the_next_start_finishes(cfg, monkeypatch):
    run = begin(cfg)
    gate = threading.Event()

    def stuck(fd, label="."):
        gate.wait(10)
        raise OSError(errno.EIO, "stuck")

    monkeypatch.setattr(run_folder, "empty_dir", stuck)
    said = []
    t0 = time.monotonic()
    end(run, said.append)
    assert time.monotonic() - t0 < 3
    assert said == [run_folder.REMOVING_LINE]
    left = cfg / "run" / f".deleting-{run.path.name}"
    assert left.exists()
    gate.set()
    with pytest.raises(OSError):
        run._ender.result(timeout=5)
    nxt = begin(cfg)
    try:
        assert not left.exists()
    finally:
        end(nxt)


def test_doctor_names_a_fallback_and_foreign_folders(cfg, tmp_path, capsys):
    from localharness.cli import doctor_cmd

    other = tmp_path / "elsewhere"
    other.mkdir()
    (cfg / "run").symlink_to(other)
    assert fallback_cause(cfg) == "a link"
    doctor_cmd._print_run_folder(cfg)
    out = capsys.readouterr().out
    assert "run/ is a link: sessions use" in out and "use --config-dir <local folder>" in out
    (cfg / "run").unlink()
    _make(cfg, "0123456789ab", mark=OTHER, scratch=True)
    _make(cfg, "ba9876543210", scratch=True)
    _make(cfg, "aaaaaaaaaaaa")
    assert foreign_count(cfg) == 2 and fallback_cause(cfg) is None
    doctor_cmd._print_run_folder(cfg)
    assert "2 run folders of another machine: left alone; exclude run/ from the sync" in capsys.readouterr().out


def test_start_app_makes_a_run_the_bash_tool_sees_and_removes_it(cfg, monkeypatch):
    from localharness.cli import start_cmd
    from localharness.tools.builtin.bash_tool import BashExecTool

    seen = {}

    async def fake(*args, run=None, **kw):
        seen["run"] = run
        seen["listing"] = sorted(p.name for p in run.path.iterdir())
        res = await BashExecTool(run=run)._execute(command='echo "$LOCALHARNESS_SCRATCH $TMPDIR"')
        seen["env"] = res.output.strip()

    monkeypatch.setattr(start_cmd, "_start_async", fake)
    monkeypatch.setattr(start_cmd, "_models_that_arrived", lambda *a: None)
    start_cmd.start_app(config_dir=str(cfg))
    run = seen["run"]
    assert seen["listing"] == [".marker", "scratch", "t"]
    assert seen["env"] == f"{run.path}/scratch {run.path}/t"
    assert not run.path.exists()


def test_start_app_makes_none_for_list_models_or_no_config(tmp_path, cfg, monkeypatch):
    from localharness.cli import start_cmd

    got = []

    async def fake(*args, run="unset", **kw):
        got.append(run)

    monkeypatch.setattr(start_cmd, "_start_async", fake)
    monkeypatch.setattr(start_cmd, "_models_that_arrived", lambda *a: None)
    start_cmd.start_app(config_dir=str(cfg), list_models=True)
    assert got == [None] and not (cfg / "run").exists()


def test_acp_session_begins_and_ends_its_own_run(cfg, monkeypatch):
    from localharness.channels.acp import AcpChannel
    from localharness.cli import start_cmd

    seen = {}

    async def fake(**kw):
        seen["run"] = kw["run"]
        seen["there"] = kw["run"].path.exists()

    monkeypatch.setattr(start_cmd, "_start_async", fake)
    ch = AcpChannel(config_dir=str(cfg))
    asyncio.run(ch._run_session())
    assert seen["there"] and not seen["run"].path.exists()


def _mobile(monkeypatch, cfg, outcomes, script):
    """Drive `_serve` with a fake server: `script(callbacks, runs)` stands for the phone."""
    import uvicorn

    from localharness.channels.mobile import server as server_mod
    from localharness.cli import mobile_cmd, start_cmd

    callbacks, runs, threads = {}, [], []

    class FakeServer:
        app = None

        def __init__(self, channel, **kw):
            callbacks.update(kw)
            callbacks["channel"] = channel

        def session_over(self):
            pass

    async def fake_start(*args, run=None, **kw):
        runs.append(run)
        outcome = outcomes.pop(0)
        if outcome == "block":
            await asyncio.Event().wait()
        return outcome

    async def fake_serve(self):
        await script(callbacks, runs)

    real_begin = run_folder.begin

    def traced(*a):
        threads.append(threading.current_thread())
        return real_begin(*a)

    monkeypatch.setattr(server_mod, "MobileServer", FakeServer)
    monkeypatch.setattr(uvicorn.Server, "serve", fake_serve)
    monkeypatch.setattr(start_cmd, "_start_async", fake_start)
    monkeypatch.setattr(run_folder, "begin", traced)
    asyncio.run(mobile_cmd._serve(
        config_dir=str(cfg), host="127.0.0.1", port=1, token="t" * 8, ui_dir=None, replay=None,
        fixtures=None, speed=1.0, verbose=False, agent=None, sleep_after=0))
    assert all(t is not threading.main_thread() for t in threads)
    return runs


async def _until(cond, timeout=5):
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.02)


def test_mobile_ends_a_run_when_its_session_ends(cfg, monkeypatch):
    async def script(cb, runs):
        cb["on_first_message"]()
        await _until(lambda: runs and not runs[0].path.exists())

    runs = _mobile(monkeypatch, cfg, [None], script)
    assert len(runs) == 1 and not runs[0].path.exists()


def test_mobile_keeps_a_sleeping_chats_run_and_ends_it_on_a_new_chat(cfg, monkeypatch):
    asleep = SimpleNamespace(resume=SimpleNamespace(slept=True))
    gone = {}

    async def script(cb, runs):
        cb["on_first_message"]()
        await _until(lambda: runs and cb["channel"].asleep)
        await asyncio.sleep(0.1)
        gone["while_asleep"] = runs[0].path.exists()
        await cb["on_new_session"]()
        await _until(lambda: len(runs) == 2)
        gone["old_after_new"] = runs[0].path.exists()

    runs = _mobile(monkeypatch, cfg, [asleep, "block"], script)
    assert gone == {"while_asleep": True, "old_after_new": False}
    assert runs[1] is not runs[0] and not runs[1].path.exists()


def test_mobile_ends_a_sleeping_chats_run_when_the_server_stops(cfg, monkeypatch):
    asleep = SimpleNamespace(resume=SimpleNamespace(slept=True))

    async def script(cb, runs):
        cb["on_first_message"]()
        await _until(lambda: runs and cb["channel"].asleep)

    runs = _mobile(monkeypatch, cfg, [asleep], script)
    assert not runs[0].path.exists()


def test_mobile_sweeps_when_the_server_starts(cfg, monkeypatch):
    dead = _make(cfg, "0123456789ab", mark=marker(), scratch=True)

    async def script(cb, runs):
        assert not dead.exists()

    _mobile(monkeypatch, cfg, [], script)
