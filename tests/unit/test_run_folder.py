"""The run folder: lifecycle, sweep, fallback, end, and who begins and ends it."""
import asyncio
import errno
import fcntl
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import pytest

from localharness.core import run_folder
from localharness.core.run_folder import begin, command_env, end, fallback_cause, foreign_count, marker, scratch_of, sweep

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
def proj(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
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
    assert sorted(p.name for p in run.path.iterdir()) == [".gitignore", ".marker", "scratch", "t"]
    assert (run.path / ".gitignore").read_text() == "*\n"
    assert (run.path / ".marker").read_text() == marker()
    assert (cfg / "run" / ".gitignore").read_text().strip() == "*"
    assert oct(run.path.stat().st_mode & 0o777) == "0o700"
    end(run)
    assert _ids(cfg) == []


def test_begin_private_makes_state_and_current_and_a_normal_begin_does_not(cfg):
    run = begin(cfg, private=True)
    assert run.private and run.state.is_dir() and oct(run.state.stat().st_mode & 0o777) == "0o700"
    assert run_folder.current() is run and run_folder.is_private()
    normal = begin(cfg)
    assert not normal.private and not normal.state.exists() and run_folder.current() is normal
    assert not run_folder.is_private()
    end(run)
    end(normal)
    assert not run_folder.is_private()


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


def test_a_fallback_under_a_linked_temp_folder_names_its_real_path(tmp_path, monkeypatch):
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "link"))
    run = run_folder._fallback()
    assert run.path.parent == (tmp_path / "real").resolve() and run.scratch.is_dir()
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


def test_scratch_of_and_command_env(cfg, proj):
    run, kept = begin(cfg), begin(cfg, home=proj / ".localharness")
    try:
        assert scratch_of(run, None) == run.scratch
        assert scratch_of(run, "x") is None and scratch_of(None, None) is None
        assert scratch_of(kept, str(proj)) == scratch_of(kept, f"{proj}/../proj") == kept.scratch
        assert scratch_of(kept, str(proj / "sub")) is None and scratch_of(run, str(proj)) is None
        assert command_env(kept, str(proj)) == {"LOCALHARNESS_COMMIT": "1", "LOCALHARNESS_SCRATCH": str(kept.scratch),
                                                "TMPDIR": str(kept.path / "t")}
        assert command_env(run, str(proj)) == {"LOCALHARNESS_COMMIT": "1"}
        assert command_env(run) == {"LOCALHARNESS_COMMIT": "1", "LOCALHARNESS_SCRATCH": str(run.path / "scratch"),
                                    "TMPDIR": str(run.path / "t")}
        assert command_env(run, "x") == command_env(None) == {"LOCALHARNESS_COMMIT": "1"}
    finally:
        end(run)
        end(kept)


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
    calls = iter([real, real + 1])
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


def _slow_finish(monkeypatch, seconds, left=()):
    real = run_folder._finish

    def slow(run):
        time.sleep(seconds)
        real(run)
        return left

    monkeypatch.setattr(run_folder, "_finish", slow)
    monkeypatch.setattr(run_folder, "_SLOW_AFTER_S", 0.05)
    monkeypatch.setattr(run_folder, "END_WAIT_S", 0.15)
    monkeypatch.setattr(run_folder, "PRIVATE_END_WAIT_S", 0.15)


def test_end_returns_an_empty_line_for_a_clean_removal(cfg):
    said = []
    assert end(begin(cfg, private=True), said.append) == "" and said == []
    assert end(None) == ""


def test_end_returns_and_says_the_line_naming_what_is_left(cfg, monkeypatch):
    run = begin(cfg)
    _slow_finish(monkeypatch, 0, left=("/x/y",))
    said = []
    line = end(run, said.append)
    assert line == "Scratch not fully removed: /x/y stays until the next start" and said == [line]
    run = begin(cfg, private=True)
    assert end(run, said.append) == "Private session not fully removed: /x/y stays until localharness next starts"


def test_a_private_end_that_hits_its_bound_names_the_folder_left_behind(cfg, monkeypatch):
    run = begin(cfg, private=True)
    _slow_finish(monkeypatch, 0)
    real = run_folder.empty_dir
    monkeypatch.setattr(run_folder, "empty_dir", lambda fd, label: time.sleep(0.4) or real(fd, label))
    said = []
    line = end(run, said.append)
    gone = run.path.with_name(f".deleting-{run.path.name}")
    assert line == f"Private session not fully removed: {gone} stays until localharness next starts" and gone.is_dir()
    assert said == ["Removing this private session… (Ctrl+C leaves it for the next start)", line]
    time.sleep(0.5)


def test_a_normal_end_that_hits_its_bound_stays_silent(cfg, monkeypatch):
    run = begin(cfg)
    _slow_finish(monkeypatch, 0.4)
    said = []
    assert end(run, said.append) == "" and said == [run_folder.REMOVING_LINE]
    time.sleep(0.5)


def test_a_second_end_returns_the_first_ends_outcome(cfg, monkeypatch):
    run = begin(cfg)
    _slow_finish(monkeypatch, 0.1, left=("/x/y",))
    got = []
    first = threading.Thread(target=lambda: got.append(end(run)))
    first.start()
    got.append(end(run))
    first.join(5)
    assert got == ["Scratch not fully removed: /x/y stays until the next start"] * 2


@pytest.mark.parametrize("cause, native, line", [
    (run_folder.LINKED, True, run_folder.PRIVATE_LINKED_RUN_LINE), (run_folder.FOREIGN, True, run_folder.PRIVATE_LINKED_RUN_LINE),
    (run_folder.UNLOCKED, True, run_folder.PRIVATE_UNLOCKED_LINE), (None, False, run_folder.PRIVATE_NEEDS_POSIX_LINE)])
def test_private_run_problem_names_the_cause_and_leaves_ending_the_run_to_its_caller(cfg, monkeypatch, cause, native, line):
    ended = []
    monkeypatch.setattr(run_folder, "end", lambda run, say=None: ended.append(run))
    monkeypatch.setattr(run_folder, "fallback_cause", lambda c: cause)
    monkeypatch.setattr(run_folder, "native", lambda: native)
    fallback = run_folder.RunFolder(cfg / "x", private=True)
    assert run_folder.private_run_problem(fallback, cfg) == line.format(path=cfg / "run")
    assert run_folder.private_run_problem(None, cfg) == run_folder.PRIVATE_NO_RUN_LINE
    assert run_folder.private_run_problem(run_folder.RunFolder(cfg / "y", parent=3, private=True), cfg) is None
    assert ended == []


def test_doctor_names_a_fallback_and_foreign_folders(cfg, tmp_path, capsys):
    from localharness.cli import doctor_cmd

    other = tmp_path / "elsewhere"
    other.mkdir()
    (cfg / "run").symlink_to(other)
    assert fallback_cause(cfg) == run_folder.LINKED == "a link"
    doctor_cmd._print_run_folder(cfg)
    out = capsys.readouterr().out
    assert f"{cfg.resolve()}/run is a link: sessions use" in out and "use --config-dir <local folder>" in out
    (cfg / "run").unlink()
    _make(cfg, "0123456789ab", mark=OTHER, scratch=True)
    _make(cfg, "ba9876543210", scratch=True)
    _make(cfg, "aaaaaaaaaaaa")
    assert foreign_count(cfg) == 2 and fallback_cause(cfg) is None
    doctor_cmd._print_run_folder(cfg)
    assert f"2 run folders of another machine in {cfg.resolve()}/run: left alone; exclude it from the sync" in " ".join(
        capsys.readouterr().out.split())


def test_doctor_names_a_projects_run_problem_and_where_sessions_go(cfg, proj, tmp_path, capsys):
    from localharness.cli import doctor_cmd

    home = proj / ".localharness"
    home.mkdir()
    other = tmp_path / "elsewhere"
    other.mkdir()
    (home / "run").symlink_to(other)
    _make(cfg, "0123456789ab", mark=OTHER, scratch=True)
    doctor_cmd._print_run_folder(cfg, home)
    out = " ".join(capsys.readouterr().out.split())
    assert f"{home}/run is a link: sessions here keep their run folder in {cfg.resolve()}/run" in out
    assert "sessions use" not in out
    assert f"1 run folders of another machine in {cfg.resolve()}/run: left alone" in out
    (home / "run").unlink()
    doctor_cmd._print_run_folder(cfg, home)
    out = " ".join(capsys.readouterr().out.split())
    assert "is a link" not in out and "1 run folders of another machine" in out


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
    assert seen["listing"] == [".gitignore", ".marker", "scratch", "t"]
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


def _mobile(monkeypatch, cfg, outcomes, script, explicit=True):
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
        config_dir=str(cfg) if explicit else None, host="127.0.0.1", port=1, token="t" * 8, ui_dir=None,
        replay=None, fixtures=None, speed=1.0, verbose=False, agent=None, sleep_after=0))
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


def _listing(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def test_a_project_run_lives_in_its_dot_folder_made_on_demand(cfg, proj):
    home = proj / ".localharness"
    run = begin(cfg, home=home)
    try:
        assert run.path.parent == home.resolve() / "run" and HEX.fullmatch(run.path.name)
        assert oct(home.stat().st_mode & 0o777) == "0o700" and (home / "run" / ".gitignore").read_text() == "*\n"
        assert sorted(p.name for p in run.path.iterdir()) == [".gitignore", ".marker", "scratch", "t"]
        assert (run.path / ".gitignore").read_text() == "*\n"
        assert not (cfg / "run").exists()
    finally:
        end(run)
    assert _listing(home) == ["run", "run/.gitignore"]


def _plant(proj, shape, outside):
    home = proj / ".localharness"
    kind, what = shape.split("-")
    place = home if kind == "dot" else home / "run"
    if kind == "run":
        home.mkdir()
    {"link": lambda: place.symlink_to(outside), "file": lambda: place.write_text("not a folder")}.get(what, lambda: None)()
    return home, place


@pytest.mark.parametrize("shape", ["dot-link", "dot-file", "run-link", "run-file", "dot-foreign", "run-foreign"])
def test_a_planted_project_dot_folder_or_run_falls_back_to_the_config_folder(cfg, proj, tmp_path, monkeypatch, shape):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("k")
    home, place = _plant(proj, shape, outside)
    if shape.endswith("foreign"):
        real, first = os.geteuid(), 0 if shape.startswith("dot") else 1
        calls = iter(range(2))
        monkeypatch.setattr(os, "geteuid", lambda: real + 1 if next(calls, 9) == first else real)
        home.mkdir(exist_ok=True)
        if shape == "run-foreign":
            place.mkdir()
    run = begin(cfg, home=home)
    try:
        assert run.path.parent == cfg.resolve() / "run"
        assert _listing(outside) == ["keep"] and (outside / "keep").read_text() == "k"
        if shape.endswith("file"):
            assert place.read_text() == "not a folder"
    finally:
        end(run)
    assert _listing(outside) == ["keep"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes anywhere")
@pytest.mark.parametrize("existing", [False, True])
def test_an_unwritable_project_falls_back_to_the_config_folder(cfg, proj, existing):
    home = proj / ".localharness"
    if existing:
        home.mkdir()
    (home if existing else proj).chmod(0o555)
    try:
        run = begin(cfg, home=home)
        assert run.path.parent == cfg.resolve() / "run"
        end(run)
        assert _listing(home) == [] if existing else not home.exists()
    finally:
        (home if existing else proj).chmod(0o755)


def _accounts(monkeypatch, members=(), primary=()):
    import grp
    import pwd
    me = os.geteuid()
    monkeypatch.setattr(grp, "getgrgid", lambda gid: SimpleNamespace(gr_name="team", gr_gid=gid, gr_mem=["me", *members]))
    monkeypatch.setattr(pwd, "getpwall", lambda: [SimpleNamespace(pw_name="me", pw_uid=me, pw_gid=os.getegid()),
                                                  *(SimpleNamespace(pw_name=n, pw_uid=me + 1 + i, pw_gid=os.getegid())
                                                    for i, n in enumerate(primary))])


@pytest.mark.parametrize("where, mode, members, primary", [
    ("root", 0o757, (), ()), ("dot", 0o707, (), ()), ("root", 0o775, ("bob",), ()), ("dot", 0o770, (), ("bob",))])
def test_a_private_session_in_a_project_another_account_can_write_keeps_its_run_in_the_config_folder(
        cfg, proj, monkeypatch, where, mode, members, primary):
    _accounts(monkeypatch, members, primary)
    home = proj / ".localharness"
    home.mkdir()
    (proj if where == "root" else home).chmod(mode)
    run = begin(cfg, private=True, home=home)
    try:
        assert run.path.parent == cfg.resolve() / "run" and list(home.iterdir()) == []
    finally:
        end(run)
    normal = begin(cfg, home=home)
    try:
        assert normal.path.parent == home.resolve() / "run"
    finally:
        end(normal)


def test_a_private_session_in_a_project_writable_only_by_a_user_private_group_keeps_its_run_there(cfg, proj, monkeypatch):
    _accounts(monkeypatch)
    proj.chmod(0o775)
    home = proj / ".localharness"
    run = begin(cfg, private=True, home=home)
    try:
        assert run.path.parent == home.resolve() / "run"
    finally:
        end(run)


@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("before", ["nothing", "dot-folder", "run-with-leftovers"])
def test_a_project_without_enforced_locks_falls_back_to_the_config_folder_and_is_left_as_it_was(
        cfg, proj, monkeypatch, before, private):
    home = proj / ".localharness"
    if before != "nothing":
        home.mkdir()
        (home / "config.yaml").write_text("{}")
    if before == "run-with-leftovers":
        _make(home, "aaaaaaaaaaaa", deleting=True, scratch=True)
        _make(home, "bbbbbbbbbbbb")
        (home / "run").chmod(0o700)
    was = [(p, (proj / p).stat().st_mode) for p in _listing(proj)]
    real, first = run_folder._enforced, iter([True])
    monkeypatch.setattr(run_folder, "_enforced", lambda name, parent: next(first, False) is False and real(name, parent))
    run = begin(cfg, private, home=home)
    try:
        assert run.path.parent == cfg.resolve() / "run"
        assert [(p, (proj / p).stat().st_mode) for p in _listing(proj)] == was
    finally:
        end(run)


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes anywhere")
@pytest.mark.parametrize("shape, cause", [
    ("dot-file", "not a folder"), ("run-file", "not a folder"),
    ("unwritable-root", "not writable"), ("unwritable-dot", "not writable"),
    ("no-locks", run_folder.UNLOCKED)])
def test_doctor_names_a_file_an_unwritable_project_and_no_locks_after_the_fallback(
        cfg, proj, monkeypatch, capsys, shape, cause):
    from localharness.cli import doctor_cmd

    home = proj / ".localharness"
    {"dot-file": lambda: home.write_text("x"), "run-file": lambda: (home.mkdir(), (home / "run").write_text("x")),
     "unwritable-root": lambda: proj.chmod(0o555), "unwritable-dot": lambda: (home.mkdir(), home.chmod(0o555)),
     "no-locks": lambda: monkeypatch.setattr(fcntl, "flock", lambda fd, op: None)}[shape]()
    try:
        if shape != "no-locks":
            end(begin(cfg, home=home))
        assert fallback_cause(home) == cause
        doctor_cmd._print_run_folder(cfg, home)
        assert f"{home}/run is {cause}: sessions here keep their run folder in" in " ".join(capsys.readouterr().out.split())
    finally:
        proj.chmod(0o755)
        if home.is_dir():
            home.chmod(0o755)


def test_a_project_sweep_follows_the_table_and_never_a_link(cfg, proj, tmp_path):
    home = proj / ".localharness"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("k")
    (outside / "dir").mkdir()
    (outside / "dir" / "inner").write_text("i")
    dying = _make(home, "aaaaaaaaaaaa", deleting=True, scratch=True)
    (dying / "evil").symlink_to(outside / "keep")
    (dying / "evil-dir").symlink_to(outside / "dir")
    empty = _make(home, "bbbbbbbbbbbb")
    foreign = _make(home, "cccccccccccc", mark=OTHER, scratch=True)
    linked_marker = _make(home, "dddddddddddd", scratch=True)
    (linked_marker / ".marker").symlink_to(outside / "keep")
    linked_name = home / "run" / "eeeeeeeeeeee"
    linked_name.symlink_to(outside / "dir")
    (home / "run" / ".gitignore").write_text("!*\n")
    run = begin(cfg, home=home)
    try:
        assert not dying.exists() and not empty.exists() and not linked_name.exists() and not linked_name.is_symlink()
        assert foreign.exists() and linked_marker.exists() and foreign_count(home) == 2
        assert _listing(outside) == ["dir", "dir/inner", "keep"] and (outside / "keep").read_text() == "k"
        assert (home / "run" / ".gitignore").read_text() == "!*\n"
    finally:
        end(run)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git on PATH")
def test_a_run_folder_is_ignored_by_git_whatever_run_gitignore_says(cfg, proj):
    subprocess.run(["git", "init", "-q"], cwd=proj, check=True)
    home = proj / ".localharness"
    (home / "run").mkdir(parents=True)
    (home / "run" / ".gitignore").write_text("!*\n")
    run = begin(cfg, home=home)
    try:
        (run.scratch / "out.txt").write_text("x")
        (run.path / "t" / "tmp").write_text("x")
        for path in (run.scratch / "out.txt", run.path / "t" / "tmp", run.path / ".marker"):
            assert subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=proj).returncode == 0
    finally:
        end(run)


def test_a_project_begin_also_sweeps_the_config_folders_run(cfg, proj):
    dead = _make(cfg, "0123456789ab", mark=marker(), scratch=True)
    mine = _make(proj / ".localharness", "ba9876543210", mark=marker(), scratch=True)
    run = begin(cfg, home=proj / ".localharness")
    try:
        assert not dead.exists() and not mine.exists()
    finally:
        end(run)


def test_a_config_folder_that_is_a_link_still_holds_its_run(cfg, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(cfg)
    run = begin(link)
    try:
        assert run.parent is not None and run.path.parent == cfg.resolve() / "run"
        assert run_folder.private_run_problem(run, link) is None
        assert foreign_count(link.resolve()) == 0
    finally:
        end(run)


def test_the_end_line_names_a_project_leftover_from_where_you_stand(cfg, proj, monkeypatch):
    run = begin(cfg, private=True, home=proj / ".localharness")
    _slow_finish(monkeypatch, 0)
    real = run_folder.empty_dir
    monkeypatch.setattr(run_folder, "empty_dir", lambda fd, label: time.sleep(0.4) or real(fd, label))
    monkeypatch.chdir(proj)
    said = []
    line = end(run, said.append)
    assert ".localharness/run/.deleting-" in line and str(proj) not in line
    assert str(proj) not in said[-1]
    time.sleep(0.5)
    assert run_folder._shown(proj / ".localharness" / "run" / "x") == ".localharness/run/x"
    monkeypatch.chdir(cfg)
    assert run_folder._shown(proj / ".localharness" / "run" / "x") == str(proj / ".localharness" / "run" / "x")


@pytest.fixture
def lived_in(tmp_path, fake_home, monkeypatch):
    """No config folder chosen: a fake home holding the config, and a project to stand in."""
    home = fake_home()
    config = home / ".localharness"
    config.mkdir()
    (config / "config.yaml").write_text("{}")
    project = tmp_path / "work"
    project.mkdir()
    monkeypatch.chdir(project)
    return SimpleNamespace(cfg=config, proj=project, dot=project.resolve() / ".localharness")


def test_start_app_keeps_a_projects_run_in_its_dot_folder(lived_in, monkeypatch):
    from localharness.cli import start_cmd
    from localharness.tools.builtin.bash_tool import BashExecTool

    seen = {}

    async def fake(*args, run=None, **kw):
        seen["run"] = run
        res = await BashExecTool(run=run)._execute(command='echo "$LOCALHARNESS_SCRATCH $TMPDIR"')
        seen["env"] = res.output.strip()

    monkeypatch.setattr(start_cmd, "_start_async", fake)
    monkeypatch.setattr(start_cmd, "_models_that_arrived", lambda *a: None)
    start_cmd.start_app(config_dir=None)
    run = seen["run"]
    assert run.path.parent == lived_in.dot / "run" and seen["env"] == f"{run.path}/scratch {run.path}/t"
    assert not run.path.exists() and _listing(lived_in.dot) == ["run", "run/.gitignore"]
    assert not (lived_in.cfg / "run").exists()


def test_a_session_with_no_project_keeps_its_run_in_the_config_folder(lived_in, tmp_path, monkeypatch):
    from localharness.cli import start_cmd

    chosen = tmp_path / "chosen"
    chosen.mkdir()
    (chosen / "config.yaml").write_text("{}")
    runs = []

    async def fake(*args, run=None, **kw):
        runs.append(run.path.parent)

    monkeypatch.setattr(start_cmd, "_start_async", fake)
    monkeypatch.setattr(start_cmd, "_models_that_arrived", lambda *a: None)
    start_cmd.start_app(config_dir=str(chosen))
    monkeypatch.setenv("LOCALHARNESS_DIR", str(chosen))
    start_cmd.start_app(config_dir=None)
    monkeypatch.delenv("LOCALHARNESS_DIR")
    monkeypatch.chdir(lived_in.cfg.parent)
    start_cmd.start_app(config_dir=None)
    assert runs == [chosen.resolve() / "run", chosen.resolve() / "run", lived_in.cfg.resolve() / "run"]
    assert not lived_in.dot.exists()


def test_acp_session_keeps_its_run_in_the_project(lived_in, monkeypatch):
    from localharness.channels.acp import AcpChannel
    from localharness.cli import start_cmd

    seen = {}

    async def fake(**kw):
        seen["run"] = kw["run"]
        seen["there"] = kw["run"].path.exists()

    monkeypatch.setattr(start_cmd, "_start_async", fake)
    asyncio.run(AcpChannel(config_dir=None)._run_session())
    assert seen["there"] and seen["run"].path.parent == lived_in.dot / "run" and not seen["run"].path.exists()
    assert not (lived_in.cfg / "run").exists()


def test_mobile_keeps_each_chats_run_in_the_project_and_sweeps_both_places(lived_in, monkeypatch):
    dead = [_make(lived_in.dot, "0123456789ab", mark=marker(), scratch=True),
            _make(lived_in.cfg, "ba9876543210", mark=marker(), scratch=True)]
    seen = {}

    async def script(cb, runs):
        seen["swept"] = [not d.exists() for d in dead]
        cb["on_first_message"]()
        await _until(lambda: runs)
        seen["parent"] = runs[0].path.parent
        seen["there"] = runs[0].path.exists()
        await cb["on_new_session"](private=True)
        await _until(lambda: len(runs) == 2)
        seen["private"] = (runs[1].path.parent, runs[1].private, runs[1].state.is_dir())

    runs = _mobile(monkeypatch, lived_in.cfg, ["block", "block"], script, explicit=False)
    assert seen["swept"] == [True, True]
    assert seen["parent"] == lived_in.dot / "run" and seen["there"]
    assert seen["private"] == (lived_in.dot / "run", True, True)
    assert not any(r.path.exists() for r in runs) and _listing(lived_in.dot) == ["run", "run/.gitignore"]
