"""TERM and HUP end a private session: in a start-up question, in the REPL, in teardown, and past a wedged thread."""
from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import textwrap
import time
from contextlib import contextmanager

import pytest
import typer
from typer.testing import CliRunner

from localharness.core import interrupt
from localharness.core.bus import EventBus

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX signals")

CHILD = textwrap.dedent('''
    import asyncio, signal, sys, time
    from pathlib import Path
    from localharness.cli import start_cmd
    from localharness.core import interrupt
    from localharness.core.run_folder import begin, end

    interrupt.DEADLINE_S, interrupt.JOIN_S, start_cmd.STAGE_S = 1.5, 0.5, 0.3
    cfg, mode = Path(sys.argv[1]), sys.argv[2]
    signal.signal(signal.SIGHUP, signal.SIG_IGN if mode == "hup-ignored" else signal.SIG_DFL)
    run = begin(cfg, private=True)
    guard = interrupt.install(("SIGTERM", "SIGHUP"), lambda: end(run))
    ready = lambda: print("ready", flush=True)

    async def main():
        task = asyncio.current_task()
        guard.repl(task)
        try:
            ready()
            if mode == "hup-ignored":
                await asyncio.sleep(1.5)
            else:
                await asyncio.Future()
        finally:
            guard.teardown()
            if mode == "wedged-stage":
                try:
                    async with start_cmd._stage(True):
                        await asyncio.sleep(100)
                except TimeoutError:
                    print("teardown continued", flush=True)
            if mode == "wedged-thread":
                await asyncio.get_running_loop().run_in_executor(None, time.sleep, 1000)

    try:
        if mode == "idle":
            ready()
            input()
        else:
            asyncio.run(main())
    except asyncio.CancelledError:
        pass
    finally:
        end(run)
        guard.restore()
''')


@contextmanager
def child(tmp_path, mode):
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "config.yaml").write_text("{}")
    proc = subprocess.Popen([sys.executable, str(script), str(cfg), mode], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        assert select.select([proc.stdout], [], [], 20)[0], "child never said ready"
        assert proc.stdout.readline().strip() == "ready"
        yield proc, cfg
    finally:
        proc.kill()
        proc.wait(timeout=5)
        proc.stdout.close()
        proc.stdin.close()


def runs(cfg):
    return [p.name for p in (cfg / "run").iterdir() if p.name != ".gitignore" and not p.name.endswith(".lock")]


def finish(proc, timeout=15):
    out = proc.stdout.read()
    return proc.wait(timeout=timeout), out


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_idle_in_a_start_up_question_ends_the_run_then_dies_by_the_signal(tmp_path, sig):
    with child(tmp_path, "idle") as (proc, cfg):
        assert runs(cfg)
        proc.send_signal(sig)
        code, _ = finish(proc)
        assert code == -sig and runs(cfg) == []


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_in_the_repl_the_task_is_cancelled_the_run_ends_and_the_status_is_the_signals(tmp_path, sig):
    with child(tmp_path, "repl") as (proc, cfg):
        proc.send_signal(sig)
        code, _ = finish(proc)
        assert code == -sig and runs(cfg) == []


def test_a_wedged_teardown_stage_times_out_and_teardown_goes_on(tmp_path):
    with child(tmp_path, "wedged-stage") as (proc, cfg):
        proc.send_signal(signal.SIGTERM)
        code, out = finish(proc)
        assert "teardown continued" in out and code == -signal.SIGTERM and runs(cfg) == []


def test_a_blocked_executor_thread_is_cut_by_the_deadline_which_ends_the_run(tmp_path):
    with child(tmp_path, "wedged-thread") as (proc, cfg):
        start = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        code, _ = finish(proc)
        assert code == 128 + signal.SIGTERM and time.monotonic() - start < 10
        assert runs(cfg) == []


def test_a_hangup_set_to_ignore_is_left_alone(tmp_path):
    with child(tmp_path, "hup-ignored") as (proc, cfg):
        proc.send_signal(signal.SIGHUP)
        code, _ = finish(proc)
        assert code == 0 and runs(cfg) == []


def test_install_skips_an_ignored_signal_and_restore_puts_every_handler_back(monkeypatch):
    before = {n: signal.getsignal(n) for n in (signal.SIGTERM, signal.SIGHUP)}
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        guard = interrupt.install(("SIGTERM", "SIGHUP", "SIGNOPE"), lambda: None)
        assert guard._signums == [signal.SIGTERM]
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) == guard._on_idle_signal
        guard.restore()
        assert signal.getsignal(signal.SIGTERM) is before[signal.SIGTERM]
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
    finally:
        for n, handler in before.items():
            signal.signal(n, handler)


async def test_stop_persisting_delivers_later_events_but_appends_none(tmp_path):
    from localharness.core.events import SystemReady

    path = tmp_path / "events.jsonl"
    bus = EventBus(persist_path=path)
    seen = []
    def ready():
        return SystemReady(config_path="c", provider_base_url="u", provider_type="vllm", detected_models=[])

    bus.subscribe(SystemReady, seen.append)
    await bus.publish(ready())
    size = path.stat().st_size
    bus.stop_persisting()
    await bus.publish(ready())
    assert len(seen) == 2 and path.stat().st_size == size


def _invoke(*args):
    from localharness.cli import start_cmd
    app = typer.Typer()
    app.command()(start_cmd.start_app)
    return CliRunner().invoke(app, list(args))


@pytest.fixture
def installs(tmp_path, monkeypatch):
    from localharness.cli import start_cmd

    calls = []

    class Fake:
        def idle(self):
            calls.append("idle")

        def restore(self):
            calls.append("restore")

    async def fake(*args, **kw):
        calls.append(("guard", kw.get("guard")))

    monkeypatch.setattr(start_cmd, "_start_async", fake)
    monkeypatch.setattr(start_cmd, "_models_that_arrived", lambda *a, **k: None)
    monkeypatch.setattr(interrupt, "install", lambda signals, on_idle: calls.append(signals) or Fake())
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "config.yaml").write_text("{}")
    return calls, cfg


def test_start_app_installs_the_guard_only_for_a_private_run(installs):
    calls, cfg = installs
    assert _invoke("--private", "--config-dir", str(cfg)).exit_code == 0
    assert calls[0] == ("SIGTERM", "SIGHUP") and calls[1][1] is not None
    assert calls[-2:] == ["idle", "restore"]
    calls.clear()
    assert _invoke("--config-dir", str(cfg)).exit_code == 0
    assert calls == [("guard", None)]


def test_a_plugins_restart_keeps_one_private_run_and_idles_the_guard_after_every_pass(installs, monkeypatch):
    from types import SimpleNamespace

    from localharness.cli import start_cmd
    from localharness.cli.session_resume import RELOAD_ACTION
    from localharness.core import run_folder

    calls, cfg = installs
    begun, ended = [], []
    real_begin, real_end = run_folder.begin, run_folder.end
    monkeypatch.setattr(run_folder, "begin", lambda *a, **k: begun.append(real_begin(*a, **k)) or begun[-1])
    monkeypatch.setattr(run_folder, "end", lambda run, **k: ended.append(run) or real_end(run, **k))
    restart = SimpleNamespace(action=RELOAD_ACTION, resume=SimpleNamespace(agent_name=None))
    passes = iter([restart, restart, None])

    async def fake(*args, run=None, **kw):
        calls.append(("pass", run))
        return next(passes)

    monkeypatch.setattr(start_cmd, "_start_async", fake)
    assert _invoke("--private", "--config-dir", str(cfg)).exit_code == 0
    assert len(begun) == 1 and ended == begun and not begun[0].path.exists()
    assert calls == [("SIGTERM", "SIGHUP"), *[("pass", begun[0]), "idle"] * 3, "restore"]


def test_a_second_end_of_a_private_run_waits_for_the_first(installs, monkeypatch):
    import threading
    from types import SimpleNamespace

    from localharness.cli import start_cmd
    from localharness.core import run_folder

    _, cfg = installs
    started, release, idle = threading.Event(), threading.Event(), []
    real = run_folder._finish

    def slow(run):
        started.set()
        release.wait(5)
        return real(run)

    async def fake(*args, run=None, **kw):
        threading.Thread(target=run_folder.end, args=(run,), daemon=True).start()
        assert started.wait(5)
        second = threading.Thread(target=idle[0], daemon=True)
        second.start()
        second.join(0.3)
        assert second.is_alive()
        release.set()
        second.join(interrupt.JOIN_S)
        assert not second.is_alive() and not run.path.exists()

    monkeypatch.setattr(run_folder, "_finish", slow)
    monkeypatch.setattr(start_cmd, "_start_async", fake)
    monkeypatch.setattr(interrupt, "install", lambda signals, on_idle: idle.append(on_idle) or SimpleNamespace(
        idle=lambda: None, restore=lambda: None))
    result = _invoke("--private", "--config-dir", str(cfg))
    assert result.exit_code == 0, result.output


async def test_a_private_run_that_ends_stops_persisting_and_swaps_its_log_handlers(tmp_path, monkeypatch):
    import logging

    from localharness.cli.start_cmd import _start_async
    from localharness.core.run_folder import begin, end
    from tests.unit.test_start_cmd import _stub_start_boundaries

    _stub_start_boundaries(tmp_path, monkeypatch)
    stopped = []
    monkeypatch.setattr(EventBus, "stop_persisting", lambda self: stopped.append(self))
    run = begin(tmp_path, private=True)
    try:
        await _start_async(None, False, False, str(tmp_path), run=run)
    finally:
        end(run)
    for name in ("localharness.memory", "localharness.agent.context"):
        handlers = logging.getLogger(name).handlers
        assert any(isinstance(h, logging.NullHandler) for h in handlers)
        assert not [h for h in handlers if isinstance(h, logging.FileHandler) and not h.stream]
    assert len(stopped) == 1


async def test_a_normal_run_keeps_its_bus_and_log_handlers(tmp_path, monkeypatch):
    import logging

    from localharness.cli.start_cmd import _start_async
    from tests.unit.test_start_cmd import _stub_start_boundaries

    _stub_start_boundaries(tmp_path, monkeypatch)
    stopped = []
    monkeypatch.setattr(EventBus, "stop_persisting", lambda self: stopped.append(self))
    await _start_async(None, False, False, str(tmp_path))
    assert stopped == []
    assert any(isinstance(h, logging.FileHandler) and h.stream for h in logging.getLogger("localharness.memory").handlers)


async def test_a_normal_runs_teardown_stage_has_no_deadline():
    from localharness.cli.start_cmd import _stage

    assert _stage(False).when() is None and _stage(True).when() is not None
