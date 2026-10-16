"""`lifecycle.UnitStrategy.stop`: the stopper is the launch's owner's (spec 20 §8.2, §9.4; D-13, D-21; START-10).

Nothing real runs. The service manager is `unit_strategy_support.ScriptedManager` (its `show` answers the
state, the verbs are answered here; a wrapper drops the unit's process from the fake `/proc` when `stop`
is answered, as the real stop removes it), `/proc` is `fake_proc`, every file is under tmp. `os.kill`,
`os.killpg` and `server.stop_server` are patched to fail wherever nothing may be signalled: a unit-owned
launch is stopped by the unit, and a foreign unit's process is named, never signalled.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from localharness.provider import lifecycle, procfs, server, units
from localharness.provider.lifecycle import UNIT_FOREIGN_STOP_LINE, UnitStrategy
from tests.unit.procfs_support import fake_proc  # noqa: F401 - the fixture
from tests.unit.unit_strategy_support import (
    UNIT, UNIT_CGROUP, FakeServe, install_process, unit_world, vanish)

PID = 4242
START = 5000
STOP_ARGV = ("systemctl", "--user", "stop", UNIT)
RESET_ARGV = ("systemctl", "--user", "reset-failed", UNIT)


class Inner:
    """The wrapped strategy, recording the stops it is asked for."""

    def __init__(self, real) -> None:
        self.real = real
        self.stops: list[tuple] = []

    async def stop(self, spec, config_dir) -> None:
        self.stops.append((spec, config_dir))

    def __getattr__(self, name):
        return getattr(self.real, name)


@pytest.fixture
def world(tmp_path, monkeypatch, fake_proc):  # noqa: F811
    made = unit_world(tmp_path, monkeypatch, fake_proc)
    monkeypatch.setattr(server, "start_server", lambda *a, **kw: pytest.fail("the harness started a server"))
    monkeypatch.setattr(server, "stop_server", lambda *a, **kw: pytest.fail("the harness stopped a process"))
    monkeypatch.setattr(os, "kill", lambda *a, **kw: pytest.fail("a process was signalled"))
    monkeypatch.setattr(os, "killpg", lambda *a, **kw: pytest.fail("a process group was signalled"))
    made.manager.answer_prefix("systemctl", "--user", "stop", out="")
    return made


def dropping_stop(world, *, drop: bool = True) -> None:
    """The unit's `stop` removes its process from the fake `/proc` (when `drop`) as the real one does."""
    manager = world.manager

    def run(argv, *, timeout, cap, **kw):
        done = manager(argv, timeout=timeout, cap=cap, **kw)
        if drop and tuple(argv)[:3] == ("systemctl", "--user", "stop") and done is not None and done.code == 0:
            vanish(world.proc, PID)
        return done

    world.monkeypatch.setattr(units, "_run_manager", run)


def record(world, spec=None, *, pid: int = PID, cgroup: str | None = UNIT_CGROUP, argv=None, start: int = START,
           pidfile: bool = True, main_pid: int | None = PID) -> None:
    """The pidfile names `pid`, `/proc` shows it running the record's llama-server in `cgroup`, and the
    manager reports `main_pid` as the unit's MainPID (None: no process)."""
    spec = spec or world.primary
    install_process(world.proc, pid, argv=argv or FakeServe.record_argv(spec), exe=str(spec.binary),
                    start_time=start, cgroup=cgroup)
    if pidfile:
        server.pid_path(world.g).parent.mkdir(parents=True, exist_ok=True)
        server.pid_path(world.g).write_text(f"{pid} {start} {spec.port}", encoding="utf-8")
    world.manager.base["MainPID"] = main_pid or 0


def stop(world, spec=None, *, strategy=None) -> None:
    spec = spec or world.primary
    strat = strategy or world.strategy(spec)
    asyncio.run(strat.stop(spec, world.g))


def verbs(world) -> list[tuple[str, ...]]:
    return world.manager.verbs


# ------------------------------------------------------------------------- a launch the unit owns


def test_an_own_unit_launch_is_stopped_through_the_unit(world):
    record(world)
    dropping_stop(world)
    stop(world)
    assert verbs(world) == [("reset-failed", UNIT), ("stop", UNIT)]
    assert procfs.read_process(PID).__class__ is procfs.ProcRefusal


def test_the_recheck_comes_first_then_reset_failed_then_stop_in_order(world):
    record(world)
    dropping_stop(world)
    stop(world)
    calls = world.manager.calls
    assert calls[0][:3] == ("systemctl", "--user", "show")
    ordered = [c for c in calls if c in (RESET_ARGV, STOP_ARGV)]
    assert ordered == [RESET_ARGV, STOP_ARGV]
    assert max(i for i, c in enumerate(calls) if c[:3] == ("systemctl", "--user", "show")) < calls.index(RESET_ARGV)


def test_the_unit_stop_removes_the_launch_so_stop_server_is_never_called(world):
    record(world)
    dropping_stop(world)
    inner = Inner(world.strategy(world.primary).inner)
    strat = world.strategy(world.primary)
    strat._inner = inner
    stop(world, strategy=strat)
    assert inner.stops == []  # and the world's patched `server.stop_server` and signals did not fire


def test_a_process_still_alive_after_the_stop_is_an_error_naming_the_pid(world):
    record(world)
    dropping_stop(world, drop=False)
    with pytest.raises(RuntimeError, match=str(PID)):
        stop(world)
    assert verbs(world) == [("reset-failed", UNIT), ("stop", UNIT)]


def test_a_reused_pid_counts_as_gone(world):
    record(world)
    manager = world.manager

    def run(argv, *, timeout, cap, **kw):
        done = manager(argv, timeout=timeout, cap=cap, **kw)
        if tuple(argv)[:3] == ("systemctl", "--user", "stop"):
            install_process(world.proc, PID, argv=("sleep", "1"), exe="/bin/sleep", start_time=START + 99,
                            cgroup=None, comm="sleep")
        return done

    world.monkeypatch.setattr(units, "_run_manager", run)
    stop(world)  # a different start time: the launch is gone


def test_a_failing_verb_is_a_runtime_error_with_the_managers_words_and_the_next_verb_is_not_issued(world):
    record(world)
    world.manager.answer_prefix("systemctl", "--user", "reset-failed", out="", code=1, err="Unit not loaded.")
    with pytest.raises(RuntimeError, match="reset-failed of localharness-llm.service failed: Unit not loaded"):
        stop(world)
    assert verbs(world) == [("reset-failed", UNIT)]


def test_a_stop_the_manager_cannot_run_is_an_error(world):
    record(world)
    world.manager.answer_prefix("systemctl", "--user", "stop", out=None)
    with pytest.raises(RuntimeError, match="the service manager did not answer"):
        stop(world)


def test_a_stop_writes_no_unit_target(world):
    record(world)
    dropping_stop(world)
    assert world.target() is None
    stop(world)
    assert world.target() is None
    units.write_target(world.g, "flash-next")
    record(world)
    stop(world)
    assert world.target() == "flash-next"


# ---------------------------------------------------------------- a unit that is no longer the harness's


def test_a_foreign_unit_with_the_launch_in_its_cgroup_is_named_and_never_signalled(world):
    record(world)
    world.manager.base["DropInPaths"] = "/x/override.conf"
    why = units.harness_unit_failure(units.unit_runtime_state(), world.g)
    assert why is not None and "drop-ins" in why
    with pytest.raises(RuntimeError) as caught:
        stop(world)
    assert str(caught.value) == UNIT_FOREIGN_STOP_LINE.format(pid=PID, why=why)
    assert verbs(world) == []
    assert str(PID) in str(caught.value) and "not stopped" in str(caught.value)


def test_a_unit_loaded_from_elsewhere_is_named_too(world):
    record(world)
    world.manager.base["FragmentPath"] = "/etc/systemd/user/localharness-llm.service"
    with pytest.raises(RuntimeError, match="is loaded from /etc/systemd/user/localharness-llm.service"):
        stop(world)
    assert verbs(world) == []


def test_a_unit_process_that_is_not_this_records_launch_is_refused_and_left_alone(world):
    record(world, argv=FakeServe.record_argv(world.peer("flash-next")))  # serves 8081, the record says 8080
    with pytest.raises(RuntimeError, match=f"pid {PID} .* not a harness launch — not signalled"):
        stop(world)
    assert verbs(world) == []


# ------------------------------------------------------------------------------- a launch it spawned


def test_a_spawned_launch_goes_to_the_wrapped_strategys_own_stop(world):
    record(world, cgroup="/user.slice/app.slice/run-r1.scope", main_pid=9999)
    strat = world.strategy(world.primary)
    inner = Inner(strat.inner)
    strat._inner = inner
    stop(world, strategy=strat)
    assert inner.stops == [(world.primary, world.g)]
    assert verbs(world) == []


def test_a_missing_main_pid_makes_every_launch_spawned(world):
    record(world, main_pid=None)  # 4242 IS in the unit's cgroup, but the unit reports no MainPID
    strat = world.strategy(world.primary)
    inner = Inner(strat.inner)
    strat._inner = inner
    stop(world, strategy=strat)
    assert len(inner.stops) == 1
    assert verbs(world) == []


@pytest.mark.parametrize("unit_name", ["Localharness-llm.service", "localharness\\x2dllm.service"])
def test_a_cgroup_unit_spelled_otherwise_is_not_the_unit(world, unit_name):
    record(world, cgroup=UNIT_CGROUP.replace(UNIT, unit_name))
    strat = world.strategy(world.primary)
    inner = Inner(strat.inner)
    strat._inner = inner
    stop(world, strategy=strat)
    assert len(inner.stops) == 1
    assert verbs(world) == []


def test_a_manager_that_cannot_be_asked_does_not_signal_a_launch_in_the_units_cgroup(world, monkeypatch):
    record(world)  # 4242 runs in localharness-llm.service's cgroup and is the unit's MainPID
    monkeypatch.setattr(units, "unit_runtime_state", lambda: None)
    strat = world.strategy(world.primary)
    inner = Inner(strat.inner)
    strat._inner = inner

    with pytest.raises(RuntimeError) as raised:
        stop(world, strategy=strat)

    assert str(raised.value) == lifecycle.UNIT_UNASKED_STOP_LINE.format(pid=PID, unit=UNIT)
    assert f"pid {PID}" in str(raised.value) and "stop the unit from a terminal" in str(raised.value)
    assert inner.stops == [] and verbs(world) == []   # the world's patched signals and stop_server did not fire
    assert isinstance(procfs.read_process(PID), procfs.ProcRecord)  # still running, untouched


def test_a_runner_that_times_out_on_every_question_is_the_same_unasked_manager(world, monkeypatch):
    # the real failure: the runner (timeout, busy user bus) answers None to the state read and to every verb
    record(world, main_pid=9999)
    monkeypatch.setattr(units, "_run_manager", lambda *a, **kw: None)
    strat = world.strategy(world.primary)
    inner = Inner(strat.inner)
    strat._inner = inner

    with pytest.raises(RuntimeError, match="could not be asked"):
        stop(world, strategy=strat)

    assert inner.stops == [] and verbs(world) == []


def test_a_manager_that_cannot_be_asked_still_lets_a_launch_outside_the_unit_go_to_the_wrapped_stop(world, monkeypatch):
    record(world, cgroup="/user.slice/app.slice/run-r1.scope", main_pid=9999)
    monkeypatch.setattr(units, "unit_runtime_state", lambda: None)
    strat = world.strategy(world.primary)
    inner = Inner(strat.inner)
    strat._inner = inner

    stop(world, strategy=strat)

    assert len(inner.stops) == 1 and verbs(world) == []


def test_the_recheck_is_asked_first_even_when_the_launch_is_spawned(world):
    record(world, cgroup=None, main_pid=9999)
    strat = world.strategy(world.primary)
    strat._inner = Inner(strat.inner)
    stop(world, strategy=strat)
    assert world.manager.calls[0][:3] == ("systemctl", "--user", "show")


# --------------------------------------------------------------------------------------- no launch


def test_no_pidfile_stops_nothing_and_calls_nothing(world):
    strat = world.strategy(world.primary)
    inner = Inner(strat.inner)
    strat._inner = inner
    stop(world, strategy=strat)
    assert inner.stops == [] and verbs(world) == []


def test_a_pidfile_whose_start_time_is_not_the_processs_stops_nothing(world):
    record(world)
    server.pid_path(world.g).write_text(f"{PID} {START + 1} 8080", encoding="utf-8")
    strat = world.strategy(world.primary)
    inner = Inner(strat.inner)
    strat._inner = inner
    stop(world, strategy=strat)
    assert inner.stops == [] and verbs(world) == []


# ------------------------------------------------------------------------------ the strategy itself


def test_the_stop_is_the_unit_strategys_own(world):
    assert "stop" in vars(UnitStrategy)


def test_the_unit_strategy_is_a_lifecycle_strategy_with_the_wrapped_identity(world):
    strat = world.strategy(world.primary)
    assert isinstance(strat, lifecycle.LifecycleStrategy)
    assert strat.launch_identity(world.primary) == strat.inner.launch_identity(world.primary)
    assert lifecycle.record_identity(world.primary) is not None
