"""`lifecycle.swap` under the harness's own autostart unit (spec 20 §8.2, §9.4; D-19, D-21, R-14-28; START-10).

`swap(..., unit=state)` hands the state to every strategy it builds: the incumbent is freed by the stopper its
OWNER calls for (`UnitStrategy.stop`: the unit's stop for a server the unit runs, the wrapped strategy's verified
stop for a spawned one), the target is launched by restarting the unit, a half-started target is stopped the same
way, and a failed swap's restore re-activates the incumbent through the unit. A target the unit cannot serve (a
docker, Ollama or LM Studio record) launches through its own strategy and takes `unit-target` away first.

Nothing real runs: the service manager is `unit_strategy_support.ScriptedManager`, the unit's `serve` is a
`FakeServe` thread (its record chosen through the real ladder where a test needs it), `/proc` is `fake_proc`, the
HTTP wait answers through `server._TRANSPORT`, and `server.start_server` is patched to fail. The spawned
incumbent's verified stop is `server.stop_server`, patched to a recorder that removes the process and its pidfile.
"""
from __future__ import annotations

import asyncio
import os
import time

import pytest
import yaml

from localharness.cli import model_ops
from localharness.cli.model_ops import RUNG_UNIT
from localharness.cli.workspace import MachineTrust
from localharness.config.loader import ConfigLoader
from localharness.config.models import AutosenseConfig, ManagedServerConfig
from localharness.provider import lifecycle, procfs, server, units
from localharness.provider.lifecycle import Heavy
from tests.unit.procfs_support import fake_proc  # noqa: F401 — the fixture
from tests.unit.unit_strategy_support import (
    PRIMARY, UNIT, UNIT_CGROUP, FakeServe, install_process, unit_world, vanish)

fcntl = pytest.importorskip("fcntl")
AUTOSENSE = AutosenseConfig()
INCUMBENT_PID = 3000
INCUMBENT_START = 4000
STOP_ARGV = ("systemctl", "--user", "stop", UNIT)


@pytest.fixture(autouse=True)
def _no_leaked_holds():
    yield
    assert lifecycle._HELD == {}


@pytest.fixture
def world(tmp_path, monkeypatch, fake_proc):  # noqa: F811
    made = unit_world(tmp_path, monkeypatch, fake_proc)
    monkeypatch.setattr(lifecycle, "GPU_FREE_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(server, "start_server", lambda *a, **kw: pytest.fail("the harness started a server"))
    monkeypatch.setattr(os, "kill", lambda *a, **kw: pytest.fail("a process was signalled"))
    monkeypatch.setattr(os, "killpg", lambda *a, **kw: pytest.fail("a process group was signalled"))
    made.manager.answer_prefix("systemctl", "--user", "stop", out="")
    made.events = []  # type: ignore[attr-defined]
    made.stop_servers_called = []  # type: ignore[attr-defined]
    yield made
    made.stop_servers()


class Reporter:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.lines: list[str] = []

    async def freed(self) -> None:
        self.events.append("freed")

    async def launched(self, live) -> None:
        self.events.append("launched")

    async def restored(self, spec) -> None:
        self.events.append("restored")

    async def line(self, text: str) -> None:
        self.lines.append(text)


def heavy(world, spec, peer=None) -> Heavy:
    return Heavy(spec, world.url(peer), peer)


def spawned_incumbent(world, spec=None, *, recorded: list | None = None) -> None:
    """A llama-server the harness spawned (a scope outside the unit's cgroup) is up and recorded in the pidfile.
    `server.stop_server` is its verified stop: a recorder that removes the process and the pidfile."""
    spec = spec or world.primary
    install_process(world.proc, INCUMBENT_PID, argv=FakeServe.record_argv(spec), exe=str(spec.binary),
                    start_time=INCUMBENT_START, cgroup="/user.slice/app.slice/run-r1.scope")
    server.pid_path(world.g).parent.mkdir(parents=True, exist_ok=True)
    server.pid_path(world.g).write_text(f"{INCUMBENT_PID} {INCUMBENT_START} {spec.port}", encoding="utf-8")
    stops = recorded if recorded is not None else world.stop_servers_called

    def fake_stop(config_dir, launch="binary", timeout_seconds=30.0, **kw):
        stops.append({"launch": launch, **kw})
        world.events.append("stop_server")
        vanish(world.proc, INCUMBENT_PID)
        server.pid_path(config_dir).unlink(missing_ok=True)
        return True

    world.monkeypatch.setattr(server, "stop_server", fake_stop)


def unit_incumbent(world, spec=None, *, pid: int = INCUMBENT_PID) -> None:
    """The unit runs the incumbent: its process is the unit's MainPID in the unit's cgroup. The unit's `stop`
    removes the process and the pidfile and the unit reports no MainPID, as the real stop does."""
    spec = spec or world.primary
    install_process(world.proc, pid, argv=FakeServe.record_argv(spec), exe=str(spec.binary),
                    start_time=INCUMBENT_START, cgroup=UNIT_CGROUP)
    server.pid_path(world.g).parent.mkdir(parents=True, exist_ok=True)
    server.pid_path(world.g).write_text(f"{pid} {INCUMBENT_START} {spec.port}", encoding="utf-8")
    world.manager.base["MainPID"] = pid
    manager = world.manager

    def run(argv, *, timeout, cap, **kw):
        done = manager(argv, timeout=timeout, cap=cap, **kw)
        if tuple(argv)[:3] == ("systemctl", "--user", "stop") and done is not None and done.code == 0:
            world.events.append("unit-stop")
            vanish(world.proc, pid)
            server.pid_path(world.g).unlink(missing_ok=True)
            manager.base["MainPID"] = 0
        return done

    world.monkeypatch.setattr(units, "_run_manager", run)
    world.monkeypatch.setattr(server, "stop_server", lambda *a, **kw: pytest.fail("the unit's server was stopped by the harness"))


def swap_now(world, incumbent, target, *, unit="own", reporter=None, timeout=60.0):
    """`lifecycle.swap` inside the launch lock, as every caller runs it. Returns `(outcome, reporter)`."""
    rep = reporter or Reporter()

    async def go():
        async with lifecycle.launch_lock(world.g):
            return await lifecycle.swap(
                incumbent, target, config_dir=world.g, autosense=AUTOSENSE,
                unit=world.state if unit == "own" else unit, reporter=rep, redact=lambda t: t,
                timeout_seconds=timeout)

    return asyncio.run(go()), rep


def flash_ready(world, *, gate=None) -> FakeServe:
    """Everything the unit's restart for flash-next needs: fast polls, an HTTP answer, a `serve`."""
    world.fast_polls()
    world.fast_loan()
    world.http(served="flash-next", gate=gate)
    world.probe()
    serving = world.serve_on_restart(world.peer("flash-next"), main_pid=False)
    # the unit reports the new MainPID only once it restarted: an incumbent the unit runs keeps its own until then
    world.manager.on_restart.append(lambda: world.manager.base.update(MainPID=serving.pid))
    return serving


def fresh(g):
    return ConfigLoader(config_dir=g).load_harness()


def ladder(g):
    return model_ops.resolve_serving(fresh(g), MachineTrust(), None, config_dir=g)


def pending_in_overlay(g):
    path = g / "overrides.yaml"
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("pending_preset") if path.exists() else None


def failing_then_serving(world, serving_first: FakeServe | None, serving_second: FakeServe) -> dict:
    """The first restart makes the unit fail (the target never comes up; `serving_first`, when given, is the
    `serve` that restart ran and is let go at the second restart); the second brings `serving_second` up as the
    unit's main process. `restarts["n"]` counts the restarts and gates the HTTP answer."""
    restarts = {"n": 0}

    def hook():
        restarts["n"] += 1
        if restarts["n"] == 1:
            if serving_first is not None:
                serving_first.start()
                deadline = time.monotonic() + 5
                while serving_first.chosen is None and time.monotonic() < deadline:
                    time.sleep(0.002)  # the unit's `serve` has read the ladder before the unit fails
            world.manager.base.update(ActiveState="failed", ExecMainStatus=1)
        else:
            if serving_first is not None:
                serving_first.stop()
            world.manager.base.update(ActiveState="active", ExecMainStatus=0, MainPID=serving_second.pid)
            serving_second.start()

    world.manager.on_restart.append(hook)
    return restarts


# ----------------------------------------------------------------------------------- outside the unit


def _docker():
    return ManagedServerConfig(runtime="vllm", launch="docker", docker_image="vllm/vllm-openai:x",
                               model="/m/ckpt", port=8000)


def _ollama():
    return ManagedServerConfig(runtime="ollama", model="llama3:8b", port=11434)


def _lmstudio():
    return ManagedServerConfig(runtime="lmstudio", model="m", port=1234)


@pytest.mark.parametrize("make", [_docker, _ollama, _lmstudio])
def test_outside_autostart_is_true_for_a_record_the_unit_cannot_serve_under_the_own_unit(world, make):
    spec = make()
    assert lifecycle.outside_autostart(spec, world.state) is True
    assert lifecycle.outside_autostart(spec, None) is False
    assert lifecycle.outside_autostart(spec, object()) is False
    assert lifecycle.outside_autostart(spec, units.UnitRuntimeState("a", 0)) is False  # raw: not own


def test_outside_autostart_is_false_for_a_record_the_unit_serves(world):
    assert lifecycle.outside_autostart(world.primary, world.state) is False
    assert lifecycle.outside_autostart(world.peer("flash-next"), world.state) is False
    vllm_binary = ManagedServerConfig(runtime="vllm", launch="binary", binary="/x/vllm", model="/m/ckpt", port=8000)
    assert lifecycle.outside_autostart(vllm_binary, world.state) is False
    assert lifecycle.outside_autostart(None, world.state) is True  # attach-only: the unit serves nothing for it


def test_the_clause_says_the_next_boot_serves_nothing():
    assert lifecycle.RUNS_OUTSIDE_AUTOSTART == "(runs outside autostart — the next boot serves nothing)"


def test_leave_unit_removes_the_target_only_for_a_record_outside_the_own_unit(world):
    units.write_target(world.g, "flash-next")
    assert lifecycle.leave_unit(world.primary, world.state, world.g) is False
    assert lifecycle.leave_unit(_docker(), None, world.g) is False
    assert lifecycle.leave_unit(_docker(), object(), world.g) is False
    assert world.target() == "flash-next"
    assert lifecycle.leave_unit(_docker(), world.state, world.g) is True
    assert world.target() is None
    assert lifecycle.leave_unit(_docker(), world.state, world.g) is True  # nothing to remove is not an error


def test_leave_unit_that_cannot_remove_the_file_refuses_before_anything_is_launched(world, monkeypatch):
    def broken(config_dir):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(units, "remove_target", broken)
    with pytest.raises(RuntimeError, match="could not leave the unit target"):
        lifecycle.leave_unit(_docker(), world.state, world.g)


@pytest.mark.parametrize("make, cls", [
    (_docker, lifecycle.ManagedVllmStrategy), (_ollama, lifecycle.DaemonStrategy),
    (_lmstudio, lifecycle.LmsStrategy)])
def test_a_target_outside_the_unit_launches_through_its_own_strategy_after_the_target_file_is_removed(
        world, monkeypatch, make, cls):
    spec = make()
    units.write_target(world.g, "flash-next")
    seen: dict = {}

    async def activate(self, spec, config_dir, base_url, **kw):
        seen["target_at_activate"] = units.read_target(config_dir)
        seen["strategy"] = type(self)
        return lifecycle.LiveEndpoint(base_url=base_url, served_models=["m"], handle=1)

    monkeypatch.setattr(cls, "activate", activate)
    outcome, rep = swap_now(world, None, Heavy(spec, "http://127.0.0.1:9999/v1"))
    assert outcome.error is None and outcome.live is not None
    assert seen == {"target_at_activate": None, "strategy": cls}   # removed BEFORE the launch, by the swap
    assert world.manager.verbs == []                                # the unit was never asked anything
    assert world.target() is None
    assert rep.events == ["freed", "launched"]


def test_swap_with_a_unit_that_is_not_an_own_unit_state_behaves_as_none(world, monkeypatch):
    units.write_target(world.g, "flash-next")
    seen: dict = {}

    async def activate(self, spec, config_dir, base_url, **kw):
        seen["target_at_activate"] = units.read_target(config_dir)
        return lifecycle.LiveEndpoint(base_url=base_url, served_models=["m"], handle=1)

    monkeypatch.setattr(lifecycle.ManagedVllmStrategy, "activate", activate)
    for unit in (object(), None, units.UnitRuntimeState("a", 0)):
        outcome, _ = swap_now(world, None, Heavy(_docker(), "http://127.0.0.1:8000/v1"), unit=unit)
        assert outcome.error is None
        assert seen["target_at_activate"] == "flash-next" and world.target() == "flash-next"  # untouched
    assert world.manager.verbs == []


# ------------------------------------------------------------------------------ freeing the incumbent


def test_a_spawned_incumbent_is_freed_through_the_wrapped_verified_stop_then_the_unit_restarts(world):
    spawned_incumbent(world)
    serving = flash_ready(world)
    world.manager.on_restart.insert(0, lambda: world.events.append("restart"))
    outcome, rep = swap_now(world, heavy(world, world.primary), heavy(world, world.peer("flash-next"), "flash-next"))
    serving.join()
    assert serving.error is None and outcome.error is None and outcome.live is not None
    assert outcome.live.served_models == ["flash-next"]
    # the spawned branch: the wrapped strategy's verified stop, with the record's identity as proof
    (stopped,) = world.stop_servers_called
    assert stopped["launch"] == "binary" and stopped["expect"] == lifecycle.strategy_for(world.primary).launch_identity(world.primary)
    assert world.events == ["stop_server", "restart"]               # freed first, then the unit restarted
    assert world.manager.verbs == [("reset-failed", UNIT), ("restart", UNIT)]
    assert world.target() == "flash-next"
    assert rep.events == ["freed", "launched"]
    assert outcome.stopped is not None and outcome.restored is None


def test_a_unit_incumbent_is_freed_through_the_unit_then_the_unit_restarts(world):
    unit_incumbent(world)
    serving = flash_ready(world)
    outcome, _ = swap_now(world, heavy(world, world.primary), heavy(world, world.peer("flash-next"), "flash-next"))
    serving.join()
    assert serving.error is None and outcome.error is None
    assert world.manager.verbs == [("reset-failed", UNIT), ("stop", UNIT), ("reset-failed", UNIT), ("restart", UNIT)]
    assert world.events == ["unit-stop"]
    assert world.target() == "flash-next"


def test_the_incumbent_that_is_the_target_is_not_stopped(world):
    unit_incumbent(world)
    freed = asyncio.run(lifecycle.free_accelerator((world.primary, "u"), world.primary, world.g, unit=world.state))
    assert freed is None and world.manager.verbs == [] and world.events == []


def test_free_accelerator_stops_a_unit_incumbent_through_the_unit_and_returns_it(world):
    unit_incumbent(world)
    flash = world.peer("flash-next")
    freed = asyncio.run(lifecycle.free_accelerator((world.primary, "u"), flash, world.g, unit=world.state))
    assert freed == (world.primary, "u")
    assert world.manager.verbs == [("reset-failed", UNIT), ("stop", UNIT)]


def test_free_accelerator_without_a_unit_is_the_wrapped_stop_alone(world):
    spawned_incumbent(world)
    flash = world.peer("flash-next")
    freed = asyncio.run(lifecycle.free_accelerator((world.primary, "u"), flash, world.g))
    assert freed == (world.primary, "u")
    assert world.stop_servers_called and world.manager.verbs == []


# --------------------------------------------------------------------------- a failed swap's restore


def test_a_failed_target_under_the_unit_restores_the_incumbent_through_the_unit(world):
    spawned_incumbent(world)
    world.fast_polls()
    world.fast_loan()
    world.probe()
    restores = world.serve(world.primary, pid=4343, when=lambda: True)  # started by the second restart
    restarts = failing_then_serving(world, None, restores)
    world.http(served=PRIMARY, gate=lambda: restarts["n"] >= 2)
    outcome, rep = swap_now(world, heavy(world, world.primary), heavy(world, world.peer("flash-next"), "flash-next"))
    restores.join()
    assert restores.error is None
    assert outcome.error is not None and outcome.stage == "launch" and outcome.live is None
    assert outcome.restored is not None and outcome.restored.spec is world.primary
    assert rep.events == ["freed", "restored"]
    assert [v for v in world.manager.verbs if v[0] == "restart"] == [("restart", UNIT), ("restart", UNIT)]
    assert world.target() == units.PRIMARY_TARGET                    # rewritten to the incumbent by its activate
    assert units.in_flight_target(world.g) is None
    assert any("Restored" in line for line in rep.lines)


def test_pending_preset_outstanding_a_swap_is_served_its_target_and_its_restore_the_incumbent(world):
    """A third record sits in `pending_preset`. The unit's `serve` resolves the REAL ladder at each restart: it
    must be served flash-next for the swap and the primary for the restore, never qwen-27b."""
    model_ops.write_pending_preset(world.harness, "qwen-27b", config_dir=world.g)
    unit_incumbent(world)
    world.fast_polls()
    world.fast_loan()
    world.probe()
    chosen: list[tuple[str, str | None]] = []

    def choose():
        got = ladder(world.g)
        chosen.append((got.record.name, got.rung))
        return got.record.spec

    first = world.serve(world.peer("flash-next"), pid=4242, choose=choose, when=lambda: True)
    second = world.serve(world.primary, pid=4343, choose=choose, when=lambda: True)
    restarts = failing_then_serving(world, first, second)
    world.http(served=PRIMARY, gate=lambda: restarts["n"] >= 2)
    outcome, _ = swap_now(world, heavy(world, world.primary), heavy(world, world.peer("flash-next"), "flash-next"))
    second.join()
    assert first.error is None and second.error is None
    assert chosen == [("flash-next", RUNG_UNIT), (PRIMARY, RUNG_UNIT)]    # never qwen-27b
    assert outcome.error is not None and outcome.restored is not None     # the target's wait failed, the incumbent is back
    assert world.target() == units.PRIMARY_TARGET
    assert pending_in_overlay(world.g) == "qwen-27b"                      # a failed swap persists nothing


def test_pending_preset_is_removed_only_by_the_landing_not_by_the_swap(world):
    model_ops.write_pending_preset(world.harness, "qwen-27b", config_dir=world.g)
    unit_incumbent(world)
    serving = flash_ready(world)
    outcome, _ = swap_now(world, heavy(world, world.primary), heavy(world, world.peer("flash-next"), "flash-next"))
    serving.join()
    assert outcome.error is None
    assert pending_in_overlay(world.g) == "qwen-27b"                      # the swap itself never touches it
    flash = next(r for r in model_ops.serving_records(fresh(world.g)) if r.name == "flash-next")
    asyncio.run(model_ops.persist_landed(fresh(world.g), flash, outcome="swap", config_dir=world.g))
    assert pending_in_overlay(world.g) is None                            # the successful landing removes it


# -------------------------------------------------------------------------------------- cancellation


def test_a_cancellation_stops_the_half_started_target_through_the_unit_and_restores_the_prior_target(world):
    units.write_target(world.g, units.PRIMARY_TARGET)                     # the prior value
    manager = world.manager
    stopped: list[int] = []

    def run(argv, *, timeout, cap, **kw):
        done = manager(argv, timeout=timeout, cap=cap, **kw)
        if tuple(argv)[:3] == ("systemctl", "--user", "stop") and done is not None and done.code == 0:
            stopped.append(1)
            vanish(world.proc, 4242)
            manager.base["MainPID"] = 0
        return done

    world.monkeypatch.setattr(units, "_run_manager", run)
    world.fast_polls()
    world.fast_loan()
    world.probe()
    serving = world.serve_on_restart(world.peer("flash-next"))
    holder: list = []
    world.http(answer_after=10**6, needs_pidfile=False,
               hook=lambda n: holder[0].cancel() if n == 3 else None)
    rep = Reporter()

    async def go():
        async def locked():
            async with lifecycle.launch_lock(world.g):
                return await lifecycle.swap(
                    None, heavy(world, world.peer("flash-next"), "flash-next"), config_dir=world.g,
                    autosense=AUTOSENSE, unit=world.state, reporter=rep, redact=lambda t: t, timeout_seconds=60)

        task = asyncio.ensure_future(locked())
        holder.append(task)
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    serving.join()
    assert serving.error is None
    assert stopped == [1]                                                   # the unit's stop took the target down
    assert [v for v in manager.verbs if v[0] == "stop"] == [("stop", UNIT)]
    assert world.target() == units.PRIMARY_TARGET                            # rolled back inside activate
    assert units.in_flight_target(world.g) is None
    assert rep.events == ["freed", "freed"]                                  # and no restore ran


# --------------------------------------------------------------------------------------------- WR-13


def test_free_accelerator_stops_only_what_the_caller_tracked_as_the_incumbent(world):
    """Threading the unit adds no stop: with no incumbent nothing is stopped or asked, whatever is running."""
    spawned_incumbent(world)
    flash = world.peer("flash-next")
    freed = asyncio.run(lifecycle.free_accelerator(None, flash, world.g, unit=world.state))
    assert freed is None
    assert world.stop_servers_called == [] and world.manager.verbs == []
    assert procfs.read_process(INCUMBENT_PID).__class__ is procfs.ProcRecord  # the CPU-only server still runs


def test_a_swap_whose_incumbent_runs_in_the_unit_while_the_manager_cannot_be_asked_signals_nothing(world, monkeypatch):
    unit_incumbent(world)
    monkeypatch.setattr(units, "_run_manager", lambda *a, **kw: None)   # every question times out
    outcome, rep = swap_now(world, heavy(world, world.primary), heavy(world, world.peer("flash-next"), "flash-next"))
    assert outcome.stage == "free" and outcome.live is None and outcome.stopped is None
    assert str(outcome.error) == lifecycle.UNIT_UNASKED_STOP_LINE.format(pid=INCUMBENT_PID, unit=UNIT)
    assert rep.events == [] and world.events == []                   # nothing freed, nothing launched
    assert rep.lines == [lifecycle.SWAP_FREE_FAILED.format(name="flash-next", exc=outcome.error)]
    assert isinstance(procfs.read_process(INCUMBENT_PID), procfs.ProcRecord)   # still running, never signalled
