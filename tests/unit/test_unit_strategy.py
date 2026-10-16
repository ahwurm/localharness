"""`lifecycle.UnitStrategy` and `strategy_for(unit=)` (spec 20 §9.4, D-19, D-20, AUTO-06, AUTO-09, RULING W3).

Everything the strategy touches is faked and nothing real runs: the service manager is
`unit_strategy_support.ScriptedManager` (successive `show` answers, a hook on `restart`), the unit's `serve` is
a `FakeServe` thread that takes the launch lock on a second file description and writes the pidfile, `/proc` is
`fake_proc`, the HTTP wait answers through `server._TRANSPORT`, and the lock's clock is a fake `time` where a
test needs time to pass. `server.start_server` is patched to raise in the activate tests: the unit's restart
launches the server, the harness never does.
"""
from __future__ import annotations

import asyncio

import pytest

from localharness.config.models import AutosenseConfig, ManagedServerConfig
from localharness.provider import lifecycle, procfs, server, units
from localharness.provider.lifecycle import UnitStrategy, strategy_for
from tests.unit.procfs_support import fake_proc  # noqa: F401 — the fixture
from tests.unit.unit_strategy_support import (
    PRIMARY, RESET_ARGV, RESTART_ARGV, UNIT, unit_world)

fcntl = pytest.importorskip("fcntl")
AUTOSENSE = AutosenseConfig()


@pytest.fixture(autouse=True)
def _no_leaked_holds():
    yield
    assert lifecycle._HELD == {}


@pytest.fixture
def world(tmp_path, monkeypatch, fake_proc):  # noqa: F811
    made = unit_world(tmp_path, monkeypatch, fake_proc)
    monkeypatch.setattr(server, "start_server", lambda *a, **kw: pytest.fail("the harness started a server"))
    yield made
    made.stop_servers()


def run_activate(world, spec, *, peer=None, name=None, timeout=60, autosense=AUTOSENSE, plans=None,
                 strategy=None, api_key=None, probe=None):
    """`UnitStrategy.activate` inside the caller's launch lock, as every harness launch runs. Returns
    `(live, lock_free_inside_after)`: whether another file description could take the lock after activate
    returned (it must not: the caller holds the lock again)."""
    strat = strategy or world.strategy(spec)

    async def go():
        async with lifecycle.launch_lock(world.g):
            live = await strat.activate(
                spec, world.g, world.url(name or peer), timeout_seconds=timeout, peer=peer, autosense=autosense,
                on_plan=(plans.append if plans is not None else None), api_key=api_key)
            return live, world.third_party_can_lock()

    return asyncio.run(go())


# ------------------------------------------------------------------------------------------- selection


def _vllm(launch="binary", **kw):
    over = {"runtime": "vllm", "launch": launch, "model": "/m/ckpt", "port": 8000, **kw}
    if launch == "binary":
        over.setdefault("binary", "/x/vllm")
    else:
        over.setdefault("docker_image", "vllm/vllm-openai:x")
    return ManagedServerConfig(**over)


def test_without_a_unit_the_strategy_is_todays(world):
    assert isinstance(strategy_for(world.primary), lifecycle.SpawnedProcessStrategy)
    assert isinstance(strategy_for(world.primary, unit=None), lifecycle.SpawnedProcessStrategy)


def test_the_harness_own_unit_wraps_a_llamacpp_record(world):
    got = strategy_for(world.primary, unit=world.state)
    assert isinstance(got, UnitStrategy)
    assert isinstance(got.inner, lifecycle.SpawnedProcessStrategy)
    assert got.unit is world.state and got.plans_launch is True


def test_the_harness_own_unit_wraps_a_vllm_binary_and_not_a_docker_record(world):
    wrapped = strategy_for(_vllm("binary"), unit=world.state)
    assert isinstance(wrapped, UnitStrategy) and isinstance(wrapped.inner, lifecycle.ManagedVllmStrategy)
    assert isinstance(strategy_for(_vllm("docker"), unit=world.state), lifecycle.ManagedVllmStrategy)
    ollama = ManagedServerConfig(runtime="ollama", model="llama3:8b", port=11434)
    assert isinstance(strategy_for(ollama, unit=world.state), lifecycle.DaemonStrategy)
    lms = ManagedServerConfig(runtime="lmstudio", model="m", port=1234)
    assert isinstance(strategy_for(lms, unit=world.state), lifecycle.LmsStrategy)


def test_a_raw_or_foreign_unit_state_selects_nothing(world):
    raw = units.UnitRuntimeState("a", 0)
    assert raw.own is False
    assert isinstance(strategy_for(world.primary, unit=raw), lifecycle.SpawnedProcessStrategy)
    assert isinstance(strategy_for(world.primary, unit="not a state"), lifecycle.SpawnedProcessStrategy)  # type: ignore[arg-type]


def test_the_wrapper_answers_identity_liveness_and_recovery_as_the_wrapped_strategy(world, tmp_path):
    spec = world.primary
    wrapper, inner = strategy_for(spec, unit=world.state), strategy_for(spec)
    assert wrapper.launch_identity(spec) == inner.launch_identity(spec)
    assert wrapper.liveness(spec, tmp_path) == inner.liveness(spec, tmp_path)
    assert wrapper.recovery_config_dir(tmp_path) == inner.recovery_config_dir(tmp_path) == tmp_path
    assert lifecycle.record_identity(spec) is not None


# ------------------------------------------------------------------------------- the activate sequence


def test_activate_sequence_writes_the_target_restarts_the_unit_and_waits_for_the_answer(world):
    world.ready()
    serving = world.serve_on_restart(world.primary)
    plans = []
    live, caller_holds_lock = run_activate(world, world.primary, plans=plans)
    serving.join()
    assert serving.error is None and serving.took_lock.is_set()
    assert world.target() == units.PRIMARY_TARGET == "server:"
    assert [c for c in world.manager.calls if c[2] != "show"] == [RESET_ARGV, RESTART_ARGV]
    assert world.manager.verbs == [("reset-failed", UNIT), ("restart", UNIT)]
    assert len(plans) == 1 and plans[0] is not None and plans[0].argv[0] == "/x/llama-server"
    assert live.base_url == world.url() and live.served_models == [PRIMARY]
    assert live.handle == 4242 and live.window is not None and live.tuned is not None
    assert caller_holds_lock is False


def test_a_peer_is_recorded_under_its_own_name(world):
    world.ready()
    world.http(served="flash-next")
    spec = world.peer("flash-next")
    serving = world.serve_on_restart(spec)
    live, _ = run_activate(world, spec, peer="flash-next")
    serving.join()
    assert world.target() == "flash-next"
    assert live.base_url == world.url("flash-next")


def test_an_untuned_activate_plans_nothing_and_says_so(world):
    world.ready()
    serving = world.serve_on_restart(world.primary)
    plans = []
    live, _ = run_activate(world, world.primary, autosense=None, plans=plans)
    serving.join()
    assert plans == [None] and live.window is None and live.tuned is None
    assert world.target() == "server:"


def test_the_harness_never_launches_a_process_itself(world, monkeypatch):
    world.ready()
    launched = []
    monkeypatch.setattr(server, "start_server", lambda *a, **kw: launched.append(a))
    stopped = []
    monkeypatch.setattr(server, "stop_server", lambda *a, **kw: stopped.append(a))
    serving = world.serve_on_restart(world.primary)
    run_activate(world, world.primary)
    serving.join()
    assert launched == [] and stopped == []


# ---------------------------------------------------------------------------- the loan ends on an event


def test_the_loan_lets_the_unit_serve_take_the_lock_only_while_it_is_lent(world):
    world.ready()
    seen = {}
    serving = world.serve_on_restart(world.primary)
    original = lifecycle.lend_launch_lock

    def spy(config_dir):
        seen["free_before"] = world.third_party_can_lock()
        return original(config_dir)

    world.monkeypatch.setattr(lifecycle, "lend_launch_lock", spy)
    live, caller_holds_lock = run_activate(world, world.primary)
    serving.join()
    assert seen["free_before"] is False       # the caller holds it until it lends it
    assert serving.took_lock.is_set()         # the unit's serve got it during the loan
    assert caller_holds_lock is False         # and the caller holds it again after
    assert live.served_models == [PRIMARY]


def test_slow_serve_slower_than_any_fixed_interval_still_gets_the_lock(world):
    world.ready()
    clock = world.fake_clock(step=100.0, freeze_at=700.0)
    serving = world.serve_on_restart(
        world.primary, when=lambda: world.manager.restarted and clock.now >= 600.0)
    live, caller_holds_lock = run_activate(world, world.primary, timeout=1800)
    serving.join()
    assert serving.took_lock.is_set() and serving.error is None
    assert clock.now >= 600.0                 # ten times the former fixed loan went by first
    assert live.served_models == [PRIMARY] and caller_holds_lock is False


def test_no_serve_ever_recording_ends_at_the_activates_own_deadline_with_the_lock_back(world):
    world.ready()
    world.fake_clock(step=100.0)

    async def go():
        async with lifecycle.launch_lock(world.g):
            with pytest.raises(TimeoutError):
                await world.strategy(world.primary).activate(
                    world.primary, world.g, world.url(), timeout_seconds=30, autosense=AUTOSENSE)
            return world.third_party_can_lock()

    assert asyncio.run(go()) is False         # the loan ended and the caller holds the lock again


def test_an_inactive_unit_ends_the_loan_at_the_next_poll(world):
    world.ready()
    world.manager.after_restart({"ActiveState": "inactive", "MainPID": 0, "ExecMainStatus": 0})

    async def go():
        async with lifecycle.launch_lock(world.g):
            with pytest.raises(RuntimeError) as caught:
                await world.strategy(world.primary).activate(
                    world.primary, world.g, world.url(), timeout_seconds=600, autosense=AUTOSENSE)
            return caught.value, world.third_party_can_lock()

    exc, free = asyncio.run(go())
    assert free is False                      # back with the caller long before the 600 s deadline
    assert lifecycle.UNIT_EXITED_LINE.format(label=PRIMARY) in str(exc)


def test_a_failed_unit_ends_the_loan_and_the_wait_with_the_failure(world):
    world.ready()
    world.manager.after_restart({"ActiveState": "failed", "ExecMainStatus": 1})

    async def go():
        async with lifecycle.launch_lock(world.g):
            with pytest.raises(RuntimeError, match="did not come up"):
                await world.strategy(world.primary).activate(
                    world.primary, world.g, world.url(), timeout_seconds=600, autosense=AUTOSENSE)
            return world.third_party_can_lock()

    assert asyncio.run(go()) is False


def test_a_nested_launch_inside_the_loan_is_refused(world):
    world.ready()
    attempts = []

    async def go():
        async with lifecycle.launch_lock(world.g):
            async with lifecycle.lend_launch_lock(world.g):
                try:
                    async with lifecycle.launch_lock(world.g):
                        attempts.append("entered")
                except RuntimeError as exc:
                    attempts.append(str(exc))

    asyncio.run(go())
    assert attempts == [lifecycle.LAUNCH_LOCK_LENT_LINE]


# --------------------------------------------------------------------------------- wait_ready(unit_poll)


def test_wait_ready_with_a_unit_poll_ends_on_its_text_and_never_reads_the_pidfile(tmp_path, monkeypatch):
    import httpx

    monkeypatch.setattr(server, "_TRANSPORT", httpx.MockTransport(lambda request: httpx.Response(503)))
    reads = []
    monkeypatch.setattr(server, "server_pid", lambda *a, **kw: reads.append(a) or None)
    monkeypatch.setattr(procfs, "read_launch", lambda *a, **kw: reads.append(a) or None)
    answers = iter([None, None, "boom"])
    calls = []

    async def poll():
        calls.append(1)
        return next(answers)

    async def go():
        return await server.wait_ready("http://127.0.0.1:8080/v1", config_dir=tmp_path, timeout_seconds=30,
                                       poll_seconds=0.001, expect=lifecycle.record_identity(
                                           ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server",
                                                               model="/m/a.gguf", port=8080)),
                                       unit_poll=poll)

    with pytest.raises(RuntimeError, match="^boom$"):
        asyncio.run(go())
    assert len(calls) == 3 and reads == []


def test_wait_ready_without_a_unit_poll_still_reads_the_pidfile(tmp_path, monkeypatch):
    import httpx

    monkeypatch.setattr(server, "_TRANSPORT", httpx.MockTransport(lambda request: httpx.Response(503)))
    with pytest.raises(RuntimeError, match="exited during startup"):
        asyncio.run(server.wait_ready("http://127.0.0.1:8080/v1", config_dir=tmp_path, timeout_seconds=5,
                                      poll_seconds=0.001))
