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
from datetime import datetime, timezone

import pytest

from localharness.config.models import AutosenseConfig, ManagedServerConfig
from localharness.provider import lifecycle, procfs, server, units
from localharness.provider.lifecycle import UnitStrategy, record_key, strategy_for
from tests.unit.procfs_support import fake_proc  # noqa: F401 — the fixture
from tests.unit.unit_strategy_support import (
    PRIMARY, RESET_ARGV, RESTART_ARGV, UNIT, FakeServe, unit_world, vanish)

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
    monkeypatch.setattr(server, "stop_server", lambda *a, **kw: pytest.fail("the harness stopped a process"))
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


def test_the_harness_neither_launches_nor_stops_a_process_itself(world, monkeypatch):
    world.ready()
    launched, stopped = [], []
    monkeypatch.setattr(server, "start_server", lambda *a, **kw: launched.append(a))
    monkeypatch.setattr(server, "stop_server", lambda *a, **kw: stopped.append(a))
    serving = world.serve_on_restart(world.primary)
    run_activate(world, world.primary)
    serving.join()
    assert launched == [] and stopped == []


def test_the_loan_has_no_fixed_bound_to_tune():
    assert not hasattr(lifecycle, "LEND_SECONDS")
    assert lifecycle.LEND_POLL_SECONDS == 0.25   # a poll interval, not a bound


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


# ================================================================================ the wait and its tolerance


def _rises(world, *, at: dict[int, dict]):
    """An HTTP hook that, on poll n, changes what the unit reports from then on (the unit restarted itself)."""
    def hook(n: int) -> None:
        if n in at:
            world.manager.base.update(at[n])
    return hook


def _activate_expecting(world, spec, error, *, timeout=60, api_key=None, autosense=AUTOSENSE):
    async def go():
        async with lifecycle.launch_lock(world.g):
            with pytest.raises(error) as caught:
                await world.strategy(spec).activate(
                    spec, world.g, world.url(), timeout_seconds=timeout, autosense=autosense, api_key=api_key)
            return caught.value, world.third_party_can_lock()

    return asyncio.run(go())


def test_one_rise_of_nrestarts_is_tolerated_when_the_plan_has_a_boot_retry(world):
    world.fast_polls()
    world.fast_loan()
    world.probe()
    world.http(answer_after=4, hook=_rises(world, at={3: {"NRestarts": 1}}))
    serving = world.serve_on_restart(world.primary)
    live, _ = run_activate(world, world.primary)
    serving.join()
    assert live.served_models == [PRIMARY]
    assert live.tuned is not None and live.tuned.spec_type is None   # the unit's own retry answered: the boot shape
    assert live.window is not None


def test_a_second_rise_of_nrestarts_stops_the_wait(world):
    world.fast_polls()
    world.fast_loan()
    world.http(answer_after=1000, hook=_rises(world, at={2: {"NRestarts": 1}, 3: {"NRestarts": 2}}))
    serving = world.serve_on_restart(world.primary)
    exc, holds = _activate_expecting(world, world.primary, RuntimeError)
    serving.join()
    assert lifecycle.UNIT_FAILED_LINE.format(label=PRIMARY) in str(exc)
    assert holds is False


def test_no_boot_retry_tolerates_no_rise_at_all(tmp_path, monkeypatch, fake_proc):  # noqa: F811
    from tests.unit.test_tuning_derive import FLASH_NEXT

    world = unit_world(tmp_path, monkeypatch, fake_proc, primary_extra={"context": 4096}, primary_header=FLASH_NEXT)
    monkeypatch.setattr(server, "start_server", lambda *a, **kw: pytest.fail("the harness started a server"))
    try:
        world.fast_polls()
        world.fast_loan()
        world.http(answer_after=1000, hook=_rises(world, at={2: {"NRestarts": 1}}))
        serving = world.serve_on_restart(world.primary)
        plans = []

        async def go():
            async with lifecycle.launch_lock(world.g):
                with pytest.raises(RuntimeError) as caught:
                    await world.strategy(world.primary).activate(
                        world.primary, world.g, world.url(), timeout_seconds=60, autosense=AUTOSENSE,
                        on_plan=plans.append)
                return caught.value

        exc = asyncio.run(go())
        serving.join()
        assert plans[0].boot_retry() is None
        assert lifecycle.UNIT_FAILED_LINE.format(label=PRIMARY) in str(exc)
    finally:
        world.stop_servers()


def test_an_untuned_activate_tolerates_no_rise(world):
    world.fast_polls()
    world.fast_loan()
    world.http(answer_after=1000, hook=_rises(world, at={2: {"NRestarts": 1}}))
    serving = world.serve_on_restart(world.primary)
    exc, _ = _activate_expecting(world, world.primary, RuntimeError, autosense=None)
    serving.join()
    assert "did not come up" in str(exc)


def test_a_tolerated_retry_relends_the_lock_to_the_new_serve(world):
    world.fast_polls()
    world.fast_loan()
    world.probe()
    first = world.serve_on_restart(world.primary, pid=4242)
    second = world.serve(world.primary, pid=5000, start_time=6000,
                         when=lambda: world.manager.base["MainPID"] == 5000)
    second.start()

    def hook(n: int) -> None:
        if n == 2:      # the unit restarts itself: the first process is gone, MainPID is 0 inside RestartSec
            vanish(world.proc, 4242)
            world.manager.base.update(NRestarts=1, MainPID=0)
        if n == 3:      # the new main process
            world.manager.base.update(MainPID=5000)

    world.http(answer_after=2, hook=hook, gate=lambda: second.recorded.is_set())
    live, holds = run_activate(world, world.primary)
    first.join()
    second.join()
    assert second.took_lock.is_set() and second.error is None and first.error is None
    assert live.served_models == [PRIMARY] and live.handle == 5000
    assert holds is False                                    # the caller holds the lock again
    assert live.tuned is not None and live.tuned.spec_type is None


def test_a_launch_that_is_neither_this_record_nor_the_units_serve_ends_the_wait_naming_its_pid(world):
    world.fast_polls()
    world.fast_loan()
    world.http(answer_after=1000)
    other = FakeServe.record_argv(world.peer("flash-next"))
    serving = world.serve_on_restart(world.primary, exec_argv=other)
    exc, _ = _activate_expecting(world, world.primary, RuntimeError)
    serving.join()
    assert server.NOT_THIS_LAUNCH_LINE.format(label="llama.cpp", pid=4242) == str(exc)


# ======================================================================== failures carry only scrubbed text


def test_a_failed_unit_raises_the_translated_line_and_the_scrubbed_journal_tail_with_a_reference(world):
    key = "sk-held-key-123456"
    world.fast_polls()
    world.fast_loan()
    world.http(answer_after=1000)
    world.manager.journal(
        f"loading qwen\nAuthorization: Bearer sk-planted\napi_keys: ****9876\nplanted {key} here\n"
        "fetch http://user:hunter2@host/x\n")
    world.manager.on_restart.append(
        lambda: world.manager.base.update(ActiveState="failed", ExecMainStatus=78, InvocationID="inv-7"))
    exc, _ = _activate_expecting(world, world.primary, RuntimeError, api_key=key)
    text = str(exc)
    assert units.STOPPED_WITHHELD_LINE.format(record=PRIMARY) in text
    assert units.JOURNAL_REF_LINE.format(invocation="inv-7") in text
    assert "loading qwen" in text
    for secret in ("sk-planted", "9876", key, "hunter2"):
        assert secret not in text
    assert exc.__cause__ is None and exc.__suppress_context__ is True
    queries = world.manager.calls_of("journalctl")
    assert queries and any("_SYSTEMD_INVOCATION_ID=inv-7" in c for c in queries)


def test_a_failure_with_no_journal_still_carries_the_reference(world):
    world.fast_polls()
    world.fast_loan()
    world.http(answer_after=1000)
    world.manager.on_restart.append(lambda: world.manager.base.update(ActiveState="failed", ExecMainStatus=1))
    exc, _ = _activate_expecting(world, world.primary, RuntimeError)
    assert "(no journal)" in str(exc) and units.JOURNAL_REF_LINE.format(invocation="inv-1") in str(exc)


def test_a_timeout_after_the_loan_carries_the_scrubbed_tail_too(world):
    world.fast_polls()
    world.fast_loan()
    world.http(answer_after=10**6)
    world.manager.journal("still loading\nBearer sk-planted\n")
    serving = world.serve_on_restart(world.primary)
    exc, holds = _activate_expecting(world, world.primary, TimeoutError, timeout=0.3)
    serving.join()
    assert "not ready after" in str(exc) and "still loading" in str(exc) and "sk-planted" not in str(exc)
    assert exc.__cause__ is None and holds is False


# ======================================================================================== the rollbacks


def _prior(world, name="flash-next"):
    units.write_target(world.g, name)
    return name


def test_a_cancellation_during_the_wait_rolls_the_target_back_to_its_prior_value(world):
    prior = _prior(world)
    world.fast_polls()
    world.fast_loan()
    holder = []
    world.http(answer_after=10**6, hook=lambda n: holder[0].cancel() if n == 2 else None)
    serving = world.serve_on_restart(world.primary)

    async def go():
        async def locked():
            async with lifecycle.launch_lock(world.g):
                await world.strategy(world.primary).activate(
                    world.primary, world.g, world.url(), timeout_seconds=60, autosense=AUTOSENSE)

        task = asyncio.ensure_future(locked())
        holder.append(task)
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    serving.join()
    assert world.target() == prior


def test_a_cancellation_with_no_prior_target_removes_the_file(world):
    world.fast_polls()
    world.fast_loan()
    holder = []
    world.http(answer_after=10**6, hook=lambda n: holder[0].cancel() if n == 2 else None)
    serving = world.serve_on_restart(world.primary)

    async def go():
        async def locked():
            async with lifecycle.launch_lock(world.g):
                await world.strategy(world.primary).activate(
                    world.primary, world.g, world.url(), timeout_seconds=60, autosense=AUTOSENSE)

        task = asyncio.ensure_future(locked())
        holder.append(task)
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    serving.join()
    assert world.target() is None


def test_a_cancellation_while_lent_keeps_the_callers_ownership_until_its_own_block_exits(world):
    prior = _prior(world)
    world.fast_polls()
    world.fast_loan()
    key = server.lock_path(world.g).resolve()
    seen = {}

    async def go():
        async def locked():
            async with lifecycle.launch_lock(world.g):
                try:
                    await world.strategy(world.primary).activate(
                        world.primary, world.g, world.url(), timeout_seconds=600, autosense=AUTOSENSE)
                finally:
                    seen["still_held"] = lifecycle._HELD.get(key, [None])[0]

        task = asyncio.ensure_future(locked())
        for _ in range(2000):                               # until the loan is open
            held = lifecycle._HELD.get(key)
            if held is not None and held[3]:
                break
            await asyncio.sleep(0.002)
        else:
            pytest.fail("the lock was never lent")
        assert lifecycle._HELD[key][0] is task
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        seen["task"] = task

    asyncio.run(go())
    assert seen["still_held"] is seen["task"]
    assert world.target() == prior


def test_a_foreign_unit_is_refused_before_any_verb_and_the_target_is_restored(world):
    prior = _prior(world)
    world.manager.base["DropInPaths"] = "/home/x/.config/systemd/user/localharness-llm.service.d/override.conf"
    world.fast_polls()
    world.fast_loan()

    async def go():
        async with lifecycle.launch_lock(world.g):
            with pytest.raises(server.LaunchRefused) as caught:
                await world.strategy(world.primary).activate(
                    world.primary, world.g, world.url(), timeout_seconds=60, autosense=AUTOSENSE)
            return caught.value

    exc = asyncio.run(go())
    assert "drop-ins" in str(exc) and "override.conf" in str(exc)
    assert world.manager.verbs == []
    assert world.target() == prior


def test_a_refused_launch_with_no_prior_target_leaves_no_file(world):
    world.manager.base["FragmentPath"] = "/etc/systemd/user/localharness-llm.service"
    exc, _ = _activate_expecting(world, world.primary, server.LaunchRefused)
    assert "FragmentPath" in str(exc) and world.manager.verbs == [] and world.target() is None


def test_a_restart_verb_that_fails_raises_without_the_key_and_restores_the_target(world):
    key = "sk-held-key-123456"
    prior = _prior(world)
    world.manager.answer_prefix("systemctl", "--user", "restart", out="", code=1,
                                err=f"Job failed. Authorization: Bearer {key} and {key}\n")
    exc, _ = _activate_expecting(world, world.primary, RuntimeError, api_key=key)
    assert "restart" in str(exc) and key not in str(exc)
    assert world.manager.verbs == [("reset-failed", UNIT), ("restart", UNIT)]
    assert world.target() == prior


def test_a_reset_failed_that_fails_stops_before_the_restart(world):
    prior = _prior(world)
    world.manager.answer_prefix("systemctl", "--user", "reset-failed", out=None)
    exc, _ = _activate_expecting(world, world.primary, RuntimeError)
    assert "did not answer" in str(exc)
    assert [v for v in world.manager.verbs if v[0] == "restart"] == []
    assert world.target() == prior


# =========================================================================================== the refusals


def test_a_typed_model_restart_is_refused_with_the_preset_names_line_and_writes_nothing(world):
    strat = world.strategy(world.primary)
    typed = world.primary.model_copy(update={"model": "/m/other.gguf"})
    exc, _ = _activate_expecting_on(world, strat, typed, server.LaunchRefused)
    assert str(exc) == lifecycle.UNIT_TYPED_LINE.format(path="/m/other.gguf")
    assert "under autostart /model takes preset names" in str(exc)
    assert world.manager.calls == [] and world.target() is None


def test_a_model_swapped_in_memory_is_refused_the_same_way(world):
    strat = world.strategy(world.primary)
    original = world.primary.model
    world.primary.model = "/m/typed.gguf"
    try:
        exc, _ = _activate_expecting_on(world, strat, world.primary, server.LaunchRefused)
    finally:
        world.primary.model = original
    assert str(exc) == lifecycle.UNIT_TYPED_LINE.format(path="/m/typed.gguf")
    assert world.manager.calls == [] and world.target() is None


def test_a_changed_served_name_is_refused(world):
    strat = world.strategy(world.primary)
    world.primary.served_name = "renamed"
    try:
        exc, _ = _activate_expecting_on(world, strat, world.primary, server.LaunchRefused)
    finally:
        world.primary.served_name = PRIMARY
    assert "preset names" in str(exc) and world.target() is None


@pytest.mark.parametrize("peer", ["my peer", "server:", "-lead", "a/b", "x" * 200])
def test_a_peer_name_the_target_cannot_hold_is_refused_and_nothing_is_written(world, peer):
    strat = world.strategy(world.peer("flash-next"))

    async def go():
        async with lifecycle.launch_lock(world.g):
            with pytest.raises(server.LaunchRefused) as caught:
                await strat.activate(world.peer("flash-next"), world.g, world.url("flash-next"),
                                     timeout_seconds=60, peer=peer, autosense=AUTOSENSE)
            return caught.value

    exc = asyncio.run(go())
    assert str(exc) == lifecycle.UNIT_NAME_LINE.format(name=peer)
    assert world.manager.calls == [] and world.target() is None


def test_a_record_with_no_binary_is_refused_by_the_launch_decision_before_anything_is_written(world):
    declared = world.primary.model_copy(update={"binary": None})
    strat = world.strategy(declared)
    exc, _ = _activate_expecting_on(world, strat, declared, server.LaunchRefused)
    assert "binary" in str(exc) and world.manager.calls == [] and world.target() is None


def _activate_expecting_on(world, strat, spec, error):
    async def go():
        async with lifecycle.launch_lock(world.g):
            with pytest.raises(error) as caught:
                await strat.activate(spec, world.g, world.url(), timeout_seconds=60, autosense=AUTOSENSE)
            return caught.value, world.third_party_can_lock()

    return asyncio.run(go())


# ================================================================================================ confirm


def _marker(world, invocation, **kw):
    fields = dict(record=record_key(world.primary), binary="/x/llama-server", invocation=invocation, cause="none",
                  shape="full", planned_context=None, full_fit=None, free_bytes=None, dropped=(),
                  at=datetime.now(timezone.utc).isoformat())
    fields.update(kw)
    return units.Marker(**fields)


def test_success_confirms_the_launch_with_the_marker_the_live_process_and_the_fresh_unit(world, monkeypatch):
    world.ready()
    marker = _marker(world, "inv-1")
    units.write_marker(world.g, marker)
    seen = []
    monkeypatch.setattr(lifecycle, "confirm_launch", lambda *a, **kw: seen.append((a, kw)))
    serving = world.serve_on_restart(world.primary)
    live, _ = run_activate(world, world.primary)
    serving.join()
    (args, kw), = seen
    assert args == (world.g, record_key(world.primary))
    assert kw["marker"] == marker and kw["live"] is live and kw["spec"] is world.primary
    assert kw["process"].pid == 4242 and kw["unit"].invocation_id == "inv-1"
    assert kw["autosense"] is AUTOSENSE and kw["header"]["block_count"] == 65


def test_the_confirm_removes_a_marker_of_this_invocation_and_leaves_another_runs(world):
    world.ready()
    units.write_marker(world.g, _marker(world, "inv-1"))
    serving = world.serve_on_restart(world.primary)
    run_activate(world, world.primary)
    serving.join()
    assert units.read_marker(world.g) is None


def test_a_marker_of_another_invocation_is_left_for_its_run(world):
    world.ready()
    units.write_marker(world.g, _marker(world, "inv-other"))
    serving = world.serve_on_restart(world.primary)
    run_activate(world, world.primary)
    serving.join()
    assert units.read_marker(world.g) is not None


def test_a_vllm_binary_record_activates_through_the_unit_without_a_window_probe(world):
    spec = _vllm("binary")
    world.fast_polls()
    world.fast_loan()
    world.http(served="/m/ckpt")
    serving = world.serve_on_restart(spec)
    live, _ = run_activate(world, spec, autosense=None)
    serving.join()
    assert live.served_models == ["/m/ckpt"] and live.window is None
    assert world.target() == "server:"
