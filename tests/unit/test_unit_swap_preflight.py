"""A swap through the harness's own unit refuses what it can know BEFORE it stops the incumbent.

`lifecycle.swap(..., unit=state)` used to free the accelerator first and meet two refusals afterwards: a peer
name `unit-target` cannot hold (`UnitStrategy.activate`), and an `autostart.env` entry the machine record does
not cover (`serve` exits 78, and the restore of the incumbent is refused the same way). Both left the box with
nothing serving, or cost the person two model loads for a swap that could never succeed. The pre-flight
(`lifecycle.unit_launch_refusal`) runs first: the incumbent keeps serving, no verb is issued, no signal is sent,
`unit-target` is not touched, and the line names the fix.

The world is `test_lifecycle_swap_unit`'s: a faked manager, fake `/proc`, a `FakeServe` thread. Nothing real runs.
"""
from __future__ import annotations

import pytest

from localharness.config import trust
from localharness.provider import lifecycle, procfs, server, units
from localharness.provider.lifecycle import Heavy
from tests.unit.procfs_support import fake_proc  # noqa: F401 — the fixture
from tests.unit.test_lifecycle_swap_unit import (  # noqa: F401 — `world` is the fixture
    INCUMBENT_PID, _no_leaked_holds, flash_ready, heavy, swap_now, unit_incumbent, world,
)
from tests.unit.unit_strategy_support import INTERPRETER, UNIT as UNIT_NAME

FIX = "localharness model --autostart on"


def assert_untouched(world, outcome, rep, *, stage: str = "refused") -> None:
    """Nothing was stopped, restarted, signalled or written, and the incumbent is still up."""
    assert outcome.live is None and outcome.stopped is None and outcome.restored is None
    assert outcome.stage == stage and isinstance(outcome.error, RuntimeError)
    assert world.manager.verbs == [] and world.events == []
    assert isinstance(procfs.read_process(INCUMBENT_PID), procfs.ProcRecord)
    assert server.pid_path(world.g).exists()
    assert rep.events == []  # `freed` is never said: nothing was freed
    (line,) = rep.lines
    assert line.endswith("No change made — the current model should still be serving.")
    assert str(outcome.error) in line


# ------------------------------------------------------------------------ WR-03: a name the unit cannot hold


@pytest.mark.parametrize("name", ["Qwen 32B", "qwen-é", "x" * 129, "-leading", "server:", ""],
                         ids=["a_space", "non_ascii", "over_the_cap", "leading_dash", "the_reserved_name", "empty"])
def test_a_peer_name_unit_target_cannot_hold_is_refused_before_the_incumbent_is_stopped(world, name):
    unit_incumbent(world)
    units.write_target(world.g, "flash-next")
    target = Heavy(world.peer("flash-next"), world.url("flash-next"), name)

    outcome, rep = swap_now(world, heavy(world, world.primary), target)

    assert_untouched(world, outcome, rep)
    assert str(outcome.error) == lifecycle.UNIT_NAME_LINE.format(name=name)
    assert world.target() == "flash-next"  # `unit-target` is as it was


def test_a_name_the_unit_holds_at_the_cap_is_not_refused_by_the_pre_flight(world):
    name = "x" * 128
    spec = world.peer("flash-next")

    assert lifecycle.unit_launch_refusal(spec, name, world.state, world.g) is None
    assert lifecycle.unit_launch_refusal(spec, name + "x", world.state, world.g) == lifecycle.UNIT_NAME_LINE.format(
        name=name + "x")


def test_the_pre_flight_name_check_is_the_one_activate_backs_up(world):
    assert lifecycle.unit_name_refusal(None) is None
    assert lifecycle.unit_name_refusal("flash-next") is None
    assert lifecycle.unit_name_refusal("a b") == lifecycle.UNIT_NAME_LINE.format(name="a b")


# ------------------------------------------------------- WR-02: an autostart.env the record does not cover


def _change_the_unit_env(world) -> None:
    """The unit file now carries another `PATH` than the one the person confirmed (the file is still a render)."""
    units.write_harness_unit(units.render_harness_unit(
        INTERPRETER, world.g, {"PATH": "/changed/bin"}, home=units.harness_home(), version="0.19.0"))


def _decline_the_new_env(world) -> None:
    """The person said No to the changed environment: the record keeps the old entry, the file holds the new."""
    _change_the_unit_env(world)
    assert trust.autostart_env_failure(world.g) == trust.AUTOSTART_ENV_CHANGED


def _forget_the_entry(world) -> None:
    trust.forget_entries(world.g, [trust.AUTOSTART_KEY])


def _no_unit_file(world) -> None:
    units.remove_harness_unit()


@pytest.mark.parametrize("break_it, why", [
    (_change_the_unit_env, trust.AUTOSTART_ENV_CHANGED),
    (_decline_the_new_env, trust.AUTOSTART_ENV_CHANGED),
    (_forget_the_entry, trust.AUTOSTART_ENV_MISSING),
    (_no_unit_file, trust.AUTOSTART_ENV_UNREADABLE),
], ids=["changed", "withheld_after_a_no", "missing", "no_file"])
def test_an_autostart_env_the_record_does_not_cover_refuses_the_swap_with_the_incumbent_serving(world, break_it, why):
    unit_incumbent(world)
    units.write_target(world.g, "flash-next")
    break_it(world)
    target = heavy(world, world.peer("flash-next"), "flash-next")

    outcome, rep = swap_now(world, heavy(world, world.primary), target)

    assert_untouched(world, outcome, rep)
    assert str(outcome.error) == lifecycle.UNIT_ENV_LINE.format(why=why)
    assert FIX in str(outcome.error) and "localharness start in a terminal" in str(outcome.error)
    assert world.target() == "flash-next"


def test_an_untouched_record_lets_the_swap_through(world):
    unit_incumbent(world)
    serving = flash_ready(world)

    outcome, _ = swap_now(world, heavy(world, world.primary), heavy(world, world.peer("flash-next"), "flash-next"))
    serving.join()

    assert serving.error is None and outcome.error is None and outcome.stage is None
    assert outcome.live is not None


def test_a_target_the_unit_cannot_serve_is_not_judged_by_the_units_environment(world, monkeypatch):
    # a docker record launches outside the unit: `serve` never runs for it, so the record's gap is no reason
    _forget_the_entry(world)
    spec = lifecycle.ManagedServerConfig(runtime="vllm", launch="docker", docker_image="vllm/vllm-openai:x",
                                         model="/m/ckpt", port=8000)

    assert lifecycle.unit_launch_refusal(spec, "dock", world.state, world.g) is None
    assert lifecycle.unit_launch_refusal(spec, "a b", world.state, world.g) is None


# ---------------------------------------- WR-10: the way back (the restore of the incumbent) is judged too


def _docker_target() -> Heavy:
    spec = lifecycle.ManagedServerConfig(runtime="vllm", launch="docker", docker_image="vllm/vllm-openai:x",
                                         model="/m/ckpt", port=8000)
    return Heavy(spec, "http://127.0.0.1:8000/v1", "dock")


def test_a_unit_incumbent_whose_restore_the_record_would_refuse_blocks_a_swap_to_a_docker_target(world):
    unit_incumbent(world)
    units.write_target(world.g, "flash-next")
    _forget_the_entry(world)   # `serve` would exit 78 for the restore of the incumbent

    outcome, rep = swap_now(world, heavy(world, world.primary), _docker_target())

    assert_untouched(world, outcome, rep)
    assert str(outcome.error) == lifecycle.UNIT_RESTORE_LINE.format(
        label="qwen", why=lifecycle.UNIT_ENV_LINE.format(why=trust.AUTOSTART_ENV_MISSING))
    assert FIX in str(outcome.error) and "could not be put back" in str(outcome.error)
    assert world.target() == "flash-next"   # `unit-target` is as it was: the docker target did not take it away


def test_an_incumbent_peer_name_unit_target_cannot_hold_blocks_a_swap_to_a_target_outside_the_unit(world):
    unit_incumbent(world)
    incumbent = Heavy(world.primary, world.url(), "Qwen 32B")

    outcome, rep = swap_now(world, incumbent, _docker_target())

    assert_untouched(world, outcome, rep)
    assert str(outcome.error) == lifecycle.UNIT_RESTORE_LINE.format(
        label="Qwen 32B", why=lifecycle.UNIT_NAME_LINE.format(name="Qwen 32B"))


def test_the_target_is_judged_first_and_named_when_both_would_be_refused(world):
    unit_incumbent(world)
    _forget_the_entry(world)

    outcome, rep = swap_now(world, heavy(world, world.primary), heavy(world, world.peer("flash-next"), "flash-next"))

    assert_untouched(world, outcome, rep)
    assert str(outcome.error) == lifecycle.UNIT_ENV_LINE.format(why=trust.AUTOSTART_ENV_MISSING)


def test_a_covered_record_lets_a_swap_to_a_docker_target_through_with_the_unit_incumbent_stopped(world, monkeypatch):
    unit_incumbent(world)

    async def activate(self, spec, config_dir, base_url, **kw):
        return lifecycle.LiveEndpoint(base_url=base_url, served_models=["m"], handle=1)

    monkeypatch.setattr(lifecycle.ManagedVllmStrategy, "activate", activate)

    outcome, _ = swap_now(world, heavy(world, world.primary), _docker_target())

    assert outcome.error is None and outcome.live is not None
    assert world.manager.verbs == [("reset-failed", UNIT_NAME), ("stop", UNIT_NAME)]


@pytest.mark.parametrize("unit", [None, object(), units.UnitRuntimeState("a", 0)], ids=["none", "object", "raw"])
def test_without_the_harness_own_unit_the_pre_flight_says_nothing(world, unit):
    _forget_the_entry(world)

    assert lifecycle.unit_launch_refusal(world.peer("flash-next"), "a b", unit, world.g) is None


# ----------------------------------------------------------------------- the check itself (config layer)


def test_autostart_env_failure_is_none_when_the_record_holds_what_the_file_holds(world):
    assert trust.autostart_env_failure(world.g) is None


def test_autostart_env_failure_names_what_serve_would_exit_78_for(world):
    _forget_the_entry(world)
    assert trust.autostart_env_failure(world.g) == trust.AUTOSTART_ENV_MISSING
    trust.record_machine(world.g, trust.machine_snapshot(world.g))
    assert trust.autostart_env_failure(world.g) is None
    _change_the_unit_env(world)
    assert trust.autostart_env_failure(world.g) == trust.AUTOSTART_ENV_CHANGED
    _no_unit_file(world)
    assert trust.autostart_env_failure(world.g) == trust.AUTOSTART_ENV_UNREADABLE


def test_autostart_env_failure_reads_and_writes_nothing_else(world, monkeypatch):
    def refuse(*a, **kw):
        raise AssertionError("the pre-flight wrote the record")

    monkeypatch.setattr(trust, "record_machine", refuse)
    monkeypatch.setattr(trust, "adopt_entries", refuse)
    _change_the_unit_env(world)

    assert trust.autostart_env_failure(world.g) == trust.AUTOSTART_ENV_CHANGED
