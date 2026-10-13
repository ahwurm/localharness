# ruff: noqa: F811  (the `popen` fixture is imported by name, then taken as a parameter)
"""TUNE-07 (death branch, spec 20 §6.4, D-16, D-35): the one retry is chosen by the plan and executed
from the same `Sensed`.

A launch with a derived spec type or a fit `-c` that dies before readiness is stopped (verified) and
relaunched once as `plan.retry()`, then given up with the log tail. A KV-only launch is not retried, a
record or `extra_args` value is never dropped, and a timeout is never retried. Nothing here starts a
process: `subprocess.Popen` is the recorder and `wait_ready` the scripted fake (`Box`).
"""
from __future__ import annotations

import pytest

from localharness.config.models import AutosenseConfig
from localharness.provider import lifecycle, server
from localharness.provider.autosense import FreeMemory
from tests.unit.launch_support import popen  # noqa: F401
from tests.unit.test_launch_fit_retry import BASE, Box, _argv, _launches
from tests.unit.test_launch_plan import FLASH_NEXT, SWIFT, _gguf_spec

GIB = 2**30
KNOBS = AutosenseConfig()
DEATH_1 = "llama.cpp exited during startup. Log tail:\nfirst death"
DEATH_2 = "llama.cpp exited during startup. Log tail:\nsecond death"


@pytest.fixture
def box(monkeypatch):
    world = Box(monkeypatch)
    world.mem = FreeMemory(40 * GIB, 48 * GIB, "nvidia-smi")
    world.weights = 18 * GIB
    return world


def _value(argv, flag) -> str:
    return argv[list(argv).index(flag) + 1]


def _has(argv, flag) -> bool:
    return flag in argv


async def _activate(tmp_path, spec, plans):
    return await lifecycle.SpawnedProcessStrategy().activate(
        spec, tmp_path, BASE, autosense=KNOBS, on_plan=plans.append)


# ------------------------------------------------------------------- the derived shape dies once


async def test_a_dead_swift_launch_is_relaunched_without_the_draft_and_at_half_the_window(
        tmp_path, box, monkeypatch):
    calls = _launches(box, monkeypatch)
    box.ready_script = [RuntimeError(DEATH_1), ["qwen"]]
    plans: list = []
    live = await _activate(tmp_path, _gguf_spec(tmp_path, SWIFT), plans)

    assert box.events == ["free_memory", "popen", "wait_ready", "stop_server", "popen", "wait_ready",
                          "probe_window"]
    first, second = _argv(calls, 0), _argv(calls, 1)
    assert _has(first, "--spec-type") and _has(first, "--spec-draft-p-min")
    assert not _has(second, "--spec-type") and not _has(second, "--spec-draft-p-min")
    assert _value(second, "--cache-type-k") == "q8_0" and _value(second, "--cache-type-v") == "q8_0"
    assert _value(second, "--cache-ram") == _value(first, "--cache-ram")
    assert int(_value(second, "-c")) == max(1, int(_value(first, "-c")) // 2)
    assert box.stops == [(tmp_path, "binary")]

    assert live.window.retried is True and live.window.asked == int(_value(second, "-c"))
    assert live.tuned.spec_type is None and live.tuned.cache_type == "q8_0"
    assert len(plans) == 2
    assert (plans[0].shape, plans[0].cause, plans[0].dropped) == ("full", None, ())
    assert (plans[1].shape, plans[1].cause, plans[1].dropped) == ("retry", "death", ("draft-mtp",))
    assert tuple(second) == plans[1].argv and live.tuned == plans[1].tuned


async def test_both_retry_stops_name_the_identity_they_must_stop(tmp_path, box, monkeypatch):
    """D-27: the stops inside `activate` pass `expect=self.launch_identity(spec)` like `stop` does."""
    _launches(box, monkeypatch)
    box.ready_script = [RuntimeError(DEATH_1), RuntimeError(DEATH_2)]
    spec = _gguf_spec(tmp_path, SWIFT)
    with pytest.raises(RuntimeError, match="second death"):
        await _activate(tmp_path, spec, [])
    identity = lifecycle.SpawnedProcessStrategy().launch_identity(spec)
    assert identity is not None
    assert box.stop_kw == [{"expect": identity}, {"expect": identity}]


async def test_the_retry_reuses_the_sensed_facts(tmp_path, box, monkeypatch):
    _launches(box, monkeypatch)
    box.ready_script = [RuntimeError(DEATH_1), ["qwen"]]
    await _activate(tmp_path, _gguf_spec(tmp_path, SWIFT), [])
    assert box.events.count("free_memory") == 1


# ------------------------------------------------------------------- the give-up paths


async def test_two_deaths_stop_twice_launch_twice_and_carry_the_second_tail(tmp_path, box, monkeypatch):
    calls = _launches(box, monkeypatch)
    box.ready_script = [RuntimeError(DEATH_1), RuntimeError(DEATH_2)]
    with pytest.raises(RuntimeError, match="second death"):
        await _activate(tmp_path, _gguf_spec(tmp_path, SWIFT), [])
    assert len(calls) == 2 and len(box.stops) == 2  # one retry, never a third launch
    assert box.events.count("probe_window") == 0


async def test_a_failing_first_stop_surfaces_the_original_death_and_does_not_relaunch(
        tmp_path, box, monkeypatch):
    calls = _launches(box, monkeypatch)
    box.ready_script = [RuntimeError(DEATH_1)]

    def stop_server(config_dir, launch="binary", timeout_seconds=30.0, **kw):
        raise RuntimeError("pid 4242 survived SIGKILL")

    monkeypatch.setattr(server, "stop_server", stop_server)
    with pytest.raises(RuntimeError, match="first death"):
        await _activate(tmp_path, _gguf_spec(tmp_path, SWIFT), [])
    assert len(calls) == 1


async def test_a_timeout_is_never_retried(tmp_path, box, monkeypatch):
    calls = _launches(box, monkeypatch)
    box.ready_script = [TimeoutError("llama.cpp not ready after 1800s.")]
    with pytest.raises(TimeoutError):
        await _activate(tmp_path, _gguf_spec(tmp_path, SWIFT), [])
    assert len(calls) == 1 and box.stops == []


async def test_a_kv_only_launch_is_not_retried(tmp_path, box, monkeypatch):
    """Flash-Next derives only the q8_0 pair and a record `context` leaves no fit `-c`: nothing the
    plan could drop or halve, so a death re-raises at once with no stop."""
    calls = _launches(box, monkeypatch)
    box.ready_script = [RuntimeError(DEATH_1)]
    plans: list = []
    spec = _gguf_spec(tmp_path, FLASH_NEXT, context=32768)
    with pytest.raises(RuntimeError, match="first death"):
        await _activate(tmp_path, spec, plans)
    assert len(calls) == 1 and box.stops == []
    assert plans[0].tuned.cache_type == "q8_0" and plans[0].retry() is None
    assert len(plans) == 1


# ------------------------------------------------------------------- what the person set is never dropped


async def test_a_record_speculative_draft_mtp_is_never_dropped_only_the_window_halves(
        tmp_path, box, monkeypatch):
    calls = _launches(box, monkeypatch)
    box.ready_script = [RuntimeError(DEATH_1), ["qwen"]]
    plans: list = []
    live = await _activate(tmp_path, _gguf_spec(tmp_path, SWIFT, speculative="draft-mtp"), plans)
    first, second = _argv(calls, 0), _argv(calls, 1)
    assert _value(second, "--spec-type") == "draft-mtp" and _value(first, "--spec-type") == "draft-mtp"
    assert int(_value(second, "-c")) == max(1, int(_value(first, "-c")) // 2)
    assert plans[1].dropped == () and plans[1].cause == "death"
    assert live.window.retried is True


async def test_a_hand_set_spec_type_in_extra_args_survives_the_retry(tmp_path, box, monkeypatch):
    calls = _launches(box, monkeypatch)
    box.ready_script = [RuntimeError(DEATH_1), ["qwen"]]
    extra = ["--spec-type", "draft-mtp", "--spec-draft-p-min", "0.9"]
    plans: list = []
    await _activate(tmp_path, _gguf_spec(tmp_path, SWIFT, extra_args=extra), plans)
    second = _argv(calls, 1)
    assert second[-len(extra) - 2:-2] == extra  # before the loopback --host pair, untouched
    assert plans[1].dropped == ()


async def test_a_derived_spec_type_is_dropped_and_a_record_context_is_kept(tmp_path, box, monkeypatch):
    calls = _launches(box, monkeypatch)
    box.ready_script = [RuntimeError(DEATH_1), ["qwen"]]
    plans: list = []
    live = await _activate(tmp_path, _gguf_spec(tmp_path, SWIFT, context=4096), plans)
    first, second = _argv(calls, 0), _argv(calls, 1)
    assert _has(first, "--spec-type") and not _has(second, "--spec-type")
    assert _value(first, "-c") == "4096" and _value(second, "-c") == "4096"
    assert plans[1].dropped == ("draft-mtp",)
    assert live.window.mode == "record" and live.tuned.spec_type is None


# ------------------------------------------------------------------- the retry is a pure function of the plan


async def test_retry_is_deterministic_and_final(tmp_path, box, monkeypatch):
    _launches(box, monkeypatch)
    plans: list = []
    await _activate(tmp_path, _gguf_spec(tmp_path, SWIFT), plans)
    plan = plans[0]
    first, again = plan.retry(), plan.retry()
    assert first is not None and first == again
    assert first.retry() is None
    assert first.shape == "retry" and plan.shape == "full"


async def test_the_retry_halves_from_the_full_fit_and_never_below_one(tmp_path, box, monkeypatch):
    from localharness.provider import autosense

    monkeypatch.setattr(autosense, "fit", lambda *a, **k: 1)
    calls = _launches(box, monkeypatch)
    box.ready_script = [RuntimeError(DEATH_1), ["qwen"]]
    plans: list = []
    live = await _activate(tmp_path, _gguf_spec(tmp_path, SWIFT), plans)
    assert _value(_argv(calls, 1), "-c") == "1" and live.window.asked == 1
    assert plans[1].full_fit == 1
