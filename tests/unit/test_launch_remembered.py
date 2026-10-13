# ruff: noqa: F811  (the `popen` fixture is imported by name, then taken as a parameter)
"""TUNE-07 / TUNE-08 (remembered branch, spec 20 §6.0, §6.4, D-17, D-18): a shape that died is
remembered by `confirm_launch` and planned again only while the fresh fit says it still applies.

Nothing here starts a process or reads the host: `subprocess.Popen` is the recorder, free memory, the
weights and the probe are faked through their module seams (`Box`), and `wait_ready` is the scripted
fake. The ledger is a real file in `tmp_path`.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from localharness.config.models import AutosenseConfig
from localharness.provider import lifecycle, server, speed_stats, units
from localharness.provider.autosense import FreeMemory
from localharness.provider.speed_stats import RetryRecord
from tests.unit.launch_support import popen  # noqa: F401
from tests.unit.test_launch_fit_retry import BASE, Box, _argv, _launches
from tests.unit.test_launch_plan import FLASH_NEXT, SWIFT, _gguf_spec

GIB = 2**30
KNOBS = AutosenseConfig()
DEATH = "llama.cpp exited during startup. Log tail:\nfirst death"
NOW = datetime(2026, 10, 13, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def box(monkeypatch):
    world = Box(monkeypatch)
    world.mem = FreeMemory(24 * GIB, 48 * GIB, "nvidia-smi")
    world.weights = 18 * GIB
    return world


def _value(argv, flag) -> str:
    return argv[list(argv).index(flag) + 1]


def _ledger(tmp_path: Path) -> Path:
    return speed_stats.default_speed_stats_path(tmp_path)


async def _activate(tmp_path, spec, plans=None, autosense=KNOBS):
    return await lifecycle.SpawnedProcessStrategy().activate(
        spec, tmp_path, BASE, autosense=autosense, on_plan=(plans.append if plans is not None else None))


def _sense(spec, tmp_path, autosense=KNOBS):
    return lifecycle.sense_launch(spec, tmp_path, autosense)


def _plan(spec, sensed, autosense=KNOBS):
    return lifecycle.plan_launch(spec, sensed, server.launch_decision(spec, False), autosense=autosense,
                                 api_key_held=False)


def _record(spec, tmp_path, **over) -> RetryRecord:
    fields = dict(full_fit=100_000, free_bytes=24 * GIB, dropped=("draft-mtp",), binary=str(spec.binary),
                  kv_cache_type="q8_0", draft_p_min=0.6, at="2026-10-01T09:00:00+00:00")
    fields.update(over)
    return RetryRecord(**fields)


def _remembered(spec, tmp_path, **over) -> lifecycle.Sensed:
    return dataclasses.replace(_sense(spec, tmp_path), remembered=_record(spec, tmp_path, **over))


# ------------------------------------------------------------------- the tracer


async def test_a_death_retry_is_remembered_planned_again_and_cleared_by_a_larger_fit(
        tmp_path, box, monkeypatch):
    calls = _launches(box, monkeypatch)
    spec = _gguf_spec(tmp_path, SWIFT)
    key = lifecycle.record_key(spec)
    assert key == speed_stats.speed_key("llamacpp", str(spec.model))

    # run 1: the full shape dies, the retry answers -> the record is written from the plan's values
    box.ready_script = [RuntimeError(DEATH), ["qwen"]]
    plans: list = []
    await _activate(tmp_path, spec, plans)
    full_fit_1 = plans[0].full_fit
    assert full_fit_1 and plans[1].cause == "death"
    rec = speed_stats.last_retry(_ledger(tmp_path), key)
    assert rec is not None
    assert (rec.full_fit, rec.free_bytes, rec.dropped) == (full_fit_1, 24 * GIB, ("draft-mtp",))
    assert (rec.binary, rec.kv_cache_type, rec.draft_p_min) == (str(spec.binary), "q8_0", 0.6)
    assert datetime.fromisoformat(rec.at).tzinfo is not None

    # run 2: same memory -> the first plan IS the retry shape, -c from the fresh fit halved
    box.ready_script = [["qwen"]]
    plans = []
    await _activate(tmp_path, spec, plans)
    first = plans[0]
    assert len(plans) == 1 and len(calls) == 3
    assert (first.shape, first.cause, first.remembered_at) == ("retry", "remembered", rec.at)
    argv = _argv(calls, 2)
    assert "--spec-type" not in argv and tuple(argv) == first.argv
    assert int(_value(argv, "-c")) == full_fit_1 // 2
    assert first.dropped == ("draft-mtp",) and first.tuned.spec_type is None
    assert speed_stats.last_retry(_ledger(tmp_path), key) == rec  # a remembered success leaves it

    # run 2b: less memory -> its own smaller fit halved, never run 1's number
    box.mem = FreeMemory(22 * GIB, 48 * GIB, "nvidia-smi")
    box.ready_script = [["qwen"]]
    plans = []
    await _activate(tmp_path, spec, plans)
    smaller = plans[0].full_fit
    assert smaller < full_fit_1 and plans[0].cause == "remembered"
    assert int(_value(_argv(calls, 3), "-c")) == smaller // 2

    # run 3: more memory, a fresh fit above the recorded one -> the full shape, which clears the record
    box.mem = FreeMemory(44 * GIB, 48 * GIB, "nvidia-smi")
    box.ready_script = [["qwen"]]
    plans = []
    await _activate(tmp_path, spec, plans)
    assert plans[0].full_fit > full_fit_1
    assert (plans[0].shape, plans[0].cause) == ("full", None)
    assert _value(_argv(calls, 4), "--spec-type") == "draft-mtp"
    assert speed_stats.last_retry(_ledger(tmp_path), key) is None


async def test_confirm_launch_writes_nothing_but_the_ledger(tmp_path, box, monkeypatch):
    _launches(box, monkeypatch)
    spec = _gguf_spec(tmp_path, SWIFT)
    box.ready_script = [RuntimeError(DEATH), ["qwen"]]
    plans: list = []
    live = await _activate(tmp_path, spec, plans)
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    ledger = _ledger(tmp_path)
    ledger.unlink()
    lifecycle.confirm_launch(tmp_path, lifecycle.record_key(spec), plan=plans[1], live=live)
    after = {p for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before and ledger in after


async def test_a_full_shape_success_with_nothing_remembered_writes_nothing(tmp_path, box, monkeypatch):
    _launches(box, monkeypatch)
    await _activate(tmp_path, _gguf_spec(tmp_path, SWIFT))
    assert not _ledger(tmp_path).exists()


async def test_an_unknown_fit_records_nothing(tmp_path, box, monkeypatch):
    """A death-retry whose plan has no fit (`full_fit` None) cannot be remembered: the next launch tries
    the full shape."""
    _launches(box, monkeypatch)
    spec = _gguf_spec(tmp_path, SWIFT, context=32768)  # the record sets -c: no fit, the spec type still derives
    box.ready_script = [RuntimeError(DEATH), ["qwen"]]
    plans: list = []
    await _activate(tmp_path, spec, plans)
    assert plans[1].shape == "retry" and plans[1].full_fit is None
    assert not _ledger(tmp_path).exists()


def test_sense_launch_reads_the_remembered_record_for_llamacpp_only(tmp_path, box):
    spec = _gguf_spec(tmp_path, SWIFT)
    speed_stats.record_retry(
        _ledger(tmp_path), lifecycle.record_key(spec), full_fit=90_000, free_bytes=None,
        dropped=["draft-mtp"], binary=str(spec.binary), kv_cache_type="q8_0", draft_p_min=0.6)
    sensed = _sense(spec, tmp_path)
    assert sensed.remembered is not None and sensed.remembered.full_fit == 90_000
    assert sensed.marker is None and sensed.prior_death is False


def test_sense_launch_without_a_unit_leaves_the_boot_fields_unset(tmp_path, box):
    sensed = lifecycle.sense_launch(_gguf_spec(tmp_path, SWIFT), tmp_path, KNOBS,
                                    unit=units.UnitRuntimeState("abc", 3))
    assert sensed.marker is None and sensed.prior_death is False  # the autostart phase reads the marker


# ------------------------------------------------------------------- a remembered record that no longer applies


def test_a_remembered_record_plans_the_retry_shape_with_the_fresh_fit_halved(tmp_path, box):
    spec = _gguf_spec(tmp_path, SWIFT)
    full = _plan(spec, _sense(spec, tmp_path))
    plan = _plan(spec, _remembered(spec, tmp_path, full_fit=full.full_fit))
    assert (plan.shape, plan.cause, plan.remembered_at) == ("retry", "remembered", "2026-10-01T09:00:00+00:00")
    assert plan.dropped == ("draft-mtp",) and "--spec-type" not in plan.argv
    assert int(_value(plan.argv, "-c")) == full.full_fit // 2 and plan.retry() is None


def test_an_unknown_fresh_fit_keeps_the_remembered_shape(tmp_path, box):
    """R-09c: a fresh fit that could not be computed is not larger than the record's."""
    box.mem = None
    spec = _gguf_spec(tmp_path, SWIFT)
    plan = _plan(spec, _remembered(spec, tmp_path))
    assert plan.shape == "retry" and plan.cause == "remembered"
    assert "--spec-type" not in plan.argv and "-c" not in plan.argv  # nothing to halve: the server sizes it


MOOT = {
    "binary": {"binary": "/x/other-llama-server"},
    "kv": {"kv_cache_type": "f16"},
    "p_min": {"draft_p_min": 0.5},
}


@pytest.mark.parametrize("field", sorted(MOOT))
def test_a_changed_binary_or_knob_makes_the_record_moot(tmp_path, box, field):
    spec = _gguf_spec(tmp_path, SWIFT)
    full = _plan(spec, _sense(spec, tmp_path))
    plan = _plan(spec, _remembered(spec, tmp_path, full_fit=full.full_fit, **MOOT[field]))
    assert (plan.shape, plan.cause) == ("full", None)
    assert plan.argv == full.argv


@pytest.mark.parametrize("over", [{"context": 32768}, {"speculative": "off"}])
def test_a_record_that_now_sets_context_or_speculative_makes_the_record_moot(tmp_path, box, over):
    spec = _gguf_spec(tmp_path, SWIFT, **over)
    sensed = _remembered(spec, tmp_path, full_fit=10**9)
    plan = _plan(spec, sensed)
    assert plan.cause is None and plan.shape == "full"
    assert plan.argv == _plan(spec, _sense(spec, tmp_path)).argv


def test_a_larger_fresh_fit_tries_the_full_shape(tmp_path, box):
    spec = _gguf_spec(tmp_path, SWIFT)
    full = _plan(spec, _sense(spec, tmp_path))
    plan = _plan(spec, _remembered(spec, tmp_path, full_fit=full.full_fit - 1))
    assert (plan.shape, plan.cause) == ("full", None)


def test_a_malformed_ledger_record_reads_as_no_record(tmp_path, box):
    spec = _gguf_spec(tmp_path, SWIFT)
    path = _ledger(tmp_path)
    path.write_text('{"%s": {"retry": {"full_fit": -5, "dropped": ["rm"]}}}' % lifecycle.record_key(spec))
    sensed = _sense(spec, tmp_path)
    assert sensed.remembered is None
    assert _plan(spec, sensed).shape == "full"


def test_a_remembered_record_without_a_dropped_draft_halves_the_window_only(tmp_path, box):
    """A `-c`-only retry on the Flash-Next header: no spec type is derived, none is dropped."""
    spec = _gguf_spec(tmp_path, FLASH_NEXT)
    full = _plan(spec, _sense(spec, tmp_path))
    plan = _plan(spec, _remembered(spec, tmp_path, full_fit=full.full_fit, dropped=()))
    assert (plan.shape, plan.cause, plan.dropped) == ("retry", "remembered", ())
    assert "--spec-type" not in plan.argv
    assert int(_value(plan.argv, "-c")) == full.full_fit // 2


def test_a_remembered_draft_drop_on_a_header_without_a_draft_and_no_fit_is_the_full_plan(tmp_path, box):
    spec = _gguf_spec(tmp_path, FLASH_NEXT, context=32768)  # no fit -c to halve, no spec type to drop
    plan = _plan(spec, _remembered(spec, tmp_path, full_fit=10**9, dropped=("draft-mtp",)))
    assert plan.shape == "full"


# ------------------------------------------------------------------- the boot verdict


def _marker(spec, tmp_path, **over) -> units.Marker:
    fields = dict(record=lifecycle.record_key(spec), binary=str(spec.binary), invocation="inv-1",
                  cause="death", shape="retry", planned_context=40_000, full_fit=80_000,
                  free_bytes=20 * GIB, dropped=("draft-mtp",), at=(NOW - timedelta(seconds=30)).isoformat())
    fields.update(over)
    return units.Marker(**fields)


def _verdict(spec, tmp_path, marker="default", unit="default", **over) -> bool:
    kw = dict(record_key=lifecycle.record_key(spec), binary=str(spec.binary), now=NOW, max_age_seconds=900)
    kw.update(over)
    m = _marker(spec, tmp_path) if marker == "default" else marker
    u = units.UnitRuntimeState("inv-2", 1) if unit == "default" else unit
    return lifecycle.boot_prior_death(m, u, **kw)


def test_boot_prior_death_needs_every_condition(tmp_path):
    spec = _gguf_spec(tmp_path, SWIFT)
    assert _verdict(spec, tmp_path) is True
    assert _verdict(spec, tmp_path, marker=_marker(spec, tmp_path, cause="none", shape="full")) is True
    assert _verdict(spec, tmp_path, marker=_marker(spec, tmp_path, cause="none", shape="retry")) is False
    assert _verdict(spec, tmp_path, marker=_marker(spec, tmp_path, cause="remembered", shape="retry")) is False
    assert _verdict(spec, tmp_path, marker=_marker(spec, tmp_path, record="llamacpp:other")) is False
    assert _verdict(spec, tmp_path, marker=_marker(spec, tmp_path, binary="/x/other")) is False
    assert _verdict(spec, tmp_path, unit=units.UnitRuntimeState("inv-2", 0)) is False
    assert _verdict(spec, tmp_path, marker=_marker(spec, tmp_path, at=(NOW - timedelta(seconds=900)).isoformat())) is False
    assert _verdict(spec, tmp_path, marker=_marker(spec, tmp_path, at=(NOW - timedelta(seconds=899)).isoformat())) is True
    assert _verdict(spec, tmp_path, marker=None) is False
    assert _verdict(spec, tmp_path, unit=None) is False


@pytest.mark.parametrize("at", ["yesterday", "", "2026-10-13T11:59:00", (NOW + timedelta(hours=9)).isoformat()])
def test_boot_prior_death_refuses_an_unparsable_naive_or_future_time(tmp_path, at):
    spec = _gguf_spec(tmp_path, SWIFT)
    assert _verdict(spec, tmp_path, marker=_marker(spec, tmp_path, at=at)) is False


def test_the_invocation_plays_no_part_in_the_boot_verdict(tmp_path):
    spec = _gguf_spec(tmp_path, SWIFT)
    assert _verdict(spec, tmp_path, marker=_marker(spec, tmp_path, invocation=None)) is True


def test_a_prior_death_plans_the_boot_retry_shape(tmp_path, box):
    spec = _gguf_spec(tmp_path, SWIFT)
    full = _plan(spec, _sense(spec, tmp_path))
    plan = _plan(spec, dataclasses.replace(_sense(spec, tmp_path), prior_death=True))
    assert (plan.shape, plan.cause, plan.dropped, plan.remembered_at) == ("retry", "boot", ("draft-mtp",), None)
    assert "--spec-type" not in plan.argv and int(_value(plan.argv, "-c")) == full.full_fit // 2


def test_a_remembered_shape_wins_over_the_boot_verdict(tmp_path, box):
    spec = _gguf_spec(tmp_path, SWIFT)
    full = _plan(spec, _sense(spec, tmp_path))
    sensed = dataclasses.replace(_remembered(spec, tmp_path, full_fit=full.full_fit), prior_death=True)
    assert _plan(spec, sensed).cause == "remembered"


def test_a_prior_death_with_nothing_to_drop_or_halve_is_the_full_plan(tmp_path, box):
    spec = _gguf_spec(tmp_path, FLASH_NEXT, context=32768)
    plan = _plan(spec, dataclasses.replace(_sense(spec, tmp_path), prior_death=True))
    assert plan.shape == "full"


# ------------------------------------------------------------------- IN-01: only a death is "after one retry"


def test_a_remembered_or_boot_plan_is_not_reported_as_retried(tmp_path, box):
    from localharness.provider.autosense import window_lines

    spec = _gguf_spec(tmp_path, SWIFT)
    full = _plan(spec, _sense(spec, tmp_path))
    remembered = _plan(spec, _remembered(spec, tmp_path, full_fit=full.full_fit))
    boot = full._shaped("boot", drop_spec=True)
    for plan in (remembered, boot):
        assert plan.report.asked == full.full_fit // 2 and plan.report.retried is False
        assert "after one retry" not in window_lines(plan.report, KNOBS)[0]
    death = full.retry()
    assert death.report.retried is True
    assert "after one retry" in window_lines(death.report, KNOBS)[0]
