"""Phase 25 (MECH-02) — offline propose→seed→experiment run→verdict for the MECHANISM axis.

Proves agent.self_check.enabled (the axis 25-01 made first-class) flows end-to-end through the
EXISTING promotion gate and reaches the gate BAND (a real verdict, exit 0-3), NOT a structural
refusal (>=4) — while the anti-reward-hacking seal stays intact on the SAME path (off-registry /
holdout / sentinel / multi-component STILL refuse, with zero bench arms run).

Fully offline: an injected FakeRunSlice supplies per-fixture success maps, so run_experiment never
builds a worktree or calls an LLM (experiment.py:451 — run_slice injected => _build_default_run_slice
is bypassed). The holdout maps are in-memory in the fake; no bench/scenarios/holdout/ slice is ever
executed. TEST-ONLY: zero src change — the mutation rides the already-wired agent.* cascade and the
v1.2-GREEN-PIN gate/Welch path UNTOUCHED.

FakeRunSlice + the band-verdict fixtures are imported verbatim from the canonical gate suite
(tests/unit/test_experiment.py) — the same seam the EXP-01..05 contract is written against.
"""
from __future__ import annotations

import json

import pytest  # noqa: F401 — asyncio_mode=auto; imported for explicit test-runner discovery

from localharness.autoresearch import experiment as exp

# The canonical injected-bench seam + band fixtures (importable: repo root on sys.path).
from tests.unit.test_experiment import (
    FakeRunSlice,
    _EQUAL_TRAIN,
    _HOLDOUT_BASE,
    _HOLDOUT_HEAD_OK,
    _IMPROVED_TRAIN_BASE,
    _IMPROVED_TRAIN_HEAD,
)

# The gate band: every promote/reject/inconclusive verdict is < 4 (refusals occupy >=4).
_BAND = (exp.EXIT_PROMOTE, exp.EXIT_REJECT_TRAIN, exp.EXIT_REJECT_HOLDOUT, exp.EXIT_INCONCLUSIVE)


async def test_mechanism_e2e_reaches_band_promote(archive_store, seeded_inflight):
    """MECH-02 e2e: agent.self_check.enabled flows seed→run_experiment→verdict and reaches the gate
    BAND (a real verdict 0-3), NOT a structural refusal. Clear train improvement + non-regressing
    holdout → EXIT_PROMOTE=0. Fully offline via the injected FakeRunSlice."""
    pid = await seeded_inflight(
        archive_store, component="agent.self_check.enabled", before=False, after=True
    )
    fake = FakeRunSlice(
        train_base=_IMPROVED_TRAIN_BASE,
        train_head=_IMPROVED_TRAIN_HEAD,
        holdout_base=_HOLDOUT_BASE,
        holdout_head=_HOLDOUT_HEAD_OK,
    )
    code = await exp.run_experiment(pid, store=archive_store, run_slice=fake)

    assert code in _BAND  # a real gate verdict (0-3)
    assert code < 4, f"mechanism axis must reach a verdict, not a structural refusal (got {code})"
    assert code == exp.EXIT_PROMOTE  # this fixture (improved train + ok holdout) promotes
    # The gate actually RAN the arms (it did not refuse before the bench):
    assert ("train", False) in fake.calls and ("train", True) in fake.calls


async def test_mechanism_e2e_train_reject_is_still_in_band(archive_store, seeded_inflight):
    """Equal arms → EXIT_REJECT_TRAIN=1 — still a BAND verdict (a real gate decision on the
    mechanism), NOT a refusal. Confirms the axis is GATED, not waved through."""
    pid = await seeded_inflight(
        archive_store, component="agent.self_check.enabled", before=False, after=True
    )
    fake = FakeRunSlice(train_base=_EQUAL_TRAIN, train_head=_EQUAL_TRAIN)
    code = await exp.run_experiment(pid, store=archive_store, run_slice=fake)

    assert code == exp.EXIT_REJECT_TRAIN
    assert code < 4  # in-band verdict, not a structural refusal
    assert ("train", False) in fake.calls and ("train", True) in fake.calls  # arms ran


async def test_mechanism_e2e_seal_still_refuses_offregistry_and_multi(archive_store, seeded_inflight):
    """The mechanism axis does NOT weaken the seal: off-registry / holdout / sentinel + a
    multi-component diff STILL refuse (>=4) on the SAME e2e path, with NO bench arm run."""
    # off-registry / holdout / sentinel surface → EXIT_REFUSE_OFFREGISTRY=6, before any bench.
    for comp in ("bench.scenarios.holdout.01", "sentinel.overfit_gap_threshold"):
        pid = await seeded_inflight(archive_store, component=comp, before="a", after="b")
        fake = FakeRunSlice(train_base=_EQUAL_TRAIN, train_head=_EQUAL_TRAIN)
        code = await exp.run_experiment(pid, store=archive_store, run_slice=fake)
        assert code == exp.EXIT_REFUSE_OFFREGISTRY
        assert code >= 4
        assert len(fake.calls) == 0  # seal refuses BEFORE any bench arm (no goalpost-moving)

    # multi-component diff (even with the mechanism axis as one leg) → EXIT_REFUSE_MULTI_COMPONENT=4.
    multi = json.dumps(
        {
            "before": {"agent.self_check.enabled": False, "agent.role": "a"},
            "after": {"agent.self_check.enabled": True, "agent.role": "x"},
        }
    )
    pid = await seeded_inflight(archive_store, component="agent.self_check.enabled", diff=multi)
    fake = FakeRunSlice(train_base=_EQUAL_TRAIN, train_head=_EQUAL_TRAIN)
    code = await exp.run_experiment(pid, store=archive_store, run_slice=fake)
    assert code == exp.EXIT_REFUSE_MULTI_COMPONENT
    assert code >= 4
    assert len(fake.calls) == 0  # refused before any bench arm
