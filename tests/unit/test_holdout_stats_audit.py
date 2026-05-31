"""AUDIT-05 — gate stats & holdout: code-audit + ONE-safe-unit characterization.

This module is the AUDIT-05 deliverable for Phase 21 (post-v1.1 bench/autoresearch
audit sweep). It is CODE-AUDIT + pure-function characterization ONLY.

SC9 (sealed holdout) — load-bearing here: the holdout slice is SEALED and must
NEVER be executed. Every test below is either a pure-function call
(``welch_regression``, ``pareto_front_2d``) or an ``inspect.getsource`` source
audit. NONE construct a ``slice="holdout"`` run, none call ``run_experiment`` far
enough to reach the holdout stage (experiment.py:444-445), and none read holdout
traces. The literal text ``slice="holdout"`` appears in this file ONLY as a grep
target inside an ``inspect.getsource`` substring assertion — never as a real call.

Findings pinned (documented in AUDIT-FINDINGS.md, plan 21-07):
- AUDIT-05a: the holdout stage reuses the SAME ``run_slice`` closure as train, so
  every Area-B seam applies identically to holdout — proven by the shared closure
  (source), NOT by running the sealed slice.
- AUDIT-05b: the TRAIN path guards ``if len(names) < 2`` (experiment.py:428-430)
  but the HOLDOUT path calls ``welch_regression(bh_vec, hh_vec)`` (experiment.py:448)
  with NO symmetric ``len(hnames) < 2`` guard. ``welch_regression`` has its OWN
  internal ``n<2`` guard (aggregator.py:124-125 -> returns False), so the practical
  blast radius is "silently returns not-regressed" (the proposal PASSES the holdout
  gate by default), NOT a raised error.
- AUDIT-05c: the seal is correctly enforced at the archive layer —
  ``pareto_front_2d(["holdout_score", "cost"])`` raises ValueError (archive.py:411-417).
"""

import inspect

import pytest

from localharness.autoresearch import experiment
from localharness.bench.aggregator import welch_regression


# ---------------------------------------------------------------------------
# AUDIT-05b — welch_regression edge-case (the practical blast radius)
# ---------------------------------------------------------------------------


def test_welch_regression_empty_and_size1_returns_false_not_raises():
    """The unguarded holdout call's practical blast radius is "silently not-regressed".

    The unguarded holdout call ``welch_regression(bh_vec, hh_vec)`` (experiment.py:448)
    would receive empty / size-1 vectors on a degenerate holdout. ``welch_regression``
    has its OWN n<2 guard (aggregator.py:124-125 -> returns False), so the practical
    effect is "silently returns not-regressed" — the proposal PASSES the holdout gate
    by default — NOT a raised StatisticsError. This documents the blast radius of the
    missing ``len(hnames) < 2`` guard (see the asymmetry test below).
    """
    # GREEN characterization: pins the current behavior. Flips RED if a future change
    # removes the internal guard and welch_regression starts raising on degenerate input.
    assert welch_regression([], [], alpha=0.05) is False
    assert welch_regression([0.5], [0.5], alpha=0.05) is False
    assert welch_regression([0.5], [], alpha=0.05) is False


def test_holdout_path_lacks_train_guard_asymmetry():
    """STATS-01 post-fix: the HOLDOUT path now has a symmetric len(hnames) < 2 guard.

    The 14-03 / test_agent_loop_uses_config_recovery_message_not_hardcoded precedent:
    prove the guard via ``inspect.getsource`` WITHOUT executing anything.
    After STATS-01, the source-level pin finds the symmetric guard present — the asymmetry
    documented by AUDIT-05b is now closed.
    """
    src = inspect.getsource(experiment.run_experiment)
    # TRAIN path HAS the inconclusive guard (experiment.py:428-430):
    assert "len(names) < 2" in src, (
        "train inconclusive guard missing — finding stale"
    )
    # HOLDOUT path calls welch_regression:
    assert "welch_regression(bh_vec, hh_vec" in src, (
        "holdout welch_regression call moved — re-anchor"
    )
    # STATS-01: the symmetric holdout guard NOW EXISTS (AUDIT-05b closed):
    assert "len(hnames) < 2" in src, (
        "symmetric holdout guard (STATS-01) is MISSING — len(hnames) < 2 must appear BEFORE "
        "the holdout welch_regression call (experiment.py) to mirror the train guard"
    )
    # statistics.mean IS guarded on the holdout side — refines AUDIT-05b: the unguarded
    # callee was welch_regression, NOT statistics.mean.
    assert "statistics.mean(hh_vec) if hh_vec else None" in src
    # NOTE (intended, NOT a finding): alpha_corr = 0.05 / trials is Bonferroni across
    # concurrent TRIALS (EXP-04 design) — confirmed intended.


# ---------------------------------------------------------------------------
# AUDIT-05a — holdout shares the broken factory (the shared run_slice closure)
# ---------------------------------------------------------------------------


def test_holdout_shares_broken_factory_documented():
    """Source-level pin: the holdout stage reuses the SAME run_slice closure as train.

    The (a) finding is a CODE-AUDIT (the shared closure), proven by source, NEVER by
    executing the sealed holdout slice. Both the train and holdout stages call
    ``run_slice(wt, slice=..., with_overlay=...)`` against the single default closure
    built at experiment.py:406-410 — so every Area-B seam (hardcoded ollama / bench
    default, probe-skip, native default mode) applies identically to holdout: the
    holdout would target the same mis-configured endpoint/model. This is AUDIT-05(a),
    documented in AUDIT-FINDINGS.md (plan 21-07), proven WITHOUT running holdout.
    """
    src = inspect.getsource(experiment.run_experiment)
    # The slice="holdout" literal below is a SOURCE SUBSTRING grep target (asserted via
    # inspect.getsource) — it is NEVER an executed run_slice call (SC9).
    assert 'run_slice(wt, slice="holdout"' in src, (
        "holdout no longer uses the shared run_slice — re-audit"
    )
    assert 'run_slice(wt, slice="train"' in src
    # Exactly ONE default run_slice closure is built (experiment.py:406-410) and reused
    # for both stages:
    assert src.count("_build_default_run_slice(") <= 1


# ---------------------------------------------------------------------------
# AUDIT-05c — the seal is enforced at the archive layer (GREEN pin)
# ---------------------------------------------------------------------------


async def test_pareto_front_2d_rejects_holdout_score(archive_store):
    """GREEN pin: pareto_front_2d refuses holdout_score as a front axis (the seal teeth).

    ``_SEALED_COLUMNS = {"holdout_score"}`` (archive.py:98); ``pareto_front_2d``
    validates BEFORE any DB access (archive.py:411-413) -> the sealed-slice invariant
    is EXECUTABLE at the query layer (Pitfall 3). The Phase-15 test
    ``test_archive_pareto.py::test_metrics_rejects_sealed_column`` already covers this;
    this is the explicit AUDIT-05 anchor (a named duplicate pin — never a weakening) so
    a refactor can't silently remove the seal.
    """
    with pytest.raises(ValueError):
        await archive_store.pareto_front_2d(metrics=["holdout_score", "cost"])
    # Forward-compat tooth (archive.py:414-417): any metric set != {train_score, cost}
    # raises — so even a holdout_score paired with train_score is refused.
    with pytest.raises(ValueError):
        await archive_store.pareto_front_2d(metrics=["train_score", "holdout_score"])
