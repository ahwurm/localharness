"""BENCH-04: aggregator math + sequential stopping rule."""
from __future__ import annotations
import math
import pytest


def test_continuous_ci_95_matches_scipy():
    """continuous_ci_95(samples) returns Student's-t 95% CI matching scipy reference."""
    import numpy as np
    from scipy import stats
    from localharness.bench.aggregator import continuous_ci_95
    samples = [1.0, 1.1, 0.9, 1.05, 0.95]
    ci = continuous_ci_95(samples)
    arr = np.array(samples)
    n = len(arr)
    mean_ref = arr.mean()
    se = arr.std(ddof=1) / math.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    half_ref = t_crit * se
    assert math.isclose(ci.mean, mean_ref, rel_tol=1e-6)
    assert math.isclose(ci.lower, mean_ref - half_ref, rel_tol=1e-6)
    assert math.isclose(ci.upper, mean_ref + half_ref, rel_tol=1e-6)


def test_continuous_ci_95_degenerate_n_lt_2():
    """continuous_ci_95 with n<2 returns half_width_pct=inf (degenerate marker)."""
    from localharness.bench.aggregator import continuous_ci_95
    ci = continuous_ci_95([1.0])
    assert math.isinf(ci.half_width_pct)


def test_wilson_ci_95_basic():
    """wilson_ci_95(5, 10) returns Wilson 95% CI with p_hat=0.5."""
    from localharness.bench.aggregator import wilson_ci_95
    ci = wilson_ci_95(5, 10)
    assert math.isclose(ci.p_hat, 0.5, rel_tol=1e-6)
    assert 0.0 < ci.lower < 0.5 < ci.upper < 1.0


def test_wilson_ci_95_zero_successes():
    """wilson_ci_95(0, 5) returns p_hat=0.0, lower=0.0, upper>0 (not Wald-broken)."""
    from localharness.bench.aggregator import wilson_ci_95
    ci = wilson_ci_95(0, 5)
    assert ci.p_hat == 0.0
    assert ci.lower == 0.0
    assert ci.upper > 0.0


def test_wilson_ci_95_all_successes():
    """wilson_ci_95(5, 5) returns p_hat=1.0, lower<1.0, upper=1.0."""
    from localharness.bench.aggregator import wilson_ci_95
    ci = wilson_ci_95(5, 5)
    assert ci.p_hat == 1.0
    assert ci.lower < 1.0
    assert ci.upper == 1.0


def test_welch_ab_test_regression_detected():
    """welch_ab_test: head > baseline + p<0.05 → returns regressed=True."""
    from localharness.bench.aggregator import welch_ab_test
    baseline = [1.0, 1.05, 0.95, 1.0, 1.02]
    head = [1.5, 1.55, 1.45, 1.5, 1.52]
    t_stat, p_value, regressed = welch_ab_test(baseline, head)
    assert regressed is True
    assert p_value < 0.05


def test_welch_ab_test_no_regression_same_means():
    """welch_ab_test: head ~= baseline → regressed=False, p>0.05."""
    from localharness.bench.aggregator import welch_ab_test
    baseline = [1.0, 1.05, 0.95, 1.0, 1.02]
    head = [1.0, 1.05, 0.95, 1.0, 1.02]
    _, p_value, regressed = welch_ab_test(baseline, head)
    assert regressed is False


def test_metrics_summary_median_p95(fake_completed_runs):
    """metrics_summary returns median, p95, mean, std, n per SCEN-02 numeric field."""
    from localharness.bench.aggregator import metrics_summary
    runs = fake_completed_runs(n=5)
    summary = metrics_summary(runs)
    for field in ("latency_ttft", "latency_total", "tokens_in", "tokens_out",
                  "iterations", "parse_failures", "stuck_recoveries", "tool_call_count"):
        assert field in summary
        assert "median" in summary[field]
        assert "p95" in summary[field]
        assert "n" in summary[field]
        assert summary[field]["n"] == 5
    assert "success_rate" in summary
    assert summary["success_rate"]["rate"] == 1.0


def test_should_stop_min_runs(fake_completed_runs):
    """should_stop returns (False, ...) before min_runs even if CI tight."""
    from localharness.bench.aggregator import should_stop
    runs = fake_completed_runs(n=2, latency_total=[1.0, 1.0])
    stop, reason = should_stop(runs, tolerance=0.10, min_runs=3, max_runs=20)
    assert stop is False
    assert "min_runs" in reason


def test_should_stop_max_runs(fake_completed_runs):
    """should_stop returns (True, max_runs_hit) at max_runs regardless of CI."""
    from localharness.bench.aggregator import should_stop
    runs = fake_completed_runs(n=20, latency_total=[1.0 + i*0.5 for i in range(20)])  # huge variance
    stop, reason = should_stop(runs, tolerance=0.10, min_runs=3, max_runs=20)
    assert stop is True
    assert "max_runs" in reason


def test_should_stop_converged(fake_completed_runs):
    """should_stop returns (True, converged) when both CIs ≤ tolerance after min_runs."""
    from localharness.bench.aggregator import should_stop
    # tightly clustered latency, all successes
    runs = fake_completed_runs(n=10, latency_total=[1.0] * 10, success=[True] * 10)
    stop, reason = should_stop(runs, tolerance=0.10, min_runs=3, max_runs=20)
    assert stop is True
    assert "converged" in reason


def test_should_stop_not_converged(fake_completed_runs):
    """should_stop returns (False, not_converged) when CI too wide."""
    from localharness.bench.aggregator import should_stop
    runs = fake_completed_runs(n=4, latency_total=[1.0, 2.0, 1.5, 3.0], success=[True]*4)
    stop, reason = should_stop(runs, tolerance=0.10, min_runs=3, max_runs=20)
    assert stop is False
    assert "not_converged" in reason


# --- Phase 17: direction-aware Welch siblings (impl lands 17-02) ---
# welch_ab_test is HIGH-is-WORSE (latency). The promotion gate scores SUCCESS RATE,
# where HIGH-is-BETTER, so 17-02 ships one-sided siblings:
#   welch_improvement(baseline, head)  -> (t, p, improved)  improved iff head > baseline at p<alpha
#   welch_regression(baseline, head, alpha) -> bool          True iff head < baseline at p<alpha
# Both share welch_ab_test's n<2 insufficient-data guard: never flag.


def test_welch_improvement_detects_higher():
    """welch_improvement: head clearly GREATER than baseline → improved=True, p<0.05 (one-sided)."""
    from localharness.bench.aggregator import welch_improvement
    t, p, improved = welch_improvement(
        baseline=[0.2, 0.3, 0.2, 0.25, 0.3], head=[0.8, 0.9, 0.85, 0.8, 0.9]
    )
    assert improved is True
    assert p < 0.05


def test_welch_improvement_no_change_not_flagged():
    """welch_improvement: equal-ish arms → improved=False, p>=0.05 (no false promote)."""
    from localharness.bench.aggregator import welch_improvement
    _, p, improved = welch_improvement(
        baseline=[0.5, 0.55, 0.5, 0.5, 0.55], head=[0.5, 0.55, 0.5, 0.5, 0.55]
    )
    assert improved is False
    assert p >= 0.05


def test_welch_improvement_insufficient_data():
    """welch_improvement: n<2 guard returns exactly (0.0, 1.0, False) — never flag."""
    from localharness.bench.aggregator import welch_improvement
    assert welch_improvement([0.5], [0.9]) == (0.0, 1.0, False)


def test_welch_regression_detects_worse():
    """welch_regression: head significantly LOWER than baseline → True (regressed)."""
    from localharness.bench.aggregator import welch_regression
    assert welch_regression(
        baseline=[0.8, 0.9, 0.85, 0.8, 0.9], head=[0.2, 0.3, 0.2, 0.25, 0.3], alpha=0.05
    ) is True


def test_welch_regression_nonregressing_passes():
    """welch_regression: equal arms → False; n<2 guard → False (never flag a regression)."""
    from localharness.bench.aggregator import welch_regression
    assert welch_regression(
        baseline=[0.5, 0.55, 0.5, 0.5, 0.55], head=[0.5, 0.55, 0.5, 0.5, 0.55], alpha=0.05
    ) is False
    assert welch_regression([0.5], [0.9], 0.05) is False


# -------------------------------------------------------------------------
# should_stop: unanimous FAILURE must NOT auto-converge like unanimous SUCCESS does — a 0%
# success rate is exactly the signal sequential sampling exists to catch.
# -------------------------------------------------------------------------


def test_should_stop_all_failure_does_not_converge_early(fake_completed_runs):
    """All-failure samples past min_runs keep sampling (not_converged), unlike all-success."""
    from localharness.bench.aggregator import should_stop
    runs = fake_completed_runs(n=5, latency_total=[1.0] * 5, success=[False] * 5)
    stop, reason = should_stop(runs, tolerance=0.10, min_runs=3, max_runs=20)
    assert stop is False
    assert "not_converged" in reason


def test_should_stop_all_failure_stops_at_max_runs_with_reason(fake_completed_runs):
    """All-failure keeps sampling to max_runs, then stops with an annotated reason."""
    from localharness.bench.aggregator import should_stop
    runs = fake_completed_runs(n=20, latency_total=[1.0] * 20, success=[False] * 20)
    stop, reason = should_stop(runs, tolerance=0.10, min_runs=3, max_runs=20)
    assert stop is True
    assert reason == "max_runs_hit (20) — all failures"


def test_should_stop_all_success_still_converges_early(fake_completed_runs):
    """Regression guard: unanimous SUCCESS must still be allowed to converge early (only the
    all-FAILURE blanket pass was removed)."""
    from localharness.bench.aggregator import should_stop
    runs = fake_completed_runs(n=5, latency_total=[1.0] * 5, success=[True] * 5)
    stop, reason = should_stop(runs, tolerance=0.10, min_runs=3, max_runs=20)
    assert stop is True
    assert "converged" in reason


def test_should_stop_not_converged_renders_non_finite_as_na(fake_completed_runs):
    """The stop-reason formatter renders a non-finite half-width (all-failure => Wilson p=0 =>
    inf) as 'n/a', not a bare 'inf%'."""
    from localharness.bench.aggregator import should_stop
    runs = fake_completed_runs(n=5, latency_total=[1.0] * 5, success=[False] * 5)
    _, reason = should_stop(runs, tolerance=0.10, min_runs=3, max_runs=20)
    assert "succ=n/a" in reason
    assert "inf" not in reason


# -------------------------------------------------------------------------
# JSON writers must not emit bare Infinity/NaN — non-finite floats (e.g. wilson_ci_95's
# half_width_pct at p=0) are sanitized to null so the file is valid JSON for a strict parser.
# -------------------------------------------------------------------------


def test_write_summary_json_sanitizes_non_finite(tmp_path):
    import json
    from localharness.bench.report import write_summary_json
    summary = {
        "success_rate": {
            "rate": 0.0,
            "successes": 0,
            "n": 5,
            "wilson_ci": {"p_hat": 0.0, "lower": 0.0, "upper": 0.5, "half_width_pct": float("inf")},
        },
    }
    path = tmp_path / "summary.json"
    write_summary_json(
        path, summary, scenario_name="s", model="m",
        stop_reason="max_runs_hit (5) — all failures", n_runs=5,
    )
    raw = path.read_text()
    assert "Infinity" not in raw
    assert json.loads(raw)["metrics"]["success_rate"]["wilson_ci"]["half_width_pct"] is None


def test_sanitize_for_json_recursive():
    """_sanitize_for_json walks nested dicts/lists, replacing only non-finite floats."""
    from localharness.bench.report import _sanitize_for_json
    payload = {
        "a": float("inf"),
        "b": [1.0, float("-inf"), {"c": float("nan"), "d": "text"}],
        "e": 3,
    }
    out = _sanitize_for_json(payload)
    assert out == {"a": None, "b": [1.0, None, {"c": None, "d": "text"}], "e": 3}
