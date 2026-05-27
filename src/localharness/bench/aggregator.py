"""BENCH-04: aggregator math + sequential stopping rule.

Statistical correctness is the entire product of the bench harness. Hand-rolled stats
are minimized: scipy.stats provides Welch's t-test and Student's-t critical values;
Wilson score interval is closed-form and computed inline. numpy provides percentile.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


# -------------------------------------------------------------------------
# CI result types
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class ContinuousCI:
    mean: float
    lower: float
    upper: float
    half_width_pct: float  # (upper - mean) / mean * 100; inf when mean is 0 or n<2


@dataclass(frozen=True)
class ProportionCI:
    p_hat: float
    lower: float
    upper: float
    half_width_pct: float


# -------------------------------------------------------------------------
# Student's-t 95% CI on the mean (continuous metric stopping rule)
# -------------------------------------------------------------------------

def continuous_ci_95(samples: list[float]) -> ContinuousCI:
    """Student's-t 95% CI on the mean. Degenerate (n<2) returns half_width_pct=inf."""
    arr = np.asarray(samples, dtype=float)
    n = len(arr)
    if n < 2:
        mean = float(arr[0]) if n else 0.0
        return ContinuousCI(mean=mean, lower=mean, upper=mean, half_width_pct=float("inf"))
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(n))
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    half = t_crit * se
    half_pct = (half / mean * 100.0) if mean != 0 else float("inf")
    return ContinuousCI(mean=mean, lower=mean - half, upper=mean + half, half_width_pct=half_pct)


# -------------------------------------------------------------------------
# Wilson score 95% CI for proportions (success_rate stopping rule)
# -------------------------------------------------------------------------

_Z_95 = 1.959963984540054  # stats.norm.ppf(0.975) — hardcoded to avoid scipy call per invocation


def wilson_ci_95(successes: int, n: int) -> ProportionCI:
    """Wilson score interval at 95%. Correct at p near 0 or 1 (Wald is broken there)."""
    if n == 0:
        return ProportionCI(p_hat=0.0, lower=0.0, upper=0.0, half_width_pct=float("inf"))
    p = successes / n
    z = _Z_95
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    lower = (centre - margin) / denom
    upper = (centre + margin) / denom
    half = (upper - lower) / 2
    half_pct = (half / p * 100.0) if p > 0 else float("inf")
    return ProportionCI(p_hat=p, lower=lower, upper=upper, half_width_pct=half_pct)


# -------------------------------------------------------------------------
# Welch's t-test (A/B early-stop in bench compare)
# -------------------------------------------------------------------------

def welch_ab_test(baseline: list[float], head: list[float], alpha: float = 0.05) -> tuple[float, float, bool]:
    """Welch's t-test for A/B comparison. Returns (t_stat, p_value, regressed).

    regressed=True iff np.mean(head) > np.mean(baseline) AND p_value < alpha.
    Direction is HIGH-is-WORSE (matches latency_total, tokens_*, iterations, etc.).
    For LOW-is-WORSE metrics (success_rate), caller inverts inputs.
    """
    if len(baseline) < 2 or len(head) < 2:
        return (0.0, 1.0, False)  # insufficient data — never flag regression
    t_stat, p_value = stats.ttest_ind(head, baseline, equal_var=False)
    regressed = bool(np.mean(head) > np.mean(baseline)) and bool(p_value < alpha)
    return (float(t_stat), float(p_value), regressed)


# -------------------------------------------------------------------------
# Helper: read a field from either an event-like object or a dict
# -------------------------------------------------------------------------

def _field(sample: Any, name: str) -> Any:
    if isinstance(sample, dict):
        return sample[name]
    return getattr(sample, name)


# -------------------------------------------------------------------------
# metrics_summary — median, p95, mean, std, n per SCEN-02 field
# -------------------------------------------------------------------------

_NUMERIC_FIELDS = (
    "latency_ttft", "latency_total",
    "tokens_in", "tokens_out",
    "iterations",
    "parse_failures", "stuck_recoveries", "tool_call_count",
)


def metrics_summary(samples: list[Any]) -> dict[str, dict]:
    """Compute median, p95, mean, std, n per SCEN-02 numeric field + success_rate with Wilson CI."""
    if not samples:
        raise ValueError("metrics_summary requires at least one sample")
    out: dict[str, dict] = {}
    for field in _NUMERIC_FIELDS:
        values = np.array([float(_field(s, field)) for s in samples], dtype=float)
        n = len(values)
        out[field] = {
            "median": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if n > 1 else 0.0,
            "n": n,
        }
    successes = sum(1 for s in samples if bool(_field(s, "success")))
    n = len(samples)
    ci = wilson_ci_95(successes, n)
    out["success_rate"] = {
        "rate": successes / n,
        "successes": successes,
        "n": n,
        "wilson_ci": {
            "p_hat": ci.p_hat,
            "lower": ci.lower,
            "upper": ci.upper,
            "half_width_pct": ci.half_width_pct,
        },
    }
    return out


# -------------------------------------------------------------------------
# should_stop — sequential adaptive sampling stopping rule
# -------------------------------------------------------------------------

def should_stop(samples: list[Any], tolerance: float, min_runs: int, max_runs: int) -> tuple[bool, str]:
    """Decide whether to stop sampling.

    Returns (stop, reason). Both gating metrics — latency_total (Student's-t CI on mean)
    and success_rate (Wilson) — must have half-width-pct <= tolerance*100 to converge.

    Order of checks:
      1. n < min_runs → (False, need_min_runs)
      2. n >= max_runs → (True, max_runs_hit)
      3. both CIs converged → (True, converged ...)
      4. else → (False, not_converged ...)
    """
    n = len(samples)
    if n < min_runs:
        return (False, f"need_min_runs ({n}/{min_runs})")
    if n >= max_runs:
        return (True, f"max_runs_hit ({n})")
    lat_ci = continuous_ci_95([float(_field(s, "latency_total")) for s in samples])
    successes = sum(1 for s in samples if bool(_field(s, "success")))
    succ_ci = wilson_ci_95(successes, n)
    tol_pct = tolerance * 100.0
    lat_ok = lat_ci.half_width_pct <= tol_pct
    # Extreme proportions (all-success or all-failure) have unbounded relative half-width
    # on the Wilson interval but a structurally unambiguous estimate at n >= min_runs.
    # Treat them as converged on the success_rate axis once we cleared the min_runs floor.
    if successes == 0 or successes == n:
        succ_ok = True
    else:
        succ_ok = succ_ci.half_width_pct <= tol_pct
    if lat_ok and succ_ok:
        return (True, f"converged (n={n}, lat={lat_ci.half_width_pct:.1f}%, succ={succ_ci.half_width_pct:.1f}%)")
    return (False, f"not_converged (lat={lat_ci.half_width_pct:.1f}%, succ={succ_ci.half_width_pct:.1f}%, target={tol_pct:.1f}%)")
