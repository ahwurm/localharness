"""BENCH-05: bench compare — regression alarm + per-metric diff report.

Exit code policy (LOCKED in CONTEXT.md):
  0 = stable (no regressions, all scenarios stable)
  1 = at least one scenario flagged a regression
  2 = infra failure (missing baseline/head dir, malformed summary.json, empty corpus)
  3 = at least one scenario in head summary has stable=False (unstable beats regressed)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from localharness.bench.aggregator import welch_ab_test, wilson_ci_95  # noqa: F401  (re-export for downstream use)
from localharness.bench.config import default_thresholds

log = logging.getLogger(__name__)


# Direction map: HIGH-is-worse for most metrics. success_rate uses LOW-is-worse via absolute_pp.
_CONTINUOUS_METRICS = (
    "latency_ttft", "latency_total",
    "tokens_in", "tokens_out",
    "iterations",
    "parse_failures", "stuck_recoveries", "tool_call_count",
)


# -------------------------------------------------------------------------
# CompareResult dataclasses — single source of truth for downstream consumers
# -------------------------------------------------------------------------

@dataclass
class ScenarioDiff:
    name: str
    regressions: dict[str, bool] = field(default_factory=dict)
    unstable: bool = False
    per_metric: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class CompareResult:
    per_scenario: dict[str, ScenarioDiff] = field(default_factory=dict)
    any_regression: bool = False
    any_unstable: bool = False


# -------------------------------------------------------------------------
# Threshold resolution (CLI > scenario YAML > bench.yaml > built-in defaults)
# -------------------------------------------------------------------------

def _parse_cli_threshold(token: str) -> tuple[str, dict[str, Any]]:
    """Parse "metric=value" into (metric, {"type": ..., "value": float}).

    Picks default type based on metric: success_rate -> absolute_pp, others -> relative.
    """
    if "=" not in token:
        raise ValueError(f"Invalid --threshold token {token!r}; expected metric=value")
    metric, raw = token.split("=", 1)
    metric = metric.strip()
    value = float(raw)
    ttype = "absolute_pp" if metric == "success_rate" else "relative"
    return metric, {"type": ttype, "value": value}


def _normalize_threshold(t: Any) -> dict[str, Any]:
    """Coerce a ThresholdSpec-or-dict into {'type': ..., 'value': ...}."""
    if isinstance(t, dict):
        return {"type": t.get("type"), "value": t.get("value")}
    return {"type": getattr(t, "type", None), "value": getattr(t, "value", None)}


def resolve_thresholds(
    cli_overrides: Optional[list[str]] = None,
    scenario_yaml_thresholds: Optional[dict[str, Any]] = None,
    bench_yaml_thresholds: Optional[dict[str, Any]] = None,
) -> dict[str, dict[str, Any]]:
    """Merge thresholds with precedence: CLI > scenario YAML > bench.yaml > built-in defaults.

    Returns dict[metric_name -> {"type": str, "value": float}]. ThresholdSpec field is `.type`,
    NOT `.kind` (locked in 11-02 schema).
    """
    merged: dict[str, dict[str, Any]] = {
        m: _normalize_threshold(t) for m, t in default_thresholds().items()
    }
    for layer in (bench_yaml_thresholds, scenario_yaml_thresholds):
        if not layer:
            continue
        for m, t in layer.items():
            merged[m] = _normalize_threshold(t)
    if cli_overrides:
        for token in cli_overrides:
            m, t = _parse_cli_threshold(token)
            merged[m] = t
    return merged


# -------------------------------------------------------------------------
# Per-metric comparison
# -------------------------------------------------------------------------

def _check_continuous(
    baseline: dict[str, Any],
    head: dict[str, Any],
    threshold: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """High-is-worse comparison. Returns (regressed, detail_record).

    threshold field name is `.type` (locked) with values: "relative" | "absolute" | "absolute_pp".
    """
    b_val = float(baseline.get("median", baseline.get("mean", 0.0)) if isinstance(baseline, dict) else 0.0)
    h_val = float(head.get("median", head.get("mean", 0.0)) if isinstance(head, dict) else 0.0)
    delta_abs = h_val - b_val
    delta_rel = (delta_abs / b_val) if b_val != 0 else (float("inf") if delta_abs > 0 else 0.0)

    ttype = threshold.get("type", "relative")
    tvalue = float(threshold.get("value", 0.0))

    if ttype == "relative":
        regressed = (delta_abs > 0) and (delta_rel > tvalue)
    elif ttype in ("absolute", "absolute_pp"):
        # "+1 over baseline" semantics: equal-to-threshold trips. Matches BENCH-05 unit tests.
        regressed = delta_abs >= tvalue
    else:
        regressed = False

    return regressed, {
        "baseline_value": b_val,
        "head_value": h_val,
        "delta_abs": delta_abs,
        "delta_rel": delta_rel,
        "threshold": threshold,
        "regressed": regressed,
    }


def _check_success_rate(
    baseline: dict[str, Any],
    head: dict[str, Any],
    threshold: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """LOW-is-worse success_rate. threshold.type must be absolute_pp; value is signed (e.g. -0.05)."""
    b_rate = float(baseline.get("rate", 0.0)) if isinstance(baseline, dict) else 0.0
    h_rate = float(head.get("rate", 0.0)) if isinstance(head, dict) else 0.0
    delta_pp = h_rate - b_rate

    tvalue = float(threshold.get("value", -0.05))
    if tvalue < 0:
        regressed = delta_pp < tvalue
    else:
        regressed = delta_pp < -tvalue

    return regressed, {
        "baseline_rate": b_rate,
        "head_rate": h_rate,
        "delta_pp": delta_pp,
        "threshold": threshold,
        "regressed": regressed,
    }


# -------------------------------------------------------------------------
# diff_summaries — returns CompareResult, not dict
# -------------------------------------------------------------------------

def _extract_scenario_block(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return a dict of {scenario_name: metric_block} from either summary shape.

    Supports:
      - roll-up summary: {"scenarios": {name: {...metrics...}}}
      - per-scenario summary.json (from write_summary_json): {"scenario": "name", "metrics": {...}}
      - flat metric dict (fallback)
    """
    if "scenarios" in summary and isinstance(summary["scenarios"], dict):
        return summary["scenarios"]
    if "scenario" in summary:
        return {summary["scenario"]: summary.get("metrics", summary)}
    return {"default": summary}


def diff_summaries(
    baseline_summary: dict[str, Any],
    head_summary: dict[str, Any],
    thresholds: dict[str, dict[str, Any]],
) -> CompareResult:
    """Compare baseline vs head summaries. Returns a CompareResult dataclass."""
    b_scenarios = _extract_scenario_block(baseline_summary)
    h_scenarios = _extract_scenario_block(head_summary)

    result = CompareResult()

    for scen_name in sorted(set(b_scenarios.keys()) & set(h_scenarios.keys())):
        b = b_scenarios[scen_name]
        h = h_scenarios[scen_name]
        sd = ScenarioDiff(name=scen_name)
        sd.unstable = not bool(h.get("stable", True))

        for metric in _CONTINUOUS_METRICS:
            t = thresholds.get(metric) or thresholds.get("default") or {"type": "relative", "value": 0.20}
            regressed, detail = _check_continuous(b.get(metric, {}), h.get(metric, {}), t)
            sd.regressions[metric] = regressed
            sd.per_metric[metric] = detail

        sr_t = thresholds.get("success_rate", {"type": "absolute_pp", "value": -0.05})
        regressed, detail = _check_success_rate(b.get("success_rate", {}), h.get("success_rate", {}), sr_t)
        sd.regressions["success_rate"] = regressed
        sd.per_metric["success_rate"] = detail

        result.per_scenario[scen_name] = sd
        if any(sd.regressions.values()):
            result.any_regression = True
        if sd.unstable:
            result.any_unstable = True

    return result


def format_diff_report(result: CompareResult, baseline_dir: Path, head_dir: Path) -> str:
    """Human-readable markdown diff report."""
    lines = [
        "# Bench Compare Report",
        "",
        f"- baseline: `{baseline_dir}`",
        f"- head: `{head_dir}`",
        "",
    ]
    if result.any_unstable:
        lines.append("**Overall verdict:** `unstable`")
    elif result.any_regression:
        lines.append("**Overall verdict:** `regressed`")
    else:
        lines.append("**Overall verdict:** `stable`")
    lines.append("")

    for scen_name, sd in result.per_scenario.items():
        suffix = "  _(UNSTABLE)_" if sd.unstable else ""
        lines.append(f"## {scen_name}{suffix}")
        lines.append("")
        lines.append("| metric | baseline | head | delta | threshold | regressed |")
        lines.append("|---|---|---|---|---|---|")
        for metric, det in sd.per_metric.items():
            if metric == "success_rate":
                base = f"{det.get('baseline_rate', 0):.3f}"
                head_v = f"{det.get('head_rate', 0):.3f}"
                delta = f"{det.get('delta_pp', 0):+.3f}pp"
            else:
                base = f"{det.get('baseline_value', 0):.3f}"
                head_v = f"{det.get('head_value', 0):.3f}"
                delta_rel = det.get("delta_rel", 0)
                if delta_rel in (float("inf"), float("-inf")):
                    delta = f"{det.get('delta_abs', 0):+.3f} (inf%)"
                else:
                    delta = f"{det.get('delta_abs', 0):+.3f} ({delta_rel*100:+.1f}%)"
            t = det.get("threshold", {})
            tstr = f"{t.get('type')}={t.get('value')}"
            marker = "**YES**" if sd.regressions.get(metric) else "no"
            lines.append(f"| {metric} | {base} | {head_v} | {delta} | {tstr} | {marker} |")
        lines.append("")
    return "\n".join(lines)


# -------------------------------------------------------------------------
# run_compare — locked exit code policy: 0=stable, 1=regressed, 2=infra, 3=unstable
# -------------------------------------------------------------------------

def _load_summaries(dir_path: Path) -> dict[str, dict[str, Any]]:
    """Scan dir_path for summary.json files; key by scenario name. Returns {} if dir missing."""
    if not dir_path.is_dir():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for jp in sorted(dir_path.rglob("summary.json")):
        try:
            data = json.loads(jp.read_text())
        except Exception:
            log.warning("malformed_summary path=%s", jp)
            continue
        if "scenarios" in data and isinstance(data["scenarios"], dict):
            for s_name, s_metrics in data["scenarios"].items():
                out[s_name] = s_metrics
        elif "scenario" in data:
            out[data["scenario"]] = data.get("metrics", data)
    return out


async def run_compare(
    baseline: Path,
    head: Path,
    threshold_overrides: Optional[list[str]] = None,
    json_output: bool = False,
) -> int:
    """Compare baseline vs head bench results. Returns locked exit code.

    Locked policy (CONTEXT.md):
      0 = stable (no regressions, no unstable scenarios)
      1 = at least one regression flagged
      2 = infra failure: missing dir, malformed summaries, empty corpus
      3 = at least one head scenario marked stable=False (unstable beats regressed)
    """
    baseline = Path(baseline)
    head = Path(head)

    if not baseline.is_dir() or not head.is_dir():
        log.error("missing_compare_dir baseline=%s head=%s", baseline, head)
        return 2

    b_summaries = _load_summaries(baseline)
    h_summaries = _load_summaries(head)
    common = sorted(set(b_summaries.keys()) & set(h_summaries.keys()))
    if not common:
        log.error("empty_corpus: no overlapping scenarios in baseline + head")
        return 2

    b_wrapped = {"scenarios": {n: b_summaries[n] for n in common}}
    h_wrapped = {"scenarios": {n: h_summaries[n] for n in common}}

    thresholds = resolve_thresholds(cli_overrides=threshold_overrides)
    result = diff_summaries(b_wrapped, h_wrapped, thresholds)

    try:
        report_path = head / "compare_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(format_diff_report(result, baseline, head))
    except Exception:
        log.exception("compare_report_write_failed")

    if json_output:
        import sys as _sys
        payload = {
            "any_regression": result.any_regression,
            "any_unstable": result.any_unstable,
            "scenarios": {
                n: {
                    "regressions": sd.regressions,
                    "unstable": sd.unstable,
                } for n, sd in result.per_scenario.items()
            },
        }
        _sys.stdout.write(json.dumps(payload, indent=2) + "\n")

    if result.any_unstable:
        return 3
    if result.any_regression:
        return 1
    return 0
