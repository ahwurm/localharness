"""BENCH-04 report writers — summary.json (machine) + summary.md (human)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def render_markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    """Return a markdown table string. Pure — no IO.

    Each row dict must contain all keys in headers. Values are stringified via str().
    """
    # Column widths: max(header, max(values))
    widths: list[int] = []
    for h in headers:
        col_vals = [str(r.get(h, "")) for r in rows]
        widths.append(max(len(h), *(len(v) for v in col_vals)) if col_vals else len(h))

    def _row(values: list[str]) -> str:
        cells = [f" {v.ljust(w)} " for v, w in zip(values, widths)]
        return "|" + "|".join(cells) + "|"

    header_line = _row(headers)
    sep_line = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body_lines = [_row([str(r.get(h, "")) for h in headers]) for r in rows]
    return "\n".join([header_line, sep_line, *body_lines])


def write_summary_json(
    summary_path: Path,
    summary: dict[str, Any],
    scenario_name: str,
    model: str,
    stop_reason: str,
    n_runs: int,
) -> None:
    """Write per-scenario summary as machine-readable JSON.

    Schema:
      {
        "model": str,
        "scenario": str,
        "n_runs": int,
        "stop_reason": str,
        "generated_at": str (UTC ISO 8601),
        "metrics": { ... metrics_summary dict ... }
      }
    """
    payload = {
        "model": model,
        "scenario": scenario_name,
        "n_runs": n_runs,
        "stop_reason": stop_reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": summary,
    }
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_summary_md(
    summary_path: Path,
    per_scenario: dict[str, dict[str, Any]],
    model: str,
) -> None:
    """Write the model's roll-up summary.md with one section per scenario.

    per_scenario shape:
      {
        "scenario_name": {
          "summary": <metrics_summary dict>,
          "stop_reason": str,
          "n_runs": int,
        },
        ...
      }
    """
    lines: list[str] = [f"# Bench Summary — {model}", ""]
    lines.append(f"_generated: {datetime.now(timezone.utc).isoformat()}_")
    lines.append("")

    for scen_name, info in per_scenario.items():
        summary = info["summary"]
        stop_reason = info["stop_reason"]
        n_runs = info["n_runs"]

        lines.append(f"## {scen_name}")
        lines.append("")
        lines.append(f"- stop_reason: `{stop_reason}`")
        lines.append(f"- n_runs: {n_runs}")
        lines.append("")

        # Numeric metrics table
        rows: list[dict[str, Any]] = []
        for metric in (
            "latency_ttft", "latency_total",
            "tokens_in", "tokens_out",
            "iterations",
            "parse_failures", "stuck_recoveries", "tool_call_count",
        ):
            m = summary.get(metric, {})
            rows.append({
                "metric": metric,
                "median": f"{m.get('median', 0):.3f}",
                "p95": f"{m.get('p95', 0):.3f}",
                "mean": f"{m.get('mean', 0):.3f}",
                "std": f"{m.get('std', 0):.3f}",
                "n": str(m.get("n", 0)),
            })
        lines.append(render_markdown_table(rows, ["metric", "median", "p95", "mean", "std", "n"]))
        lines.append("")

        # success_rate row with Wilson CI
        sr = summary.get("success_rate", {})
        wilson = sr.get("wilson_ci", {})
        sr_rows = [{
            "metric": "success_rate",
            "rate": f"{sr.get('rate', 0):.3f}",
            "wilson_lower": f"{wilson.get('lower', 0):.3f}",
            "wilson_upper": f"{wilson.get('upper', 0):.3f}",
            "successes": str(sr.get("successes", 0)),
            "n": str(sr.get("n", 0)),
        }]
        lines.append(render_markdown_table(sr_rows, ["metric", "rate", "wilson_lower", "wilson_upper", "successes", "n"]))
        lines.append("")

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines))
