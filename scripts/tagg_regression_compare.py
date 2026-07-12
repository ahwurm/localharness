"""No-grouping-regression comparator (TAGG-04, Phase 36.2 Plan 03) — the reproducible, grep-
stable machinery the attended live proof (Plan 04) consumes.

It compares a FRESH designed-month verdict.json (produced by the re-keyed pipeline, Plans
01/02) against the committed run-17 baseline (.planning/runs/2026-07-11-run17/results/
verdict.json) on pinned B1/B2/B3/B4/A1 tolerances and emits EXACTLY one grep-stable line
`regression: none|detected` at column 0.

The pre-committed RULING-D KILL reads "no grouping regression vs the run-17 baseline": Plan 04
runs the live designed-month through the re-keyed code and greps THIS comparator's artifact for
the disposition (regression -> revert: flip tag_grouping_enabled False; the migration atom_tags
data is KEPT). Pure JSON in, one line out — no sema05 eval machinery is reinvented here.

PINNED METRIC (recorded verbatim in 36.2-03-SUMMARY.md; reproducible). regression: none IFF
ALL of:
    V.verdict == "HOLDS"
    V.stage_a.a1_recall >= 0.80
    set(V.stage_b.formed_topics) >= baseline formed_topics AND b1_chapters_or_null >= baseline's
    V.stage_b.b2_ok is true AND V.stage_b.ari >= baseline.ari - 0.06 band   (run-17 0.759 -> 0.70)
    V.stage_b.noise_in_chapter == 0                                          (b3 distractor count)
    V.stage_b.b4_ok is true
else: regression: detected (notes list every failing axis).
The formed-topics set and the ARI floor are DERIVED FROM THE BASELINE dict (never hardcoded), so
the metric tracks whatever baseline it is handed; a1_recall and the b3 bar are absolute floors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_A1_RECALL_FLOOR = 0.80   # absolute Stage-A recall floor
_ARI_BAND = 0.06          # run-to-run ARI wobble tolerated below the baseline (0.759 -> 0.70)
_DEFAULT_BASELINE = ".planning/runs/2026-07-11-run17/results/verdict.json"


def _stage_b(v: dict) -> dict:
    return v.get("stage_b", {}) or {}


def _ari(v: dict) -> float:
    return float(_stage_b(v).get("ari", 0.0) or 0.0)


def _formed_topics(v: dict) -> set:
    return set(_stage_b(v).get("formed_topics", []) or [])


def compare(candidate: dict, baseline: dict) -> tuple[bool, list[str]]:
    """Grade `candidate` against `baseline` on the pinned no-regression metric. Returns
    (regression, notes): regression is True iff ANY axis fails; notes names each failing axis
    (empty when regression is False). The expected topics and the ARI floor come from the
    baseline dict — the comparator tracks the baseline it is given, never a hardcoded topic list."""
    sa, sb = candidate.get("stage_a", {}) or {}, _stage_b(candidate)
    base_topics = _formed_topics(baseline)
    base_chapters = int(_stage_b(baseline).get("b1_chapters_or_null") or 0)
    ari_floor = round(_ari(baseline) - _ARI_BAND, 2)

    notes: list[str] = []
    if candidate.get("verdict") != "HOLDS":
        notes.append(f"verdict: {candidate.get('verdict')!r} != 'HOLDS'")

    a1 = float(sa.get("a1_recall", 0.0) or 0.0)
    if a1 < _A1_RECALL_FLOOR:
        notes.append(f"a1_recall: {a1:.3f} < {_A1_RECALL_FLOOR} floor")

    cand_topics = _formed_topics(candidate)
    if not base_topics <= cand_topics:
        notes.append(f"b1_formed_topics: missing {sorted(base_topics - cand_topics)}")
    cand_chapters = int(sb.get("b1_chapters_or_null") or 0)
    if cand_chapters < base_chapters:
        notes.append(f"b1_chapters: {cand_chapters} < baseline {base_chapters}")

    if not sb.get("b2_ok"):
        notes.append("b2_ok: false")
    cand_ari = _ari(candidate)
    if cand_ari < ari_floor:
        notes.append(f"b2_ari: {cand_ari:.3f} < {ari_floor} (baseline {_ari(baseline):.3f} - {_ARI_BAND} band)")

    noise = int(sb.get("noise_in_chapter", 0) or 0)
    if noise != 0:
        notes.append(f"b3_distractors: {noise} != 0")

    if not sb.get("b4_ok"):
        notes.append("b4_ok: false")

    return (len(notes) > 0, notes)


def write_verdict(path: Any, regression: bool, notes: list[str], *,
                  candidate_path: str = "", baseline_path: str = "") -> None:
    """Write the grep-stable artifact. The FIRST column-0 line is EXACTLY `regression: none` or
    `regression: detected` (Plan 04 greps `^regression:`), followed by the failing axes and the
    two source paths (reproducibility)."""
    lines = [f"regression: {'detected' if regression else 'none'}", ""]
    if notes:
        lines.append("failing axes:")
        lines += [f"  - {n}" for n in notes]
    else:
        lines.append("all axes within tolerance: verdict HOLDS; a1_recall >= 0.80; formed-topics "
                     "superset of baseline; b2_ok + ARI within band; b3 distractors == 0; b4 arc reachable.")
    lines += ["", f"candidate: {candidate_path}", f"baseline:  {baseline_path}"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="No-grouping-regression comparator vs the run-17 baseline (TAGG-04). Emits a "
                    "grep-stable `regression: none|detected` artifact; exit is always 0 (the grep "
                    "is the gate, not the exit code).")
    p.add_argument("--candidate", required=True,
                   help="Fresh designed-month verdict.json (produced by the re-keyed pipeline).")
    p.add_argument("--baseline", default=_DEFAULT_BASELINE,
                   help=f"Baseline verdict.json (default: {_DEFAULT_BASELINE}).")
    p.add_argument("--out", required=True, help="Artifact path (grep `^regression:` is the gate).")
    args = p.parse_args(argv)

    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    regression, notes = compare(candidate, baseline)
    write_verdict(args.out, regression, notes,
                  candidate_path=args.candidate, baseline_path=args.baseline)
    print(f"regression: {'detected' if regression else 'none'} -> {args.out}")
    return 0   # always 0 — the artifact grep is the gate (Plan 04 greps it), not the exit code.


if __name__ == "__main__":
    raise SystemExit(_main())
