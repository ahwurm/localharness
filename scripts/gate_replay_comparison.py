#!/usr/bin/env python3
"""PGATE-04 provable: replay ONE real bus-events trace through BOTH gates offline and
apply the pre-committed KILL mechanically. This is the owner-facing artifact of the
Phase-35 auto-run — "an honest KILL is a successful outcome."

The harness measures the SHIPPED write decision: it imports the exact helpers the live
`PredictiveWriteGate` runs (`should_write_stat` / `correction_in_scope` / `graded_confidence`
from predictive_write_gate.py) and reuses the 34-07 walk-forward + percentile + census
machinery (`_pass1_scores` / `_pass2_signals` / `_pct` from coll_distribution_report.py) —
it is NOT a fork of either. ZERO network / model calls; refuses to write into the shared
bench results dir; only reads the trace (`EventBus.replay`), never writes near it.

Three numbers, both gates, over the same trace:
  - junk-write rate  — writes on non-surprising events (motif novelty / new-gate correction FPs).
  - capture recall   — of the surprising_failure population, how much each gate covers.
  - capture precision — THE KILL metric: new-gate write precision worse than the motif gate
                        -> revert to motifs, keep the scores as telemetry.

Two findings encoded honestly (34-RESEARCH Pitfalls 1 & 2):
  1. The v2.0 baseline is the CORRECTED full-trace offline replay of WriteGate's own logic
     (67 fires = 53 resolved_error + 14 novelty + 0 stuck_recovered over 39.68 days), NOT
     the 22 live MemoryGateFired count (WriteGate was live only 1.89 of those days).
  2. The real trace has ZERO unsurprising_failure and ZERO StuckRecovered events, so
     PGATE-02's live suppression and the stuck_recovered tier are disclosed as
     synthetic-tested-only (35-01 unit tests), never silently declared victory.

Outputs into an ISOLATED --results dir (required):
  report.md    — owner-facing comparison + mechanical verdict (hostile-read-proof)
  verdict.json — the machine verdict: gate_comparison = holds | kill (+ the metrics)

usage:
  python scripts/gate_replay_comparison.py \
      --trace ~/.localharness/agents/orchestrator/bus-events.jsonl \
      --results <ISOLATED-DIR> \
      [--hand-labels <path>/hand_labels.json]

Exit codes: 0 = report produced (HOLDS or an honest KILL are BOTH success); 1 = processing
failure / empty trace; 2 = guard refusal (missing/contaminated --results).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

# Robust src bootstrap: run from any CWD (computed from __file__, not the working dir).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# The sibling report lives beside this file; add scripts/ so `import coll_distribution_report`
# resolves whether we are launched as a subprocess or imported as a module in a test.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Reuse the 34-07 walk-forward scorer + census + percentile helpers (measure shipped code,
# do not re-derive) — the SAME functions that produced phase34's report/verdict.
from coll_distribution_report import (  # noqa: E402
    _SPECIMEN_EVENT_ID,
    _fmt,
    _pass1_scores,
    _pass2_signals,
    _pct,
    _precision_recall,
)
from localharness.config.models import PredictiveGateConfig  # noqa: E402

# The SHIPPED write decision (35-01) — imported so the comparison measures the shipped gate,
# not a fork of its logic.
from localharness.memory.predictive_write_gate import (  # noqa: E402
    correction_in_scope,
    graded_confidence,
    should_write_stat,
)

# WriteGate's resolved_error correlation TTL (memory/gate.py) — the window inside which an
# error->success pair is one recovered incident. 2 hours, as the motif gate uses.
_MOTIF_TTL_S = 2 * 3600.0  # 7200s


# ---------------------------------------------------------------------------
# The CORRECTED motif baseline — mirror WriteGate._on_observation over the walk-forward rows.
# ---------------------------------------------------------------------------
def _motif_replay(rows: list[dict]) -> dict:
    """resolved_error (an error later followed by a same-tool success within TTL) + novelty
    (first sighting of a tool). This is the corrected FULL-trace offline replay — on the real
    orchestrator trace it yields 67 fires (53 resolved_error + 14 novelty + 0 stuck_recovered),
    the honest v2.0 baseline, NOT the 22 live MemoryGateFired count (Pitfall 1).

    stuck_recovered is 0 by construction here: StuckRecovered is a SEPARATE event type absent
    from the Action/Observation replay, and the real trace has zero of them anyway — the
    stuck_recovered tier is synthetic-tested-only (disclosed, never a silent victory)."""
    pending: dict[str, tuple[float, str]] = {}  # tool -> (ts_epoch, event_id) of most-recent error
    seen: set[str] = set()
    resolved = novelty = 0
    for r in rows:  # chronological (sorted by ts_epoch upstream)
        tool, ts = r["tool"], r["ts_epoch"]
        if r["is_error"]:
            pending[tool] = (ts, r["event_id"])
            continue
        prev = pending.get(tool)
        if prev is not None and (ts - prev[0]) < _MOTIF_TTL_S:
            resolved += 1
            del pending[tool]
        elif tool not in seen:
            seen.add(tool)
            novelty += 1
    return {"resolved": resolved, "novelty": novelty, "stuck_recovered": 0}


def _motif_captured_count(sf_events: list[dict], rows: list[dict]) -> int:
    """Coverage (not fire-count): a surprising_failure error is 'motif-captured' if the SAME
    tool has a later success within TTL — the resolution the motif records the incident at.
    A same-tool retry BURST is therefore one incident with several captured errors, so this
    coverage count is >= the resolved-fire count (on the real trace: 66 of 71 captured, 5
    never — the exact gap the stat channel closes to 71/71)."""
    succ: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if not r["is_error"]:
            succ[r["tool"]].append(r["ts_epoch"])
    captured = 0
    for e in sf_events:
        te = e["ts_epoch"]
        if any(te < s < te + _MOTIF_TTL_S for s in succ.get(e["tool"], [])):
            captured += 1
    return captured


def _iso_day(ts_iso: str) -> str:
    """YYYYMMDD from an ISO timestamp — mirrors the gate's (tool, event.timestamp.strftime
    ('%Y%m%d')) day-bucket key so a same-day retry burst collapses to one write (Pitfall 6)."""
    return ts_iso[:10].replace("-", "")


# ---------------------------------------------------------------------------
# The three metrics + THE KILL — pre-committed and mechanical (importable for direct unit test).
# ---------------------------------------------------------------------------
def _gate_metrics(
    *,
    resolved: int,
    novelty: int,
    stuck: int,
    stat_writes: int,
    correction_writes: int,
    correction_tp: int,
    sf_total: int,
    sf_covered: int,
    motif_captured: int,
) -> dict:
    """The new gate is a SIBLING — motifs STILL fire, so its writes = motif fires + stat +
    correction. Capture-precision proxy per gate = precise writes / total writes:
      motif:  resolved_error fires are precise; novelty fires are telemetry (non-precise) ->
              motif_precision = resolved / (resolved + novelty + stuck).
      new:    resolved + every stat write (surprising_failure = precise by the quadrant proxy)
              + correction true-positives, over the sibling total.
    THE KILL (verbatim, mechanical): KILL iff new_gate_precision < motif_precision."""
    motif_fires = resolved + novelty + stuck
    total_new = motif_fires + stat_writes + correction_writes
    correction_fp = correction_writes - correction_tp
    motif_precision = (resolved / motif_fires) if motif_fires else None
    new_gate_precision = (
        (resolved + stat_writes + correction_tp) / total_new if total_new else None
    )
    kill = (
        motif_precision is not None
        and new_gate_precision is not None
        and new_gate_precision < motif_precision
    )
    # SENSITIVITY (critic MAJOR 1): the proxy grades EVERY stat write precise-by-quadrant
    # (credit 1.0, no per-write ground truth). Re-grade the stat writes at the motif's OWN
    # novelty base rate (motif_precision) instead of assumed-1.0 — the un-ground-truthed
    # assumption the thin HOLDS margin rests on — and report whichever verdict falls out.
    regrade = (
        (resolved + stat_writes * motif_precision + correction_tp) / total_new
        if (total_new and motif_precision is not None)
        else None
    )
    kill_regrade = (
        regrade is not None and motif_precision is not None and regrade < motif_precision
    )
    # Absolute junk-write COUNTS (not rates): the rate can fall by denominator dilution
    # (assumed-clean stat writes padding the total) while the absolute count RISES.
    junk_count_motif = novelty
    junk_count_new = novelty + correction_fp
    return {
        "motif_precision": motif_precision,
        "new_gate_precision": new_gate_precision,
        "gate_comparison": "kill" if kill else "holds",
        "new_gate_precision_regrade": regrade,
        "gate_comparison_regrade": (
            ("kill" if kill_regrade else "holds") if regrade is not None else None
        ),
        "junk_write_rate_motif": (novelty / motif_fires) if motif_fires else None,
        "junk_write_rate_new": ((novelty + correction_fp) / total_new) if total_new else None,
        "junk_write_count_motif": junk_count_motif,
        "junk_write_count_new": junk_count_new,
        "junk_write_count_delta": junk_count_new - junk_count_motif,
        "capture_recall_motif": (motif_captured / sf_total) if sf_total else None,
        "capture_recall_new": (sf_covered / sf_total) if sf_total else None,
        "correction_precision": (correction_tp / correction_writes) if correction_writes else None,
        "motif_fires": motif_fires,
        "total_new_writes": total_new,
        "correction_fp": correction_fp,
    }


def _verdict_line(m: dict) -> str:
    """The pre-committed KILL / HOLDS line, printed verbatim (no reinterpretation)."""
    x, y = _fmt(m["new_gate_precision"]), _fmt(m["motif_precision"])
    if m["gate_comparison"] == "kill":
        return (
            f"GATE COMPARISON: KILL — new-gate capture precision {x} < motif capture "
            f"precision {y}. Per ROADMAP pre-commit: revert to motifs (set "
            f"agent.memory.predictive_gate.write_live=False), keep the scores as telemetry."
        )
    return f"GATE COMPARISON: HOLDS — new-gate precision {x} >= motif precision {y}."


def _sensitivity_verdict_line(m: dict) -> str:
    """MAJOR 1: report BOTH gradings side-by-side — the shipped quadrant proxy AND the
    critic's motif-base-rate re-grade of stat writes — plus the ABSOLUTE junk-write delta.
    No massaging: print whichever verdict each grading actually produces."""
    proxy = m["gate_comparison"].upper()
    regr = (m["gate_comparison_regrade"] or "n/a").upper()
    px, mx = _fmt(m["new_gate_precision"]), _fmt(m["motif_precision"])
    rx = _fmt(m["new_gate_precision_regrade"])
    rec_new, rec_motif = _fmt(m["capture_recall_new"]), _fmt(m["capture_recall_motif"])
    return (
        f"GATE COMPARISON (methodology-sensitive): {proxy} under the quadrant proxy "
        f"(new {px} vs motif {mx}) / {regr} under the motif-base-rate re-grade of stat "
        f"writes (new {rx} vs motif {mx}). Absolute junk writes: motif "
        f"{m['junk_write_count_motif']} -> new {m['junk_write_count_new']} "
        f"(delta +{m['junk_write_count_delta']}). Recall is proxy-INDEPENDENT: "
        f"new {rec_new} vs motif {rec_motif}."
    )


_CIRCULARITY_CLAUSE = (
    "Capture precision uses the surprising_failure QUADRANT as a precise-write proxy (a stat "
    "write is 'precise' because its quadrant means a normally-reliable tool errored), NOT "
    "per-write human ground truth — the ONLY channel measured against real hand labels is the "
    "correction channel (the phase34 census). Read the precision as a mechanical KILL trip "
    "wire, not a claim that every stat write is individually correct."
)


# ---------------------------------------------------------------------------
# Report assembly (hostile-read-proof; plain language first, numbers follow).
# ---------------------------------------------------------------------------
def _build_report(*, corpus, motif, m, stat, corr, specimen_note, verdict_line, census) -> str:
    L: list[str] = []
    add = L.append
    add("# PGATE-04 Gate-Replay Comparison — motif baseline vs the new sibling gate\n")
    add(
        "Replays ONE real bus-events trace through BOTH gates offline (zero network / model "
        "calls) and applies the pre-committed capture-precision KILL mechanically. The new "
        "gate is a SIBLING: the motif floor still fires, and the stat + correction channels "
        "are ADDED on top. An honest KILL is a successful outcome.\n"
    )

    add("## Corpus\n")
    add(f"- Trace: `{corpus['trace']}`")
    add(f"- Date span: {corpus['first_ts']} → {corpus['last_ts']}")
    add(f"- Scored tool observations: **{len(corpus['rows'])}**")
    add(f"- UserMessages (full census): **{corpus['n_signals']}**")
    add(f"- surprising_failure events (the stat channel's target population): **{stat['sf_total']}**\n")

    add("## Motif baseline (corrected offline replay — Pitfall 1)\n")
    add(
        f"- This trace: resolved_error=**{motif['resolved']}**, novelty=**{motif['novelty']}**, "
        f"stuck_recovered=**{motif['stuck_recovered']}** → **{m['motif_fires']} motif fires**."
    )
    add(
        "- On the REAL orchestrator trace this is the corrected **67 = 53 + 14 + 0** baseline "
        "over 39.68 days — NOT the 22 live `MemoryGateFired` count (WriteGate was live only "
        "1.89 of those days; using 22 would compare 1.89 days of coverage against a full-trace "
        "replay — not apples to apples)."
    )
    add(
        "- stuck_recovered is **0 by construction**: StuckRecovered is a separate event type "
        "absent from the Action/Observation replay, and the real trace has zero of them — the "
        "stuck_recovered tier is **synthetic-tested-only** (35-01), never a silent victory.\n"
    )

    add("## Stat channel (new — surprising_failure)\n")
    add(
        f"- Writes on the `surprising_failure` quadrant only (`should_write_stat`, the shipped "
        f"helper). Same-day retry bursts collapse to one `(tool, day)` bucket (Pitfall 6): "
        f"**{stat['stat_writes']}** distinct stat writes covering **{stat['sf_covered']}** "
        f"surprising_failure events."
    )
    add(
        f"- Graded confidence over these events (shipped `graded_confidence`): "
        f"min={_fmt(stat['conf_min'])} median={_fmt(stat['conf_med'])} max={_fmt(stat['conf_max'])} "
        f"(all strictly < 0.7 — never enters the injected ambient block until Phase 36 promotes)."
    )
    add(
        f"- Coverage gap CLOSED: the motif floor captures **{stat['motif_captured']}/"
        f"{stat['sf_total']}** surprising_failure events (an error is motif-captured only if the "
        f"tool later recovers within TTL); the stat channel covers **{stat['sf_covered']}/"
        f"{stat['sf_total']}**. On the real trace this is the 66/71 → 71/71 gap (5 never-captured "
        f"errors the pain-only motif gate is blind to).\n"
    )

    add("## Correction channel (new — negation / correction_phrase only)\n")
    add(
        f"- Scoped to `negation`/`correction_phrase` families (`correction_in_scope`): reask "
        f"(0/18 on the census) and frustration (no census data) are EXCLUDED — this raises "
        f"write-relevant precision from 0.115 (all families) to ~0.375. Scoped writes: "
        f"**{corr['correction_writes']}**."
    )
    if census is not None:
        add(
            f"- Census cross-check (`_precision_recall` over phase34 hand labels): scoped "
            f"correction true-positives = **{corr['correction_tp']}**, scoped precision = "
            f"**{_fmt(m['correction_precision'])}**; all-families correction precision (for "
            f"contrast) = {_fmt(census['all_precision'])} / recall {_fmt(census['all_recall'])}."
        )
    else:
        add(
            "- No --hand-labels census supplied: scoped correction true-positives are "
            "**unmeasured**, so the KILL numerator counts them as **0 (conservative floor — "
            "this can only push the verdict toward KILL, never falsely toward HOLDS)**."
        )
    add(
        f"- Fireworks specimen ({_SPECIMEN_EVENT_ID}): {specimen_note}\n"
    )

    add("## The three metrics (both gates, same trace)\n")
    add("| metric | motif gate | new sibling gate |")
    add("|--------|-----------|------------------|")
    add(
        f"| capture precision | {_fmt(m['motif_precision'])} | {_fmt(m['new_gate_precision'])} |"
    )
    add(
        f"| junk-write rate | {_fmt(m['junk_write_rate_motif'])} | {_fmt(m['junk_write_rate_new'])} |"
    )
    add(
        f"| capture recall | {_fmt(m['capture_recall_motif'])} | {_fmt(m['capture_recall_new'])} |"
    )
    add(f"| total writes | {m['motif_fires']} | {m['total_new_writes']} |")
    add("")
    add(f"**{verdict_line}**\n")

    add("## Verdict sensitivity — the precision comparison is methodology-sensitive\n")
    add(
        "The capture-precision KILL is NOT robust (critic MAJOR 1): with no per-write ground "
        "truth for stat writes, the verdict depends entirely on how they are graded. Both "
        "gradings, side by side:"
    )
    add("")
    add("| grading of stat writes | new-gate precision | motif precision | verdict |")
    add("|------------------------|--------------------|-----------------|---------|")
    add(
        f"| shipped quadrant proxy (credit 1.0) | {_fmt(m['new_gate_precision'])} | "
        f"{_fmt(m['motif_precision'])} | {m['gate_comparison'].upper()} |"
    )
    add(
        f"| motif's OWN novelty base rate ({_fmt(m['motif_precision'])}) | "
        f"{_fmt(m['new_gate_precision_regrade'])} | {_fmt(m['motif_precision'])} | "
        f"{(m['gate_comparison_regrade'] or 'n/a').upper()} |"
    )
    add("")
    add(
        f"- **Absolute junk writes (not rates):** motif **{m['junk_write_count_motif']}** "
        f"(novelty) -> new sibling **{m['junk_write_count_new']}** (novelty + correction "
        f"false-positives) = **+{m['junk_write_count_delta']}**. The junk-write *rate* falls "
        f"({_fmt(m['junk_write_rate_motif'])} -> {_fmt(m['junk_write_rate_new'])}) ONLY by "
        "denominator dilution — the assumed-clean stat writes pad the total; the absolute "
        "count ROSE."
    )
    add(
        f"- **Capture recall is proxy-INDEPENDENT** (no grading assumption): the stat channel "
        f"covers **{stat['sf_covered']}/{stat['sf_total']}** surprising_failure events vs the "
        f"motif floor's **{stat['motif_captured']}/{stat['sf_total']}**. That coverage gain "
        "holds under every grading — it, not the thin/circular precision margin, is the "
        "defensible win."
    )
    add("")
    add(f"**{_sensitivity_verdict_line(m)}**\n")

    add("## Honesty footer\n")
    add(f"- Circularity clause: {_CIRCULARITY_CLAUSE}")
    add(
        f"- Zero-occurrence disclosure: unsurprising_failure events on this trace = "
        f"**{corpus['n_unsurprising_failure']}** (on the real trace: 0 — PGATE-02 suppression "
        f"is proven ONLY by the synthetic unit test in 35-01, not this replay); StuckRecovered "
        f"events = **0** (the stuck_recovered tier is synthetic-tested-only)."
    )
    add(
        "- Pitfall 1: the 22 live `MemoryGateFired` fires are NOT the baseline; the "
        f"{m['motif_fires']}-fire ({motif['resolved']}+{motif['novelty']}+{motif['stuck_recovered']}) "
        "corrected offline replay is."
    )
    add(
        "- Precision is a PROXY (quadrant, not per-write ground truth) except the correction "
        "channel (real census). junk-write rate and capture recall are read alongside it — a "
        "gate that captures more but writes junk is not a free win."
    )
    add("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
async def run(args: argparse.Namespace, results: Path) -> int:
    trace = Path(args.trace).expanduser()
    if not trace.exists():
        print(f"PROCESSING FAILURE: trace not found: {trace}", file=sys.stderr)
        return 1
    cfg = PredictiveGateConfig()  # defaults — the same config the live gate constructs

    try:
        p1 = await _pass1_scores(trace, cfg, results / "scores.jsonl")
        signals = await _pass2_signals(trace, cfg, results / "signals.jsonl")
    except Exception as exc:  # noqa: BLE001 — any replay/score failure is an honest exit 1
        print(f"PROCESSING FAILURE: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    rows = sorted(p1["rows"], key=lambda r: r["ts_epoch"])
    if not rows:
        print("PROCESSING FAILURE: no tool observations scored (empty or wrong trace?)", file=sys.stderr)
        return 1

    # (1) MOTIF baseline (the corrected offline replay).
    motif = _motif_replay(rows)

    # (2) STAT channel — surprising_failure events, day-bucket collapse; coverage vs motif.
    sf_events = [r for r in rows if should_write_stat(r["quadrant"])]
    stat_writes = len({(e["tool"], _iso_day(e["ts"])) for e in sf_events})
    sf_covered = len({e["event_id"] for e in sf_events})  # stat covers every surprising_failure
    motif_captured = _motif_captured_count(sf_events, rows)
    sf_confs = sorted(graded_confidence(e["score"]) for e in sf_events)
    n_unsurprising_failure = sum(1 for r in rows if r["quadrant"] == "unsurprising_failure")

    # (3) CORRECTION channel — scoped negation / correction_phrase (reask/frustration excluded).
    scoped = [
        s for s in signals
        if s["detected"] and correction_in_scope(s["detected"]["trigger_family"])
    ]
    correction_writes = len(scoped)
    census = None
    if args.hand_labels:
        labels = json.loads(Path(args.hand_labels).expanduser().read_text(encoding="utf-8"))
        correction_tp = sum(
            1 for s in scoped if labels.get(s["event_id"], {}).get("label") == "correction"
        )
        pr = _precision_recall(signals, labels)["per_class"]["correction"]
        census = {"all_precision": pr["precision"], "all_recall": pr["recall"]}
    else:
        # Unmeasured without a census -> conservative floor (only pushes toward KILL).
        correction_tp = 0

    # (4) The three metrics + THE mechanical kill.
    m = _gate_metrics(
        resolved=motif["resolved"],
        novelty=motif["novelty"],
        stuck=motif["stuck_recovered"],
        stat_writes=stat_writes,
        correction_writes=correction_writes,
        correction_tp=correction_tp,
        sf_total=len(sf_events),
        sf_covered=sf_covered,
        motif_captured=motif_captured,
    )
    m["correction_tp"] = correction_tp
    verdict_line = _verdict_line(m)

    # Fireworks — should be an in-scope negation correction write (broken out separately).
    specimen = next((s for s in signals if s["event_id"] == _SPECIMEN_EVENT_ID), None)
    if specimen is None:
        specimen_note = (
            "NOT PRESENT in this trace (synthetic / other trace — the real orchestrator trace "
            "has it at line 4625)."
        )
    elif specimen["detected"] and correction_in_scope(specimen["detected"]["trigger_family"]):
        specimen_note = (
            f"in-scope correction write (family={specimen['detected']['trigger_family']}, "
            f"matched='{specimen['detected']['matched_text']}') — the asymmetry the pain-only "
            "motif gate was blind to becomes a stored, reversible, provenance-stamped write."
        )
    else:
        specimen_note = (
            "present but NOT an in-scope correction (detector missed it or it fell to an "
            "excluded family) — disclosed, not smoothed."
        )

    stat = {
        "stat_writes": stat_writes,
        "sf_total": len(sf_events),
        "sf_covered": sf_covered,
        "motif_captured": motif_captured,
        "conf_min": sf_confs[0] if sf_confs else None,
        "conf_med": _pct(sf_confs, 50) if sf_confs else None,
        "conf_max": sf_confs[-1] if sf_confs else None,
    }
    corr = {"correction_writes": correction_writes, "correction_tp": correction_tp}
    corpus = {
        "trace": str(trace),
        "rows": rows,
        "n_signals": len(signals),
        "first_ts": min(r["ts"] for r in rows),
        "last_ts": max(r["ts"] for r in rows),
        "n_unsurprising_failure": n_unsurprising_failure,
    }

    # verdict.json — the machine verdict (keys the plan pins + the raw fire breakdown).
    verdict = {
        "motif_precision": m["motif_precision"],
        "new_gate_precision": m["new_gate_precision"],
        "gate_comparison": m["gate_comparison"],
        "new_gate_precision_regrade": m["new_gate_precision_regrade"],
        "gate_comparison_regrade": m["gate_comparison_regrade"],
        "junk_write_rate_motif": m["junk_write_rate_motif"],
        "junk_write_rate_new": m["junk_write_rate_new"],
        "junk_write_count_motif": m["junk_write_count_motif"],
        "junk_write_count_new": m["junk_write_count_new"],
        "junk_write_count_delta": m["junk_write_count_delta"],
        "capture_recall_motif": m["capture_recall_motif"],
        "capture_recall_new": m["capture_recall_new"],
        "stat_writes": stat_writes,
        "correction_writes": correction_writes,
        "correction_precision": m["correction_precision"],
        "n_unsurprising_failure": n_unsurprising_failure,
        "motif_fires": m["motif_fires"],
        "resolved_error": motif["resolved"],
        "novelty": motif["novelty"],
        "stuck_recovered": motif["stuck_recovered"],
        "surprising_failure_total": len(sf_events),
        "motif_captured": motif_captured,
        "correction_tp": correction_tp,
        "hand_labels_used": bool(args.hand_labels),
        "reason": verdict_line,
        "reason_sensitivity": _sensitivity_verdict_line(m),
    }
    (results / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")

    report = _build_report(
        corpus=corpus, motif=motif, m=m, stat=stat, corr=corr,
        specimen_note=specimen_note, verdict_line=verdict_line, census=census,
    )
    (results / "report.md").write_text(report, encoding="utf-8")

    # Stdout summary (owner-facing).
    print(
        f"scored {len(rows)} observations | {len(signals)} user messages | "
        f"surprising_failure {len(sf_events)} | motif fires {m['motif_fires']} "
        f"({motif['resolved']} resolved + {motif['novelty']} novelty + {motif['stuck_recovered']} stuck)"
    )
    print(verdict_line)
    print(_sensitivity_verdict_line(m))
    print(
        f"junk-write rate: motif {_fmt(m['junk_write_rate_motif'])} -> new {_fmt(m['junk_write_rate_new'])} "
        f"| absolute junk writes: motif {m['junk_write_count_motif']} -> new {m['junk_write_count_new']} "
        f"| capture recall: motif {_fmt(m['capture_recall_motif'])} -> new {_fmt(m['capture_recall_new'])}"
    )
    print(f"report: {results / 'report.md'}")
    # HOLDS or an honest KILL are BOTH a successful measurement -> exit 0.
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PGATE-04 gate-replay comparison: motif baseline vs new sibling gate + mechanical kill"
    )
    p.add_argument(
        "--trace",
        default=str(Path("~/.localharness/agents/orchestrator/bus-events.jsonl")),
        help="bus-events JSONL to replay (READ-ONLY; default = the real orchestrator trace)",
    )
    p.add_argument(
        "--results",
        required=True,
        help="ISOLATED output dir (REQUIRED). Refuses any path inside bench/results.",
    )
    p.add_argument(
        "--hand-labels",
        default=None,
        help="Optional phase34 hand_labels.json census for the real correction precision.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    results = Path(args.results).expanduser().resolve()
    # GUARD (before any output is written): never contaminate the shared bench results dir.
    if "bench/results" in str(results):
        print(
            "REFUSED: --results points inside bench/results. The shared bench results dir is "
            "contaminated (project hard rule) — use an isolated dir "
            "(e.g. ~/.localharness/gate-reports/...).",
            file=sys.stderr,
        )
        sys.exit(2)
    results.mkdir(parents=True, exist_ok=True)
    sys.exit(asyncio.run(run(args, results)))


if __name__ == "__main__":
    main()
