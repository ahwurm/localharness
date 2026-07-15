#!/usr/bin/env python3
"""COLL-05 provable: replay a REAL bus-events trace, score every historical tool
outcome with the SHIPPED surprise code (walk-forward priors, no lookahead), run the
SHIPPED trigger lexicon over every historical user message, and apply the pre-committed
kill criterion MECHANICALLY.

This script measures the shipped code — it imports the exact functions the live
predictive gate runs (`compute_surprise_score` / `classify_user_signal`) and replays the
production JSONL through the exact same walk-forward discipline `PredictiveGate` uses
(`_on_action` snapshots the prior; `_on_observation` scores THEN folds the row in, so a
row never contaminates its own prior). It does ZERO network/LLM calls and refuses to
write into the shared bench results dir (contamination rule).

Outputs into an ISOLATED --results dir (required):
  report.md     — owner-facing distribution report + mechanical verdict (hostile-read-proof)
  scores.jsonl  — one line per scored tool observation (walk-forward)
  signals.jsonl — one line per UserMessage (detected or not — the census needs negatives)
  verdict.json  — the machine verdict: statistical_channel = separates | kill

usage:
  python scripts/coll_distribution_report.py \
      --trace ~/.localharness/agents/orchestrator/bus-events.jsonl \
      --results <ISOLATED-DIR> \
      [--hand-labels <path>/hand_labels.json]

Exit codes: 0 = report produced AND the specimen specimen (if present) detected as a
correction — an honest KILL is a successful measurement, so it also exits 0; 1 = specimen
miss or processing failure; 2 = guard refusal (missing/contaminated --results).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Robust src bootstrap: run from any CWD (mirrors scripts/dogfood_memory_spine.py intent
# but computed from __file__ so it does not depend on the working directory).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from localharness.config.models import PredictiveGateConfig  # noqa: E402
from localharness.core.bus import EventBus  # noqa: E402
from localharness.core.events import (  # noqa: E402
    Action,
    Observation,
    TurnCompleted,
    UserMessage,
)
from localharness.memory.sqlite import (  # noqa: E402
    ToolPrior,
    _band_z,
    _tool_error_surprisal,
    compute_quadrant,
    compute_surprise_score,
)
from localharness.memory.user_signals import classify_user_signal, is_reask  # noqa: E402

# The motivating specimen (34-CONTEXT): the July-4th specimen correction the pain-only
# motif gate is structurally blind to. Pinned by event id — if the shipped lexicon misses
# it, that is a phase failure (exit 1), not a footnote.
_SPECIMEN_EVENT_ID = "52f90afe-5d68-401b-98ab-aa32d2410b88"

# Mechanical kill thresholds (decided in 34-07-PLAN objective; printed verbatim below).
_SURPRISING_MAX_PRIOR_ERR = 0.2   # "reliable tool" prior-error-rate ceiling
_SURPRISING_MIN_PRIOR_N = 5       # min prior history for the known-surprising set
_ROUTINE_MIN_PRIOR_N = 20         # "well-known tool" prior-history floor
_ROUTINE_MAX_PRIOR_ERR = 0.1      # routine prior-error-rate ceiling
_MIN_SET_N = 5                    # below this, verdict is an honest insufficient-history kill

_CIRCULARITY_CLAUSE = (
    "Set membership uses the error-rate prior while separation is measured on the full "
    "composite score — partial circularity is unavoidable without hand labels for tool "
    "events. What the check genuinely tests: whether the real trace has enough per-tool "
    "history for priors to be confident (vs everything cold-start neutral), and whether "
    "the latency/size z-noise swamps the error signal. The named specimens are the "
    "non-circular anchors."
)


# ---------------------------------------------------------------------------
# Walk-forward per-tool aggregates -> ToolPrior (mirrors sqlite.get_tool_prior's
# population-variance math: E[x^2]-E[x]^2 with a tiny-negative clamp, cold-start -> None).
# 34-01 pinned get_tool_prior byte-for-byte against a statistics.pvariance reference on
# identical rows; this rebuilds the SAME prior from in-memory running sums, so the
# equivalence is inherited, not re-proven here.
# ---------------------------------------------------------------------------
def _new_agg() -> dict:
    return {"n": 0, "err": 0, "dn": 0, "ds": 0.0, "dss": 0.0, "ln": 0, "ls": 0.0, "lss": 0.0}


def _prior_from_agg(tool: str, a: dict) -> ToolPrior:
    n = a["n"]
    error_rate = (a["err"] / n) if n else None
    if a["dn"]:
        lm = a["ds"] / a["dn"]
        lv = max(0.0, a["dss"] / a["dn"] - lm * lm)
    else:
        lm = lv = None
    if a["ln"]:
        sm = a["ls"] / a["ln"]
        sv = max(0.0, a["lss"] / a["ln"] - sm * sm)
    else:
        sm = sv = None
    return ToolPrior(tool, n, error_rate, lm, lv, a["dn"], sm, sv, a["ln"])


def _fold(a: dict, is_error: int, duration_ms: int, output_len: int) -> None:
    a["n"] += 1
    a["err"] += is_error
    a["dn"] += 1
    a["ds"] += duration_ms
    a["dss"] += duration_ms * duration_ms
    a["ln"] += 1
    a["ls"] += output_len
    a["lss"] += output_len * output_len


def _pct(sorted_vals: list[float], q: float) -> float | None:
    """Linear-interpolation percentile (numpy 'linear' default). q in [0, 100]."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def _fmt(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# PASS 1 — statistical channel (walk-forward, no lookahead). Mirrors PredictiveGate:
# snapshot the prior at the Action (tool_call), score at the matched Observation using
# that snapshot, THEN fold the observation into the aggregates.
# ---------------------------------------------------------------------------
async def _pass1_scores(trace: Path, cfg: PredictiveGateConfig, scores_path: Path) -> dict:
    agg: dict[str, dict] = defaultdict(_new_agg)
    pending: dict[str, tuple[ToolPrior, object]] = {}  # tool_call_id -> (prior, action_ts)
    rows: list[dict] = []
    uv_specimens: list[dict] = []  # error text has both "uv" and "not found" (live-test-night class)
    unmatched = 0
    empty_err = 0  # is_error rows whose error field is the empty string "" (shipped rule still counts them)

    bus = EventBus(persist_path=trace)  # READ-ONLY: only replay() is called, never publish()
    with scores_path.open("w", encoding="utf-8") as fh:
        async for ev in bus.replay(event_types=[Action, Observation]):
            if isinstance(ev, Action):
                if ev.action_type != "tool_call" or not ev.tool_call_id or not ev.tool_name:
                    continue
                # skip-under-load cap, mirroring the live gate (never triggers on a
                # sequential single-agent trace, kept for faithfulness).
                while len(pending) >= cfg.pending_cap:
                    pending.pop(next(iter(pending)))
                pending[ev.tool_call_id] = (_prior_from_agg(ev.tool_name, agg[ev.tool_name]), ev.timestamp)
                continue

            # Observation
            if ev.observation_type != "tool_result" or not ev.tool_call_id:
                continue
            match = pending.pop(ev.tool_call_id, None)
            if match is None:  # pre-subscribe / foreign / cap-evicted
                unmatched += 1
                continue
            prior, action_ts = match
            duration_ms = max(0, int((ev.timestamp - action_ts).total_seconds() * 1000))
            is_error = 1 if ev.error is not None else 0  # shipped rule (empty-string error counts)
            if is_error and ev.error == "":
                empty_err += 1
            output_len = len(ev.output or "")  # capped at 200 upstream
            min_n = cfg.min_prior_n
            err_s = _tool_error_surprisal(is_error, prior.error_rate, prior.n, min_n)
            z_lat = _band_z(duration_ms, prior.lat_mean_ms, prior.lat_var_ms, prior.lat_n, min_n)
            z_size = _band_z(output_len, prior.size_mean, prior.size_var, prior.size_n, min_n)
            # THE shipped composite — this is the number the live gate persists.
            score = compute_surprise_score(
                is_error, output_len, duration_ms, prior,
                min_n=min_n, latency_weight=cfg.latency_weight, size_weight=cfg.size_weight,
            )
            quadrant = compute_quadrant(is_error, prior.error_rate, prior.n, min_n)
            row = {
                "tool": ev.tool_name,
                "ts": ev.timestamp.isoformat(),
                "ts_epoch": ev.timestamp.timestamp(),
                "session_id": ev.session_id,
                "is_error": is_error,
                "output_len": output_len,
                "duration_ms": duration_ms,
                "score": score,
                "quadrant": quadrant,
                "error_surprisal": err_s,
                "z_latency": z_lat,
                "z_size": z_size,
                "prior_n": prior.n,
                "prior_error_rate": prior.error_rate,
                "event_id": ev.id,
            }
            fh.write(json.dumps(row) + "\n")
            rows.append(row)
            if is_error and ev.error is not None:
                el = ev.error.lower()
                if "uv" in el and "not found" in el:
                    uv_specimens.append({"event_id": ev.id, "tool": ev.tool_name, "score": score})
            # walk-forward: fold AFTER scoring so this row never contaminates its own prior.
            _fold(agg[ev.tool_name], is_error, duration_ms, output_len)

    return {"rows": rows, "unmatched": unmatched, "uv_specimens": uv_specimens, "empty_err": empty_err}


# ---------------------------------------------------------------------------
# PASS 2 — user-signal channel. Replays the UserSignalDetector logic exactly (same
# lexicon, same reask window, same precedence). EVERY UserMessage gets a signals.jsonl
# line so the hand-labeled census has its negatives.
# ---------------------------------------------------------------------------
async def _pass2_signals(trace: Path, cfg: PredictiveGateConfig, signals_path: Path) -> list[dict]:
    lexicon = cfg.lexicon.model_dump()
    session_messages: dict[str, list[str]] = {}
    records: list[dict] = []
    bus = EventBus(persist_path=trace)
    with signals_path.open("w", encoding="utf-8") as fh:
        async for ev in bus.replay(event_types=[UserMessage, TurnCompleted]):
            if isinstance(ev, TurnCompleted):
                continue  # detector uses this only for the corrected-turn pointer (not persisted here)
            text = (ev.content or "").strip()
            detected = None
            if text:
                match = classify_user_signal(text, lexicon)
                sid = ev.session_id or ""
                window = session_messages.setdefault(sid, [])
                if match is None and is_reask(text, window, cfg.reask_threshold):
                    match = ("correction", "reask", "")
                window.append(text)  # append AFTER the check (a message never matches itself)
                if len(window) > cfg.reask_window:
                    del window[: len(window) - cfg.reask_window]
                if len(session_messages) > 8:
                    session_messages.pop(next(iter(session_messages)))
                if match is not None:
                    detected = {"signal_type": match[0], "trigger_family": match[1], "matched_text": match[2]}
            rec = {
                "event_id": ev.id,
                "ts": ev.timestamp.isoformat(),
                "ts_epoch": ev.timestamp.timestamp(),
                "session_id": ev.session_id or "",
                "content": text,
                "detected": detected,
            }
            fh.write(json.dumps(rec) + "\n")
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def _kill_check(rows: list[dict]) -> dict:
    surprising = [r["score"] for r in rows
                  if r["is_error"] == 1 and r["prior_error_rate"] is not None
                  and r["prior_error_rate"] < _SURPRISING_MAX_PRIOR_ERR
                  and r["prior_n"] >= _SURPRISING_MIN_PRIOR_N]
    routine = [r["score"] for r in rows
               if r["is_error"] == 0 and r["prior_error_rate"] is not None
               and r["prior_error_rate"] <= _ROUTINE_MAX_PRIOR_ERR
               and r["prior_n"] >= _ROUTINE_MIN_PRIOR_N]
    n_s, n_r = len(surprising), len(routine)
    if n_s < _MIN_SET_N or n_r < _MIN_SET_N:
        return {
            "statistical_channel": "kill",
            "median_surprising": _pct(sorted(surprising), 50) if surprising else None,
            "p90_routine": _pct(sorted(routine), 90) if routine else None,
            "n_surprising": n_s, "n_routine": n_r,
            "reason": f"insufficient-history: n_surprising={n_s}, n_routine={n_r} "
                      f"(both need >= {_MIN_SET_N}; an unfalsifiable SEPARATES is worse than an honest kill)",
        }
    median_s = _pct(sorted(surprising), 50)
    p90_r = _pct(sorted(routine), 90)
    separates = median_s > p90_r
    return {
        "statistical_channel": "separates" if separates else "kill",
        "median_surprising": median_s, "p90_routine": p90_r,
        "n_surprising": n_s, "n_routine": n_r,
        "reason": (f"median(known-surprising)={median_s:.3f} > P90(routine)={p90_r:.3f}"
                   if separates else
                   f"median(known-surprising)={median_s:.3f} <= P90(routine)={p90_r:.3f} — "
                   "the latency/size z-noise swamps the error signal (or priors too cold to be confident)"),
    }


def _pct_rank(all_sorted: list[float], s: float) -> float:
    """Percentile rank of s within all scores: 100 * (fraction strictly less than s)."""
    if not all_sorted:
        return 0.0
    lo, hi = 0, len(all_sorted)
    while lo < hi:  # bisect_left
        mid = (lo + hi) // 2
        if all_sorted[mid] < s:
            lo = mid + 1
        else:
            hi = mid
    return 100.0 * lo / len(all_sorted)


_SIGNAL_CLASSES = ("correction", "confirmation", "interruption")
_FAMILY_TO_CLASS = {
    "negation": "correction", "correction_phrase": "correction", "frustration": "correction",
    "reask": "correction", "confirmation": "confirmation", "interruption": "interruption",
}


def _precision_recall(signals: list[dict], labels: dict[str, dict]) -> dict:
    """Detector output vs the hand-labeled census. Precision is CALIBRATION data for the
    Phase-35/36 look (recall-first detector), NOT a target."""
    common = [s for s in signals if s["event_id"] in labels]
    per_class = {}
    for cls in _SIGNAL_CLASSES:
        det = {s["event_id"] for s in common if s["detected"] and s["detected"]["signal_type"] == cls}
        act = {eid for eid in (s["event_id"] for s in common) if labels[eid]["label"] == cls}
        tp = len(det & act)
        fp = len(det - act)
        fn = len(act - det)
        per_class[cls] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": (tp / (tp + fp)) if (tp + fp) else None,
            "recall": (tp / (tp + fn)) if (tp + fn) else None,
        }
    # 4x4 confusion (rows = detected class incl. "none"; cols = actual hand label).
    axes = ("correction", "confirmation", "interruption", "none")
    confusion = {d: {a: 0 for a in axes} for d in axes}
    for s in common:
        d = s["detected"]["signal_type"] if s["detected"] else "none"
        a = labels[s["event_id"]]["label"]
        confusion[d][a] += 1
    # per-trigger-family precision (finer calibration granularity).
    fam_fire = Counter()
    fam_hit = Counter()
    for s in common:
        if not s["detected"]:
            continue
        fam = s["detected"]["trigger_family"]
        fam_fire[fam] += 1
        if labels[s["event_id"]]["label"] == _FAMILY_TO_CLASS.get(fam):
            fam_hit[fam] += 1
    return {"per_class": per_class, "confusion": confusion, "n_common": len(common),
            "fam_fire": dict(fam_fire), "fam_hit": dict(fam_hit)}


def _calibration_crosscheck(signals: list[dict], rows: list[dict], global_median: float | None) -> dict:
    """Free labels: for each detected correction, the max statistical score among the same
    sitting's tool observations in the 10 minutes BEFORE the message. A correction that
    arrived while the tool channel saw nothing surprising (max < global median) is a
    'miscalibrated-prior candidate' — collect-only: reported, never acted on."""
    by_session: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_session[r["session_id"] or ""].append((r["ts_epoch"], r["score"]))
    for lst in by_session.values():
        lst.sort()
    candidates = 0
    no_activity = 0
    checked = 0
    for s in signals:
        if not (s["detected"] and s["detected"]["signal_type"] == "correction"):
            continue
        checked += 1
        t = s["ts_epoch"]
        window = [sc for (ts, sc) in by_session.get(s["session_id"], []) if t - 600 <= ts <= t]
        if not window:
            no_activity += 1
            continue
        if global_median is not None and max(window) < global_median:
            candidates += 1
    return {"checked": checked, "candidates": candidates, "no_activity": no_activity}


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def _build_report(rows, signals, kill, spec, calib, pr, corpus) -> str:
    L: list[str] = []
    add = L.append
    add("# COLL-05 Distribution Report — collect-only predictive gate (Phase 34)\n")
    add("Replays the real production trace through the SHIPPED surprise scorer + trigger "
        "lexicon (walk-forward, zero network/LLM calls) and applies the pre-committed kill "
        "criterion mechanically. Plain language first; the numbers follow.\n")

    # (a) Corpus header
    add("## Corpus\n")
    add(f"- Trace: `{corpus['trace']}`")
    add(f"- Date span: {corpus['first_ts']} → {corpus['last_ts']}")
    add(f"- Scored tool observations: **{len(rows)}** (unmatched Action/Observation pairs: "
        f"{corpus['unmatched']})")
    add(f"- UserMessages (full census): **{len(signals)}**")
    add(f"- Tools seen: {', '.join(corpus['tools'])}")
    add("- CAVEAT — session_id != sitting: pre-Phase-33 ids are minted per turn, so "
        "session_id counts are NOT sitting counts; reask windows key on session_id as-is "
        "(per-turn ids make historical reask detection conservative — a real re-ask across "
        "turns is split into separate windows and under-counted).")
    add("- CAVEAT — output cap: tool output is capped at 200 chars upstream, so output-size "
        "is effectively a saturated 3-bucket signal; its z-score is near-zero for most rows "
        "and contributes little to the composite.")
    add(f"- CAVEAT — error semantics: is_error uses the shipped rule `error is not None`. "
        f"{corpus['empty_err']} of {corpus['n_err']} errors carry an empty-string error "
        "field (error==\"\"); the live gate counts those as errors too, so this report "
        "matches it (not a bug — a faithful mirror).\n")

    # (b) Score distribution
    add("## Score distribution\n")
    alls = corpus["all_sorted"]
    add(f"- Overall: P50={_fmt(_pct(alls,50))}  P75={_fmt(_pct(alls,75))}  "
        f"P90={_fmt(_pct(alls,90))}  P99={_fmt(_pct(alls,99))}  max={_fmt(_pct(alls,100))}")
    add(f"- Cold-start fraction (prior_n < min_n, score forced 0.0): "
        f"{corpus['cold_frac']:.1%} ({corpus['cold_n']}/{len(rows)})")
    add(f"- Quadrant counts: {corpus['quadrants']}\n")
    add("| tool | n | error_rate | score P50 | score P90 |")
    add("|------|---|-----------|-----------|-----------|")
    for t, st in corpus["per_tool"]:
        add(f"| {t} | {st['n']} | {st['err']:.3f} | {_fmt(st['p50'])} | {_fmt(st['p90'])} |")
    add("")

    # (c) THE KILL CHECK
    add("## The kill check (mechanical)\n")
    add(f"- known-surprising set = error observations whose walk-forward prior said the tool "
        f"was reliable (prior_error_rate < {_SURPRISING_MAX_PRIOR_ERR}, prior_n >= "
        f"{_SURPRISING_MIN_PRIOR_N}) — the 'reliable tool failed' class. n = {kill['n_surprising']}.")
    add(f"- routine set = success observations of well-known tools (prior_n >= "
        f"{_ROUTINE_MIN_PRIOR_N}, prior_error_rate <= {_ROUTINE_MAX_PRIOR_ERR}). "
        f"n = {kill['n_routine']}.")
    add(f"- SEPARATION HOLDS iff median(score | known-surprising) > P90(score | routine).")
    add(f"- median(known-surprising) = {_fmt(kill['median_surprising'])} vs "
        f"P90(routine) = {_fmt(kill['p90_routine'])}")
    add(f"- reason: {kill['reason']}\n")
    if kill["statistical_channel"] == "separates":
        add("**STATISTICAL CHANNEL: SEPARATES**\n")
    else:
        add("**STATISTICAL CHANNEL: KILL — per ROADMAP pre-commit: drop the statistical "
            "channel in Phase 35, keep user-signal only, scores remain telemetry.**\n")

    # (d) Named specimens (non-circular anchors)
    add("## Named specimens (the non-circular anchors)\n")
    add("Percentile rank = 100 * (fraction of ALL scores strictly below the specimen).\n")
    add(f"- web_fetch error observations: {len(spec['web_fetch'])} found "
        f"(the 403 class 34-RESEARCH verified). Percentile ranks: "
        f"min={_fmt(spec['wf_min'],1)}  median={_fmt(spec['wf_med'],1)}  max={_fmt(spec['wf_max'],1)}")
    add(f"- uv+not-found error observations (live-test-night class): {len(spec['uv'])} found."
        + ("" if spec["uv"] else " This orchestrator trace contains NO uv-not-found error "
           "(the errors present are 403 / permission-denied / file-not-found classes) — "
           "reported honestly, not smoothed."))
    if spec["anomalies"]:
        add(f"- ANOMALY: under a {kill['statistical_channel'].upper()} verdict, these named "
            "specimens scored below the global median (called out, not smoothed):")
        for a in spec["anomalies"]:
            add(f"    - {a['kind']} {a['event_id']}: score={_fmt(a['score'])}, pct_rank={_fmt(a['pct'],1)}")
    else:
        add("- No named specimen fell below the global median under this verdict.")
    add("")

    # (e) Signal census
    add("## Signal census (detector vs reality)\n")
    add(f"- Detector fires by signal type: {corpus['sig_types']}")
    add(f"- Detector fires by trigger family: {corpus['sig_families']}")
    add(f"- Fireworks specimen ({_SPECIMEN_EVENT_ID}): {corpus['specimen_note']}\n")

    # (f) Precision / recall vs census
    if pr is not None:
        add("## Correction-detection precision/recall vs the hand-labeled census\n")
        add("Owner framing: this is a RECALL-FIRST tripwire for a later model look — "
            "precision is CALIBRATION data for the Phase-35/36 look, NOT a target. A false "
            "trigger costs one logged record; a miss costs another specimen.\n")
        add(f"- Census size compared: {pr['n_common']} messages\n")
        add("| class | precision | recall | TP | FP | FN |")
        add("|-------|-----------|--------|----|----|----|")
        for cls in _SIGNAL_CLASSES:
            c = pr["per_class"][cls]
            add(f"| {cls} | {_fmt(c['precision'])} | {_fmt(c['recall'])} | {c['tp']} | {c['fp']} | {c['fn']} |")
        head = pr["per_class"]["correction"]
        add(f"\n**COLL-05 headline — correction precision = {_fmt(head['precision'])}, "
            f"recall = {_fmt(head['recall'])}** (calibration data, recall-first detector).\n")
        add("Confusion (rows = detected, cols = actual hand label):\n")
        axes = ("correction", "confirmation", "interruption", "none")
        add("| detected \\ actual | " + " | ".join(axes) + " |")
        add("|" + "---|" * (len(axes) + 1))
        for d in axes:
            add(f"| {d} | " + " | ".join(str(pr['confusion'][d][a]) for a in axes) + " |")
        add("\nPer-trigger-family precision (finer calibration granularity):\n")
        add("| family | fires | true-class hits | precision |")
        add("|--------|-------|-----------------|-----------|")
        for fam, fires in sorted(pr["fam_fire"].items()):
            hits = pr["fam_hit"].get(fam, 0)
            add(f"| {fam} | {fires} | {hits} | {_fmt(hits/fires if fires else None)} |")
        add("")
    else:
        add("## Correction-detection precision/recall\n")
        add("No --hand-labels provided; precision/recall omitted (run with the census to "
            "produce the COLL-05 number).\n")

    # (g) Calibration cross-check
    add("## Calibration cross-check (free labels — collect-only)\n")
    add(f"- Detected corrections checked: {calib['checked']}")
    add(f"- Corrections with NO tool activity in the 10 min before the message: {calib['no_activity']}")
    add(f"- 'miscalibrated-prior candidates' (correction arrived but the max preceding tool "
        f"score < global median): {calib['candidates']}")
    add("  These are reported, never acted on (collect-only). Phase 35 decides the reading.\n")

    # (h) Honesty footer
    add("## Honesty footer\n")
    add(f"- Circularity clause: {_CIRCULARITY_CLAUSE}")
    add("- Latency slop: duration_ms is the WALL-CLOCK delta between the tool_call Action "
        "and its Observation (queuing + logging included) — not pure tool exec time; the "
        "latency z-band is noisier than a true tool-time band would be.")
    add(f"- Unmatched Action/Observation pairs: {corpus['unmatched']} "
        + ("(clean — every tool_call resolved)" if corpus["unmatched"] == 0 else "(see caveat above)"))
    add(f"- Cold-start fraction: {corpus['cold_frac']:.1%} — with the priors this cold, the "
        "kill check leans on the tools that DID accumulate history.")
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
        scores_path = results / "scores.jsonl"
        signals_path = results / "signals.jsonl"
        p1 = await _pass1_scores(trace, cfg, scores_path)
        signals = await _pass2_signals(trace, cfg, signals_path)
    except Exception as exc:  # noqa: BLE001 — any replay/score failure is an honest exit 1
        print(f"PROCESSING FAILURE: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    rows = p1["rows"]
    if not rows:
        print("PROCESSING FAILURE: no tool observations scored (empty or wrong trace?)", file=sys.stderr)
        return 1

    # Corpus stats
    all_sorted = sorted(r["score"] for r in rows)
    global_median = _pct(all_sorted, 50)
    quadrants = dict(Counter(r["quadrant"] for r in rows))
    cold_n = quadrants.get("cold_start", 0)
    per_tool_map: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_tool_map[r["tool"]].append(r)
    per_tool = []
    for t in sorted(per_tool_map, key=lambda k: -len(per_tool_map[k])):
        sub = per_tool_map[t]
        ss = sorted(x["score"] for x in sub)
        per_tool.append((t, {"n": len(sub), "err": sum(x["is_error"] for x in sub) / len(sub),
                             "p50": _pct(ss, 50), "p90": _pct(ss, 90)}))

    # Kill check (mechanical)
    kill = _kill_check(rows)

    # Named specimens
    wf = [{"event_id": r["event_id"], "score": r["score"]} for r in rows
          if r["tool"] == "web_fetch" and r["is_error"] == 1]
    for r in wf:
        r["pct"] = _pct_rank(all_sorted, r["score"])
    for r in p1["uv_specimens"]:
        r["pct"] = _pct_rank(all_sorted, r["score"])
    wf_ranks = sorted(r["pct"] for r in wf)
    anomalies = []
    if kill["statistical_channel"] == "separates" and global_median is not None:
        for kind, lst in (("web_fetch-error", wf), ("uv-not-found", p1["uv_specimens"])):
            for r in lst:
                if r["score"] < global_median:
                    anomalies.append({"kind": kind, "event_id": r["event_id"],
                                      "score": r["score"], "pct": r["pct"]})
    spec = {"web_fetch": wf, "uv": p1["uv_specimens"], "anomalies": anomalies,
            "wf_min": wf_ranks[0] if wf_ranks else None,
            "wf_med": _pct(wf_ranks, 50) if wf_ranks else None,
            "wf_max": wf_ranks[-1] if wf_ranks else None}

    # Signal census + specimen assertion
    sig_types = dict(Counter(s["detected"]["signal_type"] for s in signals if s["detected"]))
    sig_families = dict(Counter(s["detected"]["trigger_family"] for s in signals if s["detected"]))
    specimen = next((s for s in signals if s["event_id"] == _SPECIMEN_EVENT_ID), None)
    specimen_ok = True
    if specimen is None:
        specimen_note = ("NOT PRESENT in this trace (synthetic/other trace — assertion "
                          "skipped; the real orchestrator trace has it at line 4625)")
    elif specimen["detected"] and specimen["detected"]["signal_type"] == "correction":
        specimen_note = (f"detected as CORRECTION (family="
                          f"{specimen['detected']['trigger_family']}, matched="
                          f"'{specimen['detected']['matched_text']}') — the motivating "
                          "asymmetry is closed and measured.")
    else:
        specimen_ok = False
        specimen_note = ("MISS — the shipped lexicon did NOT detect the motivating specimen "
                          "as a correction. This is a phase failure (exit 1).")

    # Precision/recall vs census (optional)
    pr = None
    labels = None
    if args.hand_labels:
        labels = json.loads(Path(args.hand_labels).expanduser().read_text(encoding="utf-8"))
        pr = _precision_recall(signals, labels)

    # Calibration cross-check
    calib = _calibration_crosscheck(signals, rows, global_median)

    corpus = {
        "trace": str(trace), "unmatched": p1["unmatched"], "tools": sorted(per_tool_map),
        "first_ts": min(r["ts"] for r in rows), "last_ts": max(r["ts"] for r in rows),
        "all_sorted": all_sorted, "quadrants": quadrants, "cold_n": cold_n,
        "cold_frac": cold_n / len(rows), "per_tool": per_tool,
        "sig_types": sig_types, "sig_families": sig_families, "specimen_note": specimen_note,
        "empty_err": p1["empty_err"],
        "n_err": sum(r["is_error"] for r in rows),
    }

    # Write verdict.json
    verdict = {k: kill[k] for k in ("statistical_channel", "median_surprising", "p90_routine",
                                    "n_surprising", "n_routine", "reason")}
    (results / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")

    # Build + write report.md
    report = _build_report(rows, signals, kill, spec, calib, pr, corpus)
    (results / "report.md").write_text(report, encoding="utf-8")

    # Stdout summary (owner-facing)
    verdict_line = ("STATISTICAL CHANNEL: SEPARATES" if kill["statistical_channel"] == "separates"
                    else "STATISTICAL CHANNEL: KILL — drop the statistical channel in Phase 35, "
                         "keep user-signal only, scores remain telemetry.")
    print(f"scored {len(rows)} observations | {len(signals)} user messages | "
          f"cold-start {corpus['cold_frac']:.1%}")
    print(verdict_line)
    print(f"reason: {kill['reason']}")
    print(f"specimen: {specimen_note}")
    if pr is not None:
        h = pr["per_class"]["correction"]
        print(f"correction precision={_fmt(h['precision'])} recall={_fmt(h['recall'])} "
              f"(recall-first calibration data)")
    print(f"report: {results / 'report.md'}")

    if not specimen_ok:
        return 1
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="COLL-05 real-trace distribution report + mechanical kill verdict")
    p.add_argument("--trace", default=str(Path("~/.localharness/agents/orchestrator/bus-events.jsonl")),
                   help="bus-events JSONL to replay (READ-ONLY; default = the real orchestrator trace)")
    p.add_argument("--results", required=True,
                   help="ISOLATED output dir (REQUIRED). Refuses any path inside bench/results.")
    p.add_argument("--hand-labels", default=None,
                   help="Optional hand_labels.json census for precision/recall.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    results = Path(args.results).expanduser().resolve()
    # GUARD (before any output is written): never contaminate the shared bench results dir.
    if "bench/results" in str(results).replace("\\", "/"):
        print("REFUSED: --results points inside bench/results. The shared bench results dir "
              "is contaminated (project hard rule) — use an isolated dir "
              "(e.g. ~/.localharness/coll-reports/...).", file=sys.stderr)
        sys.exit(2)
    results.mkdir(parents=True, exist_ok=True)
    sys.exit(asyncio.run(run(args, results)))


if __name__ == "__main__":
    main()
