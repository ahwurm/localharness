"""Trace pack — bench run files → a versioned, leak-scanned dataset artifact.

Turns a bench results tree ({model}/{scenario}/{timestamp}.jsonl bus-event traces, each
ending in a ScenarioCompleted verdict) into {manifest.json, trajectories.jsonl}: one
chat-format record per run, labeled with the gate verdict — graded trajectories usable
as a regression eval for candidate models today and as SFT material if a fine-tune is
ever justified (owner bar 2026-07-20: genuine lift, a small-model-viability play, or a
release-worthy artifact — else pass).

Bench-only by construction: files without a ScenarioCompleted verdict (live dogfood
sessions — the owner's real life) are skipped and counted, never packed. Any home-path/
key/secret pattern in packed content fails the build (PackLeakError) — fail explicit,
no silent scrubbing. Packs are regenerated per release as functionality evolves; the
manifest stamps harness_version + source so stale packs are self-describing.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import localharness

_LEAK_RX = re.compile(
    r"/home/[A-Za-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/|api[_-]?key|BEGIN [A-Z ]*PRIVATE KEY"
    r"|bearer [A-Za-z0-9_.-]{16,}|\.ssh/|secrets?\.(?:ya?ml|json|env)",
    re.IGNORECASE,
)


class PackLeakError(RuntimeError):
    """A leak pattern surfaced inside content that would have been packed."""


def _iter_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn tail line is data loss in ONE event, not a build failure
    return events


def _messages_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chat-format reconstruction: user / assistant(+tool_calls) / tool observation lines.
    Heartbeats, metrics and lifecycle events are not conversation — dropped."""
    msgs: list[dict[str, Any]] = []
    for ev in events:
        etype = ev.get("event_type")
        if etype == "UserMessage":
            msgs.append({"role": "user", "content": ev.get("content") or ""})
        elif etype == "TurnStarted":
            # Bench runs drive the loop directly — no UserMessage event; the prompt lives
            # in task_summary. REPL-shaped traces have BOTH: skip when the user turn was
            # already emitted (last message is a user turn) to avoid a double prompt.
            summary = ev.get("task_summary") or ""
            if summary and not (msgs and msgs[-1].get("role") == "user"):
                msgs.append({"role": "user", "content": summary})
        elif etype == "Action" and ev.get("action_type") == "llm_response":
            m: dict[str, Any] = {"role": "assistant", "content": ev.get("content") or ""}
            if ev.get("tool_name"):
                m["tool_calls"] = [{"name": ev["tool_name"],
                                    "params": ev.get("tool_params") or {}}]
            msgs.append(m)
        elif etype == "Observation":
            msgs.append({"role": "tool",
                         "content": ev.get("content") or ev.get("result") or ""})
    return msgs


def build_pack(results_root: Path, out_dir: Path) -> dict[str, Any]:
    """Build the pack; returns the manifest (also written to out_dir/manifest.json).

    Scans results_root recursively for *.jsonl; a file is a bench run iff it carries a
    ScenarioCompleted event. The WHOLE build fails on the first leak hit — nothing is
    written until every record passed the scan.
    """
    results_root = Path(results_root)
    out_dir = Path(out_dir)
    records: list[dict[str, Any]] = []
    skipped = 0

    for path in sorted(results_root.rglob("*.jsonl")):
        events = _iter_events(path)
        verdict = next((e for e in events if e.get("event_type") == "ScenarioCompleted"), None)
        if verdict is None:
            skipped += 1
            continue
        messages = _messages_from_events(events)
        blob = json.dumps(messages, ensure_ascii=False)
        hit = _LEAK_RX.search(blob)
        if hit:
            raise PackLeakError(
                f"leak pattern {hit.group(0)!r} in {path} "
                f"(scenario {verdict.get('scenario_name')!r}) — pack build refused"
            )
        records.append({
            "scenario": verdict.get("scenario_name"),
            "model": verdict.get("model"),
            "success": bool(verdict.get("success")),
            "harness_version": localharness.__version__,
            "run_file": str(path.relative_to(results_root)),
            "messages": messages,
            "metrics": {k: verdict.get(k) for k in (
                "tokens_in", "tokens_out", "iterations", "tool_call_count",
                "parse_failures", "stuck_recoveries") if k in verdict},
        })

    if not records:
        raise RuntimeError(
            f"no bench run files (ScenarioCompleted-verdicted *.jsonl) under {results_root}"
        )

    outcomes: dict[tuple, dict[str, Any]] = {}
    for r in records:
        key = (r["scenario"], r["model"])
        o = outcomes.setdefault(key, {"scenario": r["scenario"], "model": r["model"],
                                      "runs": 0, "successes": 0})
        o["runs"] += 1
        o["successes"] += int(r["success"])

    manifest = {
        "pack_schema": 1,
        "harness_version": localharness.__version__,
        "created_at": int(time.time()),
        "source_root": str(results_root),
        "runs_packed": len(records),
        "files_skipped": skipped,
        "outcomes": sorted(outcomes.values(), key=lambda o: (o["scenario"], o["model"])),
        "leak_scan": "passed",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "trajectories.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
