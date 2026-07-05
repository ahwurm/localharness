"""PGATE-03 mining half — the bounded transcript model-look (36-06).

The lexical tripwire (`user_signals.py`) structurally misses trigger-word-free corrections
(census recall ceiling 0.231) and never sees plain personal facts at all (live specimen:
"i got super duper sun burnt today" — not a correction, not a tool event, exactly what a
colleague would remember). This idle look mines the TRANSCRIPT for both classes: the model
IS the precision instrument, so its output writes at INJECTABLE confidence (>=0.7) — but
bounded four ways so it can neither run away nor hallucinate (CONTEXT ruling 3):

  1. WATERMARK-BOUNDED (cost): only history newer than the "mining/last_ts" watermark is
     read, and the watermark advances each pass — cost is per-window, never O(lifetime
     history). Without this the whole growing history.jsonl (`read_all` is O(whole file))
     is re-parsed and re-mined every idle cycle (Pitfall 6). Distinct from consolidation's
     own "consolidation/last_run" watermark.
  2. GROUNDED (kill discipline): every candidate line must pass `idle_llm.grounded` against
     the cited transcript span — no token in a mined fact that isn't derivable from the
     span (the number-provenance kill extended to mined facts by CONTEXT ruling 3).
  3. BUDGETED: at most `write_budget` writes per idle cycle (a bounded colleague-memory
     intake; owner-tunable). Enforced in the write loop.
  4. SUPERSEDE-NOT-OVERWRITE: the key hashes the line (`mined/{_h8(line)}`), so a re-mined
     identical line hits `store_fact`'s corroboration branch (no duplicate row) rather than
     spawning a new fact; provenance points at the transcript span.

Machine-safety: ALL LLM work routes through `idle_llm.complete_cancellable` (cooperatively
cancellable + char-capped, so a user turn never waits behind an idle generation and no
look launches an unattended long-context prefill). This function NEVER raises into the idle
scheduler — a mining fault must not break the idle cycle.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from localharness.memory.consolidation import _get_meta, _set_meta
from localharness.memory.idle_llm import complete_cancellable, grounded

log = logging.getLogger(__name__)

_MINING_WATERMARK_KEY = "mining/last_ts"  # DISTINCT from consolidation/last_run


def _h8(line: str) -> str:
    """Stable 8-hex key suffix for a mined line so an identical re-mined line collapses onto
    the same key and corroborates (store_fact) rather than duplicating (WRITE-01/02)."""
    return hashlib.sha1(line.strip().encode("utf-8")).hexdigest()[:8]


@dataclass
class MineReport:
    written: int = 0
    rejected_ungrounded: int = 0
    cancelled: bool = False


async def mine_transcript(
    store: Any,
    llm: Any,
    cancel_event: Any,
    *,
    write_budget: int = 5,
    corpus_char_cap: int = 6000,
) -> MineReport:
    """Idle model-look: mine post-watermark transcript for missed corrections + personal
    facts, write them injectable (0.7) with span provenance under a per-cycle write budget,
    then advance the watermark. Cancellable, never raises. Returns a MineReport."""
    report = MineReport()
    try:
        raw_wm = await _get_meta(store, _MINING_WATERMARK_KEY)
        try:
            watermark = int(raw_wm) if raw_wm else 0
        except (TypeError, ValueError):
            watermark = 0

        # Windowed read: only records NEWER than the watermark, with actual content. Mirrors
        # consolidation._replay_sessions' str(content)[:400] extraction so mixed live/backfill
        # provenance is tolerated uniformly.
        records = await store.get_history(limit=1000)
        window = [
            r for r in records
            if int(r.get("ts", 0) or 0) > watermark and r.get("content")
        ]
        if not window:
            return report  # nothing new — watermark unchanged; cost paid only on the read

        corpus = "\n".join(str(r.get("content", ""))[:400] for r in window)[:corpus_char_cap]
        newest = max(window, key=lambda r: int(r.get("ts", 0) or 0))
        newest_ts = int(newest.get("ts", 0) or 0)
        span_ref = newest.get("session_id") or str(newest_ts)
        provenance = f"mined-from:{span_ref}"

        prompt = (
            f"Extract at most {write_budget} durable facts a colleague would remember from "
            "this transcript — (a) preference/personal facts the user stated about "
            "themselves, (b) corrections the user made that lack obvious trigger words. "
            "One per line, no numbering. ONLY things explicitly stated below — no "
            "inference, no new numbers, and never restate what the user corrected:\n\n"
            + corpus
        )
        raw = await complete_cancellable(llm, prompt, cancel_event, char_cap=corpus_char_cap)
        if raw is None:
            # Cancelled mid-look (a user turn arrived): watermark NOT advanced so the next
            # idle pass re-mines this window.
            report.cancelled = True
            return report

        for raw_line in raw.splitlines():
            line = raw_line.strip(" -•\t")
            if not line:
                continue
            # GROUNDING GATE — the kill discipline extended to mined facts: no line whose
            # tokens aren't a majority-match in the cited span is ever written.
            if not grounded(line, corpus):
                report.rejected_ungrounded += 1
                continue
            await store.store_fact(
                key=f"mined/{_h8(line)}",
                value=line,
                tags=["mined", "pending_consolidation"],
                confidence=0.7,  # injectable (CONTEXT ruling 3) — NOT sub-0.7
                source="transcript_mining",
                provenance=provenance,
                node_kind="fact",
            )
            report.written += 1

        # Advance the watermark to the newest ts seen — the next pass starts strictly after.
        await _set_meta(store, _MINING_WATERMARK_KEY, str(newest_ts))
    except Exception:
        log.exception("mine_transcript failed (non-fatal; the idle look swallows)")
    return report
