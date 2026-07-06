"""The PRIMARY semantic feeder (MOVE 2) — the idle transcript miner.

Domain knowledge about the USER'S WORLD (what they work on, prefer, plan, decide) enters the
semantic hierarchy HERE: an idle model-look walks the un-mined transcript and extracts TYPED
atoms — `(topic, claim, evidence-span)` — writing one `sem/{topic-slug}/{h8(claim)}` fact per
atom. Same-topic atoms share a key namespace, giving clustering natural handles (shared prefix +
FTS token + graph edge to the session). This is the feeder that was previously missing: mining
shipped as a budgeted rider and the orphaned replay seam wrote unreachable `replay/*` keys — the
two near-duplicate idle extractors are now ONE (the replay seam is retired into this walk).

Four bounds keep it honest and cheap (CONTEXT ruling 3, carried forward):
  1. WATERMARK-BOUNDED (cost): only history newer than "mining/last_ts" is read; the watermark
     advances per COMPLETED chunk, so a cancelled/over-budget pass resumes where it stopped and
     cost is per-window, never O(lifetime history).
  2. CHUNKED FULL-COVERAGE: the ENTIRE un-mined window is walked in `corpus_char_cap` chunks
     (a loop, not one nibble) so a long backlog is fully mined across a pass — one LLM look per
     chunk, cancellation-checked between chunks.
  3. GROUNDED (kill discipline): each atom is attributed to its SOURCE record (best token
     overlap of its claim+evidence) and the claim must pass `grounded` against that record's
     text — no token that isn't derivable from the cited span. Ungrounded atoms are rejected
     and counted, never written.
  4. BUDGETED: at most `write_budget` writes per pass. A budget-truncated chunk does NOT advance
     the watermark, so the next pass re-mines it (identical claims corroborate, never duplicate).

Load-bearing (research doc §4): provenance is the SOURCE record's `session_id` PER ATOM, not the
mining batch — the cluster stability bar counts distinct provenance sessions, so a batch-level
provenance would make every mined atom share one session and never satisfy the ≥2-session bar.

Write confidence is 0.65 — searchable but sub-injection: a single mention is not yet ambient;
ambient status is EARNED by distinct-day recurrence (the store's ladder) or chapter membership.

Machine-safety: ALL LLM work routes through `idle_llm.complete_cancellable` (cooperatively
cancellable + char-capped). This function NEVER raises into the idle scheduler.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any

from localharness.memory.consolidation import _get_meta, _set_meta
from localharness.memory.idle_llm import complete_cancellable, grounded

log = logging.getLogger(__name__)

_MINING_WATERMARK_KEY = "mining/last_ts"  # DISTINCT from consolidation/last_run
_PER_RECORD_CHARS = 400  # per-record slice fed to the model (and grounded against)

_PROMPT = (
    "Extract durable facts about the USER'S WORLD from this transcript — what the user works "
    "on, prefers, plans, or decides. One fact per line, formatted EXACTLY as three "
    "pipe-separated fields:\n"
    "topic | claim | evidence\n"
    "where `topic` is a 1-3 word subject (e.g. subagents, kyoto trip, gpu ops), `claim` is one "
    "self-contained fact, and `evidence` is a short VERBATIM quote from the transcript that "
    "supports it. ONLY facts explicitly stated below — no inference, no new numbers:\n\n"
)


def _h8(claim: str) -> str:
    """Stable 8-hex key suffix for a claim so an identical re-mined claim collapses onto the
    same `sem/` key and corroborates (store_fact) rather than duplicating."""
    return hashlib.sha1(claim.strip().encode("utf-8")).hexdigest()[:8]


def _slug(topic: str) -> str:
    """A topic namespace slug: lowercase, non-alphanumeric runs -> single hyphen, trimmed.
    Same-topic atoms therefore share the `sem/{slug}/` prefix — a natural clustering handle."""
    s = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return s or "misc"


def _tokens(text: str, *, min_len: int = 5) -> list[str]:
    """Content tokens >= min_len chars (drops stopwords/punctuation) — the attribution probe."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= min_len]


def _parse_atoms(raw: str) -> list[tuple[str, str, str]]:
    """Parse `topic | claim | evidence` lines into typed atoms; evidence falls back to the claim
    when omitted. Malformed lines (no pipe, empty topic/claim) are skipped."""
    atoms: list[tuple[str, str, str]] = []
    for raw_line in raw.splitlines():
        line = raw_line.strip(" -•\t")
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        topic, claim = parts[0], parts[1] if len(parts) > 1 else ""
        evidence = parts[2] if len(parts) > 2 and parts[2] else claim
        if topic and claim:
            atoms.append((topic, claim, evidence))
    return atoms


def _source_record(claim: str, evidence: str, records: list[dict]) -> dict | None:
    """The record whose text best overlaps the atom's (claim+evidence) tokens — the atom's
    episodic source, whose session_id becomes the per-atom provenance. None when nothing
    overlaps (a fabricated atom with no home in the window)."""
    probe = set(_tokens(f"{claim} {evidence}"))
    if not probe:
        return None
    best, best_score = None, 0
    for r in records:
        content = str(r.get("content", ""))[:_PER_RECORD_CHARS].lower()
        score = sum(1 for t in probe if t in content)
        if score > best_score:
            best, best_score = r, score
    return best


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
    write_budget: int = 25,
    corpus_char_cap: int = 6000,
) -> MineReport:
    """Walk the un-mined transcript in `corpus_char_cap` chunks, extract typed `sem/` atoms
    (grounded, per-atom source provenance, 0.65), advancing the watermark per completed chunk.
    Cancellable, budgeted, never raises. Returns a MineReport."""
    report = MineReport()
    try:
        raw_wm = await _get_meta(store, _MINING_WATERMARK_KEY)
        try:
            watermark = int(raw_wm) if raw_wm else 0
        except (TypeError, ValueError):
            watermark = 0

        records = await store.get_history(limit=1_000_000)
        window = sorted(
            (r for r in records if int(r.get("ts", 0) or 0) > watermark and r.get("content")),
            key=lambda r: int(r.get("ts", 0) or 0),
        )
        if not window:
            return report  # nothing new — watermark unchanged; cost paid only on the read

        i = 0
        budget_hit = False
        while i < len(window) and not budget_hit:
            if getattr(cancel_event, "is_set", lambda: False)():
                report.cancelled = True
                break
            # Assemble one chunk up to corpus_char_cap chars of record content.
            chunk: list[dict] = []
            chars = 0
            while i < len(window) and chars < corpus_char_cap:
                r = window[i]
                chunk.append(r)
                chars += len(str(r.get("content", ""))[:_PER_RECORD_CHARS]) + 1
                i += 1
            corpus = "\n".join(
                str(r.get("content", ""))[:_PER_RECORD_CHARS] for r in chunk
            )[:corpus_char_cap]

            raw = await complete_cancellable(
                llm, _PROMPT + corpus, cancel_event, char_cap=corpus_char_cap + len(_PROMPT) + 64
            )
            if raw is None:
                # Cancelled mid-look — watermark NOT advanced past this chunk (next pass re-mines).
                report.cancelled = True
                break

            for topic, claim, evidence in _parse_atoms(raw):
                if report.written >= write_budget:
                    budget_hit = True
                    break
                src = _source_record(claim, evidence, chunk)
                # GROUNDING KILL-NET: reject an atom with no source or whose claim is not a
                # majority-token match in the cited record's text.
                if src is None or not grounded(claim, str(src.get("content", ""))[:_PER_RECORD_CHARS]):
                    report.rejected_ungrounded += 1
                    continue
                provenance = src.get("session_id") or str(src.get("ts", ""))
                await store.store_fact(
                    key=f"sem/{_slug(topic)}/{_h8(claim)}",
                    value=claim,
                    tags=["sem", "pending_consolidation"],
                    confidence=0.65,  # searchable, sub-injection — ambient status is EARNED
                    source="transcript_mining",
                    provenance=provenance,  # the SOURCE record's session, PER ATOM (load-bearing)
                    node_kind="fact",
                )
                report.written += 1

            if budget_hit:
                break  # partially-written chunk: don't advance — next pass re-mines (corroborates)
            newest_ts = max(int(r.get("ts", 0) or 0) for r in chunk)
            await _set_meta(store, _MINING_WATERMARK_KEY, str(newest_ts))
    except Exception:
        log.exception("mine_transcript failed (non-fatal; the idle look swallows)")
    return report
