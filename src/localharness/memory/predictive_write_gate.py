"""PredictiveWriteGate (Phase 35, PGATE-01/02/03) — the LIVE write decision.

Phase 34 measured surprise on every tool outcome and logged user-signal triggers, but
gated NOTHING (score everything, write nothing). This gate is where those measurements
start having consequences: a sibling bus subscriber that reads the already-published
`SurpriseScored` and `UserMessage` events (zero recompute, zero model calls, zero extra
DB reads for priors) and turns them into real, reversible, sub-0.7 memory writes.

Why a sibling and not an extension of WriteGate (gate.py): the phase's success bar is
"the motif floor stays provably unchanged". A sibling makes the diff on gate.py /
predictive_gate.py / user_signals.py EMPTY by construction — the strongest possible form
of "unchanged", not a re-run-and-re-diff discipline. This module reuses their pure
seams (`classify_user_signal`, `store_fact`, `staged_suspect_facts`) and edits none of
them.

Three requirements land here as tested behavior:
- PGATE-01: write on `quadrant == "surprising_failure"`; grade confidence from the
  carried score (derived from 34's real distribution); the tier carries a non-zero
  importance prior (sqlite `_IMPORTANCE_PRIORS`, added in Task 1).
- PGATE-02: `unsurprising_failure` (the priors predicted the failure) writes NOTHING —
  satisfied by construction, it is simply not the write quadrant.
- PGATE-03: a `correction_phrase` correction supersedes the staged suspect fact it is
  content-RELATED to (>=1 shared salient token — BUG #45; reversible, key-based) or, with no
  related suspect, writes a standalone quarantine fact keyed to the correction. A bare
  `negation` is quarantine-only, and `reask`/`frustration` are out of scope (measured 0/18
  and no-data respectively).

Every write stays confidence < 0.7 so it lives in the store but never enters the injected
ambient block (sqlite injection gate: confidence >= 0.7 AND retrieval_strength >= 0.2)
until Phase 36's consolidation confirms it — the CLS fast-capture / slow-integrate split.

Discipline mirrored from WriteGate: every handler swallows its own exceptions (logged,
never re-raised) so a write-gate fault can never break the user's turn or the loop.
"""
from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from localharness.memory.user_signals import classify_user_signal

if TYPE_CHECKING:
    from localharness.config.models import PredictiveGateConfig
    from localharness.core.bus import EventBus, SubscriptionHandle
    from localharness.core.events import SurpriseScored, UserMessage
    from localharness.memory.sqlite import MemoryStore

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure decision helpers — module-level so the Wave-2 replay harness measures the
# SHIPPED decision, not a fork of it (34-RESEARCH: "35 reads, not re-derives").
# ---------------------------------------------------------------------------

STAT_WRITE_QUADRANT = "surprising_failure"
CORRECTION_WRITE_FAMILIES = frozenset({"negation", "correction_phrase"})


def should_write_stat(quadrant: str) -> bool:
    """The stat channel writes ONLY on the surprising_failure quadrant.

    PGATE-02 is satisfied HERE by construction: `unsurprising_failure` (a failure the
    priors expected), `routine`, `cold_start` and `quiet_surprise` are simply not the
    write quadrant, so no suppression logic is needed — the quadrant the motif gate
    structurally cannot express just never enters the write set.
    """
    return quadrant == STAT_WRITE_QUADRANT


def graded_confidence(score: float) -> float:
    """PGATE-01: map a graded surprise score to a sub-0.7 confidence.

    Derivation from Phase 34's real distribution (phase34-20260705T004611Z, n=1145):
    routine P90 score = 0.512 anchors the floor; the surprising median score = 2.086
    grades to ~0.646; the surprising_failure quadrant's own observed minimum score
    (0.514) sits at routine P90, so the CATEGORICAL quadrant gate (should_write_stat),
    not this number, makes the write/no-write call — the score only GRADES importance
    WITHIN the tier. Clamped strictly < 0.7 so a stat fact NEVER enters the injected
    index, and floored at 0.5 so even a low-score surprise is a real candidate. Monotone
    non-decreasing in score.
    """
    return max(0.5, min(0.69, 0.5 + 0.07 * score))


def correction_in_scope(family: str | None) -> bool:
    """PGATE-03 scope: only negation / correction_phrase families write inline.

    Measured on 34's census: reask is 0/18 (the majority false-positive source) and
    frustration has no census data — excluding both raises the write-relevant precision
    from 0.115 (all families) to 0.375 (negation+correction_phrase). reask is never even
    seen here: `classify_user_signal` never returns it (it is produced only by the
    detector's difflib window, which this stateless gate does not run), so reask
    exclusion is also by construction.
    """
    return family in CORRECTION_WRITE_FAMILIES


# ---------------------------------------------------------------------------
# Tier tags + confidences (all < 0.7 — below the injection gate) and tiny local
# text helpers. Local copies (not imports from gate.py) keep gate.py an untouched,
# uncoupled file — the empty-diff proof depends on zero references into it.
# ---------------------------------------------------------------------------

_STAT_TIER_TAG = "tier:surprising_failure"
_CORRECTION_TIER_TAG = "tier:correction_pending"
_SUSPECT_SUPERSEDE_CONFIDENCE = 0.6   # < 0.7: drops a disputed suspect out of the injected block
_QUARANTINE_CONFIDENCE = 0.65         # <= 0.65 (CONTEXT ruling 3); never injected until Phase 36 promotes
# The disputed-supersede wrapper (BLOCKER 1(c)): a fixed marker PREFIX, then the FULL
# original value — never truncated, so a plain get_fact still returns the real content.
_DISPUTE_MARKER = "[disputed by user correction — pending reconciliation]"


def _h8(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8", "replace")).hexdigest()[:8]


def _preview(text: str | None, n: int = 180) -> str:
    one = " ".join((text or "").split())
    return one[: n - 1] + "…" if len(one) > n else one


# BUG #45 relatedness floor: a correction may only dispute a STAGED fact it is actually ABOUT —
# not merely whichever fact was retrieved most recently. Reuses mining._salient_words, the
# B4-defense precedent's single salient-token rule (>=4-char, stopword-filtered [a-z0-9]+ tokens;
# a short/generic shared token can never authorize a dispute), so both correction defenses key on
# the same measured salience floor.
_SUSPECT_SCAN_DEPTH = 5  # scan the few most-recent staged suspects, not just the single most recent


def _best_related_suspect(
    text: str, suspects: list[tuple[int, str, str]]
) -> tuple[int, str, str] | None:
    """The staged suspect (id, key, value) sharing the MOST salient tokens with the correction
    `text`, among the `_SUSPECT_SCAN_DEPTH` most-recently-staged — or None if none shares >=1 (the
    relatedness floor). `suspects` arrive most-recent-first, so equal-overlap ties resolve to the
    more recent. Lazy import mirrors consolidation.py's own mining import: it keeps this hot-path
    subscriber module-level-decoupled from the mining module while reusing its salient-token rule."""
    from localharness.memory.mining import _salient_words

    probe = _salient_words(text)
    if not probe:
        return None
    best: tuple[int, str, str] | None = None
    best_overlap = 0
    for suspect in suspects[:_SUSPECT_SCAN_DEPTH]:
        overlap = len(probe & _salient_words(suspect[2]))
        if overlap > best_overlap:
            best_overlap, best = overlap, suspect
    return best


class PredictiveWriteGate:
    """Subscribes to SurpriseScored + UserMessage and writes gated, sub-0.7 candidates.

    Wired beside MemoryStore / WriteGate at startup (Wave 2) when the config's write_live
    lever is on. Copies WriteGate's subscribe/react/swallow/close lifecycle verbatim; adds
    nothing to the hot path beyond one categorical check and (on a fire) one store_fact —
    the same cost class the loop already pays for WriteGate.
    """

    def __init__(
        self,
        store: "MemoryStore",
        bus: "EventBus",
        agent_id: str,
        cfg: "PredictiveGateConfig",
    ) -> None:
        self._store = store
        self._bus = bus
        self._agent_id = agent_id
        self._cfg = cfg
        self._handles: list["SubscriptionHandle"] = []
        # Cache the lexicon once (mirror UserSignalDetector) — the same measured trigger
        # lists the census scored, so the write path can never drift from the numbers.
        self._lexicon: dict[str, list[str]] = cfg.lexicon.model_dump()

    async def open(self) -> None:
        from localharness.core.events import SurpriseScored, UserMessage
        self._handles.append(
            self._bus.subscribe(SurpriseScored, self._on_surprise_scored, agent_id=self._agent_id)
        )
        self._handles.append(
            self._bus.subscribe(UserMessage, self._on_user_message, agent_id=self._agent_id)
        )

    async def close(self) -> None:
        for h in self._handles:
            self._bus.unsubscribe(h)
        self._handles.clear()

    # ------------------------------------------------------------------
    # Stat channel (PGATE-01/02) — a pure dispatch on the already-scored event.
    # ------------------------------------------------------------------

    async def _on_surprise_scored(self, event: "SurpriseScored") -> None:
        try:
            # The KILL-revert lever (Wave-2 config field). Fail-closed: absent/unreadable
            # -> no live write, scores stay pure telemetry (PGATE-04 pre-committed kill).
            if not getattr(self._cfg, "write_live", False):
                return
            if not should_write_stat(event.quadrant):
                # PGATE-02: unsurprising_failure / routine / cold_start / quiet_surprise
                # -> no write. The suppression is the absence of a branch, not a branch.
                return
            # (tool, day) bucket so a same-day retry BURST corroborates into one row
            # instead of N near-duplicate facts (Pitfall 6); MAX() on corroboration keeps
            # the day's most-surprising grade. The score already rode in on the event —
            # no recompute, no prior lookup.
            day = event.timestamp.strftime("%Y%m%d")
            await self._store.store_fact(
                key=f"predgate/surprising_failure/{event.tool_name}/{day}",
                value=(
                    f"`{event.tool_name}` had a surprising failure — a normally-reliable "
                    f"tool errored (quadrant surprising_failure). Auto-captured by the "
                    f"predictive write gate; pending consolidation."
                ),
                tags=["gate", _STAT_TIER_TAG, "pending_consolidation"],
                confidence=graded_confidence(event.score),
                source="predictive_write_gate",
                provenance=event.session_id or "",
            )
        except Exception:  # the gate must never break the loop
            log.exception("predictive-write-gate surprise handler failed (non-fatal)")

    # ------------------------------------------------------------------
    # Correction channel (PGATE-03) — reuse the shipped tripwire; coarse, reversible.
    # ------------------------------------------------------------------

    async def _on_user_message(self, event: "UserMessage") -> None:
        try:
            if not getattr(self._cfg, "write_live", False):
                return
            text = (event.content or "").strip()
            if not text:
                return
            match = classify_user_signal(text, self._lexicon)  # the SHIPPED tripwire, reused
            if match is None or not correction_in_scope(match[1]):
                # Out of scope: no trigger, or frustration/confirmation/interruption; reask
                # is never returned here (this stateless gate keeps no per-sitting window).
                return
            # BLOCKER 1(a): DEFANG the supersede. Only the EXPLICIT correction_phrase family
            # ("i meant / actually / instead / you misunderstood …") is a strong enough signal
            # to rewrite an already-retrieved fact. A bare `negation` ("no", "nah", "wrong")
            # is far too broad (a happy "no worries, thanks!" trips it), so every non-
            # correction_phrase family is QUARANTINE-ONLY — a safe additive fact that never
            # touches the staged rows. Classification is untouched (this is a write decision).
            suspects = (
                await self._store.staged_suspect_facts()
                if match[1] == "correction_phrase"
                else []
            )
            # BUG #45: dispute the staged suspect the correction is actually ABOUT, not merely the
            # most recent. Scan the few most-recent suspects and pick the BEST content-related one
            # (>=1 shared salient token); if NONE is related — or there is no suspect at all — fall
            # through to the additive quarantine so an unrelated correction can never silently
            # supersede a just-retrieved fact (reconciliation / the model's own remember-correction
            # path still act on the captured quarantine).
            target = _best_related_suspect(text, suspects)
            if target is not None:
                # Supersede the RELATED suspect only. REVERSIBLE (store_fact keeps the old row
                # queryable via history) and BLOCKER 1(c): the marker prefixes the FULL original
                # value — no truncation.
                _fid, key, value = target
                await self._store.store_fact(
                    key=key,
                    value=f"{_DISPUTE_MARKER} {value}",
                    tags=["correction", _CORRECTION_TIER_TAG, "pending_consolidation"],
                    confidence=_SUSPECT_SUPERSEDE_CONFIDENCE,
                    source="predictive_write_gate",
                    provenance=event.id,  # pointer to the trigger record (== user_signals.event_id)
                )
            else:
                if suspects:  # BUG #45: staged suspects existed but none was content-related
                    log.debug(
                        "predictive-write-gate: correction shares no salient token with any of the "
                        "%d most-recent staged suspects — disputing nothing, quarantining instead",
                        min(len(suspects), _SUSPECT_SCAN_DEPTH),
                    )
                # The no-related-suspect / no-suspect shape (and every in-scope negation): no fact
                # to supersede, so quarantine the user's own words, keyed so a repeat corroborates
                # rather than duplicating. Payload-first.
                await self._store.store_fact(
                    key=f"correction/quarantine/{_h8(event.session_id or '', match[2], text)}",
                    value=f"user correction (pending reconciliation): {_preview(text)}",
                    tags=["correction", _CORRECTION_TIER_TAG, "pending_consolidation"],
                    confidence=_QUARANTINE_CONFIDENCE,
                    source="predictive_write_gate",
                    provenance=event.id,
                )
        except Exception:  # the gate must never break the turn
            log.exception("predictive-write-gate user-message handler failed (non-fatal)")
