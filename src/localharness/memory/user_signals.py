"""COLL-02 zero-NLU user-signal channel — TRIGGERS, not classifiers (owner steer
2026-07-04 22:49).

The lexicon is a TRIPWIRE for a later model look (Phase 35 inline coarse / Phase 36 idle
deep), never a classifier: precision comes from the look + pruning, never from the
trigger. Recall-first by design — a false trigger costs one logged record, a miss costs
another missed correction. Zero tokens, zero model calls, deterministic, auditable.

The detector subscribes to UserMessage / TurnCompleted, classifies each user turn from the
trigger lexicon, and writes a look-ready labeled record (the FULL user message + the
corrected-turn pointer) via MemoryStore.record_user_signal. Corrections and confirmations
also snapshot the explicitly-staged facts as suspect / bump candidates (COLL-03) — but
NOTHING here writes to the facts table: score everything, gate nothing. Phase 35 sets
thresholds from the collected distribution.
"""
from __future__ import annotations

import difflib
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from localharness.config.models import PredictiveGateConfig
    from localharness.core.bus import EventBus, SubscriptionHandle
    from localharness.core.events import TurnCompleted, UserMessage
    from localharness.memory.sqlite import MemoryStore

log = logging.getLogger(__name__)

# Families are checked in this fixed order; the first match wins. Correction-class
# (negation / correction_phrase / frustration) precedes confirmation precedes interruption,
# so "no wait —" labels as the STRONGER correction and an interruption word never absorbs a
# correction (owner ruling 2026-07-04).
_FAMILY_ORDER: tuple[str, ...] = (
    "negation",
    "correction_phrase",
    "frustration",
    "confirmation",
    "interruption",
)
_FAMILY_TO_SIGNAL: dict[str, str] = {
    "negation": "correction",
    "correction_phrase": "correction",
    "frustration": "correction",
    "confirmation": "confirmation",
    "interruption": "interruption",
}
# Word tokens (apostrophes kept so "that's" is one token). Single-word triggers match the
# token 'no' — never the substring inside 'know' / 'now'.
_TOKEN_RE = re.compile(r"[a-z']+")


def classify_user_signal(
    text: str, lexicon: dict[str, list[str]]
) -> tuple[str, str, str] | None:
    """Return (signal_type, trigger_family, matched_text) or None.

    Single-word triggers match whole tokens; multi-word triggers match as substrings.
    Families are checked in _FAMILY_ORDER and the first hit wins (correction-class beats
    confirmation beats interruption — an interruption word never absorbs a correction).
    Recall-first by design: this is a tripwire for a later model look, not a classifier.
    """
    lower = text.lower()
    tokens = set(_TOKEN_RE.findall(lower))
    for family in _FAMILY_ORDER:
        for trigger in lexicon.get(family, []):
            t = trigger.lower()
            hit = (t in lower) if " " in t else (t in tokens)
            if hit:
                return _FAMILY_TO_SIGNAL[family], family, t
    return None


def is_reask(current: str, prior_messages: list[str], threshold: float = 0.8) -> bool:
    """True when `current` is a near-identical re-phrasing of an earlier same-sitting
    message — the user asking again = an answered-question re-ask (COLL-02 lists re-asks
    under corrections). stdlib difflib, zero NLU, zero deps.
    """
    cur = current.lower()
    return any(
        difflib.SequenceMatcher(None, cur, p.lower()).ratio() > threshold
        for p in prior_messages
    )


class UserSignalDetector:
    """Subscribes to the agent's UserMessage / TurnCompleted stream and logs COLL-02 signals.

    Wired beside MemoryStore / WriteGate at startup when
    `agent.memory.predictive_gate.enabled` (default True — collect-only). Every handler
    swallows all exceptions: a detector fault must never break the user's turn.
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
        self._lexicon: dict[str, list[str]] = cfg.lexicon.model_dump()
        # The most recent assistant-turn summary — held as the pointer to the turn being
        # corrected for the NEXT user message's record (the correction words arrive in the
        # message that FOLLOWS the turn).
        self._last_turn_summary: str | None = None
        # session_id -> prior user messages this sitting (the re-ask comparison window).
        self._session_messages: dict[str, list[str]] = {}

    async def open(self) -> None:
        from localharness.core.events import TurnCompleted, UserMessage

        self._handles.append(
            self._bus.subscribe(UserMessage, self._on_user_message, agent_id=self._agent_id)
        )
        self._handles.append(
            self._bus.subscribe(
                TurnCompleted, self._on_turn_completed, agent_id=self._agent_id
            )
        )

    async def close(self) -> None:
        for h in self._handles:
            self._bus.unsubscribe(h)
        self._handles.clear()

    async def _on_turn_completed(self, event: "TurnCompleted") -> None:
        # 500 chars is the look-ready pointer budget; the full turn is recoverable from
        # history.jsonl by ts if a later model look needs more.
        try:
            self._last_turn_summary = (event.summary or "")[:500] or None
        except Exception:
            log.exception("user_signal_turn_completed_failed")

    async def _on_user_message(self, event: "UserMessage") -> None:
        try:
            text = (event.content or "").strip()
            if not text:
                return
            match = classify_user_signal(text, self._lexicon)
            sid = event.session_id or ""
            window = self._session_messages.setdefault(sid, [])
            if match is None and is_reask(text, window, self._cfg.reask_threshold):
                # A re-ask of an answered same-sitting question — no lexical trigger, but
                # COLL-02 lists re-asks under corrections (family='reask').
                match = ("correction", "reask", "")
            # Append AFTER the re-ask check (a message never matches itself); cap the
            # window, and cap the number of live sittings tracked (sittings are serial, so
            # stale keys are just bench noise).
            window.append(text)
            if len(window) > self._cfg.reask_window:
                del window[: len(window) - self._cfg.reask_window]
            if len(self._session_messages) > 8:
                self._session_messages.pop(next(iter(self._session_messages)))
            if match is None:
                return
            signal_type, family, matched = match
            signal_id = await self._store.record_user_signal(
                session_id=sid,
                ts=int(event.timestamp.timestamp()),
                signal_type=signal_type,
                trigger_family=family,
                matched_text=matched,
                user_message=text,
                corrected_turn_summary=self._last_turn_summary,
                event_id=event.id,
            )
            # COLL-03 collect-only credit assignment, scoped to explicitly-staged facts
            # (access_count_staged > 0 — what memory_search / memory_get staged since the
            # last consolidation fold). Ambient always-injected facts are NOT staged and NOT
            # snapshotted: v1 scopes credit to explicitly-retrieved facts (34-RESEARCH Open
            # Q2); ambient credit assignment is a named, deferred extension. NOTHING here
            # writes to the facts table.
            if signal_type == "correction":
                await self._store.snapshot_staged_candidates(signal_id, "suspect")
            elif signal_type == "confirmation":
                await self._store.snapshot_staged_candidates(signal_id, "bump")
            # interruption: signal row only — a weaker label class with no candidates
            # (owner ruling; the collected data decides the reading later).
        except Exception:
            log.exception("user_signal_detect_failed")
