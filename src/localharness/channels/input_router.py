"""Type-anytime input router.

Decides whether a message the user types WHILE an agent turn is running should NUDGE the
current turn (injected at the next step boundary, like the #82 stuck-recovery nudge) or
QUEUE as the next turn (FIFO, played after the current one ends).

Two tiers, cheapest first:
  - Tier 1: deterministic lexical rules, both directions, high precision. Pure functions,
    unit-tested per rule. Message-initial interrupt/correction shapes → NUDGE;
    future-framed / new-question shapes → QUEUE; everything else abstains.
  - Tier 2: ONE bounded classification call to the harness's already-configured LLM
    (injected as an async `complete_fn`), used only when Tier 1 abstains. Output is
    code-validated to a strict two-way verdict. ANY timeout / error / invalid output → QUEUE.

Design principle (owner): model judgment lands as VALIDATED data, and uncertain → QUEUE —
a late nudge is harmless, a false nudge pollutes a running turn.

A leading `!` is a silent force-nudge escape hatch (stripped; never surfaced in help/docs).

This module is pure routing logic — no rendering, no bus, no live model. The channel/REPL
calls it and owns delivery.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Optional

CompleteFn = Callable[[list[dict]], Awaitable[str]]


class Route(str, Enum):
    NUDGE = "nudge"
    QUEUE = "queue"


@dataclass(frozen=True)
class Decision:
    """One routing decision. `tier` ∈ {force, tier1, tier2}; `reason` is the rule name or
    the tier-2 outcome — both land verbatim in the InputRouted bus event (tuning data)."""

    route: Route
    tier: str
    reason: str


FORCE_PREFIX = "!"


def strip_force(text: str) -> tuple[str, bool]:
    """Leading `!` forces a nudge (silent escape hatch). Returns (clean_text, forced)."""
    if text.startswith(FORCE_PREFIX):
        return text[len(FORCE_PREFIX):].lstrip(), True
    return text, False


# --- Tier 1: lexical tables (small, high-precision, both directions) ---------

# Leading conversational filler skipped so "please stop" / "ok, stop" read as message-initial.
_FILLERS = frozenset({"ok", "okay", "hey", "please", "um", "uh", "yeah", "so", "hmm", "well"})

# Message-initial interrupt / correction tokens → NUDGE.
_NUDGE_INITIAL = frozenset({
    "stop", "no", "nope", "wait", "hold", "halt",
    "don't", "dont", "wrong", "careful", "actually", "instead",
})

# Message-initial future / new-task leads → QUEUE.
_QUEUE_INITIAL = frozenset({
    "also", "then", "next", "later", "afterward", "afterwards",
    "additionally", "plus", "finally",
})

# "use X not Y" / "... not Z, instead" correction shapes → NUDGE.
_CORRECTION_RE = re.compile(
    r"\buse\b[^.!?]*\bnot\b|\bnot\b[^.!?]*\binstead\b|\binstead of\b",
    re.IGNORECASE,
)

# Auxiliary / verb negations ("i'm not sure", "it's not working", "do not know") are NOT
# "X not Y" corrections — the word before a bare "not" being one of these means abstain.
_NEG_AUX = frozenset({
    "do", "does", "did", "is", "are", "was", "were", "be", "been", "being", "am",
    "can", "could", "would", "should", "has", "have", "had", "will", "wo", "must",
    "may", "might", "need", "dare", "ought", "'s", "'m", "'re", "it's", "that's",
    "i'm", "im", "its", "thats", "there's", "theres", "if", "whether", "why",
    "maybe", "probably", "really", "surely", "apparently", "still", "'ve",
})


def _has_correction(low: str) -> bool:
    """A corrective 'X not Y' / 'use X not Y' / 'Y instead of X' shape → NUDGE. A bare
    'not' counts only when the preceding word is a real alternative, not an auxiliary
    negation (keeps 'i'm not sure' / 'it's not working' out)."""
    if _CORRECTION_RE.search(low):
        return True
    words = re.findall(r"[\w'./-]+", low)
    for i, w in enumerate(words):
        if w == "not" and 0 < i < len(words) - 1 and words[i - 1] not in _NEG_AUX:
            return True
    return False

# Future-framed phrases anywhere in the message → QUEUE.
_QUEUE_PHRASE_RE = re.compile(
    r"\b(when you(?:'re| are)?\s+(?:done|finished)|when done|"
    r"after (?:this|that|you)|once you(?:'re| are)?|"
    r"also do|and also|one more thing|next time|afterwards?)\b",
    re.IGNORECASE,
)


def _content_words(low: str) -> list[str]:
    """Word tokens (apostrophes kept for don't) with leading filler words removed."""
    words = re.findall(r"[a-z']+", low)
    i = 0
    while i < len(words) and words[i] in _FILLERS:
        i += 1
    return words[i:]


def classify_tier1(text: str) -> Optional[Decision]:
    """Deterministic lexical route, or None to abstain (hand off to Tier 2 / queue-default).

    Precedence: message-initial nudge token → correction shape → message-initial queue lead
    → future-framed phrase → new-question shape → abstain."""
    low = text.strip().lower()
    if not low:
        return None
    words = _content_words(low)
    first = words[0] if words else ""

    if first in _NUDGE_INITIAL:
        return Decision(Route.NUDGE, "tier1", f"nudge-initial:{first.strip(chr(39))}")
    if _has_correction(low):
        return Decision(Route.NUDGE, "tier1", "nudge-correction")
    if first in _QUEUE_INITIAL:
        return Decision(Route.QUEUE, "tier1", f"queue-initial:{first}")
    if _QUEUE_PHRASE_RE.search(low):
        return Decision(Route.QUEUE, "tier1", "queue-future-framed")
    if low.endswith("?"):
        return Decision(Route.QUEUE, "tier1", "queue-question")
    return None


# --- Tier 2: bounded LLM fallback (uncertain → queue) ------------------------

_TIER2_SYSTEM = (
    "You route a short message the user typed WHILE an agent task is already running. "
    "Decide whether it should steer the CURRENT task right now (NUDGE) or run as the NEXT "
    "task after this one finishes (QUEUE). Corrections, stop/redirect, and 'do X not Y' are "
    "NUDGE. New, independent, or future-framed requests ('after this', 'also', a fresh "
    "question) are QUEUE. If you are unsure, answer QUEUE. "
    "Reply with exactly one word: NUDGE or QUEUE."
)


def _parse_verdict(raw: str) -> Optional[Route]:
    """Strict two-way parse. Exactly one of the words present → that route; both or
    neither → None (invalid)."""
    low = (raw or "").lower()
    has_nudge = re.search(r"\bnudge\b", low) is not None
    has_queue = re.search(r"\bqueue\b", low) is not None
    if has_nudge and not has_queue:
        return Route.NUDGE
    if has_queue and not has_nudge:
        return Route.QUEUE
    return None


async def classify_tier2(
    text: str,
    context: str,
    complete_fn: CompleteFn,
    timeout: float = 5.0,
) -> Decision:
    """ONE bounded classification call. `context` is a compact running-turn summary
    (current request + latest step/tool). ANY timeout / error / invalid output → QUEUE."""
    user = (
        f"Running task: {context}\n\n"
        f"Typed message: {text}\n\n"
        "One word — NUDGE or QUEUE:"
    )
    messages = [
        {"role": "system", "content": _TIER2_SYSTEM},
        {"role": "user", "content": user},
    ]
    try:
        raw = await asyncio.wait_for(complete_fn(messages), timeout=timeout)
    except Exception:
        # timeout, provider error, cancellation of the call — never let a routing call
        # break input; a queued message is the harmless default.
        return Decision(Route.QUEUE, "tier2", "tier2-error-default-queue")
    verdict = _parse_verdict(raw)
    if verdict is None:
        return Decision(Route.QUEUE, "tier2", "tier2-invalid-default-queue")
    return Decision(verdict, "tier2", f"tier2:{verdict.value}")


async def route(
    clean_text: str,
    forced: bool,
    *,
    context: str,
    complete_fn: Optional[CompleteFn],
    tier2_enabled: bool,
    timeout: float = 5.0,
) -> Decision:
    """Full decision for a message typed during a running turn (text already `!`-stripped).

    force → NUDGE; else Tier 1; else (Tier 1 abstained) ONE Tier-2 call when enabled and a
    client is available; else QUEUE by default."""
    if forced:
        return Decision(Route.NUDGE, "force", "force-bang")
    t1 = classify_tier1(clean_text)
    if t1 is not None:
        return t1
    if tier2_enabled and complete_fn is not None:
        return await classify_tier2(clean_text, context, complete_fn, timeout=timeout)
    return Decision(Route.QUEUE, "tier1", "abstain-default-queue")
