"""Salience (S) — rung 1 of the memory forgetting ladder.

One scoring function for the whole store, in one currency (log-odds-ish nats):

    S = need + truth + stakes

- **need** — the store's OWN base-level activation, `sqlite._base_activation` in its
  day-granular variant: ln(1 + uses) − d·ln(age in days), d = 0.5 (ACT-R's canonical
  decay exponent, the one literature constant this system carries). IMPORTED, never
  re-derived: there is exactly one activation formula in this codebase and a second
  copy of it would be a second forgetting curve.
- **truth** — the log-odds of the stored confidence, ln(c / (1 − c)). Confidence is a
  probability; addition only composes evidence in log-odds, so this is the conversion
  that lets truth be ADDED to need instead of multiplied by taste.
- **stakes** — the stored `importance` column as-is (declared stakes, rung 3 reworks
  how it is set; rung 1 only reads it).

`_slow_score` is deliberately NOT reused: it is `importance + base_activation`, so
feeding it here and adding `importance` again would double-count stakes.

**The line.** S is an ordinal ranking, not a calibrated probability, so rung 1 does not
invent a threshold — it reads one out of the store: the line is the MINIMUM S over the
facts the store can PROVE were worth keeping (ever recalled, or touched by the owner's
own hand). Nothing below the worst proven-useful fact has any evidence for it. The line
is recomputed from the data every pass and is never configurable; with no proven-useful
fact yet (a cold store) there is no line and nothing is archived. Junk that gets
recalled only lowers the line — every failure direction under-archives.

Rung 4 replaces this rule with the calibrated cost-ratio line once the closed loop can
score S against settled outcomes; until then the pin floor below is the hard guarantee.

GUARDRAILS note (checked 2026-09-18): GUARDRAILS is a FILE (`<org>/GUARDRAILS.md`,
sqlite.py:771) read straight off disk into context. No fact row ever carries a guardrail
tag and no writer mints one, so there is nothing to exempt here — the org safety voice
cannot be archived because it never enters the `facts` table at all.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

from localharness.memory.sqlite import (
    USER_EDIT_PROVENANCE_PREFIX,
    USER_FORGET_PROVENANCE_PREFIX,
    _base_activation,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from localharness.memory.sqlite import Fact

# The rung this module archives for — stamped into facts_archive.archive_rung so a later
# rung's moves stay distinguishable from this one's (rung 2 archives superseded versions,
# rung 3 unconfirmed mints; each must be separately restorable and separately auditable).
ARCHIVE_RUNG_DORMANCY = "rung1_dormancy"

# Logit DOMAIN GUARD — not a tuning knob. ln(c/(1−c)) diverges at c ∈ {0, 1}, so the
# argument must be held off the poles. The offset is the store's own confidence
# RESOLUTION: every confidence this codebase writes is a 2-decimal quantity (0.9
# remember, 0.8 chapter, 0.65 mining, the +0.07 corroboration ladder, the 0.85 cap), so
# one resolution unit in from each pole is the tightest clamp that cannot distort a
# value the store can actually hold. Measured live range is 0.5–0.9, so it never bites
# today; it exists so a future writer emitting a literal 0.0/1.0 cannot produce ±inf.
CONFIDENCE_RESOLUTION = 0.01
CONFIDENCE_LOGIT_MIN = CONFIDENCE_RESOLUTION
CONFIDENCE_LOGIT_MAX = 1.0 - CONFIDENCE_RESOLUTION

# The owner's own hand, in the three shapes the store records it: the `remember` tool,
# a `user_edit@`/`user_forget@` provenance stamp (web page + `localharness memory`), and
# the schema tier (a chapter LEADS its knowledge section — owner-facing by construction).
OWNER_HAND_SOURCES = frozenset({"remember"})
OWNER_HAND_PROVENANCE_PREFIXES = (
    USER_EDIT_PROVENANCE_PREFIX,
    USER_FORGET_PROVENANCE_PREFIX,
)
OWNER_HAND_TAGS = frozenset({"tier:schema"})


@dataclass(frozen=True)
class Salience:
    """One fact's score, with its three terms kept separate — a score you cannot take
    apart is a score you cannot debug (the dry-run's per-axis report reads these)."""
    fact_id: int
    key: str
    source: str
    s: float
    need: float
    truth: float
    stakes: float
    anchor: bool
    pinned: bool


def truth_log_odds(confidence: float | None) -> float:
    """ln(c / (1 − c)) with c held inside the logit's domain (see CONFIDENCE_RESOLUTION).

    Today this axis is nearly degenerate — the measured store's p10–p90 is all
    logit(0.65) = 0.619, mining's hardcoded birth confidence — so it carries a constant
    rather than evidence. Rung 3 replaces the invented per-source confidences with
    measured per-writer precision and the axis starts doing work; the FORM is right now.
    """
    c = min(max(confidence if confidence is not None else 0.5, CONFIDENCE_LOGIT_MIN),
            CONFIDENCE_LOGIT_MAX)
    return math.log(c / (1.0 - c))


def need_now(fact: "Fact", now: int) -> float:
    """The store's own base-level activation for this fact, day-granular.

    Day granularity (not the tool path's hourly variant) because archival is a
    long-horizon decision: every fact's age advances at the same calendar-day boundary,
    so a pass's verdict cannot depend on the hour it happened to fire.
    """
    return _base_activation(
        fact.access_count, fact.last_accessed_at, fact.updated_at, now,
        day_granularity=True,
    )


def is_pinned(fact: "Fact") -> bool:
    """The owner's hand — exempt from archival at any score, forever."""
    if (fact.source or "") in OWNER_HAND_SOURCES:
        return True
    prov = fact.provenance or ""
    if any(p in prov for p in OWNER_HAND_PROVENANCE_PREFIXES):
        return True
    return bool(OWNER_HAND_TAGS.intersection(fact.tags or ()))


def is_anchor(fact: "Fact") -> bool:
    """A fact the store can PROVE was worth keeping: it was actually recalled, or the
    owner touched it. These are the only facts with evidence, so they are the only facts
    the line is allowed to be read from."""
    return (fact.access_count or 0) > 0 or is_pinned(fact)


def score_fact(fact: "Fact", now: int) -> Salience:
    need = need_now(fact, now)
    truth = truth_log_odds(fact.confidence)
    stakes = fact.importance or 0.0
    return Salience(
        fact_id=fact.id, key=fact.key, source=fact.source or "",
        s=need + truth + stakes, need=need, truth=truth, stakes=stakes,
        anchor=is_anchor(fact), pinned=is_pinned(fact),
    )


def score_facts(facts: Iterable["Fact"], now: int) -> list[Salience]:
    return [score_fact(f, now) for f in facts]


def archive_line(scored: Sequence[Salience]) -> float | None:
    """The floor of the proven-useful set — None on a cold store (no anchors, no line,
    nothing archives). COMPUTED from the store every time; never configured."""
    anchors = [s.s for s in scored if s.anchor]
    return min(anchors) if anchors else None


def select_archivable(scored: Sequence[Salience], line: float | None) -> list[Salience]:
    """Everything strictly below the line, minus the owner's hand.

    Under `archive_line` the pin filter is redundant by construction (pins are anchors,
    and the line is the anchors' minimum, so no pin can sit below it). It is kept as a
    HARD FLOOR anyway because the line is an argument, not a constant: rung 4's
    calibrated cost-ratio line is not derived from the anchor set and could sit above an
    owner-touched fact. The owner's hand is ground truth; it does not depend on which
    line rule is in force.
    """
    if line is None:
        return []
    return [s for s in scored if s.s < line and not s.pinned]


def vacuum_warranted(moved: int, active_before: int) -> bool:
    """Whether a pass that moved `moved` of `active_before` rows should VACUUM.

    VACUUM rewrites the whole database: it COPIES what remains in order to RECLAIM what
    left. So the break-even is the comparison itself — when more rows left than stayed,
    the rewrite copies less than it frees; below that, the freed pages are cheaper left
    on SQLite's freelist for the next INSERT to reuse. There is no fraction to tune here
    and none is introduced: the bar is `moved > kept`, derived from the operation's own
    cost shape and scaling with the store instead of with taste.
    """
    kept = max(0, active_before - moved)
    return moved > 0 and moved > kept


def format_pass_line(archived: int, line: float | None) -> str:
    """The owner-visible report line for one pass."""
    if line is None:
        return f"archived {archived} dormant facts (line S=none — no recalled or owner-touched fact yet)"
    return f"archived {archived} dormant facts (line S={line:.2f})"
