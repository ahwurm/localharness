"""Salience — the one scoring function of the resonance rebuild (memory spec).

One currency for the whole store, everywhere it competes (loading, search ranking's
secondary axis, archival):

    S = need + truth + stakes

- **need** — ln(1 + standing): the fact's accumulated resonance-share mass from
  dreaming's replay of the event streams. This REPLACES the transitional clock
  scorer's ACT-R activation term: there is no age, no decay exponent, no wall-clock
  anywhere. A fact untouched by new experience loses ground only RELATIVELY, because
  others gain — forgetting is the shadow cast by learning; an idle store holds still.
- **truth** — the row's log-odds belief, moved only by evidence (birth = the
  writer's measured track record; confirmations add measured weights, contradictions
  subtract). Pre-v10 rows derive it from the stored confidence.
- **stakes** — declared importance, the one human input, as-is.

**The line.** S is an ordinal ranking, so no threshold is invented — the archive
line is read out of the store: the MINIMUM S over facts the store can PROVE were
worth keeping (ever recalled, or touched by the owner's own hand). Nothing below the
worst proven-useful fact has any evidence for it. Recomputed from the data every
pass, never configurable; a cold store (no proven-useful fact yet) has no line and
archives nothing. Junk that gets recalled only lowers the line — every failure
direction under-archives.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Iterable, Sequence

from localharness.memory.sqlite import (
    USER_EDIT_PROVENANCE_PREFIX,
    USER_FORGET_PROVENANCE_PREFIX,
    _logit,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from localharness.memory.sqlite import Fact

# The owner's own hand, in the shapes the store records it: the `remember` tool and a
# `user_edit@`/`user_forget@` provenance stamp (web page + `localharness memory`).
OWNER_HAND_SOURCES = frozenset({"remember"})
OWNER_HAND_PROVENANCE_PREFIXES = (
    USER_EDIT_PROVENANCE_PREFIX,
    USER_FORGET_PROVENANCE_PREFIX,
)


@dataclass(frozen=True)
class Salience:
    """One fact's score, with its three terms kept separate — a score you cannot take
    apart is a score you cannot debug."""
    fact_id: int
    key: str
    source: str
    s: float
    need: float
    truth: float
    stakes: float
    anchor: bool
    pinned: bool


def truth_log_odds(fact: "Fact") -> float:
    """The row's belief in log-odds: the v10 column when present, else derived from
    the legacy confidence (logit with the store's domain guard)."""
    if fact.truth_logodds is not None:
        return float(fact.truth_logodds)
    return _logit(fact.confidence)


def is_pinned(fact: "Fact") -> bool:
    """The owner's hand — exempt from archival at any score, forever."""
    if (fact.source or "") in OWNER_HAND_SOURCES:
        return True
    prov = fact.provenance or ""
    return any(p in prov for p in OWNER_HAND_PROVENANCE_PREFIXES)


def is_anchor(fact: "Fact") -> bool:
    """A fact the store can PROVE was worth keeping: it was actually recalled, or the
    owner touched it. Only facts with evidence may set the line."""
    return (fact.access_count or 0) > 0 or is_pinned(fact)


def score_fact(fact: "Fact") -> Salience:
    need = math.log1p(max(0.0, fact.standing or 0.0))
    truth = truth_log_odds(fact)
    stakes = fact.importance or 0.0
    return Salience(
        fact_id=fact.id, key=fact.key, source=fact.source or "",
        s=need + truth + stakes, need=need, truth=truth, stakes=stakes,
        anchor=is_anchor(fact), pinned=is_pinned(fact),
    )


def score_facts(
    facts: Iterable["Fact"], *, also_anchor: Iterable[int] = (),
) -> list[Salience]:
    """`also_anchor`: fact ids the CALLER knows carry evidence this projection cannot
    see — rows with UNFOLDED (staged) reads. A read is a read whether or not the
    dreaming fold has moved the counter yet; missing one archives MORE than the
    evidence supports, the one direction this design does not accept."""
    ids = set(also_anchor)
    return [replace(s, anchor=True) if s.fact_id in ids else s
            for s in (score_fact(f) for f in facts)]


def archive_line(scored: Sequence[Salience]) -> float | None:
    """The floor of the proven-useful set — None on a cold store (no anchors, no
    line, nothing archives). COMPUTED from the store every time; never configured."""
    anchors = [s.s for s in scored if s.anchor]
    return min(anchors) if anchors else None


def select_archivable(scored: Sequence[Salience], line: float | None) -> list[Salience]:
    """Everything strictly below the line, minus the owner's hand.

    The pin filter is redundant under `archive_line` (pins are anchors, and the line
    is the anchors' minimum) but kept as a HARD FLOOR: the line is an argument, not a
    constant, and a future calibrated line is not derived from the anchor set. The
    owner's hand is ground truth regardless of which line rule is in force."""
    if line is None:
        return []
    return [s for s in scored if s.s < line and not s.pinned]


def vacuum_warranted(moved: int, active_before: int) -> bool:
    """VACUUM copies what remains to reclaim what left, so the break-even is the
    comparison itself: rewrite only when more rows left than stayed. No fraction to
    tune; the bar derives from the operation's own cost shape."""
    kept = max(0, active_before - moved)
    return moved > 0 and moved > kept


def format_pass_line(archived: int, line: float | None) -> str:
    """The owner-visible report line for one archival step."""
    if line is None:
        return f"archived {archived} dormant facts (line S=none — no recalled or owner-touched fact yet)"
    return f"archived {archived} dormant facts (line S={line:.2f})"
