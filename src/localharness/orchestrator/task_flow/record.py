"""The task flow's tables and its in-memory record: stages, limits and the revision rule."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

from localharness.config.models import ChildBudget

BOUNDARIES = ("reply", "new", "research", "plan", "step", "task")


class Stage(NamedTuple):
    stage: str | None
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    units: tuple[str, ...]
    floor: tuple[str, ...]
    revisions: int
    started_by: str
    next: str | None


STAGES: dict[str, Stage] = {
    "reply": Stage(None, ("draft",), (), ("panel", "revise", "panel"), ("answer", "jtbd"), 1, "seam", None),
    "new": Stage("new", ("turns", "instructions"), ("decisions.md",), ("file", "panel"), ("jtbd",), 1, "person", "plan"),
    "research": Stage("research", ("decisions.md",), ("research.md",), ("explore", "panel"),
                      ("jtbd", "factual"), 0, "root", "plan"),
    "plan": Stage("plan", ("decisions.md", "research.md"), ("plan.md",), ("file", "panel"),
                  ("jtbd", "big-picture", "parsimony"), 3, "root", "step"),
    "step": Stage("run", ("plan.md", "step-N.md"), ("step-N.md",), ("execute", "revise", "panel"),
                  ("spec-compliance",), 1, "root", "task"),
    "task": Stage(None, ("step-N.md",), ("review.md",), ("panel",), ("jtbd", "big-picture"), 0, "root", "done"),
}
TOOL_STAGES = tuple(row.stage for row in STAGES.values() if row.stage)


class Limit(NamedTuple):
    floor: Any
    default: Any
    ceiling: Any


LIMITS: dict[str, Limit] = {
    "critic": Limit(ChildBudget(actions=3, minutes=1), ChildBudget(actions=12, minutes=6),
                    ChildBudget(actions=12, minutes=6)),
}
CRITICS_AT_FLOOR = frozenset({"jtbd", "answer"})


def panel_minutes(critics: tuple[str, ...], limits: Any = None) -> float:
    return sum((LIMITS["critic"].floor if name in CRITICS_AT_FLOOR else LIMITS["critic"].default).minutes
               for name in critics)


@dataclass
class Counts:
    used: int = 0
    blockers: int | None = None
    previous: int | None = None
    revised: bool = False


@dataclass
class Row:
    digest: str
    verdict: str | None = None
    counts: Counts | None = None


Filed = dict[tuple[str, str | None], Row]


@dataclass
class Record:
    id: int
    objective: str
    turn: int
    stage: str = "new"
    filed: Filed = field(default_factory=dict)
    counts: dict[str, Counts] = field(default_factory=lambda: {"new": Counts(), "plan": Counts()})
    flags: dict[str, bool] = field(default_factory=lambda: {"new": False, "plan": False})
    halted: str | None = None
    stopped_log: str = ""
    questions: list[str] = field(default_factory=list)
    paused: bool = False


POINTS = {"new": "decisions.md", "research": "research.md", "plan": "plan.md", "task": "plan.md", "done": "review.md"}


def open_task(n: int, objective: str, turn: int) -> Record:
    return Record(n, objective, turn)


def counts_of(record: Record, boundary: str, step: int | None = None) -> Counts | None:
    if boundary in record.counts:
        return record.counts[boundary]
    row = record.filed.get(("plan.md", f"step-{step}")) if boundary == "step" else None
    return row.counts if row else None


def file_doc(record: Record, boundary: str, key: tuple[str, str | None], digest: str) -> None:
    old = record.filed.get(key)
    record.filed[key] = Row(digest, "unreviewed", old.counts if old else None)
    counts = counts_of(record, boundary)
    if counts and counts.blockers is not None:
        counts.revised = True


def set_verdict(record: Record, key: tuple[str, str | None], verdict: str) -> None:
    record.filed[key].verdict = verdict


def note_run(record: Record, boundary: str, blockers: int, step: int | None = None, counted: bool = True) -> None:
    counts = counts_of(record, boundary, step)
    if counts is None:
        return
    if counts.revised and counted:
        counts.used += 1
    counts.previous, counts.blockers, counts.revised = counts.blockers, blockers, False


def continue_revising(blockers: int, used: int, limit: int, previous: int | None) -> bool:
    return blockers > 0 and used < limit and (previous is None or blockers < previous)


def pointed(record: Record) -> str:
    return POINTS[record.stage]


def review_of(record: Record, doc: str) -> str | None:
    if (doc, None) not in record.filed:
        return None
    return f"{doc.removesuffix('.md')}-review.md"
