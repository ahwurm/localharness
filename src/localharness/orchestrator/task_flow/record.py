"""The task flow's tables and its in-memory record: stages, limits and the revision rule."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple

from localharness.channels.base import CONTROL_CHARS_RE, INVISIBLE_CHARS_RE
from localharness.config.models import ChildBudget
from localharness.core.caps import Caps

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


SENSED = ("factual", "recency", "spec-compliance")
CRITICS_AT_FLOOR = frozenset({"jtbd", "answer"})
_CEILING = ChildBudget(actions=12, minutes=6)
_FLOOR = ChildBudget(actions=3, minutes=1)

LIMITS: dict[str, Limit] = {
    "critic": Limit(_FLOOR, _CEILING, _CEILING),
    "executor": Limit(_FLOOR, ChildBudget(actions=24, minutes=25), ChildBudget(actions=48, minutes=40)),
    "explorer": Limit(_FLOOR, ChildBudget(actions=12, minutes=6), ChildBudget(actions=24, minutes=10)),
    "additions": Limit(0, 0, 3),
    "flow_cap_minutes": Limit(1.0, 180.0, 360.0),
    **{f"revisions.{b}": Limit(0, row.revisions, 1 if b == "reply" else row.revisions + 2)
       for b, row in STAGES.items() if row.revisions},
}
REPLY_RESERVE_MINUTES = ((len(STAGES["reply"].floor) + len(SENSED) + LIMITS["additions"].ceiling)
                         * (LIMITS["revisions.reply"].ceiling + 1) * LIMITS["critic"].ceiling.minutes)
_NAME = re.compile(r"[a-z-]+")


@dataclass(frozen=True)
class Limits:
    executor: ChildBudget
    explorer: ChildBudget
    flow_cap_minutes: float
    additions: dict[str, tuple[str, ...]]
    revisions: dict[str, int]
    critics: dict[str, ChildBudget]

    def critic(self, name: str) -> ChildBudget:
        return self.critics.get(name, LIMITS["critic"].floor if name in CRITICS_AT_FLOOR else LIMITS["critic"].default)


def _clip(value: Any, low: Any, high: Any) -> Any:
    return min(max(value, low), high)


def _shown(b: Any) -> str:
    return f"({b.actions}, {b.minutes:g})"


def _refusal(name: Any, kept: list[str], floor: tuple[str, ...], known: Any) -> str | None:
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        return "is not a critic name"
    if name in floor or name in kept:
        return "is already on the panel"
    if known is not None and name not in known:
        return "is not a known critic"
    if len(kept) >= LIMITS["additions"].ceiling:
        return "is past the additions ceiling"
    return None


def limited(cfg: Any = None, known: Any = None) -> tuple[Limits, tuple[str, ...]]:
    notes: list[str] = []

    def note(key: str, what: str) -> None:
        notes.append(f"task_flow.{key}: {what}")

    def number(key: str, value: Any, limit: Limit) -> Any:
        out = _clip(value, limit.floor, limit.ceiling)
        if out != value:
            note(key, f"{value} is outside {limit.floor}..{limit.ceiling}; using {out}")
        return out

    def budget(key: str, value: Any, limit: Limit) -> ChildBudget:
        out = ChildBudget(actions=_clip(value.actions, limit.floor.actions, limit.ceiling.actions),
                          minutes=_clip(value.minutes, limit.floor.minutes, limit.ceiling.minutes))
        if (out.actions, out.minutes) != (value.actions, value.minutes):
            note(key, f"{_shown(value)} is outside {_shown(limit.floor)}..{_shown(limit.ceiling)}; using {_shown(out)}")
        return out

    def given(name: str) -> Any:
        return getattr(cfg, name, None)

    own = {name: budget(name, given(name) or LIMITS[name].default, LIMITS[name]) for name in ("executor", "explorer")}
    cap = given("flow_cap_minutes")
    flow = LIMITS["flow_cap_minutes"].default if cap is None else number("flow_cap_minutes", cap, LIMITS["flow_cap_minutes"])
    revisions = {k.removeprefix("revisions."): row.default for k, row in LIMITS.items() if k.startswith("revisions.")}
    for boundary, value in (given("revisions") or {}).items():
        row = LIMITS.get(f"revisions.{boundary}")
        if row is None:
            note(f"revisions.{boundary}", "has no revisions to configure; dropped")
        else:
            revisions[boundary] = number(f"revisions.{boundary}", value, row)
    critics: dict[str, ChildBudget] = {}
    for name, value in (given("critics") or {}).items():
        if not _NAME.fullmatch(name) or (known is not None and name not in known):
            note(f"critics.{name}", "is not a known critic; dropped")
        else:
            critics[name] = budget(f"critics.{name}", value, LIMITS["critic"])
    additions: dict[str, tuple[str, ...]] = {}
    for boundary, names in (given("panels") or {}).items():
        if boundary not in STAGES:
            note(f"panels.{boundary}", "is not a boundary; dropped")
            continue
        kept: list[str] = []
        for name in names:
            reason = _refusal(name, kept, STAGES[boundary].floor, known)
            if reason:
                note(f"panels.{boundary}", f"{name!r} {reason}; dropped")
            else:
                kept.append(name)
        additions[boundary] = tuple(kept)
    return Limits(own["executor"], own["explorer"], flow, additions, revisions, critics), tuple(notes)


def panel_critics(boundary: str, limits: Limits, sensed: tuple[str, ...] = ()) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*STAGES[boundary].floor, *sensed, *limits.additions.get(boundary, ()))))


def panel_minutes(critics: tuple[str, ...], limits: Limits | None = None) -> float:
    limits = limits or limited()[0]
    return sum(limits.critic(name).minutes for name in critics)


class Unit(NamedTuple):
    boundary: str
    kind: str
    step: int | None = None


def unit_minutes(unit: Unit, limits: Limits, sensed: tuple[str, ...] = ()) -> float:
    if unit.kind == "explore":
        return limits.explorer.minutes
    if unit.kind in ("execute", "revise"):
        return limits.executor.minutes
    panels = 1 + limits.revisions["reply"] if unit.boundary == "reply" else 1
    return panels * panel_minutes(panel_critics(unit.boundary, limits, sensed), limits)


def printed_bound(minutes: float, remainder_seconds: float | None = None) -> int:
    seconds = minutes * 60 if remainder_seconds is None else min(minutes * 60, remainder_seconds)
    return math.floor(seconds / 60)


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
DOCS = {"new": "decisions.md", "research": "research.md", "plan": "plan.md", "task": "plan.md"}
STAGE_WRITES = {"opened": "new", "research": "research", "plan": "plan", "execution": "run",
                "task panel": "task", "task pass": "done", "plan accepted": "done"}
_REVIEWED = re.compile(r"(decisions|research|plan|step-[1-9][0-9]*)\.md")


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


def _row_key(boundary: str, step: int | None) -> tuple[str, str | None]:
    return ("plan.md", f"step-{step}") if boundary == "step" else (DOCS[boundary], None)


def can_revise(record: Record, boundary: str, limit: int, step: int | None = None) -> bool:
    row, counts = record.filed.get(_row_key(boundary, step)), counts_of(record, boundary, step)
    return bool(row and row.verdict == "blocked" and counts and counts.blockers is not None
                and continue_revising(counts.blockers, counts.used, limit, counts.previous))


def reopen(record: Record, boundary: str, blockers: int | None, step: int | None = None) -> None:
    counts = counts_of(record, boundary, step)
    if counts is not None:
        counts.used, counts.previous, counts.blockers, counts.revised = 0, None, blockers, False


def set_stage(record: Record, event: str) -> None:
    if event not in STAGE_WRITES:
        raise ValueError(f"unknown stage event: {event}")
    stays = (event == "research" and ("plan.md", None) in record.filed) or (event == "execution" and record.stage == "run")
    if not stays:
        record.stage = STAGE_WRITES[event]


def stop_step(record: Record, n: int, log: str) -> None:
    record.filed.pop((f"step-{n}.md", None), None)
    record.stopped_log = log[:Caps.BLOCK]


def consume_flag(record: Record, boundary: str) -> bool:
    flagged = record.flags.get(boundary, False)
    record.flags[boundary] = False
    return flagged


def clear_turn(record: Record) -> None:
    record.flags = dict.fromkeys(record.flags, False)
    record.halted = None


def pointed(record: Record) -> str:
    if record.stage != "run":
        return POINTS[record.stage]
    steps = _sections(record)
    if not steps:
        return "plan.md"
    pending = [n for n in steps if record.filed[("plan.md", f"step-{n}")].verdict not in ("pass", "accepted")]
    return f"step-{pending[0] if pending else steps[-1]}.md"


def review_of(record: Record, doc: str) -> str | None:
    if (doc, None) not in record.filed or not _REVIEWED.fullmatch(doc):
        return None
    if doc == "plan.md" and record.stage in ("task", "done") and ("review.md", None) in record.filed:
        return "review.md"
    return f"{doc.removesuffix('.md')}-review.md"


class Line(NamedTuple):
    state: str
    commands: tuple[str, ...] = ()
    tail: str = ""
    model: str = ""


_BOUND = "up to <bound> min"
_STOP = ("<stop>",)

LINES: dict[str, Line] = {
    "armed": Line("<verb> armed", _STOP, "cap <n> min"),
    "running": Line("<unit> running", _STOP, _BOUND),
    "stopping": Line("stopping", ("<cancel>",), f"after this unit · {_BOUND}"),
    "stopping idle": Line("stopping", ("<cancel>",), "before the next task call"),
    "heard": Line("heard", _STOP, "after this unit"),
    "queued": Line("queued", _STOP, "after this turn · ! nudges"),
    "queued slash": Line("queued", _STOP, "after this turn · a command waits"),
    "queued phone": Line("queued", _STOP, "after this turn"),
    "decisions filed": Line("decisions filed", (), "from <n> turns"),
    "pass": Line("<stage> <verdict>", ("<next>",), _BOUND),
    "reopened": Line("<stage> reopened", ("<cancel>",)),
    "blocked": Line("<stage> blocked", ("/task <verb> (reopens)", "show <doc>", "accept"), "<n> blockers"),
    "unreviewed": Line("<stage> unreviewed", ("/task <verb> (re-panels)", "<next>"), _BOUND,
                       "panel unfinished; the person re-runs it"),
    "interview open": Line("interview open", ("/task plan (skips the questions)",)),
    "research blocked": Line("research blocked", ("/task plan (plans anyway)", "show research"), "<n> blockers"),
    "research unreviewed": Line("research unreviewed", ("/task research (re-runs)", "plan"), _BOUND),
    "plan unreviewed": Line("plan unreviewed", ("/task plan (re-panels)",), _BOUND),
    "review blocked": Line("review blocked", ("/task plan", "show review", "accept (ends tN)"), "<n> blockers",
                           "the task review blocked the plan; call stage=plan for the findings, then revise it"),
    "review unreviewed": Line("review unreviewed", ("/task run (re-runs the task panel)",), _BOUND),
    "step pass": Line("step N/M pass", _STOP, _BOUND),
    "step pass ended": Line("step N/M pass", ("/task run (next step)",), _BOUND),
    "step pass last": Line("step N/M pass", ("/task run (task review)",), _BOUND),
    "step blocked": Line("step N/M blocked", ("/task run (retries)", "show", "accept"), "<n> blockers"),
    "step unreviewed": Line("step N/M unreviewed", ("/task run (panels it)",), _BOUND),
    "step accepted": Line("step N/M accepted", ("/task run",), _BOUND),
    "step stopped": Line("step N/M stopped, files half-edited", ("/task run (re-runs)",), _BOUND),
    "halted": Line("halted", ("<next> (continues)",), "after <unit> · clean", "halted by the person"),
    "halted cap": Line("halted", ("<next> (continues)",), "cap <n> min hit · clean",
                       "the flow cap is spent for this turn; the person continues"),
    "not started": Line("<stage> not started", ("/task <verb>",), "the model answered instead"),
    "done": Line("done", ("/task show review",)),
    "closed": Line("closed", ("/task show tN/<doc>",), "not resumable, folder kept"),
    "paused": Line("paused", ("<next>", "clear (deletes the folder)")),
    "run needs a plan": Line("run needs a plan", ("/task plan",), "",
                             "run needs a passed or accepted plan; call stage=plan"),
    "decisions unfiled": Line("decisions unfiled", (), "", "the interview is open; ask, or file stage=new first"),
    "decisions blocked": Line("decisions blocked", ("/task new (reopens)", "accept"), "",
                              "decisions.md is blocked; the person reopens or accepts it"),
    "person decides": Line("person decides", (), "", "<doc> is blocked, <why>; the person decides"),
    "nothing blocked": Line("nothing blocked", ("/task show",)),
    "nothing to revise": Line("nothing to revise", ("/task plan <change>",)),
    "research full": Line("research full", (), "", f"{Caps.QUESTIONS} questions are filed; call stage=research to review them"),
    "nothing to review": Line("nothing to review", (), "", "no unreviewed answer; next: plan"),
    "no task": Line("no task", ("/task new <objective>",), "", "no task is open; the person opens one with /task new"),
    "open and not blocked": Line("open and not blocked", ("/task plan (revises)", "new <objective> (closes tN)")),
    "step not executed": Line("step N not executed", ("/task run",)),
    "step not in the plan": Line("step N not in the plan", ("/task show plan",)),
    "research closed": Line("research closed", ("/task show research", "new <objective> (closes tN)"), "",
                            "plan.md is filed; research is closed"),
    "done refused": Line("done", ("/task show review", "new <objective>"), "",
                         "the task is done; the person starts the next"),
    "not adopted": Line("not adopted", ("/task show tN/<doc>",)),
    "not the open task": Line("not the open task", ("/task show tN/<doc>",)),
    "not a task doc": Line("not a task doc", ("/task show <doc>",)),
    "edited since filed": Line("<doc> edited since filed", ("/task accept <doc> (uses your edit)",), "",
                               "the doc changed after it was filed; the person accepts it"),
}


@dataclass(frozen=True)
class Need:
    commands: tuple[str, ...]
    reason: str
    model: str = ""


def _fill(text: str, fields: dict[str, Any]) -> str:
    for name, value in fields.items():
        text = text.replace(f"<{name}>", str(value))
    return text.replace("tN", str(fields["task"])) if "task" in fields else text


def need(row: str, **fields: Any) -> Need:
    line = LINES[row]
    return Need(tuple(_fill(c, fields) for c in line.commands), _fill(line.state, fields), _fill(line.model, fields))


_PASSED = ("pass", "accepted")
_STEP_DOC = re.compile(r"step-([1-9][0-9]*)\.md")
DOC_NAME = re.compile(r"(decisions|research|plan|review|reply|job|step-[1-9][0-9]*)")
PLAN_KEY = ("plan.md", None)


def _sections(record: Record) -> list[int]:
    return sorted(int(k[1].removeprefix("step-")) for k in record.filed if k[0] == "plan.md" and k[1])


def _executed(record: Record) -> set[int]:
    return {int(m[1]) for k in record.filed if (m := _STEP_DOC.fullmatch(k[0]))}


def _next_step(record: Record) -> int | None:
    return next((n for n in _sections(record) if record.filed[("plan.md", f"step-{n}")].verdict not in _PASSED), None)


def boundary_of(stage: str, record: Record | None) -> str:
    if stage != "run":
        return stage
    rows = [record.filed.get(("plan.md", s)) for s in (None, *(f"step-{n}" for n in _sections(record)))] if record else []
    return "task" if len(rows) > 1 and all(r and r.verdict in _PASSED for r in rows) else "step"


def valid_question(q: object) -> bool:
    return (isinstance(q, str) and 0 < len(q) <= Caps.LINE and not any(c in q for c in "\n\r\t")
            and not CONTROL_CHARS_RE.search(q) and not INVISIBLE_CHARS_RE.search(q))


def doc_key(name: str, record: Record) -> tuple[str, str | None] | None:
    key = (f"{name}.md", None)
    return key if DOC_NAME.fullmatch(name) and key in record.filed else None


def precondition(boundary: str, record: Record | None, by: str, carried: str | None) -> Need | None:
    if record is None:
        return None if boundary == "new" and by == "person" and carried else need("no task")
    if record.stage == "done":
        return None
    filed = record.filed
    if boundary == "new":
        return need("open and not blocked", task=f"t{record.id}") if by == "person" and carried is None and PLAN_KEY in filed else None
    if boundary in ("research", "plan"):
        decisions = filed.get(("decisions.md", None))
        if boundary == "research" and _executed(record):
            return need("research closed", task=f"t{record.id}")
        if decisions is None and by == "model":
            return need("decisions unfiled")
        if decisions and decisions.verdict == "blocked":
            return need("decisions blocked")
        if boundary == "research" and carried is not None and len(record.questions) >= Caps.QUESTIONS:
            return need("research full")
    if boundary in ("step", "task"):
        plan = filed.get(PLAN_KEY)
        if plan is None:
            return need("run needs a plan")
        if plan.verdict == "blocked":
            return need("review blocked", task=f"t{record.id}") if record.stage == "task" else need("blocked", stage="plan", verb="plan", doc="plan")
        if not _sections(record) or _executed(record) - set(_sections(record)):
            return need("edited since filed", doc="plan.md")
    return None


def _file(record: Record, boundary: str, carried: str | None, limits: Limits) -> bool:
    row = record.filed.get(_row_key(boundary, None))
    return carried is not None and bool(row is None or record.flags.get(boundary)
                                        or can_revise(record, boundary, limits.revisions.get(boundary, 0)))


def _explore(record: Record, boundary: str, carried: str | None, limits: Limits) -> bool:
    return carried is not None


def _execute(record: Record, boundary: str, carried: str | None, limits: Limits) -> int | None:
    n = _next_step(record)
    return n if n and (f"step-{n}.md", None) not in record.filed else None


def _revise(record: Record, boundary: str, carried: str | None, limits: Limits) -> int | None:
    n = _next_step(record)
    return n if n and can_revise(record, "step", limits.revisions["step"], n) else None


def _panel(record: Record, boundary: str, carried: str | None, limits: Limits) -> int | bool | None:
    n = _next_step(record) if boundary == "step" else None
    row = record.filed.get(_row_key(boundary, n))
    return (n or True) if row and row.verdict == "unreviewed" and record.flags.get(boundary, True) else None


PREDICATES: dict[str, Callable[[Record, str, str | None, Limits], int | bool | None]] = {
    "file": _file,
    "explore": _explore,
    "execute": _execute,
    "revise": _revise,
    "panel": _panel,
}


def _closed(record: Record) -> Need | None:
    if record.stage == "done":
        return need("done refused")
    if record.halted:
        return need("halted" if record.halted == "stopped" else "halted cap")
    return None


def _decides(record: Record, boundary: str, limits: Limits, n: int | None = None) -> Need | None:
    row, limit = record.filed.get(_row_key(boundary, n)), limits.revisions.get(boundary, 0)
    if not (limit and row and row.verdict == "blocked") or can_revise(record, boundary, limit, n):
        return None
    counts = counts_of(record, boundary, n)
    why = "revisions used up" if counts and counts.used >= limit else "blockers did not fall"
    return need("person decides", doc=f"step-{n}.md" if n else DOCS[boundary], why=why)


def next_unit(record: Record, boundary: str, carried: str | None = None, limits: Limits | None = None) -> Unit | Need | None:
    if closed := _closed(record):
        return closed
    limits = limits or limited()[0]
    if boundary in ("step", "task"):
        boundary, plan = boundary_of("run", record), record.filed.get(PLAN_KEY)
        if boundary == "task":
            return Unit("task", "panel")
        if plan and plan.verdict == "unreviewed":
            return Unit("plan", "panel")
        if plan and plan.verdict == "blocked":
            return _decides(record, "plan", limits) or Unit("plan", "file")
    n = _next_step(record) if boundary == "step" else None
    if decides := _decides(record, boundary, limits, n):
        return decides
    for kind, predicate in PREDICATES.items():
        if kind in STAGES[boundary].units and (hit := predicate(record, boundary, carried, limits)):
            return Unit(boundary, kind, None if hit is True else hit)
    row = record.filed.get(_row_key(boundary, n))
    if boundary == "research":
        return need("nothing to review")
    if carried is not None:
        return need("nothing to revise")
    return need("unreviewed", stage=boundary, verb=boundary) if row and row.verdict == "unreviewed" else None


def _words(unit: Unit) -> str:
    if unit.kind == "explore":
        return "question"
    return unit.kind if unit.step is None else f"step {unit.step}" + ("" if unit.kind == "execute" else f" {unit.kind}")


def next_line(record: Record, boundary: str, limits: Limits | None = None) -> str:
    row = record.filed.get(_row_key(boundary, None))
    if (out := _closed(record)) is None and row and row.verdict in ("pass", "accepted"):
        out = Unit(STAGES[boundary].next, STAGES[STAGES[boundary].next].units[0])
    out = out or next_unit(record, boundary, None, limits)
    if isinstance(out, Need):
        return f"next: {out.reason}" if out.reason in ("done", "halted") else out.model or out.reason
    return f"next: {STAGES[out.boundary].stage or 'run'} · {_words(out)}"
