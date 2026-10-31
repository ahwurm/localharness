"""The task flow's renderer and the person's arming verbs: a LINES row printed through notice(), and /task new, research, plan and run."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from localharness.channels.base import notice
from localharness.orchestrator.task_flow.filer import Refused
from localharness.orchestrator.task_flow.panel import turns_block, typed_turns
from localharness.orchestrator.task_flow.record import (
    DOC_NAME, DOCS, LINES, Need, Record, Unit, _PASSED, _executed, _fill, _next_step, _sections, counts_of, file_doc, next_unit,
    precondition, printed_bound, reopen, unit_minutes)
from localharness.orchestrator.task_flow.runner import INTERVIEW_SENTENCE

Levers = tuple[str, str]


def _put(text: str, fields: dict[str, Any]) -> str:
    step = fields.get("step")
    if step is not None:
        n, m = step if isinstance(step, tuple) else (step, "")
        text = text.replace("N/M", f"{n}/{m}").replace("step N", f"step {n}")
    if fields.get("n") == 1:
        text = text.replace("<n> blockers", "<n> blocker").replace("<n> turns", "<n> turn")
    return _fill(text, fields)


def _ranges(steps: tuple[int, ...]) -> str:
    runs: list[list[int]] = []
    for n in sorted(steps):
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    return ("steps " if len(steps) > 1 else "step ") + ", ".join(f"{a}-{b}" if b > a else str(a) for a, b in runs)


def render(key: str, levers: Levers, **fields: Any) -> str:
    """The `LINES` row as the finished line a channel prints, or empty for a row only the model reads."""
    line = LINES[key]
    if line.model and not line.commands:
        return ""
    fields = {"stop": levers[0], "cancel": levers[1], **fields}
    state, commands = line.state, list(line.commands)
    if key == "pass" and fields.get("stage") == "plan":
        state += f", {_ranges(fields['reopened'])} reopened" if fields.get("reopened") else ""
        commands += ["plan <change> (revises)", *(["accept decisions (unreviewed)"] * bool(fields.get("decisions_unreviewed")))]
    return notice(fields.get("task", "task"), _put(state, fields), [_put(c, fields) for c in commands], _put(line.tail, fields))


def bound(flow: Any, unit: Unit, sensed: tuple[str, ...] = ()) -> int:
    return printed_bound(unit_minutes(unit, flow.limits, sensed), flow.cap.remaining())


def cap_minutes(flow: Any) -> int:
    return printed_bound(flow.limits.flow_cap_minutes, flow.cap.unspent())


def progress(flow: Any, subject: str, state: str, commands: tuple[str, ...], tail: str, levers: Levers) -> str:
    """The emit sink's line: the levers filled in, and on the turn's first `<stop>` row `typing waits` and the cap said once."""
    spoken: list[str] = []
    if "<stop>" in commands:
        armed = any(c.startswith("cap ") for c in tail.split(" · "))
        if not armed and "typing" not in flow.said:
            spoken.append("typing waits")
        if not armed and "cap" not in flow.said:
            spoken.append(f"cap {cap_minutes(flow)} min")
        flow.said |= {"cap"} if armed else {"typing", "cap"}
    fill = {"stop": levers[0], "cancel": levers[1]}
    return notice(subject, state, [_fill(c, fill) for c in commands], " · ".join(p for p in (*spoken, tail) if p))



DECISIONS = ("decisions.md", None)
STALE = " edited since filed"
USAGE = "usage: /task new [objective] · research · plan [change] · run · show [doc] · accept [doc] · stop · clear"
BOUNDARIES = {"new": "new", "research": "research", "plan": "plan", "run": "step"}
SENTENCES = {
    "research": ('Call task(stage="research", question=...) once per question the plan needs answered from the '
                 'repository, then task(stage="research") to review the answers.'),
    "new unfiled": INTERVIEW_SENTENCE,
    "new blocked": 'Call task(stage="new") to read the findings, then call it again with doc= your revised decisions.',
    "new unreviewed": 'Call task(stage="new").',
    "new reopened": ('Call task(stage="new") to read the decisions, then call it again with doc= the decisions '
                     "revised as the person's messages since they were filed ask."),
    "plan unfiled": ('Call task(stage="plan") to read the decisions and the plan\'s shape, '
                     'then call it again with doc= your plan.'),
    "plan unreviewed": 'Call task(stage="plan").',
    "plan blocked": 'Call task(stage="plan") to read the findings, then call it again with doc= your revised plan.',
    "plan reopened": ('Call task(stage="plan") to read the plan, then call it again with doc= the plan revised '
                      "as the person's messages since it was filed ask."),
    "run": 'Call task(stage="run") and call it again after each result until it says done, blocked or halted.',
}


@dataclass(frozen=True)
class Reply:
    """Finished lines to print and the follow-up turn as (text, subtype), or none."""

    lines: tuple[str, ...] = ()
    turn: tuple[str, str] | None = None


def sentence(boundary: str, record: Record) -> str:
    """The stage sentence a verb-started turn carries, chosen from the record."""
    if boundary in ("research", "step"):
        return SENTENCES["research" if boundary == "research" else "run"]
    row = record.filed.get((DOCS[boundary], None))
    state = "unfiled" if row is None else row.verdict if row.verdict in ("blocked", "unreviewed") else "reopened"
    return SENTENCES[f"{boundary} {state}"]


def skip_interview(flow: Any) -> int:
    """Files decisions.md from the objective and the person's turns since the task opened; no panel runs. Returns the turns used."""
    record = flow.record
    turns = [(n, t) for n, t in typed_turns(flow.messages()) if n >= record.turn and t != record.objective]
    text = "\n".join(filter(None, (f"{record.objective} (turn {record.turn})", turns_block(turns))))
    file_doc(record, "new", DECISIONS, flow.docs.write(record.id, "decisions.md", text)[0])
    record.filed_turn["new"] = flow.turn()
    return len(turns) + 1


def _label(record: Record | None) -> str:
    return f"t{record.id}" if record else "task"


def _refusal(record: Record | None, refused: Need) -> str:
    commands, tail = list(refused.commands), ""
    if refused.reason in ("plan blocked", "review blocked"):
        tail = _put("<n> blockers", {"n": counts_of(record, "plan").blockers})
    elif refused.reason == "research closed" and ("research.md", None) not in record.filed:
        commands[0] = "/task show plan"
    elif refused.reason.endswith(STALE):
        doc = refused.reason.removesuffix(STALE)
        stem = doc.removesuffix(".md")
        commands = [c.replace(doc, stem) for c in commands] if DOC_NAME.fullmatch(stem) else []
    return notice(_label(record), refused.reason, commands, tail)


def _newest(record: Record) -> str | None:
    names = ["decisions", "research", "plan", *(f"step-{n}" for n in sorted(_executed(record))), "review"]
    return next((n for n in reversed(names) if (f"{n}.md", None) in record.filed), None)


def _open(flow: Any, objective: str, turn: int, levers: Levers) -> Reply:
    old, lines = flow.record, []
    if old and old.stage != "done":
        doc = _newest(old)
        lines.append(render("closed", levers, task=_label(old), doc=doc) if doc
                     else notice(_label(old), "closed", [], LINES["closed"].tail))
    record = flow.open(objective, turn)
    return Reply((*lines, render("interview open", levers, task=_label(record))), (objective, "typed"))


def _typed_since_filing(flow: Any, boundary: str) -> bool:
    filed = flow.record.filed_turn.get(boundary, 0)
    return any(n > filed for n, _ in typed_turns(flow.messages()))


def _reopen(flow: Any, boundary: str, arg: str, levers: Levers) -> list[str] | None:
    """Reopens the boundary's doc by its verdict, a blocked one first; None when a passed plan has nothing to revise."""
    record = flow.record
    row = record.filed.get((DOCS[boundary], None))
    if row is None:
        return [render("interview open", levers, task=_label(record))] if boundary == "new" else []
    if row.verdict == "blocked":
        reopen(record, boundary, counts_of(record, boundary).blockers)
        return []
    if row.verdict in _PASSED and boundary == "plan" and not arg and not _typed_since_filing(flow, boundary):
        return None
    record.flags[boundary] = True
    return [render("reopened", levers, task=_label(record), stage=boundary)]


def _steer(flow: Any, verb: str, boundary: str, arg: str, levers: Levers) -> Reply:
    record, lines = flow.record, []
    label = _label(record)
    if boundary in ("research", "plan") and record.stage == "new" and DECISIONS not in record.filed:
        lines.append(render("decisions filed", levers, task=label, n=skip_interview(flow)))
    if boundary == "step":
        unit = next_unit(record, "step", None, flow.limits)
        if isinstance(unit, Need):
            n = _next_step(record)
            blocked = unit.reason == "person decides" and n
            return Reply((render("step blocked", levers, task=label, step=(n, len(_sections(record))),
                                 n=counts_of(record, "step", n).blockers) if blocked else _refusal(record, unit),))
    elif (reopened := _reopen(flow, boundary, arg, levers)) is None:
        return Reply((render("nothing to revise", levers, task=label),))
    else:
        lines += reopened
    said, change = sentence(boundary, record), bool(arg) and boundary == "plan"
    flow.sentence = said if change else ""
    record.paused, flow.armed = False, boundary
    flow.said.add("cap")
    lines.append(render("armed", levers, task=label, verb=verb, n=cap_minutes(flow)))
    return Reply(tuple(lines), (arg, "typed") if change else (said, "request"))


def task_command(flow: Any, args: str, *, turn: int, running: bool, levers: Levers, phone: bool = False) -> Reply:
    """The person's /task verb as finished lines and the follow-up turn; synchronous, so no call lands between a check and a set."""
    parts = args.split(maxsplit=1) or [""]
    verb, arg = parts[0].lower(), "".join(parts[1:]).strip()
    boundary, record = BOUNDARIES.get(verb), flow.record
    if boundary is None:
        return Reply((USAGE,))
    try:
        if verb == "new" and arg:
            return _open(flow, arg, turn, levers)
        if record and record.stage == "done":
            return Reply((render("done refused", levers, task=_label(record)),))
        refused = precondition(boundary, record, "person", None)
        if refused:
            return Reply((_refusal(record, refused),))
        return _steer(flow, verb, boundary, arg, levers)
    except Refused as e:
        return Reply((notice(_label(record), str(e), []),))
