"""The task flow's renderer and the person's arming verbs: a LINES row printed through notice(), and /task new, research, plan and run."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from localharness.channels.base import notice, sanitize_for_display
from localharness.orchestrator.task_flow.filer import NO_REVIEW, Refused, _cut, digest, last_section, parse_plan, render_section, rerecord
from localharness.orchestrator.task_flow.panel import turns_block, typed_turns
from localharness.orchestrator.task_flow.record import (
    DOC_NAME, DOCS, LINES, PLAN_KEY, STAGES, Need, Record, Row, Unit, _PASSED, _executed, _fill, _next_step, _sections, counts_of,
    file_doc, following, need, next_unit, pointed, precondition, printed_bound, reopen, review_of, set_stage, unit_minutes)
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


_STEP = re.compile(r"step-([1-9][0-9]*)")
_FOLDER_ARG = re.compile(r"t([1-9][0-9]{0,8})")
SESSION_DOCS = {"reply": ("reply-review.md", NO_REVIEW), "job": ("jtbd.md", "no job yet")}
FOREIGN = "not adopted · not made by this session"
CLOSED = "closed · not resumable"


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


def _key_of(stem: str) -> tuple[str, str | None]:
    return ("plan.md", stem) if _STEP.fullmatch(stem) else PLAN_KEY if stem == "review" else (f"{stem}.md", None)


def _blockers(flow: Any, boundary: str, step: int | None = None) -> int:
    counts = counts_of(flow.record, boundary, step)
    if counts and counts.blockers is not None:
        return counts.blockers
    review = flow.docs.read(flow.record.id, "research-review.md") if boundary == "research" else None
    return last_section(review or "").count("- [BLOCKER]")


def _head(flow: Any, stem: str) -> str:
    """The doc's verdict header: `plan.md · blocked, 2 blockers`."""
    row, step = flow.record.filed[_key_of(stem)], _STEP.fullmatch(stem)
    boundary = "step" if step else {"decisions": "new", "research": "research"}.get(stem, "plan")
    verdict = row.verdict or "unreviewed"
    count = _put(", <n> blockers", {"n": _blockers(flow, boundary, step and int(step[1]))}) if verdict == "blocked" else ""
    return f"{stem}.md · {verdict}{count}"


def _page(head: str, body: str, findings: str, phone: bool) -> tuple[str, ...]:
    """The terminal prints the header, the doc, then the findings; the phone one notice with the findings first and the doc cut."""
    body, findings = sanitize_for_display(body), sanitize_for_display(findings)
    if phone:
        return ("\n".join(p for p in (head, findings, _cut(body)[0]) if p),)
    return tuple(p for p in (head, body, findings) if p)


def _parse(flow: Any, arg: str) -> tuple[str, int | None] | None:
    """`(stem, folder)` for a doc name or `tN/<doc>`, the open task's own folder as None; None when the name is not a task doc."""
    folder, slash, stem = arg.rpartition("/")
    found = _FOLDER_ARG.fullmatch(folder)
    if (slash and not found) or not DOC_NAME.fullmatch(stem):
        return None
    n = int(found[1]) if found else None
    return stem, None if flow.record and n == flow.record.id else n


def _alone(flow: Any, n: int, stem: str, phone: bool) -> tuple[str, ...]:
    """A folder's doc printed with no row behind it, never trusted and never adopted."""
    head = f"t{n}/{stem}.md · {CLOSED if n in flow.docs.adopted else FOREIGN}"
    return _page(head, flow.docs.read(n, f"{stem}.md") or "", "", phone)


def _absent(flow: Any, stem: str, levers: Levers) -> str | None:
    """The line for a doc or step with no row to read or accept, else None."""
    record, step, task = flow.record, _STEP.fullmatch(stem), _label(flow.record)
    if step and ("plan.md", stem) not in record.filed:
        return render("step not in the plan", levers, task=task, step=int(step[1]))
    if (f"{stem}.md", None) in record.filed:
        return None
    return render("step not executed", levers, task=task, step=int(step[1])) if step else notice(task, f"{stem}.md not filed", ["/task show"])


def _header(flow: Any, stem: str) -> str:
    """The verdict header once the doc and its section row verify, else the mismatch line naming the one that does not."""
    record, name = flow.record, f"{stem}.md"
    keys = [(name, None), *([("plan.md", stem)] * bool(_STEP.fullmatch(stem)))]
    bad = next((k for k in keys if not flow.docs.verified(record, k)), None)
    return _refusal(record, need("edited since filed", doc=bad[0])) if bad else _head(flow, stem)


def _show_open(flow: Any, stem: str, levers: Levers, phone: bool) -> tuple[str, ...]:
    record, docs, step = flow.record, flow.docs, _STEP.fullmatch(stem)
    name = f"{stem}.md"
    stopped = step and (name, None) not in record.filed and record.stopped_log and int(step[1]) == _next_step(record)
    if stopped and (log := docs.read(record.id, name)) is not None:
        return _page(f"{name} · stopped · not executed", log, "", phone)
    if line := _absent(flow, stem, levers):
        return (line,)
    review = review_of(record, name)
    findings = last_section(docs.read(record.id, review) or "") if review else ""
    return _page(_header(flow, stem), docs.read(record.id, name) or "", findings, phone)


def _show(flow: Any, arg: str, levers: Levers, phone: bool) -> tuple[str, ...]:
    record, docs = flow.record, flow.docs
    if not arg:
        if record:
            if record.stage == "new" and DECISIONS not in record.filed:
                return (notice(_label(record), "interview in progress", ["/task plan (skips the questions)"]),)
            return _show_open(flow, pointed(record).removesuffix(".md"), levers, phone)
        n = docs.highest()
        names = docs.names(n) if n else []
        return _alone(flow, n, names[-1], phone) if names else (render("no task", levers),)
    target = _parse(flow, arg)
    refused = (render("not a task doc", levers, task=_label(record)),)
    if target is None:
        return refused
    stem, n = target
    if n is not None:
        return refused if stem in SESSION_DOCS or stem not in docs.names(n) else _alone(flow, n, stem, phone)
    if stem in SESSION_DOCS:
        name, empty = SESSION_DOCS[stem]
        text = docs.read_session(name)
        return _page(name, text if text.strip() else empty, "", phone)
    return _show_open(flow, stem, levers, phone) if record else (render("no task", levers),)


def _ahead(flow: Any, boundary: str) -> tuple[str, int]:
    """The command that follows a settled doc and the bound of the unit it starts."""
    out = following(flow.record, boundary, flow.limits)
    return ("/task show", 0) if isinstance(out, Need) else (f"/task {STAGES[out.boundary].stage or 'run'}", bound(flow, out))


def _status(flow: Any) -> tuple[str, dict[str, Any]]:
    """The `LINES` row that says where the open task stands and the fields it needs."""
    record = flow.record
    fields: dict[str, Any] = {"task": f"t{record.id}"}
    if record.stage == "done":
        return "done", fields
    steps, plan = _sections(record), record.filed[PLAN_KEY] if PLAN_KEY in record.filed else None
    if record.stage == "run" and steps and plan.verdict != "unreviewed":
        n = int(_STEP.fullmatch(pointed(record).removesuffix(".md"))[1])
        ran, verdict = (f"step-{n}.md", None) in record.filed, record.filed[("plan.md", f"step-{n}")].verdict
        key = ("step stopped" if record.stopped_log else "step not executed") if not ran else {
            "blocked": "step blocked", "accepted": "step accepted", "pass": "step pass last"}.get(verdict, "step unreviewed")
        again = key in ("step stopped", "step unreviewed")
        minutes = bound(flow, Unit("step", "panel" if key == "step unreviewed" else "execute", n)) if again else _ahead(flow, "step")[1]
        return key, {**fields, "step": (n, len(steps)), "n": _blockers(flow, "step", n), "bound": minutes}
    doc = "plan" if record.stage == "run" else pointed(record).removesuffix(".md")
    boundary = "new" if doc == "decisions" else doc
    row = record.filed.get(_key_of(doc))
    fields |= {"stage": boundary, "verb": boundary, "doc": doc}
    if row is None:
        return "interview open", fields
    task = record.stage == "task"
    if row.verdict == "blocked":
        key = "review blocked" if task else "research blocked" if boundary == "research" else "blocked"
        return key, {**fields, "n": _blockers(flow, boundary)}
    if row.verdict == "unreviewed" or (task and _next_step(record) is None):
        key = "review unreviewed" if task else "unreviewed" if boundary == "new" else f"{boundary} unreviewed"
        return key, {**fields, "next": STAGES[boundary].next, "bound": bound(flow, Unit("task" if task else boundary, "panel"))}
    command, minutes = _ahead(flow, boundary)
    decisions = record.filed.get(("decisions.md", None))
    return "pass", {**fields, "verdict": row.verdict, "next": command, "bound": minutes, "reopened": record.reopened,
                    "decisions_unreviewed": bool(decisions and decisions.verdict == "unreviewed")}


def status_row(flow: Any, levers: Levers) -> str:
    key, fields = _status(flow)
    return render(key, levers, **fields)


def _note(flow: Any, doc: str) -> None:
    """Files the person's accept into the review of `doc`, before any row changes."""
    record = flow.record
    if review := review_of(record, doc):
        record.filed[(review, None)] = Row(flow.docs.append_section(record.id, review, render_section(None, (), "accepted", 0)))


def _accepted(flow: Any, levers: Levers, stem: str, header: tuple[str, ...]) -> Reply:
    """The row of what the person just accepted; an accepted plan with every step passed ends a task under review."""
    record, step = flow.record, _STEP.fullmatch(stem)
    if stem == "plan" and record.stage == "task" and _next_step(record) is None:
        set_stage(record, "plan accepted")
    if step:
        line = render("step accepted", levers, task=_label(record), step=(int(step[1]), len(_sections(record))), bound=_ahead(flow, "step")[1])
    elif stem == "plan":
        line = status_row(flow, levers)
    else:
        boundary = "new" if stem == "decisions" else stem
        command, minutes = _ahead(flow, boundary)
        line = render("pass", levers, task=_label(record), stage=boundary, verdict="accepted", next=command, bound=minutes)
    return Reply((*header, line))


def _accept_bare(flow: Any, levers: Levers) -> Reply:
    record = flow.record
    stem = pointed(record).removesuffix(".md")
    row = record.filed.get(_key_of(stem))
    if not row or row.verdict != "blocked":
        return Reply((render("nothing blocked", levers, task=_label(record)),))
    _note(flow, f"{stem}.md")
    row.verdict = "accepted"
    return _accepted(flow, levers, stem, ())


def _accept(flow: Any, arg: str, levers: Levers) -> Reply:
    record, docs = flow.record, flow.docs
    target = _parse(flow, arg) if arg else ("", None)
    if target is None or target[0] in SESSION_DOCS:
        return Reply((render("not a task doc", levers, task=_label(record)),))
    stem, n = target
    if n is not None:
        return Reply((render("not the open task" if n in docs.adopted else "not adopted", levers, task=f"t{n}", doc=stem),))
    if record is None:
        return Reply((render("no task", levers),))
    if stem == "review":
        plan = record.filed.get(PLAN_KEY)
        return _accept_bare(flow, levers) if record.stage == "task" and plan and plan.verdict == "blocked" else Reply(
            (render("nothing blocked", levers, task=_label(record)),))
    if not stem:
        return _accept_bare(flow, levers)
    if line := _absent(flow, stem, levers):
        return Reply((line,))
    head, name, key, step = _header(flow, stem), f"{stem}.md", _key_of(stem), _STEP.fullmatch(stem)
    text = docs.read(record.id, name) or ""
    sections = parse_plan(text) if stem == "plan" else []
    if isinstance(sections, str):
        return Reply((head, notice(_label(record), sections, [])))
    _note(flow, name)
    if step:
        record.filed[(name, None)] = Row(digest(text))
        record.filed[key].verdict = "accepted"
    else:
        if stem == "plan":
            record.reopened = tuple(rerecord(record, sections))
        record.filed[key] = Row(digest(text), "accepted", record.filed[key].counts)
    return _accepted(flow, levers, stem, (head,))


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
    if boundary is None and verb not in ("show", "accept"):
        return Reply((USAGE,))
    try:
        if verb == "show":
            return Reply(_show(flow, arg, levers, phone))
        if verb == "accept":
            return _accept(flow, arg, levers)
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
