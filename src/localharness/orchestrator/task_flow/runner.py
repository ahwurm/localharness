"""The task flow's runner: one unit per call, the executor's config and the two request-note providers."""
from __future__ import annotations

import asyncio
import math
import re
from contextlib import suppress
from time import monotonic
from typing import TYPE_CHECKING, Any, Callable

from localharness.agent.context import (
    COMPACT_DISABLED, CompactionPipeline, TokenCounter, make_compaction_summarize_fn, summarizer_input_chars)
from localharness.agent.subagent import EXPLORE_TOOLS, _config_child_allowed
from localharness.channels.base import one_line
from localharness.config.models import AgentConfig, BudgetConfig, PermissionConfig, ToolConfig
from localharness.core.caps import Caps
from localharness.core.events import CompactionTriggered, Observation
from localharness.orchestrator.task_flow.filer import (
    Refused, StepLog, answered, file_answer, last_section, manifest, parse_plan, render_section, rerecord)
from localharness.orchestrator.task_flow.panel import Finding, PanelInputs, PanelResult, PanelRun, fence, typed_turns, unit_floor
from localharness.orchestrator.task_flow.record import (
    DOCS, LIMITS, LINES, PLAN_KEY, STAGES, Need, Record, Row, Unit, boundary_of, can_revise, consume_flag, file_doc, following,
    need, next_line, next_unit, note_run, pointed, precondition, printed_bound, reopen, review_of, set_stage,
    set_verdict, stop_step, unit_minutes, valid_question)

if TYPE_CHECKING:
    from localharness.orchestrator.task_flow import TaskFlow

CHILD_SENTENCE = ("Say in one line what you will produce, read what it depends on before you change anything, "
                  "do the work, and write your doc as your final answer.")
INTERVIEW_SENTENCE = ("Ask the person what you need to know to write decisions.md, one question at a time, "
                      'or if the objective is clear call task(stage="new", doc=...).')
EXPLORER_ROLE = ("Answer the one question from the repository's files with read, glob and grep. Reply with one claim per "
                 "line, each with the file:line that shows it, and nothing else; reply none when the files do not answer it.")
EXECUTOR_ROLE = ("Carry out the one step you are briefed with. When you cannot proceed, take the safe option and write "
                 "why in your final answer; never ask the person.")
_RUNS = ("new", "research", "plan", "step", "task")
_FINDING = re.compile(r"- \[(BLOCKER|WARNING|NOTE)\] (.*?)(?: \(([a-z-]+)\))?")
_CAPPED = "the flow cap is spent for this turn; the person continues"


def _config(name: str, role: str, tools: ToolConfig) -> AgentConfig:
    return AgentConfig(name=name, role=role, tools=tools, permissions=PermissionConfig(budget=BudgetConfig()))


def explorer_config() -> AgentConfig:
    return _config("explorer", EXPLORER_ROLE, ToolConfig(add=list(EXPLORE_TOOLS)))


def build_executor_config(flow: TaskFlow):
    pipeline = CompactionPipeline(
        token_counter=flow.token_counter or TokenCounter(),
        llm_summarize_fn=make_compaction_summarize_fn(flow.llm, summarizer_input_chars(flow.config.context.max_context_tokens)),
        compact_md_path=COMPACT_DISABLED)
    config = _config("executor", EXECUTOR_ROLE, ToolConfig(add=list(flow.root_tools), deny=["tool_result_get"]))
    config.tools.add = _config_child_allowed(config)
    return config, pipeline, ("task_flow.executor", flow.limits.executor)


def executor_note(header: str) -> Callable[[], str]:
    text = f"{CHILD_SENTENCE}\n{one_line(header)[:Caps.LINE]}"
    return lambda: text


def _verdict(record: Record) -> str:
    doc = pointed(record)
    key = ("plan.md", doc.removesuffix(".md")) if doc.startswith("step-") else ("plan.md" if doc == "review.md" else doc, None)
    row = record.filed.get(key)
    return (row.verdict or "") if row else ""


def root_note(flow: TaskFlow) -> str:
    """The root's request note: one pointer line, the interview sentence while `new` has no doc, the pending revision block."""
    record = flow.record
    if record is None or record.paused:
        return ""
    boundary = boundary_of("run", record) if record.stage == "run" else "task" if record.stage == "done" else record.stage
    head = " ".join(p for p in (record.stage, _verdict(record)) if p)
    pointer = one_line(f"t{record.id} \u00b7 {head} \u00b7 {next_line(record, boundary, flow.limits)}")[:Caps.LINE]
    interview = INTERVIEW_SENTENCE if record.stage == "new" and ("decisions.md", None) not in record.filed else ""
    return "\n".join(p for p in (pointer, interview, flow.revision_block[:Caps.BLOCK]) if p)


def _exclude(flow: TaskFlow, seconds: float) -> None:
    """A call's wall time off the root's clock: non-finite or negative is zero, and a turn's total never passes the flow cap."""
    valid = seconds if math.isfinite(seconds) and seconds >= 0 else 0.0
    valid = min(valid, max(0.0, flow.limits.flow_cap_minutes * 60 - flow.excluded))
    flow.excluded += valid
    flow.exclude(valid)


def _text(refusal: Need) -> str:
    return refusal.model or refusal.reason


def _spent(flow: TaskFlow) -> str:
    flow.record.halted = "cap"
    return _CAPPED


def _key(unit: Unit) -> tuple[str, str | None]:
    return ("plan.md", f"step-{unit.step}") if unit.boundary == "step" else (DOCS[unit.boundary], None)


def _review_doc(unit: Unit) -> str:
    if unit.boundary == "step":
        return f"step-{unit.step}-review.md"
    return "review.md" if unit.boundary == "task" else DOCS[unit.boundary].replace(".md", "-review.md")


def _label(unit: Unit) -> str:
    return f"step {unit.step}" if unit.boundary == "step" else STAGES[unit.boundary].stage or "review"


def _running(flow: TaskFlow, unit_text: str, minutes: float) -> None:
    line = LINES["running"]
    bound = str(printed_bound(minutes, flow.cap.remaining()))
    flow.emit(f"t{flow.record.id}", line.state.replace("<unit>", unit_text), line.commands, line.tail.replace("<bound>", bound))


def _brief(flow: TaskFlow, unit: Unit) -> str:
    record = flow.record

    def here(doc: str) -> str:
        return f"{doc} ({flow.docs.path(record.id, doc)})"

    if unit.boundary == "step":
        return f"Review step {unit.step}: its section of {here('plan.md')} against {here(f'step-{unit.step}.md')}."
    steps = sum(1 for k in record.filed if k[0] == "plan.md" and k[1])
    reads = {"research": ["decisions.md"], "plan": ["decisions.md", "research.md"],
             "task": [f"step-{n}.md" for n in range(1, steps + 1)]}.get(unit.boundary, [])
    named = ", ".join(here(d) for d in reads if (d, None) in record.filed)
    return f"Review {here(DOCS[unit.boundary])}." + (f" Read also: {named}." if named else "")


def _ended(result: Any) -> str:
    """Why a child did not finish, or empty when it did."""
    return "" if result.status == "completed" else one_line(result.terminated_reason or result.status)[:Caps.LINE]


def _stale(flow: TaskFlow, *keys: tuple[str, str | None]) -> str | None:
    bad = next((k for k in keys if k in flow.record.filed and not flow.docs.verified(flow.record, k)), None)
    return _text(need("edited since filed", doc=bad[0])) if bad else None


def _findings(section: str) -> list[Finding]:
    return [Finding(m[1], m[2], "", False, m[3] or "") for line in section.splitlines() if (m := _FINDING.fullmatch(line))]


async def _panel(flow: TaskFlow, unit: Unit) -> PanelResult:
    record, key, task = flow.record, _key(unit), unit.boundary == "task"
    if task:
        set_stage(record, "task panel")

    def report(run: PanelRun) -> None:
        doc = _review_doc(unit)
        rounds = max(map(int, re.findall(r"(?m)^## round ([0-9]+)", flow.docs.read(record.id, doc) or "")), default=0) + 1
        digest = flow.docs.append_section(record.id, doc, render_section(rounds, run.findings, run.verdict, run.blockers))
        record.filed[(doc, None)] = Row(digest)
        settled = run.verdict in ("pass", "blocked")
        if task and run.verdict == "blocked":
            set_verdict(record, key, "blocked")
            reopen(record, "plan", run.blockers)
        elif task and settled:
            set_stage(record, "task pass")
        elif not task:
            set_verdict(record, key, run.verdict)
            if settled:
                note_run(record, unit.boundary, run.blockers, unit.step)
        if run.verdict == "pass" and "jtbd" in STAGES[unit.boundary].floor:
            flow.jtbd_stood = True

    _running(flow, f"{_label(unit)} panel", unit_minutes(unit, flow.limits))
    inputs = PanelInputs(_brief(flow, unit), turns=tuple(typed_turns(flow.messages())), on_run=report)
    return await flow.reply_panel.run(unit, inputs)


def _done(flow: TaskFlow, boundary: str, unit: Unit, filed=(), child: str = "", result: PanelResult | None = None,
          notes=()) -> str:
    record = flow.record
    nxt = following(record, boundary, flow.limits)
    if isinstance(nxt, Unit) and not record.halted and unit_floor(nxt, flow.limits) * 60 > flow.cap.remaining():
        record.halted = "cap"
    research = record.filed.get(("research.md", None))
    if unit.boundary == "plan" and research and research.verdict == "blocked":
        notes = [*notes, "research.md is blocked"]
    verdict = f"{_label(unit)} {result.verdict}" + (f" · {result.blockers} blockers" if result.verdict == "blocked" else "") if result else ""
    revising = (result and result.verdict == "blocked" and unit.boundary in ("new", "plan")
                and can_revise(record, unit.boundary, flow.limits.revisions[unit.boundary]))
    doc = flow.docs.read(record.id, DOCS[unit.boundary]) or "" if revising else ""
    findings = fence(result.findings) if result and result.findings else ""
    return manifest(filed, child, verdict, doc, findings, notes, next_line(record, boundary, flow.limits))


def _read(flow: TaskFlow, boundary: str) -> str:
    record, doc = flow.record, DOCS.get(boundary, "plan.md")
    if stale := _stale(flow, (doc, None)):
        return stale
    review = review_of(record, doc)
    section = last_section(flow.docs.read(record.id, review) or "") if review else last_section("")
    text = flow.docs.read(record.id, doc) or ""
    shown = fence((), text, Caps.DOC) if boundary == "research" and text else text
    return manifest((), doc=shown, findings=fence((), section), next_line=next_line(record, boundary, flow.limits))


async def _file(flow: TaskFlow, boundary: str, unit: Unit, carried: str | None) -> str:
    record, name = flow.record, DOCS[unit.boundary]
    if len(carried.encode()) > Caps.DOC:
        return f"{name} is over {Caps.DOC} bytes; file it shorter"
    sections = parse_plan(carried) if unit.boundary == "plan" else None
    if isinstance(sections, str):
        return sections
    digest, _ = flow.docs.write(record.id, name, carried)
    file_doc(record, unit.boundary, (name, None), digest)
    reopened = rerecord(record, sections) if sections else []
    if sections:
        set_stage(record, "plan")
    if consume_flag(record, unit.boundary):
        reopen(record, unit.boundary, None)
    notes = [f"steps {', '.join(map(str, reopened))} reopened"] if reopened else []
    result = await _panel(flow, Unit(unit.boundary, "panel"))
    return _done(flow, boundary, unit, (name, _review_doc(unit)), result=result, notes=notes)


async def _review(flow: TaskFlow, boundary: str, unit: Unit, carried: str | None) -> str:
    keys = [_key(unit)] + ([PLAN_KEY] if unit.boundary == "task" else [])
    if stale := _stale(flow, *keys):
        return stale
    result = await _panel(flow, unit)
    return _done(flow, boundary, unit, (_review_doc(unit),), result=result)


def _research_changed(flow: TaskFlow) -> list[str]:
    plan = flow.record.filed.get(PLAN_KEY)
    if plan and plan.verdict == "pass":
        plan.verdict = "unreviewed"
    return ["research changed"] if plan else []


async def _explore(flow: TaskFlow, boundary: str, unit: Unit, carried: str | None) -> str:
    record = flow.record
    if not valid_question(carried):
        return f"a question is one plain line of at most {Caps.LINE} characters"
    if stale := _stale(flow, ("research.md", None)):
        return stale
    n = len(record.questions) + 1
    _running(flow, f"research q{n}", flow.limits.explorer.minutes)
    granted = flow.cap.reserve(flow.limits.explorer, LIMITS["explorer"].floor.minutes)
    if granted is None:
        return _spent(flow)
    flow.reply_panel.set_phase(f"research q{n} · explorer · up to {printed_bound(granted.minutes)} min")
    started = monotonic()
    try:
        result = await flow.run_config_child(explorer_config(), f"{CHILD_SENTENCE}\nQuestion: {carried}",
                                             budget=("task_flow.explorer", granted))
    finally:
        flow.cap.settle(granted, monotonic() - started)
        flow.reply_panel.set_phase(None)
    ended = _ended(result)
    digest, _ = flow.docs.write(record.id, "research.md", file_answer(
        flow.docs.read(record.id, "research.md") or "", n, carried, "explorer", result.raw, ended))
    record.questions.append(carried)
    file_doc(record, "research", ("research.md", None), digest)
    set_stage(record, "research")
    child = f"q{n}: explorer " + (f"stopped ({ended})" if ended else "answered" if answered(result.raw) else "returned none")
    return _done(flow, boundary, unit, ("research.md",), child, notes=_research_changed(flow))


def _watch(bus: Any, log: StepLog, runtime: dict) -> list[Any]:
    if bus is None:
        return []

    def saw(event: Observation) -> None:
        if event.parent_id is not None:
            name = event.tool_name or event.observation_type
            runtime["tools"][name] = runtime["tools"].get(name, 0) + 1
            log.add(f"{name} failed: {event.error}" if event.error else name)

    def compacted(event: CompactionTriggered) -> None:
        runtime["compactions"] += event.parent_id is not None

    return [bus.subscribe(Observation, saw), bus.subscribe(CompactionTriggered, compacted)]


async def _execute(flow: TaskFlow, boundary: str, unit: Unit, carried: str | None) -> str:
    record, n = flow.record, unit.step
    key = ("plan.md", f"step-{n}")
    plan = parse_plan(flow.docs.read(record.id, "plan.md") or "")
    section = next((s for s in plan if s.n == n), None) if not isinstance(plan, str) else None
    deps = [(f"step-{d}.md", None) for d in section.depends_on] if section else []
    if stale := _stale(flow, key, *deps):
        return stale
    header = f"step {n}/{len(plan)} · {section.done}"
    parts = [CHILD_SENTENCE, f"Step {n} of {len(plan)}:", section.text,
             *(f"step-{d}.md (done before you):\n{flow.docs.read(record.id, f'step-{d}.md') or ''}" for d in section.depends_on)]
    revise = unit.kind == "revise"
    findings = _findings(last_section(flow.docs.read(record.id, f"step-{n}-review.md") or "")) if revise else []
    if findings or record.stopped_log:
        parts.append(fence(findings, record.stopped_log))
    config, pipeline, (row, budget) = build_executor_config(flow)
    _running(flow, f"step {n}" + (" revise" if revise else ""), budget.minutes)
    granted = flow.cap.reserve(budget, LIMITS["executor"].floor.minutes)
    if granted is None:
        return _spent(flow)
    if revise:
        record.filed[key].counts.revised = True
    set_stage(record, "execution")
    log, runtime = StepLog(), {"files": [], "tools": {}, "compactions": 0}
    handles = _watch(flow.bus, log, runtime)
    flow.reply_panel.set_phase(f"step {n}/{len(plan)} · executor · up to {printed_bound(granted.minutes)} min")
    started = monotonic()
    try:
        result = await flow.run_config_child(config, "\n\n".join(parts), budget=(row, granted), request_note=executor_note(header),
                                             compact_md_path=COMPACT_DISABLED, pipeline=pipeline)
    except asyncio.CancelledError:
        with suppress(Refused):
            flow.docs.write_step(record.id, n, log, "", runtime, stopped="cancelled")
        stop_step(record, n, "\n".join(log.lines))
        raise
    finally:
        flow.cap.settle(granted, monotonic() - started)
        flow.reply_panel.set_phase(None)
        for handle in handles:
            flow.bus.unsubscribe(handle)
    runtime["files"] = result.artifacts
    if not handles:
        for artifact in result.artifacts:
            log.add(f"wrote {artifact}")
    ended = _ended(result)
    digest, _ = flow.docs.write_step(record.id, n, log, result.raw, runtime, ended)
    child = f"step {n}: {len(result.artifacts)} files written, {runtime['compactions']} compactions"
    if ended:
        stop_step(record, n, "\n".join(log.lines))
        return _done(flow, boundary, unit, (f"step-{n}.md",), f"{child}; the executor stopped ({ended})")
    record.filed[(f"step-{n}.md", None)] = Row(digest)
    record.stopped_log = ""
    set_verdict(record, key, "unreviewed")
    return _done(flow, boundary, unit, (f"step-{n}.md",), child)


_KINDS = {"file": _file, "explore": _explore, "execute": _execute, "revise": _execute, "panel": _review}


async def run_unit(flow: TaskFlow, boundary: str, carried: str | None = None, by: str = "model") -> str:
    """The unit `next_unit` names for this call, run, as a manifest of names, counts, verdicts and fenced findings."""
    if boundary not in _RUNS:
        raise ValueError(f"no unit runs at {boundary!r}")
    started = monotonic()
    try:
        record = flow.record
        refused = precondition(boundary, record, by, carried) or (need("no task") if record is None else None)
        if refused:
            return _text(refused)
        unit = next_unit(record, boundary, carried, flow.limits)
        if isinstance(unit, Need):
            return _text(unit)
        return await _KINDS[unit.kind](flow, boundary, unit, carried) if unit else _read(flow, boundary)
    except Refused as e:
        return str(e)
    finally:
        _exclude(flow, monotonic() - started)
