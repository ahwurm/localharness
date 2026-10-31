"""The task flow's runner: one unit per call, the executor's config and the two request-note providers."""
from __future__ import annotations

import math
import re
from time import monotonic
from typing import TYPE_CHECKING

from localharness.orchestrator.task_flow.filer import Refused, manifest, render_section
from localharness.orchestrator.task_flow.panel import PanelInputs, PanelResult, PanelRun, fence, typed_turns
from localharness.orchestrator.task_flow.record import (
    DOCS, LINES, STAGES, Need, Row, Unit, can_revise, file_doc, need, next_line, next_unit, note_run, precondition,
    printed_bound, set_verdict, unit_minutes)

if TYPE_CHECKING:
    from localharness.orchestrator.task_flow import TaskFlow

_RUNS = ("new", "research", "plan", "step", "task")


def _exclude(flow: TaskFlow, seconds: float) -> None:
    """A call's wall time off the root's clock: non-finite or negative is zero, and a turn's total never passes the flow cap."""
    valid = seconds if math.isfinite(seconds) and seconds >= 0 else 0.0
    valid = min(valid, max(0.0, flow.limits.flow_cap_minutes * 60 - flow.excluded))
    flow.excluded += valid
    flow.exclude(valid)


def _text(need_: Need) -> str:
    return need_.model or need_.reason


def _key(unit: Unit) -> tuple[str, str | None]:
    return ("plan.md", f"step-{unit.step}") if unit.boundary == "step" else (DOCS[unit.boundary], None)


def _review_doc(unit: Unit) -> str:
    if unit.boundary == "step":
        return f"step-{unit.step}-review.md"
    return "review.md" if unit.boundary == "task" else DOCS[unit.boundary].replace(".md", "-review.md")


def _label(unit: Unit) -> str:
    return f"step {unit.step}" if unit.boundary == "step" else STAGES[unit.boundary].stage or unit.boundary


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


def _stale(flow: TaskFlow, *keys: tuple[str, str | None]) -> str | None:
    bad = next((k for k in keys if k in flow.record.filed and not flow.docs.verified(flow.record, k)), None)
    return _text(need("edited since filed", doc=bad[0])) if bad else None


async def _panel(flow: TaskFlow, unit: Unit) -> PanelResult:
    record, key = flow.record, _key(unit)

    def report(run: PanelRun) -> None:
        doc = _review_doc(unit)
        rounds = max(map(int, re.findall(r"(?m)^## round ([0-9]+)", flow.docs.read(record.id, doc) or "")), default=0) + 1
        digest = flow.docs.append_section(record.id, doc, render_section(rounds, run.findings, run.verdict, run.blockers))
        record.filed[(doc, None)] = Row(digest)
        settled = run.verdict in ("pass", "blocked")
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
    verdict = f"{_label(unit)} {result.verdict}" + (f" · {result.blockers} blockers" if result.verdict == "blocked" else "") if result else ""
    revising = (result and result.verdict == "blocked" and unit.boundary in ("new", "plan")
                and can_revise(record, unit.boundary, flow.limits.revisions[unit.boundary]))
    doc = flow.docs.read(record.id, DOCS[unit.boundary]) or "" if revising else ""
    findings = fence(result.findings) if result and result.findings else ""
    return manifest(filed, child, verdict, doc, findings, notes, next_line(record, boundary, flow.limits))


async def _file(flow: TaskFlow, boundary: str, unit: Unit, carried: str | None) -> str:
    record, name = flow.record, DOCS[unit.boundary]
    digest, cut = flow.docs.write(record.id, name, carried)
    file_doc(record, unit.boundary, (name, None), digest)
    result = await _panel(flow, Unit(unit.boundary, "panel"))
    return _done(flow, boundary, unit, (name, _review_doc(unit)), result=result,
                 notes=["truncated at the doc cap"] if cut else [])


async def _review(flow: TaskFlow, boundary: str, unit: Unit, carried: str | None) -> str:
    if stale := _stale(flow, _key(unit)):
        return stale
    result = await _panel(flow, unit)
    return _done(flow, boundary, unit, (_review_doc(unit),), result=result)


_KINDS = {"file": _file, "panel": _review}


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
        return await _KINDS[unit.kind](flow, boundary, unit, carried)
    except Refused as e:
        return str(e)
    finally:
        _exclude(flow, monotonic() - started)
