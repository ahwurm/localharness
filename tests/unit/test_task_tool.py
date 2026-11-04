"""The `task` tool over a TaskFlow: the stage whitelist, the bounded arguments, the run mapping."""
import asyncio

import pytest

from localharness.agent.subagent import _config_child_allowed
from localharness.core.caps import Caps
from localharness.orchestrator.task_flow.record import STAGES, TOOL_STAGES, Row
from localharness.tools.builtin.task_tool import TaskTool
from localharness.tools import base
from tests.unit.test_task_flow_record import TYPED, make_flow, planned

TYPED_TEXT = TYPED[-1]["content"]

NO_TASK = 'no task is open; open one with task(stage="new", doc=<the objective and how you will do it>)'


def spied(flow):
    calls = []

    async def run(boundary, carried=None, by="model"):
        calls.append((boundary, carried, by))
        return "ran"

    flow.run = run
    return calls


def refused(result, text=None):
    assert result.success is False and result.error_type == "validation_error", result
    assert result.output == ""
    if text is not None:
        assert text in result.error, result.error


def test_info_declares_the_labels_and_enumerates_the_tool_stages(tmp_path):
    info = TaskTool(make_flow(tmp_path)[0]).info()
    assert (info.name, info.group, info.gate_family, info.ingest, info.host, info.result_origin) == (
        "task", "task", "allow", "none", "safe", "untrusted")
    assert info.parameters["properties"]["stage"]["enum"] == [*TOOL_STAGES, "done"]
    assert info.parameters["required"] == ["stage"]


async def test_tracer_new_call_files_the_doc_and_names_the_next_unit(tmp_path):
    flow = make_flow(tmp_path)[0]
    flow.open("fix login", 1)
    result = await TaskTool(flow)._execute(stage="new", doc="Objective: fix login · turn 1")
    assert result.success is True
    assert flow.docs.read(1, "decisions.md") == "Objective: fix login · turn 1"
    assert result.output.splitlines()[-1] == "next: plan · file"


async def test_no_task_open_the_models_new_opens_one_on_the_persons_latest_turn(tmp_path):
    flow, _, rows, _ = make_flow(tmp_path)
    flow.turn = lambda: 1
    result = await TaskTool(flow)._execute(stage="new", doc="Objective: x, step by step")
    assert flow.record is not None and flow.record.objective == TYPED_TEXT
    assert flow.docs.read(flow.record.id, "decisions.md") == "Objective: x, step by step"
    assert rows[0][:2] == (f"t{flow.record.id}", "opened by the model") and result.output.splitlines()[-1].startswith("next:")


async def test_no_task_open_a_bare_new_still_refuses(tmp_path):
    flow = make_flow(tmp_path)[0]
    before = sorted(tmp_path.rglob("*"))
    result = await TaskTool(flow)._execute(stage="new")
    assert result.output == NO_TASK
    assert flow.record is None and sorted(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("stage", ["review", "step", "reply", "task", "", None, 3])
async def test_whitelist_refuses_every_stage_outside_the_tool_stages(tmp_path, stage):
    flow = make_flow(tmp_path)[0]
    flow.open("fix login", 1)
    calls = spied(flow)
    refused(await TaskTool(flow)._execute(stage=stage, doc="x"), "new, research, plan, run")
    assert calls == []


async def test_run_maps_to_the_step_boundary_and_returns_the_manifest(tmp_path):
    flow = make_flow(tmp_path)[0]
    flow.open("fix login", 1)
    calls = spied(flow)
    result = await TaskTool(flow)._execute(stage="run")
    assert calls == [("step", None, "model")] and result.output == "ran"


@pytest.mark.parametrize("stage", ["new", "research", "plan"])
async def test_other_stages_map_to_themselves_as_a_model_call(tmp_path, stage):
    flow = make_flow(tmp_path)[0]
    flow.open("fix login", 1)
    calls = spied(flow)
    await TaskTool(flow)._execute(stage=stage)
    assert [c[0] for c in calls] == [stage] and calls[0][2] == "model"
    assert STAGES[stage].stage == stage


def opened(tmp_path, **kw):
    flow = make_flow(tmp_path, **kw)[0]
    flow.open("fix login", 1)
    return flow


async def test_doc_of_exactly_the_cap_in_bytes_is_filed_and_one_byte_more_is_refused(tmp_path):
    flow = opened(tmp_path)
    doc = "\u00e9" * (Caps.DOC // 2)
    assert len(doc.encode()) == Caps.DOC
    assert (await TaskTool(flow)._execute(stage="new", doc=doc)).success is True
    assert flow.docs.read(1, "decisions.md") == doc
    flow = opened(tmp_path / "over")
    calls = spied(flow)
    refused(await TaskTool(flow)._execute(stage="new", doc=doc + "a"), str(Caps.DOC))
    assert calls == [] and flow.docs.read(1, "decisions.md") is None


@pytest.mark.parametrize("doc", ["", "  \n\t ", 7, ["x"]])
async def test_doc_that_is_empty_blank_or_not_text_is_refused_before_any_write(tmp_path, doc):
    flow = opened(tmp_path)
    calls = spied(flow)
    refused(await TaskTool(flow)._execute(stage="new", doc=doc))
    assert calls == [] and flow.docs.read(1, "decisions.md") is None


@pytest.mark.parametrize("stage", ["research", "run"])
async def test_doc_on_a_stage_that_takes_none_is_refused_naming_the_stage_that_does(tmp_path, stage):
    flow = opened(tmp_path)
    calls = spied(flow)
    refused(await TaskTool(flow)._execute(stage=stage, doc="x"), "new and plan")
    assert calls == []


@pytest.mark.parametrize("stage", ["new", "plan", "run"])
async def test_question_on_a_stage_that_takes_none_is_refused_naming_research(tmp_path, stage):
    flow = opened(tmp_path)
    calls = spied(flow)
    refused(await TaskTool(flow)._execute(stage=stage, question="where is login?"), "research")
    assert calls == []


@pytest.mark.parametrize("question", [7, "", "   ", "q" * (Caps.LINE + 1), "a\nb", "a\x07b", "a\u202eb", "a\u200bb"])
async def test_question_that_is_not_one_plain_line_is_refused(tmp_path, question):
    flow = opened(tmp_path)
    calls = spied(flow)
    refused(await TaskTool(flow)._execute(stage="research", question=question), str(Caps.LINE))
    assert calls == []


async def test_a_plain_question_of_the_cap_runs_one_explorer_and_the_seventh_is_full(tmp_path):
    flow = opened(tmp_path, answers={"explorer": "src/a.py:1 login"})
    tool = TaskTool(flow)
    await tool._execute(stage="new", doc="Objective: fix login")
    script = flow.run_config_child
    assert (await tool._execute(stage="research", question="q" * Caps.LINE)).success is True
    assert len(script.named("explorer")) == 1
    for n in range(Caps.QUESTIONS - 1):
        await tool._execute(stage="research", question=f"q{n}")
    full = await tool._execute(stage="research", question="one more")
    assert full.output == f"{Caps.QUESTIONS} questions are filed; call stage=research to review them"
    assert len(script.named("explorer")) == Caps.QUESTIONS


async def test_unknown_argument_is_refused(tmp_path):
    flow = opened(tmp_path)
    calls = spied(flow)
    refused(await TaskTool(flow)._execute(stage="run", action="close"), "action")
    assert calls == []


def test_description_is_one_line_per_stage_in_tool_stage_order_the_same_every_call(tmp_path):
    tool = TaskTool(make_flow(tmp_path)[0])
    text = tool.info().description
    head, where, *lines, done = text.splitlines()
    assert str(Caps.DOC) in head and "one unit" in head and "next" in head
    assert "scratch folder" in where and "not yours to write" in where
    assert done.startswith("done:") and "review.md" in done and tool.info().parameters["properties"]["stage"]["enum"][-1] == "done"
    assert [line.split(":")[0] for line in lines] == list(TOOL_STAGES)
    for line in lines:
        row = STAGES[next(b for b, r in STAGES.items() if r.stage == line.split(":")[0])]
        assert f"reads {', '.join(row.reads)}" in line and f"writes {', '.join(row.writes)}" in line
        assert "takes " in line
    assert tool.info().description == text


def test_config_child_allowed_still_strips_task_and_holds_skill(tmp_path):
    from localharness.config.models import AgentConfig, ToolConfig

    cfg = AgentConfig(name="c", role="r", tools=ToolConfig(add=["task", "read"]))
    assert _config_child_allowed(cfg) == ["read", "skill"]


@pytest.mark.parametrize("cap", [None, 1.0, 360.0])
async def test_the_registry_deadline_is_the_flow_cap_so_no_unit_is_cut_first(tmp_path, monkeypatch, cap):
    flow = opened(tmp_path, cap=cap)
    tool = TaskTool(flow)
    assert tool.timeout_s == flow.limits.flow_cap_minutes * 60
    spied(flow)
    bounds, real = [], asyncio.wait_for

    async def watch(aw, timeout=None):
        bounds.append(timeout)
        return await real(aw, timeout)

    monkeypatch.setattr(base.asyncio, "wait_for", watch)
    assert (await tool.run(stage="run")).output == "ran"
    assert bounds and bounds[0] > flow.limits.flow_cap_minutes * 60


async def test_a_unit_the_registry_cuts_still_settles_its_reservation_and_leaves_the_flow_idle(tmp_path):
    started = asyncio.Event()

    async def hang(config, task):
        started.set()
        await asyncio.sleep(3600)

    flow = (await planned(tmp_path, {"executor": [hang]}))[0]
    tool, cap = TaskTool(flow), flow.cap.remaining()
    tool.timeout_s = 0.05
    result = await tool.run(stage="run")
    assert started.is_set() and result.error_type == "timeout_error"
    assert flow.busy is False and flow.cap.unspent() == pytest.approx(cap, abs=1.0)


async def test_done_closes_a_task_finished_before_any_plan_once_decisions_passed(tmp_path):
    flow = opened(tmp_path)
    tool = TaskTool(flow)
    refused(await tool._execute(stage="done"), "doc")
    assert (await tool._execute(stage="done", doc="the note is at scratch/note.md")).output == "decisions.md must be filed first"
    flow.record.filed[("decisions.md", None)] = Row("d", "blocked")
    out = (await tool._execute(stage="done", doc="the note is at scratch/note.md")).output
    assert out.startswith("t1 closing") and flow.record.closing == "the note is at scratch/note.md" and flow.record.stage == "new"
    said = []
    flow.emit = lambda *a: said.append(a)
    flow.settle_close("blocked")
    assert flow.record.closing is None and flow.record.stage == "new" and said and said[0][1] == "not closed"
    flow.record.closing = "the note is at scratch/note.md"
    flow.settle_close("pass")
    assert flow.record.stage == "done" and flow.record.closing is None
    assert flow.docs.read(1, "review.md") == "the note is at scratch/note.md" and flow.record.filed[("review.md", None)].verdict == "pass"
    assert (await tool._execute(stage="done", doc="again")).output == "a task past its plan ends through its steps"


async def test_done_with_no_task_open_says_so(tmp_path):
    flow = make_flow(tmp_path)[0]
    assert (await TaskTool(flow)._execute(stage="done", doc="x")).output == "no task is open"


async def test_the_model_opens_only_on_this_turns_message_and_once_a_turn(tmp_path):
    flow, _, rows, _ = make_flow(tmp_path)
    flow.turn = lambda: 2
    await TaskTool(flow)._execute(stage="new", doc="x")
    assert flow.record is None
    flow.turn = lambda: 1
    await TaskTool(flow)._execute(stage="new", doc="x")
    first = flow.record
    first.stage = "done"
    await TaskTool(flow)._execute(stage="new", doc="y")
    assert flow.record is first and len([r for r in rows if r[1] == "opened by the model"]) == 1
    flow.end_turn()
    await TaskTool(flow)._execute(stage="new", doc="y")
    opened = [r for r in rows if r[1] == "opened by the model"]
    assert flow.record is not first and len(opened) == 2 and opened[0][2] == ("/task clear (removes it)",)
