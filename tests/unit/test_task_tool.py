"""The `task` tool over a TaskFlow: the stage whitelist, the bounded arguments, the run mapping."""
import pytest

from localharness.agent.subagent import _config_child_allowed
from localharness.core.caps import Caps
from localharness.orchestrator.task_flow.record import STAGES, TOOL_STAGES
from localharness.tools.builtin.task_tool import TaskTool
from tests.unit.test_task_flow_record import make_flow

NO_TASK = "no task is open; the person opens one with /task new"


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
    assert info.parameters["properties"]["stage"]["enum"] == list(TOOL_STAGES)
    assert info.parameters["required"] == ["stage"]


async def test_tracer_new_call_files_the_doc_and_names_the_next_unit(tmp_path):
    flow = make_flow(tmp_path)[0]
    flow.open("fix login", 1)
    result = await TaskTool(flow)._execute(stage="new", doc="Objective: fix login · turn 1")
    assert result.success is True
    assert flow.docs.read(1, "decisions.md") == "Objective: fix login · turn 1"
    assert result.output.splitlines()[-1] == "next: plan · file"


async def test_no_task_open_the_model_cannot_open_one(tmp_path):
    flow = make_flow(tmp_path)[0]
    before = sorted(tmp_path.rglob("*"))
    result = await TaskTool(flow)._execute(stage="new", doc="x")
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


@pytest.mark.parametrize("question", [7, "", "q" * (Caps.LINE + 1), "a\nb", "a\x07b", "a\u202eb", "a\u200bb"])
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
    head, *lines = text.splitlines()
    assert str(Caps.DOC) in head and "one unit" in head and "next" in head
    assert [line.split(":")[0] for line in lines] == list(TOOL_STAGES)
    for line in lines:
        row = STAGES[next(b for b, r in STAGES.items() if r.stage == line.split(":")[0])]
        assert f"reads {', '.join(row.reads)}" in line and f"writes {', '.join(row.writes)}" in line
        assert "takes " in line
    assert tool.info().description == text


def test_config_child_allowed_still_strips_task(tmp_path):
    from localharness.config.models import AgentConfig, ToolConfig

    cfg = AgentConfig(name="c", role="r", tools=ToolConfig(add=["task", "read"]))
    assert _config_child_allowed(cfg) == ["read"]
