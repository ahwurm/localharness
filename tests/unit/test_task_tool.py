"""The `task` tool over a TaskFlow: the stage whitelist, the bounded arguments, the run mapping."""
import pytest

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
