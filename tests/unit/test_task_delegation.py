"""Composed: delegation as a complete handoff through the real loop.

A scripted model drives the parent AgentLoop; `task` is a real TaskTool, `agent` is a real
AgentTool over the real `make_explore_agent_runner`, which spawns a real config child (yaml on
disk) holding a real WriteTool. One scripted LLM serves parent and child in order (delegation is
awaited). Assertions read the task record and the bus, never a mock of the units under test."""
import asyncio

import pytest
import yaml

from localharness.agent import subagent
from localharness.agent.permissions import PermissionEvaluator
from localharness.agent.subagent import make_explore_agent_runner
from localharness.agent.task_record import TaskState
from localharness.config.models import AgentConfig
from localharness.core.events import Observation
from localharness.tools.builtin.agent_tool import AgentTool
from localharness.tools.builtin.task_tool import TaskTool
from localharness.tools.builtin.write_tool import WriteTool
from localharness.tools.registry import ToolRegistry
from tests.unit.test_task_context import make_loop
from tests.unit.test_task_references import capture, scripted

START = ("task", {"action": "start", "objective": "Publish a reviewed note",
                  "assignment": "Get an independent review of draft.md"})
DELEGATE = ("agent", {"agent_id": "reviewer", "task": "Review draft.md",
                      "purpose": "independent review", "inputs": ["draft.md"],
                      "expected_output": "review.md"})


async def build(tmp_path, bus, llm, reviewer_budget=4):
    agents = tmp_path / "agents"
    agents.mkdir(exist_ok=True)
    for name in ("reviewer", "writer"):
        (agents / f"{name}.yaml").write_text(yaml.safe_dump({
            "name": name, "role": f"You are the {name}.", "tools": {"add": ["write"]},
            "permissions": {"mode": "unattended", "deny_patterns": [],
                            "budget": {"max_actions": reviewer_budget, "max_duration_minutes": 5}},
        }))

    def load_agent(name):
        return AgentConfig.model_validate(yaml.safe_load((agents / f"{name}.yaml").read_text()))

    write = WriteTool()
    write.workspace_root = str(tmp_path)
    base = ToolRegistry()
    await base.register(write, scope="global")
    state = TaskState(agents / "t" / "task.json", workspace=str(tmp_path))
    holder = {}
    runner = make_explore_agent_runner(
        llm=llm, bus=bus, base_registry=base, permission_evaluator=PermissionEvaluator(),
        get_parent_session_id=lambda: holder["loop"].current_session_id,
        load_agent=load_agent, config_dir=tmp_path,
    )
    results = []

    async def observed(*args):  # records what the real runner returned; changes nothing
        result = await runner(*args)
        results.append(result)
        return result

    parent = ToolRegistry()
    await parent.register(TaskTool(state), scope="global")
    await parent.register(AgentTool(agent_runner=observed, available_agents=["reviewer", "writer"]),
                          scope="global")
    holder["loop"] = make_loop(llm, bus, tmp_path, state, parent)
    return holder["loop"], state, results


def agent_observations(bus):
    return [e for e in bus.history(event_types=[Observation]) if e.tool_name == "agent"]


def tool_messages(seen, needle):
    return [m["content"] for request in seen for m in request
            if m.get("role") == "tool" and needle in str(m.get("content"))]


@pytest.mark.asyncio
async def test_completed_delegation_integrated_then_closed(tmp_path, bus, mock_llm_client):
    review = tmp_path / "review.md"
    llm = scripted(
        mock_llm_client, START, DELEGATE,
        ("write", {"path": str(review), "content": "Two claims lack sources."}),
        "Reviewed.\nHANDOFF\nstatus: completed\nartifacts: review.md\nfindings: two unsupported "
        "claims\nevidence: lines 3 and 9\nuncertainties: none\nremaining: none",
        ("task", {"action": "integrate", "delegation_id": "d1", "note": "fixed both claims"}),
        ("task", {"action": "close", "status": "complete"}),
        "Done.",
    )
    seen = capture(llm)
    loop, state, results = await build(tmp_path, bus, llm)

    assert await loop.run_turn("Publish the note after an independent review") == "Done."

    child_first = next("\n".join(str(m.get("content")) for m in request) for request in seen
                       if any("Purpose: independent review" in str(m.get("content")) for m in request))
    assert "Budget: 4 actions, 5 minutes." in child_first and "HANDOFF" in child_first
    (d,) = state.current.delegations
    assert (d.id, d.agent, d.status) == ("d1", "reviewer", "completed")
    assert d.artifacts == [str(review)] and review.read_text() == "Two claims lack sources."
    assert d.findings == "two unsupported claims" and d.integrated == "fixed both claims"
    assert state.current.closed
    (obs,) = agent_observations(bus)
    assert obs.error is None
    assert results[0].child_session_id and results[0].child_session_id != loop.current_session_id


@pytest.mark.asyncio
async def test_failed_delegation_is_named_in_reply_and_keeps_record_live(
    tmp_path, bus, mock_llm_client, monkeypatch,
):
    async def boom(*args, **kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr(subagent, "dispatch_config_subagent", boom)
    llm = scripted(mock_llm_client, START, DELEGATE,
                   ("task", {"action": "close", "status": "complete"}), "Done.")
    seen = capture(llm)
    loop, state, _ = await build(tmp_path, bus, llm)

    reply = await loop.run_turn("Publish the note after an independent review")

    (d,) = state.current.delegations
    assert d.status == "failed: execution_error" and "boom" in d.findings
    assert tool_messages(seen, "reviewer (d1): failed: execution_error, not integrated")
    assert state.current.closed and state._live() is not None  # closing never refuses
    assert reply == "Done."
    (obs,) = agent_observations(bus)
    assert obs.error and "boom" in obs.error


@pytest.mark.asyncio
async def test_malformed_handoff_is_none_and_status_stays_runtime(tmp_path, bus, mock_llm_client):
    llm = scripted(
        mock_llm_client, START, DELEGATE,
        ("write", {"path": str(tmp_path / "review.md"), "content": "ok"}),
        "Reviewed.\nHANDOFF\nstatus: maybe\nfindings: x",
        "Stopping here.",
    )
    loop, state, results = await build(tmp_path, bus, llm)

    await loop.run_turn("Publish the note after an independent review")

    assert results[0].handoff is None and results[0].status == "completed"
    (d,) = state.current.delegations
    assert d.status == "completed"
    assert d.findings != "x" and d.findings.endswith("status: maybe\nfindings: x")


@pytest.mark.asyncio
async def test_cancelled_delegation_becomes_interrupted(tmp_path, bus, mock_llm_client, monkeypatch):
    async def hang(*args, **kwargs):
        await asyncio.Event().wait()
    monkeypatch.setattr(subagent, "dispatch_config_subagent", hang)
    llm = scripted(mock_llm_client, START, DELEGATE, "Noted the interruption.")
    loop, state, _ = await build(tmp_path, bus, llm)

    turn = asyncio.create_task(loop.run_turn("Publish the note after an independent review"))
    for _ in range(200):
        await asyncio.sleep(0)
        if state.current and state.current.delegations:
            break
    assert state.current.delegations[0].status == "running"
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    reloaded, _ = TaskState.load(state.path, str(tmp_path))
    assert reloaded.current.delegations[0].status == "interrupted"
    assert "delegation d1 reviewer interrupted" in reloaded.notes

    await loop.run_turn("continue")
    assert state.current.delegations[0].status == "interrupted"
    assert "delegation d1 reviewer interrupted" in state.packet()
    assert "Delegations (1): d1 reviewer: interrupted" in state.packet()


@pytest.mark.asyncio
async def test_child_out_of_budget_is_budget_exhausted(tmp_path, bus, mock_llm_client):
    def write(n):
        return ("write", {"path": str(tmp_path / f"part{n}.md"), "content": f"part {n}"})
    llm = scripted(mock_llm_client, START, DELEGATE, write(1), write(2), write(3),
                   "Partial review: part 1 only.", "Stopping here.")
    loop, state, results = await build(tmp_path, bus, llm, reviewer_budget=1)

    await loop.run_turn("Publish the note after an independent review")

    assert results[0].terminated_reason == "budget_actions"
    (d,) = state.current.delegations
    assert d.status == "budget_exhausted"
    assert d.artifacts == [str(tmp_path / "part1.md")]
