"""run_config_child: the per-child primitive, its budget, request note and grants, and the time-budget exclusion."""
from __future__ import annotations

import inspect

import pytest

from localharness.agent.permissions import PermissionEvaluator
from localharness.agent.subagent import DelegationResult, make_explore_agent_runner
from localharness.config.models import AgentConfig, BudgetConfig, ChildBudget, PermissionConfig, ToolConfig
from tests.unit.test_capabilities import _builtin_registry


def config_child(name="cfg", add=("read", "grep")) -> AgentConfig:
    return AgentConfig(name=name, role="test child", tools=ToolConfig(add=list(add)),
                       permissions=PermissionConfig(budget=BudgetConfig()))


def capture(llm) -> list[list]:
    seen: list[list] = []
    original = llm.stream_complete

    async def wrapped(messages, *a, **kw):
        seen.append(list(messages))
        return await original(messages, *a, **kw)
    llm.stream_complete = wrapped
    return seen


async def test_a_config_child_runs_with_a_row_budget_and_a_request_note(bus, mock_llm_client):
    R = mock_llm_client.Response
    llm = mock_llm_client([R(content="README.md describes a harness."), R(content="README.md describes a harness.")])
    seen = capture(llm)
    runner = make_explore_agent_runner(
        llm=llm, bus=bus, base_registry=await _builtin_registry(), permission_evaluator=PermissionEvaluator(),
        get_parent_session_id=lambda: "sid")
    result = await runner.run_config_child(
        config_child(), "Summarise README.md",
        budget=("task_flow.executor", ChildBudget(actions=24, minutes=25)),
        request_note=lambda: "Step 1/2: summarise README.md")
    first = seen[0]
    assert first[-1]["_lh"] == {"origin": "harness", "subtype": "request_note"}
    assert first[-1]["content"] == "Step 1/2: summarise README.md"
    assert "Budget: 24 actions, 25 minutes." in next(m["content"] for m in first if m["role"] == "user")
    assert all(sum(m.get("_lh", {}).get("subtype") == "request_note" for m in req) == 1 for req in seen)
    assert result.raw == "README.md describes a harness."
    assert result.text.startswith("SUBAGENT RUN COMPLETE")
    assert isinstance(result, DelegationResult)


def test_run_config_child_signature():
    runner = make_explore_agent_runner(
        llm=object(), bus=object(), base_registry=object(), permission_evaluator=PermissionEvaluator(),
        get_parent_session_id=lambda: None)
    params = list(inspect.signature(runner.run_config_child).parameters.values())
    assert [p.name for p in params] == [
        "config", "task", "budget", "request_note", "compact_md_path", "grant_handles", "tools", "shape", "pipeline"]
    assert all(p.default is None for p in params[2:])
