"""An `agent` Observation carries how its child's run ended, from the runtime and never from text."""
from typing import get_args

import pytest

from localharness.agent.context import ContextManager
from localharness.agent.loop import AgentLoop
from localharness.agent.permissions import PermissionEvaluator
from localharness.agent.subagent import _delegation_status
from localharness.config.models import AgentConfig, PermissionConfig
from localharness.core.events import DELEGATION_WORDS, DelegationStatus, Observation
from localharness.tools.base import Tool


class _Registry:
    def __init__(self, dispatch):
        self.dispatch = dispatch

    def global_tools(self):
        return []

    def get_tools_for_agent(self, agent_id, division_id, tool_config):
        return {}


def _loop(mock_llm_client, bus, tool: str, dispatch, name: str = "root") -> AgentLoop:
    R, TC = mock_llm_client.Response, mock_llm_client.ToolCall
    llm = mock_llm_client([R(content=None, tool_calls=[TC(id=f"{name}-1", name=tool, arguments={})]),
                           R(content="Done.")])
    return AgentLoop(
        config=AgentConfig(name=name, role="Test agent.", permissions=PermissionConfig(mode="unattended")),
        llm=llm, bus=bus, context_manager=ContextManager(), tool_registry=_Registry(dispatch),
        permission_evaluator=PermissionEvaluator(),
    )


def _returning(result):
    async def dispatch(*_args, **_kwargs):
        return result
    return dispatch


async def _status(mock_llm_client, bus, tool: str, result) -> str | None:
    await _loop(mock_llm_client, bus, tool, _returning(result)).run_turn("go")
    [obs] = bus.history(event_types=[Observation])
    return obs.delegation_status


def test_the_field_and_its_words_hold_exactly_the_statuses_a_delegation_can_end_with():
    reasons = [None, "complete", "budget_actions", "kill_file", "stuck", "error"]
    runtime = {_delegation_status(r) for r in reasons} | {"failed"}
    assert set(get_args(DelegationStatus)) == set(DELEGATION_WORDS) == runtime
    assert all(DELEGATION_WORDS.values()) and DELEGATION_WORDS["failed"] == "no tool calls"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["stuck", "budget_exhausted", "killed", "completed", "failed"])
async def test_an_agent_observation_carries_the_runtime_status(mock_llm_client, bus, status):
    assert await _status(mock_llm_client, bus, "agent", Tool.ok(None, "partial", status=status)) == status


@pytest.mark.asyncio
@pytest.mark.parametrize("tool,result", [
    ("agent", Tool.ok(None, "x", status="partial")),
    ("agent", Tool.ok(None, "x", status=["stuck"])),
    ("agent", Tool.err(None, "boom", status="stuck")),
    ("probe", Tool.ok(None, "x", status="stuck")),
], ids=["outside-the-literal", "not-a-string", "failed-result", "not-the-delegate"])
async def test_any_other_status_leaves_it_none(mock_llm_client, bus, tool, result):
    assert await _status(mock_llm_client, bus, tool, result) is None


@pytest.mark.asyncio
async def test_nested_delegations_each_carry_their_own_childs_status(mock_llm_client, bus):
    async def child(*_args, **_kwargs):
        await _loop(mock_llm_client, bus, "agent", _returning(Tool.ok(None, "x", status="killed")), "child").run_turn("go")
        return Tool.ok(None, "x", status="stuck")

    await _loop(mock_llm_client, bus, "agent", child).run_turn("go")
    assert {o.agent_id: o.delegation_status for o in bus.history(event_types=[Observation])} == {
        "child": "killed", "root": "stuck"}
