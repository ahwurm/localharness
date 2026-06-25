"""Tests for the agent delegation tool."""
import pytest
from unittest.mock import AsyncMock

from localharness.tools.builtin.agent_tool import AgentTool


def _make_agent_tool(runner=None, agents=None):
    if runner is None:
        runner = AsyncMock(return_value="Task completed successfully.")
    return AgentTool(agent_runner=runner, available_agents=agents or ["coder", "researcher"])


def test_agent_tool_info_schema():
    tool = _make_agent_tool()
    schema = tool.info()
    assert schema.name == "agent"
    assert "agent_id" in schema.parameters["properties"]
    assert "task" in schema.parameters["properties"]
    assert schema.parameters["required"] == ["agent_id", "task"]
    assert schema.scope == "agent"


def test_agent_tool_description_lists_agents():
    tool = _make_agent_tool(agents=["coder", "researcher"])
    schema = tool.info()
    assert "coder" in schema.description
    assert "researcher" in schema.description


@pytest.mark.asyncio
async def test_agent_tool_delegates_to_runner():
    runner = AsyncMock(return_value="Done: created hello.py")
    tool = AgentTool(agent_runner=runner, available_agents=["coder"])
    result = await tool._execute(agent_id="coder", task="write hello.py")
    assert result.success is True
    assert "Done: created hello.py" in result.output
    runner.assert_awaited_once_with("coder", "write hello.py", None)  # 3-arg runner contract (grant_handles)


@pytest.mark.asyncio
async def test_agent_tool_unknown_agent_returns_not_found():
    runner = AsyncMock(side_effect=ValueError("not found"))
    tool = AgentTool(agent_runner=runner, available_agents=["coder"])
    result = await tool._execute(agent_id="unknown", task="anything")
    assert result.success is False
    assert result.error_type == "not_found"
    assert "unknown" in result.error


@pytest.mark.asyncio
async def test_agent_tool_runner_exception_returns_execution_error():
    runner = AsyncMock(side_effect=RuntimeError("agent crashed"))
    tool = AgentTool(agent_runner=runner, available_agents=["coder"])
    result = await tool._execute(agent_id="coder", task="do stuff")
    assert result.success is False
    assert result.error_type == "execution_error"
    assert "crashed" in result.error
