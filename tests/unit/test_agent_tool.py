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


def test_agent_tool_invites_composition_and_names_the_boundary():
    """Self-extension stays (owner ruling, Option 1), with the owner's boundary: compose
    specialists from the tools the harness has, never rebuild one of them as a script."""
    description = _make_agent_tool().info().description

    assert "BUILD a new specialist" in description
    assert "~/.localharness/agents/<name>.yaml" in description
    assert ("compose specialists from the existing tools; never write your own tool for something "
            "the harness already provides (web search, fetch, files, delegation)") in description.lower()
    assert "localharness start" in description and "~/.localharness/tools/" in description


def test_agent_tool_task_description_distills_and_gives_examples():
    """#73(b): the `task` field must steer the model to write a SELF-CONTAINED
    instruction for the subagent, never the user's verbatim sentence — with
    worked GOOD/BAD examples so the nudge lands on a local model."""
    tool = _make_agent_tool()
    task_desc = tool.info().parameters["properties"]["task"]["description"]
    lowered = task_desc.lower()
    assert "self-contained" in lowered
    assert "verbatim" in lowered
    # The worked GOOD example (a concrete directive, not "ask the joke-writer…").
    assert "Write three puns about databases." in task_desc
    # At least two GOOD/BAD contrast pairs are spelled out.
    assert lowered.count("good:") >= 2
    assert lowered.count("bad:") >= 2


@pytest.mark.asyncio
async def test_agent_tool_delegates_to_runner():
    runner = AsyncMock(return_value="Done: created hello.py")
    tool = AgentTool(agent_runner=runner, available_agents=["coder"])
    result = await tool._execute(agent_id="coder", task="write hello.py")
    assert result.success is True
    assert "Done: created hello.py" in result.output
    runner.assert_awaited_once_with("coder", "write hello.py", None)  # 3-arg runner contract (grant_handles)


@pytest.mark.asyncio
async def test_agent_tool_unknown_agent_passes_runner_message_through():
    """The runner's ValueError is actionable by design (it names what IS dispatchable and
    says a yaml can be created) — surface it verbatim. Rebuilding a generic not-found from
    the advertised list self-contradicts when that list drifts from the dispatchable set
    (qwen3-4b kospi receipts 2026-07-17: \"'data-analyst' not found. Available: ...
    data-analyst ...\")."""
    runner = AsyncMock(side_effect=ValueError(
        "Agent 'unknown' dispatch not wired (available: explore, web-researcher, "
        "search-verifier, cruncher, or any agents/<name>.yaml definition)"))
    tool = AgentTool(agent_runner=runner, available_agents=["coder"])
    result = await tool._execute(agent_id="unknown", task="anything")
    assert result.success is False
    assert result.error_type == "not_found"
    assert "dispatch not wired" in result.error
    assert "unknown" in result.error
    assert "coder" not in result.error  # the advertised list is NOT substituted


@pytest.mark.asyncio
async def test_agent_tool_keyerror_falls_back_to_advertised_list():
    """KeyError's str() is just the quoted key — useless as a message — so the generic
    not-found built from the advertised list is the right fallback there."""
    runner = AsyncMock(side_effect=KeyError("unknown"))
    tool = AgentTool(agent_runner=runner, available_agents=["coder"])
    result = await tool._execute(agent_id="unknown", task="anything")
    assert result.success is False
    assert result.error_type == "not_found"
    assert "Agent 'unknown' not found" in result.error
    assert "coder" in result.error


@pytest.mark.asyncio
async def test_agent_tool_empty_valueerror_falls_back_to_advertised_list():
    runner = AsyncMock(side_effect=ValueError())
    tool = AgentTool(agent_runner=runner, available_agents=["coder"])
    result = await tool._execute(agent_id="ghost", task="anything")
    assert result.error_type == "not_found"
    assert "Agent 'ghost' not found" in result.error


@pytest.mark.asyncio
async def test_agent_tool_runner_exception_returns_execution_error():
    runner = AsyncMock(side_effect=RuntimeError("agent crashed"))
    tool = AgentTool(agent_runner=runner, available_agents=["coder"])
    result = await tool._execute(agent_id="coder", task="do stuff")
    assert result.success is False
    assert result.error_type == "execution_error"
    assert "crashed" in result.error


# --- 0.16.5 slice 3: structured assignment in, runtime-truth result out ---------------------

STRUCTURED = dict(
    purpose="independent review", inputs=["draft.md", "pg-3"], constraints="read only",
    expected_output="review.md", checks=["every claim cited", "under 300 words"],
    stop_condition="after one pass",
)


@pytest.mark.asyncio
async def test_plain_call_passes_task_byte_for_byte():
    from localharness.tools.builtin.agent_tool import compose_brief

    task = "Summarize the retry logic in src/agent/loop.py in 5 bullets.\n  keep spacing  "
    assert compose_brief(task) is task
    runner = AsyncMock(return_value="ok")
    await _make_agent_tool(runner=runner).run(agent_id="coder", task=task)
    runner.assert_awaited_once_with("coder", task, None)


@pytest.mark.asyncio
async def test_structured_call_composes_ordered_brief_with_handoff_instruction():
    runner = AsyncMock(return_value="ok")
    await _make_agent_tool(runner=runner).run(agent_id="coder", task="Review draft.md", **STRUCTURED)
    brief = runner.await_args.args[1]
    order = ["Purpose: independent review", "Assignment: Review draft.md",
             "Inputs: draft.md, pg-3", "Constraints: read only", "Expected output: review.md",
             "Checks: every claim cited; under 300 words", "Stop when: after one pass",
             "Finish with a HANDOFF block"]
    positions = [brief.index(s) for s in order]
    assert positions == sorted(positions)
    assert brief.startswith("Purpose: ")


def test_structured_fields_are_optional_in_schema():
    params = _make_agent_tool().info().parameters
    for key in STRUCTURED:
        assert key in params["properties"]
    assert params["required"] == ["agent_id", "task"]


@pytest.mark.asyncio
async def test_str_runner_result_gets_neutral_metadata():
    result = await _make_agent_tool(runner=AsyncMock(return_value="found it")).run(
        agent_id="coder", task="x")
    assert result.success and result.output == "found it"
    assert result.metadata == {
        "delegated_to": "coder", "status": "completed", "terminated_reason": None,
        "tool_calls": 0, "artifacts": [], "child_session_id": None, "handoff": None,
    }


@pytest.mark.asyncio
async def test_delegation_result_fields_are_forwarded():
    from localharness.agent.subagent import DelegationResult

    handoff = {"status": "completed", "findings": "two claims"}
    dr = DelegationResult("SUBAGENT RUN COMPLETE ...", status="budget_exhausted",
                          terminated_reason="budget_actions", tool_calls=3,
                          artifacts=["/w/review.md"], child_session_id="s-1", handoff=handoff)
    result = await _make_agent_tool(runner=AsyncMock(return_value=dr)).run(agent_id="coder", task="x")
    assert result.success and result.output == "SUBAGENT RUN COMPLETE ..."
    assert result.metadata == {
        "delegated_to": "coder", "status": "budget_exhausted", "terminated_reason": "budget_actions",
        "tool_calls": 3, "artifacts": ["/w/review.md"], "child_session_id": "s-1", "handoff": handoff,
    }
