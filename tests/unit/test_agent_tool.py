"""Tests for the agent delegation tool."""
import asyncio
import re

import pytest
from unittest.mock import AsyncMock

from localharness.tools.builtin.agent_tool import (
    BUILTIN_AGENT_NAMES, INVALID_AGENT_NAME, ROSTER_NAME_CAP, AgentTool,
)

BOUND = (
    "You can also BUILD a new specialist as a YAML file (fields: name, role, tools: {add: [tool names]}, permissions: {budget: {max_actions, max_duration_minutes}}), then delegate to that name immediately. A project specialist goes at <P>/<name>.yaml and is read by every session started in this project; a machine specialist goes at <M>/agents/<name>.yaml and serves every project. The file name is the agent name: lowercase letters, digits, hyphens. Compose specialists from the existing tools; never write your own tool for something the harness already provides (web search, fetch, files, delegation). A specialist gets no MCP servers of its own: compose it from the harness tools (`tools.add`). A project specialist's permissions can only tighten the machine's. A script you add under <M>/tools/ runs only as an unconfirmed shell command (asked about in guarded mode) until they confirm it at their next `localharness start`."
)
UNBOUND = (
    'You can also BUILD a new specialist as a YAML file (fields: name, role, tools: {add: [tool names]}, permissions: {budget: {max_actions, max_duration_minutes}}), then delegate to that name immediately. There is no project layer in this session, so write it at <M>/agents/<name>.yaml; a project gets a layer with `localharness init --workspace` in its folder if it has none yet; it is read from the next start made from that folder without `--config-dir` or LOCALHARNESS_DIR/LOCALHARNESS_HOME. The file name is the agent name: lowercase letters, digits, hyphens. Compose specialists from the existing tools; never write your own tool for something the harness already provides (web search, fetch, files, delegation). A specialist gets no MCP servers of its own: compose it from the harness tools (`tools.add`). A script you add under <M>/tools/ runs only as an unconfirmed shell command (asked about in guarded mode) until they confirm it at their next `localharness start`.'
)


def _make_agent_tool(runner=None, agents=None, **kw):
    if runner is None:
        runner = AsyncMock(return_value="Task completed successfully.")
    return AgentTool(agent_runner=runner, available_agents=agents or ["coder", "researcher"], **kw)


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


# --- shape: lookup | review (08-09): picks only the share of the parent's remaining budget -----

def test_shape_is_an_optional_enum_in_the_schema():
    params = _make_agent_tool().info().parameters
    assert params["properties"]["shape"] == {
        "type": "string",
        "enum": ["lookup", "review"],
        "description": "lookup (default): a smaller share of your remaining budget; review: a larger one.",
    }
    assert "shape" not in params["required"]
    assert params["required"] == ["agent_id", "task"]


@pytest.mark.asyncio
async def test_no_shape_keeps_the_three_argument_runner_call():
    runner = AsyncMock(return_value="ok")
    await _make_agent_tool(runner=runner)._execute(agent_id="coder", task="write hello.py")
    runner.assert_awaited_once_with("coder", "write hello.py", None)


@pytest.mark.asyncio
async def test_a_runner_that_takes_no_shape_still_works_without_one():
    seen = []

    async def spy_runner(agent_id, task, grant_handles=None):
        seen.append((agent_id, task, grant_handles))
        return "ok"

    result = await _make_agent_tool(runner=spy_runner).run(agent_id="coder", task="x")
    assert result.success and seen == [("coder", "x", None)]


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["lookup", "review"])
async def test_a_shape_reaches_the_runner_as_a_keyword(shape):
    runner = AsyncMock(return_value="ok")
    result = await _make_agent_tool(runner=runner)._execute(
        agent_id="coder", task="write hello.py", shape=shape)
    assert result.success
    runner.assert_awaited_once_with("coder", "write hello.py", None, shape=shape)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["deep", "", "Review", "REVIEW", 3])
async def test_any_other_shape_is_refused_before_dispatch(shape):
    runner = AsyncMock(return_value="ok")
    tool = _make_agent_tool(runner=runner)
    result = await tool._execute(agent_id="coder", task="write hello.py", shape=shape)
    assert result.success is False
    assert result.error_type == "validation_error"
    assert "shape must be 'lookup' or 'review'" in result.error
    runner.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_shape_through_run_reaches_the_runner():
    runner = AsyncMock(return_value="ok")
    result = await _make_agent_tool(runner=runner).run(agent_id="coder", task="x", shape="review")
    assert result.success
    runner.assert_awaited_once_with("coder", "x", None, shape="review")


def _text(**kw):
    return _make_agent_tool(**kw).info().description


def test_the_bound_text_names_both_dirs_verbatim(tmp_path):
    machine = tmp_path / "home" / "agents"
    project = tmp_path / "[old] proj" / ".localharness" / "agents"
    description = _text(machine_agents_dir=machine, project_agents_dir=project)

    assert BOUND.replace("<P>", str(project)).replace("<M>", str(machine.parent)) in description
    assert "<P>" not in description and "<M>" not in description
    assert "read by every session started in this project" in description


def test_the_unbound_text_says_there_is_no_project_layer(tmp_path):
    machine = tmp_path / "home" / "agents"
    description = _text(machine_agents_dir=machine, project_agents_dir=None)

    assert UNBOUND.replace("<M>", str(machine.parent)) in description
    assert "no project layer in this session" in description
    assert "agent create --project" not in description
    assert "permissions can only tighten" not in description


@pytest.mark.parametrize("project", [None, "bound"])
def test_a_custom_home_is_rendered_never_the_tilde(tmp_path, project):
    home = tmp_path / "custom home"
    description = _text(machine_agents_dir=home / "agents",
                        project_agents_dir=tmp_path / "p" / "agents" if project else None)

    assert f"{home}/agents/<name>.yaml" in description and f"{home}/tools/" in description
    assert "~/.localharness" not in description


def test_no_machine_dir_renders_the_default_paths():
    for project in (None, "p"):
        description = _text(project_agents_dir=project)
        assert "~/.localharness/agents/<name>.yaml" in description
        assert "~/.localharness/tools/" in description


@pytest.mark.parametrize("project", [None, "bound"])
def test_neither_text_claims_git_or_a_confirmation_wait(tmp_path, project):
    description = _text(machine_agents_dir=tmp_path / "agents",
                        project_agents_dir=tmp_path / "p" if project else None)

    assert "waits for the user's confirmation" not in description
    assert "count only after" not in description
    assert re.search(r"\bgit\b|commit", description, re.I) is None
    assert "The file name is the agent name: lowercase letters, digits, hyphens." in description
    assert ("A specialist gets no MCP servers of its own: compose it from the harness tools "
            "(`tools.add`).") in description
    assert ("A project specialist's permissions can only tighten the machine's." in description) == bool(project)


def test_builtin_names_exist_once_in_start_order():
    assert BUILTIN_AGENT_NAMES == ("explore", "web-researcher", "cruncher", "search-verifier")
    assert INVALID_AGENT_NAME == "the file name is the agent name: lowercase letters, digits, hyphens"
    assert ROSTER_NAME_CAP == 24


def _names(n):
    return [f"a{i:02d}" for i in range(n)]


@pytest.mark.parametrize("n,tail", [(24, ""), (25, " and 1 more"), (30, " and 6 more")])
def test_the_description_caps_the_roster_at_24_names(n, tail):
    description = _text(agents=_names(n))

    assert f"Available agents: {', '.join(_names(24))}{tail}." in description
    assert ("a24" in description) is False
    assert (" more" in description) == bool(tail)


def test_the_not_found_texts_use_the_same_cap():
    expected = f"Available: {', '.join(_names(24))} and 6 more"
    for exc in (KeyError("x"), ValueError("")):
        tool = _make_agent_tool(runner=AsyncMock(side_effect=exc), agents=_names(30))
        result = asyncio.run(tool._execute(agent_id="nope", task="t"))
        assert expected in result.error


def test_a_live_iterable_roster_is_read_fresh():
    class Live:
        def __init__(self): self.names = ["explore"]
        def __iter__(self): return iter(self.names)
        def __len__(self): return len(self.names)

    live = Live()
    tool = _make_agent_tool(agents=live)
    assert "explore." in tool.info().description
    live.names.append("late-arrival")
    assert "explore, late-arrival." in tool.info().description


@pytest.mark.parametrize("agents", [None, []])
def test_no_roster_renders_none_configured(agents):
    tool = AgentTool(agent_runner=AsyncMock(), available_agents=agents)
    assert "Available agents: none configured." in tool.info().description


def test_a_roster_empty_at_construction_shows_names_added_later():
    roster: list[str] = []
    tool = AgentTool(agent_runner=AsyncMock(), available_agents=roster)
    roster.append("late-agent")
    assert "late-agent" in tool.info().description
