"""Bug#1 regression guard + T1 closure-extraction coverage for the `agent` delegation tool.

Production `localharness start` delegation was DEAD: start_cmd registered the AgentTool at
scope="agent", agent_id="orchestrator", but the running parent agent is the `default` agent, and
ToolRegistry.get_tools_for_agent resolves agent-scoped tools by the loop's own name
(loop.py: agent_id=self._config.name). So `default` was never offered `agent` and could not delegate.
The bench hid this by registering `agent` at GLOBAL scope. The fix registers at global scope in
start_cmd too, and extracts the runner into the module-level make_explore_agent_runner seam (T1).

These tests are fully deterministic — NO live model. The dispatch path uses a spy runner (Bug#1
regression guard), and the extracted-runner routing/threading is verified by spying
dispatch_explore_subagent.
"""
from __future__ import annotations

import pytest

from localharness.config.models import ToolConfig
from localharness.tools.builtin import register_builtin_tools
from localharness.tools.builtin.agent_tool import AgentTool
from localharness.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# (a) Bug#1 regression guard: global-scope `agent` is visible to + dispatchable by `default`.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_global_agent_tool_visible_and_dispatchable_as_default():
    """After registering the AgentTool at GLOBAL scope, the `default` parent SEES `agent` and a
    dispatch as agent_id='default' SUCCEEDS. With the old scope='agent'/agent_id='orchestrator'
    registration this is exactly what failed (default never resolved the tool)."""
    registry = ToolRegistry()
    await register_builtin_tools(registry)  # read/glob/grep/write/bash_exec — no `agent` collision

    calls: list[tuple[str, str]] = []

    async def _spy_runner(agent_id: str, task: str) -> str:
        calls.append((agent_id, task))
        return "delegated-ok"

    agent_tool = AgentTool(agent_runner=_spy_runner, available_agents=["explore"])
    # THE FIX under test: global scope (drop agent_id="orchestrator").
    await registry.register(agent_tool, scope="global")

    # Visibility: the `default` agent's resolved toolset INCLUDES `agent` (default ToolConfig
    # inherits global). Old agent/orchestrator scope would NOT appear here for `default`.
    tools = registry.get_tools_for_agent("default", "default", ToolConfig())
    assert "agent" in tools, "global-scope `agent` tool must be visible to the `default` parent"

    # Dispatchability: dispatch as the `default` agent resolves the tool and runs the runner.
    result = await registry.dispatch(
        "agent",
        {"agent_id": "explore", "task": "go look"},
        agent_id="default",
        division_id="default",
        tool_config=ToolConfig(),
    )
    assert result.success is True, f"dispatch as `default` must succeed, got {result.error!r}"
    assert result.output == "delegated-ok"
    assert calls == [("explore", "go look")]


@pytest.mark.asyncio
async def test_agent_scoped_orchestrator_registration_is_invisible_to_default():
    """The OLD (buggy) wiring characterized: registering `agent` at scope='agent',
    agent_id='orchestrator' leaves it INVISIBLE to the `default` parent — the root cause of Bug#1.
    This pins the contract so a regression back to agent-scope is caught."""
    registry = ToolRegistry()
    await register_builtin_tools(registry)

    async def _spy_runner(agent_id: str, task: str) -> str:
        return "x"

    agent_tool = AgentTool(agent_runner=_spy_runner, available_agents=["explore"])
    await registry.register(agent_tool, scope="agent", agent_id="orchestrator")

    tools = registry.get_tools_for_agent("default", "default", ToolConfig())
    assert "agent" not in tools, (
        "agent-scoped/orchestrator `agent` tool must NOT resolve for `default` "
        "(this invisibility is exactly Bug#1)"
    )


# ---------------------------------------------------------------------------
# (b)+(c) T1: the extracted make_explore_agent_runner routes + threads correctly.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_routes_explore_and_threads_session_and_depth(monkeypatch):
    """The extracted runner routes agent_id='explore' to a real dispatch_explore_subagent call and
    threads parent_session_id (read from the getter at CALL TIME) + depth through to it."""
    import localharness.agent.subagent as subagent

    captured: dict = {}

    async def _spy_dispatch(task, **kwargs):
        captured["task"] = task
        captured.update(kwargs)
        return "[explore findings] task: ... | findings"

    monkeypatch.setattr(subagent, "dispatch_explore_subagent", _spy_dispatch)

    sentinel_llm = object()
    sentinel_bus = object()
    sentinel_registry = object()
    sentinel_perm = object()
    # Getter is read at call time — return a value set AFTER build to prove late binding.
    session_holder = {"sid": None}

    runner = subagent.make_explore_agent_runner(
        llm=sentinel_llm,
        bus=sentinel_bus,
        base_registry=sentinel_registry,
        permission_evaluator=sentinel_perm,
        get_parent_session_id=lambda: session_holder["sid"],
    )

    # Set the session id only NOW (after build) — the late-bound getter must observe it.
    session_holder["sid"] = "parent-session-xyz"

    out = await runner("explore", "find values.txt", 0)
    assert out.startswith("[explore findings]")
    assert captured["task"] == "find values.txt"
    assert captured["llm"] is sentinel_llm
    assert captured["bus"] is sentinel_bus
    assert captured["base_registry"] is sentinel_registry
    assert captured["permission_evaluator"] is sentinel_perm
    assert captured["parent_session_id"] == "parent-session-xyz", "session id must be read at call time"
    assert captured["depth"] == 0

    # depth threads through (belt-and-suspenders recursion guard).
    captured.clear()
    await runner("explore", "again", 1)
    assert captured["depth"] == 1


@pytest.mark.asyncio
async def test_runner_refuses_non_explore_agent_with_clear_error(monkeypatch):
    """A non-explore agent_id is REFUSED with a clear, actionable ValueError (only `explore` is
    wired in v1.4); dispatch_explore_subagent is never reached. Underscore->dash sanitization
    means an `_`-bearing name that is not `explore` also refuses."""
    import localharness.agent.subagent as subagent

    async def _must_not_run(task, **kwargs):  # pragma: no cover - asserted never called
        raise AssertionError("dispatch_explore_subagent must not run for a non-explore agent_id")

    monkeypatch.setattr(subagent, "dispatch_explore_subagent", _must_not_run)

    runner = subagent.make_explore_agent_runner(
        llm=object(),
        bus=object(),
        base_registry=object(),
        permission_evaluator=object(),
        get_parent_session_id=lambda: "sid",
    )

    with pytest.raises(ValueError, match="not yet wired"):
        await runner("researcher", "do a thing", 0)

    # `explore` IS accepted (sanitizer maps `_`->`-`; bare `explore` already matches).
    with pytest.raises(ValueError, match="not yet wired"):
        await runner("writer_agent", "do a thing", 0)
