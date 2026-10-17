"""A child runs in the parent's live window, read once at spawn."""
from __future__ import annotations

import types

import pytest

import localharness.agent.subagent as subagent
from localharness.tools.registry import ToolRegistry


def _runner(window, **kw):
    return subagent.make_explore_agent_runner(
        llm=object(), bus=object(), base_registry=ToolRegistry(), permission_evaluator=object(),
        get_parent_session_id=lambda: "sid", window=window, max_subagent_depth=2, **kw,
    )


def _record(monkeypatch, name, seen):
    async def _fake(task, **kwargs):
        seen[name] = kwargs["context_manager"].max_context_tokens
        seen[name + ":tool"] = kwargs.get("child_agent_tool")
        return "ok"
    monkeypatch.setattr(subagent, name, _fake)


@pytest.mark.asyncio
async def test_children_spawn_at_the_refit_window(monkeypatch):
    root = types.SimpleNamespace(max_context_tokens=4_096)
    seen: dict = {}
    for n in ("dispatch_explore_subagent", "dispatch_web_subagent",
              "dispatch_search_verifier_subagent", "dispatch_cruncher_subagent"):
        _record(monkeypatch, n, seen)
    runner = _runner(lambda: root.max_context_tokens)
    root.max_context_tokens = 66_560

    await runner("explore", "t")
    await runner("cruncher", "t")
    await runner("web-researcher", "t")
    await seen["dispatch_web_subagent:tool"]._agent_runner("search-verifier", "t")

    assert seen["dispatch_explore_subagent"] == 66_560
    assert seen["dispatch_cruncher_subagent"] == 66_560
    assert seen["dispatch_web_subagent"] == 66_560
    assert seen["dispatch_search_verifier_subagent"] == 66_560


@pytest.mark.asyncio
async def test_a_refit_during_a_child_leaves_that_child(monkeypatch):
    root = types.SimpleNamespace(max_context_tokens=4_096)
    held: dict = {}

    async def _fake(task, **kwargs):
        ctx = kwargs["context_manager"]
        root.max_context_tokens = 66_560
        held["ctx"] = ctx
        return "ok"
    monkeypatch.setattr(subagent, "dispatch_explore_subagent", _fake)
    runner = _runner(lambda: root.max_context_tokens)

    await runner("explore", "t")

    assert held["ctx"].max_context_tokens == 4_096
    await runner("explore", "t")
    assert held["ctx"].max_context_tokens == 66_560


@pytest.mark.asyncio
@pytest.mark.parametrize("window", [None, lambda: None])
async def test_no_window_leaves_the_child_default(monkeypatch, window):
    from localharness.agent.context import ContextManager
    seen: dict = {}
    _record(monkeypatch, "dispatch_explore_subagent", seen)
    await _runner(window)("explore", "t")
    assert seen["dispatch_explore_subagent"] == ContextManager().max_context_tokens


class _Counter:
    """The system prompt costs 4,370 tokens; the schemas (a JSON list) cost nothing."""

    def estimate_messages(self, messages):
        return 0 if messages[0]["content"].startswith("[") else 4_370


async def _agent_tool(window, bus, monkeypatch, ran):
    from localharness.agent.loop import AgentLoop
    from localharness.agent.permissions import PermissionEvaluator
    from localharness.tools.builtin import register_builtin_tools
    from localharness.tools.builtin.agent_tool import AgentTool

    async def _turn(self, task, *a, **k):
        ran.append(self._config.name)
        return "done"
    monkeypatch.setattr(AgentLoop, "run_turn", _turn)
    reg = ToolRegistry()
    await register_builtin_tools(reg)
    runner = subagent.make_explore_agent_runner(
        llm=object(), bus=bus, base_registry=reg, permission_evaluator=PermissionEvaluator(),
        get_parent_session_id=lambda: "sid", token_counter=_Counter(), window=window,
    )
    return AgentTool(agent_runner=runner)


@pytest.mark.asyncio
async def test_a_child_below_its_floor_is_not_delegated(bus, monkeypatch):
    ran: list = []
    tool = await _agent_tool(lambda: 4_096, bus, monkeypatch, ran)
    res = await tool.run(agent_id="web-researcher", task="t")
    assert not res.success
    assert res.error == "web-researcher needs an 8,448-token window and the server serves 4,096; not delegated."
    assert ran == []


@pytest.mark.asyncio
async def test_a_child_at_its_floor_runs(bus, monkeypatch):
    ran: list = []
    tool = await _agent_tool(lambda: 8_448, bus, monkeypatch, ran)
    res = await tool.run(agent_id="web-researcher", task="t")
    assert res.success
    assert "web-researcher" in ran


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_id", ["explore", "search-verifier"])
async def test_every_builtin_is_checked_before_its_first_turn(bus, monkeypatch, agent_id):
    ran: list = []
    res = await (await _agent_tool(lambda: 4_096, bus, monkeypatch, ran)).run(agent_id=agent_id, task="t")
    assert res.error == f"{agent_id} needs an 8,448-token window and the server serves 4,096; not delegated."
    assert ran == []


@pytest.mark.asyncio
async def test_the_cruncher_is_checked_before_its_first_leaf(bus, monkeypatch):
    ran: list = []
    res = await (await _agent_tool(lambda: 4_096, bus, monkeypatch, ran)).run(agent_id="cruncher", task="t")
    assert res.error == "cruncher needs an 8,448-token window and the server serves 4,096; not delegated."
    assert ran == []


@pytest.mark.asyncio
async def test_a_config_child_is_checked(bus, monkeypatch):
    from localharness.config.models import AgentConfig
    ran: list = []
    tool = await _agent_tool(lambda: 4_096, bus, monkeypatch, ran)
    tool._agent_runner = subagent.make_explore_agent_runner(
        llm=object(), bus=bus, base_registry=ToolRegistry(), permission_evaluator=object(),
        get_parent_session_id=lambda: "sid", token_counter=_Counter(), window=lambda: 4_096,
        load_agent=lambda n: AgentConfig(name=n, role="r"),
    )
    res = await tool.run(agent_id="scribe", task="t")
    assert res.error == "scribe needs an 8,448-token window and the server serves 4,096; not delegated."


@pytest.mark.asyncio
async def test_an_unknown_window_runs_the_child(bus, monkeypatch):
    ran: list = []
    res = await (await _agent_tool(lambda: None, bus, monkeypatch, ran)).run(agent_id="explore", task="t")
    assert res.success and ran == ["explore"]


@pytest.mark.asyncio
async def test_builtin_role_floors_cover_the_three_roles():
    from localharness.tools.builtin import register_builtin_tools
    reg = ToolRegistry()
    await register_builtin_tools(reg)
    floors = subagent.builtin_role_floors(reg, _Counter(), object(), None)
    assert set(floors) == {"explore", "web-researcher", "search-verifier"}
    assert set(floors.values()) == {8_448}
