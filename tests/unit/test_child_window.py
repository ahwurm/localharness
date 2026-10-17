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
