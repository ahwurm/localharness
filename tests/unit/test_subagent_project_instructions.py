"""One project instructions handle reaches every loop a session builds, at any depth (11-05; D-51).

`start` hands the parent's handle to `make_explore_agent_runner(project_instructions=)`; it rides the same
chain `template_margin_ratio` does (`_run_agent` -> dispatchers -> `dispatch_cruncher_subagent` ->
`_run_chunk_summarizer` / `_cruncher_combine_turn`) and `_build_child_agent_tool` re-threads it into the
nested runner. The recorded-loops fixture is the one test_subagent_effort.py uses: it patches
`localharness.agent.loop.AgentLoop` with a kwargs recorder, so no model is ever called.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from localharness.agent import subagent as sub
from localharness.agent.permissions import PermissionEvaluator
from localharness.agent.subagent import make_explore_agent_runner
from localharness.config.models import AgentConfig, BudgetConfig, PermissionConfig
from localharness.tools.builtin import register_builtin_tools
from localharness.tools.registry import ToolRegistry

SUBAGENT_SRC = Path(sub.__file__)
SENTINEL = object()


@pytest.fixture(autouse=True)
def _ledger_off_the_repo(monkeypatch, tmp_path):
    """The search-verifier dispatch appends a ledger row; keep it out of ./reports."""
    monkeypatch.setenv("LOCALHARNESS_VERIFICATION_LEDGER_DIR", str(tmp_path / "ledger"))


@pytest.fixture
def recorded_loops(monkeypatch):
    """Record the kwargs of every AgentLoop construction; the turn itself is a no-op."""
    calls: list[dict] = []

    class _RecordingLoop:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.current_session_id = None

        def action_limit(self):
            return 0

        def time_limit(self):
            return 0.0

        async def run_turn(self, task, *a, **k):
            return ""

    monkeypatch.setattr("localharness.agent.loop.AgentLoop", _RecordingLoop)
    return calls


async def _builtin_registry() -> ToolRegistry:
    reg = ToolRegistry()
    await register_builtin_tools(reg)
    return reg


def _yaml_agent(name: str = "yaml-child") -> AgentConfig:
    return AgentConfig(
        name=name, role="test child", effort="inherit",
        permissions=PermissionConfig(budget=BudgetConfig()),
    )


async def _runner(bus, **kw):
    base = await _builtin_registry()
    return make_explore_agent_runner(
        llm=object(), bus=bus, base_registry=base, permission_evaluator=PermissionEvaluator(),
        get_parent_session_id=lambda: "parent-sess", **kw,
    )


_DISPATCHERS = [
    sub.dispatch_config_subagent,
    sub.dispatch_explore_subagent,
    sub.dispatch_web_subagent,
    sub.dispatch_search_verifier_subagent,
    sub.dispatch_cruncher_subagent,
]


@pytest.mark.parametrize("fn", [*_DISPATCHERS, sub._run_chunk_summarizer, sub._cruncher_combine_turn,
                                make_explore_agent_runner], ids=lambda f: f.__name__)
def test_every_link_of_the_chain_takes_the_keyword_only_with_default_none(fn):
    param = inspect.signature(fn).parameters["project_instructions"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is None


_CASES = [
    ("yaml-child", "do it"),
    ("explore", "find X"),
    ("web-researcher", "research X"),
    ("search-verifier", "claim: X\nentity: X\nsource_url: https://n.test/x"),
    ("cruncher", "what does it say?"),
]


async def _dispatch(bus, agent_id, task, **kw):
    """Run one child through a runner built with `kw`; a cruncher reads a granted handle."""
    if agent_id == "cruncher":
        from localharness.agent.context import ContentStore

        store = ContentStore()
        handle = store.put("a granted document")
        base = await _builtin_registry()
        runner = make_explore_agent_runner(
            llm=object(), bus=bus, base_registry=base, permission_evaluator=PermissionEvaluator(),
            get_parent_session_id=lambda: "parent-sess", parent_store=store, **kw,
        )
        await runner(agent_id, task, [handle])
        return
    runner = await _runner(bus, load_agent=lambda n: _yaml_agent(n), **kw)
    await runner(agent_id, task)


@pytest.mark.parametrize(("agent_id", "task"), _CASES)
async def test_the_handle_reaches_every_loop_of_each_dispatcher(recorded_loops, bus, agent_id, task):
    await _dispatch(bus, agent_id, task, project_instructions=SENTINEL)
    assert recorded_loops, "AgentLoop patch did not take effect at the call site"
    assert all(c["project_instructions"] is SENTINEL for c in recorded_loops)
    if agent_id == "cruncher":
        assert len(recorded_loops) >= 2, "the leaf AND the combine loop must both be built"


@pytest.mark.parametrize(("agent_id", "task"), _CASES)
async def test_a_runner_built_without_the_keyword_gives_every_child_none(recorded_loops, bus, agent_id, task):
    await _dispatch(bus, agent_id, task)
    assert recorded_loops
    assert all(c["project_instructions"] is None for c in recorded_loops)


@pytest.mark.parametrize(("agent_id", "task"), _CASES)
async def test_an_explicit_none_gives_every_child_none(recorded_loops, bus, agent_id, task):
    await _dispatch(bus, agent_id, task, project_instructions=None)
    assert recorded_loops
    assert all(c["project_instructions"] is None for c in recorded_loops)


async def test_a_grandchild_gets_the_same_handle(recorded_loops, bus):
    runner = await _runner(bus, project_instructions=SENTINEL, max_subagent_depth=3)
    await runner("web-researcher", "research X")
    agent_tool = recorded_loops[0]["tool_registry"]._tools["global"]["agent"]
    before = len(recorded_loops)
    await agent_tool._agent_runner("search-verifier", "claim: X\nentity: X\nsource_url: https://n.test/x")
    assert len(recorded_loops) > before
    assert recorded_loops[-1]["project_instructions"] is SENTINEL


async def test_a_grandchild_of_a_runner_with_none_has_none(recorded_loops, bus):
    runner = await _runner(bus, max_subagent_depth=3)
    await runner("web-researcher", "research X")
    agent_tool = recorded_loops[0]["tool_registry"]._tools["global"]["agent"]
    await agent_tool._agent_runner("search-verifier", "claim: X\nentity: X\nsource_url: https://n.test/x")
    assert recorded_loops[-1]["project_instructions"] is None


def test_the_keyword_is_forwarded_wherever_the_template_margin_is():
    """The same sites carry both: a hop that forwards one and drops the other is the bug this pins."""
    source = SUBAGENT_SRC.read_text(encoding="utf-8")
    assert source.count("project_instructions=project_instructions") == source.count(
        "template_margin_ratio=template_margin_ratio") > 0
    assert source.count("project_instructions: Any = None") == source.count(
        "template_margin_ratio: float | None = None") > 0


def test_every_agent_loop_site_passes_the_handle():
    import ast

    tree = ast.parse(SUBAGENT_SRC.read_text(encoding="utf-8"))
    sites = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "AgentLoop"]
    assert sites, "no AgentLoop( call found"
    for call in sites:
        assert "project_instructions" in {k.arg for k in call.keywords}, f"line {call.lineno}"
