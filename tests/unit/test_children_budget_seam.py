"""The five built-in child budgets live in `org.children_budget` and ride into every builder (08-06).

Builders: each takes its entry as the keyword-only `budget=`, folds `kill_file` in, never sets
`max_tool_calls`, and with no budget builds on `ChildrenBudgetConfig().for_name(<its own entry>)`
so the builder alone owns the org rung (D-13, D-14).

Runner chain: `children_budget` rides `make_explore_agent_runner` like `template_margin_ratio`, so
every child loop the runner builds carries its OWN entry of a non-default block (D-14, D-48, D-58).
The recorded-loops fixture is the one `test_subagent_effort.py` uses: every `AgentLoop(**kwargs)` is
captured and no model is ever called.
"""
from __future__ import annotations

import inspect

import pytest

from localharness.agent import subagent as sub
from localharness.agent.permissions import PermissionEvaluator
from localharness.agent.subagent import make_explore_agent_runner
from localharness.config.models import ChildBudget, ChildrenBudgetConfig
from localharness.tools.builtin import register_builtin_tools
from localharness.tools.registry import ToolRegistry

_BUILDERS = [
    ("explore", sub.build_explore_config, (8, 3.0)),
    ("web-researcher", sub.build_web_researcher_config, (56, 20.0)),
    ("search-verifier", sub.build_search_verifier_config, (12, 6.0)),
    ("cruncher", sub.build_cruncher_config, (6, 20.0)),
    ("chunk-summarizer", sub.build_chunk_summarizer_config, (3, 6.0)),
]


@pytest.mark.parametrize("name,builder,expected", _BUILDERS)
def test_a_builder_with_no_budget_builds_todays_numbers(name, builder, expected):
    budget = builder().permissions.budget
    assert (budget.max_actions, budget.max_duration_minutes) == expected
    assert budget.max_tool_calls is None
    assert budget.kill_file is None


@pytest.mark.parametrize("name,builder,expected", _BUILDERS)
def test_a_builder_with_no_budget_equals_the_schema_entry(name, builder, expected):
    entry = ChildrenBudgetConfig().for_name(name)
    budget = builder().permissions.budget
    assert (budget.max_actions, budget.max_duration_minutes) == (entry.actions, entry.minutes)


@pytest.mark.parametrize("name,builder,expected", _BUILDERS)
def test_a_given_budget_replaces_the_default_and_kill_file_still_folds_in(name, builder, expected):
    budget = builder(kill_file="/k", budget=ChildBudget(actions=5, minutes=2.0)).permissions.budget
    assert (budget.max_actions, budget.max_duration_minutes) == (5, 2.0)
    assert budget.kill_file == "/k"
    assert budget.max_tool_calls is None


@pytest.mark.parametrize("name,builder,expected", _BUILDERS)
def test_budget_is_keyword_only_with_default_none(name, builder, expected):
    param = inspect.signature(builder).parameters["budget"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is None


def test_a_renamed_build_keeps_its_builtin_entry():
    cfg = sub.build_web_researcher_config(name="web_researcher_2")
    assert cfg.permissions.budget.max_actions == ChildrenBudgetConfig().web_researcher.actions


def test_the_budget_constants_are_gone():
    gone = [n for n in vars(sub)
            if n.endswith(("_MAX_ACTIONS", "_MAX_DURATION_MINUTES", "_MAX_TOOL_CALLS"))]
    assert gone == []


# ---------------------------------------------------------------------------
# The runner chain
# ---------------------------------------------------------------------------

_BLOCK = ChildrenBudgetConfig(
    explore=ChildBudget(actions=5, minutes=2.0),
    web_researcher=ChildBudget(actions=30, minutes=10.0),
    search_verifier=ChildBudget(actions=7, minutes=4.0),
    cruncher=ChildBudget(actions=4, minutes=9.0),
    chunk_summarizer=ChildBudget(actions=2, minutes=3.0),
)
_VERIFY_TASK = "claim: X\nentity: X\nsource_url: https://n.test/x"


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

        async def run_turn(self, task, *a, **k):
            return ""

        # the loop surface later plans of phase 8 read off a child
        def action_limit(self):
            return 0

        def time_limit(self):
            return 0.0

        def remaining_budget(self):
            return (0, 0.0)

    monkeypatch.setattr("localharness.agent.loop.AgentLoop", _RecordingLoop)
    return calls


async def _builtin_registry() -> ToolRegistry:
    reg = ToolRegistry()
    await register_builtin_tools(reg)
    return reg


async def _runner(bus, **kw):
    return make_explore_agent_runner(
        llm=object(), bus=bus, base_registry=await _builtin_registry(),
        permission_evaluator=PermissionEvaluator(),
        get_parent_session_id=lambda: "parent-sess", **kw,
    )


def _pair(call: dict) -> tuple[int, float]:
    budget = call["config"].permissions.budget
    assert budget.max_tool_calls is None
    return (budget.max_actions, budget.max_duration_minutes)


_HOOKS = [
    pytest.param(None, id="no-hook-the-bench-shape"),
    pytest.param(lambda n, base: base, id="start-shape-hook"),
]


@pytest.mark.parametrize("hook", _HOOKS)
@pytest.mark.parametrize(
    ("agent_id", "task", "expected"),
    [
        ("explore", "find X", (5, 2.0)),
        ("web-researcher", "research X", (30, 10.0)),
        ("search-verifier", _VERIFY_TASK, (7, 4.0)),
    ],
)
async def test_a_built_in_child_carries_its_own_entry_of_the_block(
    recorded_loops, bus, hook, agent_id, task, expected
):
    runner = await _runner(bus, children_budget=_BLOCK, load_builtin_override=hook)
    await runner(agent_id, task)
    assert recorded_loops
    assert {_pair(c) for c in recorded_loops} == {expected}


@pytest.mark.parametrize("hook", _HOOKS)
async def test_every_web_researcher_attempt_carries_the_block(recorded_loops, bus, hook):
    """The fake makes zero tool calls, so the dispatcher builds a fresh loop for the retry too."""
    runner = await _runner(bus, children_budget=_BLOCK, load_builtin_override=hook)
    await runner("web-researcher", "research X")
    assert len(recorded_loops) == 1 + sub.WEB_ZERO_CALL_RETRIES >= 2
    assert [_pair(c) for c in recorded_loops] == [(30, 10.0)] * len(recorded_loops)


@pytest.mark.parametrize("hook", _HOOKS)
async def test_the_nested_search_verifier_carries_the_block(recorded_loops, bus, hook):
    runner = await _runner(
        bus, children_budget=_BLOCK, max_subagent_depth=3, load_builtin_override=hook)
    await runner("web-researcher", "research X")
    assert _pair(recorded_loops[0]) == (30, 10.0)
    agent_tool = recorded_loops[0]["tool_registry"]._tools["global"]["agent"]
    before = len(recorded_loops)
    await agent_tool._agent_runner("search-verifier", _VERIFY_TASK)
    assert len(recorded_loops) > before
    assert _pair(recorded_loops[-1]) == (7, 4.0)
    assert recorded_loops[-1]["config"].name == "search-verifier"


@pytest.mark.parametrize("hook", _HOOKS)
async def test_the_cruncher_leaves_and_combine_turns_carry_the_block(recorded_loops, bus, hook):
    from localharness.agent.context import ContentStore

    store = ContentStore()
    handle = store.put("a granted document")
    runner = await _runner(
        bus, children_budget=_BLOCK, parent_store=store, load_builtin_override=hook)
    await runner("cruncher", "what does it say?", [handle])
    by_name: dict[str, set] = {}
    for c in recorded_loops:
        by_name.setdefault(c["config"].name, set()).add(_pair(c))
    assert by_name == {"chunk-summarizer": {(2, 3.0)}, "cruncher": {(4, 9.0)}}


async def test_a_partial_combine_level_carries_the_block_too(monkeypatch, bus):
    """A broad query reduces in levels; the partial combine turns are cruncher loops as well."""
    from localharness.agent.context import ContentStore

    calls: list[dict] = []

    class _Loop:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.current_session_id = None
            self._name = kwargs["config"].name

        async def run_turn(self, task, *a, **k):
            return "extract " * 150 if self._name == "chunk-summarizer" else "combined"

    monkeypatch.setattr("localharness.agent.loop.AgentLoop", _Loop)
    store = ContentStore()
    handle = store.put("section text. " * 4000)
    runner = await _runner(bus, children_budget=_BLOCK, parent_store=store, max_context_tokens=2000)
    await runner("cruncher", "what does it say?", [handle])
    leaves = [c for c in calls if c["config"].name == "chunk-summarizer"]
    combines = [c for c in calls if c["config"].name == "cruncher"]
    assert len(leaves) >= 2
    assert len(combines) >= 2, "expected at least one partial level plus the final combine"
    assert {_pair(c) for c in leaves} == {(2, 3.0)}
    assert {_pair(c) for c in combines} == {(4, 9.0)}


@pytest.mark.parametrize("hook", _HOOKS)
@pytest.mark.parametrize(
    ("agent_id", "task", "expected"),
    [
        ("explore", "find X", (8, 3.0)),
        ("web-researcher", "research X", (56, 20.0)),
        ("search-verifier", _VERIFY_TASK, (12, 6.0)),
    ],
)
async def test_no_block_is_todays_numbers(recorded_loops, bus, hook, agent_id, task, expected):
    runner = await _runner(bus, load_builtin_override=hook)
    await runner(agent_id, task)
    assert {_pair(c) for c in recorded_loops} == {expected}


async def test_no_block_is_todays_numbers_for_the_cruncher(recorded_loops, bus):
    from localharness.agent.context import ContentStore

    store = ContentStore()
    handle = store.put("a granted document")
    runner = await _runner(bus, parent_store=store)
    await runner("cruncher", "what does it say?", [handle])
    assert {c["config"].name: _pair(c) for c in recorded_loops} == {
        "chunk-summarizer": (3, 6.0), "cruncher": (6, 20.0)}


async def test_the_hook_receives_a_base_already_built_on_the_block(recorded_loops, bus):
    seen: dict[str, tuple[int, float]] = {}

    def hook(name, base):
        seen[name] = (base.permissions.budget.max_actions, base.permissions.budget.max_duration_minutes)
        return base

    runner = await _runner(bus, children_budget=_BLOCK, load_builtin_override=hook)
    await runner("web-researcher", "research X")
    assert seen == {"web-researcher": (30, 10.0)}


_BLOCK_TAKERS = [
    sub.make_explore_agent_runner,
    sub.dispatch_explore_subagent,
    sub.dispatch_web_subagent,
    sub.dispatch_search_verifier_subagent,
    sub.dispatch_cruncher_subagent,
    sub._run_chunk_summarizer,
    sub._cruncher_combine_turn,
]


@pytest.mark.parametrize("fn", _BLOCK_TAKERS, ids=lambda f: f.__name__)
def test_children_budget_is_keyword_only_with_default_none(fn):
    param = inspect.signature(fn).parameters["children_budget"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is None
    names = list(inspect.signature(fn).parameters)
    assert names.index("children_budget") > names.index("server_recovery")
