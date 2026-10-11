"""Children are born with the level the person set, read at each dispatch (03-04, D-11, D-34).

Two threads ride the same plumbing from `start` to every child `AgentLoop(`:

- the parent's `(level, rung)`, through `make_explore_agent_runner(effort_fn=)` — a CALLABLE read at
  each dispatch, so a `/effort` between two dispatches reaches the second child (EFFORT-06);
- the parent loop's `template_margin_ratio`, a plain value (D-34, the AUTO-07 carry-over).

The recorded-loops fixture is the one `test_subagent_state_dir.py` uses: the dispatchers import
`AgentLoop` inside their bodies, so patching the attribute on the `localharness.agent.loop` module
captures every construction's keyword arguments and no model is ever called.
"""
from __future__ import annotations

import inspect

import pytest

from localharness.agent import subagent as sub
from localharness.agent.permissions import PermissionEvaluator
from localharness.agent.subagent import child_effort, make_explore_agent_runner
from localharness.config.models import AgentConfig, BudgetConfig, PermissionConfig
from localharness.tools.builtin import register_builtin_tools
from localharness.tools.registry import ToolRegistry


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


def _yaml_agent(effort: str = "inherit", name: str = "yaml-child") -> AgentConfig:
    return AgentConfig(
        name=name, role="test child", effort=effort,
        permissions=PermissionConfig(budget=BudgetConfig()),
    )


async def _runner(bus, **kw):
    base = await _builtin_registry()
    return make_explore_agent_runner(
        llm=object(), bus=bus, base_registry=base, permission_evaluator=PermissionEvaluator(),
        get_parent_session_id=lambda: "parent-sess", **kw,
    )


# ---------------------------------------------------------------------------
# child_effort: the pure rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("parent", "own", "expected"),
    [
        (("low", "session"), "max", ("low", "session")),     # the session rung outranks the file
        (("high", "agent"), "max", ("max", "agent")),         # no session: the child's own level
        (("high", "endpoint"), "inherit", ("high", "endpoint")),
        (("high", "org"), None, ("high", "org")),
        (("high", "default"), "low", ("low", "agent")),
        (None, "max", None),                                   # no parent level: no child level
        ((None, None), "max", None),
    ],
)
def test_child_effort_rule(parent, own, expected):
    assert child_effort(parent, own) == expected


# ---------------------------------------------------------------------------
# The runner reads effort_fn at EACH dispatch (EFFORT-06 ordering)
# ---------------------------------------------------------------------------

async def test_child_is_built_with_the_parents_live_level(recorded_loops, bus):
    state = {"effort": ("low", "session")}
    runner = await _runner(bus, effort_fn=lambda: state["effort"], effort_ratios=(0.2, 0.6))

    await runner("explore", "find X")
    state["effort"] = ("max", "session")
    await runner("explore", "find Y")

    assert [c["effort"] for c in recorded_loops] == [("low", "session"), ("max", "session")]
    assert all(c["effort_ratios"] == (0.2, 0.6) for c in recorded_loops)


async def test_effort_fn_is_called_per_dispatch_never_captured(recorded_loops, bus):
    """T-3-23: a captured value would freeze children at the start-time level."""
    seen: list[int] = []

    def _fn():
        seen.append(1)
        return ("high", "default")

    runner = await _runner(bus, effort_fn=_fn)
    assert seen == [], "effort_fn was called when the runner was BUILT"
    await runner("explore", "a")
    await runner("explore", "b")
    assert len(seen) == 2


async def test_a_yaml_agents_own_level_is_kept_unless_the_session_set_one(recorded_loops, bus):
    state = {"effort": ("high", "agent")}
    runner = await _runner(
        bus, effort_fn=lambda: state["effort"], load_agent=lambda n: _yaml_agent("max", n),
    )
    await runner("yaml-child", "do it")
    state["effort"] = ("low", "session")
    await runner("yaml-child", "do it again")

    assert recorded_loops[0]["effort"] == ("max", "agent")
    assert recorded_loops[1]["effort"] == ("low", "session"), "T-3-22: the session outranks the file"


async def test_a_built_ins_overlay_level_is_its_own(recorded_loops, bus):
    """A built-in tuned through agents/explore.yaml can carry its own level, like any agent file."""
    runner = await _runner(
        bus, effort_fn=lambda: ("high", "endpoint"),
        load_builtin_override=lambda name, base: base.model_copy(update={"effort": "low"}),
    )
    await runner("explore", "find X")
    assert recorded_loops[0]["effort"] == ("low", "agent")


async def test_a_grandchild_reads_the_live_level_too(recorded_loops, bus):
    """web-researcher is non-leaf: its nested runner gets the SAME callable, read at ITS dispatch."""
    state = {"effort": ("low", "session")}
    runner = await _runner(
        bus, effort_fn=lambda: state["effort"], effort_ratios=(0.1, 0.4),
        template_margin_ratio=0.03, max_subagent_depth=3,
    )
    await runner("web-researcher", "research X")
    web_loop = recorded_loops[0]
    assert web_loop["effort"] == ("low", "session")

    # The child's own `agent` tool wraps the nested runner; the person moves the level before the
    # web-researcher decides to delegate.
    agent_tool = web_loop["tool_registry"]._tools["global"]["agent"]
    state["effort"] = ("max", "session")
    await agent_tool._agent_runner("search-verifier", "claim: X\nentity: X\nsource_url: https://n.test/x")

    grandchild = recorded_loops[-1]
    assert grandchild["effort"] == ("max", "session"), "the grandchild captured the level"
    assert grandchild["effort_ratios"] == (0.1, 0.4)
    assert grandchild["template_margin_ratio"] == 0.03


async def test_no_effort_fn_keeps_every_child_at_effort_none(recorded_loops, bus):
    """EFFORT-06 empty case: a runner built without the keywords is today's runner."""
    runner = await _runner(bus, load_agent=lambda n: _yaml_agent("max", n))
    await runner("explore", "a")
    await runner("yaml-child", "b")
    await runner("web-researcher", "c")
    assert len(recorded_loops) >= 3
    for c in recorded_loops:
        assert c["effort"] is None
        assert c["effort_ratios"] is None
        assert c["template_margin_ratio"] is None


async def test_a_parent_without_a_level_leaves_the_child_at_none(recorded_loops, bus):
    runner = await _runner(bus, effort_fn=lambda: (None, None))
    await runner("explore", "a")
    assert recorded_loops[0]["effort"] is None


# ---------------------------------------------------------------------------
# Every dispatcher takes the keywords (signature) and the cruncher's loops carry them
# ---------------------------------------------------------------------------

_DISPATCHERS = [
    sub.dispatch_config_subagent,
    sub.dispatch_explore_subagent,
    sub.dispatch_web_subagent,
    sub.dispatch_search_verifier_subagent,
    sub.dispatch_cruncher_subagent,
]


@pytest.mark.parametrize("fn", _DISPATCHERS, ids=lambda f: f.__name__)
def test_every_dispatcher_takes_the_keywords_keyword_only(fn):
    params = inspect.signature(fn).parameters
    for name in ("effort", "effort_ratios", "template_margin_ratio"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[name].default is None


async def test_cruncher_leaf_and_combine_loops_carry_the_keywords(recorded_loops, bus):
    from localharness.agent.context import ContextManager

    ctx = ContextManager()
    handle = ctx._content_store.put("a short granted document")
    base = await _builtin_registry()
    await sub.dispatch_cruncher_subagent(
        "what does it say?", grant_handles=[handle], llm=object(), bus=bus, base_registry=base,
        parent_session_id="parent-sess", permission_evaluator=PermissionEvaluator(),
        context_manager=ctx, effort=("low", "session"), effort_ratios=(0.2, 0.6),
        template_margin_ratio=0.03,
    )
    # one leaf over the one chunk, then the final combine
    assert len(recorded_loops) >= 2
    for c in recorded_loops:
        assert c["effort"] == ("low", "session")
        assert c["effort_ratios"] == (0.2, 0.6)
        assert c["template_margin_ratio"] == 0.03


# ---------------------------------------------------------------------------
# D-34: the parent's template margin reaches every child, through all five dispatchers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("agent_id", "task"),
    [
        ("yaml-child", "do it"),
        ("explore", "find X"),
        ("web-researcher", "research X"),
        ("search-verifier", "claim: X\nentity: X\nsource_url: https://n.test/x"),
        ("cruncher", "what does it say?"),
    ],
)
async def test_the_parents_template_margin_reaches_each_dispatchers_loops(
    recorded_loops, bus, agent_id, task
):
    runner = await _runner(
        bus, template_margin_ratio=0.03, load_agent=lambda n: _yaml_agent("inherit", n),
    )
    if agent_id == "cruncher":
        # the cruncher reads granted handles from the runner's parent store
        from localharness.agent.context import ContentStore
        store = ContentStore()
        handle = store.put("a granted document")
        base = await _builtin_registry()
        runner = make_explore_agent_runner(
            llm=object(), bus=bus, base_registry=base, permission_evaluator=PermissionEvaluator(),
            get_parent_session_id=lambda: "parent-sess", template_margin_ratio=0.03,
            parent_store=store,
        )
        await runner(agent_id, task, [handle])
    else:
        await runner(agent_id, task)

    assert recorded_loops, "AgentLoop patch did not take effect at the call site"
    assert all(c["template_margin_ratio"] == 0.03 for c in recorded_loops)
    if agent_id == "cruncher":
        assert len(recorded_loops) >= 2, "the leaf AND the combine loop must both be built"


async def test_a_grandchild_gets_the_parents_template_margin(recorded_loops, bus):
    runner = await _runner(bus, template_margin_ratio=0.03, max_subagent_depth=3)
    await runner("web-researcher", "research X")
    agent_tool = recorded_loops[0]["tool_registry"]._tools["global"]["agent"]
    before = len(recorded_loops)
    await agent_tool._agent_runner("search-verifier", "claim: X\nentity: X\nsource_url: https://n.test/x")
    assert len(recorded_loops) > before
    assert recorded_loops[-1]["template_margin_ratio"] == 0.03
