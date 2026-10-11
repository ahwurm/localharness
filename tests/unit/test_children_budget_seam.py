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


# ---------------------------------------------------------------------------
# The machine knob reaches every child loop (08-09 Task 1, D-57)
# ---------------------------------------------------------------------------

def _config_child(name: str = "coder"):
    from localharness.config.models import AgentConfig, BudgetConfig, PermissionConfig

    return AgentConfig(name=name, role="test child",
                       permissions=PermissionConfig(budget=BudgetConfig()))


def _knobs(calls: list[dict]) -> list:
    return [c["unattended_max_actions"] for c in calls]


async def test_the_knob_reaches_a_config_child(recorded_loops, bus):
    runner = await _runner(bus, unattended_max_actions=100, load_agent=lambda n: _config_child(n))
    await runner("coder", "write hello.py")
    assert _knobs(recorded_loops) == [100]


@pytest.mark.parametrize("hook", _HOOKS)
@pytest.mark.parametrize(
    ("agent_id", "task"),
    [("explore", "find X"), ("web-researcher", "research X"), ("search-verifier", _VERIFY_TASK)],
)
async def test_the_knob_reaches_every_built_in_child_and_every_web_retry(
    recorded_loops, bus, hook, agent_id, task
):
    runner = await _runner(bus, unattended_max_actions=100, load_builtin_override=hook)
    await runner(agent_id, task)
    assert recorded_loops
    assert set(_knobs(recorded_loops)) == {100}
    if agent_id == "web-researcher":
        assert len(recorded_loops) == 1 + sub.WEB_ZERO_CALL_RETRIES >= 2


async def test_the_knob_reaches_the_nested_search_verifier(recorded_loops, bus):
    runner = await _runner(bus, unattended_max_actions=100, max_subagent_depth=3)
    await runner("web-researcher", "research X")
    agent_tool = recorded_loops[0]["tool_registry"]._tools["global"]["agent"]
    before = len(recorded_loops)
    await agent_tool._agent_runner("search-verifier", _VERIFY_TASK)
    assert len(recorded_loops) > before
    assert recorded_loops[-1]["unattended_max_actions"] == 100


async def test_the_knob_reaches_the_cruncher_leaves_and_combine_turns(recorded_loops, bus):
    from localharness.agent.context import ContentStore

    store = ContentStore()
    handle = store.put("a granted document")
    runner = await _runner(bus, unattended_max_actions=100, parent_store=store)
    await runner("cruncher", "what does it say?", [handle])
    assert {c["config"].name for c in recorded_loops} == {"chunk-summarizer", "cruncher"}
    assert set(_knobs(recorded_loops)) == {100}


async def test_a_partial_combine_level_carries_the_knob_too(monkeypatch, bus):
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
    runner = await _runner(bus, unattended_max_actions=100, parent_store=store, max_context_tokens=2000)
    await runner("cruncher", "what does it say?", [handle])
    assert len([c for c in calls if c["config"].name == "cruncher"]) >= 2
    assert set(_knobs(calls)) == {100}


@pytest.mark.parametrize(
    ("agent_id", "task"),
    [("coder", "write"), ("explore", "find X"), ("web-researcher", "research X"),
     ("search-verifier", _VERIFY_TASK)],
)
async def test_with_no_knob_every_child_loop_gets_none(recorded_loops, bus, agent_id, task):
    runner = await _runner(bus, load_agent=lambda n: _config_child(n))
    await runner(agent_id, task)
    assert recorded_loops
    assert set(_knobs(recorded_loops)) == {None}


_KNOB_TAKERS = _BLOCK_TAKERS + [sub.dispatch_config_subagent]


@pytest.mark.parametrize("fn", _KNOB_TAKERS, ids=lambda f: f.__name__)
def test_unattended_max_actions_is_keyword_only_with_default_none(fn):
    param = inspect.signature(fn).parameters["unattended_max_actions"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is None


def test_the_config_child_dispatcher_takes_children_budget():
    param = inspect.signature(sub.dispatch_config_subagent).parameters["children_budget"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is None


# ---------------------------------------------------------------------------
# The dispatch seam (08-09 Task 2, D-15, D-47)
# ---------------------------------------------------------------------------

from localharness.agent.loop import BudgetLabels  # noqa: E402
from localharness.config.models import AgentConfig, BudgetConfig, PermissionConfig  # noqa: E402

_DEFAULT = "org.children_budget.default"
_LOOKUP = "org.children_budget.lookup_share"
_REVIEW = "org.children_budget.review_share"


def _child(name: str = "coder", *, actions=None, minutes=None, kept=(), tool_calls=None) -> AgentConfig:
    cfg = AgentConfig(
        name=name, role="test child",
        permissions=PermissionConfig(budget=BudgetConfig(
            max_actions=actions, max_duration_minutes=minutes, max_tool_calls=tool_calls)),
    )
    cfg._budget_kept_from_workspace = frozenset(kept)
    return cfg


def _seam(cfg, name="coder", *, parent=None, shape=None, block=None):
    return sub._child_budget(
        cfg, name, children_budget=block, shape=shape,
        parent_remaining=None if parent is None else (lambda: parent))


def _written(out: AgentConfig) -> tuple:
    b = out.permissions.budget
    return (b.max_actions, b.max_duration_minutes)


def _fb(out_fb: ChildBudget) -> tuple:
    return (out_fb.actions, out_fb.minutes)


@pytest.mark.parametrize(
    ("parent", "shape", "written", "fallback", "labels"),
    [
        # unconfigured child under a finite parent: a share, capped by the org default
        ((20, 4.0), None, (5, 1.0), (5, 1.0), (_LOOKUP, _LOOKUP)),
        ((20, 4.0), "lookup", (5, 1.0), (5, 1.0), (_LOOKUP, _LOOKUP)),
        ((20, 4.0), "review", (8, 2.0), (8, 2.0), (_DEFAULT, _REVIEW)),
        # the floors: never 0, never unbounded
        ((1, 0.1), None, (1, 0.1), (1, 0.1), (_LOOKUP, _LOOKUP)),
        ((1, 0.1), "review", (1, 0.1), (1, 0.1), (_REVIEW, _REVIEW)),
        # an infinite parent: nothing written, the org rung handed to the loop
        ((0, 0.0), None, (None, None), (8, 3.0), (None, None)),
        (None, None, (None, None), (8, 3.0), (None, None)),
        # one key finite, the other infinite: each key decides alone
        ((20, 0.0), None, (5, None), (5, 3.0), (_LOOKUP, None)),
        ((0, 4.0), None, (None, 1.0), (8, 1.0), (None, _LOOKUP)),
    ],
)
def test_an_unconfigured_child_arithmetic(parent, shape, written, fallback, labels):
    out, fb, lab = _seam(_child(), parent=parent, shape=shape)
    assert _written(out) == written
    assert _fb(fb) == fallback
    assert (lab.actions, lab.minutes, lab.fallback) == (*labels, _DEFAULT)


@pytest.mark.parametrize(
    ("parent", "written", "fallback", "labels"),
    [
        ((0, 0.0), (16, 12.0), (16, 12.0), (None, None)),       # infinite: the configured value is the fallback
        ((100, 60.0), (16, 12.0), (16, 12.0), (None, None)),    # finite: the ceiling is the configured value
        ((20, 4.0), (5, 1.0), (5, 1.0), (_LOOKUP, _LOOKUP)),    # a small remainder still takes the share
    ],
)
def test_a_configured_child_keeps_its_own_ceiling(parent, written, fallback, labels):
    out, fb, lab = _seam(_child(actions=16, minutes=12.0), parent=parent)
    assert _written(out) == written
    assert _fb(fb) == fallback
    assert (lab.actions, lab.minutes) == labels


_KEPT = ("max_actions", "max_duration_minutes")


@pytest.mark.parametrize(
    ("parent", "written", "fallback", "labels"),
    [
        ((0, 0.0), (16, 12.0), (8, 3.0), (None, None)),         # nothing written; the rung is the fallback
        ((100, 60.0), (8, 3.0), (8, 3.0), (_DEFAULT, _DEFAULT)),  # the rung is the ceiling too
        ((20, 4.0), (5, 1.0), (5, 1.0), (_LOOKUP, _LOOKUP)),
    ],
)
def test_a_workspace_kept_value_cannot_climb_above_the_org_rung(parent, written, fallback, labels):
    out, fb, lab = _seam(_child(actions=16, minutes=12.0, kept=_KEPT), parent=parent)
    assert _written(out) == written
    assert _fb(fb) == fallback
    assert (lab.actions, lab.minutes) == labels


def test_one_workspace_kept_key_does_not_touch_the_other():
    _, fb, _ = _seam(_child(actions=16, minutes=12.0, kept=("max_actions",)), parent=(0, 0.0))
    assert _fb(fb) == (8, 12.0)


def test_a_workspace_kept_value_below_the_rung_is_its_own_ceiling():
    out, fb, lab = _seam(_child(actions=4, minutes=1.0, kept=_KEPT), parent=(100, 60.0))
    assert _written(out) == (4, 1.0)
    assert (lab.actions, lab.minutes) == (None, None)
    assert _fb(_seam(_child(actions=4, minutes=1.0, kept=_KEPT), parent=(0, 0.0))[1]) == (8, 3.0)


@pytest.mark.parametrize(
    ("parent", "written", "fallback"),
    [((20, 4.0), (5, 1.0), (5, 1.0)), ((0, 0.0), (8, 3.0), (8, 3.0))],
)
def test_a_built_in_child_under_the_seam(parent, written, fallback):
    out, fb, lab = _seam(sub.build_explore_config("explore"), "explore", parent=parent)
    assert _written(out) == written
    assert _fb(fb) == fallback
    assert lab.fallback == "org.children_budget.explore"


def test_the_seam_reads_the_childs_own_entry_of_the_block():
    block = ChildrenBudgetConfig(default=ChildBudget(actions=4, minutes=2.0),
                                 lookup_share=0.5, review_share=1.0)
    out, fb, lab = _seam(_child(), parent=(0, 0.0), block=block)
    assert _fb(fb) == (4, 2.0) and _written(out) == (None, None)
    out, fb, _ = _seam(_child(), parent=(6, 2.0), block=block)
    assert _written(out) == (3, 1.0)
    out, fb, _ = _seam(_child(), parent=(6, 2.0), shape="review", block=block)
    assert _written(out) == (4, 2.0)  # the share of 6 is 6, the ceiling 4 wins


def test_a_hyphenated_built_in_name_takes_its_own_entry():
    _, _, lab = _seam(sub.build_web_researcher_config("web-researcher"), "web-researcher", parent=(0, 0.0))
    assert lab.fallback == "org.children_budget.web_researcher"


def test_the_seam_never_mutates_the_input_and_never_touches_max_tool_calls():
    cfg = _child(tool_calls=7)
    before = cfg.model_dump()
    out, _, _ = _seam(cfg, parent=(20, 4.0))
    assert cfg.model_dump() == before
    assert out is not cfg and out.permissions.budget is not cfg.permissions.budget
    assert out.permissions.budget.max_tool_calls == 7
    assert out.budget_kept_from_workspace == cfg.budget_kept_from_workspace
    out2, _, _ = _seam(_child(), parent=(20, 4.0))
    assert out2.permissions.budget.max_tool_calls is None


def test_the_seam_takes_no_gate():
    params = inspect.signature(sub._child_budget).parameters
    assert "gate" not in params
    assert [n for n, p in params.items() if p.kind is inspect.Parameter.KEYWORD_ONLY] == [
        "children_budget", "shape", "parent_remaining"]


def test_a_written_share_always_validates():
    """The floors are BudgetConfig's own lower bounds: the copy stays a valid config."""
    out, _, _ = _seam(_child(), parent=(1, 0.1))
    AgentConfig.model_validate(out.model_dump())


# --- the runner: the parent's remainder is read at each dispatch ---------------------------------


@pytest.mark.parametrize("hook", _HOOKS)
async def test_parent_remaining_is_read_at_each_dispatch_not_captured(recorded_loops, bus, hook):
    state = {"rem": (20, 4.0), "reads": 0}

    def parent_remaining():
        state["reads"] += 1
        return state["rem"]

    runner = await _runner(bus, parent_remaining_fn=parent_remaining, load_builtin_override=hook)
    assert state["reads"] == 0  # building the runner reads nothing
    await runner("explore", "find X")
    assert state["reads"] == 1
    state["rem"] = (8, 2.0)
    await runner("explore", "find Y")
    assert state["reads"] == 2
    assert [_pair(c) for c in recorded_loops] == [(5, 1.0), (2, 0.5)]


async def test_the_config_child_reads_the_parent_at_each_dispatch(recorded_loops, bus):
    state = {"rem": (20, 4.0), "reads": 0}

    def parent_remaining():
        state["reads"] += 1
        return state["rem"]

    runner = await _runner(
        bus, parent_remaining_fn=parent_remaining, load_agent=lambda n: _child(n))
    await runner("coder", "write")
    state["rem"] = (0, 0.0)
    await runner("coder", "write again")
    assert state["reads"] == 2
    assert [_pair_or_none(c) for c in recorded_loops] == [(5, 1.0), (None, None)]


def _pair_or_none(call: dict) -> tuple:
    b = call["config"].permissions.budget
    return (b.max_actions, b.max_duration_minutes)


async def test_no_parent_remaining_is_an_infinite_parent(recorded_loops, bus):
    runner = await _runner(bus, load_agent=lambda n: _child(n))
    await runner("coder", "write")
    call = recorded_loops[0]
    assert _pair_or_none(call) == (None, None)
    assert call["budget_fallback"] == ChildBudget(actions=8, minutes=3.0)
    assert call["budget_labels"] == BudgetLabels(fallback=_DEFAULT)


async def test_a_config_child_gets_the_seams_copy_not_the_cached_config(recorded_loops, bus):
    cached = _child()
    runner = await _runner(bus, parent_remaining_fn=lambda: (20, 4.0), load_agent=lambda n: cached)
    await runner("coder", "write")
    assert _pair_or_none(recorded_loops[0]) == (5, 1.0)
    assert recorded_loops[0]["config"] is not cached
    assert (cached.permissions.budget.max_actions, cached.permissions.budget.max_duration_minutes) == (None, None)


@pytest.mark.parametrize("hook", _HOOKS)
async def test_a_built_in_child_carries_its_fallback_and_labels(recorded_loops, bus, hook):
    runner = await _runner(bus, load_builtin_override=hook)
    await runner("explore", "find X")
    call = recorded_loops[0]
    assert call["budget_fallback"] == ChildBudget(actions=8, minutes=3.0)
    assert call["budget_labels"] == BudgetLabels(fallback="org.children_budget.explore")
    assert _pair(call) == (8, 3.0)


@pytest.mark.parametrize("hook", _HOOKS)
async def test_every_web_researcher_attempt_carries_the_seams_result(recorded_loops, bus, hook):
    runner = await _runner(bus, parent_remaining_fn=lambda: (100, 60.0), load_builtin_override=hook)
    await runner("web-researcher", "research X")
    assert len(recorded_loops) == 1 + sub.WEB_ZERO_CALL_RETRIES >= 2
    for call in recorded_loops:
        assert _pair(call) == (25, 15.0)
        assert call["budget_labels"].actions == _LOOKUP


async def test_the_overlay_ceiling_is_the_ceiling_the_share_is_capped_by(recorded_loops, bus):
    def hook(name, base):
        base.permissions.budget.max_actions = 40
        return base

    runner = await _runner(bus, parent_remaining_fn=lambda: (400, 60.0), load_builtin_override=hook)
    await runner("web-researcher", "research X")
    assert _pair(recorded_loops[0]) == (40, 15.0)
    assert recorded_loops[0]["budget_labels"].actions is None  # the overlay's own value bound it


async def test_the_cruncher_leaves_and_combines_get_no_fallback_and_never_read_the_parent(
    recorded_loops, bus
):
    from localharness.agent.context import ContentStore

    store = ContentStore()
    handle = store.put("a granted document")
    reads: list[int] = []
    runner = await _runner(
        bus, parent_store=store, parent_remaining_fn=lambda: reads.append(1) or (20, 4.0))
    await runner("cruncher", "what does it say?", [handle])
    assert reads == []
    assert {c["config"].name for c in recorded_loops} == {"chunk-summarizer", "cruncher"}
    assert [c.get("budget_fallback") for c in recorded_loops] == [None] * len(recorded_loops)
    assert {c["config"].name: _pair(c) for c in recorded_loops} == {
        "chunk-summarizer": (3, 6.0), "cruncher": (6, 20.0)}


# --- the nested runner reads the child loop's own remainder -----------------------------------------


def _remaining_loop(monkeypatch, remaining):
    """A recording loop whose `remaining_budget` is `remaining(index)`; returns the recorded kwargs."""
    calls: list[dict] = []

    class _Loop:
        def __init__(self, **kwargs):
            self.index = len(calls)
            calls.append(kwargs)
            self.current_session_id = None

        async def run_turn(self, task, *a, **k):
            return ""

        def action_limit(self):
            return 0

        def time_limit(self):
            return 0.0

        def remaining_budget(self):
            return remaining(self.index)

    monkeypatch.setattr("localharness.agent.loop.AgentLoop", _Loop)
    return calls


async def test_the_nested_search_verifier_is_cut_from_the_web_researchers_own_remainder(
    monkeypatch, bus
):
    calls = _remaining_loop(monkeypatch, lambda i: (40, 10.0))
    runner = await _runner(bus, max_subagent_depth=3)
    await runner("web-researcher", "research X")
    agent_tool = calls[0]["tool_registry"]._tools["global"]["agent"]
    await agent_tool._agent_runner("search-verifier", _VERIFY_TASK)
    assert _pair(calls[-1]) == (10, 2.5)  # min(max(1, 0.25 x 40), 12), min(0.25 x 10, 6.0)
    assert calls[-1]["config"].name == "search-verifier"
    assert calls[-1]["budget_labels"].actions == _LOOKUP


async def test_the_nested_runner_ignores_the_root_remainder(monkeypatch, bus):
    calls = _remaining_loop(monkeypatch, lambda i: (40, 10.0))
    runner = await _runner(bus, max_subagent_depth=3, parent_remaining_fn=lambda: (4, 1.0))
    await runner("web-researcher", "research X")
    assert _pair(calls[0]) == (1, 0.25)  # the root's share
    agent_tool = calls[0]["tool_registry"]._tools["global"]["agent"]
    await agent_tool._agent_runner("search-verifier", _VERIFY_TASK)
    assert _pair(calls[-1]) == (10, 2.5)


async def test_a_retry_rebinds_the_slot_to_the_loop_now_running(monkeypatch, bus):
    calls = _remaining_loop(monkeypatch, lambda i: (40 + 4 * i, 10.0))
    runner = await _runner(bus, max_subagent_depth=3)
    await runner("web-researcher", "research X")
    attempts = len(calls)
    assert attempts == 1 + sub.WEB_ZERO_CALL_RETRIES >= 2
    agent_tool = calls[0]["tool_registry"]._tools["global"]["agent"]
    await agent_tool._agent_runner("search-verifier", _VERIFY_TASK)
    assert _pair(calls[-1])[0] == min(int(0.25 * (40 + 4 * (attempts - 1))), 12)


async def test_a_nested_runner_with_no_loop_yet_is_an_infinite_parent(monkeypatch, bus):
    calls = _remaining_loop(monkeypatch, lambda i: (0, 0.0))
    runner = await _runner(bus, max_subagent_depth=3)
    await runner("web-researcher", "research X")
    agent_tool = calls[0]["tool_registry"]._tools["global"]["agent"]
    await agent_tool._agent_runner("search-verifier", _VERIFY_TASK)
    assert _pair(calls[-1]) == (12, 6.0)
    assert calls[-1]["budget_fallback"] == ChildBudget(actions=12, minutes=6.0)


_FALLBACK_TAKERS = [sub.dispatch_explore_subagent, sub.dispatch_web_subagent,
                    sub.dispatch_search_verifier_subagent]


@pytest.mark.parametrize("fn", _FALLBACK_TAKERS, ids=lambda f: f.__name__)
def test_the_built_in_dispatchers_take_the_seams_outputs_and_the_loop_hook(fn):
    params = inspect.signature(fn).parameters
    for name in ("budget_fallback", "budget_labels", "on_child_loop"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[name].default is None
        assert list(params).index(name) > list(params).index("children_budget")


def test_parent_remaining_fn_is_keyword_only_with_default_none():
    for fn in (sub.make_explore_agent_runner, sub.dispatch_config_subagent):
        param = inspect.signature(fn).parameters["parent_remaining_fn"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is None
        names = list(inspect.signature(fn).parameters)
        assert names.index("parent_remaining_fn") > names.index("children_budget")


# --- BUDGET-05: the surface matrix through a real AgentLoop ---------------------------------------

from types import SimpleNamespace  # noqa: E402

from localharness.agent.context import ContextManager  # noqa: E402
from localharness.agent.loop import AgentLoop  # noqa: E402
from localharness.core.events import TurnFailed  # noqa: E402

_KNOB = 100


def _channel(*, local: bool) -> SimpleNamespace:
    return SimpleNamespace(channel_id="terminal" if local else "phone", local_operator=local, can_ask=False)


class _CountingRegistry:
    def __init__(self):
        self.calls: list[dict] = []

    def get_tools_for_agent(self, agent_id, division_id, tool_config):
        return {}

    async def dispatch(self, name, arguments, agent_id, division_id, tool_config, **kw):
        from localharness.tools.base import ToolResult

        self.calls.append(dict(arguments))
        return ToolResult(output=f"ran {arguments}", success=True)


def _calls(mock_llm_client, n: int) -> list:
    Response, ToolCall = mock_llm_client.Response, mock_llm_client.ToolCall
    out = [Response(content=None, tool_calls=[ToolCall(id=f"tc-{i}", name="bash", arguments={"cmd": f"echo {i}"})])
           for i in range(n)]
    out.append(Response(content="all done"))
    return out


def _parent_gate(mock_llm_client, bus, *, local: bool):
    """A real PermissionGate in `unattended` mode, with a local operator attached or not."""
    parent = AgentLoop(
        config=AgentConfig(name="root", role="root", permissions=PermissionConfig(
            mode="unattended", deny_patterns=[], budget=BudgetConfig())),
        llm=mock_llm_client([]), bus=bus, context_manager=ContextManager(),
        tool_registry=_CountingRegistry(), permission_evaluator=PermissionEvaluator())
    parent.gate.attach_channel(_channel(local=local))
    return parent.gate


def _seam_child(mock_llm_client, bus, cfg, *, local: bool, parent=(0, 0.0), responses=None):
    """The way `dispatch_config_subagent` builds the child loop, from the seam's three outputs."""
    out, fb, labels = _seam(cfg, parent=parent)
    gate = _parent_gate(mock_llm_client, bus, local=local)
    registry = _CountingRegistry()
    loop = AgentLoop(
        config=out, llm=mock_llm_client(responses or []), bus=bus, context_manager=ContextManager(),
        tool_registry=registry, permission_evaluator=PermissionEvaluator(), gate=gate,
        budget_fallback=fb, budget_labels=labels, unattended_max_actions=_KNOB)
    return loop, gate, registry


def test_an_unconfigured_child_has_no_action_limit_under_an_attended_root(mock_llm_client, bus):
    loop, gate, _ = _seam_child(mock_llm_client, bus, _child(), local=True)
    assert gate.mode == "unattended"
    assert (loop.action_limit(), loop.time_limit()) == (0, 0.0)


def test_an_unconfigured_child_runs_at_the_org_rung_remotely_and_follows_the_gate(mock_llm_client, bus):
    loop, gate, _ = _seam_child(mock_llm_client, bus, _child(), local=False)
    assert (loop.action_limit(), loop.time_limit()) == (8, 3.0)
    gate.attach_channel(_channel(local=True))   # a person sits down after dispatch: uncapped at the next check
    assert (loop.action_limit(), loop.time_limit()) == (0, 0.0)
    gate.attach_channel(_channel(local=False))
    assert loop.action_limit() == 8


async def test_the_remote_failure_names_the_org_rung(mock_llm_client, bus):
    loop, gate, registry = _seam_child(
        mock_llm_client, bus, _child(), local=False, responses=_calls(mock_llm_client, 20))
    summary = await loop.run_turn("go")
    assert len(registry.calls) == 8
    failed = [e for e in bus.history(event_types=[TurnFailed]) if e.reason == "budget_exceeded"]
    assert failed
    assert "Agent reached the maximum of 8 tool calls." in summary
    assert "(org.children_budget.default)" in summary


async def test_a_gate_leaving_unattended_after_dispatch_uncaps_the_child_mid_turn(mock_llm_client, bus):
    loop, gate, registry = _seam_child(
        mock_llm_client, bus, _child(), local=False, responses=_calls(mock_llm_client, 12))
    original = registry.dispatch

    async def dispatch(name, arguments, *a, **k):
        if len(registry.calls) == 2:
            gate.attach_channel(_channel(local=True))
        return await original(name, arguments, *a, **k)

    registry.dispatch = dispatch
    summary = await loop.run_turn("go")
    assert len(registry.calls) == 12
    assert "all done" in summary


@pytest.mark.parametrize(
    ("cfg_kw", "attended", "remote"),
    [
        ({"actions": 16, "minutes": 12.0, "kept": _KEPT}, (16, 12.0), (8, 3.0)),   # workspace-kept
        ({"actions": 16, "minutes": 12.0}, (16, 12.0), (16, 12.0)),                # operator-configured
        ({"actions": 4, "minutes": 1.0, "kept": _KEPT}, (4, 1.0), (4, 1.0)),       # kept, but below the rung
    ],
    ids=["workspace-kept", "configured", "kept-below-rung"],
)
def test_the_surface_matrix_for_a_configured_child(mock_llm_client, bus, cfg_kw, attended, remote):
    loop, gate, _ = _seam_child(mock_llm_client, bus, _child(**cfg_kw), local=True)
    assert (loop.action_limit(), loop.time_limit()) == attended
    gate.attach_channel(_channel(local=False))
    assert (loop.action_limit(), loop.time_limit()) == remote


def test_the_knob_still_caps_a_child_whose_own_limit_is_looser(mock_llm_client, bus):
    loop, gate, _ = _seam_child(mock_llm_client, bus, _child(actions=500, minutes=12.0), local=False)
    assert loop.action_limit() == _KNOB


# ---------------------------------------------------------------------------
# shape: lookup | review picks only the share (08-09 Task 3, D-18, D-31, D-50)
# ---------------------------------------------------------------------------

from localharness.tools.builtin.agent_tool import AgentTool  # noqa: E402


async def _through_the_tool(bus, agent_id, task, *, shape=None, **runner_kw):
    runner = await _runner(bus, **runner_kw)
    tool = AgentTool(agent_runner=runner, available_agents=[agent_id])
    return await tool.run(agent_id=agent_id, task=task, **({} if shape is None else {"shape": shape}))


@pytest.mark.parametrize(("shape", "expected"), [(None, (5, 1.0)), ("lookup", (5, 1.0)), ("review", (8, 2.0))])
async def test_the_shape_picks_the_share_of_an_unconfigured_child(recorded_loops, bus, shape, expected):
    result = await _through_the_tool(
        bus, "coder", "write", shape=shape,
        parent_remaining_fn=lambda: (20, 4.0), load_agent=lambda n: _child(n))
    assert result.success
    assert _pair_or_none(recorded_loops[0]) == expected


@pytest.mark.parametrize("shape", [None, "lookup", "review"])
async def test_the_shape_never_changes_a_ceiling(recorded_loops, bus, shape):
    await _through_the_tool(
        bus, "coder", "write", shape=shape,
        parent_remaining_fn=lambda: (100, 60.0), load_agent=lambda n: _child(n, actions=6, minutes=2.0))
    assert _pair_or_none(recorded_loops[0]) == (6, 2.0)


@pytest.mark.parametrize(("shape", "expected"), [(None, (5, 1.0)), ("review", (8, 2.0))])
@pytest.mark.parametrize("hook", _HOOKS)
async def test_the_shape_reaches_a_built_in_child(recorded_loops, bus, hook, shape, expected):
    await _through_the_tool(
        bus, "explore", "find X", shape=shape,
        parent_remaining_fn=lambda: (20, 4.0), load_builtin_override=hook)
    assert _pair(recorded_loops[0]) == expected


async def test_a_refused_shape_builds_no_child(recorded_loops, bus):
    result = await _through_the_tool(
        bus, "coder", "write", shape="deep",
        parent_remaining_fn=lambda: (20, 4.0), load_agent=lambda n: _child(n))
    assert result.success is False and result.error_type == "validation_error"
    assert recorded_loops == []


async def test_the_runner_still_takes_three_positional_arguments(recorded_loops, bus):
    runner = await _runner(bus, parent_remaining_fn=lambda: (20, 4.0), load_agent=lambda n: _child(n))
    await runner("coder", "write", None)
    await runner("coder", "write")
    assert [_pair_or_none(c) for c in recorded_loops] == [(5, 1.0)] * 2


def test_the_shape_is_a_keyword_with_default_none_on_the_runner_and_the_config_dispatcher():
    assert inspect.signature(sub.dispatch_config_subagent).parameters["shape"].default is None
    assert inspect.signature(sub.dispatch_config_subagent).parameters["shape"].kind is (
        inspect.Parameter.KEYWORD_ONLY)


# --- the share is never 0 and never above the parent (Nyquist) ----------------------------------


@pytest.mark.parametrize("shape", [None, "lookup", "review"])
@pytest.mark.parametrize("share", [0.01, 0.1, 0.25, 0.5, 1.0])
def test_a_finite_parent_never_hands_a_child_zero_or_more_than_it_has(shape, share):
    block = ChildrenBudgetConfig(lookup_share=share, review_share=share)
    for actions in range(1, 25):
        for minutes in (0.1, 0.15, 0.7, 3.0):
            out, fb, _ = _seam(_child(), parent=(actions, minutes), shape=shape, block=block)
            a, m = _written(out)
            assert a is not None and 1 <= a <= actions, (actions, share, a)
            assert m is not None and 0.1 <= m <= max(minutes, 0.1), (minutes, share, m)
            assert _fb(fb) == (a, m)
            AgentConfig.model_validate(out.model_dump())


@pytest.mark.parametrize("shape", [None, "lookup", "review"])
def test_an_infinite_parent_writes_nothing_into_the_childs_budget(shape):
    for cfg, name in ((_child(), "coder"), (sub.build_explore_config("explore"), "explore")):
        before = _written(cfg)
        out, fb, lab = _seam(cfg, name, parent=(0, 0.0), shape=shape)
        assert _written(out) == before
        assert (lab.actions, lab.minutes) == (None, None)
        assert fb is not None
