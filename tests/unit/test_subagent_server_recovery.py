"""The parent's reconnect supplier reaches every child loop, by identity (07-06, D-21, D-40).

`server_recovery` rides the same plumbing as `template_margin_ratio`: `start` hands one supplier to
`make_explore_agent_runner`, and the runner threads that SAME object to every `AgentLoop(` it builds
(the five dispatchers, the cruncher's leaf and combine loops, `_build_child_agent_tool`'s nested
runner). N children under one dead server therefore await the one relaunch. The recorded-loops
fixture of test_subagent_effort.py captures each construction's keywords; no model is ever called.
"""
from __future__ import annotations

import inspect

import pytest

from localharness.agent import subagent as sub
from localharness.agent.permissions import PermissionEvaluator
from localharness.agent.subagent import make_explore_agent_runner
from tests.unit.test_subagent_effort import _DISPATCHERS, _builtin_registry, _yaml_agent


@pytest.fixture(autouse=True)
def _ledger_off_the_repo(monkeypatch, tmp_path):
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

    monkeypatch.setattr("localharness.agent.loop.AgentLoop", _RecordingLoop)
    return calls


def _supplier():
    def _sup():
        return None
    return _sup


async def _runner(bus, llm=None, **kw):
    base = await _builtin_registry()
    return make_explore_agent_runner(
        llm=llm if llm is not None else object(), bus=bus, base_registry=base,
        permission_evaluator=PermissionEvaluator(),
        get_parent_session_id=lambda: "parent-sess", **kw,
    )


_SIGNATURES = [
    *_DISPATCHERS,
    sub._run_chunk_summarizer,
    sub._cruncher_combine_turn,
    make_explore_agent_runner,
]


@pytest.mark.parametrize("fn", _SIGNATURES, ids=lambda f: f.__name__)
def test_every_signature_takes_server_recovery_keyword_only(fn):
    param = inspect.signature(fn).parameters["server_recovery"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is None


def test_server_recovery_is_the_last_keyword_after_template_margin_ratio():
    for fn in _SIGNATURES:
        names = [n for n, p in inspect.signature(fn).parameters.items()
                 if p.kind is inspect.Parameter.KEYWORD_ONLY]
        assert names.index("server_recovery") == names.index("template_margin_ratio") + 1, fn.__name__


@pytest.mark.parametrize(
    ("agent_id", "task"),
    [
        ("yaml-child", "do it"),
        ("explore", "find X"),
        ("web-researcher", "research X"),
        ("search-verifier", "claim: X\nentity: X\nsource_url: https://n.test/x"),
    ],
)
async def test_each_child_loop_gets_the_runners_supplier_by_identity(recorded_loops, bus, agent_id, task):
    sup = _supplier()
    runner = await _runner(bus, server_recovery=sup, load_agent=lambda n: _yaml_agent("inherit", n))
    await runner(agent_id, task)
    assert recorded_loops, "AgentLoop patch did not take effect at the call site"
    assert all(c["server_recovery"] is sup for c in recorded_loops)


async def test_the_cruncher_leaf_and_combine_loops_get_the_supplier(recorded_loops, bus):
    from localharness.agent.context import ContentStore

    sup = _supplier()
    store = ContentStore()
    handle = store.put("a granted document")
    base = await _builtin_registry()
    runner = make_explore_agent_runner(
        llm=object(), bus=bus, base_registry=base, permission_evaluator=PermissionEvaluator(),
        get_parent_session_id=lambda: "parent-sess", parent_store=store, server_recovery=sup,
    )
    await runner("cruncher", "what does it say?", [handle])
    assert len(recorded_loops) >= 2, "the leaf AND the combine loop must both be built"
    assert all(c["server_recovery"] is sup for c in recorded_loops)


async def test_a_grandchild_through_the_nested_runner_gets_the_supplier(recorded_loops, bus):
    sup = _supplier()
    runner = await _runner(bus, server_recovery=sup, max_subagent_depth=3)
    await runner("web-researcher", "research X")
    agent_tool = recorded_loops[0]["tool_registry"]._tools["global"]["agent"]
    before = len(recorded_loops)
    await agent_tool._agent_runner("search-verifier", "claim: X\nentity: X\nsource_url: https://n.test/x")
    assert len(recorded_loops) > before
    assert recorded_loops[0]["server_recovery"] is sup
    assert recorded_loops[-1]["server_recovery"] is sup


async def test_with_no_supplier_every_child_gets_none(recorded_loops, bus):
    runner = await _runner(bus, max_subagent_depth=3, load_agent=lambda n: _yaml_agent("inherit", n))
    await runner("explore", "a")
    await runner("yaml-child", "b")
    await runner("web-researcher", "c")
    agent_tool = recorded_loops[-1]["tool_registry"]._tools["global"]["agent"]
    await agent_tool._agent_runner("search-verifier", "claim: X\nentity: X\nsource_url: https://n.test/x")
    assert len(recorded_loops) >= 4
    assert all(c["server_recovery"] is None for c in recorded_loops)


async def test_children_share_the_roots_client_so_the_bound_is_read_from_one_config(recorded_loops, bus):
    """RECON-07: the reconnect wait is read from the one LLMConfig the root loop and its children share."""
    llm = object()
    runner = await _runner(bus, llm=llm, server_recovery=_supplier(), max_subagent_depth=3)
    await runner("explore", "find X")
    await runner("web-researcher", "research X")
    agent_tool = recorded_loops[-1]["tool_registry"]._tools["global"]["agent"]
    await agent_tool._agent_runner("search-verifier", "claim: X\nentity: X\nsource_url: https://n.test/x")
    assert len(recorded_loops) >= 3
    assert all(c["llm"] is llm for c in recorded_loops)
