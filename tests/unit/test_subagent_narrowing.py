"""Per-call tool narrowing through the real delegation runner: what a narrowed child holds, is told and returns."""
from __future__ import annotations

import pytest

from localharness.agent import subagent
from localharness.agent.permissions import PermissionEvaluator
from localharness.agent.subagent import make_explore_agent_runner
from localharness.config.models import AgentConfig, BudgetConfig, PermissionConfig, ToolConfig
from localharness.tools.builtin.agent_tool import AgentTool, DelegationArgError
from tests.unit.test_capabilities import _builtin_registry

LEAF_TOOLSETS = {**subagent._STATIC_TOOLSETS, "chunk-summarizer": subagent.CHUNK_SUMMARIZER_TOOLS}


class FakeLoop:
    """Records the registry and every brief; answers with a fixed string and never calls a model."""

    loops: list["FakeLoop"] = []

    def __init__(self, **kw):
        self.tool_registry = kw["tool_registry"]
        self.config = kw["config"]
        self.briefs: list[str] = []
        self.current_session_id = "child-sid"
        self.last_terminated_reason = None
        FakeLoop.loops.append(self)

    def action_limit(self):
        return 0

    def time_limit(self):
        return 0.0

    async def run_turn(self, brief, *a, **k):
        self.briefs.append(brief)
        return "done"

    @property
    def held(self) -> set[str]:
        return set(self.tool_registry._tools["global"])


def install_fake_loop(monkeypatch) -> list[FakeLoop]:
    FakeLoop.loops = []
    monkeypatch.setattr("localharness.agent.loop.AgentLoop", FakeLoop)
    return FakeLoop.loops


def make_runner(base, llm, **kw):
    return make_explore_agent_runner(
        llm=llm, bus=object(), base_registry=base, permission_evaluator=PermissionEvaluator(),
        get_parent_session_id=lambda: "sid", **kw)


def config_child(name="cfg", add=("read", "grep", "bash_exec")) -> AgentConfig:
    return AgentConfig(name=name, role="test child", tools=ToolConfig(add=list(add)),
                       permissions=PermissionConfig(budget=BudgetConfig()))


@pytest.fixture(autouse=True)
def _ledger(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALHARNESS_VERIFICATION_LEDGER_DIR", str(tmp_path / "ledger"))


@pytest.fixture
async def base():
    return await _builtin_registry()


@pytest.fixture
def loops(monkeypatch):
    return install_fake_loop(monkeypatch)


WEB_SET = {"web_search", "web_fetch", "web_page_query"}


async def test_a_narrowed_call_does_not_leak_into_the_next(base, loops, mock_llm_client):
    runner = make_runner(base, mock_llm_client([]), max_subagent_depth=1)
    await runner("web-researcher", "t", tools=["web_fetch"])
    narrowed = len(loops)
    await runner("web-researcher", "t")
    assert loops[0].held == {"web_fetch"}
    assert loops[narrowed].held == WEB_SET
    assert base.has("web_search") and base.has("agent")


@pytest.mark.parametrize("cap", [1, 2])
async def test_web_researcher_narrowed_holds_exactly_the_name_and_no_agent(base, loops, mock_llm_client, cap):
    runner = make_runner(base, mock_llm_client([]), max_subagent_depth=cap)
    await runner("web-researcher", "t", tools=["web_fetch"])
    assert {frozenset(lp.held) for lp in loops} == {frozenset({"web_fetch"})}


async def test_explore_cannot_be_narrowed_to_agent(base, loops, mock_llm_client):
    runner = make_runner(base, mock_llm_client([]), max_subagent_depth=2)
    with pytest.raises(DelegationArgError, match="explore has no agent; its tools: read, glob, grep"):
        await runner("explore", "t", tools=["grep", "agent"])
    assert loops == []


async def test_a_name_outside_the_set_is_refused_with_the_bare_set(base, loops, mock_llm_client):
    runner = make_runner(base, mock_llm_client([]), max_subagent_depth=1)
    with pytest.raises(DelegationArgError) as exc:
        await runner("web-researcher", "t", tools=["mcp:nope"])
    assert "has no nope; its tools: web_search, web_fetch, web_page_query" in str(exc.value)
    assert "mcp:" not in str(exc.value)


async def test_an_unknown_agent_keeps_the_not_wired_error(base, loops, mock_llm_client):
    runner = make_runner(base, mock_llm_client([]), load_agent=lambda n: (_ for _ in ()).throw(KeyError(n)))
    with pytest.raises(ValueError) as exc:
        await runner("ghost", "t", tools=["read"])
    assert not isinstance(exc.value, DelegationArgError)


async def test_a_web_tool_the_base_lacks_is_refused_not_dropped(loops, mock_llm_client):
    from localharness.tools.registry import ToolRegistry
    base = await _builtin_registry()
    slim = ToolRegistry.from_allowed(["web_search", "web_fetch"], base_registry=base)
    runner = make_runner(slim, mock_llm_client([]), max_subagent_depth=1)
    with pytest.raises(DelegationArgError, match="web_page_query"):
        await runner("web-researcher", "t", tools=["web_page_query"])


async def test_cruncher_refuses_tools(base, loops, mock_llm_client):
    runner = make_runner(base, mock_llm_client([]))
    with pytest.raises(DelegationArgError, match="cruncher's tools are run by the harness"):
        await runner("cruncher", "t", tools=["chunk"])


async def test_refusals_map_to_the_agent_tool_error_types(base, loops, mock_llm_client):
    tool = AgentTool(agent_runner=make_runner(base, mock_llm_client([]), max_subagent_depth=1),
                     available_agents=["explore"])
    for agent, names in (("web-researcher", ["nope"]), ("explore", ["agent"]), ("cruncher", ["chunk"])):
        result = await tool._execute(agent_id=agent, task="t", tools=names)
        assert result.error_type == "validation_error", agent
    assert (await tool._execute(agent_id="ghost", task="t", tools=["read"])).error_type == "not_found"
