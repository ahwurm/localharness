"""Per-call tool narrowing through the real delegation runner: what a narrowed child holds, is told and returns."""
from __future__ import annotations

import pytest

from localharness.agent import subagent
from localharness.agent.permissions import PermissionEvaluator
from localharness.agent.subagent import ZERO_CALL_RETRY_PREFIX, DelegationResult, make_explore_agent_runner, prepend_toolset
from localharness.config.models import AgentConfig, BudgetConfig, PermissionConfig, ToolConfig
from localharness.tools.base import ToolResult, ToolSchema
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


class SafeTool:
    def __init__(self, name: str) -> None:
        self._name = name

    def info(self) -> ToolSchema:
        return ToolSchema(name=self._name, description="d", parameters={}, ingest="none", host="safe",
                          result_origin="trusted")

    async def run(self, **kwargs) -> ToolResult:  # pragma: no cover - never dispatched
        return ToolResult(output="")


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


VERIFIER_TASK = "claim: x\nentity: y\nsource_url: z"


async def test_web_researcher_is_told_only_what_it_holds(base, loops, mock_llm_client):
    runner = make_runner(base, mock_llm_client([]), max_subagent_depth=1)
    await runner("web-researcher", "find it", tools=["web_fetch"])
    first, retry = loops[0].briefs[0], loops[1].briefs[0]
    assert first.startswith("(Your ONLY available tools: web_fetch. You cannot delegate or use anything else")
    assert "Ignore any tool your instructions name that is not in this list." in first
    assert "web_search" not in retry and "web_fetch" in retry and retry.endswith("find it")
    loops.clear()
    await runner("web-researcher", "find it", tools=["web_search"])
    assert "web_fetch" not in loops[1].briefs[0] and "web_search" in loops[1].briefs[0]


async def test_a_web_researcher_below_the_cap_keeps_its_own_nested_agent_tool(base, loops, mock_llm_client):
    runner = make_runner(base, mock_llm_client([]), max_subagent_depth=2,
                         available_agents=["web-researcher", "search-verifier"])
    await runner("web-researcher", "t", tools=["web_fetch", "agent"])
    child = loops[0]
    nested = child.tool_registry._tools["global"]["agent"]
    assert child.held == {"web_fetch", "agent"}
    assert nested is not base._tools["global"]["agent"] and nested._available_agents == ["search-verifier"]
    assert "cannot delegate" not in child.briefs[0] and "agent" in child.briefs[0]
    loops.clear()
    await nested._agent_runner("search-verifier", VERIFIER_TASK)
    assert loops[-1].held == WEB_SET


async def test_agent_is_refused_for_a_web_researcher_at_the_cap(base, loops, mock_llm_client):
    runner = make_runner(base, mock_llm_client([]), max_subagent_depth=1)
    with pytest.raises(DelegationArgError, match="web-researcher has no agent"):
        await runner("web-researcher", "t", tools=["web_fetch", "agent"])


async def test_a_config_child_narrowed_holds_and_names_only_the_subset(base, loops, mock_llm_client):
    runner = make_runner(base, mock_llm_client([]), load_agent=lambda n: config_child(n))
    await runner("cfg", "t", tools=["read"])
    assert loops[0].held == {"read"}
    assert loops[0].briefs[0].startswith("(Your ONLY available tools: read. You cannot delegate")
    assert "Ignore any tool" in loops[0].briefs[0] and "grep" not in loops[0].briefs[0].split("\n")[0]


@pytest.mark.parametrize("entry,narrow_by,bare", [
    ("plugin:P.T", "T", "T"), ("plugin:P.T", "plugin:P.T", "T"), ("mcp:X", "X", "X"), ("mcp:X", "mcp:X", "X"),
])
async def test_prefixed_and_bare_names_narrow_alike(base, loops, mock_llm_client, entry, narrow_by, bare):
    await base.register(SafeTool("T"))
    await base.register(SafeTool("X"))
    runner = make_runner(base, mock_llm_client([]), load_agent=lambda n: config_child(n, ("read", entry)))
    await runner("cfg", "t", tools=[narrow_by])
    assert bare in loops[0].held and "read" not in loops[0].held


async def test_search_verifier_narrowed_holds_only_the_name(base, loops, mock_llm_client):
    runner = make_runner(base, mock_llm_client([]), max_subagent_depth=1)
    await runner("search-verifier", VERIFIER_TASK, tools=["web_fetch"])
    assert loops[0].held == {"web_fetch"}
    assert loops[0].briefs[0].startswith("(Your ONLY available tools: web_fetch.")
    assert VERIFIER_TASK in loops[0].briefs[0]


@pytest.mark.parametrize("agent,task,names,held", [
    ("explore", "t", ["grep"], ["grep"]),
    ("cfg", "t", ["read", "grep"], ["read", "grep"]),
    ("web-researcher", "t", ["web_fetch"], ["web_fetch"]),
    ("web-researcher", "t", ["web_fetch", "web_search"], ["web_fetch", "web_search"]),
])
async def test_the_result_carries_the_held_list(base, loops, mock_llm_client, agent, task, names, held):
    runner = make_runner(base, mock_llm_client([]), max_subagent_depth=1, load_agent=lambda n: config_child(n))
    result = await runner(agent, task, tools=names)
    assert isinstance(result, DelegationResult) and result.tools == held
    if agent == "web-researcher":
        assert result.status == "failed"
    out = await AgentTool(agent_runner=runner, available_agents=[agent])._execute(
        agent_id=agent, task=task, tools=names)
    assert out.metadata["tools"] == held


@pytest.mark.parametrize("tools", [None, []])
@pytest.mark.parametrize("agent", ["explore", "web-researcher", "search-verifier", "cfg"])
async def test_omitting_tools_leaves_every_child_as_it_was(base, loops, mock_llm_client, agent, tools):
    runner = make_runner(base, mock_llm_client([]), max_subagent_depth=1, load_agent=lambda n: config_child(n))
    task = VERIFIER_TASK if agent == "search-verifier" else "t"
    result = await runner(agent, task, **({} if tools is None else {"tools": tools}))
    assert getattr(result, "tools", None) is None
    held = {"explore": {"read", "glob", "grep"}, "cfg": {"read", "grep", "bash_exec"}}.get(agent, WEB_SET)
    assert {frozenset(lp.held) for lp in loops} == {frozenset(held)}
    if agent == "cfg":
        assert loops[0].briefs == [prepend_toolset("Budget: no action limit, no time limit.\n\nt",
                                                   ["read", "grep", "bash_exec"])]
        assert "Ignore" not in loops[0].briefs[0]
    else:
        assert loops[0].briefs == [task]
    if agent == "web-researcher":
        assert loops[1].briefs == [ZERO_CALL_RETRY_PREFIX + task]
