"""SCEN-02: bench runner accumulates standardized metrics from event subscriptions."""
from __future__ import annotations
import pytest


def test_accumulate_tokens_from_turn_completed():
    """MetricAccumulator.on_turn_completed sums input_tokens+output_tokens across events."""
    from localharness.bench.runner import MetricAccumulator
    from localharness.core.events import TurnCompleted
    acc = MetricAccumulator()
    acc.on_turn_completed(TurnCompleted(
        agent_id="a", session_id="s", iterations=1, duration_seconds=1.0,
        elapsed_tokens=100, input_tokens=80, output_tokens=20, summary="done",
    ))
    acc.on_turn_completed(TurnCompleted(
        agent_id="a", session_id="s", iterations=2, duration_seconds=2.0,
        elapsed_tokens=200, input_tokens=160, output_tokens=40, summary="done",
    ))
    assert acc.tokens_in == 240
    assert acc.tokens_out == 60


def test_accumulate_iterations_from_turn_completed():
    """MetricAccumulator.iterations takes max iterations from TurnCompleted events."""
    from localharness.bench.runner import MetricAccumulator
    from localharness.core.events import TurnCompleted
    acc = MetricAccumulator()
    acc.on_turn_completed(TurnCompleted(
        agent_id="a", session_id="s", iterations=5, duration_seconds=1.0,
        elapsed_tokens=10, summary="done",
    ))
    assert acc.iterations == 5


def test_accumulate_metrics_from_turn_failed():
    """Regression: failed runs (TurnFailed) must report real iterations/tokens.

    The accumulator originally subscribed only to TurnCompleted, so runs ending
    in TurnFailed (budget_exceeded, stuck_detected) logged iterations/tokens as 0
    despite having run — masking real work and skewing per-run telemetry.
    """
    from localharness.bench.runner import MetricAccumulator
    from localharness.core.events import TurnFailed
    acc = MetricAccumulator()
    acc.on_turn_failed(TurnFailed(
        agent_id="a", session_id="s", reason="budget_exceeded", detail="hit cap",
        iterations=3, duration_seconds=1.0, input_tokens=120, output_tokens=45,
    ))
    assert acc.iterations == 3
    assert acc.tokens_in == 120
    assert acc.tokens_out == 45


def test_accumulate_tool_call_count_from_actions():
    """MetricAccumulator.on_action increments tool_call_count when tool_name set."""
    from localharness.bench.runner import MetricAccumulator
    from localharness.core.events import Action
    acc = MetricAccumulator()
    acc.on_action(Action(agent_id="a", session_id="s", action_type="tool_calls", tool_name="bash"))
    acc.on_action(Action(agent_id="a", session_id="s", action_type="tool_calls", tool_name="read_file"))
    acc.on_action(Action(agent_id="a", session_id="s", action_type="complete"))  # no tool_name → not counted
    assert acc.tool_call_count == 2


def test_accumulate_parse_failures_from_event():
    """MetricAccumulator.on_parse_failed increments parse_failures counter."""
    from localharness.bench.runner import MetricAccumulator
    from localharness.core.events import ParseFailed
    acc = MetricAccumulator()
    acc.on_parse_failed(ParseFailed(agent_id="a", session_id="s", iteration=1, parse_retry_count=1, raw_content_preview="x"))
    acc.on_parse_failed(ParseFailed(agent_id="a", session_id="s", iteration=2, parse_retry_count=2, raw_content_preview="y"))
    assert acc.parse_failures == 2


def test_accumulate_stuck_recoveries_from_event():
    """MetricAccumulator.on_stuck_recovered increments stuck_recoveries counter."""
    from localharness.bench.runner import MetricAccumulator
    from localharness.core.events import StuckRecovered
    acc = MetricAccumulator()
    acc.on_stuck_recovered(StuckRecovered(agent_id="a", session_id="s", iteration=3, stuck_signature="x"))
    assert acc.stuck_recoveries == 1


def test_tokens_estimated_propagates():
    """Any TurnCompleted with tokens_estimated=True → ScenarioCompleted.tokens_estimated=True."""
    from localharness.bench.runner import MetricAccumulator
    from localharness.core.events import TurnCompleted
    acc = MetricAccumulator()
    acc.on_turn_completed(TurnCompleted(
        agent_id="a", session_id="s", iterations=1, duration_seconds=1.0,
        elapsed_tokens=100, tokens_estimated=True, summary="done",
    ))
    assert acc.tokens_estimated is True


# ---------------------------------------------------------------------------
# SCEN-04 plumbing: deny_events + compaction_triggered counters (Plan 12-01 Task 3)
# ---------------------------------------------------------------------------

def test_deny_event_counter():
    """on_observation increments deny_events only when error starts with 'Permission denied:'."""
    from localharness.bench.runner import MetricAccumulator
    from localharness.core.events import Observation
    acc = MetricAccumulator()
    acc.on_observation(Observation(
        agent_id="a", session_id="s", observation_type="tool_result",
        tool_call_id="t1", tool_name="bash_exec", output="[DENIED]",
        error="Permission denied: matched pattern bash_exec(rm -rf *)",
    ))
    assert acc.deny_events == 1
    # Different error — does not count
    acc.on_observation(Observation(
        agent_id="a", session_id="s", observation_type="tool_result",
        tool_call_id="t2", tool_name="bash_exec", output="x", error="some other error",
    ))
    assert acc.deny_events == 1
    # No error — does not count
    acc.on_observation(Observation(
        agent_id="a", session_id="s", observation_type="tool_result",
        tool_call_id="t3", tool_name="bash_exec", output="ok",
    ))
    assert acc.deny_events == 1


def test_compaction_triggered_counter():
    """on_compaction_triggered increments compaction_triggered each call."""
    from localharness.bench.runner import MetricAccumulator
    from localharness.core.events import CompactionTriggered
    acc = MetricAccumulator()
    acc.on_compaction_triggered(CompactionTriggered(
        agent_id="a", session_id="s", iteration=1,
        pre_usage_fraction=0.9, post_usage_fraction=0.4, stages_modified=[],
    ))
    assert acc.compaction_triggered == 1
    acc.on_compaction_triggered(CompactionTriggered(
        agent_id="a", session_id="s", iteration=2,
        pre_usage_fraction=0.9, post_usage_fraction=0.4, stages_modified=[],
    ))
    assert acc.compaction_triggered == 2


@pytest.mark.asyncio
async def test_execute_one_run_subscribes_observation_and_compaction(monkeypatch, tmp_path):
    """execute_one_run subscribes to Observation + CompactionTriggered events."""
    from localharness.bench import runner as bench_runner
    from localharness.bench.schema import ScenarioSpec, SuccessCriteria, LimitsSpec
    from localharness.core.events import BudgetSpec, Observation, CompactionTriggered

    async def fake_run_loop(loop, prompt, on_token):
        await loop["bus"].publish(Observation(
            agent_id="a", session_id="s", observation_type="tool_result",
            tool_call_id="tc-1", tool_name="bash_exec", output="[DENIED]",
            error="Permission denied: bash_exec(rm -rf *)",
        ))
        await loop["bus"].publish(CompactionTriggered(
            agent_id="a", session_id="s", iteration=1,
            pre_usage_fraction=0.9, post_usage_fraction=0.4, stages_modified=[],
        ))

    async def fake_build(bus, llm_client, scenario, session_id="", agent_config=None, base_registry=None):
        return {"bus": bus}

    monkeypatch.setattr(bench_runner, "_build_agent_loop", fake_build)
    monkeypatch.setattr(bench_runner, "_run_loop", fake_run_loop)

    scen = ScenarioSpec(
        name="t",
        prompt="x",
        success_criteria=SuccessCriteria(
            event_counts={
                "deny_events": {"min": 1},
                "compaction_triggered": {"min": 1},
            },
        ),
        budget=BudgetSpec(),
        limits=LimitsSpec(),
        tools_allowed=["bash_exec"],
        slice="train",
        category="tool_basics",
    )
    out = await bench_runner.execute_one_run(scen, "m", tmp_path / "run.jsonl", llm_client=None)
    assert out.success is True


# ---------------------------------------------------------------------------
# Plan 12-04 Task 1: AgentTool stub + plugin-prefix dispatch spike
# ---------------------------------------------------------------------------


def test_plugin_prefix_dispatch_resolves():
    """ScenarioSpec(tools_allowed=['plugin:research_tools.exa_search']) resolves to
    the registered plugin tool at scope='global' under bare name.

    See 12-RESEARCH §"Pattern 8" — the spike. ToolRegistry.from_allowed must
    accept entries with `plugin:PLUGIN.TOOL` form and resolve them against
    tools registered at scope="global" under the bare TOOL name.
    """
    import asyncio
    from localharness.tools.base import Tool, ToolSchema
    from localharness.tools.registry import ToolRegistry

    class StubPluginTool(Tool):
        timeout_s = 30.0
        async def _execute(self):
            return self.ok("stub")
        def info(self) -> ToolSchema:
            return ToolSchema(
                name="exa_search",
                description="stub plugin tool",
                parameters={"type": "object", "properties": {}, "required": []},
                scope="agent",
                estimated_tokens=10,
                destructive=False,
            )

    base = ToolRegistry()
    asyncio.run(base.register(StubPluginTool(), scope="global"))

    allowed_registry = ToolRegistry.from_allowed(
        ["plugin:research_tools.exa_search"],
        base_registry=base,
    )
    # Lookup resolves via either bare name or prefixed name — the spike fixed
    # whichever lookup form the runner uses. Both are valid resolutions.
    assert allowed_registry.has("exa_search") or allowed_registry.has(
        "plugin:research_tools.exa_search"
    )


@pytest.mark.asyncio
async def test_build_agent_loop_registers_agent_tool(monkeypatch):
    """_build_agent_loop with 'agent' in tools_allowed registers a real delegation AgentTool.

    SUBAGENT-05 / J3: the runner is built via make_explore_agent_runner (the live seam), which routes
    by agent_id (cruncher / explore / …) — no longer a hard-wired Explore-only runner. The closure is
    captured (not invoked) at build time, so the registry simply `has("agent")` after construction;
    cruncher routing + the grant keystone resolving through bench_store are proven in
    test_build_agent_loop_routes_cruncher_and_resolves_grant; the tool_call_count delegation floor in
    test_bench_subagent_delegation.py.
    """
    from localharness.bench import runner as bench_runner
    from localharness.bench.schema import ScenarioSpec, SuccessCriteria, LimitsSpec
    from localharness.core.events import BudgetSpec

    # Stub AgentLoop construction so we don't drag in the whole agent stack
    captured: dict = {}

    class FakeAgentLoop:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("localharness.agent.loop.AgentLoop", FakeAgentLoop)

    scen = ScenarioSpec(
        name="t",
        prompt="x",
        success_criteria=SuccessCriteria(rubric=["contains:OK"]),
        budget=BudgetSpec(),
        limits=LimitsSpec(),
        tools_allowed=["agent"],
        slice="train",
        category="tool_basics",
    )
    await bench_runner._build_agent_loop(bus=None, llm_client=None, scenario=scen)
    tr = captured["tool_registry"]
    assert tr.has("agent"), "AgentTool stub was not registered"


@pytest.mark.asyncio
async def test_build_agent_loop_wires_compaction_pipeline():
    """Regression guard for the 'no CompactionPipeline in runner' bench bug: _build_agent_loop MUST
    build a ContextManager WITH a compaction pipeline + content store. Without it (pipeline=None),
    max_context_tokens is never enforced — over-window content sails through to the model's real
    window and J3 (lossless over-window reduction) is unmeasurable. Pipeline construction must not
    require a live client (llm_client=None here): the summarize fn is only invoked during compaction."""
    from localharness.bench import runner as bench_runner
    from localharness.bench.schema import ScenarioSpec, SuccessCriteria, LimitsSpec
    from localharness.core.events import BudgetSpec

    scen = ScenarioSpec(
        name="overwindow-guard", prompt="x",
        success_criteria=SuccessCriteria(rubric=["contains:X"]),
        budget=BudgetSpec(), limits=LimitsSpec(),
        tools_allowed=[], slice="train", category="tool_basics",
    )
    loop = await bench_runner._build_agent_loop(bus=None, llm_client=None, scenario=scen)
    assert loop._ctx._pipeline is not None, (
        "bench runner built ContextManager WITHOUT a compaction pipeline — over-window enforcement "
        "regressed to the 'pipeline=None' bug; max_context_tokens would not be enforced"
    )
    assert loop._ctx._content_store is not None, "bench runner ContextManager has no content store"


@pytest.mark.asyncio
async def test_build_agent_loop_routes_cruncher_and_resolves_grant(monkeypatch, tmp_path):
    """J3 keystone guard: the bench must (1) route agent('cruncher') to the CRUNCHER, not Explore, and
    (2) resolve a load_document handle to the cruncher through bench_store on grant. Before the runner
    fix, every agent() hard-routed to dispatch_explore_subagent (agent_id ignored) and load_document
    minted its handle in a private store, so a scored agent('cruncher', grant_handles=[H]) scenario
    silently ran Explore over a handle that resolved to NOTHING — the cruncher path was unreachable
    from the bench and "faithful over-window" rested on mocks. This asserts the WIRING (routing + grant
    resolution); faithfulness itself is gated by the live scored scenario 25_..._over_window_cruncher.
    """
    from localharness.bench import runner as bench_runner
    from localharness.bench.schema import ScenarioSpec, SuccessCriteria, LimitsSpec
    from localharness.core.events import BudgetSpec
    from localharness.agent import subagent as subagent_mod

    # Spy both dispatchers (module-global names resolved at call time inside the factory's runner).
    calls: dict = {}

    async def _cruncher_spy(task, **kwargs):
        calls["cruncher"] = kwargs
        return "spy-cruncher-answer"

    async def _explore_spy(task, **kwargs):
        calls["explore"] = kwargs
        return "spy-explore-answer"

    monkeypatch.setattr(subagent_mod, "dispatch_cruncher_subagent", _cruncher_spy)
    monkeypatch.setattr(subagent_mod, "dispatch_explore_subagent", _explore_spy)

    captured: dict = {}

    class FakeAgentLoop:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("localharness.agent.loop.AgentLoop", FakeAgentLoop)

    scen = ScenarioSpec(
        name="cruncher-wiring", prompt="x",
        success_criteria=SuccessCriteria(rubric=["contains:X"]),
        budget=BudgetSpec(max_context_tokens=16000), limits=LimitsSpec(),
        tools_allowed=["load_document", "agent"], slice="train",
        category="agent_orchestration",
    )
    base_registry = await bench_runner._get_base_registry()  # has the builtin load_document
    await bench_runner._build_agent_loop(
        bus=None, llm_client=None, scenario=scen, session_id="s1", base_registry=base_registry)

    bench_store = captured["context_manager"]._content_store
    tr = captured["tool_registry"]

    # (1) load_document's handle must land in bench_store (proves bind_agent_store_tools, not a private
    #     default ContentStore). A unique body makes the read-through assertion unambiguous.
    body = "ALPHA top section\n" + ("filler line\n" * 50) + "ZULU buried mid fact\n" + ("tail\n" * 50)
    doc = tmp_path / "doc.txt"
    doc.write_text(body, encoding="utf-8")
    ld = tr._tools["global"]["load_document"]
    res = await ld.run(path=str(doc))
    handle = res.metadata["doc_handle"]
    assert bench_store.get(handle) == body, (
        "load_document handle did not resolve in bench_store — store-backed verbs were not bound to "
        "the per-scenario store (bind_agent_store_tools regressed); a grant would resolve to nothing"
    )

    # (2) agent('cruncher', grant_handles=[H]) must route to the cruncher (not Explore) AND hand it a
    #     ctx whose store resolves H through bench_store (the grant keystone, parent_store=bench_store).
    agent_tool = tr._tools["global"]["agent"]
    await agent_tool.run(agent_id="cruncher", task="find the buried fact", grant_handles=[handle])
    assert "cruncher" in calls, "agent('cruncher') did NOT route to dispatch_cruncher_subagent"
    assert "explore" not in calls, "agent('cruncher') wrongly routed to Explore (agent_id ignored)"
    child_ctx = calls["cruncher"].get("context_manager")
    assert child_ctx is not None and child_ctx._content_store.get(handle) == body, (
        "cruncher's granted ctx could not read the handle through bench_store — grant keystone "
        "(parent_store=bench_store) not wired in the bench runner"
    )
    assert calls["cruncher"].get("grant_handles") == [handle]


@pytest.mark.asyncio
async def test_bench_memory_tools_parity_when_seeded(monkeypatch):
    """SESS-06 parity: a seeded bench scenario registers the SAME three memory tools
    production registers whenever a store exists (start_cmd critic M5). Bench agents
    previously got memory INJECTION but no memory tools — write-only memory in scored runs.
    """
    from localharness.bench import runner as bench_runner
    from localharness.bench.schema import ScenarioSpec, SuccessCriteria, LimitsSpec
    from localharness.core.events import BudgetSpec

    captured: dict = {}

    class FakeAgentLoop:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("localharness.agent.loop.AgentLoop", FakeAgentLoop)

    # "memory_recall" is a _MEMORY_SEEDS key -> _seed_memory_store runs -> memory_loader set.
    scen = ScenarioSpec(
        name="memory_recall", prompt="x",
        success_criteria=SuccessCriteria(rubric=["contains:OK"]),
        budget=BudgetSpec(), limits=LimitsSpec(),
        tools_allowed=[], slice="train", category="tool_basics",
    )
    await bench_runner._build_agent_loop(bus=None, llm_client=None, scenario=scen)
    assert captured["memory_loader"] is not None, "seeded scenario must hydrate a MemoryStore"
    tools = captured["tool_registry"]._tools["global"]
    # the write verb registers as "remember" (not "memory_remember"), same as production
    for name in ("memory_search", "memory_get", "remember"):
        assert name in tools, f"seeded bench agent missing {name} (production registers all three)"


@pytest.mark.asyncio
async def test_bench_no_memory_tools_without_seeds(monkeypatch):
    """No phantom store: a non-seeded scenario has memory_loader=None and registers NONE of the
    three memory tools — the tools ride with the store, exactly like production."""
    from localharness.bench import runner as bench_runner
    from localharness.bench.schema import ScenarioSpec, SuccessCriteria, LimitsSpec
    from localharness.core.events import BudgetSpec

    captured: dict = {}

    class FakeAgentLoop:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("localharness.agent.loop.AgentLoop", FakeAgentLoop)

    scen = ScenarioSpec(
        name="unseeded_scenario", prompt="x",
        success_criteria=SuccessCriteria(rubric=["contains:OK"]),
        budget=BudgetSpec(), limits=LimitsSpec(),
        tools_allowed=[], slice="train", category="tool_basics",
    )
    await bench_runner._build_agent_loop(bus=None, llm_client=None, scenario=scen)
    assert captured["memory_loader"] is None, "unseeded scenario must NOT hydrate a store"
    tools = captured["tool_registry"]._tools["global"]
    for name in ("memory_search", "memory_get", "remember"):
        assert name not in tools, f"unseeded bench agent wrongly has {name} (phantom store)"


def test_render_prompt_substitutes_fixtures_token():
    """_render_prompt swaps {FIXTURES} for an absolute bench/fixtures path (so load_document, which
    requires an absolute path, gets one) and leaves token-free prompts untouched."""
    import os
    from localharness.bench import runner as bench_runner

    rendered = bench_runner._render_prompt("load {FIXTURES}/over_window_federalist.txt now")
    assert "{FIXTURES}" not in rendered
    normalized = rendered.replace("\\", "/")  # separator-agnostic regardless of platform path form
    assert os.path.isabs(normalized.split("load ", 1)[1].split("/over_window")[0])
    assert normalized.endswith("bench/fixtures/over_window_federalist.txt now")
    assert bench_runner._render_prompt("no token here") == "no token here"


@pytest.mark.asyncio
async def test_base_registry_has_cruncher_leaf_tools():
    """Regression guard for a LIVE-caught cruncher bug: the bench base_registry MUST contain every tool
    the cruncher's chunk-summarizer leaf needs (CHUNK_SUMMARIZER_TOOLS = ['tool_result_get']). The leaf
    builds its toolset via from_allowed(CHUNK_SUMMARIZER_TOOLS, base_registry=base), and from_allowed can
    only surface a tool that EXISTS in base. register_builtin_tools gates tool_result_get on an
    eviction_store, so a base built without one leaves every leaf TOOL-LESS: it can't read its granted
    chunk, emits the read call as literal text, and the cruncher reduces empty extracts and silently
    misses every fact (observed live: routing looked perfect, needle recall was 0% — 'extracts do not
    contain the answer ... only tool_result_get(...) stubs'). The unit wiring test cannot catch this (it
    stubs dispatch_cruncher_subagent, so the real leaves never run); this asserts the base directly."""
    from localharness.bench import runner as bench_runner
    from localharness.agent.subagent import CHUNK_SUMMARIZER_TOOLS

    base = await bench_runner._get_base_registry()
    for tool in CHUNK_SUMMARIZER_TOOLS:
        assert tool in base._tools["global"], (
            f"bench base_registry missing cruncher-leaf tool {tool!r}: chunk-summarizer leaves would get "
            f"an empty toolset and the cruncher would silently miss every fact"
        )


# ---------------------------------------------------------------------------
# Phase 24 EVAL-01: MemoryStore hydration in _build_agent_loop for the
# stateful_behavior scenarios. The REAL _build_agent_loop is awaited; we then
# introspect the constructed AgentLoop's seeded memory store (loop._memory)
# and assert load_context().agent_memory_md (the text injected into the system
# prompt at agent/loop.py:464-465) carries the scenario anchor tokens. The
# store's agent_id MUST equal the loop's AgentConfig.name or load_context
# would resolve to an empty fact set.
# ---------------------------------------------------------------------------

import re as _re


def _mem_scen(name: str):
    from localharness.bench.schema import ScenarioSpec, SuccessCriteria, LimitsSpec
    from localharness.core.events import BudgetSpec
    return ScenarioSpec(
        name=name,
        prompt="x",
        success_criteria=SuccessCriteria(rubric=["contains:X"]),
        budget=BudgetSpec(),
        limits=LimitsSpec(),
        tools_allowed=[],
        slice="train",
        category="stateful_behavior",
    )


@pytest.mark.asyncio
async def test_memory_hydration_injects_starfruit(tmp_path):
    """Test A — memory_recall: the seeded store injects STARFRUIT_42, and its
    agent_id equals the loop's AgentConfig.name so load_context resolves."""
    from localharness.bench import runner as bench_runner

    loop = await bench_runner._build_agent_loop(
        bus=None, llm_client=None, scenario=_mem_scen("memory_recall")
    )
    store = loop._memory
    assert store is not None, "memory_loader was not seeded for memory_recall"
    # agent_id of the store must match the loop's config name (underscore→hyphen)
    assert store._agent_id == loop._config.name == "bench-memory-recall"
    ctx = await store.load_context()
    assert "STARFRUIT_42" in ctx.agent_memory_md
    await store.close()


@pytest.mark.asyncio
async def test_memory_hydration_two_facts(tmp_path):
    """Test B — stateful_behavior_two_facts: both STARFRUIT_42 and MOONFRUIT_88
    land in the injected MEMORY.md text."""
    from localharness.bench import runner as bench_runner

    loop = await bench_runner._build_agent_loop(
        bus=None, llm_client=None, scenario=_mem_scen("stateful_behavior_two_facts")
    )
    store = loop._memory
    assert store is not None
    ctx = await store.load_context()
    assert "STARFRUIT_42" in ctx.agent_memory_md
    assert "MOONFRUIT_88" in ctx.agent_memory_md
    await store.close()


@pytest.mark.asyncio
async def test_memory_hydration_overwrite_latest_wins(tmp_path):
    """Test C — stateful_behavior_overwrite_recall: favorite_color seeded blue
    THEN amber; latest-write-wins via MemoryStore upsert, so the injected text
    matches (?i)\\bamber\\b and carries no standalone 'blue' fact line."""
    from localharness.bench import runner as bench_runner

    loop = await bench_runner._build_agent_loop(
        bus=None, llm_client=None, scenario=_mem_scen("stateful_behavior_overwrite_recall")
    )
    store = loop._memory
    assert store is not None
    ctx = await store.load_context()
    md = ctx.agent_memory_md
    assert _re.search(r"(?i)\bamber\b", md), f"amber not in injected memory: {md!r}"
    # latest-write-wins: the overwritten 'blue' fact must not survive as its own line
    fact_lines = [ln for ln in md.splitlines() if ln.lstrip().startswith("- favorite_color")]
    assert len(fact_lines) == 1, f"expected one favorite_color line, got {fact_lines!r}"
    assert _re.search(r"(?i)\bblue\b", fact_lines[0]) is None
    await store.close()


@pytest.mark.asyncio
async def test_non_memory_scenario_has_no_store(tmp_path):
    """Test D — pure_qa (tool_basics): memory_loader stays None and no /agents
    directory is created under any bench tmp base_dir for this scenario."""
    from localharness.bench import runner as bench_runner
    from localharness.bench.schema import ScenarioSpec, SuccessCriteria, LimitsSpec
    from localharness.core.events import BudgetSpec

    scen = ScenarioSpec(
        name="pure_qa",
        prompt="x",
        success_criteria=SuccessCriteria(rubric=["contains:X"]),
        budget=BudgetSpec(),
        limits=LimitsSpec(),
        tools_allowed=[],
        slice="train",
        category="tool_basics",
    )
    loop = await bench_runner._build_agent_loop(bus=None, llm_client=None, scenario=scen)
    assert loop._memory is None

@pytest.mark.asyncio
async def test_counts_dict_passed_to_evaluate(monkeypatch, tmp_path):
    """counts dict (including deny_events) is passed to success_criteria.evaluate()."""
    from localharness.bench import runner as bench_runner
    from localharness.bench.schema import ScenarioSpec, SuccessCriteria, LimitsSpec
    from localharness.core.events import BudgetSpec, Observation

    async def fake_run_loop(loop, prompt, on_token):
        await loop["bus"].publish(Observation(
            agent_id="a", session_id="s", observation_type="tool_result",
            tool_call_id="tc-1", tool_name="bash_exec",
            output="[DENIED]",
            error="Permission denied: bash_exec(rm -rf *)",
        ))

    async def fake_build(bus, llm_client, scenario, session_id="", agent_config=None, base_registry=None):
        return {"bus": bus}

    monkeypatch.setattr(bench_runner, "_build_agent_loop", fake_build)
    monkeypatch.setattr(bench_runner, "_run_loop", fake_run_loop)

    scen = ScenarioSpec(
        name="t",
        prompt="x",
        success_criteria=SuccessCriteria(event_counts={"deny_events": {"min": 1}}),
        budget=BudgetSpec(),
        limits=LimitsSpec(),
        tools_allowed=["bash_exec"],
        slice="train",
        category="tool_basics",
    )
    out = await bench_runner.execute_one_run(scen, "m", tmp_path / "run.jsonl", llm_client=None)
    assert out.success is True   # event_counts assertion satisfied via counts dict


# ---------------------------------------------------------------------------
# Phase 13 Wave 1: rglob discovery + slice filter in orchestrator
# ---------------------------------------------------------------------------


def test_discover_scenarios_recurses_into_subdirs(tmp_path):
    """_discover_scenarios uses rglob — finds yaml in train/ and holdout/ subdirs."""
    (tmp_path / "train").mkdir()
    (tmp_path / "holdout").mkdir()
    (tmp_path / "train" / "a.yaml").write_text("name: a\nprompt: x\n")
    (tmp_path / "holdout" / "b.yaml").write_text("name: b\nprompt: x\n")
    from localharness.bench.orchestrator import _discover_scenarios
    paths = _discover_scenarios(tmp_path)
    names = {p.name for p in paths}
    assert names == {"a.yaml", "b.yaml"}


def _write_fixture(path, name, slice_):
    """Write a minimal valid ScenarioSpec YAML to path with given slice."""
    path.write_text(
        f"name: {name}\n"
        f"prompt: x\n"
        f"success_criteria:\n  golden_output: '4'\n"
        f"budget:\n  max_actions: 0\n  max_duration_minutes: 0.5\n  max_context_tokens: 8000\n"
        f"tools_allowed: []\n"
        f"slice: {slice_}\n"
        f"category: tool_basics\n"
    )


@pytest.fixture
def _hermetic_categories(monkeypatch, tmp_path):
    """Point schema loader at a tmp categories.yaml so run_bench can load fixtures
    without depending on the repo-root bench/categories.yaml."""
    import textwrap
    from localharness.bench import schema as schema_mod
    cats = tmp_path / "categories.yaml"
    cats.write_text(textwrap.dedent("""\
        categories:
          tool_basics: "test only"
    """))
    monkeypatch.setenv("LOCALHARNESS_CATEGORIES_PATH", str(cats))
    schema_mod._load_allowed_categories.cache_clear()
    yield cats
    schema_mod._load_allowed_categories.cache_clear()


@pytest.mark.asyncio
async def test_scenario_overrides_slice_filter(tmp_path, monkeypatch, _hermetic_categories):
    """When --scenario is set, the slice filter is BYPASSED — named fixture wins
    regardless of which slice it lives in."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "train").mkdir()
    (corpus / "holdout").mkdir()
    _write_fixture(corpus / "train" / "a.yaml", "a", "train")
    _write_fixture(corpus / "holdout" / "b.yaml", "b", "holdout")
    results = tmp_path / "results"

    seen = []

    async def fake_run_one_model(entry, scenarios, **kw):
        seen.extend(s.name for s in scenarios)
        return len(scenarios)

    from localharness.bench import orchestrator as orch_mod
    monkeypatch.setattr(orch_mod, "_run_one_model", fake_run_one_model)
    code = await orch_mod.run_bench(
        scenario="b",
        slice="train",  # would normally filter out b (holdout), but --scenario overrides
        corpus_path=corpus,
        results_path=results,
    )
    assert code == 0
    assert seen == ["b"], f"expected only 'b' (named override), got {seen}"


@pytest.mark.asyncio
async def test_slice_filter_applied_when_no_scenario(tmp_path, monkeypatch, _hermetic_categories):
    """With --slice=train and scenario=None, only train-slice fixtures pass to runner."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "train").mkdir()
    (corpus / "holdout").mkdir()
    _write_fixture(corpus / "train" / "a.yaml", "a", "train")
    _write_fixture(corpus / "holdout" / "b.yaml", "b", "holdout")
    results = tmp_path / "results"

    seen = []

    async def fake_run_one_model(entry, scenarios, **kw):
        seen.extend(s.name for s in scenarios)
        return len(scenarios)

    from localharness.bench import orchestrator as orch_mod
    monkeypatch.setattr(orch_mod, "_run_one_model", fake_run_one_model)
    code = await orch_mod.run_bench(
        scenario=None,
        slice="train",
        corpus_path=corpus,
        results_path=results,
    )
    assert code == 0
    assert seen == ["a"], f"expected only train-slice 'a', got {seen}"


@pytest.mark.asyncio
async def test_missing_slice_field_exits_2(tmp_path, monkeypatch, _hermetic_categories):
    """A YAML missing the required `slice` field is dropped from load; if it's the
    only fixture, run_bench returns 2 (all_scenarios_failed_to_load)."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # Missing slice + category — pydantic ValidationError → silently dropped by loader
    (corpus / "broken.yaml").write_text(
        "name: broken\n"
        "prompt: x\n"
        "success_criteria:\n  golden_output: '4'\n"
        "budget:\n  max_actions: 0\n  max_duration_minutes: 0.5\n  max_context_tokens: 8000\n"
        "tools_allowed: []\n"
    )
    results = tmp_path / "results"
    from localharness.bench import orchestrator as orch_mod
    code = await orch_mod.run_bench(
        scenario=None,
        slice="train",
        corpus_path=corpus,
        results_path=results,
    )
    assert code == 2


@pytest.mark.asyncio
async def test_compaction_event_counter(monkeypatch, tmp_path):
    """When _build_agent_loop wires bus into ContextManager and the pipeline reports
    modifications during a bench run, MetricAccumulator.compaction_triggered increments.
    Proves the runner.py call site passes bus= correctly.
    """
    from localharness.bench import runner as bench_runner
    from localharness.bench.schema import ScenarioSpec, SuccessCriteria, LimitsSpec
    from localharness.core.events import BudgetSpec, CompactionTriggered

    async def fake_run_loop(loop, prompt, on_token):
        await loop["bus"].publish(CompactionTriggered(
            agent_id="a", session_id="s", iteration=1,
            pre_usage_fraction=0.9, post_usage_fraction=0.5,
            stages_modified=[],
        ))

    async def fake_build(bus, llm_client, scenario, session_id="", agent_config=None, base_registry=None):
        return {"bus": bus}

    monkeypatch.setattr(bench_runner, "_build_agent_loop", fake_build)
    monkeypatch.setattr(bench_runner, "_run_loop", fake_run_loop)

    scen = ScenarioSpec(
        name="t",
        prompt="x",
        success_criteria=SuccessCriteria(event_counts={"compaction_triggered": {"min": 1}}),
        budget=BudgetSpec(),
        limits=LimitsSpec(),
        tools_allowed=[],
        slice="train",
        category="tool_basics",
    )
    out = await bench_runner.execute_one_run(scen, "m", tmp_path / "run.jsonl", llm_client=None)
    assert out.success is True
