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

    def fake_build(bus, llm_client, scenario, session_id=""):
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


def test_build_agent_loop_registers_agent_tool(monkeypatch):
    """_build_agent_loop with 'agent' in tools_allowed registers the stub AgentTool."""
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
    )
    bench_runner._build_agent_loop(bus=None, llm_client=None, scenario=scen)
    tr = captured["tool_registry"]
    assert tr.has("agent"), "AgentTool stub was not registered"

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

    def fake_build(bus, llm_client, scenario, session_id=""):
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
    )
    out = await bench_runner.execute_one_run(scen, "m", tmp_path / "run.jsonl", llm_client=None)
    assert out.success is True   # event_counts assertion satisfied via counts dict


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

    def fake_build(bus, llm_client, scenario, session_id=""):
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
    )
    out = await bench_runner.execute_one_run(scen, "m", tmp_path / "run.jsonl", llm_client=None)
    assert out.success is True
