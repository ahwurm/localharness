"""SCEN-02 + BENCH-03: bench runner — per-run agent loop wrapper + sequential sampling driver."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from localharness.bench.aggregator import should_stop
from localharness.bench.schema import ScenarioSpec
from localharness.core.bus import EventBus
from localharness.core.events import (
    Action,
    CompactionTriggered,
    Heartbeat,
    Observation,
    ParseFailed,
    ScenarioCompleted,
    StuckRecovered,
    TurnCompleted,
)

log = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# MetricAccumulator — subscribes to events, computes SCEN-02 fields per run
# -------------------------------------------------------------------------

@dataclass
class MetricAccumulator:
    """Stateful accumulator for one scenario run.

    Handler methods are SYNC so unit tests can invoke them directly without
    asyncio plumbing. execute_one_run wraps them in async closures when
    subscribing to the EventBus (which delivers via await).
    """

    tokens_in: int = 0
    tokens_out: int = 0
    iterations: int = 0
    parse_failures: int = 0
    stuck_recoveries: int = 0
    tool_call_count: int = 0
    tokens_estimated: bool = False
    final_message: str = ""
    internal_latencies: dict[str, float] = field(default_factory=dict)
    peak_context_pct: float = 0.0
    deny_events: int = 0
    compaction_triggered: int = 0

    def on_turn_completed(self, event: TurnCompleted) -> None:
        self.tokens_in += int(getattr(event, "input_tokens", 0) or 0)
        self.tokens_out += int(getattr(event, "output_tokens", 0) or 0)
        self.iterations = max(self.iterations, int(event.iterations))
        if getattr(event, "tokens_estimated", False):
            self.tokens_estimated = True
        if event.summary:
            self.final_message = event.summary

    def on_action(self, event: Action) -> None:
        if event.tool_name:
            self.tool_call_count += 1

    def on_parse_failed(self, event: ParseFailed) -> None:
        self.parse_failures += 1

    def on_stuck_recovered(self, event: StuckRecovered) -> None:
        self.stuck_recoveries += 1

    def on_heartbeat(self, event: Heartbeat) -> None:
        self.peak_context_pct = max(self.peak_context_pct, float(event.context_utilization_pct))

    def on_observation(self, event: Observation) -> None:
        if event.error and event.error.startswith("Permission denied:"):
            self.deny_events += 1

    def on_compaction_triggered(self, event: CompactionTriggered) -> None:
        self.compaction_triggered += 1


# -------------------------------------------------------------------------
# Path resolver
# -------------------------------------------------------------------------

def resolve_run_path(results_root: Path, model: str, scenario_name: str, timestamp: str) -> Path:
    """Return bench/results/{model}/{scenario_name}/{timestamp}.jsonl path."""
    return Path(results_root) / model / scenario_name / f"{timestamp}.jsonl"


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# -------------------------------------------------------------------------
# Per-run executor
# -------------------------------------------------------------------------

async def execute_one_run(
    scen: ScenarioSpec,
    model: str,
    run_path: Path,
    llm_client: Any,
    agent_config: Any = None,
) -> ScenarioCompleted:
    """Run one scenario, capturing JSONL trace and emitting ScenarioCompleted.

    Constructs a fresh EventBus(persist_path=run_path), subscribes a MetricAccumulator,
    builds an AgentLoop bound to that bus + llm_client, runs scen.prompt, and emits
    ScenarioCompleted on the bus before returning.

    TTFT limitation (CONTEXT.md known issue): native-mode providers do not actually
    stream tokens through on_token, so latency_ttft == latency_total in current state.
    Documented in ScenarioCompleted docstring (11-01).

    internal_latencies note (CONTEXT.md Claude's Discretion): the internal latency
    breakdown (model_gen, tool_exec, mcp) is NOT instrumented in v1.0.2 — the agent
    loop does not currently emit per-segment timing events. internal_latencies stays
    an empty dict in ScenarioCompleted for now; populating it is a follow-up phase
    once the loop is instrumented. This is intentional, not a bug.
    """
    run_path = Path(run_path)
    bus = EventBus(persist_path=run_path)
    acc = MetricAccumulator()
    # session_id threaded through to ContextManager so CompactionTriggered events
    # carry stable identity for the run. Derived from run_path stem (timestamp slug).
    session_id = f"{scen.name}:{run_path.stem}"

    # Wrap sync handlers as async closures for bus delivery
    async def _h_turn(ev: TurnCompleted) -> None:
        acc.on_turn_completed(ev)

    async def _h_action(ev: Action) -> None:
        acc.on_action(ev)

    async def _h_parse(ev: ParseFailed) -> None:
        acc.on_parse_failed(ev)

    async def _h_stuck(ev: StuckRecovered) -> None:
        acc.on_stuck_recovered(ev)

    async def _h_heartbeat(ev: Heartbeat) -> None:
        acc.on_heartbeat(ev)

    async def _h_observation(ev: Observation) -> None:
        acc.on_observation(ev)

    async def _h_compaction(ev: CompactionTriggered) -> None:
        acc.on_compaction_triggered(ev)

    bus.subscribe(TurnCompleted, _h_turn)
    bus.subscribe(Action, _h_action)
    bus.subscribe(ParseFailed, _h_parse)
    bus.subscribe(StuckRecovered, _h_stuck)
    bus.subscribe(Heartbeat, _h_heartbeat)
    bus.subscribe(Observation, _h_observation)
    bus.subscribe(CompactionTriggered, _h_compaction)

    t_start = time.monotonic()
    ttft: Optional[float] = None
    timed_out = False

    async def _on_token(_token: str) -> None:
        nonlocal ttft
        if ttft is None:
            ttft = time.monotonic() - t_start

    try:
        loop = _build_agent_loop(bus=bus, llm_client=llm_client, scenario=scen, session_id=session_id, agent_config=agent_config)
        try:
            await asyncio.wait_for(
                _run_loop(loop, scen.prompt, _on_token),
                timeout=scen.limits.max_latency_s,
            )
        except asyncio.TimeoutError:
            timed_out = True
            log.warning("scenario_timeout scenario=%s model=%s budget_s=%.1f",
                        scen.name, model, scen.limits.max_latency_s)
    except Exception as e:  # noqa: BLE001 — scenario failure is data, not a crash
        log.exception("scenario_error scenario=%s model=%s", scen.name, model)
        acc.final_message = f"[scenario_error: {e!r}]"

    latency_total = (scen.limits.max_latency_s if timed_out else time.monotonic() - t_start)
    counts = {
        "tokens_in": acc.tokens_in,
        "tokens_out": acc.tokens_out,
        "iterations": acc.iterations,
        "parse_failures": acc.parse_failures,
        "stuck_recoveries": acc.stuck_recoveries,
        "tool_call_count": acc.tool_call_count,
        "deny_events": acc.deny_events,
        "compaction_triggered": acc.compaction_triggered,
    }
    success = False if timed_out else scen.success_criteria.evaluate(acc.final_message, counts=counts)

    completed = ScenarioCompleted(
        scenario_name=scen.name,
        model=model,
        success=success,
        latency_ttft=(ttft if ttft is not None else latency_total),
        latency_total=latency_total,
        tokens_in=acc.tokens_in,
        tokens_out=acc.tokens_out,
        iterations=acc.iterations,
        parse_failures=acc.parse_failures,
        stuck_recoveries=acc.stuck_recoveries,
        tool_call_count=acc.tool_call_count,
        internal_latencies=acc.internal_latencies,
        tokens_estimated=acc.tokens_estimated,
    )
    await bus.publish(completed)
    return completed


# -------------------------------------------------------------------------
# Agent loop builder (overridable shim — keeps execute_one_run testable)
# -------------------------------------------------------------------------

def _build_agent_loop(bus: EventBus, llm_client: Any, scenario: ScenarioSpec, session_id: str = "", agent_config: Any = None) -> Any:
    """Construct an AgentLoop instance for the given scenario.

    AgentLoop signature (verified from src/localharness/agent/loop.py, lines 271-285):
        AgentLoop(
            config,                # AgentConfig
            llm,                   # LLMClient
            bus,                   # EventBus
            context_manager,       # ContextManager
            tool_registry,         # ToolRegistry
            permission_evaluator,  # PermissionEvaluator
            memory_loader=None,
            kill_file_path=None,
            compact_md_path=None,
        )

    Bench synthesizes a minimal AgentConfig + ContextManager + ToolRegistry +
    PermissionEvaluator from `scenario` (scenario.budget, scenario.tools_allowed).
    Tests monkey-patch this function to return a stub loop, bypassing construction.

    If AgentLoop's exact constructor diverges across LocalHarness versions,
    this is the ONE place that needs adjusting.
    """
    # Imports are local to avoid module-import-time cycles and to keep the
    # bench package importable even when these optional modules shift.
    from localharness.agent.loop import AgentLoop
    from localharness.agent.context import ContextManager
    from localharness.agent.permissions import PermissionEvaluator
    from localharness.config.models import AgentConfig
    from localharness.tools.registry import ToolRegistry

    agent_config = agent_config if agent_config is not None else AgentConfig(
        name=f"bench-{scenario.name}",
        role=f"Bench harness execution for scenario {scenario.name}",
    )

    ctx_manager = ContextManager(
        max_context_tokens=scenario.budget.max_context_tokens,
        bus=bus,
        agent_id=session_id,
        session_id=session_id,
    )
    tool_registry = ToolRegistry.from_allowed(scenario.tools_allowed)

    # Plan 12-04 Task 1: register AgentTool stub when 'agent' is in tools_allowed.
    # The stub agent_runner returns a canned summary containing STUB_SUBAGENT_OK
    # so fixtures exercising the agent-creation tool-call shape complete
    # deterministically — NOT a real subagent (does not invoke another LLM).
    if "agent" in scenario.tools_allowed:
        from localharness.tools.builtin.agent_tool import AgentTool

        async def _stub_agent_runner(agent_id: str, task: str) -> str:
            return (
                f"STUB_SUBAGENT_OK agent_id={agent_id} task={task[:80]}"
            )

        tool_registry._tools["global"]["agent"] = AgentTool(agent_runner=_stub_agent_runner)
        tool_registry._schemas["agent"] = tool_registry._tools["global"]["agent"].info()

    # PermissionEvaluator is stateless; constructor takes no args. (Plan 12-04
    # Rule 3 fix — the prior `from_config(agent_config)` referenced a method
    # that does not exist on PermissionEvaluator and would have crashed at
    # runtime. The evaluator reads deny_patterns from the config object passed
    # to `evaluate(...)` at call time, not at construction.)
    perm_evaluator = PermissionEvaluator()

    try:
        return AgentLoop(
            config=agent_config,
            llm=llm_client,
            bus=bus,
            context_manager=ctx_manager,
            tool_registry=tool_registry,
            permission_evaluator=perm_evaluator,
            memory_loader=None,
            kill_file_path=None,
            compact_md_path=None,
        )
    except TypeError as e:
        raise RuntimeError(
            f"AgentLoop construction failed in bench runner. "
            f"The current AgentLoop signature in src/localharness/agent/loop.py has changed — "
            f"update _build_agent_loop to match. Required positional args: "
            f"(config, llm, bus, context_manager, tool_registry, permission_evaluator). "
            f"Original error: {e!r}"
        ) from e


async def _run_loop(loop: Any, prompt: str, on_token: Callable) -> None:
    """Invoke the agent loop on the given prompt. Wraps the per-loop entry method.

    The loop's public entry method is `run_turn(task, on_token=...)` in current
    LocalHarness. If it diverges, update this shim only.
    """
    if hasattr(loop, "run_turn"):
        await loop.run_turn(task=prompt, on_token=on_token)
    elif callable(loop):
        await loop(prompt)
    else:
        raise RuntimeError(f"AgentLoop {loop!r} exposes no run_turn() entry point.")


# -------------------------------------------------------------------------
# Sequential sampling driver
# -------------------------------------------------------------------------

async def accumulate_runs(
    scen: ScenarioSpec,
    model: str,
    results_root: Path,
    llm_client_factory: Callable[[ScenarioSpec], Any],
    timestamp_fn: Callable[[], str] = _timestamp_now,
    min_runs_override: Optional[int] = None,
    max_runs_override: Optional[int] = None,
    agent_config: Any = None,
) -> tuple[list[ScenarioCompleted], str]:
    """Run scen repeatedly until should_stop says converged or max_runs hit.

    Returns (samples, stop_reason). stop_reason is the second element of should_stop's
    last return — caller can derive `stable` (converged or max_runs_hit-with-converged
    substring → True; max_runs_hit-without-converged → False).
    """
    samples: list[ScenarioCompleted] = []
    min_r = min_runs_override if min_runs_override is not None else scen.min_runs
    max_r = max_runs_override if max_runs_override is not None else scen.max_runs

    while True:
        timestamp = timestamp_fn()
        run_path = resolve_run_path(results_root, model, scen.name, timestamp)
        client = llm_client_factory(scen)
        result = await execute_one_run(scen, model, run_path, client, agent_config=agent_config)
        samples.append(result)
        stop, reason = should_stop(samples, scen.tolerance, min_r, max_r)
        if stop:
            return (samples, reason)


# -------------------------------------------------------------------------
# Re-export the top-level orchestrator entrypoint.
#
# `run_bench` is implemented in localharness.bench.orchestrator so the
# orchestration layer is decoupled from the per-run execution primitives in
# this module. We re-bind it here because the CLI (and its unit tests) drive
# bench execution via `localharness.bench.runner.run_bench` — keeping the
# import path stable lets monkeypatch.setattr work as expected.
# -------------------------------------------------------------------------

def __getattr__(name: str):  # PEP 562 — lazy attribute lookup
    if name == "run_bench":
        from localharness.bench.orchestrator import run_bench as _run_bench
        return _run_bench
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
