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
    TurnFailed,
)

log = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# EVAL-01: bench-side memory seed map for stateful_behavior scenarios.
#
# Keyed by scenario NAME (NOT a ScenarioSpec field — this is bench-internal
# data, the minimum needed because the on-disk seed DB lacks facts for the
# two_facts/overwrite scenarios authored ahead of this wiring). For
# overwrite_recall ORDER MATTERS: the second favorite_color write wins via
# MemoryStore's ON CONFLICT(agent_id,key) upsert (latest-write-wins).
# -------------------------------------------------------------------------

_MEMORY_SEEDS: dict[str, list[tuple[str, str]]] = {
    "memory_recall": [("codename", "The codename from a previous session is STARFRUIT_42.")],
    "stateful_behavior_two_facts": [
        ("codename_1", "The first codename from a previous session is STARFRUIT_42."),
        ("codename_2", "The second codename from a previous session is MOONFRUIT_88."),
    ],
    "stateful_behavior_overwrite_recall": [
        ("favorite_color", "blue"),
        ("favorite_color", "amber"),
    ],
}


async def _seed_memory_store(agent_id: str, seeds: list[tuple[str, str]]) -> Any:
    """Build a seeded, flushed v1.0 MemoryStore under a per-call tmp base_dir.

    Constructed under the SAME agent_id as the loop's AgentConfig.name so
    load_context() resolves the facts (building fresh — rather than copying
    tests/fixtures/bench/memory_seed.db — resolves BOTH the underscore/hyphen
    agent_id mismatch AND the flat-file vs agents/{id}/memory.db layout
    mismatch in one stroke).

    flush_memory_md() is REQUIRED: AgentLoop injects load_context().agent_memory_md
    (the MEMORY.md "## Persistent Facts" text), NOT the raw facts table — so
    store_fact() without the flush would inject nothing.
    """
    import tempfile
    from localharness.memory.sqlite import MemoryStore

    store = MemoryStore(
        agent_id=agent_id, division_id="", org_id="",
        base_dir=tempfile.mkdtemp(prefix="bench-mem-"),
    )
    await store.open()
    for key, value in seeds:
        await store.store_fact(key, value, confidence=1.0)  # >=0.7 so flush includes it
    await store.flush_memory_md()  # load_context reads MEMORY.md, not the facts table
    return store


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

    def on_turn_failed(self, event: TurnFailed) -> None:
        # Failed runs never emit TurnCompleted; capture their real iterations/tokens.
        # success/verdict is unaffected (scored from the rubric); final_message is
        # left untouched so scoring cannot change.
        self.tokens_in += int(getattr(event, "input_tokens", 0) or 0)
        self.tokens_out += int(getattr(event, "output_tokens", 0) or 0)
        self.iterations = max(self.iterations, int(event.iterations))
        if getattr(event, "tokens_estimated", False):
            self.tokens_estimated = True

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

    async def _h_turn_failed(ev: TurnFailed) -> None:
        acc.on_turn_failed(ev)

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
    bus.subscribe(TurnFailed, _h_turn_failed)
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
        base_registry = await _get_base_registry()
        loop = await _build_agent_loop(bus=bus, llm_client=llm_client, scenario=scen, session_id=session_id, agent_config=agent_config, base_registry=base_registry)
        try:
            await asyncio.wait_for(
                _run_loop(loop, _render_prompt(scen.prompt), _on_token),
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

_BASE_REGISTRY: Any = None


async def _get_base_registry() -> Any:
    """Cached base registry populated with the builtin tools (read/write/glob/grep/
    bash_exec + tool_result_get). from_allowed() returns an EMPTY registry when given no
    base, so the bench MUST supply one or the agent gets zero tools and hallucinates on
    every tool scenario. Built once (builtin tools are stateless) and reused across runs.

    tool_result_get is registered too: register_builtin_tools gates it on an eviction_store, and
    the cruncher's chunk-summarizer leaves read their granted chunk via tool_result_get (it is the
    sole entry in CHUNK_SUMMARIZER_TOOLS). from_allowed can only surface a tool that EXISTS in this
    base, so without it every leaf gets an empty toolset, cannot read its chunk, and emits the call
    as text — the cruncher then reduces empty extracts and silently misses every fact. The store
    passed here is a placeholder: bind_agent_store_tools rebinds tool_result_get to each agent's REAL
    ContentStore per delegation (the leaf to its granted-chunk store, the root to bench_store)."""
    global _BASE_REGISTRY
    if _BASE_REGISTRY is None:
        from localharness.agent.context import ContentStore
        from localharness.tools.builtin import register_builtin_tools
        from localharness.tools.registry import ToolRegistry

        reg = ToolRegistry()
        await register_builtin_tools(reg, eviction_store=ContentStore())
        _BASE_REGISTRY = reg
    return _BASE_REGISTRY


async def _build_agent_loop(bus: EventBus, llm_client: Any, scenario: ScenarioSpec, session_id: str = "", agent_config: Any = None, base_registry: Any = None) -> Any:
    """Construct an AgentLoop instance for the given scenario.

    Async because EVAL-01 hydration seeds a MemoryStore (async open/store/flush)
    for the stateful_behavior scenarios; awaited from execute_one_run, which is
    already async — so both the seed and the loop's later load_context() run on
    the same event loop (no cross-loop aiosqlite hazard).

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
    from localharness.agent.context import CompactionPipeline, ContentStore, ContextManager, TokenCounter
    from localharness.agent.permissions import PermissionEvaluator
    from localharness.config.models import AgentConfig
    from localharness.tools.registry import ToolRegistry

    if agent_config is None:
        from localharness.config.models import BudgetConfig, PermissionConfig
        agent_config = AgentConfig(
            # Scenario names use underscores (pure_qa, single_read…) but AgentConfig's
            # name validator only allows [a-z0-9-]. Sanitize so every scenario yields a
            # valid agent name instead of crashing at construction.
            name=f"bench-{scenario.name.replace('_', '-')}",
            role=f"Bench harness execution for scenario {scenario.name}",
            permissions=PermissionConfig(
                budget=BudgetConfig(
                    # max_tool_calls (limits) acts as a ceiling on dispatch; take the
                    # tighter of the two caps so both budget.max_actions and
                    # limits.max_tool_calls are enforced by the real loop.
                    max_actions=max(1, min(scenario.budget.max_actions, scenario.limits.max_tool_calls)),
                    max_duration_minutes=scenario.budget.max_duration_minutes,
                ),
            ),
        )

    # Wire the live harness's context machinery (CompactionPipeline + ContentStore) into the bench
    # runner. Previously omitted (pipeline=None, no store), so NO compaction/eviction ran and
    # max_context_tokens was never enforced — over-window content sailed straight through to the
    # model's real window (the "no CompactionPipeline in runner" bench bug; J3 was unmeasurable).
    # This mirrors cli/start_cmd.py:391-409 so the bench enforces the window the way production does.
    cctx = agent_config.context
    # Count with the SERVED model's exact tokenizer (vLLM /tokenize) so the window gates fire on
    # real token counts — tiktoken undercounts Qwen and let over-window content slip past. Falls
    # back to the approximate tokenizer only when there is no reachable /tokenize (test mocks).
    _cfg = getattr(llm_client, "config", None)
    _tc_base = getattr(_cfg, "base_url", None)
    _tc_model = getattr(_cfg, "model", None)
    # When an endpoint is configured, use the exact server tokenizer and DO NOT swallow a /tokenize
    # failure into a silent tiktoken fallback — that would fire the window gate on wrong counts while
    # claiming "real counts". TokenCounter is server-or-fail by design; fall back to the approximate
    # tokenizer ONLY when there is no endpoint at all (test mocks with no base_url/model).
    if _tc_base and _tc_model:
        token_counter = TokenCounter(base_url=_tc_base, model=_tc_model)
    else:
        token_counter = TokenCounter()
    bench_store = ContentStore()

    from localharness.agent.context import make_compaction_summarize_fn
    bench_pipeline = CompactionPipeline(
        token_counter=token_counter,
        tool_result_cap=cctx.max_tool_output_chars,
        preserve_first_n=cctx.preserve_first_n_messages,
        preserve_last_n=cctx.preserve_last_n_messages,
        llm_summarize_fn=make_compaction_summarize_fn(llm_client),  # shared, tuple-unpack tested
        compact_md_path=None,
    )
    # eviction_store only when the scenario can RESTORE an evicted stub (has tool_result_get) — a
    # context that cannot re-pull must not stub bodies it could never recover (context.py:687-690).
    can_restore = "tool_result_get" in (scenario.tools_allowed or [])
    ctx_manager = ContextManager(
        max_context_tokens=scenario.budget.max_context_tokens,
        preserve_first_n=cctx.preserve_first_n_messages,
        preserve_last_n=cctx.preserve_last_n_messages,
        pipeline=bench_pipeline,
        bus=bus,
        agent_id=session_id,
        session_id=session_id,
        content_store=bench_store,
        eviction_store=bench_store if can_restore else None,
        token_counter=token_counter,
    )
    tool_registry = ToolRegistry.from_allowed(scenario.tools_allowed, base_registry=base_registry if base_registry is not None else ToolRegistry())

    # J3: bind the store-backed verbs (load_document / chunk / tool_result_get / web verbs) onto this
    # per-scenario registry so they hit bench_store — the SAME store the cruncher reads THROUGH on a
    # grant. Without this, load_document would mint its handle in its own private default ContentStore
    # and a cross-agent grant_handles=[H] would resolve to nothing (the cruncher path is then silently
    # unreachable from the bench). rebind_global only touches tools already present, never adds a held-
    # back capability. Mirrors cli/start_cmd.py:317-329.
    from localharness.tools.builtin import bind_agent_store_tools
    bind_agent_store_tools(tool_registry, bench_store)

    # PermissionEvaluator is stateless; constructor takes no args. (Plan 12-04
    # Rule 3 fix — the prior `from_config(agent_config)` referenced a method
    # that does not exist on PermissionEvaluator and would have crashed at
    # runtime. The evaluator reads deny_patterns from the config object passed
    # to `evaluate(...)` at call time, not at construction.)
    # Defined before the `agent` block below so the real subagent runner can capture it.
    perm_evaluator = PermissionEvaluator()

    # SUBAGENT-05 / J3: register the REAL delegation runner when 'agent' is in tools_allowed, built via
    # the SAME module-level seam the live start path uses (make_explore_agent_runner) so the bench
    # exercises PRODUCTION routing — crucially including the cruncher. The factory routes by agent_id:
    # 'cruncher' -> dispatch_cruncher_subagent (faithful over-window reduce), 'explore' -> Explore, etc.
    # The child runs a real AgentLoop on the SAME bus, so its Actions are counted (real delegation =>
    # tool_call_count >= 2). parent_store=bench_store is the grant keystone: when the model delegates
    # with grant_handles=[H], the child reads ONLY H through bench_store (where load_document retained
    # it). PRIOR BUG: the bench hard-routed EVERY agent() to Explore (agent_id ignored), so a scored
    # agent('cruncher') scenario silently ran Explore and the cruncher was UNREACHABLE from the bench —
    # every "faithful over-window" claim rested on mocked unit tests with no scored regression signal.
    if "agent" in scenario.tools_allowed:
        from localharness.agent.subagent import make_explore_agent_runner
        from localharness.tools.builtin.agent_tool import AgentTool

        _bench_agents = ["explore", "web-researcher", "data-analyst",
                         "frontend-designer", "cruncher", "search-verifier"]
        _agent_runner = make_explore_agent_runner(
            llm=llm_client,
            bus=bus,
            base_registry=base_registry,
            permission_evaluator=perm_evaluator,
            # session_id is fixed for a bench run (no per-turn re-keying), so a constant getter matches
            # the live path's late-read lambda without a not-yet-built loop reference.
            get_parent_session_id=lambda: session_id,
            # Children inherit the bench's exact /tokenize counter + the scenario's declared window so
            # the cruncher's chunk sizing (_cruncher_chunk_chars) tracks the scenario, not bare defaults.
            token_counter=token_counter,
            max_context_tokens=scenario.budget.max_context_tokens,
            depth=0,
            available_agents=_bench_agents,
            parent_store=bench_store,
            # cruncher exec policy (default off): the faithful reduce never needs host exec.
            cruncher_config=agent_config.cruncher,
        )
        tool_registry._tools["global"]["agent"] = AgentTool(
            agent_runner=_agent_runner, available_agents=_bench_agents)
        tool_registry._schemas["agent"] = tool_registry._tools["global"]["agent"].info()

    # EVAL-01: hydrate a seeded MemoryStore ONLY for the stateful_behavior
    # scenarios, keyed by scenario.name, under the loop's own agent_id
    # (agent_config.name) so load_context() resolves the seeded facts. Stays
    # None for every other scenario — no /agents dir is created for them.
    memory_loader = None
    seeds = _MEMORY_SEEDS.get(scenario.name)
    if seeds is not None:
        memory_loader = await _seed_memory_store(agent_config.name, seeds)

    try:
        return AgentLoop(
            config=agent_config,
            llm=llm_client,
            bus=bus,
            context_manager=ctx_manager,
            tool_registry=tool_registry,
            permission_evaluator=perm_evaluator,
            memory_loader=memory_loader,
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


def _render_prompt(prompt: str) -> str:
    """Substitute scenario-prompt placeholders. `{FIXTURES}` -> absolute path of bench/fixtures, so a
    scenario can hand load_document a real absolute path to a committed over-window fixture (the tool
    requires an absolute path). bench/ is resolved relative to the repo-root cwd — the same assumption
    the categories loader already makes (schema.py). No-op when the token is absent."""
    if "{FIXTURES}" in prompt:
        import os
        return prompt.replace("{FIXTURES}", os.path.abspath("bench/fixtures"))
    return prompt


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
