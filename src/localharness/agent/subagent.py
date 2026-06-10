"""Explore subagent dispatch — the harness's first real read-only subagent (SUBAGENT-01..04).

A parent agent spawns a bounded, read-only "explore" child that runs its own AgentLoop on
the SAME EventBus and returns a structured findings summary (not the raw transcript).

Design:
- Child registry = builtins {read, glob, grep} ONLY via ToolRegistry.from_allowed — it
  literally cannot write, execute, or spawn (no agent tool present): primary depth-1 guard.
- Belt-and-suspenders depth guard: `depth >= MAX_DEPTH` refuses with a clear error.
- Child events publish on the shared bus through `_ParentIdBus`, which stamps `parent_id`
  = parent session_id on every child event so an unfiltered subscriber can attribute the
  child's tool calls (mirrors bench MetricAccumulator.on_action — see MCP-SCENARIO-GAP §2).
- Child budget is DISTINCT from the parent (BUDGET-POLICY.md: max_tool_calls = max_actions+1).
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from localharness.config.models import AgentConfig, BudgetConfig, PermissionConfig

# Read-only toolset for the explore child (bare builtin names; see tools/builtin/__init__.py).
EXPLORE_TOOLS: list[str] = ["read", "glob", "grep"]

# Child budget, distinct from the parent (BUDGET-POLICY.md invariant: max_tool_calls = max_actions + 1).
EXPLORE_MAX_ACTIONS = 8
EXPLORE_MAX_TOOL_CALLS = EXPLORE_MAX_ACTIONS + 1  # 9 — kept above the budget so max_actions binds
EXPLORE_MAX_DURATION_MINUTES = 3.0

# Recursion limit: the parent runs at depth 0, the explore child at depth 1.
MAX_DEPTH = 1

EXPLORE_ROLE = (
    "You are a read-only Explore subagent. Use read, glob, and grep to investigate the "
    "codebase and answer the delegated question. You cannot write, execute, or delegate. "
    "When you have what you need, stop and reply with a concise summary of your findings."
)

# Web-research child: searches + reads pages in ITS OWN context, returns only a distilled
# summary so raw pages never enter the parent's context window.
WEB_TOOLS: list[str] = ["web_search", "web_fetch"]
WEB_MAX_ACTIONS = 12
WEB_MAX_TOOL_CALLS = WEB_MAX_ACTIONS + 1  # 13 — kept above the budget so max_actions binds
WEB_MAX_DURATION_MINUTES = 5.0

WEB_RESEARCHER_ROLE = (
    "You are a web-research subagent. Use web_search to find sources and web_fetch to read them. "
    "Research the delegated question thoroughly but efficiently — prefer a few high-quality sources "
    "over many, and don't re-fetch the same page. You cannot write, execute, or delegate. When done, "
    "stop and reply with a CONCISE, well-organized summary of your findings WITH the source URLs you "
    "used. This summary is the ONLY thing the parent agent sees, so include the key facts, figures, "
    "and citations it needs — never paste raw page text."
)

# Data-analyst child: local files + computation, no web, no write. Heavier budget than web —
# real analysis iterates (inspect format -> compute -> sanity-check); observed ~17 calls on a
# vault-sized task. Raw file dumps and intermediate outputs stay in the child's context.
DATA_TOOLS: list[str] = ["bash_exec", "read", "glob", "grep"]
DATA_MAX_ACTIONS = 16
DATA_MAX_TOOL_CALLS = DATA_MAX_ACTIONS + 1  # 17 — kept above the budget so max_actions binds
DATA_MAX_DURATION_MINUTES = 12.0

DATA_ANALYST_ROLE = (
    "You are a data-analyst subagent. Investigate LOCAL files and compute precise answers using "
    "bash_exec (python3/pandas where helpful), read, glob, and grep. If your brief names a data "
    "contract, README, or index doc, READ IT FIRST and honor what it says is the source of truth — "
    "do not re-derive facts the contract already settles. Sanity-check your numbers (signs, totals, "
    "currencies) before reporting. You cannot write files or delegate. When done, stop and reply "
    "with a CONCISE summary: the numbers, how you computed them, and the file paths used. This "
    "summary is the ONLY thing the parent agent sees — never paste raw file dumps."
)


def _sanitize_agent_name(name: str) -> str:
    """AgentConfig.name rejects underscores — map `_` -> `-` (recurring localharness gotcha)."""
    return name.replace("_", "-")


def build_explore_config(name: str = "explore", kill_file: str | None = None) -> AgentConfig:
    """Build the read-only explore-child AgentConfig with its own bounded budget.

    `kill_file=None` disables the kill switch for the child (its own short budget bounds it);
    pass a path to honor an external kill file.
    """
    return AgentConfig(
        name=_sanitize_agent_name(name),
        role=EXPLORE_ROLE,
        permissions=PermissionConfig(
            budget=BudgetConfig(
                max_actions=EXPLORE_MAX_ACTIONS,
                max_duration_minutes=EXPLORE_MAX_DURATION_MINUTES,
                kill_file=kill_file,
            ),
        ),
    )


def build_web_researcher_config(name: str = "web-researcher", kill_file: str | None = None) -> AgentConfig:
    """Build the web-researcher-child AgentConfig with its own bounded budget.

    `kill_file=None` disables the kill switch for the child (its own short budget bounds it);
    pass a path to honor an external kill file.
    """
    return AgentConfig(
        name=_sanitize_agent_name(name),
        role=WEB_RESEARCHER_ROLE,
        permissions=PermissionConfig(
            budget=BudgetConfig(
                max_actions=WEB_MAX_ACTIONS,
                max_duration_minutes=WEB_MAX_DURATION_MINUTES,
                kill_file=kill_file,
            ),
        ),
    )


class _ParentIdBus:
    """Thin pass-through wrapper over an EventBus that stamps `parent_id` on published events.

    The child AgentLoop is handed this wrapper instead of the raw bus. Every event the child
    publishes is forwarded to the SAME underlying bus (subscribers, JSONL persistence, and
    history are all the real bus's) but with `parent_id` set to the parent session_id when the
    event does not already carry one. Events are frozen Pydantic models, so we use model_copy.

    Only `publish` needs interception; any other attribute access proxies to the real bus so the
    child loop can use the bus exactly as it would the original.
    """

    def __init__(self, inner: Any, parent_id: str) -> None:
        self._inner = inner
        self._parent_id = parent_id

    async def publish(self, event: Any) -> Any:
        if getattr(event, "parent_id", None) is None:
            try:
                event = event.model_copy(update={"parent_id": self._parent_id})
            except Exception:
                pass  # non-model event — forward unchanged
        return await self._inner.publish(event)

    def __getattr__(self, item: str) -> Any:
        # Proxy everything else (subscribe, history, replay, ...) to the real bus.
        return getattr(self._inner, item)


def format_findings(task: str, summary: str, tool_calls_used: int) -> str:
    """Structured findings return (SUBAGENT-04): short header + child summary, NOT the transcript."""
    header = f"[explore findings] task: {task} | tool calls: {tool_calls_used}"
    body = (summary or "").strip() or "(no findings)"
    return f"{header}\n\n{body}"


def format_web_findings(task: str, summary: str, tool_calls_used: int) -> str:
    """Structured web-research findings return: short header + child summary, NOT the transcript."""
    header = f"[web research] task: {task} | tool calls: {tool_calls_used}"
    body = (summary or "").strip() or "(no findings)"
    return f"{header}\n\n{body}"


def build_data_analyst_config(name: str = "data-analyst", kill_file: str | None = None) -> AgentConfig:
    """Build the data-analyst-child AgentConfig with its own bounded budget."""
    return AgentConfig(
        name=_sanitize_agent_name(name),
        role=DATA_ANALYST_ROLE,
        permissions=PermissionConfig(
            budget=BudgetConfig(
                max_actions=DATA_MAX_ACTIONS,
                max_duration_minutes=DATA_MAX_DURATION_MINUTES,
                kill_file=kill_file,
            ),
        ),
    )


def format_data_findings(task: str, summary: str, tool_calls_used: int) -> str:
    """Structured data-analysis findings return: short header + child summary, NOT the transcript."""
    header = f"[data analysis] task: {task} | tool calls: {tool_calls_used}"
    body = (summary or "").strip() or "(no findings)"
    return f"{header}\n\n{body}"


async def dispatch_data_subagent(
    task: str,
    *,
    llm: Any,
    bus: Any,
    base_registry: Any,
    parent_session_id: str | None,
    permission_evaluator: Any,
    context_manager: Any = None,
    agent_name: str = "data-analyst",
    depth: int = 0,
) -> str:
    """Spawn a data-analysis child (bash/read/glob/grep, no web/write), run one turn, return findings.

    Mirrors dispatch_web_subagent: the child inspects files and computes in ITS OWN context and
    returns only a summary, so raw file dumps and intermediate outputs never reach the parent's
    window. Same depth>=MAX_DEPTH guard (a child cannot delegate further).
    """
    if depth >= MAX_DEPTH:
        raise ValueError(
            f"data-analyst subagent cannot spawn at depth {depth} (max depth {MAX_DEPTH}): "
            "subagents may not delegate further."
        )

    from localharness.agent.context import ContextManager
    from localharness.agent.loop import AgentLoop
    from localharness.tools.registry import ToolRegistry

    child_config = build_data_analyst_config(agent_name)
    child_registry = ToolRegistry.from_allowed(DATA_TOOLS, base_registry=base_registry)
    child_bus = _ParentIdBus(bus, parent_session_id) if parent_session_id is not None else bus

    child_loop = AgentLoop(
        config=child_config,
        llm=llm,
        bus=child_bus,
        context_manager=context_manager or ContextManager(),
        tool_registry=child_registry,
        permission_evaluator=permission_evaluator,
    )

    summary = await child_loop.run_turn(task)

    child_session_id = child_loop.current_session_id
    tool_calls_used = _count_session_tool_calls(bus, child_session_id)

    return format_data_findings(task, summary, tool_calls_used)


async def dispatch_explore_subagent(
    task: str,
    *,
    llm: Any,
    bus: Any,
    base_registry: Any,
    parent_session_id: str | None,
    permission_evaluator: Any,
    context_manager: Any = None,
    agent_name: str = "explore",
    depth: int = 0,
) -> str:
    """Spawn a read-only explore child, run one turn on `task`, return structured findings.

    Args:
        task: the focused "go find X" request for the child.
        llm: the LLMClient (shared with the parent).
        bus: the parent's EventBus — the child publishes on this SAME bus (via _ParentIdBus).
        base_registry: a ToolRegistry holding the builtins to draw the read-only subset from.
        parent_session_id: parent run's session_id; stamped as parent_id on child events.
        permission_evaluator: PermissionEvaluator (shared).
        context_manager: ContextManager for the child; a default is built if None.
        agent_name: child agent name (sanitized; `_` -> `-`).
        depth: caller depth. The parent is depth 0; a child runs at depth 1. depth >= MAX_DEPTH
            refuses (belt-and-suspenders recursion guard; the primary guard is the toolset).

    Returns:
        A concise findings string (header + child summary).

    Raises:
        ValueError: if invoked at depth >= MAX_DEPTH (a child trying to spawn a grandchild).
    """
    if depth >= MAX_DEPTH:
        raise ValueError(
            f"explore subagent cannot spawn at depth {depth} (max depth {MAX_DEPTH}): "
            "read-only subagents may not delegate further."
        )

    from localharness.agent.context import ContextManager
    from localharness.agent.loop import AgentLoop
    from localharness.tools.registry import ToolRegistry

    child_config = build_explore_config(agent_name)

    # Read-only registry: builtins {read, glob, grep} only — no write/bash/spawn.
    child_registry = ToolRegistry.from_allowed(EXPLORE_TOOLS, base_registry=base_registry)

    # Stamp parent_id on every child event while publishing on the shared bus.
    child_bus = _ParentIdBus(bus, parent_session_id) if parent_session_id is not None else bus

    child_loop = AgentLoop(
        config=child_config,
        llm=llm,
        bus=child_bus,
        context_manager=context_manager or ContextManager(),
        tool_registry=child_registry,
        permission_evaluator=permission_evaluator,
    )

    summary = await child_loop.run_turn(task)

    # Count the child's tool-call Actions the way the bench accumulator does (Action with a
    # tool_name), filtered to this child's session so the header reflects the child only.
    child_session_id = child_loop.current_session_id
    tool_calls_used = _count_session_tool_calls(bus, child_session_id)

    return format_findings(task, summary, tool_calls_used)


async def dispatch_web_subagent(
    task: str,
    *,
    llm: Any,
    bus: Any,
    base_registry: Any,
    parent_session_id: str | None,
    permission_evaluator: Any,
    context_manager: Any = None,
    agent_name: str = "web-researcher",
    depth: int = 0,
) -> str:
    """Spawn a web-research child (web_search/web_fetch only), run one turn, return distilled findings.

    Mirrors dispatch_explore_subagent but with the web toolset and budget. The child does all the
    searching/fetching in ITS OWN context and returns only a summary, so raw pages never reach the
    parent's window. Same depth>=MAX_DEPTH guard (a child cannot delegate further).
    """
    if depth >= MAX_DEPTH:
        raise ValueError(
            f"web-researcher subagent cannot spawn at depth {depth} (max depth {MAX_DEPTH}): "
            "subagents may not delegate further."
        )

    from localharness.agent.context import ContextManager
    from localharness.agent.loop import AgentLoop
    from localharness.tools.registry import ToolRegistry

    child_config = build_web_researcher_config(agent_name)

    # Web-only registry: builtins {web_search, web_fetch} — no write/bash/spawn.
    child_registry = ToolRegistry.from_allowed(WEB_TOOLS, base_registry=base_registry)

    child_bus = _ParentIdBus(bus, parent_session_id) if parent_session_id is not None else bus

    child_loop = AgentLoop(
        config=child_config,
        llm=llm,
        bus=child_bus,
        context_manager=context_manager or ContextManager(),
        tool_registry=child_registry,
        permission_evaluator=permission_evaluator,
    )

    summary = await child_loop.run_turn(task)

    child_session_id = child_loop.current_session_id
    tool_calls_used = _count_session_tool_calls(bus, child_session_id)

    return format_web_findings(task, summary, tool_calls_used)


def _count_session_tool_calls(bus: Any, session_id: str | None) -> int:
    """Count tool-call Actions for `session_id` from the bus's in-memory history.

    Mirrors bench MetricAccumulator.on_action: an Action with a non-empty tool_name is one
    tool call (MCP-SCENARIO-GAP §2). Returns 0 if history is unavailable.
    """
    from localharness.core.events import Action

    history_fn = getattr(bus, "history", None)
    if history_fn is None:
        return 0
    try:
        actions = history_fn(session_id=session_id, event_types=[Action])
    except Exception:
        return 0
    return sum(1 for e in actions if getattr(e, "tool_name", None))


def make_explore_agent_runner(
    *,
    llm: Any,
    bus: Any,
    base_registry: Any,
    permission_evaluator: Any,
    get_parent_session_id: Callable[[], str | None],
) -> Callable[[str, str, int], Awaitable[str]]:
    """Build the AgentTool runner for the read-only Explore subagent (module-level seam, T1).

    Mirrors the old `start_cmd._run_agent` closure exactly, but as a unit-testable factory
    (the same way Phase 27 extracted `dispatch_explore_subagent` from this closure). The returned
    async runner is what `AgentTool(agent_runner=...)` invokes. Behavior is identical to the closure:

    - Wires `explore` and `web-researcher` (after `_` -> `-` sanitization); any other agent_id raises
      a clear ValueError so the model gets an actionable error (AgentTool maps it to not_found).
    - `get_parent_session_id` is read AT CALL TIME (not captured at build time), so it reflects the
      parent loop's current session_id when the model delegates — matching the closure's late read of
      `agent_loop.current_session_id`.
    - `depth` threads through to `dispatch_explore_subagent` (the belt-and-suspenders recursion guard).
    """

    async def _run_agent(agent_id: str, task: str, depth: int = 0) -> str:
        name = _sanitize_agent_name(agent_id)
        if name == "explore":
            dispatch = dispatch_explore_subagent
        elif name == "web-researcher":
            dispatch = dispatch_web_subagent
        elif name == "data-analyst":
            dispatch = dispatch_data_subagent
        else:
            raise ValueError(
                f"Agent '{agent_id}' dispatch not wired "
                "(available: explore, web-researcher, data-analyst)"
            )
        return await dispatch(
            task,
            llm=llm,
            bus=bus,
            base_registry=base_registry,
            parent_session_id=get_parent_session_id(),
            permission_evaluator=permission_evaluator,
            depth=depth,
        )

    return _run_agent
