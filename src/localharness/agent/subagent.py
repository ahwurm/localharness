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

# Recursion limit fallback when no AgentConfig.max_subagent_depth is threaded (the parent runs
# at depth 0, a subagent at depth 1, ...). The real cap is config-driven (AgentConfig.max_subagent_depth)
# and passed into the runner/dispatches; MAX_DEPTH is the conservative default for bare callers.
MAX_DEPTH = 1

# Builtin children permitted to delegate, mapped to the agents they may spawn. A child of name N
# is given its own (fresh, depth+1) `agent` tool ONLY if N is listed here AND there is room under
# the cap — so it can nest a grandchild (e.g. web-researcher -> search-verifier, wired in P3).
# Every other builtin/config child is a leaf: it gets no `agent` tool and its role says it cannot
# delegate. Keep this in sync with the roles — a name here whose role still says "cannot delegate"
# would be a contradiction. (P3 adds the "web-researcher": ["search-verifier"] entry.)
NON_LEAF_AGENTS: dict[str, list[str]] = {}

EXPLORE_ROLE = (
    "You are a read-only Explore subagent. Use read, glob, and grep to investigate the "
    "codebase and answer the delegated question. You cannot write, execute, or delegate. "
    "When you have what you need, stop and reply with a concise summary of your findings."
)

# Web-research child: searches + reads pages in ITS OWN context, returns only a distilled
# summary so raw pages never enter the parent's context window. web_page_query (P4) lets it
# ground a claim in the FULL retained page, not just the clipped inline preview.
WEB_TOOLS: list[str] = ["web_search", "web_fetch", "web_page_query"]
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

# Frontend-designer child: builds self-contained HTML/CSS and verifies it VISUALLY via the
# packaged playwright screenshot helper before delivering. Budgets sized from live usage
# (landing-page + architecture-diagram builds ran 25-35 actions incl. screenshot iterations).
FRONTEND_TOOLS: list[str] = ["read", "write", "edit", "glob", "grep", "bash_exec"]
FRONTEND_MAX_ACTIONS = 40
FRONTEND_MAX_TOOL_CALLS = FRONTEND_MAX_ACTIONS + 1  # 41 — kept above the budget so max_actions binds
FRONTEND_MAX_DURATION_MINUTES = 20.0

FRONTEND_DESIGNER_ROLE = (
    "You are a frontend-designer subagent. Build beautiful, self-contained HTML/CSS/JS and "
    "verify it VISUALLY before delivering. Avoid generic AI-slop aesthetics: no default Inter, "
    "no purple-gradient templates — pick distinctive typography, commit to a cohesive palette "
    "via CSS custom properties, and prefer one well-orchestrated CSS animation over scattered "
    "micro-interactions. Workflow: PLAN the layout, palette, and type; BUILD a single "
    "self-contained responsive file under /tmp/designs/ (semantic HTML, no external "
    "dependencies unless asked); SCREENSHOT it with "
    "`node ~/.localharness/tools/design-screenshot.js <html-file> <output-prefix>` "
    "(captures 1440x900 and 375x812 by default and waits ~2s for fonts and CSS animations to "
    "settle — pass --wait <ms> for longer entrances); REVIEW the screenshots for layout, "
    "contrast, overflow, and responsive issues; ITERATE until polished. You cannot delegate. "
    "When done, stop and reply with the HTML file path, every screenshot path, and a brief "
    "note on the design decisions — this summary is the ONLY thing the parent agent sees."
)


def build_frontend_designer_config(name: str = "frontend-designer", kill_file: str | None = None) -> AgentConfig:
    """Build the frontend-designer-child AgentConfig with its own bounded budget."""
    return AgentConfig(
        name=_sanitize_agent_name(name),
        role=FRONTEND_DESIGNER_ROLE,
        permissions=PermissionConfig(
            budget=BudgetConfig(
                max_actions=FRONTEND_MAX_ACTIONS,
                max_duration_minutes=FRONTEND_MAX_DURATION_MINUTES,
                kill_file=kill_file,
            ),
        ),
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


# YAML-defined children with no tools.add get a safe read-only set.
CONFIG_CHILD_DEFAULT_TOOLS: list[str] = ["read", "glob", "grep"]


def prepend_toolset(task: str, allowed: list[str]) -> str:
    """Prefix a config-child brief with its exact toolset (capability fact, not inference)."""
    names = ", ".join(allowed) if allowed else "none"
    return (
        f"(Your ONLY available tools: {names}. You cannot delegate or use anything else — "
        f"if the task needs a tool you lack, say so immediately instead of improvising.)\n\n{task}"
    )


def format_child_findings(agent_name: str, task: str, summary: str, tool_calls_used: int) -> str:
    """Generic findings return for config-defined children: header + summary, NOT the transcript."""
    header = f"[{agent_name}] task: {task} | tool calls: {tool_calls_used}"
    body = (summary or "").strip() or "(no findings)"
    return f"{header}\n\n{body}"


async def dispatch_config_subagent(
    task: str,
    *,
    agent_config: Any,
    llm: Any,
    bus: Any,
    base_registry: Any,
    parent_session_id: str | None,
    permission_evaluator: Any,
    context_manager: Any = None,
    depth: int = 0,
    max_subagent_depth: int = MAX_DEPTH,
) -> str:
    """Spawn a child from a YAML-defined AgentConfig, run one turn, return distilled findings.

    This is the 'ability scales with defined subagents' seam: any agents/<name>.yaml becomes a
    dispatchable specialist — role + tools.add (its allowlist; bare, mcp:TOOL and plugin:P.TOOL
    forms all resolve) + its own budget — without touching harness code. The parent (or the
    model itself, by WRITING the yaml first) composes new specialists at runtime. The child runs
    in its own context and returns only a summary; the `agent` tool is always stripped so
    config children can never delegate further (belt to the MAX_DEPTH suspenders).
    """
    if depth >= max_subagent_depth:
        raise ValueError(
            f"subagent '{getattr(agent_config, 'name', '?')}' cannot spawn at depth {depth} "
            f"(max depth {max_subagent_depth}): subagents may not delegate further."
        )

    from localharness.agent.context import ContextManager
    from localharness.agent.loop import AgentLoop
    from localharness.tools.registry import ToolRegistry

    tool_cfg = getattr(agent_config, "tools", None)
    add = list(getattr(tool_cfg, "add", None) or []) or list(CONFIG_CHILD_DEFAULT_TOOLS)
    deny = set(getattr(tool_cfg, "deny", None) or [])
    allowed = [t for t in add if t not in deny and t.split(".")[-1].split(":")[-1] != "agent"]
    child_registry = ToolRegistry.from_allowed(allowed, base_registry=base_registry)
    # Make the toolset explicit in the brief — small models attend weakly to absent
    # tools (observed live: a bash-less child globbed for `pip` instead of saying
    # "I have no shell"). One line turns capability inference into capability fact.
    task = prepend_toolset(task, allowed)

    child_bus = _ParentIdBus(bus, parent_session_id) if parent_session_id is not None else bus
    child_loop = AgentLoop(
        config=agent_config,
        llm=llm,
        bus=child_bus,
        context_manager=context_manager or ContextManager(),
        tool_registry=child_registry,
        permission_evaluator=permission_evaluator,
    )

    summary = await child_loop.run_turn(task)
    tool_calls_used = _count_session_tool_calls(bus, child_loop.current_session_id)
    return format_child_findings(agent_config.name, task, summary, tool_calls_used)


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
    max_subagent_depth: int = MAX_DEPTH,
    child_agent_tool: Any = None,
) -> str:
    """Spawn a data-analysis child (bash/read/glob/grep, no web/write), run one turn, return findings.

    Mirrors dispatch_web_subagent: the child inspects files and computes in ITS OWN context and
    returns only a summary, so raw file dumps and intermediate outputs never reach the parent's
    window. Same depth>=cap guard (a child cannot delegate unless handed a child_agent_tool).
    """
    if depth >= max_subagent_depth:
        raise ValueError(
            f"data-analyst subagent cannot spawn at depth {depth} (max depth {max_subagent_depth}): "
            "subagents may not delegate further."
        )

    from localharness.agent.context import ContextManager
    from localharness.agent.loop import AgentLoop
    from localharness.tools.registry import ToolRegistry

    child_config = build_data_analyst_config(agent_name)
    child_registry = ToolRegistry.from_allowed(DATA_TOOLS, base_registry=base_registry)
    if child_agent_tool is not None:
        await child_registry.register(child_agent_tool, scope="global")
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


async def dispatch_frontend_subagent(
    task: str,
    *,
    llm: Any,
    bus: Any,
    base_registry: Any,
    parent_session_id: str | None,
    permission_evaluator: Any,
    context_manager: Any = None,
    agent_name: str = "frontend-designer",
    depth: int = 0,
    max_subagent_depth: int = MAX_DEPTH,
    child_agent_tool: Any = None,
) -> str:
    """Spawn a frontend-designer child (read/write/edit/glob/grep/bash), run one turn, return findings.

    The child builds HTML under /tmp/designs/ and verifies it visually via the packaged
    design-screenshot.js helper (installed to <config-dir>/tools by start_cmd) in ITS OWN
    context — only the summary (file + screenshot paths, design notes) reaches the parent.
    """
    if depth >= max_subagent_depth:
        raise ValueError(
            f"frontend-designer subagent cannot spawn at depth {depth} (max depth {max_subagent_depth}): "
            "subagents may not delegate further."
        )

    from localharness.agent.context import ContextManager
    from localharness.agent.loop import AgentLoop
    from localharness.tools.registry import ToolRegistry

    child_config = build_frontend_designer_config(agent_name)
    child_registry = ToolRegistry.from_allowed(FRONTEND_TOOLS, base_registry=base_registry)
    if child_agent_tool is not None:
        await child_registry.register(child_agent_tool, scope="global")
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
    tool_calls_used = _count_session_tool_calls(bus, child_loop.current_session_id)
    return format_child_findings(agent_name, task, summary, tool_calls_used)


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
    max_subagent_depth: int = MAX_DEPTH,
    child_agent_tool: Any = None,
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
    if depth >= max_subagent_depth:
        raise ValueError(
            f"explore subagent cannot spawn at depth {depth} (max depth {max_subagent_depth}): "
            "read-only subagents may not delegate further."
        )

    from localharness.agent.context import ContextManager
    from localharness.agent.loop import AgentLoop
    from localharness.tools.registry import ToolRegistry

    child_config = build_explore_config(agent_name)

    # Read-only registry: builtins {read, glob, grep} only — no write/bash/spawn.
    child_registry = ToolRegistry.from_allowed(EXPLORE_TOOLS, base_registry=base_registry)
    if child_agent_tool is not None:
        await child_registry.register(child_agent_tool, scope="global")

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
    max_subagent_depth: int = MAX_DEPTH,
    child_agent_tool: Any = None,
) -> str:
    """Spawn a web-research child (web_search/web_fetch[/web_page_query] only), run one turn, return findings.

    Mirrors dispatch_explore_subagent but with the web toolset and budget. The child does all the
    searching/fetching in ITS OWN context and returns only a summary, so raw pages never reach the
    parent's window. When handed a `child_agent_tool` (room left under the depth cap), the child
    becomes NON-LEAF and may dispatch a nested search-verifier; otherwise it is a leaf. The
    depth>=cap guard is the belt-and-suspenders backstop.
    """
    if depth >= max_subagent_depth:
        raise ValueError(
            f"web-researcher subagent cannot spawn at depth {depth} (max depth {max_subagent_depth}): "
            "subagents may not delegate further."
        )

    from localharness.agent.context import ContextManager
    from localharness.agent.loop import AgentLoop
    from localharness.tools.registry import ToolRegistry

    child_config = build_web_researcher_config(agent_name)

    # Web-only registry: builtins {web_search, web_fetch, web_page_query} — no write/bash.
    child_registry = ToolRegistry.from_allowed(WEB_TOOLS, base_registry=base_registry)
    # Non-leaf web-researcher: inject its own (fresh, depth+1) `agent` tool so it can spawn a
    # nested search-verifier. Registered global so the running child resolves it (same invariant
    # as start_cmd's global agent-tool registration).
    if child_agent_tool is not None:
        await child_registry.register(child_agent_tool, scope="global")

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
    load_agent: Callable[[str], Any] | None = None,
    token_counter: Any = None,
    max_context_tokens: int | None = None,
    depth: int = 0,
    max_subagent_depth: int = MAX_DEPTH,
    available_agents: list[str] | None = None,
) -> Callable[[str, str], Awaitable[str]]:
    """Build the AgentTool runner for delegation (module-level seam, T1).

    Mirrors the old `start_cmd._run_agent` closure exactly, but as a unit-testable factory
    (the same way Phase 27 extracted `dispatch_explore_subagent` from this closure). The returned
    async runner is what `AgentTool(agent_runner=...)` invokes.

    - Wires the built-ins (`explore`, `web-researcher`, `data-analyst`, after `_` -> `-`
      sanitization). Any OTHER agent_id is resolved through `load_agent` (when provided) — the
      ConfigLoader seam that turns agents/<name>.yaml into a dispatchable specialist. Pass a
      cache-bypassing loader so a yaml the model JUST WROTE is dispatchable in the same turn.
      Unresolvable names raise a clear ValueError so the model gets an actionable error.
    - `get_parent_session_id` is read AT CALL TIME (not captured at build time), so it reflects the
      parent loop's current session_id when the model delegates — matching the closure's late read of
      `agent_loop.current_session_id`.
    - This runner serves an agent at `depth`; a child it spawns runs at `depth+1`. A child listed
      in NON_LEAF_AGENTS is handed its OWN fresh runner+AgentTool closed over `depth+1` (injected
      into its registry) so it can nest a grandchild — this is the only way depth increments, since
      AgentTool calls the runner 2-arg. Nesting stops at `max_subagent_depth` (cap); `=1` disables it.
    """

    def _make_child_ctx() -> Any:
        """Fresh ContextManager per child, carrying the parent's model-aware token_counter
        (exact /tokenize counts, not tiktoken) and resolved window — so children account
        for context the same way the parent does instead of falling to bare defaults."""
        from localharness.agent.context import ContextManager
        kwargs: dict[str, Any] = {}
        if token_counter is not None:
            kwargs["token_counter"] = token_counter
        if max_context_tokens is not None:
            kwargs["max_context_tokens"] = max_context_tokens
        return ContextManager(**kwargs)

    def _build_child_agent_tool(name: str) -> Any:
        """A fresh, depth+1 `agent` tool for a non-leaf child — or None (leaf / no room under cap)."""
        child_depth = depth + 1
        delegatees = NON_LEAF_AGENTS.get(name)
        if not delegatees or child_depth >= max_subagent_depth:
            return None
        from localharness.tools.builtin.agent_tool import AgentTool
        child_runner = make_explore_agent_runner(
            llm=llm,
            bus=bus,
            base_registry=base_registry,
            permission_evaluator=permission_evaluator,
            get_parent_session_id=get_parent_session_id,
            load_agent=load_agent,
            token_counter=token_counter,
            max_context_tokens=max_context_tokens,
            depth=child_depth,
            max_subagent_depth=max_subagent_depth,
            available_agents=available_agents,
        )
        return AgentTool(agent_runner=child_runner, available_agents=delegatees)

    async def _run_agent(agent_id: str, task: str) -> str:
        name = _sanitize_agent_name(agent_id)
        child_agent_tool = _build_child_agent_tool(name)
        if name == "explore":
            dispatch = dispatch_explore_subagent
        elif name == "web-researcher":
            dispatch = dispatch_web_subagent
        elif name == "data-analyst":
            dispatch = dispatch_data_subagent
        elif name == "frontend-designer":
            dispatch = dispatch_frontend_subagent
        else:
            cfg = None
            if load_agent is not None and name != "default":
                try:
                    cfg = load_agent(name)
                except Exception:
                    cfg = None
            if cfg is None:
                raise ValueError(
                    f"Agent '{agent_id}' dispatch not wired (available: explore, "
                    "web-researcher, data-analyst, or any agents/<name>.yaml definition — "
                    "you can CREATE one with the write tool, then delegate to it by name)"
                )
            return await dispatch_config_subagent(
                task,
                agent_config=cfg,
                llm=llm,
                bus=bus,
                base_registry=base_registry,
                parent_session_id=get_parent_session_id(),
                permission_evaluator=permission_evaluator,
                context_manager=_make_child_ctx(),
                depth=depth,
                max_subagent_depth=max_subagent_depth,
            )
        return await dispatch(
            task,
            llm=llm,
            bus=bus,
            base_registry=base_registry,
            parent_session_id=get_parent_session_id(),
            permission_evaluator=permission_evaluator,
            context_manager=_make_child_ctx(),
            depth=depth,
            max_subagent_depth=max_subagent_depth,
            child_agent_tool=child_agent_tool,
        )

    return _run_agent
