# Spec 03: Agent Loop

**Component:** `src/localharness/agent/loop.py`
**Requirements:** LOOP-01, LOOP-02, LOOP-03, LOOP-04, LOOP-05
**Dependencies:** `core/events.py`, `core/bus.py`, `core/types.py`, `config/models.py`, `provider/client.py`, `tools/registry.py`, `tools/hooks.py`, `agent/context.py`, `agent/permissions.py`, `memory/history.py`

---

## Purpose

The agent loop is the execution engine of LocalHarness. It drives a single agent through a ReAct (Reason + Act) cycle: receive a task, build a request from current context, stream a response, extract tool calls, execute each call, observe results, and repeat until the task is complete or a termination condition fires.

The loop is a plain Python `while True` with explicit break conditions. It is not a state machine, not a graph, not a coroutine-per-tool pipeline. Simplicity is a hard requirement — an experienced engineer must be able to follow the execution path in a single reading.

Every external interaction (tool execution, memory writes, channel delivery) is mediated through the event bus. The loop itself holds no direct references to the tool system, memory layer, or channel adapters — it publishes `Action` events and waits for corresponding `Observation` events.

---

## File Layout

```
src/localharness/agent/
    __init__.py
    loop.py          # AgentLoop, Session, StuckDetector, BudgetTracker, KillWatcher
    context.py       # ContextManager — see spec 08
    permissions.py   # PermissionEvaluator — separate spec
```

---

## Data Structures

### `Session`

Represents the in-memory state of a single agent execution turn. Created fresh for each `run_turn()` call. Not persisted directly — persisted via the event bus JSONL log.

```python
from dataclasses import dataclass, field
from typing import Any
import time

from openai.types.chat import ChatCompletionMessage
from localharness.core.types import Message

@dataclass
class Session:
    agent_id: str
    """Stable agent identifier from config (e.g. 'morning-briefing')."""

    session_id: str
    """UUID4 generated at Session construction time. Unique per run_turn() call.
    Used to correlate all events from this execution."""

    messages: list[Message]
    """Ordered conversation history. Append-only during execution.
    Initialized with system prompt + loaded memory + task message.
    Passed to ContextManager before every LLM request for compaction check."""

    iteration: int = 0
    """Number of completed loop iterations. Incremented at the top of each loop body.
    Used for budget enforcement and stuck detection."""

    actions_taken: int = 0
    """Total tool calls executed this session. Incremented per tool call (not per turn).
    Multiple tool calls in one response each count separately."""

    start_time: float = field(default_factory=time.monotonic)
    """Session start timestamp (monotonic). Used for max_duration_minutes enforcement."""

    tool_call_log: list[dict[str, Any]] = field(default_factory=list)
    """Append-only log of every tool call: {tool_name, args_hash, iteration, timestamp}.
    Read by StuckDetector. NOT the full args — only the canonical hash."""

    summary: str = ""
    """Populated by summarize() after the loop exits. Returned to orchestrator.
    Empty string until loop completion."""

    terminated_reason: str | None = None
    """Set when the loop exits. One of:
    - None: still running
    - 'complete': natural completion (no more tool calls)
    - 'budget_actions': max_actions reached
    - 'budget_time': max_duration_minutes reached
    - 'kill_file': KILL file detected
    - 'stuck': stuck detector escalated
    - 'error': unrecoverable error"""

    def push(self, message: Message) -> None:
        """Append a message to session history.
        Messages are never removed from here — ContextManager handles pruning
        on a copy for LLM requests, not the canonical session.messages list."""

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.start_time

    def elapsed_minutes(self) -> float:
        return self.elapsed_seconds() / 60.0
```

### `BudgetTracker`

```python
@dataclass
class BudgetTracker:
    max_actions: int
    """Maximum total tool calls allowed. From agent config. 0 = unlimited."""

    max_duration_minutes: float
    """Maximum wall-clock minutes for the entire session. 0.0 = unlimited."""

    def check(self, session: Session) -> "BudgetViolation | None":
        """Return a BudgetViolation if any limit is exceeded, else None.
        Called at the top of every loop iteration, before building the LLM request."""

@dataclass
class BudgetViolation:
    reason: Literal["actions", "time"]
    limit: int | float
    current: int | float
    message: str
    """Human-readable message for the final assistant response."""
```

### `StuckDetector`

```python
import hashlib

@dataclass
class StuckDetector:
    window_size: int = 5
    """Sliding window of last N action signatures to check for repetition."""

    recovery_threshold: int = 2
    """Number of identical signatures in window before injecting recovery message."""

    escalation_threshold: int = 3
    """Number of identical signatures in window before stopping the agent."""

    def record(self, tool_name: str, args: dict) -> None:
        """Append this tool call's signature to the window.
        Window is a fixed-size deque — oldest entries are dropped automatically."""

    def check(self) -> "StuckState":
        """Analyze the current window for repetition.

        Returns:
            StuckState.CLEAR: No repeated signatures detected.
            StuckState.RECOVERING: recovery_threshold met. Caller must inject
                                   recovery message into next LLM request.
            StuckState.ESCALATE: escalation_threshold met. Caller must stop loop.
        """

    def compute_signature(self, tool_name: str, args: dict) -> str:
        """Return SHA-256 hex digest of tool_name + canonical JSON of args.

        Canonicalization: json.dumps(args, sort_keys=True, separators=(',', ':'))
        This ensures {"a": 1, "b": 2} and {"b": 2, "a": 1} produce identical signatures.
        """
        canonical = f"{tool_name}:{json.dumps(args, sort_keys=True, separators=(',', ':'))}"
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
        # 16 hex chars (64 bits) is sufficient for duplicate detection in a 5-item window

    def recovery_message(self, repeated_signature: str) -> str:
        """Return the recovery injection text for the assistant message prefix."""
        return (
            "You have attempted the same tool call multiple times with identical arguments "
            "and received the same result. That approach is not working. "
            "Consider a fundamentally different strategy: try different arguments, "
            "use a different tool, or conclude that the information is not available this way."
        )

from enum import Enum
class StuckState(Enum):
    CLEAR = "clear"
    RECOVERING = "recovering"
    ESCALATE = "escalate"
```

### `KillWatcher`

```python
from pathlib import Path

@dataclass
class KillWatcher:
    kill_file_path: Path
    """Path to watch. Default: Path.cwd() / 'KILL'.
    Configurable in org config so multiple agents watch the same file."""

    def is_killed(self) -> bool:
        """Return True if the KILL file exists at kill_file_path.
        Uses Path.exists() — no file locking, no inotify.
        Called at the top of every loop iteration.
        Stat() on a missing file is effectively free on local filesystems."""
```

---

## `AgentLoop`

```python
import asyncio
import uuid
import logging
from collections.abc import AsyncIterator, Callable, Awaitable
from typing import Any

from localharness.config.models import AgentConfig
from localharness.core.bus import EventBus
from localharness.core.events import (
    TurnStarted, TurnCompleted, TurnFailed,
    ActionEvent, ObservationEvent, StuckEvent, BudgetExceededEvent, KillEvent,
)
from localharness.core.types import Message, ToolCall, ToolSchema
from localharness.provider.client import LLMClient
from localharness.agent.context import ContextManager
from localharness.agent.loop import Session, BudgetTracker, StuckDetector, KillWatcher

log = logging.getLogger("localharness.agent.loop")

class AgentLoop:
    """ReAct while-loop agent executor.

    One instance per agent. Stateless between run_turn() calls — all session
    state lives in the Session object, which is created fresh per call.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm: LLMClient,
        bus: EventBus,
        context_manager: ContextManager,
        tool_registry: "ToolRegistry",
        permission_evaluator: "PermissionEvaluator",
        memory_loader: "MemoryLoader",
        kill_file_path: Path | None = None,
    ) -> None:
        """
        Args:
            config: Fully resolved agent configuration (after inheritance resolution).
            llm: Constructed and capability-probed LLMClient for this agent.
            bus: Shared event bus. All tool calls and observations flow through here.
            context_manager: ContextManager for this agent's token budget and compaction.
            tool_registry: Scoped tool registry for this agent (global + division + agent tools).
            permission_evaluator: Evaluates deny patterns against tool calls.
            memory_loader: Loads MEMORY.md + SQLite facts for system prompt injection.
            kill_file_path: Override KILL file path. Default: Path.cwd() / 'KILL'.
        """
        self._config = config
        self._llm = llm
        self._bus = bus
        self._ctx = context_manager
        self._tools = tool_registry
        self._permissions = permission_evaluator
        self._memory = memory_loader
        self._kill = KillWatcher(kill_file_path or Path.cwd() / "KILL")

    async def run_turn(
        self,
        task: str,
        initial_messages: list[Message] | None = None,
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Execute a full agent turn from task input to final summary.

        This is the primary external entry point. Called by the orchestrator
        for each task dispatched to this agent.

        Args:
            task: The task description. Appended as a user message after the
                  system prompt and loaded memory context.
            initial_messages: Optional prior conversation to resume from.
                              If provided, replaces the default empty history.
                              Used for context restoration (pause/resume).
            on_token: Async callback for streaming token delivery to the channel.
                      Forwarded to LLMClient.stream_complete(). May be None.

        Returns:
            Summary string for the orchestrator. Always a non-empty string.
            If the agent fails, the summary describes what was attempted and why
            it stopped — never an empty string and never a Python traceback.

        Raises:
            Never raises to the caller. All errors are caught, logged, and
            reflected in the session's terminated_reason and the returned summary.
            The caller (orchestrator) can inspect the session via the event bus
            if it needs to distinguish success from failure.

        Side effects:
            - Publishes TurnStarted, ActionEvent(s), ObservationEvent(s),
              TurnCompleted or TurnFailed to the event bus.
            - Appends to the agent's JSONL history file via the bus subscriber.
        """
        session = Session(
            agent_id=self._config.name,
            session_id=str(uuid.uuid4()),
            messages=list(initial_messages) if initial_messages else [],
        )

        await self._bus.emit(TurnStarted(
            agent_id=session.agent_id,
            session_id=session.session_id,
            task=task,
        ))

        try:
            summary = await self._execute_loop(session, task, on_token)
        except Exception as exc:
            log.exception("Unhandled error in agent loop for %s", self._config.name)
            session.terminated_reason = "error"
            summary = (
                f"Agent {self._config.name} encountered an unexpected error: {type(exc).__name__}: {exc}. "
                f"Completed {session.actions_taken} tool calls across {session.iteration} iterations "
                f"before stopping."
            )
            await self._bus.emit(TurnFailed(
                agent_id=session.agent_id,
                session_id=session.session_id,
                error=str(exc),
                iterations=session.iteration,
                actions_taken=session.actions_taken,
            ))
            return summary

        session.summary = summary
        await self._bus.emit(TurnCompleted(
            agent_id=session.agent_id,
            session_id=session.session_id,
            summary=summary,
            iterations=session.iteration,
            actions_taken=session.actions_taken,
            terminated_reason=session.terminated_reason or "complete",
        ))
        return summary

    async def step(
        self,
        session: Session,
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> "StepResult":
        """Execute one iteration of the ReAct loop.

        For testing and debugging. Exposed publicly so tests can drive the loop
        manually iteration by iteration and inspect intermediate state.

        Args:
            session: Current session state. Modified in place.
            on_token: Streaming token callback, forwarded to LLM.

        Returns:
            StepResult describing what happened in this iteration.

        Raises:
            BudgetExceeded: Budget was exceeded before the step ran.
            KillSignal: KILL file was detected before the step ran.
            Stuck: Stuck detector escalated before or after the step.
        """

    async def abort(self, session: Session, reason: str) -> None:
        """Immediately abort a running session.

        Called by external code (orchestrator escalation, test teardown) to
        forcefully stop the loop. Sets session.terminated_reason and emits TurnFailed.

        Note: If the loop is blocked inside an LLM call or tool execution,
        this cancels the underlying asyncio Task. Callers should handle
        asyncio.CancelledError when awaiting run_turn().
        """
```

### `StepResult`

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class StepResult:
    action: Literal["tool_calls", "complete", "budget", "kill", "stuck", "error"]
    """What happened in this step:
    - tool_calls: LLM issued tool calls; they were executed; loop will continue.
    - complete: LLM issued no tool calls; session is done.
    - budget: Budget limit reached; loop will stop.
    - kill: KILL file detected; loop will stop.
    - stuck: Stuck detector escalated; loop will stop.
    - error: Unrecoverable error during LLM call or tool execution.
    """
    tool_calls_executed: int = 0
    llm_response_preview: str = ""
    """First 200 chars of the LLM response content, for logging."""
    error: str | None = None
```

---

## ReAct While-Loop: Full Pseudocode

Every decision point is shown explicitly.

```
_execute_loop(session, task, on_token):

    # ── Initialization ─────────────────────────────────────────────────────────

    budget = BudgetTracker(
        max_actions = config.permissions.budget.max_actions,
        max_duration_minutes = config.permissions.budget.max_duration_minutes,
    )
    stuck_detector = StuckDetector(
        window_size = 5,
        recovery_threshold = 2,
        escalation_threshold = 3,
    )
    tools = tool_registry.get_tools_for_agent(config)  # scoped list
    tool_schemas = [t.info() for t in tools]

    # Load system prompt + memory
    system_prompt = _build_system_prompt(config)
    memory_context = await memory_loader.load(config)  # returns str
    if memory_context:
        system_prompt += "\n\n## Agent Memory\n" + memory_context

    # Initialize session messages
    session.push({"role": "system", "content": system_prompt})
    session.push({"role": "user", "content": task})

    # ── Main Loop ──────────────────────────────────────────────────────────────

    recovery_injection: str | None = None

    while True:
        session.iteration += 1

        # ── Termination Guards (checked before LLM call) ──────────────────────

        # 1. KILL file check
        if kill_watcher.is_killed():
            session.terminated_reason = "kill_file"
            await bus.emit(KillEvent(agent_id=session.agent_id, session_id=session.session_id))
            log.warning("KILL file detected, stopping agent %s after %d iterations",
                        config.name, session.iteration)
            return _format_kill_summary(session)

        # 2. Budget check
        violation = budget.check(session)
        if violation is not None:
            session.terminated_reason = f"budget_{violation.reason}"
            await bus.emit(BudgetExceededEvent(
                agent_id=session.agent_id,
                session_id=session.session_id,
                reason=violation.reason,
                limit=violation.limit,
                current=violation.current,
            ))
            log.info("Budget exceeded for %s: %s (limit=%s, current=%s)",
                     config.name, violation.reason, violation.limit, violation.current)
            return _format_budget_summary(session, violation)

        # ── Context Preparation ───────────────────────────────────────────────

        # 3. Build message list for LLM request
        #    ContextManager works on a COPY — session.messages is the canonical log.
        request_messages = context_manager.build_messages(
            session.messages,
            tool_schemas=tool_schemas,
        )
        # build_messages calls repair_tool_pairing internally.
        # Returns the (possibly compacted) message list ready for the API.

        # 4. Inject recovery message if stuck detector previously fired RECOVERING
        if recovery_injection is not None:
            # Insert as a system turn at the end of request_messages
            request_messages.append({
                "role": "system",
                "content": recovery_injection,
            })
            recovery_injection = None  # consumed; reset for next iteration

        # ── LLM Call ─────────────────────────────────────────────────────────

        # 5. Stream completion
        try:
            response_message = await llm.stream_complete(
                messages=request_messages,
                tools=tool_schemas if config.llm.tool_call_mode != "text" else None,
                on_token=on_token,
            )
        except ProviderConnectionError as exc:
            # Retry once with 2s backoff; if second call also fails, stop
            log.warning("LLM connection error in %s (iter %d): %s", config.name, session.iteration, exc)
            await asyncio.sleep(2.0)
            try:
                response_message = await llm.stream_complete(
                    messages=request_messages,
                    tools=tool_schemas if config.llm.tool_call_mode != "text" else None,
                    on_token=on_token,
                )
            except ProviderConnectionError as exc2:
                session.terminated_reason = "error"
                return _format_error_summary(session, exc2)
        except ProviderTimeoutError as exc:
            log.error("LLM timeout in %s (iter %d): %s", config.name, session.iteration, exc)
            session.terminated_reason = "error"
            return _format_error_summary(session, exc)
        except ProviderAPIError as exc:
            if exc.status_code == 400:
                # This indicates repair_tool_pairing() did not catch something.
                # Log the full message list for debugging. Do not retry.
                log.error(
                    "HTTP 400 from LLM in %s — possible orphaned tool_result. "
                    "This is a harness bug. Session messages: %s",
                    config.name, json.dumps(request_messages, default=str)
                )
            session.terminated_reason = "error"
            return _format_error_summary(session, exc)

        # 6. Push assistant response to canonical session
        session.push({"role": "assistant", "content": response_message.content,
                       "tool_calls": response_message.tool_calls})

        await bus.emit(ActionEvent(
            agent_id=session.agent_id,
            session_id=session.session_id,
            iteration=session.iteration,
            content=response_message.content,
            tool_calls=[tc.model_dump() for tc in (response_message.tool_calls or [])],
        ))

        # ── Tool Call Extraction ───────────────────────────────────────────────

        # 7. Extract tool calls (normalized — same structure regardless of mode)
        tool_calls = _extract_tool_calls(response_message, llm.config.tool_call_mode)
        # For native mode: response_message.tool_calls directly
        # For xml mode: fn_call_converter.extract_tool_calls(response_message.content)
        # For text mode: same as xml

        # 8. No tool calls → loop exits (natural completion)
        if not tool_calls:
            session.terminated_reason = "complete"
            return _format_completion_summary(session, response_message.content)

        # ── Tool Execution ─────────────────────────────────────────────────────

        # 9. Execute each tool call
        for tool_call in tool_calls:
            session.actions_taken += 1

            # 9a. Permission check
            permission_result = permission_evaluator.evaluate(tool_call, config.permissions)
            if permission_result.denied:
                # Inject denial as tool_result so model knows and can reconsider
                denial_result = _make_tool_result(
                    tool_call_id=tool_call.id,
                    content=f"Permission denied: {permission_result.reason}",
                    is_error=True,
                )
                session.push(denial_result)
                await bus.emit(ObservationEvent(
                    agent_id=session.agent_id,
                    session_id=session.session_id,
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result_preview="[DENIED]",
                    is_error=True,
                ))
                continue  # continue to next tool_call in this batch

            # 9b. Schema validation
            tool = tool_registry.get(tool_call.name)
            if tool is None:
                tool_result = _make_tool_result(
                    tool_call_id=tool_call.id,
                    content=f"Unknown tool: {tool_call.name}",
                    is_error=True,
                )
                session.push(tool_result)
                continue

            validation_error = _validate_tool_args(tool_call.arguments, tool.info())
            if validation_error:
                tool_result = _make_tool_result(
                    tool_call_id=tool_call.id,
                    content=f"Invalid arguments: {validation_error}",
                    is_error=True,
                )
                session.push(tool_result)
                continue

            # 9c. Pre-execution hook
            hook_system.run_pre_hooks(tool_call, config)
            # Pre-hooks may raise HookVeto — caught below

            # 9d. Execute via event bus
            #     Publish Action to bus; tool system subscriber executes and
            #     publishes Observation back; we await the matching observation.
            action_event = ActionEvent(
                agent_id=session.agent_id,
                session_id=session.session_id,
                iteration=session.iteration,
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                tool_args=tool_call.arguments,
            )
            await bus.emit(action_event)
            observation = await _await_observation(bus, tool_call.id, timeout=config.tool_timeout_seconds)

            # 9e. Post-execution hook (lint/typecheck gates fire here)
            hook_system.run_post_hooks(tool_call, observation, config)

            # 9f. Push tool result to session
            tool_result_message = _make_tool_result(
                tool_call_id=tool_call.id,
                content=observation.result,
                is_error=observation.is_error,
            )
            session.push(tool_result_message)

            await bus.emit(ObservationEvent(
                agent_id=session.agent_id,
                session_id=session.session_id,
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result_preview=observation.result[:200],
                is_error=observation.is_error,
            ))

            # 9g. Record to stuck detector
            stuck_detector.record(tool_call.name, tool_call.arguments)

        # ── Stuck Detection (after batch) ─────────────────────────────────────

        # 10. Check stuck state after processing all tool calls in this batch
        stuck_state = stuck_detector.check()

        if stuck_state == StuckState.RECOVERING:
            # Inject recovery hint into the NEXT iteration's request
            # (not this one — the tool results are already appended)
            repeated_sig = stuck_detector.most_repeated_signature()
            recovery_injection = stuck_detector.recovery_message(repeated_sig)
            log.info("Stuck recovery triggered for %s at iteration %d",
                     config.name, session.iteration)

        elif stuck_state == StuckState.ESCALATE:
            session.terminated_reason = "stuck"
            await bus.emit(StuckEvent(
                agent_id=session.agent_id,
                session_id=session.session_id,
                iteration=session.iteration,
                repeated_signature=stuck_detector.most_repeated_signature(),
            ))
            log.warning("Agent %s stuck after %d iterations, escalating",
                        config.name, session.iteration)
            return _format_stuck_summary(session)

        # ── Loop continues ─────────────────────────────────────────────────────
        # (back to top: kill check → budget check → context prep → LLM call)
```

---

## `repair_tool_pairing()` Algorithm

This function is the boundary guard described in PITFALLS.md Pitfall 1. It is called inside `ContextManager.build_messages()` on every message list before it is sent to the LLM. It is also exposed here for documentation because the agent loop depends on it.

**Full specification is in spec 08-context-management.md.** The contract from the agent loop's perspective:

- `build_messages()` guarantees that its returned list has no orphaned `tool_result` blocks.
- If `build_messages()` raises `RepairImpossibleError`, the agent loop must not proceed with the LLM call. It should log the error and terminate the session with `terminated_reason = "error"`.
- The agent loop never calls `repair_tool_pairing()` directly. It calls `context_manager.build_messages()` which calls it internally.

---

## Tool-Use / Tool-Result Boundary Guard

The boundary guard has two responsibilities:
1. Pre-request validation (before every LLM call)
2. Post-compaction repair (after any compaction changes the message list)

From the agent loop's perspective, both are handled by `context_manager.build_messages()`. The loop's responsibility is to ensure it never bypasses that call.

```
CORRECT:
    request_messages = context_manager.build_messages(session.messages, ...)
    response = await llm.stream_complete(messages=request_messages, ...)

WRONG:
    response = await llm.stream_complete(messages=session.messages, ...)
    # This bypasses repair_tool_pairing() and will cause HTTP 400 after compaction
```

---

## Stuck Detection: Action Signature Hashing

### Sliding Window

The stuck detector maintains a `collections.deque(maxlen=window_size)` of action signatures. Each call to `record()` appends one signature and drops the oldest if at capacity.

```python
from collections import deque

class StuckDetector:
    def __init__(self, window_size=5, recovery_threshold=2, escalation_threshold=3):
        self._window: deque[str] = deque(maxlen=window_size)
        self.window_size = window_size
        self.recovery_threshold = recovery_threshold
        self.escalation_threshold = escalation_threshold

    def record(self, tool_name: str, args: dict) -> None:
        self._window.append(self.compute_signature(tool_name, args))

    def check(self) -> StuckState:
        if len(self._window) < self.recovery_threshold:
            return StuckState.CLEAR
        counts = Counter(self._window)
        max_count = counts.most_common(1)[0][1]
        if max_count >= self.escalation_threshold:
            return StuckState.ESCALATE
        if max_count >= self.recovery_threshold:
            return StuckState.RECOVERING
        return StuckState.CLEAR

    def most_repeated_signature(self) -> str:
        if not self._window:
            return ""
        return Counter(self._window).most_common(1)[0][0]
```

### Signature Computation

```python
def compute_signature(self, tool_name: str, args: dict) -> str:
    canonical = f"{tool_name}:{json.dumps(args, sort_keys=True, separators=(',', ':'))}"
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
```

Key properties:
- `sort_keys=True`: Argument order does not matter. `{"a":1,"b":2}` and `{"b":2,"a":1}` produce identical signatures.
- 16 hex chars (64 bits): Sufficient collision resistance for 5-item windows.
- Truncated SHA-256 (not MD5): Fast, sufficient, and no external dependency.

### Recovery Protocol

```
STUCK STATE MACHINE:

CLEAR ──────────────────────────────────────────────────────────► CLEAR
  (every tool call adds to window; no repeat pattern detected)

CLEAR ──[2 identical in window]──────────────────────────────────► RECOVERING
  Action:
    Set recovery_injection = stuck_detector.recovery_message(sig)
    Log: "Stuck recovery triggered at iteration N"
    Continue loop normally (do NOT stop the agent)

RECOVERING ──[3 identical in window]────────────────────────────► ESCALATE
  Action:
    Set session.terminated_reason = "stuck"
    Emit StuckEvent to bus
    Return _format_stuck_summary(session)
    (loop exits)

RECOVERING ──[recovery injection changes behavior]───────────────► CLEAR
  (if the recovery hint worked and the model uses different args,
   new signatures dominate the window and max_count drops below threshold)
```

Recovery injection text is appended to `request_messages` as a `{"role": "system"}` message. This choice over `{"role": "user"}` is deliberate: system messages carry higher instruction weight with most models, and the recovery hint is a harness-level directive, not user input.

---

## Budget Enforcement

### `BudgetTracker.check()`

```python
def check(self, session: Session) -> BudgetViolation | None:
    if self.max_actions > 0 and session.actions_taken >= self.max_actions:
        return BudgetViolation(
            reason="actions",
            limit=self.max_actions,
            current=session.actions_taken,
            message=(
                f"Agent reached the maximum of {self.max_actions} tool calls. "
                f"Stopping to prevent runaway execution. "
                f"Increase max_actions in agent config if more calls are needed."
            ),
        )
    if self.max_duration_minutes > 0 and session.elapsed_minutes() >= self.max_duration_minutes:
        return BudgetViolation(
            reason="time",
            limit=self.max_duration_minutes,
            current=session.elapsed_minutes(),
            message=(
                f"Agent reached the time limit of {self.max_duration_minutes:.1f} minutes. "
                f"Stopping after {session.actions_taken} tool calls. "
                f"Increase max_duration_minutes in agent config if more time is needed."
            ),
        )
    return None
```

### Graceful Shutdown

When a budget limit fires, the agent loop:
1. Does NOT call the LLM for a closing message — that would consume budget and delay stop.
2. Generates the summary from the last assistant message in `session.messages`.
3. Returns the summary with the budget violation reason appended.
4. Emits `BudgetExceededEvent` before returning.

```python
def _format_budget_summary(session: Session, violation: BudgetViolation) -> str:
    last_assistant_content = _last_assistant_content(session.messages)
    return (
        f"{last_assistant_content}\n\n"
        f"[Budget limit reached: {violation.message} "
        f"Completed {session.actions_taken} tool calls in {session.elapsed_minutes():.1f} minutes.]"
    )
```

---

## Kill Switch: KILL File Watch Mechanism

The KILL file mechanism provides immediate, deterministic stop for any running agent. It requires no inter-process signaling infrastructure.

```
KILL FILE PROTOCOL:
───────────────────
Location:    Path.cwd() / "KILL"  (default)
             Configurable via kill_file_path in org config

Check point: Top of every loop iteration, before any other work

Implementation:
    Path(kill_file_path).exists()
    — No file locking
    — No inotify or filesystem watches
    — Pure stat() call — extremely fast on local filesystems
    — Checked synchronously (not async) because it's a fast local op

Stop behavior:
    - session.terminated_reason = "kill_file"
    - Emit KillEvent to bus
    - Return summary: "Agent stopped by kill signal after N iterations."
    - DO NOT delete the KILL file — that is the operator's responsibility

Creating a kill:
    touch KILL      (stops all agents watching cwd)
    echo "reason" > KILL   (optional content, ignored by harness)

Clearing:
    rm KILL         (agents in subsequent sessions will run normally)

Multiple agents:
    All agents use the same KILL file path by default.
    To kill a specific agent, use abort() from the orchestrator instead.
```

---

## Error Recovery

### Tool Failure Recovery

When a tool execution produces an error result (via `observation.is_error = True`):
- The error is pushed to `session.messages` as a `tool_result` with the error content.
- The loop continues — the model sees the error and can decide to retry with different args, use a different approach, or give up.
- Error tool results count toward `session.actions_taken`.
- The stuck detector receives the signature — consecutive identical calls after an error will trigger recovery.

```python
# Error tool result message format
{
    "role": "tool",
    "tool_call_id": "<matching tool_call id>",
    "content": "Error: <error description>",
}
```

### LLM Error Recovery

| Error Type | Recovery Action | Condition to Stop |
|-----------|----------------|------------------|
| `ProviderConnectionError` | Retry once after 2s | Second failure → stop |
| `ProviderTimeoutError` | Log and stop | No retry — timed out on a local server means something is wrong |
| `ProviderRateLimitError` | Wait `retry_after_seconds` (max 30s), retry up to 3x | 3rd failure → stop |
| `ProviderAPIError(400)` | Log full message list, stop | Always a harness bug |
| `ProviderAPIError(5xx)` | Log, stop | Server error, not retryable |
| `MalformedResponseError` | Inject retry prompt once | Second empty response → stop |

### Context Overflow Recovery

If `ContextManager.build_messages()` raises `ContextOverflowError` (context is too full to compact further):
1. Log at ERROR: "Context overflow for agent {name} at iteration {N}. Stopping."
2. Set `session.terminated_reason = "error"`
3. Return the last meaningful assistant message as the summary.
4. Do NOT attempt to continue — sending an overflowed context will produce garbage.

---

## Event Bus Integration

The agent loop communicates exclusively with other components through the event bus. No component is imported directly into the loop file except `LLMClient` and `ContextManager`, which are injected at construction.

### Events Published by Agent Loop

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class TurnStarted:
    agent_id: str
    session_id: str
    task: str

@dataclass
class TurnCompleted:
    agent_id: str
    session_id: str
    summary: str
    iterations: int
    actions_taken: int
    terminated_reason: str

@dataclass
class TurnFailed:
    agent_id: str
    session_id: str
    error: str
    iterations: int
    actions_taken: int

@dataclass
class ActionEvent:
    agent_id: str
    session_id: str
    iteration: int
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_args: dict[str, Any] | None = None

@dataclass
class ObservationEvent:
    agent_id: str
    session_id: str
    tool_call_id: str
    tool_name: str
    result_preview: str
    is_error: bool

@dataclass
class StuckEvent:
    agent_id: str
    session_id: str
    iteration: int
    repeated_signature: str

@dataclass
class BudgetExceededEvent:
    agent_id: str
    session_id: str
    reason: Literal["actions", "time"]
    limit: int | float
    current: int | float

@dataclass
class KillEvent:
    agent_id: str
    session_id: str
```

### Awaiting Observations

When the loop publishes an `ActionEvent` for a tool call, it waits for the corresponding `ObservationEvent` via a `Future`-based matching mechanism:

```python
async def _await_observation(
    bus: EventBus,
    tool_call_id: str,
    timeout: float,
) -> ObservationRaw:
    """Wait for ObservationEvent with matching tool_call_id.

    Implementation:
    - Create asyncio.Future before publishing ActionEvent
    - Register a one-shot subscriber on the bus that resolves the Future
      when ObservationEvent.tool_call_id matches
    - await asyncio.wait_for(future, timeout=timeout)
    - Deregister subscriber on completion or timeout

    Timeout:
        Uses config.tool_timeout_seconds (default 120s, separate from LLM timeout).
        Tool operations (bash, file I/O) should complete in well under this.
        On timeout: return ObservationRaw(is_error=True, result="Tool execution timed out.")
    """
```

---

## Integration Points

| Component | Interface | Direction |
|-----------|-----------|-----------|
| Event bus | `bus.emit()`, `bus.subscribe()` | Loop → Bus, Bus → Loop (observations) |
| LLM provider | `llm.stream_complete()` | Loop → LLM |
| Context manager | `ctx.build_messages()` | Loop → Context |
| Tool registry | `tools.get(name)`, `tools.get_tools_for_agent()` | Loop → Tools |
| Permission evaluator | `permissions.evaluate(call, config)` | Loop → Permissions |
| Memory loader | `memory.load(config)` | Loop → Memory (at startup only) |
| Hook system | `hooks.run_pre_hooks()`, `hooks.run_post_hooks()` | Loop → Hooks |

---

## Configuration

Agent YAML fields consumed by the agent loop:

```yaml
permissions:
  budget:
    max_actions: 100      # 0 = unlimited
    max_duration_minutes: 30  # 0.0 = unlimited

  deny_patterns: []       # evaluated by PermissionEvaluator, not loop directly

tool_timeout_seconds: 120  # timeout waiting for tool execution observation
max_iterations: 200        # hard cap on loop iterations (separate from max_actions)
                           # prevents infinite loops even with unlimited budget
```

Org-level config:
```yaml
kill_file_path: "/home/user/KILL"  # optional; default: cwd/KILL
```

---

## Implementation Notes

1. **`session.messages` is append-only.** The loop never removes from it. `ContextManager.build_messages()` returns a pruned copy for LLM requests. This ensures the canonical session history remains complete for debugging and replay.

2. **Tool calls are executed sequentially within a batch.** Even when the model returns multiple tool calls in one response, they execute one at a time. Parallel execution is a v2 feature. Sequential execution makes debugging dramatically simpler and avoids race conditions in tools that share state (filesystem, SQLite).

3. **The `_execute_loop()` method is internal.** It is separated from `run_turn()` only for error handling clarity. Do not expose it publicly. Tests should use `step()` for fine-grained control.

4. **Log every iteration.** At DEBUG level, log: `agent={name} iter={N} actions={A} elapsed={T:.1f}s context_pct={P:.0f}%`. This is the primary diagnostic for long-running agents and helps reproduce stuck loops.

5. **`on_token` callback must be non-blocking.** The loop `await`s it after each token. If the callback does significant work (e.g., writes to disk), it will slow the streaming display. Channel adapters should buffer and flush, not write per-token.

6. **Summary format convention.** All `_format_*_summary()` functions return a string starting with the agent's last meaningful assistant message content, followed by a bracketed status line. This ensures orchestrators always get actionable content in the summary, not just metadata.

7. **`repair_tool_pairing()` before every request is not optional.** Even if compaction has not run, the session may contain malformed sequences from prior error recovery paths. The check is fast (one pass through the message list). Never skip it.
