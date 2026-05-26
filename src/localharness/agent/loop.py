"""AgentLoop, Session, StuckDetector, BudgetTracker, KillWatcher, StepResult.

ReAct while-loop execution engine for LocalHarness agents.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

from localharness.core.types import Message

log = logging.getLogger("localharness.agent.loop")


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass
class Session:
    agent_id: str
    session_id: str
    messages: list[Message]
    iteration: int = 0
    actions_taken: int = 0
    start_time: float = field(default_factory=time.monotonic)
    tool_call_log: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    terminated_reason: str | None = None

    def push(self, message: Message) -> None:
        self.messages.append(message)

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.start_time

    def elapsed_minutes(self) -> float:
        return self.elapsed_seconds() / 60.0


# ---------------------------------------------------------------------------
# StuckDetector
# ---------------------------------------------------------------------------

class StuckState(Enum):
    CLEAR = "clear"
    RECOVERING = "recovering"
    ESCALATE = "escalate"


class StuckDetector:
    def __init__(
        self,
        window_size: int = 5,
        recovery_threshold: int = 2,
        escalation_threshold: int = 3,
    ) -> None:
        self.window_size = window_size
        self.recovery_threshold = recovery_threshold
        self.escalation_threshold = escalation_threshold
        self._window: deque[str] = deque(maxlen=window_size)

    def compute_signature(self, tool_name: str, args: dict) -> str:
        canonical = f"{tool_name}:{json.dumps(args, sort_keys=True, separators=(',', ':'))}"
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

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

    def recovery_message(self, repeated_signature: str) -> str:
        return (
            "You have attempted the same tool call multiple times with identical arguments "
            "and received the same result. That approach is not working. "
            "Consider a fundamentally different strategy: try different arguments, "
            "use a different tool, or conclude that the information is not available this way."
        )


# ---------------------------------------------------------------------------
# BudgetTracker / BudgetViolation
# ---------------------------------------------------------------------------

@dataclass
class BudgetViolation:
    reason: Literal["actions", "time"]
    limit: int | float
    current: int | float
    message: str


@dataclass
class BudgetTracker:
    max_actions: int
    max_duration_minutes: float

    def check(self, session: Session) -> BudgetViolation | None:
        if self.max_actions > 0 and session.actions_taken >= self.max_actions:
            return BudgetViolation(
                reason="actions",
                limit=self.max_actions,
                current=session.actions_taken,
                message=(
                    f"Agent reached the maximum of {self.max_actions} tool calls. "
                    f"Stopping to prevent runaway execution."
                ),
            )
        if self.max_duration_minutes > 0 and session.elapsed_minutes() >= self.max_duration_minutes:
            return BudgetViolation(
                reason="time",
                limit=self.max_duration_minutes,
                current=session.elapsed_minutes(),
                message=(
                    f"Agent reached the time limit of {self.max_duration_minutes:.1f} minutes. "
                    f"Stopping after {session.actions_taken} tool calls."
                ),
            )
        return None


# ---------------------------------------------------------------------------
# KillWatcher
# ---------------------------------------------------------------------------

@dataclass
class KillWatcher:
    kill_file_path: Path

    def is_killed(self) -> bool:
        return self.kill_file_path.exists()


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    action: Literal["tool_calls", "complete", "budget", "kill", "stuck", "error"]
    tool_calls_executed: int = 0
    llm_response_preview: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------

def _last_assistant_content(messages: list[Message]) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            return m["content"]
    return ""


def _format_completion_summary(session: Session, content: str | None) -> str:
    return content or _last_assistant_content(session.messages) or "Task complete."


def _format_budget_summary(session: Session, violation: BudgetViolation) -> str:
    last = _last_assistant_content(session.messages)
    prefix = f"{last}\n\n" if last else ""
    return (
        f"{prefix}"
        f"[Budget limit reached: {violation.message} "
        f"Completed {session.actions_taken} tool calls in {session.elapsed_minutes():.1f} minutes.]"
    )


def _format_kill_summary(session: Session) -> str:
    return f"Agent stopped by kill signal after {session.iteration} iterations."


def _format_stuck_summary(session: Session) -> str:
    last = _last_assistant_content(session.messages)
    prefix = f"{last}\n\n" if last else ""
    return (
        f"{prefix}"
        f"[Agent stuck: repeated identical tool calls detected after {session.iteration} iterations. "
        f"Escalating to orchestrator.]"
    )


def _format_error_summary(session: Session, exc: Exception) -> str:
    return (
        f"Agent encountered an error: {type(exc).__name__}: {exc}. "
        f"Completed {session.actions_taken} tool calls across {session.iteration} iterations."
    )


def _extract_tool_calls(response_message: Any, tool_call_mode: str) -> list:
    """Extract tool calls from an LLM response regardless of mode."""
    from localharness.core.types import ToolCall
    if tool_call_mode == "native":
        raw = getattr(response_message, "tool_calls", None) or []
        result = []
        for tc in raw:
            if hasattr(tc, "function"):
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                result.append(ToolCall(
                    name=tc.function.name,
                    arguments=args,
                    id=tc.id or str(uuid.uuid4()),
                ))
            elif isinstance(tc, dict):
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments", "{}") or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                result.append(ToolCall(
                    name=fn.get("name", ""),
                    arguments=args,
                    id=tc.get("id", str(uuid.uuid4())),
                ))
        return result
    else:
        # xml/text mode — use FnCallConverter
        try:
            from localharness.provider.fn_call import FnCallConverter
            converter = FnCallConverter()
            content = getattr(response_message, "content", "") or ""
            return converter.extract_tool_calls(content)
        except Exception:
            return []


class AgentLoop:
    """ReAct while-loop agent executor. One instance per agent."""

    def __init__(
        self,
        config: Any,  # AgentConfig
        llm: Any,  # LLMClient
        bus: Any,  # EventBus
        context_manager: Any,  # ContextManager
        tool_registry: Any,  # ToolRegistry
        permission_evaluator: Any,  # PermissionEvaluator
        memory_loader: Any = None,
        kill_file_path: Path | None = None,
        compact_md_path: Path | None = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._bus = bus
        self._ctx = context_manager
        self._tools = tool_registry
        self._permissions = permission_evaluator
        self._memory = memory_loader
        self._compact_md_path = compact_md_path
        # Determine kill file path
        if kill_file_path is not None:
            kf = kill_file_path
        elif (
            hasattr(config, "permissions")
            and hasattr(config.permissions, "budget")
            and config.permissions.budget.kill_file
        ):
            kf = Path(config.permissions.budget.kill_file).expanduser()
        else:
            kf = Path.cwd() / "KILL"
        self._kill = KillWatcher(kill_file_path=kf)
        self._current_session_id: str | None = None

    @property
    def current_session_id(self) -> str | None:
        """Return the session_id from the most recent run_turn() call."""
        return self._current_session_id

    async def run_turn(
        self,
        task: str,
        initial_messages: list[Message] | None = None,
        on_token: Callable | None = None,
    ) -> str:
        """Execute a full agent turn. Never raises — all errors become summary strings."""
        from localharness.core.events import TurnStarted, TurnCompleted, TurnFailed, BudgetSpec

        session = Session(
            agent_id=self._config.name,
            session_id=str(uuid.uuid4()),
            messages=list(initial_messages) if initial_messages else [],
        )
        self._current_session_id = session.session_id

        # Load prior session context from compact.md if present
        from localharness.agent.context import load_compact_md
        compact_path = self._compact_md_path or (Path.home() / ".localharness" / "agents" / self._config.name / "compact.md")
        compact_msg = load_compact_md(compact_path)
        if compact_msg is not None:
            insert_idx = 1 if session.messages and session.messages[0].get("role") == "system" else 0
            session.messages.insert(insert_idx, compact_msg)
            log.info("Loaded compact.md for agent %s", self._config.name)

        budget_cfg = self._config.permissions.budget
        await self._bus.publish(TurnStarted(
            agent_id=session.agent_id,
            session_id=session.session_id,
            task_summary=task[:200],
            budget=BudgetSpec(
                max_actions=budget_cfg.max_actions,
                max_duration_minutes=budget_cfg.max_duration_minutes,
            ),
        ))

        try:
            summary = await self._execute_loop(session, task, on_token)
        except Exception as exc:
            log.exception("Unhandled error in agent loop for %s", self._config.name)
            session.terminated_reason = "error"
            summary = _format_error_summary(session, exc)
            await self._bus.publish(TurnFailed(
                agent_id=session.agent_id,
                session_id=session.session_id,
                reason="internal_error",
                detail=str(exc),
                iterations=session.iteration,
                duration_seconds=session.elapsed_seconds(),
            ))
            return summary

        session.summary = summary
        reason = session.terminated_reason or "complete"
        if reason in ("budget_actions", "budget_time"):
            await self._bus.publish(TurnFailed(
                agent_id=session.agent_id,
                session_id=session.session_id,
                reason="budget_exceeded",
                detail=summary,
                iterations=session.iteration,
                duration_seconds=session.elapsed_seconds(),
            ))
        elif reason == "stuck":
            await self._bus.publish(TurnFailed(
                agent_id=session.agent_id,
                session_id=session.session_id,
                reason="stuck_detected",
                detail=summary,
                iterations=session.iteration,
                duration_seconds=session.elapsed_seconds(),
            ))
        elif reason == "kill_file":
            await self._bus.publish(TurnFailed(
                agent_id=session.agent_id,
                session_id=session.session_id,
                reason="kill_file",
                detail=summary,
                iterations=session.iteration,
                duration_seconds=session.elapsed_seconds(),
            ))
        elif reason == "error":
            await self._bus.publish(TurnFailed(
                agent_id=session.agent_id,
                session_id=session.session_id,
                reason="llm_error",
                detail=summary,
                iterations=session.iteration,
                duration_seconds=session.elapsed_seconds(),
            ))
        else:
            await self._bus.publish(TurnCompleted(
                agent_id=session.agent_id,
                session_id=session.session_id,
                iterations=session.iteration,
                duration_seconds=session.elapsed_seconds(),
                elapsed_tokens=0,
                summary=summary,
            ))
        return summary

    async def _execute_loop(self, session: Session, task: str, on_token: Callable | None) -> str:
        from localharness.provider.client import (
            ProviderConnectionError,
            ProviderTimeoutError,
            ProviderAPIError,
        )
        from localharness.core.events import Action, Observation, Escalation, Heartbeat, TaskComplete

        budget = BudgetTracker(
            max_actions=self._config.permissions.budget.max_actions,
            max_duration_minutes=self._config.permissions.budget.max_duration_minutes,
        )
        stuck_detector = StuckDetector(window_size=5, recovery_threshold=2, escalation_threshold=3)

        # Build system prompt
        system_prompt = self._config.role
        if self._memory is not None:
            try:
                ctx = await self._memory.load_context()
                parts = [system_prompt]
                if ctx.guardrails_md:
                    parts.append("## Guardrails\n" + ctx.guardrails_md)
                if ctx.division_md:
                    parts.append("## Division Context\n" + ctx.division_md)
                if ctx.agent_memory_md:
                    parts.append("## Agent Memory\n" + ctx.agent_memory_md)
                system_prompt = "\n\n".join(parts)
            except Exception:
                pass  # memory load failure is non-fatal

        # Get tool schemas
        tool_schemas: list = []
        if self._tools is not None:
            try:
                agent_id = self._config.name
                division_id = self._config.division or ""
                tool_config = self._config.tools
                tool_schemas_dict = self._tools.get_tools_for_agent(agent_id, division_id, tool_config)
                tool_schemas = list(tool_schemas_dict.values())
            except Exception:
                tool_schemas = []

        # Initialize session messages
        session.push({"role": "system", "content": system_prompt})
        session.push({"role": "user", "content": task})

        recovery_injection: str | None = None

        while True:
            session.iteration += 1
            log.debug(
                "agent=%s iter=%d actions=%d elapsed=%.1fs",
                self._config.name,
                session.iteration,
                session.actions_taken,
                session.elapsed_seconds(),
            )

            # 1. Kill check
            if self._kill.is_killed():
                session.terminated_reason = "kill_file"
                log.warning(
                    "KILL file detected, stopping agent %s after %d iterations",
                    self._config.name,
                    session.iteration,
                )
                return _format_kill_summary(session)

            # 2. Budget check
            violation = budget.check(session)
            if violation is not None:
                session.terminated_reason = f"budget_{violation.reason}"
                log.info(
                    "Budget exceeded for %s: %s (limit=%s, current=%s)",
                    self._config.name,
                    violation.reason,
                    violation.limit,
                    violation.current,
                )
                return _format_budget_summary(session, violation)

            # 3. Publish heartbeat for liveness indication
            await self._bus.publish(Heartbeat(
                agent_id=session.agent_id,
                session_id=session.session_id,
                iteration=session.iteration,
                context_utilization_pct=0.0,
                last_tool=None,
            ))

            # 4. Build request messages (copy, with repair)
            request_messages = await self._ctx.build_messages(session.messages, tool_schemas)

            # 4. Inject recovery if set
            if recovery_injection is not None:
                request_messages.append({"role": "system", "content": recovery_injection})
                recovery_injection = None

            # 5. LLM call with error handling
            tool_call_mode = getattr(
                getattr(self._llm, "config", None), "tool_call_mode", "native"
            )
            try:
                response_message = await self._llm.stream_complete(
                    messages=request_messages,
                    tools=tool_schemas if tool_call_mode != "text" else None,
                    on_token=on_token,
                )
            except ProviderConnectionError as exc:
                log.warning(
                    "LLM connection error in %s (iter %d): %s",
                    self._config.name,
                    session.iteration,
                    exc,
                )
                await asyncio.sleep(2.0)
                try:
                    response_message = await self._llm.stream_complete(
                        messages=request_messages,
                        tools=tool_schemas if tool_call_mode != "text" else None,
                        on_token=on_token,
                    )
                except ProviderConnectionError as exc2:
                    session.terminated_reason = "error"
                    return _format_error_summary(session, exc2)
            except ProviderTimeoutError as exc:
                log.error(
                    "LLM timeout in %s (iter %d): %s",
                    self._config.name,
                    session.iteration,
                    exc,
                )
                session.terminated_reason = "error"
                return _format_error_summary(session, exc)
            except ProviderAPIError as exc:
                if exc.status_code == 400:
                    log.error(
                        "HTTP 400 from LLM in %s — possible orphaned tool_result. "
                        "Request messages: %s",
                        self._config.name,
                        json.dumps(request_messages, default=str),
                    )
                session.terminated_reason = "error"
                return _format_error_summary(session, exc)

            # 6. Push assistant response to session
            content = getattr(response_message, "content", None)
            raw_tool_calls = getattr(response_message, "tool_calls", None)
            session.push({
                "role": "assistant",
                "content": content,
                "tool_calls": raw_tool_calls,
            })

            # 7. Publish Action event
            await self._bus.publish(Action(
                agent_id=session.agent_id,
                session_id=session.session_id,
                action_type="llm_response",
                content=content,
            ))

            # 8. Extract tool calls
            tool_calls = _extract_tool_calls(response_message, tool_call_mode)

            # 9. No tool calls → natural completion
            if not tool_calls:
                summary = _format_completion_summary(session, content)
                await self._bus.publish(TaskComplete(
                    agent_id=session.agent_id,
                    session_id=session.session_id,
                    success=True,
                    summary=summary,
                    duration_seconds=session.elapsed_seconds(),
                    iterations=session.iteration,
                ))
                session.terminated_reason = "complete"
                return summary

            # 10. Execute each tool call
            for tool_call in tool_calls:
                session.actions_taken += 1

                # Publish tool_call action for terminal display
                await self._bus.publish(Action(
                    agent_id=session.agent_id,
                    session_id=session.session_id,
                    action_type="tool_call",
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    tool_params=tool_call.arguments,
                ))

                # Permission check
                perm_result = self._permissions.evaluate(tool_call, self._config.permissions)
                if perm_result.denied:
                    session.push({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Permission denied: {perm_result.reason}",
                    })
                    await self._bus.publish(Observation(
                        agent_id=session.agent_id,
                        session_id=session.session_id,
                        observation_type="tool_result",
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        output="[DENIED]",
                        error=f"Permission denied: {perm_result.reason}",
                    ))
                    stuck_detector.record(tool_call.name, tool_call.arguments)
                    continue

                # Dispatch via registry
                result_content = ""
                is_error = False
                if self._tools is not None:
                    try:
                        result = await self._tools.dispatch(
                            tool_call.name,
                            tool_call.arguments,
                            self._config.name,
                            self._config.division or "",
                            self._config.tools,
                        )
                        result_content = result.output
                        is_error = not result.success
                    except Exception as exc:
                        result_content = f"Error: {exc}"
                        is_error = True
                else:
                    result_content = "No tool registry available."
                    is_error = True

                session.push({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_content,
                })

                await self._bus.publish(Observation(
                    agent_id=session.agent_id,
                    session_id=session.session_id,
                    observation_type="tool_result",
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    output=result_content[:200],
                    error=result_content if is_error else None,
                ))

                stuck_detector.record(tool_call.name, tool_call.arguments)

            # 11. Check stuck state
            stuck_state = stuck_detector.check()
            if stuck_state == StuckState.RECOVERING:
                repeated_sig = stuck_detector.most_repeated_signature()
                recovery_injection = stuck_detector.recovery_message(repeated_sig)
                log.info(
                    "Stuck recovery triggered for %s at iteration %d",
                    self._config.name,
                    session.iteration,
                )
            elif stuck_state == StuckState.ESCALATE:
                session.terminated_reason = "stuck"
                await self._bus.publish(Escalation(
                    agent_id=session.agent_id,
                    session_id=session.session_id,
                    reason="stuck_detected",
                    detail=f"Repeated identical tool calls at iteration {session.iteration}",
                    stuck_signature=stuck_detector.most_repeated_signature(),
                    iteration_at_escalation=session.iteration,
                ))
                log.warning(
                    "Agent %s stuck after %d iterations, escalating",
                    self._config.name,
                    session.iteration,
                )
                return _format_stuck_summary(session)

    async def step(self, session: Session, on_token: Callable | None = None) -> StepResult:
        """Single iteration for testing/debugging."""
        from localharness.provider.client import ProviderConnectionError, ProviderTimeoutError, ProviderAPIError

        # Guardrail checks
        if self._kill.is_killed():
            session.terminated_reason = "kill_file"
            return StepResult(action="kill")

        budget = BudgetTracker(
            max_actions=self._config.permissions.budget.max_actions,
            max_duration_minutes=self._config.permissions.budget.max_duration_minutes,
        )
        violation = budget.check(session)
        if violation is not None:
            session.terminated_reason = f"budget_{violation.reason}"
            return StepResult(action="budget")

        session.iteration += 1
        tool_call_mode = getattr(
            getattr(self._llm, "config", None), "tool_call_mode", "native"
        )
        tool_schemas: list = []
        if self._tools is not None:
            try:
                agent_id = self._config.name
                division_id = self._config.division or ""
                tool_config = self._config.tools
                tool_schemas_dict = self._tools.get_tools_for_agent(agent_id, division_id, tool_config)
                tool_schemas = list(tool_schemas_dict.values())
            except Exception:
                pass

        request_messages = await self._ctx.build_messages(session.messages, tool_schemas)

        try:
            response_message = await self._llm.stream_complete(
                messages=request_messages,
                tools=tool_schemas if tool_call_mode != "text" else None,
                on_token=on_token,
            )
        except Exception as exc:
            session.terminated_reason = "error"
            return StepResult(action="error", error=str(exc))

        content = getattr(response_message, "content", None)
        raw_tool_calls = getattr(response_message, "tool_calls", None)
        session.push({
            "role": "assistant",
            "content": content,
            "tool_calls": raw_tool_calls,
        })

        tool_calls = _extract_tool_calls(response_message, tool_call_mode)
        if not tool_calls:
            session.terminated_reason = "complete"
            return StepResult(
                action="complete",
                llm_response_preview=(content or "")[:200],
            )

        executed = 0
        for tool_call in tool_calls:
            session.actions_taken += 1
            perm = self._permissions.evaluate(tool_call, self._config.permissions)
            if perm.denied:
                session.push({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Permission denied: {perm.reason}",
                })
                executed += 1
                continue
            result_content = ""
            if self._tools is not None:
                try:
                    result = await self._tools.dispatch(
                        tool_call.name,
                        tool_call.arguments,
                        self._config.name,
                        self._config.division or "",
                        self._config.tools,
                    )
                    result_content = result.output
                except Exception as exc:
                    result_content = f"Error: {exc}"
            session.push({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_content,
            })
            executed += 1

        return StepResult(
            action="tool_calls",
            tool_calls_executed=executed,
            llm_response_preview=(content or "")[:200],
        )

    async def abort(self, session: Session, reason: str) -> None:
        """Force stop the session."""
        from localharness.core.events import TurnFailed
        session.terminated_reason = "error"
        await self._bus.publish(TurnFailed(
            agent_id=session.agent_id,
            session_id=session.session_id,
            reason="internal_error",
            detail=reason,
            iterations=session.iteration,
            duration_seconds=session.elapsed_seconds(),
        ))
