"""AgentLoop, Session, StuckDetector, BudgetTracker, KillWatcher, StepResult.

ReAct while-loop execution engine for LocalHarness agents.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from datetime import datetime
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

from localharness.core.types import Message
from localharness.tools.capabilities import CoResidenceError

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
    parse_retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_estimated: bool = False
    act_nudge_used: bool = False
    # #84/FIX-4: baton-gate nudges spent this turn (announced-next-step reply), bounded by
    # config.baton_gate.max_nudges. Was a bool (one nudge, hardcoded); now a counter so the
    # bound is configurable while the default (max_nudges=1) reproduces the original behavior.
    baton_nudges_used: int = 0
    truncated_tool_calls: int = 0  # #77: tool calls suppressed for output-ceiling truncation

    @property
    def baton_nudge_used(self) -> bool:
        """Back-compat read-only alias: True once at least one baton-gate nudge has fired."""
        return self.baton_nudges_used > 0

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
        max_nudges_per_turn: int = 3,
    ) -> None:
        self.window_size = window_size
        self.recovery_threshold = recovery_threshold
        self.escalation_threshold = escalation_threshold
        self.max_nudges_per_turn = max_nudges_per_turn
        self._window: deque[str] = deque(maxlen=window_size)
        # Per-turn ladder state (the detector is constructed fresh per turn): which
        # signatures we have already warned about, and how many warnings we have spent.
        self._warned: dict[str, int] = {}
        self._total_warnings = 0

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

    def classify(self) -> tuple[StuckState, str]:
        """Deterministic per-turn stuck ladder. Returns (action, most_repeated_signature).

        Wraps the raw window read (check) with clean-slate-after-warning bookkeeping:
        - fresh RECOVERING (a not-yet-warned signature under the nudge cap): warn once,
          CLEAR the window, and record the warning → returns RECOVERING;
        - a repeat of an ALREADY-warned signature (the model re-hit 2 identical even after
          the clean slate — the warning demonstrably failed) → ESCALATE;
        - RECOVERING once max_nudges_per_turn warnings are spent (varied flailing) → ESCALATE;
        - a raw in-window ESCALATE (non-default thresholds) passes straight through;
        - CLEAR passes through.

        One call == one decision (it mutates on a warn), so the loop calls it exactly once
        per iteration after recording that iteration's tool calls.
        """
        raw = self.check()
        sig = self.most_repeated_signature()
        if raw != StuckState.RECOVERING:
            return raw, sig
        if sig in self._warned or self._total_warnings >= self.max_nudges_per_turn:
            return StuckState.ESCALATE, sig
        self._warned[sig] = self._warned.get(sig, 0) + 1
        self._total_warnings += 1
        self._window.clear()
        return StuckState.RECOVERING, sig

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
    # Dispatch-time tool-call ceiling, DISTINCT from max_actions (BudgetConfig.max_actions is
    # ge=1 so the loop always gets its first LLM round-trip). None (default): no separate cap,
    # every existing caller is unaffected — only the bench runner sets this, from
    # scenario.limits.max_tool_calls, which — unlike max_actions — may legitimately be 0.
    max_tool_calls: int | None = None

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

    def tool_call_allowed(self, session: Session) -> bool:
        """Whether one more tool call may be DISPATCHED right now.

        Independent of check()'s max_actions gate (which only stops the NEXT iteration from
        starting): max_tool_calls=0 must refuse every dispatch while the turn keeps running,
        so the model still converges to a normal answer instead of dying budget_exceeded
        before it can use — or do without — a tool result.
        """
        if self.max_tool_calls is None:
            return True
        return session.actions_taken < self.max_tool_calls


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

def _is_confirmation(text: str | None) -> bool:
    """Bare self-check sentinel ('CONFIRMED') — means 'answer above stands', never content."""
    import re
    return bool(text) and bool(re.fullmatch(r"confirmed[.!]?", text.strip(), re.IGNORECASE))


def _last_assistant_content(messages: list[Message]) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content") and not _is_confirmation(m["content"]):
            return m["content"]
    return ""


def _format_completion_summary(session: Session, content: str | None) -> str:
    if _is_confirmation(content):
        content = None  # sentinel: surface the answer it confirmed, not the sentinel
    return content or _last_assistant_content(session.messages) or "Task complete."


# --- Baton gate (issue #84): a tool-less reply whose CLOSING move announces further work -----
# The final answer at the acceptance seam is taken verbatim; the only guard, the act-guard, arms
# ONLY at zero actions. So a turn that DID work and then ends with "Now let me read X…" drops the
# baton — an intention shipped as the result. This detector flags exactly that closing move so the
# gate can nudge once. HIGH PRECISION: it looks at the FINAL sentence only and start-anchors an
# announce opener, so a false positive (a wasted round-trip on a good turn) stays rare.
_BATON_ANNOUNCE_RE = re.compile(
    r"(?:"
    r"now\s*,?\s+let\s+me"          # now let me / now, let me
    r"|let\s+me\s+now"              # let me now
    r"|now\s+i\s*['’]?ll"           # now I'll
    r"|now\s+i\s+will"              # now I will
    r"|i\s*['’]?ll\s+now"           # I'll now
    r"|i\s+will\s+now"              # I will now
    r"|next\s*,?\s+i\s*['’]?ll"     # next I'll / next, I'll
    r"|next\s*,?\s+i\s+will"        # next I will / next, I will
    # Live-observed drops the now/next anchors missed (Gemma-4-E2B REPL, 2026-07-16: six
    # accepted finals like "I will search for Korean market holidays…", "I am executing the
    # new, specific search now.", "Please wait a moment for the search results."). Same
    # precision contract as the anchors above: final sentence, start-anchored, and a TIGHT
    # action-verb whitelist so statements of fact ("I am done", "I am confident…") and
    # idioms stay out. Deliberate misses, precision over recall: "finding" (…this confusing),
    # "working on" (…the assumption that), bare "please wait for" (instructions to the USER
    # legitimately say that); "running" carries a lookahead for out/low/late/behind.
    r"|i\s+(?:will|am\s+going\s+to)\s+(?:search|check|look|read|fetch|find|pull|execute|run|query|verify|investigate|retrieve)\b"
    r"|i\s+am\s+(?:now\s+)?(?:searching|checking|looking|reading|fetching|pulling|executing|querying|verifying|investigating|retrieving|about\s+to)\b"
    r"|i\s+am\s+(?:now\s+)?running(?!\s+(?:out|low|late|behind)\b)"
    r"|please\s+wait\s+(?:a\s+moment|while\s+i)\b"
    r"|one\s+moment\s+(?:please|while\s+i)\b"
    r")\b",
    re.IGNORECASE,
)
_BATON_SENTENCE_SPLIT_RE = re.compile(r"[.!?\n]+")

_BATON_NUDGE_MESSAGE = (
    "Your reply ends by announcing further work instead of completing it. "
    "Do that work now, or state your final answer."
)


def detect_dropped_baton(content: str) -> bool:
    """True when a tool-less reply's CLOSING move is a first-person announced next-step
    ("Now let me read X", "Next I'll check Y") rather than the work itself or a final answer —
    the 'dropped baton' (issue #84).

    High-precision by design: evaluates only the FINAL sentence, start-anchored to an announce
    opener. So a mid-reply announce followed by real content, a closing courtesy ('let me know
    if…'), a handback question ('Should I proceed?'), and 'I now understand…' (not an announce)
    all return False — a false positive costs a wasted round-trip on an otherwise good turn.
    Pure text, no I/O. Only meaningful on the no-tool-calls branch (its only caller)."""
    text = (content or "").strip()
    if not text or text.endswith("?"):
        return False  # empty, or ends by asking the user (a legitimate handback)
    # Closing move = the last non-empty sentence/line; strip any leading bullet/quote chars.
    segments = [s.strip() for s in _BATON_SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not segments:
        return False
    closing = re.sub(r"^\W+", "", segments[-1])
    return bool(_BATON_ANNOUNCE_RE.match(closing))


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


def _budget_note(session: Session, budget: BudgetTracker) -> str:
    """One-line budget status appended to each iteration's last tool result so the
    model can pace itself and summarize BEFORE the wall (Claude Code-style running
    usage warnings). Empty string when both budgets are unlimited."""
    parts = []
    nearly = False
    if budget.max_actions > 0:
        parts.append(f"{session.actions_taken}/{budget.max_actions} tool calls used")
        nearly = nearly or (budget.max_actions - session.actions_taken) <= 2
    if budget.max_duration_minutes > 0:
        elapsed = session.elapsed_minutes()
        parts.append(f"{elapsed:.1f}/{budget.max_duration_minutes:.0f} min elapsed")
        nearly = nearly or (budget.max_duration_minutes - elapsed) <= max(
            1.0, 0.2 * budget.max_duration_minutes
        )
    if not parts:
        return ""
    note = f"\n\n[budget: {', '.join(parts)}]"
    if nearly:
        note = (note[:-1] + " — wrap up NOW: give your final summary in your next reply,"
                            " before the limit cuts you off]")
    return note


def _clean_summary(text: str) -> str:
    """Remove XML tool call remnants and thinking tags from summary text."""
    import re
    from localharness.provider.fn_call import strip_thinking_tags
    text = strip_thinking_tags(text)
    # Remove any leftover <tool_call>...</tool_call> blocks
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    # Remove orphaned opening <tool_call> blocks (truncated)
    text = re.sub(r"<tool_call>.*", "", text, flags=re.DOTALL)
    return text.strip()


def _repair_json_object(s: str) -> str | None:
    """Coerce a truncated JSON object string (generation cut mid-arguments) into
    parseable JSON: drop a dangling escape, close an open string, close open
    braces/brackets. Returns parseable JSON or None if unrepairable."""
    try:
        json.loads(s)
        return s
    except (json.JSONDecodeError, TypeError):
        pass
    if not isinstance(s, str):
        return None
    t = s.strip()
    if t.endswith("\\") and not t.endswith("\\\\"):
        t = t[:-1]
    in_str = False
    esc = False
    stack: list[str] = []
    for ch in t:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    if in_str:
        t += '"'
    t += "".join(reversed(stack))
    try:
        json.loads(t)
        return t
    except json.JSONDecodeError:
        return None


def _sanitize_raw_tool_calls(raw: Any) -> Any:
    """Repair or drop tool calls whose arguments JSON is malformed BEFORE they enter
    session history. vLLM's chat template json-parses historical tool_call arguments,
    so one truncated generation otherwise 400s every subsequent request (observed:
    finish-mid-string -> 'Unterminated string' BadRequest -> turn death). Repaired
    arguments are re-serialized normalized; unrepairable calls are dropped. Returns
    the cleaned list, or None if nothing survives (preserves no-tool-call semantics)."""
    if not raw:
        return raw
    clean = []
    for tc in raw:
        fn = getattr(tc, "function", None) if not isinstance(tc, dict) else tc.get("function", {})
        if fn is None:
            continue
        args = (getattr(fn, "arguments", None) if not isinstance(fn, dict)
                else fn.get("arguments")) or "{}"
        repaired = _repair_json_object(args)
        if repaired is None:
            name = (getattr(fn, "name", "?") if not isinstance(fn, dict) else fn.get("name", "?"))
            log.warning("dropping tool call '%s': arguments JSON unrepairable (%d chars)",
                        name, len(args))
            continue
        if repaired is not args:
            normalized = json.dumps(json.loads(repaired))
            if isinstance(fn, dict):
                fn["arguments"] = normalized
            else:
                fn.arguments = normalized
            name = (getattr(fn, "name", "?") if not isinstance(fn, dict) else fn.get("name", "?"))
            log.warning("repaired truncated arguments JSON for tool call '%s'", name)
        clean.append(tc)
    return clean or None


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


# #77: fed back (tool-role) when a completion is cut at the output-token ceiling
# mid-tool-call. Names the cause AND the remedy so the model retries INFORMED instead of
# re-emitting the identical truncated call (the live 2M-token turn: same write retried 4×,
# stuck detector firing on the symptom).
_TRUNCATED_TOOL_CALL_FEEDBACK = (
    "Your last response was cut off by the output-token limit mid-tool-call; nothing was "
    "executed. Produce the file in smaller pieces (multiple write/edit calls) or shorten "
    "the arguments."
)


def _assemble_role(cfg) -> str:
    """Assemble the agent system prompt from `role` + optional orthogonal sections (MODP-01).

    Byte-identity invariant (ROADMAP success criterion 4): when every section is "" (the
    default), this returns `cfg.role` UNCHANGED — the SAME object, no extra whitespace, no
    headers, no joins. The only behavioral delta is opt-in: it appears solely when an
    experiment overlay populates a section. Populated sections are appended in the fixed
    order base -> identity -> tool_use -> stopping -> output, joined with "\n\n" (matching
    the memory-block join idiom below at :468) — but that path is never taken in the
    unmutated baseline, so it cannot affect criterion 4.
    """
    base = cfg.role
    sec = cfg.role_sections
    extra = [s for s in (sec.identity, sec.tool_use, sec.stopping, sec.output) if s]
    if not extra:
        return base                       # byte-identity: same object, zero mutation
    return "\n\n".join([base, *extra])


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
        session_id: str | None = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._bus = bus
        self._ctx = context_manager
        self._tools = tool_registry
        self._permissions = permission_evaluator
        self._memory = memory_loader
        self._compact_md_path = compact_md_path
        # Type-anytime input box: user-typed nudges routed to the CURRENT turn land here and
        # are drained into durable session history at the next step boundary (same #82 seam as
        # the stuck-recovery nudge). A one-loop, cooperative-async inbox — the REPL coroutine
        # appends, the turn coroutine drains; no lock needed. Distinct from the stuck detector.
        self._user_nudge_inbox: list[str] = []
        # Prior-session context (compact.md) is FOLDED into the single leading system message,
        # never appended as a second system message — strict chat templates reject a non-leading
        # system role (vLLM/Qwen: "System message must be at the beginning"; the SEMA-05 P0).
        # Loaded once on the first turn of a sitting, reused across that sitting's turns.
        self._prior_session_context = ""
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
        # SESS-01: one session_id per SITTING when supplied at construction. None keeps
        # the legacy per-turn uuid fallback (bench/subagent callers). Set current_session_id
        # eagerly so it is valid BEFORE the first run_turn (repl reads it to stamp the
        # turn-1 UserMessage, previously None).
        self._sitting_session_id = session_id
        self._current_session_id: str | None = session_id
        self._conversation: list[Message] = []

    @property
    def current_session_id(self) -> str | None:
        """Return the session_id from the most recent run_turn() call."""
        return self._current_session_id

    def push_user_nudge(self, text: str) -> None:
        """Queue a user-typed nudge for delivery to the running turn at its next step
        boundary (type-anytime input box). Session-persisted via the same seam as the #82
        stuck-recovery nudge, but a DISTINCT source: it never touches the stuck detector or
        its max_nudges_per_turn accounting. Multiple nudges stack (FIFO). No-op on blank."""
        text = (text or "").strip()
        if text:
            self._user_nudge_inbox.append(text)

    async def run_turn(
        self,
        task: str,
        initial_messages: list[Message] | None = None,
        on_token: Callable | None = None,
    ) -> str:
        """Execute a full agent turn. Never raises — all errors become summary strings."""
        from localharness.core.events import TurnStarted, TurnCompleted, TurnFailed, BudgetSpec

        # Session continuity: reuse prior conversation if available
        if self._conversation:
            prior = list(self._conversation)
        elif initial_messages:
            prior = list(initial_messages)
        else:
            prior = []

        session = Session(
            agent_id=self._config.name,
            session_id=self._sitting_session_id or str(uuid.uuid4()),
            messages=prior,
        )
        self._current_session_id = session.session_id

        # Load prior-session context from compact.md if no conversation history. It is STASHED
        # (not inserted as a message) and folded into the single leading system message by
        # _execute_loop — a second system message is rejected by strict chat templates
        # (vLLM/Qwen: "System message must be at the beginning"; the SEMA-05 P0, 59/59 dead turns).
        if not prior:
            from localharness.agent.context import load_compact_md
            compact_path = self._compact_md_path or (Path.home() / ".localharness" / "agents" / self._config.name / "compact.md")
            compact_msg = load_compact_md(compact_path)
            if compact_msg is not None:
                self._prior_session_context = compact_msg["content"]
                log.info("Loaded compact.md for agent %s", self._config.name)

        budget_cfg = self._config.permissions.budget
        await self._bus.publish(TurnStarted(
            agent_id=session.agent_id,
            session_id=session.session_id,
            task_summary=task[:200],
            budget=BudgetSpec(
                max_actions=budget_cfg.max_actions,
                max_duration_minutes=budget_cfg.max_duration_minutes,
                max_context_tokens=self._config.context.max_context_tokens,
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
                input_tokens=session.input_tokens,
                output_tokens=session.output_tokens,
                tokens_estimated=session.tokens_estimated,
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
                input_tokens=session.input_tokens,
                output_tokens=session.output_tokens,
                tokens_estimated=session.tokens_estimated,
            ))
        elif reason == "stuck":
            await self._bus.publish(TurnFailed(
                agent_id=session.agent_id,
                session_id=session.session_id,
                reason="stuck_detected",
                detail=summary,
                iterations=session.iteration,
                duration_seconds=session.elapsed_seconds(),
                input_tokens=session.input_tokens,
                output_tokens=session.output_tokens,
                tokens_estimated=session.tokens_estimated,
            ))
        elif reason == "kill_file":
            await self._bus.publish(TurnFailed(
                agent_id=session.agent_id,
                session_id=session.session_id,
                reason="kill_file",
                detail=summary,
                iterations=session.iteration,
                duration_seconds=session.elapsed_seconds(),
                input_tokens=session.input_tokens,
                output_tokens=session.output_tokens,
                tokens_estimated=session.tokens_estimated,
            ))
        elif reason == "error":
            await self._bus.publish(TurnFailed(
                agent_id=session.agent_id,
                session_id=session.session_id,
                reason="llm_error",
                detail=summary,
                iterations=session.iteration,
                duration_seconds=session.elapsed_seconds(),
                input_tokens=session.input_tokens,
                output_tokens=session.output_tokens,
                tokens_estimated=session.tokens_estimated,
            ))
        else:
            await self._bus.publish(TurnCompleted(
                agent_id=session.agent_id,
                session_id=session.session_id,
                iterations=session.iteration,
                duration_seconds=session.elapsed_seconds(),
                elapsed_tokens=session.input_tokens + session.output_tokens,
                input_tokens=session.input_tokens,
                output_tokens=session.output_tokens,
                tokens_estimated=session.tokens_estimated,
                summary=summary,
            ))
        return summary

    async def _execute_loop(self, session: Session, task: str, on_token: Callable | None) -> str:
        from localharness.provider.client import (
            ProviderConnectionError,
            ProviderTimeoutError,
            ProviderAPIError,
        )
        from localharness.core.events import Action, Observation, Escalation, Heartbeat, TaskComplete, ParseFailed, StuckRecovered

        budget = BudgetTracker(
            max_actions=self._config.permissions.budget.max_actions,
            max_duration_minutes=self._config.permissions.budget.max_duration_minutes,
            max_tool_calls=self._config.permissions.budget.max_tool_calls,
        )
        sd_cfg = self._config.stuck_detector
        stuck_detector = StuckDetector(
            window_size=sd_cfg.window_size,
            recovery_threshold=sd_cfg.recovery_threshold,
            escalation_threshold=sd_cfg.escalation_threshold,
            max_nudges_per_turn=sd_cfg.max_nudges_per_turn,
        )
        sc_cfg = self._config.self_check
        self_check_passes_used = 0
        bg_cfg = self._config.baton_gate

        # Build system prompt
        tool_call_mode = getattr(
            getattr(self._llm, "config", None), "tool_call_mode", "native"
        )
        system_prompt = _assemble_role(self._config)
        # Date only (no clock time) so the vLLM prefix cache churns daily, not per-turn.
        _now = datetime.now().astimezone()
        system_prompt += f"\n\nToday's date: {_now.strftime('%A, %Y-%m-%d')} ({_now.tzname()})"
        # Working directory + placement rule (#75). Without it the model invents paths under
        # $HOME instead of the project dir it was launched from ("make a folder for yourself"
        # -> wrong dir). Same deterministic-fact pattern as the date; cwd is stable across the
        # process, so it doesn't churn the prefix cache per-turn.
        system_prompt += (
            f"\n\nWorking directory: {Path.cwd()}"
            "\nUnless the user names another location, create any files or folders you make "
            "under this working directory."
        )
        # Narration nudge (belt-and-suspenders — the terminal render is the mechanism): a
        # one-line stage announcement keeps a long multi-step task legible as it runs.
        system_prompt += (
            "\n\nWhen you start a distinct phase of a multi-step task, first state what you are "
            'about to do in one short line (e.g. "Pulling the data…") before making the tool calls.'
        )
        if tool_call_mode != "native":
            system_prompt += (
                "\n\nWhen you have finished using tools, respond directly to the user. "
                "Be concise — give the answer, not your reasoning process."
            )
        if self._memory is not None:
            try:
                # Default provenance for this session's writes (WRITE-04).
                _set_sess = getattr(self._memory, "set_current_session", None)
                if _set_sess is not None:
                    _set_sess(session.session_id)
                _mem_cfg = getattr(self._config, "memory", None)
                ctx = await self._memory.load_context(
                    index_mode=getattr(_mem_cfg, "index_mode", True),
                    max_session_history=getattr(_mem_cfg, "max_session_history_entries", 8),
                )
                parts = [system_prompt]
                if ctx.guardrails_md:
                    parts.append("## Guardrails\n" + ctx.guardrails_md)
                if ctx.division_md:
                    parts.append("## Division Context\n" + ctx.division_md)
                if ctx.agent_memory_md:
                    parts.append("## Agent Memory\n" + ctx.agent_memory_md)
                system_prompt = "\n\n".join(parts)
            except Exception:
                # Non-fatal by design, but never silent (live test 2026-07-03: a
                # swallowed failure here means the agent runs amnesiac all session
                # and nothing anywhere says so).
                log.warning("memory context load failed — no memory injected this turn", exc_info=True)

        # Get tool schemas
        tool_schemas: list = []
        if self._tools is not None:
            try:
                agent_id = self._config.name
                division_id = self._config.division or ""
                tool_config = self._config.tools
                tool_schemas_dict = self._tools.get_tools_for_agent(agent_id, division_id, tool_config)
                tool_schemas = list(tool_schemas_dict.values())
            except CoResidenceError:
                raise  # capability floor must fail LOUD, never be swallowed into an empty toolset
            except Exception:
                tool_schemas = []

        # Initialize or continue session messages. Prior-session context (compact.md) is FOLDED
        # into this ONE system message's content — never a second system message (strict chat
        # templates reject a non-leading system role; the SEMA-05 P0). Folded every turn so it
        # survives the continuing-conversation refresh below.
        if self._prior_session_context:
            system_prompt = f"{system_prompt}\n\n{self._prior_session_context}"
        has_prior_turns = any(m.get("role") == "user" for m in session.messages)
        if has_prior_turns and session.messages and session.messages[0].get("role") == "system":
            # Continuing conversation — refresh system prompt, append new user message
            session.messages[0] = {"role": "system", "content": system_prompt}
        else:
            # First turn — insert the single leading system message at front
            session.messages.insert(0, {"role": "system", "content": system_prompt})
        session.push({"role": "user", "content": task})

        # MOVE 0c: a fresh turn earns fresh compaction attempts — re-arm the per-turn fire cap
        # so a prior turn's cap never suppresses this turn's compaction.
        if hasattr(self._ctx, "reset_compaction_guard"):
            self._ctx.reset_compaction_guard()

        while True:
            session.iteration += 1
            # FIX 4: wire the real per-turn iteration into the ContextManager so the emergency-floor
            # log and CompactionTriggered events carry it (production never called set_iteration).
            if hasattr(self._ctx, "set_iteration"):
                self._ctx.set_iteration(session.iteration)
            log.debug(
                "agent=%s iter=%d actions=%d elapsed=%.1fs",
                self._config.name,
                session.iteration,
                session.actions_taken,
                session.elapsed_seconds(),
            )

            # 0. Type-anytime user nudges: drain the inbox at this step boundary into durable
            # session history (#82 seam), so this iteration's build_messages replays them. A
            # distinct source from the stuck detector — deliberately outside its accounting.
            while self._user_nudge_inbox:
                nudge = self._user_nudge_inbox.pop(0)
                session.push({"role": "user", "content": nudge})
                log.info(
                    "User nudge delivered to %s at iteration %d",
                    self._config.name, session.iteration,
                )

            # 1. Kill check
            if self._kill.is_killed():
                session.terminated_reason = "kill_file"
                self._conversation = list(session.messages)
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
                summary = await self._final_summary_on_budget(session, violation, on_token)
                self._conversation = list(session.messages)
                return summary

            # 3. Build request messages first (runs compaction if needed)
            request_messages, ctx_budget = await self._ctx.build_messages(session.messages, tool_schemas)

            # SESS-03: a compaction summary must outlive the window. CompactionTriggered
            # cannot fire live (production ContextManager has no bus — start_cmd gap, noted
            # for the owner, NOT fixed here); the summary is only observable in the returned
            # request_messages. Rolling per-sitting node: supersede absorbs re-fires.
            if self._memory is not None:
                _marker = "[Context Summary]\n"
                for _m in request_messages:
                    _c = _m.get("content") or ""
                    if _m.get("role") == "assistant" and _c.startswith(_marker):
                        try:
                            from localharness.memory.hierarchy import persist_compaction_gist
                            await persist_compaction_gist(
                                self._memory, summary=_c[len(_marker):],
                                session_id=session.session_id,
                            )
                        except Exception:
                            log.warning("compaction-gist persistence failed (non-fatal)", exc_info=True)
                        break  # one summary message per build; stop at the first

            # 4. Publish heartbeat AFTER build_messages so utilization reflects post-compaction state (TELEM-01)
            raw_pct = ctx_budget.usage_fraction * 100.0
            util_pct = min(100.0, raw_pct)
            if raw_pct > 100.0:
                log.warning(
                    "context_utilization_pct overflow (%.1f%%) — compaction may be failing",
                    raw_pct,
                )
            await self._bus.publish(Heartbeat(
                agent_id=session.agent_id,
                session_id=session.session_id,
                iteration=session.iteration,
                context_utilization_pct=util_pct,
                last_tool=None,
            ))

            # 5. LLM call with error handling
            # (The stuck-recovery nudge is no longer injected here transiently — it is pushed
            # straight into session.messages at RECOVERING time [#82], so it is durable history
            # and every subsequent build_messages replays it, not just the next one.)
            tool_call_mode = getattr(
                getattr(self._llm, "config", None), "tool_call_mode", "native"
            )
            try:
                response_message, usage = await self._llm.stream_complete(
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
                    response_message, usage = await self._llm.stream_complete(
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
                        "HTTP 400 from LLM in %s — server error: %s. Request messages: %s",
                        self._config.name,
                        str(exc)[:300],
                        json.dumps(request_messages, default=str),
                    )
                session.terminated_reason = "error"
                return _format_error_summary(session, exc)

            # Accumulate per-turn token usage (TELEM-02)
            if usage is not None:
                session.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                session.output_tokens += getattr(usage, "completion_tokens", 0) or 0
            else:
                # Provider omitted usage — fall back to tiktoken estimate
                est_in = self._ctx._token_counter.count_messages(request_messages)
                est_out = self._ctx._token_counter.count(
                    getattr(response_message, "content", "") or ""
                )
                session.input_tokens += est_in
                session.output_tokens += est_out
                if not session.tokens_estimated:
                    log.warning(
                        "Provider returned no usage for iter %d — tiktoken fallback engaged",
                        session.iteration,
                    )
                session.tokens_estimated = True

            # 6. Strip thinking tags and push assistant response to session
            from localharness.provider.fn_call import (
                strip_thinking_tags,
                has_tool_call_attempt,
                truncate_after_last_tool_call,
            )
            raw_content = getattr(response_message, "content", None)
            # "length" == the completion hit the output-token ceiling (#77); any tool call
            # parsed below is truncated and must not execute (guard at step 8c).
            finish_reason = getattr(response_message, "finish_reason", None)
            # Reasoning-parser tool turns (--reasoning-parser) legitimately return
            # content=None — every token went to reasoning + tool_calls. None must never
            # enter history: each later request replays the entry and vLLM's request
            # validation rejects it ('content.str: Input should be a valid string' ->
            # HTTP 400), poisoning the rest of the session. Normalize to "" here (and
            # symmetrically at the build_messages egress); tool_calls stay untouched.
            content = strip_thinking_tags(raw_content) if raw_content else ""
            raw_tool_calls = _sanitize_raw_tool_calls(getattr(response_message, "tool_calls", None))
            try:
                response_message.tool_calls = raw_tool_calls  # extraction must match history
            except Exception:
                pass
            session.push({
                "role": "assistant",
                "content": content,
                "tool_calls": raw_tool_calls,
            })

            # 7. Extract tool calls FIRST, so the Action can carry has_tool_calls — the
            # terminal's discriminator between interstitial narration (content arrives WITH
            # tool calls) and a final answer (content alone, rendered via TaskComplete).
            # Mode-correct: native reads response_message.tool_calls, xml/text parses content.
            tool_calls = _extract_tool_calls(response_message, tool_call_mode)

            # 8. Publish Action event
            await self._bus.publish(Action(
                agent_id=session.agent_id,
                session_id=session.session_id,
                action_type="llm_response",
                content=content,
                has_tool_calls=bool(tool_calls),
            ))

            # 8b. In xml mode, populate assistant message tool_calls so
            # repair_tool_pairing doesn't strip tool results as orphaned
            if tool_call_mode != "native" and tool_calls:
                tc_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in tool_calls
                ]
                for i in range(len(session.messages) - 1, -1, -1):
                    if session.messages[i].get("role") == "assistant":
                        session.messages[i]["tool_calls"] = tc_dicts
                        # Cut prose trailing the last call block from the HISTORY copy only
                        # (the Action event above already carries the full text). Anything the
                        # model wrote after the call is pre-result speculation; replaying it
                        # lets the model trust its own fabrication over the real tool result
                        # next turn (observed live with Gemma-3-4B: invented file contents
                        # outweighed the actual read result in half of bench runs).
                        session.messages[i]["content"] = truncate_after_last_tool_call(
                            session.messages[i].get("content") or ""
                        )
                        break

            # 8c. Output-ceiling truncation guard (#77). A completion cut at the token limit
            # (finish_reason="length") mid-tool-call yields a call that MUST NOT run: arg-repair
            # (_sanitize_raw_tool_calls) can forge plausible-but-wrong JSON — a silently
            # truncated `write` (a README that ends mid-table while the turn reports "done") —
            # or drop a required field, giving "path: Field required" that the model then
            # retries byte-identically (the live 2M-token turn: 4 blind retries, ~60-70s dead
            # air + full context resend each). Suppress execution, answer each parsed call with
            # a tool-role remedy (naming cause + fix — keeps native tool_call/result pairing
            # valid, and the xml back-fill above gave xml calls matching ids), count it, and
            # continue so the model retries INFORMED. Same guard for both modes: xml truncates
            # the embedded call text, but finish_reason rides the same stream chunk.
            if finish_reason == "length" and tool_calls:
                session.truncated_tool_calls += 1
                log.warning(
                    "Output-token-ceiling truncation mid-tool-call for %s — suppressed %d "
                    "call(s), feeding remedy",
                    self._config.name, len(tool_calls),
                )
                for tool_call in tool_calls:
                    session.push({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": _TRUNCATED_TOOL_CALL_FEEDBACK,
                    })
                continue

            # 9. No tool calls — check if parse failed on an attempted tool call
            if not tool_calls:
                if has_tool_call_attempt(raw_content or "") and session.parse_retries < 3:
                    session.parse_retries += 1
                    await self._bus.publish(ParseFailed(
                        agent_id=session.agent_id,
                        session_id=session.session_id,
                        iteration=session.iteration,
                        parse_retry_count=session.parse_retries,
                        raw_content_preview=(raw_content or "")[:200],
                    ))
                    log.warning(
                        "Tool call XML parse failed (attempt %d/3) for %s",
                        session.parse_retries,
                        self._config.name,
                    )
                    session.push({
                        "role": "user",
                        "content": (
                            "Your tool call could not be parsed. Please use the correct format "
                            "and try again with your intended tool call:\n"
                            "<tool_call>\n"
                            "<name>tool_name</name>\n"
                            '<parameters>{"param_name": "value"}</parameters>\n'
                            "</tool_call>"
                        ),
                    })
                    continue

                # 9b. Act-guard: a first response that would END the turn with ZERO
                # actions taken is, empirically, usually announce-then-halt ("I'll
                # research X..." + stop) — sampling-dependent on small local models
                # (observed live: fail/succeed/fail across identical prompts). Give
                # it exactly one deterministic push before accepting a tool-less
                # completion; a genuine no-tool answer just gets repeated.
                # The no-tool fallback asks for the bare CONFIRMED sentinel —
                # _format_completion_summary surfaces the prior reply untouched
                # (issue #6: 'restate' invited meta-narrated duplicates on
                # conversational turns).
                if (session.actions_taken == 0 and not session.act_nudge_used
                        and tool_schemas):
                    session.act_nudge_used = True
                    log.info("Act-guard: tool-less first completion — nudging once")
                    session.push({"role": "user", "content": (
                        "You ended your reply with stated intentions but took no action. "
                        "Execute your plan NOW: make the tool call in this response. "
                        "If the task genuinely needs no tools, reply with exactly CONFIRMED — "
                        "your previous reply will be delivered to the user unchanged."
                    )})
                    continue

                # Natural completion — reset parse retries
                session.parse_retries = 0

                # Baton gate (#84): a tool-less reply whose CLOSING move announces further work
                # ("Now let me read X…") instead of doing it would otherwise be accepted verbatim
                # as the final answer — the act-guard only arms at zero actions, so an
                # announce-AFTER-work reply slips through. Nudge (persisted into history the same
                # way the stuck-recovery/self-check nudges are, so the model sees it), then
                # continue. Bounded by baton_gate.max_nudges per turn (default 1, i.e. the
                # original #84 behavior): once every nudge is spent, a further announced
                # intention is accepted (no loop). Runs BEFORE self_check so a genuinely-final
                # reply still gets its review.
                if (bg_cfg.enabled and session.baton_nudges_used < bg_cfg.max_nudges
                        and detect_dropped_baton(content)):
                    session.baton_nudges_used += 1
                    log.info(
                        "Baton gate: reply announces further work — nudging (%d/%d)",
                        session.baton_nudges_used, bg_cfg.max_nudges,
                    )
                    session.push({"role": "user", "content": _BATON_NUDGE_MESSAGE})
                    continue

                # Self-check (MECH-01): one bounded review pass before finalizing.
                # A loop-structure mechanism — re-enters the while-loop for one more
                # LLM round-trip (using the recovery-injection "append user turn +
                # continue" idiom, loop.py:548-551). Bounded by max_passes (ge=1,le=3)
                # so it provably terminates. Use the "user" role — vLLM rejects
                # mid-conversation system messages.
                if sc_cfg.enabled and self_check_passes_used < sc_cfg.max_passes:
                    self_check_passes_used += 1
                    session.push({"role": "user", "content": (
                        "Review your answer above for correctness and completeness. "
                        "If it is correct, reply with exactly CONFIRMED. If not, reply with "
                        "the corrected complete answer on its own — only your latest reply "
                        "is shown to the user, so never refer back to an earlier reply."
                    )})
                    continue

                summary = _clean_summary(
                    _format_completion_summary(session, content)
                )
                await self._bus.publish(TaskComplete(
                    agent_id=session.agent_id,
                    session_id=session.session_id,
                    success=True,
                    summary=summary,
                    duration_seconds=session.elapsed_seconds(),
                    iterations=session.iteration,
                ))
                session.terminated_reason = "complete"
                self._conversation = list(session.messages)
                return summary

            # 10. Deduplicate identical tool calls in same response
            seen_sigs: set[str] = set()
            unique_calls: list = []
            for tc in tool_calls:
                sig = f"{tc.name}:{json.dumps(tc.arguments, sort_keys=True, separators=(',', ':'))}"
                if sig in seen_sigs:
                    log.warning("Duplicate tool call %s in same response — skipping", tc.name)
                    continue
                seen_sigs.add(sig)
                unique_calls.append(tc)
            tool_calls = unique_calls

            # 11. Execute each tool call
            for tool_call in tool_calls:
                # Dispatch cap (FIX 3): checked BEFORE actions_taken increments or an Action
                # publishes, so a refusal is invisible to actions_taken/tool_call_count — it
                # must not trip max_actions and kill the turn, and metrics should reflect real
                # dispatches only. stuck_detector still sees it, so a model that keeps retrying
                # the same disallowed call is still caught by the stuck ladder.
                if not budget.tool_call_allowed(session):
                    session.push({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": (
                            f"Tool call not executed: this task's tool-call budget "
                            f"({budget.max_tool_calls} allowed) has been reached. Answer "
                            f"directly without calling any more tools."
                        ),
                    })
                    stuck_detector.record(tool_call.name, tool_call.arguments)
                    continue

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

                # Dispatch the tool call against the agent's registry.
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
                        is_error = not result.success
                        # Error results carry their message in .error with output "" —
                        # forward it or the model sees an empty result it can't react to.
                        result_content = (result.output if result.success
                                          else f"[tool error] {result.error or 'unknown error'}")
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

            # 11b. Append budget status to the iteration's last tool result
            note = _budget_note(session, budget)
            if note and session.messages and session.messages[-1].get("role") == "tool":
                session.messages[-1]["content"] = (session.messages[-1]["content"] or "") + note

            # Tool calls parsed and executed — reset parse retry counter
            session.parse_retries = 0

            # 12. Check stuck state (deterministic per-turn ladder: clean slate after a
            # warning, escalate only on fresh evidence — #81)
            stuck_state, repeated_sig = stuck_detector.classify()
            if stuck_state == StuckState.RECOVERING:
                await self._bus.publish(StuckRecovered(
                    agent_id=session.agent_id,
                    session_id=session.session_id,
                    iteration=session.iteration,
                    stuck_signature=repeated_sig or "",
                ))
                # Phase 14 REG-04: recovery wording is a mutable config component.
                # #82: push it into durable session history (right here, after this
                # iteration's tool results — the correct position) rather than a transient
                # request-only append, so it survives every later build_messages and the
                # history never shows the model replying to a message that isn't there.
                session.push({"role": "user", "content": self._config.recovery_injection.message})
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
                    stuck_signature=repeated_sig,
                    iteration_at_escalation=session.iteration,
                ))
                log.warning(
                    "Agent %s stuck after %d iterations, escalating",
                    self._config.name,
                    session.iteration,
                )
                # #83: return the model's partial work, not a dead notice. Telemetry
                # (terminated_reason, Escalation, TurnFailed) stays unchanged above.
                summary = await self._final_summary_on_stuck(
                    session, session.iteration, on_token
                )
                self._conversation = list(session.messages)
                return summary

    async def _final_summary_on_budget(
        self, session: Session, violation: BudgetViolation, on_token: Callable | None,
    ) -> str:
        """One forced no-tools generation so a budget-exhausted agent returns its
        FINDINGS instead of a task echo. Children previously died with only
        '[Budget limit reached]' after a full gather phase, so the parent received
        zero facts. Falls back to the plain budget notice on any provider error."""
        instruction = (
            f"[{violation.message}] You are out of tool budget — do NOT call any tools. "
            "In ONE reply, summarize everything you found relevant to the original task: "
            "concrete facts, names, numbers, and source URLs. Partial findings are valuable. "
            "If you found nothing, state in one line what you tried."
        )
        try:
            request_messages, _ = await self._ctx.build_messages(
                session.messages + [{"role": "user", "content": instruction}], None
            )
            response_message, usage = await self._llm.stream_complete(
                request_messages, tools=None, on_token=on_token,
            )
            if usage is not None:
                session.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                session.output_tokens += getattr(usage, "completion_tokens", 0) or 0
            from localharness.provider.fn_call import strip_thinking_tags
            text = strip_thinking_tags(getattr(response_message, "content", None) or "").strip()
            if not text:
                return _format_budget_summary(session, violation)
            session.push({"role": "user", "content": instruction})
            session.push({"role": "assistant", "content": text})
            return (
                f"{_clean_summary(text)}\n\n"
                f"[Budget limit reached: {violation.message} "
                f"Completed {session.actions_taken} tool calls in "
                f"{session.elapsed_minutes():.1f} minutes.]"
            )
        except Exception as exc:  # noqa: BLE001 — the summary pass must never kill the turn
            log.warning("final-summary-on-budget failed (%s); returning plain notice", exc)
            return _format_budget_summary(session, violation)

    async def _final_summary_on_stuck(
        self, session: Session, iterations: int, on_token: Callable | None,
    ) -> str:
        """One forced no-tools generation so a stuck-escalated agent returns its PARTIAL
        WORK instead of a dead '[Agent stuck: …]' notice (#83). Mirrors
        _final_summary_on_budget; falls back to the plain stuck notice on any provider
        error or empty text. Telemetry (terminated_reason, Escalation, TurnFailed) is
        unchanged — only the returned content improves."""
        instruction = (
            "[You have been stopped: you repeated the same tool call multiple times with "
            "identical arguments. Do NOT call any tools.] In ONE reply, wrap up honestly "
            "for the user: (1) what you completed so far, with concrete file paths or "
            "results; (2) what remains undone; (3) the single next step you would take. "
            "Partial work is valuable — report it plainly."
        )
        try:
            request_messages, _ = await self._ctx.build_messages(
                session.messages + [{"role": "user", "content": instruction}], None
            )
            response_message, usage = await self._llm.stream_complete(
                request_messages, tools=None, on_token=on_token,
            )
            if usage is not None:
                session.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                session.output_tokens += getattr(usage, "completion_tokens", 0) or 0
            from localharness.provider.fn_call import strip_thinking_tags
            text = strip_thinking_tags(getattr(response_message, "content", None) or "").strip()
            if not text:
                return _format_stuck_summary(session)
            session.push({"role": "user", "content": instruction})
            session.push({"role": "assistant", "content": text})
            return (
                f"{_clean_summary(text)}\n\n"
                f"[Agent stuck: repeated identical tool calls detected after {iterations} "
                f"iterations; stopped after recovery warnings failed. The summary above "
                f"reflects partial completion.]"
            )
        except Exception as exc:  # noqa: BLE001 — the summary pass must never kill the turn
            log.warning("final-summary-on-stuck failed (%s); returning plain notice", exc)
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
        if hasattr(self._ctx, "set_iteration"):  # FIX 4: carry the real iteration into ctx (step path)
            self._ctx.set_iteration(session.iteration)
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
            except CoResidenceError:
                raise  # capability floor must fail LOUD, never be swallowed into an empty toolset
            except Exception:
                pass

        request_messages, ctx_budget = await self._ctx.build_messages(session.messages, tool_schemas)

        try:
            response_message, usage = await self._llm.stream_complete(
                messages=request_messages,
                tools=tool_schemas if tool_call_mode != "text" else None,
                on_token=on_token,
            )
        except Exception as exc:
            session.terminated_reason = "error"
            return StepResult(action="error", error=str(exc))

        # Accumulate per-turn token usage (TELEM-02)
        if usage is not None:
            session.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            session.output_tokens += getattr(usage, "completion_tokens", 0) or 0
        else:
            est_in = self._ctx._token_counter.count_messages(request_messages)
            est_out = self._ctx._token_counter.count(
                getattr(response_message, "content", "") or ""
            )
            session.input_tokens += est_in
            session.output_tokens += est_out
            session.tokens_estimated = True

        content = getattr(response_message, "content", None)
        finish_reason = getattr(response_message, "finish_reason", None)
        raw_tool_calls = _sanitize_raw_tool_calls(getattr(response_message, "tool_calls", None))
        try:
            response_message.tool_calls = raw_tool_calls  # extraction must match history
        except Exception:
            pass
        session.push({
            "role": "assistant",
            "content": content,
            "tool_calls": raw_tool_calls,
        })

        tool_calls = _extract_tool_calls(response_message, tool_call_mode)

        # Populate assistant tool_calls for xml mode (repair_tool_pairing compatibility)
        if tool_call_mode != "native" and tool_calls:
            tc_dicts = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in tool_calls
            ]
            for i in range(len(session.messages) - 1, -1, -1):
                if session.messages[i].get("role") == "assistant":
                    session.messages[i]["tool_calls"] = tc_dicts
                    break

        # Output-ceiling truncation guard (#77) — same seam as _execute_loop's step 8c.
        if finish_reason == "length" and tool_calls:
            session.truncated_tool_calls += 1
            for tool_call in tool_calls:
                session.push({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": _TRUNCATED_TOOL_CALL_FEEDBACK,
                })
            return StepResult(
                action="error",
                error="output-token-ceiling truncation mid-tool-call; not executed",
                llm_response_preview=(content or "")[:200],
            )

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
