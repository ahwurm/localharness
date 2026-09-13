"""ChannelAdapter ABC — pluggable interface for all channel adapters."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from localharness.core.bus import EventBus
from localharness.core.events import (
    Action,
    CompactionTriggered,
    Escalation,
    Heartbeat,
    Observation,
    ParseFailed,
    TaskComplete,
    TurnFailed,
)


PERMISSION_DENIED_LINE = "permission denied — {tool_name}: {reason}"
"""The one line a human gets when the gate refuses a call (PRD §3.5, verification A defect D7).

One line per denial, not a block: denials arrive mid-turn while the model re-plans, and the
person needs the reason, not a report."""


PENDING_NOTICE_LINE = (
    "⏸ needs you  #{id}  {rendering}   ({total} pending · /approve {id} · /deny {id})"
)
"""The one line a human gets when `auto` PARKS a call instead of asking (owner ruling
2026-09-12).

One line, for the same reason `PERMISSION_DENIED_LINE` is one line: it arrives mid-turn while
the model carries on without the step, and the person needs the command and the two words that
answer it, not a report. Everything a reply needs is in the line — the number, the command, how
many are waiting, and both verbs spelled out — because the alternative is a notice that tells
somebody something is wrong and makes them go looking for how to fix it."""

KEPT_CONTROL_CHARS = "\t\n"
"""The two control characters a rendered question may keep: tab, and newline.

Newline is structural here — one tool call can raise several reasons and the gate renders them
as a short multi-line `display`, which the terminal prints as lines and the ACP adapter splits
into a title and a body. Every other C0 character is removed (see :func:`sanitize_for_display`).
"""

ESCAPE_SEQUENCE_RE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)?|[@-Z\\-_])?"
)
"""ANSI escape sequences: CSI (`\\x1b[…`), OSC (`\\x1b]…` up to BEL or ST), and the two-character
forms. Removed whole, so what is left of a cursor-moving sequence is not a stray `[2J` printed as
text."""

CONTROL_CHARS_RE = re.compile(
    "["
    + "".join(
        re.escape(chr(code))
        for code in (*range(0x00, 0x20), 0x7F)
        if chr(code) not in KEPT_CONTROL_CHARS
    )
    + "]"
)
"""Every C0 control and DEL except :data:`KEPT_CONTROL_CHARS` — derived from that set rather
than spelled out, so the two cannot drift apart."""


def sanitize_for_display(text: str) -> str:
    """Strip terminal control characters from text a channel is about to render.

    A permission question quotes the model's own words — a shell command, a path, an MCP tool's
    arguments — and those words reach a terminal that acts on control characters. `\\r` rewrites
    the line the human is reading, `\\x1b[2J` clears the screen, and either one can leave a
    question on screen that is not the question being answered. Both renderers of an ASK call
    this before their own escaping (rich markup escaping is about markup, not about the
    terminal), because the answer to a mangled question is a permission the human did not grant.

    Tab and newline survive (:data:`KEPT_CONTROL_CHARS`): they are layout, not control.
    """
    if not text:
        return text
    return CONTROL_CHARS_RE.sub("", ESCAPE_SEQUENCE_RE.sub("", text))


def permission_denied_reason(error: str | None) -> str | None:
    """The reason out of a denied observation's error text, or None if it is an ordinary error.

    Matches the label the loop writes (`agent/gate.DENIED_OBSERVATION_PREFIX`), imported here
    rather than repeated so the two cannot drift. The import is local: `agent/gate` reaches the
    loop and the tool registry, and `channels` must stay importable on its own.
    """
    if not error:
        return None
    from localharness.agent.gate import DENIED_OBSERVATION_PREFIX

    if not error.startswith(DENIED_OBSERVATION_PREFIX):
        return None
    return error[len(DENIED_OBSERVATION_PREFIX):].strip() or None


class ChannelAdapter(ABC):
    """
    Abstract base for all channel adapters.

    A channel adapter does two things:
      1. Receives output events from the event bus and presents them to the user.
      2. Accepts user input and publishes UserMessage events to the bus.

    Lifecycle:
      - __init__: inject bus and channel-specific config
      - start(): subscribe to bus events, begin accepting input
      - stop(): unsubscribe, flush pending output, release resources

    Threading model: all methods run in the asyncio event loop.
    Channel adapters must not block. Use asyncio primitives.

    Channels are identified by channel_id (str). The terminal channel
    uses channel_id="terminal". Future adapters use "discord", "slack", etc.
    """

    channel_id: str  # must be set as class attribute in subclasses

    can_ask: bool = False
    """Can this channel put a permission question to a human? (PRD §3.5.)

    False is the safe default and it is deliberately the BASE value: a new channel that says
    nothing is treated as unable to ask, so its ASK verdicts fail closed with a loud warning
    rather than silently running. Overridden to True by the terminal, Discord and (phase B) the
    ACP adapter.
    """

    ask_holds_dialog: bool = False
    """Does this channel hold the question open until a human answers it? (PRD §3.5.)

    True for a channel with a person in front of it — a terminal, or Zed's permission dialog —
    where PRD §3.5's table says "Timeout: none": the gate awaits the answer with no deadline,
    so stepping away from the keyboard does not silently turn into a denial. False for Discord
    and for anything message-shaped, where a question nobody reacts to has to expire; the gate
    then applies `permissions.ask.timeout_s` (or the tool-timeout derivation) and records a
    `reject_once`. False is the safe default: a new channel that says nothing gets the deadline.
    """

    has_review_surface: bool = False
    """Does an in-workspace edit land somewhere a human will see it? (PRD §3.1 choice 2.)

    True for the terminal (it prints the diff after the fact) and for an ACP client that
    advertises `fs` (Zed's diff pane, accept/reject per hunk). False everywhere else, which
    makes in-workspace edits ask ONCE per workspace instead of never — critic finding 11.
    """

    async def ask_permission(self, request: Any) -> Any:
        """Render one ASK verdict and return the human's `Decision` (PRD §3.5).

        `request` is a `PermissionRequest`; the return is a `Decision` whose kind is one of
        allow_once / allow_always / reject_once / reject_always — the four ACP option kinds every
        channel maps its UI onto. A channel that has not implemented this must also leave
        `can_ask` False; the gate never calls it, and raising here makes a mismatch between the
        two loud instead of silent.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot ask for permission; it must leave can_ask=False"
        )

    def __init__(self, bus: EventBus, config: dict[str, Any]) -> None:
        self.bus = bus
        self.config = config

    @abstractmethod
    async def start(self) -> None:
        """Subscribe to event bus topics and begin accepting user input."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Unsubscribe, flush buffered output, release resources. Safe to call multiple times."""
        ...

    @abstractmethod
    async def send_message(
        self,
        content: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Send a text message to the channel output."""
        ...

    @abstractmethod
    async def send_streaming(
        self,
        token_stream: AsyncIterator[str],
        agent_id: str | None = None,
    ) -> str:
        """Stream tokens to output as they arrive. Returns the complete assembled text."""
        ...

    @abstractmethod
    async def send_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        agent_id: str | None = None,
    ) -> None:
        """Display a tool call invocation."""
        ...

    @abstractmethod
    async def send_tool_result(
        self,
        tool_name: str,
        result: str,
        is_error: bool,
        agent_id: str | None = None,
    ) -> None:
        """Display the result of a tool call."""
        ...

    @abstractmethod
    async def send_error(
        self,
        error: str,
        detail: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Display an error to the user."""
        ...

    async def send_permission_denied(
        self,
        tool_name: str,
        reason: str,
        agent_id: str | None = None,
    ) -> None:
        """Tell the human why the gate refused a call — one line (PRD §3.5, defect D7).

        Default: through `send_error`, which every channel already renders. The terminal
        overrides it to keep the line inline with the tool lines it belongs to. A channel that
        genuinely has nowhere to put it can override with a no-op, but silence is the failure
        this exists to fix: the reason otherwise reaches only the model.
        """
        await self.send_error(
            error=PERMISSION_DENIED_LINE.format(tool_name=tool_name, reason=reason),
            agent_id=agent_id,
        )

    async def send_pending_notice(self, pending: Any, total: int) -> None:
        """Tell the human a call was parked for them — one line (:data:`PENDING_NOTICE_LINE`).

        `pending` is a `PendingCall` and `total` the size of the queue including it. Typed `Any`
        for the reason `ask_permission`'s request is: `channels` must stay importable without
        reaching into `agent`, and a renderer only ever touches `.id` and `.rendering`.

        Default: through `send_message`, so a channel that overrides nothing still says it. The
        terminal, Discord and ACP each have a better place to put it and override this; a channel
        with genuinely nowhere to put it may override with a no-op, but silence is the failure
        this exists to prevent — the model is told to carry on without the step, so if the notice
        does not land the human never learns a step was skipped at all.
        """
        await self.send_message(
            PENDING_NOTICE_LINE.format(
                id=pending.id, rendering=pending.rendering, total=total
            )
        )

    async def send_renderable(
        self,
        renderable: Any,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Present a pre-built rich renderable (e.g. the /memory tree). Default: render it to plain
        text and hand it to send_message, so channels without a rich console (Discord) degrade
        gracefully. TerminalChannel overrides this to print the renderable directly."""
        import io

        from rich.console import Console
        buf = io.StringIO()
        Console(file=buf, width=100).print(renderable)
        await self.send_message(buf.getvalue().rstrip("\n"), agent_id=agent_id, metadata=metadata)

    @abstractmethod
    async def read_input(self, prompt: str = "> ") -> str:
        """
        Read a line of input from the user.

        For interactive channels (terminal): blocks until user presses Enter.
        For non-interactive channels (file, webhook): raises NotInteractiveError.

        Raises:
            ChannelInputError: On read failure.
            NotInteractiveError: If this channel does not support user input.
            EOFError: On Ctrl-D or stream end.
        """
        ...

    # --- Default event handlers (subscribed via bus) ---

    async def on_action(self, event: Action) -> None:
        """Default handler for Action events. Calls send_tool_call for tool_call actions."""
        if event.action_type == "tool_call":
            await self.send_tool_call(
                tool_name=event.tool_name or "",
                arguments=event.tool_params or {},
                agent_id=event.agent_id,
            )

    async def on_observation(self, event: Observation) -> None:
        """Default handler for Observation events. Calls send_tool_result.

        A permission denial gets one extra line: its reason travels on `event.error`, and the
        tool-result line shows only `[DENIED]`, so without this the person watching is never
        told why a call was refused — a soft "not permitted in read-only mode" or a stored
        "never here" looked like a crash (verification A, defect D7).
        """
        result = event.output or event.error or ""
        is_error = event.error is not None
        await self.send_tool_result(
            tool_name=event.tool_name or "",
            result=result,
            is_error=is_error,
            agent_id=event.agent_id,
        )
        reason = permission_denied_reason(event.error)
        if reason is not None:
            await self.send_permission_denied(
                tool_name=event.tool_name or "", reason=reason, agent_id=event.agent_id
            )

    async def on_permission_staged(self, event: Any) -> None:
        """Default handler for PermissionStaged: draw the one-line notice.

        This event is the ONLY way a channel hears that a call was parked. The agent loop has no
        channel handle — it surfaces an ordinary denial by writing `DENIED_OBSERVATION_PREFIX`
        onto the Observation and letting `on_observation` match it — and a queue notice is not
        about one tool result, so it travels on its own event instead. A channel subscribes to it
        in its own `start()`, beside the Observation and Action subscriptions.
        """
        await self.send_pending_notice(event.pending, event.total)

    async def on_task_complete(self, event: TaskComplete) -> None:
        """Default handler for TaskComplete events. Sends the summary.

        Child-turn completions (parent_id stamped by _ParentIdBus) stay internal:
        a child's summary returns to the PARENT via the agent tool result — posting
        it to the channel reads as a premature, often contradictory 'final answer'
        (observed live: a subagent's failure apology landed in Discord minutes
        before the parent's actual answer)."""
        if getattr(event, "parent_id", None):
            return
        await self.send_message(
            content=event.summary,
            agent_id=event.agent_id,
        )

    async def on_turn_failed(self, event: TurnFailed) -> None:
        """Default handler for TurnFailed events. A failed turn must never die silently.

        Root-turn failures are fatal to the reply the user is waiting for — surface
        them as an error with the reason. Child (delegated) turns carry parent_id:
        the parent continues and still owes the real answer, so emit a one-line
        status note instead (child completions stay internal — see on_task_complete —
        but child failures surface; silence is indistinguishable from progress)."""
        if getattr(event, "parent_id", None):
            await self.send_message(
                f"⚠️ subagent {event.agent_id} failed ({event.reason}) after "
                f"{event.iterations} iterations — continuing with partial results",
                agent_id=event.agent_id,
            )
            return
        detail = (event.detail or "").strip()
        await self.send_error(
            error=f"turn failed — {event.reason}",
            detail=detail[:400] or None,
            agent_id=event.agent_id,
        )

    async def on_parse_failed(self, event: ParseFailed) -> None:
        """Default handler for ParseFailed events. Same principle as on_turn_failed: silence
        is indistinguishable from progress.

        The model emitted tool-call-shaped text that nothing parsed, so the turn did NO work
        while reading to the user as ordinary prose. That is the failure that makes a working
        model look like it is lying about what it did — observed 2026-08-05, where DeepSeek
        emitted correct native tool calls (DSML) for a whole session, every one of them was
        rendered as chat, no file was ever created, and the model took the blame for hours.
        The loop nudges it with the taught format and retries; THIS line is what tells the
        user, because a warning in a log nobody opens is the same as no warning at all."""
        await self.send_message(
            f"⚠️ tool call not parsed (retry {event.parse_retry_count}/3) — the model emitted "
            f"tool-call-shaped text the harness could not read, so nothing ran. If this "
            f"repeats, the serving runtime is not converting the model's native tool syntax: "
            f"check llama.cpp --jinja / vLLM --tool-call-parser.",
            agent_id=event.agent_id,
        )

    async def on_compaction_triggered(self, event: CompactionTriggered) -> None:
        """Default handler for CompactionTriggered. Same principle as on_parse_failed: an
        invisible context squeeze reads as the model losing its memory mid-session for no
        reason (observed 2026-08-13: a real compaction fired with zero telemetry because the
        REPL's ContextManager had no bus). One line, so the squeeze is attributable."""
        await self.send_message(
            f"context compacted: {event.pre_usage_fraction:.0%} → "
            f"{event.post_usage_fraction:.0%} of the usable budget (older detail summarized; "
            f"recent messages kept verbatim)",
            agent_id=event.agent_id,
        )

    async def on_escalation(self, event: Escalation) -> None:
        """Default handler for Escalation events."""
        await self.send_error(
            error=f"Agent {event.agent_id} escalated: {event.reason}",
            detail=event.detail,
            agent_id=event.agent_id,
        )

    async def on_heartbeat(self, event: Heartbeat) -> None:
        """Default handler for Heartbeat events. Default: no-op."""
        pass
