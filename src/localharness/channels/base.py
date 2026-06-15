"""ChannelAdapter ABC — pluggable interface for all channel adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from localharness.core.bus import EventBus
from localharness.core.events import Action, Escalation, Heartbeat, Observation, TaskComplete, TurnFailed


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
        """Default handler for Observation events. Calls send_tool_result."""
        result = event.output or event.error or ""
        is_error = event.error is not None
        await self.send_tool_result(
            tool_name=event.tool_name or "",
            result=result,
            is_error=is_error,
            agent_id=event.agent_id,
        )

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
