# Spec 11: Channel System

**Component:** `src/localharness/channels/`
**Requirements:** CHAN-01, CHAN-02, CHAN-03
**Status:** v1 (TerminalChannel only; Discord/Slack/webhook deferred to v2)

---

## Purpose

The channel system is LocalHarness's output and input delivery layer. It decouples how agent actions and observations are presented to the user from the agent loop itself.

In v1, the only channel is `TerminalChannel` — Rich-formatted stdout with streaming token output and prompt_toolkit input. The channel is designed from day one for future adapters: the `ChannelAdapter` abstract base class defines the contract that Discord, Slack, webhook, and file adapters will implement without modifying the agent loop or orchestrator.

Channels subscribe to events from the event bus. They never call agent loops or orchestrator methods directly.

---

## ChannelAdapter Abstract Base Class

```python
# src/localharness/channels/base.py

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator
from localharness.core.events import (
    EventBus,
    UserMessage,
    Action,
    Observation,
    TaskComplete,
    Heartbeat,
    Escalation,
)

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
        """
        Subscribe to event bus topics.
        Begin accepting user input (for interactive channels).
        
        The set of subscribed event types is channel-specific.
        Terminal subscribes to all event types (for full display).
        A file-output channel might subscribe to TaskComplete only.
        
        Raises:
            ChannelStartError: If subscription fails or resources unavailable.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """
        Unsubscribe from event bus.
        Flush any buffered output.
        Release resources (close file handles, disconnect sockets, etc.).
        Safe to call multiple times.
        """
        ...

    @abstractmethod
    async def send_message(
        self,
        content: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Send a text message to the channel output.
        
        This is the low-level output primitive used by event handlers.
        Callers should prefer the higher-level event-driven path, but
        the orchestrator uses this directly for system messages.
        
        Args:
            content: Text content (may include markdown).
            agent_id: If set, display as coming from this agent.
            metadata: Channel-specific rendering hints (e.g. {"color": "green"}).
        """
        ...

    @abstractmethod
    async def send_streaming(
        self,
        token_stream: AsyncIterator[str],
        agent_id: str | None = None,
    ) -> str:
        """
        Stream tokens to output as they arrive.
        
        Args:
            token_stream: Async iterator yielding token strings.
            agent_id: Source agent identifier for display labeling.
        
        Returns:
            The complete assembled text after all tokens are consumed.
        
        Raises:
            ChannelOutputError: On output write failure.
        """
        ...

    @abstractmethod
    async def send_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        agent_id: str | None = None,
    ) -> None:
        """
        Display a tool call invocation.
        Presented differently from regular message text (dimmer, prefix, etc.).
        """
        ...

    @abstractmethod
    async def send_tool_result(
        self,
        tool_name: str,
        result: str,
        is_error: bool,
        agent_id: str | None = None,
    ) -> None:
        """
        Display the result of a tool call.
        On is_error=True, the display should indicate failure clearly.
        """
        ...

    @abstractmethod
    async def send_error(
        self,
        error: str,
        detail: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        """
        Display an error to the user.
        Used for agent errors, escalations, and system errors.
        """
        ...

    @abstractmethod
    async def read_input(self, prompt: str = "> ") -> str:
        """
        Read a line of input from the user.
        
        For interactive channels (terminal): blocks until user presses Enter.
        For non-interactive channels (file, webhook): raises NotInteractiveError.
        
        Returns the input string with leading/trailing whitespace stripped.
        
        Raises:
            ChannelInputError: On read failure.
            NotInteractiveError: If this channel does not support user input.
            EOFError: On Ctrl-D or stream end.
        """
        ...

    # --- Event Handlers (subscribed via bus) ---

    async def on_action(self, event: Action) -> None:
        """
        Default handler for Action events.
        Subclasses override to customize rendering.
        Default: call send_tool_call if action is a tool call.
        """
        if event.action_type == "tool_call":
            await self.send_tool_call(
                tool_name=event.tool_name or "",
                arguments=event.arguments or {},
                agent_id=event.agent_id,
            )

    async def on_observation(self, event: Observation) -> None:
        """
        Default handler for Observation events.
        Subclasses override to customize rendering.
        Default: call send_tool_result.
        """
        await self.send_tool_result(
            tool_name=event.tool_name or "",
            result=event.content,
            is_error=event.is_error,
            agent_id=event.agent_id,
        )

    async def on_task_complete(self, event: TaskComplete) -> None:
        """
        Default handler for TaskComplete events.
        Subclasses override to customize rendering.
        Default: send the DelegateResult summary.
        """
        await self.send_message(
            content=event.result.summary,
            agent_id=event.agent_id,
        )

    async def on_escalation(self, event: Escalation) -> None:
        """
        Default handler for Escalation events.
        """
        await self.send_error(
            error=f"Agent {event.agent_id} escalated: {event.reason}",
            detail=event.detail,
            agent_id=event.agent_id,
        )

    async def on_heartbeat(self, event: Heartbeat) -> None:
        """
        Default handler for Heartbeat events. Default: no-op.
        Terminal channel may show a spinner update.
        """
        pass
```

### Error Types

```python
# src/localharness/channels/errors.py

class ChannelError(Exception):
    """Base class for channel errors."""

class ChannelStartError(ChannelError):
    """Channel failed to start (resource unavailable, bus subscription failed)."""

class ChannelOutputError(ChannelError):
    """Write to channel output failed (stdout closed, socket disconnected, etc.)."""
    def __init__(self, channel_id: str, underlying: Exception) -> None: ...

class ChannelInputError(ChannelError):
    """Read from channel input failed."""
    def __init__(self, channel_id: str, underlying: Exception) -> None: ...

class NotInteractiveError(ChannelError):
    """
    Raised by read_input() on non-interactive channels.
    """
    def __init__(self, channel_id: str) -> None: ...
```

---

## TerminalChannel

```python
# src/localharness/channels/terminal.py

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.theme import Theme
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
import asyncio
from localharness.channels.base import ChannelAdapter

TERMINAL_THEME = Theme({
    "agent.name":    "bold cyan",
    "agent.text":    "white",
    "tool.call":     "dim cyan",
    "tool.result":   "dim white",
    "tool.error":    "bold red",
    "system.info":   "dim yellow",
    "system.error":  "bold red",
    "user.input":    "bold green",
    "highlight":     "bold yellow",
    "success":       "bold green",
    "warning":       "bold yellow",
    "muted":         "dim",
})

class TerminalChannel(ChannelAdapter):
    """
    Rich-formatted terminal channel with streaming output.
    
    Output: Rich Console to stdout (formatted text, panels, spinners).
    Input: prompt_toolkit PromptSession (history, auto-suggest, key bindings).
    
    State machine:
      IDLE → STREAMING → IDLE
                       ↓
                  WAITING_INPUT → IDLE
    
    When STREAMING: output is written via Rich Live; input is suppressed.
    When WAITING_INPUT: prompt_toolkit reads user line; output is suppressed.
    When IDLE: neither streaming nor waiting.
    
    The state machine is enforced via asyncio.Lock to prevent concurrent
    stdout writes from corrupting terminal output.
    """

    channel_id = "terminal"

    def __init__(
        self,
        bus: EventBus,
        config: dict[str, Any],
        history_file: str = "~/.localharness/.repl_history",
    ) -> None:
        super().__init__(bus, config)
        self._console = Console(theme=TERMINAL_THEME, highlight=False)
        self._err_console = Console(stderr=True, theme=TERMINAL_THEME, highlight=False)
        self._session: PromptSession | None = None
        self._history_file = history_file
        self._live: Live | None = None
        self._state: str = "IDLE"    # IDLE | STREAMING | WAITING_INPUT
        self._output_lock = asyncio.Lock()
        self._current_agent_id: str | None = None
        self._spinner_task: asyncio.Task | None = None
        self._heartbeat_counter: int = 0

    async def start(self) -> None:
        """
        Initialize prompt_toolkit session.
        Subscribe to bus events: Action, Observation, TaskComplete, Escalation, Heartbeat.
        """
        self._session = PromptSession(
            history=FileHistory(self._history_file),
            auto_suggest=AutoSuggestFromHistory(),
        )
        await self.bus.subscribe("Action", self.on_action)
        await self.bus.subscribe("Observation", self.on_observation)
        await self.bus.subscribe("TaskComplete", self.on_task_complete)
        await self.bus.subscribe("Escalation", self.on_escalation)
        await self.bus.subscribe("Heartbeat", self.on_heartbeat)

    async def stop(self) -> None:
        """Unsubscribe, stop any active Live context, flush console."""
        await self.bus.unsubscribe("Action", self.on_action)
        await self.bus.unsubscribe("Observation", self.on_observation)
        await self.bus.unsubscribe("TaskComplete", self.on_task_complete)
        await self.bus.unsubscribe("Escalation", self.on_escalation)
        await self.bus.unsubscribe("Heartbeat", self.on_heartbeat)
        if self._live is not None:
            self._live.stop()
            self._live = None
        self._console.file.flush()

    async def send_message(
        self,
        content: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Print a message to the terminal.
        
        If agent_id is provided, wraps in a Panel with agent name as title.
        If no agent_id, prints as plain styled text.
        Acquires output_lock to prevent concurrent writes.
        """
        async with self._output_lock:
            self._ensure_idle()
            if agent_id:
                self._console.print(Panel(
                    Text(content, style="agent.text"),
                    title=f"[agent.name]{agent_id}[/agent.name]",
                    border_style="cyan",
                ))
            else:
                self._console.print(content)

    async def send_streaming(
        self,
        token_stream: AsyncIterator[str],
        agent_id: str | None = None,
    ) -> str:
        """
        Stream tokens to terminal using Rich Live.
        
        Opens a Rich Live context that updates in place (no scrolling).
        Tokens are accumulated and displayed as they arrive.
        The panel title shows the agent name and a streaming indicator.
        
        On completion, stops Live and prints the final panel (non-live).
        
        Threading: acquires output_lock for the entire streaming duration.
        No other send_* calls can proceed while streaming.
        """
        full_text = ""
        async with self._output_lock:
            self._state = "STREAMING"
            self._current_agent_id = agent_id
            panel_title = f"[agent.name]{agent_id or 'agent'}[/agent.name] [muted]streaming…[/muted]"
            live_panel = Panel("", title=panel_title, border_style="cyan")

            with Live(live_panel, console=self._console, refresh_per_second=20) as live:
                async for token in token_stream:
                    full_text += token
                    live_panel = Panel(
                        Text(full_text, style="agent.text"),
                        title=panel_title,
                        border_style="cyan",
                    )
                    live.update(live_panel)

            # Print final non-live panel
            self._console.print(Panel(
                Text(full_text, style="agent.text"),
                title=f"[agent.name]{agent_id or 'agent'}[/agent.name]",
                border_style="green",
            ))
            self._state = "IDLE"
        return full_text

    async def send_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        agent_id: str | None = None,
    ) -> None:
        """
        Display tool invocation inline.
        
        Format: ⚙ tool_name(arg1=val1, arg2=val2)
        Style: dim cyan
        
        Arguments are formatted compactly — long string values are truncated
        to 60 chars with "..." suffix.
        """
        args_str = _format_args_compact(arguments, max_value_len=60)
        line = f"⚙ {tool_name}({args_str})"
        async with self._output_lock:
            self._console.print(f"  [tool.call]{line}[/tool.call]")

    async def send_tool_result(
        self,
        tool_name: str,
        result: str,
        is_error: bool,
        agent_id: str | None = None,
    ) -> None:
        """
        Display tool result inline.
        
        Format: ↳ {first line of result, max 120 chars}
        If result has multiple lines, shows: ↳ {first line} (+N more lines)
        
        On is_error=True: ↳ ERROR: {result} in bold red.
        """
        lines = result.strip().split("\n")
        if is_error:
            async with self._output_lock:
                self._console.print(f"  [tool.error]↳ ERROR: {lines[0][:120]}[/tool.error]")
        else:
            summary = lines[0][:120]
            extra = f" (+{len(lines)-1} more lines)" if len(lines) > 1 else ""
            async with self._output_lock:
                self._console.print(f"  [tool.result]↳ {summary}{extra}[/tool.result]")

    async def send_error(
        self,
        error: str,
        detail: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Display error to stderr using err_console."""
        async with self._output_lock:
            self._err_console.print(f"\n[system.error]Error:[/system.error] {error}")
            if detail:
                for line in detail.split("\n"):
                    self._err_console.print(f"  {line}")
            self._err_console.print()

    async def read_input(self, prompt: str = "you> ") -> str:
        """
        Read a line from the user via prompt_toolkit.
        
        Displays the prompt and waits for Enter.
        Applies history and auto-suggest from FileHistory.
        
        Returns empty string on Ctrl-C (SIGINT), not KeyboardInterrupt.
        Raises EOFError on Ctrl-D.
        
        The output_lock is NOT held during input (allows background events
        to print while user is typing — acceptable for v1; v2 may coordinate
        more precisely using prompt_toolkit's print_formatted_text).
        """
        if self._session is None:
            raise ChannelStartError("TerminalChannel.start() must be called before read_input()")
        self._state = "WAITING_INPUT"
        try:
            line = await self._session.prompt_async(
                prompt,
                default="",
            )
            return line.strip()
        except KeyboardInterrupt:
            return ""
        finally:
            self._state = "IDLE"

    async def on_heartbeat(self, event: Heartbeat) -> None:
        """
        Update the spinner on heartbeat events.
        
        If an agent has been running > 10 seconds without new output,
        show a spinner with elapsed time. The spinner is printed inline
        (not via Live — it's a simple print that gets replaced by real output).
        
        Uses the heartbeat event's elapsed_seconds field.
        """
        if event.elapsed_seconds and event.elapsed_seconds > 10:
            self._heartbeat_counter += 1
            if self._heartbeat_counter % 3 == 0:  # Update every ~3 heartbeats
                spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
                char = spinner_chars[self._heartbeat_counter % len(spinner_chars)]
                async with self._output_lock:
                    self._console.print(
                        f"  [muted]{char} Working… ({event.elapsed_seconds:.0f}s)[/muted]",
                        end="\r"
                    )

    def _ensure_idle(self) -> None:
        """
        Assert the channel is in IDLE state. Called before terminal writes.
        If STREAMING, this indicates a programming error (concurrent write).
        Logs a warning but does not raise — output may be garbled but won't crash.
        """
        if self._state == "STREAMING":
            self._err_console.print("[warning]Warning: output during streaming state[/warning]")
```

---

## Message Formatting

### Format Contracts

All message types have defined rendering behavior:

| Event / content | Terminal format | Style |
|---|---|---|
| Orchestrator message | Plain `white` text | No panel |
| Agent final result | `Panel` with agent name, green border | `agent.text` |
| Agent streaming text | `Live Panel` with agent name, cyan border | `agent.text` |
| Tool call | `  ⚙ tool_name(args)` inline | `tool.call` (dim cyan) |
| Tool result (success) | `  ↳ first line (+N)` inline | `tool.result` (dim white) |
| Tool result (error) | `  ↳ ERROR: message` inline | `tool.error` (bold red) |
| System info | Plain `dim yellow` text | No panel |
| Error | stderr, `bold red` "Error:" prefix | `system.error` |
| User prompt | `you> ` prefix | `user.input` (bold green) |

### Tool Call Argument Formatting

```python
def _format_args_compact(arguments: dict[str, Any], max_value_len: int = 60) -> str:
    """
    Format tool call arguments for inline display.
    
    Rules:
    - String values: show quoted, truncate to max_value_len with "..."
    - Number values: show as-is
    - List values: show as [N items] if N > 3, else show first 3 items
    - Dict values: show as {N keys}
    - Bool values: show as true/false (lowercase)
    - None values: show as null
    
    Example output: query="SPX may 2026"..., num_results=5
    """
    parts = []
    for k, v in arguments.items():
        if isinstance(v, str):
            truncated = v[:max_value_len] + "…" if len(v) > max_value_len else v
            parts.append(f'{k}="{truncated}"')
        elif isinstance(v, bool):
            parts.append(f"{k}={'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k}={v}")
        elif isinstance(v, list):
            if len(v) <= 3:
                parts.append(f"{k}={v}")
            else:
                parts.append(f"{k}=[{len(v)} items]")
        elif isinstance(v, dict):
            parts.append(f"{k}={{{len(v)} keys}}")
        elif v is None:
            parts.append(f"{k}=null")
        else:
            parts.append(f"{k}=...")
    return ", ".join(parts)
```

---

## Event Bus Integration

### Subscription Contracts

The terminal channel subscribes to events after `start()` is called. The bus delivers events in FIFO order. All event handlers are async coroutines.

| Event Type | Handler | What it does |
|---|---|---|
| `Action` (tool_call) | `on_action` | `send_tool_call(tool_name, arguments, agent_id)` |
| `Action` (message) | `on_action` | `send_message(content, agent_id)` |
| `Action` (finish) | `on_action` | no output (finish is shown via TaskComplete) |
| `Observation` | `on_observation` | `send_tool_result(tool_name, result, is_error, agent_id)` |
| `TaskComplete` | `on_task_complete` | `send_message(result.summary, agent_id)` |
| `Escalation` | `on_escalation` | `send_error(reason, detail, agent_id)` |
| `Heartbeat` | `on_heartbeat` | spinner update if elapsed > 10s |

The channel does NOT subscribe to:
- `UserMessage` (channel publishes these, it doesn't receive its own)
- `TaskRequest` (orchestrator-to-agent, no user-visible output)
- `AgentCreated` (terminal channel is notified via a subsequent `send_message` from orchestrator)
- `SystemReady` (terminal channel receives the introduction message directly from orchestrator)

### Publishing UserMessage

The terminal channel is the only component that publishes `UserMessage` events (from the REPL). When the user submits input, the REPL:

```python
async def submit_input(self, text: str) -> None:
    """Called by REPL when user submits a line."""
    await self.bus.emit(UserMessage(
        id=generate_id("msg"),
        content=text,
        channel="terminal",
        channel_metadata=None,
        ts=int(time.time()),
    ))
```

---

## Future Adapter Interface

The v2 Discord adapter will implement `ChannelAdapter` with:

- `send_message`: `discord.WebhookClient.send()` to the agent's assigned subchannel
- `send_streaming`: Buffered — Discord does not support streaming natively. Buffer tokens for 1s or 2000 chars, then send as a single edit to a "thinking..." placeholder message.
- `send_tool_call` / `send_tool_result`: Formatted as Discord code blocks or embeds.
- `read_input`: `discord.Client.on_message` filtered to the agent's subchannel, awaited via `asyncio.Event`.
- `channel_id = "discord"`

The Discord adapter will require its own config block:

```yaml
# In org config, under channels.discord:
channels:
  discord:
    bot_token: "${DISCORD_BOT_TOKEN}"   # from env var; never hardcoded
    guild_id: "123456789"
    category_id: "987654321"           # channel category for agent subchannels
    orchestrator_channel_id: "111"     # where users talk to orchestrator
```

The adapter architecture ensures no changes to the agent loop, orchestrator, or event bus are needed when Discord is added.

A Slack adapter, webhook adapter (POST to URL on TaskComplete), and file adapter (write results to disk) follow the same pattern.

---

## Configuration

```yaml
# In org config, under channels.terminal:
channels:
  terminal:
    history_file: "~/.localharness/.repl_history"
    streaming: true                 # false = buffer and print on completion
    show_tool_calls: true           # false = hide tool call lines
    show_tool_results: true         # false = hide tool result lines
    max_tool_result_display_lines: 5  # truncate long results in display
    spinner_threshold_seconds: 10   # show spinner after N seconds idle
    color: true                     # false = plain text output (for terminals without color)
    panel_width: null               # null = auto (terminal width); or fixed int
```

---

## Implementation Notes

- Rich's `Console` is not thread-safe. All writes must go through `asyncio.Lock` (`_output_lock`). Since the channel runs on the asyncio event loop, `asyncio.Lock` (not `threading.Lock`) is the correct primitive.
- `prompt_toolkit`'s `PromptSession.prompt_async()` uses `asyncio` and does not block the event loop. It yields control correctly. The REPL loop can process bus events while waiting for input.
- When streaming, the `Live` context owns the terminal. Any call to `_console.print()` outside the `Live.update()` call will corrupt the output. The `_output_lock` prevents this — no other `send_*` call can proceed while `send_streaming` holds the lock.
- The Rich `Theme` is instantiated once and shared. Never create per-call Theme or Console instances.
- `send_error` writes to `_err_console` (stderr). This ensures error messages are visible even when stdout is piped.
- For `--no-stream` mode (in `agent run`): swap `send_streaming` behavior to accumulate all tokens silently and call `send_message` on completion. This is controlled by the `streaming` config option; the channel checks `self.config.get("streaming", True)` in `send_streaming`.
- The channel never re-raises exceptions from `send_*` calls to the event bus handler. A failed write is logged to stderr and the agent loop continues. A channel output failure is not a reason to abort the agent.
