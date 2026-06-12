"""TerminalChannel — Rich-formatted terminal output with streaming and prompt_toolkit input."""
from __future__ import annotations

import asyncio
import os
from typing import Any, AsyncIterator

import structlog
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

from localharness.channels.base import ChannelAdapter
from localharness.channels.errors import ChannelStartError
from localharness.core.bus import EventBus
from localharness.core.events import Action, Escalation, Heartbeat, Observation, TaskComplete, TurnFailed

log = structlog.get_logger(__name__)

# Tool call/result display characters (CONTEXT.md locked decisions)
_DIAMOND = "\u25c6"   # ◆  tool call indicator
_CHECK = "\u2713"     # ✓  tool result success
_CROSS = "\u2717"     # ✗  tool result error

# Input bubble (Claude Code style): top border + left wall in the prompt message,
# bottom border lives in the bottom_toolbar while typing, reprinted on accept so
# scrollback keeps a closed box. "noreverse" kills the toolbar's default reverse video.
INPUT_STYLE = Style.from_dict({
    "bottom-toolbar": "noreverse",
    "frame": "ansibrightblack",
    "caret": "ansicyan bold",
    "placeholder": "ansibrightblack italic",
})

TERMINAL_THEME = Theme({
    "agent.name":   "bold cyan",
    "agent.text":   "white",
    "tool.call":    "dim cyan",
    "tool.result":  "dim white",
    "tool.error":   "bold red",
    "system.info":  "dim yellow",
    "system.error": "bold red",
    "user.input":   "bold green",
    "highlight":    "bold yellow",
    "success":      "bold green",
    "warning":      "bold yellow",
    "muted":        "dim",
})


def _format_args_compact(arguments: dict[str, Any], max_value_len: int = 60) -> str:
    """Format tool call arguments for inline display.

    Rules:
    - String values: show quoted, truncate to max_value_len with "..."
    - Number values: show as-is
    - List values: show as [N items] if N > 3, else show list
    - Dict values: show as {N keys}
    - Bool values: show as true/false (lowercase)
    - None values: show as null
    """
    parts = []
    for k, v in arguments.items():
        if isinstance(v, bool):
            parts.append(f"{k}={'true' if v else 'false'}")
        elif isinstance(v, str):
            truncated = v[:max_value_len] + "..." if len(v) > max_value_len else v
            parts.append(f'{k}="{truncated}"')
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


def _get_key_arg(arguments: dict[str, Any]) -> str:
    """Return the value of the first string argument, or ''."""
    for v in arguments.values():
        if isinstance(v, str):
            return v
    return ""


class TerminalChannel(ChannelAdapter):
    """Rich-formatted terminal channel with streaming output.

    State machine:
      IDLE → STREAMING → IDLE
                       ↓
                  WAITING_INPUT → IDLE

    The _output_lock prevents concurrent stdout writes from corrupting terminal output.
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
        self._state: str = "IDLE"
        self._output_lock: asyncio.Lock = asyncio.Lock()
        self._heartbeat_counter: int = 0
        self._action_handle = None
        self._observation_handle = None
        self._task_complete_handle = None
        self._escalation_handle = None
        self._heartbeat_handle = None
        self._turn_failed_handle = None

    async def start(self) -> None:
        """Initialize prompt_toolkit session and subscribe to bus events."""
        history_path = os.path.expanduser(self._history_file)
        os.makedirs(os.path.dirname(history_path), exist_ok=True)

        self._session = PromptSession(
            history=FileHistory(history_path),
            auto_suggest=AutoSuggestFromHistory(),
        )

        self._action_handle = self.bus.subscribe(Action, self.on_action)
        self._observation_handle = self.bus.subscribe(Observation, self.on_observation)
        self._task_complete_handle = self.bus.subscribe(TaskComplete, self.on_task_complete)
        self._turn_failed_handle = self.bus.subscribe(TurnFailed, self.on_turn_failed)
        self._escalation_handle = self.bus.subscribe(Escalation, self.on_escalation)
        self._heartbeat_handle = self.bus.subscribe(Heartbeat, self.on_heartbeat)

    async def stop(self) -> None:
        """Unsubscribe from bus, stop any active Live context, flush console."""
        for handle in (
            self._action_handle,
            self._observation_handle,
            self._task_complete_handle,
            self._turn_failed_handle,
            self._escalation_handle,
            self._heartbeat_handle,
        ):
            if handle is not None:
                self.bus.unsubscribe(handle)

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
        """Print a message to the terminal. Wraps in Panel if agent_id provided."""
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
        """Stream tokens to terminal using Rich Live. Returns full assembled text."""
        full_text = ""
        async with self._output_lock:
            self._state = "STREAMING"
            panel_title = f"[agent.name]{agent_id or 'agent'}[/agent.name] [muted]streaming...[/muted]"
            live_panel = Panel("", title=panel_title, border_style="cyan")

            with Live(live_panel, console=self._console, refresh_per_second=20) as live:
                self._live = live
                async for token in token_stream:
                    full_text += token
                    live_panel = Panel(
                        Text(full_text, style="agent.text"),
                        title=panel_title,
                        border_style="cyan",
                    )
                    live.update(live_panel)
                self._live = None

            # Final non-live panel with green border
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
        """Display tool invocation inline. Format: ◆ tool_name key_arg"""
        key_arg = _get_key_arg(arguments)
        display = f"{_DIAMOND} {tool_name} {key_arg}".rstrip()
        async with self._output_lock:
            self._console.print(f"  [tool.call]{display}[/tool.call]")

    async def send_tool_result(
        self,
        tool_name: str,
        result: str,
        is_error: bool,
        agent_id: str | None = None,
    ) -> None:
        """Display tool result inline. Uses checkmark for success, cross for error."""
        lines = result.strip().split("\n") if result else [""]
        async with self._output_lock:
            if is_error:
                first = lines[0][:120] if lines else ""
                self._console.print(f"  [tool.error]{_CROSS} {tool_name} (exit 1): {first}[/tool.error]")
            else:
                line_count = len(lines)
                self._console.print(f"  [tool.result]{_CHECK} {tool_name} ({line_count} lines)[/tool.result]")

    async def send_error(
        self,
        error: str,
        detail: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Write error to stderr console."""
        self._err_console.print(f"[system.error]Error:[/system.error] {error}")
        if detail:
            for line in detail.split("\n"):
                self._err_console.print(f"  {line}")

    async def read_input(self, prompt: str = ">") -> str:
        """Read a line from the user inside a rounded input bubble. Raises ChannelStartError if not started."""
        if self._session is None:
            raise ChannelStartError("TerminalChannel.start() must be called before read_input()")
        self._state = "WAITING_INPUT"
        width = max(self._console.width, 20)
        bottom = "╰" + "─" * (width - 2) + "╯"
        message = [
            ("class:frame", "╭" + "─" * (width - 2) + "╮\n│"),
            ("class:caret", f" {prompt} "),
        ]
        try:
            line = await self._session.prompt_async(
                message,
                default="",
                placeholder=[("class:placeholder", "describe a task  (/help for commands)")],
                bottom_toolbar=[("class:frame", bottom)],
                style=INPUT_STYLE,
            )
            return line.strip()
        except KeyboardInterrupt:
            return ""
        finally:
            # prompt_toolkit erases the toolbar on exit — close the box in scrollback.
            self._console.print(bottom, style="bright_black", highlight=False)
            self._state = "IDLE"

    def _ensure_idle(self) -> None:
        """Log a warning if called while STREAMING (concurrent write detected)."""
        if self._state == "STREAMING":
            self._err_console.print("[warning]Warning: output during streaming state[/warning]")

    async def on_heartbeat(self, event: Heartbeat) -> None:
        """Show a spinner update on every 3rd heartbeat."""
        self._heartbeat_counter += 1
        if self._heartbeat_counter % 3 == 0:
            spinner_chars = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"
            char = spinner_chars[self._heartbeat_counter % len(spinner_chars)]
            async with self._output_lock:
                self._console.print(
                    f"  [muted]{char} Working...[/muted]",
                    end="\r",
                )
