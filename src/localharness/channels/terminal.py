"""TerminalChannel — Rich-formatted terminal output with streaming and prompt_toolkit input."""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

import structlog
from prompt_toolkit.application import Application
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.processors import AppendAutoSuggestion, BeforeInput
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.status import Status
from rich.text import Text
from rich.theme import Theme

from localharness.channels.base import ChannelAdapter
from localharness.channels.errors import ChannelStartError
from localharness.core.bus import EventBus
from localharness.core.events import (
    Action,
    ConsolidationFinished,
    ConsolidationStarted,
    Escalation,
    Heartbeat,
    Observation,
    TaskComplete,
    TurnFailed,
)
from localharness.tools.capabilities import UNTRUSTED_INGEST

log = structlog.get_logger(__name__)

# Tool call/result display characters (CONTEXT.md locked decisions)
_DIAMOND = "\u25c6"   # ◆  tool call indicator
_CHECK = "\u2713"     # ✓  tool result success
_CROSS = "\u2717"     # ✗  tool result error

# Official label for the background-memory ("dreaming") status (#20): middle-dot + ellipsis.
_DREAMING_LABEL = "· dreaming…"   # · dreaming…

# In-turn narration: an interstitial llm_response (content emitted ALONGSIDE tool calls)
# carries the model's progress narration ("pulling the data…"). Rendered as one dim line
# (middle-dot idiom, matching _DREAMING_LABEL / the burst separators) opening the coming
# chunk of tool activity — cropped to the first non-empty line and hard-capped so a chatty
# model can't wall-of-text the turn (per-call truth stays on the bus ledger). A tool-less
# llm_response IS the final answer (rendered by the TaskComplete panel) — never here.
_NARRATE = "·"        # ·  narration line indicator
_MAX_NARRATION = 160       # hard char cap before the ellipsis

# Braille spinner frames for the IN-FRAME working glyph. While the persistent input box is
# live, the thinking/burst indicator advances inside the box's bottom border (a
# FormattedTextControl fragment refreshed by app.invalidate) instead of a rich Status/Live —
# rich spinners under patch_stdout glue lines and, worse, FREEZE on Ctrl+C-during-burst.
_SPIN_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Burst consolidation: consecutive calls from one tool family collapse into a single
# live counter line (`◆ web_search · web_fetch · 30/30`) instead of a line per call —
# a 30-hit research burst is one scrollback line, not 60. Per-call truth (args, errors,
# timings) stays on the bus ledger (bus-events.jsonl); this is display-only. Family
# membership reuses the capability metadata. (Descriptive labels dropped 2026-07-14 —
# owner: cleaner without; the close_note security disclosure stays.)
_BURST_GROUPS: tuple[tuple[frozenset[str], str | None], ...] = (
    (UNTRUSTED_INGEST, "web results — UNTRUSTED, treated as data only"),
    (frozenset({"tool_result_get"}), None),
)


@dataclass
class _Burst:
    """Consolidation state for one open family burst (display-only)."""
    family: frozenset[str]
    close_note: str | None       # ✓ line printed once on close (None = no note)
    tools: list[str] = field(default_factory=list)  # first-use order, deduped
    calls: int = 0
    done: int = 0
    errors: int = 0
    status: Status | None = None  # live spinner while the burst is open (TTY only)


# Input bubble (Claude Code style): an inline prompt_toolkit Application drawing a
# fully closed rounded box around the buffer — bottom border hugs the input line
# (a PromptSession bottom_toolbar would pin it to the screen bottom instead).
INPUT_STYLE = Style.from_dict({
    "frame": "ansibrightblack",
    "caret": "ansicyan bold",
    "input": "ansidefault",                # typed text: terminal default fg (else inherits the dim frame class)
    "auto-suggestion": "ansibrightblack",  # ghost history completion stays dim
    "hint": "ansibrightblack italic",
    # context meter — GSD thresholds (green → yellow → orange → red at compaction)
    "ctx-low": "ansigreen",
    "ctx-mid": "ansiyellow",
    "ctx-high": "#ff8700",
    "ctx-crit": "ansired bold",
    "ctx-track": "ansibrightblack",
})


def _ctx_segments(pct: float) -> tuple[list[tuple[str, str]], int]:
    """GSD-style 10-cell context meter: ████░░░░░░ 42%. Returns (fragments, plain width).

    Color steps match gsd-statusline thresholds; localharness summary-compaction fires
    at 80% (ctx-crit), so red == "compacting now," not "out of room."
    """
    pct = max(0.0, min(100.0, pct))
    level = ("ctx-low" if pct < 50 else "ctx-mid" if pct < 65
             else "ctx-high" if pct < 80 else "ctx-crit")
    filled = min(10, int(pct // 10))
    label = f" {pct:.0f}%"
    frags = [
        (f"class:{level}", "█" * filled),
        ("class:ctx-track", "░" * (10 - filled)),
        (f"class:{level}", label),
    ]
    return frags, 10 + len(label)


def _build_input_app(
    history: FileHistory, prompt: str, hint: str, context_pct: float | None = None,
) -> Application:
    """Inline application: ╭─╮ │ > input │ ╰─ hint ──── meter ─╯. Exits with the entered line."""
    buf = Buffer(history=history, auto_suggest=AutoSuggestFromHistory(), multiline=False)
    control = BufferControl(
        buffer=buf,
        input_processors=[
            BeforeInput([("class:caret", f" {prompt} ")]),
            AppendAutoSuggestion(),
        ],
    )

    kb = KeyBindings()

    @kb.add("enter")
    def _accept(event) -> None:
        buf.append_to_history()
        event.app.exit(result=buf.text)

    @kb.add("c-c")
    def _interrupt(event) -> None:
        if buf.text:
            buf.reset()  # first Ctrl+C clears the line, like Claude Code / most shells
        else:
            event.app.exit(exception=KeyboardInterrupt(), style="class:aborting")

    @kb.add("c-d")
    def _eof(event) -> None:
        if not buf.text:
            event.app.exit(exception=EOFError(), style="class:exiting")

    def _wall(char: str) -> Window:
        return Window(width=1, char=char)

    bottom = [_wall("╰"), Window(char="─", height=1, width=1)]
    if hint:
        hint_text = f" {hint} "
        bottom.append(
            Window(FormattedTextControl([("class:hint", hint_text)]), width=len(hint_text), height=1)
        )
    bottom.append(Window(char="─", height=1))  # stretchy filler right-aligns the meter
    if context_pct is not None:
        frags, w = _ctx_segments(context_pct)
        bottom += [
            Window(FormattedTextControl(frags), width=w, height=1),
            Window(char="─", height=1, width=1),
        ]
    bottom.append(_wall("╯"))
    body = HSplit([
        VSplit([_wall("╭"), Window(char="─", height=1), _wall("╮")]),
        VSplit([_wall("│"), Window(control, wrap_lines=True, dont_extend_height=True, style="class:input"), _wall("│")]),
        VSplit(bottom),
    ], style="class:frame")

    return Application(
        layout=Layout(body, focused_element=control),
        key_bindings=kb,
        style=INPUT_STYLE,
        mouse_support=False,
    )


def _build_persistent_input_app(
    history: FileHistory,
    prompt: str,
    *,
    on_submit: Callable[[str], None],
    on_interrupt: Callable[[], None],
    on_eof: Callable[[], None],
    hint_fn: Callable[[], list[tuple[str, str]]],
    pct_fn: Callable[[], float | None],
) -> Application:
    """Long-lived input box that stays usable while turn output streams above it.

    Differs from _build_input_app in three ways, all load-bearing for the type-anytime box:
      1. Enter SUBMITS without exiting — it hands the line to on_submit and resets the buffer,
         so the same Application services every submission for the whole session (run once via
         asyncio.create_task(app.run_async()) alongside the turn, under patch_stdout(raw=True)).
      2. The bottom-border hint + context meter are DYNAMIC FormattedTextControl callables
         (hint_fn / pct_fn) refreshed by app.invalidate() — the seam the in-frame working glyph,
         `queued (N)`, and the routing-decision flash all render through.
      3. Ctrl+C (empty buffer) / Ctrl+D (empty buffer) call back into REPL policy (on_interrupt
         / on_eof) rather than raising out of run_async — the box owns raw mode for the whole
         session, so these are the only path a signal-suppressed terminal has to interrupt/exit.
    """
    buf = Buffer(history=history, auto_suggest=AutoSuggestFromHistory(), multiline=False)
    control = BufferControl(
        buffer=buf,
        input_processors=[
            BeforeInput([("class:caret", f" {prompt} ")]),
            AppendAutoSuggestion(),
        ],
    )

    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event) -> None:
        text = buf.text
        if text.strip():
            buf.append_to_history()
            on_submit(text)
        buf.reset()  # ready for the next line; the app stays alive

    @kb.add("c-c")
    def _interrupt(event) -> None:
        if buf.text:
            buf.reset()  # first Ctrl+C clears the line (Claude Code / shell idiom)
        else:
            on_interrupt()  # empty buffer → REPL cancels the turn, or arms/exits when idle

    @kb.add("c-d")
    def _eof(event) -> None:
        if not buf.text:
            on_eof()

    def _wall(char: str) -> Window:
        return Window(width=1, char=char)

    def _meter_frags() -> list[tuple[str, str]]:
        pct = pct_fn()
        if pct is None:
            return []
        frags, _ = _ctx_segments(pct)
        return frags

    def _meter_width() -> int:
        pct = pct_fn()
        if pct is None:
            return 0
        _, w = _ctx_segments(pct)
        return w

    bottom = [
        _wall("╰"),
        Window(char="─", height=1, width=1),
        Window(FormattedTextControl(hint_fn), height=1),
        Window(char="─", height=1),  # stretchy filler right-aligns the meter
        Window(FormattedTextControl(_meter_frags), width=_meter_width, height=1),
        Window(char="─", height=1, width=1),
        _wall("╯"),
    ]
    body = HSplit([
        VSplit([_wall("╭"), Window(char="─", height=1), _wall("╮")]),
        VSplit([_wall("│"), Window(control, wrap_lines=True, dont_extend_height=True, style="class:input"), _wall("│")]),
        VSplit(bottom),
    ], style="class:frame")

    return Application(
        layout=Layout(body, focused_element=control),
        key_bindings=kb,
        style=INPUT_STYLE,
        mouse_support=False,
    )


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


# write/edit carry a whole file body in their args — the view shows a one-line summary
# (path + line count), never the body. Every other tool's arg preview is capped to one
# line so a big argument can't flood the chat view either.
_FILE_WRITE_TOOLS = frozenset({"write", "edit"})
_MAX_ARG_PREVIEW = 120


def _tool_call_summary(tool_name: str, arguments: dict[str, Any]) -> str:
    """One-line, body-free summary of a tool call (no markup; caller adds ◆ + style).

    write/edit collapse to `<tool> <path> (<n> lines)` so the view never fills with file
    contents; `<n>` is the body the model actually emitted this call. Every other tool
    shows its key argument on a single line, capped to _MAX_ARG_PREVIEW characters."""
    if tool_name in _FILE_WRITE_TOOLS:
        path = str(arguments.get("path", "")).strip()
        body = arguments.get("content") or arguments.get("new_string") or ""
        n = len(body.splitlines()) if isinstance(body, str) else 0
        summary = f"{tool_name} {path}".rstrip()
        return f"{summary} ({n} lines)" if n else summary
    preview = _get_key_arg(arguments).split("\n", 1)[0]
    if len(preview) > _MAX_ARG_PREVIEW:
        preview = preview[: _MAX_ARG_PREVIEW - 1].rstrip() + "…"
    return f"{tool_name} {preview}".rstrip()


def _narration_line(content: str | None, cap: int = _MAX_NARRATION) -> str:
    """First non-empty line of `content`, hard-capped to `cap` chars with an ellipsis.

    Returns '' when there is nothing to narrate — empty/whitespace content, which
    reasoning-parser tool turns legitimately produce (all tokens went to reasoning +
    tool_calls). The caller skips rendering on ''."""
    if not content:
        return ""
    for raw in content.splitlines():
        line = raw.strip()
        if line:
            return line[: cap - 1].rstrip() + "…" if len(line) > cap else line
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
        history_file: str = ".repl_history",
    ) -> None:
        # A bare relative default; start_cmd resolves it UNDER the active config dir (#35, so
        # --config-dir isolates REPL history). Absolute/~ values are honored as-is at use.
        super().__init__(bus, config)
        self._console = Console(theme=TERMINAL_THEME, highlight=False)
        self._err_console = Console(stderr=True, theme=TERMINAL_THEME, highlight=False)
        self._history: FileHistory | None = None
        self._history_file = history_file
        self._live: Live | None = None
        self._state: str = "IDLE"
        self._sigint_armed: bool = False  # True after one Ctrl+C on an empty line; second exits
        # #49: hint drawn INSIDE the first input bubble (immune to the box repaint that drops
        # a banner hint in a real TTY). start_cmd sets it for interactive sessions; shown once.
        self.first_prompt_hint: str = ""
        self._output_lock: asyncio.Lock = asyncio.Lock()
        self._thinking: Status | None = None  # rich Status while the model is generating (REPL-02)
        self._dreaming: Status | None = None  # rich Status while a background memory pass runs (#20)
        self._burst: _Burst | None = None  # open family-burst consolidation (display-only)
        # Narration cadence guard: a narration line may open a chunk of tool activity only
        # after a tool result has rendered since the last one — never two dim lines in a row.
        # Starts True so the turn's first narration prints.
        self._tool_result_since_narration: bool = True
        self._last_agent_delegate: str | None = None  # #73: delegate name stashed on an `agent`
        # call, consumed by its completion receipt (the success result carries only the summary)
        self._context_pct: float | None = None  # latest Heartbeat utilization, shown in the input bubble
        # --- Persistent type-anytime input box (start_input_box); all inert until then ---
        self._box_active: bool = False           # True while the long-lived box owns the terminal
        self._box_app: Application | None = None
        self._box_task: asyncio.Task | None = None
        self._box_patch = None                   # patch_stdout(raw=True) ctx, held for the box's life
        self._box_ticker: asyncio.Task | None = None
        self._box_working: bool = False          # in-frame spinner glyph on (thinking/streaming/burst)
        self._queued_count: int = 0              # `queued (N)` shown in the box frame
        self._decision_flash: str = ""           # transient routing-decision line in the box frame
        self._decision_flash_task: asyncio.Task | None = None
        self._first_box_hint: str = ""           # #49 guidance hint, shown in the box until first use
        self._action_handle = None
        self._observation_handle = None
        self._task_complete_handle = None
        self._escalation_handle = None
        self._heartbeat_handle = None
        self._turn_failed_handle = None
        self._consolidation_started_handle = None
        self._consolidation_finished_handle = None

    async def start(self) -> None:
        """Initialize input history and subscribe to bus events."""
        history_path = os.path.expanduser(self._history_file)
        os.makedirs(os.path.dirname(history_path), exist_ok=True)

        self._history = FileHistory(history_path)

        self._action_handle = self.bus.subscribe(Action, self.on_action)
        self._observation_handle = self.bus.subscribe(Observation, self.on_observation)
        self._task_complete_handle = self.bus.subscribe(TaskComplete, self.on_task_complete)
        self._turn_failed_handle = self.bus.subscribe(TurnFailed, self.on_turn_failed)
        self._escalation_handle = self.bus.subscribe(Escalation, self.on_escalation)
        self._heartbeat_handle = self.bus.subscribe(Heartbeat, self.on_heartbeat)
        self._consolidation_started_handle = self.bus.subscribe(
            ConsolidationStarted, self.on_consolidation_started
        )
        self._consolidation_finished_handle = self.bus.subscribe(
            ConsolidationFinished, self.on_consolidation_finished
        )

    async def stop(self) -> None:
        """Unsubscribe from bus, stop any active Live context, flush console."""
        await self.stop_input_box()  # tear the persistent box down first (idempotent)
        for handle in (
            self._action_handle,
            self._observation_handle,
            self._task_complete_handle,
            self._turn_failed_handle,
            self._escalation_handle,
            self._heartbeat_handle,
            self._consolidation_started_handle,
            self._consolidation_finished_handle,
        ):
            if handle is not None:
                self.bus.unsubscribe(handle)

        self._stop_thinking()
        self._close_burst()  # teardown: flush an open counter so the count isn't lost

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
            self._stop_thinking()
            self._close_burst()
            self._ensure_idle()
            self._tool_result_since_narration = True  # a full message closes the chunk cadence
            if agent_id:
                self._console.print(Panel(
                    Markdown(content),
                    title=f"[agent.name]{escape(agent_id)}[/agent.name]",
                    border_style="cyan",
                ))
            else:
                self._console.print(escape(content))

    async def send_streaming(
        self,
        token_stream: AsyncIterator[str],
        agent_id: str | None = None,
    ) -> str:
        """Stream tokens to terminal using Rich Live. Returns full assembled text."""
        full_text = ""
        async with self._output_lock:
            self._stop_thinking()
            self._close_burst()  # rich allows one live display — freeze before Live starts
            self._state = "STREAMING"
            panel_title = f"[agent.name]{escape(agent_id or 'agent')}[/agent.name] [muted]streaming...[/muted]"
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

            # Final non-live panel with green border — render markdown (tables,
            # headers, bold) instead of raw text so the answer reads cleanly.
            self._console.print(Panel(
                Markdown(full_text),
                title=f"[agent.name]{escape(agent_id or 'agent')}[/agent.name]",
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
        """Display tool invocation inline. Format: ◆ tool_name key_arg

        Family tools (_BURST_GROUPS) don't print per-call: they open or extend a
        consolidated burst counter that freezes to one line when the burst ends."""
        async with self._output_lock:
            self._stop_thinking()
            group = next((g for g in _BURST_GROUPS if tool_name in g[0]), None)
            if group is None:
                self._close_burst()
                if tool_name == "agent":
                    # Stash the delegate for the completion receipt (send_tool_result): a
                    # SUCCESS Observation carries only the summary, not who ran it (#73).
                    d = arguments.get("agent_id")
                    self._last_agent_delegate = d if isinstance(d, str) else None
                summary = _tool_call_summary(tool_name, arguments)
                # no_wrap + ellipsis: a long preview stays ONE physical line (cropped to
                # the terminal), never wrapping the view open (FIX B).
                self._console.print(
                    f"  [tool.call]{_DIAMOND} {escape(summary)}[/tool.call]",
                    no_wrap=True, overflow="ellipsis",
                )
                return
            family, note = group
            if self._burst is None or self._burst.family is not family:
                self._close_burst()
                self._burst = _Burst(family=family, close_note=note)
            burst = self._burst
            if tool_name not in burst.tools:
                burst.tools.append(tool_name)
            burst.calls += 1
            self._burst_refresh(burst)

    async def send_tool_result(
        self,
        tool_name: str,
        result: str,
        is_error: bool,
        agent_id: str | None = None,
    ) -> None:
        """Display tool result inline. Uses checkmark for success, cross for error.

        Results belonging to the open burst tick its counter instead of printing;
        errors are absorbed into the count and annotated on the frozen line (full
        detail stays on the bus ledger)."""
        lines = result.strip().split("\n") if result else [""]
        async with self._output_lock:
            self._stop_thinking()
            self._tool_result_since_narration = True  # a rendered result re-opens narration
            burst = self._burst
            if burst is not None and tool_name in burst.family:
                burst.done += 1
                if is_error:
                    burst.errors += 1
                self._burst_refresh(burst)
                return
            self._close_burst()
            # #73: delegation-outcome receipt. Every `agent` completion prints a truthful,
            # system-style line whose status is taken from the TOOL RESULT (is_error), not the
            # model's later narration, and whose name is the delegate stashed from this call —
            # so a failed delegation can't be re-narrated as the subagent's own answer.
            if tool_name == "agent":
                delegate, self._last_agent_delegate = self._last_agent_delegate, None
                who = f" {escape(delegate)}" if delegate else ""
                if is_error:
                    detail = escape(lines[0].removeprefix("[tool error] ")[:120]) if lines else ""
                    self._console.print(f"  [tool.error]{_DIAMOND} agent{who} — FAILED: {detail}[/tool.error]")
                else:
                    self._console.print(f"  [tool.call]{_DIAMOND} agent{who} — completed[/tool.call]")
                return
            name = escape(tool_name)
            if is_error:
                first = escape(lines[0][:120]) if lines else ""
                self._console.print(f"  [tool.error]{_CROSS} {name} (exit 1): {first}[/tool.error]")
            elif tool_name in _FILE_WRITE_TOOLS:
                # result is a status line ("Written N bytes"), not the lines written —
                # omit the count rather than print a misleading one (FIX C).
                self._console.print(f"  [tool.result]{_CHECK} {name}[/tool.result]")
            else:
                self._console.print(f"  [tool.result]{_CHECK} {name} ({len(lines)} lines)[/tool.result]")

    async def send_error(
        self,
        error: str,
        detail: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Write error to stderr console."""
        async with self._output_lock:
            self._stop_thinking()
            self._close_burst()
        self._err_console.print(f"[system.error]Error:[/system.error] {escape(error)}")
        if detail:
            for line in detail.split("\n"):
                self._err_console.print(f"  {escape(line)}")

    async def read_input(self, prompt: str = ">") -> str:
        """Read a line from the user inside a rounded input bubble. Raises ChannelStartError if not started."""
        if self._history is None:
            raise ChannelStartError("TerminalChannel.start() must be called before read_input()")
        async with self._output_lock:
            self._stop_thinking()
            self._close_burst()  # freeze any open counter before the input bubble draws
        self._state = "WAITING_INPUT"
        # #49: the first prompt carries the guidance hint in the bubble's bottom border;
        # consume it so later prompts don't repeat it.
        hint, self.first_prompt_hint = self.first_prompt_hint, ""
        app = _build_input_app(
            self._history, prompt, hint=hint, context_pct=self._context_pct,
        )
        try:
            line = await app.run_async()
            self._sigint_armed = False
            return (line or "").strip()
        except KeyboardInterrupt:
            if self._sigint_armed:
                raise EOFError  # second consecutive Ctrl+C on an empty line → exit the REPL
            self._sigint_armed = True
            self._console.print("[system.info](Press Ctrl+C again to exit)[/system.info]")
            return ""
        finally:
            self._state = "IDLE"

    # ------------------------------------------------------------------ #
    # Persistent type-anytime input box
    # ------------------------------------------------------------------ #

    def can_run_input_box(self) -> bool:
        """The persistent box needs a real interactive TTY. Non-TTY (pipes, CI, capture)
        falls back to the classic read_input sequencing."""
        return bool(self._console.is_terminal)

    async def start_input_box(
        self, ctrl_queue: asyncio.Queue, on_interrupt: Callable[[], None]
    ) -> None:
        """Launch the long-lived input box as a sibling task, under patch_stdout(raw=True) so
        all turn output streams ABOVE it. Enter enqueues ('submit', text) onto ctrl_queue;
        Ctrl+D (empty) enqueues ('eof', None); Ctrl+C (empty) calls on_interrupt() — the REPL
        owns those policies. raw=True is load-bearing (rich ANSI is interpreted, not escaped)."""
        if self._history is None:
            raise ChannelStartError("TerminalChannel.start() must be called before start_input_box()")
        from prompt_toolkit.patch_stdout import patch_stdout

        # #49: the first prompt carries the guidance hint inside the box; consume it once.
        self._first_box_hint, self.first_prompt_hint = self.first_prompt_hint, ""
        self._box_active = True

        def _on_submit(text: str) -> None:
            ctrl_queue.put_nowait(("submit", text))

        def _on_eof() -> None:
            ctrl_queue.put_nowait(("eof", None))

        self._box_app = _build_persistent_input_app(
            self._history, ">",
            on_submit=_on_submit, on_interrupt=on_interrupt, on_eof=_on_eof,
            hint_fn=self._box_hint_frags, pct_fn=lambda: self._context_pct,
        )
        self._box_patch = patch_stdout(raw=True)
        self._box_patch.__enter__()
        self._box_task = asyncio.create_task(self._box_app.run_async())
        self._box_ticker = asyncio.create_task(self._box_tick())

    async def stop_input_box(self) -> None:
        """Tear the box down: stop the ticker, exit the app, release patch_stdout. Idempotent."""
        self._box_active = False
        for task_attr in ("_box_ticker", "_decision_flash_task"):
            task = getattr(self, task_attr)
            if task is not None:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                setattr(self, task_attr, None)
        if self._box_app is not None:
            try:
                if self._box_app.is_running:
                    self._box_app.exit()
            except Exception:
                pass
        if self._box_task is not None:
            try:
                await self._box_task
            except (asyncio.CancelledError, EOFError, KeyboardInterrupt, Exception):
                pass
            self._box_task = None
        if self._box_patch is not None:
            try:
                self._box_patch.__exit__(None, None, None)
            except Exception:
                pass
            self._box_patch = None
        self._box_app = None

    def _box_hint_frags(self) -> list[tuple[str, str]]:
        """Dynamic bottom-border content: an animated working glyph, the transient routing
        decision, the persistent `queued (N)`, or the first-run guidance hint."""
        parts: list[tuple[str, str]] = []
        if self._box_working:
            glyph = _SPIN_FRAMES[int(time.monotonic() * 8) % len(_SPIN_FRAMES)]
            parts.append(("class:hint", f" {glyph} working "))
        if self._decision_flash:
            parts.append(("class:caret", f" {self._decision_flash} "))
        elif self._queued_count:
            parts.append(("class:hint", f" queued ({self._queued_count}) "))
        elif not self._box_working and self._first_box_hint:
            parts.append(("class:hint", f" {self._first_box_hint} "))
        if not parts:
            parts.append(("class:hint", "  "))
        return parts

    def _invalidate_box(self) -> None:
        if self._box_app is not None and self._box_active:
            try:
                self._box_app.invalidate()
            except Exception:
                pass

    def box_set_queued(self, n: int) -> None:
        """Persistent `queued (N)` count shown in the box frame."""
        self._queued_count = max(0, n)
        self._invalidate_box()

    def box_flash_decision(self, text: str, seconds: float = 2.0) -> None:
        """Show a transient routing-decision line ('→ nudging current turn' / 'queued (N)')
        the instant Enter is routed; it clears itself after `seconds`."""
        self._decision_flash = text
        self._invalidate_box()
        if self._decision_flash_task is not None:
            self._decision_flash_task.cancel()
        try:
            self._decision_flash_task = asyncio.get_running_loop().create_task(
                self._clear_flash_after(seconds, text)
            )
        except RuntimeError:
            self._decision_flash_task = None  # no loop (unit context): flash persists till next change

    async def _clear_flash_after(self, seconds: float, text: str) -> None:
        try:
            await asyncio.sleep(seconds)
            if self._decision_flash == text:  # not superseded by a newer decision
                self._decision_flash = ""
                self._invalidate_box()
        except asyncio.CancelledError:
            pass

    def box_notify_working(self, working: bool) -> None:
        """Toggle the in-frame working glyph (thinking/streaming/burst active)."""
        self._box_working = working
        self._invalidate_box()

    async def _box_tick(self) -> None:
        """Animate the in-frame working glyph while the box is live (no rich Live anywhere)."""
        try:
            while self._box_active:
                if self._box_working:
                    self._invalidate_box()
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    def _ensure_idle(self) -> None:
        """Log a warning if called while STREAMING (concurrent write detected)."""
        if self._state == "STREAMING":
            self._err_console.print("[warning]Warning: output during streaming state[/warning]")

    def _burst_text(self, burst: _Burst, final: bool) -> str:
        """Render the burst counter line: ◆ tools · done/calls [· N errors]."""
        head = f"{_DIAMOND} {escape(' · '.join(burst.tools))}"
        line = f"  [tool.call]{head}[/tool.call] [muted]· {burst.done}/{burst.calls}[/muted]"
        if final and burst.errors:
            plural = "s" if burst.errors > 1 else ""
            line += f" [tool.error]· {burst.errors} error{plural}[/tool.error]"
        return line

    def _burst_refresh(self, burst: _Burst) -> None:
        """Tick the live counter. TTY-only spinner; non-TTY counts silently and
        prints just the frozen line on close (captures stay one line per burst)."""
        if self._box_active:
            # Box mode: NO rich Status (it glues lines / freezes on Ctrl+C-during-burst under
            # patch_stdout). The count accumulates in the _Burst and its frozen line prints on
            # close; the live indicator is the in-frame working glyph.
            self.box_notify_working(True)
            return
        text = self._burst_text(burst, final=False)
        if burst.status is not None:
            burst.status.update(text)
        elif self._console.is_terminal and self._state == "IDLE":
            try:
                burst.status = self._console.status(text, spinner="dots")
                burst.status.start()
            except Exception:
                burst.status = None  # a broken spinner must never break output

    def _close_burst(self) -> None:
        """Freeze an open burst into its final scrollback line (+ close note).
        Call with _output_lock held, or from teardown where no writer can race."""
        burst, self._burst = self._burst, None
        if burst is None:
            return
        if burst.status is not None:
            try:
                burst.status.stop()
            finally:
                burst.status = None
        self._console.print(self._burst_text(burst, final=True))
        if burst.close_note and burst.done:
            self._console.print(f"  [tool.result]{_CHECK} {burst.close_note}[/tool.result]")

    def _start_thinking(self) -> None:
        """Animated indicator while an LLM round-trip is in flight (REPL-02).
        IDLE-only: never over the input bubble, a streaming panel, an open burst
        counter (the burst spinner is already the live indicator), or the dreaming
        status (rich allows one live display \u2014 a turn beginning stops it first)."""
        if self._box_active:
            # Box mode: in-frame glyph, never a rich Status (freeze-safe under patch_stdout).
            self.box_notify_working(True)
            return
        if (self._thinking is None and self._dreaming is None
                and self._burst is None and self._state == "IDLE"):
            try:
                self._thinking = self._console.status(
                    "[muted]thinking\u2026[/muted]", spinner="dots"
                )
                self._thinking.start()
            except Exception:
                self._thinking = None  # a broken spinner must never break output

    def _start_dreaming(self) -> None:
        """Quiet '\u00b7 dreaming\u2026' status while a background memory consolidation/mining
        pass runs (#20). Extends _start_thinking: same console.status, IDLE-only \u2014 so it can
        never draw over the input bubble (WAITING_INPUT) or a stream, and never opens a second
        live display alongside the thinking spinner or a burst counter."""
        if self._box_active:
            return  # box mode: no rich Status; background dreaming isn't animated in-frame (v1)
        if (self._dreaming is None and self._thinking is None
                and self._burst is None and self._state == "IDLE"):
            try:
                self._dreaming = self._console.status(
                    f"[muted]{_DREAMING_LABEL}[/muted]", spinner="dots"
                )
                self._dreaming.start()
            except Exception:
                self._dreaming = None  # a broken spinner must never break output

    def _stop_dreaming(self) -> None:
        if self._dreaming is not None:
            try:
                self._dreaming.stop()
            finally:
                self._dreaming = None

    def _stop_thinking(self) -> None:
        if self._box_active:
            self.box_notify_working(False)  # box mode: drop the in-frame glyph
        if self._thinking is not None:
            try:
                self._thinking.stop()
            finally:
                self._thinking = None
        self._stop_dreaming()  # #20: real output / input tears the dreaming dot down too (stop-first)

    async def on_action(self, event: Action) -> None:
        """Render interstitial narration for an llm_response that ALSO made tool calls, then
        fall through to the base handler for tool_call display. A tool-less llm_response is
        the final answer (rendered by the TaskComplete panel) — the has_tool_calls
        discriminator keeps it from being echoed here (the double-print regression)."""
        if event.action_type == "llm_response":
            await self._render_narration(event)
            return
        await super().on_action(event)

    async def _render_narration(self, event: Action) -> None:
        """One dim line opening a chunk of tool activity. No-op unless the response also made
        tool calls (final answers render elsewhere), its content has a non-empty first line,
        and a tool result has rendered since the last narration — the cadence guard: at most
        one narration line per chunk, never two dim lines in a row."""
        if not event.has_tool_calls:
            return
        line = _narration_line(event.content)
        if not line:
            return
        async with self._output_lock:
            if not self._tool_result_since_narration:
                return
            self._stop_thinking()
            self._close_burst()  # freeze the previous chunk's counter before this one opens
            self._tool_result_since_narration = False
            self._console.print(
                f"  [muted]{_NARRATE} {escape(line)}[/muted]",
                no_wrap=True, overflow="ellipsis",
            )

    async def on_heartbeat(self, event: Heartbeat) -> None:
        """Track context utilization (shown in the next input bubble) and start the
        thinking indicator \u2014 the Heartbeat fires right before each LLM call, so this
        is the 'model is now generating' signal (REPL-02)."""
        self._context_pct = event.context_utilization_pct
        async with self._output_lock:
            self._stop_dreaming()  # a turn is beginning — replace the dreaming dot with thinking
            self._start_thinking()

    async def on_consolidation_started(self, event: ConsolidationStarted) -> None:
        """A background memory consolidation/mining pass began (#20): show the quiet
        '· dreaming…' status if the REPL is idle (never over the input bubble or a stream)."""
        async with self._output_lock:
            self._start_dreaming()

    async def on_consolidation_finished(self, event: ConsolidationFinished) -> None:
        """The background pass ended (#20): clear the dreaming status. Touches only the
        dreaming slot, so a thinking spinner started by a turn mid-pass is left intact."""
        async with self._output_lock:
            self._stop_dreaming()
