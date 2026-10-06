"""The ONE slash-command table (PAPI-07).

Four consumers read it, so they can never drift: the REPL dispatcher (`OrchestratorREPL._handle_slash`
via `find_row`), `/help` (`help_text`), the input completion menu
(`channels.terminal.SlashCommandCompleter`) and the phone's command menu (`/api/protocol`
`commands[]`). Each reads `all_rows()` when it is used, so rows added after import show everywhere.
Order here is the display order in all of them; plugin rows follow the core rows.

Plugins append rows through the session lifecycle (`set_plugin_rows`) — never by editing this file.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SlashCommand:
    """One row of the slash table. Unpacks as (name, description) — the display pair every consumer
    of the old two-tuple list reads, so /help, the completer and the phone read rows unchanged.

    `handler` is an OrchestratorREPL method name for a core row, and a bound async callable for a
    plugin row (called with the text after the name; the text — or rich renderable — it returns is
    shown to the user).
    `takes_args` rows claim "name ..." as well as "name".
    `terminal_only` rows are left out of the phone's menu; typed on another channel they answer one line."""

    name: str
    description: str
    handler: str | Callable[[str], Awaitable[Any]]
    takes_args: bool = False
    plugin: str | None = None
    terminal_only: bool = False

    def __iter__(self) -> Iterator[str]:
        return iter((self.name, self.description))


SLASH_COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("/help", "Show this help message", "_slash_help"),
    SlashCommand("/agents", "List configured agents", "_slash_agents"),
    SlashCommand("/model", "List available models; /model <name|number> to switch", "_slash_model", True),
    SlashCommand("/reasoning", "Stream the model's reasoning while it thinks; /reasoning on|off",
                 "_slash_reasoning", True),
    SlashCommand("/verbose", "Show reasoning and every tool call with its arguments; /verbose on|off",
                 "_slash_verbose", True),
    SlashCommand("/mode", "Permission mode for this session; /mode guarded|trusted|read-only",
                 "_slash_mode", True),
    SlashCommand("/pending", "Tool calls parked for you to answer", "_slash_pending"),
    SlashCommand("/approve", "Run a parked call; /approve [N] (default: the oldest)", "_slash_approve", True),
    SlashCommand("/deny", "Drop a parked call; /deny [N] (default: the oldest)", "_slash_deny", True),
    SlashCommand("/task", "Show the working task record; /task clear to delete it", "_slash_task", True),
    SlashCommand("/image", "Attach a picture to your next message; /image <path>, /image (the clipboard), "
                 "/image clear — or drop the file onto the terminal, or Ctrl+V",
                 "_slash_image", True, terminal_only=True),
    SlashCommand("/plugins", "List plugins; /plugins enable|disable <name> to switch one in this session",
                 "_slash_plugins", True, terminal_only=True),
    SlashCommand("/quit", "Exit LocalHarness", "_slash_quit"),
    SlashCommand("/exit", "Exit LocalHarness", "_slash_quit"),
)
_plugin_rows: tuple[SlashCommand, ...] = ()


def set_plugin_rows(rows: Iterable[SlashCommand]) -> list[str]:
    """Replace every plugin row (`()` removes them all). A row whose name is already taken — by core
    or an earlier plugin row — is skipped, and so is one without a callable handler (only core rows
    name a REPL method). Returns one warning per skipped row, naming its plugin."""
    global _plugin_rows
    kept: list[SlashCommand] = []
    warnings: list[str] = []
    for row in rows:
        if row.name in {r.name for r in (*SLASH_COMMANDS, *kept)}:
            warnings.append(f"plugin {row.plugin}: slash command {row.name} is already taken — skipped")
        elif not callable(row.handler):
            warnings.append(f"plugin {row.plugin}: slash command {row.name} has no callable handler — skipped")
        else:
            kept.append(row)
    _plugin_rows = tuple(kept)
    return warnings


def all_rows() -> tuple[SlashCommand, ...]:
    """Core rows, then plugin rows — what every consumer reads, at the moment it reads."""
    return SLASH_COMMANDS + _plugin_rows


def find_row(lowered: str) -> SlashCommand | None:
    """The row a lower-cased, stripped input line invokes: its exact name, or its name and a space
    for a row that takes arguments. None when no row claims the line."""
    return next((row for row in all_rows() if lowered == row.name
                 or (row.takes_args and lowered.startswith(row.name + " "))), None)


def help_text() -> str:
    """Render the /help body from `all_rows()`."""
    rows = all_rows()
    width = max(len(name) for name, _ in rows)
    lines = ["Available commands:"]
    for name, desc in rows:
        lines.append(f"  {name.ljust(width)}  {desc}")
    lines.append("")
    lines.append("Everything else is handled by the orchestrator through natural language.")
    return "\n".join(lines)
