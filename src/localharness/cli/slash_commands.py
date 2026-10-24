"""The ONE slash-command table (PAPI-07).

Four consumers read it, so they can never drift: the REPL dispatcher (`OrchestratorREPL._handle_slash`
via `find_row`), `/help` (`help_text`), the input completion menu
(`channels.terminal.SlashCommandCompleter`) and the phone's command menu (`/api/protocol`
`commands[]`). Each reads `all_rows()` when it is used, so rows added after import show everywhere.
Order here is the display order in all of them; plugin rows follow the core rows.

Plugins append rows through the session lifecycle (`set_plugin_rows`, and `add_plugin_row` for a row added
after start) — never by editing this file.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SlashCommand:
    """One row of the slash table. Unpacks as (name, description) — the display pair every consumer
    of the old two-tuple list reads, so /help, the completer and the phone read rows unchanged.

    `handler` is an OrchestratorREPL method name for a core row (`plugin` None), called with the text after
    the name as typed and lower-cased. A plugin-table row names a REPL method called with the row and the
    stripped text, or carries a bound async callable called with the stripped text; the text — or rich
    renderable — the callable returns is shown to the user.
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
    SlashCommand("/effort", "Reasoning effort for this session; /effort off|low|high|max, /effort show",
                 "_slash_effort", True),
    SlashCommand("/verbose", "Show reasoning and every tool call with its arguments; /verbose on|off",
                 "_slash_verbose", True),
    SlashCommand("/mode", "Permission mode for this session; /mode guarded|trusted|read-only",
                 "_slash_mode", True),
    SlashCommand("/pending", "Tool calls parked for you to answer", "_slash_pending"),
    SlashCommand("/approve", "Run a parked call; /approve [N] (default: the oldest)", "_slash_approve", True),
    SlashCommand("/deny", "Drop a parked call; /deny [N] (default: the oldest)", "_slash_deny", True),
    SlashCommand("/clear", "Start a fresh conversation; settings, memory and the task record stay", "_slash_clear"),
    SlashCommand("/task", "Show the working task record; /task clear to delete it", "_slash_task", True),
    SlashCommand("/instructions", "The project's instructions file (CLAUDE.md, AGENTS.md); "
                 "/instructions off|on|show", "_slash_instructions", True),
    SlashCommand("/image", "Attach a picture to your next message; /image <path>, /image (the clipboard), "
                 "/image clear — or drop the file onto the terminal, or Ctrl+V",
                 "_slash_image", True, terminal_only=True),
    SlashCommand("/plugins", "Manage plugins; /plugins enable|disable <name> switches one from the keyboard",
                 "_slash_plugins", True, terminal_only=True),
    SlashCommand("/mcp", "Manage MCP servers for the session's agent (list, add, edit, test, remove)",
                 "_slash_mcp", True, terminal_only=True),
    SlashCommand("/quit", "Exit LocalHarness", "_slash_quit"),
    SlashCommand("/exit", "Exit LocalHarness", "_slash_quit"),
)
_plugin_rows: tuple[SlashCommand, ...] = ()


_KINDS: dict[str, tuple[str, str]] = {
    "skill": ("skill {bare}", 'say "use the {bare} skill", or rename it'),
}


def _refusal(row: SlashCommand, taken: set[str]) -> str | None:
    who, fix = _KINDS.get(row.plugin, ("plugin {plugin}", "skipped"))
    who, fix = (t.format(plugin=row.plugin, bare=row.name[1:]) for t in (who, fix))
    if row.name in taken:
        return f"{who}: slash command {row.name} is already taken — {fix}"
    if not (isinstance(row.handler, str) or callable(row.handler)):
        return f"{who}: slash command {row.name} has no callable handler — skipped"
    return None


def set_plugin_rows(rows: Iterable[SlashCommand]) -> list[str]:
    """Replace every plugin-table row (`()` removes them all). A row with a taken name, or
    whose handler is neither a REPL method name nor a callable, is skipped; one warning per skipped
    row, worded by the row's kind."""
    global _plugin_rows
    kept: list[SlashCommand] = []
    warnings: list[str] = []
    for row in rows:
        warning = _refusal(row, {r.name for r in (*SLASH_COMMANDS, *kept)})
        if warning is None:
            kept.append(row)
        else:
            warnings.append(warning)
    _plugin_rows = tuple(kept)
    return warnings


def add_plugin_row(row: SlashCommand) -> str | None:
    """Append one row after start under the same check as `set_plugin_rows`; the warning, or None."""
    global _plugin_rows
    warning = _refusal(row, {r.name for r in all_rows()})
    if warning is None:
        _plugin_rows = (*_plugin_rows, row)
    return warning


def all_rows() -> tuple[SlashCommand, ...]:
    """Core rows, then plugin rows — what every consumer reads, at the moment it reads."""
    return SLASH_COMMANDS + _plugin_rows


def find_row(lowered: str) -> SlashCommand | None:
    """The row a lower-cased, stripped input line invokes: its exact name, or its name and a space
    for a row that takes arguments. None when no row claims the line."""
    return next((row for row in all_rows() if lowered == row.name
                 or (row.takes_args and lowered.startswith(row.name + " "))), None)


def help_text(keys_block: str | None = None) -> str:
    """Render the /help body from `all_rows()`.

    `keys_block`, when given, is printed after the closing sentence, one blank line between
    (spec 21 section 6.7). The REPL passes it only while the persistent input box is live, so
    every other surface reads as it always did."""
    rows = all_rows()
    width = max(len(name) for name, _ in rows)
    lines = ["Available commands:"]
    for name, desc in rows:
        lines.append(f"  {name.ljust(width)}  {desc}")
    lines.append("")
    lines.append("Everything else is handled by the orchestrator through natural language.")
    if keys_block:
        lines.append("")
        lines.append(keys_block)
    return "\n".join(lines)
