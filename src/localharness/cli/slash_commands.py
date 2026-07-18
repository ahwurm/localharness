"""Single source of truth for the REPL's slash commands.

Both /help (repl.HELP_TEXT) and the input completion menu (channels.terminal.SlashCommandCompleter)
read this one table, so they can never drift. Order here is the display order in both surfaces.
"""
from __future__ import annotations

# (name, one-line description). The name includes its leading slash.
SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help", "Show this help message"),
    ("/agents", "List configured agents"),
    ("/model", "List available models; /model <name|number> to switch"),
    ("/memory", "Browse the agent's memory by tag; show/forget/search a memory"),
    ("/quit", "Exit LocalHarness"),
    ("/exit", "Exit LocalHarness"),
]


def help_text() -> str:
    """Render the /help body from SLASH_COMMANDS (single source of truth)."""
    width = max(len(name) for name, _ in SLASH_COMMANDS)
    lines = ["Available commands:"]
    for name, desc in SLASH_COMMANDS:
        lines.append(f"  {name.ljust(width)}  {desc}")
    lines.append("")
    lines.append("Everything else is handled by the orchestrator through natural language.")
    return "\n".join(lines)
