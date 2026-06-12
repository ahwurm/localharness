"""CLI presentation — startup banner with the LocalHarness sloth mascot."""
from __future__ import annotations

from importlib import metadata
from pathlib import Path

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_FUR = "on tan"          # solid body cells (background color, Clawd-style)
_TIPS = "tan"            # foreground half-blocks for claw tips
_EYES = "bold grey15 on tan"
_ZZZ = "dim cyan"


def sloth() -> Text:
    """The mascot: a solid blocky sloth (Clawd-style), asleep on the job (it's local)."""
    t = Text()
    # head top
    t.append("  ")
    t.append(" " * 12, style=_FUR)
    t.append("   Z\n", style=_ZZZ)
    # face: wide-set sleepy eyes
    t.append("  ")
    t.append("   ", style=_FUR)
    t.append("-", style=_EYES)
    t.append("    ", style=_FUR)
    t.append("-", style=_EYES)
    t.append("   ", style=_FUR)
    t.append("  z\n", style=_ZZZ)
    # arms out
    t.append(" " * 16, style=_FUR)
    t.append("\n")
    # body
    t.append("  ")
    t.append(" " * 12, style=_FUR)
    t.append("\n")
    # three legs
    t.append("   ")
    t.append("  ", style=_FUR)
    t.append("  ")
    t.append("  ", style=_FUR)
    t.append("  ")
    t.append("  ", style=_FUR)
    t.append("\n")
    # long claw tips
    t.append("   ")
    t.append("▀▀", style=_TIPS)
    t.append("  ")
    t.append("▀▀", style=_TIPS)
    t.append("  ")
    t.append("▀▀", style=_TIPS)
    return t


def startup_banner(model: str, is_returning: bool) -> Panel:
    """Rounded panel with the sloth, wordmark, model, and cwd."""
    try:
        version = metadata.version("localharness")
    except metadata.PackageNotFoundError:
        version = "dev"
    cwd = str(Path.cwd())
    home = str(Path.home())
    if cwd.startswith(home):
        cwd = "~" + cwd[len(home):]

    info = Text()
    info.append("LocalHarness", style="bold cyan")
    info.append(f" v{version}\n", style="dim")
    info.append(f"{model}\n")
    info.append(cwd, style="dim")
    if not is_returning:
        info.append("\n\nDescribe a task, or /help for commands.", style="dim")

    grid = Table.grid(padding=(0, 4))
    grid.add_column(vertical="middle")
    grid.add_column(vertical="middle")
    grid.add_row(sloth(), info)
    return Panel(grid, border_style="dim cyan", expand=False, padding=(0, 2))
