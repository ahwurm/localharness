"""CLI presentation — startup banner with the LocalHarness snail mascot."""
from __future__ import annotations

from importlib import metadata
from pathlib import Path

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Sprite cells map to (glyph, style): B=shell, C=spiral ring, Y=body, D=eye dots.
_ON_HEAD = "grey15 on navajo_white1"
_GLYPHS = {
    "B": (" ", "on tan"),            # shell
    "C": (" ", "on grey27"),         # spiral ring
    "Y": (" ", "on navajo_white1"),  # body
    "D": (" ", "on grey15"),         # eyes on stalks
    "(": ("╰", _ON_HEAD),            # smile
    ")": ("╯", _ON_HEAD),
}
_SPRITE = [
    " D  D",
    " Y  Y     BBBB",
    " YYYY    BCCCCB",
    "Y()YY   BBCBBCBB",
    "YYYYY   BBCCCCBB",
    "YYYYYYYYYYYYYYYY",
]
_ZZZ = {1: "  Z", 2: " z"}


def mascot() -> Text:
    """The mascot: a solid blocky snail (Clawd-style) — local, slow, steady."""
    t = Text()
    for i, row in enumerate(_SPRITE):
        for ch in row:
            glyph, style = _GLYPHS.get(ch, (" ", None))
            t.append(glyph, style=style)
        if i in _ZZZ:
            t.append(_ZZZ[i], style="dim cyan")
        t.append("\n")
    t.append(" ")
    t.append("· · ·", style="dim")  # slime trail
    return t


def startup_banner(model: str, is_returning: bool) -> Panel:
    """Rounded panel with the snail, wordmark, model, and cwd."""
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
    grid.add_row(mascot(), info)
    return Panel(grid, border_style="dim cyan", expand=False, padding=(0, 2))
