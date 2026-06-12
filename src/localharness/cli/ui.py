"""CLI presentation — startup banner with the LocalHarness sloth mascot."""
from __future__ import annotations

from importlib import metadata
from pathlib import Path

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Sprite cells map to (glyph, style): S=fur, F=cream face, space=transparent.
_ON_FACE = "grey15 on navajo_white1"
_GLYPHS = {
    "S": (" ", "on tan"),
    "F": (" ", "on navajo_white1"),
    "~": ("~", _ON_FACE),   # zen eyes
    "(": ("╰", _ON_FACE),   # smile
    ")": ("╯", _ON_FACE),
}
_SPRITE = [
    "  SSSSSSSSSSSS",    # head top
    "  SFF~FFFF~FFS",    # zen eyes
    "  SFFFF()FFFFS",    # sloth smile
    "SSSSSSSSSSSSSSSS",  # arms out
    "   SS  SS  SS",     # legs
]
_ZZZ = {0: "   Z", 1: "  z"}


def sloth() -> Text:
    """The mascot: a solid blocky sloth (Clawd-style), asleep on the job (it's local)."""
    t = Text()
    for i, row in enumerate(_SPRITE):
        for ch in row:
            glyph, style = _GLYPHS.get(ch, (" ", None))
            t.append(glyph, style=style)
        if i in _ZZZ:
            t.append(_ZZZ[i], style="dim cyan")
        t.append("\n")
    t.append("   ")
    t.append("▀▀  ▀▀  ▀▀", style="tan")  # long claw tips
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
