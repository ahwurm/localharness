"""CLI presentation — startup banner: the local_harness wordmark."""
from __future__ import annotations

from importlib import metadata
from pathlib import Path

from rich.console import Group, RenderableType
from rich.text import Text

# Accent = site token oklch(80% 0.17 152). Wordmark is figlet "ANSI Shadow" for
# "local_" / "harness", hardcoded so the banner needs no figlet dep at runtime.
_GREEN = "#56dc85"
_WORDMARK = '██╗      ██████╗  ██████╗ █████╗ ██╗             \n██║     ██╔═══██╗██╔════╝██╔══██╗██║             \n██║     ██║   ██║██║     ███████║██║             \n██║     ██║   ██║██║     ██╔══██║██║             \n███████╗╚██████╔╝╚██████╗██║  ██║███████╗███████╗\n╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝\n                                                 \n██╗  ██╗ █████╗ ██████╗ ███╗   ██╗███████╗███████╗███████╗\n██║  ██║██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔════╝\n███████║███████║██████╔╝██╔██╗ ██║█████╗  ███████╗███████╗\n██╔══██║██╔══██║██╔══██╗██║╚██╗██║██╔══╝  ╚════██║╚════██║\n██║  ██║██║  ██║██║  ██║██║ ╚████║███████╗███████║███████║\n╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚══════╝\n                                                          '


def startup_banner(model: str, is_returning: bool) -> RenderableType:
    """The local_harness wordmark in green, with model and cwd."""
    try:
        version = metadata.version("localharness")
    except metadata.PackageNotFoundError:
        version = "dev"
    cwd = str(Path.cwd())
    home = str(Path.home())
    if cwd.startswith(home):
        cwd = "~" + cwd[len(home):]

    art = Text(_WORDMARK, style=f"bold {_GREEN}")
    info = Text()
    info.append(f"v{version}", style="dim")
    info.append(f"    {model}")
    info.append(f"    {cwd}", style="dim")
    if not is_returning:
        info.append("\n\nDescribe a task, or ", style="dim")
        info.append("/help", style=_GREEN)
        info.append(" for commands.", style="dim")
    return Group(art, Text(), info)
