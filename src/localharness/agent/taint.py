"""Which agents have read pages saved from the web, and what they wrote since."""
from __future__ import annotations

from pathlib import Path

from localharness.agent.gate_types import TOOL_SCRIPT_UNCONFIRMED
from localharness.config.paths import WORKSPACE_DIR_NAME

DOWNLOADS = (WORKSPACE_DIR_NAME, "downloads")

TAINTED_ASK_CLASSES = frozenset({
    "shell-destructive", "interpreter-inline", "code-exec", TOOL_SCRIPT_UNCONFIRMED, "edit-outside",
})


def downloads_dir(boundary: Path) -> Path:
    return Path(boundary).joinpath(*DOWNLOADS)


def under_downloads(path: str | Path) -> bool:
    """Does `path`, symlinks resolved, sit in a downloads folder?"""
    try:
        parts = Path(path).expanduser().resolve().parts
    except (OSError, ValueError, RuntimeError):
        parts = Path(path).parts
    return any(parts[i:i + 2] == DOWNLOADS for i in range(len(parts) - 1))
