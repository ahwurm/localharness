"""The clipboard's image, when this process can reach a clipboard at all.

A terminal cannot paste a picture: the terminal emulator turns Ctrl+V into text or nothing, so
the harness asks the OS clipboard directly, the way Claude Code does — ``wl-paste`` on Wayland,
``xclip`` on X11, ``pngpaste`` or ``osascript`` on macOS. Over SSH there is no clipboard to ask:
the picture sits on the machine the user is typing on, and this process runs on another. That
case is reported in those words rather than as "no image", because the fix is different (drop
the file onto the terminal so its path is pasted, or ``/image <path>``).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

from localharness.core.content import ImageError

NO_DISPLAY = (
    "no clipboard is reachable from this session (no DISPLAY or WAYLAND_DISPLAY — over SSH the "
    "clipboard lives on your own machine). Drop the file onto the terminal, or /image <path>"
)


def _commands() -> list[list[str]]:
    if sys.platform == "darwin":
        return [["pngpaste", "-"],
                ["osascript", "-e", "the clipboard as «class PNGf»"]]
    if os.environ.get("WAYLAND_DISPLAY"):
        return [["wl-paste", "--no-newline", "-t", "image/png"]]
    if os.environ.get("DISPLAY"):
        return [["xclip", "-selection", "clipboard", "-t", "image/png", "-o"]]
    raise ImageError(NO_DISPLAY)


def read_clipboard_image() -> bytes:
    """PNG bytes from the clipboard, or :class:`ImageError` saying exactly why not."""
    commands = _commands()
    tools = [c[0] for c in commands]
    for cmd in commands:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            run = subprocess.run(cmd, capture_output=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ImageError(f"{cmd[0]} failed: {exc}") from None
        out = run.stdout
        if cmd[0] == "osascript":   # «data PNGf89504E47…» — hex inside the AppleScript literal
            text = out.decode("utf-8", "replace").strip()
            out = bytes.fromhex(text[len("«data PNGf"):-1]) if text.startswith("«data PNGf") else b""
        if run.returncode == 0 and out:
            return out
        raise ImageError("the clipboard holds no image (copy a screenshot first)")
    raise ImageError(f"no clipboard tool found — install {' or '.join(tools)}, or /image <path>")
