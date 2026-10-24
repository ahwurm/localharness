"""The fence around project text: a per-process tag no file can predict, and the one rule that keeps text inside it."""
from __future__ import annotations

import secrets

# A file cannot predict the tag, and a file that learns it later (this process wrote it somewhere the
# model can read) has changed its own text, so the closing marker cannot be forged in advance.
TAG = secrets.token_hex(16)


def fence_project_text(header: str, text: str) -> str:
    """The header, then the text between this process's tagged markers, any line that closes the fence replaced."""
    opening = f"<project-instructions {TAG}>"
    closing = f"</project-instructions {TAG}>"
    lines = ("[closing marker removed]" if line.strip() == closing else line for line in text.splitlines())
    return "\n".join((header, opening, *lines, closing))
