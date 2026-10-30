"""An agent's interactive browser session: the snapshot window and the browser_open, click, type and scroll tools."""
from __future__ import annotations

import re

WINDOW_CHARS = 8000
UNCHANGED_CHARS = 1500
CONTEXT_LINES = 2

CHANGED_NOTE = ('[showing from near the first change; earlier lines are unchanged. '
                'Refs outside this window: web_page_query("{fid}", "next") or ("{fid}", "load more")]')
TOP_NOTE = '[refs outside this window: web_page_query("{fid}", "<button or link text>")]'
UNCHANGED_NOTE = "[the page did not change]"
SHORTER_NOTE = "[the page got shorter; showing it from the top]"

_TRANSIENT = re.compile(r" ?\[[^\]=]*\]")


def _stable(text: str) -> list[str]:
    return [_TRANSIENT.sub("", ln) for ln in text.split("\n")]


def window_start(prev: tuple[str, str] | None, url: str, text: str) -> tuple[int, int, str]:
    """Where the next snapshot's window starts, how long it is, and the note line (a template over `{fid}`)."""
    if prev is None or prev[0] != url:
        return 0, WINDOW_CHARS, TOP_NOTE
    old, new = _stable(prev[1]), _stable(text)
    first = next((i for i, (a, b) in enumerate(zip(old, new)) if a != b), min(len(old), len(new)))
    if first == len(new):
        return (0, UNCHANGED_CHARS, UNCHANGED_NOTE) if len(old) == len(new) else (0, WINDOW_CHARS, SHORTER_NOTE)
    start = sum(len(ln) + 1 for ln in text.split("\n")[:max(0, first - CONTEXT_LINES)])
    return start, WINDOW_CHARS, CHANGED_NOTE if start else TOP_NOTE
