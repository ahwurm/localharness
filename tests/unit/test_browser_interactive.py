"""window_start: the snapshot window starts where the page changed, with no browser involved."""
from __future__ import annotations

import pytest

from localharness.tools.builtin.browser_session import (
    CHANGED_NOTE, SHORTER_NOTE, TOP_NOTE, UNCHANGED_CHARS, UNCHANGED_NOTE, WINDOW_CHARS, window_start,
)

URL = "https://a.test/"
LINES = ["- heading [level=1]", "- list", "  - item one [ref=e1]", "  - item two [ref=e2]", "  - item three [ref=e3]"]


def snap(*lines: str) -> str:
    return "\n".join(lines)


def start_of(lines: list[str], index: int) -> int:
    return sum(len(ln) + 1 for ln in lines[:index])


FIRST = (0, WINDOW_CHARS, TOP_NOTE)
MORE = [*LINES, "  - item four [ref=e4]", "  - item five [ref=e5]"]
MIDDLE = [*LINES[:3], "  - ITEM TWO [ref=e2]", *LINES[4:]]


@pytest.mark.parametrize("prev,url,lines,want", [
    (None, URL, LINES, FIRST),
    ((URL, snap(*LINES)), "https://b.test/", LINES, FIRST),
    ((URL, snap(*LINES)), URL, LINES, (0, UNCHANGED_CHARS, UNCHANGED_NOTE)),
    ((URL, snap(*LINES)), URL, LINES[:3], (0, WINDOW_CHARS, SHORTER_NOTE)),
    ((URL, snap(*LINES)), URL, ["- different heading", *LINES[1:]], FIRST),
    ((URL, snap(*LINES)), URL, [LINES[0], "- other", *LINES[2:]], FIRST),
    ((URL, snap(*LINES)), URL, MORE, (start_of(MORE, 3), WINDOW_CHARS, CHANGED_NOTE)),
    ((URL, snap(*LINES)), URL, MIDDLE, (start_of(MIDDLE, 1), WINDOW_CHARS, CHANGED_NOTE)),
])
def test_window_start(prev, url, lines, want):
    assert window_start(prev, url, snap(*lines)) == want


def test_a_changed_middle_line_starts_two_lines_early_at_its_exact_offset():
    text = snap(*MIDDLE)
    start, _, note = window_start((URL, snap(*LINES)), URL, text)
    assert text[start:].startswith("- list\n  - item one") and note == CHANGED_NOTE


@pytest.mark.parametrize("marker", ["[active]", "[checked]", "[focused]"])
def test_a_marker_without_an_equals_is_not_a_change(marker):
    marked = [LINES[0], LINES[1], f"  - item one {marker} [ref=e1]", *LINES[3:]]
    assert window_start((URL, snap(*LINES)), URL, snap(*marked)) == (0, UNCHANGED_CHARS, UNCHANGED_NOTE)
    assert window_start((URL, snap(*marked)), URL, snap(*LINES))[2] == UNCHANGED_NOTE


@pytest.mark.parametrize("old,new", [("[ref=e2]", "[ref=e9]"), ("[level=2]", "[level=3]")])
def test_a_marker_with_an_equals_is_a_change(old, new):
    before = [*LINES[:2], f"  - item {old}", *LINES[3:]]
    after = [*LINES[:2], f"  - item {new}", *LINES[3:]]
    assert window_start((URL, snap(*before)), URL, snap(*after))[2] != UNCHANGED_NOTE


def test_offsets_are_taken_on_the_original_text_not_the_stripped_one():
    old = [*LINES[:3], "  - tail [ref=e9]"]
    new = [LINES[0], "- list [active]", LINES[2], "  - tail [ref=e9]", "  - more [ref=e10]", "  - last [ref=e11]"]
    text = snap(*new)
    start, _, _ = window_start((URL, snap(*old)), URL, text)
    assert start == start_of(new, 2) and text[start:].startswith("  - item one")
