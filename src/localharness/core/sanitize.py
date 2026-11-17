"""Strip what a terminal would act on from text about to be shown."""
from __future__ import annotations

import re

KEPT_CONTROL_CHARS = "\t\n"
"""The two control characters a rendered question may keep: tab, and newline.

Newline is structural here — one tool call can raise several reasons and the gate renders them
as a short multi-line `display`, which the terminal prints as lines and the ACP adapter splits
into a title and a body. Every other C0 character is removed (see :func:`sanitize_for_display`).
"""

ESCAPE_SEQUENCE_RE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)?|[@-Z\\-_])?"
)
"""ANSI escape sequences: CSI (`\\x1b[…`), OSC (`\\x1b]…` up to BEL or ST), and the two-character
forms. Removed whole, so what is left of a cursor-moving sequence is not a stray `[2J` printed as
text."""

CONTROL_CHARS_RE = re.compile(
    "["
    + "".join(
        re.escape(chr(code))
        for code in (*range(0x00, 0x20), 0x7F, *range(0x80, 0xA0))
        if chr(code) not in KEPT_CONTROL_CHARS
    )
    + "]"
)
"""Every C0 and C1 control and DEL except :data:`KEPT_CONTROL_CHARS` — derived from that set
rather than spelled out, so the two cannot drift apart. C1 (U+0080–U+009F) because a terminal that
honours 8-bit controls reads U+009B as CSI and U+009D as OSC: the same sequences without the ESC."""


INVISIBLE_CHARS_RE = re.compile(
    "["
    + "".join(
        map(
            chr,
            (
                *range(0x200B, 0x2010),
                *range(0x202A, 0x202F),
                *range(0x2060, 0x2065),
                *range(0x2066, 0x2070),
                *range(0xE0000, 0xE0080),
                0xFEFF,
                0x061C,
                0x180E,
            ),
        )
    )
    + "]"
)
"""Zero-width, bidi, invisible-operator and Unicode tag characters: they reorder or hide what the person reads.

The soft hyphen and the variation selectors stay: emoji and hyphenation rendering need them."""

NOTICE_MARK = "◆"
NEUTRAL_MARK = "◇"
LEADING_MARK_RE = re.compile(rf"(?:^|(?<=[\u2028\u2029]))([^\S\n]*){NOTICE_MARK}", re.MULTILINE)


def sanitize_for_display(text: str) -> str:
    """Strip terminal control characters from text a channel is about to render.

    A permission question quotes the model's own words — a shell command, a path, an MCP tool's
    arguments — and those words reach a terminal that acts on control characters. `\\r` rewrites
    the line the human is reading, `\\x1b[2J` clears the screen, and either one can leave a
    question on screen that is not the question being answered. Both renderers of an ASK call
    this before their own escaping (rich markup escaping is about markup, not about the
    terminal), because the answer to a mangled question is a permission the human did not grant.

    Tab and newline survive (:data:`KEPT_CONTROL_CHARS`): they are layout, not control. A
    :data:`NOTICE_MARK` that opens a line becomes :data:`NEUTRAL_MARK`, so text cannot pass for a
    harness notice.
    """
    if not text:
        return text
    bare = INVISIBLE_CHARS_RE.sub("", CONTROL_CHARS_RE.sub("", ESCAPE_SEQUENCE_RE.sub("", text)))
    return LEADING_MARK_RE.sub(rf"\1{NEUTRAL_MARK}", bare)


def one_line(text: str) -> str:
    """`sanitize_for_display` with every run of whitespace, newlines included, collapsed to one space."""
    return " ".join(sanitize_for_display(text.replace("\r", " ")).split())
