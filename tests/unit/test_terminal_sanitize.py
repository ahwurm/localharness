"""Model text reaches the terminal as text, never as terminal commands (F11).

Measured with rich 15 on the channel's own consoles (`highlight=False`): a plain `str`, `Text(...)`
and `Markdown(...)` all hand ESC sequences to the terminal verbatim (OSC 52 writes the clipboard,
`ESC[2J` clears the screen), and C1 controls (U+0080–U+009F; U+009B is an 8-bit CSI) survive every
path. A str only LOOKS clean with rich's default highlighter on, which splits the sequence with
its own styling while the raw ESC still goes out. So every model-, tool- or plugin-supplied string
passes through `sanitize_for_display` before rich sees it: the panels, the streamed answer, a
plugin's renderable (a /memory tree can hold model-written facts), and every line built by
interpolating someone's text into markup.
"""
from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.text import Text
from rich.tree import Tree

from localharness.channels.base import sanitize_for_display

OSC52 = "\x1b]52;c;ZXZpbA==\x07"
HOSTILE = f"hi {OSC52} there \x1b[2J \x9b31m"
INJECTED = ("\x1b]52", "\x1b[2J", "\x9b", "\x85")


def _channel():
    from localharness.channels.terminal import TERMINAL_THEME, TerminalChannel
    from localharness.core.bus import EventBus

    out = StringIO()
    ch = TerminalChannel(EventBus(), {})
    ch._console = Console(file=out, force_terminal=True, width=120, theme=TERMINAL_THEME,
                          highlight=False)
    return ch, out


def _clean(printed: str) -> bool:
    return not any(seq in printed for seq in INJECTED)


async def _tokens(*parts):
    for part in parts:
        yield part


def test_c1_controls_are_stripped_and_layout_survives():
    assert sanitize_for_display("a\x9b31mb\x85c") == "a31mbc"
    assert sanitize_for_display("".join(map(chr, range(0x80, 0xA0)))) == ""
    assert sanitize_for_display("tab\there\nnext é ü nbsp") == "tab\there\nnext é ü nbsp"


async def test_a_panel_message_carries_no_control_sequence():
    ch, out = _channel()
    await ch.send_message(HOSTILE, agent_id="a")
    printed = out.getvalue()
    assert "hi" in printed and "there" in printed and _clean(printed), repr(printed)


async def test_a_plain_message_carries_no_control_sequence():
    ch, out = _channel()
    await ch.send_message(HOSTILE)
    printed = out.getvalue()
    assert "hi" in printed and "there" in printed and _clean(printed), repr(printed)


async def test_the_streamed_answer_is_clean_live_and_final_and_returned_raw():
    ch, out = _channel()
    full = await ch.send_streaming(_tokens("ok ", OSC52, "\x1b[2Jdone \x9b"), agent_id="a")
    printed = out.getvalue()
    assert "done" in printed and _clean(printed), repr(printed)
    assert full == f"ok {OSC52}\x1b[2Jdone \x9b", "callers store what the model said"


async def test_a_plugin_renderable_is_sanitized_segment_by_segment():
    """Sanitized after rendering, so it reaches inside any renderable. `Text` already drops the
    BEL that ends an OSC, so the rest of that segment reads as the sequence's payload and goes
    with it — as a terminal would swallow it; the tree's next sequence ends it there."""
    ch, out = _channel()
    await ch.send_renderable(Text(f"x{OSC52}y\x9b", style="bold"))
    tree = Tree("memory")
    tree.add(Text(f"fact {OSC52} \x1b[2J end"))
    await ch.send_renderable(tree)
    printed = out.getvalue()
    assert "x" in printed and "fact" in printed and "end" in printed
    assert _clean(printed), repr(printed)
    assert "\x1b[1mx" in printed, "rich's own styling must survive"


async def test_reasoning_tool_and_error_lines_are_clean():
    """Lines built by interpolating model or tool text into markup — reasoning, a tool's output, a
    tool call's arguments: rich's string path passes ESC and C1 sequences on these consoles."""
    ch, out = _channel()
    ch.show_reasoning = True
    await ch.on_reasoning(f"thinking \x9b2J about {OSC52} it\n")
    await ch.send_tool_result("bash_exec", f"boom \x9b31m{OSC52}\x85", True)
    await ch.send_tool_call("web_fetch", {"url": "https://x.example/\x9b2J"})
    printed = out.getvalue()
    assert "thinking" in printed and "boom" in printed
    assert _clean(printed), repr(printed)
