"""Persistent type-anytime input box on the TerminalChannel.

Two things under test, both offline:
  - while the box is active, the thinking/burst indicators render as an in-frame glyph and
    NEVER start a rich Status/Live (the spike proved rich spinners under patch_stdout glue
    lines and can FREEZE on Ctrl+C-during-burst);
  - the persistent app's keybindings submit WITHOUT exiting (Enter enqueues + resets),
    Ctrl+C on an empty buffer requests interrupt, Ctrl+D on empty requests EOF.

The headless prompt_toolkit harness (create_pipe_input + DummyOutput + create_app_session)
is the first of its kind in this repo.
"""
from __future__ import annotations

from io import StringIO

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from localharness.channels.terminal import (
    TERMINAL_THEME,
    TerminalChannel,
    _build_persistent_input_app,
)
from localharness.core.bus import EventBus


def _channel(force_terminal: bool = True) -> TerminalChannel:
    from rich.console import Console

    ch = TerminalChannel(EventBus(), {})
    ch._console = Console(
        file=StringIO(), force_terminal=force_terminal, width=120,
        theme=TERMINAL_THEME, highlight=False,
    )
    return ch


class TestSpinnerSuppression:
    def test_thinking_never_starts_rich_status_when_box_active(self):
        ch = _channel()
        ch._box_active = True
        ch._start_thinking()
        assert ch._thinking is None, "no rich Status while the box is live"
        assert ch._box_working is True, "in-frame working glyph is on instead"

    async def test_burst_never_starts_rich_status_when_box_active(self):
        ch = _channel()
        ch._box_active = True
        await ch.send_tool_call("web_search", {"query": "x"})
        assert ch._burst is not None
        assert ch._burst.status is None, "no rich spinner for the burst while the box is live"
        assert ch._box_working is True
        await ch.stop()

    def test_thinking_still_uses_rich_status_without_box(self):
        # Contrast: the classic (no-box) path is unchanged.
        ch = _channel()
        assert ch._box_active is False
        ch._start_thinking()
        assert ch._thinking is not None
        ch._stop_thinking()  # tear the daemon refresh thread down
        assert ch._thinking is None


class TestHintFrame:
    def _text(self, frags) -> str:
        return "".join(t for _style, t in frags)

    def test_first_hint_shown_then_queued_and_working(self):
        ch = _channel()
        ch._box_active = True
        ch._first_box_hint = "Describe a task, or /help for commands."
        assert "Describe a task" in self._text(ch._box_hint_frags())

        ch.box_set_queued(2)
        assert "queued (2)" in self._text(ch._box_hint_frags())

        ch.box_notify_working(True)
        frags = ch._box_hint_frags()
        assert "working" in self._text(frags)

    def test_decision_flash_shows_then_can_clear(self):
        ch = _channel()
        ch._box_active = True
        ch.box_flash_decision("→ nudging current turn")
        assert "nudging" in self._text(ch._box_hint_frags())
        ch._decision_flash = ""  # simulate the timed clear
        assert "nudging" not in self._text(ch._box_hint_frags())


class TestPersistentAppKeybindings:
    async def _drive(self, feed: str):
        subs: list[str] = []
        interrupts: list[bool] = []
        eofs: list[bool] = []
        holder: dict = {}

        def on_submit(t: str) -> None:
            subs.append(t)

        def on_interrupt() -> None:
            interrupts.append(True)

        def on_eof() -> None:
            eofs.append(True)
            holder["app"].exit()

        with create_pipe_input() as inp:
            with create_app_session(input=inp, output=DummyOutput()):
                app = _build_persistent_input_app(
                    InMemoryHistory(), ">",
                    on_submit=on_submit, on_interrupt=on_interrupt, on_eof=on_eof,
                    hint_fn=lambda: [("class:hint", " ")], pct_fn=lambda: None,
                )
                holder["app"] = app
                inp.send_text(feed)
                await app.run_async()
        return subs, interrupts, eofs

    async def test_enter_submits_without_exiting_and_resets(self):
        # two Enter-terminated lines then Ctrl+D — the app stays alive across both submits.
        subs, interrupts, eofs = await self._drive("first line\rsecond line\r\x04")
        assert subs == ["first line", "second line"]
        assert eofs == [True]
        assert interrupts == []

    async def test_bang_prefix_passed_through_untouched(self):
        # the box does not strip '!'; that is the router's job (kept out of help/docs).
        subs, _i, _e = await self._drive("!keep going\r\x04")
        assert subs == ["!keep going"]

    async def test_ctrl_c_empty_buffer_requests_interrupt(self):
        subs, interrupts, _e = await self._drive("\x03\x04")
        assert interrupts == [True]
        assert subs == []

    async def test_ctrl_c_with_text_clears_line_no_interrupt(self):
        # type text, Ctrl+C clears it (no interrupt, no submit), then Ctrl+D exits.
        subs, interrupts, eofs = await self._drive("half typed\x03\x04")
        assert subs == []
        assert interrupts == []
        assert eofs == [True]
