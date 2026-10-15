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

import asyncio
import re
import time
import types
from contextlib import contextmanager
from io import StringIO

from prompt_toolkit.application import create_app_session
from pathlib import Path

from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.layout.dimension import to_dimension
from prompt_toolkit.output import DummyOutput

import localharness.channels.terminal as terminal_module
from localharness.channels.terminal import (
    PRESENCE_WINDOW_S,
    TERMINAL_THEME,
    TerminalChannel,
    _build_input_app,
    _build_persistent_input_app,
)
from localharness.config.models import TerminalKeysConfig
from localharness.core.bus import EventBus
from localharness.core.events import ConsolidationFinished, ConsolidationStarted


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
        # The first-run hint is the box's PLACEHOLDER now (dim text inside the empty box, the
        # Claude Code / Cline idiom), not a label in the border.
        assert ch._box_placeholder() == "Describe a task, or /help for commands."
        assert "tab commands" in self._text(ch._box_hint_frags()), "idle legend in the footer"

        ch.box_set_queued(2)
        assert "queued (2)" in self._text(ch._box_hint_frags())

        # FIX 2: the working glyph moved OUT of the bottom border into the status row above
        # the box. The footer keeps input-metadata (queued / decision flash / legend) only.
        ch.box_notify_working(True)
        assert "working" not in self._text(ch._box_hint_frags()), "footer no longer carries it"
        assert "working" in self._text(ch._box_status_frags()), "status row does"
        # the legend follows the mode: a running turn is when nudge/stop matter
        assert "queued (2) · alt+enter nudge · ctrl+c stop" in self._text(ch._box_hint_frags())
        ch.box_notify_working(False)
        assert "nudge" not in self._text(ch._box_hint_frags())

    async def test_first_hint_is_consumed_on_the_first_submit(self):
        """#49 says the hint shows "until first use" — submitting IS that use. It used to be set
        once in start_input_box and never cleared, so the border repeated it all session (the
        classic read_input path consumes it after one prompt; the two modes must agree)."""
        ch = _channel()
        ch._history = InMemoryHistory()  # start()'s only box prerequisite; no history file here
        ch.first_prompt_hint = "Describe a task, or /help for commands."
        q: asyncio.Queue = asyncio.Queue()
        with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
            await ch.start_input_box(q, on_interrupt=lambda: None)
            try:
                assert "Describe a task" in ch._box_placeholder()
                inp.send_text("hello\r")
                assert await asyncio.wait_for(q.get(), timeout=10.0) == ("submit", "hello")
                assert ch._box_placeholder() == ""
            finally:
                await ch.stop_input_box()

    def test_decision_flash_shows_then_can_clear(self):
        ch = _channel()
        ch._box_active = True
        ch.box_flash_decision("→ nudging current turn")
        assert "nudging" in self._text(ch._box_hint_frags())
        ch._decision_flash = ""  # simulate the timed clear
        assert "nudging" not in self._text(ch._box_hint_frags())


class TestFooterInstruments:
    """Footer-right: the local-model instrument cluster (model · measured tok/s · context
    meter). Suppliers are optional; the rate shown at rest is only ever the VERIFIED one — the
    live `~` rate belongs to the status row, so a number never appears twice on screen."""

    def _text(self, frags) -> str:
        return "".join(t for _style, t in frags)

    def test_full_cluster_model_rate_meter(self):
        ch = _channel()
        ch.model_source = lambda: "qwen3.8-27b"
        ch.tps_source = lambda: (25.3, True)
        ch._context_pct = 42.0
        frags = ch._box_instrument_frags()
        assert self._text(frags) == "qwen3.8-27b · 25.3 tok/s · ████░░░░░░ 42%"
        assert ("class:model", "qwen3.8-27b") in frags
        assert ("class:tps-yellow", "25.3 tok/s") in frags

    def test_unverified_rate_stays_on_the_status_row_only(self):
        ch = _channel()
        ch.model_source = lambda: "m"
        ch.tps_source = lambda: (28.0, False)
        assert self._text(ch._box_instrument_frags()) == "m"
        ch._box_working = True
        assert ("class:tps-yellow", "· ~28 tok/s ") in ch._box_status_frags()

    async def test_rate_leaves_the_footer_while_a_turn_runs(self):
        """Between requests of a running turn (a tool executing) the rate is verified — the
        status row's burst line shows it, so the footer must not, or it reads twice."""
        ch = _channel()
        ch._box_active = True
        ch.tps_source = lambda: (24.9, True)
        assert self._text(ch._box_instrument_frags()) == "24.9 tok/s"
        await ch.send_tool_call("web_fetch", {"url": "u"})  # opens a burst → working
        assert ("class:tps-green", "· 24.9 tok/s ") not in ch._box_status_frags()  # 24.9 is yellow
        assert "24.9 tok/s" in self._text(ch._box_status_frags())
        assert ch._box_instrument_frags() == []
        await ch.stop()

    def test_no_suppliers_is_an_empty_cluster(self):
        ch = _channel()
        assert ch._box_instrument_frags() == []
        ch._context_pct = 8.0
        assert self._text(ch._box_instrument_frags()) == "░░░░░░░░░░ 8%"

    def test_broken_model_source_never_breaks_the_footer(self):
        ch = _channel()

        def boom() -> str:
            raise RuntimeError("no client")

        ch.model_source = boom
        ch._context_pct = 8.0
        assert self._text(ch._box_instrument_frags()) == "░░░░░░░░░░ 8%"


class TestStatusRow:
    """FIX 2: the working/activity indicator renders in a one-line status row ABOVE the box
    (so it reads as the last line of the log area), not crammed into the box's bottom border.
    Empty (→ zero height) when idle; shows the live tool-burst counter; animates the between-
    turns 'dreaming' consolidation pass."""

    def _text(self, frags) -> str:
        return "".join(t for _s, t in frags)

    def test_idle_status_row_is_empty(self):
        ch = _channel()
        ch._box_active = True
        # nothing happening → empty fragments so the ConditionalContainer collapses to 0 rows
        # (no blank line wasted above the box).
        assert ch._box_status_frags() == []

    def test_working_shows_spinner_and_working(self):
        ch = _channel()
        ch._box_active = True
        ch.box_notify_working(True)
        text = self._text(ch._box_status_frags())
        assert "working" in text
        assert any(frame in text for frame in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"), "an animated braille frame"

    async def test_open_burst_shows_live_counter(self):
        ch = _channel()
        ch._box_active = True
        await ch.send_tool_call("web_search", {"query": "x"})
        await ch.send_tool_call("web_search", {"query": "y"})
        text = self._text(ch._box_status_frags())
        assert "web_search" in text, "the burst's tool family is named in the status row"
        assert "0/2" in text, "…with its live done/calls counter, not just 'working'"
        await ch.stop()

    async def test_dreaming_animates_in_status_row_in_box_mode(self):
        ch = _channel()
        ch._box_active = True
        await ch.on_consolidation_started(ConsolidationStarted())
        assert "dreaming" in self._text(ch._box_status_frags())
        await ch.on_consolidation_finished(ConsolidationFinished())
        assert ch._box_status_frags() == [], "the pass ended → row collapses again"


class TestPromptEcho:
    """FIX 1: every box submission leaves a permanent line in the scrollback (❯ <text>),
    through the same patch_stdout-safe console the tool/agent lines use — the box resets
    its buffer on submit, so without this echo the typed prompt vanishes from the transcript."""

    def _out(self, ch) -> str:
        return ch._console.file.getvalue()

    async def test_echo_prints_prompt_line_to_scrollback(self):
        ch = _channel()
        ch._box_active = True
        await ch.box_echo_prompt("index the repo")
        out = self._out(ch)
        assert "index the repo" in out
        assert "❯" in out  # ❯ prompt glyph, so a scrolled-back prompt is recognizable

    async def test_echo_with_queued_annotation(self):
        ch = _channel()
        ch._box_active = True
        await ch.box_echo_prompt("also update the changelog", annotation="queued (2)")
        out = self._out(ch)
        assert "also update the changelog" in out
        assert "queued (2)" in out

    async def test_echo_with_nudge_annotation(self):
        ch = _channel()
        ch._box_active = True
        await ch.box_echo_prompt("stop, wrong file", annotation="→ nudge")
        out = self._out(ch)
        assert "stop, wrong file" in out
        assert "→ nudge" in out

    async def test_a_multi_line_echo_prints_its_rows(self):
        ch = _channel()
        ch._box_active = True
        await ch.box_echo_prompt("a\nb")
        plain = re.sub(r"\x1b\[[0-9;]*m", "", self._out(ch))
        rows = [row.strip() for row in plain.splitlines() if row.strip()]
        assert rows[-2].endswith("a")
        assert rows[-1] == "b"

    async def test_echo_escapes_user_markup(self):
        ch = _channel()
        ch._box_active = True
        await ch.box_echo_prompt("[bold]not markup[/bold]")
        # rich markup in the user's text must render literally, never be interpreted.
        assert "[bold]not markup[/bold]" in self._out(ch)


async def _drive_box(feed: str, *, models=None, keys=None, newline_fires=None, history=None):
    """Run the persistent box on a pipe input and return (submissions, interrupts, eofs)."""
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

    extra: dict = {}
    if keys is not None:
        extra["keys"] = keys
    if newline_fires is not None:
        extra["on_newline_chord"] = newline_fires.append

    with create_pipe_input() as inp:
        with create_app_session(input=inp, output=DummyOutput()):
            app = _build_persistent_input_app(
                history if history is not None else InMemoryHistory(), ">",
                on_submit=on_submit, on_interrupt=on_interrupt, on_eof=on_eof,
                hint_fn=lambda: [("class:hint", " ")], right_fn=lambda: [],
                status_fn=lambda: [],
                model_names_fn=(lambda: list(models)) if models is not None else None,
                **extra,
            )
            holder["app"] = app
            inp.send_text(feed)
            # wait_for guard: a feed that never reaches an exit key would otherwise hang the
            # whole suite on this app instead of failing this one test.
            await asyncio.wait_for(app.run_async(), timeout=10.0)
    return subs, interrupts, eofs


async def _drive_classic(feed: str, *, keys=None, newline_fires=None) -> str:
    """Run the classic one-shot input app on a pipe input and return the line it exits with."""
    extra: dict = {}
    if keys is not None:
        extra["keys"] = keys
    if newline_fires is not None:
        extra["on_newline_chord"] = newline_fires.append
    with create_pipe_input() as inp:
        with create_app_session(input=inp, output=DummyOutput()):
            app = _build_input_app(InMemoryHistory(), ">", "", **extra)
            inp.send_text(feed)
            return await asyncio.wait_for(app.run_async(), timeout=10.0)


class TestPersistentAppKeybindings:
    async def _drive(self, feed: str, models: list[str] | None = None):
        return await _drive_box(feed, models=models)

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

    async def test_ctrl_j_is_enter_not_a_forced_nudge(self):
        # Raw LF (0x0A) is what SOME TERMINALS SEND FOR RETURN (WSL) — prompt_toolkit binds it to
        # Enter for exactly that reason. Binding it to the force-nudge instead silently prefixed
        # every submission on those terminals with '!', which sends slash commands into the
        # running turn as text instead of queueing them. It must submit plainly.
        subs, _i, _e = await self._drive("hello via LF\x0a\x04")
        assert subs == ["hello via LF"]

    async def test_ctrl_j_lines_are_plain_submissions_not_nudges(self):
        # Same for a multi-line paste on a terminal without bracketed paste: each embedded \n
        # is one ordinary submission.
        subs, _i, _e = await self._drive("check the logs\x0athen fix the parser\x0a\x04")
        assert subs == ["check the logs", "then fix the parser"]

    async def test_alt_enter_submits_as_forced_nudge(self):
        # Alt+Enter arrives as ESC,CR — the universal Meta encoding (Shift+Enter is NOT
        # bindable: prompt_toolkit remaps its xterm sequence to plain Enter).
        subs, _i, _e = await self._drive("go deeper on that\x1b\r\x04")
        assert subs == ["!go deeper on that"]

    async def test_nudge_chord_empty_buffer_is_noop(self):
        subs, interrupts, eofs = await self._drive("\x1b\r\x04")
        assert subs == []
        assert interrupts == []
        assert eofs == [True]

    async def test_escape_dismissing_a_menu_is_not_the_nudge_chord(self):
        # Esc (dismiss the menu) then Enter is an ordinary two-keystroke motion, but it arrives
        # as the same ESC,CR bytes as Alt+Enter. The dismiss binding is EAGER so Esc resolves on
        # its own press: the menu closes and the line submits NORMALLY — never as a forced nudge
        # carrying the previewed completion ('!/model') into the running turn.
        subs, _i, _e = await self._drive("/m\t\x1b\r\x04")
        assert subs == ["/m"]

    async def test_escape_takes_back_the_picker_prefix(self):
        # The /model picker pre-fills '/model ' + highlights the first model. Declining it (Esc)
        # must leave an EMPTY line — otherwise the next thing typed submits as
        # '/model rewrite the parser tests' and comes back as "Unknown model '…'".
        subs, _i, _e = await self._drive(
            "/model \t\x1brewrite the parser tests\r\x04", models=["qwen-a", "qwen-b"]
        )
        assert subs == ["rewrite the parser tests"]


class TestBoxLineBreak:
    """Backslash then Enter breaks the line with nothing configured (spec 21 §6.1)."""

    async def test_backslash_enter_breaks_the_line(self):
        subs, _i, _e = await _drive_box("x\\\ry\r\x04")
        assert subs == ["x\ny"]

    async def test_backslash_then_lf_breaks_the_line_too(self):
        # Ctrl+J shares Enter's handler (it is Return on LF terminals), so it gets the rule too.
        subs, _i, _e = await _drive_box("x\\\ny\n\x04")
        assert subs == ["x\ny"]

    async def test_double_backslash_submits_as_typed(self):
        subs, _i, _e = await _drive_box("x\\\\\r\x04")
        assert subs == ["x\\\\"]

    async def test_backslash_space_submits_and_strips_to_one_backslash(self):
        subs, _i, _e = await _drive_box("x\\ \r\x04")
        assert subs == ["x\\ "]
        assert subs[0].strip() == "x\\"

    async def test_a_lone_backslash_becomes_a_line_break(self):
        subs, _i, _e = await _drive_box("\\\ry\r\x04")
        assert subs == ["\ny"]

    async def test_backslash_enter_with_the_caret_mid_line_submits_as_typed(self):
        # The caret is not at the end: the rule is off and the text goes as typed.
        subs, _i, _e = await _drive_box("x\\\x1b[D\r\x04")
        assert subs == ["x\\"]

    async def test_a_completion_accepted_by_enter_wins_over_the_backslash(self, monkeypatch):
        rows = [("/x\\", "first"), ("/x\\y", "second")]
        monkeypatch.setattr("localharness.cli.slash_commands.all_rows", lambda: rows)
        # `/x` + Tab opens the menu on `/x\`; the first Enter applies it (no break, no submit),
        # then `z` + Enter submits `/x\z`.
        subs, _i, _e = await _drive_box("/x\t\rz\r\x04")
        assert subs == ["/x\\z"]


# The two encodings of Shift+Enter that channels/keys.py installs (spec 21 §7).
_KITTY_SHIFT_ENTER = "\x1b[13;2u"
_XTERM_SHIFT_ENTER = "\x1b[27;2;13~"
_CTRL_O = "\x0f"
_CTRL_G = "\x07"
_ALT_ENTER = "\x1b\r"


class TestBoxChords:
    """The nudge and line-break chords come from the table in channels/keys.py (spec 21 §6.2-§6.3)."""

    async def test_shift_enter_both_encodings_make_one_message(self):
        fires: list[str] = []
        subs, _i, _e = await _drive_box(
            f"a{_KITTY_SHIFT_ENTER}b{_XTERM_SHIFT_ENTER}c\r\x04", newline_fires=fires
        )
        assert subs == ["a\nb\nc"]
        assert "\r" not in subs[0]
        assert fires == ["shift-enter", "shift-enter"]

    async def test_ctrl_o_nudge_chord_forces_and_alt_enter_submits_plainly(self):
        keys = TerminalKeysConfig(nudge=["ctrl-o"])
        subs, _i, _e = await _drive_box(f"go{_CTRL_O}go{_ALT_ENTER}\x04", keys=keys)
        assert subs == ["!go", "go"]

    async def test_ctrl_g_nudge_chord_forces(self):
        keys = TerminalKeysConfig(nudge=["ctrl-g"])
        subs, _i, _e = await _drive_box(f"go{_CTRL_G}\x04", keys=keys)
        assert subs == ["!go"]

    async def test_alt_enter_as_newline_breaks_and_with_menu_open_submits(self):
        keys = TerminalKeysConfig(newline=["alt-enter"])
        assert keys.nudge == []  # precedence took alt-enter out of the defaulted nudge list
        fires: list[str] = []
        subs, _i, _e = await _drive_box(
            f"a{_ALT_ENTER}b\r\x04", keys=keys, newline_fires=fires
        )
        assert subs == ["a\nb"]
        assert fires == ["alt-enter"]
        # With the menu open the eager Escape dismisses it and the Enter submits, as for the nudge.
        subs, _i, _e = await _drive_box(f"/m\t{_ALT_ENTER}\x04", keys=keys)
        assert subs == ["/m"]

    async def test_no_nudge_chord_leaves_bang_and_plain_alt_enter(self):
        keys = TerminalKeysConfig(nudge=[])
        subs, _i, _e = await _drive_box(f"go{_ALT_ENTER}!x\r\x04", keys=keys)
        assert subs == ["go", "!x"]

    async def test_shift_enter_as_nudge_forces_and_does_not_report_a_newline(self):
        keys = TerminalKeysConfig(nudge=["shift-enter"])
        assert keys.newline == []
        fires: list[str] = []
        subs, _i, _e = await _drive_box(
            f"go{_KITTY_SHIFT_ENTER}\x04", keys=keys, newline_fires=fires
        )
        assert subs == ["!go"]
        assert fires == []

    async def test_without_shift_enter_the_table_is_untouched_and_xterm_form_is_enter(self):
        keys = TerminalKeysConfig(newline=["alt-enter"], nudge=["ctrl-o"])
        subs, _i, _e = await _drive_box(f"a{_XTERM_SHIFT_ENTER}\x04", keys=keys)
        assert subs == ["a"]
        assert ANSI_SEQUENCES[_XTERM_SHIFT_ENTER] == Keys.ControlM

    async def test_without_the_prefix_cache_the_other_chords_still_work(self, monkeypatch):
        # Simulate a prompt_toolkit that renamed the private prefix cache: the install writes
        # nothing and says so. (Deleting the real attribute would break the parser itself, which
        # reads it on every key.)
        monkeypatch.setattr(terminal_module, "install_wire_sequences", lambda: False)
        keys = TerminalKeysConfig(newline=["shift-enter", "ctrl-o"])
        subs, _i, _e = await _drive_box(
            f"a{_CTRL_O}b\\\rc\rx{_ALT_ENTER}\x04", keys=keys
        )
        assert subs == ["a\nb\nc", "!x"]
        assert ANSI_SEQUENCES[_XTERM_SHIFT_ENTER] == Keys.ControlM

    async def test_two_newline_chords_in_a_row(self):
        keys = TerminalKeysConfig(newline=["shift-enter", "ctrl-o"])
        subs, _i, _e = await _drive_box(f"a{_CTRL_O}{_CTRL_O}b\r\x04", keys=keys)
        assert subs == ["a\n\nb"]

    async def test_a_newline_chord_on_an_empty_line_submits_nothing(self):
        keys = TerminalKeysConfig(newline=["shift-enter", "ctrl-o"])
        subs, _i, eofs = await _drive_box(f"{_CTRL_O}\r\x04", keys=keys)
        assert subs == []
        assert eofs == [True]

    async def test_a_newline_chord_inserts_at_the_caret(self):
        keys = TerminalKeysConfig(newline=["shift-enter", "ctrl-o"])
        subs, _i, _e = await _drive_box(f"ab\x1b[D{_CTRL_O}\r\x04", keys=keys)
        assert subs == ["a\nb"]

    async def test_a_newline_chord_with_a_highlighted_item_does_not_accept_it(self):
        keys = TerminalKeysConfig(newline=["shift-enter", "ctrl-o"])
        subs, _i, _e = await _drive_box(f"/m\t{_CTRL_O}x\r\x04", keys=keys)
        assert subs == ["/m\nx"]

    async def test_two_nudge_chords_both_force(self):
        keys = TerminalKeysConfig(nudge=["alt-enter", "ctrl-o"])
        subs, _i, _e = await _drive_box(f"a{_ALT_ENTER}b{_CTRL_O}\x04", keys=keys)
        assert subs == ["!a", "!b"]

    async def test_a_nudge_chord_on_a_blank_line_is_a_noop(self):
        keys = TerminalKeysConfig(nudge=["alt-enter", "ctrl-o"])
        subs, _i, eofs = await _drive_box(f"{_CTRL_O}   {_ALT_ENTER}\x04", keys=keys)
        assert subs == []
        assert eofs == [True]

    async def test_a_nudge_chord_keeps_a_typed_bang(self):
        subs, _i, _e = await _drive_box(f"!x{_ALT_ENTER}\x04")
        assert subs == ["!!x"]

    def test_no_literal_alt_enter_binding_outside_the_table(self):
        import re

        source = Path(terminal_module.__file__).read_text(encoding="utf-8")
        assert not re.search(r"""["']escape["']\s*,\s*["']enter["']""", source)
        assert "KEYS[" in source


class TestClassicAppKeys:
    """The classic app binds the newline chords only: it has no turn to steer (spec 21 §6.3)."""

    async def test_alt_enter_is_a_plain_submit(self):
        assert await _drive_classic(f"go{_ALT_ENTER}") == "go"

    async def test_shift_enter_breaks_the_line(self):
        fires: list[str] = []
        got = await _drive_classic(f"a{_KITTY_SHIFT_ENTER}b\r", newline_fires=fires)
        assert got == "a\nb"
        assert fires == ["shift-enter"]

    async def test_ctrl_o_newline_breaks_the_line(self):
        keys = TerminalKeysConfig(newline=["ctrl-o"])
        assert await _drive_classic(f"a{_CTRL_O}b\r", keys=keys) == "a\nb"

    async def test_backslash_enter_breaks_the_line(self):
        assert await _drive_classic("x\\\ry\r") == "x\ny"

    async def test_double_backslash_submits_as_typed(self):
        assert await _drive_classic("x\\\\\r") == "x\\\\"


def _input_window(app):
    """The one Window of the app whose content is the input BufferControl."""
    found = [w for w in app.layout.find_all_windows() if isinstance(w.content, BufferControl)]
    assert len(found) == 1
    return found[0]


class TestBoxHeightAndHistory:
    """The input box stops growing at eight rows, and a multi-line message is one submission and one
    history entry (spec 21 §6.4, §6.5)."""

    def test_input_window_is_capped_at_eight_rows(self):
        with create_pipe_input() as inp:
            with create_app_session(input=inp, output=DummyOutput()):
                classic = _build_input_app(InMemoryHistory(), ">", "")
                persistent = _build_persistent_input_app(
                    InMemoryHistory(), ">",
                    on_submit=lambda _t: None, on_interrupt=lambda: None, on_eof=lambda: None,
                    hint_fn=lambda: [], right_fn=lambda: [], status_fn=lambda: [],
                )
                for app in (classic, persistent):
                    window = _input_window(app)
                    assert to_dimension(window.height).max == 8
                    # multiline stays off: prompt_toolkit's own Enter default is overridden anyway
                    assert not window.content.buffer.multiline()

    async def test_a_multi_line_message_is_one_history_entry(self, tmp_path):
        feed = f"a{_KITTY_SHIFT_ENTER}b{_KITTY_SHIFT_ENTER}c\r\x04"
        memory = InMemoryHistory()
        subs, _i, _e = await _drive_box(feed, history=memory)
        assert subs == ["a\nb\nc"]
        assert memory.get_strings() == ["a\nb\nc"]
        path = tmp_path / "history"
        subs, _i, _e = await _drive_box(feed, history=FileHistory(str(path)))
        assert subs == ["a\nb\nc"]
        assert list(FileHistory(str(path)).load_history_strings()) == ["a\nb\nc"]

    async def test_a_message_of_only_line_breaks_submits_nothing(self):
        subs, _i, eofs = await _drive_box(
            f"{_KITTY_SHIFT_ENTER}{_KITTY_SHIFT_ENTER}\r\x04"
        )
        assert subs == []
        assert eofs == [True]


class TestModelPickerBox:
    """box_open_model_menu writes into the LIVE box buffer — and it lands SECONDS after the user
    typed /model (the listing runs live probes first) or at an arbitrary moment when a queued
    /model plays. The box invites typing the whole time, so the picker must never take a line
    that is already being written, and must not pre-fill when there is nothing to pick."""

    @contextmanager
    def _box(self, models: list[str]):
        ch = _channel()
        ch._box_active = True
        ch.model_names_fn = lambda: list(models)
        with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
            app = _build_persistent_input_app(
                InMemoryHistory(), ">",
                on_submit=lambda t: None, on_interrupt=lambda: None, on_eof=lambda: None,
                hint_fn=lambda: [], right_fn=lambda: [], status_fn=lambda: [],
                model_names_fn=ch._model_names_for_menu,
            )
            ch._box_app = app
            yield ch, app._lh_input_buffer

    async def test_picker_prefills_an_empty_box(self):
        with self._box(["qwen-a", "qwen-b"]) as (ch, buf):
            ch.box_open_model_menu()
            await asyncio.sleep(0)  # let the async completer settle
            assert buf.text.startswith("/model ")
            # menu popped with the first model highlighted (the one-Enter picker), so the
            # buffer shows its preview — Esc restores '/model ' and then clears it.
            assert buf.complete_state is not None
            assert buf.complete_state.current_completion is not None

    async def test_picker_never_overwrites_a_line_being_typed(self):
        with self._box(["qwen-a", "qwen-b"]) as (ch, buf):
            buf.text = "summarize the last three commits"  # typed while the listing was fetched
            ch.box_open_model_menu()
            await asyncio.sleep(0)
            assert buf.text == "summarize the last three commits"  # not in history yet — sacred

    async def test_picker_skipped_when_there_is_nothing_to_pick(self):
        with self._box([]) as (ch, buf):  # server unreachable → empty menu source
            ch.box_open_model_menu()
            await asyncio.sleep(0)
            assert buf.text == ""  # no stranded '/model ' prefix in front of the next message


class TestStatusRowTps:
    """Colored tok/s readout in the box status row (speed_stats bands): shown for turn
    activity (working/burst), suppressed for swap-loading and dreaming lines, `~` marks
    the live approximation, and a broken source can never take the row down."""

    def test_working_row_shows_live_rate_with_band_and_tilde(self):
        ch = _channel()
        ch._box_active = True
        ch._box_working = True
        ch.tps_source = lambda: (28.4, False)
        assert ("class:tps-yellow", "· ~28 tok/s ") in ch._box_status_frags()

    def test_working_row_shows_verified_rate_plain(self):
        ch = _channel()
        ch._box_active = True
        ch._box_working = True
        ch.tps_source = lambda: (31.24, True)
        assert ("class:tps-green", "· 31.2 tok/s ") in ch._box_status_frags()

    def test_red_band_below_twenty(self):
        ch = _channel()
        ch._box_active = True
        ch._box_working = True
        ch.tps_source = lambda: (16.5, True)
        assert ("class:tps-red", "· 16.5 tok/s ") in ch._box_status_frags()

    def test_swap_loading_row_suppresses_stale_rate(self):
        ch = _channel()
        ch._box_active = True
        ch._box_activity = "loading qwen · 40s"  # /model load: old model's rate is stale
        ch.tps_source = lambda: (31.2, True)
        assert not any("tok/s" in t for _, t in ch._box_status_frags())

    def test_no_source_or_no_data_renders_plain_row(self):
        ch = _channel()
        ch._box_active = True
        ch._box_working = True
        assert not any("tok/s" in t for _, t in ch._box_status_frags())
        ch.tps_source = lambda: None
        assert not any("tok/s" in t for _, t in ch._box_status_frags())

    def test_broken_source_never_breaks_the_row(self):
        ch = _channel()
        ch._box_active = True
        ch._box_working = True

        def boom():
            raise RuntimeError("snapshot exploded")

        ch.tps_source = boom
        assert ch._box_status_frags()  # row still renders

    def test_classic_thinking_label_appends_verified_rate_only(self):
        ch = _channel()
        ch.tps_source = lambda: (16.5, True)
        assert ch._thinking_label() == "[muted]thinking…[/muted] [red]16.5 tok/s[/red]"
        ch.tps_source = lambda: (28.0, False)  # live approximation: classic label omits it
        assert ch._thinking_label() == "[muted]thinking…[/muted]"
        ch.tps_source = None
        assert ch._thinking_label() == "[muted]thinking…[/muted]"


class TestWorkingTallies:
    """The working row names the phase and counts its tokens (2026-09-09). With reasoning
    hidden, a bare "working" was every phase at once; a 12-minute think looked like a hang.
    Three things at most: the colored phase + tally, elapsed (or a red silence), tok/s."""

    def _text(self, frags) -> str:
        return "".join(t for _style, t in frags)

    def _row(self, snap):
        ch = _channel()
        ch._box_active = True
        ch.box_notify_working(True)
        ch.progress_source = lambda: snap
        return ch, ch._box_status_frags()

    def test_thinking_phase_is_colored_and_carries_its_tally(self):
        _, frags = self._row({"phase": "thinking", "thinking_tokens": 2300, "answer_tokens": 410,
                              "tool_call_tokens": 0, "elapsed": 72.4, "silent": 0.3})
        text = self._text(frags)
        assert "⋯ thinking 2.3k" in text and "1m12s" in text
        assert "410" not in text, "only the current phase's tally — three things, not six"
        assert "working" not in text and "silent" not in text
        assert any(style == "class:phase-thinking" and "thinking" in txt for style, txt in frags)

    def test_writing_phase_with_a_long_silence_goes_red(self):
        _, frags = self._row({"phase": "writing", "thinking_tokens": 0, "answer_tokens": 88,
                              "tool_call_tokens": 0, "elapsed": 30.0, "silent": 14.2})
        text = self._text(frags)
        assert "✎ writing 88" in text and "silent 14s" in text
        assert any(style == "class:phase-silent" for style, _ in frags)

    def test_waiting_phase_before_the_first_delta(self):
        _, frags = self._row({"phase": "waiting", "thinking_tokens": 0, "answer_tokens": 0,
                              "tool_call_tokens": 0, "elapsed": 8.0, "silent": 8.0})
        text = self._text(frags)
        assert "… waiting" in text and "8s" in text
        assert "silent" not in text, "waiting IS the silence; no double report"

    def test_no_supplier_keeps_the_plain_working(self):
        ch = _channel()
        ch._box_active = True
        ch.box_notify_working(True)
        assert "working" in self._text(ch._box_status_frags())

    def test_rate_still_follows_the_phase(self):
        ch, _ = self._row({"phase": "thinking", "thinking_tokens": 120, "answer_tokens": 0,
                           "tool_call_tokens": 0, "elapsed": 5.0, "silent": 0.2})
        ch.tps_source = lambda: (28.4, False)
        text = self._text(ch._box_status_frags())
        assert "⋯ thinking 120" in text and "~28 tok/s" in text


# --------------------------------------------------- the parked-call row + hotkeys + presence

def _pending(pid: int, rendering: str = "bash_exec: rm -rf ~/old-notes"):
    """One PendingCall, built the way the gate builds it (id + one-line rendering are all the
    box ever reads)."""
    from localharness.agent.gate_types import PendingCall, PermissionRequest

    return PendingCall(
        id=pid,
        request=PermissionRequest(
            tool_name="bash_exec", tool_params={"command": "rm -rf ~/old-notes"},
            klass="shell-destructive", key=None, grantable=False,
            reason="destructive", display=rendering,
        ),
        rendering=rendering,
        agent_label="",
        session_id="s",
        created_at=0.0,
    )


def _staged_event(pending, total: int = 1):
    from localharness.core.events import PermissionStaged

    return PermissionStaged(
        agent_id="main", session_id="s", pending=pending, total=total, channel="terminal",
    )


class _BellOutput:
    """Stands in for the box app's Output: the bell is the only call under test."""

    def __init__(self) -> None:
        self.bells = 0

    def bell(self) -> None:
        self.bells += 1


class TestPendingRow:
    """The row UNDER the bottom bar: one muted line per queue, naming the oldest parked call."""

    def _text(self, frags) -> str:
        return "".join(t for _style, t in frags)

    def test_row_is_hidden_while_nothing_is_parked(self):
        ch = _channel()
        ch._box_active = True
        assert ch._box_pending_frags() == [], "an empty queue costs no line at all"

    def test_row_names_the_oldest_and_counts_the_rest(self):
        ch = _channel()
        ch._box_active = True
        ch.box_set_pending([_pending(3), _pending(4, "bash_exec: sudo apt install x")])
        text = self._text(ch._box_pending_frags())
        assert "2 decisions pending" in text
        assert "#3" in text and "rm -rf ~/old-notes" in text, "the OLDEST is the one shown"
        assert "sudo apt" not in text, "the rest are a count, not a list"
        assert "ctrl+y run · ctrl+n skip · /pending" in text

    def test_one_parked_call_reads_as_one_decision(self):
        ch = _channel()
        ch._box_active = True
        ch.box_set_pending([_pending(1)])
        text = self._text(ch._box_pending_frags())
        assert "1 decision pending" in text and "decisions" not in text

    def test_row_is_muted_not_an_error(self):
        ch = _channel()
        ch._box_active = True
        ch.box_set_pending([_pending(1)])
        assert all(style == "class:hint" for style, _ in ch._box_pending_frags())

    async def _render(self, pending_frags):
        """Draw the real app once and return what the screen got — the row has to be IN the
        layout, not merely rendered by a function nobody mounted."""
        from prompt_toolkit.output.plain_text import PlainTextOutput

        sink, holder = StringIO(), {}
        with create_pipe_input() as inp, create_app_session(
            input=inp, output=PlainTextOutput(sink)
        ):
            app = _build_persistent_input_app(
                InMemoryHistory(), ">",
                on_submit=lambda _t: None, on_interrupt=lambda: None,
                on_eof=lambda: holder["app"].exit(),
                hint_fn=lambda: [("class:hint", "  tab commands")], right_fn=lambda: [],
                status_fn=lambda: [], pending_fn=lambda: pending_frags,
            )
            holder["app"] = app
            inp.send_text("\x04")
            await asyncio.wait_for(app.run_async(), timeout=10.0)
        return sink.getvalue()

    async def test_the_row_is_drawn_under_the_footer(self):
        screen = await self._render([("class:hint", "  ⏸ 1 decision pending")])
        assert "⏸ 1 decision pending" in screen
        assert screen.index("tab commands") < screen.index("⏸"), "under the bottom bar, not over"

    async def test_an_empty_queue_draws_no_row_at_all(self):
        assert "⏸" not in await self._render([])

    def test_answering_empties_the_row(self):
        ch = _channel()
        ch._box_active = True
        ch.box_set_pending([_pending(1)])
        ch.box_set_pending([])
        assert ch._box_pending_frags() == []


class TestStagedNotice:
    """PermissionStaged → one inline transcript line, and a bell only for somebody who is there."""

    async def test_staged_prints_the_inline_notice(self):
        ch = _channel()
        await ch.on_permission_staged(_staged_event(_pending(2), total=3))
        out = ch._console.file.getvalue()
        assert "needs you" in out and "#2" in out and "rm -rf ~/old-notes" in out
        assert "/approve" not in out, "the row under the box carries the legend, not the transcript"

    async def test_the_channel_picks_the_event_up_itself(self, tmp_path):
        """`start()` is where a channel takes its events off the bus. Without the subscription
        `send_pending_notice` is a method nobody calls — and the notice is the only thing that
        tells a person a step was skipped, since the model was told to carry on without it."""
        ch = _channel()
        ch._history_file = str(tmp_path / "history")
        await ch.start()
        try:
            await ch.bus.publish(_staged_event(_pending(7), total=2))
            assert "#7" in ch._console.file.getvalue()
        finally:
            await ch.stop()

    async def test_staged_while_present_rings_the_bell_once(self):
        ch = _channel()
        ch._box_active = True
        ch._box_app = types.SimpleNamespace(output=_BellOutput())
        ch._note_keystroke()  # somebody just typed
        await ch.on_permission_staged(_staged_event(_pending(1)))
        assert ch._box_app.output.bells == 1

    async def test_staged_while_away_is_silent(self):
        ch = _channel()
        ch._box_active = True
        ch._box_app = types.SimpleNamespace(output=_BellOutput())
        ch._last_keystroke_at = time.monotonic() - PRESENCE_WINDOW_S - 1.0
        await ch.on_permission_staged(_staged_event(_pending(1)))
        assert ch._box_app.output.bells == 0, "no beeping into an empty room"
        assert "needs you" in ch._console.file.getvalue(), "the transcript line still lands"


class TestPresence:
    async def test_a_keystroke_stamps_the_clock(self):
        ch = _channel()
        assert ch._present() is False, "nobody has typed yet"
        ch._note_keystroke()
        assert ch._present() is True

    async def test_coming_back_says_what_is_waiting_once(self):
        ch = _channel()
        ch._box_active = True
        ch.box_set_pending([_pending(1), _pending(2)])
        ch._last_keystroke_at = time.monotonic() - PRESENCE_WINDOW_S - 1.0
        ch._note_keystroke()
        ch._note_keystroke()  # the very next key is not a second return
        await asyncio.sleep(0)  # the line is printed from a task (the hook is synchronous)
        out = ch._console.file.getvalue()
        assert out.count("while you were away") == 1
        assert "2 decisions pending" in out

    async def test_no_return_line_with_an_empty_queue(self):
        ch = _channel()
        ch._box_active = True
        ch._last_keystroke_at = time.monotonic() - PRESENCE_WINDOW_S - 1.0
        ch._note_keystroke()
        await asyncio.sleep(0)
        assert ch._console.file.getvalue() == ""

    async def test_a_short_gap_is_not_a_return(self):
        ch = _channel()
        ch._box_active = True
        ch.box_set_pending([_pending(1)])
        ch._last_keystroke_at = time.monotonic() - 1.0
        ch._note_keystroke()
        await asyncio.sleep(0)
        assert "while you were away" not in ch._console.file.getvalue()


class TestPendingHotkeys:
    """ctrl+y / ctrl+n on the live box: a control event NOW, never a queued slash command."""

    async def _drive_app(self, feed: str):
        answers: list[bool] = []
        holder: dict = {}

        def on_eof() -> None:
            holder["app"].exit()

        with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
            app = _build_persistent_input_app(
                InMemoryHistory(), ">",
                on_submit=lambda _t: None, on_interrupt=lambda: None, on_eof=on_eof,
                hint_fn=lambda: [("class:hint", " ")], right_fn=lambda: [],
                status_fn=lambda: [],
                on_pending_answer=answers.append,
            )
            holder["app"] = app
            inp.send_text(feed)
            await asyncio.wait_for(app.run_async(), timeout=10.0)
        return answers

    async def test_ctrl_y_approves_and_ctrl_n_denies(self):
        assert await self._drive_app("\x19\x0e\x04") == [True, False]

    async def test_the_hotkeys_never_reach_the_line(self):
        subs: list[str] = []
        holder: dict = {}

        def on_eof() -> None:
            holder["app"].exit()

        with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
            app = _build_persistent_input_app(
                InMemoryHistory(), ">",
                on_submit=subs.append, on_interrupt=lambda: None, on_eof=on_eof,
                hint_fn=lambda: [("class:hint", " ")], right_fn=lambda: [],
                status_fn=lambda: [],
            )
            holder["app"] = app
            inp.send_text("\x19ok\r\x04")
            await asyncio.wait_for(app.run_async(), timeout=10.0)
        assert subs == ["ok"], "ctrl+y is a key, not a character"

    async def test_live_box_posts_the_control_event_for_the_oldest(self):
        ch = _channel()
        ch._history = InMemoryHistory()
        q: asyncio.Queue = asyncio.Queue()
        with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
            await ch.start_input_box(q, on_interrupt=lambda: None)
            try:
                ch.box_set_pending([_pending(1)])
                inp.send_text("\x19")
                assert await asyncio.wait_for(q.get(), timeout=10.0) == ("approve_pending", None)
                inp.send_text("\x0e")
                assert await asyncio.wait_for(q.get(), timeout=10.0) == ("deny_pending", None)
            finally:
                await ch.stop_input_box()

    async def test_with_nothing_parked_the_hotkeys_do_nothing(self):
        ch = _channel()
        ch._history = InMemoryHistory()
        q: asyncio.Queue = asyncio.Queue()
        with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
            await ch.start_input_box(q, on_interrupt=lambda: None)
            try:
                inp.send_text("\x19\x0ehello\r")
                # The submit that FOLLOWS them is the first event on the queue: neither key
                # produced one of its own.
                assert await asyncio.wait_for(q.get(), timeout=10.0) == ("submit", "hello")
            finally:
                await ch.stop_input_box()

    async def test_typing_into_the_live_box_stamps_presence(self):
        ch = _channel()
        ch._history = InMemoryHistory()
        ch._last_keystroke_at = time.monotonic() - PRESENCE_WINDOW_S - 1.0
        q: asyncio.Queue = asyncio.Queue()
        with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
            await ch.start_input_box(q, on_interrupt=lambda: None)
            try:
                ch._last_keystroke_at = time.monotonic() - PRESENCE_WINDOW_S - 1.0
                assert ch._present() is False
                inp.send_text("hi\r")
                await asyncio.wait_for(q.get(), timeout=10.0)
                assert ch._present() is True, "the key press reached the presence hook"
            finally:
                await ch.stop_input_box()
