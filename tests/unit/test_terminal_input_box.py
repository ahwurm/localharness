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
from contextlib import contextmanager
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
        assert "Describe a task" in self._text(ch._box_hint_frags())

        ch.box_set_queued(2)
        assert "queued (2)" in self._text(ch._box_hint_frags())

        # FIX 2: the working glyph moved OUT of the bottom border into the status row above
        # the box. The border keeps input-metadata (queued / decision flash / hint) only.
        ch.box_notify_working(True)
        assert "working" not in self._text(ch._box_hint_frags()), "border no longer carries it"
        assert "working" in self._text(ch._box_status_frags()), "status row does"

    def test_decision_flash_shows_then_can_clear(self):
        ch = _channel()
        ch._box_active = True
        ch.box_flash_decision("→ nudging current turn")
        assert "nudging" in self._text(ch._box_hint_frags())
        ch._decision_flash = ""  # simulate the timed clear
        assert "nudging" not in self._text(ch._box_hint_frags())


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

    async def test_echo_escapes_user_markup(self):
        ch = _channel()
        ch._box_active = True
        await ch.box_echo_prompt("[bold]not markup[/bold]")
        # rich markup in the user's text must render literally, never be interpreted.
        assert "[bold]not markup[/bold]" in self._out(ch)


class TestPersistentAppKeybindings:
    async def _drive(self, feed: str, models: list[str] | None = None):
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
                    status_fn=lambda: [],
                    model_names_fn=(lambda: list(models)) if models is not None else None,
                )
                holder["app"] = app
                inp.send_text(feed)
                # wait_for guard: a feed that never reaches an exit key would otherwise hang the
                # whole suite on this app instead of failing this one test.
                await asyncio.wait_for(app.run_async(), timeout=10.0)
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
                hint_fn=lambda: [], pct_fn=lambda: None, status_fn=lambda: [],
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
