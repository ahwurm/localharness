"""#135 — the /model picker's pre-fill is a LOAN, not a line the user wrote.

Bare `/model` renders the model list and then pre-fills the box with `/model ` + the first
model highlighted (the one-Enter picker). Whatever the user typed NEXT used to be appended to
that text and submitted as one line: `/quit` right after a bare `/model` arrived at the REPL as
`/model qwen3.8-27b/quit` — "Unknown model", and the quit silently no-oped.

These drive the REAL injection path (TerminalChannel.box_open_model_menu writing into the live
box buffer, exactly as the REPL calls it after the listing) with real keystrokes through the
headless prompt_toolkit harness, and assert on the lines the REPL is handed.
"""
from __future__ import annotations

import asyncio
from io import StringIO

from prompt_toolkit.application import create_app_session
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from localharness.channels.terminal import (
    _MODEL_PICK_PREFIX,
    TERMINAL_THEME,
    TerminalChannel,
    _build_persistent_input_app,
    _picker_loan_live,
)
from localharness.core.bus import EventBus

MODELS = ["qwen3.8-27b", "qwen3-8b"]  # first = active, i.e. what the picker highlights


async def _pick_then_feed(feed: str, models: list[str] = MODELS) -> list[str]:
    """Open the picker the way the REPL does (bare /model → box_open_model_menu) into a LIVE
    box, then send `feed` as keystrokes. Returns the lines the box submitted."""
    ch = TerminalChannel(EventBus(), {})
    ch._console = Console(file=StringIO(), force_terminal=True, width=120,
                          theme=TERMINAL_THEME, highlight=False)
    ch._box_active = True
    ch.model_names_fn = lambda: list(models)
    subs: list[str] = []
    holder: dict = {}

    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        app = _build_persistent_input_app(
            InMemoryHistory(), ">",
            on_submit=subs.append, on_interrupt=lambda: None,
            on_eof=lambda: holder["app"].exit(),
            hint_fn=lambda: [], pct_fn=lambda: None, status_fn=lambda: [],
            model_names_fn=ch._model_names_for_menu,
        )
        holder["app"] = app
        ch._box_app = app
        buf = app._lh_input_buffer
        ch.box_open_model_menu()  # pre-fill lands in the live buffer…
        for _ in range(20):       # …and the async completer previews the highlighted model
            if buf.text != "/model ":
                break
            await asyncio.sleep(0)
        assert buf.text == f"/model {models[0]}", "precondition: the picker pre-fill is in the box"
        inp.send_text(feed)
        await asyncio.wait_for(app.run_async(), timeout=10.0)
    return subs


class TestPickerPrefillIsALoan:
    async def test_typing_after_the_picker_replaces_the_prefill(self):
        # THE BUG (#135): '/quit' typed right after a bare '/model' submitted as
        # '/model qwen3.8-27b/quit' → "Unknown model", session stayed up.
        assert await _pick_then_feed("/quit\r\x04") == ["/quit"]

    async def test_pasting_over_the_prefill_replaces_it_too(self):
        # Same misparse via bracketed paste (ESC[200~ … ESC[201~) — pasting a line is typing it.
        assert await _pick_then_feed("\x1b[200~summarize the diff\x1b[201~\r\x04") == [
            "summarize the diff"
        ]

    async def test_backspace_over_the_prefill_clears_the_whole_loan(self):
        """Backspace is declining the loan, not editing it: one press leaves an EMPTY line —
        not '/model qwen3.8-27' — so the next command parses as itself. (Critic-found gap:
        default backward-delete-char expired the loan but stranded the fragment.)"""
        assert await _pick_then_feed("\x7f/quit\r\x04") == ["/quit"]

    async def test_one_enter_still_picks_the_highlighted_model(self):
        # The pre-fill stays submit-safe: Enter on it is the one-Enter picker, and the first
        # entry is the ACTIVE model, so a bare /model + Enter is a no-op switch ("already active").
        assert await _pick_then_feed("\r\x04") == ["/model qwen3.8-27b"]

    async def test_escape_still_takes_the_prefill_back(self):
        # #123 on the real injected path: declining leaves an EMPTY line, not '/model '.
        assert await _pick_then_feed("\x1brewrite the parser tests\r\x04") == [
            "rewrite the parser tests"
        ]

    async def test_the_loan_is_taken_back_once_never_from_a_typed_line(self):
        # Over-reach guard: the replace fires for the app's OWN pre-fill only. A '/model …' line
        # the user types by hand afterwards must survive keystroke for keystroke (and the first
        # line must not collapse to its last character either).
        assert await _pick_then_feed("/quit\r/model qwen3-8b\r\x04") == [
            "/quit", "/model qwen3-8b",
        ]

    def test_the_loan_covers_the_gap_before_the_menu_settles_then_expires(self):
        # The completer is async: for a beat after the injection the line is a bare '/model '
        # with no preview yet, and typing in THAT window must replace it too. And once the line
        # is the user's, the loan is dropped for good — no later '/model …' they type by hand
        # can be taken away from them.
        buf = Buffer()
        buf.text = _MODEL_PICK_PREFIX
        buf._lh_picker_loan = True
        assert _picker_loan_live(buf, _MODEL_PICK_PREFIX) is True
        buf.text = "summarize the diff"
        assert _picker_loan_live(buf, _MODEL_PICK_PREFIX) is False
        buf.text = _MODEL_PICK_PREFIX  # typed by hand this time
        assert _picker_loan_live(buf, _MODEL_PICK_PREFIX) is False
