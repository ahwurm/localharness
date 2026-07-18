"""Slash-command completion: the Completer (triggers only on a leading-slash first token, prefix
filters) and the menu keybindings driven headlessly through a prompt_toolkit pipe input — the same
create_pipe_input harness as test_terminal_input_box.py.
"""
from __future__ import annotations

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.document import Document
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from localharness.channels.terminal import SlashCommandCompleter, _build_persistent_input_app
from localharness.cli.slash_commands import SLASH_COMMANDS


def _complete(text: str) -> list:
    c = SlashCommandCompleter()
    return list(c.get_completions(Document(text, len(text)), None))


def _texts(text: str) -> set[str]:
    return {c.text for c in _complete(text)}


# --------------------------------------------------------------------------- completer unit tests
def test_completer_silent_without_leading_slash():
    assert _complete("") == []
    assert _complete("hello") == []
    assert _complete("do /help now") == []   # slash not at the start


def test_completer_lists_all_commands_on_bare_slash():
    assert _texts("/") == {name for name, _ in SLASH_COMMANDS}


def test_completer_prefix_filters():
    assert _texts("/m") == {"/memory", "/model"}   # both m-commands
    assert _texts("/me") == {"/memory"}             # /model is /mo…, excluded
    assert _texts("/mo") == {"/model"}
    assert _texts("/h") == {"/help"}


def test_completer_is_case_insensitive():
    assert _texts("/ME") == {"/memory"}


def test_completer_stops_once_into_arguments():
    # a space means we're past the command token — no menu over `/memory show 12`.
    assert _complete("/memory ") == []
    assert _complete("/memory show 12") == []


def test_completion_carries_name_and_description():
    comps = _complete("/help")
    assert len(comps) == 1
    assert comps[0].text == "/help"
    assert comps[0].display_meta_text == "Show this help message"


# --------------------------------------------------------------------------- headless menu nav
class TestSlashMenuKeybindings:
    async def _drive(self, feed: str):
        subs: list[str] = []
        holder: dict = {}

        def on_submit(t: str) -> None:
            subs.append(t)

        def on_eof() -> None:
            holder["app"].exit()

        with create_pipe_input() as inp:
            with create_app_session(input=inp, output=DummyOutput()):
                app = _build_persistent_input_app(
                    InMemoryHistory(), ">",
                    on_submit=on_submit, on_interrupt=lambda: None, on_eof=on_eof,
                    hint_fn=lambda: [("class:hint", " ")], pct_fn=lambda: None,
                    status_fn=lambda: [],
                )
                holder["app"] = app
                inp.send_text(feed)
                await app.run_async()
        return subs

    async def test_enter_submits_a_fully_typed_command_despite_open_menu(self):
        # "/help" typed in full: the menu is showing but nothing is highlighted, so Enter submits
        # (Claude Code feel — the menu must not steal Enter from a complete command).
        subs = await self._drive("/help\r\x04")
        assert subs == ["/help"]

    async def test_tab_accepts_the_matching_command_then_enter_submits(self):
        # "/mem" -> sole match "/memory"; Tab accepts it into the line, Enter submits it.
        subs = await self._drive("/mem\t\r\r\x04")
        assert subs == ["/memory"]

    async def test_arrow_navigates_open_menu_then_enter_accepts(self):
        # "/m" -> menu [/model, /memory]; Tab opens + highlights the first, Down moves to the next,
        # Enter accepts it into the line, Enter submits.
        subs = await self._drive("/m\t\x1b[B\r\r\x04")
        assert subs == ["/memory"]

    async def test_plain_text_is_unaffected_by_the_menu(self):
        # non-slash input never triggers completion; Enter submits verbatim.
        subs = await self._drive("just some text\r\x04")
        assert subs == ["just some text"]
