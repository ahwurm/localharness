"""prompt_line and confirm_line (cli/interview.py): a fresh plain-text session per ask, the cancel keys, flushed input."""
from __future__ import annotations

import asyncio
import io
import sys
import termios
from types import SimpleNamespace

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console

from localharness.cli import interview
from localharness.cli.interview import confirm_line, prompt_line

OUT = Console(file=io.StringIO())
ERR = Console(file=io.StringIO(), stderr=True)


@pytest.fixture
def sessions(monkeypatch):
    made = []

    class Recording(PromptSession):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.prompt_calls = []
            made.append(self)

        async def prompt_async(self, *a, **kw):
            self.prompt_calls.append(kw)
            return await super().prompt_async(*a, **kw)

    monkeypatch.setattr(interview, "PromptSession", Recording)
    return made


def _run(keys: str, *calls):
    async def go():
        return [await c() for c in calls]
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text(keys)
        return asyncio.run(go())


def _ask(keys: str, **kw):
    kw.setdefault("console", OUT)
    return _run(keys, lambda: prompt_line("Name", **kw))[0]


def _confirm(keys: str, default: bool = True, **kw):
    kw.setdefault("console", OUT)
    return _run(keys, lambda: confirm_line("Keep it?", default=default, **kw))[0]


def _text(session) -> str:
    return "".join(t for _, t in session.message)


def test_typed_text_comes_back_stripped():
    assert _ask("  qwen \r") == "qwen"


def test_enter_accepts_the_default_and_an_empty_default_is_empty():
    assert _ask("\r", default="main") == "main"
    assert _ask("\r") == ""


def test_secret_sets_is_password_on_the_session(sessions):
    assert _ask("hunter2\r", secret=True) == "hunter2"
    assert sessions[-1].is_password is True
    _ask("x\r")
    assert sessions[-1].is_password is False


@pytest.mark.parametrize("keys", ["\x03", "\x04"])
def test_ctrl_c_and_ctrl_d_cancel_both_callables(keys):
    assert _ask(keys) is None
    assert _confirm(keys) is None


@pytest.mark.parametrize("keys,default,expected", [
    ("y\r", False, True), ("YES\r", False, True), ("n\r", True, False), ("No\r", True, False),
    ("\r", True, True), ("\r", False, False), ("maybe\rn\r", True, False),
])
def test_confirm_reads_yes_no_or_enter_and_asks_again_on_anything_else(keys, default, expected):
    assert _confirm(keys, default=default) is expected


def test_confirm_asks_again_on_a_fresh_session_showing_default_and_tail(sessions):
    assert _confirm("maybe\rn\r", default=True, tail="Esc cancels") is False
    assert len(sessions) == 2 and sessions[0] is not sessions[1]
    assert [_text(s) for s in sessions] == ["Keep it? [Y/n]  Esc cancels "] * 2
    _confirm("\r", default=False)
    assert _text(sessions[-1]) == "Keep it? [y/N] "


def test_the_question_is_plain_fragments_never_markup(sessions):
    _run("\r", lambda: prompt_line("<b>&amp;</b>", console=OUT))
    assert sessions[-1].message == [("", "<b>&amp;</b>")]


def test_two_asks_build_two_sessions_and_the_first_answer_is_not_in_the_second_history(sessions):
    assert _run("alpha\rbeta\r", lambda: prompt_line("A", console=OUT, secret=True),
                lambda: prompt_line("B", console=OUT)) == ["alpha", "beta"]
    assert len(sessions) == 2 and sessions[0] is not sessions[1]
    assert sessions[0].history.get_strings() == ["alpha"]
    assert sessions[1].history.get_strings() == ["beta"]


def test_handle_sigint_is_off_for_the_prompt(sessions):
    _ask("x\r")
    assert sessions[-1].prompt_calls[0]["handle_sigint"] is False


def test_a_stderr_console_puts_the_prompt_on_the_tty_output(monkeypatch, sessions):
    calls, out = [], DummyOutput()
    monkeypatch.setattr(interview, "create_output", lambda **kw: calls.append(kw) or out)
    assert _ask("x\r", console=ERR) == "x"
    assert calls == [{"always_prefer_tty": True}] and sessions[-1].output is out


def test_a_stdout_console_leaves_the_output_to_the_session(monkeypatch):
    monkeypatch.setattr(interview, "create_output", lambda **kw: pytest.fail("created an output"))
    assert _ask("x\r", console=OUT) == "x"


def test_pending_input_is_flushed_before_the_session_exists(monkeypatch, sessions):
    flushed = []
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(fileno=lambda: 7))
    monkeypatch.setattr(termios, "tcflush", lambda fd, queue: flushed.append((fd, queue, len(sessions))))
    assert _ask("x\r") == "x"
    assert flushed == [(7, termios.TCIFLUSH, 0)]


def test_a_failing_or_missing_termios_is_tolerated(monkeypatch):
    def boom(fd, queue):
        raise termios.error(25, "not a tty")
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(fileno=lambda: 7))
    monkeypatch.setattr(termios, "tcflush", boom)
    assert _ask("x\r") == "x"
    monkeypatch.setitem(sys.modules, "termios", None)
    assert _ask("y\r") == "y"
