"""TerminalChannel.run_interview and can_manage: one interview node per call over the suspension's own output."""
from __future__ import annotations

import asyncio
import signal
import sys
from io import StringIO

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

import localharness.channels.terminal as terminal_mod
from localharness.channels.terminal import ERASE_APPLICATIONS_WHEN_DONE, TERMINAL_THEME, TerminalChannel
from localharness.cli import interview as interview_mod
from localharness.cli.interview import Answer, Option, Unreachable
from localharness.core.bus import EventBus

OPTIONS = [Option("a", "Alpha"), Option("b", "Beta"), Option("c", "Gamma")]
SENTINEL = object()


def _channel(force_terminal: bool = True) -> TerminalChannel:
    from rich.console import Console

    ch = TerminalChannel(EventBus(), {})
    ch._console = Console(
        file=StringIO(), force_terminal=force_terminal, width=120, theme=TERMINAL_THEME, highlight=False
    )
    return ch


class _App:
    def __init__(self, result=None, raises=None):
        self.result, self.raises, self.kw = result, raises, None

    async def run_async(self, **kw):
        self.kw = kw
        if self.raises is not None:
            raise self.raises
        return self.result


@pytest.fixture
def drawable(monkeypatch):
    monkeypatch.setattr(interview_mod, "can_draw", lambda output=None: None)


def _stub_build(monkeypatch, app):
    calls: list[tuple[tuple, dict]] = []

    def build(*args, **kw):
        calls.append((args, kw))
        return app

    monkeypatch.setattr(interview_mod, "build_interview", build)
    return calls


async def test_run_interview_draws_one_node_and_returns_the_apps_answer(monkeypatch, drawable):
    app = _App(Answer(("b", "a"), False))
    calls = _stub_build(monkeypatch, app)
    ch = _channel()

    got = await ch.run_interview("q?", OPTIONS, default=frozenset(), cancel="x", multi=True)

    assert got == Answer(("b", "a"), False)
    (args, kw), = calls
    assert args == ("q?", OPTIONS)
    assert kw == {
        "default": frozenset(), "cancel": "x", "multi": True,
        "output": ch._tty_output(), "erase_when_done": ERASE_APPLICATIONS_WHEN_DONE,
    }
    assert app.kw == {"handle_sigint": False}


@pytest.mark.parametrize("why", [Unreachable.NO_OPERATOR, Unreachable.DUMB, Unreachable.TOO_SMALL, Unreachable.ERROR])
async def test_an_unreachable_terminal_draws_nothing_and_says_why(monkeypatch, why):
    monkeypatch.setattr(interview_mod, "can_draw", lambda output=None: why)
    calls = _stub_build(monkeypatch, _App())

    got = await _channel().run_interview("q?", OPTIONS, default=0, cancel="x")

    assert got == Answer(("x",), True, reached=False, why=why)
    assert calls == []


@pytest.mark.parametrize("exc", [KeyboardInterrupt, EOFError])
async def test_a_keyboard_interrupt_or_eof_out_of_the_app_is_a_reached_cancel(monkeypatch, drawable, exc):
    _stub_build(monkeypatch, _App(raises=exc()))

    got = await _channel().run_interview("q?", OPTIONS, default=0, cancel="x")

    assert got == Answer(("x",), True)
    assert got.reached and got.why is None


async def test_the_redirected_path_draws_and_asks_on_the_tty_output(monkeypatch):
    import prompt_toolkit.output.defaults as defaults

    monkeypatch.setattr(defaults, "create_output", lambda **kw: SENTINEL)
    asked: list = []
    monkeypatch.setattr(interview_mod, "can_draw", lambda output=None: asked.append(output))
    calls = _stub_build(monkeypatch, _App(Answer(("a",), False)))
    ch = _channel(force_terminal=False)

    await ch.run_interview("q?", OPTIONS, default=0, cancel="x")

    assert asked == [SENTINEL] and calls[0][1]["output"] is SENTINEL


async def _in_span(ch, inp, keys, **kw):
    starts: list[int] = []
    real_start = ch.start_input_box

    async def counting_start(*args):
        starts.append(1)
        await real_start(*args)

    ch._history = InMemoryHistory()
    await ch.start_input_box(asyncio.Queue(), on_interrupt=lambda: None)
    ch.start_input_box = counting_start
    try:
        async with ch.suspended_box():
            inp.send_text(keys)
            got = await asyncio.wait_for(ch.run_interview("q?", OPTIONS, default=0, cancel="x", **kw), 10.0)
        return got, len(starts), ch._box_active
    finally:
        await ch.stop_input_box()


async def test_down_then_enter_inside_a_real_span_answers_the_second_option(drawable):
    ch = _channel()
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        got, restarts, active = await _in_span(ch, inp, "\x1b[B\r")

    assert got == Answer(("b",), False)
    assert restarts == 1 and active is True


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
async def test_ctrl_c_as_input_cancels_the_node_and_touches_neither_the_task_nor_sigint(drawable):
    ch = _channel()
    before = signal.getsignal(signal.SIGINT)
    me = asyncio.current_task()
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        got, _restarts, _active = await _in_span(ch, inp, "\x03")

    assert got == Answer(("x",), True)
    assert not me.cancelling() and signal.getsignal(signal.SIGINT) is before


def test_run_interview_never_opens_the_span_itself():
    import inspect

    assert "suspended_box" not in inspect.getsource(TerminalChannel.run_interview)


def test_can_manage_is_no_tty_without_asking_can_draw(monkeypatch):
    asked: list = []
    monkeypatch.setattr(terminal_mod, "cannot_ask_reason", lambda: "no tty")
    monkeypatch.setattr(interview_mod, "can_draw", lambda output=None: asked.append(output))

    assert _channel().can_manage() is Unreachable.NO_TTY
    assert asked == []


def test_can_manage_on_the_box_path_asks_can_draw_about_the_default_output(monkeypatch):
    asked: list = []
    monkeypatch.setattr(terminal_mod, "cannot_ask_reason", lambda: None)
    monkeypatch.setattr(interview_mod, "can_draw", lambda output=None: (asked.append(output), Unreachable.DUMB)[1])

    assert _channel().can_manage() is Unreachable.DUMB
    assert asked == [None]


def test_can_manage_on_the_redirected_path_asks_about_the_tty_output(monkeypatch):
    import prompt_toolkit.output.defaults as defaults

    asked: list = []
    monkeypatch.setattr(terminal_mod, "cannot_ask_reason", lambda: None)
    monkeypatch.setattr(defaults, "create_output", lambda **kw: SENTINEL)
    monkeypatch.setattr(interview_mod, "can_draw", lambda output=None: asked.append(output))

    assert _channel(force_terminal=False).can_manage() is None
    assert asked == [SENTINEL]


def _raising_create_output(monkeypatch):
    import prompt_toolkit.output.defaults as defaults

    def boom(**kw):
        raise OSError("no tty")

    monkeypatch.setattr(defaults, "create_output", boom)


async def test_run_interview_reaches_nobody_when_the_tty_output_cannot_be_built(monkeypatch):
    _raising_create_output(monkeypatch)
    calls = _stub_build(monkeypatch, _App(Answer(("a",), False)))

    got = await _channel(force_terminal=False).run_interview("q?", OPTIONS, default=0, cancel="x")

    assert got == Answer(("x",), True, reached=False, why=Unreachable.ERROR) and calls == []


def test_can_manage_reports_error_when_the_tty_output_cannot_be_built(monkeypatch):
    _raising_create_output(monkeypatch)
    monkeypatch.setattr(terminal_mod, "cannot_ask_reason", lambda: None)

    assert _channel(force_terminal=False).can_manage() is Unreachable.ERROR
