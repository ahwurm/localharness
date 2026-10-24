"""The terminal's rendering of an ASK: `[y]es once / [a]lways here / [n]o / [N]ever here`.

PRD §3.5. Driven through the same headless prompt_toolkit harness the input-box suite uses
(create_pipe_input + DummyOutput + create_app_session), so these are real keystrokes through the
real key bindings, not a stubbed prompt.
"""
from __future__ import annotations

import asyncio
import os
import signal
import sys
from io import StringIO

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from localharness.agent.gate_types import PermissionRequest
from localharness.channels.base import ChannelAdapter, sanitize_for_display
from localharness.channels.terminal import (
    PERMISSION_ANSWER_LINES,
    PERMISSION_OPTIONS_GRANTABLE,
    PERMISSION_OPTIONS_UNGRANTABLE,
    TERMINAL_THEME,
    TerminalChannel,
    _build_permission_app,
)
from localharness.core.bus import EventBus
from localharness.core.interrupt import sigint_cancels


def _channel() -> TerminalChannel:
    from rich.console import Console

    ch = TerminalChannel(EventBus(), {})
    ch._console = Console(
        file=StringIO(), force_terminal=True, width=120, theme=TERMINAL_THEME, highlight=False
    )
    return ch


def _request(grantable: bool = True, display: str | None = None) -> PermissionRequest:
    return PermissionRequest(
        tool_name="bash_exec",
        tool_params={"command": "cargo publish"},
        klass="shell-unfamiliar" if grantable else "shell-destructive",
        key="cargo publish" if grantable else None,
        grantable=grantable,
        reason="not seen in this workspace before",
        display=display or "bash_exec: cargo publish  (shell-unfamiliar — not seen before)",
    )


async def _answer(keys: str, grantable: bool = True) -> str:
    ch = _channel()
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text(keys)
        decision = await asyncio.wait_for(ch.ask_permission(_request(grantable)), timeout=10.0)
    return decision.kind


# ---------------------------------------------------------------- the four keys

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key,kind",
    [("y", "allow_once"), ("a", "allow_always"), ("n", "reject_once"), ("N", "reject_always")],
)
async def test_each_key_maps_to_its_decision(key, kind):
    assert await _answer(key) == kind


@pytest.mark.asyncio
@pytest.mark.parametrize("keys", ["\x03", "\r", "\x1b"])
async def test_interrupt_enter_and_escape_all_mean_no_this_once(keys):
    """Fail closed, and never durably: an accidental keystroke must not write a deny."""
    assert await _answer(keys) == "reject_once"


@pytest.mark.asyncio
async def test_an_ungrantable_request_offers_only_the_once_pair():
    """`[a]lways here` on a destructive call would be a lie — it asks every time."""
    assert await _answer("y", grantable=False) == "allow_once"
    assert await _answer("n", grantable=False) == "reject_once"
    # 'a' is not bound at all here, so it is ignored; the following 'n' answers.
    assert await _answer("an", grantable=False) == "reject_once"


@pytest.mark.asyncio
async def test_the_question_is_printed_where_the_user_can_read_it():
    ch = _channel()
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text("y")
        await asyncio.wait_for(ch.ask_permission(_request()), timeout=10.0)
    printed = ch._console.file.getvalue()
    assert "Permission needed" in printed
    assert "cargo publish" in printed


@pytest.mark.asyncio
async def test_control_characters_never_reach_the_screen():
    """F5: a `\\r` rewrites the line the human is reading and `\\x1b[2J` clears the screen, so a
    question could be answered that is not the question on screen. Stripped before rendering."""
    hostile = "bash_exec: echo \x1b[2Jharmless\rrm -rf /  (shell-unfamiliar)"
    ch = _channel()
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text("y")
        await asyncio.wait_for(ch.ask_permission(_request(display=hostile)), timeout=10.0)
    printed = ch._console.file.getvalue()
    assert "2J" not in printed
    assert "\r" not in printed
    assert "harmlessrm -rf /" in printed


def test_the_sanitizer_keeps_layout_and_drops_control():
    """Tab and newline are layout — the gate's multi-reason display is genuinely multi-line."""
    assert sanitize_for_display("a\tb\nc") == "a\tb\nc"
    assert sanitize_for_display("a\x1b[31mb\x07c\rd\x00e") == "abcde"
    assert sanitize_for_display("") == ""


def test_the_legends_name_every_option_the_keys_bind():
    assert "[y]es once" in PERMISSION_OPTIONS_GRANTABLE
    assert "[a]lways here" in PERMISSION_OPTIONS_GRANTABLE
    assert "[N]ever here" in PERMISSION_OPTIONS_GRANTABLE
    assert "[a]lways" not in PERMISSION_OPTIONS_UNGRANTABLE


# ------------------------------------------------------- what is left on screen

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key,line",
    [("y", "✓ allowed once"), ("a", "✓ always here"),
     ("n", "✗ denied"), ("N", "✗ never here")],
)
async def test_answering_leaves_one_confirmation_line_and_no_legend(key, line):
    """Owner, 2026-09-11: "I don't like how it persists a chat box every time I answer (a) (y)
    or whatever; it should just record my input."

    One keystroke, and what remains is the question and one short line. The legend was a menu,
    not a record, and it used to stay painted in the scrollback — along with the dead input box
    the question suspended — once per answer.
    """
    ch = _channel()
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text(key)
        await asyncio.wait_for(ch.ask_permission(_request()), timeout=10.0)
    printed = ch._console.file.getvalue()
    assert line in printed
    assert PERMISSION_OPTIONS_GRANTABLE not in printed
    assert "[y]es once" not in printed
    # exactly one confirmation, not one per redraw
    assert printed.count(line) == 1


@pytest.mark.asyncio
async def test_an_interrupted_prompt_still_says_what_it_recorded():
    """Ctrl+C answers "no, this once" — and the person has to be able to see that it did."""
    ch = _channel()
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text("\x03")
        decision = await asyncio.wait_for(ch.ask_permission(_request()), timeout=10.0)
    assert decision.kind == "reject_once"
    assert PERMISSION_ANSWER_LINES["reject_once"] in ch._console.file.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize("key,line", [
    ("y", "\u2713 trusted \u2014 remembered for this workspace"),
    ("n", "\u2717 not trusted \u2014 guarded for this session"),
])
async def test_the_trust_question_gets_its_own_confirmation(key, line):
    """"\u2713 allowed once" was a lie in both directions on the one ask that is not about a tool
    call: a yes to trusting a workspace is remembered forever, and a no denies nothing — it puts
    the session in guarded."""
    ch = _channel()
    request = PermissionRequest(
        tool_name="workspace",
        tool_params={"workspace": "/proj"},
        klass="workspace-trust",
        key="/proj",
        grantable=False,
        reason="trust this workspace?",
        display="Trust this workspace?\nAnswering yes records /proj.",
    )
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text(key)
        await asyncio.wait_for(ch.ask_permission(request), timeout=10.0)
    printed = ch._console.file.getvalue()
    assert line in printed
    assert "allowed once" not in printed and "denied" not in printed


def test_every_option_has_a_confirmation_line():
    """A key with no line would crash the answer path, which is the worst place to find out."""
    from localharness.channels.terminal import (
        PERMISSION_ANSWER_LINES_BY_CLASS,
        PERMISSION_KEYS_GRANTABLE,
    )

    kinds = set(PERMISSION_KEYS_GRANTABLE.values())
    assert kinds <= set(PERMISSION_ANSWER_LINES)
    for klass, lines in PERMISSION_ANSWER_LINES_BY_CLASS.items():
        assert kinds <= set(lines), f"{klass} can crash on a key it does not name"


def test_both_applications_erase_themselves_when_they_exit():
    """The mechanism behind the test above: prompt_toolkit's default leaves the last frame
    painted forever, which is what put a dead box and a dead legend in the scrollback per
    answer. Asserted on both apps, because the box is the half the owner actually named."""
    from prompt_toolkit.history import InMemoryHistory as _History

    from localharness.channels.terminal import _build_persistent_input_app

    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        assert _build_permission_app(PERMISSION_OPTIONS_GRANTABLE, True).erase_when_done is True
        box = _build_persistent_input_app(
            _History(), ">",
            on_submit=lambda _line: None,
            on_interrupt=lambda: None,
            on_eof=lambda: None,
            hint_fn=list,
            right_fn=list,
            status_fn=list,
        )
        assert box.erase_when_done is True


@pytest.mark.asyncio
async def test_the_app_exits_on_one_keystroke():
    """Guard on the shape: no Buffer, so a stray Enter cannot answer with leftover text."""
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        app = _build_permission_app(PERMISSION_OPTIONS_GRANTABLE, True)
        inp.send_text("a")
        assert await asyncio.wait_for(app.run_async(), timeout=10.0) == "allow_always"


# ------------------------------------------------------------------ the contract

class _Stream:
    """A stand-in for sys.stdin/stdout/stderr with a known isatty()."""

    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _streams(monkeypatch, *, stdin: bool, stdout: bool, stderr: bool = True) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream(stdin))
    monkeypatch.setattr(sys, "stdout", _Stream(stdout))
    monkeypatch.setattr(sys, "stderr", _Stream(stderr))


def test_the_terminal_declares_it_can_ask_and_shows_diffs(monkeypatch):
    _streams(monkeypatch, stdin=True, stdout=True)
    ch = _channel()
    assert ch.can_ask is True
    assert ch.has_review_surface is True
    assert ch.ask_holds_dialog is True


def test_a_piped_transcript_can_still_be_asked(monkeypatch):
    """Defect D6: `localharness start | tee session.log` keeps a real keyboard, so the ask must
    follow the INPUT side — judging it by stdout denied every prompt in a session where the
    person was sitting right there."""
    _streams(monkeypatch, stdin=True, stdout=False, stderr=True)
    assert _channel().can_ask is True


def test_a_session_with_no_keyboard_reports_that_it_cannot_ask(monkeypatch):
    """PRD §3.5's non-tty row: piped-in and --no-input runs route to the fail-closed path."""
    from localharness.channels.terminal import CANNOT_ASK_NO_STDIN, cannot_ask_reason

    _streams(monkeypatch, stdin=False, stdout=True)
    assert _channel().can_ask is False
    assert cannot_ask_reason() == CANNOT_ASK_NO_STDIN


def test_a_question_with_nowhere_to_draw_it_cannot_be_asked(monkeypatch):
    from localharness.channels.terminal import CANNOT_ASK_NO_TTY, cannot_ask_reason

    _streams(monkeypatch, stdin=True, stdout=False, stderr=False)
    assert _channel().can_ask is False
    assert cannot_ask_reason() == CANNOT_ASK_NO_TTY


@pytest.mark.asyncio
async def test_a_session_that_cannot_ask_says_so_once_at_startup(monkeypatch, tmp_path):
    """PRD §3.5 wants the fix named out loud; the gate logs it for the model, this is the line
    the human reads (defect D6)."""
    from rich.console import Console

    from localharness.channels.terminal import CANNOT_ASK_NO_STDIN

    _streams(monkeypatch, stdin=False, stdout=True)
    ch = TerminalChannel(EventBus(), {}, history_file=str(tmp_path / "hist"))
    err = StringIO()
    ch._console = Console(file=StringIO(), force_terminal=False, width=200)
    ch._err_console = Console(file=err, force_terminal=False, width=200)
    await ch.start()
    await ch.stop()
    printed = err.getvalue()
    assert "permission prompts are disabled" in printed
    assert CANNOT_ASK_NO_STDIN.split(",")[0] in printed
    assert "permissions.mode: unattended" in printed


@pytest.mark.asyncio
async def test_a_session_that_can_ask_prints_no_notice(monkeypatch, tmp_path):
    from rich.console import Console

    _streams(monkeypatch, stdin=True, stdout=True)
    ch = TerminalChannel(EventBus(), {}, history_file=str(tmp_path / "hist"))
    err = StringIO()
    ch._console = Console(file=StringIO(), force_terminal=False, width=200)
    ch._err_console = Console(file=err, force_terminal=False, width=200)
    await ch.start()
    await ch.stop()
    assert err.getvalue() == ""


def test_the_base_channel_cannot_ask_and_says_so_loudly():
    assert ChannelAdapter.can_ask is False
    assert ChannelAdapter.has_review_surface is False


@pytest.mark.asyncio
async def test_the_question_is_drawn_under_patch_stdout_and_never_blocks_a_writer():
    """D6: every other output path is protected from concurrent writers; this one was not.

    A turn does not stop while a human thinks. Subagents keep streaming, and each of those writes
    went straight onto the terminal the option legend was painting, corrupting the line the
    person answers from. Two things have to hold at once, and one of them is a trap:

    1. `patch_stdout(raw=True)` is active while the legend is up, so a concurrent write renders
       ABOVE the application instead of through it.
    2. `_output_lock` is NOT held across the wait. It is the lock every writer takes, so holding
       it until a human answers would freeze the whole session behind the dialog — the obvious
       fix and the wrong one.
    """
    from prompt_toolkit.patch_stdout import StdoutProxy

    async def _tokens():
        for token in ("sub", "agent"):
            yield token

    ch = _channel()
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        before = sys.stdout
        task = asyncio.create_task(ch.ask_permission(_request()))
        for _ in range(100):  # let the app reach its first paint
            await asyncio.sleep(0.01)
            if isinstance(sys.stdout, StdoutProxy):
                break
        assert isinstance(sys.stdout, StdoutProxy), "the question ran outside patch_stdout"

        # The load-bearing half: a subagent streaming mid-question must not wait for the human.
        streamed = await asyncio.wait_for(ch.send_streaming(_tokens(), agent_id="child"), 5.0)
        assert streamed == "subagent"

        inp.send_text("y")
        assert (await asyncio.wait_for(task, timeout=10.0)).kind == "allow_once"
        assert sys.stdout is before, "patch_stdout outlived the question"

    printed = ch._console.file.getvalue()
    assert "Permission needed" in printed and "subagent" in printed


@pytest.mark.asyncio
async def test_a_redirected_transcript_keeps_the_question_off_the_pipe(monkeypatch):
    """The patch replaces `sys.stderr` too and routes it into the app session's output — which
    on the redirected path IS the pipe. So the stdout-only guard is part of the fix, not an
    oversight: `localharness start > file` must still draw the question on the terminal."""
    from io import StringIO

    from rich.console import Console

    ch = _channel()
    ch._console = Console(file=StringIO(), force_terminal=False, width=120, theme=TERMINAL_THEME)
    err = StringIO()
    ch._err_console = Console(file=err, force_terminal=True, width=120, theme=TERMINAL_THEME)
    assert ch.can_run_input_box() is False

    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        monkeypatch.setattr(
            "prompt_toolkit.output.defaults.create_output", lambda **kw: DummyOutput()
        )
        inp.send_text("n")
        decision = await asyncio.wait_for(ch.ask_permission(_request()), timeout=10.0)

    assert decision.kind == "reject_once"
    assert "cargo publish" in err.getvalue(), "the question must go to the terminal, not the pipe"
    assert ch._console.file.getvalue() == ""


@pytest.mark.asyncio
async def test_the_persistent_box_is_restored_after_a_question():
    """The box and the question cannot both own the terminal; the box must come back."""
    ch = _channel()
    ch._history = InMemoryHistory()
    queue: asyncio.Queue = asyncio.Queue()
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        await ch.start_input_box(queue, on_interrupt=lambda: None)
        try:
            assert ch._box_active is True
            inp.send_text("y")
            decision = await asyncio.wait_for(ch.ask_permission(_request()), timeout=10.0)
            assert decision.kind == "allow_once"
            assert ch._box_active is True, "the input box was not restarted"
        finally:
            await ch.stop_input_box()


# ------------------------------------------- the human sees the denial (defect D7)

def _denied_observation(reason: str, tool_name: str = "write"):
    from localharness.core.events import Observation

    return Observation(
        agent_id="a",
        session_id="s",
        observation_type="tool_result",
        tool_call_id="tc-1",
        tool_name=tool_name,
        output="[DENIED]",
        error=f"Permission denied: {reason}",
    )


@pytest.mark.asyncio
async def test_a_denial_is_explained_to_the_person_watching(monkeypatch):
    """Verification A defect D7: the reason reached the model's observation only; the terminal
    printed `✗ write (exit 1): [DENIED]` and the human was left guessing."""
    _streams(monkeypatch, stdin=True, stdout=True)
    ch = _channel()
    await ch.on_observation(_denied_observation("not permitted in read-only mode"))
    printed = ch._console.file.getvalue()
    assert "[DENIED]" in printed
    assert "permission denied" in printed
    assert "not permitted in read-only mode" in printed
    assert printed.count("not permitted in read-only mode") == 1, "one line per denial"


@pytest.mark.asyncio
async def test_an_ordinary_tool_error_gets_no_permission_line(monkeypatch):
    from localharness.core.events import Observation

    _streams(monkeypatch, stdin=True, stdout=True)
    ch = _channel()
    await ch.on_observation(Observation(
        agent_id="a", session_id="s", observation_type="tool_result", tool_call_id="tc-1",
        tool_name="write", output="", error="Error: disk full",
    ))
    assert "permission denied" not in ch._console.file.getvalue()


def test_the_denied_label_is_one_definition():
    """The loop writes it, the channels match on it — one constant, imported by both."""
    from localharness.agent.gate import DENIED_OBSERVATION_PREFIX
    from localharness.channels.base import permission_denied_reason

    assert permission_denied_reason(f"{DENIED_OBSERVATION_PREFIX}because") == "because"
    assert permission_denied_reason("Error: something else") is None
    assert permission_denied_reason(None) is None


# ------------------------------------------------------------ the host: suspended_box

@pytest.mark.asyncio
async def test_suspend_stops_a_live_box_once_and_restarts_it_once():
    from tests.unit.test_picker import _fake_box

    ch = _channel()
    calls: list[str] = []
    _fake_box(ch, calls)
    async with ch.suspended_box():
        pass
    assert calls == ["stop", "start('queue', 'interrupt')"]


@pytest.mark.asyncio
async def test_suspend_restarts_the_box_when_the_body_raises():
    from tests.unit.test_picker import _fake_box

    ch = _channel()
    calls: list[str] = []
    _fake_box(ch, calls)
    with pytest.raises(RuntimeError, match="boom"):
        async with ch.suspended_box():
            raise RuntimeError("boom")
    assert calls == ["stop", "start('queue', 'interrupt')"]
    assert ch._suspended is False


@pytest.mark.asyncio
async def test_suspend_with_no_live_box_starts_and_stops_nothing():
    from tests.unit.test_picker import _fake_box

    ch = _channel()
    calls: list[str] = []
    _fake_box(ch, calls)
    ch._box_active = False
    async with ch.suspended_box():
        pass
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["close", "stop_before", "stop_inside"])
async def test_suspend_close_or_begin_stop_leaves_the_box_stopped_and_frees_the_lock(how):
    from tests.unit.test_picker import _fake_box

    ch = _channel()
    calls: list[str] = []
    _fake_box(ch, calls)
    if how == "stop_before":
        ch.begin_stop()
    async with ch.suspended_box() as (_output, _console, close):
        if how == "close":
            close()
        if how == "stop_inside":
            ch.begin_stop()
    assert calls == ["stop"]
    assert ch._suspended is False

    async def _again():
        async with ch.suspended_box():
            return True

    assert await asyncio.wait_for(_again(), timeout=1.0) is True


@pytest.mark.asyncio
async def test_suspend_runs_concurrent_entrants_one_after_the_other():
    from tests.unit.test_picker import _fake_box

    ch = _channel()
    calls: list[str] = []
    _fake_box(ch, calls)

    async def _span(n: int):
        async with ch.suspended_box():
            calls.append(f"enter-{n}")
            await asyncio.sleep(0.02)
            calls.append(f"exit-{n}")

    first = asyncio.create_task(_span(1))
    await asyncio.sleep(0)
    await asyncio.gather(first, _span(2))
    assert calls == [
        "stop", "enter-1", "exit-1", "start('queue', 'interrupt')",
        "stop", "enter-2", "exit-2", "start('queue', 'interrupt')",
    ]


@pytest.mark.asyncio
async def test_suspend_holds_the_stdout_patch_for_the_whole_span():
    from prompt_toolkit.patch_stdout import StdoutProxy

    ch = _channel()
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        before = sys.stdout
        async with ch.suspended_box() as (output, console, _close):
            assert isinstance(sys.stdout, StdoutProxy)
            await asyncio.sleep(0.02)
            assert isinstance(sys.stdout, StdoutProxy)
            assert output is None and console is ch._console
        assert sys.stdout is before


@pytest.mark.asyncio
async def test_suspend_on_a_redirected_stdout_yields_the_tty_output_and_leaves_stdout_alone(monkeypatch):
    from rich.console import Console

    ch = _channel()
    ch._console = Console(file=StringIO(), force_terminal=False, width=120, theme=TERMINAL_THEME)
    sentinel = DummyOutput()
    monkeypatch.setattr("prompt_toolkit.output.defaults.create_output", lambda **kw: sentinel)
    before = sys.stdout
    async with ch.suspended_box() as (output, console, _close):
        assert output is sentinel and console is ch._err_console
        assert sys.stdout is before
    assert sys.stdout is before


@pytest.mark.asyncio
async def test_suspend_starts_no_rich_status():
    from localharness.channels.terminal import _Burst
    from localharness.core.events import ConsolidationStarted

    ch = _channel()
    burst = _Burst(frozenset({"read"}), None)
    async with ch.suspended_box():
        assert ch._suspended is True
        ch._start_thinking()
        ch._start_dreaming()
        await ch.on_consolidation_started(ConsolidationStarted())
        ch._burst_refresh(burst)
        assert ch._thinking is None and ch._dreaming is None and burst.status is None
    assert ch._suspended is False
    ch._start_dreaming()
    assert ch._dreaming is not None
    ch._stop_dreaming()


@pytest.mark.asyncio
async def test_suspend_touches_no_sigint_handler(monkeypatch):
    from tests.unit.test_picker import _fake_box

    ch = _channel()
    _fake_box(ch, [])
    loop = asyncio.get_running_loop()
    seen: list[tuple[str, int]] = []
    monkeypatch.setattr(loop, "add_signal_handler", lambda sig, *a: seen.append(("add", sig)))
    monkeypatch.setattr(loop, "remove_signal_handler", lambda sig: seen.append(("remove", sig)))
    before = signal.getsignal(signal.SIGINT)
    async with ch.suspended_box():
        pass
    assert [s for s in seen if s[1] == signal.SIGINT] == []
    assert signal.getsignal(signal.SIGINT) is before


@pytest.mark.asyncio
async def test_suspend_keeps_a_typed_secret_out_of_the_box():
    from localharness.cli.interview import prompt_line

    ch = _channel()
    ch._history = InMemoryHistory()
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        await ch.start_input_box(asyncio.Queue(), on_interrupt=lambda: None)
        try:
            async with ch.suspended_box() as (_output, console, _close):
                inp.send_text("s3cret\r")
                typed = await asyncio.wait_for(
                    prompt_line("token", console=console, secret=True), timeout=10.0
                )
            assert typed == "s3cret"
            assert ch._box_active is True
            assert "s3cret" not in ch._history.get_strings()
            assert "s3cret" not in ch._box_app.current_buffer.text
        finally:
            await ch.stop_input_box()


# ------------------------------------------------- queued asks are counted

@pytest.mark.asyncio
async def test_waiting_counts_an_ask_queued_behind_an_open_span():
    ch = _channel()
    assert ch.waiting_asks() == 0
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        async with ch.suspended_box():
            task = asyncio.create_task(ch.ask_permission(_request()))
            await asyncio.sleep(0.05)
            assert ch.waiting_asks() == 1
        for _ in range(100):
            await asyncio.sleep(0.01)
            if ch._suspended:
                break
        assert ch._suspended
        assert ch.waiting_asks() == 0
        inp.send_text("y")
        assert (await asyncio.wait_for(task, timeout=10.0)).kind == "allow_once"
    assert ch.waiting_asks() == 0


@pytest.mark.asyncio
async def test_waiting_drops_an_ask_cancelled_before_it_drew():
    ch = _channel()
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        async with ch.suspended_box():
            task = asyncio.create_task(ch.ask_permission(_request()))
            await asyncio.sleep(0.05)
            assert ch.waiting_asks() == 1
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert ch.waiting_asks() == 0
    assert ch.waiting_asks() == 0


def test_base_channel_counts_no_asks_and_its_begin_stop_is_a_no_op():
    from localharness.channels.mobile.channel import MobileChannel

    assert ChannelAdapter.waiting_asks(object()) == 0
    assert ChannelAdapter.begin_stop(object()) is None
    assert MobileChannel.waiting_asks is ChannelAdapter.waiting_asks
    assert MobileChannel.begin_stop is ChannelAdapter.begin_stop
    assert MobileChannel.run_interview is ChannelAdapter.run_interview


@pytest.mark.asyncio
async def test_base_channel_run_interview_reports_nobody_was_reached():
    from localharness.cli.interview import Answer, Option, Unreachable

    answer = await ChannelAdapter.run_interview(
        object(), "q?", [Option("a", "A")], default=0, cancel="a"
    )
    assert answer == Answer(("a",), True, reached=False, why=Unreachable.NO_TTY)


@pytest.mark.asyncio
async def test_the_legend_runs_its_application_with_handle_sigint_false(monkeypatch):
    seen: dict = {}

    class _App:
        async def run_async(self, **kw):
            seen.update(kw)
            return "allow_once"

    monkeypatch.setattr("localharness.channels.terminal._build_permission_app", lambda *a, **k: _App())
    assert (await _channel().ask_permission(_request())).kind == "allow_once"
    assert seen == {"handle_sigint": False}


@pytest.mark.usefixtures("default_sigint")
def test_an_external_sigint_mid_legend_reaches_the_turns_handler():
    async def scenario():
        ch = _channel()
        before = signal.getsignal(signal.SIGINT)
        with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
            turn = asyncio.ensure_future(ch.ask_permission(_request()))
            async with sigint_cancels(turn) as cancelled:
                for _ in range(400):
                    if ch._suspended:
                        break
                    await asyncio.sleep(0.005)
                os.kill(os.getpid(), signal.SIGINT)
                try:
                    await asyncio.wait_for(turn, timeout=5.0)
                except asyncio.CancelledError:
                    if not cancelled():
                        raise
                stopped = cancelled()
        return turn.cancelled(), stopped, signal.getsignal(signal.SIGINT) is before

    assert asyncio.run(scenario()) == (True, True, True)
