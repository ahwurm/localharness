"""The terminal's rendering of an ASK: `[y]es once / [a]lways here / [n]o / [N]ever here`.

PRD §3.5. Driven through the same headless prompt_toolkit harness the input-box suite uses
(create_pipe_input + DummyOutput + create_app_session), so these are real keystrokes through the
real key bindings, not a stubbed prompt.
"""
from __future__ import annotations

import asyncio
from io import StringIO

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from localharness.agent.gate_types import PermissionRequest
from localharness.channels.base import ChannelAdapter
from localharness.channels.terminal import (
    PERMISSION_OPTIONS_GRANTABLE,
    PERMISSION_OPTIONS_UNGRANTABLE,
    TERMINAL_THEME,
    TerminalChannel,
    _build_permission_app,
)
from localharness.core.bus import EventBus


def _channel() -> TerminalChannel:
    from rich.console import Console

    ch = TerminalChannel(EventBus(), {})
    ch._console = Console(
        file=StringIO(), force_terminal=True, width=120, theme=TERMINAL_THEME, highlight=False
    )
    return ch


def _request(grantable: bool = True) -> PermissionRequest:
    return PermissionRequest(
        tool_name="bash_exec",
        tool_params={"command": "cargo publish"},
        klass="shell-unfamiliar" if grantable else "shell-destructive",
        key="cargo publish" if grantable else None,
        grantable=grantable,
        reason="not seen in this workspace before",
        display="bash_exec: cargo publish  (shell-unfamiliar — not seen before)",
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


def test_the_legends_name_every_option_the_keys_bind():
    assert "[y]es once" in PERMISSION_OPTIONS_GRANTABLE
    assert "[a]lways here" in PERMISSION_OPTIONS_GRANTABLE
    assert "[N]ever here" in PERMISSION_OPTIONS_GRANTABLE
    assert "[a]lways" not in PERMISSION_OPTIONS_UNGRANTABLE


@pytest.mark.asyncio
async def test_the_app_exits_on_one_keystroke():
    """Guard on the shape: no Buffer, so a stray Enter cannot answer with leftover text."""
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        app = _build_permission_app(PERMISSION_OPTIONS_GRANTABLE, True)
        inp.send_text("a")
        assert await asyncio.wait_for(app.run_async(), timeout=10.0) == "allow_always"


# ------------------------------------------------------------------ the contract

def test_the_terminal_declares_it_can_ask_and_shows_diffs():
    ch = _channel()
    assert ch.can_ask is True
    assert ch.has_review_surface is True


def test_a_non_tty_terminal_reports_that_it_cannot_ask():
    """PRD §3.5's non-tty row: piped and --no-input runs route to the fail-closed path."""
    from rich.console import Console

    ch = TerminalChannel(EventBus(), {})
    ch._console = Console(file=StringIO(), force_terminal=False, width=120)
    assert ch.can_ask is False


def test_the_base_channel_cannot_ask_and_says_so_loudly():
    assert ChannelAdapter.can_ask is False
    assert ChannelAdapter.has_review_surface is False


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
