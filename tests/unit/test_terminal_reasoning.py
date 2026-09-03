"""Reasoning stream on the terminal channel (terminal.show_reasoning / --show-reasoning /
/reasoning). Live motivation: a 3-minute think with the output budget spent on hidden
reasoning looked like dead air — nothing on screen until an empty reply."""
from io import StringIO

import pytest


def _channel(tmp_path):
    from localharness.channels.terminal import TERMINAL_THEME, TerminalChannel
    from localharness.core.bus import EventBus
    from rich.console import Console

    ch = TerminalChannel(EventBus(), {}, history_file=str(tmp_path / ".hist"))
    ch._console = Console(file=StringIO(), force_terminal=False, width=120, theme=TERMINAL_THEME)
    return ch


def _out(ch) -> str:
    return ch._console.file.getvalue()


@pytest.mark.asyncio
async def test_reasoning_prints_complete_lines_and_flushes_tail(tmp_path):
    from localharness.core.events import Action

    ch = _channel(tmp_path)
    ch.show_reasoning = True
    await ch.on_reasoning("Let me check the ")
    await ch.on_reasoning("skill first.\nThen the")
    assert "⋯ Let me check the skill first." in _out(ch)
    assert "Then the" not in _out(ch)                       # partial line waits
    await ch.on_reasoning(" vault.")
    # The response's Action ends the generation: the tail prints.
    await ch.on_action(Action(agent_id="a", session_id="s", action_type="llm_response",
                              content="", has_tool_calls=True))
    assert "⋯ Then the vault." in _out(ch)


@pytest.mark.asyncio
async def test_reasoning_long_paragraph_streams_in_pieces(tmp_path):
    ch = _channel(tmp_path)
    ch.show_reasoning = True
    await ch.on_reasoning("x" * 300)                         # no newline yet
    assert "⋯ " + "x" * 300 in _out(ch)


@pytest.mark.asyncio
async def test_reasoning_silent_when_off(tmp_path):
    ch = _channel(tmp_path)
    await ch.on_reasoning("secret plan\n")
    await ch.flush_reasoning()
    assert _out(ch) == ""


def test_reasoning_slash_command_is_listed():
    from localharness.cli.slash_commands import SLASH_COMMANDS
    assert any(name == "/reasoning" for name, _ in SLASH_COMMANDS)
