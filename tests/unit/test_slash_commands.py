"""The slash-command table is the single source of truth for the REPL dispatcher, /help and the
input completion menu (tests/unit/test_slash_table.py covers the phone's menu and plugin rows).

Guards against drift: /help text, the completer, and the REPL dispatcher must all agree on the set
of commands. The dispatcher reads the table, so a command cannot exist in one and not the other.
"""
from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock

from localharness.channels.keys import help_block
from localharness.channels.terminal import TerminalChannel
from localharness.cli import repl
from localharness.cli.slash_commands import SLASH_COMMANDS, all_rows, find_row, help_text
from localharness.config.models import TerminalKeysConfig
from localharness.core.bus import EventBus


def test_table_is_nonempty_name_description_pairs():
    assert SLASH_COMMANDS
    for name, desc in SLASH_COMMANDS:
        assert name.startswith("/") and desc


def test_help_text_lists_every_command_and_description():
    text = help_text()
    assert "Available commands" in text
    for name, desc in SLASH_COMMANDS:
        assert name in text and desc in text


async def test_slash_help_renders_the_live_table():
    # /help renders the table when it is typed, never a string frozen at import: plugin rows are
    # added after the REPL module is imported, so a module-level render would never show them.
    sent = []

    class _Channel:
        async def send_message(self, text, agent_id=None, metadata=None):
            sent.append((text, metadata))

    r = repl.OrchestratorREPL(orchestrator=MagicMock(), agent_loop=MagicMock(), channel=_Channel(),
                              bus=AsyncMock())
    assert await r._handle_slash("/help") is True
    assert sent == [(help_text(), {"style": "system.info"})]


def test_help_text_without_a_block_is_unchanged():
    # The pre-change rendering, rebuilt step by step: with no block /help reads exactly as before,
    # so the phone, Discord, Zed and the classic prompt are untouched (spec 21 section 6.7).
    rows = all_rows()
    width = max(len(name) for name, _ in rows)
    lines = ["Available commands:"]
    for name, desc in rows:
        lines.append(f"  {name.ljust(width)}  {desc}")
    lines.append("")
    lines.append("Everything else is handled by the orchestrator through natural language.")
    before = "\n".join(lines)
    assert help_text() == before
    assert help_text(keys_block=None) == before
    assert help_text(keys_block="") == before


def test_help_text_appends_the_keys_block():
    block = "Keys:\n  \\ then enter       line break\n  !text              send it"
    assert help_text(keys_block=block) == help_text() + "\n\n" + block


_KITTY_SETUP = r"kitty.conf: map shift+enter send_text all \x1b[13;2u"
_KITTY_ROW_1 = (
    r"  \ then enter       line break. shift+enter submits until kitty is set up — "
    + _KITTY_SETUP
)
_KITTY_ROW_2 = "  !text, alt+enter   send the line into the running turn (a nudge)"


def _help_repl_and_sent(monkeypatch, *, box_active):
    for name in ("TMUX", "TERM_PROGRAM", "ALACRITTY_WINDOW_ID", "WT_SESSION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KITTY_WINDOW_ID", "1")
    channel = TerminalChannel(EventBus(), {}, keys=TerminalKeysConfig())
    channel._box_active = box_active
    sent = []

    async def _send(text, agent_id=None, metadata=None):
        sent.append((text, metadata))

    channel.send_message = _send
    r = repl.OrchestratorREPL(orchestrator=MagicMock(), agent_loop=MagicMock(), channel=channel,
                              bus=AsyncMock())
    return r, sent


async def test_slash_help_on_a_live_box_ends_with_the_keys_block(monkeypatch):
    r, sent = _help_repl_and_sent(monkeypatch, box_active=True)
    assert await r._handle_slash("/help") is True
    block = help_block(TerminalKeysConfig(), terminal=("kitty", _KITTY_SETUP), shift_enter_seen=False)
    assert sent == [(help_text(keys_block=block), {"style": "system.info"})]
    assert sent[0][0].endswith(_KITTY_ROW_1 + "\n" + _KITTY_ROW_2)


async def test_slash_help_passes_no_block_when_the_box_is_off(monkeypatch):
    r, sent = _help_repl_and_sent(monkeypatch, box_active=False)
    assert await r._handle_slash("/help") is True
    assert sent == [(help_text(), {"style": "system.info"})]
    assert "Keys:" not in sent[0][0]


def test_table_matches_the_dispatcher_command_set():
    # The dispatcher looks every command up in the table: each row resolves to itself and each
    # core row names a real coroutine on the REPL, so the table and the dispatcher cannot drift.
    # /memory is the memory plugin's row, not core's (it reaches the table only via set_plugin_rows).
    dispatched = {"/help", "/agents", "/model", "/reasoning", "/effort", "/verbose", "/mode",
                  "/pending", "/approve", "/deny", "/task", "/instructions", "/image", "/plugins", "/mcp",
                  "/quit", "/exit"}
    table = {name for name, _ in SLASH_COMMANDS}
    assert table == dispatched and "/memory" not in table
    for row in SLASH_COMMANDS:
        assert find_row(row.name) is row
        assert inspect.iscoroutinefunction(getattr(repl.OrchestratorREPL, row.handler)), row.name
