"""The slash-command table is the single source of truth for the REPL dispatcher, /help and the
input completion menu (tests/unit/test_slash_table.py covers the phone's menu and plugin rows).

Guards against drift: /help text, the completer, and the REPL dispatcher must all agree on the set
of commands. The dispatcher reads the table, so a command cannot exist in one and not the other.
"""
from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock

from localharness.cli import repl
from localharness.cli.slash_commands import SLASH_COMMANDS, find_row, help_text


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


def test_table_matches_the_dispatcher_command_set():
    # The dispatcher looks every command up in the table: each row resolves to itself and each
    # core row names a real coroutine on the REPL, so the table and the dispatcher cannot drift.
    # /memory is the memory plugin's row, not core's (it reaches the table only via set_plugin_rows).
    dispatched = {"/help", "/agents", "/model", "/reasoning", "/verbose", "/mode",
                  "/pending", "/approve", "/deny", "/task", "/plugins", "/quit", "/exit"}
    table = {name for name, _ in SLASH_COMMANDS}
    assert table == dispatched and "/memory" not in table
    for row in SLASH_COMMANDS:
        assert find_row(row.name) is row
        assert inspect.iscoroutinefunction(getattr(repl.OrchestratorREPL, row.handler)), row.name
