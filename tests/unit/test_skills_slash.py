"""Slash rows for skills: the dispatch seam, late rows and, from a later phase, the /<name> turn and the session."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from localharness.cli.slash_commands import SLASH_COMMANDS, SlashCommand, all_rows, set_plugin_rows
from tests.unit.test_repl_unknown_slash import RecordingChannel, _build_repl


@pytest.fixture
def rows():
    assert all_rows() == SLASH_COMMANDS, "a plugin row leaked in from an earlier test"
    yield set_plugin_rows
    set_plugin_rows(())


async def test_a_plugin_table_row_with_a_method_name_is_called_with_its_row_and_args(rows):
    row = SlashCommand("/example", "an example skill", "_slash_example", takes_args=True, plugin="skill")
    assert rows([row]) == []
    repl, agent, _ = _build_repl(RecordingChannel([]))
    repl._slash_example = AsyncMock()
    assert await repl._handle_slash("/EXAMPLE  Hi There ") is True
    repl._slash_example.assert_awaited_once_with(row, "Hi There")
    repl._slash_example.reset_mock()
    assert await repl._handle_slash("/example") is True
    repl._slash_example.assert_awaited_once_with(row, "")
    agent.run_turn.assert_not_called()
