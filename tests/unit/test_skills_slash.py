"""Slash rows for skills: the dispatch seam and late rows."""
from __future__ import annotations

import ast
import inspect
import textwrap
from unittest.mock import AsyncMock

import pytest

from localharness.cli.repl import OrchestratorREPL
from localharness.cli.slash_commands import (
    SLASH_COMMANDS,
    SlashCommand,
    add_plugin_row,
    all_rows,
    set_plugin_rows,
)
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


def _skill(name: str) -> SlashCommand:
    return SlashCommand(f"/{name}", "x", "_slash_skill", takes_args=True, plugin="skill")


def _plugin(name: str, plugin: str = "example") -> SlashCommand:
    return SlashCommand(f"/{name}", "x", AsyncMock(), plugin=plugin)


def test_warnings_are_worded_by_the_rows_kind(rows):
    assert rows([_skill("help"), _plugin("help", "evil")]) == [
        'skill help: slash command /help is already taken — say "use the help skill", or rename it',
        "plugin evil: slash command /help is already taken — skipped",
    ]
    assert all_rows() == SLASH_COMMANDS


def test_empty_rows_leave_only_the_core_table(rows):
    rows([_plugin("example")])
    assert rows(()) == [] and all_rows() == SLASH_COMMANDS


def test_a_late_row_is_appended_after_every_earlier_plugin_row(rows):
    example, release, third = _plugin("example"), _skill("release-notes"), _skill("deploy")
    rows([example])
    assert add_plugin_row(release) is None
    assert all_rows() == SLASH_COMMANDS + (example, release)
    assert add_plugin_row(third) is None
    assert all_rows() == SLASH_COMMANDS + (example, release, third)


def test_a_late_row_with_no_plugin_rows_installed_is_the_only_one(rows):
    row = _skill("release-notes")
    assert add_plugin_row(row) is None
    assert all_rows() == SLASH_COMMANDS + (row,)


@pytest.mark.parametrize(("late", "warning"), [
    (_skill("example"), 'skill example: slash command /example is already taken — say "use the example skill", or rename it'),
    (_skill("help"), 'skill help: slash command /help is already taken — say "use the help skill", or rename it'),
    (_plugin("model", "twin"), "plugin twin: slash command /model is already taken — skipped"),
    (SlashCommand("/odd", "x", None, plugin="skill"), "skill odd: slash command /odd has no callable handler — skipped"),
    (SlashCommand("/odd", "x", None, plugin="twin"), "plugin twin: slash command /odd has no callable handler — skipped"),
])
def test_a_refused_late_row_returns_its_warning_and_changes_nothing(rows, late, warning):
    rows([_plugin("example")])
    before = all_rows()
    assert add_plugin_row(late) == warning
    assert all_rows() == before
    assert not warning.startswith("plugin ") or late.plugin != "skill"


def test_the_taken_name_check_is_exact(rows):
    rows([_plugin("example")])
    assert add_plugin_row(_skill("example-two")) is None
    assert [row.name for row in all_rows()][-2:] == ["/example", "/example-two"]


@pytest.mark.parametrize("handler", ["_slash_skill", AsyncMock()])
def test_a_row_naming_no_plugin_is_skipped_in_both_entry_points(rows, handler):
    row = SlashCommand("/orphan", "x", handler)
    warning = "slash command /orphan names no plugin — skipped"
    assert rows([row]) == [warning]
    assert add_plugin_row(row) == warning
    assert all_rows() == SLASH_COMMANDS


async def test_a_callable_on_a_late_row_gets_the_stripped_text_only(rows):
    handler = AsyncMock(return_value=None)
    assert add_plugin_row(SlashCommand("/late", "x", handler, takes_args=True, plugin="late")) is None
    repl, _, _ = _build_repl(RecordingChannel([]))
    assert await repl._handle_slash("/late  Hi There ") is True
    handler.assert_awaited_once_with("Hi There")


def test_the_dispatcher_does_not_key_on_the_kind():
    tree = ast.parse(textwrap.dedent(inspect.getsource(OrchestratorREPL._handle_slash)))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == "skill"]
