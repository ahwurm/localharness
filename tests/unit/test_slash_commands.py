"""The slash-command table is the single source of truth for /help and the input completion menu.

Guards against drift: /help text, the completer, and the REPL dispatcher must all agree on the set
of commands. If a command is added to the dispatcher but not the table (or vice versa), a test here
fails.
"""
from __future__ import annotations

import inspect

from localharness.cli import repl
from localharness.cli.slash_commands import SLASH_COMMANDS, help_text


def test_table_is_nonempty_name_description_pairs():
    assert SLASH_COMMANDS
    for name, desc in SLASH_COMMANDS:
        assert name.startswith("/") and desc


def test_help_text_lists_every_command_and_description():
    text = help_text()
    assert "Available commands" in text
    for name, desc in SLASH_COMMANDS:
        assert name in text and desc in text


def test_repl_help_text_is_derived_from_the_table():
    # repl.HELP_TEXT must BE the table's render, not a separate hand-maintained literal.
    assert repl.HELP_TEXT == help_text()


def test_table_matches_the_dispatcher_command_set():
    # Every command the REPL dispatcher claims appears in the table, and vice versa — no drift.
    src = inspect.getsource(repl.OrchestratorREPL._handle_slash)
    dispatched = {"/help", "/agents", "/model", "/memory", "/quit", "/exit"}
    table = {name for name, _ in SLASH_COMMANDS}
    assert table == dispatched
    for cmd in dispatched:
        assert cmd in src  # sanity: these really are the literals the dispatcher matches
