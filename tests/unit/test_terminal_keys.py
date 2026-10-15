"""`terminal.keys` — the person's own chords for the nudge and the line break (spec 21 §5, §10).

This half pins the config and the loader: the closed vocabulary, the string-or-list reading, the
precedence of a list the person set over one left at its default, the one refusal for two set lists
that share a chord, and the machine-only drop. Everything here is hermetic: pydantic and the
loader's layers on `tmp_path`, no tty, no network.
"""
from __future__ import annotations

from localharness.config.loader import HARNESS_GLOBAL_ONLY_FIELDS, is_harness_global_only
from localharness.config.models import HarnessConfig, TerminalKeysConfig
from tests.unit.test_docs_convergence import _read, check_machine_only_list


def test_defaults():
    keys = TerminalKeysConfig()
    assert keys.nudge == ["alt-enter"]
    assert keys.newline == ["shift-enter"]
    assert HarnessConfig.model_fields["terminal"].default_factory is not None
    terminal = HarnessConfig.model_fields["terminal"].default_factory()
    assert terminal.keys == TerminalKeysConfig()


def test_a_set_newline_list_takes_its_chord_from_the_defaulted_nudge():
    keys = TerminalKeysConfig(newline=["alt-enter"])
    assert keys.nudge == []
    assert keys.newline == ["alt-enter"]


def test_terminal_keys_is_one_machine_only_member():
    assert "terminal.keys" in HARNESS_GLOBAL_ONLY_FIELDS
    assert "terminal.keys.nudge" not in HARNESS_GLOBAL_ONLY_FIELDS
    assert "terminal.keys.newline" not in HARNESS_GLOBAL_ONLY_FIELDS
    assert is_harness_global_only("terminal.keys.nudge")
    assert is_harness_global_only("terminal.keys.newline")


def test_security_md_lists_terminal_keys():
    assert check_machine_only_list(_read("SECURITY.md")) == []
