"""The reader predicate behind the instructions naming rule (Phase 11, D-35, D-73).

``instruction_file_reads_only`` decides, for a ``bash_exec`` command that mentions one of the
instruction-file basenames, whether the command can only READ it. One classifier call; the
whole command is the unit; every unclear case fails toward marking (False).
"""

from __future__ import annotations

from localharness.agent.gate_types import GateSettings
from localharness.agent.shell_classify import instruction_file_reads_only

NAMES = ("CLAUDE.md", "AGENTS.md", "INSTRUCTIONS.md")


def reads_only(command: str) -> bool:
    return instruction_file_reads_only(command, NAMES)


def test_a_plain_read_is_a_reader() -> None:
    assert reads_only("cat CLAUDE.md") is True


def test_a_redirect_into_the_file_is_not() -> None:
    assert reads_only("cat > CLAUDE.md") is False


def test_a_command_that_names_nothing_is_a_reader() -> None:
    assert reads_only("ls -la") is True


def test_naming_is_case_insensitive() -> None:
    assert reads_only("cat claude.MD") is True
    assert reads_only("cat > Claude.md") is False


def test_settings_none_and_explicit_settings_agree() -> None:
    assert instruction_file_reads_only("cat CLAUDE.md", NAMES, GateSettings()) is True
    assert instruction_file_reads_only("cat > CLAUDE.md", NAMES, GateSettings()) is False
