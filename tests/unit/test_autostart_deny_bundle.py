"""The autostart deny bundle: what an agent's shell may not do to what runs at the next login.

Spec 20 section 12 ("Protected paths and deny entries", owner-confirmed 2026-10-13): the user
manager's environment, link, edit, set-property, add-wants, add-requires and enable-with-a-path
verbs, their D-Bus twins, and ``localharness model --autostart`` are refused for the agent's
``bash_exec``. The plain restart / start / stop / status of the harness's own unit stay allowed.

Every case is a PROBE STRING, evaluated in-process by ``PermissionEvaluator().evaluate`` over the
shipped ``PermissionConfig`` — nothing here builds or runs a command. The evaluator fnmatches the
raw command text (no shlex, no whitespace collapsing), which is why the word verbs are delimited
by a negated character set and the tab, newline, continuation and quote forms are pinned.

Known residual, pinned as allowed so it is visible: a verb the shell assembles at run time (quotes
inside the word) cannot be matched by any pattern over the text (spec 20 section 12 residuals).
"""
from __future__ import annotations

import fnmatch
import re

import pytest

from localharness.agent.permissions import PermissionEvaluator
from localharness.config import defaults
from localharness.config.models import PermissionConfig
from localharness.core.types import ToolCall

SC = "systemctl"
U = f"{SC} --user"
UNIT = "localharness-llm.service"

NEW_PATTERNS = [
    "bash_exec(*systemctl*set-environment*)",
    "bash_exec(*systemctl*import-environment*)",
    "bash_exec(*systemctl*unset-environment*)",
    "bash_exec(*systemctl*[!A-Za-z0-9_.@-]link[!A-Za-z0-9_.@-]*)",
    "bash_exec(*systemctl*[!A-Za-z0-9_.@-]edit[!A-Za-z0-9_.@-]*)",
    "bash_exec(*systemctl*[!A-Za-z0-9_.@-]set-property[!A-Za-z0-9_.@-]*)",
    "bash_exec(*systemctl*[!A-Za-z0-9_.@-]add-wants[!A-Za-z0-9_.@-]*)",
    "bash_exec(*systemctl*[!A-Za-z0-9_.@-]add-requires[!A-Za-z0-9_.@-]*)",
    "bash_exec(*systemctl*[!A-Za-z0-9_.@-]enable[!A-Za-z0-9_.@-]*/*)",
    "bash_exec(*busctl*org.freedesktop.systemd1*)",
    "bash_exec(*gdbus*org.freedesktop.systemd1*)",
    "bash_exec(*dbus-send*org.freedesktop.systemd1*)",
    "bash_exec(*localharness*model*--autostart*)",
]


def _denied(command: str) -> bool:
    return PermissionEvaluator().evaluate(
        ToolCall(name="bash_exec", arguments={"command": command}), PermissionConfig()
    ).denied


def _p(command: str, ident: str):
    return pytest.param(command, id=ident)


# --------------------------------------------------------------------------- the shipped list

def test_the_shipped_list_gains_exactly_the_thirteen_patterns():
    shipped = PermissionConfig().deny_patterns
    assert len(shipped) == 45
    assert shipped[-13:] == NEW_PATTERNS


@pytest.mark.parametrize("pattern", NEW_PATTERNS)
def test_every_new_pattern_passes_the_format_check_and_the_evaluator_parse(pattern):
    assert PermissionConfig(deny_patterns=[pattern]).deny_patterns == [pattern]
    assert re.match(r"^([a-z_][a-z0-9_]*)(?:\((.+)\))?$", pattern)
    assert pattern.isascii() and pattern.isprintable()


def test_no_word_verb_pattern_is_delimited_by_a_literal_space():
    for pattern in NEW_PATTERNS:
        for verb in ("link", "edit", "set-property", "add-wants", "add-requires", "enable"):
            assert f" {verb} " not in pattern


def test_the_revision_is_four():
    assert defaults.CURRENT_DEFAULTS_REVISION == 4


# --------------------------------------------------------------------------- denied

DENIED = [
    _p(f"{U} set-environment LD_PRELOAD=/x", "set_environment_user_before"),
    _p(f"{SC} set-environment --user LD_PRELOAD=/x", "set_environment_user_after"),
    _p(f"cd /tmp && {U} set-environment A=b", "set_environment_compound"),
    _p(f"{U} import-environment PATH", "import_environment"),
    _p(f"{SC} import-environment --user PATH", "import_environment_user_after"),
    _p(f"{U} unset-environment PATH", "unset_environment"),
    _p(f"{SC} unset-environment --user PATH", "unset_environment_user_after"),
    _p(f"{U} link /home/u/x.service", "link"),
    _p(f"{SC} link --user /home/u/x.service", "link_user_after"),
    _p(f"cd /tmp && {U} link /home/u/x.service", "link_compound"),
    _p(f"{U} edit {UNIT}", "edit"),
    _p(f"{U} edit --full llm.service", "edit_full"),
    _p(f"{SC} edit --user llm.service", "edit_user_after"),
    _p(f"{U} set-property llm.service MemoryMax=1G", "set_property"),
    _p(f"{SC} set-property --user llm.service MemoryMax=1G", "set_property_user_after"),
    _p(f"{U} add-wants default.target x.service", "add_wants"),
    _p(f"{U} add-requires default.target x.service", "add_requires"),
    _p(f"{U} enable /home/u/evil.service", "enable_path"),
    _p(f"{U} enable --now ./evil.service", "enable_now_relative_path"),
    _p(f"{SC} enable --user /home/u/evil.service", "enable_path_user_after"),
    _p("busctl --user call org.freedesktop.systemd1 /org/freedesktop/systemd1 X Y", "busctl_user_first"),
    _p("busctl call --user org.freedesktop.systemd1 /org/freedesktop/systemd1 X Y", "busctl_user_after"),
    _p("gdbus call --session --dest org.freedesktop.systemd1 --object-path /x --method y", "gdbus_session"),
    _p("gdbus call -e -d org.freedesktop.systemd1 -o /x -m y", "gdbus_short_flags"),
    _p("dbus-send --session --dest=org.freedesktop.systemd1 /x a.b.C", "dbus_send"),
    _p("localharness model --autostart on", "autostart_on"),
    _p("localharness model --autostart off", "autostart_off"),
    _p("python -m localharness model --autostart status", "autostart_module_entry"),
    _p("localharness model --autostart=on", "autostart_equals"),
]

# Tab, newline, backslash-newline continuation and quote forms: the evaluator matches the raw text.
DENIED_WHITESPACE = [
    _p(f"{U}\tlink /home/u/x.service", "tab_before_link"),
    _p(f"{U}\tedit {UNIT}", "tab_before_edit"),
    _p(f"{U}\tset-property llm.service MemoryMax=1G", "tab_before_set_property"),
    _p(f"{U}\tadd-wants default.target x.service", "tab_before_add_wants"),
    _p(f"{U}\tadd-requires default.target x.service", "tab_before_add_requires"),
    _p(f"{U}\tenable /home/u/evil.service", "tab_before_enable"),
    _p(f"{U} link\t/home/u/x.service", "tab_after_link"),
    _p(f"{U} enable\t/home/u/evil.service", "tab_after_enable"),
    _p(f"{U}\nedit x.service", "newline_before_edit"),
    _p(f"{U}\nlink /x.service", "newline_before_link"),
    _p(f"{U} \\\nlink /x.service", "continuation_before_link"),
    _p(f"{U} \\\nedit x.service", "continuation_before_edit"),
    _p(f"{U} 'link' /x.service", "single_quoted_link"),
    _p(f'{U} "edit" x.service', "double_quoted_edit"),
    _p(f"{U} 'set-property' x.service MemoryMax=1G", "single_quoted_set_property"),
    _p(f"{U} enable '/home/u/evil.service'", "quoted_path_enable"),
    _p(f"{SC}\t--user\tset-environment\tLD_PRELOAD=/x", "tab_separated_set_environment"),
    _p(f"{SC}\n--user\nimport-environment\nPATH", "newline_separated_import_environment"),
    _p("localharness\tmodel --autostart on", "tab_after_localharness_autostart"),
    _p("localharness model\t--autostart\ton", "tab_around_autostart_flag"),
    _p("localharness\nmodel\n--autostart on", "newline_separated_autostart"),
    _p(f"true;{U} link /x.service", "semicolon_before_link"),
    _p(f"true&&{U} edit x.service", "ampersand_before_edit"),
    _p(f"true|{U} link /x.service", "pipe_before_link"),
]


@pytest.mark.parametrize("command", DENIED)
def test_the_bundle_form_is_denied(command):
    assert _denied(command)


@pytest.mark.parametrize("command", DENIED_WHITESPACE)
def test_the_bundle_form_is_denied_through_whitespace_quotes_and_continuations(command):
    assert _denied(command)


# --------------------------------------------------------------------------- allowed

ALLOWED = [
    _p(f"{U} restart {UNIT}", "restart"),
    _p(f"{U} start {UNIT}", "start"),
    _p(f"{U} stop {UNIT}", "stop"),
    _p(f"{U} status {UNIT}", "status"),
    _p(f"{U} show-environment", "show_environment"),
    _p(f"{U} enable llm.service", "enable_bare_name"),
    _p(f"{U} is-enabled llm.service", "is_enabled"),
    _p(f"{U} status credit.service", "unit_name_contains_edit"),
    _p(f"{U} status my.link.service", "unit_name_contains_link"),
    _p(f"{U} status x-edit.service", "unit_name_ends_in_edit"),
    _p("localharness model flash-next", "model_flash_next"),
    _p("localharness model", "model_bare"),
    _p("", "empty_command"),
    _p("grep -rn edit src/", "grep_mentions_edit"),
    _p("git log --oneline -- linker.py", "git_log_mentions_link_in_a_name"),
    _p("busctl --user list", "busctl_without_the_systemd_name"),
    # the tab forms of the plain verbs stay allowed too
    _p(f"{SC}\t--user\trestart\t{UNIT}", "tab_separated_restart"),
    _p(f"{U}\tstatus {UNIT}", "tab_before_status"),
    _p(f"{U}\tstop {UNIT}", "tab_before_stop"),
    _p(f"{U} start\t{UNIT}", "tab_after_start"),
    _p(f"{U}\nstatus {UNIT}", "newline_before_status"),
]


@pytest.mark.parametrize("command", ALLOWED)
def test_the_plain_form_is_not_denied(command):
    assert not _denied(command)


def test_residual_quote_inside_word():
    """Pinned as allowed BY DESIGN: a verb with quotes inside the word (``l'in'k``) is assembled by
    the shell at run time and no pattern over the command text can match it (spec 20 section 12's
    residuals). The unit relies on its byte-equal body and the identity test, not on a closed
    shell; this test makes the residual visible instead of a later discovery."""
    assert not _denied(f"{U} l'in'k /x.service")


def test_a_bare_word_in_another_tool_is_not_the_bash_pattern():
    result = PermissionEvaluator().evaluate(
        ToolCall(name="read", arguments={"path": f"/tmp/{SC}-set-environment-notes"}),
        PermissionConfig(),
    )
    assert not result.denied


def test_the_negated_set_delimiter_is_the_unit_name_alphabet():
    pattern = "*systemctl*[!A-Za-z0-9_.@-]edit[!A-Za-z0-9_.@-]*"
    for ch in " \t\n'\";&|":
        assert fnmatch.fnmatchcase(f"systemctl --user{ch}edit{ch}x", pattern)
    for ch in "abcZ09_.@-":
        assert not fnmatch.fnmatchcase(f"systemctl --user {ch}edit{ch} x", pattern)
