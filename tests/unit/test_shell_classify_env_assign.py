"""SHELL-03 (Phase 10, plan 05): environment assignments that redirect what runs.

Task 1 pins the tables, the two fixed code constants, the settings mapping and the segment
default; later sections pin the inline matrix, the carried matrix and the verdicts.
"""
from __future__ import annotations

import dataclasses

from localharness.agent.gate_types import (
    COMMAND_VALUED_ENV_VARS,
    ENV_REDIRECT_EXCLUDED,
    EXEC_REDIRECT_ENV_VARS_DEFAULT,
    GateSettings,
    ShellSegment,
)
from localharness.config.models import AskConfig

# --------------------------------------------------------------------------- tables


def test_the_shipped_table_is_the_decided_tuple_in_order() -> None:
    assert EXEC_REDIRECT_ENV_VARS_DEFAULT == (
        "PATH", "BASH_ENV", "ENV", "LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "HOME",
        "*_CONFIG*", "GIT_EXTERNAL_DIFF", "GIT_SSH_COMMAND", "GIT_PAGER", "PAGER", "GIT_EDITOR",
        "GIT_SEQUENCE_EDITOR", "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_*",
        "GIT_DIR", "GIT_COMMON_DIR",
    )


def test_gate_settings_default_is_the_shipped_table() -> None:
    assert GateSettings().exec_redirect_env_vars == EXEC_REDIRECT_ENV_VARS_DEFAULT


def test_the_two_fixed_constants() -> None:
    assert COMMAND_VALUED_ENV_VARS == frozenset({
        "GIT_PAGER", "PAGER", "GIT_EDITOR", "GIT_SEQUENCE_EDITOR", "GIT_EXTERNAL_DIFF",
        "GIT_SSH_COMMAND",
    })
    assert ENV_REDIRECT_EXCLUDED == frozenset({"GIT_CONFIG_NOSYSTEM"})


def test_the_constants_are_code_not_configuration() -> None:
    """D-26: no layer can move a name into the exempt set or out of the excluded one."""
    names = [f.name for f in dataclasses.fields(GateSettings)] + list(AskConfig.model_fields)
    assert not [n for n in names if "command_valued" in n or "excluded" in n]


def test_ask_config_maps_onto_gate_settings() -> None:
    cfg = AskConfig(exec_redirect_env_vars=["X_TOOL"])
    assert cfg.to_gate_settings().exec_redirect_env_vars == ("X_TOOL",)


def test_ask_config_unset_keeps_the_shipped_table() -> None:
    assert AskConfig().to_gate_settings().exec_redirect_env_vars == EXEC_REDIRECT_ENV_VARS_DEFAULT


def test_segment_default_is_false_and_positional_construction_works() -> None:
    seg = ShellSegment("x", ("x",))
    assert seg.env_redirect is False
    assert ShellSegment("x", ("x",), env_redirect=True).env_redirect is True


# --------------------------------------------------------------------------- inline rule

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from localharness.agent.gate import tool_meta_from_schema  # noqa: E402
from localharness.agent.gate_types import Verdict  # noqa: E402
from localharness.agent.shell_classify import (  # noqa: E402
    OPERAND_PLACEHOLDER,
    classify_shell,
)
from localharness.agent.verdict import GateContext, evaluate  # noqa: E402
from localharness.tools.builtin.bash_tool import BashExecTool  # noqa: E402

SETTINGS = GateSettings()
BASH_META = tool_meta_from_schema(BashExecTool().info())


def _segments(command: str, settings: GateSettings = SETTINGS):
    return classify_shell(command, settings).segments


def _no_grants(workspace: Path, klass: str, key: str):
    return None


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """A workspace with its own HOME and global config dir, so nothing touches the real ones."""
    home = (tmp_path / "home").resolve()
    (home / ".localharness").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCALHARNESS_DIR", str(home / ".localharness"))
    workspace = (home / "proj").resolve()
    workspace.mkdir()
    return workspace


def _verdict(command: str, workspace: Path, mode: str):
    ctx = GateContext(
        workspace=workspace, boundary=workspace, grants=_no_grants,
        has_review_surface=True, mode=mode,  # type: ignore[arg-type]
    )
    return evaluate("bash_exec", {"command": command}, BASH_META, ctx, SETTINGS)


LEAVES_THE_TIER = [
    "GIT_EXTERNAL_DIFF=x git diff",
    "env GIT_CONFIG_PARAMETERS=\"'diff.external'='x'\" git diff",
    "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=diff.external GIT_CONFIG_VALUE_0=x git diff",
    "RIPGREP_CONFIG_PATH=f rg x",
    "BASH_ENV=s ls",
    "BASH_ENV=cat ls",
    "PATH=./d ls",
    "PATH=cat ls",
    "PATH= ls",
    "GIT_PAGER='cat; rm x' git log",
    "GIT_EXTERNAL_DIFF='sort -o' git diff",
    "HOME=./h git diff",
    "GIT_DIR=./evil git diff",
    "GIT_COMMON_DIR=./evil git diff",
    "LD_AUDIT=./a.so ls",
    "XDG_CONFIG_HOME=./x git diff",
    "GIT_CONFIG_GLOBAL=./g git log",
    "git --git-dir=./evil diff",
    "git --git-dir ./evil diff",
    "PATH=cat git diff CLAUDE.md",
    "PATH=./d timeout 5 ls",
    "env PATH=./d ls",
    "A=1 PATH=./d ls",
]


@pytest.mark.parametrize("command", LEAVES_THE_TIER)
def test_a_redirecting_assignment_takes_the_command_out_of_the_read_only_tier(command):
    segments = _segments(command)
    assert segments
    for segment in segments:
        assert segment.read_only is False, (command, segment)
        assert segment.env_redirect is True, (command, segment)


KEEPS_THE_TIER = [
    "GIT_PAGER=cat git log",
    "PAGER= git log",
    "GIT_PAGER= git log",
    "GIT_CONFIG_NOSYSTEM=1 git log",
    "GIT_PAGER=cat git diff CLAUDE.md",
    "LANG=C sort f",
    "LC_ALL=C ls",
    "TERM=dumb git log",
    "A=1 ls",
    "env A=1 ls",
    "env PATH=./d",
    "env",
    "git --work-tree=./w diff",
]


@pytest.mark.parametrize("command", KEEPS_THE_TIER)
def test_the_frequent_harmless_idioms_keep_the_tier(command):
    segments = _segments(command)
    assert segments
    for segment in segments:
        assert segment.read_only is True, (command, segment)
        assert segment.env_redirect is False, (command, segment)


def test_payloads_inherit_the_inline_assignment():
    (host_and_payload) = _segments("PATH=./d bash -c 'ls'")
    ls = [s for s in host_and_payload if s.signature == "ls"]
    assert ls and all(s.env_redirect and not s.read_only for s in ls)
    found = [s for s in _segments("PATH=./d find . -exec ls {} +") if s.signature == "ls"]
    assert found and all(s.env_redirect and not s.read_only for s in found)
    xargs = [s for s in _segments("PATH=./d xargs ls") if s.signature == "ls"]
    assert xargs and all(s.env_redirect and not s.read_only for s in xargs)
    uv = _segments("PATH=./d uv run ls")
    assert uv and all(s.env_redirect and not s.read_only for s in uv)


def test_a_harmless_assignment_does_not_taint_payloads():
    for command in ("LANG=C bash -c 'ls'", "LANG=C find . -exec ls {} +"):
        ls = [s for s in _segments(command) if s.signature == "ls"]
        assert ls and all(s.read_only and not s.env_redirect for s in ls), command


def test_env_with_no_wrapped_command_reports_no_assignment():
    (segment,) = _segments("env PATH=./d")
    assert segment.env_redirect is False and segment.read_only is True


def test_the_table_name_is_matched_case_sensitively():
    (segment,) = _segments("path=./d ls")
    assert segment.env_redirect is False and segment.read_only is True
    (segment,) = _segments("my_config=1 ls")
    assert segment.env_redirect is False


def test_the_excluded_name_is_exempt_even_when_the_table_lists_it():
    settings = GateSettings(exec_redirect_env_vars=("GIT_CONFIG_NOSYSTEM", "PATH"))
    (segment,) = _segments("GIT_CONFIG_NOSYSTEM=1 git log", settings)
    assert segment.read_only is True and segment.env_redirect is False
    (segment,) = _segments("PATH=./d ls", settings)
    assert segment.env_redirect is True


def test_a_name_a_layer_adds_leaves_the_tier_even_with_an_empty_value():
    settings = GateSettings(exec_redirect_env_vars=(*EXEC_REDIRECT_ENV_VARS_DEFAULT, "MY_TOOL"))
    (segment,) = _segments("MY_TOOL= ls", settings)
    assert segment.env_redirect is True and segment.read_only is False
    (segment,) = _segments("MY_TOOL= ls")
    assert segment.env_redirect is False and segment.read_only is True


def test_a_settings_object_cannot_make_path_command_valued():
    """COMMAND_VALUED_ENV_VARS is a code constant: PATH=cat stays out of the tier."""
    settings = GateSettings(exec_redirect_env_vars=("PATH",))
    (segment,) = _segments("PATH=cat ls", settings)
    assert segment.env_redirect is True


def test_the_command_valued_value_is_judged_with_an_operand_present():
    assert OPERAND_PLACEHOLDER == "__lh_operand__"
    for command in ("GIT_EXTERNAL_DIFF='sort -o' git diff", "GIT_EDITOR='tee' git log"):
        assert all(s.env_redirect for s in _segments(command)), command
    for command in ("GIT_PAGER=cat git log", "GIT_EXTERNAL_DIFF=cat git diff"):
        assert not any(s.env_redirect for s in _segments(command)), command


def test_a_substituted_value_leaves_the_tier():
    for command in ("GIT_PAGER=$(echo cat) git log", "GIT_PAGER=$P git log", "GIT_PAGER=`x` git log"):
        assert all(s.env_redirect for s in _segments(command) if s.signature.startswith("git")), command


def test_deeply_nested_values_terminate():
    value = "GIT_PAGER=cat"
    for _ in range(30):
        value = f"GIT_PAGER='{value} git log'"
    classify_shell(f"{value} git log", SETTINGS)


def test_signatures_are_unchanged_by_the_assignment():
    assert classify_shell("PATH=./d ls", SETTINGS).segments[0].signature == "ls"
    assert [s.signature for s in classify_shell("GIT_DIR=./e git diff", SETTINGS).segments] == ["git diff"]


# --------------------------------------------------------------------------- verdicts

def test_path_redirected_ls_asks_shell_unfamiliar_in_guarded(ws):
    result = _verdict("PATH=./d ls", ws, "guarded")
    assert result.verdict is Verdict.ASK
    assert result.request is not None
    assert result.request.klass == "shell-unfamiliar" and result.request.key == "ls"


def test_path_redirected_ls_is_denied_read_only_and_allowed_in_auto(ws):
    assert _verdict("PATH=./d ls", ws, "read-only").verdict is Verdict.DENY
    assert _verdict("PATH=./d ls", ws, "auto").verdict is Verdict.ALLOW


@pytest.mark.parametrize("mode", ("auto", "guarded", "trusted", "read-only", "unattended"))
@pytest.mark.parametrize("command", ["GIT_PAGER=cat git log", "PAGER= git log", "LANG=C ls"])
def test_harmless_assignments_stay_allow_in_every_mode(command, mode, ws):
    assert _verdict(command, ws, mode).verdict is Verdict.ALLOW
