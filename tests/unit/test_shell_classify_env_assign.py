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
