"""LAYR-02 / Phase 38 criterion 1: ONE config-dir precedence chain on every CLI flag.

Every `--config-dir` flag must resolve through `config.paths.resolve_config_dir()`, whose
chain is: explicit arg > ``LOCALHARNESS_DIR`` > ``LOCALHARNESS_HOME`` (legacy) >
``~/.localharness``.

Before this phase, typer bound only ``envvar="LOCALHARNESS_DIR"`` and supplied the literal
default ``"~/.localharness"`` — so the parameter was NEVER ``None`` and the ``LOCALHARNESS_HOME``
leg was shadowed on these five commands, while `components_cmd._build_loader()`'s bare
``ConfigLoader()`` honored it. The LOCALHARNESS_HOME-only cases below are that bug, encoded.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from localharness.config import paths as _paths
from localharness.cli.app import app

runner = CliRunner()

# Captured BEFORE any monkeypatching so the spy can call through without recursing when the
# patch target is the paths module itself (start_cmd imports the symbol function-locally).
_REAL_RESOLVE = _paths.resolve_config_dir


def _argv(base: list[str]):
    """argv builder: appends --config-dir only when an explicit dir is given."""
    return lambda d: base + (["--config-dir", str(d)] if d else [])


# name -> (argv builder, spy target).
# The four module-level importers bound the symbol by name (`from ... import
# resolve_config_dir`), so the spy must be installed on EACH command module. start_cmd keeps
# that file's function-local import idiom, so patching the paths module is what reaches it.
COMMANDS: dict[str, tuple] = {
    "doctor": (_argv(["doctor"]), "localharness.cli.doctor_cmd.resolve_config_dir"),
    "validate": (_argv(["validate"]), "localharness.cli.validate_cmd.resolve_config_dir"),
    "config-migrate": (
        _argv(["config", "migrate", "--dry-run"]),
        "localharness.cli.config_cmd.resolve_config_dir",
    ),
    "agent-list": (_argv(["agent", "list"]), "localharness.cli.agent_cmd.resolve_config_dir"),
    "start": (_argv(["start"]), "localharness.config.paths.resolve_config_dir"),
}

COMMAND_IDS = sorted(COMMANDS)


@pytest.fixture
def resolved(monkeypatch):
    """Install a recording wrapper on a command module's `resolve_config_dir` and return the
    list it appends every resolved Path to. Observes the RESOLVED DIRECTORY directly, so the
    assertions never depend on incidental command output or exit codes."""

    def install(target: str) -> list[Path]:
        seen: list[Path] = []

        def spy(config_dir=None):
            out = _REAL_RESOLVE(config_dir)
            seen.append(out)
            return out

        monkeypatch.setattr(target, spy)
        return seen

    return install


@pytest.mark.parametrize("name", COMMAND_IDS)
def test_localharness_home_only_is_honored(name, tmp_path, monkeypatch, resolved):
    """LOCALHARNESS_HOME set, LOCALHARNESS_DIR unset -> the command operates on that dir.

    THE criterion-1 regression test. Impossible to pass on pre-phase-38 code: typer's literal
    ``"~/.localharness"`` default meant the flag value was never None, so the LOCALHARNESS_HOME
    leg of the chain was unreachable from these commands.
    """
    argv, target = COMMANDS[name]
    home = tmp_path / "home-dir"
    home.mkdir()
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.setenv("LOCALHARNESS_HOME", str(home))

    seen = resolved(target)
    runner.invoke(app, argv(None))

    assert seen, f"{name} never called resolve_config_dir()"
    assert seen[0] == home
