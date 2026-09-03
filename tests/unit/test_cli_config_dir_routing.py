"""LAYR-02 / Phase 38 criterion 1: ONE config-dir precedence chain on every CLI flag.

Every `--config-dir` flag must resolve through `config.paths.resolve_config_dir()`, whose
chain is: explicit arg > ``LOCALHARNESS_DIR`` > ``LOCALHARNESS_HOME`` (legacy) >
``~/.localharness``.

Before this phase, typer bound only ``envvar="LOCALHARNESS_DIR"`` and supplied the literal
default ``"~/.localharness"`` — so the parameter was NEVER ``None`` and the ``LOCALHARNESS_HOME``
leg was shadowed on these commands, while `components_cmd._build_loader()`'s bare
``ConfigLoader()`` honored it. The LOCALHARNESS_HOME-only cases below are that bug, encoded.

The matrix is four env states x six commands. Assertions observe the RESOLVED DIRECTORY via a
recording wrapper on the chokepoint, never incidental output or exit codes — a command may
legitimately exit non-zero in a bare tmp dir (no config.yaml, no server) and still have
resolved correctly.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.conftest import _MINIMAL_CONFIG_YAML
from localharness.config import paths as _paths
from localharness.cli.app import app

runner = CliRunner()

# Captured BEFORE any monkeypatching so the spy can call through without recursing when the
# patch target is the paths module itself (start_cmd imports the symbol function-locally).
_REAL_RESOLVE = _paths.resolve_config_dir


def _argv(base: list[str]):
    """argv builder: appends --config-dir only when an explicit dir is given."""
    return lambda d: base + (["--config-dir", str(d)] if d else [])


# name -> (argv builder, spy target, needs a seeded config.yaml to reach the resolution).
# The module-level importers bound the symbol by name (`from ... import resolve_config_dir`),
# so the spy must be installed on EACH command module. start_cmd keeps that file's
# function-local import idiom, so patching the paths module is what reaches it.
#
# `init` needs the seed for the opposite reason to the others: a config.yaml already present
# makes it stop at the overwrite prompt, right after resolving — no provider detection, no
# network probes.
COMMANDS: dict[str, tuple] = {
    "doctor": (_argv(["doctor"]), "localharness.cli.doctor_cmd.resolve_config_dir", False),
    "validate": (_argv(["validate"]), "localharness.cli.validate_cmd.resolve_config_dir", False),
    "config-migrate": (
        _argv(["config", "migrate", "--dry-run"]),
        "localharness.cli.config_cmd.resolve_config_dir",
        False,
    ),
    "agent-list": (_argv(["agent", "list"]), "localharness.cli.agent_cmd.resolve_config_dir", False),
    "start": (_argv(["start"]), "localharness.config.paths.resolve_config_dir", False),
    "init": (_argv(["init"]), "localharness.cli.init_cmd.resolve_config_dir", True),
}

COMMAND_IDS = sorted(COMMANDS)

# Every command owning a --config-dir flag, as a --help surface. Superset of COMMAND_IDS:
# `agent create` shares agent_cmd's second flag declaration.
HELP_ARGV = {
    "doctor": ["doctor", "--help"],
    "validate": ["validate", "--help"],
    "config-migrate": ["config", "migrate", "--help"],
    "agent-list": ["agent", "list", "--help"],
    "agent-create": ["agent", "create", "--help"],
    "start": ["start", "--help"],
    "init": ["init", "--help"],
}


@pytest.fixture
def resolved(monkeypatch):
    """Install a recording wrapper on a command module's `resolve_config_dir` and return the
    list it appends every resolved Path to."""

    def install(target: str) -> list[Path]:
        seen: list[Path] = []

        def spy(config_dir=None):
            out = _REAL_RESOLVE(config_dir)
            seen.append(out)
            return out

        monkeypatch.setattr(target, spy)
        return seen

    return install


def _run(name, tmp_path, resolved, explicit=None) -> list[Path]:
    """Invoke `name` (with `explicit` as --config-dir when given) and return what it resolved.

    Seeds a config.yaml into whichever dir the command is expected to land on when that
    command needs one to reach its resolution point.
    """
    argv, target, needs_seed = COMMANDS[name]
    if needs_seed:
        landing = Path(explicit) if explicit else _REAL_RESOLVE(None)
        landing.mkdir(parents=True, exist_ok=True)
        (landing / "config.yaml").write_text(_MINIMAL_CONFIG_YAML, encoding="utf-8")
    seen = resolved(target)
    runner.invoke(app, argv(explicit))
    assert seen, f"{name} never called resolve_config_dir() — not routed through the chokepoint"
    return seen


# ------------------------------------------------------------------ #
# State 1 — nothing set: the ~/.localharness default leg
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("name", COMMAND_IDS)
def test_nothing_set_falls_back_to_home_dot_localharness(name, tmp_path, monkeypatch, resolved):
    """Neither env var set -> ~/.localharness. The zero-behavior-change row.

    HOME is repointed at a tmp dir so this exercises the default leg WITHOUT reading (or, for
    `start`, running the deny-defaults auto-migration against) the developer's real config.
    """
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    expected = Path("~/.localharness").expanduser()
    assert expected == tmp_path / ".localharness", "HOME repointing did not take"

    seen = _run(name, tmp_path, resolved)
    assert seen[0] == expected


# ------------------------------------------------------------------ #
# State 2 — LOCALHARNESS_HOME only: THE bug this phase fixes
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("name", COMMAND_IDS)
def test_localharness_home_only_is_honored(name, tmp_path, monkeypatch, resolved):
    """LOCALHARNESS_HOME set, LOCALHARNESS_DIR unset -> the command operates on that dir.

    THE criterion-1 regression test. Impossible to pass on pre-phase-38 code: typer's literal
    ``"~/.localharness"`` default meant the flag value was never None, so the LOCALHARNESS_HOME
    leg of the chain was unreachable from these commands.
    """
    home = tmp_path / "home-dir"
    home.mkdir()
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.setenv("LOCALHARNESS_HOME", str(home))

    seen = _run(name, tmp_path, resolved)
    assert seen[0] == home


# ------------------------------------------------------------------ #
# State 3 — both env vars: LOCALHARNESS_DIR is canonical
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("name", COMMAND_IDS)
def test_localharness_dir_beats_localharness_home(name, tmp_path, monkeypatch, resolved):
    """LOCALHARNESS_DIR (canonical) outranks LOCALHARNESS_HOME (legacy alias)."""
    canonical = tmp_path / "canonical"
    legacy = tmp_path / "legacy"
    canonical.mkdir()
    legacy.mkdir()
    monkeypatch.setenv("LOCALHARNESS_DIR", str(canonical))
    monkeypatch.setenv("LOCALHARNESS_HOME", str(legacy))

    seen = _run(name, tmp_path, resolved)
    assert seen[0] == canonical


# ------------------------------------------------------------------ #
# State 4 — LAYR-02: explicit --config-dir is full replacement
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("name", COMMAND_IDS)
def test_explicit_flag_beats_both_env_vars(name, tmp_path, monkeypatch, resolved):
    """An explicit --config-dir wins over both env vars — full replacement, not a merge."""
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    monkeypatch.setenv("LOCALHARNESS_DIR", str(tmp_path / "canonical"))
    monkeypatch.setenv("LOCALHARNESS_HOME", str(tmp_path / "legacy"))

    seen = _run(name, tmp_path, resolved, explicit=explicit)
    assert seen[0] == explicit


# ------------------------------------------------------------------ #
# --help honesty: state the real chain, never a bare "None"
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("name", sorted(HELP_ARGV))
def test_help_states_the_real_default_chain(name):
    """--help must name the whole chain. The old text advertised a literal ~/.localharness that
    the env vars silently overrode; a bare `[default: None]` would be just as dishonest."""
    result = runner.invoke(app, HELP_ARGV[name])
    assert result.exit_code == 0, result.output
    assert "LOCALHARNESS_HOME" in result.output
    assert "default: None" not in result.output
