"""config/paths.py — ONE config-dir resolution shared by the overlay, loader, model_ops
and the CLI. Bug #35: `--config-dir` isolation was a lie — the overlay path plus three
hardcoded ``~/.localharness/<name>`` defaults ignored it.

Precedence contract: explicit arg > LOCALHARNESS_DIR (the envvar every --config-dir flag
binds) > LOCALHARNESS_HOME (legacy alias) > ~/.localharness.
"""
from __future__ import annotations

from pathlib import Path


def test_resolve_config_dir_precedence(monkeypatch, tmp_path):
    from localharness.config.paths import resolve_config_dir

    explicit = tmp_path / "explicit"
    dir_env = tmp_path / "dir"
    home_env = tmp_path / "home"
    monkeypatch.setenv("LOCALHARNESS_DIR", str(dir_env))
    monkeypatch.setenv("LOCALHARNESS_HOME", str(home_env))

    # Explicit arg wins over both env vars.
    assert resolve_config_dir(explicit) == explicit
    # LOCALHARNESS_DIR (canonical) wins over legacy LOCALHARNESS_HOME.
    assert resolve_config_dir() == dir_env
    # Legacy LOCALHARNESS_HOME still honored when DIR is unset.
    monkeypatch.delenv("LOCALHARNESS_DIR")
    assert resolve_config_dir() == home_env


def test_resolve_config_dir_default(monkeypatch):
    from localharness.config.paths import resolve_config_dir

    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    assert resolve_config_dir() == Path("~/.localharness").expanduser()


def test_resolve_overlay_path_under_config_dir(tmp_path):
    from localharness.config.paths import resolve_overlay_path

    assert resolve_overlay_path(tmp_path) == tmp_path / "overrides.yaml"


def test_resolve_runtime_path_relative_under_config_dir(tmp_path):
    """A bare relative name (the new default for kill_file/audit/history) resolves UNDER the
    config dir — so the default 'KILL' lands at <config_dir>/KILL, exactly ~/.localharness/KILL
    in a default single-instance setup (back-compat invariant)."""
    from localharness.config.paths import resolve_runtime_path

    assert resolve_runtime_path("KILL", tmp_path) == tmp_path / "KILL"
    assert resolve_runtime_path("audit.jsonl", tmp_path) == tmp_path / "audit.jsonl"


def test_resolve_runtime_path_absolute_and_tilde_untouched(tmp_path):
    """An absolute or ~-prefixed user value is honored as-is — never re-rooted under config_dir."""
    from localharness.config.paths import resolve_runtime_path

    absolute = tmp_path / "custom" / "KILL"
    assert resolve_runtime_path(str(absolute), tmp_path) == absolute
    assert (
        resolve_runtime_path("~/somewhere/audit.jsonl", tmp_path)
        == Path("~/somewhere/audit.jsonl").expanduser()
    )
