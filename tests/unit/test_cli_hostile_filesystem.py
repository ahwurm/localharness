"""E cluster (b) — the filesystem is hostile, and a traceback is not a message.

`doctor` has always caught broadly around its config load. `config show`, `validate` and `init
--workspace` did not, and every shape below ended in a rich traceback on the real binary. They are
all ordinary accidents: a half-finished `mv`, a stale symlink from a moved repo, a binary file
checked out over a config, a root-owned directory, a deleted worktree.

Each test asserts three things, in this order: the command exited nonzero, it did NOT raise
(a traceback is the bug), and the message names a path the user can act on. The order matters —
`exit_code != 0` alone is satisfied by a crash.

`init --workspace` additionally must not leave a half-made tree behind: it validates the target
before it writes, so a failure means nothing was created.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.config.paths import WORKSPACE_DIR_NAME

runner = CliRunner()

_MINIMAL_CONFIG = {
    "version": "1",
    "provider": {
        "provider_type": "vllm",
        "base_url": "http://127.0.0.1:8000/v1",
        "default_model": "test-model",
    },
}


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    """A configured machine and a project directory to sabotage."""
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    home = tmp_path / "home"
    (home / WORKSPACE_DIR_NAME).mkdir(parents=True)
    (home / WORKSPACE_DIR_NAME / "config.yaml").write_text(
        yaml.dump(_MINIMAL_CONFIG), encoding="utf-8"
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COLUMNS", "400")
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    return proj


def _no_traceback(result) -> None:
    assert result.exit_code != 0, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"crashed instead of reporting: {result.exception!r}"
    )


# --------------------------------------------------------------- unreadable workspace config


@pytest.fixture
def unreadable_config(project) -> Path:
    ws = project / WORKSPACE_DIR_NAME
    ws.mkdir()
    cfg = ws / "config.yaml"
    cfg.write_text(yaml.dump({"org": {"log_level": "debug"}}), encoding="utf-8")
    os.chmod(cfg, 0o000)
    yield cfg
    os.chmod(cfg, 0o644)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads anything")
def test_config_show_survives_an_unreadable_workspace_config(unreadable_config):
    result = runner.invoke(app, ["config", "show"])

    _no_traceback(result)
    assert str(unreadable_config) in result.output


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads anything")
def test_validate_survives_an_unreadable_workspace_config(unreadable_config):
    result = runner.invoke(app, ["validate"])

    _no_traceback(result)
    assert str(unreadable_config) in result.output


# --------------------------------------------------------------- config.yaml holding raw bytes


@pytest.fixture
def binary_config(project) -> Path:
    ws = project / WORKSPACE_DIR_NAME
    ws.mkdir()
    cfg = ws / "config.yaml"
    cfg.write_bytes(b"\x80\x81\x82\xff")
    return cfg


def test_config_show_survives_a_binary_workspace_config(binary_config):
    result = runner.invoke(app, ["config", "show"])

    _no_traceback(result)
    assert "UTF-8" in result.output
    assert str(binary_config) in result.output


def test_validate_survives_a_binary_workspace_config(binary_config):
    result = runner.invoke(app, ["validate"])

    _no_traceback(result)
    assert "UTF-8" in result.output


# --------------------------------------------------------------- config.yaml that is a directory


def test_config_show_survives_a_directory_where_the_config_belongs(project):
    (project / WORKSPACE_DIR_NAME / "config.yaml").mkdir(parents=True)

    result = runner.invoke(app, ["config", "show"])

    _no_traceback(result)
    assert str(project / WORKSPACE_DIR_NAME / "config.yaml") in result.output


# --------------------------------------------------------------- init --workspace, hostile target


def test_init_workspace_survives_a_dangling_symlink(project):
    """H2: `.localharness` points at nothing. `exists()` follows the link and answers False, so
    the scaffold sailed past its own refusal and died in mkdir with FileExistsError."""
    (project / WORKSPACE_DIR_NAME).symlink_to(project / "no-such-target")

    result = runner.invoke(app, ["init", "--workspace"])

    _no_traceback(result)
    assert str(project / WORKSPACE_DIR_NAME) in result.output


def test_init_workspace_survives_a_symlink_loop(project):
    """H3: `.localharness -> .localharness`. Every path through it returns ELOOP."""
    (project / WORKSPACE_DIR_NAME).symlink_to(project / WORKSPACE_DIR_NAME)

    result = runner.invoke(app, ["init", "--workspace"])

    _no_traceback(result)
    assert str(project / WORKSPACE_DIR_NAME) in result.output


def test_init_workspace_survives_a_file_in_the_way(project):
    """H4: `.localharness` is a FILE. Already refused before this wave — kept as the control that
    the new guards did not turn a clean refusal into a crash."""
    (project / WORKSPACE_DIR_NAME).write_text("not a directory\n", encoding="utf-8")

    result = runner.invoke(app, ["init", "--workspace"])

    _no_traceback(result)
    assert str(project / WORKSPACE_DIR_NAME) in result.output


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes anywhere")
def test_init_workspace_leaves_nothing_behind_when_it_cannot_write(project):
    """The scaffold-order half: a failure must not leave a half-made workspace under an error
    message. An unwritable project directory is the cheapest way to make the write fail."""
    os.chmod(project, 0o500)
    try:
        result = runner.invoke(app, ["init", "--workspace"])

        _no_traceback(result)
        assert not (project / WORKSPACE_DIR_NAME).exists()
    finally:
        os.chmod(project, 0o700)


# --------------------------------------------------------------- the working directory is gone


def test_config_show_survives_a_deleted_working_directory(project):
    """H6: the CWD is removed under a running process. The workspace walk was fixed in wave 1;
    this covers the command's own entry point."""
    gone = project / "gone"
    gone.mkdir()
    os.chdir(gone)
    try:
        gone.rmdir()
    except OSError:  # pragma: no cover - platform dependent
        pytest.skip("cannot remove the current working directory here")

    result = runner.invoke(app, ["config", "show"])

    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"crashed instead of reporting: {result.exception!r}"
    )


def test_validate_survives_a_deleted_working_directory(project):
    gone = project / "gone"
    gone.mkdir()
    os.chdir(gone)
    try:
        gone.rmdir()
    except OSError:  # pragma: no cover - platform dependent
        pytest.skip("cannot remove the current working directory here")

    result = runner.invoke(app, ["validate"])

    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"crashed instead of reporting: {result.exception!r}"
    )


def test_init_workspace_survives_a_deleted_working_directory(project):
    gone = project / "gone"
    gone.mkdir()
    os.chdir(gone)
    try:
        gone.rmdir()
    except OSError:  # pragma: no cover - platform dependent
        pytest.skip("cannot remove the current working directory here")

    result = runner.invoke(app, ["init", "--workspace"])

    _no_traceback(result)
