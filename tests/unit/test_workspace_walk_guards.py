"""The workspace walk must survive the filesystem it is walking (A-M1/A-M2/A-L2).

Three ways a `.localharness` walk met reality and lost:
  * a directory it cannot read (chmod 000) — `is_dir()` raises PermissionError and the command dies
    with a traceback, on a directory that merely sits between you and the answer;
  * a deleted CWD — `Path.cwd()` raises FileNotFoundError, where v0.12's literal `./.localharness`
    check simply found nothing;
  * `$HOME` unset — `Path.home()` silently falls back to the passwd database, so the stop that
    exists to keep the GLOBAL config dir from being discovered as a workspace can be looking at a
    different directory than the one the config dir actually resolves to.
"""
from __future__ import annotations

import os

import pytest

from localharness.config.paths import discover_workspace_dir, workspace_is_within_repo


def test_unreadable_directory_mid_walk_is_skipped(tmp_path):
    """The walk continues past a directory it may not read, and still finds the workspace above."""
    locked = tmp_path / "locked"
    deep = locked / "proj" / "deep"
    deep.mkdir(parents=True)
    (locked / ".localharness").mkdir()
    os.chmod(locked / "proj", 0o000)
    try:
        assert discover_workspace_dir(deep) == locked / ".localharness"
    finally:
        os.chmod(locked / "proj", 0o755)


def test_deleted_cwd_means_no_workspace(tmp_path, monkeypatch):
    """v0.12 found nothing here (a relative `./.localharness` just does not exist); so do we."""
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()

    assert discover_workspace_dir() is None
    assert workspace_is_within_repo(tmp_path / ".localharness") is False


def test_home_unset_does_not_discover_the_global_config_dir(tmp_path, monkeypatch):
    """With no $HOME, `~/.localharness` still resolves — and must not be found as a workspace."""
    home = tmp_path / "home"
    (home / ".localharness").mkdir(parents=True)
    project = home / "project"
    project.mkdir()
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))
    assert discover_workspace_dir(project) is None  # premise: the home-stop holds when HOME is set

    monkeypatch.delenv("HOME")
    # Path.home() now answers with the passwd database, i.e. NOT this fixture's home — but the
    # config dir under test is still the one at `home`, pinned explicitly so the test does not
    # depend on the machine's real passwd entry.
    monkeypatch.setenv("LOCALHARNESS_HOME", str(home / ".localharness"))

    assert discover_workspace_dir(project) is None


def test_a_real_workspace_is_still_found(tmp_path, monkeypatch):
    """The guards must not cost the ordinary answer."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "proj"
    (project / ".localharness").mkdir(parents=True)
    nested = project / "src" / "deep"
    nested.mkdir(parents=True)

    assert discover_workspace_dir(nested) == project / ".localharness"


@pytest.mark.parametrize("marker", [".git", None])
def test_within_repo_survives_an_unreadable_ancestor(tmp_path, marker):
    """`workspace_is_within_repo` walks the same tree and must not raise on it either."""
    locked = tmp_path / "locked"
    deep = locked / "proj" / "deep"
    deep.mkdir(parents=True)
    (locked / ".localharness").mkdir()
    if marker:
        (locked / marker).mkdir()
    os.chmod(locked / "proj", 0o000)
    try:
        assert workspace_is_within_repo(locked / ".localharness", deep) is bool(marker)
    finally:
        os.chmod(locked / "proj", 0o755)
