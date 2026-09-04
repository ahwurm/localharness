"""A linked git worktree belongs to the project it was cut from (owner ruling R1, 2026-09-04).

`git worktree add ./wt` inside a checkout leaves a `.git` FILE at the worktree root holding one
line: `gitdir: <parent>/.git/worktrees/<name>`. The repo walk stopped at that file, so the main
checkout's `.localharness/` — one directory ABOVE the worktree — read as "config from outside the
tree you opened" and the harness asked the one-time trust question about the user's own repository
(bad-mood finding F8). The fix reads that one line, derives the parent repository's root, and counts
a workspace at or below it as inside.

Parsed, never shelled out: `config/paths.py` spawns no child process, so this works with no git
binary installed and adds nothing to every command's startup. A file that does not parse must not
widen anything — the walk keeps the answer it has today.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from localharness.config.paths import workspace_is_within_repo


def _worktree(tmp_path, *, gitdir_line: str | None, nested: bool = True) -> tuple[Path, Path]:
    """A main checkout with a workspace, plus a linked worktree whose `.git` file says
    `gitdir_line`. Returns (the workspace dir, the directory to stand in).

    `nested` places the worktree INSIDE the main checkout — the shape that actually bites, because
    that is the only way the main checkout's `.localharness/` is found by an up-walk from the
    worktree at all.
    """
    main = tmp_path / "main"
    (main / ".git" / "worktrees" / "wt").mkdir(parents=True)
    workspace = main / ".localharness"
    workspace.mkdir()
    wt = (main / "wt") if nested else (tmp_path / "wt")
    wt.mkdir(parents=True)
    if gitdir_line is not None:
        (wt / ".git").write_text(gitdir_line.format(main=main), encoding="utf-8")
    return workspace, wt


def test_a_worktrees_parent_checkout_is_inside_the_project(tmp_path):
    """The repro shape: the workspace lives in the main checkout, you stand in the worktree."""
    workspace, wt = _worktree(tmp_path, gitdir_line="gitdir: {main}/.git/worktrees/wt\n")

    assert workspace_is_within_repo(workspace, wt) is True


def test_a_relative_gitdir_line_resolves_against_the_git_file(tmp_path):
    """Not every tool writes the absolute path git writes; a relative one names the same dir."""
    workspace, wt = _worktree(tmp_path, gitdir_line="gitdir: ../.git/worktrees/wt\n")

    assert workspace_is_within_repo(workspace, wt) is True


def test_a_subdirectory_of_the_worktree_is_inside_too(tmp_path):
    """The parent root is a property of the repository, not of how deep you are standing in it."""
    workspace, wt = _worktree(tmp_path, gitdir_line="gitdir: {main}/.git/worktrees/wt\n")
    deep = wt / "src" / "deep"
    deep.mkdir(parents=True)

    assert workspace_is_within_repo(workspace, deep) is True


def test_a_workspace_above_the_parent_checkout_is_still_outside(tmp_path):
    """Widening reaches the parent repository and stops there — not one directory further."""
    _, wt = _worktree(tmp_path, gitdir_line="gitdir: {main}/.git/worktrees/wt\n")
    outer = tmp_path / ".localharness"
    outer.mkdir()

    assert workspace_is_within_repo(outer, wt) is False


def test_an_unrelated_tree_is_still_outside(tmp_path):
    """A workspace in somebody else's checkout is not reached by knowing your own parent repo."""
    _, wt = _worktree(tmp_path, gitdir_line="gitdir: {main}/.git/worktrees/wt\n")
    elsewhere = tmp_path / "elsewhere" / ".localharness"
    elsewhere.mkdir(parents=True)

    assert workspace_is_within_repo(elsewhere, wt) is False


@pytest.mark.parametrize(
    "line",
    [
        "not a gitdir line at all\n",
        "gitdir:\n",
        "",
        "gitdir: /nowhere/that/exists/.git/worktrees/wt\n",
        "gitdir: /tmp\n",  # points at no `.git` component: nothing to derive a root from
    ],
    ids=["garbage", "empty-target", "empty-file", "stale-target", "no-git-component"],
)
def test_a_git_file_that_does_not_parse_fails_closed(tmp_path, line):
    """Fail closed: an unreadable pointer keeps today's answer (the worktree root is the root),
    so the workspace above it stays outside and stays trust-gated."""
    workspace, wt = _worktree(tmp_path, gitdir_line=line)

    assert workspace_is_within_repo(workspace, wt) is False


def test_a_pointer_at_a_git_dir_that_is_gone_fails_closed(tmp_path):
    """The parent named by the file must actually be a repository. A worktree left behind by a
    deleted checkout points at a `.git` that is no longer there, and a folder that merely used to
    be a project is not the project you are standing in."""
    import shutil

    workspace, wt = _worktree(tmp_path, gitdir_line="gitdir: {main}/.git/worktrees/wt\n")
    shutil.rmtree(workspace.parent / ".git")

    assert workspace_is_within_repo(workspace, wt) is False


def test_a_worktree_of_a_home_repository_does_not_swallow_home(tmp_path, monkeypatch):
    """The home-dotfiles rule survives the widening: a parent repo AT `$HOME` must not turn every
    folder under home into one project (the same reason the plain walk stops at `$HOME`)."""
    home = tmp_path / "home"
    (home / ".git" / "worktrees" / "wt").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    wt = home / "projects" / "wt"
    wt.mkdir(parents=True)
    (wt / ".git").write_text(f"gitdir: {home}/.git/worktrees/wt\n", encoding="utf-8")
    workspace = home / "projects" / "other" / ".localharness"
    workspace.mkdir(parents=True)

    assert workspace_is_within_repo(workspace, wt) is False


def test_the_worktrees_own_workspace_still_counts(tmp_path):
    """The narrower answer the walk already gives must survive the wider one."""
    _, wt = _worktree(tmp_path, gitdir_line="gitdir: {main}/.git/worktrees/wt\n")
    own = wt / ".localharness"
    own.mkdir()

    assert workspace_is_within_repo(own, wt) is True


def test_the_repro_loads_silently_with_no_prompt(tmp_path, monkeypatch):
    """End to end through the gate: no question, and the main checkout's layer is returned."""
    from localharness.cli.workspace import resolve_workspace_layer

    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home" / ".localharness").mkdir(parents=True)
    workspace, wt = _worktree(tmp_path, gitdir_line="gitdir: {main}/.git/worktrees/wt\n")
    monkeypatch.chdir(wt)

    def _boom(*_a, **_kw):
        raise AssertionError("asked the trust question about the user's own repository")

    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)

    assert resolve_workspace_layer() == workspace
