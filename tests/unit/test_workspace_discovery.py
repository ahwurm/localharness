"""The up-tree `.localharness/` walk and the project-boundary test it is judged against.

Two pure functions in `config/paths.py`, no callers yet (39-04 wires them):

- `discover_workspace_dir(start)` — the nearest `.localharness/` at or above `start`.
  "Nearest wins" (LAYR-04) is a property of returning on the FIRST hit, never of a later
  merge step. The walk stops at `$HOME` without inspecting it, because `~/.localharness/`
  IS the global layer — without that stop, a user with no project workspace would
  "discover" their own global dir from anywhere under home and LAYR-03's byte-identical
  guarantee would not hold.
- `workspace_is_within_repo(workspace_dir, start)` — does that workspace belong to the
  project you are standing in? Answered from `.git` markers alone (a DIRECTORY in a clone,
  a FILE in a worktree/submodule), with no subprocess and no git binary. Owner ruling
  2026-09-03: only a workspace from OUTSIDE the project is trust-gated.

Env note: `Path.home()` reads `$HOME` on POSIX, so the `$HOME` cases monkeypatch it. The
project fixtures live under `tmp_path`, which is never inside the fake home.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest


# --------------------------------------------------------------------------- the walk


def test_finds_workspace_in_the_start_dir(tmp_path):
    from localharness.config.paths import discover_workspace_dir

    ws = tmp_path / ".localharness"
    ws.mkdir()
    assert discover_workspace_dir(tmp_path) == ws


def test_walks_up_to_an_ancestor_workspace(tmp_path):
    from localharness.config.paths import discover_workspace_dir

    ws = tmp_path / ".localharness"
    ws.mkdir()
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    assert discover_workspace_dir(deep) == ws


def test_nearest_workspace_wins(tmp_path):
    """LAYR-04: exactly one workspace layer, and it is the closest one up-tree."""
    from localharness.config.paths import discover_workspace_dir

    (tmp_path / ".localharness").mkdir()
    inner_ws = tmp_path / "a" / ".localharness"
    inner_ws.mkdir(parents=True)
    start = tmp_path / "a" / "b"
    start.mkdir()

    assert discover_workspace_dir(start) == inner_ws


def test_returns_none_when_nothing_up_tree(tmp_path):
    from localharness.config.paths import discover_workspace_dir

    start = tmp_path / "a" / "b"
    start.mkdir(parents=True)
    assert discover_workspace_dir(start) is None


def test_home_localharness_is_never_a_workspace(tmp_path, monkeypatch):
    """LAYR-03: `~/.localharness/` is the GLOBAL layer. Discovering it as a workspace would
    collapse the two layers for every user who has no project workspace at all."""
    from localharness.config.paths import discover_workspace_dir

    fake_home = tmp_path / "home"
    (fake_home / ".localharness").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    start = fake_home / "sub" / "dir"
    start.mkdir(parents=True)

    assert discover_workspace_dir(start) is None


def test_home_itself_is_never_a_workspace(tmp_path, monkeypatch):
    """The stop applies at `$HOME` exactly, not just below it."""
    from localharness.config.paths import discover_workspace_dir

    fake_home = tmp_path / "home"
    (fake_home / ".localharness").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))

    assert discover_workspace_dir(fake_home) is None


def test_a_localharness_file_is_not_a_workspace(tmp_path):
    """`.localharness` as a FILE is not a match — the walk continues past it."""
    from localharness.config.paths import discover_workspace_dir

    outer_ws = tmp_path / ".localharness"
    outer_ws.mkdir()
    inner = tmp_path / "a"
    inner.mkdir()
    (inner / ".localharness").write_text("not a workspace\n", encoding="utf-8")

    assert discover_workspace_dir(inner) == outer_ws


def test_symlinked_start_resolves_to_the_real_workspace(tmp_path):
    from localharness.config.paths import discover_workspace_dir

    real = tmp_path / "real"
    ws = real / ".localharness"
    ws.mkdir(parents=True)
    (real / "sub").mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    found = discover_workspace_dir(link / "sub")
    assert found == ws.resolve()
    assert found is not None and not found.is_symlink()


def test_no_argument_uses_cwd(tmp_path, monkeypatch):
    from localharness.config.paths import discover_workspace_dir

    ws = tmp_path / ".localharness"
    ws.mkdir()
    deep = tmp_path / "a"
    deep.mkdir()
    monkeypatch.chdir(deep)

    assert discover_workspace_dir() == ws.resolve()


def test_accepts_a_string_start(tmp_path):
    """PathLike, like the rest of config/paths.py: a raw flag value or a Path."""
    from localharness.config.paths import discover_workspace_dir

    ws = tmp_path / ".localharness"
    ws.mkdir()
    assert discover_workspace_dir(str(tmp_path)) == ws


def test_workspace_dir_name_is_the_single_anchor():
    """The global default DERIVES from the same constant the walk uses, so the two layers'
    directory name can never drift apart (owner rule: no bare magic strings)."""
    from localharness.config import paths

    assert paths.WORKSPACE_DIR_NAME == ".localharness"
    assert paths._DEFAULT_CONFIG_DIR == "~/.localharness"
    assert paths.GIT_DIR_NAME == ".git"


# ----------------------------------------------------------------------- the boundary


def _repo_with_workspace(root: pathlib.Path, *, git_is_file: bool = False) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".git"
    if git_is_file:
        # The git-worktree / submodule shape: `.git` is a FILE pointing elsewhere.
        marker.write_text("gitdir: /elsewhere/.git/worktrees/wt\n", encoding="utf-8")
    else:
        marker.mkdir()
    ws = root / ".localharness"
    ws.mkdir()
    return ws


def test_within_repo_when_git_is_a_directory(tmp_path):
    from localharness.config.paths import workspace_is_within_repo

    repo = tmp_path / "repo"
    ws = _repo_with_workspace(repo)
    start = repo / "a" / "b"
    start.mkdir(parents=True)

    assert workspace_is_within_repo(ws, start) is True


def test_within_repo_when_git_is_a_file(tmp_path):
    """A git worktree (and a submodule) has `.git` as a FILE — still a repo."""
    from localharness.config.paths import workspace_is_within_repo

    repo = tmp_path / "wt"
    ws = _repo_with_workspace(repo, git_is_file=True)
    start = repo / "a" / "b"
    start.mkdir(parents=True)

    assert workspace_is_within_repo(ws, start) is True


def test_within_repo_when_workspace_sits_below_the_repo_root(tmp_path):
    """Nested inherits: you are inside the project, so its config is not config from elsewhere."""
    from localharness.config.paths import workspace_is_within_repo

    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    ws = repo / "pkg" / ".localharness"
    ws.mkdir(parents=True)
    start = repo / "pkg" / "deep"
    start.mkdir()

    assert workspace_is_within_repo(ws, start) is True


def test_not_within_repo_when_workspace_sits_above_the_repo_root(tmp_path):
    """Config reaching IN from outside the tree you opened — the only trust-gated case."""
    from localharness.config.paths import workspace_is_within_repo

    outer = tmp_path / "outer"
    ws = outer / ".localharness"
    ws.mkdir(parents=True)
    inner = outer / "inner"
    (inner / ".git").mkdir(parents=True)
    start = inner / "sub"
    start.mkdir()

    assert workspace_is_within_repo(ws, start) is False


def test_not_within_repo_when_no_git_exists_at_all(tmp_path):
    from localharness.config.paths import workspace_is_within_repo

    ws = tmp_path / "proj" / ".localharness"
    ws.mkdir(parents=True)
    start = tmp_path / "proj" / "sub"
    start.mkdir()

    assert workspace_is_within_repo(ws, start) is False


def test_git_search_stops_at_home(tmp_path, monkeypatch):
    """A home-directory dotfiles repo must not make every folder under home one project."""
    from localharness.config.paths import workspace_is_within_repo

    fake_home = tmp_path / "home"
    (fake_home / ".git").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    ws = fake_home / "proj" / ".localharness"
    ws.mkdir(parents=True)
    start = fake_home / "proj" / "sub"
    start.mkdir()

    assert workspace_is_within_repo(ws, start) is False


def test_boundary_never_shells_out_to_git(tmp_path, monkeypatch):
    """Pure filesystem reads: works with no git binary and adds no process spawn to startup."""
    from localharness.config.paths import workspace_is_within_repo

    def _explode(*args, **kwargs):  # pragma: no cover - the assertion is that it never runs
        raise AssertionError("workspace_is_within_repo spawned a subprocess")

    for name in ("run", "check_output", "Popen", "call"):
        monkeypatch.setattr(subprocess, name, _explode)

    repo = tmp_path / "repo"
    ws = _repo_with_workspace(repo)
    start = repo / "a"
    start.mkdir()

    assert workspace_is_within_repo(ws, start) is True


# ------------------------------------------------------------- the import-time guard


def test_discovery_is_never_captured_at_import_time():
    """Risk #3: the workspace root varies per CWD, so a module-level constant would freeze
    a wrong answer for the whole process (config/overlay.py:32-35 is the existing trap)."""
    src = pathlib.Path(__file__).resolve().parents[2] / "src" / "localharness"
    pat = re.compile(
        r"^[A-Z_]+\s*=\s*"
        r"(discover_workspace_dir|workspace_is_within_repo|resolve_workspace_layer)\("
    )
    modules = list(src.rglob("*.py"))
    assert modules, f"source scan found no modules under {src} — the guard would be vacuous"
    offenders = [
        f"{p}:{line}"
        for p in modules
        for line in p.read_text(encoding="utf-8").splitlines()
        if pat.match(line)
    ]
    assert offenders == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX $HOME semantics")
def test_home_stop_survives_a_symlinked_home(tmp_path, monkeypatch):
    """`$HOME` reached through a symlink is still `$HOME` — both sides are realpath'd."""
    from localharness.config.paths import discover_workspace_dir

    real_home = tmp_path / "real_home"
    (real_home / ".localharness").mkdir(parents=True)
    start = real_home / "sub"
    start.mkdir()
    link_home = tmp_path / "link_home"
    link_home.symlink_to(real_home, target_is_directory=True)
    monkeypatch.setenv("HOME", str(link_home))

    assert discover_workspace_dir(start) is None
