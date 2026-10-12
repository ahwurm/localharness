"""git_read pins from real-git layouts, their fixed wording and the lexical repository walk (GITREAD-03, -06).

Every message assertion compares against a literal written out here (never built from PIN_MESSAGES), so
an edit of a template breaks a test. Real git runs in tmp_path only, under git_support's from-scratch
environment.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from localharness.tools.builtin import git_read_tool as grt
from localharness.tools.builtin.git_read_tool import (
    GIT_READ_PREFIX,
    GitReadPinFailure,
    git_read_pin_failure,
    git_read_repository_present,
)
from tests.unit.git_support import fake_git, git, make_repo, requires_git


@pytest.fixture(autouse=True)
def _scratch_home(tmp_path, monkeypatch):
    home = tmp_path / "_home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


@pytest.fixture
def base(tmp_path: Path) -> Path:
    b = Path(os.path.realpath(tmp_path)) / "w"
    b.mkdir()
    return b


@pytest.fixture
def root(base: Path) -> Path:
    return make_repo(base / "ROOT")


@pytest.fixture
def outside(base: Path) -> Path:
    return make_repo(base / "outside")


def _msg(pin: GitReadPinFailure | None) -> str:
    assert pin is not None
    return pin.message()


# ------------------------------------------------------------------ healthy and symlinks inside .git (D-71)

@requires_git
def test_healthy_repository_passes(root):
    assert git_read_pin_failure(root) is None


@requires_git
@pytest.mark.parametrize("rel", ["objects", "objects/pack", "refs/heads/main", "config"])
def test_symlink_inside_git_dir_to_outside_refused(root, outside, rel):
    link = root / ".git" / rel
    target = outside / ".git" / rel
    assert target.exists()
    if link.is_dir() and not link.is_symlink():
        shutil.rmtree(link)
    else:
        link.unlink()
    os.symlink(target, link)
    pin = git_read_pin_failure(root)
    assert pin == GitReadPinFailure("git_dir", str(link))
    text = pin.message()
    assert text == "git_read not registered: the git directory is outside the workspace root (" + repr(str(link)) + ")"
    assert "fatal" not in text


@requires_git
def test_symlink_inside_git_dir_to_inside_root_legal(root):
    pack = root / ".git" / "objects" / "pack"
    moved = root / "packs"
    shutil.move(str(pack), str(moved))
    os.symlink(moved, pack)
    assert git_read_pin_failure(root) is None


@requires_git
def test_walk_cap_fails_closed_and_admits_below(root, monkeypatch):
    entries = sum(len(d) + len(f) for _, d, f in os.walk(root / ".git"))
    monkeypatch.setattr(grt, "GIT_DIR_WALK_MAX_ENTRIES", entries - 5)
    assert git_read_pin_failure(root) == GitReadPinFailure("git_dir", str(root / ".git"))
    monkeypatch.setattr(grt, "GIT_DIR_WALK_MAX_ENTRIES", entries + 5)
    assert git_read_pin_failure(root) is None


def test_walk_cap_constant():
    assert grt.GIT_DIR_WALK_MAX_ENTRIES == 500_000


# ------------------------------------------------------------------ PATH handling (D-74)

@requires_git
def test_workspace_binary_on_relative_path_never_resolved(root, monkeypatch):
    fake_git(root / "bin", "exit 129")
    monkeypatch.setenv("PATH", "bin" + os.pathsep + os.environ["PATH"])
    monkeypatch.chdir(root)
    assert git_read_pin_failure(root) is None


@requires_git
def test_same_fake_on_absolute_path_fails_git_version(root, base, monkeypatch):
    fake_dir = fake_git(base / "fakebin", "exit 129")
    monkeypatch.setenv("PATH", str(fake_dir) + os.pathsep + os.environ["PATH"])
    assert git_read_pin_failure(root) == GitReadPinFailure("git_version", str(fake_dir / "git"))


@requires_git
def test_git_missing_from_path(root, base, monkeypatch):
    empty = base / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    pin = git_read_pin_failure(root)
    assert pin == GitReadPinFailure("git", "")
    assert pin.message() == "git_read not registered: git was not found on PATH"


@requires_git
def test_old_git_without_no_lazy_fetch(root, base, monkeypatch):
    fake_dir = fake_git(base / "fakebin", 'for a in "$@"; do [ "$a" = "--no-lazy-fetch" ] && exit 129; done')
    monkeypatch.setenv("PATH", str(fake_dir) + os.pathsep + os.environ["PATH"])
    pin = git_read_pin_failure(root)
    assert pin == GitReadPinFailure("git_version", str(fake_dir / "git"))
    assert pin.message() == (
        "git_read not registered: " + repr(str(fake_dir / "git")) + " does not support --no-lazy-fetch")


# ------------------------------------------------------------------ layouts that point outside (D-12, D-33)

@requires_git
def test_dot_git_symlink_to_outside(root, outside):
    shutil.rmtree(root / ".git")
    os.symlink(outside / ".git", root / ".git")
    assert git_read_pin_failure(root) == GitReadPinFailure("git_dir", os.path.realpath(outside / ".git"))


@requires_git
def test_gitdir_file_pointing_outside(root, outside):
    shutil.rmtree(root / ".git")
    (root / ".git").write_text(f"gitdir: {outside / '.git'}\n")
    pin = git_read_pin_failure(root)
    assert pin is not None and pin.pin == "git_dir"


@requires_git
def test_commondir_pointing_outside(root, outside):
    (root / ".git" / "commondir").write_text(str(outside / ".git") + "\n")
    pin = git_read_pin_failure(root)
    assert pin is not None and pin.pin == "common_dir"
    assert pin.message() == (
        "git_read not registered: the common git directory is outside the workspace root ("
        + repr(pin.value) + ")")


@requires_git
@pytest.mark.parametrize("content", ["", "/somewhere/objects\n"])
def test_alternates_refused(root, content):
    alt = root / ".git" / "objects" / "info" / "alternates"
    alt.parent.mkdir(parents=True, exist_ok=True)
    alt.write_text(content)
    pin = git_read_pin_failure(root)
    assert pin == GitReadPinFailure("alternates", str(alt))
    assert pin.message() == (
        "git_read not registered: the repository borrows objects from outside (" + repr(str(alt)) + ")")


@requires_git
def test_linked_worktree_outside_main_refused(base):
    main = make_repo(base / "main")
    wt = base / "wt"
    git(main, "worktree", "add", str(wt))
    pin = git_read_pin_failure(wt)
    assert pin is not None and pin.pin == "git_dir"
    assert pin.message().startswith("git_read not registered: the git directory is outside the workspace root (")


@requires_git
def test_non_repository_below_a_repository(base):
    parent = make_repo(base / "P")
    sub = parent / "sub"
    sub.mkdir()
    pin = git_read_pin_failure(sub)
    assert pin == GitReadPinFailure("toplevel", str(sub))
    assert pin.message() == (
        "git_read not registered: the repository toplevel is not the workspace root (" + repr(str(sub)) + ")")
    assert "fatal" not in pin.message() and "not a git repository" not in pin.message()


@requires_git
def test_symlinked_root_pointing_below_a_repository(base):
    parent = make_repo(base / "P")
    (parent / "sub").mkdir()
    link = base / "L"
    os.symlink(parent / "sub", link)
    pin = git_read_pin_failure(link)
    assert pin is not None and pin.pin == "toplevel"


@requires_git
def test_no_repository_anywhere(base):
    plain = base / "plain"
    plain.mkdir()
    pin = git_read_pin_failure(plain)
    assert pin is not None and pin.pin == "toplevel"


# ------------------------------------------------------------------ ownership, timeouts, spawn shape

@requires_git
def test_ownership_mismatch(root, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: os.stat(root).st_uid + 1)
    pin = git_read_pin_failure(root)
    assert pin == GitReadPinFailure("ownership", os.path.realpath(root))
    assert pin.message() == (
        "git_read not registered: " + repr(os.path.realpath(root)) + " is not owned by you (dubious ownership)")


@requires_git
def test_rev_parse_timeout_fails_closed(root, monkeypatch):
    real_run = subprocess.run

    def run(argv, **kw):
        if "rev-parse" in argv:
            raise subprocess.TimeoutExpired(argv, 5)
        return real_run(argv, **kw)

    monkeypatch.setattr(subprocess, "run", run)
    assert git_read_pin_failure(root) == GitReadPinFailure("toplevel", str(root))


@requires_git
def test_one_rev_parse_call_and_every_argv_has_the_prefix(root, monkeypatch):
    real_run = subprocess.run
    seen: list[list[str]] = []

    def run(argv, **kw):
        seen.append(list(argv))
        return real_run(argv, **kw)

    monkeypatch.setattr(subprocess, "run", run)
    assert git_read_pin_failure(root) is None
    assert all(tuple(a[: len(GIT_READ_PREFIX)]) == GIT_READ_PREFIX for a in seen)
    rev = [a for a in seen if "rev-parse" in a]
    assert len(rev) == 1
    for flag in ("--path-format=absolute", "--show-toplevel", "--git-dir", "--git-common-dir"):
        assert flag in rev[0]
    assert not any("safe.directory" in t for a in seen for t in a)


# ------------------------------------------------------------------ the fixed wording (D-31, D-44)

def test_message_templates_verbatim():
    N, R = "git_read not registered: ", "git_read refused: "
    cases = {
        "git": "git was not found on PATH",
        "git_version": "'/p/git' does not support --no-lazy-fetch",
        "ownership": "'/p/git' is not owned by you (dubious ownership)",
        "toplevel": "the repository toplevel is not the workspace root ('/p/git')",
        "git_dir": "the git directory is outside the workspace root ('/p/git')",
        "common_dir": "the common git directory is outside the workspace root ('/p/git')",
        "alternates": "the repository borrows objects from outside ('/p/git')",
    }
    for pin, text in cases.items():
        failure = GitReadPinFailure(pin, "/p/git")
        assert failure.message() == N + text
        assert failure.message(per_call=True) == R + text


def test_message_value_goes_through_repr():
    pin = GitReadPinFailure("git_dir", "/a\nb")
    assert pin.message() == "git_read not registered: the git directory is outside the workspace root ('/a\\nb')"
    assert "\n" not in pin.message()


# ------------------------------------------------------------------ git_read_repository_present (D-13)

def test_repository_present_forms(base, monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("git_read_repository_present must not spawn a process")

    monkeypatch.setattr(subprocess, "run", boom)
    d = base / "dir"
    (d / ".git").mkdir(parents=True)
    f = base / "file"
    f.mkdir()
    (f / ".git").write_text("gitdir: x\n")
    dangling = base / "dangling"
    dangling.mkdir()
    os.symlink(base / "nowhere", dangling / ".git")
    below = d / "a" / "b"
    below.mkdir(parents=True)
    for r in (d, f, dangling, below):
        assert git_read_repository_present(r) is True


def test_repository_absent(base):
    # Walk from a tree no .git can be above: tmp_path lives under pytest's basetemp, outside any repo.
    empty = base / "none"
    empty.mkdir()
    if any(os.path.lexists(p / ".git") for p in [empty, *empty.parents]):
        pytest.skip("the temp directory sits below a .git on this machine")
    assert git_read_repository_present(empty) is False
