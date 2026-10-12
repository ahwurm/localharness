"""Real-git fixtures for the git_read tests (D-65).

An importable helper module, like tests/unit/launch_support.py: its name does not start with `test_`,
so pytest collects nothing from it. A test module imports what it needs by name:

    from tests.unit.git_support import make_repo, git, requires_git  # noqa: F401

Every helper runs the system `git` under a from-scratch environment (`GIT_CONFIG_NOSYSTEM=1`,
`GIT_CONFIG_GLOBAL=/dev/null`, a scratch HOME) so the developer's own git configuration never leaks
into a fixture and a fixture never touches `~/.localharness` or the project's own `.git`.

- `requires_git` / `REAL_GIT`: skip a test when git is not installed; the absolute path of the binary.
- `git_env(home)` / `git(cwd, *args)`: run git with the scratch environment.
- `make_repo(path, files)`: a one-commit repository on branch `main`.
- `marker_script(directory, name)`: an executable that records that it ran (proof a driver executed).
- `fake_git(directory, body)`: a `git` shim that runs `body` and then the real git.

Later plans only ADD helpers here; none of these signatures changes.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REAL_GIT = shutil.which("git")
requires_git = pytest.mark.skipif(REAL_GIT is None, reason="git is not installed")


def git_env(home: Path) -> dict[str, str]:
    """A from-scratch environment: PATH, a scratch HOME, and no system or global git config."""
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }


def git(
    cwd: Path,
    *args: str,
    home: Path | None = None,
    check: bool = True,
    input: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run real git in `cwd` under `git_env`, with a fixed identity, branch name and file transport."""
    assert REAL_GIT is not None, "use @requires_git"
    return subprocess.run(
        [REAL_GIT, "-c", "user.email=t@t.t", "-c", "user.name=t", "-c", "init.defaultBranch=main",
         "-c", "protocol.file.allow=always", *args],
        cwd=cwd,
        env=git_env(home if home is not None else cwd.parent),
        capture_output=True,
        text=True,
        check=check,
        input=input,
    )


def make_repo(path: Path, files: dict[str, str] | None = None) -> Path:
    """A repository at `path` with one commit holding `files` (default: README)."""
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q")
    for name, text in (files if files is not None else {"README": "hello\n"}).items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "initial")
    return path


def marker_script(directory: Path, name: str, *, passthrough: bool = False) -> tuple[Path, Path]:
    """An executable `#!/bin/sh` at directory/name that appends "$0 $*" to directory/<name>.marker.

    With `passthrough` it also copies stdin to stdout (a textconv or clean filter that records its run).
    Returns (script, marker); the marker does not exist until the script runs.
    """
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / name
    marker = directory / (name + ".marker")
    body = f'#!/bin/sh\necho "$0 $*" >> "{marker}"\n'
    if passthrough:
        body += "cat\n"
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script, marker


def fake_git(directory: Path, body: str) -> Path:
    """Write directory/git: `#!/bin/sh`, `body`, then an exec of the real git with the same arguments.

    Returns `directory`; the caller prepends it to PATH with monkeypatch.setenv.
    """
    assert REAL_GIT is not None, "use @requires_git"
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "git"
    script.write_text(f'#!/bin/sh\n{body}\nexec "{REAL_GIT}" "$@"\n')
    script.chmod(0o755)
    return directory
