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
- `git_with_env(cwd, extra_env, *args)`: like `git` with extra environment variables (a bare-git control
  that inherits `GIT_EXTERNAL_DIFF`, `GIT_CONFIG_PARAMETERS`, ...).
- `signed_history(repo, tag)`: HEAD rewritten to carry a handcrafted `gpgsig` header, plus a signed tag.
- `partial_clone(tmp, marker_script_path)`: a `--filter=blob:none` clone whose promisor remote runs a marker.
- `submodule_repo(tmp, filter_script)`: a superproject whose submodule carries its own clean filter.

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


def git_with_env(
    cwd: Path,
    extra_env: dict[str, str],
    *args: str,
    home: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """`git` under `git_env` plus `extra_env` (which wins). For bare-git controls that need an inherited variable."""
    assert REAL_GIT is not None, "use @requires_git"
    env = git_env(home if home is not None else cwd.parent)
    env.update(extra_env)
    return subprocess.run(
        [REAL_GIT, "-c", "user.email=t@t.t", "-c", "user.name=t", "-c", "init.defaultBranch=main",
         "-c", "protocol.file.allow=always", *args],
        cwd=cwd, env=env, capture_output=True, text=True, check=check,
    )


_FAKE_SIGNATURE = (
    "-----BEGIN PGP SIGNATURE-----\n"
    "\n"
    "iQEzBAABCAAdFiEEfakefakefakefakefakefakefakefakeFAKEFAKE\n"
    "=abcd\n"
    "-----END PGP SIGNATURE-----\n"
)


def signed_history(repo: Path, tag: str = "v1") -> str:
    """Rewrite HEAD as a commit carrying a handcrafted `gpgsig` header and add a signed annotated tag.

    Reproduces RESEARCH row c: with `log.showSignature=true` a bare `git log` / `git show` runs
    `gpg.program` over the header (or the tag's trailing PGP block) however fake the signature is.
    Build this BEFORE setting `gpg.program` so creating the history runs nothing. Returns the tag name.
    """
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    tree = git(repo, "rev-parse", "HEAD^{tree}").stdout.strip()
    parents = git(repo, "rev-list", "--parents", "-n", "1", "HEAD").stdout.split()[1:]
    sig_lines = _FAKE_SIGNATURE.rstrip("\n").split("\n")
    # gpgsig continuation lines are indented by one space; an empty one is a single space.
    gpgsig = "gpgsig " + sig_lines[0] + "\n" + "".join(f" {line}\n" for line in sig_lines[1:])
    commit = (
        f"tree {tree}\n"
        + "".join(f"parent {p}\n" for p in parents)
        + "author t <t@t.t> 1700000000 +0000\n"
        + "committer t <t@t.t> 1700000000 +0000\n"
        + gpgsig
        + "\nsigned commit\n"
    )
    new = git(repo, "hash-object", "-t", "commit", "-w", "--stdin", input=commit).stdout.strip()
    git(repo, "update-ref", "HEAD", new, head)
    tag_object = (
        f"object {new}\ntype commit\ntag {tag}\ntagger t <t@t.t> 1700000000 +0000\n\n"
        f"signed tag\n{_FAKE_SIGNATURE}"
    )
    tag_sha = git(repo, "mktag", input=tag_object).stdout.strip()
    git(repo, "update-ref", f"refs/tags/{tag}", tag_sha)
    return tag


def partial_clone(tmp: Path, marker_script_path: Path) -> Path:
    """RESEARCH row g: a blobless, checkout-less clone `tmp/dst` of `tmp/src` whose promisor remote's
    `remote.origin.uploadpack` is the marker script, so any lazy fetch of a missing blob runs it.
    """
    src = make_repo(tmp / "src", {"a.txt": "one\n", "b.txt": "two\n"})
    git(src, "config", "uploadpack.allowFilter", "true")
    git(src, "config", "uploadpack.allowAnySHA1InWant", "true")
    dst = tmp / "dst"
    git(tmp, "clone", "-q", "--filter=blob:none", "--no-checkout", f"file://{src}", str(dst))
    git(dst, "config", "remote.origin.uploadpack", str(marker_script_path))
    return dst


def submodule_repo(tmp: Path, filter_script: Path) -> Path:
    """RESEARCH row h: superproject `tmp/super` with submodule `sub`.

    The submodule's own config (`.git/modules/sub/config`) sets `filter.y.clean` to `filter_script`, its
    committed `.gitattributes` says `* filter=y`, a file in its worktree is modified, and the
    superproject's `.git/config` sets `diff.submodule=log`. History is built before any filter or
    `diff.submodule` config is set, so building runs nothing; the caller removes stale markers.
    """
    subsrc = make_repo(tmp / "subsrc", {"f.txt": "x\n", ".gitattributes": "* filter=y\n"})
    sup = make_repo(tmp / "super", {"top.txt": "top\n"})
    git(sup, "submodule", "add", "-q", f"file://{subsrc}", "sub")
    git(sup, "commit", "-q", "-m", "add sub")
    git(sup / "sub", "config", "filter.y.clean", str(filter_script))
    git(sup, "config", "diff.submodule", "log")
    (sup / "sub" / "f.txt").write_text("changed\n")
    return sup
