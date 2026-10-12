"""git_read helpers: structured-argument validation, the one fixed argv prefix, the from-scratch
environment, the one synchronous git runner and the pins that keep the tool inside the workspace.

Two layers make a read-only `git` safe to hand to a critic that may be prompt-injected:

- Confinement pins keep git inside the effective boundary: the repository's toplevel is the
  workspace root, its git directories (and everything symlinked inside them) are inside the root,
  it borrows no objects from outside and it is owned by the person running the harness.
- Execution neutralisers mean git runs no repository-configured command: the argv prefix switches off
  the fsmonitor, pager, hooks and signature programs, every verb turns off external diff and textconv
  drivers, and the environment is built from scratch rather than inherited.

This module imports nothing from `agent/`; the tool class, its streaming read and the clean-filter
precheck live in later plans.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# D-04 / D-44: the one argv prefix every git call this module makes starts with. `--no-lazy-fetch`
# follows `--no-pager` so a git too old to know it exits 129 (the git_version pin) instead of
# silently fetching a missing blob over the network.
GIT_READ_PREFIX: tuple[str, ...] = (
    "git", "--no-pager", "--no-lazy-fetch",
    "-c", "core.fsmonitor=false",
    "-c", "core.pager=cat",
    "-c", "core.hooksPath=/dev/null",
    "-c", "log.showSignature=false",
    "-c", "color.ui=never",
    "-c", "diff.submodule=short",
)

# D-04 / D-43: flags every verb carries. `dirty`, not `all`, so `log -p` still shows submodule bumps
# while a worktree diff never runs a submodule's clean filter through `git status`.
VERB_GUARD_FLAGS: tuple[str, ...] = ("--no-ext-diff", "--no-textconv", "--ignore-submodules=dirty")

SUBCOMMANDS: tuple[str, ...] = ("diff", "show", "log")

# D-36: limits as module constants (read at call time so a test can patch them).
GIT_READ_TIMEOUT_SECONDS = 30   # the read itself
GIT_PIN_TIMEOUT_SECONDS = 5     # pins, precheck and rev-parse --revs-only
DEFAULT_MAX_LINES = 400
MAX_LINES_LIMIT = 4000
STDERR_CAP_CHARS = 2000

# D-03 / D-63: the phrase every rejected path sentence names.
PATH_FORM = "a path relative to the workspace root"


def _git_env(root: Path) -> dict[str, str]:
    """D-05 / D-74: the environment of every git call, built from scratch.

    PATH keeps only the absolute entries of the inherited one (so `git` can never resolve to a binary
    inside the workspace), HOME is copied when present, and five keys are fixed. Nothing else is
    inherited: no other GIT_*, LANG, LC_* or XDG_*. The ceiling comes from the realpath of the root
    because git compares it against the physical cwd (a symlinked root would otherwise never match).
    """
    env: dict[str, str] = {}
    entries = [e for e in os.environ.get("PATH", "").split(os.pathsep) if e and os.path.isabs(e)]
    if entries:
        env["PATH"] = os.pathsep.join(entries)
    if "HOME" in os.environ:
        env["HOME"] = os.environ["HOME"]
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = "/dev/null"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_NO_LAZY_FETCH"] = "1"
    env["GIT_CEILING_DIRECTORIES"] = os.path.dirname(os.path.realpath(root))
    return env


def _run_git(root: Path, args: Sequence[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[bytes]:
    """D-06 / D-37: run `git <prefix> <args>` synchronously under the from-scratch environment.

    cwd is the realpath of the root, stdin is closed, no shell. The default timeout is the pin one,
    read from the module at call time so a test can patch it. Callers parse stdout as bytes and
    `os.fsdecode` each line (paths may hold non-UTF-8 bytes).
    """
    return subprocess.run(
        [*GIT_READ_PREFIX, *args],
        cwd=os.path.realpath(root),
        env=_git_env(root),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=GIT_PIN_TIMEOUT_SECONDS if timeout is None else timeout,
    )


class GitReadValidationError(ValueError):
    """A rejected argument; str() is the one sentence the tool returns."""


@dataclass(frozen=True)
class ValidatedCall:
    subcommand: str
    revision: str | None
    paths: tuple[str, ...]
    max_lines: int
    stat: bool
    cached: bool


def _has_control(s: str) -> bool:
    return any(ord(c) < 0x20 or ord(c) == 0x7F for c in s)


def _check_path(p: object, real_root: str) -> str:
    """D-03: validate one paths[] entry; return the string that goes into argv."""
    if not isinstance(p, str):
        raise GitReadValidationError("each entry of paths must be a string")
    if not p:
        raise GitReadValidationError("a path may not be empty")
    if _has_control(p):
        raise GitReadValidationError("a path may not contain control characters")
    if ":(prefix:" in p or _magic_has_prefix(p):
        raise GitReadValidationError("the prefix: pathspec magic is not allowed; give " + PATH_FORM)
    s = p
    if os.path.isabs(p):
        try:
            resolved = Path(p).resolve()
        except (OSError, RuntimeError, ValueError):
            raise GitReadValidationError(f"{p!r} cannot be resolved; give {PATH_FORM}") from None
        if not resolved.is_relative_to(real_root):
            raise GitReadValidationError(f"{p!r} is outside the workspace; give {PATH_FORM}")
        s = os.path.relpath(resolved, real_root)
    if ".." in os.path.normpath(s).split("/"):
        raise GitReadValidationError(f"{p!r} leaves the workspace; give {PATH_FORM}")
    if s.startswith("-"):
        raise GitReadValidationError(f"{p!r} may not start with '-'; give {PATH_FORM}")
    return s


def _magic_has_prefix(p: str) -> bool:
    """True when a long-form pathspec magic `:(w1,w2,...)` carries a word starting with `prefix`."""
    if not p.startswith(":("):
        return False
    close = p.find(")")
    if close < 0:
        return False
    return any(w.strip().startswith("prefix") for w in p[2:close].split(","))


def _validate_call(
    root: Path,
    subcommand: object,
    revision: object = None,
    paths: object = None,
    max_lines: object = None,
    stat: object = False,
    cached: object = False,
) -> ValidatedCall:
    """D-01 / D-02 / D-03 / D-63: validate every caller token before any argv exists.

    Every type is re-checked because the registry's schema validator is lax (it coerces "5" and
    "true"). Nothing here spawns a process.
    """
    if not isinstance(subcommand, str) or subcommand not in SUBCOMMANDS:
        raise GitReadValidationError("subcommand must be one of: " + ", ".join(SUBCOMMANDS))
    if revision is not None:
        if not isinstance(revision, str) or not revision:
            raise GitReadValidationError("revision must be a non-empty string")
        if revision.startswith("-"):
            raise GitReadValidationError("revision may not start with '-'")
        if _has_control(revision):
            raise GitReadValidationError("revision may not contain control characters")
    if paths is None:
        raw_paths: list[object] = []
    elif isinstance(paths, list):
        raw_paths = paths
    else:
        raise GitReadValidationError("paths must be a list of strings")
    if max_lines is None:
        limit = DEFAULT_MAX_LINES
    elif isinstance(max_lines, int) and not isinstance(max_lines, bool) and 1 <= max_lines <= MAX_LINES_LIMIT:
        limit = max_lines
    else:
        raise GitReadValidationError(f"max_lines must be an integer from 1 to {MAX_LINES_LIMIT}")
    if not isinstance(stat, bool):
        raise GitReadValidationError("stat must be true or false")
    if not isinstance(cached, bool):
        raise GitReadValidationError("cached must be true or false")
    if cached and subcommand != "diff":
        raise GitReadValidationError("cached applies to diff only")
    real_root = os.path.realpath(root)
    checked = tuple(_check_path(p, real_root) for p in raw_paths)
    return ValidatedCall(subcommand, revision, checked, limit, stat, cached)


def _read_argv(call: ValidatedCall, pathspecs: Sequence[str]) -> list[str]:
    """D-04 / D-35: the argv of the read. `pathspecs` is separate because a later plan passes narrowed ones."""
    argv = [*GIT_READ_PREFIX, call.subcommand, *VERB_GUARD_FLAGS]
    if call.subcommand == "diff":
        if call.cached:
            argv.append("--cached")
        if call.stat:
            argv.append("--stat")
    elif call.subcommand == "show":
        if call.stat:
            argv.append("--stat")
    else:  # log
        argv.append("--stat" if call.stat else "-p")
    if call.revision is not None:
        argv.append(call.revision)
    argv.append("--")
    argv.extend(pathspecs)
    return argv


# ---------------------------------------------------------------------------------------------- pins

# D-76: the walk of the git directories stops here and fails closed.
GIT_DIR_WALK_MAX_ENTRIES = 500_000

NOT_REGISTERED_PREFIX = "git_read not registered: "
REFUSED_PREFIX = "git_read refused: "

# D-31 / D-44: one fixed template per pin, the value through repr() (which escapes control bytes).
# Never git's stderr: it quotes repository-controlled text.
PIN_MESSAGES: dict[str, str] = {
    "git": "git was not found on PATH",
    "git_version": "{value!r} does not support --no-lazy-fetch",
    "ownership": "{value!r} is not owned by you (dubious ownership)",
    "toplevel": "the repository toplevel is not the workspace root ({value!r})",
    "git_dir": "the git directory is outside the workspace root ({value!r})",
    "common_dir": "the common git directory is outside the workspace root ({value!r})",
    "alternates": "the repository borrows objects from outside ({value!r})",
}


@dataclass(frozen=True)
class GitReadPinFailure:
    """The first pin that failed and the value that names it."""

    pin: str
    value: str

    def message(self, *, per_call: bool = False) -> str:
        prefix = REFUSED_PREFIX if per_call else NOT_REGISTERED_PREFIX
        return prefix + PIN_MESSAGES[self.pin].format(value=self.value)


def _owner_mismatch(path: str) -> bool:
    """True when `path` exists and is not owned by the effective uid (lstat: a link counts as itself)."""
    try:
        return os.lstat(path).st_uid != os.geteuid()
    except FileNotFoundError:
        return False


def _symlink_outside(dirs: Sequence[str], real_root: str) -> GitReadPinFailure | None:
    """D-71 / D-76: any symlink inside the git directories that resolves outside the root.

    An iterative scandir walk that never follows a link: DirEntry.is_symlink() and
    is_dir(follow_symlinks=False) read the d_type from readdir, so no entry costs a stat, and
    os.path.realpath runs only for the entries that are symlinks. Every entry is checked, loose object
    files included (a symlinked loose object leaks that object). Nothing is opened. Reaching the cap, or
    an unreadable directory, fails closed as git_dir.
    """
    seen: set[str] = set()
    count = 0
    for top in dirs:
        if top in seen:
            continue
        seen.add(top)
        stack = [top]
        while stack:
            directory = stack.pop()
            try:
                with os.scandir(directory) as it:
                    for entry in it:
                        count += 1
                        if count >= GIT_DIR_WALK_MAX_ENTRIES:
                            return GitReadPinFailure("git_dir", dirs[0])
                        if entry.is_symlink():
                            if not Path(os.path.realpath(entry.path)).is_relative_to(real_root):
                                return GitReadPinFailure("git_dir", entry.path)
                        elif entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
            except OSError:
                return GitReadPinFailure("git_dir", directory)
    return None


def git_read_pin_failure(root: Path) -> GitReadPinFailure | None:
    """D-10: the one function registration, every call and doctor share.

    Checks, in order: git on the from-scratch PATH, the prefix's `version` call, ownership of the root
    and its .git, one `rev-parse` (toplevel, git dir, common dir), the git dir's owner, a walk of the
    git directories for symlinks leaving the root, and the alternates file. Returns the first failure,
    or None for a healthy repository. Synchronous by design (D-37); the tool calls it through
    asyncio.to_thread. The pin message never carries git's stderr (D-31).
    """
    real_root = os.path.realpath(root)
    env = _git_env(root)
    git_path = shutil.which("git", path=env.get("PATH", ""))
    if git_path is None:
        return GitReadPinFailure("git", "")
    try:
        if _run_git(root, ["version"]).returncode != 0:
            return GitReadPinFailure("git_version", git_path)
        # D-30 / D-62: ownership is the tool's own check; no safe.directory is ever passed (D-07).
        if hasattr(os, "geteuid"):
            for path in (real_root, os.path.join(real_root, ".git")):
                if _owner_mismatch(path):
                    return GitReadPinFailure("ownership", path)
        proc = _run_git(root, ["rev-parse", "--path-format=absolute", "--show-toplevel", "--git-dir",
                               "--git-common-dir"])
    except (subprocess.TimeoutExpired, OSError):
        return GitReadPinFailure("toplevel", str(root))
    if proc.returncode != 0:
        return GitReadPinFailure("toplevel", str(root))
    lines = [os.fsdecode(line) for line in proc.stdout.split(b"\n")]
    while lines and not lines[-1]:
        lines.pop()
    if len(lines) != 3:
        return GitReadPinFailure("toplevel", str(root))
    toplevel, git_dir, common_dir = lines
    if Path(toplevel) != Path(real_root):
        return GitReadPinFailure("toplevel", toplevel)
    if not Path(git_dir).is_relative_to(real_root):
        return GitReadPinFailure("git_dir", git_dir)
    if not Path(common_dir).is_relative_to(real_root):
        return GitReadPinFailure("common_dir", common_dir)
    if hasattr(os, "geteuid") and _owner_mismatch(git_dir):
        return GitReadPinFailure("ownership", git_dir)
    # D-71 / D-76: what git reads inside the git directories must not leave the root through a link.
    walk_dirs = [git_dir] if common_dir == git_dir else [git_dir, common_dir]
    failure = _symlink_outside(walk_dirs, real_root)
    if failure is not None:
        return failure
    alternates = os.path.join(common_dir, "objects", "info", "alternates")
    if os.path.lexists(alternates):
        return GitReadPinFailure("alternates", alternates)
    return None


def git_read_repository_present(root: Path) -> bool:
    """D-13: is there a `.git` (directory, file or dangling link) at the root or any parent? No git runs."""
    return any(os.path.lexists(d / ".git") for d in [root, *root.parents])
