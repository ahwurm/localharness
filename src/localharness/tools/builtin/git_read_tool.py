"""git_read helpers: structured-argument validation, the one fixed argv prefix, the from-scratch
environment, the one synchronous git runner and the pins that keep the tool inside the workspace.

Two layers make a read-only `git` safe to hand to a critic that may be prompt-injected:

- Confinement pins keep git inside the effective boundary: the repository's toplevel is the
  workspace root, its git directories (and everything symlinked inside them) are inside the root,
  it borrows no objects from outside and it is owned by the person running the harness.
- Execution neutralisers mean git runs no repository-configured command: the argv prefix switches off
  the fsmonitor, pager, hooks and signature programs, every verb turns off external diff and textconv
  drivers, and the environment is built from scratch rather than inherited.

This module imports nothing from `agent/`. `GitReadTool` (the declaration, the per-call pin check and
the clean-filter precheck and the streaming read) lives at the end of the module.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from localharness.tools.base import Tool, ToolResult, ToolSchema

# D-04 / D-44: the one argv prefix every git call this module makes starts with. `--no-lazy-fetch`
# follows `--no-pager` so a git too old to know it exits 129 (the git_version pin) instead of
# silently fetching a missing blob over the network.
GIT_READ_PREFIX: tuple[str, ...] = (
    "git", "--no-pager", "--no-lazy-fetch",
    "-c", "core.fsmonitor=false",
    "-c", "core.pager=cat",
    "-c", "core.hooksPath=/dev/null",
    "-c", "log.showSignature=false",
    "-c", "format.pretty=medium",
    "-c", "gpg.program=false",
    "-c", "gpg.openpgp.program=false",
    "-c", "gpg.ssh.program=false",
    "-c", "gpg.x509.program=false",
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
    _magic(p)  # D-47: an unterminated ":(" is rejected here, before any process
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


def _magic(element: str) -> tuple[str, list[str], str]:
    """D-47: split one pathspec element into (kind, magic words, path part).

    Long form `:(w1,w2,...)rest` gives its words; short form is `:` followed by any run of `/`, `!`, `^`
    and an optional `:` terminator (`/` is the word "top", `!` or `^` the word "exclude"); `:` with no
    magic letter keeps no words; anything else is plain. An element opening `:(` with no `)` is rejected.
    """
    if element.startswith(":("):
        close = element.find(")")
        if close < 0:
            raise GitReadValidationError(
                f"{element!r} has an unterminated ':(' pathspec magic; give {PATH_FORM}"
            )
        words = [w.strip() for w in element[2:close].split(",")]
        return "long", [w for w in words if w], element[close + 1:]
    if element.startswith(":"):
        i = 1
        words = []
        while i < len(element) and element[i] in "/!^":
            words.append("top" if element[i] == "/" else "exclude")
            i += 1
        if i < len(element) and element[i] == ":":
            i += 1
        return "short", words, element[i:]
    return "plain", [], element


def _narrow_for_filters(paths: Sequence[str]) -> list[str]:
    """D-15 / D-46: keep a worktree diff off every path a clean or process filter binds.

    Each positive element is emitted twice, under `:(attr:!filter)` (filter unspecified) and
    `:(attr:-filter)` (filter unset), merged into the long-form magic it already carries; `:/x` is
    normalised to `:(top)x` first. Every exclusion is kept verbatim. When no positive element exists the
    two bare positive elements come first: an exclusion with no positive pathspec applies to the whole
    tree (gitglossary), so `:!vendor/` alone would still diff every bound path through the filter (verified
    row b). A second `attr:` word in one element is fatal in git ("Only one 'attr:' specification is
    allowed", row b3), so an existing `attr:SPEC` becomes `attr:REQ SPEC` (git ANDs them), and an exclude
    word is never combined with an attr: word (`:(exclude,attr:filter)` still runs the filter, row b).
    """
    out: list[str] = []
    positive = False
    for element in paths:
        _kind, words, rest = _magic(element)
        if "exclude" in words:
            out.append(element)
            continue
        positive = True
        for requirement in ("!filter", "-filter"):
            attr = "attr:" + requirement
            others: list[str] = []
            merged = False
            for word in words:
                if word.startswith("attr:") and not merged:
                    attr = f"attr:{requirement} {word[len('attr:'):]}"
                    merged = True
                else:
                    others.append(word)
            out.append(":(" + ",".join([attr, *others]) + ")" + rest)
    if not positive:
        out = [*BARE_NARROWING, *out]
    return out


BARE_NARROWING: tuple[str, str] = (":(attr:!filter)", ":(attr:-filter)")


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
    if call.subcommand in ("show", "log"):
        # CR-02 / CR-51: a configured `format.pretty` or `pretty.*` alias holding `%G?` would verify each
        # signed commit through gpg.program; the built-in name wins over any alias (the prefix's
        # gpg.*.program=false is the second layer).
        argv.append("--pretty=medium")
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


# ---------------------------------------------------------------------------------------------- tool

# Pitfall 2: asyncio's default StreamReader limit is 64 KiB; one minified line past it would raise in
# readline. 8 MiB holds any line a 4000-line cap could usefully carry.
READ_STREAM_LIMIT = 8 * 1024 * 1024

# D-36: the line appended to a read cut at max_lines, verbatim.
TRUNCATED_LINE = "[git_read: truncated at {max_lines} lines]"

NO_OUTPUT = "(no output)"


async def _reap(proc: asyncio.subprocess.Process) -> None:
    """Kill a still-running child (and its session on POSIX) and reap it, on EVERY exit path.

    Same shape as bash_tool's `_kill_and_reap`: kill is a synchronous signal, the wait is shielded so a
    cancellation unwinding through a `finally` cannot abandon the reap, and its CancelledError is
    suppressed so the original cancellation keeps propagating.
    """
    if proc.returncode is not None:
        return
    if os.name != "nt":
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError):
        proc.kill()
    with contextlib.suppress(asyncio.CancelledError, ProcessLookupError):
        await asyncio.shield(proc.wait())


async def _drain_stderr(stream: asyncio.StreamReader | None) -> str:
    """Read stderr to EOF (an undrained pipe blocks git, Pitfall 3) and keep its first STDERR_CAP_CHARS."""
    if stream is None:
        return ""
    kept = bytearray()
    keep_bytes = STDERR_CAP_CHARS * 4  # a character is at most four utf-8 bytes
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            break
        if len(kept) < keep_bytes:
            kept.extend(chunk[: keep_bytes - len(kept)])
    return bytes(kept).decode("utf-8", errors="replace")[:STDERR_CAP_CHARS]


# D-14 / D-45: the precheck pattern and D-16: the one trailing line a narrowed diff ends with, verbatim.
FILTER_KEY_PATTERN = r"^filter\..*\.(clean|process)$"
FILTER_OMITTED_LINE = (
    "[git_read: a filter driver is configured ({key!r}); paths it binds were omitted"
    " — cached: true or a two-revision diff includes them]"
)


class GitReadPrecheckError(Exception):
    """The filter precheck could not answer; str() is the failure text and no read is spawned."""


def _filter_driver_key(root: Path) -> str | None:
    """D-14 / D-45: ask git whether a clean or process filter driver is configured.

    One `config --name-only --get-regexp` through the one prefix, environment and cwd, so includes,
    `includeIf` and `config.worktree` are seen and nothing is cached. Exit 1 means none (None); exit 0
    returns the first key, `fsdecode`d (a subsection may hold spaces, never a newline; the driver's value
    is never read). Any other exit, a signal, a timeout or an OSError fails closed.
    """
    try:
        proc = _run_git(root, ["config", "--name-only", "--get-regexp", FILTER_KEY_PATTERN])
    except subprocess.TimeoutExpired:
        raise GitReadPrecheckError(
            f"git_read timed out after {GIT_PIN_TIMEOUT_SECONDS}s checking for filter drivers"
        ) from None
    except OSError as exc:
        raise GitReadPrecheckError(
            f"git could not be started: {exc.strerror or exc.__class__.__name__}"
        ) from None
    if proc.returncode == 1:
        return None
    if proc.returncode == 0:
        for raw in proc.stdout.split(b"\n"):
            if raw:
                return os.fsdecode(raw)
        raise GitReadPrecheckError("git exited with 0 but named no filter driver")  # unreachable; fail closed
    stderr = proc.stderr.decode("utf-8", errors="replace")[:STDERR_CAP_CHARS]
    raise GitReadPrecheckError(f"git exited with {proc.returncode}: {stderr}")


def _is_two_revision(root: Path, revision: str) -> bool:
    """D-34: ask git, never parse the revision. Two or more `--revs-only` lines and exit 0 is a
    two-revision diff (blob against blob, no filter); anything else answers False, the safe direction."""
    try:
        proc = _run_git(root, ["rev-parse", "--revs-only", revision, "--"])
    except (subprocess.SubprocessError, OSError):
        return False
    if proc.returncode != 0:
        return False
    return sum(1 for line in proc.stdout.split(b"\n") if line) >= 2


class GitReadTool(Tool):
    """Read-only `git diff | show | log` of the workspace repository (D-01, D-17, D-64).

    Constructing it runs nothing. `timeout_s` is 60 so the inner read timeout (30s) and the pins (5s
    each) finish, and kill and reap their git, before `Tool.run`'s outer wait_for fires (Pitfall 1).
    """

    timeout_s: float = 60.0

    def __init__(self, root: Path) -> None:
        self._root = Path(os.path.realpath(root))

    def info(self) -> ToolSchema:
        # `ingest` is explicit: the default "untrusted" plus the root capability floor would deny the tool.
        return ToolSchema(
            name="git_read",
            group="fs.read",
            gate_family="allow", ingest="none", host="safe", result_origin="trusted",
            description=(
                "Read the git history of the workspace: diff, show or log. Read-only; it runs no "
                "repository-configured command. Paths are relative to the workspace root."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "subcommand": {
                        "type": "string",
                        "enum": list(SUBCOMMANDS),
                        "description": "Which read to run: diff, show or log.",
                    },
                    "revision": {
                        "type": "string",
                        "description": "A commit, branch, tag or range such as A..B or A...B.",
                    },
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Paths relative to the workspace root; pathspec magic such as "
                        ":(glob) is allowed.",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Stop after this many output lines.",
                        "default": DEFAULT_MAX_LINES,
                        "minimum": 1,
                        "maximum": MAX_LINES_LIMIT,
                    },
                    "stat": {
                        "type": "boolean",
                        "description": "A summary of changed files instead of the patch.",
                        "default": False,
                    },
                    "cached": {
                        "type": "boolean",
                        "description": "diff only: the index against the revision or HEAD.",
                        "default": False,
                    },
                },
                "required": ["subcommand"],
            },
            destructive=False,
            estimated_tokens=800,
        )

    async def _execute(
        self,
        subcommand: str,
        revision: str | None = None,
        paths: list[str] | None = None,
        max_lines: int | None = None,
        stat: bool = False,
        cached: bool = False,
    ) -> ToolResult:
        # D-63: validate before any process exists.
        try:
            call = _validate_call(self._root, subcommand, revision, paths, max_lines, stat, cached)
        except GitReadValidationError as exc:
            return self.err(str(exc), error_type="validation_error")
        # D-11: the layout may have changed since registration; re-run the one pin function every call.
        failure = await asyncio.to_thread(git_read_pin_failure, self._root)
        if failure is not None:
            return self.err(failure.message(per_call=True), error_type="permission_denied")
        pathspecs: list[str] = list(call.paths)
        trailing: str | None = None
        if call.subcommand == "diff" and not call.cached:
            # D-14: a worktree diff canonicalises files through clean/process filters; ask on every call.
            try:
                key = await asyncio.to_thread(_filter_driver_key, self._root)
            except GitReadPrecheckError as exc:
                return self.err(str(exc), error_type="execution_error")
            if key is not None:
                two = call.revision is not None and await asyncio.to_thread(
                    _is_two_revision, self._root, call.revision
                )
                if not two:
                    pathspecs = _narrow_for_filters(pathspecs)
                    trailing = FILTER_OMITTED_LINE.format(key=key)
        return await self._stream(_read_argv(call, pathspecs), call.max_lines, trailing=trailing)

    async def _stream(self, argv: list[str], max_lines: int, trailing: str | None = None) -> ToolResult:
        """D-06 / D-36 / D-60 / D-61: run `argv`, stop at `max_lines`, enforce the timeout, reap always.

        `trailing` is appended after the (possibly truncated) output; the clean-filter precheck passes it.
        """
        timeout = GIT_READ_TIMEOUT_SECONDS  # read at call time so a test can patch it
        extra: dict[str, bool] = {} if os.name == "nt" else {"start_new_session": True}
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(self._root),
                env=_git_env(self._root),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=READ_STREAM_LIMIT,
                **extra,
            )
        except OSError as exc:
            return self.err(f"git could not be started: {exc.strerror or exc.__class__.__name__}")
        stderr_task = asyncio.ensure_future(_drain_stderr(proc.stderr))

        async def collect() -> tuple[bytes, bool, str]:
            assert proc.stdout is not None
            lines: list[bytes] = []
            truncated = False
            try:
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    if len(lines) >= max_lines:
                        truncated = True
                        break
                    lines.append(line)
            except ValueError:
                # One line past READ_STREAM_LIMIT: stop here, as a truncation.
                truncated = True
            if truncated:
                await _reap(proc)
                return b"".join(lines), True, ""
            await proc.wait()
            return b"".join(lines), False, await stderr_task

        try:
            body, truncated, stderr = await asyncio.wait_for(collect(), timeout)
        except asyncio.TimeoutError:
            return self.err(f"git_read timed out after {GIT_READ_TIMEOUT_SECONDS}s", error_type="timeout_error")
        finally:
            await _reap(proc)
            if not stderr_task.done():
                stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await stderr_task

        if not truncated and proc.returncode != 0:
            return self.err(f"git exited with {proc.returncode}: {stderr}")
        text = body.decode("utf-8", errors="replace")
        if not text:
            text = NO_OUTPUT
        if truncated:
            if not text.endswith("\n"):
                text += "\n"
            text += TRUNCATED_LINE.format(max_lines=max_lines)
        if trailing:
            text += ("" if text.endswith("\n") else "\n") + trailing
        return self.ok(text, truncated=truncated)
