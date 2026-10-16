"""The systemd side of the harness: what a stopped user unit says about the llama-server it runs (the
init import), and the harness's own autostart unit (`localharness-llm.service`).

Two injectable subprocess seams, nothing else here spawns a process. Init's import senses through
`_run` (the program found by `which_outside_cwd`, a list argv, no shell, stdin closed, a timeout, an
output cap), and `_run` answers None for anything that is not a clean, bounded answer, which every
caller reads as "not checked"; the import never starts, stops or edits a unit. The autostart unit is
driven through `_run_manager`: the same shape, but the program is found in `MANAGER_DIRS` only (never
PATH or the current directory), its environment supplies `XDG_RUNTIME_DIR`, and it keeps the exit code.
The verbs the harness issues are the closed set `VERB_TIMEOUTS`. A test replaces `_run` and
`_run_manager`; conftest stubs both for every test.

Two sources yield the record a running process would (`procfs.ProcRecord`, with `pid=0` and
`environ`/`cwd` possibly unknown): the last llama-server run in the unit's journal, queried by the
trusted `_SYSTEMD_USER_UNIT=` and `_UID=` fields only (a forged `USER_UNIT=` entry cannot match),
and the unit file itself. Both are unit-file-grade evidence, never proof. A source that cannot be
used yields a `Skip` whose reason is a predicate on the unit ("runs run-active.sh"); init folds it
into its one no-binary line. Nothing is ever approximated: a unit file with a `$`, an unknown `%`
specifier, an `EnvironmentFile=`, drop-ins, a shell prefix or a cap hit is skipped, not guessed at.

Pure standard library. `pwd` is imported inside the function that needs it, so the module still
imports where it is absent.
"""
from __future__ import annotations

import dataclasses
import errno
import json
import os
import re
import shlex
import stat
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from localharness import resolved_version
from localharness.core.private_files import (
    atomic_write_text, open_private_dir, open_regular, read_private_text,
)
from localharness.core.which import which_outside_cwd
from localharness.provider.procfs import HARNESS_UNIT, LaunchIdentity, ProcRecord, port_of
from localharness.provider.server import server_dir, strip_url_credentials

__all__ = [
    "JOURNAL_LINES", "JOURNAL_TIMEOUT_S", "JOURNAL_OUTPUT_CAP", "SYSTEMCTL_TIMEOUT_S",
    "SYSTEMCTL_OUTPUT_CAP", "UNIT_FILE_CAP", "UNIT_LINE_CAP", "UNIT_TOKEN_CAP", "UNITS_SCANNED_CAP",
    "Skip", "UnitState", "unit_file_state", "user_units", "read_journal",
    "read_unit", "unit_record", "journal_environment_gap",
    "BOOT_MARKER_NAME", "Marker", "UnitRuntimeState", "remove_marker",
    "TARGET_NAME", "TARGET_NAME_PATTERN", "TARGET_CAP", "MARKER_CAP", "MARKER_CAUSES",
    "MARKER_SHAPES", "MARKER_DROPS", "write_target", "read_target", "remove_target", "read_marker",
    "write_marker", "PRIMARY_TARGET", "TIMEOUT_START_SECONDS", "UNIT_VALUE_PATTERN", "UNIT_HEADER_PREFIX",
    "UNIT_VERSION_PATTERN", "CAPTURED_NAMES", "CapturedShell", "harness_unit_path", "harness_home",
    "unit_config_dir", "render_harness_unit", "captured_env", "capture_shell_env", "unit_serves",
    "read_harness_unit", "write_harness_unit", "remove_harness_unit", "harness_unit_failure",
    "is_harness_unit", "MANAGER_DIRS", "RUNTIME_DIR_ROOT", "systemctl_path", "journalctl_path",
    "manager_env", "ManagerResult", "unit_runtime_state", "own_unit", "harness_unit_state",
    "VERB_TIMEOUTS", "systemctl_user", "JOURNAL_TAIL_LINES", "JOURNAL_REF_LINE", "journal_tail",
    "LINGER_DIR", "current_user", "linger_on", "Launcher", "other_launchers", "STOPPED_WITHHELD_LINE",
    "STOPPED_LIVE_PIDFILE_LINE", "STOPPED_START_LIMIT_LINE", "translate_exit",
]

JOURNAL_LINES = 200
JOURNAL_TIMEOUT_S = 5
JOURNAL_OUTPUT_CAP = 4 << 20
SYSTEMCTL_TIMEOUT_S = 5
SYSTEMCTL_OUTPUT_CAP = 1 << 20
UNIT_FILE_CAP = 64 << 10
UNIT_LINE_CAP = 16 << 10
UNIT_TOKEN_CAP = 512
UNITS_SCANNED_CAP = 64

BOOT_MARKER_NAME = "boot-attempt"
TARGET_NAME = "unit-target"
PRIMARY_TARGET = "server:"
"""The reserved `unit-target` token for the primary record (spec 20 §8.1's `server:` rung). The provider
layer cannot compute the primary record's name (it depends on `provider.default_model`), so the target
file names the primary with this token and `resolve_serving` maps it back. It fullmatches
`TARGET_NAME_PATTERN`."""
TIMEOUT_START_SECONDS = 1800
"""`TimeoutStartSec=` of the harness unit and the window a boot marker stays current (spec 20 §9.2)."""
UNIT_VALUE_PATTERN = re.compile(r"[A-Za-z0-9/._+@:=-]+")
"""Every path and value on the unit's `ExecStart` / `WorkingDirectory` / `# env` lines fullmatches this."""
CAPTURED_NAMES = ("PATH", "HF_HOME", "LD_LIBRARY_PATH")
"""The only environment names the unit captures from the shell, in the order the body carries them."""
UNIT_HEADER_PREFIX = "# written by localharness "
UNIT_VERSION_PATTERN = re.compile(r"[0-9A-Za-z.+-]{1,64}")
TARGET_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@+:-]{0,127}")
TARGET_CAP = 256
MARKER_CAP = 4096
MARKER_CAUSES = frozenset({"death", "remembered", "none"})
MARKER_SHAPES = frozenset({"full", "retry"})
MARKER_DROPS = frozenset({"draft-mtp", "parsers"})

_LLAMA_SERVER = "llama-server"
_SHOW_FIELDS = "Id,LoadState,FragmentPath,UnitFileState,DropInPaths"


@dataclass(frozen=True)
class Skip:
    """A unit source that cannot be used. `reason` is a predicate on the unit (the init line reads
    `<unit> <reason>; no llama-server found — set server.binary`); `names_model` is True when
    the unit plainly runs a llama server whose model could be named, False when it is an unrelated
    unit or has nothing to say."""

    reason: str
    names_model: bool = True


@dataclass(frozen=True)
class Marker:
    """The boot marker `serve` writes at `<server dir>/boot-attempt` before a launch it did not
    confirm (spec 20 §9.3): which record, which binary and invocation, the cause and shape of the
    attempt, the window it planned, the fit and free memory it measured, what it dropped, and when
    (ISO-8601 UTC). `write_marker` is its one writer, `read_marker` its reader; the confirm removes it."""

    record: str
    binary: str
    invocation: str | None
    cause: str
    shape: str
    planned_context: int | None
    full_fit: int | None
    free_bytes: int | None
    dropped: tuple[str, ...]
    at: str


@dataclass(frozen=True)
class UnitRuntimeState:
    """What the service manager says about the harness unit: the live `InvocationID` (None when the
    unit has none) and `NRestarts`, which the confirm reads to tell one boot attempt from the next
    (spec 20 §6.0), and, filled from the manager's `show` by the autostart phase, the `MainPID`, the
    states, the `FragmentPath` and `DropInPaths` the identity test reads, the last exit status and
    result, and `TimeoutStartSec`. Every added field is defaulted, so the two-field form stays valid.
    `own` is True once the harness's-own-unit test passed for this state."""

    invocation_id: str | None
    n_restarts: int
    main_pid: int | None = None
    active_state: str = ""
    unit_file_state: str = ""
    fragment_path: str = ""
    drop_in_paths: str = ""
    exec_main_status: int | None = None
    result: str = ""
    timeout_start_seconds: int = TIMEOUT_START_SECONDS
    own: bool = False


@dataclass(frozen=True)
class UnitState:
    """`systemctl --user show` for one unit; every field is the raw string (empty when unset)."""

    unit: str
    load_state: str
    fragment_path: str
    unit_file_state: str
    drop_in_paths: str


def _run(argv: list[str], *, timeout: float, cap: int) -> str | None:
    """The one subprocess seam: `argv[0]` resolved outside the current directory, a list argv, no
    shell, stdin closed. The stdout decoded as UTF-8, or None when the program is missing, the call
    times out or cannot start, the exit code is not 0, or the output is longer than `cap` bytes."""
    exe = which_outside_cwd(argv[0])
    if exe is None:
        return None
    try:
        done = subprocess.run([exe, *argv[1:]], capture_output=True, stdin=subprocess.DEVNULL,
                              timeout=timeout, check=False)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0 or len(done.stdout) > cap:
        return None
    return done.stdout.decode("utf-8", errors="replace")


def _show_argv(*units: str) -> list[str]:
    return ["systemctl", "--user", "show", "-p", _SHOW_FIELDS, "--", *units]


def _parse_state(block: str) -> UnitState | None:
    """One `Key=value` block of `systemctl show`, split on the first `=`; None without a LoadState."""
    fields: dict[str, str] = {}
    for line in block.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields.setdefault(key, value)
    if "LoadState" not in fields:
        return None
    return UnitState(fields.get("Id", ""), fields["LoadState"], fields.get("FragmentPath", ""),
                     fields.get("UnitFileState", ""), fields.get("DropInPaths", ""))


def unit_file_state(unit: str) -> UnitState | None:
    """What systemd says about `unit` (load state, unit file, enablement, drop-ins), or None when it
    could not be asked ("not checked"). A missing unit is an answer: `load_state == "not-found"`."""
    out = _run(_show_argv(unit), timeout=SYSTEMCTL_TIMEOUT_S, cap=SYSTEMCTL_OUTPUT_CAP)
    if out is None:
        return None
    state = _parse_state(out)
    if state is None:
        return None
    return state if state.unit else UnitState(unit, state.load_state, state.fragment_path,
                                               state.unit_file_state, state.drop_in_paths)


_SKIPPED_FILE_STATES = frozenset({"generated", "transient", "alias", "masked", "bad"})


def user_units() -> list[UnitState]:
    """The user's own service units, stopped and disabled ones included (`list-units` shows only
    loaded ones, `list-unit-files` all of them): names holding `@` (templates) and units generated,
    transient, alias, masked or bad are dropped, the first `UNITS_SCANNED_CAP` in alphabetical order
    are kept, and one batched `systemctl show` reads them. Empty when systemd could not be asked."""
    out = _run(["systemctl", "--user", "list-unit-files", "--type=service", "--plain",
                "--no-legend", "--no-pager"], timeout=SYSTEMCTL_TIMEOUT_S, cap=SYSTEMCTL_OUTPUT_CAP)
    if out is None:
        return []
    names: set[str] = set()
    for row in out.splitlines():
        cols = row.split()
        if not cols or "@" in cols[0]:
            continue
        if len(cols) > 1 and cols[1] in _SKIPPED_FILE_STATES:
            continue
        names.add(cols[0])
    kept = sorted(names)[:UNITS_SCANNED_CAP]
    if not kept:
        return []
    shown = _run(_show_argv(*kept), timeout=SYSTEMCTL_TIMEOUT_S, cap=SYSTEMCTL_OUTPUT_CAP)
    if shown is None:
        return []
    by_name: dict[str, UnitState] = {}
    for block in shown.replace("\r\n", "\n").split("\n\n"):
        state = _parse_state(block)
        if state is not None and state.unit:
            by_name.setdefault(state.unit, state)
    return [by_name[name] for name in kept if name in by_name]


# ---------------------------------------------------------------------------------- the journal


def _option_value(argv: list[str], short: str, long: str) -> str | None:
    """The value of `-m X` / `--model X` / `--model=X` (the last one wins, as llama-server's own
    parser lets it); None when absent or valueless."""
    found: str | None = None
    for i, tok in enumerate(argv):
        if tok in (short, long):
            found = argv[i + 1] if i + 1 < len(argv) else None
        elif tok.startswith(long + "="):
            found = tok[len(long) + 1:]
    return found


def read_journal(unit: str) -> ProcRecord | Skip:
    """The newest llama-server run of the user's `unit` as the record a running process gives, with
    `pid=0`, `environ=None` and `cwd=None` (the journal knows neither). Queried by the trusted
    `_SYSTEMD_USER_UNIT=` and `_UID=` fields, never `-u`, which matches the forgeable `USER_UNIT`.

    `_CMDLINE` is `/proc/<pid>/cmdline` with the NULs turned into spaces: journald does not quote it, so
    it is split on whitespace and read exactly only when it holds no quote or backslash (an argument with
    a space in it cannot be told from two). The NEWEST run decides: one the journal cannot show (a
    `_CMDLINE` journald nulled past 4096 bytes or kept as a byte array, a quote or a backslash in the
    line) is a `Skip`, never a reason to import the run before it, which may have bound otherwise."""
    uid = os.getuid()
    out = _run(["journalctl", "--user", "-o", "json", "-n", str(JOURNAL_LINES), "--all",
                "--output-fields=_EXE,_CMDLINE", "--no-pager", f"_SYSTEMD_USER_UNIT={unit}",
                f"_UID={uid}"], timeout=JOURNAL_TIMEOUT_S, cap=JOURNAL_OUTPUT_CAP)
    if out is None:
        return Skip("has an unreadable journal", names_model=False)
    newest: tuple[str, str] | None = None
    for line in reversed(out.splitlines()):
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict):
            continue
        exe, cmdline = entry.get("_EXE"), entry.get("_CMDLINE")
        if not isinstance(exe, str):
            continue  # an absent or byte-array _EXE names no program
        if not isinstance(cmdline, str):
            if os.path.basename(exe) == _LLAMA_SERVER:  # null past 4096 bytes, or a byte array
                return Skip("has a command line the journal cannot show")
            continue
        words = cmdline.split()
        if os.path.basename(exe) == _LLAMA_SERVER or (
                words and os.path.basename(words[0]) == _LLAMA_SERVER):
            newest = (exe, cmdline)
            break
    if newest is None:
        return Skip("has no llama-server in its journal", names_model=False)
    exe, cmdline = newest
    if any(ch in cmdline for ch in "\"'\\"):
        return Skip("has an argument the journal cannot show exactly")
    argv = cmdline.split()
    if not os.path.isabs(exe):
        return Skip(f"names a relative path ({exe})")
    model = _option_value(argv, "-m", "--model")
    if model is not None and not os.path.isabs(model):
        return Skip(f"names a relative path ({model})")
    if model is None or not _is_regular_file(model):
        return Skip("names no regular -m file")
    return ProcRecord(pid=0, uid=uid, argv=tuple(argv), exe=exe, cwd=None, start_time=0,
                      unit=unit, environ=None, mapped_ggufs=())


def _is_regular_file(path: str) -> bool:
    """`os.stat` says regular file; the file is never opened (a FIFO would block)."""
    try:
        return stat.S_ISREG(os.stat(path).st_mode)
    except OSError:
        return False


# ------------------------------------------------------------------------------- the unit file

_EXEC_PREFIX = "@-:+!|"
"""The characters a systemd `ExecStart=` may start with, plus `|`. `|` is NOT a systemd prefix: it is kept
on purpose (D-53) so that a command that starts with a pipe is refused as "runs through a shell" instead
of being read as a program called `|...`; for any real unit file that skip is unreachable."""


def _names_model(tokens: list[str]) -> bool:
    """A command line that plainly names a model: `-m`/`--model` or a `.gguf` token."""
    return any(t in ("-m", "--model") or t.startswith("--model=") or t.endswith(".gguf")
               for t in tokens)


def _expand(token: str) -> str | None:
    """`%%`, `%h` and `%u` in one word, as systemd expands them; None on any other specifier."""
    if "%" not in token:
        return token
    out: list[str] = []
    i = 0
    while i < len(token):
        ch = token[i]
        if ch != "%":
            out.append(ch)
            i += 1
            continue
        spec = token[i + 1:i + 2]
        if spec == "%":
            out.append("%")
        elif spec == "h":
            out.append(os.path.expanduser("~"))
        elif spec == "u":
            try:
                import pwd  # function-local: the module still imports where pwd is absent (D-71)
                out.append(pwd.getpwuid(os.getuid()).pw_name)
            except (ImportError, KeyError):
                return None
        else:
            return None
        i += 2
    return "".join(out)


def _bad_specifier(token: str) -> str:
    """The reason text for a word `_expand` refused."""
    spec = ""
    i = token.find("%")
    while i != -1:
        nxt = token[i + 1:i + 2]
        if nxt in ("%", "h", "u"):
            i = token.find("%", i + 2)
            continue
        spec = nxt
        break
    return f"uses the %{spec} specifier" if spec else "uses a stray % specifier"


def _logical_lines(text: str) -> list[str] | None:
    """The unit file's lines with each `\\`-continuation joined to its successor (the backslash and
    the newline become one space). None when a logical line is longer than `UNIT_LINE_CAP`."""
    out: list[str] = []
    buf = ""
    for raw in text.splitlines():
        if not buf and raw.lstrip().startswith(("#", ";")):
            continue
        if raw.endswith("\\"):
            buf += raw[:-1] + " "
        else:
            buf += raw
            out.append(buf)
            buf = ""
        if len(buf.encode()) > UNIT_LINE_CAP:
            return None
    if buf:
        out.append(buf)
    return None if any(len(line.encode()) > UNIT_LINE_CAP for line in out) else out


def read_unit(path: str, *, unit: str) -> ProcRecord | Skip:
    """The `ExecStart=` of a user unit file as the record a running process gives (`pid=0`), or the
    `Skip` that says why it cannot be trusted. The file is opened through `open_regular` (a FIFO
    named `*.service` cannot hang it) under `UNIT_FILE_CAP`; only `[Service]` is read, and only
    `ExecStart=`, `Environment=`, `WorkingDirectory=` and `EnvironmentFile=` of it. A `$`, a
    specifier other than `%h`, `%u`, `%%`, an `EnvironmentFile=`, a backslash escape, a shell
    prefix, a wrapper program or any cap hit skips the unit: a truncated or guessed token list could
    keep `--host 0.0.0.0` and lose `--api-key-file`."""
    try:
        with os.fdopen(open_regular(path), "rb") as fh:
            raw = fh.read(UNIT_FILE_CAP + 1)
    except OSError:
        return Skip("has an unreadable unit file", names_model=False)
    if len(raw) > UNIT_FILE_CAP:
        return Skip(f"has a unit file over {UNIT_FILE_CAP} bytes", names_model=False)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return Skip("has an unreadable unit file", names_model=False)
    lines = _logical_lines(text)
    if lines is None:
        return Skip(f"has a line over {UNIT_LINE_CAP} bytes", names_model=False)

    section = ""
    exec_starts: list[str] = []
    environments: list[str] = []
    working_dir = ""
    env_file = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped
            continue
        if section != "[Service]":
            continue
        key, sep, value = stripped.partition("=")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if key == "ExecStart":
            if value:
                exec_starts.append(value)
            else:
                exec_starts.clear()
        elif key == "Environment":
            if value:
                environments.append(value)
            else:
                environments.clear()
        elif key == "WorkingDirectory":
            working_dir = value
        elif key == "EnvironmentFile":
            env_file = True

    names = any(_names_model(_words(v)) or _LLAMA_SERVER in v for v in exec_starts)
    if env_file:
        return Skip("uses EnvironmentFile=", names_model=names)
    if len(exec_starts) != 1:
        return Skip("has no single ExecStart=", names_model=names)
    command = exec_starts[0]
    prefix = ""
    while command[len(prefix):len(prefix) + 1] and command[len(prefix)] in _EXEC_PREFIX:
        prefix += command[len(prefix)]
    if "|" in prefix:
        return Skip("runs through a shell", names_model=names)
    try:
        tokens = shlex.split(command[len(prefix):])
    except ValueError:
        return Skip("has an unparseable ExecStart=", names_model=names)
    if len(tokens) > UNIT_TOKEN_CAP:
        return Skip(f"has more than {UNIT_TOKEN_CAP} ExecStart= tokens", names_model=names)
    if not tokens or ("@" in prefix and len(tokens) < 2):
        return Skip("has an empty ExecStart=", names_model=False)
    if ";" in tokens:  # systemd's separator between two commands; the second is not llama-server's argument
        return Skip("has more than one command in ExecStart=", names_model=names)
    # Who runs is judged before anything else, so an unrelated unit reads "runs <program>" and not
    # "uses a $ expansion": with `@` the executable is the first word, else argv[0].
    program = tokens[0]
    base = os.path.basename(program)
    if base != _LLAMA_SERVER:
        return Skip(f"runs {base}", names_model=_names_model(tokens))
    for value in (command, *environments, working_dir):
        if "$" in value:
            return Skip("uses a $ expansion")
    if "\\" in command:
        return Skip("uses a backslash escape in ExecStart=")
    expanded: list[str] = []
    for tok in tokens:
        done = _expand(tok)
        if done is None:
            return Skip(_bad_specifier(tok))
        expanded.append(done)
    if os.path.dirname(expanded[0]) and not os.path.isabs(expanded[0]):
        return Skip("runs a relative path")
    if "@" in prefix:
        exe: str | None = expanded[0]
        argv = expanded[1:]
    else:
        exe, argv = None, expanded

    pairs: dict[str, str] = {}
    for value in environments:
        try:
            words = shlex.split(value)
        except ValueError:
            return Skip("has an unparseable Environment=", names_model=True)
        for word in words:
            done = _expand(word)
            if done is None:
                return Skip(_bad_specifier(word))
            name, eq, val = done.partition("=")
            if eq and name:
                pairs[name] = val
    cwd_raw = working_dir.lstrip("-") if working_dir else "~"
    cwd = _expand(cwd_raw) if cwd_raw != "~" else os.path.expanduser("~")
    if cwd is None:
        return Skip(_bad_specifier(cwd_raw))
    if not os.path.isabs(cwd):
        return Skip(f"names a relative WorkingDirectory ({cwd})")
    return ProcRecord(pid=0, uid=os.getuid(), argv=tuple(argv), exe=exe, cwd=cwd, start_time=0,
                      unit=unit, environ=MappingProxyType(pairs), mapped_ggufs=())


def _words(value: str) -> list[str]:
    """`shlex.split`, or the whitespace split of a value shlex refuses (only used to look for a
    model token in a unit that is about to be skipped anyway)."""
    try:
        return shlex.split(value)
    except ValueError:
        return value.split()


# ---------------------------------------------------------------------------- per-unit sources


def _unit_file_record(state: UnitState) -> ProcRecord | Skip:
    """The unit file as a record, or why it cannot be one: no unit file, drop-ins (a drop-in can replace
    `ExecStart=` and `Environment=`), or `read_unit`'s own skip."""
    if not state.fragment_path:
        second: ProcRecord | Skip = Skip("has no unit file", names_model=False)
    else:
        second = read_unit(state.fragment_path, unit=state.unit)
    if state.drop_in_paths:
        names = isinstance(second, ProcRecord) or second.names_model
        second = Skip(f"has drop-ins ({state.drop_in_paths})", names_model=names)
    return second


def _journal_with_unit_file(journal: ProcRecord, state: UnitState) -> ProcRecord:
    """The journal knows neither the unit's `Environment=` nor its `WorkingDirectory=`, and cannot tell
    an argument with a space in it from two (WR-04, WR-02). When the unit file is readable and names the
    SAME model, its `environ` and `cwd` complete the journal's record, and its argv replaces the journal's
    when the two spell the same words (then the unit file's tokens are the exactly-quoted ones). A unit
    file that names another model, or cannot be read, leaves the journal's record as it was."""
    unit_rec = _unit_file_record(state)
    if not isinstance(unit_rec, ProcRecord):
        return journal
    if _option_value(list(unit_rec.argv), "-m", "--model") != _option_value(list(journal.argv), "-m", "--model"):
        return journal
    same_words = " ".join(unit_rec.argv[1:]) == " ".join(journal.argv[1:])  # argv[0] is spelt per source
    return replace(journal, environ=unit_rec.environ, cwd=unit_rec.cwd,
                   argv=(journal.argv[:1] + unit_rec.argv[1:]) if same_words else journal.argv)


def journal_environment_gap(state: UnitState, record: ProcRecord) -> str | None:
    """Why the environment of a journal-sourced unit was not read, when it plausibly exists: the unit
    has a unit file and it could not be turned into a record (an `EnvironmentFile=`, a `$` or `%`
    expansion, drop-ins) or it names another model. None when `record` carries the unit's environment,
    or the unit has no unit file to read. `init` shows it, so nothing is imported silently without the
    unit's `LLAMA_ARG_*` and key."""
    if record.environ is not None or not state.fragment_path:
        return None
    unit_rec = _unit_file_record(state)
    if isinstance(unit_rec, Skip):
        return unit_rec.reason if unit_rec.names_model else None  # an unreadable or unrelated file: no claim
    return "names another model"


def unit_record(state: UnitState) -> tuple[str, ProcRecord] | Skip:
    """The first usable source for a stopped unit: ("journal", record), else ("unit", record), else
    the `Skip` init folds into its one no-binary line. The journal is tried first (a journal record is
    completed from a readable unit file that names the same model, see `_journal_with_unit_file`); the
    unit file only when the unit has no drop-ins (a drop-in can replace `ExecStart=`). When both skip,
    the first skip in source order whose unit names a model wins; otherwise the unit file's."""
    first = read_journal(state.unit)
    if isinstance(first, ProcRecord):
        return "journal", _journal_with_unit_file(first, state)
    second = _unit_file_record(state)
    if isinstance(second, ProcRecord):
        return "unit", second
    return first if first.names_model else second


def _open_server_dir(config_dir, *, create: bool) -> int:
    """A descriptor for `<config dir>/vllm`, opened `O_DIRECTORY | O_NOFOLLOW` (see
    `private_files.open_private_dir`). The directory sits where agent-run code can write, so it is
    never followed through a link; OSError (ELOOP, ENOTDIR, ENOENT, ...) propagates."""
    return open_private_dir(server_dir(config_dir), create=create)


def _unlink_in_server_dir(config_dir, name: str) -> bool:
    """Unlink `name` inside the server directory by descriptor; True when a file was removed, False
    when there was none or the directory is missing, not a directory or a link. Any other OSError
    propagates."""
    return _unlink_in_dir(server_dir(config_dir), name)


def _unlink_in_dir(directory, name: str) -> bool:
    """`_unlink_in_server_dir` over any directory path: opened `O_DIRECTORY | O_NOFOLLOW`, `name`
    unlinked by that descriptor, so a link planted at `name` is removed itself and never what it
    points at."""
    try:
        fd = open_private_dir(directory, create=False)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR, errno.ENOENT):
            return False
        raise
    try:
        os.unlink(name, dir_fd=fd)
    except FileNotFoundError:
        return False
    finally:
        os.close(fd)
    return True


def remove_marker(config_dir) -> bool:
    """Unlink `<server dir>/boot-attempt`; True when a file was removed, False when there was none.

    The server directory sits where agent-run code can write, so it is opened `O_DIRECTORY |
    O_NOFOLLOW` and the file unlinked by that descriptor: a symlinked or missing directory is never
    followed (False), and a link planted in the file's place is unlinked itself, never what it
    points at. Any other OSError propagates. Its only production caller is
    `lifecycle.confirm_launch`."""
    return _unlink_in_server_dir(config_dir, BOOT_MARKER_NAME)


# ------------------------------------------------------------------------------ the unit target


def write_target(config_dir, name: str) -> None:
    """Record `name` (a launch-record name) as the unit's target in `<server dir>/unit-target`.

    ValueError when `name` is not one record name (`TARGET_NAME_PATTERN`). The file is written
    through `atomic_write_text(dir_fd=)` inside the no-follow server directory, created if missing:
    a symlinked directory is refused with OSError, and a link planted at `unit-target` is replaced,
    never followed. Its only caller is `UnitStrategy.activate` (RULING W3); the structural test
    pins that."""
    if not isinstance(name, str) or not TARGET_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"not a record name: {name!r}")
    fd = _open_server_dir(config_dir, create=True)
    try:
        atomic_write_text(TARGET_NAME, name + "\n", dir_fd=fd)
    finally:
        os.close(fd)


def read_target(config_dir) -> str | None:
    """The record name in `<server dir>/unit-target`, or None: no file, no directory, a link, a FIFO,
    more than `TARGET_CAP` bytes, bytes that are not UTF-8, or a body that is not one record name.
    The ladder reads it (and still resolves it against fingerprinted records)."""
    try:
        fd = _open_server_dir(config_dir, create=False)
    except OSError:
        return None
    try:
        return read_private_text(TARGET_NAME, dir_fd=fd, cap=TARGET_CAP, pattern=TARGET_NAME_PATTERN)
    finally:
        os.close(fd)


def remove_target(config_dir) -> bool:
    """Unlink `<server dir>/unit-target`; True when a file was removed, False when there was none.
    The same no-follow rules as `remove_marker`: a planted link is unlinked itself."""
    return _unlink_in_server_dir(config_dir, TARGET_NAME)


# ---------------------------------------------------------------------------- the harness unit

_UNSET_ENVIRONMENT = (
    "LOCALHARNESS_DIR", "LOCALHARNESS_HOME", "LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH", "GCONV_PATH",
    "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE",
)
_UNIT_HEADER_TAIL = " — `localharness model --autostart off` removes it"
_UNIT_ENV_PREFIX = "# env "
_UNIT_DESCRIPTION = "LocalHarness model server (serves the preset `localharness model` selected)"


def harness_unit_path() -> Path:
    """`$XDG_CONFIG_HOME/systemd/user/localharness-llm.service`, else
    `~/.config/systemd/user/localharness-llm.service`, read at call time (spec 20 §9.2). An empty or
    relative `XDG_CONFIG_HOME` is ignored, as the XDG specification says."""
    xdg = os.environ.get("XDG_CONFIG_HOME", "")
    base = Path(xdg) if xdg and os.path.isabs(xdg) else Path.home() / ".config"
    return base / "systemd" / "user" / HARNESS_UNIT


def harness_home() -> str:
    """The absolute home directory the unit's `WorkingDirectory=` names."""
    return os.path.expanduser("~")


def unit_config_dir(config_dir) -> str:
    """The global config dir as the unit pins it: `~` expanded, absolute, normalised."""
    return os.path.abspath(os.path.expanduser(str(config_dir)))


def _unit_value(value) -> str:
    """`value` when it is a non-empty string made only of `UNIT_VALUE_PATTERN` characters; else
    ValueError naming it. One space, `$`, `%`, quote or newline in a unit line would change what systemd
    runs, so none is ever escaped: it is refused."""
    if not isinstance(value, str) or not UNIT_VALUE_PATTERN.fullmatch(value):
        raise ValueError(f"not allowed in the unit: {value!r}")
    return value


def _absolute_unit_path(value) -> str:
    value = _unit_value(value)
    if not value.startswith("/"):
        raise ValueError(f"not an absolute path: {value!r}")
    return value


def render_harness_unit(interpreter: str, config_dir, captured_env: Mapping[str, str], *, home: str,
                        version: str | None = None) -> str:
    """The text of the autostart unit, exactly spec 20 §9.2: fixed text plus the interpreter, the pinned
    config dir, the home and the captured `# env` lines. Pure: it reads no environment and no file, so
    the identity test can compare this render to the file byte for byte.

    `captured_env` holds only `CAPTURED_NAMES`; the lines are written in that order, for the names
    present (PATH is expected, the render does not insist: `captured_env()` does). `version` defaults to
    the running version. ValueError when the interpreter, the config dir or the home is not an absolute
    path, when any value (including `version`) is outside `UNIT_VALUE_PATTERN` / `UNIT_VERSION_PATTERN`,
    or when `captured_env` holds a name outside `CAPTURED_NAMES`."""
    interp = _absolute_unit_path(interpreter)
    cfg = _absolute_unit_path(unit_config_dir(config_dir))
    wd = _absolute_unit_path(home)
    shown = resolved_version() if version is None else version
    if not isinstance(shown, str) or not UNIT_VERSION_PATTERN.fullmatch(shown):
        raise ValueError(f"not a version the unit header can carry: {shown!r}")
    unknown = [name for name in captured_env if name not in CAPTURED_NAMES]
    if unknown:
        raise ValueError(f"not a captured name: {unknown[0]!r}")
    lines = [f"{UNIT_HEADER_PREFIX}{shown}{_UNIT_HEADER_TAIL}"]
    for name in CAPTURED_NAMES:
        if name in captured_env:
            lines.append(f"{_UNIT_ENV_PREFIX}{name}={_unit_value(captured_env[name])}")
    lines += [
        "[Unit]",
        f"Description={_UNIT_DESCRIPTION}",
        "After=network.target",
        "StartLimitIntervalSec=5400",
        "StartLimitBurst=3",
        "",
        "[Service]",
        f"WorkingDirectory={wd}",
        f"ExecStart={interp} -I -m localharness serve --config-dir {cfg}",
        f"UnsetEnvironment={' '.join(_UNSET_ENVIRONMENT)}",
        "Restart=on-failure",
        "RestartSec=5",
        "RestartPreventExitStatus=78",
        f"TimeoutStartSec={TIMEOUT_START_SECONDS}",
        "KillSignal=SIGINT",
        "",
        "[Install]",
        "WantedBy=default.target",
        "",
    ]
    return "\n".join(lines)


def _captured_value_ok(name: str, value: str) -> bool:
    """One captured value as `captured_env` accepts it: in the charset, and absolute (every element of
    `PATH` and `LD_LIBRARY_PATH`, the whole of `HF_HOME`)."""
    if not UNIT_VALUE_PATTERN.fullmatch(value):
        return False
    if name == "HF_HOME":
        return value.startswith("/")
    return all(element.startswith("/") for element in value.split(":"))


def captured_env(body: str) -> dict[str, str] | None:
    """The inverse of `render_harness_unit`: the `# env NAME=value` lines between the header line and
    `[Unit]`, in `CAPTURED_NAMES` order, or None.

    Only that block is read (an `# env` line after `[Unit]` is a comment and nothing more). None for: no
    header line, no `[Unit]` line, any line in the block that is not exactly `# env NAME=value`, a name
    outside `CAPTURED_NAMES`, a duplicate, a value outside `UNIT_VALUE_PATTERN`, an empty or relative
    element of `PATH` / `LD_LIBRARY_PATH`, a relative `HF_HOME`, or no `PATH` (the render always writes
    it, so an empty block is unparsable)."""
    if not isinstance(body, str):
        return None
    lines = body.split("\n")
    if not lines[0].startswith(UNIT_HEADER_PREFIX):
        return None
    try:
        end = lines.index("[Unit]", 1)
    except ValueError:
        return None
    found: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.startswith(_UNIT_ENV_PREFIX):
            return None
        name, eq, value = line[len(_UNIT_ENV_PREFIX):].partition("=")
        if not eq or name not in CAPTURED_NAMES or name in found or not _captured_value_ok(name, value):
            return None
        found[name] = value
    if "PATH" not in found:
        return None
    return {name: found[name] for name in CAPTURED_NAMES if name in found}


@dataclass(frozen=True)
class CapturedShell:
    """What `capture_shell_env` took from a shell: the values to render (`env`), each dropped
    `(variable, element)` of `PATH` / `LD_LIBRARY_PATH` that was empty or relative, and each
    `(variable, as written, made absolute)` of a relative `HF_HOME` (the receipt names both)."""

    env: dict[str, str]
    dropped: tuple[tuple[str, str], ...]
    absolutized: tuple[tuple[str, str, str], ...]


def capture_shell_env(environ: Mapping[str, str], cwd: str) -> CapturedShell:
    """The shell's `PATH` (always) and `HF_HOME` / `LD_LIBRARY_PATH` (when set and non-empty) as the unit
    will carry them (spec 20 §9.2). An empty or relative element of `PATH` / `LD_LIBRARY_PATH` would
    resolve against the user's home at boot, so it is dropped and reported with its variable; a relative
    `HF_HOME` is one whole value the shell meant against its working directory, so it is made absolute
    against `cwd` and reported. ValueError, naming the value, for a kept value outside
    `UNIT_VALUE_PATTERN` and for a `PATH` left with no element."""
    env: dict[str, str] = {}
    dropped: list[tuple[str, str]] = []
    absolutized: list[tuple[str, str, str]] = []
    for name in CAPTURED_NAMES:
        raw = environ.get(name)
        if name == "PATH" and raw is None:
            raw = ""  # an unset PATH has no absolute entry, which is refused below
        if raw is None or (raw == "" and name != "PATH"):
            continue
        if name == "HF_HOME":
            value = raw
            if not raw.startswith("/"):
                value = os.path.normpath(os.path.join(cwd, raw))
                absolutized.append((name, raw, value))
            if not value.startswith("/"):
                raise ValueError(f"not an absolute path: {value!r}")
            env[name] = _unit_value(value)
            continue
        kept: list[str] = []
        for element in raw.split(":"):
            if element.startswith("/"):
                kept.append(_unit_value(element))
            else:
                dropped.append((name, element))
        if kept:
            env[name] = ":".join(kept)
        elif name == "PATH":
            raise ValueError("this shell's PATH has no absolute entry")
    return CapturedShell(env, tuple(dropped), tuple(absolutized))


def _exec_start_tokens(body: str) -> list[str] | None:
    """The words of the single `ExecStart=` line of `[Service]`, or None (none, several, unsplittable)."""
    section = ""
    values: list[str] = []
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped
        elif section == "[Service]" and stripped.startswith("ExecStart="):
            values.append(stripped[len("ExecStart="):])
    if len(values) != 1:
        return None
    try:
        return shlex.split(values[0])
    except ValueError:
        return None


def unit_serves(config_dir, body: str | None = None) -> bool:
    """True only when the harness unit file's `ExecStart=` carries `--config-dir` equal to
    `unit_config_dir(config_dir)`. `body` is read from `read_harness_unit()` when None; an absent or
    unreadable file is False."""
    if body is None:
        body = read_harness_unit()
    if body is None:
        return False
    tokens = _exec_start_tokens(body)
    if tokens is None or tokens.count("--config-dir") != 1:
        return False
    at = tokens.index("--config-dir")
    return at + 1 < len(tokens) and tokens[at + 1] == unit_config_dir(config_dir)


def read_harness_unit() -> str | None:
    """The harness unit file's text, or None: no file, no `systemd/user` directory, a link, a FIFO, more
    than `UNIT_FILE_CAP` bytes, or bytes that are not UTF-8. The directory is opened `O_NOFOLLOW` and the
    file by `open_regular(dir_fd=)`, so nothing planted there is followed or blocks."""
    path = harness_unit_path()
    try:
        dir_fd = open_private_dir(path.parent, create=False)
    except OSError:
        return None
    try:
        with os.fdopen(open_regular(path.name, dir_fd=dir_fd), "rb") as fh:
            raw = fh.read(UNIT_FILE_CAP + 1)
    except OSError:
        return None
    finally:
        os.close(dir_fd)
    if len(raw) > UNIT_FILE_CAP:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def write_harness_unit(body: str) -> Path:
    """Write `body` as the harness unit file (0600) and return its path. The `systemd/user` directory is
    created if missing and opened `O_NOFOLLOW` (a symlinked one is refused with OSError, its target
    untouched); the file goes through `atomic_write_text(dir_fd=)`, so a link planted at the name is
    replaced, never followed. OSError propagates."""
    path = harness_unit_path()
    dir_fd = open_private_dir(path.parent, create=True)
    try:
        atomic_write_text(path.name, body, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)
    return path


def remove_harness_unit() -> bool:
    """Unlink the harness unit file by directory descriptor; True when a file was removed, False when
    there was none or the directory is missing, not a directory or a link. A link planted at the name
    is unlinked itself, never what it points at."""
    path = harness_unit_path()
    return _unlink_in_dir(path.parent, path.name)


def harness_unit_failure(state, config_dir, *, body: str | None = None, interpreter: str | None = None,
                         home: str | None = None) -> str | None:
    """None when the unit is the harness's own; else the phrase naming the FIRST test it failed (spec 20
    §9.2, a predicate on the unit: `status` prints `<unit> <phrase>`). In order: the service manager was
    asked, `FragmentPath` is `harness_unit_path()`, `DropInPaths` is empty, and the file is byte-equal to
    `render_harness_unit` for this interpreter, config dir and the file's own `# env` block.

    The header's version is read from the body like the captured values (it is a comment that runs
    nothing): a package upgrade that leaves the template alone keeps the unit the harness's own, while a
    change to any executable line, or to the template, still fails the byte comparison. `body` is read
    from `read_harness_unit()` when None; `interpreter` defaults to `sys.executable`, `home` to
    `harness_home()`."""
    if state is None:
        return "could not be asked: the service manager did not answer"
    expected = str(harness_unit_path())
    fragment = getattr(state, "fragment_path", "")
    if fragment != expected:
        if not fragment:
            return f"is not loaded from {expected} (FragmentPath is empty)"
        return f"is loaded from {fragment}, not {expected} (FragmentPath)"
    drop_ins = getattr(state, "drop_in_paths", "")
    if drop_ins:
        return f"has drop-ins ({drop_ins})"
    if body is None:
        body = read_harness_unit()
    if body is None:
        return "has no readable unit file"
    captured = captured_env(body)
    if captured is None:
        return "has an unparsable # env block"
    version, sep, _ = body.split("\n", 1)[0][len(UNIT_HEADER_PREFIX):].partition(" — ")
    differs = "differs from what localharness writes"
    if not sep or not UNIT_VERSION_PATTERN.fullmatch(version):
        return differs
    try:
        rendered = render_harness_unit(sys.executable if interpreter is None else interpreter, config_dir,
                                       captured, home=harness_home() if home is None else home,
                                       version=version)
    except ValueError:
        return differs
    return None if rendered == body else differs


def is_harness_unit(state, config_dir, *, body: str | None = None, interpreter: str | None = None,
                    home: str | None = None) -> bool:
    """`harness_unit_failure(...) is None`."""
    return harness_unit_failure(state, config_dir, body=body, interpreter=interpreter, home=home) is None


# ------------------------------------------------------------------------------ the boot marker

_MARKER_FIELDS = frozenset(f.name for f in dataclasses.fields(Marker))


def _opt_int(value) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool))


def _marker_from(data) -> Marker | None:
    """A `Marker` from a decoded JSON value, or None when it is off the closed shape of spec 20 §9.3."""
    if not isinstance(data, dict) or set(data) != _MARKER_FIELDS:
        return None
    for key in ("record", "binary", "at"):
        if not isinstance(data[key], str) or not data[key]:
            return None
    if data["invocation"] is not None and not isinstance(data["invocation"], str):
        return None
    if data["cause"] not in MARKER_CAUSES or data["shape"] not in MARKER_SHAPES:
        return None
    if not all(_opt_int(data[k]) for k in ("planned_context", "full_fit", "free_bytes")):
        return None
    dropped = data["dropped"]
    if not isinstance(dropped, list) or not all(isinstance(d, str) and d in MARKER_DROPS for d in dropped):
        return None
    return Marker(record=data["record"], binary=data["binary"], invocation=data["invocation"],
                  cause=data["cause"], shape=data["shape"], planned_context=data["planned_context"],
                  full_fit=data["full_fit"], free_bytes=data["free_bytes"], dropped=tuple(dropped),
                  at=data["at"])


def read_marker(config_dir) -> Marker | None:
    """The boot marker at `<server dir>/boot-attempt`, or None: no file, no directory, a link, a FIFO,
    more than `MARKER_CAP` bytes, invalid JSON, or anything off `Marker`'s closed shape. Read only:
    an unparsable marker is left in place and is replaced by the next marker `serve` writes
    (`write_marker`), because `remove_marker` may be called only inside `lifecycle.confirm_launch` and
    sensing writes nothing (spec 20 §6.0, §9.3)."""
    try:
        fd = _open_server_dir(config_dir, create=False)
    except OSError:
        return None
    try:
        with os.fdopen(open_regular(BOOT_MARKER_NAME, dir_fd=fd), "rb") as fh:
            raw = fh.read(MARKER_CAP + 1)
    except OSError:
        return None
    finally:
        os.close(fd)
    if len(raw) > MARKER_CAP:
        return None
    try:
        return _marker_from(json.loads(raw.decode("utf-8")))
    except ValueError:  # JSONDecodeError and UnicodeDecodeError are both ValueErrors
        return None


def write_marker(config_dir, marker: Marker) -> None:
    """Write `marker` as JSON (all ten keys, `dropped` a list) to `<server dir>/boot-attempt`: the one
    writer of the boot marker (spec 20 §9.3). ValueError, with nothing written, when `read_marker` would
    not read the marker back equal (a cause, shape or drop off the closed sets, an empty required string,
    a non-integer count) or its JSON is longer than `MARKER_CAP`. The file goes through
    `atomic_write_text(dir_fd=)` inside the no-follow server directory, created if missing: a symlinked
    directory is refused with OSError, and a link planted at the name is replaced, never followed."""
    if not isinstance(marker, Marker):
        raise ValueError(f"not a boot marker: {marker!r}")
    data = dataclasses.asdict(marker)
    data["dropped"] = list(marker.dropped)
    try:
        text = json.dumps(data, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"not a boot marker: {exc}") from exc
    if _marker_from(json.loads(text)) != marker:
        raise ValueError("not a boot marker `read_marker` would read back")
    if len(text.encode("utf-8")) > MARKER_CAP:
        raise ValueError(f"boot marker over {MARKER_CAP} bytes")
    fd = _open_server_dir(config_dir, create=True)
    try:
        atomic_write_text(BOOT_MARKER_NAME, text, dir_fd=fd)
    finally:
        os.close(fd)


# ------------------------------------------------------------------- the service-manager runner

MANAGER_DIRS = ("/usr/bin", "/bin")
"""The only directories the service manager's programs are looked for in. Never PATH, never the current
directory: a planted `systemctl` earlier on PATH would receive every verb the harness issues (T-14-11)."""
RUNTIME_DIR_ROOT = "/run/user"
MANAGER_STDERR_CAP = 4096


def _manager_program(name: str) -> str | None:
    """`name` as an executable regular file in the first of `MANAGER_DIRS` that has one, else None.
    A name holding a separator is not a program name and finds nothing."""
    if not name or "/" in name:
        return None
    for directory in MANAGER_DIRS:
        path = os.path.join(directory, name)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def systemctl_path() -> str | None:
    """`/usr/bin/systemctl` or `/bin/systemctl` (executable), else None; no PATH lookup (D-10)."""
    return _manager_program("systemctl")


def journalctl_path() -> str | None:
    """The journal's program, found the same way as `systemctl_path`."""
    return _manager_program("journalctl")


def manager_env() -> dict[str, str]:
    """The process environment plus `XDG_RUNTIME_DIR=/run/user/<uid>` when that is unset or empty and the
    directory exists, is a directory and is owned by this uid, so a cron-like empty environment still
    reaches the user manager. A value already set is kept as it is."""
    env = dict(os.environ)
    if not env.get("XDG_RUNTIME_DIR"):
        uid = os.getuid()
        runtime = os.path.join(RUNTIME_DIR_ROOT, str(uid))
        try:
            st = os.stat(runtime)
        except OSError:
            return env
        if stat.S_ISDIR(st.st_mode) and st.st_uid == uid:
            env["XDG_RUNTIME_DIR"] = runtime
    return env


@dataclass(frozen=True)
class ManagerResult:
    """What the service manager's program answered: the exit code and its decoded output."""

    code: int
    out: str
    err: str


def _run_manager(argv: list[str], *, timeout: float, cap: int) -> ManagerResult | None:
    """The second subprocess seam, for the autostart unit: `argv[0]` found in `MANAGER_DIRS` only, a list
    argv, no shell, stdin closed, `manager_env()` as the environment. The exit code with the decoded
    stdout / stderr (a non-zero exit is an answer), or None when the program is missing, cannot start,
    times out, or its stdout is longer than `cap` bytes. Stderr is kept to `MANAGER_STDERR_CAP` bytes."""
    exe = _manager_program(argv[0]) if argv else None
    if exe is None:
        return None
    try:
        done = subprocess.run([exe, *argv[1:]], capture_output=True, stdin=subprocess.DEVNULL,
                              timeout=timeout, check=False, env=manager_env())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    if len(done.stdout) > cap:
        return None
    return ManagerResult(done.returncode, done.stdout.decode("utf-8", errors="replace"),
                         done.stderr[:MANAGER_STDERR_CAP].decode("utf-8", errors="replace"))


# ---------------------------------------------------------------------------- the runtime state

_STATE_PROPS = "FragmentPath,DropInPaths,ActiveState,UnitFileState,NRestarts,ExecMainStatus,MainPID,InvocationID,Result"


def _fields(block: str) -> dict[str, str]:
    """`Key=value` lines of one `show` block, split on the first `=`; the first of a repeated key wins."""
    fields: dict[str, str] = {}
    for line in block.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields.setdefault(key, value)
    return fields


def _whole_number(value: str | None) -> int | None:
    return int(value) if value is not None and value.isascii() and value.isdigit() else None


def unit_runtime_state() -> UnitRuntimeState | None:
    """What the user manager says about `localharness-llm.service` right now (D-12), or None when it
    could not be asked (the runner answered None or a non-zero exit) or answered a number the harness
    cannot read (`NRestarts`, `ExecMainStatus` or `MainPID` missing or not a whole number). `MainPID=0`
    (no process) and an empty `InvocationID` are None. `own` is False: `own_unit` sets it."""
    shown = _run_manager(["systemctl", "--user", "show", "-p", _STATE_PROPS, "--", HARNESS_UNIT],
                         timeout=SYSTEMCTL_TIMEOUT_S, cap=SYSTEMCTL_OUTPUT_CAP)
    if shown is None or shown.code != 0:
        return None
    fields = _fields(shown.out)
    n_restarts = _whole_number(fields.get("NRestarts"))
    exec_status = _whole_number(fields.get("ExecMainStatus"))
    main_pid = _whole_number(fields.get("MainPID"))
    if n_restarts is None or exec_status is None or main_pid is None:
        return None
    return UnitRuntimeState(
        invocation_id=fields.get("InvocationID") or None, n_restarts=n_restarts,
        main_pid=main_pid or None, active_state=fields.get("ActiveState", ""),
        unit_file_state=fields.get("UnitFileState", ""), fragment_path=fields.get("FragmentPath", ""),
        drop_in_paths=fields.get("DropInPaths", ""), exec_main_status=exec_status,
        result=fields.get("Result", ""))


def own_unit(state, config_dir, *, interpreter: str | None = None,
             home: str | None = None) -> UnitRuntimeState | None:
    """`state` with `own=True` when it is the harness's own unit (`harness_unit_failure` finds nothing:
    loaded from `harness_unit_path()`, no drop-ins, the file byte-equal to the render), else None
    (D-13). Anything that is not a `UnitRuntimeState` is None."""
    if not isinstance(state, UnitRuntimeState):
        return None
    if harness_unit_failure(state, config_dir, interpreter=interpreter, home=home) is not None:
        return None
    return replace(state, own=True)


def harness_unit_state(config_dir, *, interpreter: str | None = None,
                       home: str | None = None) -> UnitRuntimeState | None:
    """`own_unit(unit_runtime_state(), config_dir)`: the one way a state that selects `UnitStrategy`
    is made (D-13, D-19)."""
    return own_unit(unit_runtime_state(), config_dir, interpreter=interpreter, home=home)


# -------------------------------------------------------------------------------------- the verbs

VERB_TIMEOUTS: Mapping[str, float] = MappingProxyType({
    "daemon-reload": 30, "enable": 30, "disable": 120, "reset-failed": 10,
    "restart": 120, "stop": 120, "start": 120,
})
"""The closed set of user-manager verbs the harness issues, each with its timeout in seconds
(`restart`, `stop` and `disable` wait for a model server to exit). Nothing else is ever issued from here:
no environment, link, edit, set-property or add-wants verb (R-14-13, T-14-12)."""


def systemctl_user(verb: str, *args: str) -> ManagerResult | None:
    """Issue one of the `VERB_TIMEOUTS` verbs to the user manager through `_run_manager`; ValueError,
    before anything runs, for any other verb. The result has the exit code (a non-zero exit is an
    answer); None when the manager could not be run at all."""
    if verb not in VERB_TIMEOUTS:
        raise ValueError(f"not a verb the harness issues: {verb!r}")
    return _run_manager(["systemctl", "--user", verb, *args], timeout=VERB_TIMEOUTS[verb],
                        cap=SYSTEMCTL_OUTPUT_CAP)


# ------------------------------------------------------------------------------------ the journal

JOURNAL_TAIL_LINES = 15
JOURNAL_REF_LINE = "(see journalctl --user -u localharness-llm.service, invocation {invocation})"
"""Where the raw lines of a journal tail an error quoted are: the journal keeps them, no exception does."""
_NO_JOURNAL = "(no journal)"
_INVOCATION_PATTERN = re.compile(r"[0-9A-Za-z-]{1,64}")


def journal_tail(*, invocation: str | None = None, lines: int = JOURNAL_TAIL_LINES,
                 secrets=()) -> str:
    """The last `lines` lines of the harness unit's journal as an error may quote them, SCRUBBED: the
    `secrets` given, `Bearer` tokens, llama-server's `api_keys:` line and URL credentials are masked
    (T-14-13). Queried by the trusted `_SYSTEMD_USER_UNIT=` and `_UID=` fields (never `-u`, which matches
    the forgeable `USER_UNIT`), plus `_SYSTEMD_INVOCATION_ID=` when `invocation` is given (ValueError for
    one that is not an id). `"(no journal)"` when the journal could not be read."""
    from localharness.config import redact  # function-local: the module imports light

    argv = ["journalctl", "--user", "-o", "cat", "-n", str(lines), "--no-pager",
            f"_SYSTEMD_USER_UNIT={HARNESS_UNIT}", f"_UID={os.getuid()}"]
    if invocation is not None:
        if not _INVOCATION_PATTERN.fullmatch(invocation):
            raise ValueError(f"not an invocation id: {invocation!r}")
        argv.append(f"_SYSTEMD_INVOCATION_ID={invocation}")
    read = _run_manager(argv, timeout=JOURNAL_TIMEOUT_S, cap=JOURNAL_OUTPUT_CAP)
    if read is None or read.code != 0 or not read.out.strip():
        return _NO_JOURNAL
    held = [secret for secret in secrets if isinstance(secret, str) and secret]
    return strip_url_credentials(redact.scrub(read.out.rstrip("\n"), held,
                                              patterns=redact.LOGGED_KEY_PATTERNS))


# -------------------------------------------------------------------------------------- linger

LINGER_DIR = "/var/lib/systemd/linger"
"""The login manager keeps one empty file per user with linger on. The harness reads that directory and
never runs the login manager's program (D-09)."""


def current_user() -> str:
    """This process's user name: the password database, else `USER` / `LOGNAME`, else the uid."""
    try:
        import pwd  # function-local: the module still imports where pwd is absent
        return pwd.getpwuid(os.getuid()).pw_name
    except (ImportError, KeyError):
        return os.environ.get("USER") or os.environ.get("LOGNAME") or str(os.getuid())


def linger_on(user: str | None = None) -> bool:
    """True when `<LINGER_DIR>/<user>` exists (`user` defaults to `current_user()`): the user manager
    then runs at boot, with no login. A name that is not one path component is False."""
    name = current_user() if user is None else user
    if not name or name in (".", "..") or "/" in name or "\x00" in name:
        return False
    return os.path.exists(os.path.join(LINGER_DIR, name))


# ------------------------------------------------------------------------------ other launchers

DEFAULT_LLAMA_PORT = 8080
_LAUNCHER_IDENTITY = LaunchIdentity(frozenset({"llama-server"}), None, (), "LLAMA_ARG_PORT")
_LAUNCHER_PROPS = "Id,FragmentPath,UnitFileState,ActiveState"


@dataclass(frozen=True)
class Launcher:
    """Another user unit that starts a llama-server on a port the harness serves: its name, why it
    counts (`enabled`, or `active` when it only runs now) and the port."""

    unit: str
    state: str
    port: int


def other_launchers(ports) -> list[Launcher]:
    """Every user unit other than the harness's that is `enabled` or `active` and whose unit file runs a
    llama-server on one of `ports` (`--port`, else `LLAMA_ARG_PORT`, else 8080), sorted by name (D-10).
    Read through init's `read_unit`, which yields a record for a llama-server unit only: a vLLM or other
    server unit on the port is not named (R-14-14). A unit with drop-ins is judged by its own file. Empty
    when the manager could not be asked."""
    wanted = {p for p in ports if isinstance(p, int) and not isinstance(p, bool)}
    if not wanted:
        return []
    listed = _run_manager(["systemctl", "--user", "list-unit-files", "--type=service", "--plain",
                           "--no-legend", "--no-pager"], timeout=SYSTEMCTL_TIMEOUT_S,
                          cap=SYSTEMCTL_OUTPUT_CAP)
    if listed is None or listed.code != 0:
        return []
    names: set[str] = set()
    for row in listed.out.splitlines():
        cols = row.split()
        if not cols or "@" in cols[0] or cols[0] == HARNESS_UNIT:
            continue
        if len(cols) > 1 and cols[1] in _SKIPPED_FILE_STATES:
            continue
        names.add(cols[0])
    kept = sorted(names)[:UNITS_SCANNED_CAP]
    if not kept:
        return []
    shown = _run_manager(["systemctl", "--user", "show", "-p", _LAUNCHER_PROPS, "--", *kept],
                         timeout=SYSTEMCTL_TIMEOUT_S, cap=SYSTEMCTL_OUTPUT_CAP)
    if shown is None or shown.code != 0:
        return []
    by_name: dict[str, dict[str, str]] = {}
    for block in shown.out.replace("\r\n", "\n").split("\n\n"):
        fields = _fields(block)
        if fields.get("Id") in kept:
            by_name.setdefault(fields["Id"], fields)
    found: list[Launcher] = []
    for name in kept:
        fields = by_name.get(name)
        if fields is None:
            continue
        if fields.get("UnitFileState") in ("enabled", "enabled-runtime"):
            state = "enabled"
        elif fields.get("ActiveState") == "active":
            state = "active"
        else:
            continue
        fragment = fields.get("FragmentPath", "")
        if not fragment:
            continue
        record = read_unit(fragment, unit=name)
        if not isinstance(record, ProcRecord):
            continue
        port = port_of(record, _LAUNCHER_IDENTITY) or DEFAULT_LLAMA_PORT
        if port in wanted:
            found.append(Launcher(name, state, port))
    return found


# ------------------------------------------------------------------------------ the exit, in words

STOPPED_WITHHELD_LINE = "stopped: the machine record withholds {record} — run localharness start in a terminal"
STOPPED_LIVE_PIDFILE_LINE = ("stopped: a live process holds the launch pidfile — the unit will not start "
                             "a second server beside it; localharness model <preset> switches it through the unit")
STOPPED_START_LIMIT_LINE = "stopped after 3 failed loads — journalctl --user -u localharness-llm.service"

_EXIT_WITHHELD = 78
"""`serve`'s exit when the machine record withholds its target; `RestartPreventExitStatus=78` stops retries."""
_EXIT_LIVE_PIDFILE = 75
"""`serve`'s exit (EX_TEMPFAIL) when the pidfile names a live process it cannot verify as its own."""


def translate_exit(state, *, record_name: str, not_served: str | None = None) -> str | None:
    """The line `status` and `doctor` print for a unit that stopped, or None when nothing needs saying.
    In order: `ExecMainStatus` 78 (the machine record withholds `record_name`), 75 (a live pidfile),
    `Result=start-limit-hit` (3 failed loads), then an inactive unit whose target the unit cannot serve
    (`not_served`, already worded by the caller). None for a state that is None."""
    if state is None:
        return None
    status = getattr(state, "exec_main_status", None)
    if status == _EXIT_WITHHELD:
        return STOPPED_WITHHELD_LINE.format(record=record_name)
    if status == _EXIT_LIVE_PIDFILE:
        return STOPPED_LIVE_PIDFILE_LINE
    if getattr(state, "result", "") == "start-limit-hit":
        return STOPPED_START_LIMIT_LINE
    if not_served is not None and getattr(state, "active_state", "") == "inactive":
        return not_served
    return None
