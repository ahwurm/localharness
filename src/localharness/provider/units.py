"""The systemd side of the init import: what a stopped user unit says about the llama-server it runs.

Sensing only: the harness never starts, stops or edits a unit. Every `journalctl` and `systemctl`
call goes through the one seam `_run` (the program found by `which_outside_cwd`, a list argv, no
shell, stdin closed, a timeout, an output cap), and `_run` answers None for anything that is not a
clean, bounded answer, which every caller reads as "not checked". A test replaces `_run`; nothing
else here spawns a process.

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

import json
import os
import shlex
import stat
import subprocess
from dataclasses import dataclass
from types import MappingProxyType

from localharness.core.private_files import open_regular
from localharness.core.which import which_outside_cwd
from localharness.provider.procfs import ProcRecord

__all__ = [
    "JOURNAL_LINES", "JOURNAL_TIMEOUT_S", "JOURNAL_OUTPUT_CAP", "SYSTEMCTL_TIMEOUT_S",
    "SYSTEMCTL_OUTPUT_CAP", "UNIT_FILE_CAP", "UNIT_LINE_CAP", "UNIT_TOKEN_CAP", "UNITS_SCANNED_CAP",
    "Skip", "UnitState", "unit_file_state", "read_journal",
    "read_unit", "unit_record",
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
    """The last llama-server run of the user's `unit` as the record a running process gives, with
    `pid=0`, `environ=None` and `cwd=None` (the journal knows neither). Queried by the trusted
    `_SYSTEMD_USER_UNIT=` and `_UID=` fields, never `-u`, which matches the forgeable `USER_UNIT`."""
    uid = os.getuid()
    out = _run(["journalctl", "--user", "-o", "json", "-n", str(JOURNAL_LINES), "--all",
                "--output-fields=_EXE,_CMDLINE", "--no-pager", f"_SYSTEMD_USER_UNIT={unit}",
                f"_UID={uid}"], timeout=JOURNAL_TIMEOUT_S, cap=JOURNAL_OUTPUT_CAP)
    if out is None:
        return Skip("has an unreadable journal", names_model=False)
    last: tuple[str, list[str]] | None = None
    for line in out.splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict):
            continue
        exe, cmdline = entry.get("_EXE"), entry.get("_CMDLINE")
        if not isinstance(exe, str) or not isinstance(cmdline, str):
            continue  # null past 4096 bytes, a byte array for non-UTF-8, or absent
        try:
            argv = shlex.split(cmdline)
        except ValueError:
            continue
        if os.path.basename(exe) == _LLAMA_SERVER or (
                argv and os.path.basename(argv[0]) == _LLAMA_SERVER):
            last = (exe, argv)
    if last is None:
        return Skip("has no llama-server in its journal", names_model=False)
    exe, argv = last
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


def unit_record(state: UnitState) -> tuple[str, ProcRecord] | Skip:
    """The first usable source for a stopped unit: ("journal", record), else ("unit", record), else
    the `Skip` init folds into its one no-binary line. The journal is tried first; the unit file
    only when the unit has no drop-ins (a drop-in can replace `ExecStart=`). When both skip, the
    first skip in source order whose unit names a model wins; otherwise the unit file's."""
    first = read_journal(state.unit)
    if isinstance(first, ProcRecord):
        return "journal", first
    if not state.fragment_path:
        second: ProcRecord | Skip = Skip("has no unit file", names_model=False)
    else:
        second = read_unit(state.fragment_path, unit=state.unit)
    if state.drop_in_paths:
        names = isinstance(second, ProcRecord) or second.names_model
        second = Skip(f"has drop-ins ({state.drop_in_paths})", names_model=names)
    if isinstance(second, ProcRecord):
        return "unit", second
    return first if first.names_model else second
