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

from localharness.core.which import which_outside_cwd
from localharness.provider.procfs import ProcRecord

__all__ = [
    "JOURNAL_LINES", "JOURNAL_TIMEOUT_S", "JOURNAL_OUTPUT_CAP", "SYSTEMCTL_TIMEOUT_S",
    "SYSTEMCTL_OUTPUT_CAP", "UNIT_FILE_CAP", "UNIT_LINE_CAP", "UNIT_TOKEN_CAP", "UNITS_SCANNED_CAP",
    "Skip", "UnitState", "unit_file_state", "read_journal",
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
