"""Per-root project records: the yes/off store for a project's instructions file.

``~/.localharness/project_records.yaml`` remembers, per workspace root, whether the project's own
instructions file (``CLAUDE.md`` / ``AGENTS.md`` / ``.localharness/INSTRUCTIONS.md``) is used and
which notices were last shown for that root. It is a file of its own, apart from
``trusted_workspaces.yaml`` (that one stays for outside workspaces only): a record here says
"these bytes were shown or named to a person", a trust record says "this tree may run tools".

The agent loop holds a handle built on this module and never imports it at runtime; the harness
never writes an instructions file. The only files this module writes are the store and its lock
sidecar, and a candidate is only ever opened ``O_RDONLY``.

Every store access (read or read-modify-write) runs inside one ``_store_transaction`` that holds
an exclusive ``flock`` on the sidecar ``project_records.yaml.lock``. The data file is never
locked: ``atomic_write_overlay`` replaces its inode on every write, so a lock on it would be a
lock on a file nobody holds any more. The lock is polled with ``LOCK_NB`` for a bounded time and
then the store is reported unavailable; it never proceeds unlocked.
"""
from __future__ import annotations

import copy
import errno
import hashlib
import os
import stat
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from localharness.config.overlay import atomic_write_overlay
from localharness.config.paths import global_config_dir

try:  # POSIX only; elsewhere the store falls back to an in-process lock (documented degradation)
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows only
    fcntl = None  # type: ignore[assignment]

PROJECT_RECORDS_FILE = "project_records.yaml"
"""The store's file name inside the global config dir."""

PROJECT_RECORDS_LOCK_FILE = "project_records.yaml.lock"
"""The sidecar the store's flock is taken on (never the data file)."""

STORE_VERSION = 1
"""The ``version`` field written at the top of the store, so a later reader can migrate."""

LOCK_BUDGET_SECONDS = 2.0
"""How long a caller waits for the sidecar lock before the store is reported unavailable."""

LOCK_POLL_SECONDS = 0.02
"""Pause between ``LOCK_NB`` attempts."""

ACCEPT_MAX_BYTES = 2 * 1024 * 1024
"""The most ``accept_instructions`` reads: it runs before the window is known, so it takes the
largest cap ``instructions.max_ratio`` allows on a 1M-token window."""

CANDIDATE_NAMES: tuple[str, ...] = ("CLAUDE.md", "AGENTS.md", ".localharness/INSTRUCTIONS.md")
"""The three instructions files, in the order they are tried. Nothing else is ever a candidate."""

INSTRUCTION_BASENAMES: tuple[str, ...] = ("CLAUDE.md", "AGENTS.md", "INSTRUCTIONS.md")
"""The basenames a harness call may name (the naming set covers all three candidates)."""

NOTICE_VALUE_MAX = 200
"""The longest ``value`` repr a notice stores; a huge rejected section must not bloat the store."""

_APPROX_CHARS_PER_TOKEN = 4
"""Characters per token for the size cap. Kept private and equal to
``agent.context.APPROX_CHARS_PER_TOKEN`` (a test holds them together): ``agent/__init__`` imports
the loop eagerly, so this core module cannot import that constant without pulling the agent in."""

_INSTRUCTION_STATES = frozenset({"yes", "off"})
_THREAD_LOCK = threading.Lock()


class StoreUnavailable(OSError):
    """The store's lock could not be taken in time, or its sidecar could not be opened.

    An ``OSError`` so every caller that already guards store I/O keeps working; the store is
    never read or written without the lock."""


# --------------------------------------------------------------------------- paths


def store_path(config_dir: Any = None) -> Path:
    """The store file, resolved at call time so a test's environment change takes effect."""
    return global_config_dir(config_dir) / PROJECT_RECORDS_FILE


def lock_path(config_dir: Any = None) -> Path:
    """The sidecar lock file, resolved at call time."""
    return global_config_dir(config_dir) / PROJECT_RECORDS_LOCK_FILE


def _normalize_root(root: Any) -> str:
    """The store key: absolute and resolved, so a symlinked checkout and its real path are one
    entry and a re-pointed root symlink is a new one (the ``coauthor`` spelling)."""
    return str(Path(root).resolve())


# --------------------------------------------------------------------------- lock


@contextmanager
def _store_lock(config_dir: Any = None) -> Iterator[None]:
    """Hold an exclusive lock on the sidecar for the duration of the block.

    Polled with ``LOCK_NB`` (a coroutine calling this must not stall behind another process for
    longer than the budget) and failing closed: past ``LOCK_BUDGET_SECONDS`` it raises
    ``StoreUnavailable`` rather than proceeding unlocked. The sidecar is opened with
    ``O_NOFOLLOW`` at 0600, so a symlinked sidecar is refused. Closing the descriptor releases
    the lock, so a crash leaves nothing stale."""
    if fcntl is None:  # pragma: no cover - Windows: one process-wide lock, best effort
        with _THREAD_LOCK:
            yield
        return
    fd = _open_sidecar(config_dir)
    try:
        deadline = time.monotonic() + LOCK_BUDGET_SECONDS
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise StoreUnavailable(
                        f"{PROJECT_RECORDS_LOCK_FILE} is held by another process"
                    ) from None
                time.sleep(LOCK_POLL_SECONDS)
            except OSError as exc:
                raise StoreUnavailable(f"cannot lock {PROJECT_RECORDS_LOCK_FILE}: {exc}") from exc
        yield
    finally:
        os.close(fd)


def _open_sidecar(config_dir: Any) -> int:
    """Open (creating at 0600) the sidecar; any OSError, a symlinked sidecar included, is
    reported as ``StoreUnavailable``."""
    lock_file = lock_path(config_dir)
    flags = (
        os.O_RDWR | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_file, flags, 0o600)
    except OSError as exc:
        raise StoreUnavailable(f"cannot open {PROJECT_RECORDS_LOCK_FILE}: {exc}") from exc
    try:
        os.fchmod(fd, 0o600)
    except (AttributeError, OSError):
        pass  # best effort: it was created 0600; a wider mode on an older file is not fatal
    return fd


# --------------------------------------------------------------------------- data file


def _read_store_bytes(path: Path) -> bytes | None:
    """The data file's bytes, or None when it is absent or not a regular file. Opened
    ``O_RDONLY`` without following a link and without blocking on a FIFO."""
    flags = (
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOCTTY", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError:
        return None
    finally:
        os.close(fd)


def _load(path: Path) -> dict:
    """The store as a dict. Missing, unreadable, corrupt or non-dict reads as ``{}`` — a damaged
    store means "no records", never a crashed session (the ``coauthor._load`` rule)."""
    raw = _read_store_bytes(path)
    if raw is None:
        return {}
    try:
        data = yaml.safe_load(raw.decode("utf-8"))
    except Exception:  # noqa: BLE001 — corrupt YAML or bytes: undecided, not an error
        return {}
    return data if isinstance(data, dict) else {}


@contextmanager
def _store_transaction(config_dir: Any = None) -> Iterator[dict]:
    """The one read-modify-write primitive: lock, load, yield, write if the block changed it.

    The yielded dict is ``{"version": 1, "roots": {...}}``; mutate it in place. It is written
    through ``atomic_write_overlay`` (0600) only when it differs from what was loaded, so a
    read leaves the file and its inode alone. Never nest it: a second flock from this process
    on a new descriptor would wait on the first."""
    with _store_lock(config_dir):
        loaded = _load(store_path(config_dir))
        roots = loaded.get("roots")
        data = {
            "version": loaded.get("version", STORE_VERSION),
            "roots": roots if isinstance(roots, dict) else {},
            **{k: v for k, v in loaded.items() if k not in ("version", "roots")},
        }
        before = copy.deepcopy(data)
        yield data
        if data != before:
            atomic_write_overlay(store_path(config_dir), data)


def _entry(data: dict, key: str) -> dict | None:
    """The root's entry when it is a mapping, else None."""
    entry = data["roots"].get(key)
    return entry if isinstance(entry, dict) else None


def _entry_for_write(data: dict, key: str) -> dict:
    """The root's entry, replacing a wrong-typed one with an empty mapping."""
    entry = _entry(data, key)
    if entry is None:
        entry = {}
        data["roots"][key] = entry
    return entry


def _prune(data: dict, key: str) -> None:
    """Drop a root's entry that no longer holds anything."""
    entry = _entry(data, key)
    if entry is not None and not entry:
        del data["roots"][key]


# --------------------------------------------------------------------------- records


@dataclass(frozen=True)
class ShadowRecord:
    """A refused shadowing candidate (a higher-priority file that appeared above the recorded one)."""

    name: str
    path: str
    digest: str


@dataclass(frozen=True)
class InstructionsRecord:
    """What was last accepted for a root.

    ``pending`` is ``(sitting_id, iso_utc)`` pairs sorted by id: the open marks of harness calls
    that named the file. ``harness_written`` is the digest of model-written bytes that were
    refused."""

    path: str
    name: str
    digest: str
    size: int
    state: str
    harness_written: str | None = None
    pending: tuple[tuple[str, str], ...] = ()
    shadow: ShadowRecord | None = None


def _record_to_dict(record: InstructionsRecord) -> dict:
    """Plain str / int / dict only (``yaml.safe_dump`` rejects tuples and dataclasses)."""
    out: dict[str, Any] = {
        "path": record.path,
        "name": record.name,
        "digest": record.digest,
        "size": record.size,
        "state": record.state,
    }
    if record.harness_written is not None:
        out["harness_written"] = record.harness_written
    if record.pending:
        out["pending"] = {sid: when for sid, when in sorted(record.pending)}
    if record.shadow is not None:
        out["shadow"] = {
            "name": record.shadow.name,
            "path": record.shadow.path,
            "digest": record.shadow.digest,
        }
    return out


def _pending_from_dict(raw: Any) -> tuple[tuple[str, str], ...]:
    """The open marks of an instructions mapping; a wrong-typed ``pending`` reads as none."""
    if not isinstance(raw, dict):
        return ()
    marks = raw.get("pending")
    if not isinstance(marks, dict):
        return ()
    return tuple(sorted(
        (sid, when) for sid, when in marks.items() if isinstance(sid, str) and isinstance(when, str)
    ))


def _record_from_dict(raw: Any) -> InstructionsRecord | None:
    """The typed record, or None when a required field is missing or of the wrong type or the
    state is not yes / off. Optional fields of the wrong type are dropped."""
    if not isinstance(raw, dict):
        return None
    path, name, digest = raw.get("path"), raw.get("name"), raw.get("digest")
    size, state = raw.get("size"), raw.get("state")
    if not (isinstance(path, str) and isinstance(name, str) and isinstance(digest, str)):
        return None
    if not isinstance(size, int) or isinstance(size, bool):
        return None
    if state not in _INSTRUCTION_STATES:
        return None
    written = raw.get("harness_written")
    shadow_raw = raw.get("shadow")
    shadow = None
    if isinstance(shadow_raw, dict) and all(
        isinstance(shadow_raw.get(k), str) for k in ("name", "path", "digest")
    ):
        shadow = ShadowRecord(shadow_raw["name"], shadow_raw["path"], shadow_raw["digest"])
    return InstructionsRecord(
        path=path, name=name, digest=digest, size=size, state=state,
        harness_written=written if isinstance(written, str) else None,
        pending=_pending_from_dict(raw),
        shadow=shadow,
    )


# --------------------------------------------------------------------------- accessors


def read_instructions(root: Any, *, config_dir: Any = None) -> InstructionsRecord | None:
    """The root's instructions record, or None (absent, damaged, or holding only marks)."""
    key = _normalize_root(root)
    with _store_transaction(config_dir) as data:
        entry = _entry(data, key)
        return _record_from_dict(entry.get("instructions")) if entry is not None else None


def write_instructions(
    root: Any, record: InstructionsRecord, *, config_dir: Any = None
) -> None:
    """Replace the root's instructions mapping with ``record``, keeping its notices."""
    key = _normalize_root(root)
    with _store_transaction(config_dir) as data:
        _entry_for_write(data, key)["instructions"] = _record_to_dict(record)


# --------------------------------------------------------------------------- candidates


def instruction_candidates(root: Any) -> tuple[Path, Path, Path]:
    """The three lexical candidate paths under ``root``, in the order they are tried."""
    base = Path(root)
    first, second, third = (base / name for name in CANDIDATE_NAMES)
    return first, second, third


@dataclass(frozen=True)
class Candidate:
    """An instructions file that resolved: its display ``name`` relative to the root, the
    lexical ``path`` under the root, and the ``real`` resolved path."""

    name: str
    path: Path
    real: Path


def first_candidate(root: Any) -> Candidate | None:
    """The first of ``CLAUDE.md``, ``AGENTS.md``, ``.localharness/INSTRUCTIONS.md`` that is
    usable, or None.

    Only these three names are ever candidates; nothing else under ``.claude/`` is read. A
    candidate is skipped silently when it is absent, when it is a link that leaves the resolved
    root (the contained-link rule the config layer uses), or when it is not a regular file. The
    regular-file test is ``os.stat``, which does not open the file, so a FIFO cannot block it.
    A link that stays inside the root is kept with its resolved real path."""
    root_real = Path(root).resolve()
    for name, lexical in zip(CANDIDATE_NAMES, instruction_candidates(root)):
        if not os.path.lexists(lexical):
            continue
        try:
            real = lexical.resolve()
            if not real.is_relative_to(root_real):
                continue
            if not stat.S_ISREG(os.stat(real).st_mode):
                continue
        except (OSError, RuntimeError, ValueError):
            continue
        return Candidate(name=name, path=lexical, real=real)
    return None


# --------------------------------------------------------------------------- guarded read


@dataclass(frozen=True)
class GuardedRead:
    """The outcome of ``guarded_read``: the bytes with their size and digest, or a ``failure``
    that is one of ``"too_large"``, ``"not a regular file"``, ``"unreadable"``."""

    data: bytes | None
    size: int
    digest: str | None
    failure: str | None


def _digest(data: bytes) -> str:
    """Hex digest of the raw bytes. Private on purpose: no public name carries the algorithm."""
    return hashlib.sha256(data).hexdigest()


def _refused(failure: str, size: int = 0) -> GuardedRead:
    return GuardedRead(data=None, size=size, digest=None, failure=failure)


def guarded_read(real_path: Any, root: Any, cap: int) -> GuardedRead:
    """Read an instructions file without trusting the filesystem to stay as it was.

    ``real_path`` is a recorded real path. It is re-checked on every read: it must still equal
    its own ``realpath`` (a parent directory swapped for a link is not the file recorded) and
    sit inside the resolved ``root``. It is opened read-only without following a link, without
    blocking (a FIFO opens at once) and without a controlling terminal, and ``fstat`` must show
    a regular file before a byte is read. At most ``cap + 1`` bytes are read; more than ``cap``
    is ``too_large``. The digest is over the raw bytes read."""
    path_text = os.fspath(real_path)
    try:
        contained = os.path.realpath(path_text) == path_text and Path(path_text).is_relative_to(
            Path(root).resolve()
        )
    except (OSError, RuntimeError, ValueError):
        contained = False
    if not contained:
        return _refused("not a regular file")
    flags = (
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOCTTY", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        fd = os.open(path_text, flags)
    except OSError as exc:
        return _refused("not a regular file" if exc.errno == errno.ELOOP else "unreadable")
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return _refused("not a regular file")
        if info.st_size > cap:
            return _refused("too_large", info.st_size)
        chunks: list[bytes] = []
        total = 0
        while total <= cap:
            chunk = os.read(fd, min(1 << 16, cap + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
    except OSError:
        return _refused("unreadable")
    finally:
        os.close(fd)
    if total > cap:
        return _refused("too_large", total)
    data = b"".join(chunks)
    return GuardedRead(data=data, size=len(data), digest=_digest(data), failure=None)


def accept_instructions(
    root: Any, candidate: Candidate, *, config_dir: Any = None
) -> InstructionsRecord | None:
    """Record ``state: yes`` for the candidate's current bytes: the one function every yes path
    calls (the offer prompt, the session-trust question, ``/instructions on``).

    The bytes are read through ``guarded_read`` with the ``ACCEPT_MAX_BYTES`` ceiling; a guard
    failure or an oversize file writes nothing and returns None. Existing open marks are kept;
    ``harness_written`` and ``shadow`` are cleared."""
    read = guarded_read(candidate.real, root, ACCEPT_MAX_BYTES)
    if read.failure is not None or read.digest is None:
        return None
    key = _normalize_root(root)
    with _store_transaction(config_dir) as data:
        entry = _entry_for_write(data, key)
        record = InstructionsRecord(
            path=str(candidate.real),
            name=display_name(candidate.name),
            digest=read.digest,
            size=read.size,
            state="yes",
            pending=_pending_from_dict(entry.get("instructions")),
        )
        entry["instructions"] = _record_to_dict(record)
    return record


# --------------------------------------------------------------------------- display


def format_kb(n: int) -> str:
    """A byte count in KB (1024 bytes): one decimal below 10 KB (``5.1 KB``, ``0.4 KB``), an
    integer from 10 KB (``210 KB``)."""
    kb = n / 1024
    return f"{kb:.1f} KB" if kb < 10 else f"{kb:.0f} KB"


def display_name(name: str) -> str:
    """A file name safe to print: ANSI, C0 and C1 control sequences stripped.

    The sanitiser lives in ``channels``; importing it here, not at module load, keeps the config
    package from importing channels when it loads."""
    from localharness.channels.base import sanitize_for_display

    return sanitize_for_display(name)
