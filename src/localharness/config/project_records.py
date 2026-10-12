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
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import structlog
import yaml

from localharness.config import session_presence
from localharness.config.overlay import atomic_write_overlay
from localharness.config.paths import global_config_dir
from localharness.core.events import InstructionsCause, InstructionsNotice

try:  # POSIX only; elsewhere the store falls back to an in-process lock (documented degradation)
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows only
    fcntl = None  # type: ignore[assignment]

log = structlog.get_logger(__name__)

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


# --------------------------------------------------------------------------- notices


@dataclass(frozen=True)
class Notice:
    """One print-on-change line (a narrowing warning, an oversized file).

    Hashable, so a set of notices compares whole tuples. ``value`` is the ``repr`` of the
    rejected value and is the narrowing fingerprint: the same key rejected with a different
    value is a different notice and prints again."""

    kind: str
    subject: str
    file: str
    value: str
    text: str


def notice_value(value: Any) -> str:
    """``repr(value)`` cut to ``NOTICE_VALUE_MAX`` characters, the last replaced by ``…``."""
    text = repr(value)
    if len(text) > NOTICE_VALUE_MAX:
        return text[: NOTICE_VALUE_MAX - 1] + "…"
    return text


def _notice_to_dict(notice: Notice) -> dict:
    return {
        "kind": notice.kind, "subject": notice.subject, "file": notice.file,
        "value": notice.value, "text": notice.text,
    }


def _notice_from_dict(raw: Any) -> Notice | None:
    """The notice, or None for an entry that is not a mapping of five strings."""
    if not isinstance(raw, dict):
        return None
    fields = [raw.get(k) for k in ("kind", "subject", "file", "value", "text")]
    if not all(isinstance(f, str) for f in fields):
        return None
    return Notice(*fields)


def _stored_notices(entry: dict | None) -> frozenset[Notice]:
    raw = entry.get("notices") if entry is not None else None
    if not isinstance(raw, list):
        return frozenset()
    return frozenset(n for n in map(_notice_from_dict, raw) if n is not None)


def read_notices(root: Any, *, config_dir: Any = None) -> frozenset[Notice]:
    """The notices last stored for ``root``; malformed entries are skipped."""
    key = _normalize_root(root)
    with _store_transaction(config_dir) as data:
        return _stored_notices(_entry(data, key))


def replace_notices(root: Any, notices: Any, *, config_dir: Any = None) -> None:
    """Replace the root's stored notices with ``notices``, keeping its instructions mapping.

    The stored set is the notices of the latest start, so a notice that disappears and returns
    is unseen again."""
    key = _normalize_root(root)
    wanted = sorted(
        set(notices), key=lambda n: (n.kind, n.subject, n.file, n.value, n.text)
    )
    with _store_transaction(config_dir) as data:
        if wanted:
            _entry_for_write(data, key)["notices"] = [_notice_to_dict(n) for n in wanted]
            return
        entry = _entry(data, key)
        if entry is not None:
            entry.pop("notices", None)
            _prune(data, key)


def unseen(root: Any, notices: Any, *, config_dir: Any = None) -> list[Notice]:
    """The given notices that are not in the root's stored set, in the given order, a repeat
    collapsed to its first occurrence. A pure read: it never writes the store."""
    stored = read_notices(root, config_dir=config_dir)
    out: list[Notice] = []
    for notice in notices:
        if notice not in stored and notice not in out:
            out.append(notice)
    return out


_NO_DOOR_SUBJECTS = frozenset({"permissions.budget.max_tool_calls"})
"""Narrowed keys no global file accepts (nothing derives the value), so the note names no file."""


class DispatchNoticeSink:
    """Narrowing notes made while a built-in subagent is dispatched, told once per sitting.

    ``start`` shows every narrowing it can see up front. A dispatch re-reads the overlay file, so a
    file the model (or a hook) wrote since can narrow a value ``start`` never saw. ``record`` takes
    such a note synchronously (the loader reports from synchronous code); ``flush`` publishes each
    one that is not in the start set and was not published already as a ``NarrowingNotice`` on the
    bus, where a channel shows it on its status line. Nothing here logs: a log record from a
    running turn lands on the terminal over the input box."""

    def __init__(self, start_notices: Iterable[Notice], *, global_dir: Path) -> None:
        self._start = frozenset(start_notices)
        self._global_dir = Path(global_dir)
        self._pending: list[Notice] = []
        self._shown: set[Notice] = set()

    def record(self, notice: Notice) -> None:
        if notice in self._start or notice in self._shown or notice in self._pending:
            return
        self._pending.append(notice)

    def _accept_file(self, notice: Notice) -> str:
        """The global file that would accept the value: ``""`` for a key no door accepts."""
        if notice.subject in _NO_DOOR_SUBJECTS:
            return ""
        file = Path(notice.file)
        if file.parent.name == "agents":
            return str(self._global_dir / "agents" / file.name)
        return str(self._global_dir / "config.yaml")

    async def flush(self, bus: Any, *, agent_id: str, session_id: str) -> None:
        """Publish each recorded note not shown yet, once for the sitting."""
        from localharness.core.events import NarrowingNotice

        pending, self._pending = self._pending, []
        for notice in pending:
            if notice in self._shown:
                continue
            self._shown.add(notice)
            accept = self._accept_file(notice)
            text = f"{notice.text} — {accept!r} would accept it" if accept else notice.text
            await bus.publish(NarrowingNotice(
                agent_id=agent_id, session_id=session_id, key=notice.subject,
                file=notice.file, accept_file=accept, text=text))


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


# --------------------------------------------------------------------------- the handle


def _start_text(name: str, size: int) -> str:
    """The dim line owed when a file starts being used (D-40)."""
    return f"Using {display_name(name)} ({format_kb(size)}); /instructions off"


def _harness_written_text(name: str) -> str:
    return f"{display_name(name)} changed by the model; not used — /instructions on"


def _name_rank(name: str) -> int:
    """Where a candidate name sits in the order it is tried; an unknown name sorts last."""
    try:
        return CANDIDATE_NAMES.index(name)
    except ValueError:
        return len(CANDIDATE_NAMES)


def _as_pending(marks: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(marks.items()))


def _inst_raw(data: dict, key: str) -> dict | None:
    """The root's raw instructions mapping (a record, a mark-only mapping, or None)."""
    entry = _entry(data, key)
    raw = entry.get("instructions") if entry is not None else None
    return raw if isinstance(raw, dict) else None


def _put_record(data: dict, key: str, record: InstructionsRecord, pending: dict[str, str]) -> None:
    """Write ``record`` carrying exactly the open marks in ``pending``."""
    _entry_for_write(data, key)["instructions"] = _record_to_dict(
        replace(record, pending=_as_pending(pending))
    )


def _put_marks_only(data: dict, key: str, pending: dict[str, str]) -> None:
    """Keep only the open marks for a root with no record (a mark-only instructions mapping)."""
    if pending:
        _entry_for_write(data, key)["instructions"] = {"pending": dict(sorted(pending.items()))}
        return
    entry = _entry(data, key)
    if entry is not None:
        entry.pop("instructions", None)
        _prune(data, key)


@dataclass(frozen=True)
class InstructionsState:
    """What the handle knows about the root's file right now (read-only).

    ``state`` is one of ``yes``, ``off``, ``pending`` (a record with an open mark),
    ``harness_written`` (the file currently holds bytes a harness call wrote and the record
    refused), ``too_large`` and ``none`` (no record)."""

    name: str
    path: str
    size: int
    digest: str | None
    state: str


@dataclass(frozen=True)
class _Awaiting:
    """The bytes a start line described: written as the yes record only once the line was
    delivered (D-32)."""

    name: str
    path: str
    size: int
    digest: str
    text: str


@dataclass(frozen=True)
class _Marks:
    """Open marks seen at one call: this handle's own, other sessions' live ones, dead ones.

    Under ``--no-input`` the dead ones are treated as absent (and never cleared)."""

    own: bool
    live: frozenset[str]
    dead: frozenset[str]

    @property
    def any(self) -> bool:
        return self.own or bool(self.live) or bool(self.dead)


def _state_name(
    record: InstructionsRecord | None, *, marks_open: bool, digest: str | None, too_large: bool
) -> str:
    if too_large:
        return "too_large"
    if record is None:
        return "none"
    if record.state == "off":
        return "off"
    if digest is not None and record.harness_written == digest:
        return "harness_written"
    return "pending" if marks_open else "yes"


class ProjectInstructions:
    """The one object the loop, the trust prompts, the channels and ``/instructions`` hold.

    ``section()`` answers "use this text" or "skip" at every call; the store is read (and, on a
    transition, written) under one lock acquisition per call. The constructor does no I/O.
    ``basis`` is ``layer`` (the project you stand in: the file is used by default with one dim
    line), ``trusted`` (same, a root trusted before there was a record) or ``awaiting_trust``
    (used only once a ``yes`` record exists). ``no_input`` is the CLI flag: nothing is ever
    written and no start line is owed. The handle never opens a candidate for writing."""

    def __init__(
        self,
        root: Any,
        *,
        sitting_id: str,
        max_ratio: float,
        basis: str,
        no_input: bool,
        config_dir: Any = None,
    ) -> None:
        self._root = Path(os.fspath(root))
        self._sitting_id = sitting_id
        self._max_ratio = max_ratio
        self._basis = basis
        self._no_input = no_input
        self._config_dir = config_dir
        self._candidate: Candidate | None = None
        self._declined = False
        self._marked = False
        self._had_record = False
        self._warned: set[tuple] = set()
        self._held: InstructionsNotice | None = None
        self._awaiting: _Awaiting | None = None
        self._start_notice_given = False
        self._sitting_digest: str | None = None
        self._dropped_digest: str | None = None
        self._awaiting_too_large: Notice | None = None
        self._too_large_notice: Notice | None = None
        self._start_line: str | None = None
        self._summary_entity: str | None = None
        self._recorded_path: str | None = None
        self._mem_state: Literal["yes", "off"] | None = None
        self._set_failure: str | None = None
        self._last = InstructionsState("", "", 0, None, "none")

    # ---- the public surface

    @property
    def candidates(self) -> tuple[Path, Path, Path]:
        return instruction_candidates(self._root)

    @property
    def recorded_path(self) -> Path | None:
        return Path(self._recorded_path) if self._recorded_path else None

    @property
    def state(self) -> InstructionsState:
        return self._last

    @property
    def start_line(self) -> str | None:
        return self._start_line

    @property
    def summary_entity(self) -> str | None:
        return self._summary_entity

    @property
    def too_large_notice(self) -> Notice | None:
        return self._too_large_notice

    # ---- section

    def section(self, window_tokens: int, *, cancelled: bool = False) -> tuple[
        str | None, InstructionsNotice | None
    ]:
        """The text to put in the role (or None) and the status line this call owes (or None).

        Never raises: a store failure is a ``refused`` line with reason ``unreadable store`` and
        no text. With ``cancelled=True`` the transition is still written but the line is held
        and returned by the next call that has none of its own."""
        if self._declined:
            return None, None
        try:
            text, notice = self._run(window_tokens, at_start=False)
        except (OSError, RuntimeError) as exc:
            log.debug("project_instructions_store_unavailable", error=str(exc))
            text, notice = None, self._store_refusal()
        if cancelled:
            if notice is not None:
                self._held = notice
            return text, None
        if notice is None:
            held, self._held = self._held, None
            return text, held
        self._held = None
        return text, notice

    def _run(
        self, window_tokens: int, *, at_start: bool
    ) -> tuple[str | None, InstructionsNotice | None]:
        cap = int(self._max_ratio * window_tokens * _APPROX_CHARS_PER_TOKEN)
        key = _normalize_root(self._root)
        fx: list[Callable[[], None]] = []
        with _store_transaction(self._config_dir) as data:
            result = self._decide(data, key, cap, fx, at_start=at_start)
        for effect in fx:
            effect()
        return result

    def _store_refusal(self) -> InstructionsNotice | None:
        """The ``unreadable store`` line, once per handle."""
        name, path = self._last.name, self._last.path
        key = ("refused", "unreadable store", path, 0)
        if key in self._warned:
            return None
        self._warned.add(key)
        return self._notice(
            "refused", name, path, self._last.size, self._last.state,
            reason="unreadable store",
            text=f"{display_name(name)} not used this turn (unreadable store)",
        )

    def _record_lost_refusal(self) -> InstructionsNotice | None:
        """The ``record lost`` line, once per handle: the file stays out until ``/instructions on``."""
        name, path = self._last.name, self._last.path
        key = ("refused", "record lost", path, 0)
        if key in self._warned:
            return None
        self._warned.add(key)
        return self._notice(
            "refused", name, path, self._last.size, self._last.state,
            reason="record lost",
            text=f"{display_name(name)} not used this turn (record lost); /instructions on",
        )

    @staticmethod
    def _notice(
        cause: InstructionsCause, name: str, path: str, size: int, state: str, *,
        text: str, shadowed: str | None = None, reason: str | None = None,
    ) -> InstructionsNotice:
        return InstructionsNotice(
            cause=cause, name=display_name(name), path=path, size=size, state=state,
            shadowed=shadowed, reason=reason, text=text,
        )

    def _once(self, key: tuple, fx: list[Callable[[], None]]) -> bool:
        """True the first time ``key`` is seen; remembered only if the call's writes succeed."""
        if key in self._warned:
            return False
        fx.append(lambda: self._warned.add(key))
        return True

    def _marks(self, pending: dict[str, str]) -> _Marks:
        own = self._marked or self._sitting_id in pending
        foreign = {sid for sid in pending if sid != self._sitting_id}
        live, dead = set(foreign), set()
        if foreign:
            try:
                alive = {s.session_id for s in session_presence.live_sessions(self._config_dir)}
            except Exception:  # noqa: BLE001 — cannot tell who is alive: nobody's mark is dead
                alive = set(foreign)
            live, dead = foreign & alive, foreign - alive
        if self._no_input:
            dead = set()
        return _Marks(own, frozenset(live), frozenset(dead))

    def _set_last(
        self, fx: list[Callable[[], None]], record: InstructionsRecord | None, *,
        marks_open: bool, digest: str | None, too_large: bool = False,
        name: str = "", path: str = "", size: int = 0,
    ) -> None:
        if record is not None:
            name, path, size = record.name, record.path, record.size
        state = InstructionsState(
            name or self._last.name, path or self._last.path, size,
            record.digest if record is not None else digest,
            _state_name(record, marks_open=marks_open, digest=digest, too_large=too_large),
        )
        fx.append(lambda: setattr(self, "_last", state))

    def _failure(
        self, read: GuardedRead, *, name: str, path: str, record: InstructionsRecord | None,
        cap: int, stored: frozenset[Notice], fx: list[Callable[[], None]], at_start: bool,
    ) -> InstructionsNotice | None:
        """The line for a guard failure, once per ``(cause, reason, path, size)``."""
        shown = display_name(name)
        if read.failure == "too_large":
            size = read.size
            text = (
                f"{shown} ({format_kb(size)}) over the {format_kb(cap)} limit "
                "(instructions.max_ratio); not used"
            )
            row = Notice("instructions_too_large", path, path, repr(size), text)
            self._set_last(
                fx, record, marks_open=False, digest=None, too_large=True,
                name=name, path=path, size=size,
            )
            if at_start:
                fx.append(lambda: setattr(self, "_too_large_notice", row))
                return None
            if any(
                n.kind == row.kind and n.subject == path and n.value == row.value for n in stored
            ):
                return None
            if not self._once(("too_large", None, path, size), fx):
                return None
            fx.append(lambda: setattr(self, "_awaiting_too_large", row))
            return self._notice("too_large", name, path, size, "too_large", text=text)
        if at_start:
            return None
        reason = read.failure or "unreadable"
        if not self._once(("refused", reason, path, 0), fx):
            return None
        return self._notice(
            "refused", name, path, record.size if record is not None else 0,
            self._last.state, reason=reason, text=f"{shown} not used this turn ({reason})",
        )

    def _decide(
        self, data: dict, key: str, cap: int, fx: list[Callable[[], None]], *, at_start: bool
    ) -> tuple[str | None, InstructionsNotice | None]:
        """One call of the D-28 state machine, inside the store transaction. Everything that
        changes the handle itself is queued in ``fx`` and applied only if the writes succeed."""
        raw = _inst_raw(data, key)
        record = _record_from_dict(raw)
        pending = dict(_pending_from_dict(raw))
        stored = _stored_notices(_entry(data, key))
        self._recorded_path = record.path if record is not None else None
        usable = record is not None and record.state == "yes"
        if self._mem_state == "off":
            return None, None
        if record is not None:
            fx.append(lambda: setattr(self, "_had_record", True))
        elif self._had_record or (self._marked and self._awaiting is None):
            # a record this sitting saw (or a mark it left) has vanished: deleted, damaged or
            # swapped under a running sitting. That is not a fresh root, so nothing is adopted;
            # only `/instructions on` brings the file back (CR-51, D-72)
            return None, self._record_lost_refusal()
        if not usable and self._mem_state != "yes":
            if record is None and self._basis in ("layer", "trusted") and not self._no_input:
                pass  # the no-record sitting: used, with the start line owed
            else:
                if record is not None:
                    self._set_last(fx, record, marks_open=False, digest=None)
                return None, None
        if self._awaiting is not None:
            if record is None or (record.state == "yes" and record.digest != self._awaiting.digest):
                return self._awaiting_step(cap, pending, stored, fx)
            fx.append(self._clear_awaiting)
        if not usable:
            return self._no_record(cap, stored, fx, at_start=at_start)
        return self._with_record(data, key, record, pending, cap, stored, fx, at_start=at_start)

    def _clear_awaiting(self) -> None:
        self._awaiting = None
        self._start_line = None

    def _no_record(
        self, cap: int, stored: frozenset[Notice], fx: list[Callable[[], None]], *, at_start: bool
    ) -> tuple[str | None, InstructionsNotice | None]:
        """No record: the file is used this sitting and its line is owed (D-28 step 9)."""
        cand = self._candidate or first_candidate(self._root)
        if cand is None:
            return None, None
        path = str(cand.real)
        read = guarded_read(cand.real, self._root, cap)
        if read.failure is not None or read.data is None or read.digest is None:
            notice = self._failure(
                read, name=cand.name, path=path, record=None, cap=cap, stored=stored, fx=fx,
                at_start=at_start,
            )
            return None, notice
        text = read.data.decode("utf-8", errors="replace")
        self._set_last(
            fx, None, marks_open=False, digest=read.digest, name=cand.name, path=path,
            size=read.size,
        )
        if self._no_input or self._basis not in ("layer", "trusted"):
            return text, None
        line = _start_text(cand.name, read.size)
        aw = _Awaiting(cand.name, path, read.size, read.digest, line)

        def arm() -> None:
            self._awaiting, self._start_line, self._sitting_digest = aw, line, aw.digest

        fx.append(arm)
        if at_start:
            return text, None
        fx.append(lambda: setattr(self, "_start_notice_given", True))
        return text, self._notice("start", cand.name, path, read.size, "none", text=line)

    def _awaiting_step(
        self, cap: int, pending: dict[str, str], stored: frozenset[Notice],
        fx: list[Callable[[], None]],
    ) -> tuple[str | None, InstructionsNotice | None]:
        """A start line is owed: the digest it described is the one in use this sitting."""
        aw = self._awaiting
        assert aw is not None
        read = guarded_read(aw.path, self._root, cap)
        if read.failure is not None or read.data is None or read.digest is None:
            return None, self._failure(
                read, name=aw.name, path=aw.path, record=None, cap=cap, stored=stored, fx=fx,
                at_start=False,
            )
        text = read.data.decode("utf-8", errors="replace")
        if read.digest == aw.digest:
            if self._start_notice_given:
                return text, None
            fx.append(lambda: setattr(self, "_start_notice_given", True))
            return text, self._notice("start", aw.name, aw.path, aw.size, "none", text=aw.text)
        if read.digest == self._dropped_digest:
            return None, None
        digest = read.digest
        if self._marks(pending).any:
            # a harness call named the file after the line was owed: the bytes the line
            # described stay the ones in use, the model's are dropped (D-84)
            fx.append(lambda: setattr(self, "_dropped_digest", digest))
            state = InstructionsState(
                display_name(aw.name), aw.path, read.size, digest, "harness_written"
            )
            fx.append(lambda: setattr(self, "_last", state))
            return None, self._notice(
                "harness_written", aw.name, aw.path, read.size, "harness_written",
                text=_harness_written_text(aw.name),
            )
        return text, None  # the person's own edit in a sitting with no record: no second line

    def _with_record(
        self, data: dict, key: str, record: InstructionsRecord, pending: dict[str, str],
        cap: int, stored: frozenset[Notice], fx: list[Callable[[], None]], *, at_start: bool,
    ) -> tuple[str | None, InstructionsNotice | None]:
        first = first_candidate(self._root)
        marks = self._marks(pending)
        shadowing = (
            first is not None and str(first.real) != record.path
            and _name_rank(first.name) <= _name_rank(record.name)
        )
        if shadowing and not marks.any:
            assert first is not None
            return self._take_over(
                data, key, first, record, pending, cap, stored, fx, at_start=at_start
            )
        read = guarded_read(record.path, self._root, cap)
        if read.failure is not None or read.data is None or read.digest is None:
            return None, self._failure(
                read, name=record.name, path=record.path, record=record, cap=cap, stored=stored,
                fx=fx, at_start=at_start,
            )
        text = read.data.decode("utf-8", errors="replace")
        digest = read.digest
        at_recorded = first is not None and str(first.real) == record.path
        clear = set(marks.dead) | ({self._sitting_id} & set(pending))

        if digest == record.digest:
            if shadowing:
                assert first is not None
                return text, self._shadowed(
                    data, key, first, record, pending, read, cap, fx
                )
            left = {k: v for k, v in pending.items() if k not in clear} if at_recorded else dict(pending)
            new = replace(record, shadow=None) if at_recorded else record
            if not self._no_input and (left != pending or new != record):
                _put_record(data, key, new, left)
            if at_recorded:
                fx.append(lambda: setattr(self, "_marked", False))
            self._set_last(
                fx, new, marks_open=bool(left) or (self._marked and not at_recorded), digest=digest
            )
            return text, None

        if digest == record.harness_written:
            self._set_last(fx, record, marks_open=bool(pending) or self._marked, digest=digest)
            return None, None

        if marks.any:
            if self._no_input:  # nothing is written; the model's bytes stay out this sitting
                self._set_last(fx, record, marks_open=True, digest=digest)
                return None, None
            left = {k: v for k, v in pending.items() if k not in clear}
            new = replace(record, harness_written=digest)
            _put_record(data, key, new, left)
            fx.append(lambda: setattr(self, "_marked", False))
            self._set_last(fx, new, marks_open=bool(left), digest=digest)
            return None, self._notice(
                "harness_written", record.name, record.path, read.size, "harness_written",
                text=_harness_written_text(record.name),
            )

        # a change no harness call named: the person's
        if self._no_input:
            self._set_last(fx, record, marks_open=False, digest=digest)
            return text, None
        if at_start:
            self._arm(fx, record.name, record.path, read.size, digest)
            self._set_last(fx, record, marks_open=False, digest=digest)
            return text, None
        new = replace(record, digest=digest, size=read.size, harness_written=None, shadow=None)
        _put_record(data, key, new, pending)
        self._set_last(fx, new, marks_open=bool(pending), digest=digest)
        return text, self._notice(
            "adopted", record.name, record.path, read.size, "yes",
            text=f"{display_name(record.name)} changed, using it",
        )

    def _arm(
        self, fx: list[Callable[[], None]], name: str, path: str, size: int, digest: str
    ) -> None:
        """Owe the start line for these bytes: nothing is written until it is delivered."""
        line = _start_text(name, size)
        aw = _Awaiting(name, path, size, digest, line)

        def apply() -> None:
            self._awaiting, self._start_line, self._sitting_digest = aw, line, digest

        fx.append(apply)

    def _take_over(
        self, data: dict, key: str, first: Candidate, record: InstructionsRecord,
        pending: dict[str, str], cap: int, stored: frozenset[Notice],
        fx: list[Callable[[], None]], *, at_start: bool,
    ) -> tuple[str | None, InstructionsNotice | None]:
        """A file above the recorded one, with no mark of any kind: the person's, adopted."""
        path = str(first.real)
        read = guarded_read(first.real, self._root, cap)
        if read.failure is not None or read.data is None or read.digest is None:
            return None, self._failure(
                read, name=first.name, path=path, record=record, cap=cap, stored=stored, fx=fx,
                at_start=at_start,
            )
        text = read.data.decode("utf-8", errors="replace")
        if self._no_input:
            self._set_last(fx, record, marks_open=False, digest=read.digest)
            return text, None
        if at_start:
            self._arm(fx, first.name, path, read.size, read.digest)
            self._set_last(fx, record, marks_open=False, digest=read.digest)
            return text, None
        new = InstructionsRecord(
            path=path, name=display_name(first.name), digest=read.digest, size=read.size,
            state="yes",
        )
        _put_record(data, key, new, pending)
        self._set_last(fx, new, marks_open=bool(pending), digest=read.digest)
        return text, self._notice(
            "adopted", first.name, path, read.size, "yes",
            text=f"{display_name(first.name)} changed, using it",
        )

    def _shadowed(
        self, data: dict, key: str, first: Candidate, record: InstructionsRecord,
        pending: dict[str, str], read: GuardedRead, cap: int, fx: list[Callable[[], None]],
    ) -> InstructionsNotice | None:
        """A file above the recorded one under a mark: the recorded file stays in use while its
        own digest holds, the refused file is remembered and one line says so."""
        other = guarded_read(first.real, self._root, cap)
        shadow = ShadowRecord(display_name(first.name), str(first.real), other.digest or "")
        if not self._no_input and record.shadow != shadow:
            _put_record(data, key, replace(record, shadow=shadow), pending)
        self._set_last(fx, record, marks_open=True, digest=read.digest)
        if not self._once(("shadowed", shadow.path, shadow.digest), fx):
            return None
        return self._notice(
            "shadowed", record.name, record.path, read.size, "pending", shadowed=shadow.name,
            text=(
                f"{shadow.name} (model edit) not used; using {display_name(record.name)}"
                " — /instructions on to switch"
            ),
        )

    # ---- marks

    def mark_pending(self) -> None:
        """A harness call is about to name the file: leave a mark before it runs (D-29).

        The in-memory mark is set first, so a store failure can never turn a model write into a
        person's change (D-72). Under ``--no-input`` nothing is written."""
        self._marked = True
        if self._no_input or self._declined:
            return
        try:
            key = _normalize_root(self._root)
            with _store_transaction(self._config_dir) as data:
                raw = _inst_raw(data, key)
                record = _record_from_dict(raw)
                pending = dict(_pending_from_dict(raw))
                pending[self._sitting_id] = datetime.now(timezone.utc).isoformat()
                if record is not None:
                    _put_record(data, key, record, pending)
                else:
                    _put_marks_only(data, key, pending)
        except (OSError, RuntimeError) as exc:
            log.debug("project_instructions_mark_failed", error=str(exc))

    # ---- start

    def _precheck(self, window_tokens: int) -> None:
        """The start hook's own check (D-48): one pass of the state machine in start mode.

        It decides the start line, the summary entity and the too-large notice. In start mode a
        change found with no mark is adopted through the start line (written only by
        ``on_delivered``), so one change never draws two lines."""
        try:
            _, notice = self._run(window_tokens, at_start=True)
        except (OSError, RuntimeError) as exc:
            log.debug("project_instructions_precheck_failed", error=str(exc))
            self._summary_entity = None
            self._start_line = None
            self._awaiting = None
            return
        if notice is not None and notice.cause in ("harness_written", "shadowed"):
            self._held = notice  # the first assembly's call (or any call with none of its own) says it
        self._summary_entity = self._entity()

    def _entity(self) -> str | None:
        """The counts entity text for the summary line (D-54)."""
        if self._awaiting is not None:
            return display_name(self._awaiting.name)
        state = self._last.state
        if state == "none":
            return None
        name = display_name(self._last.name)
        if state == "off":
            return f"{name} (off)"
        if state == "harness_written":
            return f"{name} (model edit; /instructions on)"
        if state == "too_large":
            return f"{name} (too large)"
        return name

    # ---- state changes

    def decline(self) -> None:
        """The workspace was declined this sitting: the handle is inert, nothing is written (D-61)."""
        self._declined = True
        self._awaiting = None
        self._start_line = None
        self._held = None

    def set_state(self, value: Literal["yes", "off"]) -> None:
        """The person's answer (D-33): ``yes`` accepts the first resolving candidate's current
        bytes, ``off`` is never re-adopted until ``yes``. Both clear an owed start line. Under
        ``--no-input`` the answer holds for this sitting only and nothing is written."""
        if self._declined:
            return
        self._awaiting = None
        self._start_line = None
        self._held = None
        self._dropped_digest = None
        self._set_failure = None
        try:
            if self._no_input:
                self._answer_in_memory(value)
            elif value == "yes":
                self._accept()
            else:
                self._turn_off()
        except (OSError, RuntimeError) as exc:
            log.debug("project_instructions_set_state_failed", error=str(exc))
            self._set_failure = "unreadable store"

    def _accept(self) -> None:
        cand = first_candidate(self._root)
        if cand is None:
            self._set_failure = "no file"
            self._last = replace(self._last, state="none")
            return
        record = accept_instructions(self._root, cand, config_dir=self._config_dir)
        if record is None:
            probe = guarded_read(cand.real, self._root, ACCEPT_MAX_BYTES)
            self._set_failure = "too large" if probe.failure == "too_large" else (
                probe.failure or "unreadable"
            )
            return
        pending = dict(record.pending)
        if self._sitting_id in pending:
            del pending[self._sitting_id]
            with _store_transaction(self._config_dir) as data:
                _put_record(data, _normalize_root(self._root), record, pending)
        self._marked = False
        self._had_record = True
        self._candidate = cand
        self._recorded_path = record.path
        self._last = InstructionsState(
            record.name, record.path, record.size, record.digest, "pending" if pending else "yes"
        )

    def _turn_off(self) -> None:
        key = _normalize_root(self._root)
        with _store_transaction(self._config_dir) as data:
            raw = _inst_raw(data, key)
            record = _record_from_dict(raw)
            pending = dict(_pending_from_dict(raw))
            if record is not None:
                new = replace(record, state="off")
            else:
                cand = self._candidate or first_candidate(self._root)
                if cand is None:
                    self._set_failure = "no file"
                    return
                read = guarded_read(cand.real, self._root, ACCEPT_MAX_BYTES)
                new = InstructionsRecord(
                    path=str(cand.real), name=display_name(cand.name), digest=read.digest or "",
                    size=read.size, state="off",
                )
            _put_record(data, key, new, pending)
        self._recorded_path = new.path
        self._had_record = True
        self._last = InstructionsState(new.name, new.path, new.size, new.digest, "off")

    def _answer_in_memory(self, value: str) -> None:
        cand = first_candidate(self._root)
        if cand is None:
            self._set_failure = "no file"
            return
        if value == "off":
            self._mem_state = "off"
            self._last = replace(self._last, state="off")
            return
        probe = guarded_read(cand.real, self._root, ACCEPT_MAX_BYTES)
        if probe.failure is not None:
            self._set_failure = "too large" if probe.failure == "too_large" else probe.failure
            return
        self._mem_state = "yes"
        self._candidate = cand
        self._last = InstructionsState(
            display_name(cand.name), str(cand.real), probe.size, probe.digest, "yes"
        )

    def _refresh(self) -> None:
        """Bring ``state`` up to date with the store and the file (for ``/instructions show``)."""
        try:
            key = _normalize_root(self._root)
            with _store_transaction(self._config_dir) as data:
                raw = _inst_raw(data, key)
                record = _record_from_dict(raw)
                pending = dict(_pending_from_dict(raw))
        except (OSError, RuntimeError):
            return
        if record is not None:
            probe = guarded_read(record.path, self._root, ACCEPT_MAX_BYTES)
            state = _state_name(
                record, marks_open=bool(pending) or self._marked, digest=probe.digest,
                too_large=False,
            )
            if self._mem_state == "off":
                state = "off"
            self._recorded_path = record.path
            self._last = InstructionsState(
                record.name, record.path, record.size, record.digest, state
            )
        elif self._awaiting is None and self._mem_state is None:
            cand = self._candidate or first_candidate(self._root)
            if cand is None:
                return
            probe = guarded_read(cand.real, self._root, ACCEPT_MAX_BYTES)
            self._last = InstructionsState(
                display_name(cand.name), str(cand.real), probe.size, probe.digest, "none"
            )

    # ---- delivery

    def on_delivered(self) -> None:
        """The start line (or the oversized line) reached a person: write what it described.

        Writes the ``yes`` record for the digest the line described, plus ``harness_written``
        for a digest dropped in the same sitting, and the oversized notice off the terminal.
        A second call, ``--no-input`` or a declined sitting writes nothing; it never raises."""
        if self._no_input or self._declined:
            return
        aw, big = self._awaiting, self._awaiting_too_large
        if aw is None and big is None:
            return
        try:
            key = _normalize_root(self._root)
            with _store_transaction(self._config_dir) as data:
                raw = _inst_raw(data, key)
                current = _record_from_dict(raw)
                pending = dict(_pending_from_dict(raw))
                if aw is not None and not (current is not None and current.state == "off"):
                    record = InstructionsRecord(
                        path=aw.path, name=display_name(aw.name), digest=aw.digest, size=aw.size,
                        state="yes", harness_written=self._dropped_digest,
                    )
                    pending.pop(self._sitting_id, None)
                    _put_record(data, key, record, pending)
                if big is not None:
                    entry = _entry_for_write(data, key)
                    rows = set(_stored_notices(entry)) | {big}
                    entry["notices"] = [
                        _notice_to_dict(n)
                        for n in sorted(rows, key=lambda n: (n.kind, n.subject, n.file, n.value, n.text))
                    ]
        except Exception as exc:  # noqa: BLE001 — a receipt must never raise into a channel
            log.debug("project_instructions_receipt_failed", error=str(exc))
            return
        self._awaiting = None
        self._start_line = None
        self._awaiting_too_large = None
        if aw is not None:
            self._had_record = True
        if aw is not None and self._dropped_digest is not None:
            self._marked = False
        self._dropped_digest = None


def open_project_instructions(
    root: Any,
    *,
    sitting_id: str,
    max_ratio: float,
    basis: str,
    no_input: bool,
    window_tokens: int,
    config_dir: Any = None,
) -> ProjectInstructions | None:
    """The handle for ``root``, or None when no instructions file resolves there (D-67).

    A default-use basis (``layer`` / ``trusted``) is demoted to ``awaiting_trust`` when the file
    is not owned by this account (D-68). The start-time pre-check runs before the handle is
    returned: it decides the start line, the summary entity and the too-large notice."""
    root_real = Path(root).resolve()
    candidate = first_candidate(root_real)
    if candidate is None:
        return None
    if basis in ("layer", "trusted") and hasattr(os, "geteuid"):
        try:
            owned = os.stat(candidate.real).st_uid == os.geteuid()
        except OSError:
            owned = False
        if not owned:
            basis = "awaiting_trust"
    handle = ProjectInstructions(
        root_real, sitting_id=sitting_id, max_ratio=max_ratio, basis=basis, no_input=no_input,
        config_dir=config_dir,
    )
    handle._candidate = candidate
    handle._last = InstructionsState(
        display_name(candidate.name), str(candidate.real), 0, None, "none"
    )
    handle._precheck(window_tokens)
    return handle


_DECLINED_REPLY = "instructions are not used in this session (the workspace was declined)"
_USAGE = "usage: /instructions off | on | show"


def run_instructions_command(handle: ProjectInstructions | None, arg: str) -> str:
    """The one ``/instructions`` handler the REPL and ACP share (D-59); the reply is the text.

    ``off`` / ``on`` / ``show`` only: no preview of the file and no diff of a change (D-01)."""
    if handle is None:
        return "no instructions file in this workspace"
    word = arg.strip().lower()
    if word not in ("off", "on", "show"):
        return _USAGE
    if word == "show":
        if not handle._declined:
            handle._refresh()
        st = handle.state
        return (
            f"{display_name(st.name)}: {st.path!r} · {format_kb(st.size)} · "
            f"sha256 {st.digest or '-'} · {st.state}"
        )
    if handle._declined:
        return _DECLINED_REPLY
    handle.set_state("off" if word == "off" else "yes")
    st = handle.state
    name = display_name(st.name)
    failure = handle._set_failure
    if failure == "no file":
        return "no instructions file in this workspace"
    if word == "off":
        if failure is not None:  # the answer could not be kept: say so rather than claim it
            return f"instructions not changed ({failure})"
        return f"instructions off for this workspace; /instructions on to use {name}"
    if failure is not None:
        return f"{name} not used this turn ({failure})"
    return f"using {name} ({format_kb(st.size)})"
