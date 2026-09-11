"""Durable "allow always" and "never here" answers, keyed by workspace realpath (PRD §3.3).

The store lives in the GLOBAL config dir — ``~/.localharness/grants.yaml`` — and NEVER in a
project tree. That is the whole security property: a cloned repo can add deny patterns, tools
and agents (things that *ask*), but it cannot ship a file that pre-approves its own
``curl … | sh``. Phase 39 already ruled this way for workspace trust
(``config/trust.py``); grants are the same threat, so they get the same shape: one YAML map
keyed by the resolved (realpath) workspace path, atomic writes, and a decision recorded only
when a human answered a prompt.

Nested workspaces inherit (the phase-39 "nested inherits" principle): a grant recorded for
``/p`` is honored in ``/p/sub``, because ``/p/sub`` is inside the project you already
answered for. The walk is upward only — a grant in a CHILD never leaks to the parent.

Provenance (``channel``, ``session_id``, ``granted_at`` / ``refused_at``) is mandatory per
PRD §3.3. A record missing any of it is not an answer anyone can audit, so it is skipped with a
warning rather than honored.

A "never here" answer lives in the same file as a NEGATIVE GRANT (``refusals:``), filed under
the same workspace key and the same key space as a grant — the shell signature, the directory,
the MCP tool the prompt actually named. It is not a text pattern: a refusal of the signature
``cp`` denies ``cp``, not ``scp``, ``cpio`` or every command whose arguments contain "cp".

Both answers are filed and matched under ``(class, key)``, never the key alone. Each ask class
has its OWN key space — a shell signature, a directory, a tool name — and the same string means
different things in different ones (``python_exec`` is both a shell signature and the
``code-exec`` tool). Matching the string alone let an answer to one question stand in for
another.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional

import yaml

from localharness.config.overlay import atomic_write_overlay
from localharness.config.paths import global_config_dir

if TYPE_CHECKING:  # pragma: no cover - typing only
    from localharness.agent.gate_types import Grant, Refusal

log = logging.getLogger(__name__)


def _record_type(name: str) -> type:
    """``gate_types.Grant`` / ``gate_types.Refusal``, imported on first use, not at import time.

    ``localharness.agent.__init__`` eagerly pulls in ``agent.loop`` → ``agent.gate`` →
    ``config.grants``, so importing ``agent.gate_types`` from this module's body made
    ``from localharness.config.grants import GrantStore`` fail in a fresh interpreter with a
    partially-initialised-module ImportError: the store could only be imported by someone who
    had already imported ``localharness.agent``. The test suite never saw it (conftest imports
    the agent package first), but any script, plugin or new test module reaching for the grant
    store first did. Deferring the import to call time breaks the cycle without moving the type,
    which stays where the contract puts it (``agent/gate_types.py``, the shared-types module).
    ``from __future__ import annotations`` keeps the annotations below string-only, so this is
    the only runtime reference.
    """
    import localharness.agent.gate_types as gate_types

    return getattr(gate_types, name)

GRANTS_FILE = "grants.yaml"
"""PRD §3.3: the file name inside the GLOBAL config dir. Sibling of
``trust.WORKSPACE_TRUST_FILE``, same directory, same realpath keying."""

GRANTS_KEY = "grants"
REFUSALS_KEY = "refusals"
"""The two lists a workspace entry may hold. "Never here" answers are written into the same file
(PRD §3.3) so one file is the whole memory of what a human answered — as a NEGATIVE GRANT in the
same key space, not as a text pattern: a refusal denies exactly the key the prompt offered."""

GRANT_REQUIRED_FIELDS: tuple[str, ...] = ("key", "class", "granted_at", "channel", "session_id")
"""PRD §3.3 record shape ``{key, class, granted_at, channel, session_id}``. Provenance is
mandatory: every one of these must be a non-empty string or the record is invalid."""

REFUSAL_REQUIRED_FIELDS: tuple[str, ...] = ("key", "class", "refused_at", "channel", "session_id")
"""Same shape and the same mandatory provenance for a "never here" answer, with ``refused_at``
in place of ``granted_at`` so a reader can tell the two lists apart at a glance."""

GRANTS_LOCK_SUFFIX = ".lock"
"""Sidecar advisory lock file next to the store (``grants.yaml.lock``).

A separate file, not the store itself, for two reasons: the store is replaced atomically by
rename (:func:`~localharness.config.overlay.atomic_write_overlay`), so a lock held on it would
be a lock on a file that no longer exists by the time the write lands; and the store may not
exist yet on the very first answer, while the lock always can."""

GRANTS_LOCK_WRITE_BUDGET_S = 0.5
"""How long one load-modify-write of the store may reasonably take, and the unit the lock's
other two numbers derive from.

It is a read of a small YAML file, a list edit, and an atomic rename — microseconds to
milliseconds on any disk. Half a second is that with room for a slow or contended filesystem
(NFS, a sync-heavy container mount), so a wait longer than this many multiples means something
is wrong rather than merely busy."""

GRANTS_LOCK_MAX_WAITERS = 20
"""How many sessions could plausibly be answering prompts against ONE global store at once — a
handful of terminals, a Zed window, a Discord instance, with headroom. It sets both the worst
case the timeout must cover and the granularity of the retry, so the two cannot drift apart."""

GRANTS_LOCK_TIMEOUT_S = GRANTS_LOCK_WRITE_BUDGET_S * GRANTS_LOCK_MAX_WAITERS
"""The longest this process waits for the lock: every plausible waiter taking its full budget."""

GRANTS_LOCK_RETRY_S = GRANTS_LOCK_WRITE_BUDGET_S / GRANTS_LOCK_MAX_WAITERS
"""How often to re-try the lock — fine enough that the whole queue drains inside one write
budget, coarse enough not to spin."""

GRANTS_LOCK_TIMEOUT_WARNING = (
    "grant store %s: could not take the write lock within %.1fs; writing anyway. A concurrent "
    "answer may be lost — check for a stale %s."
)
"""What a lock timeout says. It proceeds rather than raising: a human answered a prompt, and
losing that answer to a lock is worse than the race the lock exists to prevent — the failure
mode it guards is rare, and a refused write would be certain."""


def grants_store_path() -> Path:
    """Where the grant store lives (PRD §3.3).

    Always the GLOBAL dir, resolved at call time so a test's ``LOCALHARNESS_DIR`` takes effect —
    identical to ``trust.trust_store_path()`` on purpose.
    """
    return global_config_dir() / GRANTS_FILE


def _now() -> str:
    """ISO-8601 UTC, the ``granted_at`` format named in ``gate_types.Grant``."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _key(workspace: Path | str) -> str:
    """The realpath string a workspace is filed under (PRD §3.3; mirrors ``trust._key``)."""
    return str(Path(workspace).expanduser().resolve())


def _ancestors(workspace: Path | str) -> tuple[str, ...]:
    """The workspace key and every ancestor key, nearest first (PRD §3.3 "nested inherits")."""
    here = Path(workspace).expanduser().resolve()
    return tuple(str(p) for p in (here, *here.parents))


def _try_lock(handle: Any) -> bool:
    """One non-blocking attempt at an exclusive lock on ``handle``, on either platform.

    POSIX uses ``fcntl.flock``, which locks the open file DESCRIPTION — so two threads of one
    process holding two separate handles exclude each other, which ``fcntl.lockf`` (POSIX record
    locks, per-process) would not. Windows uses ``msvcrt.locking`` over one byte. Neither is
    imported at module scope: only one of them exists on any given machine.
    """
    try:
        import fcntl
    except ImportError:  # Windows
        import msvcrt

        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(handle: Any) -> None:
    """Release what :func:`_try_lock` took. Closing the handle would release it too; this is
    explicit so the release happens before the close, in the order a reader expects."""
    try:
        import fcntl
    except ImportError:  # Windows
        import msvcrt

        with suppress(OSError):
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    with suppress(OSError):
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold the store's write lock for one load-modify-write (PRD §3.3).

    ``add`` and ``add_refusal`` are read-modify-write over one shared file and nothing serialised
    them: twenty concurrent answers against one store kept ONE, because every writer loaded the
    same "before" state and the last rename won. A human's answer silently vanishing is the one
    failure a permission store cannot have — the next call asks again, or worse, a "never here"
    is not there any more.

    Cross-platform and advisory (:func:`_try_lock`), held only for the milliseconds of the
    edit, and it never blocks a write forever: after :data:`GRANTS_LOCK_TIMEOUT_S` it warns and
    proceeds, because losing the answer is worse than the race.
    """
    lock_path = path.with_name(path.name + GRANTS_LOCK_SUFFIX)
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "a+b")  # append mode: creates, never truncates
    except OSError:
        log.warning("grant store %s: could not open the write lock; writing anyway", path)
        yield
        return
    held = False
    try:
        deadline = time.monotonic() + GRANTS_LOCK_TIMEOUT_S
        while not (held := _try_lock(handle)):
            if time.monotonic() >= deadline:
                log.warning(
                    GRANTS_LOCK_TIMEOUT_WARNING, path, GRANTS_LOCK_TIMEOUT_S, lock_path
                )
                break
            time.sleep(GRANTS_LOCK_RETRY_S)
        yield
    finally:
        if held:
            _unlock(handle)
        handle.close()


def _valid(record: Any, required: tuple[str, ...]) -> bool:
    """A record is usable only when every provenance field is a non-empty string (PRD §3.3)."""
    if not isinstance(record, dict):
        return False
    return all(isinstance(record.get(f), str) and record[f].strip() for f in required)


class GrantStore:
    """Read/write the global grant store (PRD §3.3).

    ``path`` defaults to :func:`grants_store_path`; the parameter exists so tests and the ACP
    adapter can point at an explicit file. Nothing here ever consults a project tree, so a
    ``.localharness/grants.yaml`` committed to a repo is inert — it is simply never opened.

    The file is re-read on every call (like ``config/trust.py``): a grant written by a second
    session is visible to this one immediately, and a corrupt file degrades to "no grants"
    rather than a crashed turn.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path is not None else None

    @property
    def path(self) -> Path:
        """Resolved at access time so an env change (tests, ``--config-dir``) is honored."""
        return self._path if self._path is not None else grants_store_path()

    # ------------------------------------------------------------------ reads

    def _load(self) -> dict:
        path = self.path
        if not path.exists():
            return {}
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a corrupt store means "no grants", never a crash
            log.warning("grant store %s is unreadable; treating it as empty", path)
            return {}
        return data if isinstance(data, dict) else {}

    def _records(self, data: dict, workspace_key: str, list_key: str, required: tuple[str, ...]) -> list[dict]:
        entry = data.get(workspace_key)
        if not isinstance(entry, dict):
            return []
        raw = entry.get(list_key)
        if not isinstance(raw, list):
            return []
        out: list[dict] = []
        for record in raw:
            if _valid(record, required):
                out.append(record)
            else:
                log.warning(
                    "grant store %s: skipping invalid %s record for %s (missing provenance: %s)",
                    self.path, list_key, workspace_key, ", ".join(required),
                )
        return out

    def lookup(self, workspace: Path, klass: str, key: str) -> Optional[Grant]:
        """The grant covering ``(klass, key)`` in ``workspace``, walking ancestors (PRD §3.3).

        Nearest workspace wins; a parent's grant is inherited by a nested workspace. Returns
        None when nothing is recorded — the caller (``agent/verdict.evaluate``) then asks.

        The CLASS is half the identity. Keys are only unique within their own key space: the
        shell signature ``python_exec`` and the ``code-exec`` tool ``python_exec`` are the same
        string, and matching on the string alone let an "always" on the shell command satisfy the
        interpreter tool. A record is an answer to one question, so it is honored only for that
        question. A record with no ``class`` is invalid (:data:`GRANT_REQUIRED_FIELDS`) and is
        skipped with a warning — fail closed, ask again.
        """
        data = self._load()
        for workspace_key in _ancestors(workspace):
            for record in self._records(data, workspace_key, GRANTS_KEY, GRANT_REQUIRED_FIELDS):
                if record["key"] == key and record["class"] == klass:
                    return _record_type("Grant")(
                        key=record["key"],
                        klass=record["class"],
                        granted_at=record["granted_at"],
                        channel=record["channel"],
                        session_id=record["session_id"],
                        workspace=workspace_key,
                    )
        return None

    def refused(self, workspace: Path, klass: str, key: str) -> Optional[Refusal]:
        """The "never here" answer covering ``(klass, key)`` in ``workspace`` (PRD §3.3).

        The same key space and the same ancestor walk as :meth:`lookup` — a refusal is a
        negative grant. The caller (``agent/verdict.evaluate``) consults this BEFORE the grant,
        so a "never" wins over any later "always" on the same key and asks no more; and for a
        key that is a directory the caller walks the path upward too, so a refusal on ``/tmp/x``
        covers ``/tmp/x/y/f`` exactly as a directory grant covers its subtree.

        Matched on ``(class, key)`` for :meth:`lookup`'s reason: a key identifies a call only
        inside its own class's key space.
        """
        data = self._load()
        for workspace_key in _ancestors(workspace):
            for record in self._records(data, workspace_key, REFUSALS_KEY, REFUSAL_REQUIRED_FIELDS):
                if record["key"] == key and record["class"] == klass:
                    return _record_type("Refusal")(
                        key=record["key"],
                        klass=record["class"],
                        refused_at=record["refused_at"],
                        channel=record["channel"],
                        session_id=record["session_id"],
                        workspace=workspace_key,
                    )
        return None

    # ----------------------------------------------------------------- writes

    def _write(self, data: dict) -> None:
        atomic_write_overlay(self.path, data)

    def _record(self, workspace: Path | str, list_key: str, record: dict) -> None:
        """Load, replace-or-append one record, write — under the store's write lock.

        The lock is what makes "an answer is remembered" true when more than one session is
        answering prompts (:func:`_locked`); the load MUST happen inside it, because the bug was
        never the write, it was reading a "before" state someone else had already moved on from.

        Replaces any existing record with the same ``(key, class)`` in the same workspace, so
        re-answering refreshes provenance instead of growing the file.
        """
        workspace_key = _key(workspace)
        with _locked(self.path):
            data = self._load()
            entry = data.get(workspace_key)
            if not isinstance(entry, dict):
                entry = {}
            existing = entry.get(list_key)
            kept = [
                r for r in (existing if isinstance(existing, list) else [])
                if not (
                    isinstance(r, dict)
                    and r.get("key") == record["key"]
                    and r.get("class") == record["class"]
                )
            ]
            entry[list_key] = [*kept, record]
            data[workspace_key] = entry
            self._write(data)

    def add(self, grant: Grant) -> None:
        """Record an "allow always" answer (PRD §3.3)."""
        self._record(grant.workspace, GRANTS_KEY, {
            "key": grant.key,
            "class": grant.klass,
            "granted_at": grant.granted_at or _now(),
            "channel": grant.channel,
            "session_id": grant.session_id,
        })

    def add_refusal(self, refusal: Refusal) -> None:
        """Record a "never here" answer as a negative grant (PRD §3.3).

        Written into the same file, under the same workspace key, in the same key space as a
        grant — so "never run this command here" is stored as the command's own signature and
        denies that and nothing else. Replaces any existing refusal for the same key, so
        re-answering refreshes provenance instead of growing the file.

        A refusal wins forever after: ``agent/verdict.evaluate`` consults refusals ahead of
        grants and denies without prompting, which is the one answer a later "allow always"
        cannot undo from a prompt (editing ``grants.yaml`` is the escape hatch).
        """
        self._record(refusal.workspace, REFUSALS_KEY, {
            "key": refusal.key,
            "class": refusal.klass,
            "refused_at": refusal.refused_at or _now(),
            "channel": refusal.channel,
            "session_id": refusal.session_id,
        })


def new_grant(*, key: str, klass: str, workspace: Path | str, channel: str, session_id: str) -> Grant:
    """Build a :class:`Grant` with the current timestamp and a realpath workspace (PRD §3.3).

    The one constructor the channels use, so provenance can never be forgotten at a call site.
    """
    return _record_type("Grant")(
        key=key,
        klass=klass,
        granted_at=_now(),
        channel=channel,
        session_id=session_id,
        workspace=_key(workspace),
    )


def new_refusal(*, key: str, klass: str, workspace: Path | str, channel: str, session_id: str) -> Refusal:
    """Build a :class:`Refusal` with the current timestamp and a realpath workspace (PRD §3.3).

    :func:`new_grant`'s negative twin, and the only constructor the gate uses for a "never here"
    answer — same reason: provenance cannot be forgotten at a call site.
    """
    return _record_type("Refusal")(
        key=key,
        klass=klass,
        refused_at=_now(),
        channel=channel,
        session_id=session_id,
        workspace=_key(workspace),
    )
