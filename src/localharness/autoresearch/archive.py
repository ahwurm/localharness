"""ArchiveStore: SQLite persistence for the autoresearch mutation archive (ARCH-01).

Near-direct clone of ``localharness.memory.sqlite``'s lifecycle: aiosqlite + WAL +
``PRAGMA foreign_keys = ON`` + ``PRAGMA user_version`` migrate-on-open + integrity_check.
The event bus is the source of truth; the SQLite archive is a projection. Every
``write()`` publishes a ``MutationArchived`` event when a bus is attached.
"""
from __future__ import annotations

import json
import time
import uuid  # noqa: F401  (reserved for callers/Phase 17 id minting; keeps parity with memory/sqlite.py)
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import aiosqlite

if TYPE_CHECKING:
    from localharness.core.bus import EventBus

# ---------------------------------------------------------------------------
# Data classes  (mirror Fact / FactQuery in memory/sqlite.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchiveEntry:
    id: str
    parent_id: str | None
    component: str
    diff: str                              # JSON {"before":..,"after":..}; decode via diff_decoded
    train_score: float | None
    train_scores_per_fixture: dict[str, float] | None
    holdout_score: float | None
    p_value: float | None
    cost: float | None
    ts: int
    approved_by: str | None
    status: str

    @property
    def diff_decoded(self) -> dict:
        return json.loads(self.diff)


@dataclass(frozen=True)
class ArchiveQuery:
    component: str | None = None
    status: str | None = None
    since_ts: int | None = None
    limit: int = 20


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CURRENT_SCHEMA_VERSION = 1

SCHEMA_V1_SQL = """
CREATE TABLE IF NOT EXISTS mutations (
    id                       TEXT    PRIMARY KEY,
    parent_id                TEXT    REFERENCES mutations(id) ON DELETE RESTRICT,
    component                TEXT    NOT NULL,
    diff                     TEXT    NOT NULL,
    train_score              REAL,
    train_scores_per_fixture TEXT,
    holdout_score            REAL,
    p_value                  REAL,
    cost                     REAL,
    ts                       INTEGER NOT NULL,
    approved_by              TEXT,
    status                   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mutations_component_ts ON mutations(component, ts DESC);
CREATE INDEX IF NOT EXISTS idx_mutations_parent ON mutations(parent_id);
CREATE INDEX IF NOT EXISTS idx_mutations_status_score ON mutations(status, train_score DESC);
CREATE TABLE IF NOT EXISTS mutation_approvals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    mutation_id TEXT    NOT NULL REFERENCES mutations(id) ON DELETE RESTRICT,
    approver    TEXT    NOT NULL,
    ts          INTEGER NOT NULL,
    comment     TEXT
);
CREATE INDEX IF NOT EXISTS idx_approvals_mutation ON mutation_approvals(mutation_id);
"""

# Columns selected wherever a full ArchiveEntry is reconstructed.
_ENTRY_COLUMNS = (
    "id, parent_id, component, diff, train_score, train_scores_per_fixture, "
    "holdout_score, p_value, cost, ts, approved_by, status"
)


# ---------------------------------------------------------------------------
# ArchiveStore
# ---------------------------------------------------------------------------


class ArchiveStore:
    """Project-local SQLite archive of every attempted mutation, with lineage.

    Owns ``.localharness/archive.db``. WAL durability, ``ON DELETE RESTRICT`` on the
    self-referential ``parent_id`` FK (preserve the lineage tree, DGM convention),
    and ``PRAGMA user_version`` migrate-on-open. Unlike ``MemoryStore`` the archive
    does NOT subscribe to the bus — ``bus`` is used only to publish ``MutationArchived``
    in ``write()``.
    """

    def __init__(self, db_path: str | Path, *, bus: Optional["EventBus"] = None) -> None:
        self._db_path = Path(db_path).expanduser()
        self._bus = bus
        self._db: Optional[aiosqlite.Connection] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def open(self) -> None:
        """Open SQLite connection, enable WAL + FK, apply pending migrations."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)  # lazy materialization
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode = WAL")
        await self._db.execute("PRAGMA synchronous = NORMAL")
        await self._db.execute("PRAGMA foreign_keys = ON")  # per-connection; ON DELETE RESTRICT no-ops without it
        await self._db.execute("PRAGMA temp_store = MEMORY")
        await self._apply_migrations()

    async def _apply_migrations(self) -> None:
        assert self._db is not None
        async with self._db.execute("PRAGMA user_version") as cur:
            row = await cur.fetchone()
            current_version = row[0]
        if current_version < CURRENT_SCHEMA_VERSION:
            await self._db.executescript(SCHEMA_V1_SQL)
            await self._db.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")
            await self._db.commit()

    async def close(self) -> None:
        """Close the SQLite connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def integrity_check(self) -> list[str]:
        """Run PRAGMA integrity_check + foreign_key_check. Returns errors (empty = healthy)."""
        assert self._db is not None
        errors: list[str] = []

        async with self._db.execute("PRAGMA integrity_check") as cur:
            rows = await cur.fetchall()
        for row in rows:
            if row[0] != "ok":
                errors.append(f"integrity_check: {row[0]}")

        async with self._db.execute("PRAGMA foreign_key_check") as cur:
            rows = await cur.fetchall()
        for row in rows:
            errors.append(f"foreign_key_check: {dict(row)}")

        return errors

    # ------------------------------------------------------------------
    # CRUD  (filled in Task 2)
    # ------------------------------------------------------------------

    async def write(self, entry: ArchiveEntry) -> ArchiveEntry:
        raise NotImplementedError

    async def get(self, id: str) -> ArchiveEntry | None:
        raise NotImplementedError

    async def query(self, q: ArchiveQuery) -> list[ArchiveEntry]:
        raise NotImplementedError

    async def add_approval(self, mutation_id: str, approver: str, comment: str | None = None) -> None:
        raise NotImplementedError

    async def lineage(self, id: str) -> list[ArchiveEntry]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_entry(row: aiosqlite.Row) -> ArchiveEntry:
    tspf_raw = row["train_scores_per_fixture"]
    tspf = json.loads(tspf_raw) if tspf_raw is not None else None
    return ArchiveEntry(
        id=row["id"],
        parent_id=row["parent_id"],
        component=row["component"],
        diff=row["diff"],
        train_score=row["train_score"],
        train_scores_per_fixture=tspf,
        holdout_score=row["holdout_score"],
        p_value=row["p_value"],
        cost=row["cost"],
        ts=row["ts"],
        approved_by=row["approved_by"],
        status=row["status"],
    )
