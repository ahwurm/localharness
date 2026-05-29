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

# Sealed-slice invariant (Pitfall 3): holdout metrics must NEVER feed a Pareto
# front. pareto_front_2d rejects any metric in this set at the query layer so the
# seal is executable, not just documented. The holdout slice is sealed end-to-end.
_SEALED_COLUMNS = frozenset({"holdout_score"})


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
    # CRUD
    # ------------------------------------------------------------------

    async def write(self, entry: ArchiveEntry) -> ArchiveEntry:
        """INSERT the mutation, re-read it, then publish MutationArchived (after commit)."""
        assert self._db is not None
        tspf_json = (
            json.dumps(entry.train_scores_per_fixture)
            if entry.train_scores_per_fixture is not None
            else None
        )
        ts = entry.ts if entry.ts else int(time.time())
        await self._db.execute(
            """
            INSERT INTO mutations
                (id, parent_id, component, diff, train_score, train_scores_per_fixture,
                 holdout_score, p_value, cost, ts, approved_by, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id, entry.parent_id, entry.component, entry.diff,
                entry.train_score, tspf_json, entry.holdout_score, entry.p_value,
                entry.cost, ts, entry.approved_by, entry.status,
            ),
        )
        await self._db.commit()
        stored = await self.get(entry.id)
        assert stored is not None  # just inserted

        # Publish AFTER commit so the projection (DB) and the event agree.
        if self._bus is not None:
            from localharness.core.events import MutationArchived
            event = MutationArchived(
                mutation_id=stored.id,
                component=stored.component,
                status=stored.status,
                train_score=stored.train_score,
                holdout_score=stored.holdout_score,
                p_value=stored.p_value,
                cost=stored.cost,
                mutation_parent_id=stored.parent_id,
            )
            await self._bus.publish(event)  # never set seq; publish assigns it

        return stored

    async def get(self, id: str) -> ArchiveEntry | None:
        """Get a single mutation by full id. Returns None if not found."""
        assert self._db is not None
        async with self._db.execute(
            f"SELECT {_ENTRY_COLUMNS} FROM mutations WHERE id = ?",
            (id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return _row_to_entry(row)

    async def query(self, q: ArchiveQuery) -> list[ArchiveEntry]:
        """Query mutations with optional component/status/since_ts filters, ts DESC."""
        assert self._db is not None
        clauses: list[str] = []
        params: list[object] = []
        if q.component is not None:
            clauses.append("component = ?")
            params.append(q.component)
        if q.status is not None:
            clauses.append("status = ?")
            params.append(q.status)
        if q.since_ts is not None:
            clauses.append("ts >= ?")
            params.append(q.since_ts)

        sql = f"SELECT {_ENTRY_COLUMNS} FROM mutations"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(q.limit)

        async with self._db.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [_row_to_entry(r) for r in rows]

    async def add_approval(
        self, mutation_id: str, approver: str, comment: str | None = None
    ) -> None:
        """Append to mutation_approvals + update mutations.approved_by in ONE transaction.

        Append-only: prior approval rows are never deleted or overwritten.
        """
        assert self._db is not None
        now = int(time.time())
        await self._db.execute(
            "INSERT INTO mutation_approvals (mutation_id, approver, ts, comment) VALUES (?, ?, ?, ?)",
            (mutation_id, approver, now, comment),
        )
        await self._db.execute(
            "UPDATE mutations SET approved_by = ? WHERE id = ?",
            (approver, mutation_id),
        )
        await self._db.commit()

    async def update_verdict(
        self,
        mutation_id: str,
        *,
        status: str,
        train_score: float | None = None,
        train_scores_per_fixture: dict[str, float] | None = None,
        holdout_score: float | None = None,
        p_value: float | None = None,
        cost: float | None = None,
    ) -> "ArchiveEntry":
        """UPDATE a row's verdict (status + scores) in place, then re-publish MutationArchived.

        The Phase 17 experiment runner's write-back: in_flight → {promoted | train_rejected |
        holdout_rejected}. Only the verdict columns are touched; component/diff/parent_id/ts are
        immutable. train_scores_per_fixture MUST carry TRAIN scenario_name keys only (sealed-slice
        contract — pareto_front_per_fixture is holdout-blind; never pass holdout fixture rates here).
        Re-publishes MutationArchived AFTER commit so the event stream and the SQLite projection agree.
        """
        assert self._db is not None
        tspf = (
            json.dumps(train_scores_per_fixture)
            if train_scores_per_fixture is not None
            else None
        )
        await self._db.execute(
            "UPDATE mutations SET status=?, train_score=?, train_scores_per_fixture=?, "
            "holdout_score=?, p_value=?, cost=? WHERE id=?",
            (status, train_score, tspf, holdout_score, p_value, cost, mutation_id),
        )
        await self._db.commit()
        stored = await self.get(mutation_id)
        assert stored is not None  # caller guarantees the row exists (resolved before run)
        if self._bus is not None:
            from localharness.core.events import MutationArchived
            await self._bus.publish(
                MutationArchived(
                    mutation_id=stored.id,
                    component=stored.component,
                    status=stored.status,
                    train_score=stored.train_score,
                    holdout_score=stored.holdout_score,
                    p_value=stored.p_value,
                    cost=stored.cost,
                    mutation_parent_id=stored.parent_id,
                )
            )
        return stored

    async def lineage(self, id: str) -> list[ArchiveEntry]:
        """Walk parent_id to root. Returns chain child→...→root (git-log-oneline order)."""
        assert self._db is not None
        chain: list[ArchiveEntry] = []
        visited: set[str] = set()
        current = await self.get(id)
        while current is not None and current.id not in visited:
            visited.add(current.id)
            chain.append(current)
            if current.parent_id is None:
                break
            current = await self.get(current.parent_id)
        return chain

    # ------------------------------------------------------------------
    # Pareto fronts (ARCH-02)
    #
    # Both methods return the dumb non-dominated SET — no selection weights,
    # no sampling, no epsilon-greedy. Coverage-proportional exploration is
    # Phase 18's ParentSampler. Naive O(n^2) over a status-filtered fetch is
    # fine: the archive is low thousands of rows (CONTEXT).
    # ------------------------------------------------------------------

    async def pareto_front_per_fixture(self) -> list[ArchiveEntry]:
        """GEPA-style per-fixture front: every mutation best on >=1 train fixture.

        Sealed-slice (Pitfall 3): scoring reads ONLY ``train_scores_per_fixture``;
        ``holdout_score`` is never referenced here. The blob carries train
        scenario_name keys only (Phase 17 writer contract), so a holdout-best
        mutation cannot enter the front. Ties are INCLUDED — every candidate
        whose score equals the per-fixture max wins that fixture.
        Eligible status: ``promoted`` | ``in_flight``.
        """
        rows = await self.query(ArchiveQuery(limit=10_000))
        cands = [
            e
            for e in rows
            if e.status in ("promoted", "in_flight") and e.train_scores_per_fixture
        ]

        # Fixture universe = union of all candidates' train-fixture keys (train only).
        fixtures: set[str] = set()
        for e in cands:
            fixtures.update(e.train_scores_per_fixture.keys())

        winners: set[str] = set()
        for fx in fixtures:
            scored = [
                (e, e.train_scores_per_fixture[fx])
                for e in cands
                if fx in e.train_scores_per_fixture
            ]
            if not scored:
                continue
            best = max(score for _, score in scored)
            for e, score in scored:
                if score == best:  # ties included
                    winners.add(e.id)

        return [e for e in cands if e.id in winners]

    async def pareto_front_2d(
        self,
        metrics: list[str] | tuple[str, ...] = ("train_score", "cost"),
    ) -> list[ArchiveEntry]:
        """Global 2D non-dominated set: maximize ``train_score``, minimize ``cost``.

        Eligible rows: status ``promoted`` AND ``p_value < 0.05`` AND both
        ``train_score`` and ``cost`` present (sealed-slice + significance gate).

        Sealing teeth (validated BEFORE any DB access):
        1. Any metric in ``_SEALED_COLUMNS`` (``holdout_score``) raises ValueError —
           the executable form of the sealed-slice invariant (Pitfall 3).
        2. v1.1 supports only the (train_score, cost) pair; any other set raises
           ValueError. This is the forward-compat hook for richer fronts.
        """
        # Tooth 1 + forward-compat gate — fail before touching the DB.
        for m in metrics:
            if m in _SEALED_COLUMNS:
                raise ValueError(f"sealed column not allowed in Pareto metrics: {m}")
        if set(metrics) != {"train_score", "cost"}:
            raise ValueError(
                f"pareto_front_2d v1.1 supports only (train_score, cost), got {metrics}"
            )

        rows = await self.query(ArchiveQuery(status="promoted", limit=10_000))
        cands = [
            e
            for e in rows
            if e.p_value is not None
            and e.p_value < 0.05
            and e.train_score is not None
            and e.cost is not None
        ]

        def dominates(a: ArchiveEntry, b: ArchiveEntry) -> bool:
            return (
                a.train_score >= b.train_score
                and a.cost <= b.cost
                and (a.train_score > b.train_score or a.cost < b.cost)
            )

        return [
            b
            for b in cands
            if not any(dominates(a, b) for a in cands if a.id != b.id)
        ]


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
