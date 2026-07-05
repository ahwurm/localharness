"""MemoryStore: SQLite facts, sessions, FTS5, bus integration."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import aiosqlite

from localharness.memory.errors import (
    MemoryCorruptionError,
    MemoryVerifyError,
)
from localharness.memory.history import HistoryWriter
from localharness.memory.markdown import MarkdownMemory

if TYPE_CHECKING:
    from localharness.core.bus import EventBus, SubscriptionHandle
    from localharness.core.events import Action, Observation, UserMessage

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fact:
    key: str
    value: str
    agent_id: str
    division_id: str = ""
    org_id: str = ""
    tags: list[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = ""
    created_at: int = 0
    updated_at: int = 0
    expires_at: int | None = None
    # v2 (supersede-not-overwrite): 'active' | 'superseded'; superseded rows stay queryable
    # via get_fact_history / include_superseded but leave every hot path.
    status: str = "active"
    superseded_by: int | None = None
    # Episodic source pointer (session_id or origin) — hippocampal indexing (WRITE-04).
    provenance: str = ""
    id: int = 0
    # v3 (RANK): trust/accessibility split + ACT-R use-counters + graph node kind.
    retrieval_strength: float = 0.5
    importance: float = 0.0
    access_count: int = 0
    last_accessed_at: int | None = None
    node_kind: str = "fact"


@dataclass(frozen=True)
class FactQuery:
    text: str | None = None
    tags: list[str] = field(default_factory=list)
    min_confidence: float = 0.0
    include_scopes: list[str] = field(default_factory=lambda: ["agent"])
    limit: int = 50
    include_superseded: bool = False
    since: int | None = None   # epoch seconds, inclusive lower bound on facts.updated_at
    until: int | None = None   # epoch seconds, inclusive upper bound on facts.updated_at


@dataclass(frozen=True)
class ToolPrior:
    """A per-tool statistical prior from event history (COLL-01) — the context a
    surprise score is graded against. Computed in ONE indexed SQL aggregate; the
    None fields carry cold start honestly (no history -> no prediction)."""
    tool_name: str
    n: int                          # prior observation count (strictly earlier rows)
    error_rate: float | None        # AVG(is_error); None when n == 0
    lat_mean_ms: float | None
    lat_var_ms: float | None        # population variance
    lat_n: int
    size_mean: float | None
    size_var: float | None
    size_n: int


@dataclass(frozen=True)
class MemoryContext:
    agent_memory_md: str
    division_md: str
    guardrails_md: str
    fact_count: int
    token_estimate: int


# ---------------------------------------------------------------------------
# Phase 36 (SEMA-03/04): lesson-cluster "chapter" schema contract
# ---------------------------------------------------------------------------
# The store-side half of the chapter node: key prefix, tier tag, confidence tier, depth tag.
# DISTINCT from hierarchy.py's doc-analysis schema/doc/* gists (those are _GIST_CONFIDENCE=0.6,
# below the 0.7 line, a different feature that routes but never injects). A chapter is a
# PROMOTION over its member lessons: it must clear the 0.7 injection gate to render.
SCHEMA_KEY_PREFIX = "schema/cluster/"          # chapter nodes; never collides with schema/doc/*
SCHEMA_TIER_TAG = "tier:schema"
SCHEMA_CONFIDENCE = 0.8                          # == consolidation._PROMOTED_CONFIDENCE; >= 0.7 gate
SCHEMA_DEPTH_TAG_PREFIX = "depth:"              # depth:1 chapter-of-lessons, depth:2 chapter-of-chapters


def _schema_depth(tags: list[str]) -> int:
    """Read the depth:N tag (SEMA-03 depth cap). 0 = a plain lesson (no tag)."""
    for t in tags:
        if t.startswith(SCHEMA_DEPTH_TAG_PREFIX):
            try:
                return int(t[len(SCHEMA_DEPTH_TAG_PREFIX):])
            except ValueError:
                return 0
    return 0


# ---------------------------------------------------------------------------
# Phase 33.1 (ORCH-02): one-time root-agent rename (default -> orchestrator)
# ---------------------------------------------------------------------------
# The agent's name IS its storage identity (directory name + the agent_id column
# in facts/sessions + the bus filter), so the default->orchestrator rename must
# reconcile pre-existing 'default'-keyed data the first time the store opens under
# the new root name — otherwise every memory an existing install has is orphaned.
_LEGACY_ROOT_AGENT_ID = "default"
_ROOT_AGENT_ID = "orchestrator"


def _migrate_legacy_root_agent_dir(base_dir: Path, agent_id: str) -> None:
    """One-time, idempotent adoption of a pre-rename root store (Phase 33.1, ORCH-02).

    If this store is opening as the NEW root name and a legacy 'default' directory
    exists with no 'orchestrator' directory yet, adopt it wholesale: memory.db,
    MEMORY.md, history.jsonl, bus-events.jsonl, compact.md are all siblings in the
    same directory, so ONE atomic rename carries everything (WAL/SHM sidecars ride
    along too).

    Refuses (no-op) whenever the destination exists: never merge, never clobber a
    real 'orchestrator' agent's data — the legacy store then simply keeps opening
    under its old name (ORCH-03 collision rule). No-op for every non-root agent_id.
    """
    if (base_dir / "agents" / f"{_LEGACY_ROOT_AGENT_ID}.yaml").exists():
        # The YAML rename deletes default.yaml BEFORE any store opens; its lingering
        # presence means that rename REFUSED (the user owns an 'orchestrator' agent —
        # ORCH-03 collision) and the legacy root still lives under its old 'default'
        # name. Adopting its dir would graft the legacy root's memories into that
        # unrelated agent, falsifying the released "nothing is merged or overwritten"
        # guarantee. Never adopt while the collision-refusal marker is on disk.
        return
    if agent_id != _ROOT_AGENT_ID:
        return
    legacy_dir = base_dir / "agents" / _LEGACY_ROOT_AGENT_ID
    new_dir = base_dir / "agents" / _ROOT_AGENT_ID
    if not legacy_dir.is_dir() or new_dir.exists():
        return
    legacy_dir.rename(new_dir)
    # Honest paper trail for whoever debugs this store later (CLAUDE.md: docs for the
    # adversary): one schema-conformant session_event breadcrumb in the adopted
    # history.jsonl — carries all six HistoryWriter REQUIRED_FIELDS with a VALID_TYPES
    # type, and mirrors the existing session_event convention (v=1, integer ts) so a
    # later integrity_check()/read_all() never flags it as corruption.
    record = {
        "v": 1,
        "type": "session_event",
        "id": str(uuid.uuid4()),
        "session_id": "phase-33.1-migration",
        "agent_id": _ROOT_AGENT_ID,
        "ts": int(time.time()),
        "event": "agent_renamed",
        "from_agent_id": _LEGACY_ROOT_AGENT_ID,
        "to_agent_id": _ROOT_AGENT_ID,
    }
    with (new_dir / "history.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CURRENT_SCHEMA_VERSION = 4

# v1 kept verbatim: the v1→v2 migration test builds a v1 DB from this exact script.
SCHEMA_V1_SQL = """
CREATE TABLE IF NOT EXISTS facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    TEXT    NOT NULL,
    division_id TEXT    NOT NULL DEFAULT '',
    org_id      TEXT    NOT NULL DEFAULT '',
    key         TEXT    NOT NULL,
    value       TEXT    NOT NULL,
    tags        TEXT    NOT NULL DEFAULT '[]',
    confidence  REAL    NOT NULL DEFAULT 1.0,
    source      TEXT    NOT NULL DEFAULT '',
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL,
    expires_at  INTEGER,
    UNIQUE(agent_id, key)
);
CREATE INDEX IF NOT EXISTS idx_facts_agent_id ON facts(agent_id);
CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(agent_id, key);
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    key, value, tags,
    content=facts, content_rowid=id
);
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, key, value, tags) VALUES (new.id, new.key, new.value, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, key, value, tags) VALUES ('delete', old.id, old.key, old.value, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, key, value, tags) VALUES ('delete', old.id, old.key, old.value, old.tags);
    INSERT INTO facts_fts(rowid, key, value, tags) VALUES (new.id, new.key, new.value, new.tags);
END;
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT    PRIMARY KEY,
    agent_id        TEXT    NOT NULL,
    division_id     TEXT    NOT NULL DEFAULT '',
    org_id          TEXT    NOT NULL DEFAULT '',
    started_at      INTEGER NOT NULL,
    ended_at        INTEGER,
    turn_count      INTEGER NOT NULL DEFAULT 0,
    action_count    INTEGER NOT NULL DEFAULT 0,
    tokens_in       INTEGER NOT NULL DEFAULT 0,
    tokens_out      INTEGER NOT NULL DEFAULT 0,
    exit_reason     TEXT,
    summary         TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_agent_id ON sessions(agent_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
"""

# ---------------------------------------------------------------------------
# Schema v2 — supersede-not-overwrite (WRITE-02) + provenance (WRITE-04).
# The v1 UNIQUE(agent_id, key) is replaced by a PARTIAL unique index on ACTIVE
# rows only, so a superseded row can share its successor's key while the active
# tier keeps one-truth-per-key. The partial indexes are also the RANK-05
# hot-path guarantee: default retrieval never scans superseded rows.
# ---------------------------------------------------------------------------

_FACTS_TABLE_V2 = """
CREATE TABLE IF NOT EXISTS facts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id      TEXT    NOT NULL,
    division_id   TEXT    NOT NULL DEFAULT '',
    org_id        TEXT    NOT NULL DEFAULT '',
    key           TEXT    NOT NULL,
    value         TEXT    NOT NULL,
    tags          TEXT    NOT NULL DEFAULT '[]',
    confidence    REAL    NOT NULL DEFAULT 1.0,
    source        TEXT    NOT NULL DEFAULT '',
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL,
    expires_at    INTEGER,
    status        TEXT    NOT NULL DEFAULT 'active',
    superseded_by INTEGER,
    provenance    TEXT    NOT NULL DEFAULT ''
);
"""

_FACTS_INDEXES_V2 = """
CREATE UNIQUE INDEX IF NOT EXISTS ux_facts_active_key ON facts(agent_id, key) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_facts_agent_id ON facts(agent_id);
CREATE INDEX IF NOT EXISTS idx_facts_active_recency ON facts(agent_id, updated_at DESC) WHERE status = 'active';
"""

_FACTS_FTS_AND_TRIGGERS = """
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    key, value, tags,
    content=facts, content_rowid=id
);
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, key, value, tags) VALUES (new.id, new.key, new.value, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, key, value, tags) VALUES ('delete', old.id, old.key, old.value, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, key, value, tags) VALUES ('delete', old.id, old.key, old.value, old.tags);
    INSERT INTO facts_fts(rowid, key, value, tags) VALUES (new.id, new.key, new.value, new.tags);
END;
"""

_SESSIONS_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT    PRIMARY KEY,
    agent_id        TEXT    NOT NULL,
    division_id     TEXT    NOT NULL DEFAULT '',
    org_id          TEXT    NOT NULL DEFAULT '',
    started_at      INTEGER NOT NULL,
    ended_at        INTEGER,
    turn_count      INTEGER NOT NULL DEFAULT 0,
    action_count    INTEGER NOT NULL DEFAULT 0,
    tokens_in       INTEGER NOT NULL DEFAULT 0,
    tokens_out      INTEGER NOT NULL DEFAULT 0,
    exit_reason     TEXT,
    summary         TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_agent_id ON sessions(agent_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
"""

SCHEMA_V2_SQL = _FACTS_TABLE_V2 + _FACTS_INDEXES_V2 + _FACTS_FTS_AND_TRIGGERS + _SESSIONS_SQL

# In-place v1→v2 rebuild: SQLite cannot drop a UNIQUE table constraint, so the table is
# rebuilt (rename → recreate → copy → drop), triggers/indexes recreated, FTS re-synced.
# CRASH-SAFE (Phase-29 critic M1): the whole script is ONE transaction that stamps
# user_version as its last statement — a crash anywhere rolls back to intact v1 and the
# next open() retries cleanly; a crash after COMMIT never re-runs (version stamped).
MIGRATION_V1_TO_V2_SQL = (
    """
BEGIN IMMEDIATE;
DROP TRIGGER IF EXISTS facts_ai;
DROP TRIGGER IF EXISTS facts_ad;
DROP TRIGGER IF EXISTS facts_au;
DROP INDEX IF EXISTS idx_facts_agent_id;
DROP INDEX IF EXISTS idx_facts_key;
ALTER TABLE facts RENAME TO facts_v1_old;
"""
    + _FACTS_TABLE_V2
    + """
INSERT INTO facts (id, agent_id, division_id, org_id, key, value, tags, confidence, source,
                   created_at, updated_at, expires_at, status, superseded_by, provenance)
    SELECT id, agent_id, division_id, org_id, key, value, tags, confidence, source,
           created_at, updated_at, expires_at, 'active', NULL, ''
    FROM facts_v1_old;
DROP TABLE facts_v1_old;
"""
    + _FACTS_INDEXES_V2
    + _FACTS_FTS_AND_TRIGGERS
    + """
INSERT INTO facts_fts(facts_fts) VALUES('rebuild');
PRAGMA user_version = 2;
COMMIT;
"""
)

# ---------------------------------------------------------------------------
# Schema v3 — activation scoring (RANK-01..05). Additive:
# - ACT-R columns: access_count/last_accessed_at (BASE — the injected block's ordering
#   reads ONLY these) + *_staged twins (reads bump staging ONLY; folded at consolidation
#   boundaries so the injected block is byte-stable between consolidations, RANK-04).
# - confidence split (RANK-03): confidence stays trust (stable); retrieval_strength is
#   accessibility (decays with disuse — decay itself lands with consolidation, Phase 31;
#   supersede drops it immediately); importance is the write-time tag-heuristic prior.
# - typed graph (RANK-01): facts rows ARE the nodes (node_kind: fact|gist|schema);
#   edges(src,dst,kind) carries derived_from/member_of/supports/contradicts.
#   `supersedes` stays a facts column — it is the hot-path mechanism, not an edge.
# - facts_au trigger narrowed to indexed columns so activation bumps never churn FTS.
# ---------------------------------------------------------------------------

MIGRATION_V2_TO_V3_SQL = """
BEGIN IMMEDIATE;
ALTER TABLE facts ADD COLUMN retrieval_strength REAL NOT NULL DEFAULT 0.5;
ALTER TABLE facts ADD COLUMN importance REAL NOT NULL DEFAULT 0.0;
ALTER TABLE facts ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE facts ADD COLUMN last_accessed_at INTEGER;
ALTER TABLE facts ADD COLUMN access_count_staged INTEGER NOT NULL DEFAULT 0;
ALTER TABLE facts ADD COLUMN last_accessed_staged INTEGER;
ALTER TABLE facts ADD COLUMN node_kind TEXT NOT NULL DEFAULT 'fact';
CREATE TABLE IF NOT EXISTS edges (
    src_id     INTEGER NOT NULL,
    dst_id     INTEGER NOT NULL,
    kind       TEXT    NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (src_id, dst_id, kind)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id);
DROP TRIGGER IF EXISTS facts_au;
CREATE TRIGGER facts_au AFTER UPDATE OF key, value, tags ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, key, value, tags) VALUES ('delete', old.id, old.key, old.value, old.tags);
    INSERT INTO facts_fts(rowid, key, value, tags) VALUES (new.id, new.key, new.value, new.tags);
END;
PRAGMA user_version = 3;
COMMIT;
"""

# ---------------------------------------------------------------------------
# Schema v4 — the collect-only predictive-gate substrate (Phase 34, COLL-01..04).
# ADDITIVE ONLY: four new tables, zero touches to facts/sessions/edges — so the
# ambient injected block (_render_memory_index reads ONLY facts/sessions) is
# byte-stable by construction, not by discipline. One BEGIN IMMEDIATE ...
# PRAGMA user_version = 4; COMMIT transaction (critic M1: crash -> rollback to
# intact v3 -> clean retry), matching the v2->v3 additive precedent above.
#
# Column semantics (the schema contract plans 34-03/04/07 build against):
# - tool_observations: one row per scored Observation — the substrate for the
#   pure-SQL per-tool priors. `is_error` derives from `Observation.error IS NOT
#   NULL` (exit_code is a dead field — 100% null in production, Pitfall 1);
#   `output_len` is len of the ALREADY-CAPPED output (200 == ">=200", Pitfall 6);
#   `duration_ms` is the Action->Observation timestamp delta (zero loop
#   instrumentation); `event_id` is the source bus event's id for idempotent
#   re-ingestion (INSERT OR IGNORE); `source` in ('live','backfill').
# - surprise_scores: COLL-04's persisted SurpriseScored. `expectation_json`
#   snapshots the exact prior that produced the score (Phase 35 re-derives
#   thresholds offline under any windowing); `quadrant` in ('routine',
#   'surprising_failure','unsurprising_failure','quiet_surprise','cold_start').
# - user_signals: COLL-02's zero-NLU labels. `signal_type` in ('correction',
#   'confirmation','interruption'); `trigger_family` in ('negation',
#   'correction_phrase','frustration','reask','confirmation','interruption');
#   `user_message` stored in FULL (owner steer: look-ready records).
# - staged_snapshots: COLL-03's credit-assignment candidates. `candidate_type`
#   in ('bump','suspect').
# ---------------------------------------------------------------------------

MIGRATION_V3_TO_V4_SQL = """
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS tool_observations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id      TEXT    NOT NULL,
    session_id    TEXT    NOT NULL,
    tool_call_id  TEXT,
    tool_name     TEXT    NOT NULL,
    ts            INTEGER NOT NULL,
    is_error      INTEGER NOT NULL,
    output_len    INTEGER,
    duration_ms   INTEGER,
    event_id      TEXT    UNIQUE,
    source        TEXT    NOT NULL DEFAULT 'live'
);
CREATE INDEX IF NOT EXISTS idx_tool_obs_tool ON tool_observations(agent_id, tool_name, ts);
CREATE TABLE IF NOT EXISTS surprise_scores (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id         TEXT    NOT NULL,
    session_id       TEXT    NOT NULL,
    observation_id   INTEGER REFERENCES tool_observations(id),
    expectation_json TEXT,
    score            REAL    NOT NULL,
    quadrant         TEXT,
    scored_at        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_surprise_scores_agent ON surprise_scores(agent_id, scored_at);
CREATE TABLE IF NOT EXISTS user_signals (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id               TEXT    NOT NULL,
    session_id             TEXT    NOT NULL,
    ts                     INTEGER NOT NULL,
    signal_type            TEXT    NOT NULL,
    trigger_family         TEXT,
    matched_text           TEXT,
    user_message           TEXT    NOT NULL,
    corrected_turn_summary TEXT,
    event_id               TEXT    UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_user_signals_agent ON user_signals(agent_id, ts);
CREATE TABLE IF NOT EXISTS staged_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_signal_id  INTEGER NOT NULL REFERENCES user_signals(id),
    fact_key        TEXT    NOT NULL,
    fact_id         INTEGER,
    candidate_type  TEXT    NOT NULL,
    captured_at     INTEGER NOT NULL
);
PRAGMA user_version = 4;
COMMIT;
"""


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

class MemoryStore:
    """
    Three-tier persistent memory for a single agent.

    Owns the agent's memory.db, history.jsonl, and MEMORY.md.
    Reads (but never writes) division and org memory for context injection.
    Optionally subscribes to an EventBus for auto-diary recording.
    """

    def __init__(
        self,
        agent_id: str,
        division_id: str,
        org_id: str,
        base_dir: str,
        *,
        bus: Optional["EventBus"] = None,
    ) -> None:
        self._agent_id = agent_id
        self._division_id = division_id
        self._org_id = org_id
        self._base_dir = Path(base_dir).expanduser()

        # Agent paths
        self._agent_dir = self._base_dir / "agents" / agent_id
        self._db_path = self._agent_dir / "memory.db"
        self._history_path = self._agent_dir / "history.jsonl"
        self._notes_path = self._agent_dir / "MEMORY.md"

        # Division / org paths (read-only)
        self._division_dir = self._base_dir / "divisions" / division_id
        self._division_md_path = self._division_dir / "DIVISION.md"
        self._org_dir = self._base_dir / "orgs" / org_id
        self._guardrails_path = self._org_dir / "GUARDRAILS.md"

        self._history_writer = HistoryWriter(self._history_path)
        self._markdown_memory = MarkdownMemory(self._notes_path)
        self._bus = bus
        self._db: Optional[aiosqlite.Connection] = None
        self._subscription_handles: list["SubscriptionHandle"] = []
        # Live session id — the default provenance stamped on writes (WRITE-04).
        self._current_session_id: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def open(self) -> None:
        """Open SQLite connection, enable WAL mode, apply pending migrations.

        Phase 33.1 (ORCH-02): performs a one-time, idempotent root-rename migration
        first — a pre-rename 'default' store is adopted (directory + facts/sessions
        rows re-keyed) the first time it opens as 'orchestrator'.

        Critic M2: any failure after connect closes the connection before re-raising —
        aiosqlite's worker thread is non-daemon, and a leaked handle hangs process exit.
        """
        # Phase 33.1: must run before mkdir — mkdir would create an empty
        # agents/orchestrator/ first and the adoption rename would then refuse
        # (destination exists), silently orphaning the legacy store.
        _migrate_legacy_root_agent_dir(self._base_dir, self._agent_id)
        self._agent_dir.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        try:
            await self._open_inner()
        except BaseException:
            db, self._db = self._db, None
            try:
                await db.close()
            except Exception:
                pass
            raise

    async def _open_inner(self) -> None:
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode = WAL")
        await self._db.execute("PRAGMA synchronous = NORMAL")
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._db.execute("PRAGMA temp_store = MEMORY")
        # Overlapping writers (agent loop vs idle consolidation) wait instead of throwing
        # "database is locked" (critic CONS-02 groundwork).
        await self._db.execute("PRAGMA busy_timeout = 5000")
        # Activation scoring as registered scalar functions: zero-token ranking (RANK-02)
        # without depending on SQLite being compiled with math functions.
        await self._db.create_function("lh_slow_score", 5, _slow_score, deterministic=True)
        await self._db.create_function("lh_fused_score", 7, _fused_score, deterministic=True)
        await self._apply_migrations()

        if self._bus is not None:
            from localharness.core.events import Action, Observation, UserMessage
            self._subscription_handles.append(
                self._bus.subscribe(Action, self._on_action, agent_id=self._agent_id)
            )
            self._subscription_handles.append(
                self._bus.subscribe(Observation, self._on_observation, agent_id=self._agent_id)
            )
            self._subscription_handles.append(
                self._bus.subscribe(UserMessage, self._on_user_message, agent_id=self._agent_id)
            )

    async def _apply_migrations(self) -> None:
        """Stepwise ladder; each rewrite script is a single transaction that stamps
        user_version itself (critic M1: crash → rollback → clean retry; never a
        half-migrated DB, never a double-run)."""
        assert self._db is not None

        async def _version() -> int:
            async with self._db.execute("PRAGMA user_version") as cur:
                row = await cur.fetchone()
            return row[0]

        v = await _version()
        if v == 0:
            # Fresh DB: idempotent DDL, so stamping after is safe (a crash between
            # script and stamp re-runs harmlessly thanks to IF NOT EXISTS).
            await self._db.executescript(SCHEMA_V2_SQL)
            await self._db.execute("PRAGMA user_version = 2")
            await self._db.commit()
            v = 2
        if v == 1:
            await self._db.executescript(MIGRATION_V1_TO_V2_SQL)
            v = await _version()
        if v == 2:
            await self._db.executescript(MIGRATION_V2_TO_V3_SQL)
            v = await _version()
        if v == 3:
            await self._db.executescript(MIGRATION_V3_TO_V4_SQL)
            v = await _version()

        # Phase 33.1 (ORCH-02): one-time root-rename row fixup. Directory adoption alone
        # is NOT enough — every read filters WHERE agent_id = ?, so rows stamped 'default'
        # are invisible to a store opened as 'orchestrator'. Idempotent (matches 0 rows
        # once migrated), scoped to the root store only, and both tables commit in ONE
        # transaction (critic M1: crash -> rollback -> clean retry, never a half-migrated
        # identity). No unique-index conflict is possible: a store directory only ever
        # contains its own agent's rows, so 'default' and 'orchestrator' rows never coexist.
        if self._agent_id == _ROOT_AGENT_ID:
            await self._db.execute(
                "UPDATE facts SET agent_id = ? WHERE agent_id = ?",
                (_ROOT_AGENT_ID, _LEGACY_ROOT_AGENT_ID),
            )
            await self._db.execute(
                "UPDATE sessions SET agent_id = ? WHERE agent_id = ?",
                (_ROOT_AGENT_ID, _LEGACY_ROOT_AGENT_ID),
            )
            await self._db.commit()

    async def close(self) -> None:
        """Unsubscribe from bus, close SQLite connection."""
        if self._bus is not None:
            for handle in self._subscription_handles:
                self._bus.unsubscribe(handle)
            self._subscription_handles.clear()
        if self._db is not None:
            await self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Facts CRUD
    # ------------------------------------------------------------------

    def set_current_session(self, session_id: str | None) -> None:
        """Record the live session id — the default provenance for writes (WRITE-04)."""
        self._current_session_id = session_id

    async def store_fact(
        self,
        key: str,
        value: str,
        tags: list[str] | None = None,
        confidence: float = 1.0,
        source: str = "",
        expires_at: int | None = None,
        provenance: str | None = None,
        node_kind: str = "fact",
        _retried: bool = False,
    ) -> Fact:
        """Write a fact with supersede-not-overwrite semantics (WRITE-01/02/04).

        - No active row for `key` → insert a new active row.
        - Active row with the IDENTICAL value → corroboration touch (updated_at bumped,
          confidence = max(old, new)); no duplicate row.
        - Active row with a DIFFERENT value → the old row is marked superseded
          (status='superseded', superseded_by=<new id>) and a fresh active row is
          inserted. Nothing is overwritten or deleted; history stays queryable via
          get_fact_history / FactQuery(include_superseded=True).

        Every write is READ-BACK-VERIFIED: the active row is re-read and compared before
        the write is claimed; a mismatch raises MemoryVerifyError (the Cline
        "claims-to-write-but-didn't" class).
        """
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {confidence}")
        assert self._db is not None
        now = int(time.time())
        tags_json = json.dumps(tags or [])
        prov = provenance if provenance is not None else (self._current_session_id or "")

        existing = await self._get_fact_row(key)
        if existing is not None and existing.value == value:
            # Corroboration: same claim re-asserted — strengthen, don't duplicate.
            # expires_at/tags/node_kind follow the NEW call (critics 29-m2 + 32-m1:
            # the branch must not silently ignore caller-supplied metadata — same
            # contract as the supersede path, minus the new row).
            await self._db.execute(
                "UPDATE facts SET updated_at = ?, confidence = MAX(confidence, ?), "
                "expires_at = ?, tags = ?, node_kind = ?, "
                "source = CASE WHEN ? = '' THEN source ELSE ? END "
                "WHERE agent_id = ? AND key = ? AND status = 'active'",
                (now, confidence, expires_at, tags_json, node_kind,
                 source, source, self._agent_id, key),
            )
            await self._db.commit()
        else:
            if existing is not None:
                # Supersede: vacate the active-unique slot, insert successor, then link.
                # The loser's retrieval_strength drops immediately — interference: the old
                # memory loses the retrieval competition, it is not erased (RANK-03).
                await self._db.execute(
                    "UPDATE facts SET status = 'superseded', updated_at = ?, "
                    "retrieval_strength = MIN(retrieval_strength, 0.1) "
                    "WHERE agent_id = ? AND key = ? AND status = 'active'",
                    (now, self._agent_id, key),
                )
            try:
                cur = await self._db.execute(
                    "INSERT INTO facts (agent_id, division_id, org_id, key, value, tags, confidence, "
                    "source, created_at, updated_at, expires_at, status, provenance, importance, node_kind) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)",
                    (self._agent_id, self._division_id, self._org_id, key, value,
                     tags_json, confidence, source, now, now, expires_at, prov,
                     _importance_prior(tags or [], source), node_kind),
                )
            except sqlite3.IntegrityError:
                # Critic m4: two concurrent writers raced past the existence check (the
                # partial-unique index is the backstop). Retry once — the loser now sees
                # the winner's row and corroborates/supersedes normally.
                await self._db.rollback()
                if _retried:
                    raise
                return await self.store_fact(
                    key, value, tags=tags, confidence=confidence, source=source,
                    expires_at=expires_at, provenance=provenance, node_kind=node_kind,
                    _retried=True,
                )
            new_id = cur.lastrowid
            if existing is not None:
                await self._db.execute(
                    "UPDATE facts SET superseded_by = ? "
                    "WHERE agent_id = ? AND key = ? AND status = 'superseded' AND superseded_by IS NULL",
                    (new_id, self._agent_id, key),
                )
            await self._db.commit()

        fact = await self._get_fact_row(key)
        if fact is None or fact.value != value:
            raise MemoryVerifyError(key)
        return fact

    _FACT_COLS = (
        "key, value, agent_id, division_id, org_id, tags, confidence, source, "
        "created_at, updated_at, expires_at, status, superseded_by, provenance, id, "
        "retrieval_strength, importance, access_count, last_accessed_at, node_kind"
    )

    async def _get_fact_row(self, key: str) -> Fact | None:
        """Read the ACTIVE fact row without expiry check (internal use)."""
        assert self._db is not None
        async with self._db.execute(
            f"SELECT {self._FACT_COLS} FROM facts "
            "WHERE agent_id = ? AND key = ? AND status = 'active'",
            (self._agent_id, key),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return _row_to_fact(row)

    async def get_fact(self, key: str) -> Fact | None:
        """Get the active fact by exact key. Returns None if not found, expired, or superseded."""
        assert self._db is not None
        now = int(time.time())
        async with self._db.execute(
            f"SELECT {self._FACT_COLS} FROM facts "
            "WHERE agent_id = ? AND key = ? AND status = 'active' "
            "AND (expires_at IS NULL OR expires_at > ?)",
            (self._agent_id, key, now),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return _row_to_fact(row)

    async def get_facts_by_ids(self, ids: list[int]) -> list[Fact]:
        """Resolve graph-walk node ids back to ACTIVE facts (HIER-03), preserving input
        order. Active-only (whole-milestone critic B2): the graph may traverse THROUGH
        superseded nodes, but a hot path must never RENDER one — a stale gist presented
        with a current gist's authority is the misattribution class this project's
        number-net exists to catch. Explicit history stays on get_fact_history."""
        if not ids:
            return []
        assert self._db is not None
        qmarks = ",".join("?" * len(ids))
        async with self._db.execute(
            f"SELECT {self._FACT_COLS} FROM facts "
            f"WHERE agent_id = ? AND status = 'active' AND id IN ({qmarks})",
            [self._agent_id, *ids],
        ) as cur:
            rows = await cur.fetchall()
        by_id = {f.id: f for f in (_row_to_fact(r) for r in rows)}
        return [by_id[i] for i in ids if i in by_id]

    async def get_fact_history(self, key: str) -> list[Fact]:
        """All versions of a fact, newest first — the explicit-request path to the past
        (WRITE-02: supersede keeps history retrievable; nothing is ever silently lost)."""
        assert self._db is not None
        async with self._db.execute(
            f"SELECT {self._FACT_COLS} FROM facts "
            "WHERE agent_id = ? AND key = ? ORDER BY created_at DESC, id DESC",
            (self._agent_id, key),
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_fact(r) for r in rows]

    async def delete_fact(self, key: str) -> bool:
        """Hard-DELETE a fact by key. Returns True if a row was deleted.

        ⚠️ v2.0 (critic m7): this contradicts supersede-never-delete — it exists for
        explicit user-initiated removal only. Harness code must use store_fact
        (supersede) instead; no production path calls this."""
        assert self._db is not None
        async with self._db.execute(
            "DELETE FROM facts WHERE agent_id = ? AND key = ?",
            (self._agent_id, key),
        ) as cur:
            deleted = cur.rowcount > 0
        await self._db.commit()
        return deleted

    async def query_facts(self, query: FactQuery) -> list[Fact]:
        """Query facts with optional FTS5, tag filter, min_confidence, expiry filter."""
        assert self._db is not None
        now = int(time.time())

        status_filter = "" if query.include_superseded else "AND f.status = 'active'"
        # Temporal window on updated_at (34-05): built once, interpolated into BOTH branches
        # after {status_filter}; a fact touched by supersede/consolidation in the window is
        # temporally relevant. Tool-output concern only — the ambient render never reads it.
        temporal_filter = ""
        temporal_params: list[Any] = []
        if query.since is not None:
            temporal_filter += " AND f.updated_at >= ?"
            temporal_params.append(query.since)
        if query.until is not None:
            temporal_filter += " AND f.updated_at <= ?"
            temporal_params.append(query.until)
        fts_text = _sanitize_fts_query(query.text) if query.text else ""
        if query.text and not fts_text:
            # Critic m1: a query that sanitizes to nothing must NOT fall back to the
            # recency listing — unrelated facts would render as "matches".
            # (Same contract as origin/main's 2c7e712 inline quoting, which this
            # sanitizer supersedes — main's hyphen/colon regression tests apply.)
            return []
        prefixed_cols = ", ".join(f"f.{c}" for c in self._FACT_COLS.split(", "))

        # Tool-path ranking is the FULL fused score — fresh, staged counters included,
        # BM25 relevance + ln(confidence) precision term (RANK-04: only the tool result,
        # appended after the prefix cache, may re-rank freely on every call).
        if fts_text:
            sql = f"""
                SELECT {prefixed_cols}
                FROM facts f
                JOIN facts_fts ON facts_fts.rowid = f.id
                WHERE facts_fts MATCH ?
                  AND f.agent_id = ?
                  AND f.confidence >= ?
                  AND (f.expires_at IS NULL OR f.expires_at > ?)
                  {status_filter}
                  {temporal_filter}
                ORDER BY lh_fused_score(
                    f.importance,
                    f.access_count + f.access_count_staged,
                    COALESCE(f.last_accessed_staged, f.last_accessed_at),
                    f.updated_at, ?, f.confidence, rank) DESC
                LIMIT ?
            """
            params: list[Any] = [fts_text, self._agent_id, query.min_confidence, now, *temporal_params, now, query.limit]
        else:
            sql = f"""
                SELECT {prefixed_cols}
                FROM facts f
                WHERE f.agent_id = ?
                  AND f.confidence >= ?
                  AND (f.expires_at IS NULL OR f.expires_at > ?)
                  {status_filter}
                  {temporal_filter}
                ORDER BY lh_fused_score(
                    f.importance,
                    f.access_count + f.access_count_staged,
                    COALESCE(f.last_accessed_staged, f.last_accessed_at),
                    f.updated_at, ?, f.confidence, 0.0) DESC
                LIMIT ?
            """
            params = [self._agent_id, query.min_confidence, now, *temporal_params, now, query.limit]

        async with self._db.execute(sql, params) as cur:
            rows = await cur.fetchall()

        facts = [_row_to_fact(r) for r in rows]

        if query.tags:
            facts = [f for f in facts if any(t in f.tags for t in query.tags)]

        return facts

    # ------------------------------------------------------------------
    # Activation staging + fold (RANK-02/04)
    # ------------------------------------------------------------------

    async def touch_staged(self, keys: list[str]) -> None:
        """Record reads into the STAGING columns only — never anything the injected
        block's ordering consumes, so a plain read can never void the prefix cache
        (RANK-04, the 2026-07-02 critic's staging discipline)."""
        if not keys:
            return
        assert self._db is not None
        now = int(time.time())
        await self._db.executemany(
            "UPDATE facts SET access_count_staged = access_count_staged + 1, "
            "last_accessed_staged = ? "
            "WHERE agent_id = ? AND key = ? AND status = 'active'",
            [(now, self._agent_id, k) for k in keys],
        )
        await self._db.commit()

    async def fold_staged_access(self) -> int:
        """Consolidation-boundary fold: staged read-counters merge into the base columns
        the injected block reads. THE only moment a read can reorder the block — called
        by the idle consolidation pass (Phase 31). Returns rows folded.

        Also the RANK-03 'bumped on confirmed recall' path (Phase-31 critic minor 2:
        retrieval_strength was one-way-down): folded reads restore accessibility, so a
        heavily-used demoted/decayed fact can organically climb back above the index
        gate instead of needing a fresh supersede-write."""
        assert self._db is not None
        cur = await self._db.execute(
            "UPDATE facts SET access_count = access_count + access_count_staged, "
            "last_accessed_at = COALESCE(last_accessed_staged, last_accessed_at), "
            "retrieval_strength = MIN(1.0, retrieval_strength + 0.05 * access_count_staged), "
            "access_count_staged = 0, last_accessed_staged = NULL "
            "WHERE agent_id = ? AND access_count_staged > 0",
            (self._agent_id,),
        )
        await self._db.commit()
        return cur.rowcount

    # ------------------------------------------------------------------
    # Typed graph (RANK-01): facts rows are the nodes; edges carry structure
    # ------------------------------------------------------------------

    EDGE_KINDS = frozenset({"derived_from", "member_of", "supports", "contradicts"})

    async def add_edge(self, src_id: int, dst_id: int, kind: str) -> None:
        """Insert a typed edge (idempotent). `supersedes` is NOT an edge kind — it stays
        a facts column because it is the hot-path exclusion mechanism (RANK-05)."""
        if kind not in self.EDGE_KINDS:
            raise ValueError(f"unknown edge kind {kind!r}; allowed: {sorted(self.EDGE_KINDS)}")
        assert self._db is not None
        await self._db.execute(
            "INSERT OR IGNORE INTO edges (src_id, dst_id, kind, created_at) VALUES (?, ?, ?, ?)",
            (src_id, dst_id, kind, int(time.time())),
        )
        await self._db.commit()

    async def neighborhood(
        self, node_id: int, *, depth: int = 2, limit: int = 50
    ) -> list[tuple[int, int]]:
        """Undirected graph walk from a node: [(fact_id, min_depth)] ordered nearest-first.

        Python frontier BFS with a REAL visited set (Phase-30 critic, MAJOR): the
        recursive-CTE version's in-path guard blocked cycles but could not prune a node
        rediscovered via sibling branches — it enumerated ALL simple paths (~×avg-degree
        rows per level; 129k raw rows at depth 4 on a 200-node graph, 87-148ms on the
        every-turn retrieval path). SQLite forbids the multiple-recursive-reference
        subquery that would fix it in SQL. BFS = ≤depth round trips, identical results
        (verified across 15 start nodes), ~55× faster. Depth hard-capped at 4.
        """
        assert self._db is not None
        depth = min(max(depth, 0), 4)
        visited: dict[int, int] = {node_id: 0}
        frontier: list[int] = [node_id]
        for d in range(1, depth + 1):
            if not frontier or len(visited) >= limit * 4:
                break
            qmarks = ",".join("?" * len(frontier))
            async with self._db.execute(
                f"SELECT src_id, dst_id FROM edges "
                f"WHERE src_id IN ({qmarks}) OR dst_id IN ({qmarks})",
                [*frontier, *frontier],
            ) as cur:
                rows = await cur.fetchall()
            fset = set(frontier)
            nxt: list[int] = []
            for s, t in rows:
                for a, b in ((s, t), (t, s)):
                    if a in fset and b not in visited:
                        visited[b] = d
                        nxt.append(b)
            frontier = nxt
        items = sorted(visited.items(), key=lambda kv: (kv[1], kv[0]))[:limit]
        return items

    # ------------------------------------------------------------------
    # Predictive gate substrate (COLL-01) — pure-SQL per-tool priors
    # ------------------------------------------------------------------

    async def get_tool_prior(
        self, tool_name: str, *, before_ts: int | None = None
    ) -> ToolPrior:
        """Per-tool statistical prior from event history (COLL-01). ONE indexed
        aggregate over tool_observations, zero tokens — the two-step shape query_facts
        already uses (SQL computes the context, a pure function scores it).

        before_ts (walk-forward): only rows STRICTLY earlier count, so the scored
        observation never contaminates its own prior. Variance is the population form
        AVG(x*x) - AVG(x)*AVG(x); tiny negative results (float cancellation on a
        near-constant history) are clamped to 0.0. NULL aggregates (empty history)
        map to None — cold start carried honestly, never a fabricated 0."""
        assert self._db is not None
        async with self._db.execute(
            """
            SELECT COUNT(*),
                   AVG(is_error),
                   AVG(duration_ms),
                   AVG(duration_ms * duration_ms) - AVG(duration_ms) * AVG(duration_ms),
                   COUNT(duration_ms),
                   AVG(output_len),
                   AVG(output_len * output_len) - AVG(output_len) * AVG(output_len),
                   COUNT(output_len)
            FROM tool_observations
            WHERE agent_id = ? AND tool_name = ? AND (? IS NULL OR ts < ?)
            """,
            (self._agent_id, tool_name, before_ts, before_ts),
        ) as cur:
            row = await cur.fetchone()

        lat_var = row[3]
        if lat_var is not None and lat_var < 0.0:
            lat_var = 0.0
        size_var = row[6]
        if size_var is not None and size_var < 0.0:
            size_var = 0.0
        return ToolPrior(
            tool_name=tool_name,
            n=row[0] or 0,
            error_rate=row[1],
            lat_mean_ms=row[2],
            lat_var_ms=lat_var,
            lat_n=row[4] or 0,
            size_mean=row[5],
            size_var=size_var,
            size_n=row[7] or 0,
        )

    # ------------------------------------------------------------------
    # Recording APIs (COLL-03/04) — idempotent, collect-only. These write ONLY the
    # v4 tables; facts/sessions/edges are never touched (the byte-stability test is
    # the enforcement). Signatures are the store contract plans 34-03/34-04 call.
    # ------------------------------------------------------------------

    async def record_tool_observation(
        self, *, session_id: str, tool_call_id: str | None, tool_name: str, ts: int,
        is_error: int, output_len: int | None, duration_ms: int | None,
        event_id: str | None, source: str = "live",
    ) -> int:
        """INSERT OR IGNORE keyed on event_id (idempotent re-ingestion — a live row and a
        later backfill of the same bus event collapse to one). Returns the rowid (the
        existing row's id on ignore). One INSERT, no reads — the WriteGate cheapness class."""
        assert self._db is not None
        cur = await self._db.execute(
            "INSERT OR IGNORE INTO tool_observations "
            "(agent_id, session_id, tool_call_id, tool_name, ts, is_error, output_len, "
            "duration_ms, event_id, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (self._agent_id, session_id, tool_call_id, tool_name, ts, is_error,
             output_len, duration_ms, event_id, source),
        )
        await self._db.commit()
        if cur.rowcount == 0 and event_id is not None:
            async with self._db.execute(
                "SELECT id FROM tool_observations WHERE event_id = ?", (event_id,)
            ) as c2:
                row = await c2.fetchone()
            return row[0] if row else 0
        return cur.lastrowid

    async def record_surprise_score(
        self, *, session_id: str, observation_id: int | None,
        expectation_json: str | None, score: float, quadrant: str | None, scored_at: int,
    ) -> int:
        """One INSERT into surprise_scores. agent_id from self._agent_id."""
        assert self._db is not None
        cur = await self._db.execute(
            "INSERT INTO surprise_scores "
            "(agent_id, session_id, observation_id, expectation_json, score, quadrant, scored_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (self._agent_id, session_id, observation_id, expectation_json, score,
             quadrant, scored_at),
        )
        await self._db.commit()
        return cur.lastrowid

    async def record_user_signal(
        self, *, session_id: str, ts: int, signal_type: str, trigger_family: str | None,
        matched_text: str | None, user_message: str, corrected_turn_summary: str | None,
        event_id: str | None,
    ) -> int:
        """INSERT OR IGNORE keyed on event_id. Returns rowid (existing on ignore).
        user_message stored in FULL (owner steer: look-ready records — the future model
        look needs the verbatim text, not a preview)."""
        assert self._db is not None
        cur = await self._db.execute(
            "INSERT OR IGNORE INTO user_signals "
            "(agent_id, session_id, ts, signal_type, trigger_family, matched_text, "
            "user_message, corrected_turn_summary, event_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (self._agent_id, session_id, ts, signal_type, trigger_family, matched_text,
             user_message, corrected_turn_summary, event_id),
        )
        await self._db.commit()
        if cur.rowcount == 0 and event_id is not None:
            async with self._db.execute(
                "SELECT id FROM user_signals WHERE event_id = ?", (event_id,)
            ) as c2:
                row = await c2.fetchone()
            return row[0] if row else 0
        return cur.lastrowid

    async def snapshot_staged_candidates(
        self, user_signal_id: int, candidate_type: str
    ) -> int:
        """COLL-03 collect-only credit assignment: snapshot the facts currently staged into
        context (access_count_staged > 0 — exactly touch_staged's explicitly-retrieved
        semantics; ambient always-injected facts are NOT staged and NOT snapshotted, a
        deliberate v1 scope per 34-RESEARCH Open Q2). One SELECT + executemany INSERT.
        candidate_type: 'bump' (confirmation) | 'suspect' (correction). Returns count."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT id, key FROM facts WHERE agent_id = ? AND access_count_staged > 0",
            (self._agent_id,),
        ) as cur:
            staged = await cur.fetchall()
        if not staged:
            return 0
        now = int(time.time())
        await self._db.executemany(
            "INSERT INTO staged_snapshots "
            "(user_signal_id, fact_key, fact_id, candidate_type, captured_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [(user_signal_id, r[1], r[0], candidate_type, now) for r in staged],
        )
        await self._db.commit()
        return len(staged)

    async def staged_suspect_facts(self) -> list[tuple[int, str, str]]:
        """PGATE-03 read side: the facts explicitly staged into the current sitting's
        context (access_count_staged > 0 — the same actually-retrieved set
        snapshot_staged_candidates records as 'suspect'). Returns (id, key, value)
        MOST-RECENTLY-STAGED FIRST (max last_accessed_staged, tie-break highest id) so a
        scoped correction supersede (BLOCKER 1(b)) can target the single most-recent
        suspect deterministically instead of the whole staged sitting. Read-only: never
        resets the staged counter (that is fold_staged_access's job at the consolidation
        boundary), so it is stable within a sitting and order-independent vs
        UserSignalDetector."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT id, key, value FROM facts "
            "WHERE agent_id = ? AND access_count_staged > 0 AND status = 'active' "
            "ORDER BY last_accessed_staged DESC, id DESC",
            (self._agent_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    # ------------------------------------------------------------------
    # History delegation
    # ------------------------------------------------------------------

    async def append_history(self, record: dict[str, Any]) -> None:
        await self._history_writer.append(record)

    async def get_history(
        self,
        session_id: str | None = None,
        limit: int = 200,
        message_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        records = await self._history_writer.read_all()
        if session_id is not None:
            records = [r for r in records if r.get("session_id") == session_id]
        if message_types is not None:
            records = [r for r in records if r.get("type") in message_types]
        return records[-limit:]

    # ------------------------------------------------------------------
    # Context loading
    # ------------------------------------------------------------------

    async def load_context(
        self,
        index_mode: bool = True,
        max_session_history: int = 8,
    ) -> MemoryContext:
        """Load three-tier context for system prompt injection.

        When `index_mode` is True (default), the agent-memory block is an INDEX — one line
        per persistent fact (name + one-line description, NOT the full body) plus the most
        recent session-history entries rendered from the sessions TABLE with relative day +
        clock-time labels (`- today 11:47am: …`), hard-capped at `_SESSION_SHELF_HARD_CAP`
        (TIME-03) — instead of the entire MEMORY.md file. The full body of any fact is
        served on demand via the memory_get / memory_search tools. When False, the legacy
        behaviour (whole MEMORY.md inlined)
        is used — which became LIVE-WRITTEN in Phase 33 (SESS-02 restored the MEMORY.md
        writer: end_session -> flush_memory_md -> MarkdownMemory.regenerate writes the exact
        file this branch reads), so it now renders real facts + the latest session line
        rather than a stale/empty file no live code updated.
        """
        assert self._db is not None

        if index_mode:
            agent_md = await self._render_memory_index(max_session_history)
        else:
            agent_md = self._markdown_memory.read()

        division_md = ""
        if self._division_md_path.exists():
            division_md = self._division_md_path.read_text(encoding="utf-8")

        guardrails_md = ""
        if self._guardrails_path.exists():
            guardrails_md = self._guardrails_path.read_text(encoding="utf-8")

        async with self._db.execute(
            "SELECT COUNT(*) FROM facts WHERE agent_id = ?", (self._agent_id,)
        ) as cur:
            row = await cur.fetchone()
        fact_count = row[0] if row else 0

        token_estimate = len(agent_md + division_md + guardrails_md) // 4
        return MemoryContext(
            agent_memory_md=agent_md,
            division_md=division_md,
            guardrails_md=guardrails_md,
            fact_count=fact_count,
            token_estimate=token_estimate,
        )

    async def _render_memory_index(self, max_session_history: int) -> str:
        """Render the agent-memory INDEX: fact names + one-line descriptions (not full
        bodies) and the most recent session-history entries — the latter from the sessions
        TABLE with relative-time labels, hard-capped at `_SESSION_SHELF_HARD_CAP` (TIME-03).
        The model is told it can call memory_get(name) / memory_search(query) for detail."""
        assert self._db is not None
        now = int(time.time())
        # Injected-block ordering (RANK-02/04): importance + ACT-R base-level activation
        # over the FOLDED columns only (staged read-counters are invisible here), with age
        # quantized to DAYS — so the block's bytes change only at consolidation folds,
        # genuine writes, or a day boundary (the loop.py:592 "date not time" precedent).
        # Retrieval-strength gate (RANK-03): inaccessible facts drop out of the index
        # while staying searchable via the tool path.
        # INDEXED BY (Phase-30 critic BLOCKER 2): the rs-gate + function ORDER BY
        # combination silently defeated the planner's partial-index choice (it fell to
        # idx_facts_agent_id, fetching every superseded row on the hottest per-turn
        # query — the exact long-session degradation the owner's supersede approval
        # forbids). Forcing the partial index is deterministic, ANALYZE-independent,
        # and fails LOUD if the index name ever drifts.
        # SEMA-04 (Phase 36): schemas render FIRST as their own "### Knowledge" section —
        # "gist routes, verbatim answers" made true for EXPERIENCE (a chapter leads; its
        # member lessons are demoted OUT of the facts list by 36-04's retrieval_strength
        # drop, NOT edge-joined here — the hottest per-turn query stays flat so the forced
        # partial index keeps holding). SCOPE (RESEARCH Pitfall 4, "decide and state"): this
        # schemas-first change is the index_mode=True render path ONLY (the config default,
        # per load_context above). The legacy flush_memory_md -> MarkdownMemory.regenerate
        # end-of-session dump is DELIBERATELY out of scope — the injected ambient block is the
        # byte-stability-critical surface; the legacy dump is not on the hot path. Same gates
        # + INDEXED BY + day-quantized lh_slow_score ORDER BY as the facts query below
        # (byte-stability: folded columns only).
        async with self._db.execute(
            "SELECT key, value FROM facts INDEXED BY idx_facts_active_recency "
            "WHERE agent_id = ? AND status = 'active' AND node_kind = 'schema' "
            "AND confidence >= 0.7 AND retrieval_strength >= 0.2 "
            "AND (expires_at IS NULL OR expires_at > ?) "
            "ORDER BY lh_slow_score(importance, access_count, last_accessed_at, updated_at, ?) DESC, "
            "updated_at DESC, key ASC",
            (self._agent_id, now, now),
        ) as cur:
            schema_rows = await cur.fetchall()
        schema_lines = [f"- {r[0]}: {_one_line(r[1], 180)}" for r in schema_rows]
        # Zero bytes when absent (mirrors history_section below): an empty schemas set must
        # not change the injected block's bytes for chapter-less stores (RANK-04 byte-stability).
        schema_section = (
            f"### Knowledge ({len(schema_lines)} chapters)\n" + "\n".join(schema_lines) + "\n\n"
            if schema_lines
            else ""
        )

        async with self._db.execute(
            "SELECT key, value FROM facts INDEXED BY idx_facts_active_recency "
            "WHERE agent_id = ? AND status = 'active' AND node_kind != 'schema' "
            "AND confidence >= 0.7 "
            "AND retrieval_strength >= 0.2 "
            "AND (expires_at IS NULL OR expires_at > ?) "
            "ORDER BY lh_slow_score(importance, access_count, last_accessed_at, updated_at, ?) DESC, "
            "updated_at DESC, key ASC",
            (self._agent_id, now, now),
        ) as cur:
            rows = await cur.fetchall()

        # 180-char budget (live test 2026-07-03): at the default 100, one absolute
        # path (~50 chars) plus any prefix guillotined the payload — the injected
        # line carried an error with no filename and no resolution. Lessons must
        # survive the line render with their discriminating content intact.
        fact_lines = [f"- {r[0]}: {_one_line(r[1], 180)}" for r in rows]
        facts_block = "\n".join(fact_lines) if fact_lines else "(no persistent facts)"

        # TIME-02/03: the injected shelf renders from the sessions TABLE — started_at
        # is full-precision epoch; MEMORY.md's date-only line cannot carry clock time.
        # `summary IS NOT NULL` is one schema predicate doing two structural jobs:
        # excludes the still-open current sitting (create_session leaves summary NULL
        # until end_session) AND vacuous sittings (SESS-05: derive -> None -> NULL) —
        # "renders entries or renders nothing" (1fbdf6b), now enforced by the schema
        # instead of text filtering. Hard budget (TIME-03): min(config, cap); LIMIT
        # drops the oldest rows WHOLE (5192f27 — never mid-line). Rows for dropped
        # sittings stay in the table: absence from the prompt is not forgetting.
        shelf_n = min(max_session_history, _SESSION_SHELF_HARD_CAP)
        sess_rows: list = []
        if shelf_n > 0:
            async with self._db.execute(
                "SELECT started_at, summary FROM sessions "
                "WHERE agent_id = ? AND summary IS NOT NULL "
                "ORDER BY started_at DESC, id DESC LIMIT ?",
                (self._agent_id, shelf_n),
            ) as cur:
                sess_rows = list(await cur.fetchall())
        # Relative labels: LOCAL time (the loop.py:606 convention — this block and the
        # system prompt's date line must agree), `today` computed ONCE per render —
        # byte-stable within a day; flips only at the local day boundary, phasing with
        # the existing daily date bust (TIME-04: no new cache-bust class).
        today_local = datetime.now().astimezone().date()
        entry_lines = []
        for started_at, summary in sess_rows:
            dt_local = datetime.fromtimestamp(started_at).astimezone()
            label = _relative_day_label(dt_local.date(), today_local)
            # _one_line: newline-proof + the 180-char payload budget (5192f27) —
            # end_session stores summaries uncapped; the render must cap.
            entry_lines.append(
                f"- {label} {_clock_label(dt_local)}: {_one_line(summary, 180)}"
            )
        history_section = (
            f"\n\n### Recent Session History (last {shelf_n})\n" + "\n".join(entry_lines)
            if entry_lines
            else ""
        )

        return (
            "This is an INDEX, not the full memory. Each line below is one persistent fact "
            "(name: short description). Call `memory_get(name)` for a fact's full body, or "
            "`memory_search(query)` to search fact contents.\n\n"
            f"{schema_section}### Persistent Facts ({len(fact_lines)})\n{facts_block}"
            f"{history_section}"
        )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def create_session(
        self,
        session_id: str,
        budget: dict[str, Any],
        model: str,
        context_tokens_available: int,
    ) -> None:
        """Record session start in SQLite and append session_start to history.jsonl."""
        assert self._db is not None
        now = int(time.time())
        await self._db.execute(
            "INSERT INTO sessions (id, agent_id, division_id, org_id, started_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, self._agent_id, self._division_id, self._org_id, now),
        )
        await self._db.commit()
        await self._history_writer.append({
            "v": 1,
            "type": "session_event",
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "agent_id": self._agent_id,
            "ts": now,
            "event": "session_start",
            "data": {
                "budget": budget,
                "model": model,
                "context_tokens_available": context_tokens_available,
            },
        })

    async def end_session(
        self,
        session_id: str,
        exit_reason: str,
        summary: str | None,
        turn_count: int,
        action_count: int,
        tokens_in: int,
        tokens_out: int,
    ) -> None:
        """Record session end in SQLite + history.jsonl, flush MEMORY.md."""
        assert self._db is not None
        now = int(time.time())
        await self._db.execute(
            """
            UPDATE sessions
            SET ended_at = ?, exit_reason = ?, summary = ?,
                turn_count = ?, action_count = ?, tokens_in = ?, tokens_out = ?
            WHERE id = ?
            """,
            (now, exit_reason, summary, turn_count, action_count, tokens_in, tokens_out, session_id),
        )
        await self._db.commit()
        await self._history_writer.append({
            "v": 1,
            "type": "session_event",
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "agent_id": self._agent_id,
            "ts": now,
            "event": "session_end",
            "data": {
                "exit_reason": exit_reason,
                "summary": summary,
                "turn_count": turn_count,
                "action_count": action_count,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
            },
        })
        await self.flush_memory_md(summary)

    # ------------------------------------------------------------------
    # MEMORY.md flush
    # ------------------------------------------------------------------

    async def flush_memory_md(self, session_summary: str | None = None) -> None:
        """Regenerate MEMORY.md from current fact store. Preserves notes sections."""
        assert self._db is not None
        now = int(time.time())
        async with self._db.execute(
            "SELECT key, value, updated_at FROM facts INDEXED BY idx_facts_active_recency "
            "WHERE agent_id = ? AND status = 'active' AND confidence >= 0.7 "
            "AND retrieval_strength >= 0.2 "
            "AND (expires_at IS NULL OR expires_at > ?) "
            "ORDER BY lh_slow_score(importance, access_count, last_accessed_at, updated_at, ?) DESC, "
            "updated_at DESC, key ASC",
            (self._agent_id, now, now),
        ) as cur:
            rows = await cur.fetchall()

        facts_lines = []
        for row in rows:
            dt = datetime.fromtimestamp(row[2], tz=timezone.utc).strftime("%Y-%m-%d")
            facts_lines.append(f"- {row[0]}: {row[1]} *(updated {dt})*")
        facts_text = "\n".join(facts_lines) if facts_lines else ""

        session_entry: str | None = None
        if session_summary:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            # 180-char budget — same as the index fact-line budget (5192f27): the payload
            # must survive every render layer; 120 guillotined derived summaries mid-payload.
            session_entry = f"- {today}: {session_summary[:180]}"

        self._markdown_memory.regenerate(
            agent_id=self._agent_id,
            agent_name=self._agent_id,
            role="",
            facts_text=facts_text,
            session_entry=session_entry,
        )

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    async def integrity_check(self) -> list[str]:
        """Run SQLite integrity_check + foreign_key_check + validate history.jsonl."""
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

        try:
            await self._history_writer.read_all()
        except MemoryCorruptionError as exc:
            errors.append(f"history.jsonl corruption: {exc}")

        return errors

    # ------------------------------------------------------------------
    # Auto-diary bus handlers
    # ------------------------------------------------------------------

    async def _on_action(self, event: "Action") -> None:
        if event.action_type == "tool_call":
            record: dict[str, Any] = {
                "v": 1,
                "type": "assistant_message",
                "id": str(uuid.uuid4()),
                "session_id": event.session_id,
                "agent_id": event.agent_id,
                "ts": int(event.timestamp.timestamp()),
                "role": "assistant",
                "content": event.content,
                "tool_calls": [{
                    "id": event.tool_call_id or str(uuid.uuid4()),
                    "name": event.tool_name or "",
                    "arguments": event.tool_params or {},
                }],
                "finish_reason": "tool_calls",
                "tokens_in": 0,
                "tokens_out": 0,
                "model": "",
                "latency_ms": 0,
            }
            await self._history_writer.append(record)
        elif event.action_type == "llm_response":
            record = {
                "v": 1,
                "type": "assistant_message",
                "id": str(uuid.uuid4()),
                "session_id": event.session_id,
                "agent_id": event.agent_id,
                "ts": int(event.timestamp.timestamp()),
                "role": "assistant",
                "content": event.content,
                "tool_calls": [],
                "finish_reason": "stop",
                "tokens_in": 0,
                "tokens_out": 0,
                "model": "",
                "latency_ms": 0,
            }
            await self._history_writer.append(record)

    async def _on_observation(self, event: "Observation") -> None:
        if event.observation_type == "tool_result":
            record: dict[str, Any] = {
                "v": 1,
                "type": "tool_result",
                "id": str(uuid.uuid4()),
                "session_id": event.session_id,
                "agent_id": event.agent_id,
                "ts": int(event.timestamp.timestamp()),
                "role": "tool",
                "call_id": event.tool_call_id or "",
                "tool_name": event.tool_name or "",
                "content": event.output or "",
                "is_error": event.error is not None,
                "error_type": None,
                "truncated": event.truncated,
                "original_length": len(event.output or ""),
                "stored_length": len(event.output or ""),
            }
            await self._history_writer.append(record)

    async def _on_user_message(self, event: "UserMessage") -> None:
        record: dict[str, Any] = {
            "v": 1,
            "type": "user_message",
            "id": str(uuid.uuid4()),
            "session_id": event.session_id,
            "agent_id": event.agent_id,
            "ts": int(event.timestamp.timestamp()),
            "role": "user",
            "content": event.content,
            "channel": event.channel,
            "channel_metadata": None,
        }
        await self._history_writer.append(record)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _one_line(value: str, max_chars: int = 100) -> str:
    """First line of a fact value, truncated — the index carries a description, not the body."""
    first = (value or "").strip().splitlines()[0] if (value or "").strip() else ""
    return first if len(first) <= max_chars else first[: max_chars - 1].rstrip() + "…"


# TIME-03: the injected shelf's hard line budget — a system invariant, not a
# tunable default (config can go lower, never higher).
_SESSION_SHELF_HARD_CAP = 8


def _relative_day_label(sitting_local_date: date, today_local_date: date) -> str:
    """Relative day word for the injected shelf (TIME-02). PURE — no clock reads:
    `today_local_date` is computed ONCE per render by the caller, so the label is
    byte-stable within a day and flips only at the LOCAL day boundary, phasing
    with the loop.py:606 daily date bust (TIME-04 — no new cache-bust class).
    Negative deltas (clock skew) and >6-day deltas fall back to the absolute ISO
    date — the KILL bar's revert shape, applied per-line."""
    delta = (today_local_date - sitting_local_date).days
    if delta == 0:
        return "today"
    if delta == 1:
        return "yesterday"
    if 2 <= delta <= 6:
        return sitting_local_date.strftime("%a")  # e.g. "Tue" (older-in-week)
    return sitting_local_date.strftime("%Y-%m-%d")


def _clock_label(sitting_local_dt: datetime) -> str:
    """12-hour clock, no leading zero, portable (no %-I/%#I platform split):
    %I is 01-12, so lstrip('0') strips at most the hour's leading zero and can
    never touch the zero-padded minutes."""
    return sitting_local_dt.strftime("%I:%M%p").lstrip("0").lower()


# ---------------------------------------------------------------------------
# Activation scoring (RANK-02/03) — registered as SQLite scalar functions so ranking
# is closed-form and costs ZERO decode tokens. ACT-R–INSPIRED base-level activation
# with the canonical decay d = 0.5: a single-trace simplification ln(1+n) − d·ln(age
# since most recent use), NOT Anderson & Schooler 1991's full ln(Σ tⱼ⁻ᵈ) per-trace sum
# (Phase-30 critic minor: the criterion is freq+recency beating pure recency, which
# this satisfies — claiming exact ACT-R fidelity would overreach).
# ---------------------------------------------------------------------------

_ACTR_DECAY = 0.5

# Write-time importance prior (RANK-03): a TAG HEURISTIC, never an LLM rater (VETO #1).
_IMPORTANCE_PRIORS: dict[str, float] = {
    "tier:resolved_error": 0.3,
    "tier:stuck_recovered": 0.2,
    # Phase 35 (PGATE-01/03) stat + correction tiers. Pitfall 4: this dict is closed and
    # hand-maintained — a new tier tag with no entry here silently ranks at the 0.0
    # fallback, so the graded-surprise/correction writes must have explicit priors or
    # "graded surprise feeds importance/activation" degrades to no-op. surprising_failure
    # shares resolved_error's warrant (both an error signal worth learning);
    # correction_pending shares stuck_recovered's tier (a single-episode salient dispute).
    "tier:surprising_failure": 0.3,
    "tier:correction_pending": 0.2,
    "remember": 0.4,
    # Phase 36 (SEMA-03/04): a lesson-cluster chapter LEADS its "### Knowledge" section —
    # 0.5 sits above resolved_error's 0.3 so the schema (a promotion over its members) sorts
    # first; with no prior it would sink to the 0.0 floor (Pitfall 2) despite being the lead.
    "tier:schema": 0.5,
}


def _importance_prior(tags: list[str], source: str) -> float:
    candidates = [0.0] + [_IMPORTANCE_PRIORS[t] for t in tags if t in _IMPORTANCE_PRIORS]
    if source == "remember":
        candidates.append(_IMPORTANCE_PRIORS["remember"])
    return max(candidates)


def _base_activation(
    access_count: int | None,
    last_accessed_at: int | None,
    updated_at: int | None,
    now: int,
    *,
    day_granularity: bool,
) -> float:
    """ln(1 + n) − d·ln(age): n = folded access count; age measured from the most recent
    of last-read/last-write. Day-granular for the injected block (byte-stable within a
    day), continuous (hours) for the tool path."""
    import math

    n = access_count or 0
    stamps = [s for s in (last_accessed_at, updated_at) if s is not None]
    last = max(stamps) if stamps else now
    if day_granularity:
        # SHARED CALENDAR-DAY difference (Phase-30 critic BLOCKER 1): `(now-last)//86400`
        # was a rolling 24h window phased to each fact's own last-touch — measured 11-24
        # block reorders/day at 30-300 facts (an arbitrary-hour cache bust per fact).
        # Epoch-day difference changes for ALL facts atomically at the same boundary as
        # the system prompt's date line (loop.py:592) — measured ≤0.55 reorders/day.
        age_units = max(0, (now // 86400) - (last // 86400)) + 1
    else:
        age_units = max(0, now - last) / 3600.0 + 1.0
    return math.log(1 + n) - _ACTR_DECAY * math.log(age_units)


def _slow_score(importance, access_count, last_accessed_at, updated_at, now) -> float:
    """Injected-block score: importance + base-level over FOLDED columns, day-quantized.
    Every input moves only at consolidation folds, genuine writes, or a day boundary —
    the byte-stability discipline (RANK-04)."""
    return (importance or 0.0) + _base_activation(
        access_count, last_accessed_at, updated_at, now, day_granularity=True
    )


def _fused_score(importance, access_total, last_access, updated_at, now, confidence, bm25_rank) -> float:
    """Tool-path score (SYNTHESIS §2, additive in log-odds): importance + base-level
    (fresh, staged included, hour-granular) + ln(precision) − BM25 (SQLite bm25 is
    smaller-is-better, typically negative — negate into a goodness term)."""
    import math

    base = _base_activation(access_total, last_access, updated_at, now, day_granularity=False)
    conf = min(max(confidence if confidence is not None else 0.5, 1e-3), 1.0)
    return (importance or 0.0) + base + math.log(conf) - (bm25_rank or 0.0)


# ---------------------------------------------------------------------------
# Surprise scoring (COLL-01) — deterministic pure functions beside the activation
# scalars, same aesthetic (cold-start-graceful, never NULL/raise). NOT registered
# via create_function this phase: a 12-arg SQL scalar with no SQL-side caller is
# surface without a consumer — Phase 35 registers it when ORDER BY needs it. Module-
# level so the report script (34-07) can import and score offline.
# ---------------------------------------------------------------------------

_SURPRISE_MIN_N = 5  # default cold-start floor; callers thread the config value through


def _tool_error_surprisal(
    is_error: int, prior_error_rate: float | None, n: int, min_n: int = _SURPRISE_MIN_N
) -> float:
    """Information-theoretic surprise of one boolean outcome against this tool's own
    history: observed surprisal minus the prior's own entropy (~0 when routine, positive
    when it deviates). Cold start (n < min_n) or no prior -> 0.0 neutral (mirrors
    _base_activation's graceful n=0 — never NULL, never raises)."""
    import math

    if prior_error_rate is None or n < min_n:
        return 0.0
    p = min(max(prior_error_rate, 1e-3), 1 - 1e-3)  # guard degenerate 0%/100% rates
    observed = -math.log(p if is_error else (1 - p))
    expected = -(p * math.log(p) + (1 - p) * math.log(1 - p))  # the prior's own entropy
    return observed - expected


def _band_z(
    value: float | None,
    mean: float | None,
    variance: float | None,
    n: int,
    min_n: int = _SURPRISE_MIN_N,
) -> float:
    """Plain z-score for a continuous feature (latency, output size). None inputs,
    n < min_n, or degenerate (near-constant) variance all degrade to 0.0 rather than
    raising or dividing by ~zero."""
    import math

    if value is None or mean is None or variance is None or n < min_n or variance < 1e-6:
        return 0.0
    return (value - mean) / math.sqrt(variance)


def compute_surprise_score(
    is_error: int,
    output_len: int | None,
    duration_ms: int | None,
    prior: ToolPrior,
    *,
    min_n: int = _SURPRISE_MIN_N,
    latency_weight: float = 0.5,
    size_weight: float = 0.25,
) -> float:
    """Composite graded surprise: error-outcome surprisal + weighted ABSOLUTE latency/size
    deviations. abs(): a deviation in EITHER direction is "succeeded-but-differently" (the
    reframe doc's quiet-surprise quadrant) — a 10x-faster call is as anomalous as a
    10x-slower one. Cold-start-neutral by delegation (every term is 0.0 below min_n), so an
    empty prior yields exactly 0.0."""
    return (
        _tool_error_surprisal(is_error, prior.error_rate, prior.n, min_n)
        + latency_weight
        * abs(_band_z(duration_ms, prior.lat_mean_ms, prior.lat_var_ms, prior.lat_n, min_n))
        + size_weight
        * abs(_band_z(output_len, prior.size_mean, prior.size_var, prior.size_n, min_n))
    )


def compute_quadrant(
    is_error: int, prior_error_rate: float | None, n: int, min_n: int = _SURPRISE_MIN_N
) -> str:
    """Map an outcome onto the reframe taxonomy (the quadrants the binary gate structurally
    cannot express). Below min_n or with no prior -> 'cold_start'. predicted_fail is the
    tool's own history saying error is the base case (error_rate >= 0.5)."""
    if prior_error_rate is None or n < min_n:
        return "cold_start"
    predicted_fail = prior_error_rate >= 0.5
    if not predicted_fail:
        return "surprising_failure" if is_error else "routine"
    return "unsurprising_failure" if is_error else "quiet_surprise"


def _sanitize_fts_query(text: str, max_tokens: int = 32) -> str:
    """Quote every whitespace token so FTS5 operator/syntax characters in real-corpus
    tokens (`000660.KS`, `P/GP`, `-1.5σ`) are literal phrases, never syntax (WRITE-05).
    Embedded double-quotes are doubled per FTS5 string rules. Returns "" when no usable
    token remains (caller falls back to the non-FTS recency path)."""
    tokens = (text or "").split()
    quoted = ['"' + t.replace('"', '""') + '"' for t in tokens[:max_tokens] if t.strip('"')]
    return " ".join(quoted)


def _row_to_fact(row: aiosqlite.Row) -> Fact:
    tags_raw = row["tags"] if isinstance(row, aiosqlite.Row) else row[5]
    tags = json.loads(tags_raw) if isinstance(tags_raw, str) else (tags_raw or [])
    if isinstance(row, aiosqlite.Row):
        keys = row.keys()
        return Fact(
            key=row["key"],
            value=row["value"],
            agent_id=row["agent_id"],
            division_id=row["division_id"],
            org_id=row["org_id"],
            tags=tags,
            confidence=row["confidence"],
            source=row["source"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
            status=row["status"] if "status" in keys else "active",
            superseded_by=row["superseded_by"] if "superseded_by" in keys else None,
            provenance=row["provenance"] if "provenance" in keys else "",
            id=row["id"] if "id" in keys else 0,
            retrieval_strength=row["retrieval_strength"] if "retrieval_strength" in keys else 0.5,
            importance=row["importance"] if "importance" in keys else 0.0,
            access_count=row["access_count"] if "access_count" in keys else 0,
            last_accessed_at=row["last_accessed_at"] if "last_accessed_at" in keys else None,
            node_kind=row["node_kind"] if "node_kind" in keys else "fact",
        )
    # Positional (shouldn't happen with row_factory=aiosqlite.Row)
    return Fact(
        key=row[0], value=row[1], agent_id=row[2], division_id=row[3],
        org_id=row[4], tags=tags, confidence=row[6], source=row[7],
        created_at=row[8], updated_at=row[9], expires_at=row[10],
        status=row[11] if len(row) > 11 else "active",
        superseded_by=row[12] if len(row) > 12 else None,
        provenance=row[13] if len(row) > 13 else "",
        id=row[14] if len(row) > 14 else 0,
        retrieval_strength=row[15] if len(row) > 15 else 0.5,
        importance=row[16] if len(row) > 16 else 0.0,
        access_count=row[17] if len(row) > 17 else 0,
        last_accessed_at=row[18] if len(row) > 18 else None,
        node_kind=row[19] if len(row) > 19 else "fact",
    )
