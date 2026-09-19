"""MemoryStore: SQLite facts, sessions, FTS5, bus integration."""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
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

log = logging.getLogger(__name__)

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
    # v10 (resonance rebuild): belief as log-odds moved only by evidence (None on
    # pre-v10 rows — derived from `confidence` at read time), and the accumulated
    # resonance-share mass from dreaming's stream replay (the need axis; no clock).
    truth_logodds: float | None = None
    standing: float = 0.0


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
class MemoryContext:
    agent_memory_md: str
    division_md: str
    guardrails_md: str
    fact_count: int
    token_estimate: int
    # The atom ids rendered into the ambient shelf this load (schema chapters + persistent
    # facts) — the "injected set" for the ambient-injection activation trace (owner reversal
    # 2026-07-17). Empty in the legacy whole-MEMORY.md render path. Captured alongside the
    # rendered text; recording the trace is the loop's best-effort job, gated by the kill-switch.
    injected_fact_ids: list[int] = field(default_factory=list)


# Stimulus-text cap for activation traces (design: the digest is the sha256 of the FULL
# text plus the text truncated to ~200 chars — the hash keeps discriminating past the cap).
_STIMULUS_TEXT_CAP = 200


@dataclass(frozen=True)
class ActivationTrace:
    """One append-only retrieval-event row (tag-graph substrate P0). Records the stimulus
    digest, the atoms it fired (search/recall hits), and the subset actually injected into
    context — the non-backfillable history that later co-activation weights / pattern-
    completion retrieval will consume. Pure bookkeeping: no scoring/weights/spreading here."""
    id: int
    agent_id: str
    session_id: str
    turn: int | None
    stimulus_hash: str
    stimulus_text: str
    fired_ids: list[int]
    injected_ids: list[int]
    source: str
    ts: int


# Provenance marker stamped by forget_fact on a user-initiated forget. `<prefix><epoch>[;<orig>]`
# — lets the /memory window detect a retired-by-user row (vs a plain version supersede) and keeps
# the original provenance after the ';' for audit.
USER_FORGET_PROVENANCE_PREFIX = "user_forget@"

# Provenance marker for a HUMAN edit of a fact's content (web memory page / `localharness
# memory edit`): `<prefix><epoch>;<surface>`. Sibling of user_forget@ / promoted_from_workspace@
# — the three human-initiated verbs the store can attribute; an owner edit is ground truth, so
# it rides the normal store_fact supersede path with this stamp instead of a session id.
USER_EDIT_PROVENANCE_PREFIX = "user_edit@"

# Archival stamp, same shape as the two markers above: `<prefix><epoch>;<surface>`. It goes
# in the ARCHIVE's own metadata column (facts_archive.archive_rung), never on the fact row —
# a restored fact must come back byte-identical, so nothing about the archival may be
# written into it. The epoch duplicates facts_archive.archived_at on purpose: the stamp is
# meant to be readable on its own, the way user_edit@<epoch>;cli is.
ARCHIVE_STAMP_PREFIX = "archived@"

# WHICH decision procedure condemned the row — kept as data, not prose, so the two stay
# separately queryable forever. The first watched live run is driven by an externally
# computed consensus list, and its moves must remain distinguishable from every later
# automatic one (and from whatever scorer replaces today's).
ARCHIVE_SURFACE_FLOOR_LINE = "rung1-floor-line"     # the store's own computed proven-useful floor
ARCHIVE_SURFACE_CONSENSUS_LIST = "consensus-list"   # an external list of condemned fact ids


def archive_stamp(surface: str, epoch: int) -> str:
    return f"{ARCHIVE_STAMP_PREFIX}{epoch};{surface}"


# ---------------------------------------------------------------------------
# Phase 33.1 (ORCH-02): one-time root-agent rename (default -> orchestrator)
# ---------------------------------------------------------------------------
# The agent's name IS its storage identity (directory name + the agent_id column
# in facts/sessions + the bus filter), so the default->orchestrator rename must
# reconcile pre-existing 'default'-keyed data the first time the store opens under
# the new root name — otherwise every memory an existing install has is orphaned.
_LEGACY_ROOT_AGENT_ID = "default"
_ROOT_AGENT_ID = "orchestrator"


class LegacyStoreAwaitingAdoption(RuntimeError):
    """A non-owner open found a pre-rename `agents/default/` tree that nobody has adopted yet.

    Raised INSTEAD of creating the destination directory (v0.13 B1). The adoption below refuses
    whenever the destination exists, so a reader that mkdir'd `agents/orchestrator/` on its way
    to an empty database would orphan that tree for good — a worse outcome than the write the
    reader was avoiding. The owner's next open adopts it; until then this session reads no
    global memory and says so.
    """


def _legacy_adoption_is_pending(base_dir: Path, agent_id: str) -> bool:
    """True when `_migrate_legacy_root_agent_dir` would adopt on the owner's next open."""
    if (base_dir / "agents" / f"{_LEGACY_ROOT_AGENT_ID}.yaml").exists():
        return False        # the ORCH-03 collision marker — see below
    if agent_id != _ROOT_AGENT_ID:
        return False
    agents = base_dir / "agents"
    return (agents / _LEGACY_ROOT_AGENT_ID).is_dir() and not (agents / _ROOT_AGENT_ID).exists()


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
    # The guards live in `_legacy_adoption_is_pending` so the non-owner open path can ask the
    # same question without performing the rename. The `default.yaml` guard among them: the
    # YAML rename deletes default.yaml BEFORE any store opens; its lingering presence means
    # that rename REFUSED (the user owns an 'orchestrator' agent — ORCH-03 collision) and the
    # legacy root still lives under its old 'default' name. Adopting its dir would graft the
    # legacy root's memories into that unrelated agent, falsifying the released "nothing is
    # merged or overwritten" guarantee. Never adopt while that marker is on disk.
    if not _legacy_adoption_is_pending(base_dir, agent_id):
        return
    legacy_dir = base_dir / "agents" / _LEGACY_ROOT_AGENT_ID
    new_dir = base_dir / "agents" / _ROOT_AGENT_ID
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

CURRENT_SCHEMA_VERSION = 10

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
# Schema v5 — the activation-trace log (tag-graph substrate P0). ADDITIVE ONLY:
# one new append-only table, zero touches to facts/sessions/edges — so the ambient
# injected block stays byte-stable by construction (same discipline as v4). ONE
# BEGIN IMMEDIATE ... PRAGMA user_version = 5; COMMIT transaction (critic M1: crash
# -> rollback to intact v4 -> clean retry), matching the additive v3->v4 precedent.
#
# One row per retrieval event: the stimulus digest (stimulus_hash = sha256 of the FULL
# query/recall text; stimulus_text = that text truncated to _STIMULUS_TEXT_CAP), the
# fired atom ids (search/recall hits, JSON array), the injected subset (JSON array), the
# session/turn context, source seam, ts. Append-only — no update/delete path exists.
# Tag ids from the design are DEFERRED (tags don't exist yet): omitted, not a dead column.
# ---------------------------------------------------------------------------

MIGRATION_V4_TO_V5_SQL = """
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS activation_traces (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id       TEXT    NOT NULL,
    session_id     TEXT    NOT NULL DEFAULT '',
    turn           INTEGER,
    stimulus_hash  TEXT    NOT NULL,
    stimulus_text  TEXT    NOT NULL DEFAULT '',
    fired_ids      TEXT    NOT NULL DEFAULT '[]',
    injected_ids   TEXT    NOT NULL DEFAULT '[]',
    source         TEXT    NOT NULL DEFAULT '',
    ts             INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_activation_traces_agent ON activation_traces(agent_id, ts);
PRAGMA user_version = 5;
COMMIT;
"""


# ---------------------------------------------------------------------------
# Schema v6 — the tag graph (Amendment 4: two buckets, two layers). ADDITIVE ONLY:
# two NEW tables (tags, atom_tags); facts/sessions/activation_traces untouched, so
# _FACT_COLS / _row_to_fact / the ambient injected block are byte-stable by construction.
# ONE BEGIN IMMEDIATE ... PRAGMA user_version = 6; COMMIT (crash -> rollback to v5).
#
# DEPTH CONVENTION (pinned as the critique's fix-before-build #4): `parent_id` NULL marks a
# BUCKET (the seeded superordinate root); a non-NULL parent_id marks a CHILD of that bucket.
# v1 permits EXACTLY these two layers (bucket -> child) and NOTHING deeper — a grandchild is
# refused in create_tag (a soft, code-level constraint; a recursive SQL CHECK is not worth it
# for a two-layer schema). The seeded spine + the mint classifier both stay flat at <= 2 picks.
# ---------------------------------------------------------------------------

MIGRATION_V5_TO_V6_SQL = """
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS tags (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id          TEXT    NOT NULL,
    name              TEXT    NOT NULL,
    definition        TEXT    NOT NULL DEFAULT '',
    status            TEXT    NOT NULL DEFAULT 'proposed',   -- seeded|proposed|active|merged|retired
    parent_id         INTEGER,                                -- NULL = bucket (root); set = child. v1: two layers only.
    origin            TEXT    NOT NULL DEFAULT 'discovered',  -- seeded|discovered
    merged_into       INTEGER,                                -- surviving tag id when status = 'merged'
    distinct_sittings INTEGER NOT NULL DEFAULT 0,             -- evidence ladder: distinct sittings the members span
    reuse_count       INTEGER NOT NULL DEFAULT 0,             -- evidence ladder: trace reuse / co-fire strength
    last_accrual_ts   INTEGER,                                -- evidence ladder: recency anchor for decay
    created_at        INTEGER NOT NULL,
    updated_at        INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_tags_agent_name ON tags(agent_id, name);
CREATE INDEX IF NOT EXISTS idx_tags_agent_parent ON tags(agent_id, parent_id, status);
CREATE TABLE IF NOT EXISTS atom_tags (
    atom_id     INTEGER NOT NULL,
    tag_id      INTEGER NOT NULL,
    provenance  TEXT    NOT NULL DEFAULT 'mint',   -- mint|discovery|curation
    ts          INTEGER NOT NULL,
    PRIMARY KEY (atom_id, tag_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_atom_tags_tag ON atom_tags(tag_id);
PRAGMA user_version = 6;
COMMIT;
"""

# ---------------------------------------------------------------------------
# Schema v7 — the mining-residue ledger (the repair half of the extraction loop, owner-directed
# 2026-07-09). RECORD-level forgetting lifecycle, mirroring the atom tier's law one layer down:
# a committed history record that never sourced a written atom is enqueued PENDING; each idle
# pass re-mines a budgeted batch in ISOLATION; a record still barren after `attempt_cap` looks is
# RETIRED — permanently out of the mining window, NEVER deleted (history.jsonl stays append-only;
# retire selects, never destroys — same demote-not-delete ethic as retrieval_strength). ADDITIVE
# ONLY: one new table; facts/sessions/tags untouched, so the ambient block is byte-stable by
# construction. ONE BEGIN IMMEDIATE ... PRAGMA user_version = 7; COMMIT (crash -> rollback to v6).
# ---------------------------------------------------------------------------

MIGRATION_V6_TO_V7_SQL = """
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS mining_residue (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    TEXT    NOT NULL,
    record_id   TEXT    NOT NULL,
    content_h8  TEXT    NOT NULL DEFAULT '',
    session_id  TEXT    NOT NULL DEFAULT '',
    ts          INTEGER NOT NULL DEFAULT 0,
    chars       INTEGER NOT NULL DEFAULT 0,
    attempts    INTEGER NOT NULL DEFAULT 0,
    status      TEXT    NOT NULL DEFAULT 'pending',  -- pending|rescued|retired
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_mining_residue_agent_record ON mining_residue(agent_id, record_id);
CREATE INDEX IF NOT EXISTS idx_mining_residue_pending ON mining_residue(agent_id, status, ts);
PRAGMA user_version = 7;
COMMIT;
"""

# ---------------------------------------------------------------------------
# Schema v8 — per-turn idempotency for ambient-injection traces (owner reversal 2026-07-17 of
# the P0 exclusion: the every-turn memory shelf is now recorded source='injection'). ADDITIVE
# ONLY: one PARTIAL unique index scoped to injection rows — a second assembly of the SAME turn
# (session_id + stimulus_hash) collapses to the one row via INSERT OR IGNORE. The `source`
# discriminator column already exists (v5), so NO column is added; the injection kind REUSES it.
# The index is PARTIAL (WHERE source='injection') so pre-existing retrieval rows (memory_search /
# memory_get, and any historical source) are OUTSIDE it — untouched, still append-only, byte-
# stable. Zero injection rows have ever existed (the model never called the trace tools), so the
# index materialises empty. ONE BEGIN IMMEDIATE ... PRAGMA user_version = 8; COMMIT (crash ->
# rollback to v7). Idempotent via IF NOT EXISTS.
# ---------------------------------------------------------------------------

MIGRATION_V7_TO_V8_SQL = """
BEGIN IMMEDIATE;
CREATE UNIQUE INDEX IF NOT EXISTS ux_activation_traces_injection
    ON activation_traces(agent_id, session_id, stimulus_hash) WHERE source = 'injection';
PRAGMA user_version = 8;
COMMIT;
"""

# ---------------------------------------------------------------------------
# Schema v9 — the COLD ARCHIVE (memory-forgetting rung 1). The project's law is
# demote-never-delete; the measured consequence is a store where 99% of "active" rows are
# demoted-but-present and still in the FTS haystack. Archival is the missing go-away step:
# a row LEAVES `facts` (and with it the FTS index, every recall path, and the memory page)
# into a cold mirror table, and `mv` back is a full restore. Deletion still does not exist.
#
# The mirror is column-for-column identical to `facts` so a restore is byte-identical —
# including id (facts.id is AUTOINCREMENT, so sqlite_sequence guarantees the vacated id is
# never handed to a new row and the superseded_by chain stays valid across a round trip).
# `id` here is a plain INTEGER PRIMARY KEY: the archive never mints ids, it only carries
# them. Four columns are ADDED, never folded into the fact's own values — the fact row must
# come back unchanged, so the archival metadata lives beside it, not inside it.
#
# ADDITIVE ONLY: one new table, `facts` and its triggers untouched (an archived row's FTS
# entry is removed by the EXISTING facts_ad delete trigger — no new FTS machinery).
# ONE BEGIN IMMEDIATE ... PRAGMA user_version = 9; COMMIT (crash -> rollback to v8).
# ---------------------------------------------------------------------------

MIGRATION_V8_TO_V9_SQL = """
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS facts_archive (
    id                   INTEGER PRIMARY KEY,
    agent_id             TEXT    NOT NULL,
    division_id          TEXT    NOT NULL DEFAULT '',
    org_id               TEXT    NOT NULL DEFAULT '',
    key                  TEXT    NOT NULL,
    value                TEXT    NOT NULL,
    tags                 TEXT    NOT NULL DEFAULT '[]',
    confidence           REAL    NOT NULL DEFAULT 1.0,
    source               TEXT    NOT NULL DEFAULT '',
    created_at           INTEGER NOT NULL,
    updated_at           INTEGER NOT NULL,
    expires_at           INTEGER,
    status               TEXT    NOT NULL DEFAULT 'active',
    superseded_by        INTEGER,
    provenance           TEXT    NOT NULL DEFAULT '',
    retrieval_strength   REAL    NOT NULL DEFAULT 0.5,
    importance           REAL    NOT NULL DEFAULT 0.0,
    access_count         INTEGER NOT NULL DEFAULT 0,
    last_accessed_at     INTEGER,
    access_count_staged  INTEGER NOT NULL DEFAULT 0,
    last_accessed_staged INTEGER,
    node_kind            TEXT    NOT NULL DEFAULT 'fact',
    archived_at          INTEGER NOT NULL,
    archive_rung         TEXT    NOT NULL DEFAULT '',
    s_at_archive         REAL    NOT NULL DEFAULT 0.0,
    line_at_archive      REAL
);
CREATE INDEX IF NOT EXISTS idx_facts_archive_agent_when
    ON facts_archive(agent_id, archived_at DESC);
CREATE INDEX IF NOT EXISTS idx_facts_archive_agent_key ON facts_archive(agent_id, key);
PRAGMA user_version = 9;
COMMIT;
"""

# ---------------------------------------------------------------------------
# Schema v10 — the resonance rebuild (memory spec 2026-09-18, model-as-connection).
# Three columns join `facts` (and its archive mirror, column-for-column — the
# _assert_archive_columns contract):
#   truth_logodds — belief in the claim, log-odds currency, moved only by evidence
#                   (birth = the writer's measured track record; NULL on pre-v10 rows,
#                   derived from `confidence` at read time).
#   standing      — accumulated resonance-share mass from dreaming's replay of the
#                   event streams. The need axis; no clock anywhere.
#   embedding     — the trace's encoding vector in the subject-family model's space
#                   (float32 bytes). NULL = not yet embedded; dreaming backfills.
# Four small tables join alongside:
#   writers       — per-writer bet ledger (bets/confirmed/contradicted/paid/lost);
#                   measured precision prices every new row's birth truth.
#   memory_groups — dreaming's named bindings (labels for human legibility ONLY,
#                   never mechanism).
#   digest_marks  — per-ledger-file byte offsets: how much stream the store has
#                   digested. The amount digested is the only clock.
#   meta          — the KV table consolidation always kept (absorbed into the schema
#                   proper so every open guarantees it).
# ADDITIVE ONLY; ONE transaction (crash -> rollback to v9).
# ---------------------------------------------------------------------------

MIGRATION_V9_TO_V10_SQL = """
BEGIN IMMEDIATE;
ALTER TABLE facts ADD COLUMN truth_logodds REAL;
ALTER TABLE facts ADD COLUMN standing REAL NOT NULL DEFAULT 0.0;
ALTER TABLE facts ADD COLUMN embedding BLOB;
ALTER TABLE facts_archive ADD COLUMN truth_logodds REAL;
ALTER TABLE facts_archive ADD COLUMN standing REAL NOT NULL DEFAULT 0.0;
ALTER TABLE facts_archive ADD COLUMN embedding BLOB;
CREATE TABLE IF NOT EXISTS writers (
    agent_id     TEXT    NOT NULL,
    writer       TEXT    NOT NULL,
    bets         INTEGER NOT NULL DEFAULT 0,
    confirmed    INTEGER NOT NULL DEFAULT 0,
    contradicted INTEGER NOT NULL DEFAULT 0,
    paid         INTEGER NOT NULL DEFAULT 0,
    lost         INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (agent_id, writer)
);
CREATE TABLE IF NOT EXISTS memory_groups (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id   TEXT    NOT NULL,
    label      TEXT    NOT NULL DEFAULT '',
    member_ids TEXT    NOT NULL DEFAULT '[]',
    evidence   INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_groups_agent ON memory_groups(agent_id, evidence DESC);
CREATE TABLE IF NOT EXISTS digest_marks (
    agent_id    TEXT    NOT NULL,
    path        TEXT    NOT NULL,
    byte_offset INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (agent_id, path)
);
CREATE TABLE IF NOT EXISTS meta (
    agent_id TEXT NOT NULL,
    key      TEXT NOT NULL,
    value    TEXT NOT NULL,
    PRIMARY KEY (agent_id, key)
) WITHOUT ROWID;
PRAGMA user_version = 10;
COMMIT;
"""

# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

class MemoryStore:
    """
    Three-tier persistent memory for a single agent.

    Owns the agent's memory.db, history.jsonl, and MEMORY.md under `base_dir`.
    Reads (but never writes) division and org memory for context injection — from
    `global_base_dir`, which defaults to `base_dir`, so per-agent state may follow a
    workspace layer while org/division safety context stays on the global one.
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
        global_base_dir: Optional[str] = None,
    ) -> None:
        self._agent_id = agent_id
        self._division_id = division_id
        self._org_id = org_id
        self._base_dir = Path(base_dir).expanduser()

        # v0.13 MEMS-01 / ROADMAP critique amendment #4 (owner-ruled): per-agent STATE may follow a
        # workspace layer, but org/division SAFETY CONTEXT never does. A workspace can ADD context in
        # a later milestone; it can never silence the global voice by not having the file. Defaults
        # to base_dir, so omitting it is a no-op (bench/runner.py relies on that).
        self._global_base_dir = (
            Path(global_base_dir).expanduser() if global_base_dir else self._base_dir
        )

        # Agent paths
        self._agent_dir = self._base_dir / "agents" / agent_id
        self._db_path = self._agent_dir / "memory.db"
        self._history_path = self._agent_dir / "history.jsonl"
        self._notes_path = self._agent_dir / "MEMORY.md"

        # Division / org paths (read-only) — GLOBAL layer, always (amendment #4)
        self._division_dir = self._global_base_dir / "divisions" / division_id
        self._division_md_path = self._division_dir / "DIVISION.md"
        self._org_dir = self._global_base_dir / "orgs" / org_id
        self._guardrails_path = self._org_dir / "GUARDRAILS.md"

        self._history_writer = HistoryWriter(self._history_path)
        self._markdown_memory = MarkdownMemory(self._notes_path)
        self._bus = bus
        self._db: Optional[aiosqlite.Connection] = None
        self._subscription_handles: list["SubscriptionHandle"] = []
        # Live session id — the default provenance stamped on writes (WRITE-04).
        self._current_session_id: str | None = None
        # One-shot guard: the archive mirror still matches `facts` column-for-column (v9).
        self._archive_cols_checked = False

    # ------------------------------------------------------------------
    # Identity (read-only accessors — the two facts a caller outside this module
    # legitimately needs about a store handle without reaching for an underscore)
    # ------------------------------------------------------------------

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def base_dir(self) -> Path:
        """The config dir this store's `agents/` tree lives under."""
        return self._base_dir

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def open(self, *, owner_init: bool = True) -> None:
        """Open SQLite connection, enable WAL mode, apply pending migrations.

        Phase 33.1 (ORCH-02): performs a one-time, idempotent root-rename migration
        first — a pre-rename 'default' store is adopted (directory + facts/sessions
        rows re-keyed) the first time it opens as 'orchestrator'.

        `owner_init=False` opens a store this session does NOT own — the recall router's
        machine-global twin, held by a project session that only reads it (v0.13 B1). It
        skips the two open-time writes that belong to the store's owner: the Phase-33.1
        legacy adoption (renaming another install's `agents/default/` tree, and re-keying
        its rows, is not a reader's call to make) and `_seed_tags` (INSERTing the seeded
        tag spine into a database this session does not own).

        HONEST REMAINDER — `owner_init=False` is not a zero-write open. It still mkdirs the
        agent directory, creates `memory.db` when absent, and applies pending SCHEMA
        migrations. Neither is separable from being able to query at all: aiosqlite cannot
        connect through a missing directory, and a SELECT against an unmigrated schema is an
        error rather than a read. What a non-owner open leaves untouched is the row-level
        state a session can observe — facts, tags, activation traces, staged counters — which
        is what "a project session never writes global memory" is a claim about.

        Critic M2: any failure after connect closes the connection before re-raising —
        aiosqlite's worker thread is non-daemon, and a leaked handle hangs process exit.
        """
        # Phase 33.1: must run before mkdir — mkdir would create an empty
        # agents/orchestrator/ first and the adoption rename would then refuse
        # (destination exists), silently orphaning the legacy store.
        if owner_init:
            _migrate_legacy_root_agent_dir(self._base_dir, self._agent_id)
        elif _legacy_adoption_is_pending(self._base_dir, self._agent_id):
            # Creating the destination here would make the owner's adoption refuse FOREVER
            # (it never merges into an existing directory), orphaning the tree this open was
            # only trying to read. Refuse instead: no rename, no directory, no data lost.
            raise LegacyStoreAwaitingAdoption(
                f"{self._base_dir / 'agents' / _LEGACY_ROOT_AGENT_ID} is a pre-rename memory "
                f"tree that no session has adopted yet. Start one session outside a project "
                f"(the adoption is the store owner's to make), then this session can read it."
            )
        self._agent_dir.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        try:
            await self._open_inner(owner_init=owner_init)
        except BaseException:
            db, self._db = self._db, None
            try:
                await db.close()
            except Exception:
                pass
            raise

    async def _open_inner(self, *, owner_init: bool = True) -> None:
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode = WAL")
        await self._db.execute("PRAGMA synchronous = NORMAL")
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._db.execute("PRAGMA temp_store = MEMORY")
        # Overlapping writers (agent loop vs idle consolidation) wait instead of throwing
        # "database is locked" (critic CONS-02 groundwork).
        await self._db.execute("PRAGMA busy_timeout = 5000")
        # The one salience currency as a registered scalar (SQLite has no ln of its
        # own): ln(1+standing) + truth log-odds + declared stakes. Every ranked
        # surface — the injected block, resonance search's secondary axis, and the
        # owner's keyword search — orders by this same function.
        await self._db.create_function("lh_salience", 4, _salience_sql, deterministic=True)
        await self._apply_migrations(owner_init=owner_init)

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

    async def _apply_migrations(self, *, owner_init: bool = True) -> None:
        """Stepwise ladder; each rewrite script is a single transaction that stamps
        user_version itself (critic M1: crash → rollback → clean retry; never a
        half-migrated DB, never a double-run).

        `owner_init=False` (a non-owner open) still runs the SCHEMA ladder — a query against
        an unmigrated schema is an error, not a read — but skips the Phase-33.1 identity
        re-key below, which is data adoption rather than schema and belongs to the owner."""
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
        if v == 4:
            await self._db.executescript(MIGRATION_V4_TO_V5_SQL)
            v = await _version()
        if v == 5:
            await self._db.executescript(MIGRATION_V5_TO_V6_SQL)
            v = await _version()
        if v == 6:
            await self._db.executescript(MIGRATION_V6_TO_V7_SQL)
            v = await _version()
        if v == 7:
            await self._db.executescript(MIGRATION_V7_TO_V8_SQL)
            v = await _version()
        if v == 8:
            await self._db.executescript(MIGRATION_V8_TO_V9_SQL)
            v = await _version()
        if v == 9:
            await self._db.executescript(MIGRATION_V9_TO_V10_SQL)
            v = await _version()

        # Phase 33.1 (ORCH-02): one-time root-rename row fixup. Directory adoption alone
        # is NOT enough — every read filters WHERE agent_id = ?, so rows stamped 'default'
        # are invisible to a store opened as 'orchestrator'. Idempotent (matches 0 rows
        # once migrated), scoped to the root store only, and both tables commit in ONE
        # transaction (critic M1: crash -> rollback -> clean retry, never a half-migrated
        # identity). No unique-index conflict is possible: a store directory only ever
        # contains its own agent's rows, so 'default' and 'orchestrator' rows never coexist.
        if self._agent_id == _ROOT_AGENT_ID and owner_init:
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
        confidence: float | None = None,
        source: str = "",
        expires_at: int | None = None,
        provenance: str | None = None,
        node_kind: str = "fact",
        importance: float | None = None,
        embedding: bytes | None = None,
        truth_logodds: float | None = None,
        _retried: bool = False,
    ) -> Fact:
        """Write a fact with supersede-not-overwrite semantics (WRITE-01/02/04) —
        every write a BET, every re-sighting EVIDENCE (memory spec, write side).

        - No active row for `key` → insert a new active row. Birth truth = the
          writer's MEASURED track record (Laplace precision from the `writers`
          ledger, in log-odds), unless the caller carries `truth_logodds` or a
          legacy `confidence` forward. The write is tallied as a bet.
        - Active row with the IDENTICAL value → a re-sighting: EVIDENCE on that
          row, never a new row. From a DIFFERENT provenance (a different episode),
          the sighting writer's measured log-odds weight is ADDED to the row's
          truth and the row's original writer is tallied `confirmed` — a cold
          writer's weight is 0, so weak sources mathematically cannot push belief
          high. Same-episode re-assertion is a plain touch. No ladders, no caps.
        - Active row with a DIFFERENT value → the old row is superseded
          (status='superseded', superseded_by=<new id>) and its writer tallied
          `contradicted`; a fresh active row is inserted as a new bet. Nothing is
          overwritten or deleted; history stays queryable.

        `importance` is DECLARED stakes carried with the fact (the one human
        input); unset = 0.0 — nothing invents importance anymore.
        `embedding` is the trace's encoding vector (resonance.pack); writers that
        cannot embed leave it NULL and dreaming backfills.

        Every write is READ-BACK-VERIFIED: the active row is re-read and compared
        before the write is claimed; a mismatch raises MemoryVerifyError.
        """
        if confidence is not None and not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {confidence}")
        assert self._db is not None
        now = int(time.time())
        tags_json = json.dumps(tags or [])
        prov = provenance if provenance is not None else (self._current_session_id or "")
        writer = source or ""

        existing = await self._get_fact_row(key)
        if existing is not None and existing.value == value:
            # Re-sighting: evidence on the existing row, never a new row.
            old_truth = (existing.truth_logodds if existing.truth_logodds is not None
                         else _logit(existing.confidence))
            distinct_episode = bool(prov) and bool(existing.provenance) and existing.provenance != prov
            new_truth = old_truth
            if distinct_episode:
                new_truth = old_truth + _logit(await self.writer_precision(writer))
                if existing.source:
                    await self._bump_writer(existing.source, "confirmed")
            await self._db.execute(
                "UPDATE facts SET updated_at = ?, confidence = ?, truth_logodds = ?, "
                "expires_at = ?, tags = ?, node_kind = ?, "
                "provenance = CASE WHEN ? = '' THEN provenance ELSE ? END, "
                "source = CASE WHEN ? = '' THEN source ELSE ? END "
                "WHERE agent_id = ? AND key = ? AND status = 'active'",
                (now, _sigmoid(new_truth), new_truth, expires_at, tags_json, node_kind,
                 prov, prov, source, source, self._agent_id, key),
            )
            await self._db.commit()
        else:
            if truth_logodds is not None:
                birth_truth = float(truth_logodds)
            elif confidence is not None:
                # Legacy carry-forward (owner edits, restores): the caller holds an
                # already-priced belief; log-odds round-trips it, invents nothing.
                birth_truth = _logit(confidence)
            else:
                birth_truth = _logit(await self.writer_precision(writer))
            if existing is not None:
                # Supersede: vacate the active-unique slot, insert successor, link —
                # and the old claim's writer is tallied contradicted (an outcome).
                await self._db.execute(
                    "UPDATE facts SET status = 'superseded', updated_at = ? "
                    "WHERE agent_id = ? AND key = ? AND status = 'active'",
                    (now, self._agent_id, key),
                )
                if existing.source:
                    await self._bump_writer(existing.source, "contradicted")
            try:
                cur = await self._db.execute(
                    "INSERT INTO facts (agent_id, division_id, org_id, key, value, tags, confidence, "
                    "source, created_at, updated_at, expires_at, status, provenance, importance, "
                    "node_kind, truth_logodds, standing, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, 0.0, ?)",
                    (self._agent_id, self._division_id, self._org_id, key, value,
                     tags_json, _sigmoid(birth_truth), source, now, now, expires_at, prov,
                     0.0 if importance is None else float(importance),
                     node_kind, birth_truth, embedding),
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
                    importance=importance, embedding=embedding,
                    truth_logodds=truth_logodds, _retried=True,
                )
            new_id = cur.lastrowid
            if existing is not None:
                await self._db.execute(
                    "UPDATE facts SET superseded_by = ? "
                    "WHERE agent_id = ? AND key = ? AND status = 'superseded' AND superseded_by IS NULL",
                    (new_id, self._agent_id, key),
                )
            await self._bump_writer(writer, "bets")
            await self._db.commit()

        fact = await self._get_fact_row(key)
        if fact is None or fact.value != value:
            raise MemoryVerifyError(key)
        return fact

    # ------------------------------------------------------------------
    # Writers — the bet ledger (memory spec: "creation learns from forgetting")
    # ------------------------------------------------------------------

    async def _bump_writer(self, writer: str, column: str, n: int = 1) -> None:
        """Increment one tally for one writer (row created on first touch). Part of the
        caller's transaction — no commit here."""
        assert self._db is not None
        assert column in {"bets", "confirmed", "contradicted", "paid", "lost"}
        await self._db.execute(
            f"INSERT INTO writers (agent_id, writer, {column}) VALUES (?, ?, ?) "
            f"ON CONFLICT(agent_id, writer) DO UPDATE SET {column} = {column} + ?",
            (self._agent_id, writer, n, n),
        )

    async def writer_precision(self, writer: str) -> float:
        """The writer's measured track record as a probability — Laplace-smoothed
        precision over confirmed/contradicted outcomes. A writer with no history is
        exactly 0.5 (log-odds 0): no invented confidence, belief must be earned."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT confirmed, contradicted FROM writers WHERE agent_id = ? AND writer = ?",
            (self._agent_id, writer),
        ) as cur:
            row = await cur.fetchone()
        confirmed, contradicted = (row[0], row[1]) if row else (0, 0)
        return (confirmed + 1) / (confirmed + contradicted + 2)

    async def settle_writer_outcomes(self) -> int:
        """Recompute each writer's paid/lost tallies from the store's own tables —
        uniform statistics over everything the writer ever wrote (active, superseded,
        archived): paid = rows that were actually recalled; lost = rows that were
        archived without ever being recalled. Returns writers updated."""
        assert self._db is not None
        sql = """
            SELECT writer,
                   SUM(recalled)                    AS paid,
                   SUM(archived * (1 - recalled))   AS lost
            FROM (
                SELECT source AS writer,
                       CASE WHEN access_count + access_count_staged > 0 THEN 1 ELSE 0 END AS recalled,
                       0 AS archived
                FROM facts WHERE agent_id = :agent
                UNION ALL
                SELECT source AS writer,
                       CASE WHEN access_count + access_count_staged > 0 THEN 1 ELSE 0 END AS recalled,
                       1 AS archived
                FROM facts_archive WHERE agent_id = :agent
            )
            GROUP BY writer
        """
        async with self._db.execute(sql, {"agent": self._agent_id}) as cur:
            rows = await cur.fetchall()
        for writer, paid, lost in rows:
            await self._db.execute(
                "INSERT INTO writers (agent_id, writer, paid, lost) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(agent_id, writer) DO UPDATE SET paid = ?, lost = ?",
                (self._agent_id, writer or "", paid or 0, lost or 0, paid or 0, lost or 0),
            )
        await self._db.commit()
        return len(rows)

    _FACT_COLS = (
        "key, value, agent_id, division_id, org_id, tags, confidence, source, "
        "created_at, updated_at, expires_at, status, superseded_by, provenance, id, "
        "retrieval_strength, importance, access_count, last_accessed_at, node_kind, "
        "truth_logodds, standing"
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

    async def get_fact_by_id(self, fact_id: int) -> Fact | None:
        """Read ONE fact by id in ANY status (active|superseded), no expiry filter — the
        `/memory show` + `forget` lookup path. Active-only reads (get_fact / get_facts_by_ids)
        can't reach a superseded row, but the window must show a forgotten/replaced memory too
        (auditable history). Read-only + WAL-safe: never blocks on a concurrent live-turn write."""
        assert self._db is not None
        async with self._db.execute(
            f"SELECT {self._FACT_COLS} FROM facts WHERE agent_id = ? AND id = ?",
            (self._agent_id, fact_id),
        ) as cur:
            row = await cur.fetchone()
        return _row_to_fact(row) if row else None

    async def recent_facts(self, limit: int = 10) -> list[Fact]:
        """The most-recently-written/updated ACTIVE facts, newest first — the `/memory` overview
        feed. ONE indexed query (idx_facts_active_recency), no scoring and no model calls."""
        assert self._db is not None
        now = int(time.time())
        async with self._db.execute(
            f"SELECT {self._FACT_COLS} FROM facts INDEXED BY idx_facts_active_recency "
            "WHERE agent_id = ? AND status = 'active' AND (expires_at IS NULL OR expires_at > ?) "
            "ORDER BY updated_at DESC, id DESC LIMIT ?",
            (self._agent_id, now, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_fact(r) for r in rows]

    async def forget_fact(self, fact_id: int) -> bool:
        """User-initiated forget (WRITE-02 ethic: retire, NEVER hard-delete). Supersede the ACTIVE
        fact with a user-forget marker — flip status to 'superseded', drop retrieval_strength to
        the supersede floor (the exact interference mechanism store_fact's supersede branch uses),
        and stamp a `user_forget@<ts>` provenance marker (original provenance preserved after ';'
        for audit). The row stays queryable via get_fact_history / get_fact_by_id (the chain stays
        auditable) but leaves every active hot path — query_facts, atoms_for_tag, the ambient
        render. ATOMIC + race-safe: the UPDATE is guarded by status='active', so a fact a live turn
        just superseded returns False (nothing retired) instead of double-writing. Returns True iff
        a row was retired."""
        assert self._db is not None
        now = int(time.time())
        async with self._db.execute(
            "UPDATE facts SET status = 'superseded', "
            "retrieval_strength = MIN(retrieval_strength, 0.1), "
            "provenance = ? || ? || CASE WHEN provenance = '' THEN '' ELSE ';' || provenance END, "
            "updated_at = ? WHERE agent_id = ? AND id = ? AND status = 'active'",
            (USER_FORGET_PROVENANCE_PREFIX, now, now, self._agent_id, fact_id),
        ) as cur:
            retired = cur.rowcount > 0
        await self._db.commit()
        return retired

    # ------------------------------------------------------------------
    # Cold archive (schema v9 — memory-forgetting rung 1). A move, not a delete:
    # the row leaves the hot table (and the FTS index with it, via the existing
    # facts_ad trigger) into `facts_archive`, and restore moves it back
    # byte-identical. Both directions are ONE transaction with a read-back parity
    # check between the write and the removal — the same "never claim a write you
    # didn't verify" discipline store_fact holds, applied to a two-table move.
    # ------------------------------------------------------------------

    # The FULL physical row, in table order — NOT the `_FACT_COLS` projection. A restore
    # must return the row byte-identical, which means carrying the columns the Fact
    # projection drops (the staged read counters) as well as the ones it exposes.
    _ARCHIVE_ROW_COLS: tuple[str, ...] = (
        "id", "agent_id", "division_id", "org_id", "key", "value", "tags", "confidence",
        "source", "created_at", "updated_at", "expires_at", "status", "superseded_by",
        "provenance", "retrieval_strength", "importance", "access_count",
        "last_accessed_at", "access_count_staged", "last_accessed_staged", "node_kind",
        "truth_logodds", "standing", "embedding",
    )

    async def _assert_archive_columns(self) -> None:
        """Fail LOUDLY if `facts` has grown a column this move would silently drop.

        A migration that adds a column to `facts` and forgets `facts_archive` would make
        archival lossy and restore a liar — exactly the class the read-back verify exists
        to catch, one level up. Checked once per store (cheap), never guessed around.
        """
        if self._archive_cols_checked:
            return
        assert self._db is not None
        async with self._db.execute("PRAGMA table_info(facts)") as cur:
            live = tuple(r[1] for r in await cur.fetchall())
        if live != self._ARCHIVE_ROW_COLS:
            raise MemoryCorruptionError(
                str(self._db_path),
                f"facts columns {live} do not match the archive mirror "
                f"{self._ARCHIVE_ROW_COLS} — a migration added a column without "
                f"extending facts_archive; archival would lose it",
            )
        self._archive_cols_checked = True

    async def archive_fact(
        self,
        fact_id: int,
        *,
        surface: str,
        s_at_archive: float,
        line_at_archive: float | None,
    ) -> bool:
        """Move one ACTIVE fact out of the hot store into the cold archive.

        `surface` names WHAT condemned the row (ARCHIVE_SURFACE_*); it is stamped as
        `archived@<epoch>;<surface>` using the same epoch written to archived_at, so the
        two can never disagree. Both land in the archive's metadata columns — the fact row
        itself is written back unchanged, which is what makes restore byte-identical.

        Returns False (and changes nothing) when the row is gone, already superseded, or
        carries UNFOLDED reads — `access_count_staged > 0` means the fact was recalled
        since the last fold, so it is by definition freshly used and not dormant, whatever
        its (stale) folded counters say. THIS is the execution-time rail: it holds no
        matter which surface asked, including an external list that names the row outright.

        Raises MemoryVerifyError if the archived copy does not match the source row
        field-for-field, or if the source row survives the delete — the move is rolled
        back first, so a verify failure leaves the fact hot and intact.
        """
        assert self._db is not None
        await self._assert_archive_columns()
        cols = ", ".join(self._ARCHIVE_ROW_COLS)
        async with self._db.execute(
            f"SELECT {cols} FROM facts WHERE id = ? AND agent_id = ? "
            "AND status = 'active' AND access_count_staged = 0",
            (fact_id, self._agent_id),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return False
        source_row = tuple(row)
        key = source_row[self._ARCHIVE_ROW_COLS.index("key")]
        try:
            marks = ", ".join("?" * (len(self._ARCHIVE_ROW_COLS) + 4))
            now = int(time.time())
            await self._db.execute(
                f"INSERT INTO facts_archive ({cols}, archived_at, archive_rung, "
                f"s_at_archive, line_at_archive) VALUES ({marks})",
                (*source_row, now, archive_stamp(surface, now),
                 float(s_at_archive), line_at_archive),
            )
            async with self._db.execute(
                f"SELECT {cols} FROM facts_archive WHERE id = ?", (fact_id,)
            ) as cur:
                written = await cur.fetchone()
            if written is None or tuple(written) != source_row:
                raise MemoryVerifyError(key)
            await self._db.execute(
                "DELETE FROM facts WHERE id = ? AND agent_id = ? AND status = 'active'",
                (fact_id, self._agent_id),
            )
            async with self._db.execute(
                "SELECT COUNT(*) FROM facts WHERE id = ?", (fact_id,)
            ) as cur:
                (still_hot,) = await cur.fetchone()
            if still_hot:
                raise MemoryVerifyError(key)
        except BaseException:
            await self._db.rollback()
            raise
        await self._db.commit()
        return True

    async def restore_fact(self, fact_id: int) -> bool:
        """Move an archived fact back into the hot store, byte-identical.

        Returns False when the id is not in the archive, or when a NEWER active fact now
        holds that name (the active-key unique index refuses it — the live row wins, and
        the archived copy stays safe in the archive rather than being forced over it).
        """
        assert self._db is not None
        await self._assert_archive_columns()
        cols = ", ".join(self._ARCHIVE_ROW_COLS)
        async with self._db.execute(
            f"SELECT {cols} FROM facts_archive WHERE id = ? AND agent_id = ?",
            (fact_id, self._agent_id),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return False
        archived_row = tuple(row)
        key = archived_row[self._ARCHIVE_ROW_COLS.index("key")]
        marks = ", ".join("?" * len(self._ARCHIVE_ROW_COLS))
        try:
            await self._db.execute(
                f"INSERT INTO facts ({cols}) VALUES ({marks})", archived_row
            )
        except sqlite3.IntegrityError:
            await self._db.rollback()
            return False
        try:
            async with self._db.execute(
                f"SELECT {cols} FROM facts WHERE id = ?", (fact_id,)
            ) as cur:
                written = await cur.fetchone()
            if written is None or tuple(written) != archived_row:
                raise MemoryVerifyError(key)
            await self._db.execute("DELETE FROM facts_archive WHERE id = ?", (fact_id,))
            async with self._db.execute(
                "SELECT COUNT(*) FROM facts_archive WHERE id = ?", (fact_id,)
            ) as cur:
                (still_archived,) = await cur.fetchone()
            if still_archived:
                raise MemoryVerifyError(key)
        except BaseException:
            await self._db.rollback()
            raise
        await self._db.commit()
        return True

    async def list_archived(self, limit: int = 50, *, key: str | None = None) -> list[Fact]:
        """Archived facts, most recently archived first (the `--archived` listing)."""
        assert self._db is not None
        sql = (f"SELECT {self._FACT_COLS} FROM facts_archive WHERE agent_id = ?"
               + (" AND key = ?" if key else "")
               + " ORDER BY archived_at DESC, id DESC LIMIT ?")
        params = (self._agent_id, key, limit) if key else (self._agent_id, limit)
        async with self._db.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [_row_to_fact(r) for r in rows]

    async def count_archived(self) -> int:
        assert self._db is not None
        async with self._db.execute(
            "SELECT COUNT(*) FROM facts_archive WHERE agent_id = ?", (self._agent_id,)
        ) as cur:
            (n,) = await cur.fetchone()
        return n

    async def vacuum(self) -> None:
        """Return the pages a big archival move freed to the filesystem. Cannot run inside
        a transaction, so it is called after the moves have committed."""
        assert self._db is not None
        await self._db.execute("VACUUM")
        await self._db.commit()

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

        # Owner-surface keyword lookup: the FTS MATCH decides WHO is a hit (text
        # matching at query time is search — allowed), and the one salience currency
        # decides the ORDER. No clock, no BM25 blending, no second scorer.
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
                ORDER BY lh_salience(f.standing, f.truth_logodds, f.confidence, f.importance) DESC,
                    f.updated_at DESC, f.key ASC
                LIMIT ?
            """
            params: list[Any] = [fts_text, self._agent_id, query.min_confidence, now, *temporal_params, query.limit]
        else:
            sql = f"""
                SELECT {prefixed_cols}
                FROM facts f
                WHERE f.agent_id = ?
                  AND f.confidence >= ?
                  AND (f.expires_at IS NULL OR f.expires_at > ?)
                  {status_filter}
                  {temporal_filter}
                ORDER BY lh_salience(f.standing, f.truth_logodds, f.confidence, f.importance) DESC,
                    f.updated_at DESC, f.key ASC
                LIMIT ?
            """
            params = [self._agent_id, query.min_confidence, now, *temporal_params, query.limit]

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
        by the idle dreaming pass. Returns rows folded."""
        assert self._db is not None
        cur = await self._db.execute(
            "UPDATE facts SET access_count = access_count + access_count_staged, "
            "last_accessed_at = COALESCE(last_accessed_staged, last_accessed_at), "
            "access_count_staged = 0, last_accessed_staged = NULL "
            "WHERE agent_id = ? AND access_count_staged > 0",
            (self._agent_id,),
        )
        await self._db.commit()
        return cur.rowcount

    # ------------------------------------------------------------------
    # Resonance surface (memory spec, roles 2 + 3): vectors in, rankings and
    # standing out. The similarity JUDGMENT itself lives in memory/resonance.py —
    # the store only holds vectors and tallies.
    # ------------------------------------------------------------------

    async def resonance_search(
        self,
        query_vec: Any,
        *,
        limit: int = 10,
        since: int | None = None,
        until: int | None = None,
    ) -> list[tuple[Fact, float]]:
        """Active facts ranked by resonance with the probe vector (descending).

        Facts not yet embedded are invisible here until dreaming backfills them —
        there is deliberately NO lexical or statistical fallback ranking behind
        this method. Returns (fact, resonance) pairs."""
        from localharness.memory import resonance as _res

        assert self._db is not None
        temporal = ""
        params: list[Any] = [self._agent_id]
        if since is not None:
            temporal += " AND updated_at >= ?"
            params.append(since)
        if until is not None:
            temporal += " AND updated_at <= ?"
            params.append(until)
        async with self._db.execute(
            f"SELECT id, embedding FROM facts WHERE agent_id = ? AND status = 'active' "
            f"AND embedding IS NOT NULL{temporal}",
            params,
        ) as cur:
            id_blobs = [(r[0], r[1]) for r in await cur.fetchall()]
        # Zero/anti-resonance is NOT a match: the model judged no resemblance. The sign
        # boundary is the same non-arbitrary cut the shares arithmetic uses — no tuned
        # floor exists here, the limit does the real work.
        ranked = [(i, s) for i, s in _res.rank(query_vec, id_blobs) if s > 0.0][: max(0, limit)]
        if not ranked:
            return []
        facts = {f.id: f for f in await self.get_facts_by_ids([i for i, _ in ranked])}
        return [(facts[i], score) for i, score in ranked if i in facts]

    async def active_embedded(self) -> list[tuple[int, bytes]]:
        """(id, embedding) for every active embedded fact — dreaming's competition set."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT id, embedding FROM facts WHERE agent_id = ? AND status = 'active' "
            "AND embedding IS NOT NULL",
            (self._agent_id,),
        ) as cur:
            return [(r[0], r[1]) for r in await cur.fetchall()]

    async def facts_missing_embedding(self, *, limit: int = 512) -> list[Fact]:
        """Active facts with no vector yet (owner edits, restores, pre-v10 rows) —
        dreaming embeds these first so every trace can resonate."""
        assert self._db is not None
        async with self._db.execute(
            f"SELECT {self._FACT_COLS} FROM facts WHERE agent_id = ? AND status = 'active' "
            "AND embedding IS NULL ORDER BY id LIMIT ?",
            (self._agent_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_fact(r) for r in rows]

    async def all_embedded_fact_ids(self) -> list[int]:
        assert self._db is not None
        async with self._db.execute(
            "SELECT id FROM facts WHERE agent_id = ? AND embedding IS NOT NULL",
            (self._agent_id,),
        ) as cur:
            return [r[0] for r in await cur.fetchall()]

    async def set_fact_embedding(self, fact_id: int, blob: bytes | None) -> None:
        assert self._db is not None
        await self._db.execute(
            "UPDATE facts SET embedding = ? WHERE id = ? AND agent_id = ?",
            (blob, fact_id, self._agent_id),
        )
        await self._db.commit()

    async def add_standing(self, deltas: dict[int, float]) -> None:
        """Accumulate resonance-share mass onto facts (one dreaming batch, one txn)."""
        if not deltas:
            return
        assert self._db is not None
        await self._db.executemany(
            "UPDATE facts SET standing = standing + ? WHERE id = ? AND agent_id = ?",
            [(share, fid, self._agent_id) for fid, share in deltas.items()],
        )
        await self._db.commit()

    async def apply_digest(self, deltas: dict[int, float], marks: dict[str, int]) -> None:
        """One dreaming digest, atomically: the standing mass a batch of stream windows
        distributed AND the ledger offsets that cover them commit together, so a crash
        can never double-count a window (marks behind standing) or skip one (ahead)."""
        assert self._db is not None
        if deltas:
            await self._db.executemany(
                "UPDATE facts SET standing = standing + ? WHERE id = ? AND agent_id = ?",
                [(share, fid, self._agent_id) for fid, share in deltas.items()],
            )
        if marks:
            await self._db.executemany(
                "INSERT INTO digest_marks (agent_id, path, byte_offset) VALUES (?, ?, ?) "
                "ON CONFLICT(agent_id, path) DO UPDATE SET byte_offset = excluded.byte_offset",
                [(self._agent_id, p, off) for p, off in marks.items()],
            )
        await self._db.commit()

    # -- digest marks: how much stream has been read. The only clock. ----------

    async def get_digest_marks(self) -> dict[str, int]:
        assert self._db is not None
        async with self._db.execute(
            "SELECT path, byte_offset FROM digest_marks WHERE agent_id = ?",
            (self._agent_id,),
        ) as cur:
            return {r[0]: r[1] for r in await cur.fetchall()}

    async def set_digest_marks(self, marks: dict[str, int]) -> None:
        assert self._db is not None
        await self._db.executemany(
            "INSERT INTO digest_marks (agent_id, path, byte_offset) VALUES (?, ?, ?) "
            "ON CONFLICT(agent_id, path) DO UPDATE SET byte_offset = excluded.byte_offset",
            [(self._agent_id, p, off) for p, off in marks.items()],
        )
        await self._db.commit()

    # -- named groups: dreaming's bindings (labels for legibility, never mechanism) --

    async def upsert_group(self, member_ids: list[int], label: str | None = None) -> int:
        """Strengthen the group with exactly this member set, or mint it. A repeat
        observation bumps `evidence`; a label fills in when one arrives (first name
        wins — labels are display, not identity). Returns the group id."""
        assert self._db is not None
        members_json = json.dumps(sorted(set(member_ids)))
        now = int(time.time())
        async with self._db.execute(
            "SELECT id, label FROM memory_groups WHERE agent_id = ? AND member_ids = ?",
            (self._agent_id, members_json),
        ) as cur:
            row = await cur.fetchone()
        if row is not None:
            gid, old_label = row[0], row[1]
            await self._db.execute(
                "UPDATE memory_groups SET evidence = evidence + 1, updated_at = ?, "
                "label = CASE WHEN label = '' THEN ? ELSE label END WHERE id = ?",
                (now, label or "", gid),
            )
            await self._db.commit()
            return gid
        cur2 = await self._db.execute(
            "INSERT INTO memory_groups (agent_id, label, member_ids, evidence, created_at, updated_at) "
            "VALUES (?, ?, ?, 1, ?, ?)",
            (self._agent_id, label or "", members_json, now, now),
        )
        await self._db.commit()
        return cur2.lastrowid

    async def set_group_label(self, group_id: int, label: str) -> None:
        """Fill a group's label (display identity, never mechanism) — no evidence bump:
        naming a binding is not a second observation of it."""
        assert self._db is not None
        await self._db.execute(
            "UPDATE memory_groups SET label = ?, updated_at = ? WHERE id = ? AND agent_id = ?",
            (label, int(time.time()), group_id, self._agent_id),
        )
        await self._db.commit()

    async def list_groups(self, *, limit: int | None = None, named_only: bool = False) -> list[dict[str, Any]]:
        assert self._db is not None
        where = " AND label != ''" if named_only else ""
        lim = " LIMIT ?" if limit is not None else ""
        params: tuple[Any, ...] = (self._agent_id, limit) if limit is not None else (self._agent_id,)
        async with self._db.execute(
            f"SELECT id, label, member_ids, evidence, created_at, updated_at "
            f"FROM memory_groups WHERE agent_id = ?{where} "
            f"ORDER BY evidence DESC, updated_at DESC, id ASC{lim}",
            params,
        ) as cur:
            rows = await cur.fetchall()
        return [
            {"id": r[0], "label": r[1], "member_ids": json.loads(r[2] or "[]"),
             "evidence": r[3], "created_at": r[4], "updated_at": r[5]}
            for r in rows
        ]

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
    # Activation-trace log (tag-graph substrate P0) — append-only bookkeeping.
    # One row per retrieval event; NO update/delete path. Best-effort at the call
    # site: a trace-write failure must never fail the retrieval itself. Traces cannot
    # be backfilled, so the log ships before any consumer (co-activation weights,
    # pattern-completion retrieval) exists. Read helpers are plain log reads for tests
    # + forensics — no scoring, no weights, no spreading.
    # ------------------------------------------------------------------

    async def record_activation_trace(
        self, *, stimulus: str, fired_ids: list[int], injected_ids: list[int],
        source: str, session_id: str | None = None, turn: int | None = None,
    ) -> int:
        """Append one activation trace. `stimulus` is the raw query/recall text — the digest
        stores sha256 of the FULL text plus the text truncated to _STIMULUS_TEXT_CAP.
        `fired_ids` are the atoms the stimulus surfaced (search/recall hits); `injected_ids`
        are the subset actually rendered into context (⊆ fired). session_id falls back to the
        live session. Returns the new rowid. ONE INSERT, no reads."""
        assert self._db is not None
        text = stimulus or ""
        stim_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        sess = session_id if session_id is not None else (self._current_session_id or "")
        cur = await self._db.execute(
            "INSERT INTO activation_traces "
            "(agent_id, session_id, turn, stimulus_hash, stimulus_text, fired_ids, "
            "injected_ids, source, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (self._agent_id, sess, turn, stim_hash, text[:_STIMULUS_TEXT_CAP],
             json.dumps(fired_ids), json.dumps(injected_ids), source, int(time.time())),
        )
        await self._db.commit()
        return cur.lastrowid

    async def record_injection_trace(
        self, *, stimulus: str, injected_ids: list[int], session_id: str | None = None,
    ) -> int:
        """Append the every-turn ambient-shelf co-firing event (owner reversal 2026-07-17 of the
        P0 exclusion), source='injection'. The shelf renders exactly what it selects, so
        fired == injected. INSERT OR IGNORE keyed on the partial unique index
        (agent_id, session_id, stimulus_hash) WHERE source='injection' — a second assembly of
        the SAME turn (a retry re-render) collapses to the one existing row (per-turn dedupe).
        `stimulus` is the turn's user message; the digest stores sha256 of the FULL text plus
        the text truncated to _STIMULUS_TEXT_CAP. source='injection' keeps it DISTINGUISHABLE
        from model-initiated retrieval (memory_search/memory_get), which downstream discounts —
        the raw log keeps full fidelity. Best-effort at the call site; ONE INSERT, no reads.
        Returns the inserted rowid; on the dedupe (IGNORE) path nothing is inserted and the
        return is not meaningful — the caller (a best-effort loop hook) ignores it."""
        assert self._db is not None
        text = stimulus or ""
        stim_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        sess = session_id if session_id is not None else (self._current_session_id or "")
        ids_json = json.dumps(injected_ids)
        cur = await self._db.execute(
            "INSERT OR IGNORE INTO activation_traces "
            "(agent_id, session_id, turn, stimulus_hash, stimulus_text, fired_ids, "
            "injected_ids, source, ts) VALUES (?, ?, NULL, ?, ?, ?, ?, 'injection', ?)",
            (self._agent_id, sess, stim_hash, text[:_STIMULUS_TEXT_CAP],
             ids_json, ids_json, int(time.time())),
        )
        await self._db.commit()
        return cur.lastrowid

    _TRACE_COLS = (
        "id, agent_id, session_id, turn, stimulus_hash, stimulus_text, "
        "fired_ids, injected_ids, source, ts"
    )

    async def recent_activation_traces(self, *, limit: int = 50) -> list[ActivationTrace]:
        """Most-recent-first activation traces for this agent (forensics + tests)."""
        assert self._db is not None
        async with self._db.execute(
            f"SELECT {self._TRACE_COLS} FROM activation_traces "
            "WHERE agent_id = ? ORDER BY ts DESC, id DESC LIMIT ?",
            (self._agent_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_activation_trace(r) for r in rows]

    async def activation_traces_for_atom(
        self, atom_id: int, *, limit: int = 50
    ) -> list[ActivationTrace]:
        """Traces where `atom_id` was among the fired atoms, newest-first (forensics). P0 has
        no derived co-activation index (weights are a later rebuildable view per the design),
        so this scans the agent's log and filters on the parsed JSON — exact membership, not
        a substring LIKE that would confuse id 1 with 11. Read-only."""
        assert self._db is not None
        async with self._db.execute(
            f"SELECT {self._TRACE_COLS} FROM activation_traces "
            "WHERE agent_id = ? ORDER BY ts DESC, id DESC",
            (self._agent_id,),
        ) as cur:
            rows = await cur.fetchall()
        out: list[ActivationTrace] = []
        for r in rows:
            t = _row_to_activation_trace(r)
            if atom_id in t.fired_ids:
                out.append(t)
                if len(out) >= limit:
                    break
        return out

    # ------------------------------------------------------------------
    # Mining-residue ledger (schema v7) — the repair half of the extraction loop.
    # RECORD-level lifecycle: pending -> rescued (a later look minted from it) or
    # -> retired (barren after `attempt_cap` isolated looks). Retire = out of the
    # mining window ONLY; the history record itself is never touched (append-only).
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
        max_chars: int = 16_000,
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
            agent_md, injected_fact_ids = await self._render_memory_index_with_ids(
                max_session_history, max_chars=max_chars
            )
        else:
            agent_md, injected_fact_ids = self._markdown_memory.read(), []

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
            injected_fact_ids=injected_fact_ids,
        )

    async def _render_memory_index(self, max_session_history: int, *, max_chars: int = 16_000) -> str:
        """Render the agent-memory INDEX string (the byte-stability-critical injected surface).
        Thin wrapper over `_render_memory_index_with_ids` — returns ONLY the string, preserving
        the exact public contract every byte-identity test asserts against; the rendered atom
        ids (for the ambient-injection trace) ride out separately via load_context."""
        text, _ids = await self._render_memory_index_with_ids(max_session_history, max_chars=max_chars)
        return text

    async def _render_memory_index_with_ids(
        self,
        max_session_history: int,
        *,
        max_chars: int = 16_000,
        origin_label: str = "",
        include_preamble: bool = True,
        exclude_keys: set[str] | None = None,
    ) -> tuple[str, list[int]]:
        """Render the agent-memory INDEX under the loading budget (memory spec).

        The doorway from the store into working memory: every active fact competes in
        ONE salience currency — S = ln(1 + standing) + truth log-odds + declared
        stakes — and the char budget (`max_chars`, the config's own anchor,
        agent.memory.max_notes_chars) is the WHOLE gate: lines render in S order until
        the budget is spent. No confidence bars, no strength floors, no count caps.
        Byte-stable between store mutations by construction: standing, truth and
        stakes move only at writes and dreaming passes — no clock term anywhere.

        Dreaming's NAMED groups render first (labels for human legibility, never
        mechanism), inside the same budget. Session-history entries render from the
        sessions table as before.

        Also returns the rendered fact ids — the "injected set" for the
        ambient-injection activation trace (groups and session entries are not atoms
        and contribute none).

        `origin_label` / `include_preamble` / `exclude_keys`: the scope-merge
        contract (v0.13 MEMS-02/B5), unchanged — a merged second block drops the
        preamble and the keys the first block already rendered."""
        assert self._db is not None
        import math

        def _line(key: str, value: str, fid: int) -> str:
            prefix = f"[{origin_label}#{fid}] " if origin_label else ""
            return f"- {prefix}{key}: {_one_line(value, 180)}"

        # Named groups — the store's own bindings, display only.
        group_rows = await self.list_groups(named_only=True)
        group_lines = [
            f"- {g['label']} ({len(g['member_ids'])} memories, seen {g['evidence']}x)"
            for g in group_rows
        ]
        groups_section = (
            f"### Groups ({len(group_lines)})\n" + "\n".join(group_lines) + "\n\n"
            if group_lines
            else ""
        )

        # Every active fact competes; salience computed in Python (SQL knows no ln, and
        # the competition set is the same order of size as what the budget admits).
        async with self._db.execute(
            "SELECT key, value, id, standing, truth_logodds, confidence, importance, "
            "updated_at FROM facts WHERE agent_id = ? AND status = 'active'",
            (self._agent_id,),
        ) as cur:
            rows = await cur.fetchall()
        if exclude_keys:
            rows = [r for r in rows if r[0] not in exclude_keys]

        def _salience(r: Any) -> float:
            truth = r[4] if r[4] is not None else _logit(r[5])
            return math.log1p(max(0.0, r[3] or 0.0)) + truth + (r[6] or 0.0)

        ranked = sorted(rows, key=lambda r: (-_salience(r), -(r[7] or 0), r[0]))

        # TIME-02/03: the injected shelf renders from the sessions TABLE — started_at
        # is full-precision epoch; `summary IS NOT NULL` excludes the open sitting and
        # vacuous sittings; LIMIT drops the oldest rows WHOLE.
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
        today_local = datetime.now().astimezone().date()
        entry_lines = []
        for started_at, summary in sess_rows:
            dt_local = datetime.fromtimestamp(started_at).astimezone()
            label = _relative_day_label(dt_local.date(), today_local)
            entry_lines.append(
                f"- {label} {_clock_label(dt_local)}: {_one_line(summary, 180)}"
            )
        history_section = (
            f"\n\n### Recent Session History (last {shelf_n})\n" + "\n".join(entry_lines)
            if entry_lines
            else ""
        )

        _preamble = (
            "This is an INDEX, not the full memory. Each line below is one persistent fact "
            "(name: short description). Call `memory_get(name)` for a fact's full body, or "
            "`memory_search(query)` to search memory by meaning.\n\n"
        ) if include_preamble else ""

        # The budget clears: preamble + groups + history are the fixed furniture; fact
        # lines admit in salience order until the block would exceed max_chars.
        fixed = (len(_preamble) + len(groups_section) + len(history_section)
                 + len("### Persistent Facts ()\n") + 4)
        fact_lines: list[str] = []
        injected_ids: list[int] = []
        used = fixed
        for r in ranked:
            line = _line(r[0], r[1], r[2])
            if used + len(line) + 1 > max_chars:
                break
            fact_lines.append(line)
            injected_ids.append(r[2])
            used += len(line) + 1

        facts_block = "\n".join(fact_lines) if fact_lines else "(no persistent facts)"
        text = (
            f"{_preamble}"
            f"{groups_section}### Persistent Facts ({len(fact_lines)})\n{facts_block}"
            f"{history_section}"
        )
        return text, injected_ids

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
        """Regenerate MEMORY.md from current fact store (display artifact only).
        Ordered by the one salience currency — no confidence bars, no strength floors."""
        assert self._db is not None
        import math

        async with self._db.execute(
            "SELECT key, value, updated_at, standing, truth_logodds, confidence, importance "
            "FROM facts WHERE agent_id = ? AND status = 'active'",
            (self._agent_id,),
        ) as cur:
            rows = await cur.fetchall()
        rows = sorted(
            rows,
            key=lambda r: (
                -(math.log1p(max(0.0, r[3] or 0.0))
                  + (r[4] if r[4] is not None else _logit(r[5]))
                  + (r[6] or 0.0)),
                -(r[2] or 0), r[0],
            ),
        )

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
                "finish_reason": getattr(event, "finish_reason", None) or "stop",
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
                # #133: the REAL pre-cap size, not a re-measure of what we stored (which made
                # a capped 2k result record original_length == stored_length). None means the
                # producer didn't know it — then stored length is the best honest answer.
                "original_length": (
                    event.original_length
                    if event.original_length is not None
                    else len(event.output or "")
                ),
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
# The one salience currency, as a SQL scalar (registered as lh_salience): SQLite
# cannot compute ln, so the Python function rides along. Identical math to
# memory/salience.py's score_fact — S = ln(1+standing) + truth + stakes — with
# the same legacy-confidence derivation for pre-v10 rows.
# ---------------------------------------------------------------------------


def _salience_sql(standing, truth_logodds, confidence, importance) -> float:
    import math

    need = math.log1p(max(0.0, standing or 0.0))
    truth = truth_logodds if truth_logodds is not None else _logit(confidence)
    return need + truth + (importance or 0.0)


# Log-odds <-> probability (v10). ln(p/(1-p)) diverges at the poles; the guard is the
# CONFIDENCE column's own resolution (every value the store writes is sigmoid(logodds),
# a 2-decimal display quantity) — one unit in from each pole, not a tuning knob.
_LOGIT_GUARD = 0.01


def _logit(p: float | None) -> float:
    import math

    c = min(max(p if p is not None else 0.5, _LOGIT_GUARD), 1.0 - _LOGIT_GUARD)
    return math.log(c / (1.0 - c))


def _sigmoid(x: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-x))


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
            truth_logodds=row["truth_logodds"] if "truth_logodds" in keys else None,
            standing=row["standing"] if "standing" in keys else 0.0,
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
        truth_logodds=row[20] if len(row) > 20 else None,
        standing=row[21] if len(row) > 21 else 0.0,
    )


def _row_to_activation_trace(row: aiosqlite.Row) -> ActivationTrace:
    """Reconstruct an ActivationTrace from a _TRACE_COLS row; JSON id-arrays parsed back."""
    return ActivationTrace(
        id=row["id"],
        agent_id=row["agent_id"],
        session_id=row["session_id"],
        turn=row["turn"],
        stimulus_hash=row["stimulus_hash"],
        stimulus_text=row["stimulus_text"],
        fired_ids=json.loads(row["fired_ids"]) if row["fired_ids"] else [],
        injected_ids=json.loads(row["injected_ids"]) if row["injected_ids"] else [],
        source=row["source"],
        ts=row["ts"],
    )
