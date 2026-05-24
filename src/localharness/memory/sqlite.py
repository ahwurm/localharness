"""MemoryStore: SQLite facts, sessions, FTS5, bus integration."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import aiosqlite

from localharness.memory.errors import MemoryCorruptionError, SessionNotFoundError
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


@dataclass(frozen=True)
class FactQuery:
    text: str | None = None
    tags: list[str] = field(default_factory=list)
    min_confidence: float = 0.0
    include_scopes: list[str] = field(default_factory=lambda: ["agent"])
    limit: int = 50


@dataclass(frozen=True)
class MemoryContext:
    agent_memory_md: str
    division_md: str
    guardrails_md: str
    fact_count: int
    token_estimate: int


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CURRENT_SCHEMA_VERSION = 1

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
CREATE TABLE IF NOT EXISTS notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    TEXT    NOT NULL,
    section     TEXT    NOT NULL DEFAULT 'general',
    content     TEXT    NOT NULL,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_agent_id ON notes(agent_id, section);
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def open(self) -> None:
        """Open SQLite connection, enable WAL mode, apply pending migrations."""
        self._agent_dir.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode = WAL")
        await self._db.execute("PRAGMA synchronous = NORMAL")
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._db.execute("PRAGMA temp_store = MEMORY")
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
        assert self._db is not None
        async with self._db.execute("PRAGMA user_version") as cur:
            row = await cur.fetchone()
            current_version = row[0]
        if current_version < CURRENT_SCHEMA_VERSION:
            await self._db.executescript(SCHEMA_V1_SQL)
            await self._db.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")
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

    async def store_fact(
        self,
        key: str,
        value: str,
        tags: list[str] | None = None,
        confidence: float = 1.0,
        source: str = "",
        expires_at: int | None = None,
    ) -> Fact:
        """Upsert a fact. Preserves created_at on update."""
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {confidence}")
        assert self._db is not None
        now = int(time.time())
        tags_json = json.dumps(tags or [])
        await self._db.execute(
            """
            INSERT INTO facts (agent_id, division_id, org_id, key, value, tags, confidence, source, created_at, updated_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_id, key) DO UPDATE SET
                value      = excluded.value,
                tags       = excluded.tags,
                confidence = excluded.confidence,
                source     = excluded.source,
                updated_at = excluded.updated_at,
                expires_at = excluded.expires_at
            """,
            (self._agent_id, self._division_id, self._org_id, key, value,
             tags_json, confidence, source, now, now, expires_at),
        )
        await self._db.commit()
        return await self._get_fact_row(key)  # type: ignore[return-value]

    async def _get_fact_row(self, key: str) -> Fact | None:
        """Read a fact row without expiry check (internal use)."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT key, value, agent_id, division_id, org_id, tags, confidence, source, created_at, updated_at, expires_at "
            "FROM facts WHERE agent_id = ? AND key = ?",
            (self._agent_id, key),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return _row_to_fact(row)

    async def get_fact(self, key: str) -> Fact | None:
        """Get a single fact by exact key. Returns None if not found or expired."""
        assert self._db is not None
        now = int(time.time())
        async with self._db.execute(
            "SELECT key, value, agent_id, division_id, org_id, tags, confidence, source, created_at, updated_at, expires_at "
            "FROM facts WHERE agent_id = ? AND key = ? AND (expires_at IS NULL OR expires_at > ?)",
            (self._agent_id, key, now),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return _row_to_fact(row)

    async def delete_fact(self, key: str) -> bool:
        """Delete a fact by key. Returns True if a row was deleted."""
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

        if query.text:
            sql = """
                SELECT f.key, f.value, f.agent_id, f.division_id, f.org_id, f.tags,
                       f.confidence, f.source, f.created_at, f.updated_at, f.expires_at
                FROM facts f
                JOIN facts_fts ON facts_fts.rowid = f.id
                WHERE facts_fts MATCH ?
                  AND f.agent_id = ?
                  AND f.confidence >= ?
                  AND (f.expires_at IS NULL OR f.expires_at > ?)
                ORDER BY rank
                LIMIT ?
            """
            params: list[Any] = [query.text, self._agent_id, query.min_confidence, now, query.limit]
        else:
            sql = """
                SELECT key, value, agent_id, division_id, org_id, tags, confidence, source,
                       created_at, updated_at, expires_at
                FROM facts
                WHERE agent_id = ?
                  AND confidence >= ?
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY updated_at DESC
                LIMIT ?
            """
            params = [self._agent_id, query.min_confidence, now, query.limit]

        async with self._db.execute(sql, params) as cur:
            rows = await cur.fetchall()

        facts = [_row_to_fact(r) for r in rows]

        if query.tags:
            facts = [f for f in facts if any(t in f.tags for t in query.tags)]

        return facts

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
    # Session reconstruction
    # ------------------------------------------------------------------

    async def reconstruct_session(self, session_id: str) -> list[dict[str, Any]]:
        """Reconstruct LLM message format from history JSONL with compaction + orphan guard."""
        all_records = await self._history_writer.read_all()
        session_records = [r for r in all_records if r.get("session_id") == session_id]
        if not session_records:
            raise SessionNotFoundError(session_id)

        # Collect compacted IDs
        compacted_ids: set[str] = set()
        for r in session_records:
            if r.get("type") == "system_message" and r.get("is_compacted"):
                compacted_ids.update(r.get("replaces_ids", []))

        active = [r for r in session_records if r.get("id") not in compacted_ids]

        messages: list[dict[str, Any]] = []
        pending_tool_calls: dict[str, dict] = {}

        for r in active:
            rtype = r.get("type")
            if rtype == "system_message":
                messages.append({"role": "system", "content": r.get("content", "")})
            elif rtype == "user_message":
                messages.append({"role": "user", "content": r.get("content", "")})
            elif rtype == "assistant_message":
                tool_calls = r.get("tool_calls") or []
                msg: dict[str, Any] = {"role": "assistant", "content": r.get("content")}
                if tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("arguments", {})),
                            },
                        }
                        for tc in tool_calls
                    ]
                    for tc in tool_calls:
                        pending_tool_calls[tc["id"]] = tc
                messages.append(msg)
            elif rtype == "tool_result":
                call_id = r.get("call_id")
                if call_id in pending_tool_calls:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": r.get("content", ""),
                    })
                    del pending_tool_calls[call_id]
                # orphaned tool_result — drop silently

        # Drop orphaned assistant messages (pending_tool_calls non-empty = crash mid-turn)
        if pending_tool_calls:
            orphan_ids = set(pending_tool_calls.keys())
            messages = [
                m for m in messages
                if not (
                    m.get("role") == "assistant"
                    and any(tc["id"] in orphan_ids for tc in m.get("tool_calls", []))
                )
            ]

        return messages

    # ------------------------------------------------------------------
    # Notes / context
    # ------------------------------------------------------------------

    async def update_notes(self, section: str, content: str) -> None:
        """Replace a named MEMORY.md section. Delegates to MarkdownMemory."""
        self._markdown_memory.update_section(section, content)

    async def load_context(self) -> MemoryContext:
        """Load three-tier context for system prompt injection."""
        assert self._db is not None
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
        summary: str,
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
            "SELECT key, value, updated_at FROM facts "
            "WHERE agent_id = ? AND confidence >= 0.7 "
            "AND (expires_at IS NULL OR expires_at > ?) "
            "ORDER BY updated_at DESC",
            (self._agent_id, now),
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
            session_entry = f"- {today}: {session_summary[:120]}"

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

def _row_to_fact(row: aiosqlite.Row) -> Fact:
    tags_raw = row["tags"] if isinstance(row, aiosqlite.Row) else row[5]
    tags = json.loads(tags_raw) if isinstance(tags_raw, str) else (tags_raw or [])
    if isinstance(row, aiosqlite.Row):
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
        )
    # Positional (shouldn't happen with row_factory=aiosqlite.Row)
    return Fact(
        key=row[0], value=row[1], agent_id=row[2], division_id=row[3],
        org_id=row[4], tags=tags, confidence=row[6], source=row[7],
        created_at=row[8], updated_at=row[9], expires_at=row[10],
    )
