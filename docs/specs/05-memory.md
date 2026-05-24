# Spec 05: Memory System

**Component:** `src/localharness/memory/`
**Requirements:** MEM-01, MEM-02, MEM-03, MEM-04
**Status:** v1

---

## Purpose

The memory system provides persistent, scoped storage for agent state across sessions. It implements a three-tier hierarchy (agent → division → org) where agents read up through all tiers but write only to their own scope. The system has three storage layers per scope:

1. **SQLite facts store** — structured key/value facts and FTS5-queryable entries (`memory.db`)
2. **JSONL chat history** — append-only ordered event log, the source of truth for session reconstruction (`history.jsonl`)
3. **MEMORY.md** — human-readable markdown notes, injected into context at session start

This design is directly derived from the Mem0 multi-scope pattern and the OpenHands V1 event-sourced state model. JSONL is not auxiliary — it IS the session state. Context reconstruction means replaying the JSONL log, not querying a separate state store.

---

## File Layout

```
~/.localharness/
  orgs/{org_id}/
    GUARDRAILS.md          # persistent failure memory (org-wide)
    config.yaml            # org-level defaults
    org.db                 # SQLite: org-scoped facts (v2: FTS5 cross-agent index)

  divisions/{division_id}/
    DIVISION.md            # division narrative notes
    shared.db              # SQLite: division-scoped facts

  agents/{agent_id}/
    MEMORY.md              # agent narrative notes (read into context on every turn)
    memory.db              # SQLite: agent-scoped facts
    history.jsonl          # append-only JSONL: full session event log
    session_state.json     # current session metadata (turn count, budget used, etc.)
```

Path resolution uses `agent_id`, `division_id`, and `org_id` from `AgentConfig`. All paths are configurable per agent via `memory.base_dir` in YAML; the layout above is the default.

---

## SQLite Schema

All three scope databases use the same schema. Apply at DB creation via `PRAGMA user_version` migration guard.

### WAL Mode

Enable immediately on every connection open, before any reads or writes:

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA temp_store = MEMORY;
```

`synchronous = NORMAL` is safe with WAL mode and gives ~3x write throughput over FULL. `temp_store = MEMORY` avoids temp file creation on every query.

### Tables

```sql
-- User version for migration tracking
PRAGMA user_version = 1;

-- Structured key/value facts
CREATE TABLE IF NOT EXISTS facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    TEXT    NOT NULL,
    division_id TEXT    NOT NULL DEFAULT '',
    org_id      TEXT    NOT NULL DEFAULT '',
    key         TEXT    NOT NULL,
    value       TEXT    NOT NULL,
    tags        TEXT    NOT NULL DEFAULT '[]',   -- JSON array of strings
    confidence  REAL    NOT NULL DEFAULT 1.0,    -- 0.0–1.0
    source      TEXT    NOT NULL DEFAULT '',     -- tool name or "user" or "inference"
    created_at  INTEGER NOT NULL,               -- Unix timestamp (seconds)
    updated_at  INTEGER NOT NULL,               -- Unix timestamp (seconds)
    expires_at  INTEGER,                        -- NULL = never expires
    UNIQUE(agent_id, key)
);

CREATE INDEX IF NOT EXISTS idx_facts_agent_id    ON facts(agent_id);
CREATE INDEX IF NOT EXISTS idx_facts_key         ON facts(agent_id, key);
CREATE INDEX IF NOT EXISTS idx_facts_tags        ON facts(tags);         -- JSON index, partial

-- FTS5 full-text search over fact values (v1: per-agent only; v2: cross-agent in org.db)
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    key,
    value,
    tags,
    content     = facts,
    content_rowid = id
);

-- Triggers to keep FTS5 in sync with facts table
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

-- Session metadata
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT    PRIMARY KEY,         -- UUID4
    agent_id        TEXT    NOT NULL,
    division_id     TEXT    NOT NULL DEFAULT '',
    org_id          TEXT    NOT NULL DEFAULT '',
    started_at      INTEGER NOT NULL,
    ended_at        INTEGER,                    -- NULL = still active
    turn_count      INTEGER NOT NULL DEFAULT 0,
    action_count    INTEGER NOT NULL DEFAULT 0,
    tokens_in       INTEGER NOT NULL DEFAULT 0,
    tokens_out      INTEGER NOT NULL DEFAULT 0,
    exit_reason     TEXT,                       -- 'complete' | 'budget' | 'stuck' | 'error' | 'kill'
    summary         TEXT                        -- final summary text from agent
);

CREATE INDEX IF NOT EXISTS idx_sessions_agent_id ON sessions(agent_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started  ON sessions(started_at DESC);

-- Notes: arbitrary free-text blocks for MEMORY.md sync
-- MEMORY.md is the canonical source; this table mirrors its sections for queryability
CREATE TABLE IF NOT EXISTS notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    TEXT    NOT NULL,
    section     TEXT    NOT NULL DEFAULT 'general',
    content     TEXT    NOT NULL,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_notes_agent_id ON notes(agent_id, section);
```

### Migration Protocol

```sql
-- Check current version before applying migrations
-- This pattern runs on every DB open (cheap PRAGMA read)
-- Version 0 → 1: initial schema (above)
-- Version 1 → 2 (future): add expires_at column, cross-agent index table
```

Implementation: `MemoryStore._apply_migrations(conn)` reads `PRAGMA user_version`, compares to `CURRENT_SCHEMA_VERSION = 1`, and applies ordered migration scripts from `memory/migrations/*.sql`. Never modify existing migrations — only append new ones.

---

## JSONL Chat History Format

File: `~/.localharness/agents/{agent_id}/history.jsonl`

One JSON object per line. Each line is a complete, self-describing record. All timestamps are Unix epoch seconds (integer). All fields are always present — use `null` for absent optional values rather than omitting the field. This ensures reliable replay without field-presence checks.

### Message Types

#### user_message
```json
{
  "v": 1,
  "type": "user_message",
  "id": "msg_01abc",
  "session_id": "sess_uuid4",
  "agent_id": "morning-briefing",
  "division_id": "financial",
  "org_id": "default",
  "ts": 1748042400,
  "role": "user",
  "content": "Generate today's market briefing",
  "channel": "terminal",
  "channel_metadata": null
}
```

#### assistant_message
```json
{
  "v": 1,
  "type": "assistant_message",
  "id": "msg_02def",
  "session_id": "sess_uuid4",
  "agent_id": "morning-briefing",
  "division_id": "financial",
  "org_id": "default",
  "ts": 1748042405,
  "role": "assistant",
  "content": "I'll gather market data now.",
  "tool_calls": [
    {
      "id": "call_01",
      "name": "exa_search",
      "arguments": {"query": "market open May 23 2026", "num_results": 5}
    }
  ],
  "finish_reason": "tool_calls",
  "tokens_in": 1024,
  "tokens_out": 87,
  "model": "qwen3.5-122b-a10b",
  "latency_ms": 1230
}
```

`tool_calls` is an empty list `[]` when there are no tool calls, never `null`. `finish_reason` is one of: `"tool_calls"`, `"stop"`, `"length"`, `"content_filter"`.

#### tool_result
```json
{
  "v": 1,
  "type": "tool_result",
  "id": "msg_03ghi",
  "session_id": "sess_uuid4",
  "agent_id": "morning-briefing",
  "division_id": "financial",
  "org_id": "default",
  "ts": 1748042407,
  "role": "tool",
  "call_id": "call_01",
  "tool_name": "exa_search",
  "content": "SPX opened at 5,412.3...",
  "is_error": false,
  "error_type": null,
  "truncated": false,
  "original_length": 842,
  "stored_length": 842
}
```

`is_error: true` records use `error_type` to classify: `"tool_not_found"`, `"validation_error"`, `"execution_error"`, `"timeout"`, `"permission_denied"`. `truncated: true` means `content` was capped by the tool result budget; `original_length` records the pre-cap size.

#### system_message
```json
{
  "v": 1,
  "type": "system_message",
  "id": "msg_00sys",
  "session_id": "sess_uuid4",
  "agent_id": "morning-briefing",
  "division_id": "financial",
  "org_id": "default",
  "ts": 1748042400,
  "role": "system",
  "content": "You are morning-briefing...",
  "is_compacted": false,
  "replaces_ids": []
}
```

`is_compacted: true` means this system message is a summary that replaces the messages listed in `replaces_ids`. This is the SummaryMessageID compaction marker. The original messages are still in the JSONL file (never deleted); only in-memory replay skips them.

#### session_event
```json
{
  "v": 1,
  "type": "session_event",
  "id": "evt_01",
  "session_id": "sess_uuid4",
  "agent_id": "morning-briefing",
  "division_id": "financial",
  "org_id": "default",
  "ts": 1748042400,
  "event": "session_start",
  "data": {
    "budget": {"max_actions": 100, "max_duration_minutes": 30},
    "model": "qwen3.5-122b-a10b",
    "context_tokens_available": 131072
  }
}
```

`event` values: `"session_start"`, `"session_end"`, `"compaction"`, `"budget_warning"`, `"stuck_detected"`, `"kill_signal"`.

---

## MEMORY.md Format

File: `~/.localharness/agents/{agent_id}/MEMORY.md`

Human-readable, git-committable, read in full at session start. Format is strict to enable automated append:

```markdown
# Memory: {agent_name}

Last updated: {ISO 8601 datetime}
Agent ID: {agent_id}
Division: {division_id}

## Identity

{agent_role_description}

## Persistent Facts

- {fact_key}: {fact_value} *(updated {date})*
- {fact_key}: {fact_value} *(updated {date})*

## Working Notes

{free-form narrative notes added by the agent}

## Learned Behaviors

{patterns, preferences, and corrections accumulated over sessions}

## Session History

- {YYYY-MM-DD}: {one-line summary of what was accomplished}
- {YYYY-MM-DD}: {one-line summary}
```

### Auto-Update Rules

MEMORY.md is updated by `MemoryStore.flush_memory_md()` after each session completes. Rules:

1. **Facts section**: regenerated from the `facts` table (agent scope only). All facts with `confidence >= 0.7` are included, sorted by `updated_at DESC`.
2. **Session History**: one line appended per completed session. Line format: `- {date}: {session.summary[:120]}`. Never truncate existing history lines; prepend new entries at the top of the list.
3. **Working Notes and Learned Behaviors**: never overwritten programmatically. Only the agent's LLM may append to these sections via `update_notes()`. The agent receives the current content of these sections as context and returns updated text.
4. **Last updated**: always refreshed to current UTC datetime.
5. **Identity**: written once at agent creation from `agent_config.role`. Never modified after.

The file is the canonical human-readable view. If the file is manually edited by the user, the next `flush_memory_md()` call preserves the edited Working Notes and Learned Behaviors sections verbatim.

### Read-Into-Context Protocol

At session start, `MemoryStore.load_context()` returns a `MemoryContext` object with the full contents of:

1. Agent's `MEMORY.md` (always)
2. Division's `DIVISION.md` (always, if it exists)
3. Org's `GUARDRAILS.md` (always, if it exists and non-empty)

These are injected into the system prompt in that order, separated by `---` markers. The agent loop injects them before the user's task message, not as part of the conversation history. This means they consume system prompt tokens, not message tokens.

---

## Public Interfaces

### MemoryStore

```python
# src/localharness/memory/sqlite.py

import aiosqlite
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class Fact:
    key: str
    value: str
    agent_id: str
    division_id: str
    org_id: str
    tags: list[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = ""
    created_at: int = 0      # Unix seconds
    updated_at: int = 0      # Unix seconds
    expires_at: int | None = None

@dataclass(frozen=True)
class FactQuery:
    """Parameters for querying facts."""
    text: str | None = None          # FTS5 search query
    tags: list[str] = field(default_factory=list)
    min_confidence: float = 0.0
    include_scopes: list[str] = field(default_factory=lambda: ["agent"])
    # include_scopes: ["agent"], ["agent", "division"], ["agent", "division", "org"]
    limit: int = 50

@dataclass(frozen=True)
class MemoryContext:
    """Loaded context for injection into system prompt."""
    agent_memory_md: str          # full MEMORY.md content, empty string if not exists
    division_md: str              # full DIVISION.md content, empty string if not exists
    guardrails_md: str            # full GUARDRAILS.md content, empty string if not exists
    fact_count: int               # number of facts loaded
    token_estimate: int           # rough token count (len(joined) // 4)

class MemoryStore:
    """
    Three-tier persistent memory for a single agent.
    
    Owns the agent's memory.db, history.jsonl, and MEMORY.md.
    Reads (but never writes) division and org memory for context injection.
    
    Thread safety: aiosqlite connections are not thread-safe. One MemoryStore
    instance per agent loop coroutine. Do not share across asyncio tasks.
    """

    def __init__(
        self,
        agent_id: str,
        division_id: str,
        org_id: str,
        base_dir: str,           # e.g. "~/.localharness"
    ) -> None: ...

    async def open(self) -> None:
        """
        Open SQLite connection, enable WAL mode, apply pending migrations.
        Must be called before any other method.
        Raises MemoryCorruptionError if integrity check fails.
        """
        ...

    async def close(self) -> None:
        """
        Flush pending writes, close SQLite connection.
        Safe to call multiple times.
        """
        ...

    async def store_fact(
        self,
        key: str,
        value: str,
        tags: list[str] | None = None,
        confidence: float = 1.0,
        source: str = "",
        expires_at: int | None = None,
    ) -> Fact:
        """
        Upsert a fact into the agent's fact store.
        
        If a fact with the same key already exists, updates value, confidence,
        source, updated_at, and tags. Does not reset created_at.
        
        Args:
            key: Fact identifier. Use dot notation for namespacing: "portfolio.last_rebalance".
            value: Fact value. Always stored as text; caller serializes complex values to JSON.
            tags: Optional list of string labels for filtering.
            confidence: 0.0–1.0. Facts below 0.5 are excluded from MEMORY.md output.
            source: Origin of the fact: tool name, "user", "inference", "system".
            expires_at: Unix timestamp after which the fact is excluded from queries. None = never.
        
        Returns:
            The stored Fact with populated id and timestamps.
        
        Raises:
            MemoryWriteError: On SQLite write failure.
            ValueError: If confidence not in [0.0, 1.0].
        """
        ...

    async def query_facts(
        self,
        query: FactQuery,
    ) -> list[Fact]:
        """
        Query facts across one or more scopes.
        
        Scope order: agent facts first, then division, then org.
        Within each scope, results are ordered by relevance (FTS5 rank) when
        text is provided, otherwise by updated_at DESC.
        
        Expired facts (expires_at <= now()) are never returned.
        
        Args:
            query: FactQuery specifying search criteria.
        
        Returns:
            List of matching Facts, deduplicated by key (agent scope wins over
            division wins over org for same key).
        
        Raises:
            MemoryReadError: On SQLite read failure.
        """
        ...

    async def get_fact(self, key: str) -> Fact | None:
        """
        Get a single fact by exact key from agent scope.
        Returns None if not found or expired.
        """
        ...

    async def delete_fact(self, key: str) -> bool:
        """
        Delete a fact by key from agent scope.
        Returns True if a row was deleted, False if key not found.
        """
        ...

    async def get_history(
        self,
        session_id: str | None = None,
        limit: int = 200,
        message_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Load message history from JSONL, optionally filtered by session_id.
        
        Reads history.jsonl sequentially (it is ordered by append time).
        When session_id is provided, returns only records with that session_id.
        When session_id is None, returns the last `limit` records across all sessions.
        
        This is the primary method for context reconstruction (session replay).
        Compaction markers (system_message with is_compacted=True) are honored:
        messages listed in replaces_ids are excluded from the returned list, and
        the compacted summary message is included in their place.
        
        Args:
            session_id: Filter to a specific session. None = recent history.
            limit: Maximum number of records to return.
            message_types: Filter by type. None = all types returned.
        
        Returns:
            List of message dicts in chronological order (oldest first).
            Each dict matches the JSONL record schema exactly.
        
        Raises:
            MemoryReadError: On file I/O failure.
            MemoryCorruptionError: If any JSONL line fails JSON parsing.
        """
        ...

    async def reconstruct_session(
        self,
        session_id: str,
    ) -> list[dict[str, Any]]:
        """
        Reconstruct the full message list for an LLM request from a session.
        
        Returns messages in the format expected by the LLM provider:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
        
        Handles:
        - Compaction markers: replaces sequences with summary messages
        - tool_calls reconstruction: assembles OpenAI-format tool_calls arrays
        - tool_result attachment: pairs each tool_result with its tool_call
        - Orphan guard: removes any tool_result without a preceding tool_call
        
        Raises:
            MemoryReadError: On I/O failure.
            SessionNotFoundError: If session_id has no records in history.jsonl.
        """
        ...

    async def update_notes(
        self,
        section: str,
        content: str,
    ) -> None:
        """
        Replace the content of a named section in MEMORY.md.
        
        Valid sections: "working_notes", "learned_behaviors".
        The "identity", "persistent_facts", and "session_history" sections
        are managed automatically and cannot be updated via this method.
        
        Args:
            section: Section name (snake_case, matches MEMORY.md heading).
            content: Full replacement text for the section. Overwrites entirely.
        
        Raises:
            ValueError: If section is not "working_notes" or "learned_behaviors".
            MemoryWriteError: On file I/O failure.
        """
        ...

    async def load_context(self) -> MemoryContext:
        """
        Load all memory context for system prompt injection.
        
        Reads MEMORY.md (agent), DIVISION.md (division), GUARDRAILS.md (org).
        Missing files return empty strings — not an error.
        
        Returns a MemoryContext with pre-formatted text ready for system prompt injection.
        """
        ...

    async def append_history(
        self,
        record: dict[str, Any],
    ) -> None:
        """
        Append a single record to history.jsonl.
        
        The record must match one of the defined JSONL schemas (user_message,
        assistant_message, tool_result, system_message, session_event).
        
        Validates the record has required fields: v, type, id, session_id,
        agent_id, ts. Raises ValueError for missing fields.
        
        O_APPEND write — atomic on POSIX for records under 4KB (PIPE_BUF safe).
        
        Raises:
            MemoryWriteError: On file I/O failure.
            ValueError: If record is missing required fields or type is unknown.
        """
        ...

    async def flush_memory_md(
        self,
        session_summary: str | None = None,
    ) -> None:
        """
        Regenerate MEMORY.md from current fact store state.
        
        Preserves "working_notes" and "learned_behaviors" sections verbatim.
        Regenerates "persistent_facts" from SQLite.
        Appends session_summary to "session_history" if provided.
        
        Idempotent: safe to call multiple times. Does not acquire exclusive lock
        (race condition on concurrent flush_memory_md calls is acceptable — the
        last writer wins, which is fine since only one agent loop runs at a time).
        
        Raises:
            MemoryWriteError: On file I/O failure.
        """
        ...

    async def create_session(
        self,
        session_id: str,
        budget: dict[str, Any],
        model: str,
        context_tokens_available: int,
    ) -> None:
        """
        Record session start in SQLite sessions table and append session_start
        event to history.jsonl.
        """
        ...

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
        """
        Record session end in SQLite sessions table and append session_end
        event to history.jsonl. Calls flush_memory_md(summary).
        """
        ...

    async def integrity_check(self) -> list[str]:
        """
        Run SQLite PRAGMA integrity_check and PRAGMA foreign_key_check.
        Validate history.jsonl is parseable (scan all lines).
        
        Returns list of error strings. Empty list = healthy.
        Does not raise — returns errors for caller to handle.
        """
        ...
```

### HistoryWriter

```python
# src/localharness/memory/history.py

import asyncio
from pathlib import Path
from typing import Any

class HistoryWriter:
    """
    Low-level append-only JSONL writer for chat history.
    
    Used internally by MemoryStore. Callers should use MemoryStore.append_history().
    Exposed separately for the audit logger to reuse the same append pattern.
    
    Uses asyncio file I/O with O_APPEND flag. On POSIX, O_APPEND writes are
    atomic up to PIPE_BUF (4096 bytes) — sufficient for all message records.
    Records exceeding 4096 bytes are still written correctly but not atomically
    (a concurrent writer could interleave). Agent loops are single-coroutine,
    so this is not a practical concern for v1.
    """

    def __init__(self, path: Path) -> None: ...

    async def append(self, record: dict[str, Any]) -> None:
        """
        Serialize record to JSON and append to file with newline.
        Creates file if it does not exist.
        
        Raises:
            MemoryWriteError: On file I/O failure.
        """
        ...

    async def read_all(self) -> list[dict[str, Any]]:
        """
        Read and parse all records from the file.
        Returns empty list if file does not exist.
        
        Raises:
            MemoryCorruptionError: If any line fails JSON parsing.
                Includes line number and raw line content in error message.
        """
        ...

    async def read_last_n(self, n: int) -> list[dict[str, Any]]:
        """
        Efficiently read the last n records without loading entire file.
        Uses seek-from-end strategy for files larger than 1MB.
        Falls back to read_all() + tail for smaller files.
        """
        ...
```

### MarkdownMemory

```python
# src/localharness/memory/markdown.py

from pathlib import Path

VALID_WRITABLE_SECTIONS = frozenset({"working_notes", "learned_behaviors"})

class MarkdownMemory:
    """
    MEMORY.md and DIVISION.md file manager.
    
    Parses the markdown file into named sections for selective update.
    Section boundaries are defined by ## headings.
    """

    def __init__(self, path: Path) -> None: ...

    def exists(self) -> bool: ...

    def read(self) -> str:
        """Read full file content. Returns empty string if file does not exist."""
        ...

    def get_section(self, section_slug: str) -> str:
        """
        Extract content of a named section by its slug (heading text in snake_case).
        Returns empty string if section not found.
        
        Example: section_slug="working_notes" matches "## Working Notes"
        """
        ...

    def update_section(self, section_slug: str, content: str) -> None:
        """
        Replace content of a named section in place.
        
        Raises:
            ValueError: If section_slug not in VALID_WRITABLE_SECTIONS.
            FileNotFoundError: If the markdown file does not exist.
        """
        ...

    def regenerate(
        self,
        agent_id: str,
        agent_name: str,
        role: str,
        facts_text: str,
        session_entry: str | None,
    ) -> None:
        """
        Rewrite the file, preserving working_notes and learned_behaviors verbatim.
        Creates the file if it does not exist.
        
        Args:
            facts_text: Pre-formatted text for the Persistent Facts section.
            session_entry: One-line entry to prepend to Session History, or None.
        """
        ...
```

---

## Scope Resolution

### Read Scope

When an agent calls `query_facts(FactQuery(include_scopes=["agent", "division", "org"]))`:

1. Query agent's `memory.db` — returns agent-scoped facts.
2. Open division's `shared.db` (read-only) — returns division-scoped facts.
3. Open org's `org.db` (read-only) — returns org-scoped facts.
4. Merge results, deduplicating by key: agent scope wins over division, division wins over org.

The MemoryStore holds the path to its own DB and resolves sibling paths from `base_dir`, `division_id`, and `org_id`. It opens division and org DBs with `aiosqlite.connect(..., uri=True)` using `?mode=ro` URI parameter — read-only mode prevents accidental writes even if a code path tries to write.

### Write Scope

Agents write only to their own `memory.db`. Division and org DBs are opened read-only. An agent that attempts to write to a parent scope (which should never happen through the public API) receives `PermissionError`. The MemoryStore API provides no method to write to division or org scope.

Division-level shared facts are written by the orchestrator only, via a separate `DivisionMemoryStore` that wraps the shared division DB with write access. `DivisionMemoryStore` exposes the same `store_fact` / `query_facts` interface but targets `shared.db` instead of `memory.db`.

---

## Session Reconstruction

Reconstruction from JSONL is the recovery path used when an agent loop restarts after a crash, or when the orchestrator needs to review a past session.

### Algorithm

```python
async def reconstruct_session(session_id: str) -> list[dict]:
    records = await history_writer.read_all()
    session_records = [r for r in records if r["session_id"] == session_id]

    # Identify compaction ranges: system_messages with is_compacted=True
    compacted_ids: set[str] = set()
    for r in session_records:
        if r["type"] == "system_message" and r.get("is_compacted"):
            compacted_ids.update(r.get("replaces_ids", []))

    # Exclude compacted messages; keep summary system_messages
    active = [r for r in session_records if r["id"] not in compacted_ids]

    # Convert to LLM message format
    messages = []
    pending_tool_calls: dict[str, dict] = {}  # call_id -> tool_call dict

    for r in active:
        if r["type"] == "system_message":
            messages.append({"role": "system", "content": r["content"]})
        elif r["type"] == "user_message":
            messages.append({"role": "user", "content": r["content"]})
        elif r["type"] == "assistant_message":
            msg = {"role": "assistant", "content": r["content"]}
            if r["tool_calls"]:
                msg["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                    for tc in r["tool_calls"]
                ]
                for tc in r["tool_calls"]:
                    pending_tool_calls[tc["id"]] = tc
            messages.append(msg)
        elif r["type"] == "tool_result":
            call_id = r["call_id"]
            if call_id in pending_tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": r["content"],
                })
                del pending_tool_calls[call_id]
            # Orphaned tool_results (no matching tool_call) are silently dropped
            # This is the boundary guard during reconstruction

    # Drop trailing orphaned tool_calls from pending_tool_calls
    # (crash occurred mid-turn before results arrived)
    if pending_tool_calls:
        # Remove the assistant message that contains these orphaned calls
        messages = [
            m for m in messages
            if not (m.get("role") == "assistant" and
                    any(tc["id"] in pending_tool_calls for tc in m.get("tool_calls", [])))
        ]

    return messages
```

### Crash Recovery

If a crash occurs mid-turn (after tool calls dispatched but before results received), `pending_tool_calls` will be non-empty at reconstruction time. The algorithm removes the orphaned assistant message containing those tool calls. The reconstructed session ends at the last clean turn, allowing the agent loop to re-issue the last message safely.

---

## Error Handling

### Error Types

```python
# src/localharness/memory/errors.py

class MemoryError(Exception):
    """Base class for all memory errors."""
    pass

class MemoryWriteError(MemoryError):
    """
    SQLite write or file I/O write failure.
    Attributes: path (str), underlying (Exception)
    """
    def __init__(self, path: str, underlying: Exception) -> None: ...

class MemoryReadError(MemoryError):
    """
    SQLite read or file I/O read failure.
    """
    def __init__(self, path: str, underlying: Exception) -> None: ...

class MemoryCorruptionError(MemoryError):
    """
    Detected corruption: PRAGMA integrity_check failed, or JSONL line
    fails JSON parsing.
    Attributes: path (str), detail (str)
    """
    def __init__(self, path: str, detail: str) -> None: ...

class SessionNotFoundError(MemoryError):
    """
    session_id has no records in history.jsonl.
    """
    def __init__(self, session_id: str) -> None: ...

class DiskFullError(MemoryWriteError):
    """
    Write failed due to ENOSPC. Subclass of MemoryWriteError.
    The agent loop catches this specifically to trigger emergency shutdown
    and log the event to stderr (disk-full means JSONL write also failed).
    """
    pass
```

### Concurrent Access (WAL Mode)

SQLite WAL mode allows multiple simultaneous readers with one writer. The agent loop is the sole writer to `memory.db`. The orchestrator may read concurrently for indexing. With WAL mode:

- Multiple readers never block each other.
- One writer at a time; writer does not block readers.
- Readers see a consistent snapshot from when their transaction began.

`aiosqlite` uses `asyncio.to_thread` to run SQLite operations on a thread pool, avoiding blocking the event loop. The MemoryStore does not implement additional locking; SQLite's WAL handles isolation.

### Disk Full

When `append_history` or any write method raises an `OSError` with `errno.ENOSPC`, the MemoryStore raises `DiskFullError`. The agent loop catches `DiskFullError`, logs to stderr (since JSONL writes also fail), invokes the kill switch, and terminates the session. The orchestrator is notified via an `Escalation` event published before the kill.

### Corruption Detection

On `open()`, `MemoryStore` runs:
```sql
PRAGMA integrity_check;
PRAGMA foreign_key_check;
```

If either returns anything other than `"ok"`, `open()` raises `MemoryCorruptionError`. The agent loop catches this, logs the error, and refuses to start the session (safe: won't make corrupt state worse). A CLI message directs the user to `localharness doctor` for recovery options.

Recovery options (exposed by `localharness doctor`):
1. Delete `memory.db` and reconstruct facts from MEMORY.md (lossy — facts in DB but not in MEMORY.md are lost).
2. Copy a WAL checkpoint backup (SQLite WAL files can be used for point-in-time recovery if the `-wal` and `-shm` files are intact).

---

## Configuration

```yaml
# In agent YAML config, under the memory: key
memory:
  base_dir: "~/.localharness"     # root for all memory files
  max_facts: 10000                # hard cap on fact count per agent scope
  fact_expiry_days: null          # null = facts never auto-expire
  history_retention_sessions: 50  # keep last N session IDs in history.jsonl
                                  # older sessions are archived to history.YYYY-MM.jsonl
  max_history_file_mb: 100        # rotate history.jsonl when it exceeds this size
  memory_md_max_tokens: 4096      # truncate MEMORY.md injection if over this estimate
  include_division_context: true  # read DIVISION.md into system prompt
  include_org_context: true       # read GUARDRAILS.md into system prompt
```

All memory paths computed from `base_dir`:
- Agent DB: `{base_dir}/agents/{agent_id}/memory.db`
- Agent history: `{base_dir}/agents/{agent_id}/history.jsonl`
- Agent MEMORY.md: `{base_dir}/agents/{agent_id}/MEMORY.md`
- Division DB: `{base_dir}/divisions/{division_id}/shared.db`
- Division MD: `{base_dir}/divisions/{division_id}/DIVISION.md`
- Org DB: `{base_dir}/orgs/{org_id}/org.db`
- Org GUARDRAILS: `{base_dir}/orgs/{org_id}/GUARDRAILS.md`

---

## Implementation Notes

- Use `aiosqlite` exclusively — never `sqlite3` directly, as synchronous DB calls block the asyncio event loop for 10-50ms per query.
- Use parameterized queries everywhere. No string interpolation in SQL. `aiosqlite` supports `:named` parameters.
- The `tags` column stores a JSON array as text. Index lookups use `json_each()`: `WHERE EXISTS (SELECT 1 FROM json_each(tags) WHERE value = ?)`. This is SQLite's built-in JSON function, available in Python's bundled sqlite3 since Python 3.9.
- JSONL files are never rewritten — only appended. Compaction is a logical operation (tracking `replaces_ids`) not a physical one. Physical cleanup (archiving old sessions) happens via a maintenance task on session close when `history_retention_sessions` is exceeded.
- The `MemoryStore.open()` call creates the directory tree (`base_dir/agents/{agent_id}/`) if it does not exist, using `Path.mkdir(parents=True, exist_ok=True)`.
- MEMORY.md regeneration uses `MEMORY.md.tmp` write + atomic rename (`os.replace`) to prevent partial-write corruption.
