# Spec 12: Audit & Observability

**Component:** `src/localharness/audit/`
**Requirements:** v1 (structlog JSONL), v2 (hash-chained audit, GUARDRAILS.md)
**Status:** v1 core + v2 design spec

---

## Purpose

The audit system provides tamper-evident, complete observability for every agent action, tool call, and decision. It answers: what did the agent do, why, when, what was the result, and could it be undone?

In v1, audit logs are structlog-powered JSONL files — one per agent, plus an org-level aggregate. In v2, logs are SHA-256 hash-chained (tamper-evident) with a Rust PyO3 hot-path writer.

The `GUARDRAILS.md` file is a persistent failure memory: when an agent fails the same pattern 3+ times, a guardrail is automatically appended. It is read into agent context at session start, making the harness self-improving.

---

## AuditLogger Class

```python
# src/localharness/audit/logger.py

import structlog
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

@dataclass(frozen=True)
class AuditEvent:
    """Base record written to the JSONL audit log. All fields always present."""
    v: int                          # schema version, currently 1
    record_type: str                # 'tool_call' | 'decision' | 'session' | 'error' | 'guardrail'
    id: str                         # UUID4 prefixed by type: "tool_01abc"
    session_id: str
    agent_id: str
    division_id: str
    org_id: str
    ts: int                         # Unix epoch seconds
    ts_iso: str                     # ISO 8601 for human readability: "2026-05-23T05:30:00Z"
    # v2 only (None in v1):
    prev_hash: str | None           # SHA-256 of previous record (hex string)
    self_hash: str | None           # SHA-256 of this record excluding self_hash field

@dataclass(frozen=True)
class ToolCallRecord(AuditEvent):
    """Audit record for a single tool invocation."""
    record_type: str = "tool_call"
    tool_name: str = ""
    tool_scope: str = ""            # "global" | "division" | "agent" | "mcp"
    arguments: dict[str, Any] = field(default_factory=dict)
    result_summary: str = ""        # first 200 chars of result
    result_length: int = 0          # full result length in bytes
    is_error: bool = False
    error_type: str | None = None   # None on success
    duration_ms: int = 0
    permission_result: str = ""     # "allowed" | "denied"
    permission_rule: str | None = None  # deny pattern that matched, if denied
    # Decision provenance
    risk_annotation: str | None = None    # inline LLM risk self-report (v2 feature)
    reversible: bool | None = None        # None = unknown; True = can undo; False = cannot
    reversibility_note: str | None = None # explanation if reversible=False

@dataclass(frozen=True)
class DecisionRecord(AuditEvent):
    """
    Audit record for orchestrator or agent decisions that don't produce tool calls.
    Used for: routing decisions, escalations, workflow stage transitions,
    agent creation, config changes.
    """
    record_type: str = "decision"
    decision_type: str = ""         # "routing" | "escalation" | "creation" | "config" | "compaction"
    description: str = ""           # human-readable: what was decided
    inputs: dict[str, Any] = field(default_factory=dict)   # relevant inputs to the decision
    outcome: str = ""               # what happened
    # Provenance fields
    identity: str = ""              # who made the decision: agent_id or "orchestrator"
    reasoning_summary: str | None = None  # brief summary of why (from agent's text before tool call)
    lineage: list[str] = field(default_factory=list)  # IDs of prior records this decision depends on
    model: str | None = None        # model version that made this decision
    turn_number: int = 0
    action_number: int = 0

@dataclass(frozen=True)
class SessionRecord(AuditEvent):
    """Audit record for session lifecycle events."""
    record_type: str = "session"
    event: str = ""                 # "start" | "end" | "compaction" | "stuck" | "kill"
    data: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ErrorRecord(AuditEvent):
    """Audit record for harness-level errors (not agent tool errors)."""
    record_type: str = "error"
    error_class: str = ""           # Python exception class name
    error_message: str = ""
    component: str = ""             # which component raised the error
    traceback_hash: str | None = None  # SHA-256 of traceback (not stored — just the hash)
    fatal: bool = False

@dataclass(frozen=True)
class GuardrailRecord(AuditEvent):
    """Audit record written when a guardrail is appended to GUARDRAILS.md."""
    record_type: str = "guardrail"
    pattern_id: str = ""            # hash of the failure pattern
    trigger: str = ""               # the failure condition
    instruction: str = ""           # what the agent should do instead
    reason: str = ""                # why this guardrail was added
    occurrences: int = 0            # how many times the pattern fired before guardrail
    affected_sessions: list[str] = field(default_factory=list)  # session IDs

class AuditLogger:
    """
    Structured JSONL audit logger for a single agent.
    
    Uses structlog for the processor pipeline (timestamp, level, JSON render).
    Writes to per-agent audit file and org-level aggregate.
    
    Thread safety: all writes are synchronous O_APPEND. Async callers use
    asyncio.to_thread() to avoid blocking the event loop.
    
    v1: Plain JSONL (no hash chaining).
    v2: Hash-chained JSONL via Rust PyO3 extension (same public API).
    
    The AuditLogger subscribes to all event types on the bus and writes
    a record for each relevant event. It is a passive observer — it never
    modifies events or agent state.
    """

    def __init__(
        self,
        agent_id: str,
        division_id: str,
        org_id: str,
        base_dir: str,
        enable_hash_chain: bool = False,   # v2: set True to enable hash chaining
    ) -> None:
        self._agent_id = agent_id
        self._division_id = division_id
        self._org_id = org_id
        self._base_dir = Path(base_dir).expanduser()
        self._enable_hash_chain = enable_hash_chain

        self._agent_log_path = self._base_dir / "agents" / agent_id / "audit.jsonl"
        self._org_log_path = self._base_dir / "orgs" / org_id / "audit.jsonl"

        self._log = structlog.get_logger().bind(
            agent_id=agent_id,
            division_id=division_id,
            org_id=org_id,
        )
        self._last_hash: str | None = None     # v2: running hash chain state
        self._record_count: int = 0
        self._guardrail_tracker: "GuardrailTracker | None" = None

    async def open(self) -> None:
        """
        Create log directories and files if they don't exist.
        Configure structlog processor pipeline.
        Initialize GuardrailTracker.
        
        Raises:
            AuditInitError: If log directories cannot be created.
        """
        ...

    async def close(self) -> None:
        """
        Flush any buffered writes.
        Write final stats to structlog (record_count, session summary).
        """
        ...

    async def log_tool_call(
        self,
        session_id: str,
        tool_name: str,
        tool_scope: str,
        arguments: dict[str, Any],
        result: str,
        is_error: bool,
        error_type: str | None,
        duration_ms: int,
        permission_result: str,
        permission_rule: str | None = None,
        reversible: bool | None = None,
        reversibility_note: str | None = None,
        turn_number: int = 0,
        action_number: int = 0,
    ) -> str:
        """
        Write a ToolCallRecord to the agent audit log and org aggregate.
        
        Returns the record ID (used for lineage tracking in subsequent decisions).
        
        The result is truncated to 200 chars for result_summary; full length
        is stored in result_length. This keeps the audit log compact.
        
        Raises:
            AuditWriteError: On file I/O failure.
        """
        ...

    async def log_decision(
        self,
        session_id: str,
        decision_type: str,
        description: str,
        inputs: dict[str, Any],
        outcome: str,
        identity: str,
        reasoning_summary: str | None = None,
        lineage: list[str] | None = None,
        model: str | None = None,
        turn_number: int = 0,
        action_number: int = 0,
    ) -> str:
        """
        Write a DecisionRecord to the audit log.
        
        Used for routing decisions, escalations, workflow transitions.
        
        Returns the record ID.
        """
        ...

    async def log_session(
        self,
        session_id: str,
        event: str,
        data: dict[str, Any] | None = None,
    ) -> str:
        """
        Write a SessionRecord for lifecycle events.
        
        event: "start" | "end" | "compaction" | "stuck" | "kill"
        """
        ...

    async def log_error(
        self,
        session_id: str,
        error_class: str,
        error_message: str,
        component: str,
        traceback: str | None = None,
        fatal: bool = False,
    ) -> str:
        """
        Write an ErrorRecord.
        
        If traceback is provided, stores SHA-256(traceback) as traceback_hash
        but not the traceback itself (keeps log compact; hash enables dedup).
        
        Always writes to org-level aggregate regardless of fatal flag.
        """
        ...

    async def log_guardrail(
        self,
        session_id: str,
        pattern_id: str,
        trigger: str,
        instruction: str,
        reason: str,
        occurrences: int,
        affected_sessions: list[str],
    ) -> str:
        """
        Write a GuardrailRecord to the audit log.
        Called by GuardrailTracker after appending to GUARDRAILS.md.
        """
        ...

    async def _write_record(self, record: AuditEvent) -> None:
        """
        Serialize record to JSON and append to both agent and org log files.
        
        v1: plain JSON line (no hash).
        v2: compute prev_hash = last record's self_hash, compute self_hash,
            write record with both hashes populated.
        
        Uses asyncio.to_thread() to avoid blocking the event loop on file I/O.
        O_APPEND file open — atomic on POSIX for records < PIPE_BUF (4096 bytes).
        
        Raises:
            AuditWriteError: On file I/O failure.
            AuditDiskFullError: On ENOSPC.
        """
        ...
```

---

## JSONL Record Schema

Each line in `audit.jsonl` is a complete JSON object. All fields in the base `AuditEvent` are present on every record. Type-specific fields are also always present (with `null` for absent optional values).

### Complete record examples

#### tool_call record (v1, no hash chain)

```json
{
  "v": 1,
  "record_type": "tool_call",
  "id": "tool_01a2b3c4",
  "session_id": "sess_uuid4",
  "agent_id": "morning-briefing",
  "division_id": "financial",
  "org_id": "default",
  "ts": 1748042407,
  "ts_iso": "2026-05-23T05:33:27Z",
  "prev_hash": null,
  "self_hash": null,
  "tool_name": "exa_search",
  "tool_scope": "agent",
  "arguments": {"query": "SPX May 23 2026 market open", "num_results": 5},
  "result_summary": "SPX opened at 5,412.3. VIX at 18.2. Treasury yields...",
  "result_length": 3847,
  "is_error": false,
  "error_type": null,
  "duration_ms": 1231,
  "permission_result": "allowed",
  "permission_rule": null,
  "risk_annotation": null,
  "reversible": null,
  "reversibility_note": null
}
```

#### tool_call record (v2, hash-chained, write action)

```json
{
  "v": 2,
  "record_type": "tool_call",
  "id": "tool_02d4e5f6",
  "session_id": "sess_uuid4",
  "agent_id": "morning-briefing",
  "division_id": "financial",
  "org_id": "default",
  "ts": 1748042412,
  "ts_iso": "2026-05-23T05:33:32Z",
  "prev_hash": "a3f2b1c9d8e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1",
  "self_hash": "b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3",
  "tool_name": "write_file",
  "tool_scope": "global",
  "arguments": {"path": "/tmp/briefing-2026-05-23.md", "content": "..."},
  "result_summary": "Written 4,821 bytes to /tmp/briefing-2026-05-23.md",
  "result_length": 52,
  "is_error": false,
  "error_type": null,
  "duration_ms": 12,
  "permission_result": "allowed",
  "permission_rule": null,
  "risk_annotation": "Write to /tmp is reversible — file can be deleted. Low risk.",
  "reversible": true,
  "reversibility_note": "File can be deleted with rm /tmp/briefing-2026-05-23.md"
}
```

#### decision record

```json
{
  "v": 1,
  "record_type": "decision",
  "id": "dec_01abc",
  "session_id": "sess_uuid4",
  "agent_id": "orchestrator",
  "division_id": "default",
  "org_id": "default",
  "ts": 1748042400,
  "ts_iso": "2026-05-23T05:33:20Z",
  "prev_hash": null,
  "self_hash": null,
  "decision_type": "routing",
  "description": "Routed task to morning-briefing agent",
  "inputs": {
    "task": "Generate today's market briefing",
    "candidates": ["morning-briefing", "portfolio"],
    "scores": {"morning-briefing": 0.72, "portfolio": 0.21}
  },
  "outcome": "Delegated to morning-briefing (score=0.72)",
  "identity": "orchestrator",
  "reasoning_summary": null,
  "lineage": [],
  "model": null,
  "turn_number": 0,
  "action_number": 0
}
```

#### session record

```json
{
  "v": 1,
  "record_type": "session",
  "id": "sess_evt_01",
  "session_id": "sess_uuid4",
  "agent_id": "morning-briefing",
  "division_id": "financial",
  "org_id": "default",
  "ts": 1748042400,
  "ts_iso": "2026-05-23T05:33:20Z",
  "prev_hash": null,
  "self_hash": null,
  "event": "start",
  "data": {
    "budget": {"max_actions": 100, "max_duration_minutes": 30},
    "model": "qwen3.5-122b-a10b",
    "context_tokens_available": 131072
  }
}
```

#### error record

```json
{
  "v": 1,
  "record_type": "error",
  "id": "err_01abc",
  "session_id": "sess_uuid4",
  "agent_id": "morning-briefing",
  "division_id": "financial",
  "org_id": "default",
  "ts": 1748042500,
  "ts_iso": "2026-05-23T05:35:00Z",
  "prev_hash": null,
  "self_hash": null,
  "error_class": "MemoryWriteError",
  "error_message": "Failed to write to memory.db: disk full",
  "component": "memory.sqlite",
  "traceback_hash": "c9d8e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1",
  "fatal": true
}
```

---

## Decision Provenance

Every audit record carries enough information to reconstruct why a decision was made. Six provenance dimensions:

| Dimension | Fields | Purpose |
|---|---|---|
| **Identity** | `agent_id`, `identity` | Who made this decision |
| **Temporal** | `ts`, `ts_iso`, `turn_number`, `action_number` | When, in what context |
| **Reasoning** | `reasoning_summary`, `description` | Why (agent's stated rationale) |
| **Tools** | `tool_name`, `arguments`, `result_summary` | What was used |
| **Lineage** | `lineage: list[str]` | Which prior records this depended on |
| **Reversibility** | `reversible`, `reversibility_note` | Can this be undone |

Provenance enables:
- Post-hoc audit: "why did the agent write to that file?"
- Lineage tracing: "what sequence of actions led to this outcome?"
- Recovery planning: "which actions cannot be undone?"
- GUARDRAILS.md generation: "what pattern keeps failing?"

---

## structlog Configuration

```python
# src/localharness/audit/config.py

import structlog
import logging
from pathlib import Path

def configure_structlog(
    agent_id: str,
    log_path: Path,
    debug: bool = False,
) -> None:
    """
    Configure the structlog processor pipeline.
    
    Pipeline (applied in order):
      1. add_log_level — adds "level" field
      2. add_timestamp — adds "ts" and "ts_iso" fields (overrides structlog default)
      3. add_agent_context — adds agent_id, division_id, org_id (from bound logger)
      4. JSONRenderer — serializes to compact JSON (no indentation, no separators spaces)
    
    Output: FileHandler opened in append mode ("a") at log_path.
    Backup: also log ERROR+ to stderr (structlog stdlib integration).
    
    This function is idempotent — safe to call multiple times.
    """
    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _add_timestamps,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]

    if debug:
        shared_processors.append(structlog.dev.ConsoleRenderer())
    else:
        shared_processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # File handler for JSONL output
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    # stderr handler for errors
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(logging.ERROR)

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stderr_handler)
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

def _add_timestamps(logger, method_name, event_dict):
    """structlog processor: add ts (Unix int) and ts_iso (ISO 8601 string)."""
    import time
    from datetime import datetime, timezone
    now = int(time.time())
    event_dict["ts"] = now
    event_dict["ts_iso"] = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return event_dict
```

---

## File Layout

```
~/.localharness/
  agents/{agent_id}/
    audit.jsonl              # per-agent audit log (every record type)
    
  orgs/{org_id}/
    audit.jsonl              # org-level aggregate (copy of all records from all agents)
    audit.YYYY-MM.jsonl      # rotated monthly archives
    GUARDRAILS.md            # persistent failure memory (auto-appended)
```

The org-level `audit.jsonl` contains every record from every agent in the org. It is written to simultaneously with the agent log — both receive the same record. This enables org-wide search without reading N agent files.

---

## v2: SHA-256 Hash Chain Spec

### How Hash Chaining Works

Each record includes `prev_hash` and `self_hash`. This creates a chain: modifying or deleting any record breaks the chain, making tampering detectable.

```
Record 1: prev_hash=NULL,         self_hash=H(record_1_without_self_hash)
Record 2: prev_hash=H(record_1),  self_hash=H(record_2_without_self_hash)
Record 3: prev_hash=H(record_2),  self_hash=H(record_3_without_self_hash)
...
```

### Hash Computation

```python
def compute_record_hash(record: dict) -> str:
    """
    Compute SHA-256 of a record for hash chaining.
    
    Process:
      1. Create copy of record with self_hash field set to "" (empty string).
      2. Serialize to JSON with sorted keys, no indentation, no extra whitespace.
         json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
      3. Encode to UTF-8 bytes.
      4. Compute SHA-256.
      5. Return lowercase hex string (64 chars).
    
    The sort_keys=True ensures deterministic serialization regardless of dict
    insertion order in Python 3.7+. ensure_ascii=True ensures byte-for-byte
    reproducibility across platforms.
    """
    import json
    import hashlib
    record_copy = {**record, "self_hash": ""}
    canonical = json.dumps(record_copy, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

### Chain Verification

```python
async def verify_chain(log_path: Path) -> list[str]:
    """
    Verify the hash chain of an audit log file.
    
    Returns list of error strings. Empty list = chain intact.
    
    Algorithm:
      1. Read all records from log_path.
      2. For each record (except the first):
         a. Compute expected_self_hash = compute_record_hash(record)
         b. Assert record["self_hash"] == expected_self_hash (hash not modified)
         c. Assert record["prev_hash"] == previous_record["self_hash"] (chain not broken)
      3. Report each failed assertion with: record ID, record number, what was expected.
    
    The first record must have prev_hash == null (genesis record).
    Any record with schema version v1 (no hash fields) is skipped — the chain
    starts from the first v2 record encountered.
    """
    ...
```

### Rust Extension (v2)

When hash chaining is enabled (`enable_hash_chain=True`), the Python `_write_record` method delegates to a Rust extension:

```rust
// src/rust/audit-writer/src/lib.rs

use pyo3::prelude::*;
use sha2::{Sha256, Digest};
use std::fs::OpenOptions;
use std::io::Write;

#[pyfunction]
fn append_hashed_record(
    path: &str,
    record_json: &str,  // record as JSON string with self_hash=""
    prev_hash: Option<&str>,
) -> PyResult<String> {  // returns self_hash
    // 1. Substitute prev_hash into the record
    // 2. Compute self_hash = SHA256(record_with_prev_hash_and_empty_self_hash)
    // 3. Substitute self_hash into the record
    // 4. Append to file with O_APPEND (atomic on POSIX < PIPE_BUF)
    // 5. Return self_hash for next record's prev_hash
    ...
}
```

The Python `_write_record` passes the serialized record JSON and the current `self._last_hash`. The Rust function computes both hashes and performs the O_APPEND write in a single operation, then returns the new `self_hash` for the next call. This is the bottleneck operation — the Rust implementation is ~100x faster than Python for high-frequency tool events.

---

## GUARDRAILS.md

### Purpose

GUARDRAILS.md is the org-level persistent failure memory. When an agent repeatedly fails the same pattern, the harness automatically appends a guardrail. The guardrail is read into every agent's system prompt at session start, preventing the same failures from recurring.

### Format

```markdown
# GUARDRAILS

This file records patterns that have caused repeated agent failures.
Each guardrail is added automatically after 3+ occurrences of the same pattern.
Read this file before acting. Obey all instructions below without question.

---

## G-001: Do not use bash to install packages
**Trigger:** bash_execute with arguments matching `pip install|npm install|apt install`
**Instruction:** Never install packages via bash during a session. Packages must be
pre-installed. If a required package is missing, escalate to the user.
**Reason:** Package installation caused unrecoverable environment corruption in 4 sessions.
**Added:** 2026-05-23T05:33:20Z
**Sessions:** sess_01abc, sess_02def, sess_03ghi, sess_04jkl
**Provenance:** guardrail_evt_01xyz

---

## G-002: Verify file paths before write
**Trigger:** write_file immediately after read_file returns an error
**Instruction:** If a file read fails, do not proceed with writing to the same path.
Confirm the path is correct with the user before writing.
**Reason:** Write-on-failed-read caused data loss in 3 sessions.
**Added:** 2026-05-23T08:15:00Z
**Sessions:** sess_05mno, sess_06pqr, sess_07stu
**Provenance:** guardrail_evt_02xyz
```

### GuardrailTracker

```python
# src/localharness/audit/guardrails.py

import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

@dataclass
class FailurePattern:
    """A tracked failure pattern in the rolling window."""
    pattern_id: str              # SHA-256 of normalized pattern description
    description: str             # human-readable pattern
    trigger_condition: str       # what triggers this pattern
    occurrences: list[str]       # session IDs where this pattern occurred
    first_seen: int              # Unix timestamp
    last_seen: int               # Unix timestamp
    guardrail_appended: bool = False  # True once a guardrail has been written

class GuardrailTracker:
    """
    Tracks failure patterns across sessions and appends guardrails when threshold is met.
    
    Failure patterns are identified by normalizing error messages and tool call
    sequences into a canonical form, then hashing. If the same normalized pattern
    appears in 3 or more distinct sessions, a guardrail is appended.
    
    Pattern normalization:
      - Tool name + error type: "write_file:PermissionError"
      - Tool sequence: "read_file → write_file → error"
      - Budget exhaustion on specific tool: "exa_search:budget_exceeded"
    
    The tracker is per-org (shared across all agents in an org).
    State is persisted to ~/.localharness/orgs/{org_id}/guardrail_state.json.
    """

    GUARDRAIL_THRESHOLD = 3   # occurrences before guardrail is appended

    def __init__(
        self,
        org_id: str,
        base_dir: str,
        audit_logger: AuditLogger,
    ) -> None: ...

    async def load(self) -> None:
        """Load persisted state from guardrail_state.json."""
        ...

    async def save(self) -> None:
        """Persist current state to guardrail_state.json."""
        ...

    def normalize_pattern(
        self,
        tool_name: str,
        error_type: str | None,
        arguments: dict[str, Any],
        context: str,
    ) -> str:
        """
        Normalize a failure event into a canonical pattern string.
        
        Normalization rules:
          - Drop argument values (keep only argument keys)
          - Normalize error messages: strip line numbers, file paths, process IDs
          - Keep tool name and error type
          - Append high-level context ("budget_exhausted", "stuck", "permission_denied")
        
        Example:
          tool_name="write_file", error_type="PermissionError", 
          arguments={"path": "/etc/hosts", "content": "..."}, context="permission_denied"
          → "write_file:permission_denied:PermissionError"
        
        Returns a normalized string. Caller should hash it for pattern_id.
        """
        ...

    async def record_failure(
        self,
        session_id: str,
        tool_name: str,
        error_type: str | None,
        arguments: dict[str, Any],
        context: str,
        suggested_instruction: str,
    ) -> bool:
        """
        Record a failure event. If threshold is reached, append guardrail.
        
        Args:
            session_id: Current session ID.
            tool_name: Tool that failed.
            error_type: Exception class name, or None.
            arguments: Tool arguments (values will be normalized away).
            context: High-level failure context.
            suggested_instruction: If a guardrail is appended, this text becomes
                                   the instruction. Caller generates this from the
                                   error context.
        
        Returns:
            True if a guardrail was appended (threshold just reached).
            False if just recorded (below threshold).
        """
        ...

    async def _append_guardrail(
        self,
        pattern: FailurePattern,
        instruction: str,
    ) -> None:
        """
        Append a new guardrail entry to GUARDRAILS.md.
        
        Guardrail numbering: G-{N:03d} where N = current guardrail count + 1.
        
        Write process:
          1. Read current GUARDRAILS.md (create from template if not exists).
          2. Append the new guardrail entry (formatted as per spec).
          3. Write to GUARDRAILS.md.tmp, then os.replace() (atomic).
          4. Call audit_logger.log_guardrail() to record in audit log.
          5. Mark pattern.guardrail_appended = True.
          6. Save state.
        
        Raises:
            AuditWriteError: On file I/O failure.
        """
        ...

    def _generate_guardrail_text(
        self,
        guardrail_id: str,
        trigger: str,
        instruction: str,
        reason: str,
        pattern: FailurePattern,
        provenance_id: str,
    ) -> str:
        """Generate the formatted markdown text for a single guardrail entry."""
        ...
```

---

## Rotation and Retention Policy

### Per-Agent Audit Log

- Rotate when `audit.jsonl` exceeds 50MB (configurable via `audit.max_file_mb`).
- Rotation: rename `audit.jsonl` → `audit.YYYY-MM-DDTHH:MM:SS.jsonl`, create new empty `audit.jsonl`.
- Keep last 10 rotated files (configurable via `audit.max_rotated_files`). Delete oldest on rotation if over limit.
- Rotation is triggered at session start, not mid-session (avoid rotating while writing).

### Org-Level Audit Log

- Rotate monthly: on the first write of each new month, archive `audit.jsonl` → `audit.YYYY-MM.jsonl`.
- Keep all monthly archives (no deletion — org-level audit is the tamper-evident record).
- Monthly archive files are never written to after rotation (immutable after archiving).

### Retention Configuration

```yaml
# In org config, under audit:
audit:
  enabled: true
  max_file_mb: 50                  # per-agent log rotation threshold
  max_rotated_files: 10            # per-agent: keep last N rotated files
  hash_chain: false                # v2: enable SHA-256 hash chaining
  org_aggregate: true              # write all records to org-level log
  result_summary_max_chars: 200    # truncate tool results in audit records
  debug: false                     # if true, structlog ConsoleRenderer to stderr
```

---

## Error Handling

```python
# src/localharness/audit/errors.py

class AuditError(Exception):
    """Base class for audit errors."""

class AuditInitError(AuditError):
    """Failed to create log directories or open log files."""
    def __init__(self, path: str, underlying: Exception) -> None: ...

class AuditWriteError(AuditError):
    """Failed to write audit record to JSONL file."""
    def __init__(self, path: str, underlying: Exception) -> None: ...

class AuditDiskFullError(AuditWriteError):
    """Write failed due to ENOSPC."""
    pass

class AuditCorruptionError(AuditError):
    """
    Detected audit log corruption: invalid JSON on a line, or broken hash chain.
    Not fatal to the agent — the agent continues running.
    The corruption is reported to the user via the terminal channel.
    """
    def __init__(self, path: str, line_number: int, detail: str) -> None: ...

class HashChainBrokenError(AuditCorruptionError):
    """
    Hash chain verification failed. Indicates tampering or file corruption.
    
    Includes: record ID, expected prev_hash, actual prev_hash.
    """
    def __init__(self, record_id: str, expected: str, actual: str) -> None: ...
```

### Write Failure Policy

Audit write failures are **never fatal to the agent**. The agent loop continues running even if the audit log cannot be written. Write failures are:

1. Logged to stderr via the error console.
2. Counted on the `AuditLogger` instance (`_write_failure_count`).
3. After 5 consecutive write failures, the `AuditLogger` disables itself (sets `_enabled = False`) to avoid flooding stderr.
4. On session end, if any write failures occurred, the session record includes `"audit_write_failures": N`.

Rationale: An agent that can't write its audit log is less observable, but stopping the agent is worse than losing audit records. The user can investigate via `localharness doctor`.

### Disk Full

`AuditDiskFullError` is raised when `ENOSPC` is detected. The agent loop catches this and:
1. Attempts to write a final "audit disabled: disk full" record to stderr.
2. Disables the audit logger for the session.
3. Continues execution — audit loss is recoverable; stopping the agent is not.

The distinct `AuditDiskFullError` subclass allows the agent loop to surface a user-visible warning separately from generic write failures.

---

## Implementation Notes

- The `AuditLogger` subscribes to the event bus as a passive observer. It receives the same events as the terminal channel and the memory system but never modifies them.
- structlog bound context (`agent_id`, `division_id`, `org_id`) is set once at `open()` and attached to every log call automatically. No field duplication in call sites.
- The org-level `audit.jsonl` is written to with the same `O_APPEND` pattern as the agent log. Since multiple agents (in v2) may write concurrently, records from different agents interleave in the org log. This is correct — the org log is a merged stream, queried by `agent_id` when needed.
- `compute_record_hash` must be deterministic: `sort_keys=True` and `separators=(',', ':')` are mandatory. Any deviation produces a different hash and breaks chain verification.
- GUARDRAILS.md is written with `os.replace()` (atomic rename on POSIX) to prevent partial-write corruption. The `.tmp` file is in the same directory to ensure rename is atomic (same filesystem).
- The `GuardrailTracker` pattern state (`guardrail_state.json`) uses the same atomic-write pattern. Concurrent access is prevented by the single-threaded asyncio model: only one agent loop runs at a time in v1.
- In v2 (parallel agents), `GuardrailTracker.save()` must use a file lock (`fcntl.flock`) since multiple agent loops may update it concurrently.
