"""The memory plugin's settings — `agent.memory.*` (its AgentConfigModel). Pydantic only: memory/plugin.py imports this at module level."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class MemoryConsolidationConfig(BaseModel):
    """Idle-time dreaming (the memory spec's consolidation pass).

    An in-harness feature (default-on, config-off; NO cron job, no daemon assumption):
    triggered by a session-start staleness check + an in-session idle timer, and
    cooperatively cancelled the instant a user turn arrives. The pass replays the NEW
    event-stream windows since the last digest, distributes each moment's attention
    over stored facts by resonance in the model's own space, binds co-fired facts
    into named groups, folds read counters, and settles the writer bet ledger. The
    amount digested is the pass's only clock — no decay constants, no caps, no tiers.
    All fields auto-enumerate as `agent.memory.consolidation.*` registry axes."""
    model_config = ConfigDict(frozen=False, extra="forbid")

    enabled: bool = Field(
        default=True,
        description=(
            "If True (default), the harness dreams during idle: embeds un-embedded "
            "facts, digests new event-stream windows into resonance standing, binds "
            "and names co-fired groups, folds staged read-counters, and settles "
            "writer tallies. Set False to disable all background memory work."
        ),
    )
    idle_minutes: float = Field(
        default=10.0, ge=0.5, le=1440.0,
        description="In-session idle trigger: minutes with no user activity before a pass may fire.",
    )
    staleness_hours: float = Field(
        default=6.0, ge=0.1, le=720.0,
        description="Session-start trigger: run a pass at startup when the last one is older than this.",
    )
    iteration_cap: int = Field(
        default=200, ge=1, le=10_000,
        description=(
            "Hard per-pass work cap (infinite-loop-class guardrail): max stream windows "
            "digested in one pass. The un-digested tail is simply the next pass's stream — "
            "a throughput bound, never a correctness one."
        ),
    )


class MemoryArchivalConfig(BaseModel):
    """Dormancy archival — the forgetting half of consolidation (memory rung 1).

    A fact that scores below the store's own proven-useful floor moves out of the hot
    `facts` table into the cold `facts_archive` mirror: off every recall path, out of the
    FTS haystack and the memory page, restorable byte-identical at any time. Nothing is
    ever deleted.

    ONE axis only, and it is a ROLLOUT GATE rather than a tuning knob — the line the step
    archives against is COMPUTED from the store every pass (the minimum salience over the
    facts that were actually recalled or touched by the owner), never configured."""
    model_config = ConfigDict(frozen=False, extra="forbid")

    enabled: bool = Field(
        default=False,
        description=(
            "OFF by default, deliberately: archival is the first mechanism in this project "
            "that REMOVES rows from the hot store, and the owner watches its first real run "
            "before it is ever allowed to fire on its own. Leave it False and the "
            "consolidation step no-ops with zero side effects — `localharness memory "
            "archive --dry-run` still reports exactly what it WOULD move (it moves nothing), "
            "and `localharness memory archive` still performs an explicit, owner-triggered "
            "run. Set True only once a dry-run has been read on the store in question; then "
            "every idle consolidation pass archives dormant facts automatically and reports "
            "the count and the line it used. Mutable via `localharness components set "
            "agent.memory.archival.enabled <true|false>`."
        ),
    )


class MemoryConfig(BaseModel):
    """Memory backend configuration for an agent."""
    model_config = ConfigDict(frozen=False, extra="forbid")

    # Paths here are declarative only — MemoryStore derives its real paths from base_dir
    # (memory/sqlite.py:712-740). Left unset they stay None; nothing in src/ reads them.
    sqlite_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to the SQLite facts store for this agent. "
            "Unset: MemoryStore derives it from its base_dir (<config dir>/agents/{name}/memory.db)."
        ),
    )

    history_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to the JSONL chat history file. "
            "Unset: MemoryStore derives it from its base_dir (<config dir>/agents/{name}/history.jsonl)."
        ),
    )

    notes_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to the MEMORY.md persistent notes file. "
            "Unset: MemoryStore derives it from its base_dir (<config dir>/agents/{name}/MEMORY.md)."
        ),
    )

    max_notes_chars: int = Field(
        default=16_000,
        ge=0,
        le=200_000,
        description=(
            "Maximum characters of MEMORY.md to inject into context on each turn. "
            "Notes are injected from the top (most recent entries are at the bottom — "
            "the whole file is included until this limit is reached)."
        ),
    )

    shared_read: list[Literal["division", "org"]] = Field(
        default_factory=list,
        description=(
            "Which memory scopes this agent can read from in addition to its own. "
            "'division': can read the division's shared.db and DIVISION.md. "
            "'org': can read the org-level GUARDRAILS.md (v2)."
        ),
    )

    recall_scope: Literal["workspace", "global", "both"] = Field(
        default="workspace",
        description=(
            "Which physical memory STORE recall reads from when a workspace layer applies: "
            "'workspace' (default) reads only this project's own store; 'global' reads only the "
            "machine-global store; 'both' merges them, this project's store first, with an origin "
            "token on every injected line. READS ONLY — remember, the write gate and consolidation "
            "always write to this session's own store whatever this says; `/memory promote <id>` is "
            "the one way a memory crosses into the global store. Distinct from shared_read, which "
            "is the ORG-HIERARCHY axis (division/org context files), not the workspace/global "
            "physical-store axis. Inert with no workspace layer (LAYR-03): exactly one store exists "
            "and all three values behave identically. `localharness components set "
            "agent.memory.recall_scope <workspace|global|both>` sets this MACHINE-WIDE, in the "
            "global overrides.yaml — every project on this machine, not just the one you are "
            "standing in. For one project only, put `memory: {recall_scope: ...}` in that "
            "project's `.localharness/agents/<name>.yaml`."
        ),
    )

    inject_into_context: bool = Field(
        default=True,
        description=(
            "If True, inject MEMORY.md contents and recent SQLite facts into the system prompt "
            "at the start of each turn. If False, memory is available via tools only."
        ),
    )

    index_mode: bool = Field(
        default=True,
        description=(
            "If True (default), inline only a MEMORY INDEX — one line per persistent fact "
            "(name + one-line description, not the full body) plus the most recent "
            "`max_session_history_entries` session entries. Full fact bodies are served on "
            "demand via the memory_get / memory_search tools, so the per-turn memory tax "
            "stays small instead of growing with the whole MEMORY.md. If False, the entire "
            "MEMORY.md is inlined every turn (legacy behaviour)."
        ),
    )

    max_session_history_entries: int = Field(
        default=8,
        ge=0,
        le=200,
        description=(
            "When index_mode is True, how many of the most recent Session History entries to "
            "inline. Older entries stay in MEMORY.md / history.jsonl and are not injected. "
            "The injected index hard-caps this at 8 lines (TIME-03) — values above 8 render "
            "8; lower values render fewer."
        ),
    )

    trace_ambient_injection: bool = Field(
        default=True,
        description=(
            "Record an activation trace for the every-turn ambient memory shelf (owner reversal "
            "2026-07-17 of the P0 exclusion). Each turn's injected fact set is a co-firing event: "
            "one best-effort, per-turn-deduped row tagged source='injection' — DISTINGUISHABLE from "
            "model-initiated retrieval (memory_search / memory_get), which downstream consumers "
            "discount (downstream consumers may discount it). A trace-write failure "
            "never disturbs the turn. False restores the pre-reversal behavior: no injection-trace "
            "rows, the injected block byte-identical. Mutable via `localharness components set "
            "agent.memory.trace_ambient_injection <true|false>`."
        ),
    )

    embedding_model: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        description=(
            "The subject-family embedding model whose representation space carries every "
            "memory similarity judgment (search ranking, dreaming's stream replay, "
            "write-time re-sighting). Runs locally on CPU via sentence-transformers "
            "(`uv sync --extra embeddings`); weights download to the HF cache on first "
            "use. Changing it triggers a full re-embed at the next dreaming pass "
            "(vectors from different models are not comparable). There is NO fallback: "
            "with the model unavailable, memory search and remember fail loudly."
        ),
    )

    consolidation: MemoryConsolidationConfig = Field(
        default_factory=MemoryConsolidationConfig,
        description="Idle-time consolidation pass (v2.0 CONS) — see MemoryConsolidationConfig.",
    )

    archival: MemoryArchivalConfig = Field(
        default_factory=MemoryArchivalConfig,
        description=(
            "Dormancy archival, the forgetting half of consolidation (rung 1) — OFF by "
            "default, see MemoryArchivalConfig."
        ),
    )
