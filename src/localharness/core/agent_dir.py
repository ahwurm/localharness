"""Agent state-dir layout migrations that run for every start, memory on or off (moved out of memory/sqlite.py so core never imports the memory plugin)."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Phase 33.1 (ORCH-02): one-time root-agent rename (default -> orchestrator)
# ---------------------------------------------------------------------------
# The agent's name IS its storage identity (directory name + the agent_id column
# in facts/sessions + the bus filter), so the default->orchestrator rename must
# reconcile pre-existing 'default'-keyed data the first time the store opens under
# the new root name — otherwise every memory an existing install has is orphaned.
_LEGACY_ROOT_AGENT_ID = "default"
_ROOT_AGENT_ID = "orchestrator"


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
