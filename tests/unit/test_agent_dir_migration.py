"""The root-agent state-dir rename (Phase 33.1, ORCH-02) lives in core: `start` runs it with memory
on or off, so it must not live in (or import) the memory plugin it is about to be separated from."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import localharness.core.agent_dir as agent_dir
from localharness.core.agent_dir import (
    _LEGACY_ROOT_AGENT_ID,
    _ROOT_AGENT_ID,
    _legacy_adoption_is_pending,
    _migrate_legacy_root_agent_dir,
)


def test_migration_lives_in_core():
    assert (_LEGACY_ROOT_AGENT_ID, _ROOT_AGENT_ID) == ("default", "orchestrator")
    tree = ast.parse(Path(agent_dir.__file__).read_text())
    imported = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imported += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not [m for m in imported if m.startswith("localharness.memory")]


def test_memory_store_uses_the_core_function():
    import localharness.memory.sqlite as sqlite
    assert sqlite._migrate_legacy_root_agent_dir is _migrate_legacy_root_agent_dir


def test_migration_renames_the_legacy_root_dir(tmp_path):
    legacy = tmp_path / "agents" / "default"
    legacy.mkdir(parents=True)
    (legacy / "memory.db").write_bytes(b"db")
    (legacy / "history.jsonl").write_text("")
    assert _legacy_adoption_is_pending(tmp_path, "orchestrator")
    _migrate_legacy_root_agent_dir(tmp_path, "orchestrator")
    new = tmp_path / "agents" / "orchestrator"
    assert not legacy.exists() and (new / "memory.db").read_bytes() == b"db"
    (record,) = [json.loads(line) for line in (new / "history.jsonl").read_text().splitlines()]
    assert (record["event"], record["from_agent_id"], record["to_agent_id"]) == (
        "agent_renamed", "default", "orchestrator")
    _migrate_legacy_root_agent_dir(tmp_path, "orchestrator")  # idempotent
    assert len((new / "history.jsonl").read_text().splitlines()) == 1


def test_migration_refuses_when_the_destination_exists_or_the_collision_marker_is_there(tmp_path):
    agents = tmp_path / "agents"
    (agents / "default").mkdir(parents=True)
    (agents / "default.yaml").write_text("")
    _migrate_legacy_root_agent_dir(tmp_path, "orchestrator")
    assert (agents / "default").is_dir() and not (agents / "orchestrator").exists()
    (agents / "default.yaml").unlink()
    (agents / "orchestrator").mkdir()
    _migrate_legacy_root_agent_dir(tmp_path, "orchestrator")
    assert (agents / "default").is_dir()
    _migrate_legacy_root_agent_dir(tmp_path, "worker")  # non-root: no-op
    assert (agents / "default").is_dir()
