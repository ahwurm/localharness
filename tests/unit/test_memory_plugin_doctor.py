"""The memory doctor checks (47-05, MEMP-08): each memory.db opened read-only, the embedding model
and package looked up locally — no LLM, no session, no network, nothing loaded or downloaded."""
from __future__ import annotations

import importlib.util
import socket
import sqlite3

import huggingface_hub
import pytest

from localharness.config.models import MemoryConfig
from localharness.core.bus import EventBus
from localharness.memory.plugin import MemoryPlugin
from localharness.memory.sqlite import CURRENT_SCHEMA_VERSION, MemoryStore
from localharness.plugins.api import PluginContext, PluginPaths
from localharness.tools.registry import ToolRegistry

MODEL = "Qwen/Qwen3-Embedding-0.6B"


def _ctx(tmp_path, agent_config=None) -> PluginContext:
    return PluginContext(bus=EventBus(), tools=ToolRegistry(), hooks=None, config=None,
                         agent_config=agent_config, paths=PluginPaths(tmp_path, None, tmp_path), llm=None)


@pytest.fixture
def cache(monkeypatch):
    """Model present and package installed unless a test says otherwise."""
    state = {"model": True, "package": True, "asked": []}

    def lookup(repo_id, filename, **kw):
        state["asked"].append((repo_id, filename))
        return f"/cache/{repo_id}/{filename}" if state["model"] else None

    real = importlib.util.find_spec
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lookup)
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: (object() if state["package"] else None)
                        if name == "sentence_transformers" else real(name, *a))
    return state


def _rows(tmp_path, name, agent_config=None):
    return [c for c in MemoryPlugin().doctor(_ctx(tmp_path, agent_config)) if c.name == name]


async def _real_db(tmp_path):
    store = MemoryStore(agent_id="orchestrator", division_id="default", org_id="default",
                        base_dir=str(tmp_path), global_base_dir=str(tmp_path))
    await store.open()
    await store.close()
    return tmp_path / "agents" / "orchestrator" / "memory.db"


def test_no_database_skips(tmp_path, cache) -> None:
    rows = _rows(tmp_path, "memory-db")
    assert len(rows) == 1
    assert (rows[0].status, rows[0].detail) == ("skip", "no memory database yet — created on first start")


async def test_a_real_database_passes_read_only(tmp_path, cache) -> None:
    db = await _real_db(tmp_path)
    before = (db.stat().st_mtime_ns, db.stat().st_size)
    [row] = _rows(tmp_path, "memory-db")
    assert row.status == "pass"
    assert str(db) in row.detail and f"schema {CURRENT_SCHEMA_VERSION}" in row.detail
    assert (db.stat().st_mtime_ns, db.stat().st_size) == before
    # The store runs in WAL mode: a read-only reader may leave SQLite's -shm/-wal side files behind
    # (it cannot checkpoint), but the WAL holds no frames — nothing was written.
    wal = db.with_name("memory.db-wal")
    assert not wal.exists() or wal.stat().st_size == 0


def test_a_corrupt_database_fails(tmp_path, cache) -> None:
    db = tmp_path / "agents" / "x" / "memory.db"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"this is not sqlite at all " * 100)
    [row] = _rows(tmp_path, "memory-db")
    assert row.status == "fail" and str(db) in row.detail and row.hint


async def test_a_newer_schema_fails(tmp_path, cache) -> None:
    db = await _real_db(tmp_path)
    conn = sqlite3.connect(db)
    conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION + 1}")
    conn.commit()
    conn.close()
    [row] = _rows(tmp_path, "memory-db")
    assert row.status == "fail" and "newer localharness" in row.hint


async def test_one_row_per_database(tmp_path, cache) -> None:
    await _real_db(tmp_path)
    bad = tmp_path / "agents" / "zz" / "memory.db"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"x" * 4096)
    assert [r.status for r in _rows(tmp_path, "memory-db")] == ["pass", "fail"]


def test_embedding_model_present(tmp_path, cache) -> None:
    [row] = _rows(tmp_path, "memory-embedding")
    assert row.status == "pass" and MODEL in row.detail
    assert set(cache["asked"]) == {(MODEL, "config.json"), (MODEL, "modules.json")}


def test_embedding_model_follows_agent_config(tmp_path, cache) -> None:
    [row] = _rows(tmp_path, "memory-embedding", MemoryConfig(embedding_model="org/other-embed"))
    assert row.status == "pass" and "org/other-embed" in row.detail


def test_embedding_model_missing(tmp_path, cache) -> None:
    cache["model"] = False
    [row] = _rows(tmp_path, "memory-embedding")
    assert row.status == "fail"
    assert MODEL in row.hint and "the first memory search would otherwise download it" in row.hint


def test_embedding_package_missing(tmp_path, cache) -> None:
    cache["package"] = False
    [row] = _rows(tmp_path, "memory-embedding")
    assert row.status == "fail" and "uv sync --extra embeddings" in row.hint


async def test_doctor_needs_no_llm_session_or_network(tmp_path, cache, monkeypatch) -> None:
    await _real_db(tmp_path)

    def no_network(*a, **kw): raise OSError("network used")
    monkeypatch.setattr(socket, "socket", no_network)
    ctx = _ctx(tmp_path)
    assert ctx.llm is None and ctx.session is None
    p = MemoryPlugin()
    assert await p.configure(ctx) == "ready"
    assert [(c.name, c.status) for c in p.doctor(ctx)] == [("memory-db", "pass"), ("memory-embedding", "pass")]
