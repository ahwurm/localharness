"""Store spine for the `/memory` window: get_fact_by_id (any status), recent_facts (newest
first), forget_fact (user-forget supersede — retire, NEVER hard-delete, chain stays auditable).

These are the read/lifecycle primitives the window renders over; they must be WAL-safe and reuse
the existing supersede mechanism, so they get their own store-level coverage before the CLI layer.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from localharness.memory.sqlite import (
    USER_FORGET_PROVENANCE_PREFIX,
    FactQuery,
    MemoryStore,
)


def make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        agent_id="test-agent", division_id="test-div", org_id="default",
        base_dir=str(tmp_path),
    )


@pytest.mark.asyncio
async def test_get_fact_by_id_reaches_active_and_superseded(tmp_path):
    store = make_store(tmp_path)
    await store.open()
    try:
        # A supersede chain: v1 -> v2 for the same key.
        v1 = await store.store_fact(key="port", value="vLLM on 8081", confidence=0.9)
        v2 = await store.store_fact(key="port", value="vLLM on 8090", confidence=0.9)
        assert v1.id != v2.id
        # The active row is reachable by id.
        got_active = await store.get_fact_by_id(v2.id)
        assert got_active is not None and got_active.status == "active"
        # The SUPERSEDED row is reachable by id too (active-only reads can't reach it).
        got_super = await store.get_fact_by_id(v1.id)
        assert got_super is not None and got_super.status == "superseded"
        assert await store.get_facts_by_ids([v1.id]) == []  # proof: active-only read misses it
        # Unknown id -> None.
        assert await store.get_fact_by_id(999999) is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_recent_facts_newest_first_active_only(tmp_path):
    store = make_store(tmp_path)
    await store.open()
    try:
        now = int(time.time())
        a = await store.store_fact(key="a", value="one", confidence=0.8)
        b = await store.store_fact(key="b", value="two", confidence=0.8)
        c = await store.store_fact(key="c", value="three", confidence=0.8)
        # Nudge updated_at so ordering is deterministic (a<b<c).
        await store._db.execute("UPDATE facts SET updated_at=? WHERE id=?", (now - 30, a.id))
        await store._db.execute("UPDATE facts SET updated_at=? WHERE id=?", (now - 20, b.id))
        await store._db.execute("UPDATE facts SET updated_at=? WHERE id=?", (now - 10, c.id))
        await store._db.commit()
        recent = await store.recent_facts(limit=2)
        assert [f.key for f in recent] == ["c", "b"]  # newest first, limited
        # Forgetting b drops it from the recency feed (active-only).
        await store.forget_fact(b.id)
        recent2 = await store.recent_facts(limit=5)
        assert [f.key for f in recent2] == ["c", "a"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_forget_fact_supersedes_with_marker_and_stays_auditable(tmp_path):
    store = make_store(tmp_path)
    await store.open()
    try:
        f = await store.store_fact(key="secret", value="the passphrase is hunter2",
                                   confidence=0.9, source="remember", provenance="sess-1")
        ok = await store.forget_fact(f.id)
        assert ok is True
        # Retired: gone from every active hot path.
        assert await store.get_fact("secret") is None
        assert await store.query_facts(FactQuery(text="passphrase", limit=10)) == []
        # NEVER hard-deleted: the row survives, now superseded, with a user-forget provenance.
        row = await store.get_fact_by_id(f.id)
        assert row is not None and row.status == "superseded"
        assert row.provenance.startswith(USER_FORGET_PROVENANCE_PREFIX)
        assert "sess-1" in row.provenance  # original provenance preserved for audit
        assert row.value == "the passphrase is hunter2"  # content intact (auditable)
        # The chain stays auditable via history.
        hist = await store.get_fact_history("secret")
        assert any(h.id == f.id and h.status == "superseded" for h in hist)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_forget_fact_already_superseded_is_noop(tmp_path):
    store = make_store(tmp_path)
    await store.open()
    try:
        v1 = await store.store_fact(key="k", value="v1", confidence=0.9)
        await store.store_fact(key="k", value="v2", confidence=0.9)  # supersedes v1
        # v1 is already superseded -> forgetting it retires nothing (atomic status guard).
        assert await store.forget_fact(v1.id) is False
        # Unknown id -> False.
        assert await store.forget_fact(424242) is False
    finally:
        await store.close()
