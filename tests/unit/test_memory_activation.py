"""Phase 30 (v2.0 Hierarchical Memory) — graph substrate + activation scoring: RANK-01..05.

Covers: typed graph with cycle-guarded traversal (RANK-01), ACT-R activation in SQL +
staging discipline (RANK-02/04), the confidence split + importance prior (RANK-03),
partial-index hot-path guarantee (RANK-05), and the v3 schema ladder.
RANK-06 (GB10 micro-bench) is a committed script (scripts/microbench_prefix_cache.py);
its live run is deferred by owner ruling.
"""
import time
from pathlib import Path

import pytest

from localharness.memory.sqlite import FactQuery, MemoryStore


@pytest.fixture
async def store(tmp_path: Path):
    s = MemoryStore(agent_id="act-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


# ---------------------------------------------------------------------------
# Schema v3 ladder
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fresh_db_lands_on_v3_with_graph(store: MemoryStore):
    async with store._db.execute("PRAGMA user_version") as cur:
        row = await cur.fetchone()
    assert row[0] == 3
    async with store._db.execute("PRAGMA table_info(facts)") as cur:
        cols = {r[1] for r in await cur.fetchall()}
    assert {"retrieval_strength", "importance", "access_count", "last_accessed_at",
            "access_count_staged", "last_accessed_staged", "node_kind"} <= cols
    async with store._db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='edges'"
    ) as cur:
        assert await cur.fetchone() is not None


# ---------------------------------------------------------------------------
# RANK-04: staging discipline — reads never move the injected block
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_staged_reads_keep_injected_block_byte_stable(store: MemoryStore):
    await store.store_fact("a-fact", "alpha body")
    await store.store_fact("b-fact", "beta body")

    before = await store._render_memory_index(10)
    for _ in range(5):
        await store.touch_staged(["b-fact"])
    after_reads = await store._render_memory_index(10)
    assert after_reads == before  # reads are invisible to the block (RANK-04)

    folded = await store.fold_staged_access()
    assert folded == 1
    after_fold = await store._render_memory_index(10)
    assert after_fold != before
    lines = [ln for ln in after_fold.splitlines() if ln.startswith("- ")]
    assert lines[0].startswith("- b-fact:")  # the used fact now ranks first


@pytest.mark.asyncio
async def test_fold_is_idempotent(store: MemoryStore):
    await store.store_fact("k", "v")
    await store.touch_staged(["k"])
    assert await store.fold_staged_access() == 1
    assert await store.fold_staged_access() == 0  # nothing staged → no-op
    fact = await store.get_fact("k")
    assert fact.access_count == 1


# ---------------------------------------------------------------------------
# RANK-02: ACT-R activation beats pure recency
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_frequently_used_old_fact_outranks_fresh_unused(store: MemoryStore):
    now = int(time.time())
    await store.store_fact("old-workhorse", "used constantly")
    await store._db.execute(
        "UPDATE facts SET updated_at = ?, created_at = ?, access_count = 10, "
        "last_accessed_at = ? WHERE agent_id = ? AND key = 'old-workhorse'",
        (now - 10 * 86400, now - 10 * 86400, now - 86400, "act-agent"),
    )
    await store._db.commit()
    await store.store_fact("new-never-used", "just written")

    index = await store._render_memory_index(10)
    lines = [ln for ln in index.splitlines() if ln.startswith("- ")]
    # Pure recency would put new-never-used first; ACT-R puts the workhorse first:
    # ln(11) − 0.5·ln(2 days) ≈ 2.05  >  ln(1) − 0.5·ln(1) = 0.
    assert lines[0].startswith("- old-workhorse:")


# ---------------------------------------------------------------------------
# RANK-03: confidence split + importance prior
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_supersede_drops_retrieval_strength_and_leaves_index(store: MemoryStore):
    await store.store_fact("thesis", "old view")
    await store.store_fact("thesis", "new view")
    history = await store.get_fact_history("thesis")
    old = next(f for f in history if f.value == "old view")
    assert old.retrieval_strength <= 0.1  # lost the retrieval competition
    index = await store._render_memory_index(10)
    assert "old view" not in index and "new view" in index


@pytest.mark.asyncio
async def test_importance_prior_from_tags_not_llm(store: MemoryStore):
    f_remember = await store.store_fact("r", "v", source="remember", confidence=0.9)
    f_gate = await store.store_fact("g", "v", tags=["gate", "tier:resolved_error"], confidence=0.65)
    f_plain = await store.store_fact("p", "v")
    assert f_remember.importance == 0.4
    assert f_gate.importance == 0.3
    assert f_plain.importance == 0.0


@pytest.mark.asyncio
async def test_fused_search_ranks_used_trusted_first(store: MemoryStore):
    await store.store_fact("hot", "vpn is required for deploys", confidence=1.0)
    await store.store_fact("cold", "vpn also mentioned here once", confidence=0.4)
    await store._db.execute(
        "UPDATE facts SET access_count = 20, last_accessed_at = ? "
        "WHERE agent_id = ? AND key = 'hot'",
        (int(time.time()), "act-agent"),
    )
    await store._db.commit()
    results = await store.query_facts(FactQuery(text="vpn", min_confidence=0.0))
    assert [f.key for f in results][0] == "hot"


# ---------------------------------------------------------------------------
# RANK-01: typed graph, cycle-guarded traversal
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_neighborhood_survives_a_cycle(store: MemoryStore):
    a = await store.store_fact("node-a", "a")
    b = await store.store_fact("node-b", "b")
    c = await store.store_fact("node-c", "c")
    await store.add_edge(a.id, b.id, "supports")
    await store.add_edge(b.id, c.id, "supports")
    await store.add_edge(c.id, a.id, "supports")  # the cycle machine-written edges WILL make

    walk = await store.neighborhood(a.id, depth=10, limit=50)  # depth hard-capped at 4
    ids = {node_id for node_id, _ in walk}
    assert ids == {a.id, b.id, c.id}  # terminates; every node once
    depths = dict(walk)
    assert depths[a.id] == 0 and depths[b.id] == 1 and depths[c.id] == 1  # undirected


@pytest.mark.asyncio
async def test_edge_kind_is_validated(store: MemoryStore):
    a = await store.store_fact("x", "1")
    b = await store.store_fact("y", "2")
    with pytest.raises(ValueError):
        await store.add_edge(a.id, b.id, "supersedes")  # column, not edge — by design
    await store.add_edge(a.id, b.id, "derived_from")  # idempotent
    await store.add_edge(a.id, b.id, "derived_from")


# ---------------------------------------------------------------------------
# RANK-05: superseded rows are out of the hot path via partial index
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_default_scan_uses_partial_active_index(store: MemoryStore):
    await store.store_fact("k1", "v1")
    async with store._db.execute(
        "EXPLAIN QUERY PLAN SELECT key FROM facts "
        "WHERE agent_id = ? AND status = 'active' ORDER BY updated_at DESC",
        ("act-agent",),
    ) as cur:
        plan = " ".join(str(tuple(r)) for r in await cur.fetchall())
    assert "idx_facts_active_recency" in plan or "ux_facts_active_key" in plan, plan


# ---------------------------------------------------------------------------
# Read tools bump staging (integration with memory_tools)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_and_get_touch_staging(store: MemoryStore):
    from localharness.tools.builtin.memory_tools import MemoryGetTool, MemorySearchTool

    await store.store_fact("stag", "staging target value")
    await MemorySearchTool(store)._execute(query="staging")
    await MemoryGetTool(store)._execute(name="stag")

    async with store._db.execute(
        "SELECT access_count, access_count_staged FROM facts WHERE key='stag'"
    ) as cur:
        base, staged = await cur.fetchone()
    assert base == 0 and staged == 2  # both reads staged, none folded
