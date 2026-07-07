"""P0 activation-trace log (tag-graph substrate): pure append-only bookkeeping.

One row per retrieval event — stimulus digest (hash + truncated text), the atoms it
fired, and the subset actually injected into context. Traces cannot be backfilled, so
the log ships before any consumer (co-activation weights, pattern-completion retrieval)
exists. These tests pin: the search/recall seams append a row, the injected subset is
recorded distinctly, the log is append-only (no mutation API), a trace-write failure
never breaks retrieval, and the v4->v5 migration is additive + idempotent.
"""
from pathlib import Path

import pytest

from localharness.memory.sqlite import CURRENT_SCHEMA_VERSION, MemoryStore
from localharness.tools.builtin.memory_tools import MemoryGetTool, MemorySearchTool


def make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        agent_id="test-agent",
        division_id="test-div",
        org_id="default",
        base_dir=str(tmp_path),
    )


# ---------------------------------------------------------------------------
# Seam: memory_search appends one trace per retrieval event
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_appends_trace_with_fired_ids_and_stimulus(tmp_path: Path):
    """A search that returns hits appends ONE trace row: fired_ids == the hit atom ids,
    stimulus_text == the query, injected == fired (all hits are rendered), source tagged."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("recipe_key", "banana smoothie recipe with honey")
        await store.store_fact("car_key", "car maintenance schedule")
        recipe_id = (await store.get_fact("recipe_key")).id

        res = await MemorySearchTool(store).run(query="smoothie")
        assert res.success and "recipe_key" in res.output

        traces = await store.recent_activation_traces()
        assert len(traces) == 1
        t = traces[0]
        assert t.fired_ids == [recipe_id]
        assert t.injected_ids == [recipe_id]        # every hit was rendered
        assert t.stimulus_text == "smoothie"
        assert t.stimulus_hash                        # digest present
        assert t.source == "memory_search"
        assert t.agent_id == "test-agent"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_search_no_hits_records_no_trace(tmp_path: Path):
    """A retrieval event with zero hits is not an activation — no row (the early 'No facts
    matched' return path fires before the trace hook)."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("recipe_key", "banana smoothie recipe with honey")
        res = await MemorySearchTool(store).run(query="thismatchesnothingxyz")
        assert res.success
        assert await store.recent_activation_traces() == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_memory_get_appends_recall_trace(tmp_path: Path):
    """memory_get(name) surfaces one atom to the model — a recall event: fired == injected
    == [that atom id], stimulus == the requested name, source tagged."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("deploy_note", "deploy requires the vpn")
        atom_id = (await store.get_fact("deploy_note")).id

        res = await MemoryGetTool(store).run(name="deploy_note")
        assert res.success

        traces = await store.recent_activation_traces()
        assert len(traces) == 1
        assert traces[0].fired_ids == [atom_id]
        assert traces[0].injected_ids == [atom_id]
        assert traces[0].stimulus_text == "deploy_note"
        assert traces[0].source == "memory_get"
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Store contract: injected subset, digest cap, append-only
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_injected_subset_recorded_distinctly(tmp_path: Path):
    """The injected set is a recorded subset of the fired set (the schema must carry both,
    for future seams that fire N candidates but inject only top-k)."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.record_activation_trace(
            stimulus="what did we decide", fired_ids=[1, 2, 3], injected_ids=[1, 3],
            source="test",
        )
        t = (await store.recent_activation_traces())[0]
        assert t.fired_ids == [1, 2, 3]
        assert t.injected_ids == [1, 3]
        assert t.injected_ids != t.fired_ids
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_stimulus_digest_hashes_full_but_truncates_text(tmp_path: Path):
    """Stimulus digest = hash of the FULL query text + the text truncated to the ~200-char
    cap. Two long stimuli that share a 200-char prefix still get distinct hashes."""
    store = make_store(tmp_path)
    await store.open()
    try:
        long_a = "x" * 250 + "AAA"
        long_b = "x" * 250 + "BBB"
        await store.record_activation_trace(stimulus=long_a, fired_ids=[1], injected_ids=[1], source="t")
        await store.record_activation_trace(stimulus=long_b, fired_ids=[1], injected_ids=[1], source="t")
        traces = await store.recent_activation_traces()
        # text is capped...
        assert all(len(t.stimulus_text) <= 200 for t in traces)
        # ...but the hash is of the full text, so the two rows differ despite equal prefixes.
        assert traces[0].stimulus_hash != traces[1].stimulus_hash
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_log_is_append_only(tmp_path: Path):
    """Append-only: the log only grows, re-recording never overwrites, and NO mutation API
    (update/delete) is exposed on the store."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.record_activation_trace(stimulus="q", fired_ids=[1], injected_ids=[1], source="t")
        await store.record_activation_trace(stimulus="q", fired_ids=[1], injected_ids=[1], source="t")
        assert len(await store.recent_activation_traces()) == 2  # grew, never coalesced
        # No mutation surface exists on the store.
        assert not hasattr(store, "update_activation_trace")
        assert not hasattr(store, "delete_activation_trace")
        assert not hasattr(store, "clear_activation_traces")
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_traces_for_atom_forensics(tmp_path: Path):
    """The traces-for-atom read helper returns exactly the events an atom fired in — plain
    log read, no scoring/weights/spreading."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.record_activation_trace(stimulus="a", fired_ids=[5, 6], injected_ids=[5], source="t")
        await store.record_activation_trace(stimulus="b", fired_ids=[6, 7], injected_ids=[6], source="t")
        assert len(await store.activation_traces_for_atom(5)) == 1
        assert len(await store.activation_traces_for_atom(6)) == 2
        assert len(await store.activation_traces_for_atom(99)) == 0
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Best-effort: a trace-write failure never breaks the retrieval
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_trace_write_failure_does_not_break_search(tmp_path: Path, monkeypatch):
    """If the trace write raises, memory_search still returns its hits (best-effort). The
    hook IS invoked (proving it is wired) but its failure is swallowed."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("recipe_key", "banana smoothie recipe with honey")
        called = {"n": 0}

        async def boom(**kwargs):
            called["n"] += 1
            raise RuntimeError("trace store down")

        monkeypatch.setattr(store, "record_activation_trace", boom)
        res = await MemorySearchTool(store).run(query="smoothie")
        assert res.success                       # retrieval survived the trace failure
        assert "recipe_key" in res.output
        assert called["n"] == 1                  # the hook fired (best-effort wrapper caught it)
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Schema migration v4 -> v5: additive + idempotent on an existing store
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_migration_v4_to_v5_adds_activation_traces(tmp_path: Path):
    """A hand-built v4 store opens at v5 with the activation_traces table present and the
    pre-existing fact row byte-unchanged (additive-only, matching the v3->v4 precedent)."""
    import aiosqlite

    from localharness.memory.sqlite import (
        MIGRATION_V2_TO_V3_SQL,
        MIGRATION_V3_TO_V4_SQL,
        SCHEMA_V2_SQL,
    )

    agent_dir = tmp_path / "agents" / "test-agent"
    agent_dir.mkdir(parents=True)
    db_path = agent_dir / "memory.db"
    conn = await aiosqlite.connect(str(db_path))
    try:
        await conn.executescript(SCHEMA_V2_SQL)
        await conn.executescript(MIGRATION_V2_TO_V3_SQL)
        await conn.executescript(MIGRATION_V3_TO_V4_SQL)  # stamps user_version = 4
        await conn.execute(
            "INSERT INTO facts (agent_id, key, value, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("test-agent", "pre-v5-fact", "survives migration", 111, 222),
        )
        await conn.commit()
        async with conn.execute("PRAGMA user_version") as cur:
            assert (await cur.fetchone())[0] == 4  # the fixture really is a v4 DB
        async with conn.execute(
            "SELECT id, agent_id, key, value, status FROM facts"
        ) as cur:
            fact_before = tuple(await cur.fetchone())
    finally:
        await conn.close()

    store = make_store(tmp_path)
    await store.open()
    try:
        async with store._db.execute("PRAGMA user_version") as cur:
            assert (await cur.fetchone())[0] == CURRENT_SCHEMA_VERSION  # ladder carried v4 -> v5
        async with store._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = 'activation_traces'"
        ) as cur:
            assert (await cur.fetchone()) is not None
        async with store._db.execute(
            "SELECT id, agent_id, key, value, status FROM facts"
        ) as cur:
            assert tuple(await cur.fetchone()) == fact_before  # byte-unchanged
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_migration_v5_idempotent_reopen(tmp_path: Path):
    """Open + close + reopen stays at v5 with exactly one activation_traces table, and an
    appended trace survives the reopen."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.record_activation_trace(stimulus="persist me", fired_ids=[1], injected_ids=[1], source="t")
    finally:
        await store.close()

    store2 = make_store(tmp_path)
    await store2.open()
    try:
        async with store2._db.execute("PRAGMA user_version") as cur:
            assert (await cur.fetchone())[0] == CURRENT_SCHEMA_VERSION
        async with store2._db.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name = 'activation_traces'"
        ) as cur:
            assert (await cur.fetchone())[0] == 1  # reopen did not re-create
        traces = await store2.recent_activation_traces()
        assert len(traces) == 1 and traces[0].stimulus_text == "persist me"
    finally:
        await store2.close()
