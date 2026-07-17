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


# ---------------------------------------------------------------------------
# Ambient-injection traces (owner reversal 2026-07-17 of the P0 exclusion): the
# every-turn memory shelf is a co-firing event too — recorded source='injection',
# per-turn-deduped, best-effort. The raw log keeps full fidelity; the discount that
# distinguishes it from model-initiated retrieval lives downstream (discovery).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_injection_trace_records_source_tagged_row(tmp_path: Path):
    """record_injection_trace writes ONE row tagged source='injection', fired == injected ==
    the rendered atom ids, stimulus == the turn's user message."""
    store = make_store(tmp_path)
    await store.open()
    try:
        rid = await store.record_injection_trace(
            stimulus="where do api keys live?", injected_ids=[7, 9], session_id="sit-1",
        )
        assert rid  # a real rowid
        traces = await store.recent_activation_traces()
        assert len(traces) == 1
        t = traces[0]
        assert t.source == "injection"
        assert t.fired_ids == [7, 9]
        assert t.injected_ids == [7, 9]        # shelf renders exactly what it selects
        assert t.stimulus_text == "where do api keys live?"
        assert t.stimulus_hash                  # digest present
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_injection_trace_per_turn_dedupe(tmp_path: Path):
    """One firing per turn assembly: a second assembly of the SAME turn (same session +
    stimulus) collapses to the existing row; a new stimulus (next turn) or a new session
    (sitting) is a distinct row."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.record_injection_trace(stimulus="turn one", injected_ids=[1, 2], session_id="s1")
        await store.record_injection_trace(stimulus="turn one", injected_ids=[1, 2], session_id="s1")  # re-assembly
        injection = [t for t in await store.recent_activation_traces() if t.source == "injection"]
        assert len(injection) == 1  # deduped — not two rows for one turn

        await store.record_injection_trace(stimulus="turn two", injected_ids=[1, 3], session_id="s1")
        injection = [t for t in await store.recent_activation_traces() if t.source == "injection"]
        assert len(injection) == 2  # a different stimulus is a distinct turn

        await store.record_injection_trace(stimulus="turn one", injected_ids=[1, 2], session_id="s2")
        injection = [t for t in await store.recent_activation_traces() if t.source == "injection"]
        assert len(injection) == 3  # same stimulus, different sitting -> distinct
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_retrieval_traces_not_deduped_by_injection_index(tmp_path: Path):
    """The dedup index is PARTIAL (WHERE source='injection') — retrieval-source rows
    (memory_search / memory_get) stay append-only exactly as before, even when identical."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.record_activation_trace(stimulus="q", fired_ids=[1], injected_ids=[1], source="memory_search")
        await store.record_activation_trace(stimulus="q", fired_ids=[1], injected_ids=[1], source="memory_search")
        retrieval = [t for t in await store.recent_activation_traces() if t.source == "memory_search"]
        assert len(retrieval) == 2  # unchanged — no dedup on the retrieval path
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_load_context_exposes_injected_ids_without_changing_render(tmp_path: Path):
    """load_context exposes the atom ids it rendered into the shelf (the injected set) while the
    rendered markdown stays byte-identical to _render_memory_index (the string contract is the
    thin wrapper's; the ids are captured alongside, not by re-querying)."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("deploy_note", "deploy requires the vpn", confidence=0.9)
        await store.store_fact("api_key_loc", "api keys live in the vault", confidence=0.9)
        ids = {(await store.get_fact("deploy_note")).id, (await store.get_fact("api_key_loc")).id}

        ctx = await store.load_context()
        assert set(ctx.injected_fact_ids) == ids
        # byte-identity: the wrapper returns exactly _render_memory_index's string.
        assert ctx.agent_memory_md == await store._render_memory_index(8)
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Schema migration v7 -> v8: additive partial-unique injection dedup index
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_migration_v7_to_v8_adds_injection_dedup_index(tmp_path: Path):
    """A hand-built v7 store opens at v8 with the partial-unique injection index present and a
    pre-existing NON-injection trace row byte-unchanged (additive: the index only covers
    source='injection')."""
    import aiosqlite

    from localharness.memory.sqlite import (
        MIGRATION_V2_TO_V3_SQL,
        MIGRATION_V3_TO_V4_SQL,
        MIGRATION_V4_TO_V5_SQL,
        MIGRATION_V5_TO_V6_SQL,
        MIGRATION_V6_TO_V7_SQL,
        SCHEMA_V2_SQL,
    )

    agent_dir = tmp_path / "agents" / "test-agent"
    agent_dir.mkdir(parents=True)
    conn = await aiosqlite.connect(str(agent_dir / "memory.db"))
    try:
        for sql in (SCHEMA_V2_SQL, MIGRATION_V2_TO_V3_SQL, MIGRATION_V3_TO_V4_SQL,
                    MIGRATION_V4_TO_V5_SQL, MIGRATION_V5_TO_V6_SQL, MIGRATION_V6_TO_V7_SQL):
            await conn.executescript(sql)
        await conn.execute(
            "INSERT INTO activation_traces (agent_id, session_id, turn, stimulus_hash, "
            "stimulus_text, fired_ids, injected_ids, source, ts) "
            "VALUES (?, '', NULL, 'h0', 'q', '[1]', '[1]', 'memory_search', 100)",
            ("test-agent",),
        )
        await conn.commit()
        async with conn.execute("PRAGMA user_version") as cur:
            assert (await cur.fetchone())[0] == 7  # the fixture really is a v7 DB
    finally:
        await conn.close()

    store = make_store(tmp_path)
    await store.open()
    try:
        async with store._db.execute("PRAGMA user_version") as cur:
            assert (await cur.fetchone())[0] == CURRENT_SCHEMA_VERSION  # ladder carried v7 -> v8
        async with store._db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name = 'ux_activation_traces_injection'"
        ) as cur:
            assert (await cur.fetchone()) is not None  # the partial-unique index exists
        rows = await store.recent_activation_traces()
        assert len(rows) == 1 and rows[0].source == "memory_search"  # pre-existing row survived
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_migration_v8_idempotent_reopen(tmp_path: Path):
    """Open + close + reopen stays at v8 with exactly one injection index, and dedup still holds
    across the reopen (re-recording the same turn is a no-op)."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.record_injection_trace(stimulus="m", injected_ids=[1, 2], session_id="s")
    finally:
        await store.close()

    store2 = make_store(tmp_path)
    await store2.open()
    try:
        async with store2._db.execute("PRAGMA user_version") as cur:
            assert (await cur.fetchone())[0] == CURRENT_SCHEMA_VERSION
        async with store2._db.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' "
            "AND name = 'ux_activation_traces_injection'"
        ) as cur:
            assert (await cur.fetchone())[0] == 1  # reopen did not re-create
        await store2.record_injection_trace(stimulus="m", injected_ids=[1, 2], session_id="s")
        injection = [t for t in await store2.recent_activation_traces() if t.source == "injection"]
        assert len(injection) == 1  # dedup survives the reopen
    finally:
        await store2.close()
