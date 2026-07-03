"""Feature 1: memory as a queryable handle.

- load_context(index_mode=True) renders an INDEX (fact names + one-line descriptions),
  NOT full bodies.
- session-history cap inlines only the last N entries.
- memory_get returns a fact's full body; memory_search finds a seeded fact (FTS5).
"""
from pathlib import Path

import pytest

from localharness.memory.sqlite import MemoryStore
from localharness.tools.builtin.memory_tools import MemoryGetTool, MemorySearchTool


def make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        agent_id="test-agent",
        division_id="test-div",
        org_id="default",
        base_dir=str(tmp_path),
    )


@pytest.mark.asyncio
async def test_index_has_names_not_bodies(tmp_path: Path):
    store = make_store(tmp_path)
    await store.open()
    try:
        # Multi-line body: the index carries only the FIRST line (truncated), so deeper
        # lines must be absent.
        long_body = "First summary line of the procedure.\n" + ("DEEP_BODY_TOKEN " * 50)
        await store.store_fact("deploy_procedure", long_body)
        ctx = await store.load_context(index_mode=True)
        md = ctx.agent_memory_md
        # Name + its one-line description appear in the index...
        assert "deploy_procedure" in md
        assert "First summary line of the procedure." in md
        # ...but the full body's later lines do NOT.
        assert "DEEP_BODY_TOKEN" not in md
        assert long_body not in md
        # Index instructs the model how to retrieve detail.
        assert "memory_get" in md and "memory_search" in md
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_index_mode_false_inlines_full_file(tmp_path: Path):
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("k", "v")
        await store.flush_memory_md("a session summary")  # writes MEMORY.md
        ctx = await store.load_context(index_mode=False)
        # legacy mode returns the raw MEMORY.md (has the file header)
        assert "# Memory:" in ctx.agent_memory_md
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_session_history_cap(tmp_path: Path):
    store = make_store(tmp_path)
    await store.open()
    try:
        # 5 sessions => 5 prepended history entries (newest first).
        for i in range(5):
            await store.flush_memory_md(f"session number {i}")
        ctx = await store.load_context(index_mode=True, max_session_history=2)
        md = ctx.agent_memory_md
        # Only the 2 most-recent entries inline (4 and 3); 0 must be excluded.
        assert "session number 4" in md
        assert "session number 3" in md
        assert "session number 0" not in md
        # The populated path renders the section header (the twin of the empty test below).
        assert "Recent Session History" in md
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_session_history_section_absent_when_empty(tmp_path: Path):
    """v2.0 audit FINDING-A: no dead promises in the injected block — with zero recorded
    sessions the index omits the 'Recent Session History' section entirely (no header,
    no '(no sessions recorded)' placeholder). It self-restores once any history entry
    exists (see test_session_history_cap's header assert)."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("k", "v")
        md = (await store.load_context(index_mode=True)).agent_memory_md
        assert "Recent Session History" not in md
        assert "(no sessions recorded)" not in md
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_memory_get_returns_full_body(tmp_path: Path):
    store = make_store(tmp_path)
    await store.open()
    try:
        body = "The full multi-line\nbody of the fact." * 10
        await store.store_fact("big_fact", body)
        tool = MemoryGetTool(store)
        res = await tool.run(name="big_fact")
        assert res.success
        assert res.output == body
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_memory_get_missing(tmp_path: Path):
    store = make_store(tmp_path)
    await store.open()
    try:
        tool = MemoryGetTool(store)
        res = await tool.run(name="nope")
        assert not res.success
        assert res.error_type == "not_found"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_memory_search_finds_seeded_fact(tmp_path: Path):
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("recipe_key", "banana smoothie recipe with honey")
        await store.store_fact("car_key", "car maintenance schedule")
        tool = MemorySearchTool(store)
        res = await tool.run(query="smoothie")
        assert res.success
        assert "recipe_key" in res.output
        assert "car_key" not in res.output
    finally:
        await store.close()
