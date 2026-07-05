"""Feature 1: memory as a queryable handle.

- load_context(index_mode=True) renders an INDEX (fact names + one-line descriptions),
  NOT full bodies.
- session-history cap inlines only the last N entries.
- memory_get returns a fact's full body; memory_search finds a seeded fact (FTS5).
"""
import time
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

import pytest

from localharness.memory.sqlite import MemoryStore
from localharness.tools.builtin.memory_tools import (
    MemoryGetTool,
    MemorySearchTool,
    resolve_time_expr,
)


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
        # 5 REAL closed sittings with distinct explicit starts (i=4 newest, all today/
        # yesterday). The injected shelf now renders from the sessions TABLE, so seed real
        # rows and reach into store._db to set started_at — the established seed pattern
        # (cf. test_memory_store.py's session-history tests).
        base = int(time.time()) - 5 * 3600
        for i in range(5):
            sid = f"sit-{i}"
            await store.create_session(sid, {}, "m", 0)
            await store.end_session(sid, "complete", f"session number {i}", 1, 1, 0, 0)
            await store._db.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?", (base + i * 3600, sid)
            )
        await store._db.commit()
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


# ---------------------------------------------------------------------------
# 34-05: resolve_time_expr grammar (closed enum + ISO) → LOCAL epoch seconds
# ---------------------------------------------------------------------------

def _local_day_epoch(day: date, *, end: bool) -> int:
    """Independent reference: LOCAL start/end-of-day epoch (no call to the unit under test)."""
    boundary = dtime(23, 59, 59) if end else dtime.min
    return int(datetime.combine(day, boundary).astimezone().timestamp())


def test_resolve_today():
    today = datetime.now().astimezone().date()
    assert resolve_time_expr("today", end=False) == _local_day_epoch(today, end=False)
    assert resolve_time_expr("today", end=True) == _local_day_epoch(today, end=True)


def test_resolve_yesterday_window():
    yday = datetime.now().astimezone().date() - timedelta(days=1)
    start = resolve_time_expr("yesterday", end=False)
    finish = resolve_time_expr("yesterday", end=True)
    assert start == _local_day_epoch(yday, end=False)
    assert finish == _local_day_epoch(yday, end=True)
    assert finish - start == 86399  # brackets the whole local day 00:00:00..23:59:59


def test_resolve_this_week():
    today = datetime.now().astimezone().date()
    monday = today - timedelta(days=today.weekday())  # Monday itself → today 00:00
    assert resolve_time_expr("this_week", end=False) == _local_day_epoch(monday, end=False)


def test_resolve_iso_date_and_datetime():
    # bare date → local start-of-day (end=False) / end-of-day (end=True)
    assert resolve_time_expr("2026-07-01", end=False) == _local_day_epoch(date(2026, 7, 1), end=False)
    assert resolve_time_expr("2026-07-01", end=True) == _local_day_epoch(date(2026, 7, 1), end=True)
    # datetime precision → exact local time, end flag ignored (both ends identical)
    exact = int(datetime(2026, 7, 1, 9, 30).astimezone().timestamp())
    assert resolve_time_expr("2026-07-01T09:30", end=False) == exact
    assert resolve_time_expr("2026-07-01T09:30", end=True) == exact


def test_resolve_garbage():
    with pytest.raises(ValueError) as exc:
        resolve_time_expr("banana")
    msg = str(exc.value)
    assert "today|yesterday|this_week" in msg
    assert "ISO" in msg


@pytest.mark.asyncio
async def test_search_temporal_since_filters(tmp_path: Path):
    """memory_search(query, since='today') returns only facts touched today — 'what happened
    this morning?' is answerable by construction."""
    store = make_store(tmp_path)
    await store.open()
    try:
        now = int(time.time())
        await store.store_fact("fresh_note", "temporal search marker fresh")
        await store.store_fact("stale_note", "temporal search marker stale")
        await store._db.execute(
            "UPDATE facts SET updated_at = ? WHERE key = ?", (now - 2 * 86400, "stale_note")
        )
        await store._db.commit()
        tool = MemorySearchTool(store)
        res = await tool.run(query="marker", since="today")
        assert res.success
        assert "fresh_note" in res.output
        assert "stale_note" not in res.output
        res_all = await tool.run(query="marker")  # no filter → both
        assert "fresh_note" in res_all.output and "stale_note" in res_all.output
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_search_temporal_bad_expr_readable_error(tmp_path: Path):
    """must_have #4: an unparseable time expression is a readable tool error naming the
    accepted grammar — NEVER an exception into the loop."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("k", "some searchable value")
        tool = MemorySearchTool(store)
        res = await tool.run(query="value", since="banana")
        assert res.success is False
        blob = (res.error or "") + (res.output or "")
        assert "today|yesterday|this_week" in blob
        assert "ISO" in blob
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_search_temporal_params_optional(tmp_path: Path):
    """Plain memory_search (no since/until) is byte-unchanged — FactQuery gets since=None,
    until=None and behaves exactly as before the rider."""
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
