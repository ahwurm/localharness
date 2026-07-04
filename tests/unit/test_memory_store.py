"""Tests for MemoryStore: SQLite facts, sessions, FTS5, bus integration."""
import time
import uuid
from pathlib import Path

import pytest

from localharness.core.bus import EventBus
from localharness.core.events import Action, Observation, UserMessage
from localharness.memory.sqlite import Fact, FactQuery, MemoryContext, MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_store(tmp_path: Path, bus=None) -> MemoryStore:
    return MemoryStore(
        agent_id="test-agent",
        division_id="test-div",
        org_id="default",
        base_dir=str(tmp_path),
        bus=bus,
    )


def new_session_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# open / directory / migration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_open_creates_db(tmp_path: Path):
    store = make_store(tmp_path)
    await store.open()
    try:
        db_path = tmp_path / "agents" / "test-agent" / "memory.db"
        assert db_path.exists()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_open_creates_directories(tmp_path: Path):
    store = make_store(tmp_path)
    await store.open()
    try:
        agent_dir = tmp_path / "agents" / "test-agent"
        assert agent_dir.is_dir()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_open_applies_migration(tmp_path: Path):
    store = make_store(tmp_path)
    await store.open()
    try:
        async with store._db.execute("PRAGMA user_version") as cur:
            row = await cur.fetchone()
        from localharness.memory.sqlite import CURRENT_SCHEMA_VERSION
        assert row[0] == CURRENT_SCHEMA_VERSION
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# store_fact / get_fact
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_store_fact_insert(memory_store: MemoryStore):
    fact = await memory_store.store_fact("key1", "value1")
    assert isinstance(fact, Fact)
    assert fact.key == "key1"
    assert fact.value == "value1"

    retrieved = await memory_store.get_fact("key1")
    assert retrieved is not None
    assert retrieved.value == "value1"


@pytest.mark.asyncio
async def test_store_fact_upsert(memory_store: MemoryStore):
    """v2 semantics (WRITE-02): a different value SUPERSEDES — new active row, old row
    kept as history. The v1 destructive in-place upsert is gone by design."""
    await memory_store.store_fact("key1", "v1")

    fact2 = await memory_store.store_fact("key1", "v2")
    assert fact2.value == "v2"
    assert fact2.status == "active"

    history = await memory_store.get_fact_history("key1")
    assert [f.value for f in history[:2]] == ["v2", "v1"] or {f.value for f in history} == {"v1", "v2"}
    old = next(f for f in history if f.value == "v1")
    assert old.status == "superseded"
    assert old.superseded_by == fact2.id


@pytest.mark.asyncio
async def test_store_fact_with_tags(memory_store: MemoryStore):
    await memory_store.store_fact("tagged_key", "val", tags=["tag1", "tag2"])
    retrieved = await memory_store.get_fact("tagged_key")
    assert retrieved is not None
    assert "tag1" in retrieved.tags
    assert "tag2" in retrieved.tags


@pytest.mark.asyncio
async def test_store_fact_with_confidence(memory_store: MemoryStore):
    await memory_store.store_fact("conf_key", "val", confidence=0.8)
    retrieved = await memory_store.get_fact("conf_key")
    assert retrieved is not None
    assert abs(retrieved.confidence - 0.8) < 0.001


@pytest.mark.asyncio
async def test_store_fact_with_expiry(memory_store: MemoryStore):
    past = int(time.time()) - 10  # already expired
    await memory_store.store_fact("expired_key", "val", expires_at=past)
    retrieved = await memory_store.get_fact("expired_key")
    assert retrieved is None


@pytest.mark.asyncio
async def test_store_fact_invalid_confidence(memory_store: MemoryStore):
    with pytest.raises(ValueError):
        await memory_store.store_fact("k", "v", confidence=1.5)

    with pytest.raises(ValueError):
        await memory_store.store_fact("k", "v", confidence=-0.1)


@pytest.mark.asyncio
async def test_get_fact_missing(memory_store: MemoryStore):
    result = await memory_store.get_fact("nonexistent")
    assert result is None


# ---------------------------------------------------------------------------
# delete_fact
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_fact(memory_store: MemoryStore):
    await memory_store.store_fact("del_key", "val")
    deleted = await memory_store.delete_fact("del_key")
    assert deleted is True
    assert await memory_store.get_fact("del_key") is None


@pytest.mark.asyncio
async def test_delete_fact_missing(memory_store: MemoryStore):
    result = await memory_store.delete_fact("nonexistent")
    assert result is False


# ---------------------------------------------------------------------------
# query_facts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_query_facts_by_text(memory_store: MemoryStore):
    await memory_store.store_fact("alpha_key", "banana smoothie recipe")
    await memory_store.store_fact("beta_key", "car maintenance schedule")
    results = await memory_store.query_facts(FactQuery(text="banana"))
    keys = [f.key for f in results]
    assert "alpha_key" in keys
    assert "beta_key" not in keys


@pytest.mark.asyncio
async def test_query_facts_by_tags(memory_store: MemoryStore):
    await memory_store.store_fact("t1", "v1", tags=["food"])
    await memory_store.store_fact("t2", "v2", tags=["transport"])
    results = await memory_store.query_facts(FactQuery(tags=["food"]))
    keys = [f.key for f in results]
    assert "t1" in keys
    assert "t2" not in keys


@pytest.mark.asyncio
async def test_query_facts_excludes_expired(memory_store: MemoryStore):
    past = int(time.time()) - 10
    future = int(time.time()) + 3600
    await memory_store.store_fact("expired", "old", expires_at=past)
    await memory_store.store_fact("active", "new", expires_at=future)
    results = await memory_store.query_facts(FactQuery())
    keys = [f.key for f in results]
    assert "expired" not in keys
    assert "active" in keys


@pytest.mark.asyncio
async def test_query_facts_min_confidence(memory_store: MemoryStore):
    await memory_store.store_fact("high", "val", confidence=0.9)
    await memory_store.store_fact("low", "val", confidence=0.3)
    results = await memory_store.query_facts(FactQuery(min_confidence=0.5))
    keys = [f.key for f in results]
    assert "high" in keys
    assert "low" not in keys


@pytest.mark.asyncio
async def test_query_facts_hyphenated_text(memory_store: MemoryStore):
    """Regression: raw hyphens leaked FTS5 column syntax — 'built-in subagents' raised
    'no such column: in' (memory_search exit 1, observed live 2026-07-02)."""
    await memory_store.store_fact("subagents", "the built-in subagents ship with the harness")
    results = await memory_store.query_facts(FactQuery(text="built-in subagents"))
    assert [f.key for f in results] == ["subagents"]


@pytest.mark.asyncio
async def test_query_facts_fts_operator_chars(memory_store: MemoryStore):
    """FTS5 operator/quote characters in user text must never raise."""
    await memory_store.store_fact("k1", "plain value")
    for text in ('col:filter', 'a AND (b', 'quo"te', 'NEAR(x, y)', 'trailing-'):
        await memory_store.query_facts(FactQuery(text=text))  # must not raise


@pytest.mark.asyncio
async def test_query_facts_punctuation_only_text(memory_store: MemoryStore):
    """Nothing searchable in the query → empty result, not an FTS error or a full scan."""
    await memory_store.store_fact("k1", "plain value")
    assert await memory_store.query_facts(FactQuery(text="--- ::: !!")) == []


# ---------------------------------------------------------------------------
# persistence across close/reopen
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fact_persistence(tmp_path: Path):
    store1 = make_store(tmp_path)
    await store1.open()
    await store1.store_fact("persist_key", "persist_value")
    await store1.close()

    store2 = make_store(tmp_path)
    await store2.open()
    try:
        fact = await store2.get_fact("persist_key")
        assert fact is not None
        assert fact.value == "persist_value"
    finally:
        await store2.close()


# ---------------------------------------------------------------------------
# session management
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_session(memory_store: MemoryStore):
    sid = new_session_id()
    await memory_store.create_session(sid, {"max_actions": 100}, "test-model", 128000)

    async with memory_store._db.execute(
        "SELECT id FROM sessions WHERE id = ?", (sid,)
    ) as cur:
        row = await cur.fetchone()
    assert row is not None

    records = await memory_store.get_history(session_id=sid)
    types = [r["type"] for r in records]
    assert "session_event" in types


@pytest.mark.asyncio
async def test_end_session(memory_store: MemoryStore):
    sid = new_session_id()
    await memory_store.create_session(sid, {}, "m", 0)
    await memory_store.end_session(sid, "complete", "did some work", 3, 5, 1000, 500)

    async with memory_store._db.execute(
        "SELECT ended_at, exit_reason, summary FROM sessions WHERE id = ?", (sid,)
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    assert row[0] is not None  # ended_at set
    assert row[1] == "complete"
    assert row[2] == "did some work"

    records = await memory_store.get_history(session_id=sid)
    types = [r["type"] for r in records]
    assert types.count("session_event") >= 2  # start + end


@pytest.mark.asyncio
async def test_session_resume(tmp_path: Path):
    store = make_store(tmp_path)
    await store.open()
    sid = new_session_id()
    await store.create_session(sid, {}, "m", 0)
    await store.store_fact("resume_key", "resume_value")
    await store.end_session(sid, "complete", "session summary", 1, 1, 100, 50)
    await store.close()

    store2 = make_store(tmp_path)
    await store2.open()
    try:
        fact = await store2.get_fact("resume_key")
        assert fact is not None
        assert fact.value == "resume_value"
        ctx = await store2.load_context()
        # MEMORY.md should have been flushed and now contains content
        assert isinstance(ctx.agent_memory_md, str)
    finally:
        await store2.close()


# ---------------------------------------------------------------------------
# load_context
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_load_context_agent_only(memory_store: MemoryStore):
    ctx = await memory_store.load_context()
    assert isinstance(ctx, MemoryContext)
    assert isinstance(ctx.agent_memory_md, str)
    assert ctx.division_md == ""
    assert ctx.guardrails_md == ""


@pytest.mark.asyncio
async def test_load_context_with_division(tmp_path: Path):
    # Create DIVISION.md
    div_dir = tmp_path / "divisions" / "test-div"
    div_dir.mkdir(parents=True)
    (div_dir / "DIVISION.md").write_text("# Division Notes\nSome division content.", encoding="utf-8")

    store = make_store(tmp_path)
    await store.open()
    try:
        ctx = await store.load_context()
        assert "Division Notes" in ctx.division_md or "Some division" in ctx.division_md
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_load_context_with_org(tmp_path: Path):
    # Create GUARDRAILS.md
    org_dir = tmp_path / "orgs" / "default"
    org_dir.mkdir(parents=True)
    (org_dir / "GUARDRAILS.md").write_text("# Guardrails\nDo not harm.", encoding="utf-8")

    store = make_store(tmp_path)
    await store.open()
    try:
        ctx = await store.load_context()
        assert "Guardrails" in ctx.guardrails_md or "Do not harm" in ctx.guardrails_md
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# history delegation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_append_history_delegates(memory_store: MemoryStore):
    sid = new_session_id()
    record = {
        "v": 1,
        "type": "user_message",
        "id": str(uuid.uuid4()),
        "session_id": sid,
        "agent_id": "test-agent",
        "ts": int(time.time()),
        "role": "user",
        "content": "hello",
        "channel": "terminal",
        "channel_metadata": None,
    }
    await memory_store.append_history(record)
    records = await memory_store.get_history(session_id=sid)
    assert len(records) == 1
    assert records[0]["content"] == "hello"


@pytest.mark.asyncio
async def test_get_history_delegates(memory_store: MemoryStore):
    sid1, sid2 = new_session_id(), new_session_id()
    for i, sid in enumerate([sid1, sid2]):
        record = {
            "v": 1,
            "type": "user_message",
            "id": str(uuid.uuid4()),
            "session_id": sid,
            "agent_id": "test-agent",
            "ts": int(time.time()),
            "role": "user",
            "content": f"msg-{i}",
            "channel": "terminal",
            "channel_metadata": None,
        }
        await memory_store.append_history(record)

    s1_records = await memory_store.get_history(session_id=sid1)
    assert len(s1_records) == 1
    assert s1_records[0]["content"] == "msg-0"


@pytest.mark.asyncio
async def test_flush_memory_md(tmp_path: Path):
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact("f1", "fact value one", confidence=0.9)
        await store.flush_memory_md(session_summary="Did some work today")
        md = store._markdown_memory.read()
        assert "fact value one" in md
        assert "Did some work today" in md
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# TIME-02/04: pure relative-time label helpers. The day-flip seam is dependency
# injection (a plain `today` argument) — zero clock mocking, freezegun is NOT a dep.
# ---------------------------------------------------------------------------

def test_relative_day_label_boundaries():
    """Day-delta boundaries for the injected shelf's relative day word. `today` is a plain
    argument (no clock read), so the flip is provable without any mocking library."""
    from datetime import date

    from localharness.memory.sqlite import _relative_day_label

    today = date(2026, 7, 4)
    assert _relative_day_label(date(2026, 7, 4), today) == "today"      # delta 0
    assert _relative_day_label(date(2026, 7, 3), today) == "yesterday"  # delta 1
    # delta 2 and 6 → the sitting's own weekday abbrev (locale-immune: compare to strftime)
    assert _relative_day_label(date(2026, 7, 2), today) == date(2026, 7, 2).strftime("%a")
    assert _relative_day_label(date(2026, 6, 28), today) == date(2026, 6, 28).strftime("%a")
    # delta 7 → absolute ISO fallback
    assert _relative_day_label(date(2026, 6, 27), today) == "2026-06-27"
    # delta -1 (clock skew: sitting "tomorrow") → ISO fallback, NEVER "today"
    skewed = _relative_day_label(date(2026, 7, 5), today)
    assert skewed == "2026-07-05" and skewed != "today"


def test_clock_label_edges():
    """12-hour clock, no leading zero, portable — %I is 01-12 so lstrip('0') only ever
    strips the hour's leading zero, never the zero-padded minutes."""
    from datetime import datetime

    from localharness.memory.sqlite import _clock_label

    assert _clock_label(datetime(2026, 7, 4, 9, 5)) == "9:05am"
    assert _clock_label(datetime(2026, 7, 4, 0, 5)) == "12:05am"   # midnight
    assert _clock_label(datetime(2026, 7, 4, 12, 5)) == "12:05pm"  # noon
    assert _clock_label(datetime(2026, 7, 4, 23, 59)) == "11:59pm"


# ---------------------------------------------------------------------------
# session-history render contract (SESS-05): payload-first, 180-char, placeholder-proof
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_history_renders_after_real_end_session(tmp_path: Path):
    """A real end_session summary renders in the injected index's history shelf — the
    1fbdf6b suppression self-restores once one real entry exists."""
    store = make_store(tmp_path)
    await store.open()
    try:
        sid = new_session_id()
        await store.create_session(sid, {}, "m", 8192)
        await store.end_session(
            sid, "complete",
            "resolved: uv: command not found; 5 turns, 12 tool calls (bash_exec, read)",
            5, 12, 1000, 200,
        )
        md = (await store.load_context(index_mode=True, max_session_history=10)).agent_memory_md
        assert "### Recent Session History" in md
        assert "uv: command not found" in md
        async with store._db.execute(
            "SELECT ended_at FROM sessions WHERE id = ?", (sid,)
        ) as cur:
            row = await cur.fetchone()
        assert row is not None and row[0] is not None  # ended_at NOT NULL
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_vacuous_summary_keeps_shelf_suppressed(tmp_path: Path):
    """summary=None must not un-suppress the shelf, and must never write a placeholder the
    render could pick up — empty-means-empty (the 1fbdf6b contract)."""
    store = make_store(tmp_path)
    await store.open()
    try:
        sid = new_session_id()
        await store.create_session(sid, {}, "m", 0)
        await store.end_session(sid, "complete", None, 0, 0, 0, 0)
        md = (await store.load_context(index_mode=True)).agent_memory_md
        assert "Recent Session History" not in md
        # MEMORY.md exists but its Session History section carries no entry line.
        assert store._markdown_memory.exists()
        history = store._markdown_memory.get_section("session_history")
        assert not [ln for ln in history.splitlines() if ln.lstrip().startswith("- ")]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_placeholder_never_leaks_into_index(tmp_path: Path):
    """Render-time filter defeats a legacy on-disk placeholder: seed a pre-fix MEMORY.md
    whose Session History body is the literal placeholder, then a real end_session renders
    exactly ONE entry line and never surfaces the placeholder."""
    store = make_store(tmp_path)
    await store.open()
    try:
        # Simulate pre-fix on-disk state (a MEMORY.md written before this plan landed).
        store._markdown_memory._path.write_text(
            "# Memory: test-agent\n\n"
            "## Persistent Facts\n\n(No facts recorded yet.)\n\n"
            "## Working Notes\n\n(No working notes yet.)\n\n"
            "## Learned Behaviors\n\n(No learned behaviors yet.)\n\n"
            "## Session History\n\n(No sessions recorded yet.)\n",
            encoding="utf-8",
        )
        sid = new_session_id()
        await store.create_session(sid, {}, "m", 0)
        await store.end_session(sid, "complete", "resolved: real work this sitting", 1, 1, 10, 5)
        md = (await store.load_context(index_mode=True)).agent_memory_md
        assert "resolved: real work this sitting" in md
        # Exactly one entry line in the whole index (no facts stored → the entry is the
        # only '- ' line) and the legacy placeholder is filtered out at render time.
        dash_lines = [ln for ln in md.splitlines() if ln.lstrip().startswith("- ")]
        assert len(dash_lines) == 1
        assert "resolved: real work this sitting" in dash_lines[0]
        assert "(No sessions recorded" not in md
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_summary_survives_to_180_chars(tmp_path: Path):
    """The history line budget is 180 (5192f27), not the old [:120] guillotine: a marker
    past char 120 but within 180 survives into the rendered entry."""
    store = make_store(tmp_path)
    await store.open()
    try:
        summary = "X" * 140 + "MARKER-BEYOND-120" + "Y" * 63  # 220 chars; marker at 140..156
        assert len(summary) == 220
        sid = new_session_id()
        await store.create_session(sid, {}, "m", 0)
        await store.end_session(sid, "complete", summary, 1, 1, 0, 0)
        md = (await store.load_context(index_mode=True)).agent_memory_md
        assert "MARKER-BEYOND-120" in md  # the old [:120] cap would have cut it
        entry = next(ln for ln in md.splitlines() if "MARKER-BEYOND-120" in ln)
        payload = entry.split(": ", 1)[1]  # strip "- YYYY-MM-DD: "
        assert len(payload) <= 180
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_legacy_index_mode_renders_truth_after_end_session(tmp_path: Path):
    """LEGACY-INJECT rider (SESS-06): index_mode=False reads the WHOLE MEMORY.md — and
    since Phase 33's flush restored a live writer (end_session -> flush_memory_md ->
    regenerate), that whole-file read now reflects real data. A real fact AND a fresh
    session line both survive the legacy path — rider closed by proof, zero new code on
    the legacy branch."""
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact(
            "learned/bash_exec/resolved_error",
            "uv fix: use .venv/bin/python",
            confidence=0.9,  # >= 0.7 so the flush includes it
        )
        sid = new_session_id()
        await store.create_session(sid, {}, "m", 8192)
        await store.end_session(
            sid, "complete", "resolved: uv: command not found; 3 turns", 3, 4, 500, 100
        )
        ctx = await store.load_context(index_mode=False)
        assert "uv fix: use .venv/bin/python" in ctx.agent_memory_md  # facts survived
        assert "uv: command not found" in ctx.agent_memory_md  # fresh session line
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# SESS-04: the zero-tool "what did we do last sitting?" answer survives a
# process boundary (a fresh MemoryStore instance over the same base_dir)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fresh_sitting_answers_last_sitting_zero_tool(tmp_path: Path):
    """SESS-04 phase provable: a FRESH sitting (new MemoryStore instance = new process
    lifetime over one base_dir) answers 'what did we do last sitting?' from the injected
    block ALONE — zero tool calls. The disk-persisted MEMORY.md session_history is the
    cross-process seam: end_session flushes it (sitting N) -> a brand-new instance's
    load_context renders it (sitting N+1), which loop.py injects verbatim into the next
    system prompt. Two instances, sequential (first CLOSED before second opens)."""
    # Sitting N: a real close-out flushes a payload-first history line to MEMORY.md, then
    # the process ends (store_a.close()).
    store_a = MemoryStore(agent_id="test-agent", division_id="", org_id="",
                          base_dir=str(tmp_path))
    await store_a.open()
    await store_a.create_session("sit-1", {}, "dogfood", 8192)
    await store_a.end_session(
        "sit-1", exit_reason="complete",
        summary="resolved: uv: command not found; 5 turns, 12 tool calls (bash_exec, read)",
        turn_count=5, action_count=12, tokens_in=1000, tokens_out=200,
    )
    await store_a.close()

    # Sitting N+1: a NEW instance opens the SAME base_dir and renders the injected block.
    store_b = MemoryStore(agent_id="test-agent", division_id="", org_id="",
                          base_dir=str(tmp_path))
    await store_b.open()
    try:
        agent_md = (await store_b.load_context(index_mode=True)).agent_memory_md
        assert "### Recent Session History" in agent_md
        assert "uv: command not found" in agent_md  # the previous sitting's payload
    finally:
        await store_b.close()


# ---------------------------------------------------------------------------
# integrity_check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_integrity_check(memory_store: MemoryStore):
    errors = await memory_store.integrity_check()
    assert errors == []


# ---------------------------------------------------------------------------
# bus auto-diary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bus_auto_diary_action(tmp_path: Path):
    bus = EventBus()
    store = make_store(tmp_path, bus=bus)
    await store.open()
    try:
        sid = new_session_id()
        event = Action(
            agent_id="test-agent",
            session_id=sid,
            action_type="tool_call",
            tool_name="test_tool",
            tool_call_id="call_123",
            tool_params={"k": "v"},
        )
        await bus.publish(event)
        records = await store.get_history(session_id=sid)
        assert len(records) == 1
        assert records[0]["type"] == "assistant_message"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_bus_auto_diary_observation(tmp_path: Path):
    bus = EventBus()
    store = make_store(tmp_path, bus=bus)
    await store.open()
    try:
        sid = new_session_id()
        event = Observation(
            agent_id="test-agent",
            session_id=sid,
            observation_type="tool_result",
            tool_call_id="call_123",
            tool_name="test_tool",
            output="some output",
        )
        await bus.publish(event)
        records = await store.get_history(session_id=sid)
        assert len(records) == 1
        assert records[0]["type"] == "tool_result"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_bus_auto_diary_user_message(tmp_path: Path):
    bus = EventBus()
    store = make_store(tmp_path, bus=bus)
    await store.open()
    try:
        sid = new_session_id()
        event = UserMessage(
            agent_id="test-agent",
            session_id=sid,
            content="Hello from user",
            channel="terminal",
        )
        await bus.publish(event)
        records = await store.get_history(session_id=sid)
        assert len(records) == 1
        assert records[0]["type"] == "user_message"
        assert records[0]["content"] == "Hello from user"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_close_unsubscribes(tmp_path: Path):
    bus = EventBus()
    store = make_store(tmp_path, bus=bus)
    await store.open()
    # Verify subscription exists
    assert bus.subscriber_count > 0
    await store.close()
    # After close, subscriber_count should be 0 (all handles unsubscribed)
    assert bus.subscriber_count == 0


# ---------------------------------------------------------------------------
# Phase 33.1 (ORCH-02): one-time root rename default -> orchestrator
#
# The agent's name IS its storage identity (directory name + the agent_id column
# in facts/sessions). Renaming the root without migrating would orphan every
# memory an existing install has ("amnesia the same week it learned to
# remember"). These composed tests build a PRE-rename store under "default", then
# open the SAME base_dir as "orchestrator" (the migration trigger) and prove the
# old facts + session history are fully reachable, rows are RE-KEYED (not
# directory-aliased), re-opens are a clean no-op, non-root opens never touch the
# legacy dir, and a real user "orchestrator" is never merged or clobbered.
# ---------------------------------------------------------------------------

async def _build_legacy_default_store(tmp_path: Path) -> None:
    """A pre-rename root store: one high-confidence fact + one ended sitting, closed."""
    store = MemoryStore(agent_id="default", division_id="", org_id="",
                        base_dir=str(tmp_path))
    await store.open()
    await store.store_fact(
        "learned/bash_exec/resolved_error", "uv fix: use .venv/bin/python",
        confidence=0.9,
    )
    await store.create_session("sit-1", {}, "dogfood", 8192)
    await store.end_session(
        "sit-1", exit_reason="complete",
        summary="resolved: uv: command not found; 5 turns, 12 tool calls (bash_exec, read)",
        turn_count=5, action_count=12, tokens_in=1000, tokens_out=200,
    )
    await store.close()


@pytest.mark.asyncio
async def test_orchestrator_migration_pre_rename_store_fully_reachable(tmp_path: Path):
    """THE quality-gate composed test: a store built + closed under the OLD root name
    "default" is fully reachable after opening the SAME base_dir as "orchestrator" —
    facts AND session history render, rows are RE-KEYED (not just directory-aliased),
    the legacy dir is gone, the new dir holds memory.db, integrity stays clean, and an
    honest rename breadcrumb lands in history.jsonl."""
    await _build_legacy_default_store(tmp_path)

    # Simulated upgrade: open under the NEW root name over the SAME base_dir.
    store_b = MemoryStore(agent_id="orchestrator", division_id="", org_id="",
                          base_dir=str(tmp_path))
    await store_b.open()  # <- triggers the migration (dir adoption + SQL row fixup)
    try:
        # (a) facts + session history render under the new root name
        agent_md = (await store_b.load_context(index_mode=True)).agent_memory_md
        assert "uv fix: use .venv/bin/python" in agent_md      # old fact reachable
        assert "### Recent Session History" in agent_md
        assert "uv: command not found" in agent_md             # old session line reachable
        # (b) the row was RE-KEYED, not directory-aliased
        fact = await store_b.get_fact("learned/bash_exec/resolved_error")
        assert fact is not None and fact.agent_id == "orchestrator"
        # (c) directory adoption: legacy gone, new dir holds the db
        assert not (tmp_path / "agents" / "default").exists()
        assert (tmp_path / "agents" / "orchestrator" / "memory.db").exists()
        # (d) breadcrumb is schema-conformant — integrity stays clean
        assert await store_b.integrity_check() == []
        # (e) honest paper trail: the rename is recorded in history.jsonl
        assert "agent_renamed" in (
            tmp_path / "agents" / "orchestrator" / "history.jsonl"
        ).read_text()
    finally:
        await store_b.close()


@pytest.mark.asyncio
async def test_orchestrator_migration_idempotent_across_reopens(tmp_path: Path):
    """Second and later opens as "orchestrator" are a clean no-op: the dir-rename branch
    no-ops (legacy gone), the row UPDATE matches 0 rows, and EXACTLY ONE breadcrumb ever
    exists — the facts + session history still render."""
    await _build_legacy_default_store(tmp_path)

    store_b = MemoryStore(agent_id="orchestrator", division_id="", org_id="",
                          base_dir=str(tmp_path))
    await store_b.open()
    await store_b.close()

    store_c = MemoryStore(agent_id="orchestrator", division_id="", org_id="",
                          base_dir=str(tmp_path))
    await store_c.open()  # second open — must not raise, must not re-migrate
    try:
        agent_md = (await store_c.load_context(index_mode=True)).agent_memory_md
        assert "uv fix: use .venv/bin/python" in agent_md
        assert "uv: command not found" in agent_md
        fact = await store_c.get_fact("learned/bash_exec/resolved_error")
        assert fact is not None and fact.agent_id == "orchestrator"
        # exactly one rename breadcrumb — no re-append on the idempotent second open
        breadcrumbs = [
            ln for ln in (tmp_path / "agents" / "orchestrator" / "history.jsonl")
            .read_text().splitlines()
            if "agent_renamed" in ln
        ]
        assert len(breadcrumbs) == 1
    finally:
        await store_c.close()


@pytest.mark.asyncio
async def test_orchestrator_migration_noop_for_other_agents(tmp_path: Path):
    """The migration is scoped to the root: opening any OTHER agent_id never touches the
    legacy "default" directory, and the legacy store stays fully reachable under its old
    name (nothing moves for non-root opens)."""
    await _build_legacy_default_store(tmp_path)

    # A non-root agent opens over the same base_dir — legacy dir must be untouched.
    cruncher = MemoryStore(agent_id="cruncher", division_id="", org_id="",
                           base_dir=str(tmp_path))
    await cruncher.open()
    try:
        assert (tmp_path / "agents" / "default" / "memory.db").exists()  # untouched
        assert await cruncher.get_fact("learned/bash_exec/resolved_error") is None
    finally:
        await cruncher.close()

    # The legacy root still opens — and still holds its fact — under the OLD name.
    legacy = MemoryStore(agent_id="default", division_id="", org_id="",
                         base_dir=str(tmp_path))
    await legacy.open()
    try:
        fact = await legacy.get_fact("learned/bash_exec/resolved_error")
        assert fact is not None and fact.agent_id == "default"
    finally:
        await legacy.close()


# --- Collision refusal (ORCH-03): a user's OWN "orchestrator" is never merged/clobbered ---

async def _build_collision_stores(tmp_path: Path) -> None:
    """A user's OWN pre-existing "orchestrator" agent AND a legacy "default" root — both
    real, both closed. The migration must REFUSE to touch either (destination exists)."""
    theirs = MemoryStore(agent_id="orchestrator", division_id="", org_id="",
                         base_dir=str(tmp_path))
    await theirs.open()
    await theirs.store_fact("user/own/fact", "THEIRS-marker", confidence=0.9)
    await theirs.close()

    legacy = MemoryStore(agent_id="default", division_id="", org_id="",
                         base_dir=str(tmp_path))
    await legacy.open()
    await legacy.store_fact("learned/legacy/fact", "LEGACY-marker", confidence=0.9)
    await legacy.close()


@pytest.mark.asyncio
async def test_orchestrator_migration_refuses_collision_never_clobbers(tmp_path: Path):
    """Collision = refusal: when a real "orchestrator" agent already exists, opening it
    NEVER merges the legacy "default" data in and NEVER deletes the legacy dir. Their own
    fact stays intact, the legacy fact does not leak, nothing is clobbered, and no false
    rename breadcrumb is written on the refused migration."""
    await _build_collision_stores(tmp_path)

    theirs = MemoryStore(agent_id="orchestrator", division_id="", org_id="",
                         base_dir=str(tmp_path))
    await theirs.open()  # destination exists -> migration REFUSES
    try:
        assert await theirs.get_fact("user/own/fact") is not None      # their data intact
        assert await theirs.get_fact("learned/legacy/fact") is None    # NO merge/leak
        assert (tmp_path / "agents" / "default" / "memory.db").exists()  # nothing deleted
        # No false breadcrumb on a refused migration. store_fact writes no history, so the
        # orchestrator's history.jsonl may not exist at all — guard the read.
        hist = tmp_path / "agents" / "orchestrator" / "history.jsonl"
        assert (not hist.exists()) or ("agent_renamed" not in hist.read_text())
    finally:
        await theirs.close()


@pytest.mark.asyncio
async def test_orchestrator_migration_collision_legacy_still_opens_as_default(tmp_path: Path):
    """After a refused collision, the un-migrated legacy root keeps working under its OLD
    "default" name — its fact is still reachable and still renders (ORCH-03: keep working
    under the old name when the new name is already taken)."""
    await _build_collision_stores(tmp_path)

    # Refuse once (open+close the colliding orchestrator), then prove the legacy still opens.
    theirs = MemoryStore(agent_id="orchestrator", division_id="", org_id="",
                         base_dir=str(tmp_path))
    await theirs.open()
    await theirs.close()

    legacy = MemoryStore(agent_id="default", division_id="", org_id="",
                         base_dir=str(tmp_path))
    await legacy.open()
    try:
        fact = await legacy.get_fact("learned/legacy/fact")
        assert fact is not None and fact.agent_id == "default"
        agent_md = (await legacy.load_context(index_mode=True)).agent_memory_md
        assert "LEGACY-marker" in agent_md
    finally:
        await legacy.close()
