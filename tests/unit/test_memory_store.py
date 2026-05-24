"""Tests for MemoryStore: SQLite facts, sessions, FTS5, bus integration."""
import time
import uuid
from pathlib import Path

import pytest

from localharness.core.bus import EventBus
from localharness.core.events import Action, Observation, UserMessage
from localharness.memory.errors import SessionNotFoundError
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
        assert row[0] == 1
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
    fact1 = await memory_store.store_fact("key1", "v1")
    created_at = fact1.created_at

    fact2 = await memory_store.store_fact("key1", "v2")
    assert fact2.value == "v2"
    assert fact2.created_at == created_at  # created_at preserved


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


# ---------------------------------------------------------------------------
# reconstruct_session
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reconstruct_session_basic(memory_store: MemoryStore):
    sid = new_session_id()
    ts = int(time.time())

    await memory_store.append_history({
        "v": 1, "type": "system_message", "id": str(uuid.uuid4()),
        "session_id": sid, "agent_id": "test-agent", "ts": ts,
        "role": "system", "content": "You are a test agent.",
        "is_compacted": False, "replaces_ids": [],
    })
    await memory_store.append_history({
        "v": 1, "type": "user_message", "id": str(uuid.uuid4()),
        "session_id": sid, "agent_id": "test-agent", "ts": ts + 1,
        "role": "user", "content": "Hello agent", "channel": "terminal",
        "channel_metadata": None,
    })
    call_id = str(uuid.uuid4())
    await memory_store.append_history({
        "v": 1, "type": "assistant_message", "id": str(uuid.uuid4()),
        "session_id": sid, "agent_id": "test-agent", "ts": ts + 2,
        "role": "assistant", "content": "Using a tool.",
        "tool_calls": [{"id": call_id, "name": "some_tool", "arguments": {}}],
        "finish_reason": "tool_calls", "tokens_in": 100, "tokens_out": 20,
        "model": "test-model", "latency_ms": 100,
    })
    await memory_store.append_history({
        "v": 1, "type": "tool_result", "id": str(uuid.uuid4()),
        "session_id": sid, "agent_id": "test-agent", "ts": ts + 3,
        "role": "tool", "call_id": call_id, "tool_name": "some_tool",
        "content": "Tool result here.", "is_error": False, "error_type": None,
        "truncated": False, "original_length": 16, "stored_length": 16,
    })

    messages = await memory_store.reconstruct_session(sid)
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "tool"]


@pytest.mark.asyncio
async def test_reconstruct_session_orphan_guard(memory_store: MemoryStore):
    sid = new_session_id()
    ts = int(time.time())
    orphan_call_id = str(uuid.uuid4())

    await memory_store.append_history({
        "v": 1, "type": "assistant_message", "id": str(uuid.uuid4()),
        "session_id": sid, "agent_id": "test-agent", "ts": ts,
        "role": "assistant", "content": "Using a tool.",
        "tool_calls": [{"id": orphan_call_id, "name": "tool", "arguments": {}}],
        "finish_reason": "tool_calls", "tokens_in": 100, "tokens_out": 20,
        "model": "test-model", "latency_ms": 100,
    })
    # No matching tool_result — orphaned tool call

    messages = await memory_store.reconstruct_session(sid)
    # The assistant message with unmatched tool_calls should be dropped
    for m in messages:
        assert m.get("role") != "assistant" or not m.get("tool_calls")


@pytest.mark.asyncio
async def test_reconstruct_session_not_found(memory_store: MemoryStore):
    with pytest.raises(SessionNotFoundError):
        await memory_store.reconstruct_session("nonexistent-session-id")


# ---------------------------------------------------------------------------
# notes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_notes(tmp_path: Path):
    store = make_store(tmp_path)
    await store.open()
    try:
        # Create initial MEMORY.md via flush
        await store.flush_memory_md()
        await store.update_notes("working_notes", "new working note content")
        md = store._markdown_memory.read()
        assert "new working note content" in md
    finally:
        await store.close()


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
