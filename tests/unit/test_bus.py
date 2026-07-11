"""Tests for localharness.core.bus — EventBus publish/subscribe/replay/wait_for/history."""
import asyncio
import json
import sys
from pathlib import Path

import pytest

from localharness.core.bus import EventBus, SubscriptionHandle
from localharness.core.events import Action, Heartbeat, Observation, UserMessage
from localharness.core.types import AgentID, EventSeq, SessionID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action(agent_id: str = "a", session_id: str = "s", action_type: str = "tool_call") -> Action:
    return Action(agent_id=AgentID(agent_id), session_id=SessionID(session_id), action_type=action_type)


def _obs(agent_id: str = "a", session_id: str = "s") -> Observation:
    return Observation(agent_id=AgentID(agent_id), session_id=SessionID(session_id), observation_type="tool_result")


# ---------------------------------------------------------------------------
# Sequence and immutability
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_publish_assigns_seq(bus: EventBus):
    e1 = await bus.publish(_action())
    e2 = await bus.publish(_action())
    assert e1.seq == 0
    assert e2.seq == 1


@pytest.mark.asyncio
async def test_publish_immutable(bus: EventBus):
    original = _action()
    assert original.seq is None
    published = await bus.publish(original)
    assert original.seq is None  # original unchanged
    assert published.seq == 0


# ---------------------------------------------------------------------------
# Subscribe and decorator on()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subscribe_receives_events(bus: EventBus):
    received = []

    async def handler(event: Action):
        received.append(event)

    bus.subscribe(Action, handler)
    await bus.publish(_action())
    await bus.publish(_action())
    assert len(received) == 2


@pytest.mark.asyncio
async def test_decorator_on_receives_events(bus: EventBus):
    received = []

    @bus.on(Action)
    async def handler(event: Action):
        received.append(event)

    await bus.publish(_action())
    assert len(received) == 1


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_filter_by_agent_id(bus: EventBus):
    received = []

    async def handler(event: Action):
        received.append(event)

    bus.subscribe(Action, handler, agent_id=AgentID("agent-a"))
    await bus.publish(_action(agent_id="agent-a"))  # should receive
    await bus.publish(_action(agent_id="agent-b"))  # should NOT receive
    assert len(received) == 1
    assert received[0].agent_id == "agent-a"


@pytest.mark.asyncio
async def test_filter_by_session_id(bus: EventBus):
    received = []

    async def handler(event: Action):
        received.append(event)

    bus.subscribe(Action, handler, session_id=SessionID("session-x"))
    await bus.publish(_action(session_id="session-x"))  # should receive
    await bus.publish(_action(session_id="session-y"))  # should NOT receive
    assert len(received) == 1
    assert received[0].session_id == "session-x"


# ---------------------------------------------------------------------------
# FIFO ordering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fifo_ordering(bus: EventBus):
    received_seqs = []

    async def handler(event: Action):
        received_seqs.append(event.seq)

    bus.subscribe(Action, handler)
    for _ in range(5):
        await bus.publish(_action())

    assert received_seqs == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Exception isolation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subscriber_exception_isolated(bus: EventBus):
    received_b = []

    async def handler_a(event: Action):
        raise RuntimeError("handler A explodes")

    async def handler_b(event: Action):
        received_b.append(event)

    bus.subscribe(Action, handler_a)
    bus.subscribe(Action, handler_b)
    await bus.publish(_action())
    # handler_b should still receive despite handler_a raising
    assert len(received_b) == 1


# ---------------------------------------------------------------------------
# Sync (plain def) handlers — first-class dispatch, no await-None TypeError
# ---------------------------------------------------------------------------

class _LogSpy:
    """Records the bus's error-isolation calls (log.exception) with the active exception."""

    def __init__(self):
        self.exceptions: list[tuple[str, BaseException | None]] = []

    def exception(self, event_name, **kw):
        self.exceptions.append((event_name, sys.exc_info()[1]))

    def __getattr__(self, name):  # warning/error/info/debug — ignore
        return lambda *a, **kw: None


@pytest.mark.asyncio
async def test_sync_handler_dispatches_without_typeerror(bus: EventBus, monkeypatch):
    """A plain `def` handler must be first-class: its side effect runs and NO
    await-None TypeError is absorbed into the subscriber_error isolation log.
    (Live bug: subscribe()'s `filtered` wrapper does `await handler(event)`
    unconditionally — a sync handler returns None -> `await None` -> TypeError
    on every real event, masking real errors in the publishing turn.)"""
    spy = _LogSpy()
    monkeypatch.setattr("localharness.core.bus.log", spy)
    received = []

    def sync_handler(event: Action):  # deliberately NOT async
        received.append(event)

    bus.subscribe(Action, sync_handler)
    await bus.publish(_action())

    assert len(received) == 1
    assert spy.exceptions == []


@pytest.mark.asyncio
async def test_sync_and_async_handlers_coexist(bus: EventBus):
    """Async handlers are still genuinely awaited when sync handlers share the event."""
    got_sync, got_async = [], []

    def sync_handler(event: Action):
        got_sync.append(event)

    async def async_handler(event: Action):
        await asyncio.sleep(0)  # real suspension point — must be awaited to land
        got_async.append(event)

    bus.subscribe(Action, sync_handler)
    bus.subscribe(Action, async_handler)
    await bus.publish(_action())

    assert len(got_sync) == 1
    assert len(got_async) == 1


@pytest.mark.asyncio
async def test_sync_handler_own_exception_still_isolated_not_swallowed(bus: EventBus, monkeypatch):
    """A RAISING sync handler's own error still flows into the standard
    subscriber_error isolation (fixing await-None must not swallow handler
    errors), and other subscribers still receive the event."""
    spy = _LogSpy()
    monkeypatch.setattr("localharness.core.bus.log", spy)
    received_b = []

    def bad_sync_handler(event: Action):
        raise ValueError("sync boom")

    async def handler_b(event: Action):
        received_b.append(event)

    bus.subscribe(Action, bad_sync_handler)
    bus.subscribe(Action, handler_b)
    await bus.publish(_action())

    assert len(received_b) == 1
    assert len(spy.exceptions) == 1
    _, exc = spy.exceptions[0]
    assert isinstance(exc, ValueError)  # its OWN error — not an await-None TypeError
    assert "sync boom" in str(exc)


# ---------------------------------------------------------------------------
# wait_for
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wait_for_returns_matching(bus: EventBus):
    async def publisher():
        await asyncio.sleep(0.01)
        await bus.publish(_action(action_type="finish"))

    task = asyncio.create_task(publisher())
    result = await bus.wait_for(Action, timeout=5.0, predicate=lambda e: e.action_type == "finish")
    await task
    assert result.action_type == "finish"


@pytest.mark.asyncio
async def test_wait_for_timeout(bus: EventBus):
    with pytest.raises(asyncio.TimeoutError):
        await bus.wait_for(Action, timeout=0.05)


# ---------------------------------------------------------------------------
# Double-publish guard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_double_publish_raises(bus: EventBus):
    published = await bus.publish(_action())
    with pytest.raises(ValueError, match="already published"):
        await bus.publish(published)  # seq is already set


# ---------------------------------------------------------------------------
# Unsubscribe
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery(bus: EventBus):
    received = []

    async def handler(event: Action):
        received.append(event)

    handle = bus.subscribe(Action, handler)
    await bus.publish(_action())
    assert len(received) == 1
    bus.unsubscribe(handle)
    await bus.publish(_action())
    assert len(received) == 1  # no new events after unsubscribe


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_history_in_memory(bus: EventBus):
    e1 = await bus.publish(_action())
    e2 = await bus.publish(_obs())
    hist = bus.history()
    assert len(hist) == 2
    assert hist[0].seq == 0
    assert hist[1].seq == 1


@pytest.mark.asyncio
async def test_history_filter(bus: EventBus):
    await bus.publish(_action())
    await bus.publish(_obs())
    await bus.publish(_action())
    actions = bus.history(event_types=[Action])
    assert len(actions) == 2
    assert all(isinstance(e, Action) for e in actions)


# ---------------------------------------------------------------------------
# JSONL persistence — replay
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_replay_from_jsonl(bus_with_persistence: EventBus, tmp_path: Path):
    """Write 3 events with persist_path, create new bus, replay yields same 3 events."""
    a1 = await bus_with_persistence.publish(_action(action_type="llm_response"))
    a2 = await bus_with_persistence.publish(_action(action_type="tool_call"))
    a3 = await bus_with_persistence.publish(_obs())

    # Create new bus pointing to same file
    persist_path = tmp_path / "events.jsonl"
    new_bus = EventBus(persist_path=persist_path)
    replayed = []
    async for event in new_bus.replay():
        replayed.append(event)

    assert len(replayed) == 3
    assert replayed[0].id == a1.id
    assert replayed[1].id == a2.id
    assert replayed[2].id == a3.id


@pytest.mark.asyncio
async def test_replay_partial_line_skipped(tmp_path: Path):
    """Truncated last line in JSONL does not raise, earlier events still returned."""
    persist_path = tmp_path / "events.jsonl"
    bus = EventBus(persist_path=persist_path)
    a1 = await bus.publish(_action())
    a2 = await bus.publish(_action())

    # Append a partial/corrupt JSON line
    with open(persist_path, "a") as f:
        f.write('{"event_type": "Action", "id": "partial\n')  # truncated

    new_bus = EventBus(persist_path=persist_path)
    replayed = []
    async for event in new_bus.replay():
        replayed.append(event)

    assert len(replayed) == 2  # partial line skipped
    assert replayed[0].id == a1.id
    assert replayed[1].id == a2.id


# ---------------------------------------------------------------------------
# Per-session JSONL
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_per_session_jsonl(tmp_path: Path):
    """Events with different session_ids written to separate sessions/{session_id}.jsonl files."""
    persist_path = tmp_path / "events.jsonl"
    bus = EventBus(persist_path=persist_path)

    sess1 = SessionID("session-1")
    sess2 = SessionID("session-2")
    await bus.publish(Action(agent_id=AgentID("a"), session_id=sess1, action_type="tool_call"))
    await bus.publish(Action(agent_id=AgentID("a"), session_id=sess2, action_type="llm_response"))

    sess1_file = tmp_path / "sessions" / "session-1.jsonl"
    sess2_file = tmp_path / "sessions" / "session-2.jsonl"
    assert sess1_file.exists(), "Per-session file for session-1 should exist"
    assert sess2_file.exists(), "Per-session file for session-2 should exist"

    # Verify content
    lines1 = [l for l in sess1_file.read_text().splitlines() if l.strip()]
    lines2 = [l for l in sess2_file.read_text().splitlines() if l.strip()]
    assert len(lines1) == 1
    assert len(lines2) == 1
    data1 = json.loads(lines1[0])
    data2 = json.loads(lines2[0])
    assert data1["session_id"] == "session-1"
    assert data2["session_id"] == "session-2"


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_event_count_property(bus: EventBus):
    assert bus.event_count == 0
    await bus.publish(_action())
    assert bus.event_count == 1
    await bus.publish(_action())
    assert bus.event_count == 2


@pytest.mark.asyncio
async def test_subscriber_count_property(bus: EventBus):
    assert bus.subscriber_count == 0

    async def handler1(event): pass
    async def handler2(event): pass

    h1 = bus.subscribe(Action, handler1)
    assert bus.subscriber_count == 1
    h2 = bus.subscribe(Observation, handler2)
    assert bus.subscriber_count == 2
    bus.unsubscribe(h1)
    assert bus.subscriber_count == 1
