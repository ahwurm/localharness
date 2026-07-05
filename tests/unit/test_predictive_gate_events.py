"""COLL-04 live-contract tests for the collect-only PredictiveGate (Phase 34-03).

Task 1 (this section): the five-event correlated sequence per tool call
(Action -> ExpectationAttached -> Observation -> OutcomeObserved -> SurpriseScored)
plus the two persisted rows, and the inertness of non-tool / unmatched traffic.

Events are constructed EXACTLY as agent/loop.py:950 (the tool_call Action) and
loop.py:1009 (the tool_result Observation) publish them, so the tests exercise the
real wire shape. The subscriber mirrors WriteGate (memory/gate.py).
"""
import json
import logging
import statistics
import time
from pathlib import Path

from localharness.config.models import PredictiveGateConfig
from localharness.core.bus import EventBus
from localharness.core.events import (
    Action,
    ExpectationAttached,
    Observation,
    OutcomeObserved,
    SurpriseScored,
    TurnCompleted,
    UserMessage,
)
from localharness.memory.predictive_gate import PredictiveGate
from localharness.memory.sqlite import MemoryStore

AGENT = "test-agent"
SESSION = "sitting-1"


def make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        agent_id=AGENT, division_id="test-div", org_id="default",
        base_dir=str(tmp_path), bus=None,
    )


def action(tool_call_id: str, tool_name: str = "bash_exec", *, agent_id: str = AGENT) -> Action:
    """A tool_call Action, byte-shaped like agent/loop.py:950."""
    return Action(
        agent_id=agent_id, session_id=SESSION, action_type="tool_call",
        tool_call_id=tool_call_id, tool_name=tool_name, tool_params={"cmd": "ls"},
    )


def observation(
    tool_call_id: str, tool_name: str = "bash_exec", *,
    output: str | None = "ok", error: str | None = None, agent_id: str = AGENT,
) -> Observation:
    """A tool_result Observation, byte-shaped like agent/loop.py:1009."""
    return Observation(
        agent_id=agent_id, session_id=SESSION, observation_type="tool_result",
        tool_call_id=tool_call_id, tool_name=tool_name, output=output, error=error,
    )


class Probe:
    """Records every event of the types it subscribes to, in arrival order."""

    def __init__(self) -> None:
        self.seen: list = []

    async def record(self, event) -> None:
        self.seen.append(event)

    def watch(self, bus: EventBus, *types, agent_id: str = AGENT) -> None:
        for t in types:
            bus.subscribe(t, self.record, agent_id=agent_id)

    def types(self) -> list[str]:
        return [type(e).__name__ for e in self.seen]


async def test_bus_contract_ordering(tmp_path: Path):
    """One tool call -> the full correlated five-event sequence, in order, all three
    new events keyed to the same tool_call_id + tool_name."""
    store = make_store(tmp_path)
    await store.open()
    bus = EventBus()
    probe = Probe()
    # Probe subscribes BEFORE the gate opens: with inline delivery the gate publishes
    # its derived events from inside the loop-event handler, so a probe subscribed
    # after the gate would see the derived event first. Subscribing first models a
    # watcher on the real stream — loop event, then the harness reaction.
    probe.watch(bus, Action, Observation, ExpectationAttached, OutcomeObserved, SurpriseScored)
    gate = PredictiveGate(store, bus, AGENT, PredictiveGateConfig())
    await gate.open()
    try:
        await bus.publish(action("tc-1"))
        await bus.publish(observation("tc-1"))

        assert probe.types() == [
            "Action", "ExpectationAttached", "Observation", "OutcomeObserved", "SurpriseScored",
        ]
        for e in probe.seen:
            if isinstance(e, (ExpectationAttached, OutcomeObserved, SurpriseScored)):
                assert e.tool_call_id == "tc-1"
                assert e.tool_name == "bash_exec"
    finally:
        await store.close()


async def test_rows_persisted(tmp_path: Path):
    """After one pair: exactly one tool_observations row (is_error=0, event_id == the
    Observation's id) and one surprise_scores row whose expectation_json is a dict
    carrying 'error_rate' and 'n'."""
    store = make_store(tmp_path)
    await store.open()
    bus = EventBus()
    gate = PredictiveGate(store, bus, AGENT, PredictiveGateConfig())
    await gate.open()
    try:
        await bus.publish(action("tc-1"))
        obs = await bus.publish(observation("tc-1", output="ok"))

        async with store._db.execute(
            "SELECT is_error, event_id FROM tool_observations"
        ) as cur:
            rows = await cur.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 0
        assert rows[0][1] == obs.id

        async with store._db.execute("SELECT expectation_json FROM surprise_scores") as cur:
            srows = await cur.fetchall()
        assert len(srows) == 1
        exp = json.loads(srows[0][0])
        assert "error_rate" in exp and "n" in exp
    finally:
        await store.close()


async def test_error_outcome(tmp_path: Path):
    """An Observation carrying error -> OutcomeObserved.is_error True and is_error=1
    persisted (is_error derives from Observation.error IS NOT NULL)."""
    store = make_store(tmp_path)
    await store.open()
    bus = EventBus()
    probe = Probe()
    probe.watch(bus, OutcomeObserved)
    gate = PredictiveGate(store, bus, AGENT, PredictiveGateConfig())
    await gate.open()
    try:
        await bus.publish(action("tc-e"))
        await bus.publish(observation("tc-e", output="[DENIED]", error="[tool error] boom"))

        outcomes = [e for e in probe.seen if isinstance(e, OutcomeObserved)]
        assert len(outcomes) == 1
        assert outcomes[0].is_error is True

        async with store._db.execute("SELECT is_error FROM tool_observations") as cur:
            assert (await cur.fetchone())[0] == 1
    finally:
        await store.close()


async def test_unmatched_observation_ignored(tmp_path: Path):
    """An Observation for a tool_call_id never seen (pre-subscribe, evicted, foreign)
    publishes nothing, persists nothing, raises nothing."""
    store = make_store(tmp_path)
    await store.open()
    bus = EventBus()
    probe = Probe()
    probe.watch(bus, OutcomeObserved, SurpriseScored)
    gate = PredictiveGate(store, bus, AGENT, PredictiveGateConfig())
    await gate.open()
    try:
        await bus.publish(observation("never-seen"))

        assert probe.seen == []
        async with store._db.execute("SELECT COUNT(*) FROM tool_observations") as cur:
            assert (await cur.fetchone())[0] == 0
        async with store._db.execute("SELECT COUNT(*) FROM surprise_scores") as cur:
            assert (await cur.fetchone())[0] == 0
    finally:
        await store.close()


async def test_non_tool_events_ignored(tmp_path: Path):
    """A non-tool Action (llm_response) and a non-tool_result Observation produce
    nothing — the gate keys strictly on action_type/observation_type."""
    store = make_store(tmp_path)
    await store.open()
    bus = EventBus()
    probe = Probe()
    probe.watch(bus, ExpectationAttached, OutcomeObserved, SurpriseScored)
    gate = PredictiveGate(store, bus, AGENT, PredictiveGateConfig())
    await gate.open()
    try:
        await bus.publish(Action(
            agent_id=AGENT, session_id=SESSION, action_type="llm_response", content="hi",
        ))
        await bus.publish(Observation(
            agent_id=AGENT, session_id=SESSION, observation_type="user_input", output="hello",
        ))

        assert probe.seen == []
        assert gate._pending == {}
        async with store._db.execute("SELECT COUNT(*) FROM tool_observations") as cur:
            assert (await cur.fetchone())[0] == 0
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Task 2: hot-path proofs — the ROADMAP criterion-1 "event-count + timing proof"
# and the reframe doc's "skips under load; never blocks or raises into the hot
# path" binding, encoded as executable regression tests (not prose).
# ---------------------------------------------------------------------------


def scripted_events() -> list:
    """A FRESH loop-originated instance list per call — BaseEvent is frozen and
    publish() rejects an already-sequenced instance, so the two buses cannot share
    instances. Three tool pairs + one UserMessage + one TurnCompleted."""
    return [
        action("tc-1"), observation("tc-1"),
        action("tc-2"), observation("tc-2", output="[DENIED]", error="[tool error] denied"),
        action("tc-3"), observation("tc-3", output="done"),
        UserMessage(agent_id=AGENT, session_id=SESSION, content="thanks", channel="cli"),
        TurnCompleted(agent_id=AGENT, session_id=SESSION, iterations=3,
                      duration_seconds=1.0, elapsed_tokens=42, summary="ok"),
    ]


async def test_event_count_identity(tmp_path: Path):
    """Zero behavior change: the loop's own event stream is identical (count, order, and
    payload modulo seq/id/timestamp) with and without the gate attached; the gate ADDS
    only the three predictive types and never mutates or re-publishes a loop event."""
    loop_types = (Action, Observation, UserMessage, TurnCompleted)

    # Bus WITHOUT the gate — nothing reacts, no store needed.
    bus0 = EventBus()
    probe0 = Probe()
    probe0.watch(bus0, *loop_types)
    for e in scripted_events():
        await bus0.publish(e)

    # Bus WITH the gate.
    store = make_store(tmp_path)
    await store.open()
    bus1 = EventBus()
    probe1 = Probe()
    probe1.watch(bus1, *loop_types)
    added = Probe()
    added.watch(bus1, ExpectationAttached, OutcomeObserved, SurpriseScored)
    gate = PredictiveGate(store, bus1, AGENT, PredictiveGateConfig())
    await gate.open()
    try:
        for e in scripted_events():
            await bus1.publish(e)
    finally:
        await store.close()

    def sig(e) -> tuple:
        return (
            type(e).__name__,
            getattr(e, "tool_call_id", None),
            getattr(e, "output", None),
            getattr(e, "error", None),
            getattr(e, "content", None),
        )

    assert probe1.types() == probe0.types()                                  # count + order
    assert [sig(e) for e in probe1.seen] == [sig(e) for e in probe0.seen]    # payloads
    # The gate's ONLY bus footprint: exactly one of each new type per tool pair.
    assert added.types().count("ExpectationAttached") == 3
    assert added.types().count("OutcomeObserved") == 3
    assert added.types().count("SurpriseScored") == 3
    assert len(added.seen) == 9


async def test_timing_bound(tmp_path: Path):
    """ROADMAP criterion 1's timing proof: with a warm 50-row prior, each Action+
    Observation pair is one indexed SELECT + two INSERTs — median wall time per pair
    well under a generous 100ms CI bound (the claim is a cost class, not a microbench)."""
    store = make_store(tmp_path)
    await store.open()
    bus = EventBus()
    gate = PredictiveGate(store, bus, AGENT, PredictiveGateConfig())
    await gate.open()
    try:
        for i in range(50):
            await store.record_tool_observation(
                session_id=SESSION, tool_call_id=f"seed-{i}", tool_name="bash_exec",
                ts=1000 + i, is_error=int(i % 5 == 0), output_len=100 + i,
                duration_ms=50 + i, event_id=f"seed-ev-{i}",
            )
        timings = []
        for i in range(20):
            t0 = time.perf_counter()
            await bus.publish(action(f"tc-{i}"))
            await bus.publish(observation(f"tc-{i}"))
            timings.append(time.perf_counter() - t0)
        assert statistics.median(timings) < 0.1
    finally:
        await store.close()


async def test_never_raises(tmp_path: Path, monkeypatch, caplog):
    """A store method that raises never surfaces to the publisher: loop events are still
    delivered, no SurpriseScored fires for the failed pair, and the GATE logged the
    exception itself (not the bus) — so a future bus change can't weaponize a scorer bug."""
    store = make_store(tmp_path)
    await store.open()
    bus = EventBus()
    probe = Probe()
    probe.watch(bus, Action, Observation)   # loop-originated only
    added = Probe()
    added.watch(bus, SurpriseScored)
    gate = PredictiveGate(store, bus, AGENT, PredictiveGateConfig())
    await gate.open()
    try:
        async def boom(*a, **k):
            raise RuntimeError("store down")
        monkeypatch.setattr(store, "record_tool_observation", boom)

        with caplog.at_level(logging.ERROR):
            await bus.publish(action("tc-1"))       # nothing propagates
            await bus.publish(observation("tc-1"))  # nothing propagates

        assert probe.types() == ["Action", "Observation"]   # loop stream intact
        assert added.seen == []                             # no score for the failed pair
        assert any(
            "predictive" in r.getMessage().lower() and r.levelname == "ERROR"
            for r in caplog.records
        )
    finally:
        await store.close()


async def test_pending_cap_evicts(tmp_path: Path):
    """Load-shed at the cap: with pending_cap=8, 12 unobserved Actions leave exactly the
    8 newest pending (4 oldest evicted); a late Observation for an evicted id is inert."""
    store = make_store(tmp_path)
    await store.open()
    bus = EventBus()
    added = Probe()
    added.watch(bus, OutcomeObserved, SurpriseScored)
    gate = PredictiveGate(store, bus, AGENT, PredictiveGateConfig(pending_cap=8))
    await gate.open()
    try:
        for i in range(12):
            await bus.publish(action(f"tc-{i}"))
        assert len(gate._pending) == 8
        assert set(gate._pending) == {f"tc-{i}" for i in range(4, 12)}
        assert all(f"tc-{i}" not in gate._pending for i in range(4))

        await bus.publish(observation("tc-0"))   # a since-evicted id
        assert added.seen == []
        async with store._db.execute("SELECT COUNT(*) FROM tool_observations") as cur:
            assert (await cur.fetchone())[0] == 0
    finally:
        await store.close()


async def test_close_unsubscribes(tmp_path: Path):
    """After close() the gate is fully detached — a new Action produces no
    ExpectationAttached."""
    store = make_store(tmp_path)
    await store.open()
    bus = EventBus()
    probe = Probe()
    probe.watch(bus, ExpectationAttached)
    gate = PredictiveGate(store, bus, AGENT, PredictiveGateConfig())
    await gate.open()
    try:
        await gate.close()
        await bus.publish(action("tc-1"))
        assert probe.seen == []
    finally:
        await store.close()
