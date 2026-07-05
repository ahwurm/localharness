"""COLL-03 collect-only credit assignment (Phase 34-04), end-to-end through a REAL
EventBus + REAL MemoryStore.

A correction marks the explicitly-staged facts 'suspect'; a confirmation marks them
'bump'; an interruption logs alone (weaker class, no candidates); a neutral message
logs nothing. And the facts table is provably UNTOUCHED by all of it — collect-only, no
store write to facts is gated on any signal. The motivating fireworks specimen produces a
look-ready labeled record (full message + corrected-turn pointer + sitting id + ts).
"""
import pytest

from localharness.config.models import PredictiveGateConfig
from localharness.core.bus import EventBus
from localharness.core.events import TurnCompleted, UserMessage
from localharness.memory.sqlite import MemoryStore
from localharness.memory.user_signals import UserSignalDetector

_AGENT = "snap-agent"


async def _harness(tmp_path):
    # The store is given bus=None so ONLY the detector reacts to UserMessage — otherwise
    # MemoryStore.open() would also subscribe its own UserMessage handler and muddy the
    # assertions. The detector calls the store's methods directly, no bus needed there.
    store = MemoryStore(agent_id=_AGENT, division_id="", org_id="", base_dir=str(tmp_path))
    await store.open()
    bus = EventBus()
    det = UserSignalDetector(store, bus, _AGENT, PredictiveGateConfig())
    await det.open()
    return store, bus


def _um(content):
    return UserMessage(
        agent_id=_AGENT, session_id="s1", content=content, channel="terminal"
    )


async def _stage_two_facts(store):
    await store.store_fact("fact-a", "alpha")
    await store.store_fact("fact-b", "beta")
    await store.store_fact("fact-c", "gamma")  # never staged -> never a candidate
    await store.touch_staged(["fact-a", "fact-b"])


async def _signal_rows(store):
    async with store._db.execute(
        "SELECT id, signal_type FROM user_signals ORDER BY id"
    ) as cur:
        return [tuple(r) for r in await cur.fetchall()]


async def _snapshots(store):
    async with store._db.execute(
        "SELECT fact_key, candidate_type, user_signal_id FROM staged_snapshots"
    ) as cur:
        return [tuple(r) for r in await cur.fetchall()]


@pytest.mark.asyncio
async def test_correction_snapshots_suspect(tmp_path):
    store, bus = await _harness(tmp_path)
    try:
        await _stage_two_facts(store)
        await bus.publish(_um("no that's wrong"))
        sigs = await _signal_rows(store)
        assert len(sigs) == 1 and sigs[0][1] == "correction"
        snaps = await _snapshots(store)
        assert {s[0] for s in snaps} == {"fact-a", "fact-b"}  # untouched fact-c excluded
        assert all(s[1] == "suspect" for s in snaps)
        assert all(s[2] == sigs[0][0] for s in snaps)  # linked by user_signal_id
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_confirmation_snapshots_bump(tmp_path):
    store, bus = await _harness(tmp_path)
    try:
        await _stage_two_facts(store)
        await bus.publish(_um("exactly what i needed"))
        sigs = await _signal_rows(store)
        assert len(sigs) == 1 and sigs[0][1] == "confirmation"
        snaps = await _snapshots(store)
        assert len(snaps) == 2 and all(s[1] == "bump" for s in snaps)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_interruption_logs_but_snapshots_nothing(tmp_path):
    store, bus = await _harness(tmp_path)
    try:
        await _stage_two_facts(store)
        await bus.publish(_um("hold on"))
        sigs = await _signal_rows(store)
        assert len(sigs) == 1 and sigs[0][1] == "interruption"
        assert await _snapshots(store) == []  # weaker class -> zero candidates
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_no_signal_no_rows(tmp_path):
    store, bus = await _harness(tmp_path)
    try:
        await _stage_two_facts(store)
        await bus.publish(_um("please check the weather forecast"))
        assert await _signal_rows(store) == []
        assert await _snapshots(store) == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_facts_untouched_by_signals(tmp_path):
    """Collect-only: no store write to facts is gated on any signal — confidence,
    access_count, and importance are byte-identical before and after the whole barrage."""
    store, bus = await _harness(tmp_path)
    try:
        await _stage_two_facts(store)

        async def _fact_state():
            async with store._db.execute(
                "SELECT key, confidence, access_count, importance FROM facts "
                "WHERE status = 'active' ORDER BY key"
            ) as cur:
                return [tuple(r) for r in await cur.fetchall()]

        before = await _fact_state()
        await bus.publish(_um("no that's wrong"))       # correction -> suspect snapshots
        await bus.publish(_um("exactly what i needed"))  # confirmation -> bump snapshots
        await bus.publish(_um("hold on"))                # interruption -> nothing
        after = await _fact_state()
        assert before == after
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_fireworks_record_is_look_ready(tmp_path):
    """The phase's motivating specimen: a casually-stated preference correction with NO
    tool error and NO stuck loop (invisible to the pain-only gate) now produces a
    look-ready labeled record — full message + corrected-turn pointer + sitting id + ts,
    with the staged facts marked suspect and linked to this signal."""
    store, bus = await _harness(tmp_path)
    try:
        await _stage_two_facts(store)
        await bus.publish(
            TurnCompleted(
                agent_id=_AGENT, session_id="s1", iterations=1, duration_seconds=1.0,
                elapsed_tokens=10,
                summary="set your fireworks preference to 'hide from view'",
            )
        )
        specimen = "nah id rather watch the fireworks from the park with friends tomorrow"
        await bus.publish(_um(specimen))
        async with store._db.execute(
            "SELECT signal_type, trigger_family, matched_text, user_message, "
            "corrected_turn_summary, session_id, ts FROM user_signals"
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert (row[0], row[1], row[2]) == ("correction", "negation", "nah")
        assert row[3] == specimen                # FULL message, not a preview
        assert "fireworks preference" in row[4]  # the corrected-turn pointer is held
        assert row[5] == "s1" and row[6] > 0     # sitting id + ts
        assert all(s[1] == "suspect" for s in await _snapshots(store))
    finally:
        await store.close()
