"""Phase 31 (v2.0 Hierarchical Memory) — idle consolidation: CONS-01..06.

Covers: staleness/idle trigger logic + watermark (CONS-01), cooperative cancellation of
in-flight LLM work (CONS-02), fold + cross-episode recurrence promotion + guardrails
(CONS-03/04), the SOFT cap trim (CONS-05), and churn/sample quality proxies (CONS-06).
"""
import asyncio
import time
from pathlib import Path

import pytest

from localharness.config.models import MemoryConsolidationConfig
from localharness.memory.consolidation import ConsolidationPass, ConsolidationScheduler
from localharness.memory.sqlite import FactQuery, MemoryStore


@pytest.fixture
async def store(tmp_path: Path):
    s = MemoryStore(agent_id="cons-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


def _cfg(**over) -> MemoryConsolidationConfig:
    base = dict(enabled=True, idle_minutes=0.5, staleness_hours=6.0,
                max_active_facts=256, decay_half_life_days=30.0, iteration_cap=200)
    base.update(over)
    return MemoryConsolidationConfig(**base)


async def _seed_candidate(store, tool: str, session: str, detail: str, tier="resolved_error"):
    """A gate candidate exactly as WriteGate writes them (key shape + tags + provenance)."""
    await store.store_fact(
        key=f"gate/{tier}/{tool}/{abs(hash(session + detail)) % 10**8}",
        value=f"Tool `{tool}` failed then succeeded. Error was: {detail}",
        tags=["gate", f"tier:{tier}", "pending_consolidation"],
        confidence=0.65,
        source="write_gate",
        provenance=session,
    )


# ---------------------------------------------------------------------------
# CONS-03: fold + recurrence promotion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_two_episode_recurrence_promotes_one_off_does_not(store: MemoryStore):
    await _seed_candidate(store, "bash_exec", "sess-1", "command not found: uvx")
    await _seed_candidate(store, "bash_exec", "sess-2", "command not found: uvx (again)")
    await _seed_candidate(store, "grep", "sess-1", "bad regex")  # one episode only

    report = await ConsolidationPass(store, _cfg()).run()
    assert report.promoted == 1

    promoted = await store.get_fact("learned/bash_exec/resolved_error")
    assert promoted is not None
    assert promoted.confidence >= 0.7  # crosses the injection threshold
    assert "2 episodes" in promoted.value
    assert await store.get_fact("learned/grep/resolved_error") is None  # one-off stays candidate

    # Index now carries the promoted lesson; candidates left the index-eligible set.
    index = await store._render_memory_index(10)
    assert "learned/bash_exec/resolved_error" in index
    assert "gate/resolved_error/bash_exec" not in index

    # derived_from edges link the promoted record to its source candidates.
    walk = await store.neighborhood(promoted.id, depth=1)
    assert len(walk) == 3  # self + 2 candidates


@pytest.mark.asyncio
async def test_pass_folds_staged_reads(store: MemoryStore):
    await store.store_fact("k", "v")
    await store.touch_staged(["k"])
    report = await ConsolidationPass(store, _cfg()).run()
    assert report.folded == 1
    assert (await store.get_fact("k")).access_count == 1


@pytest.mark.asyncio
async def test_re_promotion_with_new_episode_supersedes(store: MemoryStore):
    await _seed_candidate(store, "web_fetch", "s1", "timeout")
    await _seed_candidate(store, "web_fetch", "s2", "timeout too")
    await ConsolidationPass(store, _cfg()).run()
    await _seed_candidate(store, "web_fetch", "s3", "fresh third failure")
    await ConsolidationPass(store, _cfg()).run()

    history = await store.get_fact_history("learned/web_fetch/resolved_error")
    assert len(history) == 2  # merged record changed → superseded, never overwritten
    assert history[0].status == "active" or history[1].status == "active"


# ---------------------------------------------------------------------------
# CONS-02: cooperative cancellation (the serial inference gate must be released)
# ---------------------------------------------------------------------------

class _SlowLLM:
    def __init__(self, delay: float = 10.0):
        self.delay = delay
        self.cancelled = False

    async def complete(self, prompt: str) -> str:
        try:
            await asyncio.sleep(self.delay)
            return "a lesson"
        except asyncio.CancelledError:
            self.cancelled = True
            raise


@pytest.mark.asyncio
async def test_user_turn_cancels_in_flight_generation(store: MemoryStore):
    await store.append_history({"v": 1, "agent_id": "cons-agent", "type": "assistant_message",
                                "content": "VPNACCESS matters", "session_id": "s", "id": "1", "ts": 1})
    llm = _SlowLLM(delay=10.0)
    cons = ConsolidationPass(store, _cfg(), llm=llm)

    t0 = time.monotonic()
    task = asyncio.create_task(cons.run())
    await asyncio.sleep(0.1)
    cons.cancel()  # what a user turn does via the scheduler
    report = await asyncio.wait_for(task, timeout=3.0)

    assert report.cancelled
    assert time.monotonic() - t0 < 3.0  # nobody waited 10s behind the generation
    assert llm.cancelled  # the generation task was truly cancelled (gate released)


# ---------------------------------------------------------------------------
# CONS-04: replay guardrails (fake LLM — live quality iteration deferred)
# ---------------------------------------------------------------------------

class _FakeLLM:
    def __init__(self, text: str):
        self.text = text

    async def complete(self, prompt: str) -> str:
        return self.text


@pytest.mark.asyncio
async def test_replay_verifies_against_leaf_and_dedups(store: MemoryStore):
    await store.append_history({"v": 1, "agent_id": "cons-agent", "type": "assistant_message", "id": "1",
                                "session_id": "s", "ts": 1, "content": "Deploys require VPNACCESS before anything works"})
    llm = _FakeLLM(
        "Deploys require VPNACCESS enabled\n"          # grounded (VPNACCESS in corpus) → kept
        "Unicorns fabricate quarterly numbers\n"        # confabulated (no ≥6-char token) → dropped
    )
    report = await ConsolidationPass(store, _cfg(), llm=llm).run()
    assert report.replayed_claims == 1
    stored = await store.query_facts(FactQuery(text="VPNACCESS", min_confidence=0.0))
    assert any("pending_consolidation" in f.tags for f in stored)

    # dedup-before-generate: the identical claim on a second pass is not re-stored.
    report2 = await ConsolidationPass(store, _cfg(), llm=llm).run()
    assert report2.replayed_claims == 0


# ---------------------------------------------------------------------------
# RANK-03 time axis: decay
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unused_fact_decays_out_of_index_never_out_of_store(store: MemoryStore):
    await store.store_fact("fading", "unused for 90 days")
    ninety_days_ago = int(time.time()) - 90 * 86400
    await store._db.execute(
        "UPDATE facts SET updated_at = ?, last_accessed_at = ? WHERE key = 'fading'",
        (ninety_days_ago, ninety_days_ago),
    )
    await store._db.commit()

    report = await ConsolidationPass(store, _cfg(decay_half_life_days=30.0)).run()
    assert report.decayed == 1
    fact = await store.get_fact("fading")
    assert fact is not None            # never deleted
    assert fact.retrieval_strength < 0.2   # out of the injected index
    assert "fading" not in await store._render_memory_index(10)
    hits = await store.query_facts(FactQuery(text="unused", min_confidence=0.0))
    assert hits and hits[0].key == "fading"  # still searchable via the tool path


# ---------------------------------------------------------------------------
# CONS-05: the SOFT cap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_over_cap_admission_never_blocks_and_pass_trims(store: MemoryStore):
    cfg = _cfg(max_active_facts=8)
    for i in range(12):
        await store.store_fact(f"fact-{i:02d}", f"body {i}")  # admission always succeeds

    async with store._db.execute(
        "SELECT COUNT(*) FROM facts WHERE status='active' AND retrieval_strength >= 0.2"
    ) as cur:
        (before,) = await cur.fetchone()
    assert before == 12  # soft: over-cap state is allowed to exist

    report = await ConsolidationPass(store, cfg).run()
    assert report.demoted == 4
    async with store._db.execute(
        "SELECT COUNT(*), (SELECT COUNT(*) FROM facts) FROM facts "
        "WHERE status='active' AND retrieval_strength >= 0.2"
    ) as cur:
        in_index, total = await cur.fetchone()
    assert in_index == 8   # trimmed back under the bound at the consolidation boundary
    assert total == 12     # nothing deleted — demoted facts remain searchable


# ---------------------------------------------------------------------------
# CONS-06: quality proxies
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_churn_metric_and_sample_hook(store: MemoryStore):
    samples: list = []

    async def hook(facts):
        samples.extend(facts)

    await _seed_candidate(store, "bash_exec", "s1", "err A")
    await _seed_candidate(store, "bash_exec", "s2", "err B")
    report = await ConsolidationPass(store, _cfg(), on_promotion_sample=hook).run()
    assert report.promoted == 1
    assert samples and samples[0].key == "learned/bash_exec/resolved_error"

    # Supersede the promoted fact → next pass's churn rate reflects it (junk signal).
    await store.store_fact("learned/bash_exec/resolved_error", "manually corrected")
    report2 = await ConsolidationPass(store, _cfg()).run()
    assert report2.churn_rate > 0.0


# ---------------------------------------------------------------------------
# CONS-01: scheduler triggers + watermark
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_staleness_gate_watermark_and_idempotent_launch(store: MemoryStore):
    from localharness.core.bus import EventBus

    bus = EventBus()
    sched = ConsolidationScheduler(store, bus, "cons-agent", _cfg())

    # Fresh DB, no watermark, but no work either → no run.
    assert not await sched.should_run()
    await _seed_candidate(store, "bash_exec", "s1", "e1")
    assert await sched.should_run()  # stale + work

    sched.launch()
    sched.launch()  # idempotent while running
    await asyncio.wait_for(sched._run_task, timeout=5.0)
    assert sched.last_report is not None

    assert not await sched.should_run()  # watermark fresh now


@pytest.mark.asyncio
async def test_user_activity_cancels_running_pass(store: MemoryStore):
    from localharness.core.bus import EventBus

    await store.append_history({"v": 1, "agent_id": "cons-agent", "type": "assistant_message", "id": "1",
                                "session_id": "s", "ts": 1, "content": "SOMETOKEN here"})
    bus = EventBus()
    sched = ConsolidationScheduler(store, bus, "cons-agent", _cfg(), llm=_SlowLLM(10.0))
    sched.launch()
    await asyncio.sleep(0.1)
    await sched._on_user_activity(None)  # what the UserMessage subscription does
    await asyncio.wait_for(sched._run_task, timeout=3.0)
    assert sched.last_report is not None and sched.last_report.cancelled


@pytest.mark.asyncio
async def test_disabled_scheduler_is_inert(store: MemoryStore):
    from localharness.core.bus import EventBus

    sched = ConsolidationScheduler(store, EventBus(), "cons-agent", _cfg(enabled=False))
    await sched.start()
    assert sched._timer_task is None and not sched._handles
    assert not await sched.should_run()
    await sched.stop()
