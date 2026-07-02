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
    """A gate candidate exactly as WriteGate writes them — key shape MIRRORS gate.py
    (whole-milestone critic B1: a fixture with its own key scheme made composition
    bugs invisible): gate/<tier>/<tool>/<LESSON=h8(tool,detail)>/<SESSION=h8(session)>."""
    from localharness.memory.gate import _h8 as _gate_h8
    await store.store_fact(
        key=f"gate/{tier}/{tool}/{_gate_h8(tool, detail)}/{_gate_h8(session)}",
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
    await _seed_candidate(store, "bash_exec", "sess-2", "command not found: uvx")  # SAME lesson, new episode
    await _seed_candidate(store, "grep", "sess-1", "bad regex")  # one episode only

    report = await ConsolidationPass(store, _cfg()).run()
    assert report.promoted == 1

    assert report.promoted_keys[0].startswith("learned/bash_exec/resolved_error/")
    promoted = await store.get_fact(report.promoted_keys[0])
    assert promoted is not None
    assert promoted.confidence >= 0.7  # crosses the injection threshold
    assert "2 episodes" in promoted.value
    assert not [k for k in report.promoted_keys if "/grep/" in k]  # one-off stays candidate

    # Index now carries the promoted lesson; candidates left the index-eligible set.
    index = await store._render_memory_index(10)
    assert report.promoted_keys[0] in index
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
    await _seed_candidate(store, "web_fetch", "s2", "timeout")
    r1 = await ConsolidationPass(store, _cfg()).run()
    await _seed_candidate(store, "web_fetch", "s3", "timeout")  # same lesson, 3rd episode
    await ConsolidationPass(store, _cfg()).run()

    history = await store.get_fact_history(r1.promoted_keys[0])
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
# Phase-31 critic dispositions (BLOCKER 2, M1, M4)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_same_run_promotion_survives_cap_trim(store: MemoryStore):
    """Critic BLOCKER 2: a freshly promoted record (zero access history = lowest slow
    score) must not be demoted by the cap-trim of the very pass that promoted it."""
    import time as _t
    cfg = _cfg(max_active_facts=8)
    now = int(_t.time())
    for i in range(9):  # established facts with real access history, already at cap
        await store.store_fact(f"veteran-{i}", f"old body {i}")
    await store._db.execute(
        "UPDATE facts SET access_count = 5, last_accessed_at = ? WHERE key LIKE 'veteran-%'",
        (now,),
    )
    await store._db.commit()
    await _seed_candidate(store, "bash_exec", "s1", "same err")
    await _seed_candidate(store, "bash_exec", "s2", "same err")

    report = await ConsolidationPass(store, cfg).run()
    assert report.promoted == 1 and report.demoted > 0
    promoted = await store.get_fact(report.promoted_keys[0])
    assert promoted.retrieval_strength >= 0.2  # visible in the index, NOT self-demoted
    assert report.promoted_keys[0] in await store._render_memory_index(10)


@pytest.mark.asyncio
async def test_salient_single_episode_promotes(store: MemoryStore):
    """Critic M1: the APPROACH §C salience-flag route — a stuck-recovery (tagged
    salient by the gate) promotes on ONE episode; recurrence is not required."""
    await store.store_fact(
        key="gate/stuck_recovered/abc12345",
        value="Agent got stuck (repeated `read:x`) and recovered at iteration 7.",
        tags=["gate", "tier:stuck_recovered", "pending_consolidation", "salient"],
        confidence=0.6, source="write_gate", provenance="s1",
    )
    report = await ConsolidationPass(store, _cfg()).run()
    assert report.promoted == 1
    promoted = await store.get_fact("learned/abc12345/stuck_recovered")
    assert promoted is not None and promoted.confidence >= 0.7


@pytest.mark.asyncio
async def test_verify_against_leaf_rejects_shared_common_word(store: MemoryStore):
    """Critic M4: a confabulated claim sharing ONE common ≥6-char word with the corpus
    ('contains') must be rejected — majority-token grounding required."""
    await store.append_history({"v": 1, "agent_id": "cons-agent", "type": "assistant_message",
                                "id": "1", "session_id": "s", "ts": 1,
                                "content": "The database contains customer records"})
    llm = _FakeLLM("The production database contains unencrypted admin passwords for all customers")
    report = await ConsolidationPass(store, _cfg(), llm=llm).run()
    assert report.replayed_claims == 0  # confidently-alarming junk stays OUT


@pytest.mark.asyncio
async def test_demoted_fact_recovers_through_use(store: MemoryStore):
    """Critic minor 2 / RANK-03 'bumped on confirmed recall': heavy use restores a
    demoted fact's retrieval strength via the fold — demotion is not a one-way door."""
    await store.store_fact("comeback", "demoted then heavily used")
    await store._db.execute("UPDATE facts SET retrieval_strength = 0.15 WHERE key='comeback'")
    await store._db.commit()
    for _ in range(3):
        await store.touch_staged(["comeback"])
    await store.fold_staged_access()
    fact = await store.get_fact("comeback")
    assert fact.retrieval_strength >= 0.2  # 0.15 + 3*0.05 → back above the index gate


# ---------------------------------------------------------------------------
# CONS-06: quality proxies
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_churn_metric_and_sample_hook(store: MemoryStore):
    samples: list = []

    async def hook(facts):
        samples.extend(facts)

    await _seed_candidate(store, "bash_exec", "s1", "err A")
    await _seed_candidate(store, "bash_exec", "s2", "err A")
    report = await ConsolidationPass(store, _cfg(), on_promotion_sample=hook).run()
    assert report.promoted == 1
    assert samples and samples[0].key == report.promoted_keys[0]

    # Supersede the promoted fact → next pass's churn rate reflects it (junk signal).
    await store.store_fact(report.promoted_keys[0], "manually corrected")
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


# ---------------------------------------------------------------------------
# Whole-milestone critic B1: the COMPOSED gate→consolidation path (real WriteGate
# keys, not fixture keys) — recurrence semantics must hold in both directions.
# ---------------------------------------------------------------------------

class _NullBus:
    def subscribe(self, *a, **k):
        return object()

    def unsubscribe(self, h):
        pass

    async def publish(self, e):
        pass


@pytest.mark.asyncio
async def test_composed_gate_consolidation_recurrence_semantics(store: MemoryStore):
    """Same lesson across 2 sessions → promotes; two DIFFERENT one-off errors on one
    tool → must NOT merge into a fabricated 'recurring' record (critic B1's false
    positive), and the recurring lesson must not self-supersede into invisibility
    (B1's false negative)."""
    from localharness.core.events import Observation
    from localharness.memory.gate import WriteGate

    gate = WriteGate(store, _NullBus(), "cons-agent")

    async def cycle(sess: str, tool: str, err: str):
        await gate._on_observation(Observation(
            agent_id="cons-agent", session_id=sess, observation_type="tool_result",
            tool_name=tool, output="", error=err))
        await gate._on_observation(Observation(
            agent_id="cons-agent", session_id=sess, observation_type="tool_result",
            tool_name=tool, output="ok", error=None))

    await cycle("s1", "bash_exec", "permission denied: /etc/passwd")
    await cycle("s2", "bash_exec", "permission denied: /etc/passwd")  # SAME lesson recurs
    await cycle("s1", "edit", "old_string not found")                 # two DIFFERENT
    await cycle("s2", "edit", "target file missing")                  # one-offs

    report = await ConsolidationPass(store, _cfg()).run()
    assert report.promoted == 1  # exactly the recurring bash_exec lesson
    assert report.promoted_keys[0].startswith("learned/bash_exec/resolved_error/")
    promoted = await store.get_fact(report.promoted_keys[0])
    assert "2 episodes" in promoted.value
    assert not [k for k in report.promoted_keys if "/edit/" in k]
