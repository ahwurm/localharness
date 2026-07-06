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
# CONS-04: grounding kill-net (the retired replay seam's discipline now lives in mining —
# see test_transcript_mining.py; MOVE 2 removed the orphaned replay/* extractor).
# ---------------------------------------------------------------------------

class _FakeLLM:
    def __init__(self, text: str):
        self.text = text

    async def complete(self, prompt: str) -> str:
        return self.text


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


@pytest.mark.asyncio
async def test_promoted_lesson_payload_survives_injected_block(store: MemoryStore):
    """Live-test regression (2026-07-03): chat #3 fumbled WITH the lesson nominally
    in context, because the promoted value led with bookkeeping ("Recurring (2
    episodes): tier — …") and the index line's char budget guillotined the payload
    before the filename. The full seam — real WriteGate observations (loop-shaped
    "[tool error] " prefix included), real ConsolidationPass, real load_context()
    injection — must deliver the DISCRIMINATING content: what failed, what worked."""
    from localharness.core.events import Observation
    from localharness.memory.gate import WriteGate

    gate = WriteGate(store, _NullBus(), "cons-agent")
    err = "[tool error] File not found: /home/user/localharness/docs/VISION.md"
    fix = "1 # Spec 00: Architecture Overview for LocalHarness — layers, event bus"
    for sess in ("sitting-1", "sitting-2"):
        await gate._on_observation(Observation(
            agent_id="cons-agent", session_id=sess, observation_type="tool_result",
            tool_name="read", output="", error=err))
        await gate._on_observation(Observation(
            agent_id="cons-agent", session_id=sess, observation_type="tool_result",
            tool_name="read", output=fix, error=None))

    report = await ConsolidationPass(store, _cfg()).run()
    assert report.promoted == 1

    block = (await store.load_context(index_mode=True)).agent_memory_md
    line = next(ln for ln in block.splitlines() if "learned/read/resolved_error/" in ln)
    # The two facts that make the lesson actionable — the failure's subject and
    # the resolution's head — must survive every render layer.
    assert "docs/VISION.md" in line
    assert "Spec 00" in line
    # Payload-first at every layer: lesson before recurrence bookkeeping, and the
    # loop's presentation prefix stripped at capture.
    assert line.find("File not found") < line.find("[recurring: 2 episodes")
    assert "[tool error]" not in line


# ---------------------------------------------------------------------------
# SESS-01 + FLOW2-BUS-HOP: the session-unit ruling made executable, and the real
# bus→filtered-subscription→gate delivery seam proven with live EventBus objects.
# ---------------------------------------------------------------------------

async def _facts_like(store: MemoryStore, pattern: str) -> list[tuple[str, str]]:
    """(key, provenance) rows matching a LIKE pattern — the raw-SQL lookup consolidation.py
    itself uses (150-156) to find candidates without knowing the lesson hash."""
    assert store._db is not None
    async with store._db.execute(
        "SELECT key, provenance FROM facts WHERE agent_id = ? AND key LIKE ? ORDER BY key",
        (store._agent_id, pattern),
    ) as cur:
        return list(await cur.fetchall())


@pytest.mark.asyncio
async def test_same_sitting_double_stumble_does_not_promote(store: MemoryStore):
    """The owner's session-unit ruling (SESS-01) made executable: a double-stumble WITHIN
    one sitting is ONE provenance, so it does NOT promote by recurrence — 'recurring ≥2
    episodes' now honestly means '≥2 sittings'. The SAME lesson in a second sitting (2
    distinct provenances) DOES promote (the positive control)."""
    from localharness.core.events import Observation
    from localharness.memory.gate import WriteGate

    gate = WriteGate(store, _NullBus(), "cons-agent")

    async def cycle(sess: str):
        await gate._on_observation(Observation(
            agent_id="cons-agent", session_id=sess, observation_type="tool_result",
            tool_name="bash_exec", output="", error="permission denied: /etc/shadow"))
        await gate._on_observation(Observation(
            agent_id="cons-agent", session_id=sess, observation_type="tool_result",
            tool_name="bash_exec", output="ok", error=None))

    # Same sitting stumbling twice on the same lesson: one provenance → no recurrence.
    await cycle("sit-1")
    await cycle("sit-1")
    report = await ConsolidationPass(store, _cfg()).run()
    assert report.promoted == 0
    assert await _facts_like(store, "learned/bash_exec/%") == []

    # A SECOND sitting hits the same lesson → 2 distinct provenances → now it promotes.
    await cycle("sit-2")
    report2 = await ConsolidationPass(store, _cfg()).run()
    assert report2.promoted == 1
    assert report2.promoted_keys[0].startswith("learned/bash_exec/resolved_error/")
    assert "2 episodes" in (await store.get_fact(report2.promoted_keys[0])).value


@pytest.mark.asyncio
async def test_bus_publish_reaches_gate_composed(store: MemoryStore):
    """FLOW2-BUS-HOP closed: the whole delivery seam with REAL objects — bus.publish → the
    EventBus agent_id-filtered subscription → the gate handler → store — NOT a direct
    handler call. Also proves the filter hop itself: another agent's identical event is
    dropped at the subscription and never produces a fact."""
    from localharness.core.bus import EventBus
    from localharness.core.events import Observation
    from localharness.memory.gate import WriteGate

    bus = EventBus()
    gate = WriteGate(store, bus, "cons-agent")
    await gate.open()
    try:
        # A DIFFERENT agent's full error→success cycle must be filtered out at the
        # subscription (agent_id mismatch); if it leaked a fact under "other-sess" appears.
        for out, err in (("", "boom"), ("ok", None)):
            await bus.publish(Observation(
                agent_id="other-agent", session_id="other-sess", observation_type="tool_result",
                tool_name="bash_exec", output=out, error=err))
        assert await _facts_like(store, "gate/resolved_error/bash_exec/%") == []

        # The real hop for THIS agent: publish → filtered handler → gate writes the fact.
        for out, err in (("", "boom"), ("ok", None)):
            await bus.publish(Observation(
                agent_id="cons-agent", session_id="s1", observation_type="tool_result",
                tool_name="bash_exec", output=out, error=err))
        rows = await _facts_like(store, "gate/resolved_error/bash_exec/%")
        assert len(rows) == 1
        assert rows[0][1] == "s1"  # provenance == the resolving session
    finally:
        await gate.close()


@pytest.mark.asyncio
async def test_has_work_ignores_non_gate_pending_facts(store: MemoryStore):
    # minor 1: predgate/ surprising_failure telemetry carries pending_consolidation forever —
    # it never matches the gate/ promotion prefix and 36-04 drains it piggybacking on
    # real-work passes, so it must NEVER trip the staleness gate (Phase 36 DELIBERATELY does
    # not add surprising_failure to _has_work). correction_pending is the deliberate premise
    # change: 36's reconciliation consumer clears it, so it counts as work WHEN reconcile is
    # enabled (the new default) but not when disabled (old optimization preserved for non-36).
    from localharness.core.bus import EventBus

    off = ConsolidationScheduler(store, EventBus(), "cons-agent", _cfg(reconcile_enabled=False))
    on = ConsolidationScheduler(store, EventBus(), "cons-agent", _cfg(reconcile_enabled=True))

    await store.store_fact(
        key="predgate/surprising_failure/web_fetch/20260705", value="stat telemetry",
        tags=["gate", "tier:surprising_failure", "pending_consolidation"], confidence=0.646,
    )
    # surprising_failure alone is NEVER work — neither reconcile mode counts it (36-04 drains it).
    assert await off._has_work() is False
    assert await on._has_work() is False

    # A correction_pending quarantine row: work ONLY when reconcile is enabled (the consumer
    # must fire); with reconcile off the pre-36 staleness optimization holds exactly.
    await store.store_fact(
        key="correction/quarantine/abc123", value="quarantined user words",
        tags=["correction", "tier:correction_pending", "pending_consolidation"], confidence=0.65,
    )
    assert await off._has_work() is False   # reconcile disabled -> old behavior preserved
    assert await on._has_work() is True     # reconcile enabled -> the consumer fires (new premise)

    # A DISPUTED gate/-keyed row (correction supersede keeps the gate/ key but carries
    # tier:correction_pending, which promotion skips) is likewise the reconciler's job: NOT
    # work when disabled — never the un-untaggable "pins _has_work forever" anti-pattern.
    await store.store_fact(
        key="gate/resolved_error/mytool/lesson0/sess0", value="[disputed] old lesson text",
        tags=["gate", "tier:correction_pending", "pending_consolidation"], confidence=0.6,
    )
    assert await off._has_work() is False
    # a real gate/-keyed promotion candidate IS work regardless of reconcile mode.
    await store.store_fact(
        key="gate/resolved_error/mytool/lesson1/sess1", value="a real recurring lesson",
        tags=["gate", "tier:resolved_error", "pending_consolidation"], confidence=0.65,
    )
    assert await off._has_work() is True


# ---------------------------------------------------------------------------
# Phase 36 (the chapter-writer): the LOAD-BEARING composed proof — a single
# ConsolidationPass.run() with an LLM wired must write a chapter, reconcile a
# disputed fact (revert RESTORES the original), AND mine a personal fact, all
# three inside run() (must-have #4 / plan-check WARNING 1). The no-LLM path is
# provably inert (the deterministic core is byte-unchanged).
# ---------------------------------------------------------------------------

# Two `read` lessons sharing the salient tokens "absolute"/"resolved"/"path" — they cluster
# via the FTS relatedness signal (fixture-fake seed text; allowed in tests per 36-CONTEXT).
_READ_A = "The read tool returned FileNotFound on a relative path; retrying with the absolute path resolved it."
_READ_B = "The read tool raised a permission problem on a protected path; the absolute path form resolved it cleanly."
_ORIG_FACT = "user's preferred editor is neovim"          # the wrongly-disputed remembered fact
_SUNBURN = "i got super duper sunburnt today at the beach"  # the live personal-fact specimen (36-CONTEXT)


class _DispatchLLM:
    """One fake that dispatches on prompt content so a SINGLE pass drives all three idle steps:
    REVERT for the reconcile look (shape (a) prompt says "disputed"), a grounded TYPED atom for
    the miner (MOVE 2 prompt says "USER'S WORLD"), and a grounded corpus slice for the
    chapter-writer ("Write ONE")."""
    async def complete(self, prompt: str) -> str:
        if "disputed" in prompt:                      # reconciliation shape (a)
            return "REVERT"
        if "USER'S WORLD" in prompt:                  # MOVE 2 transcript mining (topic|claim|evidence)
            return "health | user got sunburnt today | super duper sunburnt today at the beach"
        if "Write ONE" in prompt:                     # chapter-writer — echo a verbatim corpus slice
            corpus = prompt.split("\n\n")[-1]
            lines = [ln for ln in corpus.splitlines() if ln.strip()]
            return " ".join(lines[0].split()[:12]) if lines else "chapter"
        return ""                                     # anything else: inert


async def _seed_cluster_lesson(store, tool, tier, body, sess, lesson):
    """A promoted lesson (learned/{tool}/{tier}/{lesson}, 0.8) PLUS its derived_from gate/ source
    carrying provenance=session — mirrors 36-01/36-04 seeding so the cluster's session spread is
    derivable from neighborhood() like the real graph. The gate lesson-part ("{lesson}src") differs
    from the learned suffix so _step_promote_recurring never re-promotes the seed (no collision)."""
    promoted = await store.store_fact(
        key=f"learned/{tool}/{tier}/{lesson}", value=body,
        tags=["consolidated", f"tier:{tier}"], confidence=0.8,
        source="consolidation", provenance="consolidated:1-episodes",
    )
    cand = await store.store_fact(
        key=f"gate/{tier}/{tool}/{lesson}src/{sess}",
        value=f"Tool `{tool}` episode in {sess}: {body}",
        tags=["gate", f"tier:{tier}", "pending_consolidation"],
        confidence=0.65, source="write_gate", provenance=sess,
    )
    await store.add_edge(promoted.id, cand.id, "derived_from")
    return promoted


async def _seed_sem(store, body, session, *, topic="topic", conf=0.65):
    import hashlib
    h = hashlib.sha1(body.strip().encode("utf-8")).hexdigest()[:8]
    return await store.store_fact(
        key=f"sem/{topic}/{h}", value=body, tags=["sem", "pending_consolidation"],
        confidence=conf, source="transcript_mining", provenance=session, node_kind="fact",
    )


async def _seed_phase36(store) -> str:
    """Seed the three specimens the composed pass must consume; returns the disputed key K."""
    from localharness.memory.predictive_write_gate import _DISPUTE_MARKER

    # (i) a stable 2-lesson cluster spanning 2 sittings (s1, s2).
    await _seed_sem(store, _READ_A, "s1")
    await _seed_sem(store, _READ_B, "s2")
    # (ii) a SHAPE (a) disputed fact: a clean antecedent THEN the exact gate supersede-wrap on the
    # SAME key, so get_fact_history holds a pre-dispute row and REVERT RESTORES (marker disappears).
    k = "recall/editor_pref"
    await store.store_fact(key=k, value=_ORIG_FACT, tags=["remember"], confidence=0.8)
    await store.store_fact(
        key=k, value=f"{_DISPUTE_MARKER} {_ORIG_FACT}",
        tags=["correction", "tier:correction_pending", "pending_consolidation"], confidence=0.6,
    )
    # (iii) a mineable transcript span past the (zero) mining watermark.
    await store.append_history({"v": 1, "agent_id": "cons-agent", "type": "user_message",
                                "id": "h1000", "session_id": "mine-sess", "ts": 1_000_000,
                                "content": _SUNBURN})
    return k


async def _schema_rows(store) -> list[tuple[str, str]]:
    async with store._db.execute(
        "SELECT key, value FROM facts WHERE agent_id = ? AND status = 'active' AND node_kind = 'schema'",
        (store._agent_id,),
    ) as cur:
        return list(await cur.fetchall())


@pytest.mark.asyncio
async def test_phase36_pass_writes_schema_reconciles_and_mines(store: MemoryStore):
    """ONE ConsolidationPass.run() with an LLM wired writes a chapter (rendered schemas-first),
    reverts+RESTORES a disputed fact, AND mines a personal fact — every new step proven wired
    inside run() (must-have #4; mining_enabled=True is the plan-check WARNING 1 assertion)."""
    from localharness.memory.predictive_write_gate import _DISPUTE_MARKER

    k = await _seed_phase36(store)
    cfg = _cfg(schema_writer_enabled=True, reconcile_enabled=True, mining_enabled=True)
    report = await ConsolidationPass(store, cfg, llm=_DispatchLLM()).run()

    # (1) SCHEMA — one chapter written and it renders in the "### Knowledge" section.
    assert report.schemas_written >= 1
    schemas = await _schema_rows(store)
    assert len(schemas) >= 1
    index = await store._render_memory_index(10)
    assert "### Knowledge" in index
    assert schemas[0][0] in index                       # the schema key routes in the index

    # (2) RECONCILE — the dispute is reverted and the ORIGINAL value is RESTORED (not cleared).
    assert report.reconciled >= 1
    restored = await store.get_fact(k)
    assert restored is not None
    assert restored.value == _ORIG_FACT                 # pre-dispute value back, verbatim
    assert _DISPUTE_MARKER not in restored.value        # the marker is gone

    # (3) MINE — a typed semantic atom is mined and written at 0.65 (searchable, sub-injection —
    # ambient status is EARNED by recurrence/membership); proves _step_mine is wired in run().
    assert report.mined >= 1
    mined = await store.query_facts(FactQuery(tags=["sem"]))
    assert mined and mined[0].key.startswith("sem/")
    assert mined[0].confidence == 0.65


@pytest.mark.asyncio
async def test_phase36_deterministic_pass_unchanged_without_llm(store: MemoryStore):
    """The same seed with llm=None: the three new steps are INERT (report fields 0, no schema
    written, the dispute stays quarantined, nothing mined) — the deterministic core is unchanged."""
    from localharness.memory.predictive_write_gate import _DISPUTE_MARKER

    k = await _seed_phase36(store)
    report = await ConsolidationPass(store, _cfg(), llm=None).run()

    assert report.schemas_written == 0
    assert report.reconciled == 0
    assert report.mined == 0
    assert await _schema_rows(store) == []                       # no chapter without an LLM
    assert (await store.get_fact(k)).value.startswith(_DISPUTE_MARKER)  # dispute untouched
    assert await store.query_facts(FactQuery(tags=["mined"])) == []     # nothing mined
