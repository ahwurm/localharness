"""#90 — the turn-end micro-pass: a bounded tail-work drain that fires after a turn's answer is
delivered, so the naming/promotion/backfill tail (starved by cancellation in the big idle pass)
actually runs.

Units per firing, oldest-first, in atomic units (a cancel between units loses nothing):
  1. heal legacy #88 bucket conflicts + backfill-classify up to 5 untagged atoms (drains
     remember-legacy),
  2. NAME up to 2 eligible `proposed` discovery candidates (one model call each — the step that
     never runs live),
  3. promotion + stale-candidate prune (pure SQL, no model calls).
Hard wall-clock budget; cancellation reuses the cancel-on-user-activity machinery. Config-off
restores today's behavior. No live model needed (fake classifier/namer + injected clock).
"""
import hashlib
import time

import pytest

from localharness.config.models import MemoryConsolidationConfig
from localharness.core.bus import EventBus
from localharness.core.events import TurnEndMicroPassCompleted
from localharness.memory.consolidation import (
    ConsolidationPass,
    ConsolidationScheduler,
    TurnEndMicroPass,
)
from localharness.memory.discovery import _NAME_MARKER
from localharness.memory.sqlite import MemoryStore
from localharness.memory.tag_classify import _BUCKET_MARKER, _CHILD_MARKER


@pytest.fixture
async def store(tmp_path):
    s = MemoryStore(agent_id="micro-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


def _cfg(**over) -> MemoryConsolidationConfig:
    base = dict(enabled=True, idle_minutes=0.5, tag_discovery_enabled=False,
                turn_end_micro_pass_enabled=True, turn_end_micro_pass_budget_seconds=60.0)
    base.update(over)
    return MemoryConsolidationConfig(**base)


class _ManualClock:
    def __init__(self, start=0.0):
        self.t = start

    def __call__(self):
        return self.t


class _FakeLLM:
    """Prompt-aware classifier + namer. Optionally advances a clock per call to model budget spend."""

    def __init__(self, *, bucket="project", child="ops", name="hardware", clock=None, per_call=0.0):
        self.bucket, self.child, self.name = bucket, child, name
        self.clock, self.per_call = clock, per_call
        self.calls = 0

    async def complete(self, prompt: str) -> str:
        self.calls += 1
        if self.clock is not None:
            self.clock.t += self.per_call
        if _BUCKET_MARKER in prompt:
            return self.bucket
        if _CHILD_MARKER in prompt:
            return self.child
        if _NAME_MARKER in prompt:
            return self.name
        return ""


def _h8(s):
    return hashlib.sha1(s.encode()).hexdigest()[:8]


async def _seed_untagged(store, topic, value, *, conf=0.65, sess="s1"):
    """A pool-visible sem/ atom with NO bucket tag — a backfill-classify target."""
    return await store.store_fact(key=f"sem/{topic}/{_h8(value)}", value=value,
                                  tags=["sem", "pending_consolidation"], confidence=conf,
                                  source="transcript_mining", provenance=sess, node_kind="fact")


async def _seed_eligible_candidate(store, suffix, bodies, sittings, now):
    """A `proposed` discovered candidate whose STORED evidence already clears the incorporation
    ladder (>=2 members across >=2 sittings, score>=3) but was never named (cancellation)."""
    project = await store.get_tag("project")
    cand = await store.create_tag(f"cand-{suffix}", "discovery candidate (unincorporated)",
                                  parent_id=project.id, status="proposed", origin="discovered")
    for body, sess in zip(bodies, sittings):
        f = await store.store_fact(key=f"sem/c{suffix}/{_h8(body)}", value=body,
                                   tags=["sem", "pending_consolidation"], confidence=0.65,
                                   source="transcript_mining", provenance=sess, node_kind="fact")
        await store.add_bucket_tag(f.id, project.id, "mint")
        await store.add_atom_tag(f.id, cand.id, "discovery")
    await store.bump_tag_evidence(cand.id, distinct_sittings=len(set(sittings)),
                                  reuse_count=2, last_accrual_ts=now)
    return cand


async def _seed_recurring_candidate(store, tool, detail):
    """Two gate candidates (same lesson, 2 sessions) — a promotion-eligible pair for _step_promote."""
    from localharness.memory.gate import _h8 as _gh8
    for sess in ("gsess-1", "gsess-2"):
        await store.store_fact(
            key=f"gate/resolved_error/{tool}/{_gh8(tool, detail)}/{_gh8(sess)}",
            value=f"`{tool}` failed then succeeded: {detail}",
            tags=["gate", "tier:resolved_error", "pending_consolidation"],
            confidence=0.65, source="write_gate", provenance=sess)


async def _bucket_names(store, atom_id):
    return {t.name for t in await store.tags_for_atom(atom_id) if t.parent_id is None}


# ---------------------------------------------------------------------------
# Unit 1 — backfill-classify + #88 heal
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backfill_classifies_untagged_and_caps_at_5(store):
    ids = [(await _seed_untagged(store, "t", f"untagged atom number {i}")).id for i in range(7)]
    report = await TurnEndMicroPass(store, _cfg(), llm=_FakeLLM()).run()
    assert report.classified == 5                                # cap 5
    # oldest-first: the 5 lowest-id atoms got a bucket; the 2 newest stayed untagged.
    for a in ids[:5]:
        assert await _bucket_names(store, a) == {"project"}
    for a in ids[5:]:
        assert await _bucket_names(store, a) == set()


@pytest.mark.asyncio
async def test_micro_heals_legacy_bucket_conflict(store):
    f = await _seed_untagged(store, "h", "an atom with two buckets")
    ids = {b.name: b.id for b in await store.buckets()}
    for tag_id, ts in ((ids["personal"], 1000), (ids["project"], 1001)):
        await store._db.execute("INSERT INTO atom_tags (atom_id, tag_id, provenance, ts) VALUES (?,?, 'mint', ?)",
                                (f.id, tag_id, ts))
    await store._db.commit()

    report = await TurnEndMicroPass(store, _cfg(), llm=_FakeLLM()).run()
    assert report.bucket_conflicts_healed == 1
    assert await _bucket_names(store, f.id) == {"personal"}      # collapsed to the earliest


# ---------------------------------------------------------------------------
# Unit 2 — name eligible discovery candidates (the step that never runs live)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_names_eligible_candidate_and_replaces_placeholder(store):
    now = int(time.time())
    cand = await _seed_eligible_candidate(
        store, "hw", ["the gpu runs at 119 GiB unified memory", "the gb10 blackwell chip is fast"],
        ["s1", "s2"], now)

    report = await TurnEndMicroPass(store, _cfg(), llm=_FakeLLM(name="hardware")).run()
    assert report.named == 1
    healed = await store.get_tag_by_id(cand.id)
    assert healed.status == "active" and healed.name == "hardware"
    assert healed.definition != "discovery candidate (unincorporated)"   # placeholder replaced


@pytest.mark.asyncio
async def test_name_cap_is_2(store):
    now = int(time.time())
    for i in range(3):
        await _seed_eligible_candidate(store, f"c{i}", [f"body {i}a about x", f"body {i}b about x"],
                                       ["s1", "s2"], now)
    report = await TurnEndMicroPass(store, _cfg(), llm=_FakeLLM(name="topicname")).run()
    assert report.named == 2                                     # cap 2 model-name calls


@pytest.mark.asyncio
async def test_under_evidenced_candidate_not_named(store):
    """Evidence rules unchanged: a candidate that has NOT cleared the ladder is left alone."""
    project = await store.get_tag("project")
    cand = await store.create_tag("cand-weak", "discovery candidate (unincorporated)",
                                  parent_id=project.id, status="proposed", origin="discovered")
    f = await _seed_untagged(store, "weak", "a lone atom")
    await store.add_atom_tag(f.id, cand.id, "discovery")         # 1 member, no evidence bump
    report = await TurnEndMicroPass(store, _cfg(), llm=_FakeLLM()).run()
    assert report.named == 0
    assert (await store.get_tag_by_id(cand.id)).status == "proposed"


# ---------------------------------------------------------------------------
# Unit 3 — promotion + prune (pure SQL, no model call)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pure_sql_units_run_without_llm(store):
    """heal + promote + prune run with NO llm (pure SQL); classify/name are skipped."""
    # a promotable recurring pair
    await _seed_recurring_candidate(store, "bash_exec", "permission denied")
    # a stale prunable candidate
    project = await store.get_tag("project")
    old = await store.create_tag("cand-stale", "discovery candidate (unincorporated)",
                                 parent_id=project.id, status="proposed", origin="discovered")
    await store.bump_tag_evidence(old.id, distinct_sittings=0, reuse_count=0,
                                  last_accrual_ts=int(time.time()) - 30 * 86400)  # 30d stale

    report = await TurnEndMicroPass(store, _cfg(), llm=None).run()
    assert report.promoted == 1
    assert report.pruned == 1
    assert report.classified == 0 and report.named == 0
    assert (await store.get_tag_by_id(old.id)).status == "retired"


# ---------------------------------------------------------------------------
# Budget + cancellation (atomic units — nothing half-done)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_budget_stops_work_cleanly(store):
    for i in range(5):
        await _seed_untagged(store, "b", f"budget atom {i}")
    clock = _ManualClock()
    # each model call burns 3s; budget 8s → only the first atom or two are reached, then a clean stop.
    llm = _FakeLLM(clock=clock, per_call=3.0)
    report = await TurnEndMicroPass(store, _cfg(turn_end_micro_pass_budget_seconds=8.0),
                                    llm=llm, clock=clock).run()
    assert report.classified < 5                                 # did not process the full cap
    assert report.budget_spent_s >= 8.0                          # stopped at/after the budget


@pytest.mark.asyncio
async def test_cancel_between_units_commits_only_finished_work(store):
    ids = [(await _seed_untagged(store, "x", f"cancel atom {i}")).id for i in range(4)]
    micro = TurnEndMicroPass(store, _cfg(), llm=_FakeLLM())

    # A classifier that cancels the pass the instant it files the FIRST atom's bucket.
    class _CancelAfterFirst:
        def __init__(self, m):
            self.m, self.filed = m, 0

        async def complete(self, prompt):
            if _BUCKET_MARKER in prompt:
                return "project"
            if _CHILD_MARKER in prompt:
                self.filed += 1
                if self.filed == 1:
                    self.m.cancel()                              # user activity mid-drain
                return "ops"
            return ""

    micro._llm = _CancelAfterFirst(micro)
    report = await micro.run()
    assert report.cancelled is True
    assert report.classified == 1                                # exactly the finished atom
    assert await _bucket_names(store, ids[0]) == {"project"}     # fully filed, not half
    for a in ids[1:]:
        assert await _bucket_names(store, a) == set()            # rest untouched
    assert report.named == 0 and report.promoted == 0            # later units skipped by the cancel


@pytest.mark.asyncio
async def test_config_off_makes_run_inert(store):
    for i in range(3):
        await _seed_untagged(store, "off", f"atom {i}")
    report = await TurnEndMicroPass(store, _cfg(turn_end_micro_pass_enabled=False), llm=_FakeLLM()).run()
    assert (report.classified, report.named, report.promoted, report.pruned,
            report.bucket_conflicts_healed) == (0, 0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Scheduler wiring (turn-end trigger, no-double-run, cancel-on-activity, event)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_turn_end_launches_micro_and_emits_event(store):
    await _seed_untagged(store, "e", "an atom to drain on turn end")
    seen = []

    async def _cap(e):
        seen.append(e)

    bus = EventBus()
    bus.subscribe(TurnEndMicroPassCompleted, _cap)
    sched = ConsolidationScheduler(store, bus, "micro-agent", _cfg(), llm=_FakeLLM())

    await sched._on_turn_ended(None)
    assert sched._micro_task is not None
    await sched._micro_task
    assert len(seen) == 1 and seen[0].classified == 1


@pytest.mark.asyncio
async def test_config_off_scheduler_launches_no_micro(store):
    sched = ConsolidationScheduler(store, EventBus(), "micro-agent",
                                   _cfg(turn_end_micro_pass_enabled=False), llm=_FakeLLM())
    await sched._on_turn_ended(None)
    assert sched._micro_task is None


@pytest.mark.asyncio
async def test_micro_skips_while_full_pass_active(store):
    import asyncio
    sched = ConsolidationScheduler(store, EventBus(), "micro-agent", _cfg(), llm=_FakeLLM())
    sched._run_task = asyncio.create_task(asyncio.sleep(5))       # a full pass "in flight"
    try:
        await sched._on_turn_ended(None)
        assert sched._micro_task is None                         # deferred — no double-run
    finally:
        sched._run_task.cancel()


@pytest.mark.asyncio
async def test_full_pass_skips_while_micro_active(store):
    import asyncio
    sched = ConsolidationScheduler(store, EventBus(), "micro-agent", _cfg(), llm=_FakeLLM())
    sched._micro_task = asyncio.create_task(asyncio.sleep(5))     # a micro-pass in flight
    try:
        sched.launch()
        assert sched._run_task is None                           # full pass deferred
    finally:
        sched._micro_task.cancel()


@pytest.mark.asyncio
async def test_new_turn_cancels_running_micro(store):
    import asyncio
    for i in range(6):
        await _seed_untagged(store, "c", f"slow atom {i}")

    class _SlowLLM:
        async def complete(self, prompt):
            await asyncio.sleep(10.0)
            return "project"

    sched = ConsolidationScheduler(store, EventBus(), "micro-agent", _cfg(), llm=_SlowLLM())
    await sched._on_turn_ended(None)
    assert sched._micro_task is not None
    await asyncio.sleep(0.05)
    await sched._on_turn_started(None)                           # a new turn begins → cancel
    await asyncio.wait_for(sched._micro_task, timeout=3.0)       # finished promptly, no 10s hang
    assert sched._micro is None                                  # ran to its finally after the cancel
