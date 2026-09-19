"""Dreaming — the idle consolidation pass of the resonance rebuild.

What is under test:
- streams: closed turn windows off the session ledgers, incremental byte marks.
- digest: each window's one unit of attention distributes over facts by resonance
  share (zero-sum per moment); standing accumulates; an empty pass changes nothing.
- bind: co-fired above-uniform facts become a named group (fake LLM names it);
  a repeat observation strengthens evidence.
- embed-backfill: un-embedded facts get vectors; a changed model re-embeds all.
- settle: writer paid/lost tallies recompute from the store's own tables.
- forget: gated OFF by default; ON archives below the store's own line.
- scheduler: user activity cancels a running pass; disabled is inert; a turn in
  flight defers launches; _has_work sees unread ledger bytes and missing vectors.

The engine double is the dep-free HashingEmbedder behind the engine interface —
tests exercise the INTERFACE and the arithmetic, never the real model.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np
import pytest

from localharness.config.models import MemoryArchivalConfig, MemoryConsolidationConfig
from localharness.core.bus import EventBus
from localharness.memory import resonance as res
from localharness.memory.consolidation import (
    ConsolidationPass,
    ConsolidationScheduler,
    _get_meta,
)
from localharness.memory.embeddings import HashingEmbedder
from localharness.memory.sqlite import MemoryStore
from localharness.memory.streams import read_new_windows


class FakeEngine:
    model_name = "hash-fake"

    def __init__(self, dim: int = 64) -> None:
        self._h = HashingEmbedder(dim=dim)

    def embed_docs(self, texts):
        return np.asarray(self._h.embed(texts), dtype=np.float32)

    def embed_query(self, text):
        return np.asarray(self._h.embed([text])[0], dtype=np.float32)


class FakeLLM:
    def __init__(self, answer: str = "Test Topic") -> None:
        self.answer = answer
        self.calls: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.answer


@pytest.fixture
async def store(tmp_path: Path):
    s = MemoryStore(agent_id="orchestrator", division_id="default", org_id="default",
                    base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


def _cfg(**kw) -> MemoryConsolidationConfig:
    return MemoryConsolidationConfig(**kw)


def _write_session(agent_dir: Path, session_id: str, turns: list[list[dict]]) -> Path:
    """Write a session ledger: each turn = TurnStarted + its events + TurnCompleted."""
    sessions = agent_dir / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    path = sessions / f"{session_id}.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        for events in turns:
            fh.write(json.dumps({"event_type": "TurnStarted", "session_id": session_id}) + "\n")
            for ev in events:
                fh.write(json.dumps(ev) + "\n")
            fh.write(json.dumps({"event_type": "TurnCompleted", "session_id": session_id}) + "\n")
    return path


async def _seed(store: MemoryStore, engine: FakeEngine, key: str, value: str, **kw):
    return await store.store_fact(
        key, value, embedding=res.pack(engine.embed_docs([f"{key}: {value}"])[0]), **kw
    )


# --------------------------------------------------------------------- streams

def test_windows_close_on_completion_and_marks_advance(tmp_path: Path):
    agent_dir = tmp_path / "agents" / "orchestrator"
    _write_session(agent_dir, "s1", [
        [{"event_type": "UserMessage", "content": "hello world"}],
    ])
    # an OPEN second turn: started, no closer yet
    with open(agent_dir / "sessions" / "s1.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event_type": "TurnStarted", "session_id": "s1"}) + "\n")
        fh.write(json.dumps({"event_type": "UserMessage", "content": "still typing"}) + "\n")

    windows, marks = read_new_windows(agent_dir / "sessions", {})
    assert len(windows) == 1
    assert "hello world" in windows[0].text()

    # nothing new digested on a re-read from the marks; the open turn stays unread
    windows2, marks2 = read_new_windows(agent_dir / "sessions", marks)
    assert windows2 == [] and marks2 == marks

    # closing the open turn yields exactly it
    with open(agent_dir / "sessions" / "s1.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event_type": "TurnCompleted", "session_id": "s1"}) + "\n")
    windows3, _ = read_new_windows(agent_dir / "sessions", marks)
    assert len(windows3) == 1 and "still typing" in windows3[0].text()


def test_windows_max_bound_defers_the_tail(tmp_path: Path):
    agent_dir = tmp_path / "agents" / "orchestrator"
    _write_session(agent_dir, "s1", [
        [{"event_type": "UserMessage", "content": f"turn number {i}"}] for i in range(5)
    ])
    w1, marks = read_new_windows(agent_dir / "sessions", {}, max_windows=2)
    assert len(w1) == 2
    w2, _ = read_new_windows(agent_dir / "sessions", marks)
    assert len(w2) == 3  # the deferred tail, nothing lost, nothing doubled


# --------------------------------------------------------------------- shares

def test_shares_are_zero_sum_and_nonnegative():
    eng = FakeEngine()
    docs = eng.embed_docs(["banana smoothie recipe", "vllm server port config"])
    blobs = [(1, res.pack(docs[0])), (2, res.pack(docs[1]))]
    probe = eng.embed_query("smoothie with banana")
    sh = res.shares(probe, blobs)
    assert sh and abs(sum(sh.values()) - 1.0) < 1e-6
    assert all(v > 0 for v in sh.values())
    assert sh.get(1, 0.0) > sh.get(2, 0.0)  # the resonant trace wins the moment


# --------------------------------------------------------------------- digest

async def test_digest_accumulates_standing_and_is_idempotent_via_marks(store: MemoryStore):
    eng = FakeEngine()
    await _seed(store, eng, "recipes", "banana smoothie recipe with honey")
    await _seed(store, eng, "ops", "vllm server listens on port 8000")
    _write_session(store._agent_dir, "s1", [
        [{"event_type": "UserMessage", "content": "make me a banana smoothie recipe"}],
    ])
    p = ConsolidationPass(store, _cfg(), engine=eng)
    report = await p.run()
    assert report.windows_digested == 1
    recipes = await store.get_fact("recipes")
    ops = await store.get_fact("ops")
    assert recipes.standing > ops.standing >= 0.0

    # an empty pass forgets nothing and adds nothing — the store holds still
    before = (recipes.standing, ops.standing)
    report2 = await ConsolidationPass(store, _cfg(), engine=eng).run()
    assert report2.windows_digested == 0
    after = ((await store.get_fact("recipes")).standing, (await store.get_fact("ops")).standing)
    assert after == before


async def test_digest_without_engine_skips_gracefully(store: MemoryStore):
    await store.store_fact("k", "v", source="w")
    _write_session(store._agent_dir, "s1", [
        [{"event_type": "UserMessage", "content": "anything"}],
    ])
    report = await ConsolidationPass(store, _cfg(), engine=None).run()
    assert report.windows_digested == 0 and report.embedded_backfill == 0


# --------------------------------------------------------------------- bind

async def test_cofired_facts_bind_into_a_named_group(store: MemoryStore):
    eng = FakeEngine()
    await _seed(store, eng, "fruit-a", "banana smoothie recipe with honey")
    await _seed(store, eng, "fruit-b", "banana bread recipe with honey butter")
    await _seed(store, eng, "ops", "kernel scheduler tuning knobs sysctl")
    llm = FakeLLM("Banana Recipes")
    _write_session(store._agent_dir, "s1", [
        [{"event_type": "UserMessage", "content": "banana recipe with honey please"}],
    ])
    report = await ConsolidationPass(store, _cfg(), engine=eng, llm=llm).run()
    assert report.groups_observed == 1 and report.groups_named == 1
    groups = await store.list_groups(named_only=True)
    assert len(groups) == 1 and groups[0]["label"] == "Banana Recipes"
    member_keys = {(await store.get_fact_by_id(i)).key for i in groups[0]["member_ids"]}
    assert member_keys == {"fruit-a", "fruit-b"}

    # the same co-firing observed again strengthens the group, never duplicates it
    _write_session(store._agent_dir, "s2", [
        [{"event_type": "UserMessage", "content": "another banana recipe with honey"}],
    ])
    await ConsolidationPass(store, _cfg(), engine=eng, llm=llm).run()
    groups2 = await store.list_groups(named_only=True)
    assert len(groups2) == 1 and groups2[0]["evidence"] == 2


async def test_bind_without_llm_leaves_group_unnamed_and_unrendered(store: MemoryStore):
    eng = FakeEngine()
    await _seed(store, eng, "a", "banana smoothie recipe honey")
    await _seed(store, eng, "b", "banana bread recipe honey")
    await _seed(store, eng, "c", "kernel scheduler sysctl tuning")
    _write_session(store._agent_dir, "s1", [
        [{"event_type": "UserMessage", "content": "banana recipe honey"}],
    ])
    report = await ConsolidationPass(store, _cfg(), engine=eng, llm=None).run()
    assert report.groups_observed == 1 and report.groups_named == 0
    assert await store.list_groups(named_only=True) == []
    assert len(await store.list_groups()) == 1


# --------------------------------------------------------------------- backfill

async def test_backfill_embeds_missing_and_model_change_reembeds_all(store: MemoryStore):
    eng = FakeEngine()
    await store.store_fact("plain", "no vector at birth", source="w")
    report = await ConsolidationPass(store, _cfg(), engine=eng).run()
    assert report.embedded_backfill == 1
    assert await store.facts_missing_embedding() == []
    assert await _get_meta(store, "resonance/embed_model") == "hash-fake"

    class OtherEngine(FakeEngine):
        model_name = "hash-fake-v2"

    report2 = await ConsolidationPass(store, _cfg(), engine=OtherEngine()).run()
    assert report2.reembedded_all is True
    assert report2.embedded_backfill == 1
    assert await _get_meta(store, "resonance/embed_model") == "hash-fake-v2"


# --------------------------------------------------------------------- settle

async def test_settle_recomputes_paid_and_lost_from_the_stores_own_tables(store: MemoryStore):
    await store.store_fact("won", "recalled later", source="writer-a")
    await store.store_fact("lost", "never recalled", source="writer-a")
    await store.touch_staged(["won"])
    await store.fold_staged_access()
    lost = await store.get_fact("lost")
    await store.archive_fact(lost.id, surface="test", s_at_archive=0.0, line_at_archive=None)

    await ConsolidationPass(store, _cfg(), engine=None).run()
    async with store._db.execute(
        "SELECT paid, lost FROM writers WHERE agent_id = ? AND writer = ?",
        (store.agent_id, "writer-a"),
    ) as cur:
        paid, lost_n = await cur.fetchone()
    assert paid == 1 and lost_n == 1


async def test_writer_track_record_prices_birth_truth(store: MemoryStore):
    """A writer with no history births at log-odds 0 (confidence 0.5); a contradicted
    writer births lower; a confirmed writer births higher. Belief is earned."""
    f0 = await store.store_fact("first", "claim", source="w")
    assert f0.truth_logodds == 0.0 and abs(f0.confidence - 0.5) < 1e-9
    # a supersede by another writer tallies w contradicted
    await store.store_fact("first", "corrected claim", source="owner-hand")
    f1 = await store.store_fact("second", "new claim", source="w")
    assert f1.truth_logodds < 0.0  # precision (0+1)/(0+1+2) = 1/3 → negative log-odds


# --------------------------------------------------------------------- forget gate

async def test_forget_step_respects_the_archival_gate(store: MemoryStore):
    await store.store_fact("anchor", "recalled", source="w")
    await store.touch_staged(["anchor"])
    await store.fold_staged_access()
    a = await store.get_fact("anchor")
    await store._db.execute("UPDATE facts SET standing = 5.0 WHERE id = ?", (a.id,))
    await store._db.commit()
    await store.store_fact("junk", "never recalled", source="w")

    off = await ConsolidationPass(store, _cfg(), engine=None,
                                  archival=MemoryArchivalConfig(enabled=False)).run()
    assert off.archived == 0 and await store.get_fact("junk") is not None

    on = await ConsolidationPass(store, _cfg(), engine=None,
                                 archival=MemoryArchivalConfig(enabled=True)).run()
    assert on.archived == 1
    assert await store.get_fact("junk") is None
    assert await store.get_fact("anchor") is not None


# --------------------------------------------------------------------- scheduler

async def test_disabled_scheduler_is_inert(store: MemoryStore):
    bus = EventBus()
    sched = ConsolidationScheduler(store, bus, store.agent_id, _cfg(enabled=False))
    await sched.start()
    assert sched._timer_task is None and sched._handles == []
    await sched.stop()


async def test_user_activity_cancels_running_pass(store: MemoryStore):
    bus = EventBus()
    sched = ConsolidationScheduler(store, bus, store.agent_id, _cfg())
    sched._running = ConsolidationPass(store, _cfg(), engine=None)
    await sched._on_user_activity(object())
    assert sched._running.cancelled is True
    await sched.stop()


async def test_turn_in_flight_defers_launch(store: MemoryStore):
    bus = EventBus()
    sched = ConsolidationScheduler(store, bus, store.agent_id, _cfg())
    sched._turn_in_flight = True
    sched.launch()
    assert sched._run_task is None
    sched._turn_in_flight = False
    sched.launch()
    assert sched._run_task is not None
    await sched.stop()


async def test_has_work_sees_unread_ledger_and_missing_vectors(store: MemoryStore):
    bus = EventBus()
    sched = ConsolidationScheduler(store, bus, store.agent_id, _cfg())
    assert await sched._has_work() is False

    await store.store_fact("k", "v", source="w")     # un-embedded fact = work
    assert await sched._has_work() is True
    eng = FakeEngine()
    await ConsolidationPass(store, _cfg(), engine=eng).run()
    assert await sched._has_work() is False

    _write_session(store._agent_dir, "s1", [
        [{"event_type": "UserMessage", "content": "new experience"}],
    ])                                                # unread stream = work
    assert await sched._has_work() is True
    await ConsolidationPass(store, _cfg(), engine=eng).run()
    assert await sched._has_work() is False
    await sched.stop()


async def test_cancelled_pass_reports_cancelled(store: MemoryStore):
    p = ConsolidationPass(store, _cfg(), engine=None)
    p.cancel()
    report = await p.run()
    assert report.cancelled is True
