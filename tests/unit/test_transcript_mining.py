"""PGATE-03 mining half (mining.py) — the bounded transcript model-look that closes the
correction-recall gap the lexical tripwire structurally leaves (census ceiling 0.231) and
mines plain personal facts a colleague would remember (live specimen: the sunburn line).

These lock the load-bearing properties (CONTEXT ruling 3 — transcript-mined writes are
injectable, budgeted, and grounded):
  1. a grounded personal fact is written INJECTABLE (>=0.7, node_kind='fact') with span
     provenance, and the watermark advances to the newest ts seen;
  2. an ungrounded line (a hallucinated detail) is REJECTED — the number-provenance kill
     discipline extends to mined facts;
  3. a re-run with no new records writes nothing and the watermark HOLDS (cost is bounded
     per-window, never O(lifetime history));
  4. a SET cancel event returns fast, reports cancelled, writes nothing, and never hangs
     or raises (machine-safety: the box hard-hangs on unattended long-context prefill).
  5. [Task 2] the per-cycle write budget caps writes; a re-mined identical line supersedes
     (corroborates) rather than duplicating.
"""
import asyncio
from pathlib import Path

import pytest

from localharness.memory.consolidation import _get_meta
from localharness.memory.mining import _MINING_WATERMARK_KEY, MineReport, mine_transcript
from localharness.memory.sqlite import FactQuery, MemoryStore


@pytest.fixture
async def store(tmp_path: Path):
    s = MemoryStore(agent_id="mine-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


class _FakeLLM:
    """The precision instrument, stubbed — returns fixed lines regardless of prompt."""

    def __init__(self, text: str):
        self.text = text

    async def complete(self, prompt: str) -> str:
        return self.text


class _SlowLLM:
    """A generation that never finishes on its own — proves cancel doesn't hang."""

    def __init__(self, delay: float = 10.0):
        self.delay = delay

    async def complete(self, prompt: str) -> str:
        await asyncio.sleep(self.delay)
        return "slow result"  # pragma: no cover — must be cancelled first


def _rec(ts: int, content: str, sid: str = "s1", typ: str = "user_message") -> dict:
    """A history.jsonl-shaped record (the fields HistoryWriter requires + content)."""
    return {"v": 1, "agent_id": "mine-agent", "type": typ, "id": f"h{ts}",
            "session_id": sid, "ts": ts, "content": content}


async def _seed_sunburn(store: MemoryStore) -> None:
    # The live personal-fact specimen. Single-word 'sunburnt' (an 8-char token) is used so
    # the grounding gate is exercised on a REAL >=6-char token match, not a vacuous
    # empty-token pass — the write must earn its grounding.
    await store.append_history(_rec(10, "i got super duper sunburnt today at the beach"))
    await store.append_history(_rec(20, "ok noted — hope it heals soon", typ="assistant_message"))


@pytest.mark.asyncio
async def test_personal_fact_written_injectable_and_watermark_advances(store: MemoryStore):
    """A grounded personal fact writes at INJECTABLE confidence (0.7, node_kind='fact')
    with span provenance, and the watermark advances to the newest ts seen (20)."""
    await _seed_sunburn(store)
    report = await mine_transcript(store, _FakeLLM("user got sunburnt today"), asyncio.Event())

    assert isinstance(report, MineReport)
    assert report.written == 1
    assert report.cancelled is False

    mined = await store.query_facts(FactQuery(tags=["mined"]))
    assert len(mined) == 1
    f = mined[0]
    assert f.value == "user got sunburnt today"
    assert f.confidence == 0.7  # injectable — CONTEXT ruling 3, NOT sub-0.7
    assert f.node_kind == "fact"
    assert f.provenance.startswith("mined-from:")
    assert "pending_consolidation" in f.tags

    assert int(await _get_meta(store, _MINING_WATERMARK_KEY)) == 20


@pytest.mark.asyncio
async def test_ungrounded_line_is_rejected_not_written(store: MemoryStore):
    """A line whose >=6-char token ('lottery') is absent from the transcript span is
    REJECTED — no un-derivable token enters a mined fact (kill discipline)."""
    await _seed_sunburn(store)
    report = await mine_transcript(store, _FakeLLM("user won the lottery"), asyncio.Event())

    assert report.written == 0
    assert report.rejected_ungrounded == 1
    assert await store.query_facts(FactQuery(tags=["mined"])) == []


@pytest.mark.asyncio
async def test_watermark_holds_on_rerun_without_new_records(store: MemoryStore):
    """A second pass with no post-watermark records writes nothing and the watermark HOLDS
    — cost is per-window, never a full re-mine of the growing history."""
    await _seed_sunburn(store)
    llm = _FakeLLM("user got sunburnt today")
    first = await mine_transcript(store, llm, asyncio.Event())
    assert first.written == 1
    wm = int(await _get_meta(store, _MINING_WATERMARK_KEY))

    second = await mine_transcript(store, llm, asyncio.Event())
    assert second.written == 0
    assert int(await _get_meta(store, _MINING_WATERMARK_KEY)) == wm  # held


@pytest.mark.asyncio
async def test_set_cancel_event_reports_cancelled_without_hanging(store: MemoryStore):
    """A pre-SET cancel event (a user turn already waiting) returns fast, reports
    cancelled, writes nothing, and does NOT advance the watermark (next pass re-mines)."""
    await _seed_sunburn(store)
    cancel = asyncio.Event()
    cancel.set()

    report = await asyncio.wait_for(
        mine_transcript(store, _SlowLLM(delay=10.0), cancel), timeout=3.0
    )

    assert report.cancelled is True
    assert report.written == 0
    assert await store.query_facts(FactQuery(tags=["mined"])) == []
    assert await _get_meta(store, _MINING_WATERMARK_KEY) is None  # watermark untouched
