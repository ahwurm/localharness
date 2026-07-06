"""MOVE 2 mining — the PRIMARY semantic feeder (mining.py). Domain knowledge about the user's
world enters the semantic hierarchy here as TYPED atoms: `topic | claim | evidence` ->
`sem/{topic-slug}/{h8(claim)}` at 0.65, node_kind='fact', provenance = the SOURCE record's
session (PER ATOM), grounded against its cited record.

These lock the load-bearing properties:
  1. a grounded typed atom writes at 0.65 (searchable, sub-injection) under sem/{slug}/, tags
     ['sem','pending_consolidation'], provenance = the source record's session_id;
  2. an ungrounded atom (a token in no source record) is REJECTED and counted — the kill net;
  3. PER-ATOM provenance (research doc §4): two atoms from two different sessions carry the two
     DISTINCT sessions, never one batch provenance — the ≥2-session cluster stability bar needs it;
  4. the CHUNKED WALK covers the ENTIRE un-mined window in one pass (an atom grounded only in the
     LAST chunk is still mined, and the watermark advances to the final record — not one nibble);
  5. a SET cancel returns fast, reports cancelled, writes nothing, does not advance the watermark;
  6. the per-pass write budget caps writes; a re-mined identical claim corroborates (no duplicate).
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
    """The precision instrument, stubbed — returns fixed `topic | claim | evidence` lines."""

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
        return "topic | slow | slow"  # pragma: no cover — must be cancelled first


def _rec(ts: int, content: str, sid: str = "s1", typ: str = "user_message") -> dict:
    return {"v": 1, "agent_id": "mine-agent", "type": typ, "id": f"h{ts}",
            "session_id": sid, "ts": ts, "content": content}


async def _seed_sunburn(store: MemoryStore) -> None:
    await store.append_history(_rec(10, "i got super duper sunburnt today at the beach", sid="beach-day"))
    await store.append_history(_rec(20, "ok noted — hope it heals soon", typ="assistant_message", sid="beach-day"))


@pytest.mark.asyncio
async def test_typed_atom_written_semantic_with_source_provenance(store: MemoryStore):
    """A grounded typed atom writes at 0.65 under sem/{slug}/{h8}, node_kind='fact', tags
    ['sem',...], provenance = the SOURCE record's session ('beach-day'), watermark -> 20."""
    await _seed_sunburn(store)
    llm = _FakeLLM("health | user got sunburnt today | super duper sunburnt today at the beach")
    report = await mine_transcript(store, llm, asyncio.Event())

    assert isinstance(report, MineReport)
    assert report.written == 1 and report.cancelled is False

    mined = await store.query_facts(FactQuery(tags=["sem"]))
    assert len(mined) == 1
    f = mined[0]
    assert f.value == "user got sunburnt today"
    assert f.key.startswith("sem/health/")
    assert f.confidence == 0.65          # searchable, sub-injection — ambient status is EARNED
    assert f.node_kind == "fact"
    assert f.provenance == "beach-day"   # the SOURCE record's session, PER ATOM
    assert "pending_consolidation" in f.tags

    assert int(await _get_meta(store, _MINING_WATERMARK_KEY)) == 20


@pytest.mark.asyncio
async def test_ungrounded_atom_is_rejected_not_written(store: MemoryStore):
    """An atom whose claim token ('lottery') is in no source record is REJECTED and counted."""
    await _seed_sunburn(store)
    llm = _FakeLLM("luck | user won the lottery jackpot | i got super duper sunburnt")
    report = await mine_transcript(store, llm, asyncio.Event())

    assert report.written == 0
    assert report.rejected_ungrounded == 1
    assert await store.query_facts(FactQuery(tags=["sem"])) == []


@pytest.mark.asyncio
async def test_per_atom_provenance_is_source_session(store: MemoryStore):
    """Load-bearing (research doc §4): two atoms mined from two DIFFERENT sessions carry the two
    distinct sessions — NOT one batch provenance — so the ≥2-session cluster stability bar can
    ever be met. A batch-level provenance (the SEMA-05 defect) would collapse both onto one."""
    await store.append_history(_rec(10, "i am building a summarizer subagent for the harness", sid="mon"))
    await store.append_history(_rec(20, "i am adding a citation subagent to the harness", sid="tue"))
    llm = _FakeLLM(
        "subagents | building a summarizer subagent for the harness | summarizer subagent\n"
        "subagents | adding a citation subagent to the harness | citation subagent"
    )
    report = await mine_transcript(store, llm, asyncio.Event())

    assert report.written == 2
    provs = {f.provenance for f in await store.query_facts(FactQuery(tags=["sem"]))}
    assert provs == {"mon", "tue"}       # two distinct source sessions, not a single mining batch


@pytest.mark.asyncio
async def test_chunked_walk_covers_the_full_window(store: MemoryStore):
    """The walk mines the ENTIRE un-mined window in one pass (a loop of chunks, not one nibble):
    an atom grounded ONLY in the LAST record is still written and the watermark reaches it."""
    for ts in range(1, 6):
        await store.append_history(_rec(ts * 10, f"early filler record number {ts} about nothing", sid="s1"))
    await store.append_history(_rec(100, "the user prefers the ristretto espresso blend", sid="s1"))
    # Grounds only in the LAST record; corpus_char_cap tiny -> forces a multi-chunk walk.
    llm = _FakeLLM("coffee | user prefers the ristretto espresso blend | ristretto espresso blend")
    report = await mine_transcript(store, llm, asyncio.Event(), corpus_char_cap=80)

    assert report.written == 1
    mined = await store.query_facts(FactQuery(tags=["sem"]))
    assert mined and mined[0].value == "user prefers the ristretto espresso blend"
    assert int(await _get_meta(store, _MINING_WATERMARK_KEY)) == 100  # walked to the last chunk


@pytest.mark.asyncio
async def test_watermark_holds_on_rerun_without_new_records(store: MemoryStore):
    await _seed_sunburn(store)
    llm = _FakeLLM("health | user got sunburnt today | sunburnt today at the beach")
    first = await mine_transcript(store, llm, asyncio.Event())
    assert first.written == 1
    wm = int(await _get_meta(store, _MINING_WATERMARK_KEY))

    second = await mine_transcript(store, llm, asyncio.Event())
    assert second.written == 0
    assert int(await _get_meta(store, _MINING_WATERMARK_KEY)) == wm  # held


@pytest.mark.asyncio
async def test_set_cancel_event_reports_cancelled_without_hanging(store: MemoryStore):
    await _seed_sunburn(store)
    cancel = asyncio.Event()
    cancel.set()

    report = await asyncio.wait_for(
        mine_transcript(store, _SlowLLM(delay=10.0), cancel), timeout=3.0
    )

    assert report.cancelled is True
    assert report.written == 0
    assert await store.query_facts(FactQuery(tags=["sem"])) == []
    assert await _get_meta(store, _MINING_WATERMARK_KEY) is None  # watermark untouched


@pytest.mark.asyncio
async def test_write_budget_caps_writes_per_pass(store: MemoryStore):
    """10 grounded atoms with write_budget=5 write at MOST 5 this pass; the budget-truncated
    chunk does NOT advance the watermark (the rest are re-mined next pass, corroborating)."""
    await store.append_history(
        _rec(5, "the user reported several distinct preference items worth remembering", sid="s1")
    )
    llm = _FakeLLM("\n".join(
        f"prefs | user reported preference item {i} worth remembering | reported several distinct preference"
        for i in range(10)
    ))
    report = await mine_transcript(store, llm, asyncio.Event(), write_budget=5)

    assert report.written == 5
    assert len(await store.query_facts(FactQuery(tags=["sem"]))) == 5
    assert await _get_meta(store, _MINING_WATERMARK_KEY) is None  # not advanced past a capped chunk


@pytest.mark.asyncio
async def test_repeated_claim_corroborates_not_duplicates(store: MemoryStore):
    """A re-mined identical claim hits the SAME sem/ key -> corroboration, no duplicate row."""
    from localharness.memory.mining import _h8, _slug

    claim = "user reported a preference for dark mode"
    await store.append_history(_rec(5, "the user reported a preference for dark mode theme", sid="s1"))
    llm = _FakeLLM(f"prefs | {claim} | reported a preference for dark mode")

    r1 = await mine_transcript(store, llm, asyncio.Event())
    assert r1.written == 1

    await store.append_history(_rec(50, "the user reported a preference for dark mode theme", sid="s1"))
    r2 = await mine_transcript(store, llm, asyncio.Event())
    assert r2.written == 1

    history = await store.get_fact_history(f"sem/{_slug('prefs')}/{_h8(claim)}")
    assert len(history) == 1  # corroboration on repeat: identical value == no duplicate row


# ---------------------------------------------------------------------------
# Ruling 3 (run-2 forensics): corrections must SUPERSEDE stale atoms. Claim-hash keys
# (sem/{topic}/{h8(claim)}) can never collide, so the port-8000 atom stayed active beside the
# 8081 correction (B4 fail). The miner is shown the topic's existing active atoms in-prompt and
# may mark `replaces=<id>`; on replaces, store_fact writes the NEW value on the OLD key —
# supersede chain, history preserved. Grounding still applies.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replaces_marker_supersedes_stale_atom(store: MemoryStore):
    from localharness.memory.mining import _h8

    stale_claim = "vLLM server listens on port 8000"
    await store.append_history(
        _rec(10, "for reference: our vLLM server listens on port 8000. remember that", sid="day2")
    )
    r1 = await mine_transcript(
        store, _FakeLLM(f"gpu ops | {stale_claim} | server listens on port 8000"), asyncio.Event()
    )
    assert r1.written == 1
    old_key = f"sem/gpu-ops/{_h8(stale_claim)}"
    assert (await store.get_fact(old_key)).value == stale_claim

    prompts: list[str] = []

    class _CorrectingLLM:
        async def complete(self, prompt: str) -> str:
            prompts.append(prompt)
            return (
                "gpu ops | vLLM server moved to port 8081 | moved the vLLM server to port 8081 "
                f"| replaces={old_key}"
            )

    await store.append_history(
        _rec(20, "correction: we moved the vLLM server to port 8081, not 8000", sid="day3")
    )
    r2 = await mine_transcript(store, _CorrectingLLM(), asyncio.Event())
    assert r2.written == 1

    # The miner was shown the topic's existing active atoms in-prompt (id + claim, capped).
    assert any(old_key in p and stale_claim in p for p in prompts), (
        "existing active atoms never reached the miner's prompt"
    )

    # Exactly ONE active port atom — the corrected value, ON THE OLD KEY (supersede chain).
    active = await store.query_facts(FactQuery(tags=["sem"]))
    port_atoms = [f for f in active if "port" in f.value]
    assert len(port_atoms) == 1, f"stale atom still active beside the correction: {port_atoms}"
    assert port_atoms[0].value == "vLLM server moved to port 8081"
    assert port_atoms[0].key == old_key
    assert port_atoms[0].provenance == "day3"  # per-atom source provenance, as ever

    history = await store.get_fact_history(old_key)
    assert len(history) == 2  # supersede chain intact: 8000 -> 8081
    assert any(h.value == stale_claim and h.status == "superseded" for h in history)
