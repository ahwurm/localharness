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
import logging
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


# ---------------------------------------------------------------------------
# FIX 2/3 (run-3 forensics): the explicit `replaces=<id>` marker NEVER fired in run 3 — Qwen
# instead pasted a known atom's KEY into the `topic` field, which _slug() mangled into a shadow-
# duplicate ('sem/sem-vllm-port-…'), leaving the stale value active beside its correction (twice
# in 41 turns). FIX 2a coerces a key-shaped topic into an implied replaces (full-key) or a slug
# recovery (prefix); 2b shows short opaque ids + accepts replaces=aN; 2c persists the raw
# completion; FIX 3 nudges stable, reused topic labels.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_key_shaped_topic_coerces_to_supersede(store: MemoryStore):
    """FIX 2a (the run-3 port bug, id 70->74): the model pastes a known atom's FULL KEY into the
    topic field with a corrected value and NO replaces marker. The parser coerces that into an
    implied replaces → the correction supersedes on the OLD key; no shadow 'sem/sem-…' duplicate."""
    from localharness.memory.mining import _h8

    stale = "vLLM server listens on port 8000"
    await store.append_history(_rec(10, "for reference our vLLM server listens on port 8000", sid="day2"))
    r1 = await mine_transcript(
        store, _FakeLLM(f"vllm port | {stale} | server listens on port 8000"), asyncio.Event()
    )
    assert r1.written == 1
    old_key = f"sem/vllm-port/{_h8(stale)}"
    assert (await store.get_fact(old_key)).value == stale

    # Run-3 shape: old_key pasted into the TOPIC field, corrected value, NO replaces marker.
    await store.append_history(_rec(20, "correction we moved the vLLM server to port 8081 not 8000", sid="day3"))
    r2 = await mine_transcript(
        store,
        _FakeLLM(f"{old_key} | vLLM server moved to port 8081 | moved the vLLM server to port 8081"),
        asyncio.Event(),
    )
    assert r2.written == 1

    active = await store.query_facts(FactQuery(tags=["sem"]))
    assert not any(f.key.startswith("sem/sem-") for f in active), "shadow mangled-slug dup minted"
    port_atoms = [f for f in active if "port" in f.value]
    assert len(port_atoms) == 1, f"stale atom still active beside the correction: {port_atoms}"
    assert port_atoms[0].key == old_key
    assert port_atoms[0].value == "vLLM server moved to port 8081"

    history = await store.get_fact_history(old_key)
    assert len(history) == 2  # supersede chain 8000 -> 8081, history preserved
    assert any(h.value == stale and h.status == "superseded" for h in history)


@pytest.mark.asyncio
async def test_short_id_replaces_marker_accepted(store: MemoryStore):
    """FIX 2b: the miner is shown short opaque ids ([a1]) for known atoms and may emit
    `replaces=a1`; the parser resolves the short id to the atom's key and supersedes."""
    from localharness.memory.mining import _h8

    stale = "vLLM server listens on port 8000"
    await store.append_history(_rec(10, "our vLLM server listens on port 8000 for now", sid="day2"))
    r1 = await mine_transcript(
        store, _FakeLLM(f"vllm port | {stale} | server listens on port 8000"), asyncio.Event()
    )
    assert r1.written == 1
    old_key = f"sem/vllm-port/{_h8(stale)}"

    prompts: list[str] = []

    class _ShortIdLLM:
        async def complete(self, prompt: str) -> str:
            prompts.append(prompt)
            return ("vllm port | vLLM server moved to port 8081 | "
                    "moved the vLLM server to port 8081 | replaces=a1")

    await store.append_history(_rec(20, "we moved the vLLM server to port 8081 not 8000", sid="day3"))
    r2 = await mine_transcript(store, _ShortIdLLM(), asyncio.Event())
    assert r2.written == 1
    assert any("[a1]" in p for p in prompts), "short opaque id never reached the miner's prompt"

    port_atoms = [f for f in await store.query_facts(FactQuery(tags=["sem"])) if "port" in f.value]
    assert len(port_atoms) == 1
    assert port_atoms[0].key == old_key
    assert port_atoms[0].value == "vLLM server moved to port 8081"


@pytest.mark.asyncio
async def test_slug_prefix_topic_recovers_slug_without_superseding(store: MemoryStore):
    """FIX 2a (documented deviation from the literal spec, run-3 summarizer case id 57->62): a
    topic pasted as a known atom's `sem/<slug>` PREFIX is AMBIGUOUS and was a NEW fact on the
    topic, not a correction. Superseding the distinct existing atom would LOSE data, so we recover
    the real slug (killing the shadow-duplicate mangled key) but do NOT supersede — both stay
    active under the clean slug."""
    from localharness.memory.mining import _h8

    first = "first subagent is a summarizer that condenses long text files"
    await store.append_history(_rec(10, "the first subagent is a summarizer that condenses long text files", sid="day1"))
    r1 = await mine_transcript(
        store, _FakeLLM(f"summarizer subagent | {first} | summarizer that condenses long text"),
        asyncio.Event(),
    )
    assert r1.written == 1
    first_key = f"sem/summarizer-subagent/{_h8(first)}"
    assert (await store.get_fact(first_key)).value == first

    second = "summarizer subagent output is capped at 200 words"
    await store.append_history(_rec(20, "the summarizer subagent output is capped at 200 words per file", sid="day2"))
    r2 = await mine_transcript(
        store, _FakeLLM(f"sem/summarizer-subagent | {second} | summarizer subagent output is capped"),
        asyncio.Event(),
    )
    assert r2.written == 1

    active = await store.query_facts(FactQuery(tags=["sem"]))
    assert not any(f.key.startswith("sem/sem-") for f in active), "mangled shadow slug minted"
    assert {f.value for f in active} == {first, second}          # BOTH facts survive (no data loss)
    assert (await store.get_fact(first_key)).value == first      # original not superseded


@pytest.mark.asyncio
async def test_invalid_key_shaped_topic_falls_back_to_normal_mint(store: MemoryStore):
    """FIX 2a: a key-shaped topic resolving to NO known active atom is not coerced — it falls
    back to a normal mint (never a spurious supersede)."""
    await store.append_history(_rec(10, "the user prefers the ristretto espresso blend daily", sid="s1"))
    r = await mine_transcript(
        store,
        _FakeLLM("sem/ghost-topic/deadbeef | user prefers the ristretto espresso blend | ristretto espresso blend"),
        asyncio.Event(),
    )
    assert r.written == 1
    active = await store.query_facts(FactQuery(tags=["sem"]))
    assert len(active) == 1 and active[0].value == "user prefers the ristretto espresso blend"


@pytest.mark.asyncio
async def test_raw_completion_persisted_to_log(store: MemoryStore):
    """FIX 2c: run-3's raw miner completions were unrecoverable, making root-cause inferential.
    mine_transcript appends each chunk's RAW completion (pre-parse) to a caller-supplied log."""
    await _seed_sunburn(store)
    completions: list[dict] = []
    raw = "health | user got sunburnt today | super duper sunburnt today at the beach"
    await mine_transcript(store, _FakeLLM(raw), asyncio.Event(), completions_log=completions)
    assert completions and any(raw in c["raw"] for c in completions)


def test_prompt_nudges_stable_topic_reuse():
    """FIX 3: the miner must reuse a known topic's label verbatim when a claim continues it, and
    keep labels short/generic — the designed same-slug fast path was dead all of run 3 (0 shared
    slugs). Fully general: no manifest/eval vocabulary."""
    from localharness.memory.mining import _PROMPT

    p = _PROMPT.lower()
    assert "reuse" in p and "topic" in p
    assert "generic" in p or "short" in p
    for banned in ("manifest", "designed", "grading", "chapter"):
        assert banned not in p, f"eval-specific vocabulary leaked into the general prompt: {banned}"


# ---------------------------------------------------------------------------
# FIX 2 (run-10 forensics, ids 51/52): a SINGLE completion emitted the same corrected fact TWICE
# with two different replaces= targets, superseding BOTH stale atoms -> two active rows, identical
# value, same slug. The in-pass dedupe skips a mint whose (slug, normalized value) already landed
# on a DIFFERENT key this pass (or is already active on one) — same-KEY writes stay corroboration.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_double_replace_markers_dedupe_to_one_active_row(store: MemoryStore):
    """One completion, the same corrected value twice, two DISTINCT valid replaces targets on one
    slug: exactly ONE corrected row is active afterwards (the duplicate mint is skipped), and the
    second stale atom it pointed at is left untouched rather than clobbered into a duplicate."""
    from localharness.memory.mining import _h8

    await store.append_history(_rec(
        10, "our vLLM server listens on port 8000 and the GPU box has 119 GiB unified memory",
        sid="day2"))
    stale_a = "vLLM server listens on port 8000"
    stale_b = "GPU box has 119 GiB unified memory"
    r1 = await mine_transcript(store, _FakeLLM(
        f"gpu ops | {stale_a} | vLLM server listens on port 8000\n"
        f"gpu ops | {stale_b} | GPU box has 119 GiB unified memory"), asyncio.Event())
    assert r1.written == 2
    key_a = f"sem/gpu-ops/{_h8(stale_a)}"
    key_b = f"sem/gpu-ops/{_h8(stale_b)}"
    assert (await store.get_fact(key_a)).value == stale_a
    assert (await store.get_fact(key_b)).value == stale_b

    # One completion emits the SAME corrected fact twice, replacing BOTH known atoms (the run-10 shape).
    corrected = "vLLM server now runs on port 8081"
    await store.append_history(_rec(
        20, "correction: the vLLM server now runs on port 8081 after the move", sid="day3"))
    dup = (f"gpu ops | {corrected} | vLLM server now runs on port 8081 | replaces={key_a}\n"
           f"gpu ops | {corrected} | vLLM server now runs on port 8081 | replaces={key_b}")
    r2 = await mine_transcript(store, _FakeLLM(dup), asyncio.Event())

    active = await store.query_facts(FactQuery(tags=["sem"]))
    corrected_rows = [f for f in active if f.value == corrected]
    assert len(corrected_rows) == 1, f"duplicate corrected rows still active: {corrected_rows}"
    assert r2.written == 1  # the second identical atom was deduped, not minted


@pytest.mark.asyncio
async def test_dedupe_leaves_same_key_corroboration_untouched(store: MemoryStore):
    """The dedupe is KEY-AWARE: a re-mined identical claim lands on the SAME key (corroboration,
    not a duplicate) and must NOT be skipped — regression guard beside the run-10 fix."""
    from localharness.memory.mining import _h8, _slug

    claim = "user prefers the ristretto espresso blend"
    await store.append_history(_rec(5, "the user prefers the ristretto espresso blend daily", sid="s1"))
    llm = _FakeLLM(f"coffee | {claim} | prefers the ristretto espresso blend")
    assert (await mine_transcript(store, llm, asyncio.Event())).written == 1

    await store.append_history(_rec(50, "again the user prefers the ristretto espresso blend daily", sid="s1"))
    r2 = await mine_transcript(store, llm, asyncio.Event())
    assert r2.written == 1  # same-key corroboration is not a duplicate — it proceeds
    assert len(await store.get_fact_history(f"sem/{_slug('coffee')}/{_h8(claim)}")) == 1


# ---------------------------------------------------------------------------
# FIX 2 (extraction-yield): no per-chunk yield FLOOR/RETRY existed — a SUBSTANTIVE chunk that
# returns zero atoms (a flat refusal "I cannot extract any facts…", live in run-5 chunk4; or a bad
# roll) silently advanced the watermark with nothing re-mined. A zero-PARSE chunk with >=K records
# is re-mined exactly ONCE (bounded — never loops); a tiny (<K) chunk is legitimately empty and is
# NOT retried. Instrumented: a log line on every retry + per-chunk {atoms_yielded, retried} and the
# retry's raw kept in the completions log, so run-12 forensics separate coverage (FIX 1) from
# retry-recovered atoms (FIX 2). K = 3.
# ---------------------------------------------------------------------------


class _RefusalThenAtomsLLM:
    """A flat refusal on the FIRST mining look, real atoms on the re-mine. Mint-time tag-classify
    calls are answered inertly and NEVER counted as mining looks (keyed on the mining prompt)."""

    def __init__(self, atoms: str):
        self._atoms = atoms
        self.mining_calls = 0

    async def complete(self, prompt: str) -> str:
        if "Extract durable facts" not in prompt:  # a tag-classify call — inert, uncounted
            return "none"
        self.mining_calls += 1
        return ("I cannot extract any facts because the provided text lacks a transcript to analyze."
                if self.mining_calls == 1 else self._atoms)


class _AlwaysRefusesLLM:
    """Refuses every mining look — proves the re-mine is capped at ONE even when it also whiffs."""

    def __init__(self):
        self.mining_calls = 0

    async def complete(self, prompt: str) -> str:
        if "Extract durable facts" not in prompt:
            return "none"
        self.mining_calls += 1
        return "I cannot extract any facts; please provide the transcript."


@pytest.mark.asyncio
async def test_zero_yield_substantive_chunk_is_remined_once(store: MemoryStore, caplog):
    """A >=K-record chunk whose first look yields ZERO atoms (a refusal) is re-mined EXACTLY once;
    the retry's atoms land, a retry log line is emitted, and the completions log records the retried
    chunk (atoms_yielded>0, retried=True) AND keeps the retry's raw completion for forensics."""
    await store.append_history(_rec(10, "the user prefers the ristretto espresso blend daily", sid="s1"))
    await store.append_history(_rec(20, "some filler chatter about the weather outside today", sid="s1"))
    await store.append_history(_rec(30, "more idle filler conversation with no durable facts", sid="s1"))
    atoms = "coffee | user prefers the ristretto espresso blend | ristretto espresso blend"
    llm = _RefusalThenAtomsLLM(atoms)
    completions: list[dict] = []

    with caplog.at_level(logging.WARNING, logger="localharness.memory.mining"):
        report = await mine_transcript(store, llm, asyncio.Event(), completions_log=completions)

    assert llm.mining_calls == 2                    # first refusal + exactly one re-mine
    assert report.written == 1
    mined = await store.query_facts(FactQuery(tags=["sem"]))
    assert len(mined) == 1 and mined[0].value == "user prefers the ristretto espresso blend"
    # Instrumentation: a retried chunk that recovered atoms, and the retry's raw kept in the log.
    assert any(c.get("retried") and c.get("atoms_yielded", 0) > 0 for c in completions)
    assert any(atoms in c["raw"] for c in completions)      # retry raw preserved for forensics
    assert "re-mined zero-yield chunk" in caplog.text       # the per-retry attribution line


@pytest.mark.asyncio
async def test_tiny_empty_chunk_below_k_records_does_not_retry(store: MemoryStore):
    """A legitimately-empty chunk with FEWER than K records is NOT re-mined — the floor only fires
    on a substantive chunk, so tiny trailing chunks (run-5's 1-record refusal) don't burn a look."""
    await store.append_history(_rec(10, "short note about nothing much here today ok", sid="s1"))
    await store.append_history(_rec(20, "another brief filler line with no facts stated", sid="s1"))
    llm = _AlwaysRefusesLLM()

    report = await mine_transcript(store, llm, asyncio.Event())

    assert llm.mining_calls == 1                    # 2 records (<K) -> no re-mine
    assert report.written == 0


@pytest.mark.asyncio
async def test_zero_yield_retry_is_capped_at_one(store: MemoryStore):
    """A >=K-record chunk that whiffs on BOTH the first look AND the re-mine is not looked at a
    THIRD time — the retry is bounded to one (machine-safety: never loop on a stubborn refusal)."""
    for ts in (10, 20, 30):
        await store.append_history(_rec(ts, f"idle filler record at {ts} with no durable facts", sid="s1"))
    llm = _AlwaysRefusesLLM()

    report = await mine_transcript(store, llm, asyncio.Event())

    assert llm.mining_calls == 2                    # first look + exactly one re-mine, then proceed
    assert report.written == 0
