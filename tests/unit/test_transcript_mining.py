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


def _rec(ts: int, content: str, sid: str = "s1", typ: str = "user_message",
         tool_name: str | None = None) -> dict:
    r = {"v": 1, "agent_id": "mine-agent", "type": typ, "id": f"h{ts}",
         "session_id": sid, "ts": ts, "content": content}
    if tool_name is not None:
        r["tool_name"] = tool_name  # tool_result records carry the emitting tool's name
    return r


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

    class _PerCorpusLLM:
        """Models the real miner under FIX 3 per-session chunking: each session is shown its OWN
        corpus in a separate chunk, so it emits only the atom grounded in what that chunk contains
        (a fixed cross-session completion would half-ground the other session's atom onto the wrong
        source record, since 'subagent'/'harness' are shared — the very mis-attribution session
        chunking must avoid)."""
        async def complete(self, prompt: str) -> str:
            if "summarizer subagent for the harness" in prompt.rsplit("\n\n", 1)[-1]:
                return "subagents | building a summarizer subagent for the harness | summarizer subagent"
            return "subagents | adding a citation subagent to the harness | citation subagent"

    report = await mine_transcript(store, _PerCorpusLLM(), asyncio.Event())

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


# ---------------------------------------------------------------------------
# FIX 2 (production budget — owner "necessity"): the pass-wide write_budget aborts the walk once
# hit. STEP-1 finding (traced, not assumed): the watermark advances ONLY per fully-mined chunk and
# a budget-tripped chunk breaks BEFORE its commit, so the un-mined TAIL stays ts > watermark and the
# NEXT pass re-mines it — DEFERRED, never permanently lost (production is recurring; only the single-
# pass eval is lossy, which is why it overrides to 500). So the fix is (a) lock that no-loss property
# and (b) make a sane production budget expressible (the le=50 ceiling couldn't) with a higher default.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_abort_defers_tail_never_permanent_loss(store: MemoryStore):
    """FIX 2 necessity, multi-chunk: early chunks mint and advance the watermark; a later chunk trips
    the pass-wide budget and is NOT committed; a SECOND pass recovers the deferred tail. The watermark
    only ever advances over records actually mined — the tail is never silently dropped."""
    await store.append_history(_rec(10, "reminder the user prefers dark mode in the code editor", sid="s1"))
    await store.append_history(_rec(20, "for the record the user timezone is pacific standard time", sid="s1"))
    await store.append_history(_rec(30, "note the user drives a blue subaru outback wagon here", sid="s1"))
    # One fixed completion; per-chunk grounding admits only the atom whose source record is in-chunk.
    llm = _FakeLLM(
        "prefs | user prefers dark mode in the code editor | prefers dark mode\n"
        "timezone | user timezone is pacific standard time | pacific standard time\n"
        "vehicle | user drives a blue subaru outback wagon | blue subaru outback"
    )
    # cap=40 -> one record per chunk (3 chunks); budget=2. The fixed completion emits all 3 lines
    # every chunk, so chunk 2 trips the cap at its 3rd line (after prefs+timezone) -> chunk 2 is a
    # partially-mined chunk that is NOT committed; only chunk 1 committed (watermark = 10).
    r1 = await mine_transcript(store, llm, asyncio.Event(),
                               write_budget=2, corpus_char_cap=40, file_tags=False)
    assert r1.written == 2
    vals1 = {f.value for f in await store.query_facts(FactQuery(tags=["sem"]))}
    assert not any("subaru" in v for v in vals1)                    # tail NOT mined this pass
    assert int(await _get_meta(store, _MINING_WATERMARK_KEY)) == 10  # advanced only over the mined chunk

    # A second pass resumes at the deferred tail — the subaru fact is recovered, never lost (the
    # re-mined timezone atom corroborates onto its existing key, so written counts it too).
    await mine_transcript(store, llm, asyncio.Event(), corpus_char_cap=40, file_tags=False)
    vals2 = {f.value for f in await store.query_facts(FactQuery(tags=["sem"]))}
    assert any("subaru" in v for v in vals2)                        # tail recovered next pass
    assert int(await _get_meta(store, _MINING_WATERMARK_KEY)) == 30  # now walked to the end


def test_config_expresses_a_production_scale_mining_budget():
    """FIX 2: the le=50 ceiling couldn't express a sane production budget (the eval had to bypass the
    ctor to set 500). The raised ceiling makes a production-scale per-pass budget configurable, and the
    default is lifted off the run-6-starving 25; deferral (test above) stays the safety net past it."""
    from localharness.config.models import MemoryConsolidationConfig
    assert MemoryConsolidationConfig().mining_write_budget >= 50       # better-justified default (was 25)
    assert MemoryConsolidationConfig(mining_write_budget=200).mining_write_budget == 200  # expressible now


# ---------------------------------------------------------------------------
# FIX 3 (per-session chunking + configurable chunk size): today mine_transcript flattens ALL
# sittings by ts and walks a HARDCODED 6000-char window, so a chunk can straddle two sessions. 3a
# cuts chunks at session_id boundaries (session-by-session, chronological), sub-splitting an
# oversized session with the existing char-cap loop. 3b makes the chunk size a config knob. The
# critic's risk: +40% chunk count scrolls the known-atoms window (was 30) faster, so a same-pass
# `replaces=` targeting an atom minted many chunks earlier can fall OUT of the window and silently
# mint a shadow duplicate — the window is now a config knob defaulting >= the write budget.
# ---------------------------------------------------------------------------


class _CaptureLLM:
    """Records every prompt the miner builds; mints nothing (empty completion) so the known-block
    stays empty and each prompt's only content markers come from that chunk's corpus."""

    def __init__(self):
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return ""


@pytest.mark.asyncio
async def test_chunks_never_span_a_session_boundary(store: MemoryStore):
    """FIX 3a: two sittings interleaved in ts are mined session-by-session, so no single chunk's
    corpus ever straddles two sessions (today's flat ts-walk co-mingles all four records in one
    chunk). Sessions are walked in chronological (min-ts) order; both are covered."""
    await store.append_history(_rec(10, "alpha one the subagent design notes go here now", sid="sessA"))
    await store.append_history(_rec(20, "bravo one the kyoto trip planning notes are here", sid="sessB"))
    await store.append_history(_rec(30, "alpha two more subagent design details noted here", sid="sessA"))
    await store.append_history(_rec(40, "bravo two more kyoto trip planning details here", sid="sessB"))
    llm = _CaptureLLM()

    await mine_transcript(store, llm, asyncio.Event(), file_tags=False)

    assert llm.prompts, "the miner never ran"
    for p in llm.prompts:
        assert not ("alpha" in p and "bravo" in p), "a chunk straddled two sessions"
    assert any("alpha" in p for p in llm.prompts) and any("bravo" in p for p in llm.prompts)


def test_corpus_char_cap_is_configurable_with_default_6000():
    """FIX 3b: the mining chunk size is a config knob (was a hardcoded 6000), tunable for a later
    empirical sweep; the default preserves today's behaviour."""
    from localharness.config.models import MemoryConsolidationConfig
    assert MemoryConsolidationConfig().mining_corpus_char_cap == 6000
    assert MemoryConsolidationConfig(mining_corpus_char_cap=9000).mining_corpus_char_cap == 9000


@pytest.mark.asyncio
async def test_step_mine_threads_config_chunk_and_known_caps(store: MemoryStore, monkeypatch):
    """FIX 3b wiring (owner rule: prove the knob is WIRED, not merely present): _step_mine passes the
    config's corpus_char_cap AND known_atoms_cap into mine_transcript, not the hardcoded constants."""
    from localharness.config.models import MemoryConsolidationConfig
    from localharness.memory.consolidation import ConsolidationPass, ConsolidationReport

    captured: dict = {}

    async def _fake_mine(store_, llm_, cancel_, **kw):
        captured.update(kw)
        return MineReport()

    monkeypatch.setattr("localharness.memory.mining.mine_transcript", _fake_mine)
    cfg = MemoryConsolidationConfig(mining_corpus_char_cap=1234, mining_known_atoms_cap=42)
    await ConsolidationPass(store, cfg, llm=object())._step_mine(ConsolidationReport())

    assert captured.get("corpus_char_cap") == 1234
    assert captured.get("known_atoms_cap") == 42


@pytest.mark.asyncio
async def test_correction_supersedes_across_many_intervening_mints(store: MemoryStore):
    """FIX 3 (the critic's finding): per-session chunking multiplies chunk count, scrolling the
    known-atoms window faster. A same-pass `replaces=` targeting an atom minted many chunks earlier —
    with >30 OTHER atoms minted in between — must STILL supersede, not mint a shadow duplicate. The
    known-atoms cap now defaults >= the write budget so the target stays visible across the pass."""
    from localharness.memory.mining import _h8

    target_claim = "vLLM endpoint listens on port 8000"
    target_key = f"sem/gpu-ops/{_h8(target_claim)}"
    corrected = "vLLM moved to port 8081 instead"
    # ts=1 mints the target; ts=2..32 mint 31 UNRELATED fillers (>30 intervening); ts=100 corrects it.
    await store.append_history(_rec(1, "the vLLM endpoint listens on port 8000 noted", sid="s1"))
    for k in range(2, 33):
        await store.append_history(
            _rec(k, f"widgetnum{k} gadgetnum{k} are the two distinct markers here", sid="s1")
        )
    await store.append_history(_rec(100, "correction the vLLM moved to port 8081 instead now", sid="s1"))
    # One fixed completion carrying all lines; per-chunk grounding admits only the in-chunk atom, and
    # the correction only grounds against ts=100 (its salient token 'instead' lives nowhere else).
    lines = (
        [f"gpu ops | {target_claim} | endpoint listens on port 8000"]
        + [f"filler | widgetnum{k} gadgetnum{k} | widgetnum{k} gadgetnum{k}" for k in range(2, 33)]
        + [f"gpu ops | {corrected} | moved to port 8081 instead | replaces={target_key}"]
    )
    llm = _FakeLLM("\n".join(lines))
    # Tiny cap -> one record per chunk (33 chunks) so the known window scrolls between target and fix.
    await mine_transcript(store, llm, asyncio.Event(),
                          write_budget=100, corpus_char_cap=40, file_tags=False)

    active = await store.query_facts(FactQuery(tags=["sem"]))
    gpu = [f for f in active if f.key.startswith("sem/gpu-ops/")]
    assert len(gpu) == 1, f"correction did not supersede across the window (shadow dup): {gpu}"
    assert gpu[0].key == target_key and gpu[0].value == corrected


# ---------------------------------------------------------------------------
# FIX 4 (provenance-collapse guard — OPERATIVE-SURFACE allowlist): per-session chunking created a
# latent risk. When the agent RE-READS its own stored content, memory_search/memory_get echo a prior
# fact VERBATIM into a LATER session's tool_result. If mining walked that echo it would re-mine the
# fact and store_fact's distinct-day ladder would ADVANCE its provenance to the later session,
# collapsing the >=2-distinct-session evidence a chapter needs (and starving A1, which keys recall on
# provenance-day). The guard is STRUCTURAL, at INPUT CONSTRUCTION: mining fetches only the OPERATIVE
# CONVERSATIONAL SURFACE (user + assistant messages) via get_history(message_types=...), so tool I/O
# — every store read-back, plus any FUTURE echo tool — is never even read. An allowlist (not a
# denylist of named echo tools) needs no per-tool maintenance and is proven no-loss on the designed
# month (all 17 atoms ground in conversation; 0 need tool I/O). The double below is DELIBERATELY
# UNFAITHFUL — a fixed completion that re-states its atom on EVERY chunk, echo or not — so these
# prove the guard holds WITHOUT assuming the miner self-censors.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_read_echo_is_never_mined_provenance_preserved(store: MemoryStore):
    """Load-bearing guard: a memory_get tool_result echoing a day-1 fact into day-2 is NOT part of
    the mineable surface, so an UNFAITHFUL miner cannot re-mine it — the day-1 fact keeps
    provenance=day1 (>=2-session evidence intact). Day-2's OWN conversational fact still mines
    (the operative surface is not over-excluded)."""
    from localharness.memory.mining import _h8

    f_claim = "user got sunburnt at the beach"
    g_claim = "user favorite programming language is python"
    await store.append_history(_rec(10, "i got sunburnt at the beach yesterday afternoon", sid="day1"))
    await store.append_history(_rec(20, "my favorite programming language is python nowadays", sid="day2"))
    # The read-back: memory_get echoes the day-1 fact VERBATIM into day-2's transcript.
    await store.append_history(_rec(30, "user got sunburnt at the beach", sid="day2",
                                     typ="tool_result", tool_name="memory_get"))
    # Fixed == unfaithful: BOTH lines emitted for every chunk; per-chunk grounding admits only the
    # atom whose source record is present, and the echo is never in the fetched record stream.
    llm = _FakeLLM(f"health | {f_claim} | sunburnt at the beach\n"
                   f"prefs | {g_claim} | favorite programming language is python")

    report = await mine_transcript(store, llm, asyncio.Event(), file_tags=False)

    assert report.written == 2
    f = await store.get_fact(f"sem/health/{_h8(f_claim)}")
    assert f.provenance == "day1"          # NOT advanced to day2 — the echo was never mined
    assert f.confidence == 0.65            # never corroborated from a distinct day (no ladder step)
    g = await store.get_fact(f"sem/prefs/{_h8(g_claim)}")
    assert g.provenance == "day2"          # day-2's conversational fact still mines


@pytest.mark.asyncio
async def test_including_tool_result_surface_reproduces_the_collapse(store: MemoryStore):
    """Negative control (the guard is LOAD-BEARING, not decorative): widen the operative surface to
    INCLUDE tool_result and the SAME unfaithful double + echo DOES collapse — the day-1 fact is
    re-mined off the day-2 echo, store_fact's distinct-day ladder advances its provenance to day2 and
    steps confidence. This is exactly the failure the operative-surface restriction prevents (and the
    A1-starving mechanism a live unfaithful miner would otherwise be exposed to)."""
    from localharness.memory.mining import _h8

    f_claim = "user got sunburnt at the beach"
    await store.append_history(_rec(10, "i got sunburnt at the beach yesterday afternoon", sid="day1"))
    await store.append_history(_rec(30, "user got sunburnt at the beach", sid="day2",
                                     typ="tool_result", tool_name="memory_get"))
    llm = _FakeLLM(f"health | {f_claim} | sunburnt at the beach")

    await mine_transcript(store, llm, asyncio.Event(), file_tags=False,
                          operative_message_types=["user_message", "assistant_message", "tool_result"])

    f = await store.get_fact(f"sem/health/{_h8(f_claim)}")
    assert f.provenance == "day2"          # collapsed — re-mined off the echo, provenance advanced
    assert f.confidence > 0.65             # distinct-day ladder stepped it (0.65 -> 0.72)


@pytest.mark.asyncio
async def test_tool_result_content_is_structurally_out_of_scope(store: MemoryStore):
    """STEP 4 (structural exclusion of tool I/O): a fact stated ONLY in a tool_result is NOT mined by
    default — tool output (file reads, command output, store read-backs) is out of mining's operative
    surface. The SAME fact stated in a user message IS mined, so the surface is conversational, not
    empty."""
    from localharness.memory.mining import _h8

    claim = "the deploy runbook lives in docs slash ops"
    llm = _FakeLLM(f"ops | {claim} | deploy runbook lives in docs")

    # Only-in-a-tool_result: NOT mined.
    await store.append_history(_rec(10, "the deploy runbook lives in docs/ops per the config file",
                                     sid="s1", typ="tool_result", tool_name="read"))
    r1 = await mine_transcript(store, llm, asyncio.Event(), file_tags=False)
    assert r1.written == 0
    assert await store.query_facts(FactQuery(tags=["sem"])) == []

    # The same fact in a user message IS mined (the surface is conversational, not over-excluded).
    await store.append_history(_rec(20, "the deploy runbook lives in docs/ops per the config file",
                                     sid="s2"))
    r2 = await mine_transcript(store, llm, asyncio.Event(), file_tags=False)
    assert r2.written == 1
    assert (await store.get_fact(f"sem/ops/{_h8(claim)}")).provenance == "s2"


@pytest.mark.asyncio
async def test_operative_surface_is_config_driven_extensible(store: MemoryStore):
    """The operative surface is a config knob, not hardcoded: adding a record type widens what mining
    reads with NO code change (here tool_result becomes mineable). Robust by construction — a new
    echo-source needs no per-tool denylist entry to stay out; it is out unless the surface is widened."""
    from localharness.memory.mining import _h8

    claim = "the deploy runbook lives in docs slash ops"
    await store.append_history(_rec(10, "the deploy runbook lives in docs/ops per the config file",
                                     sid="s1", typ="tool_result", tool_name="read"))
    llm = _FakeLLM(f"ops | {claim} | deploy runbook lives in docs")

    await mine_transcript(store, llm, asyncio.Event(), file_tags=False,
                          operative_message_types=["user_message", "assistant_message", "tool_result"])

    assert (await store.get_fact(f"sem/ops/{_h8(claim)}")).provenance == "s1"


def test_config_default_operative_message_types_is_conversation():
    """The default operative surface is the conversational records (user + assistant); tool I/O is
    excluded by construction."""
    from localharness.config.models import MemoryConsolidationConfig
    assert (MemoryConsolidationConfig().mining_operative_message_types
            == ["user_message", "assistant_message"])
    cfg = MemoryConsolidationConfig(mining_operative_message_types=["user_message"])
    assert cfg.mining_operative_message_types == ["user_message"]  # narrowable/extensible verbatim


@pytest.mark.asyncio
async def test_step_mine_threads_operative_message_types_config(store: MemoryStore, monkeypatch):
    """Owner rule (prove the knob is WIRED, not merely present): _step_mine passes the config's
    mining_operative_message_types into mine_transcript — otherwise the guard would never fire in
    production, where the live transcript DOES carry memory read-backs."""
    from localharness.config.models import MemoryConsolidationConfig
    from localharness.memory.consolidation import ConsolidationPass, ConsolidationReport

    captured: dict = {}

    async def _fake_mine(store_, llm_, cancel_, **kw):
        captured.update(kw)
        return MineReport()

    monkeypatch.setattr("localharness.memory.mining.mine_transcript", _fake_mine)
    cfg = MemoryConsolidationConfig(mining_operative_message_types=["user_message"])
    await ConsolidationPass(store, cfg, llm=object())._step_mine(ConsolidationReport())

    assert captured.get("operative_message_types") == ["user_message"]


# ---------------------------------------------------------------------------
# REVIEW FIXES (36.1 extraction-yield review pass): two edges the run-10/FIX-3 mechanisms
# don't reach. (a) The dedupe identity ignored punctuation, so 'X' vs 'X.' minted duplicate
# active rows — the exact class run-10 closed for byte-identical strings. (b) The known-atoms
# window is a prompt-VISIBILITY cap, but supersede correctness silently depended on it: a
# same-pass replaces target that scrolled out (mints-in-pass > cap) fell through to a shadow
# duplicate. Correctness must come from an in-pass minted registry, not from keeping two
# independently-bounded config knobs manually in sync.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_punctuation_variant_value_dedupes_to_one_active_row(store: MemoryStore):
    """A re-mined claim differing from an active atom only by a trailing '.' is the SAME value:
    it must fold into the existing row (dedupe skip), not land on a fresh _h8 key as a second
    active duplicate."""
    await store.append_history(_rec(10, "the vLLM server listens on port 8081 now", sid="day2"))
    claim = "vLLM server listens on port 8081"
    r1 = await mine_transcript(store, _FakeLLM(
        f"gpu ops | {claim} | vLLM server listens on port 8081"), asyncio.Event())
    assert r1.written == 1

    await store.append_history(_rec(
        20, "yes the vLLM server listens on port 8081. confirmed", sid="day3"))
    r2 = await mine_transcript(store, _FakeLLM(
        f"gpu ops | {claim}. | vLLM server listens on port 8081."), asyncio.Event())

    active = await store.query_facts(FactQuery(tags=["sem"]))
    gpu = [f for f in active if f.key.startswith("sem/gpu-ops/")]
    assert len(gpu) == 1, (
        f"punctuation variant minted a duplicate active row: {[(f.key, f.value) for f in gpu]}")
    assert r2.written == 0  # skipped as a duplicate of the active row, not minted


@pytest.mark.asyncio
async def test_correction_supersedes_atom_scrolled_out_of_known_window(store: MemoryStore):
    """A same-pass `replaces=` whose target was minted this pass but scrolled OUT of the capped
    known window (mints-in-pass > known_atoms_cap) must still supersede via the in-pass minted
    registry — never mint a shadow duplicate. Guards the config space the raised write-budget
    ceiling opened (write_budget may legally exceed the cap's own upper bound)."""
    from localharness.memory.mining import _h8

    target_claim = "vLLM endpoint listens on port 8000"
    target_key = f"sem/gpu-ops/{_h8(target_claim)}"
    corrected = "vLLM moved to port 8081 instead"
    # ts=1 mints the target; ts=2..9 mint 8 fillers (> cap=5, target scrolls out); ts=100 corrects.
    await store.append_history(_rec(1, "the vLLM endpoint listens on port 8000 noted", sid="s1"))
    for k in range(2, 10):
        await store.append_history(
            _rec(k, f"widgetnum{k} gadgetnum{k} are the two distinct markers here", sid="s1"))
    await store.append_history(_rec(100, "correction the vLLM moved to port 8081 instead now", sid="s1"))
    lines = (
        [f"gpu ops | {target_claim} | endpoint listens on port 8000"]
        + [f"filler | widgetnum{k} gadgetnum{k} | widgetnum{k} gadgetnum{k}" for k in range(2, 10)]
        + [f"gpu ops | {corrected} | moved to port 8081 instead | replaces={target_key}"]
    )
    # Tiny corpus cap -> one record per chunk, so the known window (cap=5) scrolls past the target.
    await mine_transcript(store, _FakeLLM("\n".join(lines)), asyncio.Event(),
                          write_budget=100, corpus_char_cap=40, known_atoms_cap=5, file_tags=False)

    active = await store.query_facts(FactQuery(tags=["sem"]))
    gpu = [f for f in active if f.key.startswith("sem/gpu-ops/")]
    assert len(gpu) == 1, f"scrolled-out target not superseded (shadow dup): {gpu}"
    assert gpu[0].key == target_key and gpu[0].value == corrected


# ---------------------------------------------------------------------------
# STAGE 1 (extraction science plan, 2026-07-09): the COVERAGE/RESIDUE metric — recall
# observability, zero behavior change. Per pass: which committed records were never the SOURCE of a
# WRITTEN atom (the residue). The residue's stable record identity feeds the cross-run intersection
# (R) that adjudicates systematic vs stochastic under-extraction BEFORE any repair mechanism is
# built (research/2026-07-09-extraction-science-plan.md — thresholds frozen pre-data). A record
# whose only atom was rejected-ungrounded is residue: that is a recall failure, not coverage.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coverage_residue_identifies_uncited_records(store: MemoryStore):
    """3 committed records, 1 written atom grounding in the first: the other two are residue,
    carrying stable id + session + ts + chars (the cross-run intersection needs the id)."""
    await store.append_history(_rec(10, "i am building a summarizer subagent for the harness"))
    await store.append_history(_rec(20, "we are planning a kyoto trip in november this year"))
    await store.append_history(_rec(30, "ok sounds good thanks"))
    llm = _FakeLLM("subagents | building a summarizer subagent for the harness | summarizer subagent")

    report = await mine_transcript(store, llm, asyncio.Event(), file_tags=False)

    assert report.written == 1
    assert report.records_seen == 3 and report.records_cited == 1
    assert {r["id"] for r in report.residue} == {"h20", "h30"}
    kyoto = next(r for r in report.residue if r["id"] == "h20")
    assert kyoto["session_id"] == "s1" and kyoto["ts"] == 20
    assert kyoto["chars"] == len("we are planning a kyoto trip in november this year")
    # Stable cross-run identity: record ids are per-run uuids in production, so the Stage-3
    # intersection keys on the content fingerprint (scripted turns are byte-stable across runs).
    import hashlib
    assert kyoto["content_h8"] == hashlib.sha1(
        b"we are planning a kyoto trip in november this year").hexdigest()[:8]


@pytest.mark.asyncio
async def test_coverage_rejected_ungrounded_atom_does_not_cite(store: MemoryStore):
    """An atom killed by the grounding net does NOT mark its source record cited — the record
    yielded no WRITTEN atom, so it is residue (extraction failed there)."""
    await store.append_history(_rec(10, "i got super duper sunburnt today at the beach"))
    llm = _FakeLLM("luck | user won the lottery jackpot | i got super duper sunburnt")

    report = await mine_transcript(store, llm, asyncio.Event(), file_tags=False)

    assert report.written == 0 and report.rejected_ungrounded == 1
    assert report.records_seen == 1 and report.records_cited == 0
    assert [r["id"] for r in report.residue] == ["h10"]


@pytest.mark.asyncio
async def test_step_mine_surfaces_coverage_metric(store: MemoryStore, monkeypatch):
    """The consolidation report carries the coverage fields (wired, not merely present) so the
    eval harness can persist per-run residue for the Stage-3 cross-run intersection."""
    from localharness.config.models import MemoryConsolidationConfig
    from localharness.memory.consolidation import ConsolidationPass, ConsolidationReport

    async def _fake_mine(store_, llm_, cancel_, **kw):
        return MineReport(written=1, records_seen=3, records_cited=1,
                          residue=[{"id": "h20", "session_id": "s1", "ts": 20,
                                    "chars": 50, "preview": "we are planning a kyoto trip"}])

    monkeypatch.setattr("localharness.memory.mining.mine_transcript", _fake_mine)
    report = ConsolidationReport()
    await ConsolidationPass(store, MemoryConsolidationConfig(), llm=object())._step_mine(report)

    assert report.mined == 1
    assert report.mined_records_seen == 3 and report.mined_records_cited == 1
    assert report.mining_residue and report.mining_residue[0]["id"] == "h20"


# ---------------------------------------------------------------------------
# RESIDUE LEDGER (core repair loop, owner-directed 2026-07-09): encode -> check -> repair ->
# forget. The miner's first pass is attention-limited and lossy BY DESIGN (run-12 Kyoto: a
# crowded chunk minted 3 unrelated atoms and skimmed 9 Kyoto records IN THE SAME CHUNK). The
# repair is AMORTIZED, never a same-pass second look: uncited non-trivial records enter a durable
# ledger (schema v7 side table); the NEXT idle pass re-mines them in ISOLATION (a residue-only
# chunk, where they cannot lose the attention contest), sequentially, budgeted. A record that
# stays barren after `attempt_cap` isolated looks is RETIRED — permanently out of the mining
# window but NEVER deleted (history.jsonl is append-only; same demote-never-destroy law as the
# atom tier's retrieval_strength). K/budget/intake are config hyperparameters — swept later,
# never taste-picked.
# ---------------------------------------------------------------------------


class _AttentionLimitedLLM:
    """Models the measured live failure: in a CROWDED corpus the miner extracts the loudest fact
    and skims the rest; shown the skimmed content in ISOLATION it extracts it. Branch order makes
    'summarizer' always win a crowded corpus, then port, then kyoto."""

    def __init__(self):
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        corpus = prompt.rsplit("\n\n", 1)[-1]
        if "summarizer subagent" in corpus:
            return "subagents | building a summarizer subagent for the harness | summarizer subagent"
        if "listens on port 8081" in corpus:
            return "gpu ops | vllm server listens on port 8081 | server listens on port 8081"
        if "kyoto trip" in corpus:
            return "kyoto trip | planning a kyoto trip in november | planning a kyoto trip in november"
        return ""


@pytest.mark.asyncio
async def test_residue_enqueued_then_rescued_by_isolated_drain_next_pass(store: MemoryStore):
    """The amortized loop end-to-end: pass 1 skims the kyoto record (residue -> ledger, and the
    SAME pass never drains what it just enqueued); pass 2 — with NO new records — re-mines the
    residue in isolation, rescues the fact, and provenance stays the ORIGINAL record's session."""
    await store.append_history(_rec(10, "i am building a summarizer subagent for the harness"))
    await store.append_history(_rec(20, "we are planning a kyoto trip in november this year"))
    llm = _AttentionLimitedLLM()

    r1 = await mine_transcript(store, llm, asyncio.Event(), file_tags=False)
    assert r1.written == 1
    assert r1.residue_drained == 0          # amortized: a pass never drains its own residue
    assert r1.residue_enqueued == 1         # the skimmed kyoto record entered the ledger
    assert [p["record_id"] for p in await store.residue_pending(cap=10)] == ["h20"]

    r2 = await mine_transcript(store, llm, asyncio.Event(), file_tags=False)  # no new records
    assert r2.residue_drained == 1 and r2.residue_rescued == 1 and r2.written == 1
    kyoto = [f for f in await store.query_facts(FactQuery(tags=["sem"]))
             if f.key.startswith("sem/kyoto-trip/")]
    assert len(kyoto) == 1 and kyoto[0].provenance == "s1"  # original session, not the drain pass
    assert await store.residue_pending(cap=10) == []        # rescued — out of the queue
    # Isolation: the drain corpus carries ONLY the residue record (the loud fact is absent).
    assert "summarizer" not in llm.prompts[-1].rsplit("\n\n", 1)[-1]


@pytest.mark.asyncio
async def test_residue_retired_at_attempt_cap_history_never_deleted(store: MemoryStore):
    """The forgetting half: a record that yields nothing after `attempt_cap` ISOLATED looks is
    retired — out of the mining window forever — while the raw history record survives untouched
    (append-only law: retire selects, never destroys)."""
    await store.append_history(_rec(10, "i am building a summarizer subagent for the harness"))
    await store.append_history(_rec(20, "the weather was surprisingly nice around here today"))
    llm = _AttentionLimitedLLM()  # never emits a weather fact

    r1 = await mine_transcript(store, llm, asyncio.Event(), file_tags=False, residue_attempt_cap=2)
    assert r1.residue_enqueued == 1
    r2 = await mine_transcript(store, llm, asyncio.Event(), file_tags=False, residue_attempt_cap=2)
    assert r2.residue_drained == 1 and r2.residue_rescued == 0 and r2.residue_retired == 0
    r3 = await mine_transcript(store, llm, asyncio.Event(), file_tags=False, residue_attempt_cap=2)
    assert r3.residue_drained == 1 and r3.residue_retired == 1   # attempts hit the cap
    r4 = await mine_transcript(store, llm, asyncio.Event(), file_tags=False, residue_attempt_cap=2)
    assert r4.residue_drained == 0                                # retired = never re-fed
    assert await store.residue_pending(cap=10) == []
    # The substrate is intact: the record is still in history, readable by anything else.
    assert any(x.get("id") == "h20" for x in await store.get_history(limit=100))


@pytest.mark.asyncio
async def test_residue_intake_filter_skips_trivial_records(store: MemoryStore):
    """Belt to the cap's suspenders: a trivially short record ('ok') stays VISIBLE in the metric
    but never enters the ledger — the queue stays small by construction, not by re-chewing."""
    await store.append_history(_rec(10, "i am building a summarizer subagent for the harness"))
    await store.append_history(_rec(20, "ok"))
    llm = _AttentionLimitedLLM()

    r1 = await mine_transcript(store, llm, asyncio.Event(), file_tags=False, residue_min_chars=20)
    assert {x["id"] for x in r1.residue} == {"h20"}   # the METRIC still sees it (honest coverage)
    assert r1.residue_enqueued == 0                   # the LEDGER never chews it
    assert await store.residue_pending(cap=10) == []


@pytest.mark.asyncio
async def test_drain_runs_after_new_window_and_isolated_from_it(store: MemoryStore):
    """A pass with BOTH new records and pending residue does the new work first, then drains the
    residue in its own isolated chunk — sequential (one look at a time), never fanned out, and the
    two corpora never mix (residue must not lose the attention contest AGAIN)."""
    await store.append_history(_rec(10, "i am building a summarizer subagent for the harness"))
    await store.append_history(_rec(20, "we are planning a kyoto trip in november this year"))
    llm = _AttentionLimitedLLM()
    await mine_transcript(store, llm, asyncio.Event(), file_tags=False)   # kyoto -> ledger

    await store.append_history(_rec(30, "the vllm server listens on port 8081 now", sid="s2"))
    r2 = await mine_transcript(store, llm, asyncio.Event(), file_tags=False)

    assert r2.written == 2 and r2.residue_rescued == 1
    facts = {f.key.split("/")[1]: f.provenance for f in await store.query_facts(FactQuery(tags=["sem"]))}
    assert facts["gpu-ops"] == "s2" and facts["kyoto-trip"] == "s1"
    main_corpus = llm.prompts[-2].rsplit("\n\n", 1)[-1]
    drain_corpus = llm.prompts[-1].rsplit("\n\n", 1)[-1]
    assert "port 8081" in main_corpus and "kyoto" not in main_corpus
    assert "kyoto" in drain_corpus and "port 8081" not in drain_corpus


@pytest.mark.asyncio
async def test_residue_disabled_is_inert(store: MemoryStore):
    """The kill switch: residue_enabled=False neither enqueues nor drains (metric unaffected)."""
    await store.append_history(_rec(10, "i am building a summarizer subagent for the harness"))
    await store.append_history(_rec(20, "we are planning a kyoto trip in november this year"))
    llm = _AttentionLimitedLLM()

    r1 = await mine_transcript(store, llm, asyncio.Event(), file_tags=False, residue_enabled=False)
    assert r1.residue_enqueued == 0 and await store.residue_pending(cap=10) == []
    assert {x["id"] for x in r1.residue} == {"h20"}   # metric still reports coverage
    r2 = await mine_transcript(store, llm, asyncio.Event(), file_tags=False, residue_enabled=False)
    assert r2.residue_drained == 0


@pytest.mark.asyncio
async def test_residue_ledger_schema_v7_idempotent_enqueue(store: MemoryStore):
    """Schema v7 carries the ledger; enqueue is keyed on (agent, record) — a re-enqueue of the
    same record is a no-op that PRESERVES its attempt count (re-surfacing never resets the K clock)."""
    async with store._db.execute("PRAGMA user_version") as cur:
        assert (await cur.fetchone())[0] == 7
    e = {"id": "hx", "session_id": "s1", "ts": 5, "chars": 30, "content_h8": "aabbccdd"}
    assert await store.residue_enqueue([e]) == 1
    assert await store.residue_enqueue([e]) == 0
    await store.residue_bump(["hx"], attempt_cap=5)
    assert await store.residue_enqueue([e]) == 0
    assert (await store.residue_pending(cap=10))[0]["attempts"] == 1


@pytest.mark.asyncio
async def test_step_mine_threads_residue_config_and_surfaces_counters(store: MemoryStore, monkeypatch):
    """The four knobs are WIRED (config -> mine_transcript) and the counters surface on
    ConsolidationReport so the eval persists them per run."""
    from localharness.config.models import MemoryConsolidationConfig
    from localharness.memory.consolidation import ConsolidationPass, ConsolidationReport

    captured: dict = {}

    async def _fake_mine(store_, llm_, cancel_, **kw):
        captured.update(kw)
        return MineReport(residue_enqueued=2, residue_drained=3, residue_rescued=1, residue_retired=1)

    monkeypatch.setattr("localharness.memory.mining.mine_transcript", _fake_mine)
    cfg = MemoryConsolidationConfig(mining_residue_enabled=True, mining_residue_attempt_cap=3,
                                    mining_residue_record_budget=7, mining_residue_min_chars=5)
    report = ConsolidationReport()
    await ConsolidationPass(store, cfg, llm=object())._step_mine(report)

    assert captured.get("residue_enabled") is True
    assert captured.get("residue_attempt_cap") == 3
    assert captured.get("residue_record_budget") == 7
    assert captured.get("residue_min_chars") == 5
    assert report.mining_residue_enqueued == 2 and report.mining_residue_drained == 3
    assert report.mining_residue_rescued == 1 and report.mining_residue_retired == 1
