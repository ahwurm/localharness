"""Phase 35 PredictiveWriteGate — the LIVE write decision (PGATE-01/02/03).

The gate is a sibling bus subscriber that turns Phase-34's measured surprise signals
into real, reversible, sub-0.7 memory writes, while gate.py / predictive_gate.py /
user_signals.py stay byte-untouched (a sibling makes that diff empty by construction).

This file is built in two passes on the same shared branch:
- Task 1 (this top section): the two additive store seams — the new _IMPORTANCE_PRIORS
  tiers (Pitfall 4 regression pins) and the read-only staged_suspect_facts().
- Task 3 (appended below): the PGATE-01/02/03 behaviors driven over a REAL EventBus.
"""
from pathlib import Path

from localharness.config.models import MemoryConsolidationConfig, PredictiveGateConfig
from localharness.core.bus import EventBus
from localharness.core.events import SurpriseScored, UserMessage
from localharness.memory.consolidation import ConsolidationPass
from localharness.memory.predictive_write_gate import PredictiveWriteGate
from localharness.memory.sqlite import (
    FactQuery,
    MemoryStore,
    _importance_prior,
    _IMPORTANCE_PRIORS,
)
from localharness.tools.builtin.memory_tools import MemorySearchTool

AGENT = "pwg-agent"


def make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        agent_id=AGENT, division_id="", org_id="", base_dir=str(tmp_path)
    )


# ---------------------------------------------------------------------------
# Task 1 — additive store seams
# ---------------------------------------------------------------------------


def test_importance_priors_carry_the_two_new_tiers():
    # Pitfall 4: a new tier tag with no _IMPORTANCE_PRIORS entry silently ranks 0.0.
    # These two entries are the regression pin that keeps PGATE-01/03 writes non-zero.
    assert _IMPORTANCE_PRIORS["tier:surprising_failure"] == 0.3
    assert _IMPORTANCE_PRIORS["tier:correction_pending"] == 0.2
    # The v2.0 tiers must be untouched by the additive change (back-compat).
    assert _IMPORTANCE_PRIORS["tier:resolved_error"] == 0.3
    assert _IMPORTANCE_PRIORS["tier:stuck_recovered"] == 0.2


def test_importance_prior_resolves_surprising_failure_tag_nonzero():
    # A stat fact's tag yields the tier prior, not the 0.0 fallback.
    assert _importance_prior(["gate", "tier:surprising_failure"], "predictive_write_gate") == 0.3
    # And an unknown tag set still degrades gracefully to 0.0 (unchanged behavior).
    assert _importance_prior(["gate", "tier:not_a_real_tier"], "predictive_write_gate") == 0.0


async def test_staged_suspect_facts_returns_the_staged_set(tmp_path: Path):
    store = make_store(tmp_path)
    await store.open()
    try:
        # Nothing staged yet -> empty.
        assert await store.staged_suspect_facts() == []
        # A stored-but-unstaged fact is still not a suspect (store_fact never stages).
        await store.store_fact(key="k1", value="value one", confidence=0.65)
        assert await store.staged_suspect_facts() == []
        # touch_staged is what memory_search/memory_get do on retrieval -> now it is staged.
        await store.touch_staged(["k1"])
        rows = await store.staged_suspect_facts()
        assert len(rows) == 1
        _fid, key, value = rows[0]
        assert key == "k1"
        assert value == "value one"
        assert isinstance(_fid, int)
    finally:
        await store.close()


async def test_staged_suspect_facts_is_read_only_and_active_only(tmp_path: Path):
    store = make_store(tmp_path)
    await store.open()
    try:
        await store.store_fact(key="k1", value="v1", confidence=0.65)
        await store.touch_staged(["k1"])
        # Read-only: a second call returns the SAME row (never resets access_count_staged;
        # that is fold_staged_access's job at the consolidation boundary).
        first = await store.staged_suspect_facts()
        second = await store.staged_suspect_facts()
        assert first == second == [(first[0][0], "k1", "v1")]
        # Superseding k1 leaves the OLD row status='superseded' but with access_count_staged
        # still == 1 (supersede never touches the staged counter). Without the status='active'
        # filter that stale suspect would leak; with it, the superseded row is excluded and
        # the fresh active row (staged=0) is not a suspect -> empty.
        await store.store_fact(key="k1", value="v2-different", confidence=0.6)
        assert await store.staged_suspect_facts() == []
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Task 3 — PGATE-01/02/03 behaviors over a REAL EventBus + MemoryStore.
# The gate is opened, events are published on the bus, then store state is asserted:
# the "reachable end-to-end, not a green test on unwired code" bar (CLAUDE.md).
# ---------------------------------------------------------------------------


def _cfg_write_live() -> PredictiveGateConfig:
    cfg = PredictiveGateConfig()
    # Wave-2 (35-02) adds `write_live` to the model; this wave the field does not exist
    # yet and PredictiveGateConfig is extra="forbid" (plain setattr would raise), so set
    # it directly on the instance. The gate reads it fail-closed via getattr.
    object.__setattr__(cfg, "write_live", True)
    return cfg


async def _open_gate(tmp_path: Path) -> tuple[MemoryStore, EventBus]:
    store = make_store(tmp_path)
    await store.open()
    bus = EventBus()
    gate = PredictiveWriteGate(store, bus, AGENT, _cfg_write_live())
    await gate.open()
    return store, bus


def _surprise(
    quadrant: str, score: float, *, tool_name: str = "web_fetch", session_id: str = "s1"
) -> SurpriseScored:
    # Published DIRECTLY (not derived) so the WRITE decision is isolated from the SCORING
    # (predictive_gate.py's tested job): quadrant + score are set explicitly on the event.
    return SurpriseScored(
        agent_id=AGENT, session_id=session_id, tool_call_id="tc-1",
        tool_name=tool_name, score=score, quadrant=quadrant,
    )


def _um(content: str, session_id: str = "s1") -> UserMessage:
    return UserMessage(
        agent_id=AGENT, session_id=session_id, content=content, channel="terminal"
    )


async def _active_facts(store: MemoryStore) -> list:
    return await store.query_facts(FactQuery(limit=200))


async def test_surprising_failure_writes_graded(tmp_path: Path):
    store, bus = await _open_gate(tmp_path)
    try:
        await bus.publish(_surprise("surprising_failure", 2.086, tool_name="web_fetch"))
        hits = [
            f for f in await _active_facts(store)
            if f.key.startswith("predgate/surprising_failure/web_fetch/")
        ]
        assert len(hits) == 1
        f = hits[0]
        # PGATE-01: graded from the carried score (0.5 + 0.07*2.086 ≈ 0.646), strictly < 0.7.
        assert abs(f.confidence - 0.646) < 0.002
        assert f.confidence < 0.7
        # non-zero importance prior (Pitfall 4).
        assert f.importance > 0
        # never rendered in the injected ambient block (confidence < 0.7 fails the gate).
        ctx = await store.load_context(index_mode=True)
        assert f.key not in ctx.agent_memory_md
        assert "surprising failure" not in ctx.agent_memory_md
    finally:
        await store.close()


async def test_importance_prior_nonzero(tmp_path: Path):
    store, bus = await _open_gate(tmp_path)
    try:
        await bus.publish(_surprise("surprising_failure", 2.086, tool_name="bash_exec"))
        hits = [
            f for f in await _active_facts(store)
            if f.key.startswith("predgate/surprising_failure/bash_exec/")
        ]
        assert len(hits) == 1
        # the tier prior (0.3), NOT the silent 0.0 fallback — the Pitfall-4 regression pin.
        assert hits[0].importance == 0.3
    finally:
        await store.close()


async def test_unsurprising_failure_no_write(tmp_path: Path):
    store, bus = await _open_gate(tmp_path)
    try:
        # PGATE-02: the real 34 trace has 0 of these, so this is SYNTHETIC. Every non-write
        # quadrant must produce no fact — the write set is exactly {surprising_failure}.
        for q in ("unsurprising_failure", "routine", "cold_start", "quiet_surprise"):
            await bus.publish(_surprise(q, 3.0))
        assert await _active_facts(store) == []
    finally:
        await store.close()


async def test_correction_phrase_supersedes_single_staged_suspect(tmp_path: Path):
    store, bus = await _open_gate(tmp_path)
    try:
        # A fact retrieved into the corrected turn: seed it active (above the injection
        # gate) and stage it, exactly as memory_search/memory_get would.
        await store.store_fact(key="k1", value="the capital is Sydney", confidence=0.8)
        await store.touch_staged(["k1"])
        # BLOCKER 1(a): only the correction_phrase family ("i meant …") supersedes a staged
        # suspect; a bare negation is quarantine-only (proven in the next test). BUG #45: the
        # correction must also be content-RELATED to the suspect — "capital" is the shared
        # salient token here (the old placeholder "i meant the other one" shared none).
        await bus.publish(_um("actually the capital is Canberra"))
        # k1's active row is now the disputed marker at 0.6 — a real, key-based supersede
        # that also drops the disputed suspect below the 0.7 injection gate.
        active = await store.get_fact("k1")
        assert active is not None
        assert active.value.startswith("[disputed")
        # BLOCKER 1(c): the FULL original value is preserved behind the marker (no truncation).
        assert "the capital is Sydney" in active.value
        assert active.confidence == 0.6
        # REVERSIBLE: the original value is still reachable through history (supersede,
        # not overwrite) — a wrong correction is recoverable.
        history = await store.get_fact_history("k1")
        assert any(v.value == "the capital is Sydney" for v in history)
    finally:
        await store.close()


async def test_negation_with_suspects_quarantines_not_supersede(tmp_path: Path):
    # BLOCKER 1(a) repro (the critic's footgun): a BENIGN negation ("no worries …") must
    # NEVER mangle a staged fact. Pre-fix the gate superseded EVERY staged suspect on any
    # negation; post-fix a bare negation is quarantine-only, leaving retrieved memory intact.
    store, bus = await _open_gate(tmp_path)
    try:
        await store.store_fact(key="profile/name", value="Alexander Wurm", confidence=1.0)
        await store.touch_staged(["profile/name"])
        await bus.publish(_um("no worries, thanks! that all looks great"))
        # the staged fact is byte-untouched — same value, same confidence, still injected.
        active = await store.get_fact("profile/name")
        assert active is not None
        assert active.value == "Alexander Wurm"
        assert active.confidence == 1.0
        # the negation is captured ADDITIVELY as a standalone quarantine fact, not a supersede.
        quar = [
            f for f in await _active_facts(store) if f.key.startswith("correction/quarantine/")
        ]
        assert len(quar) == 1
    finally:
        await store.close()


async def test_supersede_targets_the_related_suspect_not_the_most_recent(tmp_path: Path):
    # BUG #45 (this INVERTS the pre-fix pin, which superseded the single most-recently-staged
    # suspect regardless of content): with several staged suspects, the dispute targets the one
    # the correction is content-RELATED to — NOT merely the most recent. The blast radius is
    # still exactly ONE fact.
    store, bus = await _open_gate(tmp_path)
    try:
        # k1 (RELATED to the coming correction) is staged FIRST / older; k2 (UNRELATED) is
        # staged LAST / most-recent — so recency and relatedness point at different facts.
        await store.store_fact(key="k1", value="the vLLM server listens on port 8000", confidence=0.8)
        await store.store_fact(key="k2", value="the user's dog is named Rex", confidence=0.8)
        await store.touch_staged(["k1", "k2"])
        await store._db.execute("UPDATE facts SET last_accessed_staged = 1000 WHERE key = 'k1'")
        await store._db.execute("UPDATE facts SET last_accessed_staged = 2000 WHERE key = 'k2'")
        await store._db.commit()
        # shares {vllm, server, port} with k1, nothing with k2.
        await bus.publish(_um("actually the vLLM server port is 8081"))
        k1 = await store.get_fact("k1")
        k2 = await store.get_fact("k2")
        # the RELATED (older) suspect is disputed; the UNRELATED most-recent one is untouched.
        assert k1 is not None and k1.value.startswith("[disputed") and k1.confidence == 0.6
        assert k2 is not None and k2.value == "the user's dog is named Rex" and k2.confidence == 0.8
    finally:
        await store.close()


async def test_supersede_preserves_full_value_no_truncation(tmp_path: Path):
    # BLOCKER 1(c): the disputed wrapper preserves the FULL original value — the pre-fix
    # _preview truncated it to 180 chars, corrupting the active row until reconciliation.
    store, bus = await _open_gate(tmp_path)
    try:
        long_value = "CAPITAL FACT: " + "x" * 300  # >> the old 180-char _preview cap
        await store.store_fact(key="k1", value=long_value, confidence=0.8)
        await store.touch_staged(["k1"])
        # BUG #45: correction is content-related to the suspect (shares "capital"/"fact").
        await bus.publish(_um("actually the capital fact changed"))
        active = await store.get_fact("k1")
        assert active is not None
        assert active.value.startswith("[disputed")
        # the ENTIRE original survives behind the marker (no 180-char truncation).
        assert long_value in active.value
    finally:
        await store.close()


async def test_unrelated_correction_disputes_nothing(tmp_path: Path):
    # BUG #45 (the exact live repro): fact A is retrieved (staged), then the user corrects an
    # UNRELATED fact B two turns later. Pre-fix, A was superseded to "[disputed …]" at 0.6 —
    # silently vanishing from MEMORY.md. Post-fix, a correction only disputes a suspect it is
    # content-related to; with none related, NOTHING is disputed and the correction is captured
    # additively as a quarantine (so reconciliation / the model's own path can still act on it).
    store, bus = await _open_gate(tmp_path)
    try:
        await store.store_fact(key="pet", value="the dog is named Rex", confidence=0.9)
        await store.touch_staged(["pet"])
        # correction_phrase family ("actually"), but shares NO salient token with the dog fact.
        await bus.publish(_um("actually the bakery is in Boulder"))
        # A is byte-untouched — same value, same confidence, still injectable.
        a = await store.get_fact("pet")
        assert a is not None
        assert a.value == "the dog is named Rex"
        assert a.confidence == 0.9
        # nothing anywhere carries the dispute marker ...
        assert not any(
            f.value.startswith("[disputed") for f in await _active_facts(store)
        )
        # ... and the unrelated correction was captured additively (not lost, not a supersede).
        quar = [
            f for f in await _active_facts(store) if f.key.startswith("correction/quarantine/")
        ]
        assert len(quar) == 1
    finally:
        await store.close()


async def test_correction_about_the_staged_fact_still_disputes(tmp_path: Path):
    # BUG #45 true-positive guard (the sema05 eval's correction shape depends on this path): a
    # correction that IS about the staged fact — sharing salient tokens like port/vLLM/server —
    # still supersedes it exactly as before. Only content-UNRELATED disputes are suppressed.
    store, bus = await _open_gate(tmp_path)
    try:
        await store.store_fact(
            key="cfg/vllm_port", value="the vLLM server listens on port 8000", confidence=0.8
        )
        await store.touch_staged(["cfg/vllm_port"])
        await bus.publish(_um("actually the vLLM server now runs on port 8081"))
        active = await store.get_fact("cfg/vllm_port")
        assert active is not None
        assert active.value.startswith("[disputed")
        assert "the vLLM server listens on port 8000" in active.value  # full value preserved
        assert active.confidence == 0.6
        assert "tier:correction_pending" in active.tags
        # reversible: the original survives in history.
        history = await store.get_fact_history("cfg/vllm_port")
        assert any(v.value == "the vLLM server listens on port 8000" for v in history)
    finally:
        await store.close()


async def test_disputed_gate_candidate_is_not_promoted_into_injected_block(tmp_path: Path):
    # MAJOR 2 (exact critic repro): a correction_phrase dispute on a STAGED gate/ candidate
    # writes the disputed row back onto the ORIGINAL gate/ key (tier:correction_pending), so
    # the existing Phase-31 consolidation groups + promotes it to 0.8 — into the injected
    # ambient block, violating the "<0.7 until Phase 36" invariant. Driven end-to-end
    # (real gate + real deterministic pass, no LLM).
    store, bus = await _open_gate(tmp_path)
    try:
        tool, lesson, sess = "mytool", "abc12345", "sess0001"
        gate_key = f"gate/resolved_error/{tool}/{lesson}/{sess}"
        learned_key = f"learned/{tool}/resolved_error/{lesson}"
        # a prior legit promotion already injected — the reachability condition (an existing
        # learned/ record makes a single fresh episode promotable).
        await store.store_fact(
            key=learned_key, value="mytool errored: boom then fixed",
            tags=["consolidated", "tier:resolved_error"], confidence=0.8,
            source="consolidation", provenance="consolidated:2-episodes",
        )
        # a staged gate/ candidate — the suspect the user is about to dispute.
        await store.store_fact(
            key=gate_key,
            value="`mytool` error resolved: boom -> then succeeded (auto-captured; pending consolidation)",
            tags=["gate", "tier:resolved_error", "pending_consolidation"], confidence=0.65,
            source="write_gate", provenance=sess,
        )
        await store.touch_staged([gate_key])
        # dispute it via a correction_phrase trigger (the class that supersedes post-BLOCKER-1);
        # BUG #45: content-related to the gate suspect ("mytool"/"error" are shared salient tokens).
        await bus.publish(_um("actually the mytool error is different"))
        disputed = await store.get_fact(gate_key)
        assert disputed is not None and disputed.value.startswith("[disputed")
        assert "tier:correction_pending" in disputed.tags

        report = await ConsolidationPass(store, MemoryConsolidationConfig()).run()

        # the disputed correction row is EXCLUDED from promotion candidates -> nothing promotes.
        assert report.promoted == 0
        # the learned/ record is NOT re-promoted from the disputed text ...
        promoted = await store.get_fact(learned_key)
        assert promoted is not None and "[disputed" not in promoted.value
        # ... and the disputed marker never reaches the injected ambient block.
        ctx = await store.load_context(index_mode=True)
        assert "disputed by user correction" not in ctx.agent_memory_md
    finally:
        await store.close()


async def test_reask_excluded(tmp_path: Path):
    store, bus = await _open_gate(tmp_path)
    try:
        # Two near-identical messages with NO lexical trigger (a reask shape). This gate
        # keeps no per-sitting window, so it never fires on a reask — by construction.
        await bus.publish(_um("what is the vllm flag for prefix caching"))
        await bus.publish(_um("what is the vllm flag for prefix caching again"))
        assert await _active_facts(store) == []
    finally:
        await store.close()


async def test_frustration_excluded(tmp_path: Path):
    store, bus = await _open_gate(tmp_path)
    try:
        # "ugh" is signal_type=correction but family=frustration — out of scope (no census
        # data; only negation/correction_phrase write inline).
        await bus.publish(_um("ugh"))
        assert await _active_facts(store) == []
    finally:
        await store.close()


async def test_fireworks_quarantine(tmp_path: Path):
    store, bus = await _open_gate(tmp_path)
    try:
        # No staged fact -> no supersede target (and a bare negation is quarantine-only
        # anyway, post-BLOCKER-1) -> standalone quarantine of the user's own words: the
        # specimen that was structurally invisible to the v2.0 pain-only gate.
        msg = _um("nah id rather watch the fireworks from the park with friends tomorrow")
        await bus.publish(msg)
        hits = [
            f for f in await _active_facts(store)
            if f.key.startswith("correction/quarantine/")
        ]
        assert len(hits) == 1
        f = hits[0]
        assert f.confidence == 0.65
        assert f.provenance == msg.id  # provenance points at the trigger record
        # never in the injected block ...
        ctx = await store.load_context(index_mode=True)
        assert f.key not in ctx.agent_memory_md
        # ... but reachable by memory_search (Phase 36's reconciliation can still find it).
        res = await MemorySearchTool(store)._execute(query="fireworks")
        assert res.success
        assert "correction/quarantine/" in res.output
    finally:
        await store.close()
