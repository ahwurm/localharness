"""Cross-feature seam 4 — one consolidation pass, all three chapter mechanisms with work.

The seam no unit test crosses: the containment guard (5d19ede), the staleness re-check
(18dccb6), and the healthy-chapter idempotency law each have a unit test in ISOLATION, but a
real idle pass runs them together, in a fixed order (recheck → write_schemas), sharing the same
report counters (both steps `+=` the same containment fields). This module drives ONE
ConsolidationPass over a store where simultaneously:
  - an existing chapter's evidentiary base eroded (a member superseded out from under it) —
    the staleness re-check must re-draft it on the survivors;
  - a fresh candidate cluster strictly contains two existing chapters — the writer must mint/
    merge and supersede the stranded one (the gap chapter_refresh_overlap alone leaves);
  - a healthy chapter has full support — it must be left byte-for-byte untouched.

Then it asserts the EXACT final state: append-only supersede chains, exactly the expected
active chapters, self-consistent counters, and a byte-stable second pass. Behavior is correct
under the real code; this locks it in.
"""
from __future__ import annotations

import hashlib

import pytest

from localharness.config.models import MemoryConsolidationConfig
from localharness.memory.consolidation import ConsolidationPass
from localharness.memory.sqlite import MemoryStore


@pytest.fixture
async def store(tmp_path):
    s = MemoryStore(agent_id="chap-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


class _DispatchLLM:
    """The idle-LLM contract is `async def complete(prompt) -> str` (what complete_cancellable
    awaits). For a chapter-write prompt, echo the corpus tail verbatim (always grounded, no
    invented number); the healthy revalidation path never calls this at all."""

    def __init__(self):
        self.calls = 0

    async def complete(self, prompt: str) -> str:
        self.calls += 1
        if "Write ONE" in prompt:
            corpus = prompt.split("\n\n")[-1]
            lines = [ln for ln in corpus.splitlines() if ln.strip()]
            return " ".join(lines[0].split()[:12]) if lines else "chapter"
        return ""


async def _seed_sem(store, body, session, *, topic, clustered):
    """One pool atom (sem/{topic}/{hash}) at pool-entry confidence. `clustered=True` attaches an
    active child tag so co-tagged atoms across >=2 sessions form a stable cluster; False keeps
    the atom OUT of the clustering graph (used for the never-re-clustered healthy/stale members)."""
    h = hashlib.sha1(body.strip().encode()).hexdigest()[:8]
    fact = await store.store_fact(
        key=f"sem/{topic}/{h}", value=body, tags=["sem", "pending_consolidation"],
        confidence=0.65, source="transcript_mining", provenance=session, node_kind="fact")
    if clustered:
        tag = await store.get_tag(topic)
        if tag is None:
            tag = await store.create_tag(topic, f"serves {topic}; example {topic}",
                                         parent_id=(await store.get_tag("project")).id,
                                         status="active", origin="discovered")
        await store.add_atom_tag(fact.id, tag.id, "discovery")
    return fact


async def _seed_chapter(store, suffix, member_ids, body):
    ch = await store.store_fact(
        key=f"schema/cluster/{suffix}", value=body,
        tags=["schema", "tier:schema", "depth:1"], confidence=0.8,
        source="chapter_writer", provenance="cluster:s0|s1", node_kind="schema")
    for mid in member_ids:
        await store.add_edge(ch.id, mid, "member_of")
    return ch


async def _active_chapter_keys(store):
    async with store._db.execute(
        "SELECT key FROM facts WHERE agent_id=? AND status='active' AND node_kind='schema'",
        (store._agent_id,)) as cur:
        return {r[0] for r in await cur.fetchall()}


async def _active_chapter_snapshot(store):
    async with store._db.execute(
        "SELECT key, value, id, status FROM facts WHERE agent_id=? AND status='active' "
        "AND node_kind='schema'", (store._agent_id,)) as cur:
        return {(r[0], r[1], r[2], r[3]) for r in await cur.fetchall()}


def _cfg():
    # schema-writer + staleness recheck + containment guard all ON (defaults); silence the other
    # idle passes so this test isolates the three chapter mechanisms (and never touches an embedder).
    return MemoryConsolidationConfig(
        enabled=True, schema_writer_enabled=True, mining_enabled=False,
        reconcile_enabled=False, mint_tagging_enabled=False, tag_discovery_enabled=False,
    )


@pytest.mark.asyncio
async def test_triple_composition_single_pass(store):
    # (1) HEALTHY chapter — 2 grounded members, single session, unclustered (never re-derives).
    ra = "the harness runs every subagent read only by default for safety"
    rb = "the orchestrator delegates reading and searching to cheaper sonnet workers"
    h1 = await _seed_sem(store, ra, "s1", topic="reads", clustered=False)
    h2 = await _seed_sem(store, rb, "s1", topic="reads", clustered=False)
    ch_healthy = await _seed_chapter(store, "healthy1", [h1.id, h2.id], ra)

    # (2) STALE chapter — the '7-day taper' erosion shape: M3 is the sole bearer of "7".
    m1 = "the runner trains on tuesdays and saturdays every week without fail"
    m2 = "the runner is preparing for a september marathon race this autumn"
    m3 = "the coach added a 7 day taper to the training plan before race"
    taper_body = "the runner trains tuesdays saturdays with a 7 day taper for september marathon"
    r1 = await _seed_sem(store, m1, "s2", topic="fitness", clustered=False)
    r2 = await _seed_sem(store, m2, "s2", topic="fitness", clustered=False)
    r3 = await _seed_sem(store, m3, "s2", topic="fitness", clustered=False)
    ch_stale = await _seed_chapter(store, "stale1", [r1.id, r2.id, r3.id], taper_body)
    # Erode M3 on its own key — the active corpus loses "7", so the chapter's figure is unsupported.
    await store.store_fact(key=r3.key, value="the coach dropped the extra week from the plan",
                           tags=["sem", "pending_consolidation"], confidence=0.65,
                           source="transcript_mining", provenance="s2", node_kind="fact")

    # (3) CONTAINMENT — two grounded 1-member stubs whose members are a STRICT subset of a fresh
    # 2-session "subagents" cluster the writer will form. (Two stubs are required to exercise the
    # containment-SUPERSEDE counter: refresh-overlap re-keys the single best-overlap chapter, and
    # the guard supersedes the OTHER stranded one — the gap the guard was built to close.)
    sa1 = "the subagent runs read only by default in the harness environment always"
    sa2 = "the subagent build order is summarizer first then the citation checker"
    a1 = await _seed_sem(store, sa1, "s0", topic="subagents", clustered=True)
    a2 = await _seed_sem(store, sa2, "s1", topic="subagents", clustered=True)
    await _seed_chapter(store, "sub1", [a1.id], sa1)
    sub2 = await _seed_chapter(store, "sub2", [a2.id], sa2)

    assert await _active_chapter_keys(store) == {
        "schema/cluster/healthy1", "schema/cluster/stale1",
        "schema/cluster/sub1", "schema/cluster/sub2",
    }

    # --- ONE pass: recheck (stale) → write_schemas (containment), healthy untouched throughout ---
    llm = _DispatchLLM()
    report = await ConsolidationPass(store, _cfg(), llm=llm).run()
    assert not report.cancelled

    # All three mechanisms fired, in one pass, with self-consistent counters.
    assert report.chapters_redrafted_stale == 1          # the eroded taper chapter
    assert report.chapters_retired_stale == 0
    assert report.chapters_superseded_by_containment == 1  # the stranded stub the guard supersedes
    assert report.chapters_folded == 0
    assert report.schemas_written == 1                   # the merged "subagents" chapter
    # Healthy + both grounded stubs revalidate (all before the writer supersedes sub2). The healthy
    # path invokes the LLM zero times — only the stale re-draft and the writer generate.
    assert report.chapters_revalidated == 3
    assert llm.calls == 2

    # Exactly the expected active chapters (sub2 folded away under the merged subagents chapter).
    assert await _active_chapter_keys(store) == {
        "schema/cluster/healthy1", "schema/cluster/stale1", "schema/cluster/sub1",
    }
    active_sub = await store.get_fact("schema/cluster/sub1")

    # Append-only supersede chains — nothing deleted, every superseded row still queryable.
    sub2_hist = await store.get_fact_history("schema/cluster/sub2")
    assert [h.status for h in sub2_hist] == ["superseded"]
    assert sub2_hist[0].id == sub2.id
    assert sub2_hist[0].superseded_by == active_sub.id  # superseded BY the surviving chapter
    assert await store.get_fact("schema/cluster/sub2") is None

    stale_hist = await store.get_fact_history("schema/cluster/stale1")
    assert [h.status for h in stale_hist] == ["active", "superseded"]  # newest first
    assert stale_hist[1].id == ch_stale.id
    assert stale_hist[1].superseded_by == stale_hist[0].id

    healthy_hist = await store.get_fact_history("schema/cluster/healthy1")
    assert [h.status for h in healthy_hist] == ["active"]  # single row, never rewritten

    # Healthy chapter is byte-for-byte untouched (recheck's healthy branch writes nothing).
    healthy_now = await store.get_fact("schema/cluster/healthy1")
    assert healthy_now.id == ch_healthy.id
    assert healthy_now.value == ch_healthy.value
    assert healthy_now.updated_at == ch_healthy.updated_at

    # Stale chapter healed: the unsupported figure is gone from the live re-draft.
    stale_now = await store.get_fact("schema/cluster/stale1")
    assert stale_now.id != ch_stale.id
    assert "7" not in stale_now.value

    # --- SECOND pass changes nothing: converged, no churn (fold-corroboration is not a change). ---
    snap_before = await _active_chapter_snapshot(store)
    healthy_ts = healthy_now.updated_at
    report2 = await ConsolidationPass(store, _cfg(), llm=_DispatchLLM()).run()
    assert report2.chapters_redrafted_stale == 0
    assert report2.chapters_retired_stale == 0
    assert report2.chapters_superseded_by_containment == 0
    assert report2.schemas_written == 0  # the re-derived cluster folds into the existing chapter
    assert await _active_chapter_snapshot(store) == snap_before  # identical (key, value, id, status)
    assert (await store.get_fact("schema/cluster/healthy1")).updated_at == healthy_ts
