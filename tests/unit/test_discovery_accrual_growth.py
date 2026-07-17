"""#94 — tag-discovery evidence accrual fires ONLY on real growth.

`discover_tags` unconditionally called `bump_tag_evidence(..., last_accrual_ts=now)` whenever an
unchanged cluster still matched a proposed candidate. That refreshed the recency anchor every pass,
so the 21d staleness prune could NEVER fire on a candidate that keeps weakly matching but never
grows — the live store had 41 candidates, 0 eligible, 0 prunable, 40/41 sharing one bulk
last_accrual_ts (zombie candidates).

Fix: bump only when the matched cluster added >= 1 genuinely new member atom or a new distinct
sitting vs the candidate's stored membership. Unchanged clusters no longer refresh last_accrual_ts,
so the existing prune becomes reachable. Floors/score are untouched — single-sitting candidates now
age out at 21d unless the topic genuinely recurs (by design).
"""
import asyncio
import hashlib

import pytest

from localharness.memory.discovery import discover_tags, prune_stale_candidates, _NAME_MARKER
from localharness.memory.sqlite import MemoryStore


@pytest.fixture
async def store(tmp_path):
    s = MemoryStore(agent_id="disc-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


class _OneHotEmbedder:
    def __init__(self, vocab):
        self.vocab = vocab

    def embed(self, texts):
        return [[1.0 if w in t.lower() else 0.0 for w in self.vocab] for t in texts]


class _NamerLLM:
    def __init__(self, name="x"):
        self.name = name

    async def complete(self, prompt: str) -> str:
        return self.name if _NAME_MARKER in prompt else ""


async def _seed_bucket_atom(store, value, session, *, bucket="project"):
    h = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    f = await store.store_fact(key=f"sem/disc/{h}", value=value, tags=["sem", "pending_consolidation"],
                               confidence=0.65, source="transcript_mining", provenance=session,
                               node_kind="fact")
    b = await store.get_tag(bucket)
    await store.add_atom_tag(f.id, b.id, "mint")
    return f


async def _cofire(store, atoms):
    await store.record_activation_trace(stimulus="probe", fired_ids=[a.id for a in atoms],
                                        injected_ids=[a.id for a in atoms], source="memory_search")


_DAY = 86400


@pytest.mark.asyncio
async def test_unchanged_cluster_does_not_refresh_accrual(store):
    """A single-sitting candidate that re-matches the SAME atoms a later pass keeps its original
    last_accrual_ts — the recency anchor is NOT reset (bump did not fire)."""
    a = await _seed_bucket_atom(store, "clusterx alpha detail", "d1")
    b = await _seed_bucket_atom(store, "clusterx bravo detail", "d1")  # same sitting -> stays proposed
    await _cofire(store, [a, b])
    emb = _OneHotEmbedder(["clusterx"])
    t1 = 1_000_000
    r1 = await discover_tags(store, _NamerLLM(), asyncio.Event(), embedder=emb, now=t1)
    assert r1.proposed, "a candidate must form"
    name = r1.proposed[0]
    assert (await store.get_tag(name)).last_accrual_ts == t1

    # Second pass, SAME cluster, later clock — nothing grew.
    await discover_tags(store, _NamerLLM(), asyncio.Event(), embedder=emb, now=t1 + 5 * _DAY)
    assert (await store.get_tag(name)).last_accrual_ts == t1


@pytest.mark.asyncio
async def test_new_member_refreshes_accrual(store):
    """Adding a genuinely new member atom to a matched cluster IS growth — bump fires, refreshing
    the recency anchor."""
    a = await _seed_bucket_atom(store, "clustery alpha detail", "d1")
    b = await _seed_bucket_atom(store, "clustery bravo detail", "d1")
    await _cofire(store, [a, b])
    emb = _OneHotEmbedder(["clustery"])
    t1 = 1_000_000
    r1 = await discover_tags(store, _NamerLLM(), asyncio.Event(), embedder=emb, now=t1)
    name = r1.proposed[0]
    assert (await store.get_tag(name)).last_accrual_ts == t1

    c = await _seed_bucket_atom(store, "clustery gamma detail", "d1")  # NEW member
    await _cofire(store, [a, b, c])
    t2 = t1 + 5 * _DAY
    await discover_tags(store, _NamerLLM(), asyncio.Event(), embedder=emb, now=t2)
    assert (await store.get_tag(name)).last_accrual_ts == t2


@pytest.mark.asyncio
async def test_new_sitting_refreshes_accrual(store):
    """A new DISTINCT sitting for the same members is also growth (distinct_sittings rises)."""
    a = await _seed_bucket_atom(store, "clusterq alpha detail", "d1")
    b = await _seed_bucket_atom(store, "clusterq bravo detail", "d1")
    await _cofire(store, [a, b])
    emb = _OneHotEmbedder(["clusterq"])
    t1 = 1_000_000
    r1 = await discover_tags(store, _NamerLLM(), asyncio.Event(), embedder=emb, now=t1)
    name = r1.proposed[0]

    # A third atom from a DIFFERENT sitting joins -> distinct_sittings goes 1 -> 2.
    c = await _seed_bucket_atom(store, "clusterq gamma detail", "d2")
    await _cofire(store, [a, b, c])
    t2 = t1 + 3 * _DAY
    await discover_tags(store, _NamerLLM(), asyncio.Event(), embedder=emb, now=t2)
    assert (await store.get_tag(name)).last_accrual_ts == t2


@pytest.mark.asyncio
async def test_unchanged_matching_candidate_ages_out(store):
    """The payoff: a candidate that keeps matching an UNCHANGED cluster is now reachable by the 21d
    staleness prune (previously the every-pass bump made it immortal)."""
    a = await _seed_bucket_atom(store, "clusterz alpha detail", "d1")
    b = await _seed_bucket_atom(store, "clusterz bravo detail", "d1")
    await _cofire(store, [a, b])
    emb = _OneHotEmbedder(["clusterz"])
    t1 = 1_000_000
    r1 = await discover_tags(store, _NamerLLM(), asyncio.Event(), embedder=emb, now=t1)
    name = r1.proposed[0]

    # 22 days later the SAME cluster still re-matches, but nothing grew.
    t2 = t1 + 22 * _DAY
    await discover_tags(store, _NamerLLM(), asyncio.Event(), embedder=emb, now=t2)

    pruned = await prune_stale_candidates(store, now=t2)
    assert pruned >= 1
    assert (await store.get_tag(name)).status == "retired"
