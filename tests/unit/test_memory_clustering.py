"""SEMA-01 lesson clustering (Phase 36) — pure-computation cluster discovery.

Covers: connected-component detection over the promoted-lesson pool via graph
neighborhood + FTS similarity (Task 1); cross-sitting stability filtering (Task 2);
adjacent tier:surprising_failure aux-member attachment (Task 3, PGATE-03 rider).
Every test also guards the invariant: clustering issues ZERO store writes.
"""
import pytest

from localharness.memory.clustering import (
    Cluster,
    _connected_components,
    _load_pool,
    _relatedness_edges,
    find_stable_clusters,
)
from localharness.memory.sqlite import MemoryStore

# Two `read` lessons that share the salient tokens "absolute"/"resolved" (FTS-related);
# a `grep` lesson that shares none (isolated). Content is fixture-fake (allowed in tests;
# only the SEMA-05 provable forbids fabricated lesson text — see 36-CONTEXT.md).
_READ_A = "The read tool returned FileNotFound on a relative path; retrying with the absolute path resolved the failure."
_READ_B = "The read tool raised a permission problem on a protected path; the absolute path form resolved it cleanly."
_READ_C = "The read tool timed out once then, using the absolute path directly, resolved on the second attempt."
_GREP_C = "The grep command required the fixed-string flag for literal square brackets during matching."
# A second domain sharing "docker"/"registry"/"container" but NO token with the read lessons.
_DOCK_A = "The docker build failed pulling from the registry; a container cache prune cleared the broken layer."
_DOCK_B = "The docker container refused to start until the registry credentials were refreshed before the pull."


@pytest.fixture
async def store(tmp_path):
    s = MemoryStore(agent_id="clus-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


async def _seed_learned(store, tool, tier, body, sessions, *, lesson="", node_kind="fact"):
    """A promoted lesson exactly as consolidation.py writes it (key learned/{tool}/{tier},
    confidence 0.8, provenance consolidated:{n}-episodes) PLUS its derived_from source
    gate/* candidates — one per session, each carrying provenance=session — wired with the
    same derived_from edge consolidation.py:242 creates, so the session spread is derivable
    from neighborhood(depth=1) exactly like the real graph. Returns the promoted Fact."""
    key = f"learned/{tool}/{tier}" + (f"/{lesson}" if lesson else "")
    promoted = await store.store_fact(
        key=key, value=body, tags=["consolidated", f"tier:{tier}"],
        confidence=0.8, source="consolidation",
        provenance=f"consolidated:{len(sessions)}-episodes", node_kind=node_kind,
    )
    for i, sess in enumerate(sessions):
        cand = await store.store_fact(
            key=f"gate/{tier}/{tool}/{lesson or 'L'}{i}/{sess}",
            value=f"Tool `{tool}` episode in {sess}: {body}",
            tags=["gate", f"tier:{tier}", "pending_consolidation"],
            confidence=0.65, source="write_gate", provenance=sess,
        )
        await store.add_edge(promoted.id, cand.id, "derived_from")
    return promoted


async def _fact_count(store) -> int:
    async with store._db.execute("SELECT COUNT(*) FROM facts") as cur:
        return (await cur.fetchone())[0]


# ---------------------------------------------------------------------------
# Task 1 — pool + relatedness graph + connected components
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_related_lessons_form_one_component(store):
    """FTS signal: two content-overlapping `read` lessons collapse into one component;
    an unrelated `grep` lesson stays a singleton."""
    await _seed_learned(store, "read", "resolved_error", _READ_A, ["s1"], lesson="a1")
    await _seed_learned(store, "read", "permission", _READ_B, ["s2"], lesson="b2")
    await _seed_learned(store, "grep", "flag_missing", _GREP_C, ["s3"], lesson="c3")

    pool = await _load_pool(store)
    assert len(pool) == 3  # only the promoted lessons — sub-0.7 gate candidates excluded

    adj = await _relatedness_edges(store, pool, fts_top_k=5, graph_depth=2)
    comps = _connected_components(pool, adj)
    assert sorted(len(c) for c in comps) == [1, 2]  # {read,read} related; grep isolated


@pytest.mark.asyncio
async def test_graph_shared_candidate_forms_component(store):
    """Graph signal (independent of FTS): two lessons with NO shared salient tokens but a
    shared derived_from candidate are connected through the depth-2 neighborhood."""
    d = await store.store_fact("learned/alpha/x/d", "alpha bravo charlie delta echo",
                               tags=["consolidated", "tier:x"], confidence=0.8,
                               provenance="consolidated:1-episodes")
    e = await store.store_fact("learned/omega/y/e", "foxtrot golfing hotels indiana juliett",
                               tags=["consolidated", "tier:y"], confidence=0.8,
                               provenance="consolidated:1-episodes")
    shared = await store.store_fact("gate/x/alpha/shared/sess", "shared candidate body",
                                    tags=["gate", "pending_consolidation"],
                                    confidence=0.65, provenance="sess")
    await store.add_edge(d.id, shared.id, "derived_from")
    await store.add_edge(e.id, shared.id, "derived_from")

    pool = await _load_pool(store)
    adj = await _relatedness_edges(store, pool, fts_top_k=5, graph_depth=2)
    comps = _connected_components(pool, adj)
    assert [len(c) for c in comps] == [2]  # linked purely via the shared candidate node


@pytest.mark.asyncio
async def test_cluster_contract_has_aux_hook():
    """The contract dataclass is frozen, carries the four fields, and aux_members defaults
    to an empty list (the PGATE-03 rider hook Task 3 populates)."""
    c = Cluster(members=[], sessions=frozenset(), depth=0)
    assert c.aux_members == []
    with pytest.raises(Exception):
        c.depth = 1  # frozen


# ---------------------------------------------------------------------------
# Task 2 — cross-sitting stability filter + find_stable_clusters entrypoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cross_sitting_related_pair_is_stable(store):
    """Two related lessons whose sources span 2 distinct sittings → one stable cluster."""
    await _seed_learned(store, "read", "resolved_error", _READ_A, ["s1"], lesson="a1")
    await _seed_learned(store, "read", "permission", _READ_B, ["s2"], lesson="b2")

    clusters = await find_stable_clusters(store)
    assert len(clusters) == 1
    assert len(clusters[0].members) == 2
    assert clusters[0].sessions == frozenset({"s1", "s2"})
    assert clusters[0].aux_members == []  # Task 3 populates; empty here


@pytest.mark.asyncio
async def test_single_sitting_no_cluster(store):
    """Two related lessons captured in ONE sitting are a double-stumble, not a chapter:
    the component exists but min_sessions filters it out (SEMA-01 'not one hot evening')."""
    await _seed_learned(store, "read", "resolved_error", _READ_A, ["s1"], lesson="a1")
    await _seed_learned(store, "read", "permission", _READ_B, ["s1"], lesson="b2")

    # the pair IS one component...
    pool = await _load_pool(store)
    adj = await _relatedness_edges(store, pool, fts_top_k=5, graph_depth=2)
    assert [len(c) for c in _connected_components(pool, adj)] == [2]
    # ...but not a STABLE cluster (single session).
    assert await find_stable_clusters(store) == []


@pytest.mark.asyncio
async def test_three_lesson_three_sitting_cluster(store):
    """A 3-lesson component spanning 3 sittings → one cluster, sessions of size 3."""
    await _seed_learned(store, "read", "resolved_error", _READ_A, ["s1"], lesson="a1")
    await _seed_learned(store, "read", "permission", _READ_B, ["s2"], lesson="b2")
    await _seed_learned(store, "read", "timeout", _READ_C, ["s3"], lesson="c3")

    clusters = await find_stable_clusters(store)
    assert len(clusters) == 1
    assert len(clusters[0].members) == 3
    assert clusters[0].sessions == frozenset({"s1", "s2", "s3"})


@pytest.mark.asyncio
async def test_clusters_sorted_biggest_first(store):
    """Deterministic order: the biggest chapter leads (writer's per-cycle budget)."""
    await _seed_learned(store, "read", "resolved_error", _READ_A, ["s1"], lesson="a1")
    await _seed_learned(store, "read", "permission", _READ_B, ["s2"], lesson="b2")
    await _seed_learned(store, "read", "timeout", _READ_C, ["s3"], lesson="c3")
    await _seed_learned(store, "docker", "build_fail", _DOCK_A, ["s4"], lesson="d1")
    await _seed_learned(store, "docker", "start_fail", _DOCK_B, ["s5"], lesson="d2")

    clusters = await find_stable_clusters(store)
    assert [len(c.members) for c in clusters] == [3, 2]  # read (3) before docker (2)
    # deterministic across calls
    again = await find_stable_clusters(store)
    assert [c.sessions for c in again] == [c.sessions for c in clusters]


@pytest.mark.asyncio
async def test_find_stable_clusters_issues_no_writes(store, monkeypatch):
    """The whole entrypoint is pure read: any write attempt must raise."""
    await _seed_learned(store, "read", "resolved_error", _READ_A, ["s1"], lesson="a1")
    await _seed_learned(store, "read", "permission", _READ_B, ["s2"], lesson="b2")
    before = await _fact_count(store)

    async def _boom(*a, **k):
        raise AssertionError("clustering must not write")

    monkeypatch.setattr(store, "store_fact", _boom)
    monkeypatch.setattr(store, "add_edge", _boom)

    clusters = await find_stable_clusters(store)
    assert len(clusters) == 1  # still produced the cluster
    assert await _fact_count(store) == before
