"""SEMA-01 lesson clustering (Phase 36) — pure-computation cluster discovery.

Covers: connected-component detection over the promoted-lesson pool via graph
neighborhood + FTS similarity (Task 1); cross-sitting stability filtering (Task 2);
adjacent tier:surprising_failure aux-member attachment (Task 3, PGATE-03 rider).
Every test also guards the invariant: clustering issues ZERO store writes.
"""
import pytest

from localharness.memory.clustering import (
    Cluster,
    _attach_aux_failures,
    _connected_components,
    _load_failure_queue,
    _load_pool,
    _relatedness_edges,
    find_stable_clusters,
)
from localharness.memory.sqlite import MemoryStore

# Two `read` lessons that share the salient tokens "absolute"/"resolved" (FTS-related);
# a `grep` lesson that shares none (isolated). Content is fixture-fake (allowed in tests;
# only the SEMA-05 provable forbids fabricated lesson text — see 36-CONTEXT.md).
_READ_A = "The read tool returned FileNotFound on a relative path; retrying with the absolute path resolved it."
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


async def _seed_failure(store, tool, day="20260705", *, rs=None, value=None):
    """A tier:surprising_failure queue row exactly as predictive_write_gate.py:181-192 writes
    it (key predgate/surprising_failure/{tool}/{day}, sub-0.7 confidence). Optionally fade its
    retrieval_strength (direct SQL — tests may write; the harness never fades via clustering)."""
    f = await store.store_fact(
        key=f"predgate/surprising_failure/{tool}/{day}",
        value=value or (f"`{tool}` had a surprising failure — a normally-reliable tool "
                        "errored (quadrant surprising_failure). Pending consolidation."),
        tags=["gate", "tier:surprising_failure", "pending_consolidation"],
        confidence=0.6, source="predictive_write_gate", provenance="sess-fail",
    )
    if rs is not None:
        await store._db.execute("UPDATE facts SET retrieval_strength = ? WHERE id = ?", (rs, f.id))
        await store._db.commit()
    return f


async def _seed_sem(store, body, session, *, topic="topic", conf=0.65):
    """A SEMANTIC atom exactly as MOVE-2 mining writes it: key sem/{topic}/{h8(body)}, node_kind
    'fact', provenance = the SOURCE session (so the cluster's session-spread comes from the atom's
    OWN provenance — no gate/ derived_from needed), confidence 0.65 (pool-entry, sub-injection).
    This is the post-MOVE-1 population; the old `learned/*` operational rows are a separate track."""
    import hashlib
    h = hashlib.sha1(body.strip().encode("utf-8")).hexdigest()[:8]
    return await store.store_fact(
        key=f"sem/{topic}/{h}", value=body, tags=["sem", "pending_consolidation"],
        confidence=conf, source="transcript_mining", provenance=session, node_kind="fact",
    )


# ---------------------------------------------------------------------------
# Task 1 — pool + relatedness graph + connected components
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_load_pool_is_semantic(store):
    """MOVE 1: _load_pool is the SEMANTIC population — sem/, mined/, schema nodes, and settled
    corrections (tier:reconcile_confirmed) at/above 0.6 — and EXCLUDES the operational track
    (gate/, predgate/, learned/), which is a separate hierarchy (owner ruling c). A settled
    correction on a gate/ key (shape-b confirm keeps its key) stays IN via the reconcile arm."""
    a = await _seed_sem(store, "the user is researching HBM memory makers this month", "s1")
    b = await store.store_fact(key="mined/deadbeef", value="the user runs a 27B model on vLLM",
                               tags=["mined"], confidence=0.65, provenance="s2", node_kind="fact")
    c = await store.store_fact(key="schema/cluster/xyz", value="a chapter body", tags=["schema"],
                               confidence=0.8, provenance="cluster:x", node_kind="schema")
    # A settled correction that KEPT its gate/ key (shape-b confirm) — must stay IN the pool.
    d = await store.store_fact(key="gate/correction/read/x/s3", value="vLLM listens on port 8081",
                               tags=["gate", "tier:reconcile_confirmed"], confidence=0.65, provenance="s3")
    # Operational rows + a sub-floor atom — all EXCLUDED.
    await store.store_fact(key="learned/read/resolved_error", value="op lesson", tags=["consolidated"],
                           confidence=0.8, provenance="consolidated:2-episodes")
    await _seed_failure(store, "bash_exec")  # predgate/
    await store.store_fact(key="gate/novelty/read/x/s9", value="raw op candidate",
                           tags=["gate", "pending_consolidation"], confidence=0.65, provenance="s9")
    await _seed_sem(store, "a low-confidence atom below the floor", "s4", conf=0.55)  # < 0.6

    pool_keys = {f.key for f in await _load_pool(store)}
    assert pool_keys == {a.key, b.key, c.key, d.key}, f"unexpected pool: {pool_keys}"

@pytest.mark.asyncio
async def test_related_lessons_form_one_component(store):
    """FTS signal: two content-overlapping `read` lessons collapse into one component;
    an unrelated `grep` lesson stays a singleton."""
    await _seed_sem(store, _READ_A, "s1")
    await _seed_sem(store, _READ_B, "s2")
    await _seed_sem(store, _GREP_C, "s3")

    pool = await _load_pool(store)
    assert len(pool) == 3  # only the promoted lessons — sub-0.7 gate candidates excluded

    adj = await _relatedness_edges(store, pool, fts_top_k=5, graph_depth=2)
    comps = _connected_components(pool, adj)
    assert sorted(len(c) for c in comps) == [1, 2]  # {read,read} related; grep isolated


@pytest.mark.asyncio
async def test_graph_shared_candidate_forms_component(store):
    """Graph signal (independent of FTS): two lessons with NO shared salient tokens but a
    shared derived_from candidate are connected through the depth-2 neighborhood."""
    d = await _seed_sem(store, "alpha bravo charlie delta echo", "s1")
    e = await _seed_sem(store, "foxtrot golfing hotels indiana juliett", "s2")
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
    await _seed_sem(store, _READ_A, "s1")
    await _seed_sem(store, _READ_B, "s2")

    clusters = await find_stable_clusters(store)
    assert len(clusters) == 1
    assert len(clusters[0].members) == 2
    assert clusters[0].sessions == frozenset({"s1", "s2"})
    assert clusters[0].aux_members == []  # Task 3 populates; empty here


@pytest.mark.asyncio
async def test_single_sitting_no_cluster(store):
    """Two related lessons captured in ONE sitting are a double-stumble, not a chapter:
    the component exists but min_sessions filters it out (SEMA-01 'not one hot evening')."""
    await _seed_sem(store, _READ_A, "s1")
    await _seed_sem(store, _READ_B, "s1")

    # the pair IS one component...
    pool = await _load_pool(store)
    adj = await _relatedness_edges(store, pool, fts_top_k=5, graph_depth=2)
    assert [len(c) for c in _connected_components(pool, adj)] == [2]
    # ...but not a STABLE cluster (single session).
    assert await find_stable_clusters(store) == []


@pytest.mark.asyncio
async def test_three_lesson_three_sitting_cluster(store):
    """A 3-lesson component spanning 3 sittings → one cluster, sessions of size 3."""
    await _seed_sem(store, _READ_A, "s1")
    await _seed_sem(store, _READ_B, "s2")
    await _seed_sem(store, _READ_C, "s3")

    clusters = await find_stable_clusters(store)
    assert len(clusters) == 1
    assert len(clusters[0].members) == 3
    assert clusters[0].sessions == frozenset({"s1", "s2", "s3"})


@pytest.mark.asyncio
async def test_clusters_sorted_biggest_first(store):
    """Deterministic order: the biggest chapter leads (writer's per-cycle budget)."""
    await _seed_sem(store, _READ_A, "s1")
    await _seed_sem(store, _READ_B, "s2")
    await _seed_sem(store, _READ_C, "s3")
    await _seed_sem(store, _DOCK_A, "s4")
    await _seed_sem(store, _DOCK_B, "s5")

    clusters = await find_stable_clusters(store)
    assert [len(c.members) for c in clusters] == [3, 2]  # read (3) before docker (2)
    # deterministic across calls
    again = await find_stable_clusters(store)
    assert [c.sessions for c in again] == [c.sessions for c in clusters]


@pytest.mark.asyncio
async def test_find_stable_clusters_issues_no_writes(store, monkeypatch):
    """The whole entrypoint is pure read: any write attempt must raise."""
    await _seed_sem(store, _READ_A, "s1")
    await _seed_sem(store, _READ_B, "s2")
    before = await _fact_count(store)

    async def _boom(*a, **k):
        raise AssertionError("clustering must not write")

    monkeypatch.setattr(store, "store_fact", _boom)
    monkeypatch.setattr(store, "add_edge", _boom)

    clusters = await find_stable_clusters(store)
    assert len(clusters) == 1  # still produced the cluster
    assert await _fact_count(store) == before


# ---------------------------------------------------------------------------
# Task 3 — adjacent tier:surprising_failure aux_members (PGATE-03 rider, pure read)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_load_failure_queue_excludes_faded(store):
    """The queue = active, pending, not-yet-faded (retrieval_strength >= 0.2). Faded rows
    are 36-04's drain concern, never offered as aux."""
    live = await _seed_failure(store, "read", "20260701")            # rs default 0.5
    faded = await _seed_failure(store, "read", "20260702", rs=0.1)   # below the 0.2 gate
    ids = {f.id for f in await _load_failure_queue(store)}
    assert live.id in ids
    assert faded.id not in ids


@pytest.mark.asyncio
async def test_same_tool_failure_attaches_as_aux(store):
    """A surprising_failure row on a cluster member's tool attaches (domain match); the
    sub-0.7 row is attached, NEVER promoted."""
    await _seed_sem(store, _READ_A, "s1")
    await _seed_sem(store, _READ_B, "s2")
    fail = await _seed_failure(store, "read", value="a recorded failure where the absolute path resolved it")

    clusters = await find_stable_clusters(store)
    assert len(clusters) == 1
    assert fail.key in {f.key for f in clusters[0].aux_members}
    assert all(f.confidence < 0.7 for f in clusters[0].aux_members)  # untouched, sub-line


@pytest.mark.asyncio
async def test_unrelated_tool_failure_not_attached(store):
    """A generic surprising_failure row on an unrelated tool with no graph/FTS link does
    not attach to the read cluster."""
    await _seed_sem(store, _READ_A, "s1")
    await _seed_sem(store, _READ_B, "s2")
    fail = await _seed_failure(store, "bash_exec")  # different tool, generic content

    clusters = await find_stable_clusters(store)
    attached = {f.key for c in clusters for f in c.aux_members}
    assert fail.key not in attached


@pytest.mark.asyncio
async def test_fts_adjacent_failure_attaches_despite_tool(store):
    """Graph/FTS adjacency REINFORCES the tool match, it doesn't gate it: a queue row on a
    different tool but sharing content tokens with a member attaches via the FTS branch."""
    await _seed_sem(store, _READ_A, "s1")
    await _seed_sem(store, _READ_B, "s2")
    fail = await _seed_failure(
        store, "write_file",
        value="A recorded failure where the absolute path resolved differently than usual.",
    )

    clusters = await find_stable_clusters(store)
    assert fail.key in {f.key for f in clusters[0].aux_members}  # tool mismatch, shared tokens


@pytest.mark.asyncio
async def test_aux_members_capped(store):
    """aux_cap bounds the folded corpus even when many FTS-adjacent rows match. (Post-MOVE-1 the
    members are semantic — no tool axis — so the 10 rows attach via the FTS arm on the shared
    'absolute/resolved' tokens; fts_top_k is widened so all 10 candidates surface and aux_cap,
    not the probe width, is what bounds them to 8.)"""
    await _seed_sem(store, _READ_A, "s1")
    await _seed_sem(store, _READ_B, "s2")
    for i in range(10):
        await _seed_failure(store, "read", f"2026070{i}", value=f"absolute path resolved case {i}")
    clusters = await find_stable_clusters(store, aux_cap=8, fts_top_k=15)
    assert len(clusters[0].aux_members) == 8


@pytest.mark.asyncio
async def test_attach_aux_issues_no_writes(store, monkeypatch):
    """Aux attachment is a pure READ — any write attempt must raise; queue rows untouched."""
    await _seed_sem(store, _READ_A, "s1")
    await _seed_sem(store, _READ_B, "s2")
    await _seed_failure(store, "read", value="absolute path resolved here")
    before = await _fact_count(store)

    async def _boom(*a, **k):
        raise AssertionError("aux attachment must not write")

    monkeypatch.setattr(store, "store_fact", _boom)
    monkeypatch.setattr(store, "add_edge", _boom)

    clusters = await find_stable_clusters(store)
    assert clusters[0].aux_members  # aux was attached
    assert await _fact_count(store) == before


@pytest.mark.asyncio
async def test_attach_aux_failures_direct(store):
    """Unit-level: _attach_aux_failures matches same-tool, dedups, and returns the row."""
    m1 = await _seed_learned(store, "read", "resolved_error", _READ_A, ["s1"], lesson="a1")
    m2 = await _seed_learned(store, "read", "permission", _READ_B, ["s2"], lesson="b2")
    fail = await _seed_failure(store, "read")
    queue = await _load_failure_queue(store)
    aux = await _attach_aux_failures(store, [m1, m2], queue, fts_top_k=5, graph_depth=2, aux_cap=8)
    assert [f.key for f in aux] == [fail.key]
