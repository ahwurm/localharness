"""SEMA-01 lesson clustering (Phase 36, the chapter-writer) — PURE computation.

This module DISCOVERS which promoted lessons belong together. It performs NO LLM call
and NO store write: it only reads the graph + FTS index and composes them. The
chapter-writer (36-04) consumes find_stable_clusters()'s output and summarizes each
cluster into one schema node — "100 lessons -> one chapter" starts here, by deciding
WHICH lessons are one chapter.

Design (Claude's-discretion knobs are the signature defaults, real numbers):
  - Population = the PROMOTED lessons: active `learned/*` facts + `schema` nodes at/above
    the 0.7 injection line (_load_pool). The raw sub-0.7 `gate/*` candidates are NEVER the
    population — Phase 31 already clustered episodes->lesson; 36 clusters lesson->chapter
    one level up. The sub-0.7 `predgate/surprising_failure/*` stat rows are likewise
    excluded here by construction (conf<0.7); they arrive as aux_members (Task 3).
  - Relatedness (undirected): two pool lessons are RELATED if graph-adjacent within
    `graph_depth` hops (store.neighborhood) OR FTS-similar (they share a salient content
    token, probed via store.query_facts). Connected components >= min_cluster_size are
    candidate clusters.
  - Stability (find_stable_clusters): a component is a STABLE cluster only if its members'
    source sittings span >= min_sessions distinct sessions — recurring experience across
    evenings, not one hot evening (SEMA-01, enforced on real Phase-33 session units).
  - Each stable cluster is then enriched with adjacent tier:surprising_failure aux_members
    (PGATE-03 rider) — a pure READ; those sub-0.7 rows are attached for 36-04 to fold +
    drain, and are NEVER promoted into primary membership.

FTS note: this store's FTS5 combines query terms with implicit AND, so a full-value query
would collapse to near-exact-duplicate detection — useless for grouping related-but-distinct
lessons. The similarity signal here is therefore a per-salient-token union (each token an
independent probe), filtered to the >=0.7 promoted population so sub-0.7 candidates can
never dilute the top-k.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from localharness.memory.sqlite import Fact, FactQuery, _row_to_fact


@dataclass(frozen=True)
class Cluster:
    members: list[Fact]              # promoted lessons/schemas grouped together (>=2)
    sessions: frozenset[str]         # distinct source sittings the members span
    depth: int                       # max member schema-depth (0 for plain lessons); writer adds +1
    aux_members: list[Fact] = field(default_factory=list)  # adjacent tier:surprising_failure rows (Task 3); sub-0.7, folded+drained by 36-04, NEVER promoted


# ---------------------------------------------------------------------------
# Candidate pool + relatedness signals (pure reads)
# ---------------------------------------------------------------------------

async def _load_pool(store) -> list[Fact]:
    """The PROMOTED population: active lessons/schemas at/above the 0.7 injection line.
    Sub-0.7 gate/* + predgate/surprising_failure/* rows never clear the filter — the
    schema is the visibility artifact — so they cannot enter primary membership."""
    assert store._db is not None
    now = int(time.time())
    async with store._db.execute(
        f"SELECT {store._FACT_COLS} FROM facts "
        "WHERE agent_id = ? AND status = 'active' AND confidence >= 0.7 "
        "AND (key LIKE 'learned/%' OR node_kind = 'schema') "
        "AND (expires_at IS NULL OR expires_at > ?)",
        (store._agent_id, now),
    ) as cur:
        return [_row_to_fact(r) for r in await cur.fetchall()]


async def _load_failure_queue(store) -> list[Fact]:
    """The LIVE tier:surprising_failure queue: active, pending_consolidation, not-yet-faded
    (retrieval_strength >= 0.2). Faded rows are 36-04's drain concern — offering them as aux
    would blur the fold-vs-drain split — so they are withheld here. These sub-0.7 rows are
    attached to clusters as aux, NEVER promoted (raw observations never enter the index).
    Mirrors _load_pool's SELECT idiom + consolidation.py's `tags LIKE '%\"...\"%'` predicate."""
    assert store._db is not None
    now = int(time.time())
    async with store._db.execute(
        f"SELECT {store._FACT_COLS} FROM facts "
        "WHERE agent_id = ? AND status = 'active' "
        "AND tags LIKE '%\"tier:surprising_failure\"%' "
        "AND tags LIKE '%\"pending_consolidation\"%' "
        "AND retrieval_strength >= 0.2 "
        "AND (expires_at IS NULL OR expires_at > ?)",
        (store._agent_id, now),
    ) as cur:
        return [_row_to_fact(r) for r in await cur.fetchall()]


def _salient_tokens(value: str, *, max_tokens: int = 8) -> list[str]:
    """A bounded, deduped set of >=5-char content tokens — the FTS similarity probe.
    >=5 chars drops stopwords/punctuation and matches the grounding-check token floor
    used elsewhere in the subsystem."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in value.split():
        w = raw.strip("`.,:;!?()[]{}\"'").lower()
        if len(w) >= 5 and w not in seen:
            seen.add(w)
            out.append(w)
            if len(out) >= max_tokens:
                break
    return out


async def _relatedness_edges(store, pool, *, fts_top_k, graph_depth) -> dict[int, set[int]]:
    """Undirected relatedness adjacency over the pool. F~G iff G is within F's graph
    neighborhood (depth<=graph_depth) OR G shares a salient FTS token with F. Both signals
    are pure reads and only connect pool members (the min_confidence>=0.7 FTS filter keeps
    sub-0.7 candidates from surfacing as false neighbors)."""
    pool_ids = {f.id for f in pool}
    adj: dict[int, set[int]] = {f.id: set() for f in pool}

    def _link(a: int, b: int) -> None:
        adj[a].add(b)
        adj[b].add(a)

    for f in pool:
        # (a) GRAPH signal: derived_from / member_of neighborhood.
        for nid, _depth in await store.neighborhood(f.id, depth=graph_depth):
            if nid in pool_ids and nid != f.id:
                _link(f.id, nid)
        # (b) FTS signal: per salient token (implicit-AND store -> per-token union), scoped
        # to the >=0.7 promoted population so candidates never dilute the top-k.
        for tok in _salient_tokens(f.value):
            for hit in await store.query_facts(FactQuery(text=tok, limit=fts_top_k, min_confidence=0.7)):
                if hit.id in pool_ids and hit.id != f.id:
                    _link(f.id, hit.id)
    return adj


def _connected_components(pool, adj) -> list[list[Fact]]:
    """Connected components over the pool given the relatedness adjacency (iterative BFS)."""
    by_id = {f.id: f for f in pool}
    seen: set[int] = set()
    comps: list[list[Fact]] = []
    for f in pool:
        if f.id in seen:
            continue
        seen.add(f.id)
        stack = [f.id]
        comp: list[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj.get(cur, ()):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append([by_id[i] for i in comp])
    return comps


# ---------------------------------------------------------------------------
# Cross-sitting stability + the public entrypoint
# ---------------------------------------------------------------------------

def _depth_from_tags(tags: list[str]) -> int:
    """Schema depth from a `depth:N` tag (0 for a plain lesson). Local until 36-02 makes
    one canonical; 36-04 uses whichever lands. Tag-based (not a column) is zero-migration,
    consistent with the existing tier:X / salient tag convention."""
    for t in tags:
        if t.startswith("depth:"):
            try:
                return int(t.split(":", 1)[1])
            except ValueError:
                return 0
    return 0


async def _component_sessions(store, members) -> frozenset[str]:
    """The distinct sittings a component spans. A promoted lesson's own provenance is
    `consolidated:{N}-episodes` (NOT a session), so the session spread lives on its depth-1
    derived_from source candidates (gate/* keys), whose provenance IS a session id
    (consolidation.py:180). We union: (i) any member provenance that already looks like a
    session (defensive — non-`consolidated:` shapes), and (ii) the gate/* sources one hop out."""
    sessions: set[str] = set()
    for m in members:
        if m.provenance and not m.provenance.startswith("consolidated:"):
            sessions.add(m.provenance)
        src_ids = [nid for nid, _d in await store.neighborhood(m.id, depth=1) if nid != m.id]
        for src in await store.get_facts_by_ids(src_ids):
            if src.key.startswith("gate/") and src.provenance:
                sessions.add(src.provenance)
    return frozenset(sessions)


def _cluster_key(cluster: Cluster) -> str:
    """Stable tiebreak key: the sorted member keys (order-independent, deterministic)."""
    return "|".join(sorted(m.key for m in cluster.members))


def _member_tool(key: str) -> str | None:
    """learned/{tool}/... -> tool. Schema/other keys have no tool axis -> None."""
    parts = key.split("/")
    return parts[1] if len(parts) >= 2 and parts[0] == "learned" else None


def _failure_tool(key: str) -> str | None:
    """predgate/surprising_failure/{tool}/{day} -> tool (predictive_write_gate.py:182)."""
    parts = key.split("/")
    return parts[2] if len(parts) >= 3 and parts[0] == "predgate" else None


async def _attach_aux_failures(
    store, cluster_members, failure_queue, *, fts_top_k, graph_depth, aux_cap: int = 8,
) -> list[Fact]:
    """PGATE-03 rider, pure READ. A surprising_failure queue row Q is auxiliary to a cluster
    iff EITHER (a) DOMAIN: Q's tool matches any member's tool, OR (b) GRAPH/FTS: Q is within a
    member's graph neighborhood OR surfaces on a member's salient FTS token. Deduped, capped
    at aux_cap (bounded corpus). The rows' sub-0.7 confidence is left untouched — 36-04 folds
    them under the schema (member_of + a consumed tag) and drains them; they are never promoted."""
    if not failure_queue:
        return []
    q_ids = {q.id for q in failure_queue}
    member_tools = {t for t in (_member_tool(m.key) for m in cluster_members) if t}

    graph_adjacent: set[int] = set()
    fts_adjacent: set[int] = set()
    for m in cluster_members:
        for nid, _depth in await store.neighborhood(m.id, depth=graph_depth):
            if nid in q_ids:
                graph_adjacent.add(nid)
        for tok in _salient_tokens(m.value):
            # No min_confidence here: the queue rows ARE sub-0.7 and must surface.
            for hit in await store.query_facts(FactQuery(text=tok, limit=fts_top_k)):
                if hit.id in q_ids:
                    fts_adjacent.add(hit.id)

    matched: list[Fact] = []
    seen: set[int] = set()
    for q in failure_queue:
        if q.id in seen:
            continue
        tool = _failure_tool(q.key)
        if (tool is not None and tool in member_tools) or q.id in graph_adjacent or q.id in fts_adjacent:
            matched.append(q)
            seen.add(q.id)
            if len(matched) >= aux_cap:
                break
    return matched


async def find_stable_clusters(
    store, *, min_cluster_size: int = 2, min_sessions: int = 2,
    fts_top_k: int = 5, graph_depth: int = 2, aux_cap: int = 8,
) -> list[Cluster]:
    """Discover STABLE lesson clusters over the promoted population. A cluster is returned
    iff it has >= min_cluster_size (2) related members spanning >= min_sessions (2) distinct
    sittings — recurrence across evenings, not one hot evening (SEMA-01), enforced explicitly
    on real Phase-33 session units rather than merely inherited from each lesson's own
    promotion warrant. Each stable cluster is then enriched with adjacent
    tier:surprising_failure aux_members (PGATE-03 rider) — a pure READ that attaches the
    sub-0.7 stat rows for 36-04 to fold + drain; they are NEVER promoted into membership.
    Returned biggest-first (deterministic tiebreak) so the writer's per-cycle budget takes
    the largest chapters first. Pure read+compute: zero LLM, zero writes."""
    pool = await _load_pool(store)
    if len(pool) < min_cluster_size:
        return []
    adj = await _relatedness_edges(store, pool, fts_top_k=fts_top_k, graph_depth=graph_depth)
    failure_queue = await _load_failure_queue(store)  # loaded ONCE for the whole pass

    clusters: list[Cluster] = []
    for members in _connected_components(pool, adj):
        if len(members) < min_cluster_size:
            continue
        sessions = await _component_sessions(store, members)
        if len(sessions) < min_sessions:
            continue  # single-sitting grouping — not yet a stable chapter
        depth = max(_depth_from_tags(m.tags) for m in members)
        ordered = sorted(members, key=lambda m: m.key)
        aux = (
            await _attach_aux_failures(
                store, ordered, failure_queue,
                fts_top_k=fts_top_k, graph_depth=graph_depth, aux_cap=aux_cap,
            )
            if failure_queue else []
        )
        clusters.append(Cluster(members=ordered, sessions=sessions, depth=depth, aux_members=aux))
    return sorted(clusters, key=lambda c: (-len(c.members), _cluster_key(c)))
