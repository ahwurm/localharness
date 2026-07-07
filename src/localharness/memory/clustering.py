"""SEMA-01 lesson clustering (Phase 36, the chapter-writer) — PURE computation.

This module DISCOVERS which promoted lessons belong together. It performs NO LLM call
and NO store write: it only reads the graph + FTS index and composes them. The
chapter-writer (36-04) consumes find_stable_clusters()'s output and summarizes each
cluster into one schema node — "100 lessons -> one chapter" starts here, by deciding
WHICH lessons are one chapter.

Design (Claude's-discretion knobs are the signature defaults, real numbers):
  - Population = the SEMANTIC pool (MOVE 1): active `sem/*`/`mined/*` atoms + `schema` nodes +
    settled corrections (`tier:reconcile_confirmed`) at/above pool-entry 0.6 (_load_pool). This
    describes the USER'S WORLD, not tool lessons: operational memory (`gate/*`, `predgate/*`,
    `learned/*`) is a SEPARATE track and is excluded (owner ruling c). Grouping runs over the
    episodic population and the chapter IS the promotion (>=0.7) — un-inverting CLS. The sub-0.7
    `predgate/surprising_failure/*` stat rows arrive only as aux_members (Task 3).
  - Relatedness (undirected, precision-tiered — run-2 ruling 1): two pool atoms are RELATED if
    graph-adjacent within `graph_depth` hops (store.neighborhood), OR same `sem/{topic}/` slug
    with >=1 shared salient token, OR different slugs with >=2 distinct shared salient tokens
    after dropping tokens above 30% pool document-frequency (generic-verb guard — one shared
    'requested' must not weld the pool into a mega-component). Connected components
    >= min_cluster_size are candidate clusters.
  - Stability (find_stable_clusters): a component is a STABLE cluster only if its members'
    source sittings span >= min_sessions distinct sessions — recurring experience across
    evenings, not one hot evening (SEMA-01, enforced on real Phase-33 session units).
  - Each stable cluster is then enriched with adjacent tier:surprising_failure aux_members
    (PGATE-03 rider) — a pure READ; those sub-0.7 rows are attached for 36-04 to fold +
    drain, and are NEVER promoted into primary membership.

FTS note: this store's FTS5 combines query terms with implicit AND, so a full-value query
would collapse to near-exact-duplicate detection. Member relatedness therefore intersects
_salient_tokens in memory (FTS can neither count distinct shared tokens nor see pool-level
document frequency); FTS probes remain only in aux-failure attachment (_attach_aux_failures).
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

from localharness.memory.sqlite import Fact, FactQuery, _row_to_fact

# Co-tag generic-hub guard (Stage B): a child tag on > _TAG_DF_FRACTION of the pool forms no
# edges, floored at _TAG_DF_FLOOR members so a small homogeneous cluster survives a tiny pool.
_TAG_DF_FRACTION = 0.30
_TAG_DF_FLOOR = 3


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
    """The SEMANTIC population (MOVE 1): active facts at/above pool-entry 0.6 that describe the
    USER'S WORLD — mined atoms (`sem/`, `mined/`), discovered schemas (chapters), settled user
    corrections (`tier:reconcile_confirmed`), AND user-remembered facts (the `remember` tag —
    tag-graph critique item 1: a `remember()`-sourced fact is declarative user/world content, not
    operational telemetry, so it is a first-class taggable pool member, mirroring the `mined/`
    fix). This filter is INCLUDE-ONLY by design:
    operational memory (`gate/`, `predgate/`, `learned/`) matches no arm and is thereby EXCLUDED
    — a separate track, never in the ontology (owner ruling c). A settled correction that happens
    to sit on a `gate/` key (shape-b confirm keeps its key) is a user-confirmed fact and MUST stay
    in, so it enters via the reconcile_confirmed arm — NOT despite a blanket prefix ban that would
    wrongly drop it. Pool entry (0.6) is BELOW the 0.7 injection line: grouping runs over the
    episodic population and admission stops doing the consolidator's job (un-inverts CLS); the
    chapter written from a stable cluster is the promotion, at >=0.7 as before."""
    assert store._db is not None
    now = int(time.time())
    async with store._db.execute(
        f"SELECT {store._FACT_COLS} FROM facts "
        "WHERE agent_id = ? AND status = 'active' AND confidence >= 0.6 "
        "AND (key LIKE 'sem/%' OR key LIKE 'mined/%' OR node_kind = 'schema' "
        "     OR tags LIKE '%\"tier:reconcile_confirmed\"%' "
        "     OR tags LIKE '%\"remember\"%') "
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


def _topic_slug(key: str) -> str | None:
    """sem/{slug}/{h8} -> slug; anything else (schemas, mined/ legacy, settled gate/ keys) has
    no topic axis -> None (such pairs take the strict cross-topic rule)."""
    parts = key.split("/")
    return parts[1] if len(parts) >= 3 and parts[0] == "sem" else None


async def _relatedness_edges(store, pool, *, fts_top_k, graph_depth) -> dict[int, set[int]]:
    """Undirected relatedness adjacency over the pool — GRAPH signal + CO-TAG signal. Word-overlap
    edges are REPLACED by shared-tag edges (tag-graph M3): chapters now form from the ontological
    relatedness word-matching structurally cannot see (run-3 markets: 3 one-topic atoms, <=1 shared
    token, 0 token edges). Edge rule:
      - graph: F~G iff G is within F's neighborhood (depth<=graph_depth) — unchanged (this is real
        derived_from/member_of structure, not word overlap, so it stays);
      - co-tag: F~G iff F and G share an active CHILD tag. BUCKET tags NEVER form edges — they are
        navigation, and 'project' as an edge source would mega-blob everything project
        (child_tags_for_atoms returns children only, so this is structural, not just policy).
    tag-df GUARD (the mega-blob defense, carried from tokens to tags): a child tag attached to
    > 30% of the pool forms NO edges (generic hub), floored at > 2 members so the statistic needs
    support in a tiny pool (mirrors the old token generic_cut). Pure read. `fts_top_k` is unused
    here (kept for signature stability; aux attachment still probes FTS)."""
    pool_ids = {f.id for f in pool}
    adj: dict[int, set[int]] = {f.id: set() for f in pool}

    def _link(a: int, b: int) -> None:
        adj[a].add(b)
        adj[b].add(a)

    # (a) GRAPH signal: derived_from / member_of neighborhood — unchanged.
    for f in pool:
        for nid, _depth in await store.neighborhood(f.id, depth=graph_depth):
            if nid in pool_ids and nid != f.id:
                _link(f.id, nid)

    # (b) CO-TAG signal (REPLACES word overlap): two atoms sharing an active CHILD tag link.
    child_tags = await store.child_tags_for_atoms(list(pool_ids), edge_eligible=True)
    tag_df = Counter(t for tags in child_tags.values() for t in tags)
    # generic-hub guard: a child tag on > 30% of the pool forms no edges. Floored at count 3
    # (not the token precedent's 2): the old token guard EXEMPTED same-slug pairs, so a same-topic
    # cluster always linked; co-tag has no such exemption, so the absolute floor must protect a
    # legitimate small (<=3-member) homogeneous cluster in a tiny pool from being read as a hub.
    # In a real (large) pool the 30% fraction dominates the floor, so scale behaviour is unchanged.
    df_cut = max(_TAG_DF_FLOOR, _TAG_DF_FRACTION * len(pool))
    generic = {t for t, c in tag_df.items() if c > df_cut}
    members_by_tag: dict[int, list[int]] = {}
    for aid, tags in child_tags.items():
        for t in tags - generic:
            members_by_tag.setdefault(t, []).append(aid)
    for member_ids in members_by_tag.values():
        for i, a in enumerate(member_ids):
            for b in member_ids[i + 1:]:
                _link(a, b)
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


async def _known_session_ids(store) -> frozenset[str]:
    """All session ids this agent has ever opened — the authoritative sitting registry."""
    assert store._db is not None
    async with store._db.execute(
        "SELECT id FROM sessions WHERE agent_id = ?", (store._agent_id,)
    ) as cur:
        return frozenset(r[0] for r in await cur.fetchall())


def _is_sitting(prov: str, known: frozenset[str]) -> bool:
    """Run-2 ruling 2: a bookkeeping provenance ('confirm:4517cb1b-…') counted as a session and
    faked cross-sitting stability. A provenance counts as a sitting iff it IS a sessions-table
    row, or it matches the session-id convention: no ':' — every derived/bookkeeping provenance
    the subsystem mints is 'marker:detail' (confirm:/revert-of:/retire:/consolidated:/cluster:/
    mined-from:), while real sitting ids (uuid4, sema05-*, designed-*) never carry a colon."""
    return bool(prov) and (prov in known or ":" not in prov)


async def _component_sessions(store, members) -> frozenset[str]:
    """The distinct SITTINGS a component spans (ruling 2: sessions must be sittings — never a
    reconcile/consolidation breadcrumb). We union: (i) member provenance that passes _is_sitting
    (a schema node's `cluster:sessA|sessB` provenance is EXPANDED into its constituent sitting
    ids first — counting the raw string as one fake session was the old MAJOR-5 pollution, and
    dropping it wholesale would make chapters-of-chapters permanently unstable; each expanded id
    still passes the filter), and (ii) the gate/* sources one hop out, whose provenance IS a
    session id by construction (consolidation.py:180) — same filter applied."""
    known = await _known_session_ids(store)
    sessions: set[str] = set()
    for m in members:
        provs = (
            m.provenance[len("cluster:"):].split("|")
            if m.provenance.startswith("cluster:") else [m.provenance]
        )
        sessions.update(p for p in provs if _is_sitting(p, known))
        src_ids = [nid for nid, _d in await store.neighborhood(m.id, depth=1) if nid != m.id]
        for src in await store.get_facts_by_ids(src_ids):
            if src.key.startswith("gate/") and _is_sitting(src.provenance, known):
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
