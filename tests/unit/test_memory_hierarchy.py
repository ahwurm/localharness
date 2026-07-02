"""Phase 32 (v2.0 Hierarchical Memory) — hierarchy depth: HIER-01..04.

Covers: the factored reduce (HIER-01, byte-identical batching/termination), gist-tree
persistence with derived_from/member_of edges (HIER-02), structure-aware retrieval
(HIER-03), and the number-provenance net extended to memory (HIER-04).
"""
from pathlib import Path

import pytest

from localharness.core.reduce import ReduceLevel, batch_by_budget, hierarchical_reduce
from localharness.memory.hierarchy import flag_unverified_figures, persist_gist_tree
from localharness.memory.sqlite import MemoryStore


@pytest.fixture
async def store(tmp_path: Path):
    s = MemoryStore(agent_id="hier-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


# ---------------------------------------------------------------------------
# HIER-01: the factored reduce is the cruncher's exact algorithm
# ---------------------------------------------------------------------------

def test_batch_by_budget_matches_inline_greedy_semantics():
    items = ["a" * 40, "b" * 40, "c" * 40, "d" * 100, "e" * 10]
    batches = batch_by_budget(items, budget=90)
    # Greedy (verbatim the cruncher's inline loop): close a batch when the NEXT item
    # would exceed the budget; an oversized item lands alone and closes immediately.
    assert batches == [["a" * 40, "b" * 40], ["c" * 40], ["d" * 100], ["e" * 10]]
    assert [x for b in batches for x in b] == items  # nothing dropped, order kept


@pytest.mark.asyncio
async def test_hierarchical_reduce_terminates_and_traces():
    calls: list[list[str]] = []

    async def combine(batch: list[str]) -> str:
        calls.append(batch)
        return f"[gist of {len(batch)}]"

    items = ["x" * 50 for _ in range(8)]  # 400 chars + separators > budget 120
    out, level, trace = await hierarchical_reduce(items, budget=120, combine_partial=combine)
    assert level >= 1 and len(out) < 8
    assert len("\n\n".join(out)) <= 120 or len(out) == 1
    # Trace records every batch→gist pair, level by level (the HIER-02 seam).
    assert [t.level for t in trace] == list(range(1, level + 1))
    for t in trace:
        assert len(t.batches) == len(t.outputs)


@pytest.mark.asyncio
async def test_small_input_is_zero_levels_zero_calls():
    async def combine(batch):  # pragma: no cover - must not be called
        raise AssertionError("combine called on a fitting input")

    out, level, trace = await hierarchical_reduce(["short"], budget=1000, combine_partial=combine)
    assert out == ["short"] and level == 0 and trace == []


# ---------------------------------------------------------------------------
# HIER-04: the number net, extended to memory
# ---------------------------------------------------------------------------

def test_flag_unverified_figures_reuses_the_shipped_net():
    assert flag_unverified_figures("Revenue was $5,140 million", ["revenue: $5,140 million"]) == []
    flags = flag_unverified_figures("Revenue was $9,999 million", ["revenue: $5,140 million"])
    assert flags and "9,999" in flags[0]


# ---------------------------------------------------------------------------
# HIER-02: gist-tree persistence
# ---------------------------------------------------------------------------

def _fake_trace() -> tuple[list[str], list[ReduceLevel], str]:
    leaves = [f"[section {i}] fact {i}: value {i * 11}" for i in range(1, 5)]
    l1 = ReduceLevel(
        level=1,
        batches=[leaves[:2], leaves[2:]],
        outputs=["gist A: values 11 and 22", "gist B: values 33 and 44"],
    )
    l2 = ReduceLevel(
        level=2,
        batches=[["gist A: values 11 and 22", "gist B: values 33 and 44"]],
        outputs=["combined gist: 11, 22, 33, 44"],
    )
    final = "final answer citing 11 and 44"
    return leaves, [l1, l2], final


@pytest.mark.asyncio
async def test_persist_gist_tree_builds_schema_gists_edges(store: MemoryStore):
    leaves, trace, final = _fake_trace()
    index_before = await store._render_memory_index(10)

    written = await persist_gist_tree(
        store, question="what are the values?", leaf_extracts=leaves, trace=trace,
        final_answer=final, session_id="sess-7", source_handles=["h1"],
    )
    assert written == 1 + 3 + 1  # schema + 3 gists + final

    from localharness.memory.sqlite import FactQuery
    all_facts = await store.query_facts(FactQuery(min_confidence=0.0, limit=100))
    schemas = [f for f in all_facts if f.node_kind == "schema"]
    gists = [f for f in all_facts if f.node_kind == "gist"]
    assert len(schemas) == 1 and len(gists) == 4
    assert all("sess-7" in f.provenance for f in schemas + gists)  # verbatim pointer

    # L2 gist derived_from both L1 gists; final derived_from L2; all member_of schema.
    l2 = next(f for f in gists if f.key.endswith("L2-0"))
    walk = dict(await store.neighborhood(l2.id, depth=1, limit=20))
    l1_ids = {f.id for f in gists if "/L1-" in f.key}
    assert l1_ids <= set(walk)  # both parents reachable at depth 1
    assert schemas[0].id in walk

    # Gists route, they don't inject: the block is unchanged (below 0.7 threshold).
    assert await store._render_memory_index(10) == index_before


@pytest.mark.asyncio
async def test_gist_with_ungrounded_figure_is_tagged(store: MemoryStore):
    leaves = ["[section 1] margin was 12%"]
    bad = ReduceLevel(level=1, batches=[leaves], outputs=["margin was 47%"])  # DRM lure
    await persist_gist_tree(
        store, question="q", leaf_extracts=leaves, trace=[bad],
        final_answer="", session_id="s", source_handles=[],
    )
    from localharness.memory.sqlite import FactQuery
    gist = [f for f in await store.query_facts(FactQuery(min_confidence=0.0, limit=50))
            if f.node_kind == "gist"][0]
    assert "unverified-figures" in gist.tags


# ---------------------------------------------------------------------------
# HIER-03: structure-aware retrieval — FTS entry point → graph neighborhood
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_memory_search_surfaces_graph_neighborhood(store: MemoryStore):
    from localharness.tools.builtin.memory_tools import MemorySearchTool

    leaf = await store.store_fact("mu-hbm-fact", "Micron HBM supply is tight", confidence=0.9)
    gist = await store.store_fact(
        "gist/run1/L1-0", "memory-chip supply gists", confidence=0.6,
        source="cruncher", node_kind="gist",
    )
    schema = await store.store_fact(
        "schema/doc/run1", "semiconductor supply analysis", confidence=0.6,
        source="cruncher", node_kind="schema",
    )
    await store.add_edge(gist.id, leaf.id, "derived_from")
    await store.add_edge(gist.id, schema.id, "member_of")

    result = await MemorySearchTool(store)._execute(query="HBM supply")
    assert result.success
    assert "Related (graph neighborhood of top hit):" in result.output
    assert "gist/run1/L1-0 [gist]" in result.output  # the hit's gist context surfaced
