"""Phase 36 (SEMA-03/04): schema-fact store contract + schemas-first index render.

Task 1 proves the contract primitives (the tier:schema importance prior, the depth-tag
reader). Task 2 proves the index_mode=True render path renders a chapter in a "### Knowledge"
section ABOVE Persistent Facts, byte-stable across renders, zero bytes when there are no
chapters. Hand-written FIXTURE schema facts only — the chapter-writer (36-04) is not needed
to exercise this contract.
"""
from pathlib import Path

import pytest

from localharness.memory.sqlite import (
    SCHEMA_KEY_PREFIX,
    MemoryStore,
    _importance_prior,
    _schema_depth,
)


@pytest.fixture
async def store(tmp_path: Path):
    s = MemoryStore(agent_id="schema-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


# --- Task 1: contract primitives -------------------------------------------------

def test_schema_tier_importance_prior():
    """tier:schema carries a non-zero prior (0.5) so a chapter LEADS its section; with no
    explicit entry it would sink to the 0.0 floor (Pitfall 2) despite being a promotion."""
    assert _importance_prior(["tier:schema"], "") == 0.5


def test_schema_depth_parses_tag():
    assert _schema_depth(["schema", "depth:2"]) == 2


def test_schema_depth_defaults_zero_for_plain_lesson():
    assert _schema_depth(["schema"]) == 0


# --- Task 2: schemas-first render (index_mode=True path) -------------------------

async def _seed_schema(store, key_suffix: str, value: str):
    """A chapter fact exactly as 36-04 will write one: schema key prefix, node_kind='schema',
    tier:schema + depth:1 tags, promoted confidence (>= the 0.7 injection gate)."""
    await store.store_fact(
        key=SCHEMA_KEY_PREFIX + key_suffix,
        value=value,
        tags=["schema", "tier:schema", "depth:1"],
        confidence=0.8,
        node_kind="schema",
    )


@pytest.mark.asyncio
async def test_schema_renders_above_facts_and_not_inside_them(store: MemoryStore):
    """A chapter renders under '### Knowledge' ABOVE '### Persistent Facts', and its key
    never double-renders into the facts list (SEMA-04: gist routes, verbatim answers)."""
    await store.store_fact("fact/one", "first persistent fact body", confidence=0.8)
    await store.store_fact("fact/two", "second persistent fact body", confidence=0.8)
    await _seed_schema(store, "abc", "How vLLM behaves on this box: prefill wedges past 96k.")

    rendered = await store._render_memory_index(8)

    assert "### Knowledge" in rendered
    assert rendered.index("### Knowledge") < rendered.index("### Persistent Facts")  # schemas first
    facts_block = rendered[rendered.index("### Persistent Facts"):]
    assert (SCHEMA_KEY_PREFIX + "abc") not in facts_block  # not double-rendered as a fact
    assert "fact/one" in facts_block and "fact/two" in facts_block  # plain facts still render


@pytest.mark.asyncio
async def test_index_byte_stable_across_renders(store: MemoryStore):
    """RANK-04 / TIME-04: two renders with NO intervening write are byte-identical (the
    16.1s TTFT re-prefill @32k is real — the injected block must not churn within a sitting)."""
    await store.store_fact("fact/one", "a persistent fact body", confidence=0.8)
    await _seed_schema(store, "xyz", "A chapter that must render identically twice.")

    first = await store._render_memory_index(8)
    second = await store._render_memory_index(8)
    assert first == second


@pytest.mark.asyncio
async def test_empty_schema_set_adds_zero_bytes(store: MemoryStore):
    """No chapters -> the '### Knowledge' section is entirely absent (zero added bytes),
    preserving byte-identity for chapter-less stores."""
    await store.store_fact("fact/only", "just a plain fact", confidence=0.8)
    rendered = await store._render_memory_index(8)
    assert "### Knowledge" not in rendered
