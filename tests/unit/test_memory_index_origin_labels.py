"""Origin-labelled render mode on the injected memory index (Phase 42, plan 01, task 2).

`_render_memory_index_with_ids` gains two keyword-only parameters, both of whose defaults
reproduce today's exact bytes:

  * `origin_label` — when non-empty, every rendered line carries a composite token,
    `[workspace#12]` / `[global#7]`. `facts.id` is AUTOINCREMENT **per database**, so a
    scope-merged blend showing a bare `#12` from two stores would be ambiguous.
  * `include_preamble` — False drops the leading INDEX instructions, so the 42-02 router
    merging two blocks owns exactly one preamble.

This plan ships NO consumer. The file exists so 42-02/42-03 can be graded on routing
alone: the byte-identity test below is an explicit expected STRING, not `default ==
default`, so a future reflow of the injected preamble reddens THIS file rather than
silently changing every user's prompt.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from localharness.memory.sqlite import MemoryStore

# Copied character-for-character from sqlite.py's render. Duplicated ON PURPOSE:
# if the source literal is reflowed, this constant is what turns that into a failure.
PREAMBLE = (
    "This is an INDEX, not the full memory. Each line below is one persistent fact "
    "(name: short description). Call `memory_get(name)` for a fact's full body, or "
    "`memory_search(query)` to search fact contents.\n\n"
)


def make_store(tmp_path: Path) -> MemoryStore:
    """The make_store idiom from tests/unit/test_memory_store.py:18."""
    return MemoryStore(
        agent_id="test-agent",
        division_id="test-div",
        org_id="default",
        base_dir=str(tmp_path),
    )


@pytest.fixture
async def store(tmp_path: Path) -> MemoryStore:
    s = make_store(tmp_path)
    await s.open()
    yield s
    await s.close()


# ------------------------------------------------------------------ #
# 1. Defaults render TODAY'S EXACT BYTES.
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_default_render_is_byte_identical_to_the_shipped_surface(store: MemoryStore) -> None:
    """The whole default-preserving claim, written as a literal rather than a tautology.

    One chapter + one fact + no sittings, so the expected string is fully determined
    (no cross-row ORDER BY tie to make this flaky)."""
    chapter = await store.store_fact("hbm-cluster", "chapter body", node_kind="schema")
    fact = await store.store_fact("a-fact", "alpha body")

    text, ids = await store._render_memory_index_with_ids(8)

    expected = (
        PREAMBLE
        + "### Knowledge (1 chapters)\n"
        + "- hbm-cluster: chapter body\n"
        + "\n"
        + "### Persistent Facts (1)\n"
        + "- a-fact: alpha body"
    )
    assert text == expected
    # The ids ride out separately and are NOT rendered under the default.
    assert ids == [chapter.id, fact.id]
    # No origin token of ANY shape under the default — including the empty-label
    # `[#1]` an unconditionally-applied prefix would produce.
    assert re.search(r"\[\w*#\d+\]", text) is None, text


# ------------------------------------------------------------------ #
# 2. A labelled render carries the composite token on fact lines.
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_origin_label_prefixes_fact_lines_with_store_and_row(store: MemoryStore) -> None:
    fact = await store.store_fact("a-fact", "alpha body")

    text, ids = await store._render_memory_index_with_ids(8, origin_label="workspace")

    assert f"- [workspace#{fact.id}] a-fact: alpha body" in text
    assert ids == [fact.id]
    # The label names the STORE, so it must be the caller's word, not a constant.
    other, _ = await store._render_memory_index_with_ids(8, origin_label="global")
    assert f"- [global#{fact.id}] a-fact: alpha body" in other


# ------------------------------------------------------------------ #
# 3. A labelled render has NO unlabelled line — schema chapters too.
#    (Same store, same id space; one shared line builder is why.)
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_schema_chapters_carry_the_token_too(store: MemoryStore) -> None:
    chapter = await store.store_fact("hbm-cluster", "chapter body", node_kind="schema")
    await store.store_fact("a-fact", "alpha body")
    await store.store_fact("b-fact", "beta body")

    text, _ids = await store._render_memory_index_with_ids(8, origin_label="workspace")

    assert f"- [workspace#{chapter.id}] hbm-cluster: chapter body" in text
    bullets = [ln for ln in text.splitlines() if ln.startswith("- ")]
    assert len(bullets) == 3  # one chapter + two facts, no sittings seeded
    unlabelled = [ln for ln in bullets if not re.match(r"^- \[workspace#\d+\] ", ln)]
    assert unlabelled == [], unlabelled


# ------------------------------------------------------------------ #
# 4. include_preamble=False drops EXACTLY the preamble.
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_include_preamble_false_drops_exactly_the_preamble(store: MemoryStore) -> None:
    await store.store_fact("hbm-cluster", "chapter body", node_kind="schema")
    await store.store_fact("a-fact", "alpha body")

    full, full_ids = await store._render_memory_index_with_ids(8)
    bare, bare_ids = await store._render_memory_index_with_ids(8, include_preamble=False)

    assert full.startswith(PREAMBLE)
    assert PREAMBLE not in bare
    # The RELATIONSHIP, not just "something got shorter": "it dropped something"
    # must not be able to pass for "it dropped exactly the preamble".
    assert full.replace(PREAMBLE, "", 1) == bare
    assert bare_ids == full_ids


# ------------------------------------------------------------------ #
# 5. Pitfall 1: fact ids are per-database. Two stores WILL collide.
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_same_id_in_two_stores_renders_distinguishably(tmp_path: Path) -> None:
    ws = make_store(tmp_path / "workspace")
    gl = make_store(tmp_path / "global")
    await ws.open()
    await gl.open()
    try:
        ws_fact = await ws.store_fact("shared-key", "same body")
        gl_fact = await gl.store_fact("shared-key", "same body")
        # Assert the COLLISION first — otherwise this test could pass because the two
        # stores happened to hand out different ids, proving nothing about the token.
        assert ws_fact.id == gl_fact.id

        ws_text, _ = await ws._render_memory_index_with_ids(8, origin_label="workspace")
        gl_text, _ = await gl._render_memory_index_with_ids(8, origin_label="global")
        ws_line = next(ln for ln in ws_text.splitlines() if "shared-key" in ln)
        gl_line = next(ln for ln in gl_text.splitlines() if "shared-key" in ln)

        assert ws_line != gl_line
        assert ws_line == f"- [workspace#{ws_fact.id}] shared-key: same body"
        assert gl_line == f"- [global#{gl_fact.id}] shared-key: same body"
        # They differ ONLY in the token: strip it and the payloads are identical.
        assert re.sub(r"^- \[[a-z]+#\d+\] ", "", ws_line) == re.sub(r"^- \[[a-z]+#\d+\] ", "", gl_line)
    finally:
        await ws.close()
        await gl.close()
