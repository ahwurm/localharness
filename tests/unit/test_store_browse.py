"""46-03 (WEBP-03): StoreBrowse — the memory slot's transitional occupant (ROADMAP D3) — answers the
five MemoryBrowse verbs over a REAL MemoryStore with today's phone record shape and provenance, so
the phone's memory routes can read through the slot without changing what the phone gets."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from localharness.memory.browse import StoreBrowse
from localharness.memory.sqlite import MemoryStore
from localharness.plugins.api import BrowseQuery

pytestmark = pytest.mark.asyncio

ROW_KEYS = {"name", "value", "status", "confidence", "source", "node_kind", "tags", "updated_at",
            "provenance"}
LONG = "the owner prefers terse replies " * 40  # longer than any display clip


def make_store(base: Path, global_base: Path | None = None) -> MemoryStore:
    return MemoryStore(agent_id="test-agent", division_id="test-div", org_id="default",
                       base_dir=str(base),
                       global_base_dir=str(global_base) if global_base is not None else None)


@pytest.fixture
async def store(tmp_path):
    s = make_store(tmp_path)
    await s.open()
    try:
        yield s
    finally:
        await s.close()


async def _seed(store):
    await store.store_fact(key="pref/style", value=LONG, tags=["pref"], confidence=0.8,
                           source="user", provenance="seed", node_kind="preference")
    await store.store_fact(key="proj/db", value="sqlite with fts5 zebra", tags=["proj"],
                           confidence=0.6, source="user", provenance="seed")


async def test_search_returns_full_rows_with_the_nine_keys(store):
    await _seed(store)
    rows = await StoreBrowse(store).search(BrowseQuery(text="", min_confidence=0.0, limit=200))
    assert {r["name"] for r in rows} == {"pref/style", "proj/db"}
    assert all(set(r) == ROW_KEYS for r in rows)
    assert next(r for r in rows if r["name"] == "pref/style")["value"] == LONG  # unclipped


async def test_search_text_narrows_and_default_floor_is_zero(store):
    await _seed(store)
    b = StoreBrowse(store)
    assert [r["name"] for r in await b.search(BrowseQuery(text="zebra"))] == ["proj/db"]
    assert len(await b.search(BrowseQuery())) == 2  # min_confidence None -> 0.0, nothing filtered


async def test_get_returns_fact_and_history_or_none(store):
    await _seed(store)
    b = StoreBrowse(store)
    await b.edit("pref/style", "short replies")
    got = await b.get("pref/style")
    assert got["fact"]["value"] == "short replies" and set(got["fact"]) == ROW_KEYS
    assert LONG in [h["value"] for h in got["history"]]
    assert await b.get("missing") is None


async def test_edit_unchanged_edited_and_missing(store):
    await _seed(store)
    b = StoreBrowse(store)
    assert await b.edit("pref/style", LONG + "  ") == {"status": "unchanged", "name": "pref/style"}
    assert await b.edit("pref/style", "brief", origin="web") == \
        {"status": "edited", "name": "pref/style"}
    f = await store.get_fact("pref/style")
    assert f.value == "brief" and f.source == "user_edit"
    assert re.fullmatch(r"user_edit@\d+;web", f.provenance), f.provenance
    assert (list(f.tags), f.confidence, f.node_kind) == (["pref"], 0.8, "preference")
    await b.edit("proj/db", "postgres")
    g = await store.get_fact("proj/db")
    assert re.fullmatch(r"user_edit@\d+", g.provenance), g.provenance
    assert g.node_kind == "fact"
    assert await b.edit("missing", "x") == {"status": "missing", "name": "missing"}


async def test_forget_retires_and_keeps_history(store):
    await _seed(store)
    b = StoreBrowse(store)
    assert await b.forget("proj/db") is True
    got = await b.get("proj/db")
    assert got["fact"] is None and got["history"]
    assert await b.forget("missing") is False
