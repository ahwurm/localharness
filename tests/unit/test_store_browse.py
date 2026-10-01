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


# --------------------------------------------------------------------------- promote (no phone route)


async def test_the_occupant_satisfies_memory_browse(store):
    from localharness.plugins.api import MemoryBrowse

    b = StoreBrowse(store)
    assert isinstance(b, MemoryBrowse) and b.browse() is b


async def test_promote_with_a_project_layer_copies_into_the_global_store(tmp_path):
    from localharness.cli.memory_cmd import PROMOTE_PROVENANCE_PREFIX
    from localharness.memory.router import RecallRouter

    ws = make_store(tmp_path / "proj")
    await ws.open()
    gl = make_store(tmp_path / "home")  # constructed, NOT opened — the router opens it once
    router = RecallRouter(ws, gl)
    try:
        await ws.store_fact(key="notes/x", value="measure before claiming", tags=["lesson"],
                            confidence=0.9, source="user", provenance="seed")
        got = await StoreBrowse(ws, router, workspace_identity="/proj").promote("notes/x")
        assert got["promoted"] is True and got["message"].startswith("Promoted."), got
        copy = await gl.get_fact("notes/x")
        assert copy is not None and copy.status == "active"
        assert copy.provenance.startswith(PROMOTE_PROVENANCE_PREFIX) and "/proj" in copy.provenance
        assert (await ws.get_fact("notes/x")).value == "measure before claiming"  # original stays
    finally:
        await router.close()
        await ws.close()


async def test_promote_without_a_project_layer_is_refused(store):
    await _seed(store)
    got = await StoreBrowse(store, None).promote("proj/db")
    assert got["promoted"] is False
    assert got["message"].startswith("Promotion needs a project layer"), got


async def test_promote_missing_names_it(store):
    assert await StoreBrowse(store).promote("nope") == \
        {"promoted": False, "message": "No memory named 'nope' — nothing to promote."}
