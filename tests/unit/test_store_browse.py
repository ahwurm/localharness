"""46-03 (WEBP-03): StoreBrowse — the memory plugin's browse API (its browse() returns one) — answers the
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


async def test_the_browse_object_satisfies_memory_browse(store):
    from localharness.plugins.api import MemoryBrowse

    b = StoreBrowse(store)
    assert isinstance(b, MemoryBrowse)  # a plain browse object now; the memory plugin's browse() returns it


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


# ------------------------------------------- 48-02: plugin-internal delegates (NOT MemoryBrowse verbs)
# The value of these methods is the access boundary — /memory, `localharness memory` and the bench
# reach the store only through the plugin's own browse class — not new behaviour: each returns
# exactly what the store returns.


def _same(a, b):
    """Two store results equal field for field (Fact dataclasses, dicts, lists, scalars)."""
    return repr(a) == repr(b)


async def test_every_delegate_returns_exactly_what_the_store_returns(store):
    await _seed(store)
    b = StoreBrowse(store)
    fid = (await store.get_fact("proj/db")).id
    from localharness.memory.sqlite import FactQuery
    q = FactQuery(text=None, min_confidence=0.0)
    pairs = [
        (await b.list_groups(), await store.list_groups()),
        (await b.list_groups(limit=1, named_only=True), await store.list_groups(limit=1, named_only=True)),
        (await b.recent_facts(5), await store.recent_facts(5)),
        (await b.get_fact("proj/db"), await store.get_fact("proj/db")),
        (await b.get_fact_by_id(fid), await store.get_fact_by_id(fid)),
        (await b.get_fact_history("proj/db"), await store.get_fact_history("proj/db")),
        (await b.query_facts(q), await store.query_facts(q)),
        (await b.list_archived(), await store.list_archived()),
        (await b.list_archived(10, key="proj/db"), await store.list_archived(10, key="proj/db")),
        (await b.count_archived(), await store.count_archived()),
        (await b.restore_fact(999999), await store.restore_fact(999999)),
    ]
    for got, want in pairs:
        assert _same(got, want)
    assert await b.get_fact_by_id(fid) is not None


async def test_archive_delegates_return_what_consolidation_returns(store):
    from localharness.memory.consolidation import archive_dormant_facts, archive_listed_facts
    await _seed(store)
    b = StoreBrowse(store)
    fid = (await store.get_fact("proj/db")).id
    now = 2_000_000_000
    assert _same(await b.archive_dormant(dry_run=True, now=now),
                 await archive_dormant_facts(store, dry_run=True, now=now))
    assert _same(await b.archive_listed(ids=[fid], dry_run=True, now=now),
                 await archive_listed_facts(store, [fid], dry_run=True, now=now))


async def test_forget_fact_retires_only_the_id_previewed(store):
    """M4: a version a live turn superseded is never retired in its place."""
    b = StoreBrowse(store)
    old = await store.store_fact("proj/db", "postgres")
    await store.store_fact("proj/db", "sqlite")  # a live turn supersedes the previewed version
    assert await b.forget_fact(old.id) is False
    assert (await store.get_fact("proj/db")).value == "sqlite"


async def test_store_seeds_one_fact_exactly_as_store_fact(store):
    """Today's bench seed is `store_fact(key, value, confidence=1.0)`; the seed verb writes the
    same row (the store's own clamp and empty provenance included), with no embedding."""
    b = StoreBrowse(store)
    assert await b.store("k", "v") is None
    await store.store_fact("direct", "v", confidence=1.0)
    row = lambda f: (f.value, f.confidence, f.source, f.provenance, f.node_kind, f.tags, f.status)
    seeded, direct = await store.get_fact("k"), await store.get_fact("direct")
    assert row(seeded) == row(direct) and seeded.value == "v" and seeded.source == ""
    assert getattr(seeded, "embedding", None) is None  # no embedder ran
    await b.store("k2", "v2", confidence=0.5)
    await store.store_fact("direct2", "v2", confidence=0.5)
    assert (await store.get_fact("k2")).confidence == (await store.get_fact("direct2")).confidence


async def _render(reply) -> str:
    if isinstance(reply, str):
        return reply
    from rich.console import Console
    console = Console(record=True, width=100, force_terminal=False, color_system=None)
    console.print(reply)
    return console.export_text()


async def test_memory_dispatch_answers_the_same_over_the_browse_class(tmp_path):
    from localharness.cli.memory_cmd import dispatch
    a, c = make_store(tmp_path / "a"), make_store(tmp_path / "c")
    await a.open(), await c.open()
    try:
        for s in (a, c):
            await _seed(s)
        fid = (await a.get_fact("proj/db")).id
        assert fid == (await c.get_fact("proj/db")).id
        for arg in ("", f"show {fid}", "search zebra", f"forget {fid}", f"forget {fid} confirm",
                    f"show {fid}", "show nope", "bogus"):
            assert await _render(await dispatch(StoreBrowse(a), arg)) == await _render(await dispatch(c, arg)), arg
    finally:
        await a.close(), await c.close()


async def test_the_protocol_stays_five_verbs_and_a_five_verb_fake_still_passes(store):
    import typing
    from localharness.plugins.api import MemoryBrowse

    members = (typing.get_protocol_members(MemoryBrowse) if hasattr(typing, "get_protocol_members")
               else MemoryBrowse.__protocol_attrs__)
    assert sorted(members) == ["edit", "forget", "get", "promote", "search"]

    class Five:
        async def search(self, query): ...
        async def get(self, name): ...
        async def edit(self, name, content, origin=""): ...
        async def forget(self, name): ...
        async def promote(self, name): ...

    assert isinstance(Five(), MemoryBrowse)
    assert isinstance(StoreBrowse(store), MemoryBrowse)
