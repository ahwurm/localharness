"""`/memory` window rendering (cli.memory_cmd.dispatch) over a seeded fixture store.

Model-free: every assertion is on the plain text a subcommand returns. Covers the overview, tag-path
listings (full path, bare child, bucket, paging, unknown), show (detail + supersede chain + ambient
line), forget (preview / confirm / already-retired / unknown), and search.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from localharness.cli import memory_cmd
from localharness.memory.sqlite import USER_FORGET_PROVENANCE_PREFIX, MemoryStore


def make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        agent_id="test-agent", division_id="test-div", org_id="default",
        base_dir=str(tmp_path),
    )


async def _file(store, key, value, conf, bucket, child=None, source="mined"):
    """Store a fact and file it under bucket[/child] via the tag graph (the real filing path)."""
    f = await store.store_fact(key=key, value=value, confidence=conf, source=source)
    b = await store.get_tag(bucket)
    await store.add_bucket_tag(f.id, b.id)
    if child:
        c = await store.get_tag(child)
        await store.add_atom_tag(f.id, c.id)
    return f


async def seeded(tmp_path) -> MemoryStore:
    store = make_store(tmp_path)
    await store.open()
    await _file(store, "port", "vLLM serves on port 8081", 0.9, "project", "ops")
    await _file(store, "gpu", "the GB10 has 119 GiB unified memory", 0.85, "project", "ops")
    await _file(store, "subagents", "subagents are read-only unless stated", 0.8, "project", "conventions")
    await _file(store, "stocks", "follows HBM / semiconductor stocks", 0.75, "personal", "preferences")
    await _file(store, "taper", "adds a pre-race taper before a 10k", 0.6, "personal", "health")
    # A discovery candidate (proposed child under personal).
    personal = await store.get_tag("personal")
    await store.create_tag("onsen", "discovery candidate (unincorporated)",
                           parent_id=personal.id, status="proposed")
    return store


# --------------------------------------------------------------------------- overview
@pytest.mark.asyncio
async def test_overview_shows_buckets_children_proposed_and_recent(tmp_path):
    store = await seeded(tmp_path)
    try:
        out = await memory_cmd.dispatch(store, "")
        # Both buckets, with children paths.
        assert "personal" in out and "project" in out
        assert "project/ops" in out and "personal/preferences" in out
        # ops has 2 filed atoms.
        assert "project/ops" in out and "(2)" in out
        # Proposed candidate rendered on its own (dim) line, with a count.
        assert "proposed" in out.lower() and "onsen" in out
        # Recent feed carries ids + confidence.
        assert "Most recent" in out and "#" in out and "conf" in out
    finally:
        await store.close()


# --------------------------------------------------------------------------- listing
@pytest.mark.asyncio
async def test_listing_full_path_lists_only_that_child(tmp_path):
    store = await seeded(tmp_path)
    try:
        out = await memory_cmd.dispatch(store, "project/ops")
        assert "vLLM serves on port 8081" in out
        assert "GB10" in out
        assert "read-only" not in out  # a conventions fact, not ops
        assert "project/ops" in out
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_listing_accepts_bare_child_name(tmp_path):
    store = await seeded(tmp_path)
    try:
        out = await memory_cmd.dispatch(store, "preferences")
        assert "HBM" in out
        assert "personal/preferences" in out  # resolved to its full path
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_listing_bucket_lists_all_its_atoms(tmp_path):
    store = await seeded(tmp_path)
    try:
        out = await memory_cmd.dispatch(store, "project")
        assert "vLLM serves on port 8081" in out and "read-only" in out  # ops + conventions
        assert "HBM" not in out  # a personal fact
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_listing_unknown_path_is_clear(tmp_path):
    store = await seeded(tmp_path)
    try:
        out = await memory_cmd.dispatch(store, "nonsense/bucket")
        assert "Unknown tag path" in out or "unknown tag path" in out.lower()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_listing_pages_at_twenty(tmp_path):
    store = make_store(tmp_path)
    await store.open()
    try:
        for i in range(25):
            await _file(store, f"c{i}", f"conventions fact number {i}", 0.8, "project", "conventions")
        page1 = await memory_cmd.dispatch(store, "project/conventions")
        assert page1.count("#") >= 20  # a full page of rows
        assert "more" in page1.lower()  # a paging hint
        assert "2" in page1  # points at the next page
        page2 = await memory_cmd.dispatch(store, "project/conventions 2")
        assert "page 2" in page2.lower()
        # The two pages don't overlap on the newest row.
        assert page1 != page2
    finally:
        await store.close()


# --------------------------------------------------------------------------- show
@pytest.mark.asyncio
async def test_show_full_detail_and_ambient_eligible(tmp_path):
    store = await seeded(tmp_path)
    try:
        f = await store.get_fact("port")
        out = await memory_cmd.dispatch(store, f"show {f.id}")
        assert "vLLM serves on port 8081" in out  # value, unclipped
        assert "port" in out  # key
        assert "project/ops" in out  # tag path
        assert "mined" in out  # source
        assert "0.90" in out  # confidence
        # The ambient-eligibility teaching line (eligible: conf >= floor).
        assert "ambient-eligible: yes" in out
        assert ">= 0.70 floor" in out
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_show_low_confidence_not_ambient(tmp_path):
    store = await seeded(tmp_path)
    try:
        f = await store.get_fact("taper")  # confidence 0.60 < floor
        out = await memory_cmd.dispatch(store, f"show {f.id}")
        assert "ambient-eligible: no" in out
        assert "0.60 < 0.70 floor" in out
        assert "not auto-injected" in out
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_show_supersede_chain_both_directions(tmp_path):
    store = make_store(tmp_path)
    await store.open()
    try:
        v1 = await store.store_fact(key="k", value="first value", confidence=0.9)
        v2 = await store.store_fact(key="k", value="second value", confidence=0.9)
        show_v2 = await memory_cmd.dispatch(store, f"show {v2.id}")
        assert f"#{v1.id}" in show_v2 and "replaces" in show_v2.lower()
        show_v1 = await memory_cmd.dispatch(store, f"show {v1.id}")
        assert f"#{v2.id}" in show_v1 and "replaced" in show_v1.lower()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_show_bad_and_unknown_id(tmp_path):
    store = await seeded(tmp_path)
    try:
        assert "Usage" in await memory_cmd.dispatch(store, "show abc")
        assert "No memory with id 987654" in await memory_cmd.dispatch(store, "show 987654")
    finally:
        await store.close()


# --------------------------------------------------------------------------- forget
@pytest.mark.asyncio
async def test_forget_preview_then_confirm(tmp_path):
    store = await seeded(tmp_path)
    try:
        f = await store.get_fact("stocks")
        # Bare form = preview; nothing retired yet.
        preview = await memory_cmd.dispatch(store, f"forget {f.id}")
        assert "About to forget" in preview
        assert f"/memory forget {f.id} confirm" in preview
        assert await store.get_fact("stocks") is not None  # still active
        # Confirm form retires it.
        done = await memory_cmd.dispatch(store, f"forget {f.id} confirm")
        assert "Forgotten" in done
        assert await store.get_fact("stocks") is None  # gone from active
        row = await store.get_fact_by_id(f.id)
        assert row.status == "superseded" and row.provenance.startswith(USER_FORGET_PROVENANCE_PREFIX)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_forget_already_retired_and_unknown(tmp_path):
    store = make_store(tmp_path)
    await store.open()
    try:
        v1 = await store.store_fact(key="k", value="v1", confidence=0.9)
        await store.store_fact(key="k", value="v2", confidence=0.9)  # supersedes v1
        msg = await memory_cmd.dispatch(store, f"forget {v1.id} confirm")
        assert "already retired" in msg.lower()
        assert "nothing to forget" in (await memory_cmd.dispatch(store, "forget 55555")).lower()
        assert "Usage" in await memory_cmd.dispatch(store, "forget notanid")
    finally:
        await store.close()


# --------------------------------------------------------------------------- search
@pytest.mark.asyncio
async def test_search_hits_and_miss(tmp_path):
    store = await seeded(tmp_path)
    try:
        hits = await memory_cmd.dispatch(store, "search vLLM")
        assert "8081" in hits and "#" in hits  # a hit with an id to follow with show/forget
        miss = await memory_cmd.dispatch(store, "search zzzznotpresent")
        assert "No memories matched" in miss
        assert "Usage" in await memory_cmd.dispatch(store, "search")
    finally:
        await store.close()


# --------------------------------------------------------------------------- guardrails
@pytest.mark.asyncio
async def test_dispatch_without_store_is_graceful():
    out = await memory_cmd.dispatch(None, "")
    assert "not available" in out.lower() or "isn't available" in out.lower()
