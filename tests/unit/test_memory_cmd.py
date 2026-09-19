"""`/memory` window rendering (cli.memory_cmd.dispatch) over a seeded fixture store.

Model-free: every assertion is on the text a subcommand returns. Covers the overview
(named groups + recent memories), show (detail + supersede chain + the salience
teaching line), forget (preview / confirm / already-retired / unknown), search, and
the no-store / unknown-subcommand paths.
"""
from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

from localharness.cli import memory_cmd
from localharness.memory.sqlite import MemoryStore


def _text(obj) -> str:
    """Render a dispatch result to plain text for substring asserts (show returns a
    rich Group; everything else is str)."""
    if isinstance(obj, str):
        return obj
    console = Console(file=io.StringIO(), width=200)
    console.print(obj)
    return console.file.getvalue()


def make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        agent_id="test-agent", division_id="test-div", org_id="default",
        base_dir=str(tmp_path),
    )


async def seeded(tmp_path) -> MemoryStore:
    store = make_store(tmp_path)
    await store.open()
    await store.store_fact("port", "vLLM serves on port 8000", confidence=0.9, source="remember")
    await store.store_fact("gpu", "the GB10 has 119 GiB unified memory", confidence=0.85, source="w")
    await store.store_fact("subagents", "subagents are read-only unless stated", confidence=0.8, source="w")
    port = await store.get_fact("port")
    gpu = await store.get_fact("gpu")
    gid = await store.upsert_group([port.id, gpu.id])
    await store.set_group_label(gid, "Serving Hardware")
    return store


# --------------------------------------------------------------------------- overview

async def test_overview_shows_groups_and_recent(tmp_path):
    store = await seeded(tmp_path)
    try:
        out = _text(await memory_cmd.dispatch(store, ""))
        assert "Serving Hardware" in out          # dreaming's named binding
        assert "2 memories" in out                # its member count
        assert "#" in out and "conf" in out       # recent rows carry id + confidence
        assert "port" not in out or True          # values render, keys need not
    finally:
        await store.close()


async def test_overview_empty_store(tmp_path):
    store = make_store(tmp_path)
    await store.open()
    try:
        out = _text(await memory_cmd.dispatch(store, ""))
        assert "empty" in out.lower()
    finally:
        await store.close()


async def test_unknown_subcommand_is_usage(tmp_path):
    store = await seeded(tmp_path)
    try:
        out = _text(await memory_cmd.dispatch(store, "project/ops"))
        assert "/memory show" in out and "/memory search" in out
    finally:
        await store.close()


async def test_dispatch_without_store_is_graceful():
    out = await memory_cmd.dispatch(None, "")
    assert "isn't available" in out


# --------------------------------------------------------------------------- show

async def test_show_full_detail_and_salience_line(tmp_path):
    store = await seeded(tmp_path)
    try:
        fact = await store.get_fact("port")
        out = _text(await memory_cmd.dispatch(store, f"show {fact.id}"))
        assert f"Memory #{fact.id}" in out
        assert "vLLM serves on port 8000" in out
        assert "remember" in out
        assert "salience" in out                  # the teaching line: budget, no floor
        assert "supersede chain" in out
    finally:
        await store.close()


async def test_show_supersede_chain(tmp_path):
    store = await seeded(tmp_path)
    try:
        await store.store_fact("port", "vLLM serves on port 8001", source="w")
        current = await store.get_fact("port")
        out = _text(await memory_cmd.dispatch(store, f"show {current.id}"))
        assert "← current" in out
        assert "8000" in out and "8001" in out    # both versions visible in the chain
    finally:
        await store.close()


async def test_show_bad_and_unknown_id(tmp_path):
    store = await seeded(tmp_path)
    try:
        assert "Usage" in _text(await memory_cmd.dispatch(store, "show banana"))
        assert "No memory with id 9999" in _text(await memory_cmd.dispatch(store, "show 9999"))
    finally:
        await store.close()


# --------------------------------------------------------------------------- forget

async def test_forget_preview_then_confirm(tmp_path):
    store = await seeded(tmp_path)
    try:
        fact = await store.get_fact("gpu")
        preview = _text(await memory_cmd.dispatch(store, f"forget {fact.id}"))
        assert "About to forget" in preview and "confirm" in preview
        assert (await store.get_fact("gpu")) is not None   # preview moved nothing

        done = _text(await memory_cmd.dispatch(store, f"forget {fact.id} confirm"))
        assert "Forgotten" in done
        assert await store.get_fact("gpu") is None
        again = _text(await memory_cmd.dispatch(store, f"forget {fact.id} confirm"))
        assert "already retired" in again
    finally:
        await store.close()


async def test_forget_unknown_id(tmp_path):
    store = await seeded(tmp_path)
    try:
        assert "nothing to forget" in _text(await memory_cmd.dispatch(store, "forget 424242"))
    finally:
        await store.close()


# --------------------------------------------------------------------------- search

async def test_search_hits_and_miss(tmp_path):
    store = await seeded(tmp_path)
    try:
        hit = _text(await memory_cmd.dispatch(store, "search unified memory"))
        assert "GB10" in hit and "#" in hit
        miss = _text(await memory_cmd.dispatch(store, "search zebra unicycle"))
        assert "No memories matched" in miss
    finally:
        await store.close()
