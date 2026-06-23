"""P2 — per-agent store cutover: bind_agent_store_tools rebinds verb tools onto each agent's OWN
ContentStore (isolation), and the latent tool_result_get root-store leak is closed. Deterministic.
"""
from __future__ import annotations

import pytest

from localharness.agent.context import ContentStore
from localharness.tools.builtin import bind_agent_store_tools, register_builtin_tools
from localharness.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_bind_rebinds_web_tools_to_given_store():
    reg = ToolRegistry()
    await register_builtin_tools(reg)            # web_fetch / web_page_query present, bare (default)
    store = ContentStore()
    bind_agent_store_tools(reg, store)
    assert reg._tools["global"]["web_fetch"]._store is store
    assert reg._tools["global"]["web_page_query"]._store is store


@pytest.mark.asyncio
async def test_bind_is_noop_for_absent_tools():
    # A read-only child registry has no web/get tools — binding must NOT add a withheld capability.
    base = ToolRegistry()
    await register_builtin_tools(base)
    child = ToolRegistry.from_allowed(["read", "glob", "grep"], base_registry=base)
    bind_agent_store_tools(child, ContentStore())
    assert set(child._tools["global"].keys()) == {"read", "glob", "grep"}
    assert "web_fetch" not in child._tools["global"]
    assert "tool_result_get" not in child._tools["global"]


@pytest.mark.asyncio
async def test_web_fetch_and_query_share_the_bound_store():
    # web_fetch writes and web_page_query reads the SAME per-agent store after binding.
    reg = ToolRegistry()
    await register_builtin_tools(reg)
    store = ContentStore()
    bind_agent_store_tools(reg, store)
    h = store.put_web("the body of a page about widgets and gadgets")
    q = await reg._tools["global"]["web_page_query"].run(fetch_id=h, pattern="gadgets")
    assert q.success and "gadgets" in q.output


@pytest.mark.asyncio
async def test_tool_result_get_scoped_per_agent():
    """The latent leak fix: a child's tool_result_get reads ITS OWN store, never the root's bodies."""
    root_store = ContentStore()
    handle = root_store.put("ROOT-ONLY evicted body", origin="trusted")

    # A registry that HAS tool_result_get (e.g. a config child that requested it), then bound to the
    # child's OWN (empty) store — the exact per-agent rebind the dispatch path performs.
    base = ToolRegistry()
    await register_builtin_tools(base, eviction_store=ContentStore())  # registers tool_result_get
    child = ToolRegistry.from_allowed(["tool_result_get"], base_registry=base)
    child_store = ContentStore()
    bind_agent_store_tools(child, child_store)

    tool = child._tools["global"]["tool_result_get"]
    assert tool._store is child_store
    res = await tool.run(id=handle)
    assert not res.success and res.error_type == "not_found"   # cannot see the root's body
    assert root_store.get(handle) == "ROOT-ONLY evicted body"  # but the root still can
