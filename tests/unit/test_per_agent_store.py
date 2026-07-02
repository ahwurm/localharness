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


# --- dispatch-level isolation (8.6): per-agent stores never share bodies ------

def _fake_httpx_by_url(monkeypatch, mapping):
    from localharness.tools.builtin import web_tool

    class _Resp:
        def __init__(self, text):
            self.text = text
            self.headers = {"content-type": "text/plain"}
            self.url = "https://x.test/"
            self.encoding = "utf-8"

        def raise_for_status(self):
            pass

        def json(self):
            return None

        async def aiter_bytes(self):
            yield self.text.encode("utf-8")

    def _resp_for(url):
        for key, text in mapping.items():
            if key in url:
                return _Resp(text)
        return _Resp("unmapped")

    class _Stream:
        def __init__(self, url):
            self._url = url

        async def __aenter__(self):
            return _resp_for(self._url)

        async def __aexit__(self, *a):
            return False

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, **k):
            return _resp_for(url)

        def stream(self, method, url, **k):
            return _Stream(url)

    monkeypatch.setattr(web_tool.httpx, "AsyncClient", _Client)


@pytest.mark.asyncio
async def test_per_agent_store_isolation_across_dispatch(mock_llm_client, bus, tmp_path, monkeypatch):
    """Two dispatched agents fetch different pages into their OWN stores; neither store ever holds
    the other agent's body. This is the per-agent isolation that keeps the blind verifier blind."""
    from localharness.agent.context import ContextManager, _content_handle
    from localharness.agent.permissions import PermissionEvaluator
    from localharness.agent.subagent import (
        dispatch_search_verifier_subagent,
        dispatch_web_subagent,
    )

    monkeypatch.setenv("RESEARCH_RIGOR", "fast")  # no nested verifier; keep the researcher a leaf
    monkeypatch.setenv("LOCALHARNESS_VERIFICATION_LEDGER_DIR", str(tmp_path))
    page_r = "RESEARCHER PAGE about alpha widgets. " * 50
    page_v = "VERIFIER PAGE: SPCX joined the S&P 500. " * 50
    _fake_httpx_by_url(monkeypatch, {"r.test": page_r, "v.test": page_v})

    Response, ToolCall = mock_llm_client.Response, mock_llm_client.ToolCall
    llm_r = mock_llm_client([
        Response(content=None, tool_calls=[ToolCall(id="f1", name="web_fetch",
                 arguments={"url": "https://r.test/a"})]),
        Response(content="summary: widgets"),
    ])
    llm_v = mock_llm_client([
        Response(content=None, tool_calls=[ToolCall(id="f1", name="web_fetch",
                 arguments={"url": "https://v.test/b"})]),
        Response(content='{"verdict":"SUPPORTED","evidence":"on page"}'),
    ])

    base = ToolRegistry()
    await register_builtin_tools(base)
    store_r, store_v = ContentStore(), ContentStore()
    await dispatch_web_subagent(
        "research widgets", llm=llm_r, bus=bus, base_registry=base, parent_session_id="run",
        permission_evaluator=PermissionEvaluator(), context_manager=ContextManager(content_store=store_r),
    )
    await dispatch_search_verifier_subagent(
        "claim: SPCX in S&P 500\nentity: SPCX\nsource_url: https://v.test/b",
        llm=llm_v, bus=bus, base_registry=base, parent_session_id="run",
        permission_evaluator=PermissionEvaluator(), context_manager=ContextManager(content_store=store_v),
    )

    # each agent retained ITS OWN page under its own pg-1
    assert "RESEARCHER PAGE" in (store_r.get("pg-1") or "")
    assert "VERIFIER PAGE" in (store_v.get("pg-1") or "")
    # and neither store holds the other's body (by content handle) — no cross-agent leak
    assert store_v.get(_content_handle(page_r)) is None
    assert store_r.get(_content_handle(page_v)) is None


# --- the capability grant (P-CRUNCH A): read-through ONLY granted parent handles ----

@pytest.mark.asyncio
async def test_tool_result_get_reads_granted_parent_handle():
    """The cruncher grant end-to-end: a child whose store is built with (parent, granted={h})
    resolves the GRANTED parent handle via tool_result_get, but NOT an ungranted one."""
    parent = ContentStore()
    granted_h = parent.put("granted parent body")
    secret_h = parent.put("ungranted secret body")

    base = ToolRegistry()
    await register_builtin_tools(base, eviction_store=ContentStore())
    child = ToolRegistry.from_allowed(["tool_result_get"], base_registry=base)
    bind_agent_store_tools(child, ContentStore(parent=parent, granted=frozenset({granted_h})))

    tool = child._tools["global"]["tool_result_get"]
    ok = await tool.run(id=granted_h)
    assert ok.success and ok.output == "granted parent body"   # the capability
    denied = await tool.run(id=secret_h)
    assert not denied.success and denied.error_type == "not_found"  # ungranted stays invisible
