"""P1 — canonical web-stack hardening: SSRF guard, UNTRUSTED banner, JS-salvage, SearXNG."""
from __future__ import annotations

import sys
import types

import pytest

from localharness.agent.context import ContentStore
from localharness.tools.builtin import web_tool
from localharness.tools.builtin.web_tool import WebFetchTool, WebPageQueryTool, WebSearchTool


def _fake_httpx(monkeypatch, *, text="", json_data=None, content_type="text/html"):
    """Patch web_tool.httpx.AsyncClient to return a fixed body / JSON.

    `get` serves the SearXNG search path; `stream` serves the capped-download fetch path."""
    class _Resp:
        def __init__(self):
            self.text = text
            self.headers = {"content-type": content_type}
            self.url = "https://example.test/page"
            self.encoding = "utf-8"
        def raise_for_status(self): pass
        def json(self): return json_data
        async def aiter_bytes(self):
            yield text.encode("utf-8")

    class _Stream:
        async def __aenter__(self): return _Resp()
        async def __aexit__(self, *a): return False

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, **k): return _Resp()
        def stream(self, method, url, **k): return _Stream()

    monkeypatch.setattr(web_tool.httpx, "AsyncClient", _Client)


# --- SSRF guard ---------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [
    "http://127.0.0.1/x", "https://10.1.2.3/x", "http://192.168.0.5/x",
    "http://localhost:8000/x", "http://172.16.0.1/x", "http://[::1]/x",
])
async def test_web_fetch_ssrf_rejects_internal(bad):
    result = await WebFetchTool().run(url=bad)
    assert result.success is False
    assert "internal addresses" in result.error


@pytest.mark.asyncio
async def test_web_fetch_ssrf_allows_public(monkeypatch):
    _fake_httpx(monkeypatch, text="A real public article body. " * 10, content_type="text/plain")
    result = await WebFetchTool().run(url="https://example.test/page")
    assert result.success is True


# --- UNTRUSTED banner ---------------------------------------------------------

@pytest.mark.asyncio
async def test_web_fetch_untrusted_banner_on_body(monkeypatch):
    _fake_httpx(monkeypatch, text="A legitimate article body. " * 20, content_type="text/plain")
    result = await WebFetchTool().run(url="https://example.test/page")
    assert result.success is True
    assert result.output.startswith(web_tool._UNTRUSTED)


# --- JS-salvage (HTML only) ---------------------------------------------------

@pytest.mark.asyncio
async def test_web_fetch_js_salvage_rescues_title(monkeypatch):
    html = "<html><head><title>QNT hits new high</title></head><body><div></div></body></html>"
    _fake_httpx(monkeypatch, text=html, content_type="text/html")
    result = await WebFetchTool().run(url="https://example.test/js")
    assert result.success is True
    assert "low extractable text" in result.output
    assert "QNT hits new high" in result.output
    assert result.output.startswith(web_tool._UNTRUSTED)


@pytest.mark.asyncio
async def test_web_fetch_short_plaintext_is_not_js_salvaged(monkeypatch):
    # text/plain short bodies are legitimately short, not JS-blocked — no salvage note.
    _fake_httpx(monkeypatch, text="ok", content_type="text/plain")
    result = await WebFetchTool().run(url="https://example.test/api")
    assert result.success is True
    assert "low extractable text" not in result.output
    assert result.output.startswith(web_tool._UNTRUSTED + "ok")  # body shown, then the fetch_id tail


# --- search backends ----------------------------------------------------------

@pytest.mark.asyncio
async def test_web_search_ddgs_default_with_banner(monkeypatch):
    monkeypatch.setattr(web_tool, "_SEARXNG_URL", None)
    fake_ddgs = types.ModuleType("ddgs")
    class _DDGS:
        def text(self, q, max_results=5):
            return [{"title": "T1", "href": "https://a.test", "body": "snippet one"}]
    fake_ddgs.DDGS = _DDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_ddgs)
    result = await WebSearchTool().run(query="qnt")
    assert result.success is True
    assert result.output.startswith(web_tool._UNTRUSTED)
    assert "https://a.test" in result.output


@pytest.mark.asyncio
async def test_web_search_searxng_when_env_set(monkeypatch):
    monkeypatch.setattr(web_tool, "_SEARXNG_URL", "http://searx.test/search")
    _fake_httpx(monkeypatch, json_data={"results": [
        {"title": "SX1", "url": "https://sx.test/1", "content": "sx snippet"},
    ]}, content_type="application/json")
    result = await WebSearchTool().run(query="qnt")
    assert result.success is True
    assert result.output.startswith(web_tool._UNTRUSTED)
    assert "https://sx.test/1" in result.output
    assert "sx snippet" in result.output


# --- P4: lossless retrieve-and-verify (page store + web_page_query) ------------

@pytest.mark.asyncio
async def test_web_fetch_retains_full_page_and_returns_fetch_id(monkeypatch):
    store = ContentStore()
    page = "lorem ipsum " * 800  # ~9600 chars — well past the 5000 inline window
    _fake_httpx(monkeypatch, text=page, content_type="text/plain")
    result = await WebFetchTool(store).run(url="https://example.test/big")
    assert result.success is True
    assert result.truncated is True
    fid = result.metadata["fetch_id"]
    assert fid and f"fetch_id={fid}" in result.output
    # retained length == FULL page, not the inline cap
    assert len(store.get(fid)) == len(page)


@pytest.mark.asyncio
async def test_web_page_query_surfaces_content_past_the_inline_cap(monkeypatch):
    web_tool._reset_page_store()
    filler = "background prose. " * 400          # ~7200 chars of filler
    needle = "SPCX was added to the S&P 500 index on 2026-01-15."
    page = filler + needle + " and some trailing text."
    _fake_httpx(monkeypatch, text=page, content_type="text/plain")

    fetched = await WebFetchTool().run(url="https://example.test/article")
    fid = fetched.metadata["fetch_id"]
    assert needle not in fetched.output, "the disconfirming sentence is PAST the inline preview"

    q = await WebPageQueryTool().run(fetch_id=fid, pattern="S&P 500")
    assert q.success is True
    assert needle in q.output, "web_page_query must surface past-cap content losslessly"
    assert q.output.startswith(web_tool._UNTRUSTED)


@pytest.mark.asyncio
async def test_web_page_query_unknown_fetch_id_errors():
    web_tool._reset_page_store()
    q = await WebPageQueryTool().run(fetch_id="pg-does-not-exist", pattern="x")
    assert q.success is False
    assert "no retained page" in q.error


@pytest.mark.asyncio
async def test_web_page_query_no_match_is_explicit(monkeypatch):
    web_tool._reset_page_store()
    _fake_httpx(monkeypatch, text="a page about cats and dogs. " * 50, content_type="text/plain")
    fetched = await WebFetchTool().run(url="https://example.test/pets")
    q = await WebPageQueryTool().run(fetch_id=fetched.metadata["fetch_id"], pattern="quarterly earnings")
    assert q.success is True
    assert "no match" in q.output


@pytest.mark.asyncio
async def test_web_page_query_pages_under_registry_cap(monkeypatch):
    web_tool._reset_page_store()
    # 20 well-separated matches; with a wide window their combined context exceeds the output
    # budget, so the tool PAGES (drops the tail with a note) rather than returning a lossy 50k cut.
    block = "NEEDLE" + ("filler " * 700)  # ~4900 chars apart so windows don't all merge
    page = block * 20
    _fake_httpx(monkeypatch, text=page, content_type="text/plain")
    fetched = await WebFetchTool().run(url="https://example.test/many")
    q = await WebPageQueryTool().run(fetch_id=fetched.metadata["fetch_id"], pattern="NEEDLE", window=8000)
    assert q.success is True
    assert len(q.output) < 50_000, "must stay under the 50k registry cap (no lossy truncation)"
    assert "stay under the size cap" in q.output
