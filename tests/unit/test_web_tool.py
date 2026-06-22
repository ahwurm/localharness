"""P1 — canonical web-stack hardening: SSRF guard, UNTRUSTED banner, JS-salvage, SearXNG."""
from __future__ import annotations

import sys
import types

import pytest

from localharness.tools.builtin import web_tool
from localharness.tools.builtin.web_tool import WebFetchTool, WebSearchTool


def _fake_httpx(monkeypatch, *, text="", json_data=None, content_type="text/html"):
    """Patch web_tool.httpx.AsyncClient to return a fixed body / JSON."""
    class _Resp:
        def __init__(self):
            self.text = text
            self.headers = {"content-type": content_type}
            self.url = "https://example.test/page"
        def raise_for_status(self): pass
        def json(self): return json_data

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, **k): return _Resp()

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
    assert result.output == web_tool._UNTRUSTED + "ok"


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
