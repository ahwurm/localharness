"""web_fetch download cap: a huge body stops at _FETCH_MAX_BODY_BYTES and the retained
(lossless-store) text carries the cap notice — truncation is surfaced, never silent."""
from __future__ import annotations

from types import SimpleNamespace

from localharness.agent.context import ContentStore
from localharness.tools.builtin import web_tool as web_tool_mod
from localharness.tools.builtin.web_tool import WebFetchTool


class _FakeResp:
    headers = {"content-type": "text/plain"}
    encoding = "utf-8"
    url = "http://example.com/big"

    def raise_for_status(self):
        return None

    async def aiter_bytes(self):
        for _ in range(4):
            yield b"x" * 2_000_000  # 8 MB offered vs 4 MB cap


class _FakeStream:
    async def __aenter__(self):
        return _FakeResp()

    async def __aexit__(self, *exc):
        return False


class _FakeClient:
    def __init__(self, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url):
        return _FakeStream()


async def test_fetch_download_capped(monkeypatch):
    monkeypatch.setattr(
        web_tool_mod,
        "httpx",
        SimpleNamespace(AsyncClient=_FakeClient, HTTPError=web_tool_mod.httpx.HTTPError),
    )
    store = ContentStore()
    res = await WebFetchTool(store).run(url="http://example.com/big")
    assert res.success
    retained = store.get(res.metadata["fetch_id"])
    cap = web_tool_mod._FETCH_MAX_BODY_BYTES
    assert len(retained) <= cap + 200  # capped body + the notice line
    assert "download capped" in retained
