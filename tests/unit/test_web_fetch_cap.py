"""web_fetch download cap: a huge body stops at _FETCH_MAX_BODY_BYTES and the retained
(lossless-store) text carries the cap notice — truncation is surfaced, never silent."""
from __future__ import annotations

import httpx

from localharness.agent.context import ContentStore
from localharness.tools.builtin import web_tool as web_tool_mod
from localharness.tools.builtin.web_tool import WebFetchTool
from tests.unit.test_web_fetch_guard import public_web


class _EightMegabytes(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.offered = 0

    async def __aiter__(self):
        for _ in range(4):
            self.offered += 1
            yield b"x" * 2_000_000  # 8 MB offered vs 4 MB cap


async def test_fetch_download_capped(monkeypatch):
    public_web(monkeypatch)
    body = _EightMegabytes()
    monkeypatch.setattr(web_tool_mod, "_TRANSPORT", httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"content-type": "text/plain"}, stream=body)))
    store = ContentStore()
    res = await WebFetchTool(store).run(url="http://example.com/big")
    assert res.success
    retained = store.get(res.metadata["fetch_id"])
    cap = web_tool_mod._FETCH_MAX_BODY_BYTES
    assert len(retained) <= cap + 200  # capped body + the notice line
    assert "download capped" in retained
    assert body.offered < 4, "the download stopped at the cap instead of reading the whole body"
