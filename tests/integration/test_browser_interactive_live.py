"""browser_open on a real Chromium, over fixture pages served for a fake public host."""
from __future__ import annotations

import asyncio
from urllib.parse import urlsplit

from localharness.agent.context import ContentStore
from localharness.tools.builtin.browser_plugin import BrowserConfig
from localharness.tools.builtin.browser_session import WINDOW_CHARS, BrowserOpenTool
from localharness.tools.builtin.browser_tool import BrowserRuntime
from localharness.tools.builtin.web_tool import UNTRUSTED_BANNER, WebPageQueryTool
from tests.integration.test_browser_reach_live import SANDBOX, pytestmark, runtime  # noqa: F401
from tests.unit.test_web_fetch_guard import public_web

HOST = "fixture.localharness.test"


def serve_pages(rt, monkeypatch, pages: dict[str, str]) -> str:
    """Every https://HOST/path GET is answered from `pages`; a non-GET document request falls through to the factory's abort."""
    public_web(monkeypatch)
    make = rt._new_context

    async def new_context(browser):
        context, flags = await make(browser)

        async def fulfil(route):
            request = route.request
            if request.method != "GET" and request.resource_type == "document":
                await route.fallback()
            elif (body := pages.get(urlsplit(request.url).path)) is not None:
                await route.fulfill(status=200, content_type="text/html", body=body)
            else:
                await route.fulfill(status=404, content_type="text/plain", body="")

        await context.route(f"https://{HOST}/**", fulfil)
        return context, flags

    monkeypatch.setattr(rt, "_new_context", new_context)
    return f"https://{HOST}"


LIST = "<!doctype html><title>List</title><h1>Long list</h1><ul>" + "".join(
    f'<li><a href="/item/{i}">link number {i}</a></li>' for i in range(400)) + "</ul>"


async def test_a_long_page_opens_as_a_windowed_snapshot_with_refs_and_the_rest_is_in_the_store(runtime, monkeypatch):  # noqa: F811
    base = serve_pages(runtime, monkeypatch, {"/list": LIST})
    store = ContentStore()
    tool = BrowserOpenTool(runtime, store)
    got = await tool.run(url=f"{base}/list")
    assert got.success, got.error
    header = f"{UNTRUSTED_BANNER}URL: {base}/list\n\n"
    assert got.output.startswith(header)
    window, _, footer = got.output[len(header):].partition("\n\n[chars ")
    assert "[ref=e" in window and len(window) <= WINDOW_CHARS
    assert footer.startswith(f"0-{WINDOW_CHARS} of ") and "start_index" not in got.output
    assert 'web_page_query("pg-1", pattern)' in footer and got.output.endswith('"<button or link text>")]')
    assert "link number 399" not in got.output
    found = await WebPageQueryTool(store).run(fetch_id="pg-1", pattern="link number 399")
    assert found.success and "link number 399" in found.output
    await tool.release_store(store)
    assert not runtime._sessions and not runtime._session_slot.locked()


async def test_chromium_closes_after_idle_close_s_once_the_last_session_is_released(runtime, monkeypatch):  # noqa: F811
    own = BrowserRuntime(BrowserConfig(idle_close_s=1, sandbox=SANDBOX))
    try:
        base = serve_pages(own, monkeypatch, {"/list": LIST})
        store = ContentStore()
        tool = BrowserOpenTool(own, store)
        assert (await tool.run(url=f"{base}/list")).success
        await asyncio.sleep(1.5)
        assert own._browser is not None and own._browser.is_connected()
        await tool.release_store(store)
        await asyncio.sleep(2.5)
        assert own._browser is None and not own._session_slot.locked()
    finally:
        await own.close()
