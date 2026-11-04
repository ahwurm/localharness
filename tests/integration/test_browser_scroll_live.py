"""Live: browser_fetch's scroll in real Chromium against offline pages that each break a naive scroll loop."""
from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import pytest

from localharness.tools.builtin import browser_plugin, browser_tool
from localharness.tools.builtin.browser_tool import SCROLL_S, BrowserRuntime

pytestmark = [
    pytest.mark.skipif(not os.environ.get("LOCALHARNESS_LIVE_BROWSER"),
                       reason="live browser opt-in (LOCALHARNESS_LIVE_BROWSER=1)"),
]
# Read at import, before the suite's autouse fixtures point HOME at a temporary directory.
BROWSERS = str(browser_plugin._cache_root())
SANDBOX = os.environ.get("LOCALHARNESS_LIVE_BROWSER_SANDBOX", "on") != "off"
H = "https://fixture.test"

FEED = """<!doctype html><body><h1>Quotes</h1><div id="q"></div><script>
let page = 1, more = true;
const load = p => fetch('/api?page=' + p).then(r => r.json()).then(d => {
  for (const t of d.items) { const e = document.createElement('div'); e.style.height = '110px'; e.textContent = t;
    document.getElementById('q').appendChild(e); }
  more = d.more; });
load(page);
addEventListener('scroll', () => {
  if (more && Math.abs(scrollY - (document.documentElement.scrollHeight - innerHeight)) <= 1) load(++page); });
</script></body>"""

SHELL = """<!doctype html><body style="margin:0;overflow:hidden;height:100vh">
<header style="position:fixed;top:0;left:0;right:0;height:60px;z-index:2">Site header</header>
<main id="m" style="position:absolute;top:60px;bottom:0;left:0;right:0;overflow:auto">
<div id="list"></div><div id="s">loading more</div></main><script>
let n = 0; const list = document.getElementById('list');
const more = () => { for (let i = 0; i < 20; i++) { const p = document.createElement('p'); p.style.height = '40px';
  p.textContent = 'Item ' + (++n) + ' of the inner scroller'; list.appendChild(p); } };
more();
new IntersectionObserver(es => { if (es[0].isIntersecting && n < 100) setTimeout(more, 200); },
  {root: document.getElementById('m')}).observe(document.getElementById('s'));
</script></body>"""

VIRTUAL = """<!doctype html><body style="margin:0"><div id="v" style="height:100vh;overflow:auto">
<div id="sp" style="position:relative"></div></div><script>
const N = 500, R = 30, v = document.getElementById('v'), sp = document.getElementById('sp'); sp.style.height = N * R + 'px';
const draw = () => { const a = Math.max(0, Math.floor(v.scrollTop / R) - 5), b = Math.min(N, a + Math.ceil(v.clientHeight / R) + 10);
  sp.innerHTML = ''; for (let i = a; i < b; i++) { const d = document.createElement('div');
    d.style.cssText = 'position:absolute;top:' + i * R + 'px;height:' + R + 'px'; d.textContent = 'Row ' + i + ' lorem ipsum dolor';
    sp.appendChild(d); } };
v.addEventListener('scroll', draw); draw();
</script></body>"""

FOREVER = """<!doctype html><body><div id="f"></div><script>
let n = 0; const add = () => { for (let i = 0; i < 20; i++) { const p = document.createElement('p');
  p.style.height = '60px'; p.textContent = 'Post ' + (++n) + ' in a feed that never ends'; document.getElementById('f').appendChild(p); } };
add(); addEventListener('scroll', () => { if (scrollY + innerHeight > document.documentElement.scrollHeight - 300) add(); });
</script></body>"""

FORGED = """<!doctype html><body><p>static text on a page that lies about its height</p><script>
const big = 'x'.repeat(30000000);
Object.defineProperty(Element.prototype, 'scrollHeight', {get() { return big; }, configurable: true});
</script></body>"""

HANG = """<!doctype html><body style="margin:0"><div id="h" style="height:100vh;overflow:auto">
<div style="height:5000px">inner content of a page whose wheel handler blocks</div></div><script>
document.getElementById('h').addEventListener('wheel', () => { const t = Date.now(); while (Date.now() - t < 15000) {} },
  {passive: false});
</script></body>"""

NEXT = ("""<!doctype html><body><article><h1>Article one</h1>"""
        + "<p style='height:200px'>one paragraph of the first article</p>" * 10 + """</article><script>
let done = false; addEventListener('scroll', () => {
  if (!done && scrollY + innerHeight >= document.documentElement.scrollHeight - 2) { done = true;
    const a = document.createElement('article');
    a.innerHTML = '<h1>ARTICLE TWO</h1>' + '<p style=height:200px>two paragraph of another article</p>'.repeat(10);
    document.body.appendChild(a); history.replaceState(null, '', '/article-2'); } });
</script></body>""")

PAGED = ("""<!doctype html><body><article><h1>Chapter</h1><div id="c">"""
         + "<p style='height:200px'>first part of the chapter text</p>" * 8 + """</div></article><script>
let part = 1; addEventListener('scroll', () => {
  if (part < 4 && scrollY + innerHeight >= document.documentElement.scrollHeight - 2) { part++;
    for (let i = 0; i < 8; i++) { const p = document.createElement('p'); p.style.height = '200px';
      p.textContent = 'part ' + part + ' of the same chapter'; document.getElementById('c').appendChild(p); }
    history.replaceState(null, '', '/chapter/page/' + part); } });
</script></body>""")

STATIC = "<!doctype html><body>" + "".join(f"<p style='height:100px'>paragraph {i}</p>" for i in range(200)) + "</body>"

PAGES = {"/feed": FEED, "/shell": SHELL, "/virtual": VIRTUAL, "/forever": FOREVER, "/forged": FORGED,
         "/hang": HANG, "/article": NEXT, "/chapter": PAGED, "/static": STATIC}


async def _fixtures(route) -> None:
    url = route.request.url
    if not url.startswith(H):
        return await route.abort()
    path = url[len(H):].split("?")[0].split("#")[0]
    if path == "/api":
        import asyncio
        page = int(url.split("page=")[1])
        await asyncio.sleep(0.3)
        items = [f"Quote {page}.{i} " + "lorem ipsum " * 8 for i in range(10)]
        return await route.fulfill(status=200, content_type="application/json",
                                   body=json.dumps({"items": items, "more": page < 10}))
    await route.fulfill(status=200, content_type="text/html", body=PAGES[path])


@pytest.fixture
async def runtime(monkeypatch):
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", BROWSERS)
    monkeypatch.setattr(browser_tool, "_cost_route", _fixtures)
    r = BrowserRuntime(SimpleNamespace(idle_close_s=120, executable_path="", sandbox=SANDBOX))
    yield r
    await r.close()


async def timed(runtime, path):
    t = time.monotonic()
    page = await runtime.render(H + path)
    return page, time.monotonic() - t


async def test_a_feed_that_loads_only_at_the_very_bottom_arrives_whole(runtime):
    page, _ = await timed(runtime, "/feed")
    assert "Quote 10.9" in page.text and not page.growing


async def test_an_inner_scroller_under_a_fixed_header_arrives_whole(runtime):
    page, _ = await timed(runtime, "/shell")
    assert "Item 100 of the inner scroller" in page.text


async def test_a_virtual_list_keeps_its_first_screen(runtime):
    page, _ = await timed(runtime, "/virtual")
    assert "Row 0 lorem ipsum" in page.text


async def test_a_feed_that_never_ends_returns_in_budget_and_says_so(runtime):
    page, took = await timed(runtime, "/forever")
    assert "Post 1 in a feed" in page.text and page.growing and took <= SCROLL_S + 5


async def test_a_page_that_forges_its_height_stops_quickly_without_a_large_read(runtime):
    page, took = await timed(runtime, "/forged")
    assert "static text" in page.text and took < 6


async def test_a_page_whose_wheel_handler_blocks_returns_its_first_screen(runtime):
    page, took = await timed(runtime, "/hang")
    assert "inner content" in page.text and took <= SCROLL_S + 8


async def test_a_page_that_swaps_in_the_next_article_is_cited_and_read_as_the_first(runtime):
    page, _ = await timed(runtime, "/article")
    assert page.url == H + "/article"
    assert "ARTICLE TWO" not in page.text and "Article one" in page.text


async def test_a_reader_whose_address_follows_the_reading_position_arrives_whole(runtime):
    page, _ = await timed(runtime, "/chapter")
    assert "part 4 of the same chapter" in page.text and page.url == H + "/chapter"


async def test_a_static_page_reads_the_same_with_little_overhead(runtime):
    page, took = await timed(runtime, "/static")
    assert "paragraph 199" in page.text and not page.growing and took < 5
