"""The browser's reach on a real Chromium: the launch checks, a probing page, the in-page cut and a client-rendered page."""
from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import os
import socket

import pytest

from localharness.tools.builtin import browser_plugin, browser_tool, netguard
from localharness.tools.builtin.browser_plugin import BrowserConfig
from localharness.tools.builtin.browser_tool import BrowserFetchTool, BrowserRuntime
from localharness.tools.builtin.web_tool import FETCH_MAX_BODY_BYTES, WebFetchTool, get_default_store

pytestmark = [
    pytest.mark.real_sockets,
    pytest.mark.skipif(
        not os.environ.get("LOCALHARNESS_LIVE_BROWSER"),
        reason="live browser opt-in (set LOCALHARNESS_LIVE_BROWSER=1 with localharness[browser] and Playwright's headless shell installed)"),
]

CLIENT_RENDERED = os.environ.get("LOCALHARNESS_LIVE_BROWSER_URL", "https://excalidraw.com/")
# Read at import, before the suite's autouse fixtures point HOME at a temporary directory.
BROWSERS = str(browser_plugin._cache_root())
SANDBOX = os.environ.get("LOCALHARNESS_LIVE_BROWSER_SANDBOX", "on") != "off"
_PROBE = "https://reach-probe.localharness.test/"
_OPTIONS = dict(accept_downloads=False, service_workers="block", java_script_enabled=True, bypass_csp=False,
                ignore_https_errors=False)

_PROBE_PAGE = """<!doctype html><body><script>
window.__probe = (async () => {
  const wait = ms => new Promise(r => setTimeout(r, ms));
  await Promise.allSettled([
    fetch("https://127.0.0.1:__PORT__/", {mode: "no-cors"}),
    fetch("https://169.254.169.254/", {mode: "no-cors"}),
    fetch("https://__HOST__/", {mode: "no-cors"}),
  ]);
  const ws = new WebSocket("wss://localhost/");
  await Promise.race([new Promise(r => { ws.onerror = ws.onclose = r; }), wait(5000)]);
  const rtc = typeof RTCPeerConnection === "function";
  const candidates = [];
  let ended = "none";
  if (rtc) {
    const pc = new RTCPeerConnection({iceServers: []});
    pc.createDataChannel("probe");
    const gathered = new Promise(done => {
      pc.onicecandidate = e => e.candidate ? candidates.push(e.candidate.candidate) : done("complete");
      setTimeout(() => done("timeout"), 5000);
    });
    await pc.setLocalDescription(await pc.createOffer());
    ended = await gathered;
    pc.close();
  }
  const opened = window.open("file:///etc/passwd") !== null;
  return {rtc, ended, candidates, opened};
})();
</script></body>"""

_FORGER = """<body><p>the true text of this page</p><script>
String.prototype.slice = String.prototype.charCodeAt = Reflect.apply = () => "forged";
Object.defineProperty(HTMLElement.prototype, "innerText", {get() { return "forged"; }, configurable: true});
Object.defineProperty(Document.prototype, "body", {get() { return null; }, configurable: true});
try { window.__lh_text = () => [false, "forged"]; } catch (e) {}
</script></body>"""


@pytest.fixture
async def runtime(monkeypatch):
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", BROWSERS)
    if importlib.util.find_spec("playwright") is None:
        pytest.fail("LOCALHARNESS_LIVE_BROWSER is set but playwright is not installed: uv sync --inexact --extra dev --extra browser")
    config = BrowserConfig(sandbox=SANDBOX)
    if browser_plugin._binary(config) is None:
        pytest.fail(f"LOCALHARNESS_LIVE_BROWSER is set but no headless shell is in {BROWSERS}: "
                    "uv run localharness plugins enable browser downloads it")
    r = BrowserRuntime(config)
    yield r
    await r.close()


@contextlib.asynccontextmanager
async def _context(runtime, html, init=False):
    context = await (await runtime.ensure()).new_context(**_OPTIONS)
    try:
        if init:
            await context.add_init_script(browser_tool._INIT)

        async def serve(route):
            await route.fulfill(status=200, content_type="text/html", body=html)

        await context.route(_PROBE, serve)
        yield context
    finally:
        await context.close()


async def _cut(runtime, html, n):
    async with _context(runtime, html, init=True) as context:
        page = await context.new_page()
        await page.goto(_PROBE, wait_until="load")
        return await page.evaluate(browser_tool._TEXT, n)


async def test_the_launch_passes_its_canary_and_flag_assertion_through_a_real_chromium(runtime):
    browser = await runtime.ensure()
    assert browser.is_connected()
    want = f"127.0.0.1:{runtime._proxy.port}"
    argv = next((a for a in browser_tool._command_lines()
                 if not any(x.startswith("--type=") for x in a)
                 and any(x.startswith("--proxy-server=") and x.endswith(want) for x in a)), [])
    assert argv, f"no browser process carries --proxy-server ending {want}"
    assert any(x.startswith("--proxy-bypass-list=") and "<-loopback>" in x for x in argv), argv
    assert all(flag in argv for flag in browser_tool._ARGS), argv
    assert not (SANDBOX and "--no-sandbox" in argv), argv
    refused = list(runtime._proxy.refused)
    assert any(r.startswith("127.0.0.1:") and r.endswith(" not public") for r in refused), \
        f"the launch canary left no loopback refusal; refused={refused} argv={argv}"


async def test_the_doctor_launch_and_sandbox_rows_pass_on_a_real_launch(runtime):
    rows = await browser_plugin._launch_rows(BrowserConfig(sandbox=SANDBOX))
    assert [(r.name, r.status) for r in rows] == [("browser-launch", "pass"), ("browser-sandbox", "pass" if SANDBOX else "warn")], \
        [(r.name, r.status, r.detail, r.hint) for r in rows]


async def test_a_page_reaches_no_private_address_no_udp_path_and_no_local_scheme(runtime):
    host = socket.gethostname().lower()
    try:
        await netguard.check(f"https://{host}/", allow_private=False)
    except netguard.Refused:
        pass
    else:
        pytest.fail(f"this box's name {host} resolves to a public address, so the probe has no non-public name of its own")
    accepted = 0

    def accept(_reader, writer):
        nonlocal accepted
        accepted += 1
        writer.close()

    server = await asyncio.start_server(accept, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    html = _PROBE_PAGE.replace("__PORT__", str(port)).replace("__HOST__", host)
    popups = []
    try:
        async with _context(runtime, html) as context:
            runtime._proxy.refused.clear()
            page = await context.new_page()
            context.on("page", popups.append)
            await page.goto(_PROBE, wait_until="load")
            result = await page.evaluate("window.__probe")
            await asyncio.sleep(1)
            refused, urls = list(runtime._proxy.refused), [p.url for p in context.pages]
            contents = [await p.content() for p in popups if not p.is_closed()]
    finally:
        server.close()
        await server.wait_closed()
    assert not accepted, f"the canary listener accepted {accepted} connection(s): Chromium dialled loopback past the proxy"
    for authority in (f"127.0.0.1:{port}", "169.254.169.254:443", "localhost:443", f"{host}:443"):
        assert f"{authority} not public" in refused, \
            f"{authority} is not in the proxy's refused log, and the canary saw {accepted} connections: Chromium refused or bypassed it before the proxy; refused={refused}"
    assert result["rtc"], "RTCPeerConnection does not exist, so an empty candidate list would prove nothing"
    assert result["ended"] in ("complete", "timeout"), result
    assert not [c for c in result["candidates"] if " typ host" in c], f"WebRTC gathered a host candidate: {result['candidates']}"
    assert not [u for u in urls if u.startswith("file:")], f"a page is on a file: URL: {urls}"
    assert not [c for c in contents if "root:x:0:0" in c], "a popup holds the text of /etc/passwd"


async def test_a_page_over_the_budget_is_cut_inside_the_page(runtime):
    html = f'<body><script>document.body.textContent = "x".repeat({FETCH_MAX_BODY_BYTES + 1000})</script></body>'
    got = await _cut(runtime, html, FETCH_MAX_BODY_BYTES)
    assert got[0] is True and len(got[1]) == FETCH_MAX_BODY_BYTES, (got[0], len(got[1]))


async def test_a_page_that_replaces_the_natives_cannot_change_the_cut(runtime):
    assert await _cut(runtime, _FORGER, 1000) == [False, "the true text of this page"]


async def test_a_cut_inside_a_surrogate_pair_leaves_no_lone_surrogate(runtime):
    got = await _cut(runtime, "<body>aaaaaaaaa&#x1F600;bbbbbbbbbb</body>", 10)
    assert got == [True, "a" * 9]
    assert "�" not in got[1]


async def test_a_client_rendered_page_reads_through_browser_fetch_and_not_web_fetch(runtime):
    web = await WebFetchTool().run(url=CLIENT_RENDERED)
    assert web.success and "low extractable text" in web.output, \
        f"web_fetch read enough text without a browser; set LOCALHARNESS_LIVE_BROWSER_URL to a client-rendered https page: {web.output[:200]!r}"
    res = await BrowserFetchTool(runtime).run(url=CLIENT_RENDERED)
    assert res.success and res.metadata.get("fetch_id"), res.output or res.error
    assert len(get_default_store().get(res.metadata["fetch_id"]).strip()) > 80
