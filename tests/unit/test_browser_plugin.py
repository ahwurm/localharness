"""The browser plugin's runtime on a fake Chromium: the real proxy and its checks, no browser, no playwright."""
from __future__ import annotations

import asyncio
import base64
import re
from types import SimpleNamespace

import pytest

from localharness import __version__
from localharness.agent.context import ContentStore
from localharness.tools.base import StoreBound
from localharness.tools.builtin import browser_tool
from localharness.tools.builtin.browser_tool import BrowserFetchTool, BrowserRuntime, Rendered
from localharness.tools.builtin.web_tool import UNTRUSTED_BANNER, WebFetchTool, WebPageQueryTool
from tests.unit.test_web_fetch_guard import public_web

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/156.0.0.0 Safari/537.36"
CANARY = re.compile(r"https://127\.0\.0\.1:(\d+)/")


class FakeError(Exception):
    pass


class FakeTimeout(FakeError):
    pass


class FakeResponse:
    def __init__(self, status, content_type):
        self.status, self.headers = status, {"content-type": content_type}


class FakePage:
    def __init__(self, context):
        self.context, self.url = context, "about:blank"

    async def goto(self, url, wait_until=None, timeout=None):
        world = self.context.browser.world
        world.events.append(("goto", url))
        world.gotos.append({"url": url, "wait_until": wait_until, "timeout": timeout})
        if match := CANARY.fullmatch(url):
            await world.dial_canary(self.context.browser.kwargs["proxy"], int(match[1]))
        if world.on_goto:
            await world.on_goto(url)
        self.url = url
        status, _, content_type = world.pages[url]
        return FakeResponse(status, content_type)

    async def wait_for_timeout(self, ms):
        self.context.browser.world.waits.append(ms)

    async def evaluate(self, expression, arg=None):
        world = self.context.browser.world
        if expression == "navigator.userAgent":
            return UA
        if world.eval_error:
            raise world.eval_error
        world.slices.append((expression, arg))
        return [world.capped, world.pages[self.url][1]]


class FakeContext:
    def __init__(self, browser, options):
        self.browser, self.options, self.routes, self.open = browser, options, [], True

    async def route(self, pattern, handler):
        self.browser.world.events.append(("route", pattern))
        self.routes.append((pattern, handler))

    async def new_page(self):
        return FakePage(self)

    async def close(self):
        self.open = False


class FakeBrowser:
    def __init__(self, world, kwargs):
        self.world, self.kwargs, self.connected, self.closed = world, kwargs, True, False

    def is_connected(self):
        return self.connected

    async def new_context(self, **options):
        context = FakeContext(self, options)
        self.world.contexts.append(context)
        return context

    async def close(self):
        self.closed, self.connected = True, False


class FakePlaywright:
    def __init__(self, world):
        self.world, self.chromium = world, self

    async def launch(self, **kwargs):
        if self.world.launch_error:
            raise self.world.launch_error
        browser = FakeBrowser(self.world, kwargs)
        self.world.launches.append(kwargs)
        self.world.browsers.append(browser)
        return browser

    async def stop(self):
        self.world.stopped += 1


class World:
    """Every launch, context and navigation of one test, and what the fake browser does on a canary."""

    def __init__(self):
        self.launches, self.browsers, self.contexts = [], [], []
        self.events, self.gotos, self.waits, self.slices = [], [], [], []
        self.stopped = 0
        self.pages = {"https://example.com/": (200, "hello world", "text/html")}
        self.capped = False
        self.canary = "proxy"
        self.launch_error = None
        self.on_goto = None
        self.eval_error = None
        self.argv_edit = lambda argv: argv
        self.api = SimpleNamespace(
            Error=FakeError, TimeoutError=FakeTimeout,
            async_playwright=lambda: SimpleNamespace(start=self._start))

    async def _start(self):
        return FakePlaywright(self)

    def command_lines(self):
        if not self.launches:
            return [["/usr/bin/other"]]
        kwargs = self.launches[-1]
        proxy = kwargs["proxy"]
        argv = ["chrome", "--remote-debugging-pipe", f"--proxy-server={proxy['server']}",
                f"--proxy-bypass-list={proxy['bypass']}", *kwargs["args"]]
        if not kwargs["chromium_sandbox"]:
            argv.append("--no-sandbox")
        return [["chrome", "--type=renderer", f"--proxy-server={proxy['server']}"], self.argv_edit(argv)]

    async def dial_canary(self, proxy, port):
        if self.canary in ("proxy", "both"):
            reader, writer = await asyncio.open_connection("127.0.0.1", int(proxy["server"].rsplit(":", 1)[1]))
            token = base64.b64encode(f"{proxy['username']}:{proxy['password']}".encode()).decode()
            writer.write(f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\nProxy-Authorization: Basic {token}\r\n\r\n".encode())
            assert b" 403 " in await reader.readuntil(b"\r\n\r\n")
            writer.close()
        if self.canary in ("direct", "both"):
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            await reader.read()
            writer.close()
        raise FakeError("net::ERR_TUNNEL_CONNECTION_FAILED")

    def open_contexts(self):
        return sum(c.open for c in self.contexts)


def config(**kw):
    return SimpleNamespace(**{"idle_close_s": 120, "executable_path": "", "sandbox": True, **kw})


@pytest.fixture
def world(monkeypatch, tmp_path):
    w = World()
    monkeypatch.setattr(browser_tool, "_API", w.api)
    monkeypatch.setattr(browser_tool, "_command_lines", w.command_lines)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    return w


@pytest.fixture
async def runtime(world):
    r = BrowserRuntime(config())
    yield r
    await r.close()


async def test_the_first_render_launches_through_the_proxy_passes_its_checks_and_returns_the_text(world, runtime):
    got = await runtime.render("https://example.com/")
    assert got == Rendered("https://example.com/", 200, "hello world", False)
    (kwargs,) = world.launches
    proxy = kwargs.pop("proxy")
    assert proxy["server"] == f"http://127.0.0.1:{runtime._proxy.port}" and proxy["bypass"] == "<-loopback>"
    assert (proxy["username"], proxy["password"]) == (runtime._proxy.username, runtime._proxy.password)
    assert kwargs == {"headless": True, "chromium_sandbox": True, "executable_path": None, "handle_sigint": False,
                      "args": ["--disable-quic", "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
                               "--disable-breakpad"]}
    assert any(e.endswith(" not public") and e.startswith("127.0.0.1:") for e in runtime._proxy.refused)
    await runtime.close()
    assert world.open_contexts() == 0 and world.browsers[0].closed and world.stopped == 1


async def unavailable(runtime, url="https://example.com/"):
    with pytest.raises(browser_tool.RenderError) as e:
        await runtime.render(url)
    assert e.value.kind == "unavailable"
    return e.value.detail


async def refuses_connections(port):
    with pytest.raises(OSError):
        await asyncio.open_connection("127.0.0.1", port)


@pytest.mark.parametrize("canary", ["direct", "none", "both"])
async def test_a_canary_that_is_not_refused_by_the_proxy_alone_fails_the_launch_closed(world, runtime, canary):
    world.canary = canary
    assert await unavailable(runtime) == "reach check failed"
    port = int(world.launches[0]["proxy"]["server"].rsplit(":", 1)[1])
    assert world.browsers[0].closed and world.stopped == 1
    await refuses_connections(port)
    assert await unavailable(runtime) == "reach check failed"
    assert len(world.launches) == 1


async def test_with_the_canary_failing_no_tool_context_is_ever_created(world, runtime):
    world.canary = "direct"
    await unavailable(runtime)
    assert [c.options for c in world.contexts] == [{}]
    assert all(url.startswith("https://127.0.0.1:") for kind, url in world.events if kind == "goto")


FLAGS = ["--proxy-server=", "--proxy-bypass-list=", "--disable-quic", "--force-webrtc-ip-handling-policy=",
         "--disable-breakpad"]


@pytest.mark.parametrize("flag", FLAGS)
async def test_a_command_line_missing_any_flag_fails_the_launch(world, runtime, flag):
    world.argv_edit = lambda argv: [a for a in argv if not a.startswith(flag)]
    assert await unavailable(runtime) == "reach check failed"
    assert world.browsers[0].closed and world.open_contexts() == 0


async def test_a_bypass_list_without_the_loopback_rule_fails_the_launch(world, runtime):
    world.argv_edit = lambda argv: [("--proxy-bypass-list=localhost" if a.startswith("--proxy-bypass") else a) for a in argv]
    assert await unavailable(runtime) == "reach check failed"


async def test_an_unreadable_command_line_fails_the_launch_closed(world, monkeypatch, runtime):
    monkeypatch.setattr(browser_tool, "_command_lines", lambda: [])
    assert await unavailable(runtime) == "reach check failed"


async def test_a_sandboxed_launch_whose_command_line_carries_no_sandbox_is_refused(world, runtime):
    world.argv_edit = lambda argv: [*argv, "--no-sandbox"]
    assert await unavailable(runtime) == "Chromium started without its sandbox"
    assert world.browsers[0].closed


async def test_the_machine_level_sandbox_off_launches_without_it_and_passes(world):
    runtime = BrowserRuntime(config(sandbox=False))
    assert (await runtime.render("https://example.com/")).text == "hello world"
    assert world.launches[0]["chromium_sandbox"] is False
    await runtime.close()


@pytest.mark.parametrize("text,reason", [
    ("BrowserType.launch: Host system is missing dependencies to run browsers.\nplease install", "system libraries are missing"),
    ("Failed to launch: No usable sandbox! If you are running on Ubuntu 23.10+", "the Chromium sandbox could not start"),
    ("Executable doesn't exist at /x/chrome\n╔═══╗", "Chromium is not installed"),
    ("boom\nmore", "launch failed: boom"),
    ("BrowserType.launch: boom\nmore", "launch failed: boom"),
])
async def test_a_launch_error_becomes_the_reason_a_person_can_act_on(world, runtime, text, reason):
    world.launch_error = FakeError(text)
    assert await unavailable(runtime) == reason
    assert world.stopped == 1
    world.launch_error = None
    assert await unavailable(runtime) == reason
    assert world.launches == []


async def test_an_unknown_launch_error_is_cut_at_200_characters(world, runtime):
    world.launch_error = FakeError("x" * 500)
    assert await unavailable(runtime) == "launch failed: " + "x" * 200


async def test_the_proxy_is_closed_when_the_launch_fails(world, runtime, monkeypatch):
    seen = []
    real = browser_tool.serve

    async def spy(upstream):
        seen.append(await real(upstream))
        return seen[-1]

    monkeypatch.setattr(browser_tool, "serve", spy)
    world.launch_error = FakeError("boom")
    await unavailable(runtime)
    await refuses_connections(seen[0].port)


@pytest.mark.parametrize("path,passed", [("/opt/chromium/chrome", "/opt/chromium/chrome"), ("", None)])
async def test_executable_path_is_passed_through_only_when_set(world, path, passed):
    runtime = BrowserRuntime(config(executable_path=path))
    await runtime.render("https://example.com/")
    assert world.launches[0]["executable_path"] == passed
    await runtime.close()


async def test_the_user_agent_keeps_chromiums_own_string_and_appends_the_token(world, runtime):
    await runtime.render("https://example.com/")
    ua = world.contexts[-1].options["user_agent"]
    assert ua == f"{UA} LocalHarness/{__version__} (+https://localharness.dev)" and "HeadlessChrome" in ua


async def test_only_unowned_real_stale_profile_directories_are_removed_before_the_launch(
        world, tmp_path, runtime, monkeypatch):
    live, stale = (tmp_path / f"playwright_chromiumdev_profile-{n}" for n in ("live", "stale"))
    other, target = tmp_path / "unrelated-dir", tmp_path / "link-target"
    for d in (live, stale, target, other):
        d.mkdir()
    link = tmp_path / "playwright_chromiumdev_profile-link"
    link.symlink_to(target, target_is_directory=True)
    plain_file = tmp_path / "playwright_chromiumdev_profile-file"
    plain_file.write_text("x")
    first = world.command_lines
    seen = []

    def lines():
        seen.append(stale.exists())
        return [*first(), ["chrome", f"--user-data-dir={live}"]]

    monkeypatch.setattr(browser_tool, "_command_lines", lines)
    await runtime.render("https://example.com/")
    assert not stale.exists() and seen[0] is True
    assert live.exists() and other.exists() and plain_file.exists() and link.is_symlink() and target.exists()


async def test_nothing_is_swept_when_the_running_processes_cannot_be_read(world, tmp_path, monkeypatch):
    stale = tmp_path / "playwright_chromiumdev_profile-stale"
    stale.mkdir()
    monkeypatch.setattr(browser_tool, "_command_lines", lambda: [])
    await unavailable(BrowserRuntime(config()))
    assert stale.exists()


@pytest.mark.skipif(not browser_tool.Path("/proc/self/cmdline").exists(), reason="needs /proc")
def test_the_real_command_lines_include_this_process():
    mine = browser_tool.Path("/proc/self/cmdline").read_bytes().decode().split("\0")[0]
    assert any(argv[:1] == [mine] for argv in browser_tool._command_lines())


async def until(predicate, seconds=2.0):
    for _ in range(int(seconds / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


def tool_contexts(world):
    return [c for c in world.contexts if c.options]


async def test_a_render_opens_one_context_with_the_exact_options_and_routes_before_it_navigates(world, runtime):
    await runtime.render("https://example.com/")
    (context,) = tool_contexts(world)
    assert context.options == {
        "accept_downloads": False, "service_workers": "block", "java_script_enabled": True, "bypass_csp": False,
        "ignore_https_errors": False, "user_agent": f"{UA} LocalHarness/{__version__} (+https://localharness.dev)"}
    assert [e for e in world.events if e[1] == "**/*" or e[1] == "https://example.com/"] == [
        ("route", "**/*"), ("goto", "https://example.com/")]
    assert world.gotos[-1] == {"url": "https://example.com/", "wait_until": "load", "timeout": 30000}
    assert world.waits == [500]
    assert world.slices == [(browser_tool._SLICE, browser_tool.FETCH_MAX_BODY_BYTES)]
    assert "innerText" in browser_tool._SLICE and "0xD800" in browser_tool._SLICE


class FakeRoute:
    def __init__(self, resource_type):
        self.request, self.did = SimpleNamespace(resource_type=resource_type), None

    async def abort(self):
        self.did = "abort"

    async def continue_(self):
        self.did = "continue"


@pytest.mark.parametrize("kind,did", [
    ("image", "abort"), ("media", "abort"), ("font", "abort"), ("document", "continue"), ("script", "continue"),
    ("stylesheet", "continue"), ("xhr", "continue"), ("fetch", "continue"), ("websocket", "continue")])
async def test_only_image_media_and_font_requests_are_aborted(world, runtime, kind, did):
    await runtime.render("https://example.com/")
    ((pattern, handler),) = tool_contexts(world)[0].routes
    route = FakeRoute(kind)
    await handler(route)
    assert pattern == "**/*" and route.did == did


@pytest.mark.parametrize("text,expected", [("a  b\t\tc\n\n\n\nd", "a b c\n\nd"), ("  \n padded \n", "padded"), ("", "")])
async def test_whitespace_is_collapsed_the_way_web_fetch_does_and_an_empty_body_is_empty(world, runtime, text, expected):
    world.pages["https://example.com/"] = (200, text, "text/html")
    assert (await runtime.render("https://example.com/")).text == expected


async def test_the_pages_capped_flag_comes_back(world, runtime):
    world.capped = True
    assert (await runtime.render("https://example.com/")).capped is True


async def test_an_http_error_status_is_returned_for_the_tool_to_judge(world, runtime):
    world.pages["https://example.com/"] = (404, "gone", "text/html")
    assert (await runtime.render("https://example.com/")).status == 404


@pytest.mark.parametrize("make,kind,detail", [
    (lambda: FakeTimeout("Page.goto: Timeout 30000ms exceeded."), "timeout", ""),
    (lambda: FakeError("Page.goto: Download is starting"), "file", ""),
    (lambda: FakeError("net::ERR_TUNNEL_CONNECTION_FAILED at https://x/\ncall log"), "failed",
     "net::ERR_TUNNEL_CONNECTION_FAILED at https://x/"),
    (lambda: FakeError("Page.goto: net::ERR_NAME_NOT_RESOLVED at https://x/\ncall log"), "failed",
     "net::ERR_NAME_NOT_RESOLVED at https://x/")])
async def test_a_navigation_error_becomes_one_kind_and_the_context_is_closed(world, runtime, make, kind, detail):
    async def fail(url):
        raise make()

    world.on_goto = fail
    with pytest.raises(browser_tool.RenderError) as e:
        await runtime.render("https://example.com/")
    assert (e.value.kind, e.value.detail) == (kind, detail)
    assert len(tool_contexts(world)) == 1 and world.open_contexts() == 0


async def test_a_pdf_response_is_a_file_and_every_outcome_closes_its_context(world, runtime):
    world.pages["https://example.com/doc.pdf"] = (200, "", "application/pdf; charset=binary")
    with pytest.raises(browser_tool.RenderError) as e:
        await runtime.render("https://example.com/doc.pdf")
    assert e.value.kind == "file" and world.open_contexts() == 0
    await runtime.render("https://example.com/")
    assert len(tool_contexts(world)) == 2 and world.open_contexts() == 0


async def test_an_error_while_reading_the_text_is_a_failed_render_with_the_context_closed(world, runtime):
    world.eval_error = FakeError("Page.evaluate: Execution context was destroyed")
    with pytest.raises(browser_tool.RenderError) as e:
        await runtime.render("https://example.com/")
    assert (e.value.kind, e.value.detail) == ("failed", "Execution context was destroyed")
    assert world.open_contexts() == 0


def gate_on(world, *urls):
    gates = {u: asyncio.Event() for u in urls}

    async def wait(url):
        if url in gates:
            await gates[url].wait()

    world.on_goto = wait
    for u in urls:
        world.pages[u] = (200, u, "text/html")
    return gates


async def test_two_renders_at_once_use_two_contexts_and_a_third_waits_for_one_to_close(world, runtime):
    await runtime.ensure()
    gates = gate_on(world, *(f"https://example.com/{i}" for i in range(3)))
    tasks = [asyncio.create_task(runtime.render(u)) for u in gates]
    await until(lambda: len(tool_contexts(world)) == 2)
    await asyncio.sleep(0.05)
    assert len(tool_contexts(world)) == 2 and world.open_contexts() == 2
    gates["https://example.com/0"].set()
    await until(lambda: len(tool_contexts(world)) == 3)
    for g in gates.values():
        g.set()
    results = await asyncio.gather(*tasks)
    assert [r.text for r in results] == list(gates)
    assert len({id(c) for c in tool_contexts(world)}) == 3 and world.open_contexts() == 0


async def test_two_first_renders_started_together_launch_once(world, runtime):
    a, b = await asyncio.gather(runtime.render("https://example.com/"), runtime.render("https://example.com/"))
    assert a == b and len(world.launches) == 1


async def test_ensure_twice_launches_once_and_close_twice_is_a_no_op(world, runtime):
    first = await runtime.ensure()
    assert await runtime.ensure() is first and len(world.launches) == 1
    await runtime.close()
    await runtime.close()
    assert world.stopped == 1 and first.closed


async def test_a_disconnected_browser_is_relaunched_by_the_next_render(world, runtime):
    await runtime.render("https://example.com/")
    world.browsers[0].connected = False
    await runtime.render("https://example.com/")
    assert len(world.launches) == 2 and world.browsers[0].closed


async def test_the_browser_closes_after_the_idle_time_and_the_next_render_launches_again(world, runtime):
    runtime._idle_s = 0.05
    await runtime.render("https://example.com/")
    await until(lambda: world.browsers[0].closed, 1)
    assert world.stopped == 1
    await runtime.render("https://example.com/")
    assert len(world.launches) == 2


async def test_a_render_in_flight_holds_off_the_idle_close(world, runtime):
    runtime._idle_s = 0.05
    await runtime.render("https://example.com/")
    gates = gate_on(world, "https://example.com/slow")
    task = asyncio.create_task(runtime.render("https://example.com/slow"))
    await until(lambda: len(tool_contexts(world)) == 2)
    await asyncio.sleep(0.2)
    assert not world.browsers[0].closed
    gates["https://example.com/slow"].set()
    assert (await task).text == "https://example.com/slow"
    await until(lambda: world.browsers[0].closed, 1)
    assert len(world.launches) == 1


async def test_close_cancels_a_pending_idle_close(world, runtime):
    runtime._idle_s = 0.05
    await runtime.render("https://example.com/")
    await runtime.close()
    await asyncio.sleep(0.15)
    assert world.stopped == 1


DESCRIPTION = (
    "Like web_fetch, but runs the page's JavaScript first (slow, up to 30 s). If web_fetch returned "
    "\"low extractable text\" for an https URL, call this once with the same URL — web_fetch's "
    "\"do NOT re-fetch\" means do not call web_fetch again, not this tool. Otherwise use web_fetch. "
    "Same start_index and fetch_id; search the full page with web_page_query.")
PAGE = "x" * 3000 + " needle " + "y" * 2992


@pytest.fixture
def fetch(world, runtime, monkeypatch):
    public_web(monkeypatch)
    world.pages["https://example.com/a"] = (200, PAGE, "text/html")
    store = ContentStore()
    return BrowserFetchTool(runtime, store), store


async def test_a_rendered_page_is_retained_in_the_tools_store_and_found_by_web_page_query(fetch):
    tool, store = fetch
    got = await tool.run(url="https://example.com/a", max_chars=500)
    assert got.success
    assert got.output == (UNTRUSTED_BANNER + PAGE[:500] + "\n\n[chars 0-500 of 6000 | fetch_id=pg-1 | "
                          'web_page_query("pg-1", pattern) to search the full retained page; '
                          "or browser_fetch start_index=500 for the next window]")
    assert got.metadata == {"url": "https://example.com/a", "fetch_id": "pg-1", "start_index": 0, "next_start_index": 500}
    found = await WebPageQueryTool(store).run(fetch_id="pg-1", pattern="needle")
    assert found.success and "needle" in found.output


def test_the_schema_is_web_fetchs_with_the_one_description_and_its_name():
    tool = BrowserFetchTool(object(), ContentStore())
    schema, web = tool.info(), WebFetchTool(ContentStore()).info()
    assert schema.description == DESCRIPTION
    assert schema.parameters == web.parameters
    assert list(schema.parameters["properties"]) == list(web.parameters["properties"])
    assert schema.name == "browser_fetch" and isinstance(tool, StoreBound)
    assert (schema.group, schema.estimated_tokens, schema.destructive) == (web.group, web.estimated_tokens, False)
    assert tool.timeout_s == browser_tool.NAV_TIMEOUT_S + 20


NOT_HTTPS = "browser_fetch renders https pages only. For http://… use web_fetch."
EMPTY = "[rendered text is under 80 characters — the page is empty or blocking the browser; do NOT retry. Use the search snippet.]"
RETRY = "Do not retry this URL; use the search snippet or another source."


def not_public(host):
    return (f"browser_fetch: {host} is not a public address; browser_fetch reaches public sites only. "
            "Do not retry this host; use the search snippet or another source.")


async def refused(tool, world, url, line, error_type="validation_error"):
    got = await tool.run(url=url)
    assert (got.success, got.error, got.error_type) == (False, line, error_type)
    assert world.launches == []


@pytest.mark.parametrize("url", ["http://example.com/", "ftp://example.com/", "", "example.com/a"])
async def test_a_url_that_is_not_https_is_refused_before_any_browser_work(fetch, world, url):
    await refused(fetch[0], world, url, NOT_HTTPS)


@pytest.mark.parametrize("url, host", [
    ("https://127.0.0.1/", "127.0.0.1"), ("https://10.0.0.1/", "10.0.0.1"), ("https://169.254.169.254/", "169.254.169.254"),
    ("https://100.64.0.1/", "100.64.0.1"), ("https://[::1]/", "::1"), ("https://224.0.0.1/", "224.0.0.1"),
])
async def test_a_non_public_address_is_refused_in_one_line_naming_the_host(fetch, world, url, host):
    await refused(fetch[0], world, url, not_public(host))


async def test_a_name_resolving_to_a_private_address_is_refused_with_the_name(fetch, world, monkeypatch):
    public_web(monkeypatch, "10.0.0.5")
    await refused(fetch[0], world, "https://intranet.example/", not_public("intranet.example"))


async def test_the_private_allowlist_is_not_honoured_by_the_browser(fetch, world, monkeypatch):
    monkeypatch.setattr("localharness.tools.builtin.netguard._ALLOW_PRIVATE", ("10.0.0.0/8",))
    await refused(fetch[0], world, "https://10.0.0.5/", not_public("10.0.0.5"))


@pytest.mark.parametrize("url, line", [
    ("https://user:pw@example.com/", "refused: a URL with a user name or password in it is not fetched"),
    ("https://", "refused: the URL names no host"),
    ("https://address.example/", "refused: could not resolve address.example"),
    ("https://[address]/", "refused: 'https://[address]/' is not a valid URL"),
])
async def test_any_other_refusal_answers_netguards_own_line_whatever_words_the_url_holds(fetch, world, monkeypatch, url, line):
    monkeypatch.setattr("localharness.tools.builtin.netguard._resolve", lambda h, p: _fail())
    await refused(fetch[0], world, url, line)


async def _fail():
    raise OSError("no such host")


@pytest.mark.parametrize("status", [404, 503])
async def test_an_http_error_status_answers_the_status_line(fetch, world, status):
    world.pages["https://example.com/a"] = (status, PAGE, "text/html")
    got = await fetch[0].run(url="https://example.com/a")
    assert (got.success, got.error) == (False, f"browser_fetch failed: HTTP {status}. {RETRY}")


@pytest.mark.parametrize("error, line", [
    (FakeTimeout("Timeout 30000ms exceeded"), "browser_fetch: the page did not finish loading in 30 s. Do not retry; use the search snippet."),
    (FakeError("Download is starting"), "browser_fetch: that URL is a file, not a page. Use the search snippet, or find an HTML version."),
    (FakeError("net::ERR_TUNNEL_CONNECTION_FAILED at https://example.com/a"),
     f"browser_fetch failed: net::ERR_TUNNEL_CONNECTION_FAILED at https://example.com/a. {RETRY}"),
])
async def test_a_navigation_failure_answers_its_one_line(fetch, world, error, line):
    async def boom(url):
        raise error

    world.on_goto = boom
    got = await fetch[0].run(url="https://example.com/a")
    assert (got.success, got.error) == (False, line)


async def test_a_browser_that_cannot_launch_answers_the_not_available_line(world, monkeypatch):
    public_web(monkeypatch)
    world.canary = "direct"
    runtime = BrowserRuntime(config())
    got = await BrowserFetchTool(runtime, ContentStore()).run(url="https://example.com/")
    assert (got.success, got.error) == (False, "browser_fetch is not available on this machine (reach check failed); "
                                              "use the search snippet or another source. The person can run `localharness doctor`.")
    await runtime.close()


async def test_a_page_under_80_characters_answers_the_empty_line_and_retains_nothing(fetch, world):
    tool, store = fetch
    world.pages["https://example.com/a"] = (200, "x" * 60, "text/html")
    got = await tool.run(url="https://example.com/a")
    assert got.success and got.output == UNTRUSTED_BANNER + "URL: https://example.com/a\n\n" + EMPTY
    assert got.metadata == {"url": "https://example.com/a"}
    world.pages["https://example.com/a"] = (200, PAGE, "text/html")
    assert (await tool.run(url="https://example.com/a")).metadata["fetch_id"] == "pg-1"


async def test_a_text_cut_in_the_page_ends_with_the_capped_sentence_before_it_is_retained(fetch, world):
    world.capped = True
    got = await fetch[0].run(url="https://example.com/a", max_chars=20000)
    end = (f"[rendered text capped at {browser_tool.FETCH_MAX_BODY_BYTES} characters — the rest of this page was not read; "
           "the retained text is a PREFIX, not the full page]")
    assert got.output.endswith(PAGE + "\n" + end + f"\n\n[chars 0-{len(PAGE) + 1 + len(end)} of {len(PAGE) + 1 + len(end)} | "
                               'fetch_id=pg-1 | web_page_query("pg-1", pattern) to search the full retained page]')


async def test_the_https_test_ignores_case(fetch, world):
    world.pages["HTTPS://example.com/a"] = world.pages["https://example.com/a"]
    got = await fetch[0].run(url="HTTPS://example.com/a")
    assert got.success and got.metadata["fetch_id"] == "pg-1"


async def test_no_line_names_the_allowlist_and_only_the_not_https_line_names_web_fetch(fetch, world, monkeypatch):
    tool, _ = fetch
    lines = [(await tool.run(url=u)).error for u in
             ("http://example.com/", "https://127.0.0.1/", "https://[::1]/", "https://224.0.0.1/", "https://user:pw@example.com/", "https://")]
    for status in (404, 503):
        world.pages["https://example.com/a"] = (status, PAGE, "text/html")
        lines.append((await tool.run(url="https://example.com/a")).error)
    for error in (FakeTimeout("t"), FakeError("Download is starting"), FakeError("net::ERR_FAILED")):
        async def boom(url, error=error):
            raise error

        world.on_goto = boom
        lines.append((await tool.run(url="https://example.com/a")).error)
    assert all(lines) and not any("allow_private" in line for line in lines)
    assert [line for line in lines if "web_fetch" in line] == [NOT_HTTPS]


async def test_the_same_url_twice_gives_two_fetch_ids_with_equal_windows(fetch):
    tool, store = fetch
    one, two = await tool.run(url="https://example.com/a"), await tool.run(url="https://example.com/a")
    assert (one.metadata["fetch_id"], two.metadata["fetch_id"]) == ("pg-1", "pg-2")
    assert one.output.split("\n\n[chars")[0] == two.output.split("\n\n[chars")[0]
    for fid in ("pg-1", "pg-2"):
        assert (await WebPageQueryTool(store).run(fetch_id=fid, pattern="needle")).success


async def test_two_concurrent_calls_on_one_bound_copy_get_distinct_fetch_ids(fetch):
    tool, store = fetch
    a, b = await asyncio.gather(tool.run(url="https://example.com/a"), tool.run(url="https://example.com/a"))
    assert {a.metadata["fetch_id"], b.metadata["fetch_id"]} == {"pg-1", "pg-2"}


async def test_info_is_stable_and_never_touches_the_runtime(world, runtime):
    tool = BrowserFetchTool(runtime, ContentStore())
    assert tool.info() == tool.info() and world.launches == [] and runtime._browser is None


async def test_bound_copies_share_one_browser_and_keep_their_own_store(fetch, world):
    tool, _ = fetch
    world.pages["https://example.com/b"] = (200, "z" * 100, "text/html")
    s1, s2 = ContentStore(), ContentStore()
    one, two = tool.bound_to(s1), tool.bound_to(s2)
    a, b = await asyncio.gather(one.run(url="https://example.com/a"), two.run(url="https://example.com/b"))
    assert (a.metadata["fetch_id"], b.metadata["fetch_id"]) == ("pg-1", "pg-1") and len(world.launches) == 1
    again = await one.run(url="https://example.com/a")
    assert again.metadata["fetch_id"] == "pg-2"
    assert not (await WebPageQueryTool(s2).run(fetch_id="pg-2", pattern="needle")).success
