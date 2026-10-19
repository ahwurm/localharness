"""The browser plugin's runtime on a fake Chromium: the real proxy and its checks, no browser, no playwright."""
from __future__ import annotations

import asyncio
import base64
import re
from types import SimpleNamespace

import pytest

from localharness import __version__
from localharness.tools.builtin import browser_tool
from localharness.tools.builtin.browser_tool import BrowserRuntime, Rendered

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
    monkeypatch.setattr(browser_tool, "serve", lambda upstream: seen.append(real(upstream)) or seen[-1])
    world.launch_error = FakeError("boom")
    await unavailable(runtime)
    await refuses_connections((await seen[0]).port)


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
    live, stale, target = (tmp_path / f"playwright_chromiumdev_profile-{n}" for n in ("live", "stale", "real"))
    other = tmp_path / "unrelated-dir"
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
