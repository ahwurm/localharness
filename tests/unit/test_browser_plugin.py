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
