"""The browser plugin's runtime: one sandboxed headless Chromium behind the checking proxy, verified at every launch before any page."""
from __future__ import annotations

import asyncio
import contextlib
import importlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localharness import __version__
from localharness.tools.builtin.browser_proxy import NAV_TIMEOUT_S, Proxy, serve
from localharness.tools.builtin.web_tool import FETCH_MAX_BODY_BYTES

MAX_CONTEXTS = 2
REACH_FAILED = "reach check failed"
_ARGS = ("--disable-quic", "--force-webrtc-ip-handling-policy=disable_non_proxied_udp", "--disable-breakpad")
_SKIPPED_TYPES = frozenset({"image", "media", "font"})
# Counts UTF-16 code units, so a cut can leave a lone high surrogate, which would not survive the driver pipe.
_SLICE = """n => {
  const t = document.body ? document.body.innerText : '';
  let s = t.slice(0, n);
  const c = s.charCodeAt(s.length - 1);
  if (c >= 0xD800 && c <= 0xDBFF) s = s.slice(0, -1);
  return [t.length > n, s];
}"""
_API: Any = None  # the tests' stand-in for playwright.async_api


@dataclass(frozen=True)
class Rendered:
    url: str
    status: int
    text: str
    capped: bool


class RenderError(Exception):
    """`kind` is unavailable, timeout, file or failed."""

    def __init__(self, kind: str, detail: str = ""):
        super().__init__(kind, detail)
        self.kind, self.detail = kind, detail


def _command_lines() -> list[list[str]]:
    """The argv of every running process."""
    if Path("/proc").is_dir():
        out = []
        for p in Path("/proc").glob("[0-9]*/cmdline"):
            with contextlib.suppress(OSError):
                out.append([a for a in p.read_bytes().decode(errors="replace").split("\0") if a])
        return out
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        ps = subprocess.run(["ps", "-ax", "-o", "command="], capture_output=True, text=True, timeout=10)
        return [line.split() for line in ps.stdout.splitlines()]
    return []


async def _cost_route(route) -> None:
    await (route.abort() if route.request.resource_type in _SKIPPED_TYPES else route.continue_())


class BrowserRuntime:
    def __init__(self, config):
        self._config = config
        self._idle_s = config.idle_close_s
        self._lock = asyncio.Lock()
        self._api: Any = None
        self._proxy: Proxy | None = None
        self._pw: Any = None
        self._browser: Any = None
        self._ua = ""

    async def ensure(self):
        async with self._lock:
            if self._browser is not None and self._browser.is_connected():
                return self._browser
            await self._teardown()
            await self._launch()
            return self._browser

    async def _launch(self) -> None:
        c = self._config
        self._proxy = await serve(None)
        self._api = _API or importlib.import_module("playwright.async_api")
        self._pw = await self._api.async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=True, chromium_sandbox=c.sandbox, executable_path=c.executable_path or None, handle_sigint=False,
            # Chromium bypasses the proxy for loopback unless `<-loopback>` subtracts that rule: https://chromium.googlesource.com/chromium/src/+/HEAD/net/docs/proxy.md
            proxy={"server": f"http://127.0.0.1:{self._proxy.port}", "bypass": "<-loopback>",
                   "username": self._proxy.username, "password": self._proxy.password},
            args=list(_ARGS))
        await self._assert_flags()
        await self._canary()

    async def _assert_flags(self) -> None:
        want = f"127.0.0.1:{self._proxy.port}"
        argv = next((a for a in await asyncio.to_thread(_command_lines)
                     if not any(x.startswith("--type=") for x in a)
                     and any(x.startswith("--proxy-server=") and x.endswith(want) for x in a)), [])
        if not (any(x.startswith("--proxy-bypass-list=") and "<-loopback>" in x for x in argv)
                and all(flag in argv for flag in _ARGS)):
            raise RenderError("unavailable", REACH_FAILED)

    async def _canary(self) -> None:
        """Loopback must reach the proxy and be refused there, never the listener."""
        accepted = 0

        def accept(_reader, writer):
            nonlocal accepted
            accepted += 1
            writer.close()

        server = await asyncio.start_server(accept, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        context = await self._browser.new_context()
        try:
            page = await context.new_page()
            ua = await page.evaluate("navigator.userAgent")
            with contextlib.suppress(self._api.Error):
                await page.goto(f"https://127.0.0.1:{port}/", timeout=NAV_TIMEOUT_S * 1000)
        finally:
            await context.close()
            server.close()
            await server.wait_closed()
        if f"127.0.0.1:{port} not public" not in self._proxy.refused or accepted:
            raise RenderError("unavailable", REACH_FAILED)
        self._ua = f"{ua} LocalHarness/{__version__} (+https://localharness.dev)"

    async def render(self, url: str) -> Rendered:
        browser = await self.ensure()
        context = await browser.new_context(
            accept_downloads=False, service_workers="block", java_script_enabled=True, bypass_csp=False,
            ignore_https_errors=False, user_agent=self._ua)
        try:
            await context.route("**/*", _cost_route)
            page = await context.new_page()
            response = await page.goto(url, wait_until="load", timeout=NAV_TIMEOUT_S * 1000)
            await page.wait_for_timeout(500)
            capped, text = await page.evaluate(_SLICE, FETCH_MAX_BODY_BYTES)
            text = re.sub(r"\n\s*\n\s*\n+", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()
            return Rendered(page.url, response.status if response else 200, text, capped)
        finally:
            await context.close()

    async def close(self) -> None:
        async with self._lock:
            await self._teardown()

    async def _teardown(self) -> None:
        owned, self._browser, self._pw, self._proxy = (self._browser, self._pw, self._proxy), None, None, None
        for handle, stop in zip(owned, ("close", "stop", "close")):
            if handle is not None:
                with contextlib.suppress(Exception):
                    await getattr(handle, stop)()
