"""The browser plugin's runtime: one sandboxed headless Chromium behind the checking proxy, verified at every launch before any page."""
from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localharness import __version__
from localharness.tools.base import Tool, ToolResult, ToolSchema
from localharness.tools.builtin import netguard
from localharness.tools.builtin.browser_proxy import NAV_TIMEOUT_S, Proxy, serve
from localharness.tools.builtin.web_tool import (
    FETCH_DEFAULT_CHARS, FETCH_MAX_BODY_BYTES, UNTRUSTED_BANNER, WebFetchTool, _get_default_store, retain_window)

MAX_CONTEXTS = 2
RETRY_AFTER_S = 60
REACH_FAILED = "reach check failed"
SANDBOX_FAILED = "the Chromium sandbox could not start"
SANDBOX_OFF = "Chromium started without its sandbox"
MISSING_LIBS = "system libraries are missing"
NOT_INSTALLED = "Chromium is not installed"
_LAUNCH_ERRORS = (("Host system is missing dependencies", MISSING_LIBS), ("No usable sandbox", SANDBOX_FAILED),
                  ("Executable doesn't exist", NOT_INSTALLED))
_PROFILE_PREFIX = "playwright_chromiumdev_profile-"
_ARGS = ("--disable-quic", "--force-webrtc-ip-handling-policy=disable_non_proxied_udp", "--disable-breakpad")
_SKIPPED_TYPES = frozenset({"image", "media", "font"})
# Runs before any page script, so the natives are taken before the page can replace them. Counts UTF-16 code units,
# so a cut can leave a lone high surrogate, which would not survive the driver pipe.
_INIT = """(() => {
  const apply = Reflect.apply;
  const text = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'innerText').get;
  const body = Object.getOwnPropertyDescriptor(Document.prototype, 'body').get;
  const slice = String.prototype.slice, charCodeAt = String.prototype.charCodeAt;
  Object.defineProperty(window, '__lh_text', {value: n => {
    const b = apply(body, document, []);
    const t = b ? apply(text, b, []) : '';
    let s = apply(slice, t, [0, n]);
    const c = apply(charCodeAt, s, [s.length - 1]);
    if (c >= 0xD800 && c <= 0xDBFF) s = apply(slice, s, [0, -1]);
    return [t.length > n, s];
  }, writable: false, configurable: false, enumerable: false});
})()"""
_TEXT = "n => window.__lh_text(n)"
_API: Any = None  # the tests' stand-in for playwright.async_api
log = logging.getLogger(__name__)
_now = time.monotonic  # the tests' clock


@dataclass(frozen=True)
class Rendered:
    url: str
    status: int
    text: str
    capped: bool


class RenderError(Exception):
    """`kind` is unavailable, timeout, file, not_public (`detail` is the host) or failed."""

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


def _first_line(e: Exception) -> str:
    """The error's first line without the driver's `Class.method: ` prefix."""
    lines = str(e).strip().splitlines()
    return re.sub(r"^[A-Za-z]+\.[A-Za-z]+: ", "", lines[0]) if lines else type(e).__name__


def _unavailable_reason(e: Exception) -> str:
    return next((reason for needle, reason in _LAUNCH_ERRORS if needle in str(e)), f"launch failed: {_first_line(e)[:200]}")


def _sweep_profiles() -> None:
    """Remove profile directories a crashed launch left behind; nothing is touched unless the process list is readable."""
    if not (lines := _command_lines()):
        return
    live = {a.partition("=")[2] for argv in lines for a in argv if a.startswith("--user-data-dir=")}
    uid = os.getuid() if hasattr(os, "getuid") else None
    for d in Path(tempfile.gettempdir()).glob(_PROFILE_PREFIX + "*"):
        with contextlib.suppress(OSError):
            if d.is_dir() and not d.is_symlink() and str(d) not in live and (uid is None or d.stat().st_uid == uid):
                shutil.rmtree(d, ignore_errors=True)


async def _cost_route(route) -> None:
    await (route.abort() if route.request.resource_type in _SKIPPED_TYPES else route.continue_())


class BrowserRuntime:
    def __init__(self, config):
        self._config = config
        self._idle_s = config.idle_close_s
        self._lock = asyncio.Lock()
        self.unavailable: str | None = None
        self._retry_at: float | None = None
        self._slots = asyncio.Semaphore(MAX_CONTEXTS)
        self._in_flight = 0
        self._idle: asyncio.TimerHandle | None = None
        self._idle_task: asyncio.Task | None = None
        self._api: Any = None
        self._proxy: Proxy | None = None
        self._pw: Any = None
        self._browser: Any = None
        self._ua = ""

    async def ensure(self):
        async with self._lock:
            if self._retry_at is not None and _now() >= self._retry_at:
                self.unavailable = self._retry_at = None
            if self.unavailable:
                raise RenderError("unavailable", self.unavailable)
            if self._browser is not None and self._browser.is_connected():
                return self._browser
            await self._teardown()
            try:
                await self._launch()
            except BaseException as e:
                await self._teardown()
                if not isinstance(e, Exception):
                    raise
                self.unavailable = e.detail if isinstance(e, RenderError) else _unavailable_reason(e)
                if self.unavailable not in (REACH_FAILED, SANDBOX_OFF):
                    self._retry_at = _now() + RETRY_AFTER_S
                raise RenderError("unavailable", self.unavailable) from None
            return self._browser

    async def _launch(self) -> None:
        c = self._config
        await asyncio.to_thread(_sweep_profiles)
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
        if self._config.sandbox and "--no-sandbox" in argv:
            raise RenderError("unavailable", SANDBOX_OFF)

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
        self._cancel_idle()
        self._in_flight += 1
        try:
            browser = await self.ensure()
            async with self._slots:
                return await self._render(browser, url)
        finally:
            self._in_flight -= 1
            if not self._in_flight and self._browser is not None:
                self._idle = asyncio.get_running_loop().call_later(self._idle_s, self._start_idle_close)

    async def _render(self, browser, url: str) -> Rendered:
        api, proxy = self._api, self._proxy
        refusals = proxy.refusals
        context = await browser.new_context(
            accept_downloads=False, service_workers="block", java_script_enabled=True, bypass_csp=False,
            ignore_https_errors=False, user_agent=self._ua)
        try:
            await context.route("**/*", _cost_route)
            await context.add_init_script(_INIT)
            page = await context.new_page()
            response = await page.goto(url, wait_until="load", timeout=NAV_TIMEOUT_S * 1000)
            if response and "application/pdf" in response.headers.get("content-type", ""):
                raise RenderError("file")
            await page.wait_for_timeout(500)
            capped, text = await page.evaluate(_TEXT, FETCH_MAX_BODY_BYTES)
            if not (isinstance(capped, bool) and isinstance(text, str)):
                raise RenderError("failed")
            text = re.sub(r"\n\s*\n\s*\n+", "\n\n", re.sub(r"[ \t]+", " ", text[:FETCH_MAX_BODY_BYTES])).strip()
            return Rendered(page.url, response.status if response else 200, text, capped)
        except api.TimeoutError:
            raise RenderError("timeout") from None
        except api.Error as e:
            if "Download is starting" in str(e):
                raise RenderError("file") from None
            log.info("browser_fetch navigation failed: %s", _first_line(e)[:200])
            if proxy.refusals > refusals:
                raise RenderError("not_public", proxy.refused[-1].removesuffix(" not public").rpartition(":")[0]) from None
            raise RenderError("failed") from None
        finally:
            with contextlib.suppress(Exception):
                await context.close()

    def _cancel_idle(self) -> None:
        if self._idle is not None:
            self._idle.cancel()
            self._idle = None

    def _start_idle_close(self) -> None:
        self._idle_task = asyncio.ensure_future(self._close_if_idle())

    async def _close_if_idle(self) -> None:
        async with self._lock:
            if not self._in_flight:
                await self._teardown()

    async def close(self) -> None:
        self._cancel_idle()
        async with self._lock:
            await self._teardown()

    async def _teardown(self) -> None:
        owned, self._browser, self._pw, self._proxy = (self._browser, self._pw, self._proxy), None, None, None
        for handle, stop in zip(owned, ("close", "stop", "close")):
            if handle is not None:
                with contextlib.suppress(Exception):
                    await getattr(handle, stop)()


_DESCRIPTION = (
    "Like web_fetch, but runs the page's JavaScript first (slow, up to 30 s). If web_fetch returned "
    "\"low extractable text\" for an https URL, call this once with the same URL — web_fetch's "
    "\"do NOT re-fetch\" means do not call web_fetch again, not this tool. Otherwise use web_fetch. "
    "Same start_index and fetch_id; search the full page with web_page_query.")
_CAPPED = (f"[rendered text capped at {FETCH_MAX_BODY_BYTES} characters — the rest of this page was not read; "
           "the retained text is a PREFIX, not the full page]")
_NOT_HTTPS = "browser_fetch renders https pages only. For http://… use web_fetch."
_NOT_PUBLIC = ("browser_fetch: {host} is not a public address; browser_fetch reaches public sites only. "
               "Do not retry this host; use the search snippet or another source.")
_NO_RETRY = "Do not retry this URL; use the search snippet or another source."
_HTTP_FAILED = f"browser_fetch failed: HTTP {{status}}. {_NO_RETRY}"
_FAILED = f"browser_fetch failed: the page could not be loaded. {_NO_RETRY}"
_TIMED_OUT = f"browser_fetch: the page did not finish loading in {NAV_TIMEOUT_S} s. Do not retry; use the search snippet."
_FILE = "browser_fetch: that URL is a file, not a page. Use the search snippet, or find an HTML version."
_EMPTY = ("[rendered text is under 80 characters — the page is empty or blocking the browser; do NOT retry. "
          "Use the search snippet.]")
_NOT_AVAILABLE = ("browser_fetch is not available on this machine ({reason}); use the search snippet or another source. "
                  "The person can run `localharness doctor`.")


class BrowserFetchTool(Tool):
    timeout_s = NAV_TIMEOUT_S + 20

    def __init__(self, runtime: BrowserRuntime, store: Any = None) -> None:
        self._runtime = runtime
        self._store = store if store is not None else _get_default_store()

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="browser_fetch", group="web",
            gate_family="network", ingest="untrusted", host="safe", result_origin="untrusted",
            description=_DESCRIPTION, parameters=WebFetchTool(store=self._store).info().parameters,
            destructive=False, estimated_tokens=FETCH_DEFAULT_CHARS // 4)

    def bound_to(self, store: Any) -> "BrowserFetchTool":
        return BrowserFetchTool(self._runtime, store)

    async def _execute(self, url: str, max_chars: int = FETCH_DEFAULT_CHARS, start_index: int = 0) -> ToolResult:
        if not url.lower().startswith("https://"):
            return self.err(_NOT_HTTPS, error_type="validation_error")
        try:
            await netguard.check(url, allow_private=False)
        except netguard.NotPublic as exc:
            return self.err(_NOT_PUBLIC.format(host=exc.host), error_type="validation_error")
        except netguard.Refused as exc:
            return self.err(str(exc), error_type="validation_error")
        try:
            page = await self._runtime.render(url)
        except RenderError as e:
            return self.err({"timeout": _TIMED_OUT, "file": _FILE, "unavailable": _NOT_AVAILABLE.format(reason=e.detail),
                             "not_public": _NOT_PUBLIC.format(host=e.detail)}.get(e.kind) or _FAILED)
        if page.status >= 400:
            return self.err(_HTTP_FAILED.format(status=page.status))
        if len(page.text.strip()) < 80:
            return self.ok(f"{UNTRUSTED_BANNER}URL: {page.url}\n\n{_EMPTY}", url=page.url)
        text = page.text + (f"\n{_CAPPED}" if page.capped else "")
        return retain_window(self._store, text, page.url, start_index, max_chars, refetch="browser_fetch")
