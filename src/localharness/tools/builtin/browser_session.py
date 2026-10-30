"""An agent's interactive browser session: the snapshot window and the browser_open, click, type and scroll tools."""
from __future__ import annotations

import asyncio
import re
from typing import Any

from localharness.tools.base import Tool, ToolResult, ToolSchema
from localharness.tools.builtin.browser_proxy import NAV_TIMEOUT_S
from localharness.tools.builtin.browser_tool import RenderError, _HTTP_FAILED, fail, refusal, render_error, render_line
from localharness.tools.builtin.web_tool import FETCH_MAX_BODY_BYTES, UNTRUSTED_BANNER, get_default_store, retain_window

SESSION_IDLE_S = 600
LOCK_WAIT_S = 5
REF_S = 5
ACT_S = 5
SNAPSHOT_S = 10
URL_MAX = 2048
MIN_REFS = 5
WINDOW_CHARS = 8000
UNCHANGED_CHARS = 1500
CONTEXT_LINES = 2

CHANGED_NOTE = ('[showing from near the first change; earlier lines are unchanged. '
                'Refs outside this window: web_page_query("{fid}", "next") or ("{fid}", "load more")]')
TOP_NOTE = '[refs outside this window: web_page_query("{fid}", "<button or link text>")]'
UNCHANGED_NOTE = "[the page did not change]"
SHORTER_NOTE = "[the page got shorter; showing it from the top]"

LOW_CONTENT = ("[this page is nearly empty — the site may be blocking the browser; do not retry. "
               "Use the search snippet or another source.]")
_NO_STORE = ("The browser session needs an agent of its own; delegate the browsing to web-researcher. "
             "If you called a plugin:-prefixed name, call browser_open instead.")
_BUSY = ("Another agent is using the browser session; use browser_fetch for one read. "
         "Do not retry this more than once.")
_CLOSED = ("The browser session is closed; call browser_open with the URL again. "
           "Page state such as a dismissed cookie wall was lost.")
_DESCRIPTION = ("Opens a page you need to click, type or scroll on, and returns it with element refs. "
                "For one read use web_fetch, or browser_fetch if web_fetch said low extractable text. "
                "Do not use this just to read.")

_TRANSIENT = re.compile(r" ?\[[^\]=]*\]")


def _stable(text: str) -> list[str]:
    return [_TRANSIENT.sub("", ln) for ln in text.split("\n")]


def window_start(prev: tuple[str, str] | None, url: str, text: str) -> tuple[int, int, str]:
    """Where the next snapshot's window starts, how long it is, and the note line (a template over `{fid}`)."""
    if prev is None or prev[0] != url:
        return 0, WINDOW_CHARS, TOP_NOTE
    old, new = _stable(prev[1]), _stable(text)
    first = next((i for i, (a, b) in enumerate(zip(old, new)) if a != b), min(len(old), len(new)))
    if first == len(new):
        return (0, UNCHANGED_CHARS, UNCHANGED_NOTE) if len(old) == len(new) else (0, WINDOW_CHARS, SHORTER_NOTE)
    start = sum(len(ln) + 1 for ln in text.split("\n")[:max(0, first - CONTEXT_LINES)])
    return start, WINDOW_CHARS, CHANGED_NOTE if start else TOP_NOTE


async def _bounded(awaitable, seconds: float):
    try:
        return await asyncio.wait_for(awaitable, seconds)
    except asyncio.TimeoutError:
        raise RenderError("timeout") from None


class Session:
    """One agent store's browser context and page, owned by the runtime and ended only by `close()`."""

    def __init__(self, runtime, store: Any) -> None:
        self.runtime, self.store = runtime, store
        self.context = self.page = None
        self.lock = asyncio.Lock()
        self.prev: tuple[str, str] | None = None
        self.closed = False
        self._timer: asyncio.TimerHandle | None = None
        self._expiry: asyncio.Task | None = None

    def live(self) -> bool:
        browser = self.runtime._browser
        return browser is not None and browser.is_connected() and (self.page is None or not self.page.is_closed())

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._disarm()
        if self.context is not None:
            try:
                await self.context.close()
            except Exception:  # noqa: BLE001
                pass
        self.runtime.release(self)
        self.prev = self.store = None

    async def release(self) -> None:
        """Close after waiting for an action to finish, but not past `LOCK_WAIT_S`."""
        held = False
        try:
            await asyncio.wait_for(self.lock.acquire(), LOCK_WAIT_S)
            held = True
        except asyncio.TimeoutError:
            pass
        try:
            await self.close()
        finally:
            if held:
                self.lock.release()

    def _disarm(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _arm(self) -> None:
        self._disarm()
        self._timer = asyncio.get_running_loop().call_later(SESSION_IDLE_S, self._expire)

    def _expire(self) -> None:
        self._expiry = asyncio.ensure_future(self.release())

    async def open(self, url: str) -> ToolResult:
        async with self.lock:
            if self.closed:
                return fail(_CLOSED)
            try:
                return await self._open(url)
            finally:
                if not self.closed:
                    self._arm()

    async def _open(self, url: str) -> ToolResult:
        runtime = self.runtime
        if self.page is None:
            try:
                await _bounded(self._start(), NAV_TIMEOUT_S)
            except BaseException as e:
                await self.close()
                if not isinstance(e, Exception):
                    raise
                return fail(render_line(e if isinstance(e, RenderError) else RenderError("failed")))
        if (proxy := runtime._proxy) is None or self.closed:
            return fail(_CLOSED)
        api, refusals, navigating = runtime._api, proxy.refusals, True
        try:
            response = await _bounded(self.page.goto(url, wait_until="load", timeout=NAV_TIMEOUT_S * 1000), NAV_TIMEOUT_S)
            navigating = False
            if self.closed:
                return fail(_CLOSED)
            if response and "application/pdf" in response.headers.get("content-type", ""):
                raise RenderError("file")
            if response and response.status >= 400:
                return fail(_HTTP_FAILED.format(status=response.status))
            return await self._snapshot()
        except RenderError as e:
            return fail(render_line(e))
        except (api.TimeoutError, api.Error) as e:
            return fail(render_line(render_error(api, e, proxy, refusals, navigating)))

    async def _start(self) -> None:
        browser = await self.runtime.ensure()
        self.context, _ = await self.runtime._new_context(browser)
        self.page = await self.context.new_page()

    async def _snapshot(self) -> ToolResult:
        text = await _bounded(self.page.aria_snapshot(mode="ai", timeout=SNAPSHOT_S * 1000), SNAPSHOT_S)
        text = "\n".join(f"> {ln}" if ln.startswith("URL:") else ln
                         for ln in (text[:FETCH_MAX_BODY_BYTES] or "[empty page]").split("\n"))
        url = self.page.url
        start, cap, note = window_start(self.prev, url, text)
        got = retain_window(self.store, text, url, start, cap, None)
        if not got.success:
            return got
        self.prev = (url, text)
        body = got.output[len(UNTRUSTED_BANNER):]
        low = f"\n{LOW_CONTENT}" if text.count("[ref=") < MIN_REFS else ""
        output = f"{UNTRUSTED_BANNER}URL: {url[:200]}\n\n{body}{low}\n{note.format(fid=got.metadata['fetch_id'])}"
        return got.model_copy(update={"output": output})


class BrowserOpenTool(Tool):
    timeout_s = 2 * NAV_TIMEOUT_S + 40

    def __init__(self, runtime, store: Any = None) -> None:
        self._runtime = runtime
        self._store = store if store is not None else get_default_store()

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="browser_open", group="web",
            gate_family="network", ingest="untrusted", host="safe", result_origin="untrusted",
            description=_DESCRIPTION,
            parameters={"type": "object", "properties": {"url": {"type": "string", "description": "The https URL to open."}},
                        "required": ["url"]},
            destructive=False, estimated_tokens=WINDOW_CHARS // 4)

    def bound_to(self, store: Any) -> "BrowserOpenTool":
        return BrowserOpenTool(self._runtime, store)

    async def _execute(self, url: str) -> ToolResult:
        if len(url) > URL_MAX:
            return self.err(f"url is over {URL_MAX} characters", error_type="validation_error")
        if refused := await refusal(url):
            return refused
        if self._store is get_default_store():
            return self.err(_NO_STORE, error_type="validation_error")
        runtime = self._runtime
        session = runtime._sessions.get(id(self._store))
        if session is not None and not session.live():
            await session.close()
            session = None
        if session is None:
            if not await runtime.take_slot():
                return self.err(_BUSY)
            session = Session(runtime, self._store)
            runtime.track(session)
        return await session.open(url)

    async def release_store(self, store: Any) -> None:
        session = self._runtime._sessions.get(id(store))
        if session is not None and session.store is store:
            await session.release()
