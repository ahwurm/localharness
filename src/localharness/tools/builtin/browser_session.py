"""An agent's interactive browser session: the snapshot window and the browser_open, click, type and scroll tools."""
from __future__ import annotations

import asyncio
import contextlib
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
POPUP_S = 2
TEXT_MAX = 256
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
_POST = ("That would submit a form (POST); browsing is read-only and the page is unchanged. "
         "Do not retry; follow a link from this snapshot, or use web_search for this site.")
_STALE = "ref {ref} is not on the current page; use a ref from your latest snapshot."
_BAD_REF = "ref must be one from the snapshot, like e12."
_NO_PAGE = "No browser page is open; call browser_open with a URL first."
_POPUP = "A link opened a new window: {url}. Call browser_open with it to go there."
_NEW_WINDOW = "A link opened a new window that never loaded; it was closed."
_PASSWORD = "That is a password field; this browser does not log in. Use a public page."
_REF = re.compile(r"[a-z0-9]{1,16}")
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
        self.context = self.page = self.flags = None
        self._popups: list[asyncio.Task] = []
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
        self.runtime._ended.add(self.store)
        self.runtime.release(self)
        self.prev = self.store = None
        self._popups = []

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
        self.context, self.flags = await self.runtime._new_context(browser)
        self.page = await self.context.new_page()
        self.context.on("page", self._on_page)

    def _on_page(self, page) -> None:
        if page is not self.page and not self.closed:
            self._popups.append(asyncio.ensure_future(self._popup(page)))

    async def _popup(self, page) -> str | None:
        """The URL a new window committed within `POPUP_S`, or None; the window is closed either way."""
        url = None
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                page.wait_for_url(lambda u: u != "about:blank", wait_until="commit", timeout=POPUP_S * 1000), POPUP_S + 1)
            url = page.url
        with contextlib.suppress(Exception):
            await page.close()
        return None if url in (None, "about:blank") else url

    async def _gone(self) -> bool:
        if self.closed or not self.live():
            await self.close()
            return True
        return False

    async def act(self, do) -> ToolResult:
        async with self.lock:
            try:
                return fail(_CLOSED) if await self._gone() else await self._act(do)
            finally:
                if not self.closed:
                    self._arm()

    async def _act(self, do) -> ToolResult:
        proxy = self.runtime._proxy
        if proxy is None:
            return fail(_CLOSED)
        api, refusals = self.runtime._api, proxy.refusals
        self.flags.post_refused = False
        try:
            if (early := await do(self.page)) is not None:
                return early
            await _bounded(self.page.wait_for_load_state(timeout=NAV_TIMEOUT_S * 1000), NAV_TIMEOUT_S)
            await self.page.wait_for_timeout(500)
            popups, self._popups = self._popups, []
            urls = await asyncio.gather(*popups, return_exceptions=True)
            if self.flags.post_refused:
                return fail(_POST)
            if await self._gone():
                return fail(_CLOSED)
            return await self._snapshot([_POPUP.format(url=u[:200]) if isinstance(u, str) else _NEW_WINDOW for u in urls])
        except (RenderError, api.TimeoutError, api.Error) as e:
            if self.flags.post_refused:
                return fail(_POST)
            if await self._gone():
                return fail(_CLOSED)
            return fail(render_line(e if isinstance(e, RenderError) else render_error(api, e, proxy, refusals, True)))

    async def _snapshot(self, extra: list[str] = ()) -> ToolResult:
        text = await _bounded(self.page.aria_snapshot(mode="ai", timeout=SNAPSHOT_S * 1000), SNAPSHOT_S)
        if self.closed:
            return fail(_CLOSED)
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
        lines = "".join(f"\n{ln}" for ln in extra)
        output = f"{UNTRUSTED_BANNER}URL: {url[:200]}\n\n{body}{low}{lines}\n{note.format(fid=got.metadata['fetch_id'])}"
        return got.model_copy(update={"output": output})


async def _element(page, ref: str):
    locator = page.get_by_ref(ref)
    if await _bounded(locator.count(), REF_S) == 0:
        return fail(_STALE.format(ref=ref))
    return await _bounded(locator.element_handle(timeout=REF_S * 1000), REF_S)


async def _click(page, ref: str) -> ToolResult | None:
    handle = await _element(page, ref)
    if isinstance(handle, ToolResult):
        return handle
    await _bounded(handle.click(timeout=ACT_S * 1000), ACT_S)


async def _type(page, ref: str, text: str, submit: bool) -> ToolResult | None:
    handle = await _element(page, ref)
    if isinstance(handle, ToolResult):
        return handle
    if ((await _bounded(handle.get_attribute("type"), ACT_S)) or "").lower() == "password":
        return fail(_PASSWORD, "validation_error")
    await _bounded(handle.fill(text, timeout=ACT_S * 1000), ACT_S)
    if submit:
        await _bounded(handle.press("Enter", timeout=ACT_S * 1000), ACT_S)


async def _scroll(page, direction: str) -> None:
    height = (page.viewport_size or {}).get("height", 720)
    await _bounded(page.mouse.wheel(0, height if direction == "down" else -height), ACT_S)


def _invalid(ref: str | None = None, text: str = "", direction: str = "down") -> str | None:
    if ref is not None and not (isinstance(ref, str) and _REF.fullmatch(ref)):
        return _BAD_REF
    if len(text) > TEXT_MAX:
        return f"text is over {TEXT_MAX} characters"
    if direction not in ("down", "up"):
        return "direction must be down or up"
    return None


class _SessionTool(Tool):
    timeout_s = NAV_TIMEOUT_S + 30
    _name = _description = ""
    _properties: dict = {}
    _required: tuple[str, ...] = ()

    def __init__(self, runtime, store: Any = None) -> None:
        self._runtime = runtime
        self._store = store if store is not None else get_default_store()

    def info(self) -> ToolSchema:
        return ToolSchema(
            name=self._name, group="web",
            gate_family="network", ingest="untrusted", host="safe", result_origin="untrusted",
            description=self._description,
            parameters={"type": "object", "properties": self._properties, "required": list(self._required)},
            destructive=False, estimated_tokens=WINDOW_CHARS // 4)

    def bound_to(self, store: Any) -> "_SessionTool":
        return type(self)(self._runtime, store)

    def _no_store(self) -> ToolResult | None:
        return self.err(_NO_STORE, error_type="validation_error") if self._store is get_default_store() else None

    async def _act(self, do, invalid: str | None) -> ToolResult:
        if invalid:
            return self.err(invalid, error_type="validation_error")
        if refused := self._no_store():
            return refused
        runtime = self._runtime
        if (session := runtime._sessions.get(id(self._store))) is None:
            return fail(_CLOSED if self._store in runtime._ended else _NO_PAGE)
        return await session.act(do)

    async def release_store(self, store: Any) -> None:
        session = self._runtime._sessions.get(id(store))
        if session is not None and session.store is store:
            await session.release()


class BrowserOpenTool(_SessionTool):
    timeout_s = 2 * NAV_TIMEOUT_S + 40
    _name, _description = "browser_open", _DESCRIPTION
    _properties = {"url": {"type": "string", "description": "The https URL to open."}}
    _required = ("url",)

    async def _execute(self, url: str) -> ToolResult:
        if len(url) > URL_MAX:
            return self.err(f"url is over {URL_MAX} characters", error_type="validation_error")
        if refused := await refusal(url):
            return refused
        if refused := self._no_store():
            return refused
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


class BrowserClickTool(_SessionTool):
    _name = "browser_click"
    _description = "Clicks the element with this ref from your latest browser snapshot. Returns the updated page."
    _properties = {"ref": {"type": "string", "description": "A ref from your latest snapshot, like e12."}}
    _required = ("ref",)

    async def _execute(self, ref: str) -> ToolResult:
        return await self._act(lambda page: _click(page, ref), _invalid(ref))


class BrowserTypeTool(_SessionTool):
    _name = "browser_type"
    _description = ("Types into the text field with this ref from your latest snapshot; submit presses Enter. "
                    "For a site's search box, not for logging in. Returns the updated page.")
    _properties = {"ref": {"type": "string", "description": "A ref from your latest snapshot, like e12."},
                   "text": {"type": "string", "description": f"The text to type, at most {TEXT_MAX} characters."},
                   "submit": {"type": "boolean", "description": "Press Enter after typing."}}
    _required = ("ref", "text")

    async def _execute(self, ref: str, text: str, submit: bool = False) -> ToolResult:
        return await self._act(lambda page: _type(page, ref, text, submit), _invalid(ref, text))


class BrowserScrollTool(_SessionTool):
    _name = "browser_scroll"
    _description = "Scrolls one screen on the open page, for lists that load as you scroll. Returns the updated page."
    _properties = {"direction": {"type": "string", "enum": ["down", "up"], "description": "down or up."}}
    _required = ("direction",)

    async def _execute(self, direction: str) -> ToolResult:
        return await self._act(lambda page: _scroll(page, direction), _invalid(direction=direction))
