"""The ASGI app: SSE down, POST up, the static shell, and the auth that holds them.

Why SSE + POST rather than a WebSocket (§4.1, LOCKED). Three rows decided it and the third is
the one that matters:

* **Reconnect** is built into `EventSource`, with `Last-Event-ID` resume in the spec. A socket
  gives you nothing and you hand-roll it every time.
* **The replay buffer already exists on disk** — the bus persists with a monotonic `seq` BEFORE
  delivering — so the cursor is not invented here.
* **Answering a permission ask after a drop.** The answer is a POST carrying `request_id`; it
  does not care which connection delivered the question or whether it still exists. On a socket
  it is tempting to answer on the socket, which couples a human decision to a TCP connection a
  locked phone has already dropped — converting "the owner stepped away" into "the decision was
  lost". An idempotent POST removes that failure mode structurally.

iOS seals it: backgrounded sockets are closed and may not fire `onclose`/`onerror`, so the
client believes it is connected when it is not. MCP's own Streamable HTTP transport is
SSE-down/POST-up for the same reason.

Starlette + uvicorn, not aiohttp and not a hand-rolled h11 loop: the PRD names an ASGI app, ASGI
is what lets the whole surface be driven in-process by `httpx.ASGITransport` with no socket and
no model (which is how every test in `tests/unit/channels/test_web_*.py` runs), and both are
already in the tree as transitive dependencies of `mcp`. They are declared explicitly in the
`web` extra anyway — a transitive dependency is not a promise, and this repo has already been
bitten by treating one as though it were.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import structlog
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from starlette.responses import StreamingResponse
from starlette.routing import Route

from localharness.config import session_presence

from . import auth, push
from .channel import WebChannel
from .protocol import (
    COLLAPSIBLE_GROUPS,
    FINISH_REASONS_KNOWN,
    NEVER_FIRED_EVENTS,
    PROTOCOL_VERSION,
    TRANSCRIPT_RULES,
    event_schemas,
    frame_schemas,
)

log = structlog.get_logger(__name__)

JSON_CONTENT_TYPE = "application/json"

SIMPLE_CONTENT_TYPES: frozenset[str] = frozenset({
    "text/plain", "application/x-www-form-urlencoded", "multipart/form-data",
})
"""The three content types a cross-origin `<form>` or a `text/plain` `fetch` can send with NO
preflight. Refused outright on every POST — for a verb that can approve a permission or set
`unattended` mode, leaning on the cookie's `SameSite` alone is not the right posture."""

UI_INDEX = "index.html"

PACKAGED_UI_DIR = Path(__file__).parent / "ui"
"""Where the reference page ships. Served from a DIRECTORY, live: the owner edits the file and
pulls to refresh — no server restart, no build step, no bundler (WIN-B). The worker beside it
caches nothing for the same reason, so a restyle reaches an installed phone on the next reload."""

MAX_BODY_BYTES = 1 << 20
"""1 MiB. Sourced from the measured message corpus: the longest real user message is 10,900
characters, so this is two orders of magnitude of headroom for a text verb while still refusing
to buffer something that is not a message at all."""

KEEPALIVE_S = 15.0
"""Comment-only `: ping` cadence. Derived from the failure it prevents: proxies kill idle
connections at 30-60 seconds, so half of the tightest of those is the margin that still holds if
a proxy is stricter than documented."""

QUEUE_POLL_S = 1.0
"""How long the SSE loop waits on the client queue before checking whether a keepalive is due.

It is a POLL, not a cadence: it exists so a stream with nothing to say still wakes often enough
to notice the keepalive deadline, and it has to be well under `KEEPALIVE_S` or the ping drifts
late. One second is the coarsest value that keeps a 15-second ping inside its own window.
"""

INTENTS: frozenset[str] = frozenset({"auto", "nudge", "queue"})

DEFAULT_MID_TURN_INTENT = "queue"
"""An omitted `intent` on a MID-TURN send means queue, not auto (§4.4.1).

Measured from the router's own source, not assumed: tier-2's LLM fallback runs `permit_wait=30`
plus `timeout=5`, and the budget is their SUM because the classifier shares a capacity-1
inference gate with the very turn it is asking about. So on a phone a message tier-1 does not
confidently classify can spend ~35 seconds in the router and resolve to QUEUE anyway — while
somebody stands on a platform watching a composer that swallowed their message. Bug #92 was
exactly this. Paying a model call and half a minute to reach the answer we would have chosen for
free is the worst available trade. Written down so a later "simplification" back to `auto` does
not silently reintroduce the stall.
"""


SESSION_LIST_CAP = 50
"""How many logs `GET /api/sessions` returns, newest first.

The drawer it feeds is a recency surface on a phone, not an archive browser: fifty rows is
already several screens of thumb-scroll, and each row costs a title scan (below). A box that
outgrows this wants `ls sessions/`, not a longer drawer.
"""

TITLE_SCAN_LINES = 200
"""How deep into a log the title scan reads before giving up on finding a `UserMessage`.

The first user message is normally within the first handful of lines; the cap is there so one
bring-up-heavy or malformed log cannot cost a full-file read per listing row.
"""

TITLE_MAX_CHARS = 100
"""A history row's title is a recognition cue, not the message: one drawer line's worth."""


TOKEN_FRAGMENT_KEY = "t"
"""The URL-FRAGMENT key the enrolment QR carries the app token in: `https://host/#t=<token>`.

§7.3 says the token is never placed in a URL, and this honours the reason that rule exists
rather than its letter. What the rule is about is the query string, which lands in server logs,
in proxy logs and in `Referer` headers. A fragment is sent to no server at all — it never leaves
the browser — and the page below strips it from the address bar the moment it has been read, so
it does not survive into history or a screenshot either. The alternative is hand-typing a
256-bit secret on a phone keyboard, which is the setup step people abandon.
"""

MANIFEST_CONTENT_TYPE = "application/manifest+json"

MANIFEST: dict[str, Any] = {
    "name": "localharness",
    "short_name": "harness",
    "description": "The agent harness on your box, from your phone.",
    "id": "/",
    "scope": "/",
    "start_url": "/",
    # `standalone`, not `browser`: WIN-A is "beats opening ChatGPT", and a thing you reach by
    # unlocking the phone, opening Safari and typing a .ts.net URL has lost that contest before
    # the model is consulted. The competitor is a home-screen icon.
    "display": "standalone",
    "orientation": "portrait",
    # Both are the reference page's ground (`--color-bg` from localharness.dev's theme, the same
    # value its `theme-color` meta carries). They are what iOS paints the splash screen and the
    # surround with BEFORE a line of the page has run, so a mismatch here is a white flash on
    # every cold launch of a dark app.
    "background_color": "#0D0F15",
    "theme_color": "#0D0F15",
    "icons": [
        {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any"},
        {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any"},
        {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"},
        {"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml"},
    ],
}
"""The install identity. Deliberately plain, and the icon is a placeholder meant to be replaced —
drop a new `icon-192.png` / `icon-512.png` into the UI directory and it is done, no build step,
which is the same promise the reference page itself makes (WEBCH-20)."""


SECURITY_HEADERS: dict[str, str] = {
    # THE one that matters here. The shell is served without a credential and is inert without
    # one — but `localStorage` is scoped to the ORIGIN, not to the frame embedding it, so an
    # iframe of this origin inherits an enrolled session wholesale. Both headers, because
    # `frame-ancestors` is the modern rule and `X-Frame-Options` is what an older WebView obeys.
    "X-Frame-Options": "DENY",
    # `frame-ancestors` alone, NOT a `default-src`: the reference page is deliberately one file
    # with inline `<style>` and an inline module, and a `default-src 'self'` would break the very
    # page this serves. `base-uri` and `form-action` are free alongside it and close a `<base>`
    # rewrite and a cross-origin form post.
    "Content-Security-Policy": "frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
    "X-Content-Type-Options": "nosniff",
    # The token is never put in a URL, and this keeps any future one out of a referrer anyway.
    "Referrer-Policy": "no-referrer",
}
"""Stamped on EVERY response by :class:`_SecurityHeaders`, the static shell included.

Not on the API alone: the shell is the thing an attacker wants to frame, precisely because it is
the one route that answers without a credential.
"""


class _SecurityHeaders(BaseHTTPMiddleware):
    """Add :data:`SECURITY_HEADERS` to every response, including the streaming one."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response


def _json(payload: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status)


def _unauthorized() -> JSONResponse:
    return _json({"error": auth.UNAUTHENTICATED_ERROR}, status=401)


class WebServer:
    """Routes, auth and the SSE loop for one `localharness web` process."""

    def __init__(
        self,
        channel: WebChannel,
        *,
        token: str,
        ui_dir: Optional[Path] = None,
        on_first_message: Optional[Any] = None,
        on_new_session: Optional[Any] = None,
        replay: Any = None,
        config_dir: Optional[str | Path] = None,
    ) -> None:
        self.channel = channel
        self.token = token
        self.ui_dir = (ui_dir or PACKAGED_UI_DIR).resolve()
        self.on_first_message = on_first_message
        self.on_new_session = on_new_session
        self.replay = replay
        self.config_dir = config_dir
        self._bringup_started = False
        self.app = self._build()

    # ------------------------------------------------------------------ auth

    def _authed(self, request: Request, *, post: bool) -> Optional[JSONResponse]:
        """None when the request may proceed, otherwise the refusal.

        Two different checks for two different transports, because `EventSource` cannot send an
        `Authorization` header:

        * **POST** requires `Authorization: Bearer <token>` AND `Content-Type: application/json`.
          Either alone forces a CORS preflight this server answers for no foreign origin, so the
          POST surface is structurally CSRF-safe rather than relying on `SameSite`.
        * **GET** (the stream included) accepts the bearer header or the `SameSite=Strict`
          cookie set at enrolment. The token is never placed in a URL — a query string lands in
          logs and referrers.
        """
        if post:
            ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
            if ctype in SIMPLE_CONTENT_TYPES or ctype != JSON_CONTENT_TYPE:
                return _json({"error": auth.CONTENT_TYPE_ERROR}, status=415)
            presented = self._bearer(request)
        else:
            presented = self._bearer(request) or request.cookies.get(auth.AUTH_COOKIE)
        return None if auth.constant_time_match(presented, self.token) else _unauthorized()

    @staticmethod
    def _bearer(request: Request) -> Optional[str]:
        header = request.headers.get("authorization") or ""
        scheme, _, value = header.partition(" ")
        return value.strip() if scheme.lower() == "bearer" and value.strip() else None

    async def _body(self, request: Request) -> dict:
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            raise ValueError("body too large")
        if not raw:
            return {}
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("body must be a JSON object")
        return parsed

    # ------------------------------------------------------------------ routes

    def _build(self) -> Starlette:
        """Routes, plus the headers that keep the unauthenticated shell from being weaponised.

        `SECURITY_HEADERS` is not decoration. The static page is served without a credential and
        argued inert — every `/api` route refuses an unauthenticated caller — but the enrolled
        token lives in `localStorage`, which is scoped to the ORIGIN, not to whatever frames it.
        Without a framing rule, a malicious page the owner's browser happens to visit can iframe
        this origin, inherit an enrolled session, and UI-redress BOTH taps of the `_always`
        confirm — defeating by clicks the one control built specifically so a single request
        cannot write a permanent, global, unrevokable grant. Found by an adversarial review, not
        by the tests.
        """
        routes = [
            Route("/api/auth/enroll", self.enroll, methods=["POST"]),
            Route("/api/stream", self.stream, methods=["GET"]),
            Route("/api/health", self.health, methods=["GET"]),
            Route("/api/protocol", self.protocol, methods=["GET"]),
            Route("/api/schema", self.schema, methods=["GET"]),
            Route("/api/tools", self.tools, methods=["GET"]),
            Route("/api/grants", self.grants, methods=["GET"]),
            Route("/api/permissions", self.permissions, methods=["GET"]),
            Route("/api/permissions/{request_id}/answer", self.answer, methods=["POST"]),
            Route("/api/pending/{pending_id}/{verb}", self.pending, methods=["POST"]),
            Route("/api/bringup/abort", self.abort_bringup, methods=["POST"]),
            Route("/api/push/key", self.push_key, methods=["GET"]),
            Route("/api/push/subscribe", self.push_subscribe, methods=["POST"]),
            # Before the catch-all, and a ROUTE rather than a file, because an authenticated
            # caller gets a `start_url` that pairs the installed app (see `manifest`).
            Route("/manifest.webmanifest", self.manifest, methods=["GET"]),
            Route("/api/tool-results/{eviction_id}", self.tool_result, methods=["GET"]),
            Route("/api/sessions", self.sessions, methods=["GET"]),
            Route("/api/sessions/new", self.new_session, methods=["POST"]),
            Route("/api/sessions/{session_id}/events", self.events, methods=["GET"]),
            Route("/api/sessions/{session_id}/message", self.message, methods=["POST"]),
            Route("/api/sessions/{session_id}/cancel", self.cancel, methods=["POST"]),
            Route("/api/sessions/{session_id}/mode", self.mode, methods=["POST"]),
            Route("/api/sessions/{session_id}/command", self.command, methods=["POST"]),
            Route("/{path:path}", self.static, methods=["GET"]),
        ]
        return Starlette(routes=routes, middleware=[Middleware(_SecurityHeaders)])

    # ------------------------------------------------------------------ static

    async def static(self, request: Request) -> Response:
        """Serve the UI directory, realpath-confined (WEBCH-40).

        The shell itself is served without a credential and is INERT without one: every `/api`
        route below refuses an unauthenticated caller, so reaching this port still is not a
        shell. It is served openly because the alternative — a token in the URL of the page that
        bootstraps enrolment — is exactly what §7.3 refuses to do.

        Confinement is this method's own, not `StaticFiles`': both sides are resolved, so a
        symlink inside the UI directory pointing at `/etc/passwd` resolves outside the root and
        is refused where a string-prefix check would have served it.
        """
        rel = request.path_params.get("path") or ""
        if not rel or rel.endswith("/"):
            rel = f"{rel}{UI_INDEX}"
        target = auth.confine(self.ui_dir, rel)
        if target is None or not target.is_file():
            return PlainTextResponse("not found", status_code=404)
        # `no-cache` means REVALIDATE, not "don't store": with the ETag FileResponse already
        # sends, an unchanged page is a 304 and costs nothing. Without this header there is NO
        # policy at all, and iOS applies heuristic caching to the installed app's shell — which
        # is how the owner spent a night hard-refreshing to see fixes that were already live
        # (2026-09-15, "very weird spot where i have to hard refresh to see the intended ui").
        return FileResponse(target, headers={"Cache-Control": "no-cache"})

    async def enroll(self, request: Request) -> Response:
        """Trade the app token for the `SameSite=Strict` cookie `EventSource` can carry.

        The one POST that does not already require the cookie, because it is what mints it.
        """
        refusal = self._authed(request, post=True)
        if refusal is not None:
            return refusal
        response = _json({"status": "enrolled", "protocol_version": PROTOCOL_VERSION})
        response.set_cookie(
            auth.AUTH_COOKIE, self.token,
            httponly=True, samesite="strict", secure=True, path="/",
        )
        return response

    # ------------------------------------------------------------------ the stream

    async def stream(self, request: Request) -> Response:
        """Backfill from the session log, then tail live, de-duplicating on `seq` at the seam.

        The composition, and why it is in this order (§4.2.3):

        1. **Attach the live client FIRST.** Everything published from this instant is buffered
           for it. Backfilling first would leave a window in which an event is in neither half.
        2. **Stream the per-session file** from the cursor. Read DIRECTLY, not through
           `bus.replay()` — that reads the per-AGENT `bus-events.jsonl`, every session that agent
           has ever run, and filters in Python afterwards, which on a corpus already at 12 MB
           makes every reconnect an O(agent-lifetime) scan. `replay_and_resubmit()` is worse
           still and is deliberately NOT used: it re-delivers history to EVERY current subscriber
           of that type, so catching one phone up would hand a simultaneously-attached terminal a
           pile of events it already rendered.
        3. **Drain the live buffer**, skipping anything whose `seq` the backfill already served.

        The order can duplicate but can never drop, and the `seq` de-dup removes the duplicates.
        That is the safe side of the off-by-one this seam has sitting next to it.
        """
        refusal = self._authed(request, post=False)
        if refusal is not None:
            return refusal
        cursor = self._cursor(request)
        await self._maybe_prewarm()
        client = self.channel.attach_client()
        return StreamingResponse(
            self._sse(client, cursor),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store",
                # Nginx and friends buffer text/event-stream by default, which turns a live
                # stream into one long pause followed by everything at once.
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @staticmethod
    def _cursor(request: Request) -> Optional[int]:
        """The client's resume point, normalised to "the first seq I still want".

        Two sources, and **they do not mean the same thing** — which is the whole reason this is
        a function rather than one `int()` call:

        * `?from=N` is the client's own explicit request and already means "from N inclusive":
          the page computes it as `last_seen + 1`.
        * `Last-Event-ID: N` is the BROWSER's, attached automatically on `EventSource`'s own
          auto-reconnect, and names the last event it RECEIVED. Serving from N would re-deliver
          it — one duplicated event on every ordinary wifi blip, with no app code involved.

        So the header is advanced by one and the query parameter is not. Getting this wrong is
        invisible in the cold-relaunch test everyone writes (which sets `?from=`) and shows up
        only on the commonest reconnect there is.

        `?from=` wins when both are present: a cold relaunch is the case that matters — iOS
        reclaims the page and `EventSource`'s own resume state goes with it — so the cursor the
        client persisted to `localStorage` is the authoritative one (WEBCH-42).
        """
        raw = request.query_params.get("from")
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                pass
        header = request.headers.get("last-event-id")
        if header is not None:
            try:
                return int(header) + 1
            except (TypeError, ValueError):
                pass
        return None

    async def _sse(self, client: Any, cursor: Optional[int]) -> AsyncIterator[bytes]:
        served_upto: Optional[int] = None
        try:
            hello = await self.channel.hello(client, cursor)
            if self.replay is not None:
                hello = hello.model_copy(update={"synthetic": True})
            yield _frame(hello.frame_type, None, hello.model_dump_json())

            backfill_sid = self.channel.session_id
            if cursor is not None and self.channel.session_id:
                async for name, seq, payload in self._backfill(self.channel.session_id, cursor):
                    served_upto = seq if seq is not None else served_upto
                    yield _frame(name, seq, payload)
                gap = self.channel.gap_against_log(self.channel.session_id, served_upto)
                if gap is not None:
                    log.error("web_replay_gap", from_seq=gap.from_seq, to_seq=gap.to_seq)
                    yield _frame(gap.frame_type, None, gap.model_dump_json())

            if self.channel.bringup is not None:
                yield _frame(
                    self.channel.bringup.frame_type, None, self.channel.bringup.model_dump_json()
                )

            last_ping = time.monotonic()
            while True:
                try:
                    name, seq, payload = await asyncio.wait_for(
                        client.queue.get(), timeout=QUEUE_POLL_S
                    )
                except asyncio.TimeoutError:
                    now = time.monotonic()
                    if now - last_ping >= KEEPALIVE_S:
                        last_ping = now
                        # A comment, not a frame: it keeps proxies from killing an idle
                        # connection at 30-60s and is invisible to `EventSource`.
                        yield b": ping\n\n"
                    continue
                if seq is not None:
                    if served_upto is not None and seq <= served_upto:
                        # The de-dup guards the backfill/live seam of ONE session's seq line —
                        # and seq lines RESTART when a new chat swaps the bus (2026-09-15,
                        # live: a phone holding yesterday's cursor watched frames arrive while
                        # every event of the fresh session was eaten right here). A row from a
                        # different session than the one backfilled is not a duplicate of that
                        # backfill, whatever its seq number says. Parsing only on the drop
                        # path keeps the common case free.
                        try:
                            row_sid = json.loads(payload).get("session_id")
                        except (ValueError, TypeError):
                            row_sid = None
                        if row_sid == backfill_sid:
                            continue  # the backfill already served it — the seam's de-dup
                    served_upto = seq
                    client.last_seq = seq
                yield _frame(name, seq, payload)
        except asyncio.CancelledError:
            raise
        finally:
            self.channel.detach_client(client)

    async def _backfill(self, session_id: str, from_seq: int) -> AsyncIterator[tuple]:
        """Stream the per-session JSONL from `from_seq`, verbatim, skipping torn lines."""
        path = self._session_log(session_id)
        if path is None or not path.exists():
            return
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn line is skipped, exactly as bus.replay() does
            seq = data.get("seq")
            if seq is None or seq < from_seq:
                continue
            yield data.get("event_type") or "Event", seq, line

    def _session_log(self, session_id: str) -> Optional[Path]:
        if self.replay is not None:
            return self.replay.path
        base = self.channel._session_dir
        return None if base is None else base / f"{session_id}.jsonl"

    async def _maybe_prewarm(self) -> None:
        """Opening the app IS the signal — for a live session, and for a replay (LOCKED, §6.1).

        ACP cannot know a user is coming until they submit a prompt; this channel can, because it
        has a connect event the others lack. So a client connecting begins bring-up in the
        background immediately and by the time a thumb has finished typing the session is usually
        already up. The single cheapest thing that serves WIN-A.

        **But a phone in a pocket must not be able to spin the GPU.** A connect is a much weaker
        signal of intent than a message — a backgrounded page reconnects on its own — and bring-up
        will start a harness-managed model server if one is configured. So the connect path
        pre-warms ONLY when the provider already answers its probe, which is the normal case on a
        warm box and costs nothing there. A cold server waits for an actual message, where a human
        has demonstrably asked for something. `POST .../message` goes through `_ensure_session`,
        which has no such condition.

        The replay case uses the same signal for a different reason, found by driving the real
        command: starting the playback when the SERVER starts means a browser opened a few
        seconds later has already missed the opening of the session it came to watch — and "edit,
        refresh, watch it again" is the entire development loop `--replay` exists for.
        """
        if self.replay is not None:
            self.replay.start()
            return
        if self._bringup_started or self.on_first_message is None:
            return
        if self.channel.session_id is not None:
            return
        if await self.channel.probe_model() is not True:
            log.info("web_prewarm_skipped", reason="provider not answering; waiting for a message")
            return
        self._bringup_started = True
        self.on_first_message()

    # ------------------------------------------------------------------ read verbs

    async def health(self, request: Request) -> Response:
        refusal = self._authed(request, post=False)
        if refusal is not None:
            return refusal
        await self.channel.probe_model()
        return _json({
            "protocol_version": PROTOCOL_VERSION,
            "model_state": self.channel.model_state(),
            "session_id": self.channel.session_id,
            "agent_id": self.channel.agent_id,
            "session_live": self.channel.session_id is not None,
            "turn_in_progress": self.channel._turn_running,
            "mode": getattr(self.channel._gate, "mode", None),
            "clients": self.channel.client_count,
            "replay": self.replay is not None,
            "push_enrolled": await asyncio.to_thread(self._push_enrolled),
            # WEBCH-29. The bring-up warning is one line on the wire and therefore invisible to a
            # phone that connected afterwards; this is the same fact as state, which a client
            # arriving at any time can read.
            # Re-scanned per request, not snapshotted at bring-up: the first session would
            # otherwise report an empty list forever, however many joined it afterwards.
            "other_sessions": session_presence.summary(
                await asyncio.to_thread(self.channel.live_co_tenants)),
        })

    def _push_enrolled(self) -> int:
        """How many devices would buzz. The page uses it to tell "notifications are on" from
        "notifications are on, on some other phone"."""
        try:
            return len(push.SubscriptionStore(self.config_dir).all())
        except Exception:  # noqa: BLE001 — health must answer even when push is broken
            return 0

    async def protocol(self, request: Request) -> Response:
        """The contract, served from the same places the code reads it from.

        `commands[]` comes from the REPL's own table and `modes[]` from the gate's own mode list
        rather than being hardcoded here: a one-thumb mode chip or command menu that duplicates
        those lists client-side is exactly the drift the generated event schema exists to
        prevent.
        """
        refusal = self._authed(request, post=False)
        if refusal is not None:
            return refusal
        from localharness.agent.gate import MODE_STRICTNESS
        from localharness.cli.slash_commands import SLASH_COMMANDS

        return _json({
            "protocol_version": PROTOCOL_VERSION,
            "events": [
                {"name": name, "never_fires": name in NEVER_FIRED_EVENTS}
                for name in sorted(event_schemas())
            ],
            "sse_only": sorted(frame_schemas()),
            "verbs": [
                {"method": m, "path": p, "note": n} for m, p, n in _VERBS
            ],
            "commands": [{"name": n, "description": d} for n, d in SLASH_COMMANDS],
            "modes": sorted(MODE_STRICTNESS, key=lambda m: MODE_STRICTNESS[m]),
            "intents": sorted(INTENTS),
            "default_mid_turn_intent": DEFAULT_MID_TURN_INTENT,
            "collapsible_groups": list(COLLAPSIBLE_GROUPS),
            "finish_reasons_known": list(FINISH_REASONS_KNOWN),
            "transcript_rules": list(TRANSCRIPT_RULES),
        })

    async def schema(self, request: Request) -> Response:
        refusal = self._authed(request, post=False)
        if refusal is not None:
            return refusal
        return _json({
            "protocol_version": PROTOCOL_VERSION,
            "events": event_schemas(),
            "sse_only": frame_schemas(),
        })

    async def tools(self, request: Request) -> Response:
        """The live registry: name → group, destructive.

        The client keys its "never collapse this" rule off `group` — not a hardcoded tool list
        and not the dead `risk_level`, which is null on every `Action` ever recorded — so a
        newly-added dangerous tool cannot silently start collapsing.
        """
        refusal = self._authed(request, post=False)
        if refusal is not None:
            return refusal
        return _json({
            "collapsible_groups": list(COLLAPSIBLE_GROUPS),
            "tools": _tool_rows(self.channel._tool_registry),
        })

    async def grants(self, request: Request) -> Response:
        """What have I permanently allowed? (§5.3b)

        Read-only. `allow_always` writes to the global grant store keyed by workspace realpath,
        never expires, and has NO revoke command — so the least this endpoint owes is making the
        consequences visible from the phone. Revocation itself stays out of scope, named rather
        than buried in YAML.
        """
        refusal = self._authed(request, post=False)
        if refusal is not None:
            return refusal
        from localharness.config.grants import GrantStore

        try:
            store = GrantStore()
            data = store._load()
        except Exception as exc:  # noqa: BLE001
            return _json({"error": f"could not read the grant store: {exc}", "grants": []})
        rows: list[dict] = []
        for workspace, block in (data or {}).items():
            if not isinstance(block, dict):
                continue
            for kind in ("grants", "refusals"):
                for record in block.get(kind) or []:
                    if isinstance(record, dict):
                        rows.append({"workspace": workspace, "kind": kind, **record})
        return _json({
            "path": str(GrantStore().path),
            "revocable": False,
            "note": "grants are global, keyed by workspace realpath, and never expire. There is "
                    "no revoke command — edit the file to remove one.",
            "grants": rows,
        })

    async def permissions(self, request: Request) -> Response:
        """Everything awaiting a human, as STATE: `{blocking, parked}`.

        Both halves, deliberately. Without `parked`, a cold client would have to fold every
        `PermissionStaged` not yet matched by a `PermissionResolved` out of the whole session log
        by hand.
        """
        refusal = self._authed(request, post=False)
        if refusal is not None:
            return refusal
        gate = self.channel._gate
        parked = []
        for pending in list(getattr(gate, "pending", {}).values()):
            parked.append({
                "id": pending.id,
                "rendering": pending.rendering,
                "agent_label": pending.agent_label,
                "session_id": pending.session_id,
                "created_at": pending.created_at,
                "tool_name": getattr(pending.request, "tool_name", ""),
                "klass": getattr(pending.request, "klass", ""),
            })
        return _json({"blocking": self.channel.open_asks(), "parked": parked})

    async def tool_result(self, request: Request) -> Response:
        """A ContentStore-evicted body, live session only.

        Explicitly NOT a general "tap for the full output": `Observation.output` is already
        bounded by the registry's `result_size_cap_chars`, and beyond that cap the full body is
        not on the bus AT ALL. The ContentStore is in-process, cleared on restart, and holds only
        bodies bulky enough to have been evicted for context-window reasons. So when there is
        nothing, this says why rather than implying the data exists somewhere.
        """
        refusal = self._authed(request, post=False)
        if refusal is not None:
            return refusal
        eviction_id = request.path_params["eviction_id"]
        store = getattr(getattr(self.channel._agent_loop, "_context_manager", None), "store", None)
        body = None
        if store is not None:
            try:
                body = store.get(eviction_id)
            except Exception:  # noqa: BLE001
                body = None
        if body is None:
            return _json({
                "error": "no stored body for that id",
                "detail": "the ContentStore holds only results EVICTED for context-window "
                          "reasons, in this process, since the last restart. A result truncated "
                          "at the registry's size cap no longer exists anywhere — "
                          "`original_length` records what was lost.",
            }, status=404)
        return _json({"eviction_id": eviction_id, "body": body})

    async def sessions(self, request: Request) -> Response:
        """The history list: every session log on disk, newest first (the drawer behind ☰).

        Reading is free, exactly as `events` below — no session, no GPU, no bring-up. `title`
        is the first `UserMessage` in the log, because "what did I ask" is how a human
        recognises a conversation; a log whose scan finds none gets null and the client
        falls back to the id.
        """
        refusal = self._authed(request, post=False)
        if refusal is not None:
            return refusal

        def scan() -> tuple[list[dict], Optional[str]]:
            if self.replay is not None:
                files = [self.replay.path] if self.replay.path.exists() else []
            else:
                base = self.channel._session_dir
                if base is None:
                    return [], ("no log directory yet — it binds when the first session "
                                "comes up, so a cold box has no history to list")
                if not base.is_dir():
                    return [], None
                files = sorted(base.glob("*.jsonl"),
                               key=lambda p: p.stat().st_mtime, reverse=True)
            rows: list[dict] = []
            for path in files[:SESSION_LIST_CAP]:
                title: Optional[str] = None
                try:
                    with path.open(encoding="utf-8", errors="replace") as fh:
                        for lineno, raw in enumerate(fh):
                            if lineno >= TITLE_SCAN_LINES:
                                break
                            try:
                                data = json.loads(raw)
                            except json.JSONDecodeError:
                                continue  # a torn line is skipped, exactly as _backfill does
                            if data.get("event_type") == "UserMessage":
                                text = (data.get("content") or "").strip()
                                title = text[:TITLE_MAX_CHARS] or None
                                break
                    stat = path.stat()
                except OSError:
                    continue  # deleted between glob and read: a listing must not 500 over it
                rows.append({
                    "session_id": path.stem,
                    "title": title,
                    "modified_unix": stat.st_mtime,
                    "size_bytes": stat.st_size,
                    "live": path.stem == self.channel.session_id,
                })
            return rows, None

        listed, note = await asyncio.to_thread(scan)
        return _json({"sessions": listed, "note": note})

    async def events(self, request: Request) -> Response:
        """Replay/backfill off disk. Reading is always free — it needs nothing live.

        Stated explicitly because the obvious implementation of "tap a chat" wires it to the
        expensive path, which would make idle curiosity cost you a running turn.
        """
        refusal = self._authed(request, post=False)
        if refusal is not None:
            return refusal
        session_id = request.path_params["session_id"]
        try:
            from_seq = int(request.query_params.get("from") or 0)
        except (TypeError, ValueError):
            from_seq = 0
        rows = [line async for _, _, line in self._backfill(session_id, from_seq)]
        return Response(
            "\n".join(rows), media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store"},
        )

    # ------------------------------------------------------------------ write verbs

    async def message(self, request: Request) -> Response:
        """A user turn. `intent` ∈ auto | nudge | queue.

        `nudge` and `queue` SKIP the classifier entirely — a phone can offer two buttons where a
        terminal has to guess, and a human who already said which they meant should not be made
        to pay a model call and up to ~35 seconds to be told the same thing (§4.4.1).
        """
        refusal = self._authed(request, post=True)
        if refusal is not None:
            return refusal
        try:
            body = await self._body(request)
        except (ValueError, json.JSONDecodeError) as exc:
            return _json({"error": str(exc)}, status=400)
        text = (body.get("text") or "").strip()
        if not text:
            return _json({"error": "text is required"}, status=400)
        intent = body.get("intent") or ""
        if intent and intent not in INTENTS:
            return _json({"error": f"intent must be one of {sorted(INTENTS)}"}, status=400)

        started = self._ensure_session()
        if not self.channel._turn_running:
            # Between turns there is nothing to nudge: every intent is the same "run it next".
            self.channel.submit(text)
            return _json({"status": "queued", "intent": "queue", "bringing_up": started})

        effective = intent or DEFAULT_MID_TURN_INTENT
        if effective == "nudge":
            if not await self.channel.nudge(text):
                return _json({
                    "status": "no_resolver",
                    "detail": "this session cannot steer a running turn yet — the REPL installs "
                              "that handle when the session comes up.",
                }, status=409)
            return _json({"status": "nudged", "intent": "nudge"})
        if effective == "auto":
            # The terminal's behaviour, kept available for a caller that wants the classifier.
            # Not the default here, and §4.4.1 says why at length.
            self.channel.submit(text)
            return _json({"status": "queued", "intent": "auto"})
        self.channel.submit(text)
        return _json({"status": "queued", "intent": "queue"})

    async def cancel(self, request: Request) -> Response:
        refusal = self._authed(request, post=True)
        if refusal is not None:
            return refusal
        cancelled = await self.channel.cancel_turn()
        if not cancelled:
            return _json({"status": "nothing_to_cancel"})
        return _json({"status": "cancelled"})

    async def new_session(self, request: Request) -> Response:
        """The + button: end the current session and build a fresh one (owner, 2026-09-14).

        Refused mid-turn on purpose — the stop verb exists, and killing a generating session
        under a thumb that wanted a clean topic break discards real GPU work. Refused in replay
        and wherever the runner installed no restart handle: this endpoint must never pretend.
        The old session's log stays on disk, so the drawer keeps the chat that just ended.
        """
        refusal = self._authed(request, post=True)
        if refusal is not None:
            return refusal
        if self.on_new_session is None:
            return _json({"error": "this server cannot restart the session — a replay serves "
                                   "a finished log, and only the live runner installs the "
                                   "restart handle"}, status=409)
        if self.channel._turn_running:
            return _json({"error": "a turn is running — stop it first, then start the new "
                                   "chat"}, status=409)
        await self.on_new_session()
        return _json({
            "status": "starting",
            "note": "the fresh session is building; the old chat stays in the history list",
        })

    async def answer(self, request: Request) -> Response:
        """Answer a blocking ask. Idempotent; the `_always` kinds take a second, server-checked tap."""
        refusal = self._authed(request, post=True)
        if refusal is not None:
            return refusal
        try:
            body = await self._body(request)
        except (ValueError, json.JSONDecodeError) as exc:
            return _json({"error": str(exc)}, status=400)
        kind = body.get("kind") or ""
        result = self.channel.answer_ask(
            request.path_params["request_id"], kind, body.get("confirm_token")
        )
        status = {
            "unknown": 404, "invalid": 400, "confirm_required": 409,
        }.get(result.get("status", ""), 200)
        return _json(result, status=status)

    async def pending(self, request: Request) -> Response:
        """Answer a PARKED call. The answer closes it on every surface, via `PermissionResolved`."""
        refusal = self._authed(request, post=True)
        if refusal is not None:
            return refusal
        verb = request.path_params["verb"]
        if verb not in ("approve", "deny"):
            return _json({"error": "verb must be approve or deny"}, status=404)
        try:
            pending_id = int(request.path_params["pending_id"])
        except (TypeError, ValueError):
            return _json({"error": "pending id must be a number"}, status=400)
        try:
            ok = await self.channel.resolve_pending(verb, pending_id)
        except KeyError as exc:
            return _json({"error": str(exc)}, status=404)
        if not ok:
            return _json({
                "status": "no_resolver",
                "detail": "this session has no pending handle yet — it is installed as the "
                          "session comes up.",
            }, status=409)
        return _json({
            "status": "recorded",
            "verb": verb,
            "note": "it runs when the model re-issues the call, not now — an approval is a "
                    "one-shot pass on that exact call, and the loop is the only thing that ever "
                    "dispatches a tool.",
        })

    async def mode(self, request: Request) -> Response:
        refusal = self._authed(request, post=True)
        if refusal is not None:
            return refusal
        try:
            body = await self._body(request)
        except (ValueError, json.JSONDecodeError) as exc:
            return _json({"error": str(exc)}, status=400)
        gate = self.channel._gate
        if gate is None:
            return _json({"error": "no permission gate on this session yet"}, status=409)
        try:
            gate.set_mode(str(body.get("mode") or ""), from_channel=True)
        except ValueError as exc:
            return _json({"error": str(exc)}, status=400)
        return _json({"status": "set", "mode": gate.mode})

    async def command(self, request: Request) -> Response:
        """A slash command, routed through the REPL's OWN dispatcher.

        Not re-implemented here, deliberately: ACP had to re-implement `/pending`, `/approve` and
        `/deny` because it took the self-driving shape, and that is exactly the drift between
        surfaces this channel exists not to repeat. The command goes onto the same inbound queue
        an ordinary message does, and `_dispatch_input` claims it.
        """
        refusal = self._authed(request, post=True)
        if refusal is not None:
            return refusal
        try:
            body = await self._body(request)
        except (ValueError, json.JSONDecodeError) as exc:
            return _json({"error": str(exc)}, status=400)
        text = (body.get("text") or "").strip()
        if not text:
            return _json({"error": "text is required"}, status=400)
        if not text.startswith("/"):
            text = "/" + text
        self._ensure_session()
        self.channel.submit(text)
        return _json({"status": "queued", "command": text.split()[0]})

    async def abort_bringup(self, request: Request) -> Response:
        refusal = self._authed(request, post=True)
        if refusal is not None:
            return refusal
        return _json({"status": "aborting" if self.channel.abort_bringup() else "not_building"})

    # ------------------------------------------------------------------ the app (A2)

    async def push_key(self, request: Request) -> Response:
        """The VAPID public key, so the page can call `pushManager.subscribe()`.

        Authenticated even though a public key is not a secret: an unauthenticated caller has no
        business learning that this box exists, let alone enough to start an enrolment.
        """
        refusal = self._authed(request, post=False)
        if refusal is not None:
            return refusal
        # to_thread: this reads (and on first use writes) a key file, and it runs on the
        # same loop that is streaming tokens to every other attached client.
        keys = await asyncio.to_thread(push.load_or_create_vapid, self.config_dir)
        return _json({"application_server_key": keys.application_server_key})

    async def push_subscribe(self, request: Request) -> Response:
        """Register a device for Web Push.

        **This is a credential-gated verb and the gate is the point.** A push subscription is a
        standing channel into the owner's lock screen, carrying deep links to the very things the
        harness wants approved; letting an unauthenticated caller register one would hand a
        stranger both the notifications and a map of what to tap. It takes the same two-part POST
        check as every other verb — bearer token AND a JSON content type — so a cross-origin
        form, which is the shape that would otherwise skip a preflight, cannot reach it.
        """
        refusal = self._authed(request, post=True)
        if refusal is not None:
            return refusal
        try:
            body = await self._body(request)
        except ValueError as exc:
            return _json({"error": str(exc)}, status=400)
        subscription = push.valid_subscription(body)
        if subscription is None:
            return _json({"error": push.SUBSCRIPTION_ERROR}, status=400)
        store = push.SubscriptionStore(self.config_dir)
        devices = await asyncio.to_thread(_store_add, store, subscription)
        return _json({"status": "subscribed", "devices": devices})

    async def manifest(self, request: Request) -> Response:
        """The PWA manifest — and the one place the install story gets honest.

        iOS keeps a home-screen web app's storage in a DIFFERENT jar from Safari's. So a phone
        that paired in Safari by scanning the QR installs an app that knows nothing: same origin,
        empty `localStorage`, no cookie. The obvious fix — put the token in `start_url` — would
        publish it to every unauthenticated caller of this route, so instead the token-bearing
        `start_url` is served ONLY to a caller that already has the credential, which the page
        arranges by asking for the manifest with `crossorigin="use-credentials"`.

        Belt and braces, because that credentialed manifest fetch is browser behavior we cannot
        force: when it does not happen the generic manifest is served, the app installs anyway,
        and the shell shows its pairing field. The install is never blocked — only pre-paired.
        """
        authed = self._authed(request, post=False) is None
        body = dict(MANIFEST)
        if authed:
            body["start_url"] = f"/#{TOKEN_FRAGMENT_KEY}={self.token}"
        response = _json(body)
        response.headers["Content-Type"] = MANIFEST_CONTENT_TYPE
        # Never let a proxy or a shared cache keep the token-bearing variant.
        response.headers["Cache-Control"] = "no-store"
        return response

    def _ensure_session(self) -> bool:
        """Start session bring-up if it has not begun. Returns True if this call started it."""
        if self._bringup_started or self.on_first_message is None or self.replay is not None:
            return False
        if self.channel.session_id is not None:
            return False
        self._bringup_started = True
        self.on_first_message()
        return True


_VERBS: tuple[tuple[str, str, str], ...] = (
    ("POST", "/api/sessions/new", "end the current session, build a fresh one; the old log stays"),
    ("POST", "/api/sessions/{id}/message", "a user turn; body {text, intent?}"),
    ("POST", "/api/sessions/{id}/cancel", "cancel the in-flight turn, then emit TurnCancelled"),
    ("POST", "/api/sessions/{id}/mode", "set the permission mode; body {mode}"),
    ("POST", "/api/sessions/{id}/command", "a slash command through the REPL's dispatcher"),
    ("POST", "/api/permissions/{request_id}/answer",
     "answer a blocking ask; idempotent; the _always kinds need a second POST with confirm_token"),
    ("POST", "/api/pending/{pending_id}/{approve|deny}", "answer a parked call"),
    ("POST", "/api/bringup/abort", "abandon a wedged session bring-up"),
    ("POST", "/api/auth/enroll", "trade the bearer token for the stream cookie"),
    ("POST", "/api/push/subscribe", "register this device for Web Push; body is a PushSubscription"),
    ("GET", "/api/push/key", "the VAPID application server key for pushManager.subscribe()"),
    ("GET", "/api/stream", "the SSE event stream; ?from={seq} resumes"),
    ("GET", "/api/sessions", "the session logs on disk, newest first: [{session_id, title, live, …}]"),
    ("GET", "/api/sessions/{id}/events", "replay off disk as NDJSON; ?from={seq}"),
    ("GET", "/api/permissions", "everything awaiting a human: {blocking, parked}"),
    ("GET", "/api/tools", "the live registry: name -> group, destructive"),
    ("GET", "/api/health", "model reachability and session state"),
    ("GET", "/api/grants", "the permanent grants in force (read-only)"),
    ("GET", "/api/tool-results/{eviction_id}", "a ContentStore-evicted body, live session only"),
    ("GET", "/api/schema", "JSON Schema for every event and frame"),
    ("GET", "/api/protocol", "this document, as data"),
)


def _store_add(store: Any, subscription: dict) -> int:
    """Add and count in ONE worker-thread hop, rather than two round trips to the loop."""
    store.add(subscription)
    return len(store.all())


def _tool_rows(registry: Any) -> list[dict]:
    """Every live tool with the two fields a collapse rule may key off.

    Walks the registry's buckets rather than resolving one agent's toolset: the UI's question is
    "what could appear in this transcript", and a subagent's tools appear in it too.
    """
    if registry is None:
        return []
    seen: dict[str, dict] = {}

    def _add(name: str, tool: Any) -> None:
        if name in seen:
            return
        try:
            schema = tool.info()
        except Exception:  # noqa: BLE001 — one broken tool must not empty the list
            return
        seen[name] = {
            "name": getattr(schema, "name", name),
            "group": getattr(schema, "group", "other") or "other",
            "destructive": bool(getattr(schema, "destructive", False)),
            "description": (getattr(schema, "description", "") or "")[:200],
        }

    buckets = list(getattr(registry, "_tools", {}).values())
    buckets += list(getattr(registry, "_division_tools", {}).values())
    buckets += list(getattr(registry, "_agent_tools", {}).values())
    for bucket in buckets:
        if isinstance(bucket, dict):
            for name, tool in bucket.items():
                _add(name, tool)
    return sorted(seen.values(), key=lambda row: row["name"])


def _frame(event_name: str, seq: Optional[int], payload: str) -> bytes:
    """One SSE frame.

    The `id:` line is written ONLY for a real bus event. An SSE-only frame stamping its own id
    would poison `Last-Event-ID` and the cursor the client persists, because neither would then
    name something a replay can serve.
    """
    head = f"id: {seq}\n" if seq is not None else ""
    return f"{head}event: {event_name}\ndata: {payload}\n\n".encode("utf-8")
