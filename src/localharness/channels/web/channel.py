"""`WebChannel` — the harness's bus, faithfully, over one SSE connection.

The shape of this file follows three rules from the PRD, and every oddity in it is one of them:

1. **The wire is a faithful mirror of the bus.** Every bus event this channel subscribes to
   reaches the client verbatim, as the same `model_dump_json()` bytes the session JSONL gets.
   No base-class handler is on the live path (§4.2.2) — not `on_action`, not `on_observation`,
   and not `on_task_complete`/`on_turn_failed`, whose base implementations DROP a child turn's
   completion and DOWNGRADE a child's failure. Those are RENDERING defaults that belong to the
   client; a filter at the wire would be the facade trap arriving by the back door — invisible,
   well-intentioned, and impossible for a UI to undo.

2. **A closed phone cannot hang the agent.** In `auto` — the default mode — the gate never calls
   the asker at all: it PARKS the call, hands the model a refusal it can route around, and
   returns immediately. So the phone's primary permission surface is a queue with a badge, not a
   modal, and `ask_holds_dialog` is False because the screen may be in a pocket.

3. **Fail closed, always.** Every path out of `ask_permission` that is not a human's answer ends
   in a rejection. There is no error, no disconnect, no shutdown and no bug in this file that
   can turn a question nobody answered into permission granted.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import structlog

from localharness.channels.base import ChannelAdapter, sanitize_for_display
from localharness.channels.errors import NotInteractiveError
from localharness.core.bus import EventBus
# Only the four types this channel REACTS to beyond forwarding. Everything else reaches the
# client through `EVENT_TYPE_MAP` in `start()`, which is the point: nothing here is a list of
# what the wire carries.
from localharness.core.events import (
    Action,
    Escalation,
    Heartbeat,
    PermissionResolved,
    PermissionStaged,
    TurnCompleted,
    TurnFailed,
    TurnStarted,
)

from . import push as push_mod
from .protocol import (
    AskExpired,
    AskOption,
    BlockingAsk,
    BringUpStage,
    GapDetected,
    Hello,
    Lagged,
    Notice,
    ReasoningDelta,
    StatusTick,
    StreamClosed,
    TokenDelta,
    TurnCancelled,
    WireFrame,
)

log = structlog.get_logger(__name__)

# --------------------------------------------------------------------------- tunables

STATUS_TICK_HZ = 4.0
"""How often the instrument cluster is pushed WHILE A TURN RUNS.

Sourced from the surface it mirrors, not chosen: the terminal's status row is a live repaint a
person reads as continuous motion, and 4 Hz is the slowest rate at which a token counter still
reads as counting rather than stepping. It ticks only during a turn, so an idle phone holds a
silent stream and the cost at rest is zero.
"""

STATUS_TICK_INTERVAL_S = 1.0 / STATUS_TICK_HZ
"""Derived from the rate above rather than written twice."""

PUSH_SUMMARY_CHARS = 120
"""How much of a turn summary or a parked call's rendering a notification body may carry.

A lock screen truncates around here anyway, and the encrypted push record has a fixed size — so
a 40 KB tool error is cut where the notification is composed rather than failing at the
transport, where the only symptom would be a push that silently never arrived.
"""

SSE_KEEPALIVE_S = 15.0
"""Comment-only `: ping` interval.

Derived from the failure it prevents: proxies kill idle connections at 30-60 seconds, so the
keepalive must be comfortably under the tightest of those. Half of 30 is the margin that still
holds if a proxy is stricter than documented.
"""

CLIENT_QUEUE_MAX_FRAMES = 2048
"""How far behind one client may fall before its buffer is dropped instead of grown.

Sourced from a real turn's event count: the measured corpus tops out in the low hundreds of
events for a long agentic turn, so this is roughly an order of magnitude of headroom above the
worst observed case — enough that a phone sleeping through an entire long turn still catches up
without a resync, and bounded so a client that is gone forever cannot make the server hold the
session in memory forever. Overflow is VISIBLE (`Lagged`), never silent; the client re-syncs
through the same `?from={seq}` cursor reconnect already uses.
"""

CONFIRM_TOKEN_TTL_S = 120.0
"""How long a server-minted `_always` confirmation token stays valid.

Long enough for a person to read a second dialog and mean it; short enough that a token captured
from a log is useless by the time anybody reads the log. Single-use besides.
"""

ASK_FALLBACK_DECISION = "reject_once"
"""What every non-answer resolves to. Fail closed — SECURITY.md's "deny on doubt".

This is the load-bearing constant in the file. A dismissed dialog, a dropped connection, a
shutdown mid-question, a client that never comes back, an exception on the render path: all of
them are this, and none of them is an allow.
"""

TRUST_KLASS = "workspace-trust"
"""The one ask whose answer is permanent even though it arrives `grantable=False` (§5.6)."""

TRUST_ASK_TIMEOUT_S = 300.0
"""How long the workspace-trust question waits for an attached client before denying.

Sourced from the harness's own definition of "somebody is still here": the terminal's
`PRESENCE_WINDOW_S` is 300 seconds, and a question that has gone unanswered for longer than the
harness believes a person is present has no person behind it. It needs a deadline of its OWN —
unlike every other ask, this one is called directly by `resolve_workspace_layer` rather than by
the gate, so there is no `permissions.ask.timeout_s` to inherit and nothing else would ever end
the wait. The wait is visible (`BringUpStage`) and abortable while it runs.
"""

TRUST_OPTION_NAMES: dict[str, str] = {
    "allow_always": "Trust this workspace",
    "reject_once": "Not now",
}
"""Its own two buttons.

Drawing it with the generic ungrantable pair would label a permanent decision "Allow once" — the
opposite of what it does. Mirrors what the ACP adapter had to do for the same question.
"""

GRANTABLE_OPTION_NAMES: dict[str, str] = {
    "allow_once": "Allow once",
    "allow_always": "Always allow here",
    "reject_once": "Deny once",
    "reject_always": "Never here",
}

UNGRANTABLE_KINDS: tuple[str, ...] = ("allow_once", "reject_once")
"""A request that cannot be remembered offers only the `_once` pair — an "always" button on a
class that asks every time by construction would be a lie."""

ALWAYS_KINDS: frozenset[str] = frozenset({"allow_always", "reject_always"})
"""The two kinds that write durable, global, never-expiring state, and therefore the two that
take a server-enforced second tap (§5.3a)."""

NO_RESOLVER_NOTICE = (
    "this session has no {what} handle yet — the harness is still coming up. Try again in a "
    "moment."
)
"""Said out loud rather than letting a tap do nothing.

Discord's `PENDING_NO_RESOLVER_LINE` precedent: the REPL installs these handles, and a channel
whose handle is missing must SAY so, because a button that silently does nothing is the defect
that teaches somebody the feature is broken.
"""


class _Client:
    """One attached SSE consumer: a bounded queue and the cursor it has been served up to."""

    __slots__ = ("id", "queue", "last_seq", "lagged", "dropped")

    def __init__(self, client_id: str) -> None:
        self.id = client_id
        self.queue: asyncio.Queue[tuple[str, Optional[int], str]] = asyncio.Queue(
            maxsize=CLIENT_QUEUE_MAX_FRAMES
        )
        self.last_seq: Optional[int] = None
        self.lagged = False
        self.dropped = 0


class _OpenAsk:
    """One blocking question, its answer future, and the outcome once it has one.

    The outcome is kept AFTER the future resolves so the answering POST is idempotent: a
    duplicate or late answer returns what was recorded, never a second decision. A retry on a
    flaky phone connection must not be able to allow something twice.
    """

    __slots__ = ("request_id", "request", "future", "frame", "asked_at", "outcome", "confirm")

    def __init__(self, request_id: str, request: Any, frame: BlockingAsk) -> None:
        self.request_id = request_id
        self.request = request
        self.frame = frame
        self.future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self.asked_at = time.monotonic()
        self.outcome: Optional[str] = None
        self.confirm: Optional[tuple[str, str, float]] = None  # (token, kind, expires_at)


class WebChannel(ChannelAdapter):
    """The channel behind `localharness web`.

    Driven the HYBRID way (§6.0): the HTTP server is up before any session exists (ACP's
    reachability), but once a session is built it is `OrchestratorREPL` that drives this channel,
    pulling from the inbound queue through `read_input()`. That choice is not a detail — taking
    ACP's self-driving shape instead would have cost the slash-command surface, the input router,
    the pending resolver and `UserMessage` publishing, and would have produced a session history
    with no user turns in it at all.
    """

    channel_id = "web"

    can_ask = True
    """There is a human with a screen. Declaring False would silently convert every ASK into a
    denial, which is a worse answer than asking."""

    ask_holds_dialog = False
    """...but the screen may be in a pocket. The Discord posture, not the Zed posture, and the
    honest one: Zed holds a dialog open because a person is demonstrably in front of it. So the
    gate keeps its deadline here and a question nobody reaches expires into `reject_once`.
    Setting this True would let one unanswered question hang a turn forever."""

    has_review_surface = False
    """No diff pane in v1, so an in-workspace edit asks ONCE per workspace rather than never.
    Claiming a review surface the UI lacks silently REMOVES a gate. Flipping this flag is the
    entire change if a diff view is ever built — and it must not be flipped before then."""

    streams_tokens = True
    """The first channel in the harness with live answer text in its own UI. The terminal passes
    `on_token=None`; ACP gives streaming to Zed but not to anything the harness draws. So "parity
    with the terminal" was never the bar here — the web channel exceeds it."""

    has_display_toggles = True
    """`/reasoning` and `/verbose` act on this channel: it owns `show_reasoning` and `verbose`
    and streams reasoning of its own, so the REPL's handlers apply here as they do in the
    terminal."""

    # Declared as class attributes because the REPL installs each one ONLY if the channel already
    # declares it as None (`if getattr(self._channel, "_pending_resolver", "absent") is None`).
    # A channel that does not declare the attribute is skipped with NO error — which fails
    # silently, at a tap, much later.
    _pending_resolver: Optional[Callable[[str, int], Awaitable[None]]] = None
    _nudge_resolver: Optional[Callable[[str, str], Awaitable[bool]]] = None
    _cancel_resolver: Optional[Callable[[], Awaitable[bool]]] = None

    def __init__(self, bus: EventBus, config: dict[str, Any]) -> None:
        super().__init__(bus, config)
        self._clients: dict[str, _Client] = {}
        self._inbound: asyncio.Queue[str] = asyncio.Queue()
        self._handles: list[Any] = []
        self._open_asks: dict[str, _OpenAsk] = {}
        self._started = False

        # Session identity, filled by bind_runtime() once _start_async has built one.
        self.session_id: Optional[str] = None
        self.agent_id: Optional[str] = None
        self._gate: Any = None
        self._tool_registry: Any = None
        self._llm: Any = None
        self._agent_loop: Any = None

        # Display toggles — the same two attributes the terminal owns, so the REPL's /reasoning
        # and /verbose handlers act on this channel through one generalized code path.
        self.show_reasoning: bool = False
        self.verbose: bool = False

        # The instrument sources, wired by `_start_async` to exactly what the terminal gets.
        # Declared here rather than sprung on the object later so a reader of this class can see
        # what the status frame is actually made of, and so a missing wiring is `None` rather
        # than an AttributeError inside a 4 Hz loop.
        self.tps_source: Optional[Callable[[], Any]] = None
        self.progress_source: Optional[Callable[[], Optional[dict]]] = None
        self.model_source: Optional[Callable[[], Optional[str]]] = None

        # Last real reachability answer for the provider endpoint, refreshed by `GET /api/health`.
        # Cached rather than probed on every frame: `model_state()` is read on every connect and
        # every health poll, and a TCP connect per read would put the phone's polling in front of
        # the model server.
        self._model_reachable: Optional[bool] = None

        # Streaming state. `_stream_id` is minted at the first token of a generation and retired
        # by the llm_response Action that supersedes it (§4.2.1).
        self._stream_id: Optional[str] = None
        self._status_task: Optional[asyncio.Task] = None
        self._turn_running = False
        self._context_pct: Optional[float] = None

        # Bring-up progress, so a client connecting mid-build sees the stage rather than a number.
        self._bringup: Optional[BringUpStage] = None
        self._bringup_abort: Optional[Callable[[], None]] = None

        # The highest seq this channel actually forwarded, per session. Compared against what the
        # session log can serve, so a swallowed persist failure surfaces as a gap (§4.2.3).
        self._forwarded_max: dict[str, int] = {}
        self._session_dir: Optional[Path] = None

        # Web Push, when a phone has enrolled. Optional on purpose: `--replay` sets none, and a
        # box nobody has paired a phone with must behave exactly as it did before A2.
        self._push: Any = None
        self._push_tasks: set[asyncio.Task] = set()

        # Other live sessions on this agent (WEBCH-29): the bring-up list, refreshed by
        # `live_co_tenants()` whenever health is read. Carried so a phone that connects LATER can
        # still see a co-tenant it was never present to be warned about — the warning itself is a
        # one-shot line on the wire, and a one-shot line is invisible to a client that arrives
        # after it.
        self._co_tenants: list[Any] = []
        self._co_tenant_rescan: Optional[Callable[[], list[Any]]] = None

    # ---------------------------------------------------------------- lifecycle

    def rebind_bus(self, bus: Any) -> None:
        """Point the event subscriptions at THE session's bus.

        THE 2026-09-15 BUG, foundational: this channel is constructed before any session
        exists, so the bus it subscribed to at start() was a placeholder nothing ever
        publishes on. Live bus events therefore NEVER reached a phone — frames (direct
        callbacks: the pulse, token streaming, asks) flowed, and every EVENT a client ever
        saw came from a connect-time backfill, which is why the app only looked right after
        a refresh. The unit tests never caught it because they build the channel and the
        publisher on ONE bus; production built them on two. Every bring-up hands the real
        bus over here — a new chat's fresh bus follows the same path.
        """
        if bus is None or bus is self.bus:
            return
        for handle in self._handles:
            with contextlib.suppress(Exception):
                self.bus.unsubscribe(handle)
        self.bus = bus
        from localharness.core.events import EVENT_TYPE_MAP

        raw = self._forward_event
        self._handles = [self.bus.subscribe(t, raw) for t in EVENT_TYPE_MAP.values()]

    def bind_runtime(
        self,
        *,
        session_id: str,
        agent_id: str,
        gate: Any = None,
        tool_registry: Any = None,
        llm: Any = None,
        agent_loop: Any = None,
        session_dir: Optional[Path] = None,
        bus: Any = None,
    ) -> None:
        """Hand the channel the session objects the HTTP surface has to answer questions about.

        Called from `_start_async`'s `web` branch once the session exists. Kept as one explicit
        call rather than reaching through the bus for each: `/api/tools`, `/api/permissions` and
        `/api/health` are *state* questions, and a channel that has to reconstruct state from an
        event stream it also forwards is a channel with two sources of truth.
        """
        self.rebind_bus(bus)
        self.session_id = session_id
        self.agent_id = agent_id
        self._gate = gate
        self._tool_registry = tool_registry
        self._llm = llm
        self._agent_loop = agent_loop
        self._session_dir = session_dir
        # The session exists from this line on, with two client-visible consequences. First,
        # "ready" is published HERE — the bring-up task only returns when the whole session
        # ends, so publishing ready from its `else` branch meant the build row and the ribbon
        # said "starting" for the session's entire life. Second, the give-up abort is disarmed:
        # it is wired to the session TASK's cancel, and once the build is over that handle
        # cancels a live session — which is exactly what a stray thumb did on 2026-09-14.
        self.set_bringup_abort(None)
        self.set_bringup("ready", elapsed=self._bringup.elapsed if self._bringup else 0.0)

    def reset_session(self) -> None:
        """Forget the bound session so the next bring-up binds fresh — the new-chat verb's half.

        The runtime handles go stale the moment the session task is cancelled; dropping them
        makes `/api/health` answer cold instead of describing a corpse. Open blocking asks die
        with the session — their answers have nowhere to land and their `PermissionResolved`
        is never coming. `_session_dir` deliberately survives: the drawer must keep listing
        history between sessions, and the next bind rewrites it anyway.
        """
        self.session_id = None
        self.agent_id = None
        self._gate = None
        self._llm = None
        self._agent_loop = None
        self._turn_running = False
        self._open_asks.clear()
        self.set_bringup_abort(None)

    async def start(self) -> None:
        """Subscribe to the bus. Idempotent — the REPL starts the channel that already exists.

        Every subscription here is a DIRECT one, and every handler forwards the raw event. The
        base class's `on_action`/`on_observation` would route through
        `send_tool_call`/`send_tool_result`, whose signatures have no `tool_call_id` parameter —
        so anything going through that seam loses the very key the UI needs to pair a call with
        its result. ACP hit this and routed around it; this channel does the same, deliberately.

        Subscribed to EVERY event type the harness declares, from `EVENT_TYPE_MAP` — not a list
        kept here. That is §4.2's promise made literal: "a new event type added anywhere in the
        harness reaches the phone with no channel change". A hand-kept list is the same drift a
        hand-written DTO per event would be, one type at a time instead of one field at a time,
        and it was already drifting: a replayed log carried memory-gate and predictive-gate events
        that a LIVE session did not, so the two modes disagreed about what the wire contains.
        """
        if self._started:
            return
        self._started = True
        from localharness.core.events import EVENT_TYPE_MAP

        raw = self._forward_event
        self._handles = [self.bus.subscribe(t, raw) for t in EVENT_TYPE_MAP.values()]

    async def stop(self) -> None:
        """Unsubscribe, stop the ticker, and fail every open question closed.

        The last of those is the point. A shutdown with a question still on the wire must resolve
        it as a denial rather than leaving a future nobody will ever complete: the gate awaits
        that future, and a turn that outlives this call would otherwise hang on a screen that is
        already gone.
        """
        for handle in self._handles:
            with contextlib.suppress(Exception):
                self.bus.unsubscribe(handle)
        self._handles = []
        self._started = False
        await self._stop_status_ticker()
        for ask in list(self._open_asks.values()):
            self._settle(ask, ASK_FALLBACK_DECISION)
        self._open_asks.clear()
        # A turn-finished push fired moments before shutdown is the one most worth not dropping:
        # it is the notification saying the thing you walked away from is done. Bounded, because
        # a push service that never answers must not be able to hold Ctrl-C.
        await self.flush_push(timeout=push_mod.SHUTDOWN_FLUSH_S)

    # ---------------------------------------------------------------- fan-out

    def attach_client(self) -> _Client:
        """Register a new SSE consumer and return its handle."""
        client = _Client(secrets.token_hex(8))
        self._clients[client.id] = client
        return client

    def detach_client(self, client: _Client) -> None:
        self._clients.pop(client.id, None)

    @property
    def client_count(self) -> int:
        """How many streams are attached. The in-surface presence signal A2's push gating needs,
        and the `Hello` frame's honest answer to "is anybody watching"."""
        return len(self._clients)

    def _emit(self, type_name: str, seq: Optional[int], payload: str) -> None:
        """Put one wire message on every attached client's queue.

        Synchronous and non-blocking on purpose: this runs inside a bus handler, and the bus
        isolates handlers but still awaits them under a 30-second timeout. A slow or absent
        consumer must never be able to slow the publisher down, so a full queue drops the client
        to a resync instead of applying backpressure to the agent loop.
        """
        for client in list(self._clients.values()):
            if client.lagged:
                continue
            try:
                client.queue.put_nowait((type_name, seq, payload))
            except asyncio.QueueFull:
                client.lagged = True
                client.dropped = client.queue.qsize()
                log.warning("web_client_lagged", client=client.id, dropped=client.dropped)
                with contextlib.suppress(asyncio.QueueFull):
                    while not client.queue.empty():
                        client.queue.get_nowait()
                    client.queue.put_nowait(
                        self._wire(Lagged(
                            session_id=self.session_id,
                            from_seq=client.last_seq,
                            dropped=client.dropped,
                        ))
                    )

    @staticmethod
    def _wire(frame: WireFrame) -> tuple[str, Optional[int], str]:
        """Serialize an SSE-only frame.

        The `None` in the middle is load-bearing: **an SSE-only frame carries no `id:` line.**
        The SSE id IS the bus seq and nothing else, so `Last-Event-ID` and the cursor a client
        persists to `localStorage` always name a real, replayable event. A frame that stamped its
        own id would silently poison the resume cursor with a number no replay can honour.
        """
        return (frame.frame_type, None, frame.model_dump_json())

    def push(self, frame: WireFrame) -> None:
        """Send one SSE-only frame to every attached client."""
        type_name, seq, payload = self._wire(frame)
        self._emit(type_name, seq, payload)

    async def _forward_event(self, event: Any) -> None:
        """Every subscribed bus event, verbatim, as the same bytes the session JSONL gets.

        `model_dump_json()` is called once here and the result serves the phone; the bus called
        it separately for disk. Same method, same model, so the two cannot drift — which is the
        whole reason the wire is the bus event stream rather than a hand-written DTO per event.
        A DTO layer is how tool visibility gets lost one field at a time.
        """
        payload = event.model_dump_json()
        seq = getattr(event, "seq", None)
        sid = getattr(event, "session_id", None)
        if sid is not None and seq is not None:
            prev = self._forwarded_max.get(sid)
            if prev is None or seq > prev:
                self._forwarded_max[sid] = seq
        self._emit(type(event).__name__, seq, payload)
        await self._react(event)

    # ---------------------------------------------------------------- push (A2)

    def set_co_tenants(
        self, others: list[Any], rescan: Optional[Callable[[], list[Any]]] = None
    ) -> None:
        """Record the other live sessions on this agent, for `GET /api/health` (WEBCH-29).

        `rescan` re-reads the registry at request time. Without it the answer is frozen at
        bring-up, and the session that started FIRST is exactly the one that can never learn of
        the second — which is backwards, because it is the first session's history the second
        one interleaves with.
        """
        self._co_tenants = list(others or [])
        self._co_tenant_rescan = rescan

    @property
    def co_tenants(self) -> list[Any]:
        """The last known list. `live_co_tenants()` is what a reader should ask for."""
        return self._co_tenants

    def live_co_tenants(self) -> list[Any]:
        """The co-tenants right NOW — blocking, so call it off the event loop.

        Falls back to the snapshot when nothing wired a rescan (a test channel, a driver that
        never announced), which is the pre-WEBCH-29 behaviour and never worse than it.
        """
        if self._co_tenant_rescan is None:
            return self._co_tenants
        try:
            self._co_tenants = list(self._co_tenant_rescan())
        except Exception:  # noqa: BLE001 — health answers even when the registry is unreadable
            log.warning("co_tenant_rescan_failed", exc_info=True)
        return self._co_tenants

    def set_push(self, service: Any) -> None:
        """Attach the Web Push service. Absent, every trigger site below is inert."""
        self._push = service

    def _push_fire(self, message: Any) -> None:
        """Deliver one push WITHOUT waiting for it.

        Scheduled rather than awaited because this runs inside bus-event handling, on the event
        loop that is also generating tokens: a push service that takes ten seconds to answer
        would otherwise pause the turn it is announcing. The task is held in a set because a
        bare `ensure_future` can be garbage-collected mid-flight — the same footgun the harness
        hit once already elsewhere.
        """
        if self._push is None or message is None:
            return
        task = asyncio.ensure_future(self._deliver(message))
        self._push_tasks.add(task)
        task.add_done_callback(self._push_tasks.discard)

    async def _deliver(self, message: Any) -> None:
        try:
            await self._push.deliver(message)
        except Exception:  # noqa: BLE001 — a convenience never takes the turn down with it
            log.warning("web_push_failed", exc_info=True)

    async def flush_push(self, timeout: Optional[float] = None) -> None:
        """Wait for in-flight pushes. Used by `stop()` so a turn-finished push is not dropped on
        the way out, and by tests, which would otherwise race the loop.

        `stop()` passes a bound and that bound is load-bearing: each send carries a ten-second
        socket budget, so a few devices behind a dead network path would hold Ctrl-C for minutes
        — while uvicorn's own graceful-shutdown budget claims two seconds. An undelivered
        notification is worth waiting a moment for and is not worth a hung process.
        """
        while self._push_tasks:
            pending = list(self._push_tasks)
            if timeout is None:
                await asyncio.gather(*pending, return_exceptions=True)
                continue
            done, still_running = await asyncio.wait(pending, timeout=timeout)
            for task in still_running:
                task.cancel()
            self._push_tasks.difference_update(still_running)
            return

    def _push_policy(self) -> Any:
        return None if self._push is None else self._push.policy

    async def _react(self, event: Any) -> None:
        """The few events this channel does something about beyond forwarding them.

        Every turn-boundary reaction is gated on `parent_id is None` — i.e. the ROOT turn. A
        subagent's turn publishes its own `TurnStarted`/`TurnCompleted` (stamped with the parent's
        session id by `_ParentIdBus`), and 45% of real sessions delegate, so without the guard a
        child finishing mid-task set `_turn_running = False`: the instrument cluster went dead for
        the rest of the root turn and the provisional streaming bubble was dropped, while the root
        carried on generating. Found by test, not by reading.
        """
        root = getattr(event, "parent_id", None) is None
        policy = self._push_policy()
        if isinstance(event, TurnStarted):
            if root:
                self._turn_running = True
                self._stream_id = None
                await self._start_status_ticker()
                if policy is not None:
                    policy.turn_started(self.session_id)
        elif isinstance(event, (TurnCompleted, TurnFailed)):
            if root:
                self._turn_running = False
                await self._stop_status_ticker()
                self._close_stream(None)
                if policy is not None:
                    # `duration_seconds` is the harness's OWN measurement of the turn; preferred
                    # over this channel's wall-clock bookkeeping, which cannot see a turn that
                    # began before the page connected.
                    self._push_fire(policy.turn_finished(
                        self.session_id,
                        clients_attached=self.client_count,
                        duration=getattr(event, "duration_seconds", None),
                        summary=(getattr(event, "summary", "") or "")[:PUSH_SUMMARY_CHARS],
                    ))
        elif isinstance(event, PermissionStaged):
            # NOT gated on `root`: a subagent's parked call needs a human exactly as much as the
            # orchestrator's does, and 45% of real sessions delegate.
            if policy is not None:
                pending = getattr(event, "pending", None)
                self._push_fire(policy.needs_you(
                    self.session_id, kind="parked",
                    pending_id=getattr(pending, "id", None),
                    detail=(getattr(pending, "rendering", "") or "")[:PUSH_SUMMARY_CHARS],
                ))
        elif isinstance(event, PermissionResolved):
            # From ANY surface — answering in Discord has to bring this phone's badge down too.
            if policy is not None:
                policy.resolved(self.session_id, pending_id=getattr(event, "pending_id", None))
        elif isinstance(event, Escalation):
            if policy is not None:
                self._push_fire(policy.needs_you(
                    self.session_id, kind="escalation",
                    detail=(getattr(event, "reason", "") or "")[:PUSH_SUMMARY_CHARS],
                ))
        elif isinstance(event, Heartbeat):
            self._context_pct = event.context_utilization_pct
        elif isinstance(event, Action):
            # The hand-off §4.2.1 refuses to make the client infer: the Action that supersedes
            # the provisional bubble names itself, by seq.
            if event.action_type == "llm_response" and event.parent_id is None:
                self._close_stream(event.seq)

    def _close_stream(self, superseded_by_seq: Optional[int]) -> None:
        if self._stream_id is None:
            return
        stream_id, self._stream_id = self._stream_id, None
        self.push(StreamClosed(
            session_id=self.session_id,
            stream_id=stream_id,
            superseded_by_seq=superseded_by_seq,
        ))

    # ---------------------------------------------------------------- streaming

    async def on_token(self, text: str) -> None:
        """The model's answer as it generates — the callback `run_turn(on_token=...)` calls.

        This makes the web the FIRST channel with live answer text in the harness's own UI: the
        terminal passes `on_token=None` and `send_streaming` has zero call sites in `src/`. So
        "parity with the terminal" was never the bar here; the web channel exceeds it.
        """
        if not text:
            return
        if self._stream_id is None:
            self._stream_id = uuid.uuid4().hex
        phase = "writing"
        snap = self._snapshot()
        if snap:
            phase = snap.get("phase") or phase
        self.push(TokenDelta(
            session_id=self.session_id, stream_id=self._stream_id, text=text, phase=phase,
        ))

    async def on_reasoning(self, text: str) -> None:
        """The model's thinking as it generates.

        Reached by widening one wiring line — `llm.on_reasoning` is already a plain attribute
        carrying the raw text, merely gated to `TerminalChannel` at the call site. No provider
        change was needed. Gated on the same toggle the terminal uses so `/reasoning off` means
        what it says on both surfaces; the sink stays wired either way so the toggle works live.
        """
        if not text or not (self.show_reasoning or self.verbose):
            return
        self.push(ReasoningDelta(session_id=self.session_id, text=text))

    def _snapshot(self) -> Optional[dict]:
        src = getattr(self, "progress_source", None)
        if src is None:
            return None
        try:
            return src()
        except Exception:  # noqa: BLE001 — an instrument must never break the render path
            return None

    def status_frame(self) -> StatusTick:
        """Build one instrument frame from the direct callables `_start_async` wires in.

        Each source is guarded exactly as the terminal guards its own: a broken instrument
        reports nothing rather than raising into the path that is trying to show progress.
        """
        snap = self._snapshot() or {}
        tps: Optional[float] = None
        verified = False
        tps_src = getattr(self, "tps_source", None)
        if tps_src is not None:
            try:
                got = tps_src()
                if got:
                    tps, verified = float(got[0]), bool(got[1])
            except Exception:  # noqa: BLE001
                pass
        model: Optional[str] = None
        model_src = getattr(self, "model_source", None)
        if model_src is not None:
            try:
                model = model_src()
            except Exception:  # noqa: BLE001
                pass
        return StatusTick(
            session_id=self.session_id,
            phase=snap.get("phase") or "waiting",
            thinking_tokens=int(snap.get("thinking_tokens") or 0),
            answer_tokens=int(snap.get("answer_tokens") or 0),
            tool_call_tokens=int(snap.get("tool_call_tokens") or 0),
            elapsed=float(snap.get("elapsed") or 0.0),
            silent=float(snap.get("silent") or 0.0),
            tps=tps,
            tps_verified=verified,
            model=model,
            context_pct=self._context_pct,
        )

    async def _start_status_ticker(self) -> None:
        if self._status_task is not None and not self._status_task.done():
            return
        self._status_task = asyncio.ensure_future(self._status_loop())

    async def _stop_status_ticker(self) -> None:
        task, self._status_task = self._status_task, None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def _status_loop(self) -> None:
        """Push one `StatusTick` per interval for as long as a turn runs, and no longer."""
        try:
            while self._turn_running:
                if self._clients:
                    self.push(self.status_frame())
                await asyncio.sleep(STATUS_TICK_INTERVAL_S)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — the instrument never takes the session with it
            log.warning("web_status_ticker_failed", exc_info=True)

    # ---------------------------------------------------------------- bring-up

    def set_bringup(
        self, stage: str, *, detail: str = "", failed: bool = False, elapsed: float = 0.0
    ) -> None:
        """Name the bring-up stage now running, and remember it for a client that connects late."""
        self._bringup = BringUpStage(
            session_id=self.session_id, stage=stage, detail=detail, failed=failed,
            elapsed=elapsed, abortable=self._bringup_abort is not None,
        )
        self.push(self._bringup)

    @property
    def bringup(self) -> Optional[BringUpStage]:
        return self._bringup

    def set_bringup_abort(self, abort: Optional[Callable[[], None]]) -> None:
        """Install the way out of a stuck build (WEBCH-43). Cancelling a TURN is a different state
        machine, and without this there is no way out of a bring-up that wedged."""
        self._bringup_abort = abort

    def abort_bringup(self) -> bool:
        if self._bringup_abort is None:
            return False
        self._bringup_abort()
        return True

    # ---------------------------------------------------------------- input

    async def read_input(self, prompt: str = "") -> str:
        """Block until a client POSTs a message. The REPL's pull side of the push/pull bridge.

        Identical in shape to Discord's: an `asyncio.Queue` fed by the inbound HTTP verb. Taking
        this shape is what earns the web channel slash commands, the input router, the pending
        resolver and `UserMessage` publishing without re-implementing any of them.
        """
        if not self._started:
            raise NotInteractiveError("WebChannel.start() must be called before read_input()")
        return await self._inbound.get()

    def submit(self, text: str) -> None:
        """Queue a user message for the REPL's next `read_input()`.

        This IS `intent=queue`: the classic REPL loop does not read input while a turn runs, so a
        message posted mid-turn simply waits for the turn to end and then runs as the next one.
        Nothing is dropped and nothing interleaves.
        """
        self._inbound.put_nowait(text)

    async def nudge(self, text: str) -> bool:
        """Steer the RUNNING turn at its next step boundary (`intent=nudge`).

        Reaches `AgentLoop.push_user_nudge` through a resolver the REPL installs, for the same
        reason the pending resolver exists: this channel holds no loop handle and should not
        start holding one. Returns False when no handle is installed yet, so the caller can SAY
        so rather than letting a tap do nothing.
        """
        resolver = self._nudge_resolver
        if resolver is None:
            return False
        return await resolver(text, "nudge")

    async def cancel_turn(self) -> bool:
        """Cancel the in-flight turn, then say so on the wire.

        `TurnCancelled` is minted here because the loop publishes NEITHER `TurnCompleted` NOR
        `TurnFailed` on cancel — the REPL prints its own line and nothing reaches the bus. Without
        this frame the phone shows a turn that never ends.
        """
        resolver = self._cancel_resolver
        if resolver is None:
            return False
        cancelled = await resolver()
        if cancelled:
            self._turn_running = False
            await self._stop_status_ticker()
            self._close_stream(None)
            self.push(TurnCancelled(session_id=self.session_id))
        return cancelled

    async def resolve_pending(self, action: str, pending_id: int) -> bool:
        """Answer a PARKED call (`gate.approve`/`gate.deny`) through the REPL's resolver.

        The UI must say "it runs when the model re-issues it", never "ran": `gate.approve`
        records the answer and dispatches nothing — the model re-issues the call after a nudge.
        """
        resolver = self._pending_resolver
        if resolver is None:
            return False
        await resolver(action, pending_id)
        return True

    # ---------------------------------------------------------------- permissions

    async def ask_permission(self, request: Any) -> Any:
        """Put a BLOCKING question on the wire and wait for a human (§5.3).

        No timeout of its own. `PermissionGate` already awaits this under
        `permissions.ask.timeout_s` (or the tool's own timeout) and records a `reject_once` when
        it runs out, so a second deadline here would only be a second place to get it wrong. The
        gate's deadline arrives as a CANCEL, and the ask is closed and announced expired on the
        way out — a question still showing live buttons after it stopped mattering is a lie
        somebody will tap.

        Every exit that is not a human's answer is `reject_once`.
        """
        from localharness.agent.gate_types import Decision

        request_id = uuid.uuid4().hex
        ask = _OpenAsk(request_id, request, self._ask_frame(request_id, request))
        self._open_asks[request_id] = ask
        self.push(ask.frame)
        policy = self._push_policy()
        if policy is not None:
            # A blocking ask is the one that stops the turn dead, so it is the one most worth a
            # buzz — and it is the class most likely to be sitting on a screen nobody is looking
            # at, since the gate holds the loop until somebody answers.
            self._push_fire(policy.needs_you(
                self.session_id, kind="blocking", request_id=request_id,
                detail=(getattr(ask.frame, "text", "") or getattr(request, "display", "")
                        or "")[:PUSH_SUMMARY_CHARS],
            ))
        try:
            kind = await ask.future
            return Decision(kind=kind)
        except asyncio.CancelledError:
            ask.outcome = ASK_FALLBACK_DECISION
            self.push(AskExpired(
                session_id=self.session_id,
                request_id=request_id,
                decision=ASK_FALLBACK_DECISION,
                waited_s=time.monotonic() - ask.asked_at,
            ))
            raise
        except Exception:  # noqa: BLE001 — a render fault denies; it never allows
            log.warning("web_ask_failed", request_id=request_id, exc_info=True)
            return Decision(kind=ASK_FALLBACK_DECISION)
        finally:
            self._open_asks.pop(request_id, None)
            # HERE, not in the two success paths, and that placement is the whole point. The
            # badge counts what is outstanding, and this question stops being outstanding on
            # EVERY exit — answered, expired by the gate's deadline, or failed while rendering.
            # Expiry is not the rare path either: it is the designed one, because a pocket does
            # not answer questions. And no bus-side wiring could cover it — `PermissionResolved`
            # carries no `request_id` — so if this line is not exhaustive, nothing else is.
            if policy is not None:
                policy.resolved(self.session_id, request_id=request_id)

    @staticmethod
    def ask_options(klass: str, grantable: bool) -> list[AskOption]:
        """The buttons a question of this class and grantability gets.

        One derivation, used by the live ask AND by `--fixtures`, so a scripted dialog cannot
        offer a different set from the real thing — which would let a UI be built against buttons
        that never appear in a session.
        """
        if klass == TRUST_KLASS:
            kinds: tuple[str, ...] = ("allow_always", "reject_once")
            names = TRUST_OPTION_NAMES
        elif grantable:
            kinds = ("allow_once", "allow_always", "reject_once", "reject_always")
            names = GRANTABLE_OPTION_NAMES
        else:
            kinds = UNGRANTABLE_KINDS
            names = GRANTABLE_OPTION_NAMES
        return [
            AskOption(
                kind=k,
                name=names.get(k, k),
                # The workspace-trust answer is permanent but writes no grant, so it takes no
                # second tap: the question ITSELF already says it is permanent, and a confirm on
                # a two-button dialog that has already explained itself is friction with no
                # guardrail attached.
                confirm_required=(k in ALWAYS_KINDS and klass != TRUST_KLASS),
            )
            for k in kinds
        ]

    def _ask_frame(self, request_id: str, request: Any) -> BlockingAsk:
        from localharness.tools.capabilities import UNTRUSTED_INGEST

        klass = getattr(request, "klass", "")
        grantable = bool(getattr(request, "grantable", False))
        tool_name = getattr(request, "tool_name", "") or ""
        return BlockingAsk(
            session_id=self.session_id,
            request_id=request_id,
            tool_name=tool_name,
            tool_params=dict(getattr(request, "tool_params", None) or {}),
            klass=klass,
            key=getattr(request, "key", None),
            grantable=grantable,
            reason=sanitize_for_display(getattr(request, "reason", "") or ""),
            display=sanitize_for_display(getattr(request, "display", "") or ""),
            agent_id=getattr(request, "agent_id", None),
            call_id=getattr(request, "call_id", None),
            options_legend=getattr(request, "options_legend", None),
            options=self.ask_options(klass, grantable),
            untrusted_ingest=tool_name in UNTRUSTED_INGEST,
        )

    def open_asks(self) -> list[dict[str, Any]]:
        """Every blocking question still waiting, as STATE.

        This is what makes a blocking ask survive a reconnect: the question is served from
        `GET /api/permissions`, so a client that reconnects, connects fresh, or is a second
        device renders it without ever having received the frame that announced it.
        """
        return [json.loads(a.frame.model_dump_json()) for a in self._open_asks.values()]

    def answer_ask(self, request_id: str, kind: str, confirm: Optional[str] = None) -> dict:
        """Answer a blocking ask. Idempotent, and two-phase for the `_always` kinds.

        **The second tap is enforced HERE, on the server, not in the page.** `allow_always`
        writes a grant to the global `~/.localharness/grants.yaml` keyed by workspace realpath
        that never expires and has no revoke command — a fat-thumbed tap is forever. The
        reference page is explicitly the owner's disposable half, so anything else holding the
        bearer token (a Shortcuts automation, `curl`, a future UI, a bug in this page's state
        machine) would otherwise write a permanent global grant in ONE request. The gate already
        sets this precedent by downgrading an ungrantable request server-side "rather than
        trusted to every channel's UI to get right".
        """
        ask = self._open_asks.get(request_id)
        if ask is None:
            return {"status": "unknown", "detail": "no open question with that id"}
        if ask.outcome is not None:
            return {"status": "already_answered", "decision": ask.outcome}
        if kind not in {o.kind for o in ask.frame.options}:
            return {"status": "invalid", "detail": f"{kind!r} is not offered for this question"}

        if kind in ALWAYS_KINDS and ask.frame.klass != TRUST_KLASS:
            now = time.monotonic()
            held = ask.confirm
            live = held if (held and held[1] == kind and held[2] > now) else None
            if confirm and live and secrets.compare_digest(confirm, live[0]):
                ask.confirm = None
            else:
                # A WRONG or absent token does not rotate a token that is still live: reissuing
                # on every failed attempt would invalidate the one a legitimate client is holding,
                # so an ordinary double-submit on a flaky phone connection would never be able to
                # complete the confirm. Repeating the first tap is idempotent — same token, same
                # deadline — and only an expired or kind-mismatched one is replaced.
                if live is None:
                    live = (secrets.token_urlsafe(16), kind, now + CONFIRM_TOKEN_TTL_S)
                    ask.confirm = live
                return {
                    "status": "confirm_required",
                    "confirm_token": live[0],
                    "expires_in": round(live[2] - now, 3),
                    "detail": "this answer is permanent, global and has no revoke command; "
                              "repeat the request with confirm_token to record it",
                }
        self._settle(ask, kind)
        return {"status": "recorded", "decision": kind}

    def _settle(self, ask: _OpenAsk, kind: str) -> None:
        ask.outcome = kind
        if not ask.future.done():
            ask.future.set_result(kind)
        policy = self._push_policy()
        if policy is not None:
            # An answered question is no longer outstanding, however it was answered — including
            # by the shutdown path below, which settles every open ask as a denial.
            policy.resolved(self.session_id, request_id=ask.request_id)

    def register_fixture_ask(self, frame: BlockingAsk) -> None:
        """Make a `--fixtures` blocking ask genuinely answerable (§8).

        No gate awaits this one — it is scripted, not raised by a real tool call — but it goes
        through the same `_open_asks` map, so `GET /api/permissions` lists it, the two-phase
        `_always` confirm applies to it, and answering it completes the round trip. Without that,
        `--fixtures` would draw a dialog whose buttons do nothing, and a UI author could build a
        modal that looks right and has never once finished its own handshake.

        A fixture that names no options gets the ones a REAL question of that class and
        grantability would get. Making the author restate them would be a second place for the
        button set to be wrong, and a minimal fixture would otherwise draw a dialog with no
        buttons at all.
        """
        if not frame.options:
            frame = frame.model_copy(
                update={"options": self.ask_options(frame.klass, frame.grantable)}
            )
        ask = _OpenAsk(frame.request_id, None, frame)
        self._open_asks[frame.request_id] = ask
        self.push(frame)

    # ---------------------------------------------------------------- trust (§5.6)

    def trust_asker(self) -> Callable[[str], bool]:
        """Bridge the SYNCHRONOUS workspace-trust question onto this event loop.

        `cli/workspace.resolve_workspace_layer` takes a plain `Callable[[str], bool]` and runs on
        a worker thread, so the answer has to come back across the thread boundary —
        `run_coroutine_threadsafe`, exactly as the ACP adapter documents. Skipping this does not
        fail loudly: it silently makes an outside `.localharness/` invisible forever, which is
        precisely the failure the trust dialog was added to prevent.

        Fails CLOSED. A client that goes away mid-question leaves the workspace untrusted and
        re-askable; a trust answer is never inferred from a dropped connection.
        """
        loop = asyncio.get_running_loop()

        def _ask(question: str) -> bool:
            try:
                return asyncio.run_coroutine_threadsafe(self._ask_trust(question), loop).result()
            except (Exception, asyncio.CancelledError):  # noqa: BLE001
                # CancelledError is caught EXPLICITLY: it is a BaseException, so a bare
                # `except Exception` would let a cancelled bridge escape as an exception out of a
                # worker thread instead of as the denial it means.
                log.warning("web_trust_ask_failed", exc_info=True)
                return False

        return _ask

    async def _ask_trust(self, question: str) -> bool:
        """The trust question, drawn with its own two permanent-sounding buttons.

        The ONE ask in the harness with no gate behind it — `resolve_workspace_layer` calls its
        asker directly, so there is no `permissions.ask.timeout_s` to inherit and nothing else
        will ever time this out. It therefore carries its own deadline, and both of its escape
        hatches fail closed:

        * **Nobody attached → don't ask at all.** Asking into an empty room and then waiting is
          how bring-up hangs on a question no screen ever showed.
        * **Attached but silent → deny at the deadline.** The workspace stays untrusted and
          re-askable next time; a trust answer is never inferred from a dropped connection.
        """
        from localharness.agent.gate_types import PermissionRequest

        if not self._clients:
            log.info("web_trust_no_client", detail="no attached client to ask; leaving untrusted")
            return False
        try:
            decision = await asyncio.wait_for(
                self.ask_permission(PermissionRequest(
                    tool_name="workspace",
                    tool_params={"question": question},
                    klass=TRUST_KLASS,
                    key=None,
                    grantable=False,
                    reason=question,
                    display=question,
                )),
                timeout=TRUST_ASK_TIMEOUT_S,
            )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return False
        return bool(getattr(decision, "allowed", False))

    # ---------------------------------------------------------------- gap detection

    def gap_against_log(self, session_id: str, log_max_seq: Optional[int]) -> Optional[GapDetected]:
        """Compare what was forwarded live against what the log can serve (§4.2.3).

        `EventBus._append_jsonl` swallows write failures and `publish()` delivers regardless, so
        an event can be live-visible and permanently absent from the file a reconnecting client
        replays from. The diff-against-the-JSONL test of WEBCH-06 would still PASS while
        describing a client that missed something. This is the check that makes it visible.
        """
        forwarded = self._forwarded_max.get(session_id)
        if forwarded is None:
            return None
        highest = -1 if log_max_seq is None else log_max_seq
        if forwarded <= highest:
            return None
        return GapDetected(
            session_id=session_id,
            from_seq=highest + 1,
            to_seq=forwarded,
            detail="these events were delivered live but are missing from the session log — the "
                   "bus logs a persist failure and delivers anyway, so a disk fault can leave "
                   "this hole. Nothing that happened is lost from the transcript you already "
                   "have; a fresh client cannot replay it.",
        )

    # ---------------------------------------------------------------- prose out

    async def send_message(
        self,
        content: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Prose the harness addressed to the human — slash-command output above all.

        Without this the entire slash-command surface would be invisible on the phone: the REPL
        talks to a person exclusively through `send_message` / `send_error` / `send_renderable`.
        `metadata["style"]` is carried rather than dropped — it is a dead key everywhere else, so
        info-versus-error intent is currently thrown away at the terminal.
        """
        style = "system.info"
        if isinstance(metadata, dict):
            style = str(metadata.get("style") or style)
        self.push(Notice(
            session_id=self.session_id, text=content, style=style, agent_id=agent_id,
        ))

    async def send_error(
        self, error: str, detail: str | None = None, agent_id: str | None = None
    ) -> None:
        self.push(Notice(
            session_id=self.session_id, text=error, style="system.error",
            detail=detail, agent_id=agent_id,
        ))

    async def send_renderable(
        self,
        renderable: Any,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """The `/memory` tree and anything else built as a rich renderable.

        Flattened to text and marked `preformatted`, so the box-drawing survives in a `<pre>`.
        This is the one place the web channel is WORSE than the terminal, and memory legibility is
        this project's north star — a JSON memory endpoint is the first thing after Phase B, not
        vague later work.
        """
        import io

        from rich.console import Console

        buf = io.StringIO()
        Console(file=buf, width=100).print(renderable)
        self.push(Notice(
            session_id=self.session_id,
            text=buf.getvalue().rstrip("\n"),
            style="renderable",
            agent_id=agent_id,
            preformatted=True,
        ))

    async def on_permission_staged(self, event: Any) -> None:
        """A parked call reaches this client as the TYPED event and nothing else.

        The base default would also draw a one-line prose notice through `send_message`, which on
        a typed wire is the same fact twice — once as a `PermissionStaged` the UI can badge and
        queue, once as a sentence it cannot. The event carries the whole `PendingCall`, so
        nothing is lost by declining the prose; it is already forwarded by the raw subscription.
        """
        return None

    async def hello(self, client: _Client, resume_seq: Optional[int]) -> Hello:
        """The first frame on a connection."""
        return Hello(
            session_id=self.session_id,
            agent_id=self.agent_id,
            mode=getattr(self._gate, "mode", None),
            resume_seq=resume_seq,
            turn_in_progress=self._turn_running,
            model_state=self.model_state(),
            session_live=self.session_id is not None,
        )

    def model_state(self) -> str:
        """ready | cold | unreachable | building | unknown — what `/api/health` reports.

        Distinct states matter: "cold" and "unreachable" are different problems with different
        actions, and neither is an indefinite spinner (WEBCH-26).
        """
        if self._model_reachable is False:
            return "unreachable"
        if self._bringup is not None and self._bringup.failed:
            return "unreachable"
        if self.session_id is None:
            return "building" if self._bringup is not None else "cold"
        if self._model_reachable is None:
            return "unknown"
        return "ready"

    async def probe_model(self) -> Optional[bool]:
        """TCP-probe the provider endpoint and cache the answer for `model_state()`.

        A real connect, not a guess: the chat list has to be able to say "cold" or "unreachable"
        BEFORE the owner commits to sending, and the difference between those two is the whole
        value of saying it. Returns None when there is no endpoint to probe yet.
        """
        from urllib.parse import urlparse

        from localharness.provider.client import _probe_reachable

        base = getattr(getattr(self._llm, "config", None), "base_url", None)
        if not base:
            return None
        parsed = urlparse(base)
        if not parsed.hostname:
            return None
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            self._model_reachable = await _probe_reachable(parsed.hostname, port)
        except Exception:  # noqa: BLE001 — an unprobeable endpoint is "unknown", never "ready"
            self._model_reachable = None
        return self._model_reachable

    # --------------------------------------------------- ABC methods NOT on the live path

    async def send_streaming(self, token_stream: Any, agent_id: str | None = None) -> str:
        """Unused. Streaming here is `run_turn(on_token=...)` → `TokenDelta`, not a pull over an
        async iterator; `send_streaming` has zero call sites anywhere in `src/`. Kept because the
        ABC declares it."""
        parts: list[str] = []
        async for piece in token_stream:
            parts.append(piece)
            await self.on_token(piece)
        return "".join(parts)

    async def send_tool_call(
        self, tool_name: str, arguments: dict[str, Any], agent_id: str | None = None
    ) -> None:
        """Unused: `Action` is forwarded raw, because THIS signature has no `tool_call_id`
        parameter and the UI needs it to pair a call with its result (§4.2.2). Kept because the
        ABC declares it."""
        return None

    async def send_tool_result(
        self, tool_name: str, result: str, is_error: bool, agent_id: str | None = None
    ) -> None:
        """Unused, for the same reason as `send_tool_call`. Kept because the ABC declares it."""
        return None
