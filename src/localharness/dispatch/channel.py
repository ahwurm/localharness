"""DispatchChannel — the platform-neutral chat channel core: messages in as user turns, replies out.

Everything any chat platform would do the same way lives here: the inbound queue bridging the
platform's push to the REPL's `read_input` pull, the allow-list (users, optionally conversations;
the same user set gates reactions), the ack reaction at dequeue, reply routing to the originating
conversation, chunking at the adapter's `message_limit`, permission asks answered by reaction, the
parked-call (pending) notices, silent tool activity, `send_error` and stop. Platform verbs live in
one `ChatAdapter` per platform (`dispatch/adapters/`); this module never imports a platform SDK.

Moved from channels/discord.py (Phase 49): `_chunk`, the reaction tables and the permission /
pending constants are copied verbatim (the message limit is now a parameter); the behaviour is
pinned by tests/integration/test_dispatch_discord_golden.py against both implementations.

Config dict: `adapter` (ChatAdapter), `allow` (user ids, required to start), `channels`
(conversation ids; empty = any), `ack` (emoji reacted at dequeue, default ✅; "" disables),
`state_dir` (Path | None; read by the outbound attachment path).
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Awaitable, Callable, Protocol

import structlog

from localharness.channels.base import ChannelAdapter, sanitize_for_display
from localharness.channels.errors import ChannelStartError
from localharness.core.artifacts import artifact_root
from localharness.core.bus import EventBus
from localharness.core.content import image_label
from localharness.core.events import (
    ARTIFACT_MIMES,
    Action,
    Escalation,
    Heartbeat,
    Observation,
    ParseFailed,
    PermissionResolved,
    PermissionStaged,
    TaskComplete,
    TurnFailed,
)

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class InboundMessage:
    """One platform message, normalised by the adapter. `handle` is the platform message (the ack
    and reply target); `conversation` is where replies are sent. `attachments` is metadata only.
    `load_images`, when the platform message carries pictures, fetches them as
    `(image parts, note lines)`; the allow gate awaits it only after admitting the author and
    conversation, then stores the parts in `images` and appends each note (a picture that could
    not be read, named) to `text`. A message with no text but an image is a turn."""

    author_id: str
    conversation_id: str
    text: str
    is_bot: bool
    handle: Any
    conversation: Any
    attachments: tuple = ()
    images: list = field(default_factory=list)
    load_images: Callable[[], Awaitable[tuple[list, list]]] | None = None


OnMessage = Callable[[InboundMessage], Awaitable[None]]
OnReaction = Callable[[str, int, str], Awaitable[None]]


class ChatAdapter(Protocol):
    """The platform verbs. `platform` becomes the channel's `channel_id` (history rows, trust
    records and ask-rate keys carry it), `title` is the human name in the start banner."""

    platform: str
    title: str
    message_limit: int

    async def connect(self, on_message: OnMessage, on_reaction: OnReaction) -> None: ...
    async def close(self) -> None: ...
    async def send(self, conversation: Any, text: str) -> Any: ...
    async def react(self, message: Any, emoji: str) -> None: ...
    async def edit(self, handle: Any, text: str) -> None: ...
    async def reply(self, handle: Any, text: str) -> Any: ...
    async def send_file(self, conversation: Any, path: Any, mime: str) -> Any: ...


def _chunk(text: str, limit: int) -> list[str]:
    """Split text into <=limit pieces, preferring newline then space boundaries."""
    text = text or ""
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks



PERMISSION_REACTIONS: dict[str, str] = {
    "✅": "allow_once",
    "♾️": "allow_always",
    "❌": "reject_once",
}
"""PRD §3.5: "message with the reactions from the allowlisted user".

The PRD writes the middle option as a doubled check; a Discord reaction is ONE emoji, so
"always here" is the infinity sign — forever, unmistakable next to the single check, and not a
near-twin of it the way a boxed check would be. There is no "never here" option: the PRD's
Discord row lists three reactions, and a durable deny written by a mis-tap in a chat window is
the one answer that cannot be undone from a prompt.
"""

PERMISSION_REACTIONS_UNGRANTABLE: dict[str, str] = {
    "✅": "allow_once",
    "❌": "reject_once",
}
"""PRD §3.5: an ungrantable request offers only the `_once` pair — it asks every time by
construction, so an "always" reaction would be a lie."""

PERMISSION_ALWAYS_CONFIRM = (
    "♾️ is permanent: it is remembered for this workspace and cannot be undone from chat. "
    "Tap ✅ on THIS message to confirm it, or ✅ on the question for just this once.")
"""The second tap a permanent grant takes, as on the phone (the mobile channel's server-checked
confirm). Nobody can add the same reaction twice, so the confirm is its own message, pre-reacted
✅ — and the bot's own ✅ answers nothing: the adapter drops every reaction the bot makes. Until it
is tapped the question stays open as it was: ✅ on the question is once, ❌ is no, a second ♾️
changes nothing, and the gate's deadline still ends the whole ask."""

PERMISSION_CONFIRM_REACTION = "✅"
"""The one reaction that confirms :data:`PERMISSION_ALWAYS_CONFIRM`."""

PERMISSION_DEFAULT_DECISION = "reject_once"
"""Fail closed when there is nowhere to post the question, and what the gate records when the
wait runs out (PRD §3.5 "deny on timeout")."""

PERMISSION_TIMEOUT_LINE = "⌛ No answer within {seconds:.0f}s — **denied**. Ask again to retry."
"""What the 🛑 message says once the gate's deadline has passed (D7).

Until this, an expired question stayed on screen exactly as it looked when it was asked: three
reactions, no answer, no sign it had stopped mattering. Someone coming back to their phone ten
minutes later tapped ✅ and nothing happened — the waiter was gone, so the tap was a silent
no-op, and the only visible evidence pointed the other way. The line closes the message: the
answer is recorded, the call was denied, and the way to change that is to ask again."""

PERMISSION_TIMEOUT_POST_TIMEOUT_S = 10.0
"""How long the expiry annotation may take before it is abandoned.

It is posted while the gate is already cancelling this coroutine, so it must not be able to
hold the turn open: a wedged gateway costs the annotation, never the session. Ten seconds is
Discord's own REST timeout order — long enough that an ordinary edit always lands."""

PERMISSION_MESSAGE = "🛑 **Permission needed**\n`{display}`\n{legend}"
PERMISSION_LEGEND_GRANTABLE = "✅ allow once  ·  ♾️ always in this workspace  ·  ❌ no"
PERMISSION_LEGEND_UNGRANTABLE = (
    "✅ allow once  ·  ❌ no    (this one asks every time — it cannot be remembered)"
)
"""The message body. Written out rather than generated from the reaction maps because it is the
sentence someone reads on a phone, and the wording is the design."""


PENDING_REACTIONS: dict[str, str] = {"✅": "approve", "❌": "deny"}
"""The two answers a PARKED call takes (owner ruling 2026-09-12, the staging queue).

The same ✅/❌ pair a blocking ask offers, and deliberately NOT :data:`PERMISSION_REACTIONS`'
♾️: ``auto`` stages only the ungrantable tier, an approval there buys exactly one re-run of the
one command a human read (``gate.APPROVED_ONCE_REASON``) and nothing durable is ever written, so
an "always here" reaction would promise something this queue cannot keep.

The values are the verbs :attr:`DispatchChannel._pending_resolver` is called with, so the emoji
map and the resolver contract cannot drift apart.
"""

PENDING_NOTICE_MESSAGE = (
    "⏸ **needs you**  #{id}  `{rendering}`   ({total} pending) — "
    "react ✅ to run it, ❌ to skip, or reply `/approve {id}` · `/deny {id}`"
)
"""One line, the Discord shape of :data:`~localharness.channels.base.PENDING_NOTICE_LINE`.

Both ways to answer are spelled out in the line itself. A reaction is the fast one on a phone,
but a reaction is lost the moment the bot cannot see the message (a reconnect that misses the
raw event, a channel the bot loses reaction permission in), and the text commands go through the
REPL's own slash router — so the notice never rests on the reaction path having worked.
"""

PENDING_RESOLVED_LINES: dict[str, str] = {
    "approve": "✅ ran #{id} — approved; the model re-issues the call.",
    "deny": "❌ skipped #{id} — denied.",
}
"""What the notice becomes once it has been answered, whichever surface answered it.

Same reasoning as :data:`PERMISSION_TIMEOUT_LINE`: a notice still showing two live reactions
after the call was answered is a lie somebody will tap, and the tap is a silent no-op because
the waiter behind it is gone. "ran" is the approval's honest word only because the approval is
spent on a re-run the MODEL issues — the gate dispatches nothing itself.
"""

PENDING_NO_RESOLVER_LINE = (
    "⚠️ nothing in this session is wired to reactions — answer with `/approve {id}` or `/deny {id}`"
)
"""Posted when a reaction arrives and :attr:`DispatchChannel._pending_resolver` was never
installed (the REPL owns that wiring). The alternative is exactly the D7 failure: the person
taps, nothing happens anywhere, and the message still says a reaction is how you answer."""

PENDING_RAN_DECISIONS: frozenset[str] = frozenset({"allow_once", "allow_always"})
"""The ``PermissionResolved.decision`` values that mean the call may run.

Everything else the event can carry — ``reject_once``, ``reject_always``,
``CANCELLED_RESOLUTION`` — reads as "skipped": from this message's point of view they are one
outcome, the command did not run.
"""

PENDING_TRUNCATION_SUFFIX = "…"
"""Marks a rendering cut down to fit the adapter's ``message_limit``; see :func:`pending_notice_body`."""


def pending_notice_body(pending: Any, total: int, limit: int) -> str:
    """Render the notice, with the command truncated so the whole line fits ONE message.

    The budget is derived, not chosen: whatever ``limit`` (the adapter's ``message_limit``) allows, minus what the
    template costs at these numbers. Chunking is the wrong answer here even though
    :func:`_chunk` exists — a split notice leaves the reactions on a message whose text no longer
    shows the command they answer, and the whole point of the line is that the number a person
    taps is the number they read.
    """
    frame = PENDING_NOTICE_MESSAGE.format(id=pending.id, rendering="", total=total)
    room = limit - len(frame)
    rendering = sanitize_for_display(pending.rendering)
    if len(rendering) > room:
        rendering = rendering[: room - len(PENDING_TRUNCATION_SUFFIX)] + PENDING_TRUNCATION_SUFFIX
    return PENDING_NOTICE_MESSAGE.format(id=pending.id, rendering=rendering, total=total)


class _ConfirmTaps:
    """The always-confirm message's reaction waiter: it hands each tap to the question's own queue,
    marked as a confirm tap, so one loop reads both messages in the order the taps arrived."""

    def __init__(self, question: asyncio.Queue) -> None:
        self._question = question

    def put_nowait(self, emoji: str) -> None:
        self._question.put_nowait(("confirm", emoji))


@dataclass
class _PendingNotice:
    """One posted notice, and the reaction waiter watching it."""

    pending: Any
    message: Any
    body: str
    waiter: asyncio.Queue
    task: asyncio.Task | None = None



class DispatchChannel(ChannelAdapter):
    """A chat platform as a channel: allowlisted messages in, agent replies out.

    Push (the adapter's on_message) is bridged to the REPL's pull (read_input) via an
    asyncio.Queue. Turns are processed serially — messages that arrive mid-turn queue up and run in
    order.

    Parked calls (``auto``'s staging queue) are answered here by reaction. This channel holds no
    gate handle, so the tap is handed back to whoever owns the gate: :attr:`_pending_resolver` is
    that handle, and the REPL installs it (``channel._pending_resolver = ...``). Until it is
    installed a reaction is logged and the message says so (:data:`PENDING_NO_RESOLVER_LINE`).
    """

    can_ask = True
    """PRD §3.5: a chat platform renders an ASK as reactions on a message."""

    has_review_surface = False
    """Nothing here shows a diff, so an in-workspace edit asks once per workspace (PRD §3.1
    choice 2, critic finding 11) instead of never."""

    ask_holds_dialog = False
    """PRD §3.5, Discord row: a message nobody reacts to has to expire, so the gate keeps its
    deadline here (`permissions.ask.timeout_s`, else the tool-timeout derivation) and a timeout
    resolves as `reject_once`."""

    bare_mode_command = True
    """PRD §3.4: no slash convention in chat, so a bare `mode <name>` is the mode command."""

    def __init__(self, bus: EventBus, config: dict[str, Any]) -> None:
        super().__init__(bus, config)
        self._adapter: ChatAdapter = config["adapter"]
        self.channel_id = self._adapter.platform
        self.start_banner = f"Dispatch mode: {self._adapter.title} — listening for allowlisted messages."
        self._allow: set[str] = {str(u) for u in config.get("allow", ()) if str(u).strip()}
        self._channels: set[str] = {str(c) for c in config.get("channels", ()) if str(c).strip()}
        self._ack: str = config.get("ack", "✅")
        self._state_dir = config.get("state_dir")
        self._queue: asyncio.Queue = asyncio.Queue()
        self._ready: asyncio.Event = asyncio.Event()
        self._images: list[dict] = []  # parts of the message read_input last returned, until taken
        self._current_msg: InboundMessage | None = None  # the message being answered (reply target)
        self._handles: list[Any] = []
        # message id -> queue of emoji, for ask_permission and pending notices: push (a reaction
        # event) bridged to pull (an awaiting gate) the same way on_message is bridged to read_input.
        self._reaction_waiters: dict[int, asyncio.Queue | _ConfirmTaps] = {}
        # pending id -> the notice posted for it, insertion-ordered (oldest first), for as long
        # as nobody has answered it. Its message is edited in place when the answer arrives.
        self._pending_notices: dict[int, _PendingNotice] = {}
        self._pending_resolver: Callable[[str, int], Awaitable[None]] | None = None
        """``await resolver(action, pending_id)``, action one of :data:`PENDING_REACTIONS`' values.
        Installed by the REPL, which owns the gate and the nudge; None = reactions cannot answer."""

    async def start(self) -> None:
        if not self._allow:
            raise ChannelStartError(
                f"{self._adapter.title} allowlist empty — set dispatch.{self.channel_id}.allow to "
                f"your user id(s) (LOCALHARNESS_{self.channel_id.upper()}_ALLOW still works until "
                f"0.17.0); refusing to listen to everyone"
            )
        self._handles = [
            self.bus.subscribe(Action, self.on_action),
            self.bus.subscribe(Observation, self.on_observation),
            self.bus.subscribe(TaskComplete, self.on_task_complete),
            self.bus.subscribe(TurnFailed, self.on_turn_failed),
            self.bus.subscribe(Escalation, self.on_escalation),
            self.bus.subscribe(ParseFailed, self.on_parse_failed),
            self.bus.subscribe(Heartbeat, self.on_heartbeat),
            # The only way this channel hears about the staging queue: the loop holds no channel
            # handle, so a parked call and its answer both travel as their own events.
            self.bus.subscribe(PermissionStaged, self.on_permission_staged),
            self.bus.subscribe(PermissionResolved, self.on_permission_resolved),
        ]
        try:
            await self._adapter.connect(self._on_message, self._on_reaction)
        except BaseException:
            self._unsubscribe()
            raise
        self._ready.set()

    async def _on_message(self, msg: InboundMessage) -> None:
        """The allow gate, in today's order: bot, user, conversation, then (admitted only) the
        image fetch, then empty — no text and no image."""
        if msg.is_bot:
            return
        if msg.author_id not in self._allow:
            return
        if self._channels and msg.conversation_id not in self._channels:
            return
        if msg.load_images is not None:
            images, notes = await msg.load_images()
            msg = replace(msg, text="\n".join(t for t in (msg.text or "", *notes) if t),
                          images=list(images), load_images=None)
        if not (msg.text or "").strip() and not msg.images:
            return
        await self._queue.put(msg)

    async def _on_reaction(self, user_id: str, message_id: int, emoji: str) -> None:
        # The allowlist is the SAME set that gates inbound messages — one definition of "who may
        # drive this session".
        if str(user_id) not in self._allow:
            return
        waiter = self._reaction_waiters.get(int(message_id or 0))
        if waiter is not None:
            waiter.put_nowait(str(emoji))

    def _unsubscribe(self) -> None:
        for h in self._handles:
            if h is not None:
                self.bus.unsubscribe(h)
        self._handles = []

    async def stop(self) -> None:
        self._unsubscribe()
        # Reaction waiters outlive the turn that staged them, so shutdown is the only thing that
        # ends them. Left running they would keep a closed session's tasks alive.
        for notice in list(self._pending_notices.values()):
            self._drop_notice(notice)
        await self._adapter.close()

    async def read_input(self, prompt: str = "") -> str:
        """Block until the next allowlisted message arrives; ack it; return its text."""
        if not self._ready.is_set():
            raise ChannelStartError("DispatchChannel.start() must be called before read_input()")
        msg = await self._queue.get()
        self._current_msg = msg
        self._images = list(msg.images)
        if self._ack:
            try:
                await self._adapter.react(msg.handle, self._ack)
            except Exception:  # noqa: BLE001 — a failed reaction must never drop the turn
                pass
        return (msg.text or "").strip() or " ".join(image_label(p) for p in msg.images)

    def take_images(self) -> list[dict]:
        images, self._images = self._images, []
        return images

    async def ask_permission(self, request: Any) -> Any:
        """Post the question and wait for an allowlisted reaction (PRD §3.5).

        No timeout of its own: `PermissionGate` awaits this under its deadline and records a
        `reject_once` when it runs out. The deadline arrives here as a cancel, and the message is
        annotated on the way out (:data:`PERMISSION_TIMEOUT_LINE`) before the cancel is re-raised,
        so an expired question never looks live. A reaction from anyone not on the allowlist is
        ignored, not counted as an answer. ♾️ is not an answer on its own: it posts
        :data:`PERMISSION_ALWAYS_CONFIRM`, and only ✅ on that message records "always".
        """
        from localharness.agent.gate_types import Decision

        target = self._current_msg
        if target is None or not self._ready.is_set():
            log.warning(f"{self.channel_id}_permission_no_target", tool=getattr(request, "tool_name", ""))
            return Decision(kind=PERMISSION_DEFAULT_DECISION)

        options = PERMISSION_REACTIONS if request.grantable else PERMISSION_REACTIONS_UNGRANTABLE
        legend = (
            PERMISSION_LEGEND_GRANTABLE if request.grantable else PERMISSION_LEGEND_UNGRANTABLE
        )
        body = PERMISSION_MESSAGE.format(
            display=sanitize_for_display(request.display), legend=legend
        )
        sent = await self._adapter.send(target.conversation, body)
        waiter: asyncio.Queue = asyncio.Queue()
        self._reaction_waiters[int(sent.id)] = waiter
        confirm: Any = None  # the always-confirm message, once ♾️ was tapped
        asked_at = time.monotonic()
        try:
            for emoji in options:
                try:
                    await self._adapter.react(sent, emoji)
                except Exception:  # noqa: BLE001 — a failed reaction must not drop the question
                    log.warning(f"{self.channel_id}_permission_reaction_failed", emoji=emoji)
            while True:
                tap = await waiter.get()
                if isinstance(tap, tuple):  # on the confirm message: only its ✅ means anything
                    if tap[1] == PERMISSION_CONFIRM_REACTION:
                        return Decision(kind="allow_always")
                    continue
                kind = options.get(tap)
                if kind == "allow_always":
                    if confirm is None:
                        confirm = await self._send_always_confirm(target)
                        if confirm is not None:  # its waiter first, so no quick tap is lost
                            self._reaction_waiters[int(confirm.id)] = _ConfirmTaps(waiter)
                            try:
                                await self._adapter.react(confirm, PERMISSION_CONFIRM_REACTION)
                            except Exception:  # noqa: BLE001 — the message names the reaction
                                log.warning(f"{self.channel_id}_permission_reaction_failed",
                                            emoji=PERMISSION_CONFIRM_REACTION)
                    continue
                if kind is not None:
                    return Decision(kind=kind)
        except asyncio.CancelledError:
            await self._annotate_expired(sent, body, time.monotonic() - asked_at)
            raise
        finally:
            self._reaction_waiters.pop(int(sent.id), None)
            if confirm is not None:
                self._reaction_waiters.pop(int(confirm.id), None)

    async def _send_always_confirm(self, target: InboundMessage) -> Any:
        """Post :data:`PERMISSION_ALWAYS_CONFIRM` beside the question; None when the post fails,
        which leaves the question open as it was (once and no still work; the next ♾️ tries
        again). The caller registers the waiter: the moment the message exists, it owns it."""
        try:
            return await self._adapter.send(target.conversation, PERMISSION_ALWAYS_CONFIRM)
        except Exception:  # noqa: BLE001 — a failed post must not drop the question
            log.warning(f"{self.channel_id}_permission_confirm_failed")
            return None

    async def _annotate_expired(self, sent: Any, body: str, waited_s: float) -> None:
        """Mark the 🛑 message as expired and denied (D7): an edit, falling back to a reply; a
        failure of both is logged and dropped (this runs during the gate's cancellation). The
        seconds are MEASURED, not the configured deadline."""
        line = PERMISSION_TIMEOUT_LINE.format(seconds=waited_s)
        try:
            await asyncio.wait_for(
                self._edit_or_reply(sent, body, line), PERMISSION_TIMEOUT_POST_TIMEOUT_S
            )
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001 — never mask the cancel
            log.warning(f"{self.channel_id}_permission_timeout_note_failed",
                        message_id=getattr(sent, "id", None))

    async def send_pending_notice(self, pending: Any, total: int) -> None:
        """Post the parked call as its own message, with the two answers pre-reacted, and return
        WITHOUT waiting for the human (``auto`` parks a call so the loop can carry on): the wait
        lives in a detached task. The reactions are added before returning — a notice telling
        someone to tap ✅ before ✅ is there is one they will try to answer by hand."""
        target = self._current_msg
        if target is None or not self._ready.is_set():
            log.warning(f"{self.channel_id}_pending_no_target", pending_id=getattr(pending, "id", None))
            return
        body = pending_notice_body(pending, total, self._adapter.message_limit)
        sent = await self._adapter.send(target.conversation, body)
        waiter: asyncio.Queue = asyncio.Queue()
        self._reaction_waiters[int(sent.id)] = waiter
        for emoji in PENDING_REACTIONS:
            try:
                await self._adapter.react(sent, emoji)
            except Exception:  # noqa: BLE001 — a failed reaction must not drop the notice
                log.warning(f"{self.channel_id}_pending_reaction_failed", emoji=emoji)
        notice = _PendingNotice(pending=pending, message=sent, body=body, waiter=waiter)
        self._pending_notices[int(pending.id)] = notice
        notice.task = asyncio.create_task(self._await_pending_reaction(notice))

    async def _await_pending_reaction(self, notice: _PendingNotice) -> None:
        """Wait — with no deadline: a parked call is the one a person answers later — for the first
        mapped reaction, then resolve it. Cancelled by :meth:`on_permission_resolved` when the same
        call was answered in text."""
        try:
            while True:
                action = PENDING_REACTIONS.get(await notice.waiter.get())
                if action is not None:
                    break
        finally:
            self._reaction_waiters.pop(int(notice.message.id), None)
        resolver = self._pending_resolver
        if resolver is None:
            log.warning(f"{self.channel_id}_pending_no_resolver", pending_id=notice.pending.id,
                        action=action)
            # The notice stays registered: the human still has the text commands, and when they
            # use them the resolution event edits this same message.
            await self._edit_or_reply(
                notice.message, notice.body, PENDING_NO_RESOLVER_LINE.format(id=notice.pending.id)
            )
            return
        self._pending_notices.pop(int(notice.pending.id), None)
        await resolver(action, int(notice.pending.id))
        await self._close_pending(notice, action)

    async def on_permission_resolved(self, event: PermissionResolved) -> None:
        """Close the notice for a call answered somewhere else (``/approve 3`` in chat). Cancelling
        the reaction waiter is the point: a live ✅ on an answered message would resolve it twice."""
        notice = self._notice_for(event)
        if notice is None:
            return
        self._drop_notice(notice)
        action = "approve" if event.decision in PENDING_RAN_DECISIONS else "deny"
        await self._close_pending(notice, action)

    def _drop_notice(self, notice: _PendingNotice) -> None:
        """Forget a notice and stop listening for its reactions. The waiter is popped HERE: a task
        cancelled before it has run once never reaches its own ``finally``."""
        self._pending_notices.pop(int(notice.pending.id), None)
        self._reaction_waiters.pop(int(notice.message.id), None)
        if notice.task is not None:
            notice.task.cancel()

    def _notice_for(self, event: PermissionResolved) -> _PendingNotice | None:
        """The notice this resolution closes, by `pending_id` — and only by that: a blocking ask
        resolves through the same event with no pending id and the same pairing fields."""
        if event.pending_id is None:
            return None
        return next(
            (n for n in self._pending_notices.values() if n.pending.id == event.pending_id), None
        )

    async def _close_pending(self, notice: _PendingNotice, action: str) -> None:
        """Stamp the verdict onto the notice message (:data:`PENDING_RESOLVED_LINES`); a failure is
        logged and dropped — the call is already resolved in the gate."""
        line = PENDING_RESOLVED_LINES[action].format(id=notice.pending.id)
        try:
            await self._edit_or_reply(notice.message, notice.body, line)
        except Exception:  # noqa: BLE001 — the answer is recorded; the annotation is cosmetic
            log.warning(f"{self.channel_id}_pending_note_failed",
                        message_id=getattr(notice.message, "id", None))

    async def _edit_or_reply(self, sent: Any, body: str, line: str) -> None:
        try:
            await self._adapter.edit(sent, f"{body}\n{line}")
        except Exception:  # noqa: BLE001 — fall back to saying it somewhere the person will see
            await self._adapter.reply(sent, line)

    async def _send(self, content: str) -> None:
        msg = self._current_msg
        if msg is None:
            log.warning(f"{self.channel_id}_send_no_target", preview=(content or "")[:80])
            return
        for chunk in _chunk(content, self._adapter.message_limit):
            try:
                await self._adapter.send(msg.conversation, chunk)
            except Exception as e:  # noqa: BLE001
                log.error(f"{self.channel_id}_send_failed", error=str(e))

    async def send_message(
        self,
        content: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self._send(content)

    async def send_streaming(
        self,
        token_stream: AsyncIterator[str],
        agent_id: str | None = None,
    ) -> str:
        """v1: assemble tokens and post once (no live message editing). Returns full text."""
        full = ""
        async for tok in token_stream:
            full += tok
        await self._send(full)
        return full

    async def send_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        agent_id: str | None = None,
    ) -> None:
        # Keep the channel clean — tool activity stays silent in v1; the final summary is the reply.
        pass

    async def send_tool_result(
        self,
        tool_name: str,
        result: str,
        is_error: bool,
        agent_id: str | None = None,
    ) -> None:
        # Intentionally silent: intermediate tool results (incl. recoverable failures the agent
        # works around) read as crashes mid-turn. The user sees the ack and the final reply.
        return

    async def send_error(
        self,
        error: str,
        detail: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        text = f"❌ {error}"
        if detail:
            text += f"\n{detail[:600]}"
        await self._send(text)

    async def on_heartbeat(self, event: Heartbeat) -> None:
        # No spinner in chat. Typing indicators can come in a later iteration.
        pass

    async def on_observation(self, event: Observation) -> None:
        """Today's tool-result handling, then a typed artifact (G3) posted as a file reply to the
        conversation being answered. The file is resolved only as `<id><suffix>` under core's
        `artifact_root(state_dir, ref.plugin)` and must be one regular, non-symlink file with an
        allowlisted mime; anything else is logged and nothing is sent."""
        await super().on_observation(event)
        ref, msg = event.artifact, self._current_msg
        if ref is None or msg is None:
            return
        suffix = ARTIFACT_MIMES.get(ref.mime)
        path = (artifact_root(self._state_dir, ref.plugin) / f"{ref.id}{suffix}"
                if self._state_dir is not None and suffix else None)
        try:
            ok = path is not None and not path.is_symlink() and path.is_file()
        except OSError:
            ok = False
        if not ok:
            log.warning(f"{self.channel_id}_artifact_skipped", artifact=ref.id, mime=ref.mime,
                        state_dir=self._state_dir is not None)
            return
        try:
            await self._adapter.send_file(msg.conversation, path, ref.mime)
        except Exception as e:  # noqa: BLE001 — a failed upload must never end the turn
            log.error(f"{self.channel_id}_send_file_failed", error=str(e))
