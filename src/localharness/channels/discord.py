"""DiscordChannel — inbound Discord messages as user turns, agent replies posted back.

Mirrors the OpenClaw dispatch pattern: a discord.py gateway client receives messages,
gates them against an allowlist, and queues them. The existing OrchestratorREPL pulls each
message via read_input() (no REPL changes needed), runs a turn, and the agent's TaskComplete
output flows back through on_task_complete -> send_message, posted to the originating channel.

Config dict (built by discord_config_from_env):
    token: bot token (str)
    allow_users: iterable of Discord user IDs permitted to talk to the agent
    allow_channels: optional iterable of channel IDs to restrict to (empty = any visible channel)
    ack_emoji: emoji reacted to each accepted message (default "\U0001f440"); "" to disable
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

import structlog

from localharness.channels.base import ChannelAdapter, sanitize_for_display
from localharness.channels.errors import ChannelStartError
from localharness.core.bus import EventBus
from localharness.core.events import (
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

_DISCORD_LIMIT = 2000  # Discord's hard per-message character cap


def _chunk(text: str, limit: int = _DISCORD_LIMIT) -> list[str]:
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


def discord_config_from_env() -> dict[str, Any]:
    """Assemble Discord channel config from env (token may fall back to the OpenClaw .env).

    Env vars:
        LOCALHARNESS_DISCORD_TOKEN / DISCORD_BOT_TOKEN  — bot token
        LOCALHARNESS_DISCORD_ALLOW     — comma-separated user IDs (required)
        LOCALHARNESS_DISCORD_CHANNELS  — comma-separated channel IDs (optional)
        LOCALHARNESS_DISCORD_ACK       — ack emoji (optional)
    """
    token = os.environ.get("LOCALHARNESS_DISCORD_TOKEN") or os.environ.get("DISCORD_BOT_TOKEN") or ""
    if not token:
        # Reuse the OpenClaw bot token if present (Alex's "steal the bot" path).
        env_file = Path.home() / ".claude" / "channels" / "discord" / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("DISCORD_BOT_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    def _split(name: str) -> list[str]:
        return [p.strip() for p in os.environ.get(name, "").split(",") if p.strip()]

    return {
        "token": token,
        "allow_users": _split("LOCALHARNESS_DISCORD_ALLOW"),
        "allow_channels": _split("LOCALHARNESS_DISCORD_CHANNELS"),
        "ack_emoji": os.environ.get("LOCALHARNESS_DISCORD_ACK", "✅"),
    }


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

The values are the verbs :attr:`DiscordChannel._pending_resolver` is called with, so the emoji
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
"""Posted when a reaction arrives and :attr:`DiscordChannel._pending_resolver` was never
installed (the REPL owns that wiring). The alternative is exactly the D7 failure: the person
taps, nothing happens anywhere, and the message still says a reaction is how you answer."""

PENDING_RAN_DECISIONS: frozenset[str] = frozenset({"allow_once", "allow_always"})
"""The ``PermissionResolved.decision`` values that mean the call may run.

Everything else the event can carry — ``reject_once``, ``reject_always``,
``CANCELLED_RESOLUTION`` — reads as "skipped": from this message's point of view they are one
outcome, the command did not run.
"""

PENDING_TRUNCATION_SUFFIX = "…"
"""Marks a rendering cut down to fit :data:`_DISCORD_LIMIT`; see :func:`pending_notice_body`."""


def pending_notice_body(pending: Any, total: int) -> str:
    """Render the notice, with the command truncated so the whole line fits ONE message.

    The budget is derived, not chosen: whatever :data:`_DISCORD_LIMIT` allows, minus what the
    template costs at these numbers. Chunking is the wrong answer here even though
    :func:`_chunk` exists — a split notice leaves the reactions on a message whose text no longer
    shows the command they answer, and the whole point of the line is that the number a person
    taps is the number they read.
    """
    frame = PENDING_NOTICE_MESSAGE.format(id=pending.id, rendering="", total=total)
    room = _DISCORD_LIMIT - len(frame)
    rendering = sanitize_for_display(pending.rendering)
    if len(rendering) > room:
        rendering = rendering[: room - len(PENDING_TRUNCATION_SUFFIX)] + PENDING_TRUNCATION_SUFFIX
    return PENDING_NOTICE_MESSAGE.format(id=pending.id, rendering=rendering, total=total)


@dataclass
class _PendingNotice:
    """One posted notice, and the reaction waiter watching it."""

    pending: Any
    message: Any
    body: str
    waiter: asyncio.Queue
    task: asyncio.Task | None = None


class DiscordChannel(ChannelAdapter):
    """Discord gateway channel: messages in, agent replies out.

    Push (discord on_message) is bridged to the REPL's pull (read_input) via an asyncio.Queue.
    Turns are processed serially — messages that arrive mid-turn queue up and run in order.

    Parked calls (``auto``'s staging queue) are answered here by reaction. This channel holds no
    gate handle — it never has — so the tap has to be handed back to whoever owns the gate:
    :attr:`_pending_resolver` is that handle, and the REPL installs it
    (``channel._pending_resolver = ...``, calling ``gate.approve``/``gate.deny`` and nudging the
    model). Until it is installed a reaction is logged and the message says so
    (:data:`PENDING_NO_RESOLVER_LINE`) rather than silently doing nothing.
    """

    channel_id = "discord"

    can_ask = True
    """PRD §3.5: Discord renders an ASK as reactions on a message."""

    has_review_surface = False
    """Nothing here shows a diff, so an in-workspace edit asks once per workspace (PRD §3.1
    choice 2, critic finding 11) instead of never."""

    ask_holds_dialog = False
    """PRD §3.5, Discord row: a message nobody reacts to has to expire, so the gate keeps its
    deadline here (`permissions.ask.timeout_s`, else the tool-timeout derivation) and a timeout
    resolves as `reject_once`. Stated explicitly rather than inherited: this is the one channel
    the timeout is FOR."""

    def __init__(self, bus: EventBus, config: dict[str, Any]) -> None:
        super().__init__(bus, config)
        self._token: str = config.get("token") or ""
        self._allow_users: set[str] = {str(u) for u in config.get("allow_users", []) if str(u).strip()}
        self._allow_channels: set[str] = {str(c) for c in config.get("allow_channels", []) if str(c).strip()}
        self._ack_emoji: str = config.get("ack_emoji", "✅")
        self._client: Any = None
        self._client_task: asyncio.Task | None = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._ready: asyncio.Event = asyncio.Event()
        self._current_msg: Any = None  # discord.Message being answered (reply routing target)
        self._handles: list[Any] = []
        # message id -> queue of emoji, for ask_permission. Push (a gateway reaction event) is
        # bridged to pull (an awaiting gate) the same way on_message is bridged to read_input.
        self._reaction_waiters: dict[int, asyncio.Queue] = {}
        # pending id -> the notice posted for it, insertion-ordered (oldest first), for as long
        # as nobody has answered it. Its message is edited in place when the answer arrives.
        self._pending_notices: dict[int, _PendingNotice] = {}
        self._pending_resolver: Callable[[str, int], Awaitable[None]] | None = None
        """How a reaction reaches the gate: ``await resolver(action, pending_id)`` where action is
        one of :data:`PENDING_REACTIONS`' values ("approve"/"deny"). Installed by the REPL, which
        owns the gate and the nudge; None here means reactions cannot be answered yet."""

    async def start(self) -> None:
        try:
            import discord
        except ImportError as e:
            raise ChannelStartError(
                "discord.py not installed — run: uv pip install 'discord.py>=2.3' "
                "(or install the 'dispatch' extra)"
            ) from e
        if not self._token:
            raise ChannelStartError(
                "Discord bot token missing — set LOCALHARNESS_DISCORD_TOKEN or DISCORD_BOT_TOKEN"
            )
        if not self._allow_users:
            raise ChannelStartError(
                "Discord allowlist empty — set LOCALHARNESS_DISCORD_ALLOW to your user id(s); "
                "refusing to listen to everyone"
            )

        intents = discord.Intents.default()
        intents.message_content = True  # privileged intent — must be enabled on the bot
        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_ready() -> None:
            log.info("discord_ready", bot=str(self._client.user),
                     channels=sorted(self._allow_channels) or "any")
            self._ready.set()

        @self._client.event
        async def on_message(msg: Any) -> None:
            if self._client.user is not None and msg.author.id == self._client.user.id:
                return
            if msg.author.bot:
                return
            if str(msg.author.id) not in self._allow_users:
                return
            if self._allow_channels and str(msg.channel.id) not in self._allow_channels:
                return
            if not (msg.content or "").strip():
                return
            await self._queue.put(msg)

        @self._client.event
        async def on_raw_reaction_add(payload: Any) -> None:
            # RAW, not on_reaction_add: the raw event needs no message cache, so a permission
            # question still resolves after a reconnect. The allowlist is the SAME set that
            # gates inbound messages — one definition of "who may drive this session".
            if str(getattr(payload, "user_id", "")) not in self._allow_users:
                return
            waiter = self._reaction_waiters.get(int(getattr(payload, "message_id", 0) or 0))
            if waiter is not None:
                waiter.put_nowait(str(getattr(payload, "emoji", "")))

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
        self._client_task = asyncio.create_task(self._client.start(self._token))
        await self._ready.wait()

    async def stop(self) -> None:
        for h in self._handles:
            if h is not None:
                self.bus.unsubscribe(h)
        self._handles = []
        # Reaction waiters outlive the turn that staged them, so shutdown is the only thing that
        # ends them. Left running they would keep a closed session's tasks alive.
        for notice in list(self._pending_notices.values()):
            self._drop_notice(notice)
        if self._client is not None:
            await self._client.close()
        if self._client_task is not None:
            try:
                await self._client_task
            except (asyncio.CancelledError, Exception):
                pass

    async def read_input(self, prompt: str = "") -> str:
        """Block until the next allowlisted Discord message arrives; return its text."""
        if self._client is None:
            raise ChannelStartError("DiscordChannel.start() must be called before read_input()")
        msg = await self._queue.get()
        self._current_msg = msg
        if self._ack_emoji:
            try:
                await msg.add_reaction(self._ack_emoji)
            except Exception:  # noqa: BLE001 — a failed reaction must never drop the turn
                pass
        return (msg.content or "").strip()

    async def ask_permission(self, request: Any) -> Any:
        """Post the question and wait for an allowlisted reaction (PRD §3.5).

        No timeout of its own: `PermissionGate` already awaits this under
        `permissions.ask.timeout_s` (or the tool's own timeout) and records a `reject_once` when
        it runs out, so a second deadline here would only be a second place to get it wrong. A
        reaction from anyone not on the allowlist is ignored, not counted as an answer.

        The gate's deadline arrives here as a cancel, and the message is annotated on the way out
        (:data:`PERMISSION_TIMEOUT_LINE`) before the cancel is re-raised. That annotation is the
        only thing standing between an expired question and a lie: the 🛑 message and its three
        reactions look identical whether the answer is still wanted or was recorded as a denial
        five minutes ago, and a late tap on ✅ does nothing at all, because the waiter it would
        have reached is gone.
        """
        from localharness.agent.gate_types import Decision

        target = self._current_msg
        if target is None or self._client is None:
            log.warning("discord_permission_no_target", tool=getattr(request, "tool_name", ""))
            return Decision(kind=PERMISSION_DEFAULT_DECISION)

        options = PERMISSION_REACTIONS if request.grantable else PERMISSION_REACTIONS_UNGRANTABLE
        legend = (
            PERMISSION_LEGEND_GRANTABLE if request.grantable else PERMISSION_LEGEND_UNGRANTABLE
        )
        body = PERMISSION_MESSAGE.format(
            display=sanitize_for_display(request.display), legend=legend
        )
        sent = await target.channel.send(body)
        waiter: asyncio.Queue = asyncio.Queue()
        self._reaction_waiters[int(sent.id)] = waiter
        asked_at = time.monotonic()
        try:
            for emoji in options:
                try:
                    await sent.add_reaction(emoji)
                except Exception:  # noqa: BLE001 — a failed reaction must not drop the question
                    log.warning("discord_permission_reaction_failed", emoji=emoji)
            while True:
                kind = options.get(await waiter.get())
                if kind is not None:
                    return Decision(kind=kind)
        except asyncio.CancelledError:
            await self._annotate_expired(sent, body, time.monotonic() - asked_at)
            raise
        finally:
            self._reaction_waiters.pop(int(sent.id), None)

    async def _annotate_expired(self, sent: Any, body: str, waited_s: float) -> None:
        """Mark the 🛑 message as expired and denied (D7).

        An edit rather than a new message, so the annotation is attached to the question it
        answers — someone scrolling back reads one closed exchange, not a question here and a
        verdict thirty lines later. A channel where the bot cannot edit (someone else's message,
        a permissions change) falls back to a reply, and a failure of both is logged and dropped:
        this runs during the gate's cancellation and must never become the reason a turn hangs
        or raises something other than the cancel it was given.

        The seconds are MEASURED, not the configured deadline — what the person actually waited
        is the honest number, and this coroutine is never told what the gate's budget was.
        """
        line = PERMISSION_TIMEOUT_LINE.format(seconds=waited_s)
        try:
            await asyncio.wait_for(
                self._edit_or_reply(sent, body, line), PERMISSION_TIMEOUT_POST_TIMEOUT_S
            )
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001 — never mask the cancel
            log.warning("discord_permission_timeout_note_failed", message_id=getattr(sent, "id", None))

    async def send_pending_notice(self, pending: Any, total: int) -> None:
        """Post the parked call as its own message, with the two answers pre-reacted.

        The one rule this method exists to keep: it must NOT wait for the human. ``auto`` stages
        a call precisely so the agent loop can carry on without it, and this runs on the bus
        handler for that staging — awaiting a reaction here would put the stall back one layer
        down. So the wait lives in a detached task and this returns as soon as the message and
        its reactions are up.

        The reactions ARE added before returning, unlike the wait: they are two bounded REST
        calls, and a notice that tells someone to tap ✅ before ✅ is there is a notice they will
        try to answer by hand.
        """
        target = self._current_msg
        if target is None or self._client is None:
            log.warning("discord_pending_no_target", pending_id=getattr(pending, "id", None))
            return
        body = pending_notice_body(pending, total)
        sent = await target.channel.send(body)
        waiter: asyncio.Queue = asyncio.Queue()
        self._reaction_waiters[int(sent.id)] = waiter
        for emoji in PENDING_REACTIONS:
            try:
                await sent.add_reaction(emoji)
            except Exception:  # noqa: BLE001 — a failed reaction must not drop the notice
                log.warning("discord_pending_reaction_failed", emoji=emoji)
        notice = _PendingNotice(pending=pending, message=sent, body=body, waiter=waiter)
        self._pending_notices[int(pending.id)] = notice
        notice.task = asyncio.create_task(self._await_pending_reaction(notice))

    async def _await_pending_reaction(self, notice: _PendingNotice) -> None:
        """Wait — for as long as it takes — for the first mapped reaction, then resolve it.

        Detached from the turn that staged the call, so it has no deadline: a parked call is
        exactly the one a person is expected to answer later, and the timeout that guards a
        blocking ask (:data:`PERMISSION_TIMEOUT_LINE`) would here delete an answer the queue is
        still holding open. Cancellation is the only other way out, and it comes from
        :meth:`on_permission_resolved` when the same call was answered in text.
        """
        try:
            while True:
                action = PENDING_REACTIONS.get(await notice.waiter.get())
                if action is not None:
                    break
        finally:
            self._reaction_waiters.pop(int(notice.message.id), None)
        resolver = self._pending_resolver
        if resolver is None:
            log.warning("discord_pending_no_resolver", pending_id=notice.pending.id, action=action)
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
        """Close the notice for a call that was answered somewhere else (``/approve 3`` in chat).

        The reaction waiter is cancelled here, and that cancel is the point: without it the ✅
        still sitting on an answered message stays live, and a tap on it minutes later would
        resolve the call a second time — approving something the person already denied by text.

        `PermissionResolved` carries no pending id (it is the same event a blocking ask
        publishes), so the notice is matched on the pairing key that event does carry — session,
        tool, class, grant key — oldest first. HONEST LIMIT: two calls of the same tool and class
        parked at once (two different `rm -rf` paths) are indistinguishable at this event, and the
        older notice takes the verdict. The gate's own queue is unaffected; what can be wrong is
        which of two messages gets annotated.
        """
        notice = self._notice_for(event)
        if notice is None:
            return
        self._drop_notice(notice)
        action = "approve" if event.decision in PENDING_RAN_DECISIONS else "deny"
        await self._close_pending(notice, action)

    def _drop_notice(self, notice: _PendingNotice) -> None:
        """Forget a notice and stop listening for its reactions.

        The waiter is popped HERE rather than left to the task's own ``finally``: a task
        cancelled before it has run once never reaches that ``finally``, and a staged call
        answered in the same breath it was posted is exactly that case. A live waiter for a
        closed notice is a reaction that resolves nothing and pins the queue entry forever.
        """
        self._pending_notices.pop(int(notice.pending.id), None)
        self._reaction_waiters.pop(int(notice.message.id), None)
        if notice.task is not None:
            notice.task.cancel()

    def _notice_for(self, event: PermissionResolved) -> _PendingNotice | None:
        """The notice this resolution closes, by `pending_id` — and only by that.

        A blocking ask (guarded/trusted) resolves through the same event with no pending id,
        and its pairing fields (session, tool, class, key) are the SAME ones a parked call of
        that class carries, so matching on them would let a dialog answer close a notice whose
        call is still queued — a message reading "ran" over an item nobody ran.
        """
        if event.pending_id is None:
            return None
        return next(
            (n for n in self._pending_notices.values() if n.pending.id == event.pending_id), None
        )

    async def _close_pending(self, notice: _PendingNotice, action: str) -> None:
        """Stamp the verdict onto the notice message (:data:`PENDING_RESOLVED_LINES`).

        An edit, not a new message, for the reason :meth:`_annotate_expired` gives: the answer
        belongs to the question, not thirty lines below it. A failure to say it at all is logged
        and dropped — the call is already resolved in the gate, and no channel write may undo
        that or raise into whatever answered it.
        """
        line = PENDING_RESOLVED_LINES[action].format(id=notice.pending.id)
        try:
            await self._edit_or_reply(notice.message, notice.body, line)
        except Exception:  # noqa: BLE001 — the answer is recorded; the annotation is cosmetic
            log.warning("discord_pending_note_failed", message_id=getattr(notice.message, "id", None))

    async def _edit_or_reply(self, sent: Any, body: str, line: str) -> None:
        try:
            await sent.edit(content=f"{body}\n{line}")
        except Exception:  # noqa: BLE001 — fall back to saying it somewhere the person will see
            await sent.reply(line)

    async def _send(self, content: str) -> None:
        msg = self._current_msg
        if msg is None:
            log.warning("discord_send_no_target", preview=(content or "")[:80])
            return
        for chunk in _chunk(content):
            try:
                await msg.channel.send(chunk)
            except Exception as e:  # noqa: BLE001
                log.error("discord_send_failed", error=str(e))

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
        # works around, and empty-error false positives) shouldn't reach the user mid-turn — they
        # read as crashes. The user sees the ack reaction and the final reply; genuinely fatal
        # failures surface via the turn-end path. (Status messaging is a future enhancement.)
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
        # No-op on Discord (no spinner). Typing indicators can come in a later iteration.
        pass
