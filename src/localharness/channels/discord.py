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
from pathlib import Path
from typing import Any, AsyncIterator

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


class DiscordChannel(ChannelAdapter):
    """Discord gateway channel: messages in, agent replies out.

    Push (discord on_message) is bridged to the REPL's pull (read_input) via an asyncio.Queue.
    Turns are processed serially — messages that arrive mid-turn queue up and run in order.
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
        ]
        self._client_task = asyncio.create_task(self._client.start(self._token))
        await self._ready.wait()

    async def stop(self) -> None:
        for h in self._handles:
            if h is not None:
                self.bus.unsubscribe(h)
        self._handles = []
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
