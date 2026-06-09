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
from pathlib import Path
from typing import Any, AsyncIterator

import structlog

from localharness.channels.base import ChannelAdapter
from localharness.channels.errors import ChannelStartError
from localharness.core.bus import EventBus
from localharness.core.events import Action, Escalation, Heartbeat, Observation, TaskComplete

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
        "ack_emoji": os.environ.get("LOCALHARNESS_DISCORD_ACK", "\U0001f440"),
    }


class DiscordChannel(ChannelAdapter):
    """Discord gateway channel: messages in, agent replies out.

    Push (discord on_message) is bridged to the REPL's pull (read_input) via an asyncio.Queue.
    Turns are processed serially — messages that arrive mid-turn queue up and run in order.
    """

    channel_id = "discord"

    def __init__(self, bus: EventBus, config: dict[str, Any]) -> None:
        super().__init__(bus, config)
        self._token: str = config.get("token") or ""
        self._allow_users: set[str] = {str(u) for u in config.get("allow_users", []) if str(u).strip()}
        self._allow_channels: set[str] = {str(c) for c in config.get("allow_channels", []) if str(c).strip()}
        self._ack_emoji: str = config.get("ack_emoji", "\U0001f440")
        self._client: Any = None
        self._client_task: asyncio.Task | None = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._ready: asyncio.Event = asyncio.Event()
        self._current_msg: Any = None  # discord.Message being answered (reply routing target)
        self._handles: list[Any] = []

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

        self._handles = [
            self.bus.subscribe(Action, self.on_action),
            self.bus.subscribe(Observation, self.on_observation),
            self.bus.subscribe(TaskComplete, self.on_task_complete),
            self.bus.subscribe(Escalation, self.on_escalation),
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
        if is_error:
            await self._send(f"⚠️ {tool_name} failed: {(result or '')[:300]}")

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
