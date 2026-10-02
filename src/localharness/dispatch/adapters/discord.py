"""The Discord adapter: discord.py's Client behind the ChatAdapter verbs. The only module under src/
that touches `discord`, and only inside connect() — importing this file needs no discord.py.

Holds what is Discord-specific and nothing else: the client and its privileged `message_content`
intent, event wiring (`on_ready`, `on_message`, `on_raw_reaction_add`), the self/bot author filter,
normalising a discord.Message into an InboundMessage, and the send / react / edit / reply / file
verbs. The allow-list, queue, ack (default ✅), chunking and asks live in DispatchChannel."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import structlog

from localharness.channels.errors import ChannelStartError
from localharness.dispatch.channel import InboundMessage, OnMessage, OnReaction

log = structlog.get_logger(__name__)


def _attachment_meta(msg: Any) -> tuple:
    """`(filename, size, content_type)` per uploaded file — metadata only, never the bytes."""
    return tuple((str(getattr(a, "filename", "")), int(getattr(a, "size", 0) or 0),
                  getattr(a, "content_type", None)) for a in (getattr(msg, "attachments", ()) or ()))


class DiscordAdapter:
    platform = "discord"
    title = "Discord"
    message_limit = 2000  # Discord's hard per-message character cap

    def __init__(self, token: str) -> None:
        self._token = token
        self._client: Any = None
        self._client_task: asyncio.Task | None = None
        self._discord: Any = None

    async def connect(self, on_message: OnMessage, on_reaction: OnReaction) -> None:
        """Log in and return once the gateway is ready. Refuses before any network: no discord.py,
        or no token (the token itself is never in a message)."""
        try:
            import discord
        except ImportError as e:
            raise ChannelStartError(
                "discord.py not installed — install the dispatch extra: uv sync --extra dispatch "
                "(or pip install 'localharness[dispatch]')"
            ) from e
        if not self._token:
            raise ChannelStartError(
                "Discord bot token missing — set dispatch.discord.token "
                "(LOCALHARNESS_DISCORD_TOKEN / DISCORD_BOT_TOKEN still work until 0.17.0)"
            )
        self._discord = discord
        deliver = on_message  # the channel's gate; the name is reused by discord's event below
        intents = discord.Intents.default()
        intents.message_content = True  # privileged intent — must be enabled on the bot
        client = self._client = discord.Client(intents=intents)
        ready = asyncio.Event()

        @client.event
        async def on_ready() -> None:
            log.info("discord_ready", bot=str(client.user))
            ready.set()

        @client.event
        async def on_message(msg: Any) -> None:
            if client.user is not None and msg.author.id == client.user.id:
                return
            await deliver(InboundMessage(
                author_id=str(msg.author.id), conversation_id=str(msg.channel.id),
                text=msg.content or "", is_bot=bool(msg.author.bot), handle=msg,
                conversation=msg.channel, attachments=_attachment_meta(msg),
            ))

        @client.event
        async def on_raw_reaction_add(payload: Any) -> None:
            # RAW, not on_reaction_add: the raw event needs no message cache, so a permission
            # question still resolves after a reconnect.
            await on_reaction(str(getattr(payload, "user_id", "")),
                              int(getattr(payload, "message_id", 0) or 0),
                              str(getattr(payload, "emoji", "")))

        self._client_task = asyncio.create_task(client.start(self._token))
        await ready.wait()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
        if self._client_task is not None:
            try:
                await self._client_task
            except (asyncio.CancelledError, Exception):
                pass

    async def send(self, conversation: Any, text: str) -> Any:
        return await conversation.send(text)

    async def react(self, message: Any, emoji: str) -> None:
        await message.add_reaction(emoji)

    async def edit(self, handle: Any, text: str) -> None:
        await handle.edit(content=text)

    async def reply(self, handle: Any, text: str) -> Any:
        return await handle.reply(text)

    async def send_file(self, conversation: Any, path: Any, mime: str) -> Any:
        return await conversation.send(file=self._discord.File(str(path), filename=Path(path).name))
