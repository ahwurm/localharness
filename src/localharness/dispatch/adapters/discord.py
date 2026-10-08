"""The Discord adapter: discord.py's Client behind the ChatAdapter verbs. The only module under src/
that touches `discord`, and only inside connect() — importing this file needs no discord.py.

Holds what is Discord-specific and nothing else: the client and its privileged `message_content`
intent, its mention policy (nobody pinged but the person replied to), event wiring (`on_ready`,
`on_message`, `on_raw_reaction_add`), the self/bot author filter and the self reaction filter,
normalising a discord.Message into an InboundMessage, and the send / react / edit / reply / file
verbs (text sent with every masked link's address shown). The allow-list, queue, ack (default ✅),
chunking and asks live in DispatchChannel."""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import structlog

from localharness.channels.errors import ChannelStartError
from localharness.core.content import MAX_IMAGE_BYTES, ImageError, image_part
from localharness.dispatch.channel import InboundMessage, OnMessage, OnReaction

log = structlog.get_logger(__name__)


TOKEN_MISSING = (
    "Discord bot token missing — run `localharness plugins enable dispatch` to set "
    "dispatch.discord.token (~/.claude/channels/discord/.env is no longer read)")
"""The one line a start with no token gets. The last clause is for the person whose bot just
stopped starting: it borrowed Claude Code's token file, which belongs to another program."""

_MASKED_LINK = re.compile(
    r"\[([^\[\]\n]+)\]\(\s*<?(https?://(?:[^()<>\s]|\([^()<>\s]*\))+)>?\s*\)", re.IGNORECASE)
"""`[text](url)`, `[text](<url>)` and spaced forms, a URL with one level of parentheses kept whole
(a Wikipedia link)."""

_LINK_SEAM = re.compile(r"\]\((?=\s*<?https?://)", re.IGNORECASE)
"""What is left of a masked link :data:`_MASKED_LINK` did not take whole (nested brackets or a
newline in its text): breaking the `](` seam is enough for Discord to show it as plain text."""


def plain_links(text: str) -> str:
    """Every masked link as its text followed by its address — `click here (<https://…>)` — so
    model text cannot show one destination and open another. The angle brackets keep the address
    from unfurling into a preview."""
    return _LINK_SEAM.sub("] (", _MASKED_LINK.sub(r"\1 (<\2>)", text))


def _attachment_meta(msg: Any) -> tuple:
    """`(filename, size, content_type)` per uploaded file — metadata only, never the bytes."""
    return tuple((str(getattr(a, "filename", "")), int(getattr(a, "size", 0) or 0),
                  getattr(a, "content_type", None)) for a in (getattr(msg, "attachments", ()) or ()))


MAX_IMAGES_PER_MESSAGE = 4
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp")


def _is_image(a: Any) -> bool:
    return str(getattr(a, "content_type", "") or "").startswith("image/") \
        or str(getattr(a, "filename", "")).lower().endswith(_IMAGE_SUFFIXES)


async def _read_images(msg: Any) -> tuple[list[dict], list[str]]:
    """Image parts for the picture attachments (header-validated by core.content), plus one
    `[attachment <name> not read: <reason>]` note per picture that could not become a part — too
    big, over the per-message cap, unreadable, not really an image. Never a silent drop. Other
    files stay metadata only."""
    parts: list[dict] = []
    notes: list[str] = []
    for a in (getattr(msg, "attachments", ()) or ()):
        if not _is_image(a):
            continue
        name, size = str(getattr(a, "filename", "")), int(getattr(a, "size", 0) or 0)
        if len(parts) >= MAX_IMAGES_PER_MESSAGE:
            notes.append(f"[attachment {name} not read: only {MAX_IMAGES_PER_MESSAGE} images per message]")
            continue
        if size > MAX_IMAGE_BYTES:
            notes.append(f"[attachment {name} not read: {size / 1024 / 1024:.1f} MiB is over the "
                         f"{MAX_IMAGE_BYTES // 1024 // 1024} MiB limit]")
            continue
        try:
            parts.append(image_part(await a.read(), name=name))
        except ImageError as e:
            notes.append(f"[attachment {name} not read: {e}]")
        except Exception as e:  # noqa: BLE001 — a failed download is named, never fatal
            log.warning("discord_attachment_read_failed", name=name, error=f"{type(e).__name__}: {e}")
            notes.append(f"[attachment {name} not read: download failed ({type(e).__name__})]")
    return parts, notes


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
            raise ChannelStartError(TOKEN_MISSING)
        self._discord = discord
        deliver = on_message  # the channel's gate; the name is reused by discord's event below
        intents = discord.Intents.default()
        intents.message_content = True  # privileged intent — must be enabled on the bot
        # No mass, role or user pings from model text; a reply still notifies the person replied
        # to. Set on the client, it is the default of every send, edit, reply and file below.
        client = self._client = discord.Client(
            intents=intents,
            allowed_mentions=discord.AllowedMentions(everyone=False, users=False, roles=False, replied_user=True))
        ready = asyncio.Event()

        @client.event
        async def on_ready() -> None:
            log.info("discord_ready", bot=str(client.user))
            ready.set()

        @client.event
        async def on_message(msg: Any) -> None:
            if client.user is not None and msg.author.id == client.user.id:
                return
            # The bytes are fetched by the channel, and only after its allow gate admits the
            # author and conversation: a stranger's upload is never downloaded.
            await deliver(InboundMessage(
                author_id=str(msg.author.id), conversation_id=str(msg.channel.id),
                text=msg.content or "", is_bot=bool(msg.author.bot), handle=msg,
                conversation=msg.channel, attachments=_attachment_meta(msg),
                load_images=(lambda: _read_images(msg)) if any(map(_is_image, msg.attachments or ())) else None,
            ))

        @client.event
        async def on_raw_reaction_add(payload: Any) -> None:
            # RAW, not on_reaction_add: the raw event needs no message cache, so a permission
            # question still resolves after a reconnect. The bot pre-reacts to its own questions
            # and the gateway echoes those back: its reactions are never answers.
            if client.user is not None and str(getattr(payload, "user_id", "")) == str(client.user.id):
                return
            await on_reaction(str(getattr(payload, "user_id", "")),
                              int(getattr(payload, "message_id", 0) or 0),
                              str(getattr(payload, "emoji", "")))

        task = self._client_task = asyncio.create_task(client.start(self._token))
        waiter = asyncio.create_task(ready.wait())
        try:  # a login that fails never fires on_ready: wait for whichever comes first
            await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
        if not ready.is_set():
            err = None if task.cancelled() else task.exception()
            why = f"{type(err).__name__}: {err}" if err else "the client stopped before it was ready"
            raise ChannelStartError(f"Discord login failed — {why.replace(self._token, '**********')}") from None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
        if self._client_task is not None:
            try:
                await self._client_task
            except (asyncio.CancelledError, Exception):
                pass

    async def send(self, conversation: Any, text: str) -> Any:
        return await conversation.send(plain_links(text))

    async def react(self, message: Any, emoji: str) -> None:
        await message.add_reaction(emoji)

    async def edit(self, handle: Any, text: str) -> None:
        await handle.edit(content=plain_links(text))

    async def reply(self, handle: Any, text: str) -> Any:
        return await handle.reply(plain_links(text))

    async def send_file(self, conversation: Any, path: Any, mime: str) -> Any:
        return await conversation.send(file=self._discord.File(str(path), filename=Path(path).name))
