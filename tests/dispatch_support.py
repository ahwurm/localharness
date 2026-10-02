"""Shared support for every dispatch test (Phase 49): env isolation + a recording fake `discord`.

SAFETY: this box's `~/.claude/channels/discord/.env` holds a live bot token that drives the
owner's dispatch fleet, and the dispatch plugin's env fallback reads it. Every
dispatch test calls `isolate_discord_env` FIRST, so `Path.home()` is a tmp dir and the five env
token/allow sources are gone — no test can ever see, print or log in with the real token.

The fake `discord` module stands in at the discord.py API boundary (Intents, Client.event /
start / close, messages with send / add_reaction / edit / reply). It never touches the network.
Every outbound call lands as one `(op, target, payload)` tuple in ONE shared `fake.log`, in call
order: `target` is a fake id string (`c<channel id>` for a channel, `m<message id>` for a
message), `payload` the exact text or emoji. Ids are deterministic counters, never time-derived.
Generalised (copied, not imported) from tests/unit/test_discord_pending_notice.py and
tests/unit/test_channel_permission_asks.py.
"""
from __future__ import annotations

import asyncio
import os
import sys
import types
from pathlib import Path
from typing import Any

DISCORD_ENV = (
    "LOCALHARNESS_DISCORD_TOKEN",
    "DISCORD_BOT_TOKEN",
    "LOCALHARNESS_DISCORD_ALLOW",
    "LOCALHARNESS_DISCORD_CHANNELS",
    "LOCALHARNESS_DISCORD_ACK",
)
BOT_USER_ID = 999

Record = tuple[str, "str | None", str]


def isolate_discord_env(monkeypatch, tmp_path: Path) -> Path:
    """HOME -> `tmp_path/home` (created); the five Discord env sources deleted."""
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    for name in DISCORD_ENV:
        monkeypatch.delenv(name, raising=False)
    assert Path.home().resolve().is_relative_to(tmp_path.resolve()), Path.home()
    assert not any(os.environ.get(n) for n in DISCORD_ENV)
    return home


class FakeDiscord:
    """The recording fake. `module` is what sits in `sys.modules["discord"]`."""

    def __init__(self) -> None:
        self.log: list[Record] = []
        self.client: Any = None
        self.files: list[Any] = []
        self.fail_edits = False
        self._next_id = 1000
        self._channels: dict[int, Any] = {}
        self._last_sent: Any = None
        self.module = self._build_module()

    # ---------------------------------------------------------------- ids, channels, messages

    def _id(self) -> int:
        self._next_id += 1
        return self._next_id

    def channel(self, channel_id: int) -> "_FakeChannel":
        if channel_id not in self._channels:
            self._channels[channel_id] = _FakeChannel(self, channel_id)
        return self._channels[channel_id]

    def message(self, author_id: int, channel_id: int, content: str, *, bot: bool = False,
                attachments: list | None = None) -> "_FakeMessage":
        author = types.SimpleNamespace(id=author_id, bot=bot)
        return _FakeMessage(self, self._id(), self.channel(channel_id), content, author,
                            attachments=attachments or [])

    @property
    def last_sent(self) -> "_FakeMessage":
        return self._last_sent

    async def deliver(self, msg: "_FakeMessage") -> None:
        await self.client.events["on_message"](msg)

    async def react(self, message_id: int, user_id: int, emoji: str) -> None:
        payload = types.SimpleNamespace(user_id=user_id, message_id=message_id, emoji=emoji)
        await self.client.events["on_raw_reaction_add"](payload)

    # ---------------------------------------------------------------- the module

    def _build_module(self) -> types.ModuleType:
        fake = self
        mod = types.ModuleType("discord")

        class Intents:
            def __init__(self) -> None:
                self.message_content = False

            @classmethod
            def default(cls) -> "Intents":
                return cls()

        class Client:
            def __init__(self, *, intents: Any = None, **_: Any) -> None:
                self.intents = intents
                self.user = types.SimpleNamespace(id=BOT_USER_ID)
                self.events: dict[str, Any] = {}
                self._closed = asyncio.Event()
                fake.client = self

            def event(self, fn):
                self.events[fn.__name__] = fn
                return fn

            async def start(self, token: str) -> None:
                fake.log.append(("connect", "client",
                                 f"message_content={getattr(self.intents, 'message_content', None)}"))
                await self.events["on_ready"]()
                await self._closed.wait()

            async def close(self) -> None:
                fake.log.append(("close", "client", ""))
                self._closed.set()

        class File:
            def __init__(self, fp: Any, filename: str | None = None, **_: Any) -> None:
                self.fp = fp
                self.filename = filename or Path(str(fp)).name
                fake.files.append(self)

        mod.Intents = Intents
        mod.Client = Client
        mod.File = File
        return mod


class _FakeChannel:
    def __init__(self, fake: FakeDiscord, channel_id: int) -> None:
        self._fake = fake
        self.id = channel_id

    async def send(self, content: str = "", **kw: Any) -> "_FakeMessage":
        self._fake.log.append(("send", f"c{self.id}", content))
        if kw.get("file") is not None:
            self._fake.log.append(("file", f"c{self.id}", kw["file"].filename))
        author = types.SimpleNamespace(id=BOT_USER_ID, bot=True)
        msg = _FakeMessage(self._fake, self._fake._id(), self, content, author)
        self._fake._last_sent = msg
        return msg


class _FakeMessage:
    def __init__(self, fake: FakeDiscord, message_id: int, channel: _FakeChannel, content: str,
                 author: Any, *, attachments: list | None = None) -> None:
        self._fake = fake
        self.id = message_id
        self.channel = channel
        self.content = content
        self.author = author
        self.attachments = attachments or []

    async def add_reaction(self, emoji: str) -> None:
        self._fake.log.append(("react", f"m{self.id}", str(emoji)))

    async def edit(self, content: str = "", **_: Any) -> None:
        if self._fake.fail_edits:
            self._fake.log.append(("edit_failed", f"m{self.id}", content))
            raise RuntimeError("cannot edit this message")
        self._fake.log.append(("edit", f"m{self.id}", content))
        self.content = content

    async def reply(self, content: str = "", **_: Any) -> None:
        self._fake.log.append(("reply", f"m{self.id}", content))


def install_fake_discord(monkeypatch) -> FakeDiscord:
    fake = FakeDiscord()
    monkeypatch.setitem(sys.modules, "discord", fake.module)
    return fake


def build_dispatch_discord(bus, *, allow, channels, ack, state_dir=None):
    """DispatchChannel over the real DiscordAdapter (token "t"), on the same fake `discord`.
    `state_dir` (49-08) is where the outbound artifact path looks; None = no artifacts."""
    from localharness.dispatch.adapters.discord import DiscordAdapter
    from localharness.dispatch.channel import DispatchChannel

    return DispatchChannel(bus, {"adapter": DiscordAdapter(token="t"), "allow": set(allow),
                                 "channels": set(channels), "ack": ack, "state_dir": state_dir})


def discord_events(ch) -> dict[str, Any]:
    """The handlers the Discord adapter registered on its (fake) client, by event name."""
    return ch._adapter._client.events
