"""FixtureChatAdapter — an in-memory second chat platform for the DISP-03 proof (Phase 49-08).

Test-only. Registered by monkeypatching one line into `localharness.dispatch.adapters.ADAPTERS`
and one name into the dispatch manifest's `channels` tuple; nothing under src/ names it.
Every verb appends one `(verb, target, payload)` tuple to `self.log`; `push` delivers an inbound
message and `react` an inbound reaction through the callbacks `connect` was handed.
`token` is accepted because DispatchPlugin.make_channel builds every adapter as `cls(token=...)`
from its one (Discord-shaped) settings section; this fixture ignores it.
"""
from __future__ import annotations

import itertools
import types
from typing import Any

from localharness.dispatch.channel import InboundMessage

INSTANCES: list["FixtureChatAdapter"] = []  # the test finds the adapter start built here


class FixtureChatAdapter:
    platform = "fixturechat"
    title = "Fixturechat"
    message_limit = 50

    def __init__(self, token: str = "") -> None:
        self.log: list[tuple] = []
        self._ids = itertools.count(1)
        self._on_message = self._on_reaction = None
        INSTANCES.append(self)

    async def connect(self, on_message, on_reaction) -> None:
        self._on_message, self._on_reaction = on_message, on_reaction
        self.log.append(("connect", None, ""))

    async def close(self) -> None:
        self.log.append(("close", None, ""))

    def _handle(self, conversation: Any, text: str) -> Any:
        return types.SimpleNamespace(id=next(self._ids), conversation=conversation, text=text)

    async def push(self, author: str, conversation: str, text: str, *, attachments: tuple = ()) -> Any:
        handle = self._handle(conversation, text)
        await self._on_message(InboundMessage(author_id=author, conversation_id=conversation, text=text,
                                              is_bot=False, handle=handle, conversation=conversation,
                                              attachments=attachments))
        return handle

    async def react_as(self, user: str, message_id: int, emoji: str) -> None:
        await self._on_reaction(user, message_id, emoji)

    async def send(self, conversation: Any, text: str) -> Any:
        handle = self._handle(conversation, text)
        self.log.append(("send", conversation, text, handle.id))
        return handle

    async def react(self, message: Any, emoji: str) -> None:
        self.log.append(("react", message.id, emoji))

    async def edit(self, handle: Any, text: str) -> None:
        self.log.append(("edit", handle.id, text))

    async def reply(self, handle: Any, text: str) -> Any:
        self.log.append(("reply", handle.id, text))
        return self._handle(handle.conversation, text)

    async def send_file(self, conversation: Any, path: Any, mime: str) -> Any:
        self.log.append(("file", conversation, str(path), mime))
