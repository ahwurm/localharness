"""Discord's rendering of a PARKED call — the `auto` staging queue (owner ruling 2026-09-12).

The staged twin of `test_channel_permission_asks.py`: there a question holds the loop open until
somebody answers, here nothing is held at all. So the thing under test is as much what the
adapter does NOT do — `send_pending_notice` returns with the reaction unanswered and the caller
carries on — as what it posts.

The fakes are the ones that file uses (a posted message that records its reactions, edits and
replies), copied rather than imported so neither file's fixtures can quietly change the other's
meaning.
"""
from __future__ import annotations

import asyncio
import sys
import types

import pytest

from localharness.agent.gate_types import PendingCall, PermissionRequest
from localharness.channels.discord import (
    PENDING_REACTIONS,
    DiscordChannel,
    pending_notice_body,
)
from localharness.core.bus import EventBus
from localharness.core.events import PermissionResolved, PermissionStaged

SESSION = "sess-1"


# ------------------------------------------------------------------ fakes (see module docstring)

class _SentMessage:
    def __init__(self, message_id: int, content: str = "", *, can_edit: bool = True) -> None:
        self.id = message_id
        self.content = content
        self.reactions: list[str] = []
        self.replies: list[str] = []
        self.can_edit = can_edit

    async def add_reaction(self, emoji: str) -> None:
        self.reactions.append(emoji)

    async def edit(self, content: str) -> None:
        if not self.can_edit:
            raise RuntimeError("cannot edit this message")
        self.content = content

    async def reply(self, content: str) -> None:
        self.replies.append(content)


class _Channel:
    def __init__(self, *, can_edit: bool = True) -> None:
        self.sent: list[str] = []
        self.messages: list[_SentMessage] = []
        self._can_edit = can_edit

    async def send(self, content: str) -> _SentMessage:
        self.sent.append(content)
        msg = _SentMessage(1000 + len(self.messages), content, can_edit=self._can_edit)
        self.messages.append(msg)
        return msg


class _InboundMessage:
    def __init__(self, *, can_edit: bool = True) -> None:
        self.channel = _Channel(can_edit=can_edit)


def _discord_channel(*, can_edit: bool = True) -> DiscordChannel:
    ch = DiscordChannel(EventBus(), {"token": "t", "allow_users": ["42"]})
    ch._client = object()  # the notice path only checks it is not None
    ch._current_msg = _InboundMessage(can_edit=can_edit)
    return ch


def _react(ch: DiscordChannel, message_id: int, emoji: str) -> None:
    ch._reaction_waiters[message_id].put_nowait(emoji)


# ------------------------------------------------------------------ the parked call

def _pending(pending_id: int = 3, rendering: str = "bash_exec: rm -rf ~/old-notes") -> PendingCall:
    return PendingCall(
        id=pending_id,
        request=PermissionRequest(
            tool_name="bash_exec",
            tool_params={"command": "rm -rf ~/old-notes"},
            klass="shell-destructive",
            key=None,
            grantable=False,
            reason="destructive command outside the project",
            display=rendering,
        ),
        rendering=rendering,
        agent_label="",
        session_id=SESSION,
        created_at=0.0,
    )


def _resolved(pending: PendingCall, decision: str) -> PermissionResolved:
    """The event `gate.approve`/`gate.deny` publish: it names the parked call by `pending_id`."""
    return PermissionResolved(
        agent_id="",
        session_id=pending.session_id,
        tool_name=pending.request.tool_name,
        klass=pending.request.klass,
        key=None,
        decision=decision,
        pending_id=pending.id,
    )


class _Resolver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def __call__(self, action: str, pending_id: int) -> None:
        self.calls.append((action, pending_id))


async def _close(ch: DiscordChannel) -> None:
    """Shut down the notice waiters without `stop()`: these channels hold a placeholder client."""
    for notice in list(ch._pending_notices.values()):
        if notice.task is not None:
            notice.task.cancel()
    ch._pending_notices.clear()


async def _settle(ch: DiscordChannel) -> None:
    """Let the detached waiter task run to its next await."""
    for _ in range(4):
        await asyncio.sleep(0)


# ------------------------------------------------------------------ the notice

@pytest.mark.asyncio
async def test_a_staged_call_posts_one_notice_with_both_answers():
    """The default `on_permission_staged` handler, driven with the real event."""
    ch = _discord_channel()
    pending = _pending()
    await ch.on_permission_staged(
        PermissionStaged(
            agent_id="a", session_id=SESSION, pending=pending, total=1, channel="discord"
        )
    )
    try:
        assert len(ch._current_msg.channel.messages) == 1
        sent = ch._current_msg.channel.messages[0]
        assert sent.reactions == list(PENDING_REACTIONS) == ["✅", "❌"]
        assert "#3" in sent.content and "rm -rf ~/old-notes" in sent.content
        assert "/approve 3" in sent.content and "/deny 3" in sent.content
        assert "1 pending" in sent.content
    finally:
        await _close(ch)


@pytest.mark.asyncio
async def test_the_notice_returns_without_waiting_for_an_answer():
    """The whole point of staging: nothing here holds the caller. No reaction is ever sent."""
    ch = _discord_channel()
    await asyncio.wait_for(ch.send_pending_notice(_pending(), 1), timeout=1.0)
    try:
        notice = ch._pending_notices[3]
        assert notice.task is not None and not notice.task.done()
    finally:
        await _close(ch)


@pytest.mark.asyncio
@pytest.mark.parametrize("emoji,action", list(PENDING_REACTIONS.items()))
async def test_a_reaction_resolves_the_call_and_closes_the_message(emoji, action):
    ch = _discord_channel()
    resolver = _Resolver()
    ch._pending_resolver = resolver
    await ch.send_pending_notice(_pending(), 1)
    sent = ch._current_msg.channel.messages[0]

    _react(ch, sent.id, emoji)
    await _settle(ch)

    assert resolver.calls == [(action, 3)]
    assert ("ran #3" if action == "approve" else "skipped #3") in sent.content
    assert ch._pending_notices == {} and ch._reaction_waiters == {}
    await _close(ch)


@pytest.mark.asyncio
async def test_a_reaction_from_outside_the_allowlist_never_reaches_the_notice(monkeypatch):
    """Driven through the listener `start()` registers, so the gate on WHO may answer is the
    production one — the same allowlist that gates inbound messages."""
    stub = types.ModuleType("discord")

    class _Intents:
        message_content = False

        @staticmethod
        def default():
            return _Intents()

    class _Client:
        def __init__(self, **kw):
            self.user = None
            self.events: dict = {}

        def event(self, fn):
            self.events[fn.__name__] = fn
            return fn

        async def start(self, token):
            await asyncio.sleep(3600)

        async def close(self):
            pass

    stub.Intents = _Intents
    stub.Client = _Client
    monkeypatch.setitem(sys.modules, "discord", stub)

    ch = DiscordChannel(EventBus(), {"token": "t", "allow_users": ["42"]})
    ch._ready.set()
    await ch.start()
    try:
        ch._current_msg = _InboundMessage()
        resolver = _Resolver()
        ch._pending_resolver = resolver
        await ch.send_pending_notice(_pending(), 1)
        sent = ch._current_msg.channel.messages[0]
        handler = ch._client.events["on_raw_reaction_add"]

        await handler(types.SimpleNamespace(user_id=99, message_id=sent.id, emoji="✅"))
        await _settle(ch)
        assert resolver.calls == [], "a stranger's reaction answered a parked call"

        await handler(types.SimpleNamespace(user_id=42, message_id=sent.id, emoji="✅"))
        await _settle(ch)
        assert resolver.calls == [("approve", 3)]
    finally:
        if ch._client_task:
            ch._client_task.cancel()
        await ch.stop()


@pytest.mark.asyncio
async def test_an_answer_typed_as_text_cancels_the_reaction_waiter():
    """`/approve 3` goes to the gate, not to this channel. The resolution event is how the notice
    finds out — and the waiter must die with it, or a later tap resolves the call twice."""
    ch = _discord_channel()
    resolver = _Resolver()
    ch._pending_resolver = resolver
    pending = _pending()
    await ch.send_pending_notice(pending, 1)
    sent = ch._current_msg.channel.messages[0]
    task = ch._pending_notices[3].task

    await ch.on_permission_resolved(_resolved(pending, "allow_once"))
    await _settle(ch)

    assert "ran #3" in sent.content
    assert task is not None and task.cancelled()
    assert ch._pending_notices == {} and ch._reaction_waiters == {}

    # the stale ✅ is now inert: nothing is listening, and the resolver was never called
    assert resolver.calls == []
    await _close(ch)


@pytest.mark.asyncio
async def test_a_denial_typed_as_text_closes_the_message_as_skipped():
    ch = _discord_channel()
    pending = _pending()
    await ch.send_pending_notice(pending, 1)
    sent = ch._current_msg.channel.messages[0]

    await ch.on_permission_resolved(_resolved(pending, "reject_once"))
    assert "skipped #3" in sent.content
    await _close(ch)


@pytest.mark.asyncio
async def test_a_resolution_for_something_else_leaves_the_notice_alone():
    ch = _discord_channel()
    await ch.send_pending_notice(_pending(), 1)
    sent = ch._current_msg.channel.messages[0]
    body = sent.content

    await ch.on_permission_resolved(
        PermissionResolved(
            agent_id="", session_id=SESSION, tool_name="file_write",
            klass="protected-path", key=None, decision="allow_once",
        )
    )
    assert sent.content == body and 3 in ch._pending_notices
    await _close(ch)


@pytest.mark.asyncio
async def test_without_a_resolver_the_tap_is_logged_and_said_out_loud(capsys):
    """The REPL installs `_pending_resolver`; until it does, a reaction must not vanish quietly."""
    ch = _discord_channel()
    await ch.send_pending_notice(_pending(), 1)
    sent = ch._current_msg.channel.messages[0]

    _react(ch, sent.id, "✅")
    await _settle(ch)

    # structlog renders to the console here, so the warning is read off the captured stream
    logged = capsys.readouterr().out
    assert "discord_pending_no_resolver" in logged and "warning" in logged
    assert "/approve 3" in sent.content  # the message points at the path that still works
    assert 3 in ch._pending_notices, "the text commands can still answer it"
    await _close(ch)


def test_a_long_command_is_truncated_into_one_message():
    """One message, never a chunked notice: the reactions have to stay on the line that shows
    the command they answer."""
    from localharness.channels.discord import _DISCORD_LIMIT, PENDING_TRUNCATION_SUFFIX

    body = pending_notice_body(_pending(rendering="bash_exec: " + "x" * (_DISCORD_LIMIT * 2)), 1)
    assert len(body) <= _DISCORD_LIMIT
    assert body.endswith(PENDING_TRUNCATION_SUFFIX) or PENDING_TRUNCATION_SUFFIX in body


@pytest.mark.asyncio
async def test_a_dialog_answer_without_a_pending_id_never_closes_a_notice():
    """A guarded-mode dialog resolves through the same event with the same pairing fields; only
    a resolution that names the parked call may close its notice (critic finding, 2026-09-12)."""
    ch = _discord_channel()
    ch._pending_resolver = _Resolver()
    pending = _pending(1)
    await ch.send_pending_notice(pending, 1)
    event = _resolved(pending, "allow_once")
    event = event.model_copy(update={"pending_id": None})
    await ch.on_permission_resolved(event)
    assert any(n.pending.id == 1 for n in ch._pending_notices.values()), "nobody answered #1"
