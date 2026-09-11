"""Discord's rendering of an ASK, and the `/mode` command on both channels (PRD §3.4, §3.5).

The Discord half uses the suite's existing fake-`discord`-module pattern (a `types.ModuleType`
installed into `sys.modules` before `start()` runs its own `import discord`), so the real
adapter code executes unmodified against a fake gateway — the reaction listener under test is
the one the production `start()` registers, not a stand-in.
"""
from __future__ import annotations

import asyncio
import sys
import types

import pytest

from localharness.agent.gate import PermissionGate
from localharness.agent.gate_types import PermissionRequest
from localharness.channels.discord import (
    PERMISSION_REACTIONS,
    PERMISSION_REACTIONS_UNGRANTABLE,
    DiscordChannel,
)
from localharness.config.grants import GrantStore
from localharness.core.bus import EventBus


def _request(grantable: bool = True) -> PermissionRequest:
    return PermissionRequest(
        tool_name="bash_exec",
        tool_params={"command": "cargo publish"},
        klass="shell-unfamiliar" if grantable else "shell-destructive",
        key="cargo publish" if grantable else None,
        grantable=grantable,
        reason="not seen in this workspace before",
        display="bash_exec: cargo publish  (shell-unfamiliar — not seen before)",
    )


class _SentMessage:
    def __init__(self, message_id: int) -> None:
        self.id = message_id
        self.reactions: list[str] = []

    async def add_reaction(self, emoji: str) -> None:
        self.reactions.append(emoji)


class _Channel:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.messages: list[_SentMessage] = []

    async def send(self, content: str) -> _SentMessage:
        self.sent.append(content)
        msg = _SentMessage(1000 + len(self.messages))
        self.messages.append(msg)
        return msg


class _InboundMessage:
    def __init__(self) -> None:
        self.channel = _Channel()


def _discord_channel() -> DiscordChannel:
    ch = DiscordChannel(EventBus(), {"token": "t", "allow_users": ["42"]})
    ch._client = object()  # ask_permission only checks it is not None
    ch._current_msg = _InboundMessage()
    return ch


def _react(ch: DiscordChannel, message_id: int, emoji: str) -> None:
    ch._reaction_waiters[message_id].put_nowait(emoji)


# ------------------------------------------------------------------ the reactions

@pytest.mark.asyncio
@pytest.mark.parametrize("emoji,kind", list(PERMISSION_REACTIONS.items()))
async def test_each_reaction_maps_to_its_decision(emoji, kind):
    ch = _discord_channel()
    task = asyncio.ensure_future(ch.ask_permission(_request()))
    await asyncio.sleep(0)
    sent = ch._current_msg.channel.messages[0]
    assert sent.reactions == list(PERMISSION_REACTIONS)
    _react(ch, sent.id, emoji)
    assert (await asyncio.wait_for(task, timeout=5.0)).kind == kind


@pytest.mark.asyncio
async def test_the_question_text_carries_the_request():
    ch = _discord_channel()
    task = asyncio.ensure_future(ch.ask_permission(_request()))
    await asyncio.sleep(0)
    posted = ch._current_msg.channel.sent[0]
    assert "Permission needed" in posted and "cargo publish" in posted
    _react(ch, ch._current_msg.channel.messages[0].id, "❌")
    await asyncio.wait_for(task, timeout=5.0)


@pytest.mark.asyncio
async def test_an_ungrantable_request_offers_only_two_reactions():
    ch = _discord_channel()
    task = asyncio.ensure_future(ch.ask_permission(_request(grantable=False)))
    await asyncio.sleep(0)
    sent = ch._current_msg.channel.messages[0]
    assert sent.reactions == list(PERMISSION_REACTIONS_UNGRANTABLE)
    assert "♾️" not in sent.reactions
    _react(ch, sent.id, "✅")
    assert (await asyncio.wait_for(task, timeout=5.0)).kind == "allow_once"


@pytest.mark.asyncio
async def test_an_unrelated_reaction_is_not_an_answer():
    ch = _discord_channel()
    task = asyncio.ensure_future(ch.ask_permission(_request()))
    await asyncio.sleep(0)
    sent = ch._current_msg.channel.messages[0]
    _react(ch, sent.id, "🎉")
    await asyncio.sleep(0)
    assert not task.done(), "a bystander's emoji answered the question"
    _react(ch, sent.id, "✅")
    assert (await asyncio.wait_for(task, timeout=5.0)).kind == "allow_once"


@pytest.mark.asyncio
async def test_the_gate_times_the_wait_out_and_denies(tmp_path):
    """PRD §3.5: the deadline lives in the gate, and a timeout is a `reject_once`."""
    from localharness.agent.gate_types import GateSettings, ToolMeta

    ch = _discord_channel()
    workspace = tmp_path / "project"
    workspace.mkdir()
    gate = PermissionGate(
        boundary=workspace,
        workspace=workspace,
        grants=GrantStore(tmp_path / "grants.yaml"),
        asker=ch.ask_permission,
        channel_name="discord",
        settings=GateSettings(ask_timeout_s=0.05),
    )
    outcome = await gate.check(
        "bash_exec", {"command": "cargo publish"}, ToolMeta(group="shell"),
        agent_id="a", session_id="s",
    )
    assert not outcome.allowed and "no answer" in outcome.reason
    assert ch._reaction_waiters == {}, "the waiter leaked after the timeout"


@pytest.mark.asyncio
async def test_with_nowhere_to_post_it_fails_closed():
    ch = DiscordChannel(EventBus(), {"token": "t", "allow_users": ["42"]})
    assert (await ch.ask_permission(_request())).kind == "reject_once"


def test_discord_declares_it_can_ask_and_has_no_review_surface():
    assert DiscordChannel.can_ask is True
    assert DiscordChannel.has_review_surface is False


@pytest.mark.asyncio
async def test_only_allowlisted_reactions_reach_the_waiter(monkeypatch):
    """The reaction listener the production `start()` registers, driven for real."""
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
        waiter: asyncio.Queue = asyncio.Queue()
        ch._reaction_waiters[7] = waiter
        handler = ch._client.events["on_raw_reaction_add"]

        await handler(types.SimpleNamespace(user_id=99, message_id=7, emoji="✅"))
        assert waiter.empty(), "a reaction from outside the allowlist was accepted"

        await handler(types.SimpleNamespace(user_id=42, message_id=7, emoji="✅"))
        assert waiter.get_nowait() == "✅"
    finally:
        if ch._client_task:
            ch._client_task.cancel()
        await ch.stop()


# --------------------------------------------------------------------- /mode

class _RecordingChannel:
    channel_id = "terminal"

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_message(self, text, agent_id=None, metadata=None) -> None:
        self.sent.append(text)


def _repl(channel, gate):
    from localharness.cli.repl import OrchestratorREPL

    return OrchestratorREPL(
        orchestrator=types.SimpleNamespace(active_workflow=None),
        agent_loop=None,
        channel=channel,
        bus=EventBus(),
        gate=gate,
    )


def _gate(tmp_path) -> PermissionGate:
    workspace = tmp_path / "project"
    workspace.mkdir(exist_ok=True)
    return PermissionGate(
        boundary=workspace, workspace=workspace, grants=GrantStore(tmp_path / "g.yaml"),
        channel_name="terminal",
    )


@pytest.mark.asyncio
async def test_slash_mode_switches_and_reports(tmp_path):
    channel, gate = _RecordingChannel(), _gate(tmp_path)
    repl = _repl(channel, gate)

    assert await repl._handle_slash("/mode read-only") is True
    assert gate.mode == "read-only"
    assert "read-only" in channel.sent[-1]

    assert await repl._handle_slash("/mode") is True
    assert "Permission mode: read-only" in channel.sent[-1]


@pytest.mark.asyncio
async def test_slash_mode_refuses_unattended(tmp_path):
    """PRD §3.4: a chat message must never be able to switch off asking."""
    channel, gate = _RecordingChannel(), _gate(tmp_path)
    repl = _repl(channel, gate)

    await repl._handle_slash("/mode read-only")
    await repl._handle_slash("/mode unattended")
    assert gate.mode == "read-only", "unattended was settable from a channel command"
    assert "cannot be set from a channel" in channel.sent[-1]


@pytest.mark.asyncio
async def test_slash_mode_rejects_an_unknown_name(tmp_path):
    channel, gate = _RecordingChannel(), _gate(tmp_path)
    await _repl(channel, gate)._handle_slash("/mode yolo")
    assert gate.mode == "guarded"
    assert "unknown mode" in channel.sent[-1]


@pytest.mark.asyncio
async def test_discord_uses_the_bare_word(tmp_path):
    channel, gate = _RecordingChannel(), _gate(tmp_path)
    channel.channel_id = "discord"
    repl = _repl(channel, gate)

    assert await repl._dispatch_input("mode trusted") is None
    assert gate.mode == "trusted"


@pytest.mark.asyncio
async def test_the_terminal_keeps_the_bare_word_as_a_message(tmp_path):
    """"mode" is an ordinary English word; the terminal has /mode for the command."""
    channel, gate = _RecordingChannel(), _gate(tmp_path)
    repl = _repl(channel, gate)
    started: list[str] = []

    async def _turn(task, on_token=None):
        started.append(task)
        return "done"

    repl._agent = types.SimpleNamespace(
        _config=types.SimpleNamespace(name="a"), current_session_id="s", run_turn=_turn
    )
    repl._detect_creation_intent = lambda _text: False  # type: ignore[method-assign]

    turn = await repl._dispatch_input("mode trusted")
    if turn is not None:
        await turn
    assert started == ["mode trusted"], "the line did not reach the model as a message"
    assert gate.mode == "guarded", "a plain sentence changed the permission mode"


# ------------------------------------------- the human sees the denial (defect D7)

@pytest.mark.asyncio
async def test_discord_posts_one_line_when_a_call_is_denied():
    """Defect D7: Discord's send_tool_result is deliberately silent, so without this the person
    is never told why the agent stopped short."""
    from localharness.core.events import Observation

    ch = _discord_channel()
    await ch.on_observation(Observation(
        agent_id="a", session_id="s", observation_type="tool_result", tool_call_id="tc-1",
        tool_name="bash_exec", output="[DENIED]",
        error="Permission denied: you answered 'never here' for this: bash_exec(*cargo publish*)",
    ))
    sent = ch._current_msg.channel.sent
    assert len(sent) == 1
    assert "permission denied" in sent[0]
    assert "never here" in sent[0]
