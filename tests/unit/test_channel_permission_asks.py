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
from localharness.agent.gate_types import ToolMeta
from localharness.cli.repl import (
    MODE_EFFECTS,
    MODE_SETTABLE_NAMES,
    NOTHING_PENDING,
)
from localharness.cli.slash_commands import SLASH_COMMANDS
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
    """A posted message, with the two ways the adapter can annotate it afterwards.

    `can_edit=False` stands in for a channel where the edit fails (a permissions change), which
    is the case the reply fallback exists for.
    """

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
    ch._client = object()  # ask_permission only checks it is not None
    ch._current_msg = _InboundMessage(can_edit=can_edit)
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
    """PRD §3.5: the deadline lives in the gate, and a timeout is a `reject_once`.

    ``mode`` is pinned to ``guarded`` because this test is about the ask machinery: v0.14.1
    moved the default to ``auto`` (owner ruling 2026-09-11), which allows an unfamiliar
    ``cargo publish`` silently and so would never open a question to time out.
    """
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
        mode="guarded",
        settings=GateSettings(ask_timeout_s=0.05),
    )
    outcome = await gate.check(
        "bash_exec", {"command": "cargo publish"}, ToolMeta(group="shell"),
        agent_id="a", session_id="s",
    )
    assert not outcome.allowed and "no answer" in outcome.reason
    assert ch._reaction_waiters == {}, "the waiter leaked after the timeout"


async def _expire(ch: DiscordChannel, seconds: float = 0.05) -> None:
    """Put the question through the gate's own deadline — the only thing that expires it."""
    task = asyncio.ensure_future(ch.ask_permission(_request()))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(task, seconds)


@pytest.mark.asyncio
async def test_an_expired_question_says_so_on_the_message():
    """D7: the 🛑 message used to stand unchanged forever after the gate had already denied the
    call. Three reactions, no answer, no sign it had stopped mattering — and a tap on ✅ an hour
    later was a silent no-op, because the waiter it would have reached was gone."""
    from localharness.channels.discord import PERMISSION_TIMEOUT_LINE

    ch = _discord_channel()
    await _expire(ch)

    posted = ch._current_msg.channel.messages[0]
    assert "Permission needed" in posted.content, "the question itself must survive the edit"
    assert PERMISSION_TIMEOUT_LINE.split("{")[0] in posted.content
    assert "denied" in posted.content
    assert ch._reaction_waiters == {}


@pytest.mark.asyncio
async def test_an_answered_question_is_never_annotated():
    from localharness.channels.discord import PERMISSION_TIMEOUT_LINE

    ch = _discord_channel()
    task = asyncio.ensure_future(ch.ask_permission(_request()))
    await asyncio.sleep(0)
    posted = ch._current_msg.channel.messages[0]
    _react(ch, posted.id, "✅")
    assert (await asyncio.wait_for(task, timeout=5.0)).kind == "allow_once"
    assert PERMISSION_TIMEOUT_LINE.split("{")[0] not in posted.content
    assert posted.replies == []


@pytest.mark.asyncio
async def test_a_channel_that_refuses_the_edit_gets_a_reply_instead():
    """Saying it in the wrong shape beats not saying it: the verdict has to be visible."""
    ch = _discord_channel(can_edit=False)
    await _expire(ch)

    posted = ch._current_msg.channel.messages[0]
    assert len(posted.replies) == 1
    assert "denied" in posted.replies[0]


@pytest.mark.asyncio
async def test_a_broken_gateway_costs_the_note_and_nothing_else():
    """The annotation runs while the gate is already cancelling this coroutine, so it must never
    become the reason a turn hangs or raises something other than the cancel it was handed."""
    ch = _discord_channel()

    async def _never(*_a, **_kw):
        await asyncio.sleep(3600)

    ch._current_msg.channel._can_edit = True
    task = asyncio.ensure_future(ch.ask_permission(_request()))
    await asyncio.sleep(0)
    posted = ch._current_msg.channel.messages[0]
    posted.edit = _never  # type: ignore[method-assign]
    posted.reply = _never  # type: ignore[method-assign]

    from localharness.channels import discord as discord_module

    # The wait is bounded by a named constant; shortened here so the test does not sit for it.
    original = discord_module.PERMISSION_TIMEOUT_POST_TIMEOUT_S
    discord_module.PERMISSION_TIMEOUT_POST_TIMEOUT_S = 0.05
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(task, 0.05)
    finally:
        discord_module.PERMISSION_TIMEOUT_POST_TIMEOUT_S = original
    assert ch._reaction_waiters == {}, "the waiter leaked when the annotation hung"


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
    """The gate the `/mode` tests below drive, started in GUARDED.

    ``mode`` is pinned rather than left to :data:`DEFAULT_MODE`: v0.14.1 moved the default to
    ``auto`` (owner ruling 2026-09-11), and these tests read the mode back to prove a command
    either did or did NOT change it — which needs a known, named starting point that is not
    whatever the default happens to be that release.
    """
    workspace = tmp_path / "project"
    workspace.mkdir(exist_ok=True)
    return PermissionGate(
        boundary=workspace, workspace=workspace, grants=GrantStore(tmp_path / "g.yaml"),
        channel_name="terminal", mode="guarded",
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
    # the bare command lists everything settable, loosest first — `auto` included, which is the
    # mode a person switching BACK from read-only wants and could not name before v0.14.1
    assert MODE_SETTABLE_NAMES in channel.sent[-1]
    assert "auto" in MODE_SETTABLE_NAMES and "unattended" in MODE_SETTABLE_NAMES


@pytest.mark.asyncio
async def test_slash_mode_accepts_auto_the_new_default(tmp_path):
    """`auto` is a real mode as of v0.14.1 and has to be reachable by name: a session that got
    tightened mid-task has to be able to get back."""
    channel, gate = _RecordingChannel(), _gate(tmp_path)
    repl = _repl(channel, gate)

    assert await repl._handle_slash("/mode auto") is True
    assert gate.mode == "auto"
    assert MODE_EFFECTS["auto"] in channel.sent[-1]


@pytest.mark.asyncio
async def test_slash_mode_sets_unattended_and_says_what_it_means(tmp_path):
    """`/mode unattended` goes through, and the REPL reports the mode it just entered.

    It was refused until v0.14.1, on the reasoning that a chat message must never switch off
    asking. The owner met that rule from inside his own terminal — "I can't swap my active
    localharness session to unattended without exiting" (2026-09-11) — and the person typing
    `/mode` there is the person the gate protects. What must not change is that the human is
    TOLD what they just turned off, so the reported line carries this mode's own effect text.
    """
    channel, gate = _RecordingChannel(), _gate(tmp_path)
    repl = _repl(channel, gate)

    await repl._handle_slash("/mode read-only")
    await repl._handle_slash("/mode unattended")
    assert gate.mode == "unattended"
    assert MODE_EFFECTS["unattended"] in channel.sent[-1]
    assert "unattended" in channel.sent[-1]


@pytest.mark.asyncio
async def test_slash_mode_rejects_an_unknown_name(tmp_path):
    channel, gate = _RecordingChannel(), _gate(tmp_path)
    await _repl(channel, gate)._handle_slash("/mode yolo")
    assert gate.mode == "guarded", "an unknown name changed the mode"
    # Loosest-first, from the strictness table: the list a person is offered after a typo.
    assert f"unknown mode 'yolo'; choose one of: {MODE_SETTABLE_NAMES}" in channel.sent[-1]


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


# ----------------------------------------------------- /pending, /approve, /deny

class _RecordingAgent:
    """Stands in for the AgentLoop: the only thing `/approve` asks of it is the nudge seam."""

    def __init__(self) -> None:
        self.nudges: list[str] = []

    def push_user_nudge(self, text: str) -> None:
        self.nudges.append(text)


async def _stage_one(gate: PermissionGate, command: str = "rm -rf ~/old-notes"):
    """Park one call the way a real turn does: through `check` in the default mode."""
    gate.set_mode("auto")
    gate.asker = _noop_asker
    return await gate.check(
        "bash_exec", {"command": command}, ToolMeta(group="shell"),
        agent_id="main", session_id="s",
    )


async def _noop_asker(request):
    raise AssertionError("the asker was called; `auto` must stage, never ask")


@pytest.mark.asyncio
async def test_slash_pending_lists_what_is_waiting(tmp_path):
    channel, gate = _RecordingChannel(), _gate(tmp_path)
    repl = _repl(channel, gate)

    assert await repl._handle_slash("/pending") is True
    assert channel.sent[-1] == NOTHING_PENDING

    await _stage_one(gate)
    assert await repl._handle_slash("/pending") is True
    assert "#1" in channel.sent[-1] and "rm -rf" in channel.sent[-1]


@pytest.mark.asyncio
async def test_slash_approve_answers_the_gate_and_nudges_a_running_turn(tmp_path):
    channel, gate = _RecordingChannel(), _gate(tmp_path)
    agent = _RecordingAgent()
    repl = _repl(channel, gate)
    repl._agent = agent
    repl._turn_task = asyncio.ensure_future(asyncio.sleep(5))
    try:
        await _stage_one(gate)

        assert await repl._handle_slash("/approve 1") is True
        assert gate.pending == {}
        assert "approved" in channel.sent[-1] and "#1" in channel.sent[-1]
        # The approval reaches the MODEL as words at the next step boundary — never as a second
        # path into tool dispatch.
        assert len(agent.nudges) == 1
        assert "approved pending #1" in agent.nudges[0]
        assert repl._slash_followup is None, "a running turn must not also start a new one"
    finally:
        repl._turn_task.cancel()


@pytest.mark.asyncio
async def test_slash_approve_on_an_idle_session_leaves_a_turn_to_start(tmp_path):
    """With no turn in flight there is nothing to nudge, so the approval becomes an ordinary
    user turn — started by `_dispatch_input`, which owns "a line became a turn"."""
    channel, gate = _RecordingChannel(), _gate(tmp_path)
    repl = _repl(channel, gate)
    await _stage_one(gate)

    assert await repl._handle_slash("/approve") is True
    assert repl._slash_followup is not None
    assert "approved pending #1" in repl._slash_followup


@pytest.mark.asyncio
async def test_slash_deny_drops_it_and_never_starts_a_turn(tmp_path):
    channel, gate = _RecordingChannel(), _gate(tmp_path)
    repl = _repl(channel, gate)
    await _stage_one(gate)

    assert await repl._handle_slash("/deny 1") is True
    assert gate.pending == {}
    assert "denied" in channel.sent[-1]
    # Nothing for an idle session to do about a refusal: there is no call to run.
    assert repl._slash_followup is None


@pytest.mark.asyncio
async def test_slash_approve_reports_an_unknown_number(tmp_path):
    channel, gate = _RecordingChannel(), _gate(tmp_path)
    repl = _repl(channel, gate)
    assert await repl._handle_slash("/approve 9") is True
    assert "No pending call #9" in channel.sent[-1]
    assert repl._slash_followup is None


@pytest.mark.asyncio
async def test_slash_approve_rejects_a_non_number(tmp_path):
    channel, gate = _RecordingChannel(), _gate(tmp_path)
    repl = _repl(channel, gate)
    assert await repl._handle_slash("/approve latest") is True
    assert "/approve takes a pending number" in channel.sent[-1]


def test_the_three_commands_are_in_the_one_table():
    """/help and the completion menu both read SLASH_COMMANDS: a command missing there is a
    command nobody discovers."""
    names = [name for name, _ in SLASH_COMMANDS]
    assert {"/pending", "/approve", "/deny"} <= set(names)
