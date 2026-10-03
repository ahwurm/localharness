"""`channels.remote_unattended`: the machine's lock on what the phone and Discord may loosen.

The default (true) keeps 0.16's behaviour: a remote channel may switch the session to
`unattended` and answer "always" — the owner drives his own harness that way. Set to false in the
machine config, the phone and Discord are refused both: `/mode unattended` gets one line naming
the setting, and a question offers only the `_once` answers (an "always" that arrives anyway
counts once). The terminal and Zed (ACP, over stdio) are local operators and are never limited.

Driven through the gate, the REPL's `/mode` handler (which also serves Discord's bare
`mode <name>` and the web `command` route), the web `/mode` route over ASGI, and a Discord ask
through the real adapter on the fake gateway.
"""
from __future__ import annotations

import asyncio
import types
from pathlib import Path

import pytest

import localharness.agent.gate as gate_mod
from localharness.agent.gate import PermissionGate
from localharness.agent.gate_types import Decision, PermissionRequest, ToolMeta
from localharness.config.grants import GrantStore
from localharness.core.bus import EventBus

SHELL = ToolMeta(group="shell")
CARGO = {"command": "cargo build --release"}  # guarded: shell-unfamiliar, grantable


def _gate(tmp_path: Path, **kw) -> PermissionGate:
    workspace = tmp_path / "project"
    workspace.mkdir(exist_ok=True)
    return PermissionGate(boundary=workspace, workspace=workspace,
                          grants=GrantStore(tmp_path / "grants.yaml"), mode="guarded", **kw)


class _Remote:
    """A remote channel as the gate sees it: an id, an asker, and no `local_operator`."""

    can_ask = True

    def __init__(self, channel_id: str, answer: str = "allow_always") -> None:
        self.channel_id = channel_id
        self.answer = answer
        self.seen: list[PermissionRequest] = []
        self.sent: list[tuple[str, dict | None]] = []

    async def ask_permission(self, request: PermissionRequest) -> Decision:
        self.seen.append(request)
        return Decision(kind=self.answer)

    async def send_message(self, text, agent_id=None, metadata=None) -> None:
        self.sent.append((text, metadata))


class _Local(_Remote):
    local_operator = True


async def _check(gate: PermissionGate):
    return await gate.check("bash_exec", CARGO, SHELL, agent_id="a", session_id="s")


def _granted(gate: PermissionGate, asked: PermissionRequest) -> bool:
    """Did an answer to `asked` leave a grant behind, under any key the question carried?"""
    keys = asked.grant_keys or ((asked.klass, asked.key),)
    return any(gate.grants.lookup(gate.workspace, klass, key) is not None for klass, key in keys)


# ------------------------------------------------------------------- mode, through the gate

@pytest.mark.parametrize("channel_id", ["web", "discord"])
def test_the_lock_refuses_unattended_from_a_remote_channel(tmp_path, channel_id):
    gate = _gate(tmp_path, remote_unattended=False)
    gate.attach_channel(_Remote(channel_id))
    with pytest.raises(ValueError) as e:
        gate.set_mode("unattended", from_channel=True)
    assert str(e.value) == gate_mod.REMOTE_UNATTENDED_REFUSAL
    assert "channels.remote_unattended" in str(e.value) and "\n" not in str(e.value)
    assert gate.mode == "guarded"


def test_the_lock_leaves_every_other_switch_alone(tmp_path):
    """Only `unattended` from the channel is locked: the other modes still switch from the phone,
    and the config/trust paths (no `from_channel`) are the machine's own decisions."""
    gate = _gate(tmp_path, remote_unattended=False)
    gate.attach_channel(_Remote("web"))
    assert gate.set_mode("trusted", from_channel=True) == "trusted"
    assert gate.set_mode("guarded", from_channel=True) == "guarded"
    assert gate.set_mode("unattended") == "unattended"
    assert gate.set_mode("unattended", from_channel=True) == "unattended", "a no-op switch is no escalation"


def _terminal(tmp_path: Path):
    from localharness.channels.terminal import TerminalChannel
    return TerminalChannel(EventBus(), {}, history_file=str(tmp_path / ".h"))


def _acp(tmp_path: Path):
    from localharness.channels.acp import AcpChannel
    return AcpChannel(config_dir=str(tmp_path / "config"))


@pytest.mark.parametrize("build", [_terminal, _acp], ids=["terminal", "acp"])
def test_a_local_operator_is_never_locked(tmp_path, build):
    channel = build(tmp_path)
    assert channel.local_operator is True
    gate = _gate(tmp_path, remote_unattended=False)
    gate.attach_channel(channel)
    assert gate.set_mode("unattended", from_channel=True) == "unattended"


def test_remote_channels_are_not_local_operators():
    from localharness.channels.base import ChannelAdapter
    from localharness.channels.web.channel import WebChannel
    from localharness.dispatch.channel import DispatchChannel

    assert ChannelAdapter.local_operator is False
    assert WebChannel.local_operator is False and DispatchChannel.local_operator is False


async def test_the_default_keeps_remote_unattended_and_always(tmp_path):
    """Friction pin: with nothing set, the phone and Discord do exactly what they did in 0.16."""
    gate = _gate(tmp_path)
    channel = _Remote("discord")
    gate.attach_channel(channel)

    assert (await _check(gate)).allowed
    assert channel.seen[0].grantable is True
    assert gate_mod.REMOTE_ALWAYS_NOTE not in channel.seen[0].display
    assert _granted(gate, channel.seen[0]), "the default must still write the grant an always answer asks for"

    assert gate.set_mode("unattended", from_channel=True) == "unattended"


# ------------------------------------------------------------------- "always", through the gate

async def test_a_locked_remote_question_offers_no_always_and_counts_it_once(tmp_path):
    gate = _gate(tmp_path, remote_unattended=False)
    channel = _Remote("discord", answer="allow_always")
    gate.attach_channel(channel)

    first = await _check(gate)
    assert first.allowed, "the answer still lets the call run, once"
    asked = channel.seen[0]
    assert asked.grantable is False
    assert asked.display.endswith("\n" + gate_mod.REMOTE_ALWAYS_NOTE)
    assert "channels.remote_unattended" in gate_mod.REMOTE_ALWAYS_NOTE
    assert not _granted(gate, asked), "a locked remote always wrote a permanent grant"

    await _check(gate)
    assert len(channel.seen) == 2, "the same call must ask again: nothing was remembered"


async def test_a_locked_machine_still_lets_the_terminal_answer_always(tmp_path):
    gate = _gate(tmp_path, remote_unattended=False)
    channel = _Local("terminal")
    gate.attach_channel(channel)

    assert (await _check(gate)).allowed
    assert channel.seen[0].grantable is True
    assert gate_mod.REMOTE_ALWAYS_NOTE not in channel.seen[0].display
    assert _granted(gate, channel.seen[0])


# ------------------------------------------------------------------- the REPL's /mode handler

def _repl(channel, gate):
    from localharness.cli.repl import OrchestratorREPL
    return OrchestratorREPL(orchestrator=types.SimpleNamespace(active_workflow=None), agent_loop=None,
                            channel=channel, bus=EventBus(), gate=gate)


async def test_the_repl_mode_command_shows_the_refusal_as_one_error_line(tmp_path):
    gate = _gate(tmp_path, remote_unattended=False)
    channel = _Remote("discord")
    channel.bare_mode_command = True
    gate.attach_channel(channel)
    repl = _repl(channel, gate)

    await repl._handle_mode_cmd("unattended")
    assert channel.sent[-1] == (gate_mod.REMOTE_UNATTENDED_REFUSAL, {"style": "system.error"})
    assert await repl._dispatch_input("mode unattended") is None  # Discord's bare word
    assert channel.sent[-1] == (gate_mod.REMOTE_UNATTENDED_REFUSAL, {"style": "system.error"})
    assert gate.mode == "guarded"


# ------------------------------------------------------------------- the web /mode route

async def test_the_web_mode_route_answers_400_with_the_refusal(tmp_path):
    from tests.unit.channels.test_web_server import JSON, _stack

    gate = _gate(tmp_path, remote_unattended=False)
    _, channel, _, client = await _stack(tmp_path, runtime={"gate": gate})
    gate.attach_channel(channel)

    refused = await client.post("/api/sessions/s1/mode", json={"mode": "unattended"}, headers=JSON)
    assert refused.status_code == 400
    assert refused.json() == {"error": gate_mod.REMOTE_UNATTENDED_REFUSAL}
    assert gate.mode == "guarded"
    ok = await client.post("/api/sessions/s1/mode", json={"mode": "trusted"}, headers=JSON)
    assert ok.json() == {"status": "set", "mode": "trusted"}


async def test_the_web_mode_route_still_switches_with_the_default(tmp_path):
    from tests.unit.channels.test_web_server import JSON, _stack

    gate = _gate(tmp_path)
    _, channel, _, client = await _stack(tmp_path, runtime={"gate": gate})
    gate.attach_channel(channel)
    got = await client.post("/api/sessions/s1/mode", json={"mode": "unattended"}, headers=JSON)
    assert got.json() == {"status": "set", "mode": "unattended"}


# ------------------------------------------------------------------- the web's confirm step

def _grantable() -> PermissionRequest:
    return PermissionRequest(tool_name="bash_exec", tool_params={"command": "ls"}, klass="shell-new",
                             key="bash:ls", grantable=True, reason="new command", display="bash_exec: ls")


async def test_a_malformed_confirm_token_is_refused_never_raised(tmp_path):
    """`secrets.compare_digest` raises TypeError on a non-ASCII str or a non-str — a 500 from
    anything holding the bearer. Every malformed token is just "not the token"."""
    from tests.unit.channels.test_web_server import _stack

    _, channel, _, _ = await _stack(tmp_path)
    task = asyncio.ensure_future(channel.ask_permission(_grantable()))
    await asyncio.sleep(0)
    request_id = next(iter(channel._open_asks))
    token = channel.answer_ask(request_id, "allow_always")["confirm_token"]

    for bad in ("é" + token[1:], "é…", "\ud800" + token[1:], 123, ["x"], {"t": token}):
        got = channel.answer_ask(request_id, "allow_always", confirm=bad)  # type: ignore[arg-type]
        assert got["status"] == "confirm_required" and got["confirm_token"] == token, bad
    assert not task.done()
    assert channel.answer_ask(request_id, "allow_always", confirm=token)["status"] == "recorded"
    assert (await task).kind == "allow_always"


async def test_a_malformed_answer_over_the_wire_is_a_4xx_never_a_500(tmp_path):
    from tests.unit.channels.test_web_server import JSON, _stack

    _, channel, _, client = await _stack(tmp_path)
    task = asyncio.ensure_future(channel.ask_permission(_grantable()))
    await asyncio.sleep(0)
    request_id = next(iter(channel._open_asks))
    url = f"/api/permissions/{request_id}/answer"

    for body in ({"kind": "allow_always", "confirm_token": 123},
                 {"kind": "allow_always", "confirm_token": "é…"},
                 {"kind": ["allow_always"]}):
        got = await client.post(url, json=body, headers=JSON)
        assert 400 <= got.status_code < 500, (body, got.status_code)
    assert not task.done()
    task.cancel()


# ------------------------------------------------------------------- a Discord ask, end to end

async def test_a_locked_discord_question_carries_no_infinity(tmp_path, monkeypatch):
    """The real DispatchChannel on the fake gateway: the posted question has no ♾️ reaction, says
    why, and an allowlisted ✅ lets the call run once."""
    from tests.dispatch_support import build_dispatch_discord, install_fake_discord, isolate_discord_env

    isolate_discord_env(monkeypatch, tmp_path)
    fake = install_fake_discord(monkeypatch)
    ch = build_dispatch_discord(EventBus(), allow=("42",), channels=(), ack="")
    await ch.start()
    try:
        await fake.deliver(fake.message(42, 7, "build it"))
        await ch.read_input()
        gate = _gate(tmp_path, remote_unattended=False)
        gate.attach_channel(ch)

        check = asyncio.ensure_future(_check(gate))
        for _ in range(6):
            await asyncio.sleep(0)
        question = fake.last_sent
        assert gate_mod.REMOTE_ALWAYS_NOTE in question.content
        reactions = [r[2] for r in fake.log if r[0] == "react" and r[1] == f"m{question.id}"]
        assert reactions == ["✅", "❌"]
        await fake.react(question.id, 42, "✅")
        assert (await asyncio.wait_for(check, 1.0)).allowed
        assert not gate.grants.path.exists(), "nothing durable may be written from a locked remote"
    finally:
        await ch.stop()
