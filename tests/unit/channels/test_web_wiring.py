"""The wiring: the `web` branch, the unknown-`--channel` fix, and the REPL handles.

These are the "is it actually reachable" tests. A green test on a channel nobody constructs is a
checkmark on a lie, so each one here asserts something about the path a real `localharness web`
takes: which branch picks the channel, which resolvers the REPL installs, which command surfaces
stop refusing.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest
import typer

from localharness.channels.web.channel import WebChannel
from localharness.core.bus import EventBus

pytestmark = pytest.mark.asyncio


# ------------------------------------------------------------------ the CLI surface

async def test_the_web_command_is_registered_and_visible():
    """A channel that exists only in the docs is a channel nobody finds."""
    from localharness.cli.app import app

    names = {c.name for c in app.registered_commands}
    assert "web" in names
    web = next(c for c in app.registered_commands if c.name == "web")
    assert web.hidden is not True


async def test_start_async_takes_a_prebuilt_web_channel():
    """Passed IN, like ACP's, because the HTTP server answers long before a session exists."""
    from localharness.cli.start_cmd import _start_async

    params = inspect.signature(_start_async).parameters
    assert "web_channel" in params and params["web_channel"].default is None


async def test_an_unknown_channel_is_refused_instead_of_silently_becoming_the_terminal(tmp_path):
    """WEBCH-01's second half. `--channel discrod` used to start an ordinary terminal session
    with nothing anywhere saying the flag had been ignored."""
    from localharness.cli.start_cmd import KNOWN_CHANNEL_MODES, _start_async

    assert KNOWN_CHANNEL_MODES == {"terminal", "discord", "web"}
    with pytest.raises(typer.BadParameter) as exc:
        await _start_async(None, False, False, str(tmp_path), channel_mode="discrod")
    assert "discrod" in str(exc.value)
    assert "terminal" in str(exc.value) and "web" in str(exc.value)


async def test_the_refusal_happens_before_any_session_is_built(tmp_path):
    """Refusing at the CLI boundary alone would leave every other caller of `_start_async` with
    the old silent fallback."""
    from localharness.cli.start_cmd import _start_async

    # An empty config dir exits early with the welcome message; a bad channel must be refused
    # regardless of what else is or is not on disk.
    (tmp_path / "config.yaml").write_text("version: '1'\n")
    with pytest.raises(typer.BadParameter):
        await _start_async(None, False, False, str(tmp_path), channel_mode="nope")


# ------------------------------------------------------------------ the REPL handles

def _repl(channel):
    """A REPL with the collaborators these tests touch and nothing else.

    Built by hand rather than through `_start_async` on purpose: what is under test is the
    handshake between the REPL and a channel, and a full session build would drown it.
    """
    from localharness.cli.repl import OrchestratorREPL

    repl = OrchestratorREPL.__new__(OrchestratorREPL)
    repl._channel = channel
    repl._bus = EventBus()
    repl._pending_handles = []
    repl._turn_task = None
    return repl


async def test_the_repl_installs_the_nudge_and_cancel_handles():
    """§6.0.1: the REPL installs each one ONLY if the channel declares it as None, and a channel
    that does not is skipped with NO error — which fails silently, at a tap, much later."""
    channel = WebChannel(bus=EventBus(), config={})
    repl = _repl(channel)
    repl._subscribe_pending()

    assert channel._pending_resolver is not None
    assert channel._nudge_resolver is not None
    assert channel._cancel_resolver is not None


async def test_a_channel_that_declares_nothing_is_left_alone():
    """The terminal declares none of the three; installing on it would be a behaviour change."""
    class _Bare:
        pass

    bare = _Bare()
    repl = _repl(bare)
    repl._subscribe_pending()
    for name in ("_pending_resolver", "_nudge_resolver", "_cancel_resolver"):
        assert not hasattr(bare, name)


async def test_the_nudge_handle_reaches_push_user_nudge_only_while_a_turn_runs():
    """`push_user_nudge` is a public method built for precisely this, drained at the next step
    boundary. With nothing running there is nothing to steer, and the honest answer is False."""
    pushed: list[str] = []

    class _Agent:
        def push_user_nudge(self, text):
            pushed.append(text)

    repl = _repl(WebChannel(bus=EventBus(), config={}))
    repl._agent = _Agent()

    assert await repl._nudge_from_channel("too late") is False
    assert pushed == []

    async def _forever():
        await asyncio.sleep(3600)

    repl._turn_task = asyncio.ensure_future(_forever())
    try:
        assert await repl._nudge_from_channel("  use ripgrep  ") is True
        assert pushed == ["use ripgrep"]
        assert await repl._nudge_from_channel("   ") is False
    finally:
        repl._turn_task.cancel()


async def test_the_cancel_handle_cancels_the_turn_not_the_session():
    repl = _repl(WebChannel(bus=EventBus(), config={}))
    assert await repl._cancel_from_channel() is False

    async def _forever():
        await asyncio.sleep(3600)

    repl._turn_task = asyncio.ensure_future(_forever())
    assert await repl._cancel_from_channel() is True
    with pytest.raises(asyncio.CancelledError):
        await repl._turn_task


# ------------------------------------------------------------------ streaming opt-in

async def test_a_turn_streams_only_when_the_channel_offers_a_token_sink():
    """WEBCH-05. The terminal declares no `on_token` and keeps passing None — it has never
    streamed answer text — so this adds live text to the web channel and changes nothing else."""
    from localharness.channels.base import ChannelAdapter
    from localharness.channels.terminal import TerminalChannel

    assert ChannelAdapter.streams_tokens is False          # the safe default
    assert TerminalChannel.streams_tokens is False         # unchanged: it has never streamed
    assert WebChannel.streams_tokens is True
    assert callable(getattr(WebChannel, "on_token", None))

    seen: dict = {}

    class _Agent:
        _config = type("C", (), {"name": "orchestrator"})()
        current_session_id = "s1"

        async def run_turn(self, task, on_token=None, **kw):
            seen["on_token"] = on_token
            return "done"

    channel = WebChannel(bus=EventBus(), config={})
    repl = _repl(channel)
    repl._agent = _Agent()
    task = await repl._start_user_turn("hello")
    await task
    assert seen["on_token"] == channel.on_token


# ------------------------------------------------------------------ /reasoning and /verbose

async def test_the_display_toggles_are_capability_gated_not_class_gated():
    """WEBCH-19: they were `isinstance(TerminalChannel)` checks that refused on every other
    surface — right while the terminal was the only one that could show reasoning, and wrong the
    moment a second one could."""
    from localharness.channels.base import ChannelAdapter
    from localharness.channels.terminal import TerminalChannel

    assert ChannelAdapter.has_display_toggles is False   # safe default
    assert TerminalChannel.has_display_toggles is True
    assert WebChannel.has_display_toggles is True


async def test_reasoning_and_verbose_act_on_the_web_channel():
    said: list[str] = []

    channel = WebChannel(bus=EventBus(), config={})

    async def _say(content, agent_id=None, metadata=None):
        said.append(content)

    channel.send_message = _say  # type: ignore[method-assign]
    repl = _repl(channel)

    await repl._handle_reasoning_cmd("on")
    assert channel.show_reasoning is True
    await repl._handle_reasoning_cmd("off")
    assert channel.show_reasoning is False
    await repl._handle_verbose_cmd("on")
    assert channel.verbose is True
    assert not any("terminal-channel setting" in line for line in said)


async def test_a_channel_without_the_toggles_still_gets_an_honest_refusal():
    """Discord has no reasoning stream, and saying so is better than pretending to toggle one."""
    said: list[str] = []

    class _Chan:
        channel_id = "discord"
        has_display_toggles = False

        async def send_message(self, content, agent_id=None, metadata=None):
            said.append(content)

    repl = _repl(_Chan())
    await repl._handle_reasoning_cmd("on")
    await repl._handle_verbose_cmd("on")
    assert len(said) == 2
    assert all("does not" in line or "not have one" in line for line in said)


# ------------------------------------------------------------------ bring-up staging

async def test_bringup_stages_are_named_and_abortable():
    """WEBCH-43: a rising number reports elapsed time, not health. A wedged memory lock, a
    failing MCP server and a sibling process holding the inference flock all look identical to a
    healthy slow start."""
    channel = WebChannel(bus=EventBus(), config={})
    await channel.start()
    client = channel.attach_client()

    assert channel.bringup is None
    assert channel.abort_bringup() is False        # nothing to abort, said honestly

    aborted: list[bool] = []
    channel.set_bringup_abort(lambda: aborted.append(True))
    channel.set_bringup("opening memory", detail="waiting on the WAL lock", elapsed=4.0)

    _, _, payload = client.queue.get_nowait()
    import json
    frame = json.loads(payload)
    assert frame["frame_type"] == "BringUpStage"
    assert frame["stage"] == "opening memory" and frame["abortable"] is True
    assert channel.abort_bringup() is True and aborted == [True]

    channel.set_bringup("failed", detail="no model server", failed=True)
    assert channel.model_state() == "unreachable"
