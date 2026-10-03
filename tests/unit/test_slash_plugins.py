"""`/plugins` in a terminal session (RULINGS R1, R2, R3, R10, R14).

1. The restart decision: `plugins_cmd.switch_decision` and `plugins_overview`, over stub plugins
   swapped into BUILTIN_PLUGINS and a real config dir. Does `/plugins enable|disable NAME` restart
   the session (None), or answer one line and stay? The decision is made BEFORE teardown, from a
   fresh plan over the session's own layers, and asks and writes nothing.
2. The REPL: the terminal-only `/plugins` row and `_handle_plugins_cmd`. One line on a channel
   that cannot switch plugins, with no session hook, for a wrong spelling, or while a call is
   parked; otherwise the hook's line, or the restart: `restart_request` set and the REPL ended
   as /quit ends it, in the classic loop and in the box loop, typed-ahead lines kept.
3. The restart indicator (R3): after the rebuild, "Restarted with NAME on. Your conversation
   continues." and one status line — did the plugin come up, and how? The composed restart
   itself is tests/integration/test_setup_in_session_e2e.py.

Tests that call asyncio.run are plain `def`: asyncio_mode is "auto". Every REPL here gets an
explicit gate: on the MagicMock agent, `agent.gate.pending` is truthy and would read as a parked
call.
"""
from __future__ import annotations

import asyncio
import dataclasses
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from rich.console import Console

from localharness.channels.base import ChannelAdapter
from localharness.channels.terminal import TerminalChannel
from localharness.channels.web.channel import WebChannel
from localharness.cli import plugins_cmd, start_cmd
from localharness.cli.repl import OrchestratorREPL
from localharness.cli.slash_commands import all_rows, find_row
from localharness.plugins import builtin, discovery
from tests.unit.test_plugin_step import Reach
from tests.unit.test_plugins_enable_setup import _CONFIG, Needy, Stub
from tests.unit.test_repl_input_box import FakeBoxChannel, _pending_turn, _repl
from tests.unit.test_repl_unknown_slash import RecordingChannel, _build_repl

_REAL_DISCOVER = discovery.discover
NEEDY_LINE = ("needy needs its install extra first: install `localharness[nosuchextra]`, "
              "then /plugins enable needy again")
USAGE_TAIL = "/plugins enable <name> turns one on here; /plugins disable <name> turns one off."


# --- 1. the restart decision -------------------------------------------------------------------

@pytest.fixture
def g(tmp_path: Path, monkeypatch) -> Path:
    """stub (off by default, one question) and needy (on by default, its extra missing) as the
    bundled plugins, over a real global config dir — returned."""
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Stub, Needy))
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [
        f for f in _REAL_DISCOVER(global_config_dir) if f.source == "folder"])
    g = tmp_path / "g"
    g.mkdir()
    (g / "config.yaml").write_text(yaml.safe_dump(_CONFIG), encoding="utf-8")
    return g


def _overrides(directory: Path, data: dict) -> None:
    (directory / "overrides.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def _decide(name: str, on: bool, running: set[str], g: Path, workspace: Path | None = None):
    return asyncio.run(plugins_cmd.switch_decision(name, on, frozenset(running), g, workspace))


def test_decision_unknown_name_is_one_line(g):
    assert _decide("nope", True, set(), g) == "Unknown plugin: nope. /plugins lists them."


def test_decision_running_and_set_up_is_already_on(g):
    _overrides(g, {"stub": {"enabled": True, "url": "http://ok"}})
    assert _decide("stub", True, {"stub"}, g) == "stub is already on."


def test_decision_not_running_restarts(g):
    assert _decide("stub", True, set(), g) is None


def test_decision_running_but_its_step_still_has_work_restarts(g):
    # on, but its question has no answer: the step's questions are asked outside the input box
    _overrides(g, {"stub": {"enabled": True}})
    assert _decide("stub", True, {"stub"}, g) is None


def test_decision_needs_its_extra_is_one_line(g):
    assert _decide("needy", True, set(), g) == NEEDY_LINE


def test_decision_needs_its_extra_even_when_turned_off(g):
    # the plan reads `off` now; the decision asks for the extra directly, as the step does
    _overrides(g, {"needy": {"enabled": False}})
    assert _decide("needy", True, set(), g) == NEEDY_LINE


def test_decision_a_project_that_turns_it_off_still_wins(g, tmp_path):
    ws = tmp_path / "proj" / ".localharness"
    ws.mkdir(parents=True)
    (ws / "config.yaml").write_text(yaml.safe_dump({"stub": {"enabled": False}}), encoding="utf-8")
    assert _decide("stub", True, set(), g, ws) == (
        f"this project turns stub off ({ws}), and that still wins here — "
        "`localharness plugins enable stub --workspace` changes it for this project")


def test_decision_disable_of_an_enabled_plugin_restarts(g):
    _overrides(g, {"stub": {"enabled": True, "url": "http://ok"}})
    assert _decide("stub", False, {"stub"}, g) is None


def test_decision_disable_of_a_plugin_not_enabled_is_already_off(g):
    assert _decide("stub", False, set(), g) == "stub is already off."


def test_decision_asks_and_writes_nothing(g, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the decision must not ask or write")
    monkeypatch.setattr(plugins_cmd.typer, "prompt", boom)
    monkeypatch.setattr(plugins_cmd, "atomic_write_overlay", boom)
    _overrides(g, {"stub": {"enabled": True}})
    before = sorted(p.name for p in g.iterdir())
    assert _decide("stub", True, {"stub"}, g) is None
    assert _decide("stub", False, {"stub"}, g) is None
    assert sorted(p.name for p in g.iterdir()) == before


def test_overview_lists_every_plugin_and_how_to_switch_one(g):
    text = plugins_cmd.plugins_overview(frozenset(), g, None)
    assert text.splitlines() == [
        "Plugins:",
        "  stub   off — turn on: localharness plugins enable stub",
        "  needy  on (install `localharness[nosuchextra]` to use it)",
        USAGE_TAIL,
    ]


def test_overview_says_which_run_in_this_session(g):
    _overrides(g, {"stub": {"enabled": True, "url": "http://ok"}})
    lines = plugins_cmd.plugins_overview(frozenset({"stub"}), g, None).splitlines()
    assert lines[1] == "  stub   running in this session"
    assert lines[-1] == USAGE_TAIL


def test_decision_disable_of_a_plugin_running_here_restarts_even_when_its_flag_is_off(g):
    # its flag was turned off on disk mid-session (a shell `plugins disable`): it still runs here,
    # so "already off" would be false — the restart is what stops it
    assert _decide("stub", False, {"stub"}, g) is None


def test_decision_enable_of_a_plugin_running_here_whose_flag_is_off_restarts(g):
    # running and set up, but its flag on disk is off: "already on" would leave it off at the
    # next start — the restart's step is what writes the flag
    _overrides(g, {"stub": {"url": "http://ok"}})
    assert _decide("stub", True, {"stub"}, g) is None


# --- 2. the REPL: the terminal-only row and its handler ----------------------------------------

INFO = {"style": "system.info"}
TERMINAL_ONLY = ("/plugins works only in a terminal session. From a shell: "
                 "localharness plugins enable <name>")
NO_SESSION = "/plugins needs a session started with `localharness start`."
USAGE = "Usage: /plugins, /plugins enable <name> or /plugins disable <name>"
PARKED = "Answer the parked calls first (/pending lists them): a restart would drop them."


class Terminalish(RecordingChannel):
    """RecordingChannel that, like the terminal, can restart the session for /plugins."""
    can_switch_plugins = True


class BoxTerminal(FakeBoxChannel):
    can_switch_plugins = True


def _classic(channel, *, pending=None, **kw):
    """_build_repl's REPL, built again with the /plugins keywords (the hook, the queued lines),
    and an explicit gate holding `pending`."""
    base, agent, bus = _build_repl(channel)
    repl = OrchestratorREPL(orchestrator=base._orchestrator, agent_loop=agent, channel=channel,
                            bus=bus, **kw)
    repl._gate = SimpleNamespace(pending=pending or {})
    return repl, agent


def _box(**kw):
    """test_repl_input_box's `_repl`, built again with the /plugins keywords."""
    base, channel, agent = _repl(BoxTerminal())
    repl = OrchestratorREPL(orchestrator=base._orchestrator, agent_loop=agent, channel=channel,
                            bus=base._bus, **kw)
    repl._box_ctrl_q = asyncio.Queue()
    repl._gate = SimpleNamespace(pending={})
    return repl, channel, agent


def test_the_plugins_row_is_the_one_terminal_only_row():
    row = find_row("/plugins")
    assert row is not None and row.terminal_only and row.takes_args
    assert row.handler == "_slash_plugins" and row.plugin is None
    assert find_row("/plugins enable image") is row
    assert [r.name for r in all_rows() if r.terminal_only] == ["/plugins"]


def test_only_the_terminal_can_switch_plugins():
    assert ChannelAdapter.can_switch_plugins is False
    assert TerminalChannel.can_switch_plugins is True
    assert WebChannel.can_switch_plugins is False


async def test_plugins_on_a_channel_that_cannot_switch_is_one_line_and_never_calls_the_hook():
    hook = AsyncMock(return_value=None)
    channel = RecordingChannel(["/plugins enable image"])
    repl, _ = _classic(channel, on_plugin_switch=hook)
    await repl.run()
    assert channel.sent == [(TERMINAL_ONLY, INFO)]
    hook.assert_not_awaited()
    assert repl.restart_request is None


async def test_plugins_without_a_session_hook_says_so():
    channel = Terminalish(["/plugins enable image"])
    repl, _ = _classic(channel)
    await repl.run()
    assert channel.sent == [(NO_SESSION, INFO)]
    assert repl.restart_request is None


@pytest.mark.parametrize("line", ["/plugins frobnicate x", "/plugins enable", "/plugins enable a b"])
async def test_plugins_with_a_wrong_spelling_shows_the_usage(line):
    hook = AsyncMock(return_value=None)
    channel = Terminalish([line])
    repl, _ = _classic(channel, on_plugin_switch=hook)
    await repl.run()
    assert channel.sent == [(USAGE, INFO)]
    hook.assert_not_awaited()


async def test_plugins_refuses_the_restart_while_a_call_is_parked():
    hook = AsyncMock(return_value=None)
    channel = Terminalish(["/plugins enable image", "/help"])
    repl, _ = _classic(channel, pending={1: object()}, on_plugin_switch=hook)
    await repl.run()
    assert channel.sent[0] == (PARKED, INFO)
    assert channel.sent[1][0].startswith("Available commands:")  # the session went on
    hook.assert_not_awaited()
    assert repl.restart_request is None


async def test_plugins_shows_the_hooks_line_and_reads_the_next_input():
    hook = AsyncMock(return_value="image is already on.")
    channel = Terminalish(["/plugins enable image", "/help"])
    repl, _ = _classic(channel, on_plugin_switch=hook)
    await repl.run()
    hook.assert_awaited_once_with("enable", "image")
    assert channel.sent[0] == ("image is already on.", INFO)
    assert channel.sent[1][0].startswith("Available commands:")
    assert repl.restart_request is None


@pytest.mark.parametrize(("line", "action", "said"), [
    ("/plugins enable image", ("enable", "image"), "Restarting with image on — your conversation is kept."),
    ("/Plugins Disable Image", ("disable", "image"), "Restarting with image off — your conversation is kept."),
])
async def test_plugins_restart_ends_the_repl_without_reading_on(line, action, said):
    hook = AsyncMock(return_value=None)
    channel = Terminalish([line, "never read"])
    repl, agent = _classic(channel, on_plugin_switch=hook)
    await repl.run()
    hook.assert_awaited_once_with(*action)
    assert channel.sent == [(said, INFO)]
    assert repl.restart_request == action
    assert channel._inputs == ["never read"]
    agent.run_turn.assert_not_called()


@pytest.mark.parametrize("line", ["/plugins", "/PLUGINS  "])
async def test_bare_plugins_sends_the_list(line):
    hook = AsyncMock(return_value="Plugins:\n  image  on")
    channel = Terminalish([line])
    repl, _ = _classic(channel, on_plugin_switch=hook)
    await repl.run()
    hook.assert_awaited_once_with("list", "")
    assert channel.sent == [("Plugins:\n  image  on", INFO)]
    assert repl.restart_request is None


async def test_classic_mode_plays_resumed_lines_before_reading_input():
    channel = Terminalish([])
    repl, agent = _classic(channel, on_plugin_switch=AsyncMock(return_value=None), queued=("hello",))
    await repl.run()
    agent.run_turn.assert_awaited_once()
    assert agent.run_turn.call_args.kwargs["task"] == "hello"
    assert repl.queued == ()


async def test_classic_mode_keeps_the_lines_after_a_replayed_restart():
    channel = Terminalish(["never read"])
    repl, _ = _classic(channel, on_plugin_switch=AsyncMock(return_value=None),
                       queued=("/plugins enable y", "/help"))
    await repl.run()
    assert repl.restart_request == ("enable", "y")
    assert repl.queued == ("/help",)
    assert channel._inputs == ["never read"]


async def test_box_mode_plugins_enable_ends_the_loop_and_keeps_typed_ahead_lines():
    hook = AsyncMock(return_value=None)
    repl, channel, _ = _box(on_plugin_switch=hook)
    await _pending_turn(repl)
    running = repl._turn_task
    try:
        assert await repl._handle_box_event("submit", "/plugins enable x") is True
        assert await repl._handle_box_event("submit", "/help") is True
        assert repl.queued == ("/plugins enable x", "/help")  # slash lines mid-turn are queued
        hook.assert_not_awaited()
        done = asyncio.ensure_future(asyncio.sleep(0))
        await done
        assert await repl._handle_box_event("turn_done", done) is False
        hook.assert_awaited_once_with("enable", "x")
        assert repl.restart_request == ("enable", "x")
        assert repl.queued == ("/help",)
        assert ("Restarting with x on — your conversation is kept.", INFO) in channel.sent
    finally:
        running.cancel()


async def test_box_mode_replays_resumed_lines_first():
    hook = AsyncMock(return_value=None)
    repl, channel, _ = _box(on_plugin_switch=hook, queued=("/plugins enable y", "/help"))
    await asyncio.wait_for(repl._run_with_box(), 5)
    hook.assert_awaited_once_with("enable", "y")
    assert repl.restart_request == ("enable", "y")
    assert repl.queued == ("/help",)
    assert channel.box_started is True and channel.box_stopped is True


# --- 3. the restart indicator ------------------------------------------------------------------

def _resume(action=("enable", "x"), **kw):
    fields = dict(action=action, agent_name="orchestrator", conversation=(), prior_context="",
                  eviction_store=None, queued=(), gate_mode="auto", previous_sitting_id="s1")
    return start_cmd.Resume(**(fields | kw))


def _started(loaded=(), failed=None, unconfigured=None):
    """What start_plugins reported, in LifecycleResult's shape."""
    return SimpleNamespace(loaded_names=list(loaded), failed=failed or {}, unconfigured=unconfigured or {})


def test_indicator_the_restarted_line():
    assert start_cmd.RESTARTED_LINE.format(name="x", state="on") == \
        "Restarted with x on. Your conversation continues."


def test_indicator_the_handle_is_plain_frozen_data():
    resume = _resume()
    assert dataclasses.is_dataclass(resume)
    with pytest.raises(dataclasses.FrozenInstanceError):
        resume.gate_mode = "unattended"
    restart = start_cmd.Restart(("enable", "x"), resume)
    assert restart.action == ("enable", "x") and restart.resume is resume


def test_indicator_running():
    assert start_cmd._resume_status(_resume(), _started(["x"])) == "x: on in this session"


def test_indicator_its_check_failed():
    assert start_cmd._resume_status(_resume(failed_check="no answer at http://c"), _started(["x"])) == \
        "x: on, but its check failed: no answer at http://c"


def test_indicator_not_set_up_yet():
    assert start_cmd._resume_status(_resume(), _started(unconfigured={"x": "x.url"})) == \
        "x: on, but not set up yet — run /plugins enable x to set it up"


def test_indicator_a_skipped_check_is_not_set_up_yet():
    """Deferred #23: a check the step skipped (web's "not enrolled yet", autoresearch with no
    proposer) is a plugin that is not set up yet, never "its check failed"."""
    web = _resume(("enable", "web"), skipped_check="not enrolled yet")
    assert start_cmd._resume_status(web, _started(["web"])) == "web: on, but not set up yet — not enrolled yet"


def test_indicator_it_could_not_start():
    assert start_cmd._resume_status(_resume(), _started(failed={"x": "start() raised OSError: boom"})) == \
        "x: on, but it could not start: start() raised OSError: boom"


def test_indicator_its_step_stopped():
    stopped = _resume(step_stopped=True)
    assert start_cmd._resume_status(stopped, _started()) == \
        "x: not on in this session — its setup stopped before it finished"
    assert start_cmd._resume_status(stopped, _started(["x"])) == \
        "x: on in this session — its setup stopped before it finished"


def test_indicator_disabled():
    off = _resume(("disable", "x"))
    assert start_cmd._resume_status(off, _started(["y"])) == "x: off in this session"
    assert start_cmd._resume_status(off, _started(["x"])) == "x: still on in this session — /plugins shows why"


def test_indicator_otherwise_not_running():
    assert start_cmd._resume_status(_resume(), _started()) == \
        "x: not running in this session — /plugins shows its state"
    assert start_cmd._resume_status(_resume(), None) == \
        "x: not running in this session — /plugins shows its state"


def test_indicator_a_failing_setup_action_row_is_a_failed_check(g, monkeypatch):
    """Deferred item 4: in the session's step, a setup-action row that does not pass while the
    offline doctor check passes reaches the indicator as "its check failed", not "on"."""
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Stub, Needy, Reach))
    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    monkeypatch.setattr(plugins_cmd.typer, "prompt", lambda text, default=None, **kw: "http://far")
    monkeypatch.setattr(plugins_cmd, "console", Console(file=io.StringIO(), width=400))

    outcome = plugins_cmd.session_step(("enable", "reach"), str(g))

    assert outcome == plugins_cmd.StepOutcome("reach", True, failed_check="no answer from http://far")
    resume = dataclasses.replace(_resume(("enable", "reach")), failed_check=outcome.failed_check,
                                 step_stopped=outcome.stopped)
    assert start_cmd._resume_status(resume, _started(["reach"])) == \
        "reach: on, but its check failed: no answer from http://far"
