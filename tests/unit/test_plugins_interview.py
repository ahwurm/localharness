"""The Plugins flow driven through a scripted channel: the root, its close and the reload decision."""
from __future__ import annotations

import asyncio
import io
import os
import signal
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, get_args

import click
import pytest
import yaml
from pydantic import BaseModel, field_validator
from rich.console import Console

from localharness.cli import plugins_cmd, plugins_interview
from localharness.cli.interview import Answer, Unreachable
from localharness.cli.plugins_cmd import (
    CONSENT, CTRLC, CTRLC_DOES, CTRLC_IGNORED, INSTALLING, OVERRIDE_LINE, THIRD_PARTY_CONFIRM, Kind, PassProgress, PassResult, PluginSession, Refusal, ReloadPlan, StepOutcome, TailKey, Touch,
    reload_names,
)
from localharness.cli.plugins_interview import CLOSE_LABELS, PLUGINS_QUESTION, at_close, run
from localharness.cli.update_cmd import InstallRoute, Manual
from localharness.plugins import builtin, lifecycle
from localharness.plugins import resolve as resolve_module
from localharness.plugins.api import Check, Plugin, PluginManifest, SetupField
from localharness.plugins.plan import PlanEntry
from localharness.plugins.setup import AGENT_PROMPT_LEAD
from tests.unit.interview_support import CANCEL, UNREACHED, Asked
from tests.unit.test_plugin_resolve import write_folder_plugin
from tests.unit.test_extra_install import SPEC, _Proc
from tests.unit.test_extra_install import dist as _dist
from tests.unit.test_plugins_bookkeeping import _CONFIG, Stub
from tests.unit.test_plugins_pass import CALLS, DOWNLOAD, Ask, Mob
from tests.unit.test_plugins_pass import _on as _overlay
from tests.unit.test_plugins_pass import folder as _folder
from tests.unit.test_plugins_pass import layers as _layers
from tests.unit.test_plugins_pass import mobile as _mobile

layers, folder, dist, mobile = _layers, _folder, _dist, _mobile
HANG = object()


class Label:
    def __init__(self, label: str) -> None:
        self.label = label


class Unreached:
    def __init__(self, why: Unreachable) -> None:
        self.why = why


class FakeChannel:
    """Answers `run_interview` from a script and records the span, the questions and the console."""

    def __init__(self, *answers, waiting: int = 0) -> None:
        self.answers, self.waiting = list(answers), waiting
        self.asked: list[Asked] = []
        self.entered = 0
        self.closed = False
        self.console = Console(file=io.StringIO(), width=400)
        self.out_at_exit: int | None = None

    @asynccontextmanager
    async def suspended_box(self):
        self.entered += 1

        def close() -> None:
            self.closed = True

        try:
            yield None, self.console, close
        finally:
            self.out_at_exit = len(self.text)

    @property
    def text(self) -> str:
        return self.console.file.getvalue()

    @property
    def lines(self) -> list[str]:
        return self.text.splitlines()

    def waiting_asks(self) -> int:
        return self.waiting

    async def run_interview(self, question, options, *, default, cancel, multi=False) -> Answer:
        self.asked.append(Asked(question, tuple(options), default, cancel, multi))
        if not self.answers:
            raise AssertionError("asked")
        answer = self.answers.pop(0)
        if callable(answer):
            answer = answer(options)
        if answer is HANG:
            await asyncio.sleep(100)
        if answer is CANCEL:
            return Answer((cancel,), True)
        if answer is UNREACHED:
            return Answer((cancel,), True, reached=False, why=Unreachable.NO_TTY)
        if isinstance(answer, Unreached):
            return Answer((cancel,), True, reached=False, why=answer.why)
        if isinstance(answer, Label):
            return Answer((next(o.key for o in options if o.label == answer.label),), False)
        if isinstance(answer, (set, frozenset)):
            return Answer(tuple(o.key for o in options if o.key in answer), False)
        return Answer((answer,), False)


def _rows(asked) -> list:
    return [o for o in asked.options if not o.key.startswith("+")]


def _session(layers, **kw) -> PluginSession:
    return PluginSession(layers.g, layers.ws, kw.pop("running", frozenset()), kw.pop("unconfigured", {}),
                         kw.pop("failed", {}), **kw)


def _should_reload(running=frozenset(), parked=0, waiting=0):
    return lambda touched: ReloadPlan(reload_names(touched, frozenset(running)), parked, waiting)


async def _run(layers, channel, *, running=frozenset(), parked=0, waiting=0, **kw) -> bool:
    return await run(channel, _session(layers, running=running, **kw), _should_reload(running, parked, waiting))


@pytest.mark.parametrize("answer", ["+close", CANCEL], ids=["close", "esc"])
async def test_reload_root_asks_one_row_per_plugin_then_close_and_prints_nothing(layers, answer) -> None:
    channel = FakeChannel(answer)

    assert await _run(layers, channel) is False

    assert channel.entered == 1 and not channel.closed and channel.text == ""
    (asked,) = channel.asked
    rows = [o for o in asked.options if not o.key.startswith("+")]
    assert asked.question == PLUGINS_QUESTION == "Plugins — Esc or Ctrl-C closes"
    assert [o.key for o in rows] == ["0", "1", "2"]
    assert [o.label for o in rows] == ["stub", "ask", "mob"]
    assert (asked.options[-1].key, asked.options[-1].label) == ("+close", "Close")
    assert asked.cancel == "+close" and asked.default == [o.key for o in asked.options].index("0")


async def test_reload_a_picked_plugin_row_opens_its_menu_and_back_draws_the_root_again(layers) -> None:
    channel = FakeChannel("1", CANCEL, "+close")

    assert await _run(layers, channel) is False

    assert len(channel.asked) == 3 and channel.entered == 1


async def test_reload_close_with_a_plugin_that_will_load_closes_the_box_and_returns_true(layers) -> None:
    channel = FakeChannel("+close")

    assert await _run(layers, channel, touched={"stub": Touch(False, True, None)}) is True

    assert channel.closed and channel.text == ""
    assert channel.asked[0].options[-1].label == CLOSE_LABELS["restarts"].format(plugins="1 plugin")


async def test_reload_close_is_true_only_for_a_close_not_for_a_reload_that_waits(layers) -> None:
    channel = FakeChannel("+close")

    assert await _run(layers, channel, touched={"stub": Touch(False, True, None)}, waiting=1) is False

    assert not channel.closed


async def test_reload_with_nothing_touched_names_nothing_and_the_close_row_reads_close(layers) -> None:
    channel = FakeChannel("+close")

    assert await _run(layers, channel) is False

    assert channel.asked[0].options[-1].label == "Close" and not channel.closed


async def test_reload_two_opens_share_the_touched_dict_the_caller_holds(layers) -> None:
    touched: dict = {}
    first = FakeChannel(lambda options: touched.update(stub=Touch(False, True, None)) or "+close")

    assert await _run(layers, first, touched=touched) is True
    second = FakeChannel("+close")

    assert await _run(layers, second, touched=touched) is True
    assert touched == {"stub": Touch(False, True, None)}


@pytest.mark.parametrize(("running", "touch", "names"), [
    (set(), Touch(False, True, None), {"p"}),
    (set(), Touch(True, False, None), set()),
    ({"p"}, Touch(False, False, None), {"p"}),
    ({"p"}, Touch(False, True, None), set()),
    ({"p"}, Touch(True, True, None), {"p"}),
    ({"q"}, Touch(True, False, None), set()),
], ids=["starts", "rebind-while-off", "stops", "unchanged", "rebinds", "other-runs"])
def test_reload_names_holds_the_plugins_whose_loading_differs_or_that_run_and_rebind(running, touch, names) -> None:
    assert reload_names({"p": touch}, frozenset(running)) == frozenset(names)


def test_reload_names_compares_the_plan_names_exactly() -> None:
    assert reload_names({"Stub": Touch(False, True, None)}, frozenset({"stub"})) == frozenset({"Stub"})


def test_reload_at_close_phrases() -> None:
    assert at_close(ReloadPlan(frozenset({"p"}), 0, 0)) == "when you close Plugins"
    assert at_close(ReloadPlan(frozenset({"p"}), 1, 0)) == "when you next close Plugins with no asks pending"


CLOSING = "when you close Plugins"
LATER = "when you next close Plugins with no asks pending"


def _entry(state="on", name="p", extra=None):
    manifest = PluginManifest(name=name, version="0.1.0", kind="tools", requires_extra=extra)
    return PlanEntry(name, True, state, "built in", "does things", "", manifest)


def _hint(entry, *, running=(), touched=None, unconfigured=None, failed=None, names=(), parked=0, waiting=0):
    session = PluginSession(Path("/nowhere"), None, frozenset(running), unconfigured or {}, failed or {}, touched or {})
    return plugins_interview._row_state(entry, session, ReloadPlan(frozenset(names), parked, waiting))


def _touch(probe=None):
    return {"p": Touch(False, True, probe)}


@pytest.mark.parametrize(("entry", "kw", "hint"), [
    (_entry(), dict(running={"p"}, names={"p"}, touched=_touch(("failed", "x"))),
     f"on — check failed, stops {CLOSING}; Info shows why"),
    (_entry(), dict(running={"p"}, names={"p"}, parked=1, touched=_touch(("failed", "x"))),
     f"on — check failed, stops {LATER}; Info shows why"),
    (_entry(), dict(touched=_touch(("failed", "x"))), "on — check failed; Info shows why"),
    (_entry(), dict(running={"p"}, touched=_touch(("failed", "x"))), "on — check failed; Info shows why"),
    (_entry(), dict(touched=_touch(("unconfigured", "p.url"))), "on — not set up; Re-run setup"),
    (_entry(), dict(running={"p"}, names={"p"}, touched=_touch(("unconfigured", "p.url"))),
     f"on — not set up, stops {CLOSING}; Re-run setup"),
    (_entry(), dict(unconfigured={"p": "p.url"}), "on — not set up; Re-run setup"),
    (_entry(), dict(unconfigured={"p": "p.url"}, running={"p"}, names={"p"}),
     f"on — not set up, stops {CLOSING}; Re-run setup"),
    (_entry(), dict(unconfigured={"p": "p.url"}, names={"p"}, touched=_touch(("unconfigured", "p.url"))),
     "on — not set up; Re-run setup"),
    (_entry(), dict(unconfigured={"p": "p.url"}, names={"p"}, touched=_touch(("on", ""))), f"on — starts {CLOSING}"),
    (_entry(), dict(names={"p"}, touched=_touch()), f"on — starts {CLOSING}"),
    (_entry(), dict(names={"p"}, parked=2, touched=_touch()), f"on — starts {LATER}"),
    (_entry("off"), dict(running={"p"}, names={"p"}), f"off {CLOSING}"),
    (_entry("available"), dict(running={"p"}, names={"p"}, waiting=1), f"off {LATER}"),
    (_entry(), dict(running={"p"}, names={"p"}), f"on — changed, applies {CLOSING}"),
    (_entry(), dict(failed={"p": "boom"}), "on — did not load (Info shows why)"),
    (_entry(), dict(failed={"p": "boom"}, touched=_touch(("on", ""))), "on — not running; Re-run setup starts it"),
    (_entry(), dict(), "on — not running; Re-run setup starts it"),
    (_entry(), dict(running={"p"}), "on"),
    (_entry("off"), dict(), "off"),
    (_entry("off"), dict(touched=_touch()), "off"),
    (_entry("available"), dict(), "available"),
    (_entry("refused"), dict(), "refused"),
], ids=["probe_failed_stops", "probe_failed_stops_deferred", "probe_failed", "probe_failed_running_unqueued",
        "unconfigured_probe", "unconfigured_probe_stops", "unconfigured_build", "unconfigured_build_stops",
        "unconfigured_beats_starts", "passing_probe_outranks_the_build", "starts", "starts_deferred", "stops",
        "stops_available", "rebinds", "load_failed", "load_failed_needs_no_later_probe", "not_running",
        "on_plain", "off_untouched", "off_disabled_not_running", "available_plain", "state_word"])
def test_row_state_hint(entry, kw, hint) -> None:
    assert _hint(entry, **kw) == hint


def test_row_state_probe_failed_stops_outranks_rebinds_and_load_failed_outranks_not_running() -> None:
    both = dict(running={"p"}, names={"p"}, touched={"p": Touch(True, False, ("failed", "x"))})
    assert _hint(_entry(), **both).startswith("on — check failed, stops")
    assert _hint(_entry(), failed={"p": "boom"}) == "on — did not load (Info shows why)"


@pytest.mark.parametrize(("route", "hint"), [
    (SimpleNamespace(argv=["pip", "install", "x"]), "needs localharness[mobile] — Install and set up"),
    (SimpleNamespace(argv=None), "needs localharness[mobile] — Show install step"),
    (None, "needs localharness[mobile] — install it by hand, then Re-run setup"),
], ids=["needs_extra_argv", "needs_extra_manual", "needs_extra_third_party"])
def test_row_state_a_plugin_holding_for_its_extra_names_the_way_to_get_it(monkeypatch, route, hint) -> None:
    monkeypatch.setattr(plugins_cmd, "install_route_for", lambda entry: route)

    assert _hint(_entry("needs-extra", extra="mobile")) == hint


def test_row_state_off_needs_extra_names_the_extra(monkeypatch) -> None:
    monkeypatch.setattr(plugins_cmd, "extra_for", lambda entry: "mobile")

    assert _hint(_entry("off", extra="mobile")) == "off (needs localharness[mobile])"
    assert _hint(_entry("on", extra="mobile"), running={"p"}) == "on"


async def test_root_rows_carry_the_row_state_hint(layers) -> None:
    (layers.g / "overrides.yaml").write_text(yaml.safe_dump({"stub": {"enabled": True}}), encoding="utf-8")
    channel = FakeChannel("+close")

    await _run(layers, channel)

    assert [o.hint for o in _rows(channel.asked[0])[:3]] == ["on — not running; Re-run setup starts it", "off", "off"]


async def test_root_is_resolved_again_on_every_render(layers) -> None:
    def enable(options):
        (layers.g / "overrides.yaml").write_text(yaml.safe_dump({"stub": {"enabled": True}}), encoding="utf-8")
        return "1"

    channel = FakeChannel(enable, CANCEL, "+close")

    await _run(layers, channel)

    first, second = (_rows(channel.asked[i])[0].hint for i in (0, 2))
    assert first == "off" and second == "on — not running; Re-run setup starts it"


async def test_root_lists_a_third_party_folder_plugin_without_importing_it(folder) -> None:
    write_folder_plugin(folder.g, "swatch")
    channel = FakeChannel("+close")

    await _run(folder, channel)

    (asked,) = channel.asked
    assert "swatch" in [o.label for o in asked.options]
    assert [o.hint for o in asked.options if o.label == "swatch"] == ["available"]
    assert not (Path(os.environ["LH_TEST_SENTINEL_DIR"]) / "swatch").exists()


async def test_root_two_refused_entries_with_one_name_get_their_own_rows(layers, monkeypatch) -> None:
    twice = tuple(_entry("refused", name="dup") for _ in range(2))
    monkeypatch.setattr(resolve_module, "resolve", lambda loader, **kw: SimpleNamespace(plan=SimpleNamespace(entries=twice)))
    channel = FakeChannel("+close")

    await _run(layers, channel)

    assert [(o.key, o.label) for o in channel.asked[0].options] == [("0", "dup"), ("1", "dup"), ("+close", "Close")]


async def test_root_with_no_plugin_entries_shows_only_close_and_closing_prints_nothing(layers, monkeypatch) -> None:
    monkeypatch.setattr(resolve_module, "resolve", lambda loader, **kw: SimpleNamespace(plan=SimpleNamespace(entries=())))
    channel = FakeChannel("+close")

    assert await _run(layers, channel) is False

    assert [o.key for o in channel.asked[0].options] == ["+close"] and channel.asked[0].default == 0
    assert channel.text == ""


_THREE = {n: Touch(False, True, None) for n in ("stub", "ask", "mob")}


@pytest.mark.parametrize(("touched", "parked", "waiting", "label"), [
    ({}, 0, 0, "Close"),
    ({}, 2, 1, "Close"),
    ({"stub": _THREE["stub"]}, 0, 0, "Close — restarts to apply (1 plugin)"),
    (_THREE, 0, 0, "Close — restarts to apply (3 plugins)"),
    (_THREE, 1, 0,
     "Close — restart deferred: 1 ask to answer after closing (in /pending), then open and close Plugins"),
    (_THREE, 3, 0,
     "Close — restart deferred: 3 asks to answer after closing (in /pending), then open and close Plugins"),
    (_THREE, 0, 1,
     "Close — restart deferred: 1 ask to answer after closing (shown next), then open and close Plugins"),
    (_THREE, 0, 3,
     "Close — restart deferred: 3 asks to answer after closing (shown next), then open and close Plugins"),
    (_THREE, 1, 3,
     "Close — restart deferred: 4 asks to answer after closing (in /pending and shown next), "
     "then open and close Plugins"),
], ids=["plain", "plain_with_asks_and_nothing_touched", "restarts_1", "restarts_3", "blocked_parked_1",
        "blocked_parked_3", "blocked_waiting_1", "blocked_waiting_3", "blocked_both"])
async def test_close_label(layers, touched, parked, waiting, label) -> None:
    channel = FakeChannel("+close")

    await _run(layers, channel, touched=dict(touched), parked=parked, waiting=waiting)

    assert channel.asked[0].options[-1].label == label


@pytest.mark.parametrize(("parked", "line"), [
    (1, "1 ask parked — answer in /pending after closing; the keys that answer asks are off here"),
    (3, "3 asks parked — answer in /pending after closing; the keys that answer asks are off here"),
])
async def test_asks_pending_prints_once_at_open_and_never_again_in_that_open(layers, parked, line) -> None:
    channel = FakeChannel("1", CANCEL, "1", CANCEL, "+close")

    await _run(layers, channel, parked=parked)

    assert channel.lines == [line] and len(channel.asked) == 5


async def test_asks_pending_is_silent_with_nothing_parked_even_when_asks_wait(layers) -> None:
    channel = FakeChannel("+close")

    await _run(layers, channel, waiting=2)

    assert channel.text == ""


async def test_asks_pending_with_a_call_parked_a_pending_change_defers_with_the_blocked_close(layers) -> None:
    channel = FakeChannel("+close")

    assert await _run(layers, channel, touched={"stub": Touch(False, True, None)}, parked=1) is False

    assert not channel.closed and "restart deferred: 1 ask" in channel.asked[0].options[-1].label


def _flow(layers) -> plugins_interview.Flow:
    channel = FakeChannel()
    return plugins_interview.Flow(channel, _session(layers), _should_reload(), channel.console, lambda: None,
                                  None, None, ReloadPlan(frozenset(), 0, 0))


def press() -> None:
    os.kill(os.getpid(), signal.SIGINT)


async def _hang() -> None:
    await asyncio.sleep(100)


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit, click.Abort, click.exceptions.Exit, EOFError])
async def test_guarded_a_stop_is_the_sentinel_and_leaves_the_session_task_alone(layers, exc) -> None:
    async def node():
        raise exc()

    assert await plugins_interview.guarded(_flow(layers), node) is plugins_interview._STOP
    assert asyncio.current_task().cancelling() == 0


@pytest.mark.parametrize(("phase", "install"), [("question_installed", "done"), ("question", "none")])
async def test_guarded_a_refusal_is_the_refused_result(layers, phase, install) -> None:
    async def node():
        raise Refusal("no", ())

    got = await plugins_interview.guarded(_flow(layers), node, names=("stub",), progress=PassProgress(phase=phase))

    assert got == plugins_cmd.refused(_session(layers), "stub", "no", install)
    assert (got.kind, got.install, got.detail) == ("refused", install, "no")


async def test_guarded_an_error_is_abandoned_with_its_first_line_only(layers) -> None:
    async def node():
        raise RuntimeError("boom\nsecond")

    with pytest.raises(plugins_interview._Abandon) as caught:
        await plugins_interview.guarded(_flow(layers), node)

    assert caught.value.error == "RuntimeError: boom"


async def test_guarded_a_refusal_with_no_name_to_report_is_an_error(layers) -> None:
    async def node():
        raise Refusal("no", ())

    with pytest.raises(plugins_interview._Abandon) as caught:
        await plugins_interview.guarded(_flow(layers), node)

    assert caught.value.error == "Refusal: no"


@pytest.mark.usefixtures("default_sigint")
def test_guarded_one_press_stops_a_hanging_node_and_restores_the_handler(layers) -> None:
    async def scenario():
        before = signal.getsignal(signal.SIGINT)
        task = []

        async def node():
            task.append(asyncio.current_task())
            await _hang()

        asyncio.get_running_loop().call_soon(press)
        got = await plugins_interview.guarded(_flow(layers), node)
        return got, task[0].cancelled(), signal.getsignal(signal.SIGINT) is before, asyncio.current_task().cancelling()

    assert asyncio.run(scenario()) == (plugins_interview._STOP, True, True, 0)


@pytest.mark.usefixtures("default_sigint")
def test_guarded_a_parent_cancel_propagates_and_restores_the_handler(layers) -> None:
    async def scenario():
        before = signal.getsignal(signal.SIGINT)
        parent = asyncio.ensure_future(plugins_interview.guarded(_flow(layers), _hang))
        await asyncio.sleep(0.05)
        parent.cancel()
        with pytest.raises(asyncio.CancelledError):
            await parent
        return signal.getsignal(signal.SIGINT) is before

    assert asyncio.run(scenario())


@pytest.mark.usefixtures("default_sigint")
def test_guarded_a_parent_cancel_during_the_root_leaves_run_with_nothing_printed(layers) -> None:
    async def scenario():
        before, channel = signal.getsignal(signal.SIGINT), FakeChannel(HANG)
        parent = asyncio.ensure_future(_run(layers, channel))
        await asyncio.sleep(0.05)
        parent.cancel()
        with pytest.raises(asyncio.CancelledError):
            await parent
        return channel.text, channel.closed, signal.getsignal(signal.SIGINT) is before

    assert asyncio.run(scenario()) == ("", False, True)


@pytest.mark.usefixtures("default_sigint")
@pytest.mark.parametrize(("touched", "reloads"), [({}, False), ({"stub": Touch(False, True, None)}, True)],
                         ids=["nothing_touched", "a_plugin_will_load"])
def test_guarded_an_external_sigint_on_the_root_runs_the_close_time_computation(layers, touched, reloads) -> None:
    async def scenario():
        before = signal.getsignal(signal.SIGINT)
        asyncio.get_running_loop().call_later(0.05, press)
        got = await _run(layers, FakeChannel(HANG), touched=touched)
        return got, signal.getsignal(signal.SIGINT) is before

    assert asyncio.run(scenario()) == (reloads, True)


async def test_guarded_a_stop_while_resolving_the_root_closes_it(layers, monkeypatch) -> None:
    def stop(loader, **kw):
        raise EOFError

    monkeypatch.setattr(resolve_module, "resolve", stop)
    channel = FakeChannel()

    assert await _run(layers, channel) is False

    assert channel.text == "" and channel.asked == []


_TOO_SMALL = "1 plugin change saved, not applied: the window got too small. Enlarge the window, then open /plugins and close it"
_NO_DRAW = "1 plugin change saved, not applied: the screen could not be drawn. Quit and start localharness again to apply"
_ONE = {"stub": Touch(False, True, None)}


@pytest.mark.parametrize(("answer", "line"), [
    (Unreached(Unreachable.TOO_SMALL), _TOO_SMALL),
    (UNREACHED, _NO_DRAW),
    (Unreached(Unreachable.DUMB), _NO_DRAW),
], ids=["too_small", "no_tty", "dumb"])
async def test_not_applied_an_unreached_node_with_changes_prints_the_one_line_before_the_span_exits(
        layers, answer, line) -> None:
    channel = FakeChannel(answer)

    assert await _run(layers, channel, touched=dict(_ONE)) is False

    assert channel.lines == [line] and channel.out_at_exit == len(channel.text) and not channel.closed


@pytest.mark.parametrize("why", list(Unreachable), ids=lambda why: why.name.lower())
async def test_unreached_with_no_changes_prints_unavailable_and_returns_false(layers, why) -> None:
    channel = FakeChannel(Unreached(why))

    assert await _run(layers, channel) is False

    assert channel.lines == [plugins_interview.unavailable(why)]


@pytest.mark.parametrize(("why", "line"), [
    (Unreachable.TOO_SMALL, "Plugins can't be drawn here — enlarge the window, then open /plugins again"),
    (Unreachable.NO_TTY, "Plugins can't be drawn here — use /plugins enable <name>"),
    (Unreachable.ERROR, "Plugins can't be drawn here — use /plugins enable <name>"),
])
def test_unreached_unavailable_texts(why, line) -> None:
    assert plugins_interview.unavailable(why) == line


@pytest.mark.parametrize(("touched", "parked", "waiting", "line"), [
    (_ONE, 1, 0, "1 plugin change saved, not applied: a restart would drop 1 ask. "
                 "Answer the asks in /pending, then open /plugins and close it"),
    (_ONE, 0, 3, "1 plugin change saved, not applied: a restart would drop 3 asks. "
                 "Answer the asks shown next, then open /plugins and close it"),
    (_THREE, 2, 1, "3 plugin changes saved, not applied: a restart would drop 3 asks. "
                   "Answer the asks in /pending and shown next, then open /plugins and close it"),
], ids=["parked", "waiting", "both"])
async def test_not_applied_a_close_with_asks_pending_prints_the_asks_case(layers, touched, parked, waiting, line) -> None:
    channel = FakeChannel("+close")

    assert await _run(layers, channel, touched=dict(touched), parked=parked, waiting=waiting) is False

    assert channel.lines[-1:] == [line] and not channel.closed
    assert len([ln for ln in channel.lines if "saved, not applied" in ln]) == 1


async def test_not_applied_an_unreached_node_while_asks_are_pending_prints_only_the_unreached_row(layers) -> None:
    channel = FakeChannel(Unreached(Unreachable.TOO_SMALL))

    await _run(layers, channel, touched=dict(_ONE), parked=1, waiting=2)

    assert [ln for ln in channel.lines if "saved, not applied" in ln] == [_TOO_SMALL]
    assert "restart would drop" not in channel.text


async def test_not_applied_a_close_with_nothing_touched_and_asks_waiting_prints_nothing(layers) -> None:
    channel = FakeChannel("+close")

    assert await _run(layers, channel, waiting=2) is False

    assert channel.text == ""


def _failing(monkeypatch, message: str, exc=RuntimeError) -> None:
    def fail(loader, **kw):
        raise exc(message)

    monkeypatch.setattr(resolve_module, "resolve", fail)


async def test_closed_on_error_with_changes_prints_the_closed_early_case(layers, monkeypatch) -> None:
    _failing(monkeypatch, "boom")
    channel = FakeChannel()

    assert await _run(layers, channel, touched=dict(_ONE)) is False

    assert channel.lines == ["1 plugin change saved, not applied: Plugins closed on an error: RuntimeError: boom. "
                             "Quit and start localharness again to apply"]


async def test_closed_on_error_with_no_changes_prints_one_clean_line(layers, monkeypatch) -> None:
    _failing(monkeypatch, "boom\x1b]0;t\x07 https://u:p@h.test/x\nsecond")
    channel = FakeChannel()

    assert await _run(layers, channel) is False

    (line,) = channel.lines
    assert line.startswith("Plugins closed on an error: RuntimeError: boom") and "https://h.test/x" in line
    assert not any(bad in line for bad in ("\x1b", "\x07", "u:p@", "second"))
    assert channel.out_at_exit == len(channel.text)


async def test_closed_on_error_an_error_outside_any_node_closes_plugins_with_its_line(layers) -> None:
    class Broken(FakeChannel):
        async def run_interview(self, *args, **kw):
            return None

    channel = Broken()

    assert await _run(layers, channel) is False

    (line,) = channel.lines
    assert line.startswith("Plugins closed on an error: AttributeError: ")


_NAMED = frozenset({"p"})


def _flat(alias) -> set[str]:
    return {m for arm in get_args(alias) for m in get_args(arm)}


def _outcome(kind, *, running=False, names=_NAMED, parked=0, progress=None, key="", **kw):
    session = PluginSession(Path("/nowhere"), None, frozenset({"p"} if running else ()), {}, {})
    result = PassResult(kind, kw.pop("step", StepOutcome("p", True)), **kw)
    plan = ReloadPlan(frozenset(names), parked, 0)
    return plugins_interview.result_line("p", result, session, plan, progress or PassProgress(), key=key)


_FAILED = StepOutcome("p", True, failed_check="no answer")
_SKIPPED = StepOutcome("p", True, skipped_check="not paired")
_PROJECT = Path("/w/config.yaml")


def test_result_line_kinds_are_the_outcome_keys_plus_the_two_block_kinds() -> None:
    assert set(get_args(Kind)) == set(plugins_interview.RESULT_OUTCOMES) | {"install_skipped", "enabled_manual"}
    assert plugins_interview.RESULT == "{name}: {outcome}"
    assert plugins_interview.RESULT_OUTCOMES["override"] is OVERRIDE_LINE


@pytest.mark.parametrize(("kind", "kw", "line"), [
    ("enabled", dict(), f"enabled; starts {CLOSING}"),
    ("enabled", dict(parked=1), f"enabled; starts {LATER}"),
    ("enabled", dict(running=True, names=_NAMED), f"enabled; applies {CLOSING}"),
    ("enabled", dict(running=True, names=()), "enabled; already running"),
    ("enabled_unconfigured", dict(questions="skipped"),
     "enabled, not set up — setup questions skipped; Re-run setup to finish, then it starts"),
    ("enabled_unconfigured", dict(running=True, names=_NAMED, questions="skipped"),
     f"not set up — setup questions skipped; Re-run setup to finish; stops {CLOSING}"),
    ("enabled_unconfigured", dict(detail="unconfigured — set ask.url"),
     "enabled, not set up — set ask.url; Re-run setup to finish, then it starts"),
    ("enabled_unconfigured", dict(detail="set ask.url", running=True, names=_NAMED, parked=2),
     f"not set up — set ask.url; Re-run setup to finish; stops {LATER}"),
    ("enabled_unconfigured", dict(), "enabled, not set up — setup questions skipped; Re-run setup to finish, then it starts"),
    ("check_failed", dict(step=_FAILED), f"enabled, but its check failed: no answer; starts {CLOSING}"),
    ("check_failed", dict(step=_FAILED, detail="from detail"), f"enabled, but its check failed: from detail; starts {CLOSING}"),
    ("check_failed", dict(step=_FAILED, running=True, names=_NAMED),
     f"enabled, but its check failed: no answer; applies {CLOSING}"),
    ("check_failed", dict(step=_FAILED, running=True, names=()), "enabled, but its check failed: no answer; already running"),
    ("check_failed_noload", dict(detail="no answer"), "enabled, but not loading — no answer; Re-run setup to retry"),
    ("check_failed_noload", dict(detail="no answer", running=True, names=_NAMED),
     f"enabled, but not loading — no answer; Re-run setup to retry; stops {CLOSING}"),
    ("check_skipped", dict(step=_SKIPPED), f"enabled — not paired; starts {CLOSING}"),
    ("check_skipped", dict(step=_SKIPPED, running=True, names=()), "enabled — not paired; already running"),
    ("check_skipped", dict(step=_SKIPPED, running=True, names=_NAMED), f"enabled — not paired; applies {CLOSING}"),
    ("enabled_not_loading", dict(detail="needs localharness[mobile]"), "enabled — not loading: needs localharness[mobile]"),
    ("enabled_not_loading", dict(detail="import broke", running=True, names=_NAMED),
     f"enabled — not loading: import broke; stops {CLOSING}"),
    ("rerun_ok", dict(), "setup re-run — check passed"),
    ("rerun_ok", dict(loads=True), f"setup re-run — check passed; starts {CLOSING}"),
    ("rerun_ok", dict(loads=True, parked=1), f"setup re-run — check passed; starts {LATER}"),
    ("rerun_ok", dict(running=True, names=_NAMED, rebind=True), f"setup re-run — check passed; applies {CLOSING}"),
    ("rerun_ok", dict(running=True, loads=True, names=()), "setup re-run — check passed"),
    ("rerun_failed", dict(step=_FAILED), "setup re-run — check failed: no answer"),
    ("rerun_failed", dict(step=_FAILED, loads=True), f"setup re-run — check failed: no answer; starts {CLOSING}"),
    ("rerun_failed", dict(step=_FAILED, running=True, names=_NAMED, rebind=True),
     f"setup re-run — check failed: no answer; applies {CLOSING}"),
    ("rerun_blocked", dict(detail=lifecycle.DOWNLOAD_RUNNING), f"setup not re-run — {lifecycle.DOWNLOAD_RUNNING}"),
    ("disabled", dict(running=True, names=_NAMED), f"off {CLOSING}"),
    ("disabled", dict(running=True, names=_NAMED, parked=1), f"off {LATER}"),
    ("disabled", dict(), "off"),
    ("saved", dict(loads=True), f"color saved — applies {CLOSING}"),
    ("saved", dict(loads=True, parked=1), f"color saved — applies {LATER}"),
    ("saved", dict(), "color saved — applies when it starts"),
    ("saved_overridden", dict(override=(None, _PROJECT)),
     "color saved on this machine, but this project's /w/config.yaml still wins here"),
    ("declined", dict(), "stays off"),
    ("refused", dict(detail="boom"), "settings not changed — boom"),
    ("refused", dict(detail="boom", install="skipped"), "settings not changed — boom"),
    ("refused", dict(detail="boom", install="done"), "settings not changed — boom (package installed — Enable to finish)"),
    ("install_skipped", dict(), None),
    ("enabled_manual", dict(), None),
])
def test_result_line_each_kind(kind, kw, line) -> None:
    assert _outcome(kind, key="color", **kw) == line


def test_result_line_refused_after_an_install_names_re_run_setup_when_the_plugin_was_already_enabled() -> None:
    line = _outcome("refused", detail="boom", install="done", progress=PassProgress(written=True))

    assert line == "settings not changed — boom (package installed — Re-run setup to finish)"


@pytest.mark.parametrize(("wrote", "saved"), [(False, ""), (True, "saved on this machine, but ")])
@pytest.mark.parametrize(("state", "flip"), [("on", "disable"), ("off", "enable")])
def test_result_line_override_renders_the_one_sentence_and_saves_only_when_the_pass_wrote(wrote, saved, state, flip) -> None:
    line = _outcome("override", override=(state, _PROJECT), progress=PassProgress(wrote=wrote))

    assert line == OVERRIDE_LINE.format(saved=saved, state=state, file=_PROJECT, flip=flip, name="p")
    assert line.startswith(saved + "this project turns it ")


@pytest.mark.parametrize("kind", ["enabled_unconfigured", "check_failed_noload", "enabled_not_loading", "rerun_blocked", "refused"])
def test_result_line_a_detail_is_stripped_and_redacted(kind) -> None:
    line = _outcome(kind, detail="boom\x1b]0;t\x07 https://u:p@h.test/x")

    assert "https://h.test/x" in line
    assert not any(bad in line for bad in ("\x1b", "\x07", "u:p@"))


@pytest.mark.parametrize("step", [
    StepOutcome("p", True, failed_check="boom\x1b[2J https://u:p@h.test/x"),
    StepOutcome("p", True, skipped_check="boom\x1b[2J https://u:p@h.test/x"),
])
def test_result_line_a_step_detail_is_stripped_and_redacted(step) -> None:
    kind = "check_failed" if step.failed_check else "check_skipped"
    line = _outcome(kind, step=step)

    assert "https://h.test/x" in line and "\x1b" not in line and "u:p@" not in line


_TAIL_PLAN = ReloadPlan(frozenset({"p"}), 0, 0)


def _tail(progress, *, names=("p",), plan=_TAIL_PLAN, running=(), touched=None, **kw):
    session = PluginSession(Path("/nowhere"), None, frozenset(running), {}, {}, touched or {})
    return plugins_interview.stop_tail(names, progress, session, plan, **kw)


def test_stop_tail_keys_are_the_tail_keys() -> None:
    assert _flat(TailKey) == set(plugins_interview.STOP_TAILS)
    assert plugins_interview.STOP_LINE == "{names}: stopped — {tail}"
    assert plugins_interview.TURN_SEVERAL_ON == "Turn several on…"


@pytest.mark.parametrize("phase", ["consent", "question", "edit", "disable"])
def test_stop_tail_changes_nothing_before_a_write(phase) -> None:
    assert _tail(PassProgress(phase=phase)) == "nothing changed"


@pytest.mark.parametrize(("written", "tail"), [
    (False, "install interrupted — Enable to retry"),
    (True, "install interrupted — Install and set up to retry"),
])
def test_stop_tail_install(written, tail) -> None:
    assert _tail(PassProgress(phase="install", written=written)) == tail


def test_stop_tail_install_batch_names_the_multi_select_and_ignores_written() -> None:
    progress = PassProgress(phase="install", written=True)

    assert _tail(progress, names=("a", "b"), in_batch=True) == "install interrupted — pick them again under Turn several on…"


@pytest.mark.parametrize(("written", "tail"), [
    (False, "package installed, not set up — Enable to finish"),
    (True, "package installed, not set up — Re-run setup to finish"),
])
def test_stop_tail_question_installed(written, tail) -> None:
    assert _tail(PassProgress(phase="question_installed", written=written)) == tail


@pytest.mark.parametrize(("phase", "text"), [
    ("download", "enabled, but its download did not finish — Re-run setup"),
    ("check", "enabled, but not checked — Re-run setup"),
])
@pytest.mark.parametrize(("busy", "retry"), [(False, ""), (True, " in a minute")])
def test_stop_tail_download_and_check_retry_in_a_minute_only_while_downloading(monkeypatch, phase, text, busy, retry) -> None:
    monkeypatch.setattr(lifecycle, "downloading", lambda name: busy and name == "p")

    assert _tail(PassProgress(phase=phase, wrote=True)) == text + retry


def test_stop_tail_download_asks_downloading_of_the_first_name_only(monkeypatch) -> None:
    monkeypatch.setattr(lifecycle, "downloading", lambda name: name == "b")

    assert _tail(PassProgress(phase="download"), names=("a", "b")).endswith("Re-run setup")


@pytest.mark.parametrize(("running", "touched", "plan", "applies"), [
    ((), {}, _TAIL_PLAN, "when it starts"),
    (("p",), {}, _TAIL_PLAN, CLOSING),
    (("p",), {"p": Touch(True, True, None)}, ReloadPlan(frozenset({"p"}), 1, 0), LATER),
    ((), {"p": Touch(False, True, None)}, _TAIL_PLAN, CLOSING),
    (("p",), {"p": Touch(False, False, None)}, _TAIL_PLAN, "when it starts"),
])
def test_stop_tail_edit_written_applies_from_the_stored_or_seeded_touch(running, touched, plan, applies) -> None:
    tail = _tail(PassProgress(phase="edit", wrote=True), running=running, touched=touched, plan=plan, key="color")

    assert tail == f"color saved — applies {applies}; the audit entry may be missing"


def test_stop_tail_edit_written_for_an_overridden_write_says_where_the_value_lost() -> None:
    progress = PassProgress(phase="edit", wrote=True, overridden=True)

    assert _tail(progress, key="color", file=_PROJECT) == (
        "color saved on this machine, but this project's /w/config.yaml still wins here; the audit entry may be missing")


def test_stop_tail_a_stop_line_joins_every_name() -> None:
    tail = _tail(PassProgress(phase="install"), names=("a", "b"), in_batch=True)

    assert plugins_interview.STOP_LINE.format(names="a, b", tail=tail) == (
        "a, b: stopped — install interrupted — pick them again under Turn several on…")


_MANUAL = InstallRoute("x", None, Manual("", False, "uv sync --extra mobile"))
_ARGV = InstallRoute("x", ["pip", "install", "x"], None)


def _patched(monkeypatch, *, states=None, enabled=None, extra=()) -> None:
    real = resolve_module.resolve

    def patched(loader, **kw):
        found = real(loader, **kw)
        entries = tuple(replace(e, state=states[e.name]) if e.name in (states or {}) else e
                        for e in found.plan.entries) + tuple(extra)
        return replace(found, plan=replace(found.plan, entries=entries), enabled={**found.enabled, **(enabled or {})})

    monkeypatch.setattr(resolve_module, "resolve", patched)


def _on_disk(layers, **flags) -> None:
    (layers.g / "overrides.yaml").write_text(yaml.safe_dump({n: {"enabled": v} for n, v in flags.items()}), encoding="utf-8")


def _labels(asked) -> list[str]:
    return [o.label for o in asked.options]


async def test_menu_a_picked_row_asks_its_state_with_back_and_distinct_keys(layers) -> None:
    _on_disk(layers, stub=True)
    channel = FakeChannel("0", CANCEL, "+close")

    await _run(layers, channel)

    menu = channel.asked[1]
    assert menu.question == "stub — on — not running; Re-run setup starts it (Esc = back)"
    assert (menu.default, menu.cancel) == (0, "back")
    assert _labels(menu) == ["Re-run setup", "Disable", "Info"] and len({o.key for o in menu.options}) == 3
    assert channel.asked[2].question == PLUGINS_QUESTION


async def test_menu_is_resolved_again_when_entered(layers) -> None:
    def enable_ask(options):
        _on_disk(layers, ask=True)
        return "1"

    channel = FakeChannel(enable_ask, CANCEL, "+close")

    await _run(layers, channel)

    assert channel.asked[0].options[1].hint == "off"
    assert channel.asked[1].question == "ask — on — not running; Re-run setup starts it (Esc = back)"


async def test_menu_a_plugin_that_vanished_sends_the_person_back_to_the_root(layers, monkeypatch) -> None:
    real, calls = resolve_module.resolve, []

    def vanish(loader, **kw):
        found = real(loader, **kw)
        calls.append(1)
        if len(calls) == 1:
            return found
        return replace(found, plan=replace(found.plan, entries=tuple(e for e in found.plan.entries if e.name != "stub")))

    monkeypatch.setattr(resolve_module, "resolve", vanish)
    channel = FakeChannel("0", "+close")

    assert await _run(layers, channel) is False

    assert [a.question for a in channel.asked] == [PLUGINS_QUESTION] * 2 and channel.text == ""


@pytest.mark.parametrize(("state", "enabled", "route", "labels"), [
    ("on", True, None, ["Re-run setup", "Disable", "Info"]),
    ("off", False, None, ["Enable", "Info"]),
    ("available", False, None, ["Enable", "Info"]),
    ("needs-extra", True, _MANUAL, ["Show install step", "Disable", "Info"]),
    ("needs-extra", True, None, ["Disable", "Info"]),
    ("needs-extra", True, _ARGV, ["Install and set up", "Disable", "Info"]),
    ("failed", True, None, ["Disable", "Info"]),
    ("refused", False, None, ["Info"]),
    ("skipped", False, None, ["Info"]),
])
def test_menu_actions_follow_the_state_the_flag_and_the_route(layers, monkeypatch, state, enabled, route, labels) -> None:
    _patched(monkeypatch, enabled={"stub": enabled})
    actions = plugins_interview.actions_for(_entry(state, name="stub"), _session(layers), route)

    assert [o.label for o, _ in actions] == labels
    assert len({o.key for o, _ in actions}) == len(actions)


async def test_menu_a_third_party_plugin_that_needs_an_extra_offers_disable_and_info_only(layers, monkeypatch) -> None:
    _on_disk(layers, mob=True)
    _patched(monkeypatch, states={"mob": "needs-extra"})
    monkeypatch.setattr(plugins_cmd, "install_route_for", lambda entry: None)
    channel = FakeChannel("2", CANCEL, "+close")

    await _run(layers, channel)

    assert _labels(channel.asked[1]) == ["Disable", "Info"]


async def test_disable_a_running_plugin_says_off_at_close_and_the_close_label_restarts(layers) -> None:
    _on_disk(layers, stub=True)
    touched: dict = {}
    channel = FakeChannel("0", Label("Disable"), "+close")

    assert await _run(layers, channel, running={"stub"}, touched=touched) is True

    assert channel.lines == ["stub: off when you close Plugins"] and channel.closed
    assert touched == {"stub": Touch(rebind=False, loads=False, probe=None)}
    assert channel.asked[-1].options[-1].label == "Close — restarts to apply (1 plugin)"
    assert yaml.safe_load((layers.g / "overrides.yaml").read_text(encoding="utf-8")) == {"stub": {"enabled": False}}


async def test_disable_a_plugin_that_is_not_running_says_off_and_the_close_label_stays(layers) -> None:
    _on_disk(layers, stub=True)
    channel = FakeChannel("0", Label("Disable"), "+close")

    assert await _run(layers, channel) is False

    assert channel.lines == ["stub: off"] and not channel.closed
    assert channel.asked[-1].options[-1].label == "Close"


async def test_disable_under_a_project_that_turns_it_on_saves_and_says_which_file_wins(layers) -> None:
    (layers.ws / "config.yaml").write_text(yaml.safe_dump({"stub": {"enabled": True}}), encoding="utf-8")
    touched: dict = {}
    channel = FakeChannel("0", Label("Disable"), "+close")

    assert await _run(layers, channel, running={"stub"}, touched=touched) is False

    assert channel.lines == [
        f"stub: saved on this machine, but this project turns it on ({layers.ws / 'config.yaml'}) and that "
        "still wins — localharness plugins disable stub --workspace changes it"]
    assert touched["stub"].loads is True and channel.asked[-1].options[-1].label == "Close"


async def test_disable_a_stop_changes_nothing_and_stores_no_touch(layers, monkeypatch) -> None:
    async def stop(*args, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(plugins_interview, "disable_pass", stop)
    _on_disk(layers, stub=True)
    before, touched = (layers.g / "overrides.yaml").read_bytes(), {}
    channel = FakeChannel("0", Label("Disable"), "+close")

    await _run(layers, channel, running={"stub"}, touched=touched)

    assert channel.lines == ["stub: stopped — nothing changed"] and touched == {}
    assert (layers.g / "overrides.yaml").read_bytes() == before
    assert channel.asked[-1].options[-1].label == "Close"


async def test_disable_a_refusal_prints_one_line_and_leaves_the_close_label(layers, monkeypatch) -> None:
    async def refuse(*args, **kw):
        raise Refusal("stub cannot be turned off: x", ())

    monkeypatch.setattr(plugins_interview, "disable_pass", refuse)
    _on_disk(layers, stub=True)
    touched: dict = {}
    channel = FakeChannel("0", Label("Disable"), "+close")

    await _run(layers, channel, running={"stub"}, touched=touched)

    assert channel.lines == ["stub: settings not changed — stub cannot be turned off: x"] and touched == {}
    assert channel.asked[-1].options[-1].label == "Close"


async def test_disable_prints_no_next_steps(layers) -> None:
    _on_disk(layers, ask=True)
    channel = FakeChannel("1", Label("Disable"), "+close")

    await _run(layers, channel)

    assert channel.lines == ["ask: off"] and "ASK-NEXT" not in channel.text


async def test_info_prints_the_builds_failure_and_asks_the_same_menu_again(layers) -> None:
    _on_disk(layers, stub=True)
    channel = FakeChannel("0", Label("Info"), CANCEL, "+close")

    await _run(layers, channel, failed={"stub": "import broke"})

    assert "import broke" in channel.text
    assert len(channel.asked) == 4 and channel.asked[1].question == channel.asked[2].question


async def test_info_prints_the_last_probe_detail(layers) -> None:
    _on_disk(layers, stub=True)
    channel = FakeChannel("0", Label("Info"), CANCEL, "+close")

    await _run(layers, channel, touched={"stub": Touch(False, False, ("failed", "no answer"))})

    assert "no answer" in channel.text


async def test_info_on_a_refused_entry_prints_its_name_without_the_escape_sequence(layers, monkeypatch) -> None:
    evil = PlanEntry("evil\x1b[31mx", False, "refused", "entry point", "(not loaded)", "bad \x1b[31m reason")
    _patched(monkeypatch, extra=(evil,))
    channel = FakeChannel("3", Label("Info"), CANCEL, "+close")

    await _run(layers, channel)

    assert _labels(channel.asked[1]) == ["Info"]
    assert "evil" in channel.text and "\x1b" not in channel.text


async def test_show_install_prints_the_manual_block_and_writes_nothing(layers, monkeypatch) -> None:
    _on_disk(layers, mob=True)
    _patched(monkeypatch, states={"mob": "needs-extra"})
    monkeypatch.setattr(plugins_cmd, "install_route_for", lambda entry: _MANUAL)
    before, touched = (layers.g / "overrides.yaml").read_bytes(), {}
    channel = FakeChannel("2", Label("Show install step"), "+close")

    await _run(layers, channel, touched=touched)

    assert channel.asked[1].question == "mob — needs localharness[mobile] — Show install step (Esc = back)"
    assert channel.lines == ["mob: enabled in config; this install must be done by hand.", "In another terminal run:",
                             "uv sync --extra mobile", "then Re-run setup under /plugins → mob"]
    assert (layers.g / "overrides.yaml").read_bytes() == before and touched == {}


def _flow_for(layers, *, running=(), touched=None):
    channel = FakeChannel()
    session = _session(layers, running=frozenset(running), touched={} if touched is None else touched)
    flow = plugins_interview.Flow(channel, session, _should_reload(running), channel.console, lambda: None, None, None,
                                  ReloadPlan(frozenset(), 0, 0))
    return flow, channel


async def test_stop_path_a_written_edit_records_the_touch_then_says_when_it_applies(layers) -> None:
    touched: dict = {}
    flow, channel = _flow_for(layers, running={"stub"}, touched=touched)
    progress = PassProgress()

    async def node():
        progress.phase, progress.wrote = "edit", True
        raise KeyboardInterrupt

    got = await plugins_interview.guarded(flow, node, names=("stub",), progress=progress, key="color")

    assert got is plugins_interview._STOP and touched["stub"] == Touch(True, True, None)
    assert channel.lines == ["stub: stopped — color saved — applies when you close Plugins; the audit entry may be missing"]
    after = FakeChannel("+close")
    await run(after, _session(layers, running=frozenset({"stub"}), touched=touched), _should_reload({"stub"}))
    assert after.asked[0].options[-1].label == "Close — restarts to apply (1 plugin)"


@pytest.mark.parametrize("phase", ["consent", "question", "edit", "disable"])
async def test_stop_path_before_a_write_stores_no_touch(layers, phase) -> None:
    flow, channel = _flow_for(layers, running={"stub"})
    progress = PassProgress(phase=phase)

    async def node():
        raise EOFError

    await plugins_interview.guarded(flow, node, names=("stub",), progress=progress)

    assert channel.lines == ["stub: stopped — nothing changed"] and flow.session.touched == {}


async def test_stop_path_a_batch_install_stop_names_every_pick(layers) -> None:
    flow, channel = _flow_for(layers)
    progress = PassProgress(phase="install")

    async def node():
        raise click.Abort

    await plugins_interview.guarded(flow, node, names=("a", "b"), progress=progress, in_batch=True)

    assert channel.lines == ["a, b: stopped — install interrupted — pick them again under Turn several on…"]


async def test_stop_path_a_node_with_no_holder_prints_nothing(layers) -> None:
    flow, channel = _flow_for(layers)

    async def node():
        raise EOFError

    assert await plugins_interview.guarded(flow, node) is plugins_interview._STOP
    assert channel.text == ""


def test_stop_path_stopping_line_prints_for_the_install_phase_only() -> None:
    console = Console(file=io.StringIO(), width=400)

    plugins_interview.stopping_line(console, ("a", "b"), PassProgress(phase="download"))
    assert console.file.getvalue() == ""
    plugins_interview.stopping_line(console, ("a", "b"), PassProgress(phase="install"))

    assert console.file.getvalue() == "a, b: stopping — pressing Ctrl-C two more times quits localharness\n"


@pytest.mark.usefixtures("default_sigint")
@pytest.mark.parametrize(("unwind", "lines"), [
    (0, ["stub: stopped — install interrupted — Enable to retry"]),
    (1.3, ["stub: stopping — pressing Ctrl-C two more times quits localharness",
           "stub: stopped — install interrupted — Enable to retry"]),
], ids=["quick", "slow"])
def test_stop_path_stopping_prints_only_when_an_install_stop_outlasts_a_second(layers, unwind, lines) -> None:
    async def scenario():
        flow, channel = _flow_for(layers)

        async def node():
            try:
                await _hang()
            except asyncio.CancelledError:
                await asyncio.sleep(unwind)
                raise

        asyncio.get_running_loop().call_soon(press)
        await plugins_interview.guarded(flow, node, names=("stub",), progress=PassProgress(phase="install"))
        return channel.lines

    assert asyncio.run(scenario()) == lines


class Said:
    """Replaces the module's prompt and confirm: answers in order, records what was asked."""

    def __init__(self, monkeypatch, answers=(), yes=()) -> None:
        self.answers, self.yes, self.asked, self.confirmed = list(answers), list(yes), [], []

        async def prompt(text, *, console, default="", secret=False):
            self.asked.append((text, secret))
            return self._next(self.answers)

        async def confirm(text, *, console, default, tail=""):
            self.confirmed.append((text, default, tail))
            return self._next(self.yes)

        monkeypatch.setattr(plugins_interview, "prompt_line", prompt)
        monkeypatch.setattr(plugins_interview, "confirm_line", confirm)

    @staticmethod
    def _next(script):
        answer = script.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer() if callable(answer) else answer


def _written(layers) -> dict:
    return yaml.safe_load((layers.g / "overrides.yaml").read_text(encoding="utf-8"))


def _enable_script(name, label="Enable"):
    return (Label(name), Label(label), "+close")


async def test_enable_installs_asks_and_checks_in_one_pass(layers, mobile, dist, monkeypatch) -> None:
    said = Said(monkeypatch, ["http://box"], [True])
    channel = FakeChannel(*_enable_script("mob"))
    touched: dict = {}

    assert await _run(layers, channel, touched=touched) is True

    assert len(mobile.spawns) == 1 and mobile.spawns[0][0][-1] == SPEC
    (text, default, tail), = said.confirmed
    assert (text, default) == ("", True) and CTRLC.format(does=CTRLC_DOES["prompt"]) in tail
    assert CTRLC.format(does=CTRLC_DOES["batch"]) not in tail and said.asked == [("Address: ", False)]
    lines = channel.lines
    assert lines[0] == CONSENT[0].format(names="mob") and f"mob: {DOWNLOAD}" in lines
    assert lines[-2:] == ["mob: enabled; starts when you close Plugins", "MOB-NEXT"]
    assert DOWNLOAD not in [text for text, *_ in said.confirmed]
    assert _written(layers) == {"mob": {"enabled": True, "url": "http://box"}}
    assert _rows(channel.asked[-1])[2].hint == "on — starts when you close Plugins"
    assert channel.asked[-1].options[-1].label == "Close — restarts to apply (1 plugin)" and channel.closed
    assert touched["mob"].loads is True


async def test_enable_a_failed_check_prints_the_result_then_the_help_then_the_next_steps(layers, monkeypatch) -> None:
    Said(monkeypatch, ["http://box", "s3cr3t-typed"])
    touched: dict = {}
    channel = FakeChannel(*_enable_script("ask"))

    await _run(layers, channel, touched=touched)

    assert [x for x in channel.lines if x] == [
        "ask: enabled, but its check failed: no answer; starts when you close Plugins",
        "ASK-HELP", AGENT_PROMPT_LEAD, "  Run it at http://box.", "ASK-NEXT"]
    assert channel.lines[-2:] == ["", "ASK-NEXT"]
    assert "s3cr3t-typed" not in channel.text and "s3cr3t-typed" not in repr(touched)


async def test_enable_a_secret_answer_is_asked_hidden_and_never_reaches_the_console_or_touched(layers, monkeypatch) -> None:
    said = Said(monkeypatch, ["http://box-ok", "s3cr3t-typed"])
    touched: dict = {}
    channel = FakeChannel(*_enable_script("ask"))

    await _run(layers, channel, touched=touched)

    assert said.asked == [("Address: ", False), ("Token: ", True)]
    assert "s3cr3t-typed" not in channel.text + repr(touched) and "http://box-ok" not in channel.text
    assert channel.lines == ["ask: enabled; starts when you close Plugins", "ASK-NEXT"]


async def test_enable_a_declined_install_prints_the_skipped_block_once_then_the_next_steps(layers, mobile, dist, monkeypatch) -> None:
    Said(monkeypatch, yes=[False])
    touched: dict = {}
    channel = FakeChannel(*_enable_script("mob"))

    assert await _run(layers, channel, touched=touched) is False

    lines = channel.lines
    assert [x for x in lines if x.startswith("mob: enabled in config, package not installed")] == [lines[4]]
    assert lines[-1] == "MOB-NEXT" and not any(x.startswith("mob: enabled") and "package" not in x for x in lines)
    assert mobile.spawns == [] and touched["mob"].loads is False
    assert channel.asked[-1].options[-1].label == "Close"
    assert _written(layers) == {"mob": {"enabled": True}}


async def test_enable_a_manual_route_prints_the_by_hand_block_and_no_result_line(layers, mobile, monkeypatch) -> None:
    monkeypatch.setattr(plugins_cmd, "install_route_for", lambda entry: _MANUAL)
    monkeypatch.setattr(plugins_cmd, "extra_for", lambda entry: "mobile")
    touched: dict = {}
    channel = FakeChannel(*_enable_script("mob"))

    assert await _run(layers, channel, touched=touched) is False

    assert channel.lines == [
        "mob: enabled in config; this install must be done by hand.", "In another terminal run:",
        "uv sync --extra mobile", "then Re-run setup under /plugins → mob", "MOB-NEXT"]
    assert not touched["mob"].loads and channel.asked[-1].options[-1].label == "Close"


async def test_enable_a_plugin_with_nothing_to_set_up_runs_no_probe_and_prints_no_next_steps(layers, monkeypatch) -> None:
    def boom(*a, **kw):
        raise AssertionError("probed")

    monkeypatch.setattr(plugins_cmd, "probe", boom)
    touched: dict = {}
    channel = FakeChannel(*_enable_script("stub"))

    await _run(layers, channel, touched=touched)

    assert channel.lines == ["stub: enabled; starts when you close Plugins"] and touched["stub"].loads is True


async def test_enable_a_plugin_you_installed_asks_the_trust_question_with_the_prompt_tail_first(folder, monkeypatch) -> None:
    write_folder_plugin(folder.g, "swatch")
    said = Said(monkeypatch, yes=[True])
    touched: dict = {}
    channel = FakeChannel(*_enable_script("swatch"))

    await _run(folder, channel, touched=touched)

    assert said.confirmed == [(THIRD_PARTY_CONFIRM.format(name="swatch"), False, CTRLC.format(does=CTRLC_DOES["prompt"]))]
    assert channel.lines == ["swatch: enabled; starts when you close Plugins"] and touched["swatch"].loads is True


async def test_enable_a_no_to_the_trust_question_stays_off_and_writes_nothing(folder, monkeypatch) -> None:
    write_folder_plugin(folder.g, "swatch")
    Said(monkeypatch, yes=[False])
    touched: dict = {}
    channel = FakeChannel(*_enable_script("swatch"))

    await _run(folder, channel, touched=touched)

    assert channel.lines == ["swatch: stays off"] and touched == {}
    assert not (folder.g / "overrides.yaml").exists()


async def test_enable_a_plugin_you_installed_whose_extra_nobody_provides_is_not_loading(folder, monkeypatch) -> None:
    write_folder_plugin(folder.g, "swatch", manifest=', requires_extra="nosuchextra"')
    Said(monkeypatch, yes=[True])
    channel = FakeChannel(*_enable_script("swatch"))

    await _run(folder, channel)

    (line,) = channel.lines
    assert line.startswith("swatch: enabled — not loading: ") and "nosuchextra" in line


async def test_enable_a_plugin_you_installed_that_fails_to_import_is_not_loading(folder, monkeypatch) -> None:
    write_folder_plugin(folder.g, "swatch", prelude="raise RuntimeError('boom')")
    Said(monkeypatch, yes=[True])
    channel = FakeChannel(*_enable_script("swatch"))

    await _run(folder, channel)

    (line,) = channel.lines
    assert line.startswith("swatch: enabled — not loading: ") and "could not be imported" in line


async def test_enable_a_resolve_that_raises_after_the_install_is_one_refused_line_and_writes_nothing(
        layers, mobile, dist, monkeypatch) -> None:
    real, fired = resolve_module.resolve, []

    def once(loader, **kw):
        if mobile.installed and not fired:
            fired.append(1)
            raise RuntimeError("boom http://user:pw@host/x")
        return real(loader, **kw)

    once.__kwdefaults__ = real.__kwdefaults__
    monkeypatch.setattr(resolve_module, "resolve", once)
    Said(monkeypatch, yes=[True])
    touched: dict = {}
    channel = FakeChannel(*_enable_script("mob"))

    await _run(layers, channel, touched=touched)

    line = channel.lines[-1]
    assert line.startswith("mob: settings not changed — RuntimeError: boom") and "pw" not in line
    assert line.endswith("(package installed — Enable to finish)")
    assert not (layers.g / "overrides.yaml").exists() and touched == {}


async def test_enable_install_and_set_up_runs_the_same_pass_for_an_enabled_plugin_missing_its_extra(
        layers, mobile, dist, monkeypatch) -> None:
    _on_disk(layers, mob=True)
    said = Said(monkeypatch, ["http://box"], [True])
    touched: dict = {}
    channel = FakeChannel(*_enable_script("mob", "Install and set up"))

    assert await _run(layers, channel, touched=touched) is True

    assert _labels(channel.asked[1]) == ["Install and set up", "Disable", "Info"]
    assert len(mobile.spawns) == 1 and len(said.confirmed) == 1
    assert channel.lines[-2:] == ["mob: enabled; starts when you close Plugins", "MOB-NEXT"]


def _rerun_entry(**manifest):
    return PlanEntry("p", True, "on", "built in", "does things", "", PluginManifest(name="p", version="0.1.0", kind="tools", **manifest))


_ASKS = dict(setup=(SetupField(key="url", prompt="Address"),))


@pytest.mark.parametrize(("manifest", "kw", "offered"), [
    ({}, dict(running={"p"}), False),
    ({}, dict(), True),
    ({}, dict(running={"p"}, touched={"p": Touch(False, True, ("failed", "x"))}), True),
    ({}, dict(running={"p"}, touched={"p": Touch(False, True, ("on", ""))}), False),
    ({}, dict(running={"p"}, touched={"p": Touch(False, True, None)}), False),
    (_ASKS, dict(running={"p"}), True),
    (dict(setup_action="Download?"), dict(running={"p"}), True),
], ids=["running_no_step", "not_running", "stored_failed_probe", "stored_passing_probe", "stored_no_probe", "asks", "action"])
def test_rerun_row_is_offered_for_a_step_a_plugin_not_running_or_a_probe_that_is_not_on(layers, manifest, kw, offered) -> None:
    session = _session(layers, running=frozenset(kw.get("running", ())), touched=kw.get("touched", {}))

    labels = [o.label for o, _ in plugins_interview.actions_for(_rerun_entry(**manifest), session, None)]

    assert ("Re-run setup" in labels) is offered


@pytest.mark.parametrize("state", ["off", "available", "needs-extra", "failed", "refused"])
def test_rerun_row_is_only_for_a_plugin_that_is_on(layers, state) -> None:
    entry = replace(_rerun_entry(**_ASKS), state=state)

    assert "Re-run setup" not in [o.label for o, _ in plugins_interview.actions_for(entry, _session(layers), None)]


async def test_rerun_a_plugin_that_is_on_and_not_running_probes_and_starts_at_close(layers) -> None:
    _on_disk(layers, stub=True)
    touched: dict = {}
    channel = FakeChannel(*_enable_script("stub", "Re-run setup"))

    assert await _run(layers, channel, touched=touched) is True

    assert channel.lines == ["stub: setup re-run — check passed; starts when you close Plugins"]
    assert touched["stub"].loads is True and channel.asked[-1].options[-1].label == "Close — restarts to apply (1 plugin)"


async def test_rerun_a_healthy_running_plugin_asks_nothing_and_leaves_names_empty(layers, monkeypatch) -> None:
    _overlay(layers, "ask", url="http://box-ok", token="tok")
    said = Said(monkeypatch)
    touched: dict = {}
    channel = FakeChannel(*_enable_script("ask", "Re-run setup"))

    assert await _run(layers, channel, running={"ask"}, touched=touched) is False

    assert channel.lines == ["ask: setup re-run — check passed", "ASK-NEXT"]
    assert said.asked == [] and said.confirmed == [] and not channel.closed
    assert channel.asked[-1].options[-1].label == "Close"


async def test_rerun_answers_written_on_a_running_plugin_apply_at_close(layers, monkeypatch) -> None:
    _on_disk(layers, ask=True)
    Said(monkeypatch, ["http://box-ok", "tok"])
    channel = FakeChannel(*_enable_script("ask", "Re-run setup"))

    assert await _run(layers, channel, running={"ask"}) is True

    assert channel.lines[0] == "ask: setup re-run — check passed; applies when you close Plugins"
    assert _written(layers) == {"ask": {"enabled": True, "url": "http://box-ok", "token": "tok"}}


async def test_rerun_with_every_question_skipped_says_what_is_left_to_set(layers, monkeypatch) -> None:
    _on_disk(layers, ask=True)
    Said(monkeypatch, ["", ""])
    touched: dict = {}
    channel = FakeChannel(*_enable_script("ask", "Re-run setup"))

    await _run(layers, channel, touched=touched)

    assert channel.lines[0] == "ask: enabled, not set up — set ask.url; Re-run setup to finish, then it starts"
    assert touched["ask"].loads is False and channel.asked[-1].options[-1].label == "Close"


async def test_rerun_asks_the_plugins_own_download_confirm(layers, mobile, monkeypatch) -> None:
    mobile.installed = True
    _overlay(layers, "mob", url="http://box")
    said = Said(monkeypatch, ["http://box"], [True])
    channel = FakeChannel(*_enable_script("mob", "Re-run setup"))

    await _run(layers, channel)

    assert said.confirmed == [(DOWNLOAD, True, "")]
    assert channel.lines == ["mob: setup re-run — check passed; starts when you close Plugins", "MOB-NEXT"]


async def test_rerun_while_a_download_still_stops_hints_it_and_is_blocked_with_nothing_changed(layers, monkeypatch) -> None:
    monkeypatch.setattr(lifecycle, "downloading", lambda name: True)
    _on_disk(layers, ask=True)
    said = Said(monkeypatch)
    touched: dict = {}
    channel = FakeChannel(*_enable_script("ask", "Re-run setup"))

    await _run(layers, channel, running={"ask"}, touched=touched)

    (rerun,) = [o for o in channel.asked[1].options if o.label == "Re-run setup"]
    assert rerun.hint == "download still stopping"
    assert channel.lines == ["ask: setup not re-run — " + lifecycle.DOWNLOAD_RUNNING]
    assert channel.lines[0].endswith("quit and restart localharness") and touched == {} and said.asked == []


def _break_probe(monkeypatch, layers) -> None:
    real = lifecycle._doctor_row

    async def boom(*a, **kw):
        if (layers.g / "overrides.yaml").exists():
            raise RuntimeError("boom")
        return await real(*a, **kw)

    monkeypatch.setattr(lifecycle, "_doctor_row", boom)


async def test_rerun_a_probe_that_raises_is_a_failed_check_with_no_restart_and_a_failed_row(layers, monkeypatch) -> None:
    _on_disk(layers, stub=True)
    _break_probe(monkeypatch, layers)
    touched: dict = {}
    channel = FakeChannel(Label("stub"), Label("Re-run setup"), "+close")

    assert await _run(layers, channel, touched=touched) is False

    (line,) = channel.lines
    assert line == "stub: enabled, but not loading — could not check it now: RuntimeError: boom; Re-run setup to retry"
    assert touched["stub"].probe[0] == "failed" and touched["stub"].loads is False
    assert _rows(channel.asked[-1])[0].hint == "on — check failed; Info shows why"
    assert channel.asked[-1].options[-1].label == "Close"


async def test_enable_a_probe_that_raises_is_a_failed_check_with_the_help_and_no_restart(layers, monkeypatch) -> None:
    Said(monkeypatch, ["http://box-ok", "tok"])
    _break_probe(monkeypatch, layers)
    touched: dict = {}
    channel = FakeChannel(*_enable_script("ask"))

    assert await _run(layers, channel, touched=touched) is False

    assert channel.lines[0] == "ask: enabled, but not loading — could not check it now: RuntimeError: boom; Re-run setup to retry"
    assert "ASK-HELP" in channel.lines and "  Run it at http://box-ok." in channel.lines and channel.lines[-1] == "ASK-NEXT"
    assert touched["ask"].probe[0] == "failed" and _rows(channel.asked[-1])[1].hint == "on — check failed; Info shows why"


@pytest.mark.parametrize(("label", "enabled"), [("Enable", False), ("Re-run setup", True)])
async def test_enable_and_rerun_a_write_phase_resolve_that_raises_is_the_same_failed_check(layers, monkeypatch, label, enabled) -> None:
    if enabled:
        _on_disk(layers, ask=True)
    real, armed, fired = resolve_module.resolve, [], []

    def patched(loader, **kw):
        if armed and not fired:
            fired.append(1)
            raise RuntimeError("boom")
        return real(loader, **kw)

    patched.__kwdefaults__ = real.__kwdefaults__
    monkeypatch.setattr(resolve_module, "resolve", patched)
    Said(monkeypatch, ["http://box-ok", lambda: armed.append(1) or "tok"])
    touched: dict = {}
    channel = FakeChannel(*_enable_script("ask", label))

    assert await _run(layers, channel, touched=touched) is False

    assert channel.lines[0].startswith("ask: enabled, but not loading — could not check it now: RuntimeError: boom")
    assert "ASK-HELP" in channel.lines and "  Run it at http://127.0.0.1:1." in channel.lines
    assert touched["ask"].probe[0] == "failed" and not channel.closed


async def test_rerun_a_plugin_the_build_failed_whose_probe_still_fails_does_not_load(layers, monkeypatch) -> None:
    _overlay(layers, "ask", url="box-dead", token="tok")
    Said(monkeypatch, ["", ""])
    touched: dict = {}
    channel = FakeChannel(*_enable_script("ask", "Re-run setup"))

    assert await _run(layers, channel, failed={"ask": "import broke"}, touched=touched) is False

    assert touched["ask"].loads is False and "check passed" not in channel.text
    assert channel.lines[0].startswith("ask: enabled, but not loading — ") and "RuntimeError: dead" in channel.lines[0]
    assert channel.asked[-1].options[-1].label == "Close"


class _Crashing(_Proc):
    async def _read(self, n):
        raise KeyboardInterrupt


async def test_stop_at_the_consent_changes_nothing(layers, mobile, dist, monkeypatch) -> None:
    Said(monkeypatch, yes=[None])
    touched: dict = {}
    channel = FakeChannel(*_enable_script("mob"))

    await _run(layers, channel, touched=touched)

    assert channel.lines[-1] == "mob: stopped — nothing changed" and mobile.spawns == []
    assert touched == {} and not (layers.g / "overrides.yaml").exists()


@pytest.mark.parametrize(("enabled", "label", "again"), [
    (False, "Enable", "Enable"), (True, "Install and set up", "Install and set up")], ids=["enable", "install_row"])
async def test_stop_in_the_install_terminates_the_child_and_names_the_retry(layers, mobile, dist, monkeypatch, enabled, label, again) -> None:
    mobile.proc = _Crashing()
    if enabled:
        _on_disk(layers, mob=True)
    Said(monkeypatch, yes=[True])
    channel = FakeChannel(*_enable_script("mob", label))

    await _run(layers, channel)

    assert channel.lines[-1] == f"mob: stopped — install interrupted — {again} to retry"
    assert mobile.proc.calls == ["terminate", "wait"] and channel.asked[-1].question == PLUGINS_QUESTION


async def test_stop_at_the_first_question_after_the_install_names_the_finish(layers, mobile, dist, monkeypatch) -> None:
    Said(monkeypatch, [None], [True])
    touched: dict = {}
    channel = FakeChannel(*_enable_script("mob"))

    await _run(layers, channel, touched=touched)

    assert channel.lines[-1] == "mob: stopped — package installed, not set up — Enable to finish"
    assert mobile.installed and touched == {} and not (layers.g / "overrides.yaml").exists()


async def test_stop_at_the_first_question_changes_nothing(layers, monkeypatch) -> None:
    Said(monkeypatch, [None])
    touched: dict = {}
    channel = FakeChannel(*_enable_script("ask"))

    await _run(layers, channel, touched=touched)

    assert channel.lines == ["ask: stopped — nothing changed"] and touched == {}


@pytest.mark.parametrize(("busy", "retry"), [(False, ""), (True, " in a minute")])
async def test_stop_at_the_download_confirm_keeps_the_enable_and_names_the_rerun(layers, mobile, monkeypatch, busy, retry) -> None:
    mobile.installed = True
    monkeypatch.setattr(lifecycle, "downloading", lambda name: busy)
    Said(monkeypatch, ["http://box"], [None])
    touched: dict = {}
    channel = FakeChannel(*_enable_script("mob"))

    assert await _run(layers, channel, touched=touched) is True

    assert channel.lines[-1] == f"mob: stopped — enabled, but its download did not finish — Re-run setup{retry}"
    assert touched["mob"].loads is True and _written(layers)["mob"]["enabled"] is True
    assert channel.asked[-1].options[-1].label == "Close — restarts to apply (1 plugin)"


class _StopMob(Mob):
    def setup_action(self, ctx):
        raise KeyboardInterrupt


async def test_stop_a_keyboard_interrupt_in_the_download_thread_is_one_stop_line_and_the_session_goes_on(layers, mobile, monkeypatch) -> None:
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Stub, Ask, _StopMob))
    mobile.installed = True
    Said(monkeypatch, ["http://box"], [True])
    channel = FakeChannel(*_enable_script("mob"))

    await _run(layers, channel)

    assert channel.lines == ["mob: stopped — enabled, but its download did not finish — Re-run setup"]
    assert asyncio.current_task().cancelling() == 0 and channel.asked[-1].question == PLUGINS_QUESTION


async def test_stop_a_system_exit_in_the_probe_names_the_unchecked_enable(layers, monkeypatch) -> None:
    async def leave(*a, **kw):
        raise SystemExit

    monkeypatch.setattr(plugins_cmd, "probe", leave)
    Said(monkeypatch, ["http://box-ok", "tok"])
    touched: dict = {}
    channel = FakeChannel(*_enable_script("ask"))

    await _run(layers, channel, touched=touched)

    assert channel.lines == ["ask: stopped — enabled, but not checked — Re-run setup"] and touched["ask"].loads is True
    assert asyncio.current_task().cancelling() == 0 and channel.asked[-1].question == PLUGINS_QUESTION


async def test_ctrlc_ignored_prints_before_the_install_and_before_the_download_and_nowhere_else(layers, mobile, dist, monkeypatch) -> None:
    said = Said(monkeypatch, ["http://box"], [True])
    channel = FakeChannel(*_enable_script("mob"))

    with monkeypatch.context() as ignoring:
        ignoring.setattr(signal, "getsignal", lambda signum: signal.SIG_IGN)
        await _run(layers, channel)

    lines = channel.lines
    install, download = (CTRLC_IGNORED.format(step=step) for step in ("install", "download"))
    assert lines.count(install) == lines.count(download) == 1
    assert lines[lines.index(install) + 1].startswith(INSTALLING.format(installer="pip", spec=SPEC))
    assert lines.index(install) < lines.index(download) < lines.index("mob: enabled; starts when you close Plugins")
    assert [x for x in lines if "Ctrl-C is ignored" in x] == [install, download]
    assert CTRLC.format(does=CTRLC_DOES["prompt"]) in said.confirmed[0][2]


async def test_ctrlc_ignored_prints_nothing_for_a_declined_install_or_a_question_only_enable(layers, mobile, dist, monkeypatch) -> None:
    Said(monkeypatch, ["http://box-ok", "tok"], [False])
    channel = FakeChannel(*_enable_script("mob"), *_enable_script("ask"))

    with monkeypatch.context() as ignoring:
        ignoring.setattr(signal, "getsignal", lambda signum: signal.SIG_IGN)
        await _run(layers, channel)

    assert "Ctrl-C is ignored" not in channel.text


class _Hostile(Ask):
    manifest = Ask.manifest.model_copy(update={
        "next_steps": "run localharness[browser] then [/x]\x1b]0;owned\x07", "setup_help": "see [b]docs[/b]\x1b[31m"})


async def test_next_steps_and_help_with_markup_and_an_osc_sequence_print_as_text_with_it_stripped(layers, monkeypatch) -> None:
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Stub, _Hostile, Mob))
    Said(monkeypatch, ["http://box", "tok"])
    channel = FakeChannel(*_enable_script("ask"))

    await _run(layers, channel)

    assert "run localharness[browser] then [/x]" in channel.lines and "see [b]docs[/b]" in channel.lines
    assert "\x1b" not in channel.text and "\x07" not in channel.text and "owned" not in channel.text


DOWNLOAD2 = "Fetch the second model?"
DL_ASK = "Fetch the dl model?"
BATCH_CTRLC = CTRLC.format(does=CTRLC_DOES["batch"])


class _Url(BaseModel):
    url: str = ""


class Mob2(Plugin):
    """needs the mobile extra, asks a port and downloads a model"""

    manifest = PluginManifest(
        name="mob2", version="0.1.0", kind="tools", enabled_by_default=False, requires_extra="mobile",
        setup=(SetupField(key="url", prompt="Port", default="1"),), setup_action=DOWNLOAD2)
    ConfigModel = _Url

    async def configure(self, ctx):
        return "ready"

    def setup_action(self, ctx):
        CALLS.append("mob2")
        return [Check(name="mob2", status="pass", detail="fetched")]

    def doctor(self, ctx):
        return [Check(name="mob2", status="pass" if "mob2" in CALLS else "fail", detail="ready")]


class Dl(Plugin):
    """downloads a model and needs no extra"""

    manifest = PluginManifest(name="dl", version="0.1.0", kind="tools", enabled_by_default=False, setup_action=DL_ASK)
    ConfigModel = Stub.ConfigModel

    def setup_action(self, ctx):
        CALLS.append("dl")
        return [Check(name="dl", status="pass", detail="fetched")]

    def doctor(self, ctx):
        return [Check(name="dl", status="pass" if "dl" in CALLS else "fail", detail="ready")]


@pytest.fixture
def batch(layers, mobile, dist, monkeypatch):
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Stub, Mob, Mob2, Dl))
    return layers


def _pick(*names):
    return lambda options: {o.key for o in options if o.label in names}


def _several(*picks, then="+close"):
    return (Label("Turn several on…"), _pick(*picks), then)


def _plugins(monkeypatch, *classes) -> None:
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", classes)


def _numbered(count):
    return tuple(type(f"P{i}", (Stub,), {"manifest": PluginManifest(
        name=f"p{i}", version="0.1.0", kind="tools", enabled_by_default=False)}) for i in range(1, count + 1))


@pytest.mark.parametrize("classes", [(), (Stub,)], ids=["none", "one"])
async def test_several_row_is_absent_below_two_switchable_plugins(layers, monkeypatch, classes) -> None:
    _plugins(monkeypatch, *classes)
    channel = FakeChannel("+close")

    await _run(layers, channel)

    assert [o.key for o in channel.asked[0].options if o.key.startswith("+")] == ["+close"]


async def test_several_row_leads_the_root_at_two_and_the_default_is_the_first_plugin_row(layers, monkeypatch) -> None:
    _plugins(monkeypatch, Stub, Dl)
    channel = FakeChannel("+close")

    await _run(layers, channel)

    (asked,) = channel.asked
    first = asked.options[0]
    assert (first.key, first.label, first.hint) == ("+several", "Turn several on…", "turn on more than one at once")
    assert asked.default == 1 and asked.options[asked.default].label == "stub"


async def test_several_row_counts_the_plugins_that_can_be_switched_on_here(batch, monkeypatch) -> None:
    channel = FakeChannel(Label("Turn several on…"), CANCEL, "+close")

    await _run(batch, channel)

    assert [o.label for o in channel.asked[1].options] == ["stub", "mob", "mob2", "dl"]


@pytest.mark.parametrize("hold", ["overridden", "manual"])
async def test_several_row_never_counts_a_project_override_or_a_manual_route(batch, monkeypatch, hold) -> None:
    if hold == "overridden":
        (batch.ws / "config.yaml").write_text(yaml.safe_dump({"stub": {"enabled": False}, "dl": {"enabled": False}}), encoding="utf-8")
    else:
        real = plugins_cmd.install_route_for
        monkeypatch.setattr(plugins_cmd, "install_route_for", lambda entry: _MANUAL if entry.name.startswith("mob") else real(entry))
    channel = FakeChannel(Label("Turn several on…"), CANCEL, "+close")

    await _run(batch, channel)

    assert [o.label for o in channel.asked[1].options] == (["mob", "mob2"] if hold == "overridden" else ["stub", "dl"])


async def test_several_row_goes_when_the_only_other_candidate_is_held_off(batch) -> None:
    (batch.ws / "config.yaml").write_text(yaml.safe_dump({"stub": {"enabled": False}, "mob": {"enabled": False},
                                                          "mob2": {"enabled": False}}), encoding="utf-8")
    channel = FakeChannel("+close")

    await _run(batch, channel)

    assert "+several" not in [o.key for o in channel.asked[0].options]


async def test_several_which_is_a_multi_select_with_nothing_ticked_and_esc_back(batch) -> None:
    channel = FakeChannel(Label("Turn several on…"), CANCEL, "+close")

    assert await _run(batch, channel) is False

    asked = channel.asked[1]
    assert (asked.question, asked.default, asked.cancel, asked.multi) == ("Turn on which? (Esc = back)", frozenset(), "back", True)
    assert channel.text == "" and len(channel.asked) == 3


async def test_several_an_empty_confirmed_set_returns_to_the_root_with_nothing_printed(batch) -> None:
    channel = FakeChannel(Label("Turn several on…"), _pick(), "+close")

    assert await _run(batch, channel) is False

    assert channel.text == "" and channel.asked[2].question == PLUGINS_QUESTION and not _on_disk_exists(batch)


def _on_disk_exists(layers) -> bool:
    return (layers.g / "overrides.yaml").exists()


async def test_several_an_unreached_question_closes_with_the_unreached_line(batch) -> None:
    channel = FakeChannel(Label("Turn several on…"), Unreached(Unreachable.NO_TTY))

    assert await _run(batch, channel) is False

    assert channel.lines == [plugins_interview.unavailable(Unreachable.NO_TTY)]


async def test_batch_one_consent_and_one_install_cover_the_picks_that_need_it(batch, mobile, monkeypatch) -> None:
    said = Said(monkeypatch, ["http://box", "8080"], [True, True])
    touched: dict = {}
    channel = FakeChannel(*_several("mob", "mob2", "dl"))

    assert await _run(batch, channel, touched=touched) is True

    lines = channel.lines
    assert lines[0] == BATCH_CTRLC and lines.count(BATCH_CTRLC) == 1
    assert len(mobile.spawns) == 1 and mobile.spawns[0][0][-1] == SPEC
    assert lines[1] == CONSENT[0].format(names="mob, mob2")
    assert CONSENT[2].format(name="mob", question=DOWNLOAD) in lines and CONSENT[2].format(name="mob2", question=DOWNLOAD2) in lines
    assert not any(DL_ASK in line for line in lines)
    (consent, default, tail), (own, own_default, own_tail) = said.confirmed
    assert (consent, default) == ("", True) and BATCH_CTRLC in tail
    assert (own, own_default, own_tail) == (DL_ASK, True, BATCH_CTRLC) and all(DOWNLOAD not in c[0] and DOWNLOAD2 not in c[0] for c in said.confirmed)
    assert said.asked == [("Address: ", False), ("Port: ", False)]
    progress = [x for x in lines if x in ("1 of 3: mob", "2 of 3: mob2", "3 of 3: dl")]
    assert progress == ["1 of 3: mob", "2 of 3: mob2", "3 of 3: dl"]
    for name, nxt in (("mob", "2 of 3: mob2"), ("mob2", "3 of 3: dl")):
        assert lines.index(f"{name}: enabled; starts when you close Plugins") < lines.index(nxt)
    assert lines[-2:] == ["dl: enabled; starts when you close Plugins", "3 of 3 enabled; restarts when you close Plugins"]
    assert _written(batch) == {"mob": {"enabled": True, "url": "http://box"}, "mob2": {"enabled": True, "url": "8080"},
                               "dl": {"enabled": True}}
    assert set(touched) == {"mob", "mob2", "dl"} and channel.closed and channel.entered == 1


async def test_batch_the_first_line_is_the_ctrlc_line_even_when_nothing_needs_an_install(batch, monkeypatch) -> None:
    said = Said(monkeypatch, yes=[True])
    channel = FakeChannel(*_several("stub", "dl"))

    assert await _run(batch, channel) is True

    lines = channel.lines
    assert lines == [BATCH_CTRLC, "1 of 2: stub", "stub: enabled; starts when you close Plugins", "2 of 2: dl",
                     "dl: enabled; starts when you close Plugins", "2 of 2 enabled; restarts when you close Plugins"]
    assert said.confirmed == [(DL_ASK, True, BATCH_CTRLC)] and not any(CONSENT[0].split("{")[0] in x for x in lines)


async def test_batch_picks_run_in_the_order_they_are_listed_not_the_order_ticked(batch, mobile, monkeypatch) -> None:
    Said(monkeypatch, yes=[True])
    channel = FakeChannel(*_several("dl", "stub"))

    await _run(batch, channel)

    assert [x for x in channel.lines if x.endswith(("stub", "dl")) and " of 2: " in x] == ["1 of 2: stub", "2 of 2: dl"]


def _skipped(names, why="") -> str:
    return f"{names}: enabled in config, package not installed{why}"


async def test_batch_a_declined_install_prints_one_skipped_block_and_still_runs_the_pick_that_needed_none(batch, mobile, monkeypatch) -> None:
    said = Said(monkeypatch, yes=[False, True])
    touched: dict = {}
    channel = FakeChannel(*_several("mob", "mob2", "dl"))

    assert await _run(batch, channel, touched=touched) is True

    lines = channel.lines
    assert [x for x in lines if x.startswith(_skipped("mob, mob2"))] == [_skipped("mob, mob2")]
    assert mobile.spawns == [] and said.asked == [] and [c[0] for c in said.confirmed] == ["", DL_ASK]
    assert _written(batch) == {"mob": {"enabled": True}, "mob2": {"enabled": True}, "dl": {"enabled": True}}
    assert lines[-1] == "1 of 3 enabled, install still needed: 2; restarts when you close Plugins"
    assert touched["mob"].loads is False and touched["dl"].loads is True


async def test_batch_a_declined_install_with_nothing_else_to_start_has_no_restart_part(batch, mobile, monkeypatch) -> None:
    Said(monkeypatch, yes=[False])
    channel = FakeChannel(*_several("mob", "mob2"))

    assert await _run(batch, channel) is False

    assert channel.lines[-1] == "0 of 2 enabled, install still needed: 2" and channel.asked[-1].options[-1].label == "Close"


async def test_batch_a_failed_install_gives_the_same_shape_with_the_reason(batch, mobile, monkeypatch) -> None:
    mobile.proc = _Proc([b"boom\n"], code=1)
    Said(monkeypatch, yes=[True, True])
    channel = FakeChannel(*_several("mob", "mob2", "dl"))

    await _run(batch, channel)

    lines = channel.lines
    why = " — the install failed (exit 1; see the output above)"
    assert [x for x in lines if x.startswith(_skipped("mob, mob2"))] == [_skipped("mob, mob2", why)] and "boom" in lines
    assert lines[-1] == "1 of 3 enabled, install still needed: 2; restarts when you close Plugins"


async def test_batch_a_stop_in_the_shared_install_names_every_pick_and_terminates_the_child(batch, mobile, monkeypatch) -> None:
    mobile.proc = _Crashing()
    Said(monkeypatch, yes=[True])
    touched: dict = {}
    channel = FakeChannel(*_several("mob", "mob2", "dl"))

    assert await _run(batch, channel, touched=touched) is False

    assert channel.lines[-3:] == [
        "mob, mob2, dl: stopped — install interrupted — pick them again under Turn several on…",
        "not started: mob, mob2, dl — enable them from the list", "0 of 3 enabled"]
    assert mobile.proc.calls == ["terminate", "wait"] and touched == {} and not _on_disk_exists(batch)
    assert channel.asked[-1].question == PLUGINS_QUESTION


async def test_batch_a_stop_at_the_shared_consent_changes_nothing(batch, mobile, monkeypatch) -> None:
    Said(monkeypatch, yes=[None])
    channel = FakeChannel(*_several("mob", "mob2", "dl"))

    await _run(batch, channel)

    assert channel.lines[-3:] == ["mob, mob2, dl: stopped — nothing changed",
                                  "not started: mob, mob2, dl — enable them from the list", "0 of 3 enabled"]
    assert mobile.spawns == [] and not _on_disk_exists(batch)


async def test_batch_a_stop_at_a_picks_question_after_the_install_names_the_finish_and_what_did_not_start(batch, mobile, monkeypatch) -> None:
    Said(monkeypatch, ["http://box", None], [True])
    touched: dict = {}
    channel = FakeChannel(*_several("mob", "mob2", "dl"))

    assert await _run(batch, channel, touched=touched) is True

    assert channel.lines[-3:] == [
        "mob2: stopped — package installed, not set up — Enable to finish",
        "not started: dl — enable them from the list", "1 of 3 enabled; restarts when you close Plugins"]
    assert set(touched) == {"mob"} and "3 of 3: dl" not in channel.lines and _written(batch) == {"mob": {"enabled": True, "url": "http://box"}}


def _stop_at(monkeypatch, name) -> None:
    real = plugins_interview.enable_pass

    async def wrapped(session, picked, **kw):
        if picked == name:
            raise KeyboardInterrupt
        return await real(session, picked, **kw)

    monkeypatch.setattr(plugins_interview, "enable_pass", wrapped)


@pytest.mark.parametrize(("stopped", "tail", "summary"), [
    ("p1", "not started: p2, p3, p4 and 2 more — enable them from the list", "0 of 6 enabled"),
    ("p3", "not started: p4, p5, p6 — enable them from the list", "2 of 6 enabled; restarts when you close Plugins"),
    ("p5", "not started: p6 — enable them from the list", "4 of 6 enabled; restarts when you close Plugins"),
    ("p6", "nothing left to start", "5 of 6 enabled; restarts when you close Plugins"),
], ids=["first", "three-left", "one-left", "last"])
async def test_batch_a_stop_inside_a_pass_names_the_picks_after_it(layers, monkeypatch, stopped, tail, summary) -> None:
    _plugins(monkeypatch, *_numbered(6))
    _stop_at(monkeypatch, stopped)
    channel = FakeChannel(Label("Turn several on…"), _pick(*(f"p{i}" for i in range(1, 7))), "+close")

    await _run(layers, channel)

    assert channel.lines[-3:] == [f"{stopped}: stopped — nothing changed", tail, summary]


async def test_batch_names_with_an_escape_sequence_print_without_it(layers) -> None:
    flow, channel = _flow_for(layers)
    picks = [SimpleNamespace(name=name, manifest=None, bundled=True) for name in ("p\x1b[31m1", "q\x1b]0;owned\x07", "r")]

    async def stopping(session, name, **kw):
        raise KeyboardInterrupt

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(plugins_interview, "enable_pass", stopping)
        await plugins_interview._batch(flow, picks)

    assert "\x1b" not in channel.text and "\x07" not in channel.text and "owned" not in channel.text
    assert channel.lines[-4:] == ["1 of 3: p1", "p1: stopped — nothing changed", "not started: q, r — enable them from the list", "0 of 3 enabled"]


async def test_batch_summary_counts_the_enabled_kinds_and_never_an_install_skipped_pass(layers) -> None:
    flow, _ = _flow_for(layers)
    results = [(k, PassResult(k, StepOutcome(k, True))) for k in plugins_interview.BATCH_ENABLED | {"install_skipped", "declined", "refused"}]
    results.append(("stopped", None))

    done = len(plugins_interview.BATCH_ENABLED)
    assert plugins_interview._summary(flow, [None] * len(results), results) == f"{done} of {len(results)} enabled, install still needed: 1"
    assert plugins_interview.BATCH_ENABLED == {"enabled", "enabled_unconfigured", "check_failed", "check_failed_noload",
                                               "check_skipped", "enabled_manual", "enabled_not_loading"}


class _ConfConfig(BaseModel):
    flag: bool = False
    mode: Literal["a", "b"] = "a"
    count: int = 0
    label: str | None = None
    items: list[str] = []

    @field_validator("label")
    @classmethod
    def _not_banned(cls, value: str | None) -> str | None:
        if value and "banned" in value:
            raise ValueError(f"label {value!r} is banned")
        return value


class Conf(Plugin):
    """has a switch, a choice, a number, a label and a list"""

    manifest = PluginManifest(name="conf", version="0.1.0", kind="tools", enabled_by_default=False)
    ConfigModel = _ConfConfig


@pytest.fixture
def conf(layers, monkeypatch):
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Stub, Conf))
    layers.audit = layers.g / "audit.jsonl"
    (layers.g / "config.yaml").write_text(yaml.safe_dump(_CONFIG | {"org": {"audit_log_path": str(layers.audit)}}), encoding="utf-8")
    _on_disk(layers, conf=True)
    return layers


def _typing(monkeypatch, *answers):
    """Replaces the module's prompt: answers in order, records (text, default, secret)."""
    asked, script = [], list(answers)

    async def prompt(text, *, console, default="", secret=False):
        asked.append((text, default, secret))
        answer = script.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(plugins_interview, "prompt_line", prompt)
    return asked


def _settings(*edits):
    return (Label("conf"), Label("Configure…"), *edits, CANCEL, CANCEL, "+close")


def _audit_lines(layers) -> list[str]:
    return layers.audit.read_text(encoding="utf-8").splitlines() if layers.audit.exists() else []


_STATES = [
    ("on", True, None, False, ["Configure…", "Re-run setup", "Disable", "Info"]),
    ("on", True, None, True, ["Configure…", "Disable", "Info"]),
    ("off", False, None, False, ["Enable", "Info"]),
    ("available", False, None, False, ["Enable", "Info"]),
    ("needs-extra", True, _ARGV, False, ["Install and set up", "Disable", "Info"]),
    ("needs-extra", True, _MANUAL, False, ["Show install step", "Disable", "Info"]),
    ("needs-extra", True, None, False, ["Disable", "Info"]),
    ("failed", True, None, False, ["Disable", "Info"]),
    ("failed", False, None, False, ["Info"]),
    ("refused", False, None, False, ["Info"]),
    ("skipped", True, None, False, ["Disable", "Info"]),
    ("skipped", False, None, False, ["Info"]),
]


@pytest.mark.parametrize(("state", "enabled", "route", "running", "labels"), _STATES)
def test_actions_table_per_state_for_a_plugin_with_settings(conf, monkeypatch, state, enabled, route, running, labels) -> None:
    _patched(monkeypatch, enabled={"conf": enabled})
    session = _session(conf, running=frozenset({"conf"}) if running else frozenset())

    actions = plugins_interview.actions_for(_entry(state, name="conf"), session, route)

    assert [o.label for o, _ in actions] == labels
    assert len({o.key for o, _ in actions}) == len(actions)


def test_actions_table_a_failed_probe_on_a_running_plugin_offers_rerun_after_configure(conf) -> None:
    session = _session(conf, running=frozenset({"conf"}), touched={"conf": Touch(False, True, ("failed", "x"))})

    labels = [o.label for o, _ in plugins_interview.actions_for(_entry("on", name="conf"), session, None)]

    assert labels == ["Configure…", "Re-run setup", "Disable", "Info"]


def test_actions_table_configure_is_absent_without_settings_and_present_with(conf) -> None:
    session = _session(conf, running=frozenset({"conf", "stub"}))

    assert "Configure…" not in [o.label for o, _ in plugins_interview.actions_for(_entry("on", name="stub"), session, None)]
    assert "Configure…" in [o.label for o, _ in plugins_interview.actions_for(_entry("on", name="conf"), session, None)]


async def test_configure_lists_the_plugins_settings_with_value_and_layer_then_done(conf) -> None:
    (conf.g / "overrides.yaml").write_text(yaml.safe_dump({"conf": {"enabled": True, "count": 7}}), encoding="utf-8")
    channel = FakeChannel(*_settings())

    await _run(conf, channel)

    asked = channel.asked[2]
    assert asked.question == "conf — settings (Esc = back)" and (asked.default, asked.cancel) == (0, "done")
    rows = {o.label: o.hint for o in asked.options}
    assert list(rows) == ["count", "flag", "items", "label", "mode", "Done"]
    assert rows["count"] == "7 · global-overrides" and rows["flag"] == "False · default" and rows["label"] == "None · default"
    assert rows["items"] == "[] · default" and rows["mode"] == "'a' · default" and rows["Done"] == ""
    assert [o.key for o in asked.options] == ["0", "1", "2", "3", "4", "done"]
    assert channel.text == ""


@pytest.mark.parametrize("leave", [Label("Done"), CANCEL], ids=["done", "esc"])
async def test_configure_done_and_esc_return_to_the_plugins_menu(conf, leave) -> None:
    channel = FakeChannel(Label("conf"), Label("Configure…"), leave, CANCEL, "+close")

    await _run(conf, channel)

    assert channel.asked[3].question.startswith("conf — on") and channel.asked[3].question.endswith("(Esc = back)")
    assert channel.text == ""


@pytest.mark.parametrize(("running", "applies"), [(False, "when it starts"), (True, "when you close Plugins")])
async def test_configure_a_switch_goes_through_on_off_and_says_when_it_applies(conf, running, applies) -> None:
    channel = FakeChannel(*_settings(Label("flag"), Label("on")))

    assert await _run(conf, channel, running={"conf"} if running else frozenset()) is running

    ask = channel.asked[3]
    assert (ask.question, [o.label for o in ask.options], ask.default) == ("flag: on or off?", ["on", "off"], 1)
    assert channel.lines == [f"conf: flag saved — applies {applies}"]
    assert _written(conf) == {"conf": {"enabled": True, "flag": True}} and len(_audit_lines(conf)) == 1


async def test_configure_a_choice_asks_which_over_its_values_with_the_current_one_ahead(conf) -> None:
    channel = FakeChannel(*_settings(Label("mode"), Label("b")))

    await _run(conf, channel)

    ask = channel.asked[3]
    assert (ask.question, [o.label for o in ask.options], ask.default) == ("mode: which?", ["a", "b"], 0)
    assert _written(conf)["conf"]["mode"] == "b" and channel.lines == ["conf: mode saved — applies when it starts"]


async def test_configure_a_number_is_typed_at_a_prompt_that_shows_its_value_and_is_written_as_a_number(conf, monkeypatch) -> None:
    asked = _typing(monkeypatch, "7")
    channel = FakeChannel(*_settings(Label("count")))

    await _run(conf, channel)

    assert asked == [("count [0]: ", "", False)]
    assert _written(conf)["conf"]["count"] == 7 and len(_audit_lines(conf)) == 1


async def test_configure_a_refused_number_prints_one_line_and_asks_again(conf, monkeypatch) -> None:
    asked = _typing(monkeypatch, "x", "8")
    channel = FakeChannel(*_settings(Label("count")))

    await _run(conf, channel)

    assert len(asked) == 2 and len(channel.lines) == 2
    assert channel.lines[0].startswith("conf: settings not changed — Cannot coerce 'x' to int")
    assert channel.lines[1] == "conf: count saved — applies when it starts"
    assert _written(conf)["conf"]["count"] == 8 and len(_audit_lines(conf)) == 1


async def test_configure_an_optional_text_takes_a_value_and_none_clears_it(conf, monkeypatch) -> None:
    _typing(monkeypatch, "hello", "none")
    channel = FakeChannel(*_settings(Label("label"), Label("label")))

    await _run(conf, channel)

    assert _written(conf)["conf"]["label"] is None and len(_audit_lines(conf)) == 2
    assert channel.lines == ["conf: label saved — applies when it starts"] * 2


async def test_configure_a_list_is_read_only_and_names_the_path_and_the_machine_file(conf, monkeypatch) -> None:
    before = (conf.g / "overrides.yaml").read_bytes()
    asked = _typing(monkeypatch)
    channel = FakeChannel(*_settings(Label("items")))

    await _run(conf, channel)

    assert channel.lines == [f"read-only here — edit conf.items in {conf.g / 'overrides.yaml'}"]
    assert asked == [] and (conf.g / "overrides.yaml").read_bytes() == before and _audit_lines(conf) == []


@pytest.mark.parametrize("row", ["count", "label"])
async def test_configure_enter_on_a_prompt_keeps_the_stored_value_and_writes_nothing(conf, monkeypatch, row) -> None:
    before = (conf.g / "overrides.yaml").read_bytes()
    _typing(monkeypatch, "")
    channel = FakeChannel(*_settings(Label(row)))

    await _run(conf, channel)

    assert channel.text == "" and (conf.g / "overrides.yaml").read_bytes() == before and _audit_lines(conf) == []


async def test_configure_a_value_the_plugins_own_model_refuses_writes_nothing(conf, monkeypatch) -> None:
    before = (conf.g / "overrides.yaml").read_bytes()
    asked = _typing(monkeypatch, "banned!", "")
    channel = FakeChannel(*_settings(Label("label")))

    await _run(conf, channel)

    assert len(asked) == 2 and len(channel.lines) == 1
    assert channel.lines[0].startswith("conf: settings not changed — Validation failed for conf.label") and "banned" in channel.lines[0]
    assert (conf.g / "overrides.yaml").read_bytes() == before and _audit_lines(conf) == []


@pytest.mark.parametrize("row", ["flag", "mode"])
async def test_configure_esc_at_a_choice_is_no_edit(conf, row) -> None:
    before = (conf.g / "overrides.yaml").read_bytes()
    channel = FakeChannel(*_settings(Label(row), CANCEL))

    await _run(conf, channel)

    assert channel.text == "" and (conf.g / "overrides.yaml").read_bytes() == before and _audit_lines(conf) == []


async def test_configure_a_choice_nobody_could_be_asked_closes_plugins_with_the_unavailable_line(conf) -> None:
    channel = FakeChannel(Label("conf"), Label("Configure…"), Label("flag"), Unreached(Unreachable.NO_TTY))

    assert await _run(conf, channel) is False

    assert channel.lines == [plugins_interview.unavailable(Unreachable.NO_TTY)] and _audit_lines(conf) == []


async def test_configure_the_workspace_file_that_wins_a_setting_stays_as_it_was(conf, monkeypatch) -> None:
    project = conf.ws / "overrides.yaml"
    project.write_text(yaml.safe_dump({"conf": {"count": 9}}), encoding="utf-8")
    before, touched = project.read_bytes(), {}
    _typing(monkeypatch, "7")
    channel = FakeChannel(*_settings(Label("count")))

    assert await _run(conf, channel, running={"conf"}, touched=touched) is False

    assert channel.lines == [f"conf: count saved on this machine, but this project's {project} still wins here"]
    assert touched["conf"].rebind is False and project.read_bytes() == before
    assert _written(conf)["conf"]["count"] == 7


async def test_configure_settings_are_read_again_after_every_edit(conf, monkeypatch) -> None:
    _typing(monkeypatch, "7")
    channel = FakeChannel(*_settings(Label("count")))

    await _run(conf, channel)

    first, second = (channel.asked[2], channel.asked[3])
    assert first.options[0].hint == "0 · default" and second.options[0].hint == "7 · global-overrides"
