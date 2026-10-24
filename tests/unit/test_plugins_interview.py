"""The Plugins flow driven through a scripted channel: the root, its close and the reload decision."""
from __future__ import annotations

import asyncio
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from rich.console import Console

from localharness.cli import plugins_cmd, plugins_interview
from localharness.cli.interview import Answer, Unreachable
from localharness.cli.plugins_cmd import PluginSession, ReloadPlan, Touch, reload_names
from localharness.cli.plugins_interview import CLOSE_LABELS, PLUGINS_QUESTION, at_close, run
from localharness.plugins import resolve as resolve_module
from localharness.plugins.api import PluginManifest
from localharness.plugins.plan import PlanEntry
from tests.unit.interview_support import CANCEL, UNREACHED, Asked
from tests.unit.test_plugin_resolve import write_folder_plugin
from tests.unit.test_plugins_pass import folder as _folder
from tests.unit.test_plugins_pass import layers as _layers

layers, folder = _layers, _folder
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


async def test_reload_a_picked_plugin_row_draws_the_root_again(layers) -> None:
    channel = FakeChannel("1", Label("ask"), "+close")

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

    assert [o.hint for o in channel.asked[0].options[:3]] == ["on — not running; Re-run setup starts it", "off", "off"]


async def test_root_is_resolved_again_on_every_render(layers) -> None:
    def enable(options):
        (layers.g / "overrides.yaml").write_text(yaml.safe_dump({"stub": {"enabled": True}}), encoding="utf-8")
        return "1"

    channel = FakeChannel(enable, "+close")

    await _run(layers, channel)

    first, second = (asked.options[0].hint for asked in channel.asked)
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
    channel = FakeChannel("1", "1", "+close")

    await _run(layers, channel, parked=parked)

    assert channel.lines == [line] and len(channel.asked) == 3


async def test_asks_pending_is_silent_with_nothing_parked_even_when_asks_wait(layers) -> None:
    channel = FakeChannel("+close")

    await _run(layers, channel, waiting=2)

    assert channel.text == ""


async def test_asks_pending_with_a_call_parked_a_pending_change_defers_with_the_blocked_close(layers) -> None:
    channel = FakeChannel("+close")

    assert await _run(layers, channel, touched={"stub": Touch(False, True, None)}, parked=1) is False

    assert not channel.closed and "restart deferred: 1 ask" in channel.asked[0].options[-1].label
