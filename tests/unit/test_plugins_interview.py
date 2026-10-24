"""The Plugins flow driven through a scripted channel: the root, its close and the reload decision."""
from __future__ import annotations

import asyncio
import io
from contextlib import asynccontextmanager

import pytest
from rich.console import Console

from localharness.cli.interview import Answer, Unreachable
from localharness.cli.plugins_cmd import PluginSession, ReloadPlan, Touch, reload_names
from localharness.cli.plugins_interview import CLOSE_LABELS, PLUGINS_QUESTION, at_close, run
from tests.unit.interview_support import CANCEL, UNREACHED, Asked
from tests.unit.test_plugins_pass import layers as _layers

layers = _layers
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
