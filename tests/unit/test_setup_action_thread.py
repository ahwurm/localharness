"""setup_action_rows runs a plain-def setup_action on the default executor behind lifecycle's
_downloading claim: the loop keeps ticking, the thread sees the task's context, a second call is
refused while the first runs, and exactly one side clears the flag whichever finishes first."""
from __future__ import annotations

import asyncio
import threading
import time
from contextvars import ContextVar

import pytest

from localharness.plugins import lifecycle
from localharness.plugins.api import Check, PluginPaths
from localharness.plugins.lifecycle import DOWNLOAD_RUNNING, downloading, setup_action_rows
from tests.unit.test_plugin_lifecycle import _plugin, _resolution

NAME = "dl"
OK = [Check(name=NAME, status="pass", detail="downloaded")]
REFUSED = (Check(name=NAME, status="fail", detail=DOWNLOAD_RUNNING),)


def _rows(tmp_path, setup_action):
    return setup_action_rows(_resolution(bundled=(_plugin(NAME, setup_action=setup_action),)),
                             NAME, PluginPaths(tmp_path, None, tmp_path))


class _Counting(set):
    """_downloading with its discard calls counted."""

    cleared = 0

    def discard(self, x):
        self.cleared += 1
        super().discard(x)


@pytest.fixture
def flags(monkeypatch) -> _Counting:
    flags = _Counting()
    monkeypatch.setattr(lifecycle, "_downloading", flags)
    return flags


async def _until(pred, seconds=5.0):
    deadline = time.monotonic() + seconds
    while not pred() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert pred()


async def test_a_plain_def_runs_off_the_loop_thread_while_the_loop_keeps_ticking(tmp_path, flags):
    ticks, seen = [], {}

    def action(self, ctx):
        time.sleep(0.3)
        seen["ticked"], seen["thread"] = bool(ticks), threading.get_ident()
        return OK

    asyncio.get_running_loop().call_later(0.1, ticks.append, 1)
    assert await _rows(tmp_path, action) == tuple(OK)
    assert seen["ticked"] and seen["thread"] != threading.get_ident()
    assert not downloading(NAME) and flags.cleared == 1  # the thread cleared it before the awaiter woke


async def test_the_action_sees_the_tasks_context_vars(tmp_path, flags):
    var: ContextVar[str] = ContextVar("var")
    var.set("from the task")
    rows = await _rows(tmp_path, lambda self, ctx: [Check(name=NAME, status="pass", detail=var.get("unset"))])
    assert rows[0].detail == "from the task"


async def test_an_async_action_is_awaited_on_the_loop_thread(tmp_path, flags):
    async def action(self, ctx):
        return [Check(name=NAME, status="pass", detail=str(threading.get_ident()))]

    rows = await _rows(tmp_path, action)
    assert rows[0].detail == str(threading.get_ident())
    assert not downloading(NAME) and flags.cleared == 1


async def test_a_sync_wrapper_returning_a_coroutine_is_awaited(tmp_path, flags):
    async def inner():
        return OK

    assert await _rows(tmp_path, lambda self, ctx: inner()) == tuple(OK)
    assert not downloading(NAME) and flags.cleared == 1


async def test_a_second_call_while_the_first_runs_is_refused(tmp_path, flags):
    release = threading.Event()

    def action(self, ctx):
        release.wait(5)
        return OK

    first = asyncio.create_task(_rows(tmp_path, action))
    await _until(lambda: downloading(NAME))
    assert await _rows(tmp_path, action) == REFUSED
    release.set()
    assert await first == tuple(OK)
    assert not downloading(NAME) and flags.cleared == 1


async def test_two_sequential_calls_run_the_action_twice_and_leave_nothing_flagged(tmp_path, flags):
    runs = []

    def action(self, ctx):
        runs.append(downloading(NAME))
        return OK

    for n in (1, 2):
        assert await _rows(tmp_path, action) == tuple(OK)
        assert runs == [True] * n and not downloading(NAME) and flags.cleared == n


async def test_an_empty_list_is_no_rows_and_leaves_nothing_flagged(tmp_path, flags):
    assert await _rows(tmp_path, lambda self, ctx: []) == ()
    assert not downloading(NAME) and flags.cleared == 1


async def test_an_awaiter_cancelled_mid_download_leaves_the_flag_to_the_thread(tmp_path, flags):
    started, release = threading.Event(), threading.Event()

    def action(self, ctx):
        started.set()
        release.wait(5)
        return OK

    task = asyncio.create_task(_rows(tmp_path, action))
    await _until(started.is_set)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert downloading(NAME) and flags.cleared == 0
    release.set()
    await _until(lambda: not downloading(NAME))
    assert flags.cleared == 1


async def test_an_awaiter_cancelled_before_the_thread_claims_clears_it_and_the_thread_does_nothing(
        tmp_path, flags, monkeypatch):
    loop, submitted, ran = asyncio.get_running_loop(), [], []

    def submit(executor, fn, *args):
        submitted.append((fn, args))
        return loop.create_future()

    monkeypatch.setattr(loop, "run_in_executor", submit)
    task = asyncio.create_task(_rows(tmp_path, lambda self, ctx: ran.append(1) or OK))
    await asyncio.sleep(0)
    assert downloading(NAME)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not downloading(NAME) and flags.cleared == 1
    (fn, args), = submitted
    assert fn(*args) is None and ran == [] and flags.cleared == 1


@pytest.mark.parametrize("escapes", [True, False])
async def test_a_submit_that_raises_discards_the_name(tmp_path, flags, monkeypatch, escapes):
    exc = KeyboardInterrupt() if escapes else RuntimeError("executor is shut down")

    def submit(*args):
        raise exc

    monkeypatch.setattr(asyncio.get_running_loop(), "run_in_executor", submit)
    if escapes:
        with pytest.raises(KeyboardInterrupt):
            await _rows(tmp_path, lambda self, ctx: OK)
    else:
        assert await _rows(tmp_path, lambda self, ctx: OK) == (Check(
            name=NAME, status="fail", detail="its setup action raised RuntimeError: executor is shut down"),)
    assert not downloading(NAME) and flags.cleared == 1


async def test_an_action_that_raises_in_the_thread_is_one_failing_row_and_clears_the_flag(tmp_path, flags):
    def action(self, ctx):
        raise SystemExit(3)

    rows = await _rows(tmp_path, action)
    assert rows == (Check(name=NAME, status="fail", detail="its setup action raised SystemExit: 3"),)
    assert not downloading(NAME) and flags.cleared == 1
