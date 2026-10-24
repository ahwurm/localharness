"""`sigint_cancels` on a real loop on the main thread, pressed with real SIGINTs."""
from __future__ import annotations

import asyncio
import os
import signal
import sys

import pytest

from localharness.core.interrupt import sigint_cancels

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals"),
    pytest.mark.usefixtures("default_sigint"),
]
SIGINT = signal.SIGINT


def press() -> None:
    os.kill(os.getpid(), SIGINT)


async def sleepy(cleanup: float = 0.0) -> None:
    try:
        await asyncio.sleep(100)
    finally:
        if cleanup:
            await asyncio.sleep(cleanup)


async def step(sub: asyncio.Task, **kw) -> str:
    """One guarded step, the turn handler's shape: "stopped" on a user stop, else re-raise."""
    async with sigint_cancels(sub, **kw) as cancelled:
        try:
            await sub
        except asyncio.CancelledError:
            if not cancelled():
                raise
            return "stopped"
    return "done"


def test_the_first_press_cancels_the_subtask_and_is_a_stop():
    async def scenario():
        before = signal.getsignal(SIGINT)
        sub = asyncio.ensure_future(sleepy())
        asyncio.get_running_loop().call_soon(press)
        result = await step(sub)
        return result, sub.cancelled(), signal.getsignal(SIGINT) is before

    assert asyncio.run(scenario()) == ("stopped", True, True)


def test_the_second_press_hands_sigint_back_to_the_previous_handler():
    seen: list = []
    original = signal.signal(SIGINT, lambda *_: seen.append("press"))
    recorder = signal.getsignal(SIGINT)

    async def presses():
        press()
        await asyncio.sleep(0.05)
        press()
        await asyncio.sleep(0.05)
        assert signal.getsignal(SIGINT) is recorder
        press()
        await asyncio.sleep(0.05)

    async def scenario():
        sub = asyncio.ensure_future(sleepy(cleanup=0.5))
        presser = asyncio.ensure_future(presses())
        result = await step(sub)
        await presser
        return result

    try:
        assert asyncio.run(scenario()) == "stopped"
    finally:
        signal.signal(SIGINT, original)
    assert seen == ["press"]


def test_a_second_press_during_a_slow_unwind_is_still_a_stop():
    async def scenario():
        loop = asyncio.get_running_loop()
        sub = asyncio.ensure_future(sleepy(cleanup=0.2))
        loop.call_soon(press)
        loop.call_later(0.05, press)
        return await step(sub)

    assert asyncio.run(scenario()) == "stopped"


def test_a_parent_cancel_during_the_unwind_propagates():
    async def scenario():
        loop = asyncio.get_running_loop()
        sub = asyncio.ensure_future(sleepy(cleanup=0.2))
        parent = asyncio.ensure_future(step(sub))
        loop.call_soon(press)
        loop.call_later(0.05, parent.cancel)
        with pytest.raises(asyncio.CancelledError):
            await parent
        return sub.cancelled()

    assert asyncio.run(scenario())


def test_a_parent_cancel_in_the_tick_the_subtask_finished_propagates():
    async def scenario():
        sub = asyncio.ensure_future(sleepy())

        async def parent_body():
            me = asyncio.current_task()
            sub.add_done_callback(lambda _: me.cancel())
            return await step(sub)

        parent = asyncio.ensure_future(parent_body())
        asyncio.get_running_loop().call_soon(press)
        with pytest.raises(asyncio.CancelledError):
            await parent

    asyncio.run(scenario())


def test_a_parent_that_swallowed_an_earlier_cancel_still_stops():
    async def parent_body():
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            pass
        assert asyncio.current_task().cancelling() == 1
        sub = asyncio.ensure_future(sleepy())
        asyncio.get_running_loop().call_soon(press)
        return await step(sub)

    async def scenario():
        parent = asyncio.ensure_future(parent_body())
        await asyncio.sleep(0)
        parent.cancel()
        return await parent

    assert asyncio.run(scenario()) == "stopped"


def test_an_ignore_launch_installs_nothing():
    original = signal.signal(SIGINT, signal.SIG_IGN)

    async def scenario():
        loop = asyncio.get_running_loop()
        calls: list = []
        loop.add_signal_handler = lambda *a: calls.append("add")
        loop.remove_signal_handler = lambda *a: calls.append("remove")
        sub = asyncio.ensure_future(sleepy())
        loop.call_soon(press)
        loop.call_later(0.05, sub.cancel)
        with pytest.raises(asyncio.CancelledError):
            await step(sub)
        return calls, signal.getsignal(SIGINT)

    try:
        assert asyncio.run(scenario()) == ([], signal.SIG_IGN)
    finally:
        signal.signal(SIGINT, original)


def test_on_stopping_fires_about_a_second_into_a_slow_stop():
    async def scenario():
        loop = asyncio.get_running_loop()
        fired: list = []
        sub = asyncio.ensure_future(sleepy(cleanup=1.3))
        t0 = loop.time()
        loop.call_soon(press)
        result = await step(sub, on_stopping=lambda: fired.append(loop.time() - t0))
        return result, fired

    result, fired = asyncio.run(scenario())
    assert result == "stopped"
    assert len(fired) == 1 and 1.0 <= fired[0] < 1.3


def test_on_stopping_stays_quiet_for_a_quick_stop_or_no_press():
    async def scenario():
        loop = asyncio.get_running_loop()
        fired: list = []
        quick = asyncio.ensure_future(asyncio.sleep(0))
        assert await step(quick, on_stopping=lambda: fired.append("no press")) == "done"
        sub = asyncio.ensure_future(sleepy(cleanup=0.1))
        loop.call_soon(press)
        assert await step(sub, on_stopping=lambda: fired.append("quick stop")) == "stopped"
        await asyncio.sleep(1.1)
        return fired

    assert asyncio.run(scenario()) == []


def test_a_cancel_without_a_press_propagates():
    async def scenario():
        before = signal.getsignal(SIGINT)
        sub = asyncio.ensure_future(sleepy())
        asyncio.get_running_loop().call_soon(sub.cancel)
        with pytest.raises(asyncio.CancelledError):
            await step(sub)
        return signal.getsignal(SIGINT) is before

    assert asyncio.run(scenario())


def test_the_step_runs_when_the_loop_has_no_signal_support():
    async def scenario():
        loop = asyncio.get_running_loop()
        removed: list = []

        def _raise(*a):
            raise RuntimeError("no signal support")

        loop.add_signal_handler = _raise
        loop.remove_signal_handler = lambda *a: removed.append(a)
        before = signal.getsignal(SIGINT)
        quick = asyncio.ensure_future(asyncio.sleep(0))
        return await step(quick), removed, signal.getsignal(SIGINT) is before

    assert asyncio.run(scenario()) == ("done", [], True)
