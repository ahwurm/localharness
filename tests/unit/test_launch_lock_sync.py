"""AUTO-07 (spec 20 §9.3, D-14, R-14-09, R-14-36): the launch lock has a synchronous twin for `serve`
(`launch_lock_sync`) and the async lock's holder can lend the flock to it (`lend_launch_lock`) while the
unit's `serve` writes its pidfile.

Hermetic like tests/unit/test_launch_lock.py: every config dir is `tmp_path`; "another process" is a raw fd
in this process (flock is per open file description, so a second fd conflicts exactly like a second process
would); no subprocess, no real server, nothing under the owner's `~/.localharness`. Nothing here sleeps for
longer than a few 50 ms polls.
"""
from __future__ import annotations

import asyncio
import errno
import logging
import os
import stat
import types

import pytest

from localharness.provider import lifecycle, server
from tests.unit.test_launch_lock import _hold, _lockable, _OpenSpy

fcntl = pytest.importorskip("fcntl")


@pytest.fixture(autouse=True)
def _no_leaked_holds():
    yield
    assert lifecycle._HELD == {}


@pytest.fixture
def lock_file(tmp_path):
    path = server.lock_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class _Clock:
    """A fake `time` for `lifecycle`: `sleep` advances `monotonic`, and may run a hook per sleep."""

    def __init__(self, on_sleep=None):
        self.now = 1000.0
        self.sleeps = 0
        self.on_sleep = on_sleep

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep(self.sleeps)


def _fake_time(monkeypatch, clock: _Clock) -> None:
    monkeypatch.setattr(lifecycle, "time", types.SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep))


def _closed(fd: int) -> bool:
    try:
        os.fstat(fd)
    except OSError:
        return True
    return False


# ------------------------------------------------------------------- launch_lock_sync


def test_the_sync_lock_takes_the_same_file_exclusively_and_not_inheritably(tmp_path, monkeypatch):
    path = server.lock_path(tmp_path)
    spy = _OpenSpy(path)
    monkeypatch.setattr(os, "open", spy)
    with lifecycle.launch_lock_sync(tmp_path, timeout_seconds=5):
        assert path.is_file() and stat.S_IMODE(path.stat().st_mode) == 0o600
        assert len(spy.fds) == 1 and os.get_inheritable(spy.fds[0]) is False
        assert not _lockable(path)
    assert _closed(spy.fds[0])
    assert _lockable(path)


def test_the_sync_lock_creates_the_server_directory(tmp_path):
    assert not server.server_dir(tmp_path).exists()
    with lifecycle.launch_lock_sync(tmp_path, timeout_seconds=1):
        assert server.lock_path(tmp_path).is_file()


def test_a_contended_sync_lock_times_out_with_the_contract_line(tmp_path, lock_file, monkeypatch):
    holder = _hold(lock_file)
    clock = _Clock()
    _fake_time(monkeypatch, clock)
    spy = _OpenSpy(lock_file)
    monkeypatch.setattr(os, "open", spy)
    try:
        with pytest.raises(TimeoutError) as raised:
            with lifecycle.launch_lock_sync(tmp_path, timeout_seconds=5):
                pytest.fail("the block must not run while another fd holds the lock")
        assert str(raised.value) == lifecycle.LAUNCH_LOCK_TIMEOUT_LINE.format(seconds=5)
        assert str(raised.value) == "another LocalHarness is launching the model server — gave up after 5s"
        assert clock.now - 1000.0 >= 5 and clock.sleeps >= 99  # it polled every 50 ms until the deadline
        assert _closed(spy.fds[0])
    finally:
        os.close(holder)


def test_a_zero_timeout_fails_at_once_when_held(tmp_path, lock_file):
    holder = _hold(lock_file)
    try:
        with pytest.raises(TimeoutError):
            with lifecycle.launch_lock_sync(tmp_path, timeout_seconds=0):
                pytest.fail("held")
    finally:
        os.close(holder)


def test_a_sync_lock_released_mid_wait_is_acquired(tmp_path, lock_file, monkeypatch):
    holder = _hold(lock_file)

    def release_after_three(count):
        if count == 3:
            os.close(holder)

    clock = _Clock(release_after_three)
    _fake_time(monkeypatch, clock)
    with lifecycle.launch_lock_sync(tmp_path, timeout_seconds=5):
        assert clock.sleeps == 3
        assert not _lockable(lock_file)
    assert _lockable(lock_file)


def test_a_planted_symlink_degrades_with_one_warning_and_the_block_runs(tmp_path, caplog):
    server.server_dir(tmp_path).mkdir(parents=True, exist_ok=True)
    target = tmp_path / "elsewhere"
    target.write_text("x")
    server.lock_path(tmp_path).symlink_to(target)
    ran = False
    with caplog.at_level(logging.WARNING, logger=lifecycle.log.name):
        with lifecycle.launch_lock_sync(tmp_path, timeout_seconds=1):
            ran = True
    assert ran and target.read_text() == "x"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "launch lock unavailable" in warnings[0].getMessage()


def test_a_flock_failing_with_another_errno_degrades_and_closes_the_fd(tmp_path, caplog, monkeypatch):
    spy = _OpenSpy(server.lock_path(tmp_path))
    monkeypatch.setattr(os, "open", spy)

    def refuse(fd, op):
        raise OSError(errno.ENOLCK, "no locks available")

    monkeypatch.setattr(lifecycle.fcntl, "flock", refuse)
    ran = False
    with caplog.at_level(logging.WARNING, logger=lifecycle.log.name):
        with lifecycle.launch_lock_sync(tmp_path, timeout_seconds=1):
            ran = True
            assert _closed(spy.fds[0])
    assert ran
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "flock failed" in warnings[0].getMessage()


def test_without_fcntl_the_sync_lock_runs_the_block_unlocked(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(lifecycle, "fcntl", None)
    ran = False
    with caplog.at_level(logging.WARNING, logger=lifecycle.log.name):
        with lifecycle.launch_lock_sync(tmp_path, timeout_seconds=1):
            ran = True
    assert ran and any("no fcntl" in r.getMessage() for r in caplog.records)


def test_the_sync_lock_closes_the_fd_when_the_block_raises(tmp_path, monkeypatch):
    spy = _OpenSpy(server.lock_path(tmp_path))
    monkeypatch.setattr(os, "open", spy)
    with pytest.raises(ValueError):
        with lifecycle.launch_lock_sync(tmp_path, timeout_seconds=1):
            raise ValueError("boom")
    assert _closed(spy.fds[0]) and _lockable(server.lock_path(tmp_path))


async def test_the_async_and_the_sync_lock_exclude_each_other(tmp_path):
    """One lock, two entry points: the sync lock cannot be taken while the async one is held, and the
    async one waits while the sync one is."""
    async with lifecycle.launch_lock(tmp_path):
        with pytest.raises(TimeoutError):
            with lifecycle.launch_lock_sync(tmp_path, timeout_seconds=0.1):
                pytest.fail("the async lock is held")

    entered = asyncio.Event()

    async def launch():
        async with lifecycle.launch_lock(tmp_path):
            entered.set()

    with lifecycle.launch_lock_sync(tmp_path, timeout_seconds=1):
        task = asyncio.create_task(launch())
        await asyncio.sleep(0.2)
        assert not entered.is_set()
    await asyncio.wait_for(task, 1.0)
    assert entered.is_set()


# ------------------------------------------------------------------- lend_launch_lock


async def test_the_loan_frees_the_flock_for_its_length_and_takes_it_back(tmp_path, lock_file):
    key = lock_file.resolve()
    me = asyncio.current_task()
    async with lifecycle.launch_lock(tmp_path):
        entry = lifecycle._HELD[key]
        assert not _lockable(lock_file) and entry[3] is False
        async with lifecycle.lend_launch_lock(tmp_path):
            assert _lockable(lock_file)  # the unit's serve can take it
            assert lifecycle._HELD[key] is entry and entry[0] is me and entry[3] is True
        assert not _lockable(lock_file) and entry[3] is False
        assert lifecycle._HELD[key][0] is me
    assert _lockable(lock_file)


async def test_the_unit_serve_can_take_and_release_the_lock_inside_the_loan(tmp_path, lock_file):
    async with lifecycle.launch_lock(tmp_path):
        async with lifecycle.lend_launch_lock(tmp_path):
            with lifecycle.launch_lock_sync(tmp_path, timeout_seconds=1):
                assert not _lockable(lock_file)
        assert not _lockable(lock_file)  # the lender has it again
    assert _lockable(lock_file)


async def test_the_reclaim_waits_for_a_second_holder_and_never_breaks_it(tmp_path, lock_file, monkeypatch):
    real_sleep = asyncio.sleep
    other: list[int] = []
    polls = 0

    async def sleep(seconds, *a, **kw):
        nonlocal polls
        polls += 1
        if polls == 4 and other:
            os.close(other.pop())
        await real_sleep(0)

    async with lifecycle.launch_lock(tmp_path):
        async with lifecycle.lend_launch_lock(tmp_path):
            other.append(_hold(lock_file))  # a second launcher wins the flock during the loan
            monkeypatch.setattr(lifecycle.asyncio, "sleep", sleep)
        assert polls == 4 and other == []  # it waited exactly until the other released
        assert not _lockable(lock_file)
        assert lifecycle._HELD[lock_file.resolve()][3] is False
    assert _lockable(lock_file)


async def test_a_cancelled_reclaim_still_clears_the_lent_flag(tmp_path, lock_file):
    key = lock_file.resolve()
    ended = asyncio.Event()
    seen: dict = {}

    async def lender():
        async with lifecycle.launch_lock(tmp_path):
            seen["entry"] = lifecycle._HELD[key]
            async with lifecycle.lend_launch_lock(tmp_path):
                seen["other"] = _hold(lock_file)
                ended.set()
                await asyncio.sleep(0)

    task = asyncio.create_task(lender())
    await ended.wait()
    await asyncio.sleep(0.15)  # the reclaim is polling a lock somebody else holds
    assert seen["entry"][3] is True and not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    os.close(seen["other"])
    assert seen["entry"][3] is False
    assert lifecycle._HELD == {}


async def test_a_loan_by_a_task_that_holds_nothing_is_a_no_op(tmp_path, lock_file):
    async with lifecycle.lend_launch_lock(tmp_path):
        assert lifecycle._HELD == {}

    holder_in = asyncio.Event()
    release = asyncio.Event()

    async def holder():
        async with lifecycle.launch_lock(tmp_path):
            holder_in.set()
            await release.wait()

    task = asyncio.create_task(holder())
    await holder_in.wait()
    async with lifecycle.lend_launch_lock(tmp_path):  # this task is not the holder
        assert not _lockable(lock_file) and lifecycle._HELD[lock_file.resolve()][3] is False
    release.set()
    await task


async def test_without_fcntl_the_loan_is_a_no_op_and_nothing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(lifecycle, "fcntl", None)
    async with lifecycle.launch_lock(tmp_path):
        entry = lifecycle._HELD[server.lock_path(tmp_path).resolve()]
        assert entry[2] is None
        async with lifecycle.lend_launch_lock(tmp_path):
            assert entry[3] is False


async def test_a_nested_loan_is_a_no_op(tmp_path, lock_file):
    async with lifecycle.launch_lock(tmp_path):
        async with lifecycle.lend_launch_lock(tmp_path):
            async with lifecycle.lend_launch_lock(tmp_path):
                assert _lockable(lock_file)
            assert _lockable(lock_file)  # the inner exit did not take it back early
        assert not _lockable(lock_file)


# ------------------------------------------------------------------- R-14-36: in-process ownership across the loan


async def test_a_second_task_in_process_waits_through_the_whole_loan(tmp_path, lock_file):
    key = lock_file.resolve()
    second_in = asyncio.Event()
    owners: dict = {}

    async def second():
        async with lifecycle.launch_lock(tmp_path):
            second_in.set()
            owners["inside"] = lifecycle._HELD[key][0]
            owners["me"] = asyncio.current_task()

    async def first():
        me = asyncio.current_task()
        async with lifecycle.launch_lock(tmp_path):
            async with lifecycle.lend_launch_lock(tmp_path):
                t2 = asyncio.create_task(second())
                for _ in range(6):
                    await asyncio.sleep(0.06)
                    assert not second_in.is_set()
                    assert lifecycle._HELD[key][0] is me and lifecycle._HELD[key][3] is True
                    assert _lockable(lock_file)  # a second process (the unit's serve) can still take it
            # the loan is over: the lender's own nested acquire is re-entrant, not a self-deadlock
            async with asyncio.timeout(1.0):
                async with lifecycle.launch_lock(tmp_path):
                    assert lifecycle._HELD[key][0] is me
            assert not second_in.is_set() and lifecycle._HELD[key][0] is me
        return t2

    t2 = await first()
    await asyncio.wait_for(t2, 2.0)
    assert second_in.is_set() and owners["inside"] is owners["me"] is t2
    assert key not in lifecycle._HELD


async def test_a_nested_acquire_by_the_lender_inside_its_loan_is_refused(tmp_path, lock_file):
    key = lock_file.resolve()
    me = asyncio.current_task()
    async with lifecycle.launch_lock(tmp_path):
        entry = lifecycle._HELD[key]
        async with lifecycle.lend_launch_lock(tmp_path):
            with pytest.raises(RuntimeError) as raised:
                async with lifecycle.launch_lock(tmp_path):
                    pytest.fail("a loan's block must not launch")
            assert str(raised.value) == lifecycle.LAUNCH_LOCK_LENT_LINE
            assert entry[1] == 1  # the refused acquire left the depth alone
        assert lifecycle._HELD[key] is entry and entry[0] is me and not _lockable(lock_file)
        async with lifecycle.launch_lock(tmp_path):  # and after the loan it is re-entrant again
            assert entry[1] == 2
    assert _lockable(lock_file)


def test_the_lent_line_is_the_contract():
    assert lifecycle.LAUNCH_LOCK_LENT_LINE == (
        "the launch lock is lent to the autostart unit — a launch inside the loan would race it")


async def test_two_tasks_outside_any_loan_still_serialise_and_leave_no_entry(tmp_path):
    order: list[str] = []

    async def body(name):
        async with lifecycle.launch_lock(tmp_path):
            order.append(f"in:{name}")
            await asyncio.sleep(0.1)
            order.append(f"out:{name}")

    await asyncio.gather(body("a"), body("b"))
    assert [o.split(":")[0] for o in order] == ["in", "out", "in", "out"]
    assert order[0][3:] == order[1][4:] and order[2][3:] == order[3][4:]


async def test_the_wait_announcement_still_fires_when_the_holder_is_a_task_of_this_process(
        tmp_path, monkeypatch):
    from localharness.provider import client as client_mod

    monkeypatch.setattr(client_mod, "_QUEUE_VISIBILITY_SECONDS", 0.1)
    told: list[str] = []

    async def on_wait():
        told.append(lifecycle.launch_lock_wait_line(tmp_path))

    async def second():
        async with lifecycle.launch_lock(tmp_path, on_wait=on_wait):
            pass

    async with lifecycle.launch_lock(tmp_path):
        task = asyncio.create_task(second())
        await asyncio.sleep(0.4)
    await asyncio.wait_for(task, 1.0)
    assert told == [lifecycle.LAUNCH_LOCK_LOCAL_WAIT_LINE]
