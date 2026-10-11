"""The cross-process launch lock (D-26, D-28, D-29, D-47): one advisory flock on `launch.lock` beside the
pidfile, re-entrant per owning task, cancellation-safe, degrading to an in-process lock.

Hermetic (D-34): every config dir is `tmp_path`; the "other process" is a raw fd in this process (flock
is per open file description, so a second fd conflicts exactly like a second process would); no
subprocess, no real server, nothing under the owner's `~/.localharness`.
"""
from __future__ import annotations

import asyncio
import logging
import os
import stat

import pytest

from localharness.provider import client as client_mod
from localharness.provider import lifecycle, server

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


def _hold(path) -> int:
    """Take the lock from a second fd, the way another LocalHarness process would."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _lockable(path) -> bool:
    """True when a fresh fd takes LOCK_EX | LOCK_NB at once (the file is unlocked)."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


class _OpenSpy:
    """Records every `os.open` of one path (the lock file) and the fd it returned."""

    def __init__(self, path):
        self.path = os.fspath(path)
        self.fds: list[int] = []
        self._real = os.open

    def __call__(self, p, *args, **kw):
        fd = self._real(p, *args, **kw)
        if os.fspath(p) == self.path:
            self.fds.append(fd)
        return fd


@pytest.fixture
def open_spy(monkeypatch, lock_file):
    spy = _OpenSpy(lock_file)
    monkeypatch.setattr(os, "open", spy)
    return spy


def test_the_wait_line_is_the_contract():
    assert lifecycle.LAUNCH_LOCK_WAIT_LINE == "another LocalHarness is launching the model server — waiting"


def test_lock_path_sits_beside_the_pidfile(tmp_path):
    assert server.lock_path(tmp_path) == tmp_path / "vllm" / "launch.lock"
    assert server.lock_path(tmp_path).parent == server.pid_path(tmp_path).parent


async def test_a_held_lock_makes_the_next_launch_wait_until_it_is_closed(tmp_path, lock_file):
    holder = _hold(lock_file)
    entered = asyncio.Event()

    async def launch():
        async with lifecycle.launch_lock(tmp_path):
            entered.set()

    task = asyncio.create_task(launch())
    await asyncio.sleep(0.2)
    assert not entered.is_set() and not task.done()
    os.close(holder)
    await asyncio.wait_for(task, 0.5)
    assert entered.is_set()


async def test_two_tasks_run_their_bodies_one_after_the_other(tmp_path):
    order: list[str] = []

    async def body(name):
        async with lifecycle.launch_lock(tmp_path):
            order.append(f"in:{name}")
            await asyncio.sleep(0.1)
            order.append(f"out:{name}")

    await asyncio.gather(body("a"), body("b"))
    assert order[0].startswith("in:") and order[1].startswith("out:")
    assert order[0][3:] == order[1][4:]
    assert order[2].startswith("in:") and order[3].startswith("out:")
    assert order[2][3:] == order[3][4:]


async def test_the_holder_re_enters_without_a_second_fd(tmp_path, open_spy, lock_file):
    async with lifecycle.launch_lock(tmp_path):
        async with asyncio.timeout(1.0):
            async with lifecycle.launch_lock(tmp_path):
                async with lifecycle.launch_lock(tmp_path):
                    pass
        launch_opens = len(open_spy.fds)
        assert not _lockable(lock_file)  # still held after the nested exits
    assert launch_opens == 1  # the nested acquires opened nothing
    assert _lockable(lock_file)


async def test_a_task_spawned_inside_the_lock_does_not_inherit_it(tmp_path):
    """Re-entrancy is per owning task, not per context: a child task waits for the holder."""
    child_in = asyncio.Event()

    async def child():
        async with lifecycle.launch_lock(tmp_path):
            child_in.set()

    async with lifecycle.launch_lock(tmp_path):
        task = asyncio.create_task(child())
        await asyncio.sleep(0.2)
        assert not child_in.is_set()
    await asyncio.wait_for(task, 0.5)
    assert child_in.is_set()


async def test_held_is_empty_after_an_exception_in_the_body(tmp_path, lock_file):
    with pytest.raises(ValueError):
        async with lifecycle.launch_lock(tmp_path):
            raise ValueError("boom")
    assert lifecycle._HELD == {}
    assert _lockable(lock_file)


async def test_on_wait_is_awaited_once_when_the_lock_stays_held(monkeypatch, tmp_path, lock_file):
    monkeypatch.setattr(client_mod, "_QUEUE_VISIBILITY_SECONDS", 0.05)
    holder = _hold(lock_file)
    told: list[int] = []

    async def on_wait():
        told.append(1)

    task = asyncio.create_task(_enter(tmp_path, on_wait))
    await asyncio.sleep(0.4)
    os.close(holder)
    await asyncio.wait_for(task, 0.5)
    assert told == [1]


async def _enter(config_dir, on_wait=None):
    async with lifecycle.launch_lock(config_dir, on_wait=on_wait):
        pass


async def test_on_wait_is_never_awaited_when_the_lock_is_free(monkeypatch, tmp_path):
    monkeypatch.setattr(client_mod, "_QUEUE_VISIBILITY_SECONDS", 0.05)
    told: list[int] = []

    async def on_wait():
        told.append(1)

    await _enter(tmp_path, on_wait)
    assert told == []


async def test_on_wait_threshold_is_read_at_call_time(monkeypatch, tmp_path, lock_file):
    """A short wait under the default 2.0 s tells nobody; the constant is read through the module."""
    holder = _hold(lock_file)
    told: list[int] = []

    async def on_wait():
        told.append(1)

    task = asyncio.create_task(_enter(tmp_path, on_wait))
    await asyncio.sleep(0.2)
    os.close(holder)
    await asyncio.wait_for(task, 0.5)
    assert told == []


async def test_cancelling_a_waiter_leaves_the_file_unlocked(tmp_path, lock_file):
    holder = _hold(lock_file)
    task = asyncio.create_task(_enter(tmp_path))
    await asyncio.sleep(0.15)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    os.close(holder)
    assert _lockable(lock_file)


async def test_cancelling_the_holder_leaves_the_file_unlocked(tmp_path, lock_file):
    inside = asyncio.Event()

    async def hold_forever():
        async with lifecycle.launch_lock(tmp_path):
            inside.set()
            await asyncio.sleep(60)

    task = asyncio.create_task(hold_forever())
    await asyncio.wait_for(inside.wait(), 1.0)
    assert not _lockable(lock_file)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _lockable(lock_file)


async def test_the_held_fd_is_not_inheritable_and_the_file_is_private(tmp_path, open_spy, lock_file):
    async with lifecycle.launch_lock(tmp_path):
        (fd,) = open_spy.fds
        assert os.get_inheritable(fd) is False
        mode = stat.S_IMODE(os.stat(lock_file).st_mode)
        assert mode & 0o077 == 0


async def test_without_fcntl_launches_still_serialise_and_one_warning_names_the_cause(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setattr(lifecycle, "fcntl", None)
    caplog.set_level(logging.WARNING, logger=lifecycle.log.name)
    order: list[str] = []

    async def body(name):
        async with lifecycle.launch_lock(tmp_path):
            order.append(f"in:{name}")
            await asyncio.sleep(0.05)
            order.append(f"out:{name}")

    await asyncio.gather(body("a"), body("b"))
    assert order[1].startswith("out:") and order[3].startswith("out:")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "launch lock" in warnings[0].getMessage()
    assert not server.lock_path(tmp_path).exists()  # no file was opened at all


async def test_a_symlinked_lock_file_is_refused_and_never_written(tmp_path, caplog):
    lock = server.lock_path(tmp_path)
    lock.parent.mkdir(parents=True)
    target = tmp_path / "elsewhere"
    lock.symlink_to(target)
    caplog.set_level(logging.WARNING, logger=lifecycle.log.name)
    order: list[str] = []

    async def body(name):
        async with lifecycle.launch_lock(tmp_path):
            order.append(f"in:{name}")
            await asyncio.sleep(0.05)
            order.append(f"out:{name}")

    await asyncio.gather(body("a"), body("b"))
    assert order[1].startswith("out:") and order[3].startswith("out:")
    assert not target.exists()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "launch lock" in warnings[0].getMessage()


async def test_a_dead_holder_never_leaves_a_stale_lock(tmp_path, lock_file):
    """Closing the fd is what a dying process does: the next launch gets the lock at once."""
    holder = _hold(lock_file)
    os.close(holder)
    await asyncio.wait_for(_enter(tmp_path), 0.5)
    assert lock_file.exists() and lock_file.read_bytes() == b""  # the empty file left behind is harmless
