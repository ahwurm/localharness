"""Phase-31 critic BLOCKER 1 regression: cancelling a task WAITING on the inference
gate must never wedge the shared lockfile.

The old acquire (`await asyncio.to_thread(fcntl.flock, fd, LOCK_EX)`) parked a real OS
thread; cancellation closed the fd in `finally` while the thread's flock was in-flight —
the kernel then granted the lock to a struct-file no fd named, so it could never be
released: every later gate acquire blocked forever. The LOCK_NB poll-loop acquire is
cancellation-safe by construction; this test locks the wedge out of coming back.
"""
import asyncio

import pytest

fcntl = pytest.importorskip("fcntl")

from localharness.provider.client import _inference_gate  # noqa: E402


class _Cfg:
    is_local = True
    base_url = "http://localhost:59999/test-gate-cancel"


@pytest.mark.asyncio
async def test_cancel_while_waiting_does_not_wedge_the_lock(tmp_path, monkeypatch):
    import localharness.provider.client as client_mod

    lock_path = str(tmp_path / "gate.lock")
    monkeypatch.setattr(client_mod, "_inference_lock_path", lambda base_url: lock_path)
    monkeypatch.setattr(client_mod, "_INFERENCE_LOCK_ENABLED", True)
    monkeypatch.setattr(client_mod, "_inference_sem", asyncio.Semaphore(3))

    holder_entered = asyncio.Event()
    release_holder = asyncio.Event()

    async def holder():
        async with _inference_gate(_Cfg()):
            holder_entered.set()
            await release_holder.wait()

    async def waiter():
        async with _inference_gate(_Cfg()):
            pass  # pragma: no cover — must be cancelled before entry

    h = asyncio.create_task(holder())
    await asyncio.wait_for(holder_entered.wait(), timeout=5)

    w = asyncio.create_task(waiter())
    await asyncio.sleep(0.2)  # waiter is now polling for the lock
    w.cancel()
    with pytest.raises(asyncio.CancelledError):
        await w

    release_holder.set()
    await asyncio.wait_for(h, timeout=5)

    # THE regression assertion: after holder release + waiter cancellation, a third
    # acquirer gets the gate promptly. Under the old code this blocked forever
    # (orphaned-thread flock on a closed fd owned the lock with no releaser).
    async def third():
        async with _inference_gate(_Cfg()):
            return True

    assert await asyncio.wait_for(third(), timeout=5) is True
