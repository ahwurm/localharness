"""Phase 7 behaviour-audit gaps: proxy/protocol transport errors map to connection errors, the
degraded (no fcntl) launch lock is re-entrant, and a cancelled in-process waiter releases nothing
it did not take. Hermetic: tmp_path config dirs, no sockets, no subprocess."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from localharness.provider import lifecycle, server
from localharness.provider.client import LLMClient, LLMConfig, ProviderConnectionError

fcntl = pytest.importorskip("fcntl")


@pytest.fixture(autouse=True)
def _no_leaked_holds():
    yield
    assert lifecycle._HELD == {}


def _client() -> LLMClient:
    return LLMClient(LLMConfig(base_url="http://127.0.0.1:9/v1", model="m", is_local=False,
                               tool_call_mode="native"))


@pytest.mark.parametrize("exc", [httpx.ProxyError("proxy refused"),
                                 httpx.UnsupportedProtocol("no scheme")])
def test_proxy_and_protocol_errors_are_connection_errors(exc):
    out = _client()._wrap_error(exc)
    assert type(out) is ProviderConnectionError
    assert out.cause is exc


def _lockable(path) -> bool:
    import os
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


async def test_a_cancelled_in_process_waiter_leaves_the_holders_lock_held(tmp_path):
    inside, release = asyncio.Event(), asyncio.Event()

    async def holder():
        async with lifecycle.launch_lock(tmp_path):
            inside.set()
            await release.wait()

    async def waiter():
        async with lifecycle.launch_lock(tmp_path):
            pytest.fail("the waiter must never get in while the holder holds")

    h = asyncio.create_task(holder())
    await asyncio.wait_for(inside.wait(), 1.0)
    w = asyncio.create_task(waiter())
    await asyncio.sleep(0.15)
    w.cancel()
    with pytest.raises(asyncio.CancelledError):
        await w
    assert not _lockable(server.lock_path(tmp_path))  # holder's lock untouched
    release.set()
    await asyncio.wait_for(h, 1.0)
    assert _lockable(server.lock_path(tmp_path))


async def test_the_degraded_lock_re_enters_within_one_task(monkeypatch, tmp_path):
    monkeypatch.setattr(lifecycle, "fcntl", None)
    async with asyncio.timeout(1.0):
        async with lifecycle.launch_lock(tmp_path):
            async with lifecycle.launch_lock(tmp_path):
                pass
    # and a second task still serialises behind a holder afterwards
    async with lifecycle.launch_lock(tmp_path):
        pass
