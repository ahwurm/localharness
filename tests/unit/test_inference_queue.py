"""BUG #62: the inference gate must (a) FAIL FAST on a dead endpoint with a cheap TCP
reachability probe BEFORE consuming a queue slot, (b) SURFACE a long queue wait once past a
short threshold, and (c) bound the queue wait by a config ceiling — never the generation itself.

These exercise the gate mechanics directly (probe seam, semaphore ceiling, flock ceiling,
visibility log) with the HTTP layer mocked, so no real model server is needed."""
from __future__ import annotations

import asyncio
import logging
import socket
from types import SimpleNamespace

import pytest

from localharness.provider import client as client_mod
from localharness.provider.client import (
    LLMClient,
    LLMConfig,
    ProviderConnectionError,
    ProviderTimeoutError,
)

fcntl = pytest.importorskip("fcntl")


def _cfg(**kw) -> LLMConfig:
    kw.setdefault("base_url", "http://127.0.0.1:8000/v1")
    kw.setdefault("model", "m")
    kw.setdefault("timeout_seconds", 300.0)
    kw.setdefault("is_local", True)
    kw.setdefault("tool_call_mode", "native")
    return LLMConfig(**kw)


class _Tracker:
    """Fake completions endpoint recording how many times the HTTP path was actually reached."""

    def __init__(self):
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0

    async def create(self, **kwargs):
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        msg = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


def _wire(client: LLMClient, tracker: _Tracker) -> None:
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=tracker.create))
    )


class _CountingSem:
    """A semaphore wrapper that counts acquires — proves the gate slot is never consumed."""

    def __init__(self, value: int = 1):
        self.acquires = 0
        self._real = asyncio.Semaphore(value)

    async def acquire(self):
        self.acquires += 1
        return await self._real.acquire()

    def release(self):
        self._real.release()

    def locked(self):
        return self._real.locked()


@pytest.fixture(autouse=True)
def _fresh_queue_state(monkeypatch):
    """Fresh per-test gate state: a 1-permit semaphore, flock off, default threshold, empty
    probe cache. The probe ENABLE flag is set per-test (the global conftest disables it)."""
    monkeypatch.setattr(client_mod, "_inference_sem", asyncio.Semaphore(1))
    monkeypatch.setattr(client_mod, "_INFERENCE_LOCK_ENABLED", False)
    monkeypatch.setattr(client_mod, "_QUEUE_VISIBILITY_SECONDS", 2.0)
    client_mod._probe_cache.clear()
    yield
    client_mod._probe_cache.clear()


# ---------------------------------------------------------------------------
# (a) FAIL FAST on a dead endpoint — before entering the queue.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dead_endpoint_fails_fast_without_acquiring_gate(monkeypatch):
    """An unreachable endpoint raises ProviderConnectionError immediately, never touching the
    semaphore (no queue slot consumed) and never reaching the HTTP path."""
    monkeypatch.setattr(client_mod, "_INFERENCE_PROBE_ENABLED", True)

    async def _unreachable(host, port):
        return False

    monkeypatch.setattr(client_mod, "_probe_reachable", _unreachable)
    sem = _CountingSem(1)
    monkeypatch.setattr(client_mod, "_inference_sem", sem)

    c = LLMClient(_cfg())
    tracker = _Tracker()
    _wire(c, tracker)

    t0 = asyncio.get_event_loop().time()
    with pytest.raises(ProviderConnectionError, match="unreachable"):
        await c.complete([{"role": "user", "content": "hi"}])
    elapsed = asyncio.get_event_loop().time() - t0

    assert sem.acquires == 0          # no gate slot consumed
    assert tracker.calls == 0         # HTTP never reached
    assert elapsed < 1.0              # fast — not a 90s queue wait


@pytest.mark.asyncio
async def test_probe_reachable_true_for_open_port_false_for_closed():
    """The real TCP probe (connect + close, no HTTP route) returns True for a listening port
    and False for a closed one."""
    # A real listening socket on an ephemeral port.
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    open_port = srv.getsockname()[1]
    try:
        assert await client_mod._probe_reachable("127.0.0.1", open_port) is True
    finally:
        srv.close()
    # Now that port is closed → connection refused → unreachable.
    client_mod._probe_cache.clear()
    assert await client_mod._probe_reachable("127.0.0.1", open_port) is False


@pytest.mark.asyncio
async def test_healthy_path_serializes_with_probe_on(monkeypatch):
    """Probe ON + reachable endpoint: the healthy path is untouched — requests still succeed and
    still serialize one-at-a-time through the gate."""
    monkeypatch.setattr(client_mod, "_INFERENCE_PROBE_ENABLED", True)

    async def _reachable(host, port):
        return True

    monkeypatch.setattr(client_mod, "_probe_reachable", _reachable)

    c = LLMClient(_cfg())
    tracker = _Tracker()
    _wire(c, tracker)
    await asyncio.gather(*[c.complete([{"role": "user", "content": "hi"}]) for _ in range(5)])

    assert tracker.calls == 5
    assert tracker.max_in_flight == 1  # serialization preserved


# ---------------------------------------------------------------------------
# (c) CEILING bounds ONLY the queue wait — on both the flock and the semaphore.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ceiling_fires_with_message_when_flock_held(monkeypatch, tmp_path):
    """Holding the cross-process flock, a request with a small queue-wait ceiling gives up with a
    clear ProviderTimeoutError naming the model slot — never blocking unbounded."""
    monkeypatch.setattr(client_mod, "_INFERENCE_PROBE_ENABLED", False)  # isolate to the flock wait
    lock_path = str(tmp_path / "gate.lock")
    monkeypatch.setattr(client_mod, "_inference_lock_path", lambda base_url: lock_path)
    monkeypatch.setattr(client_mod, "_INFERENCE_LOCK_ENABLED", True)
    monkeypatch.setattr(client_mod, "_inference_sem", asyncio.Semaphore(1))

    held = client_mod.os.open(lock_path, client_mod.os.O_CREAT | client_mod.os.O_RDWR, 0o666)
    fcntl.flock(held, fcntl.LOCK_EX)
    try:
        cfg = _cfg(queue_wait_seconds=0.4)
        t0 = asyncio.get_event_loop().time()
        with pytest.raises(ProviderTimeoutError, match="model slot"):
            async with client_mod._inference_gate(cfg):
                pass  # pragma: no cover — the gate must never open while the flock is held
        elapsed = asyncio.get_event_loop().time() - t0
        assert 0.3 <= elapsed < 2.0  # bounded by the ceiling, not unbounded
    finally:
        fcntl.flock(held, fcntl.LOCK_UN)
        client_mod.os.close(held)


@pytest.mark.asyncio
async def test_ceiling_fires_when_semaphore_held(monkeypatch):
    """The ceiling also bounds the in-process semaphore wait: with the sole permit held, a small
    ceiling gives up with ProviderTimeoutError."""
    monkeypatch.setattr(client_mod, "_INFERENCE_PROBE_ENABLED", False)
    sem = asyncio.Semaphore(1)
    await sem.acquire()  # hold the only permit
    monkeypatch.setattr(client_mod, "_inference_sem", sem)
    try:
        cfg = _cfg(queue_wait_seconds=0.4)
        t0 = asyncio.get_event_loop().time()
        with pytest.raises(ProviderTimeoutError, match="model slot"):
            async with client_mod._inference_gate(cfg):
                pass  # pragma: no cover
        assert asyncio.get_event_loop().time() - t0 < 2.0
    finally:
        sem.release()


@pytest.mark.asyncio
async def test_ceiling_zero_disables_the_bound(monkeypatch):
    """queue_wait_seconds=0 disables the ceiling: the gate waits (no ProviderTimeoutError) and
    proceeds once the slot frees."""
    monkeypatch.setattr(client_mod, "_INFERENCE_PROBE_ENABLED", False)
    monkeypatch.setattr(client_mod, "_QUEUE_VISIBILITY_SECONDS", 0.05)
    sem = asyncio.Semaphore(1)
    await sem.acquire()
    monkeypatch.setattr(client_mod, "_inference_sem", sem)

    cfg = _cfg(queue_wait_seconds=0)
    entered = asyncio.Event()

    async def _enter():
        async with client_mod._inference_gate(cfg):
            entered.set()

    task = asyncio.create_task(_enter())
    await asyncio.sleep(0.3)                 # far longer than any small ceiling would allow
    assert not task.done()                   # still waiting — the ceiling did NOT fire
    sem.release()                            # free the slot
    await asyncio.wait_for(task, timeout=2)  # now it proceeds
    assert entered.is_set()


# ---------------------------------------------------------------------------
# (b) VISIBILITY — one honest INFO signal once the wait passes the threshold.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_visibility_info_emitted_once_after_threshold(monkeypatch, caplog):
    """Once a queue wait passes the visibility threshold, exactly one INFO is logged naming the
    in-flight request — not spammed every poll."""
    monkeypatch.setattr(client_mod, "_INFERENCE_PROBE_ENABLED", False)
    monkeypatch.setattr(client_mod, "_QUEUE_VISIBILITY_SECONDS", 0.1)
    sem = asyncio.Semaphore(1)
    await sem.acquire()
    monkeypatch.setattr(client_mod, "_inference_sem", sem)

    cfg = _cfg(queue_wait_seconds=None)  # ceiling disabled — isolate the visibility behaviour
    entered = asyncio.Event()

    async def _enter():
        async with client_mod._inference_gate(cfg):
            entered.set()

    with caplog.at_level(logging.INFO, logger="localharness.provider.client"):
        task = asyncio.create_task(_enter())
        await asyncio.sleep(0.35)  # past the 0.1s threshold
        waiting = [r for r in caplog.records if "waiting for a model slot" in r.getMessage()]
        assert len(waiting) == 1   # surfaced once, not per poll
        sem.release()
        await asyncio.wait_for(task, timeout=2)
    assert entered.is_set()


# ---------------------------------------------------------------------------
# Config wiring: the ceiling is a provider config field (default 600), threaded to LLMConfig.
# ---------------------------------------------------------------------------


def test_provider_config_queue_wait_default_and_override():
    from localharness.config.models import ProviderConfig

    p = ProviderConfig(provider_type="vllm", base_url="http://x/v1", default_model="m")
    assert p.inference_queue_wait_seconds == 600.0  # GENEROUS default (multi-session is legit)

    p2 = ProviderConfig(provider_type="vllm", base_url="http://x/v1", default_model="m",
                        inference_queue_wait_seconds=30.0)
    assert p2.inference_queue_wait_seconds == 30.0
    p3 = ProviderConfig(provider_type="vllm", base_url="http://x/v1", default_model="m",
                        inference_queue_wait_seconds=0)
    assert p3.inference_queue_wait_seconds == 0  # 0 disables the bound


def test_llmconfig_queue_wait_default():
    assert LLMConfig(base_url="http://x/v1", model="m").queue_wait_seconds == 600.0
