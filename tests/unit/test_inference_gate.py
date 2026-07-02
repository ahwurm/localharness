"""Inference gate: local requests serialize in-process, remote stay ungated, and the
xml→fallback path never deadlocks on the 1-permit gate (it re-enters via a second acquire)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest

from localharness.provider import client as client_mod
from localharness.provider.client import LLMClient, LLMConfig


def _cfg(**kw) -> LLMConfig:
    kw.setdefault("base_url", "http://127.0.0.1:9")
    kw.setdefault("model", "m")
    kw.setdefault("timeout_seconds", 300.0)
    return LLMConfig(**kw)


class _Tracker:
    """Fake completions endpoint that records overlap; optionally rejects the first call."""

    def __init__(self, fail_first: Exception | None = None):
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls = 0
        self._fail_first = fail_first

    async def create(self, **kwargs):
        self.calls += 1
        if self._fail_first is not None and self.calls == 1:
            exc, self._fail_first = self._fail_first, None
            raise exc
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


@pytest.fixture(autouse=True)
def _fresh_gate(monkeypatch):
    # Module-level gate state: fresh 1-permit semaphore per test; flock layer is
    # cross-process scope, off for unit tests.
    monkeypatch.setattr(client_mod, "_inference_sem", asyncio.Semaphore(1))
    monkeypatch.setattr(client_mod, "_INFERENCE_LOCK_ENABLED", False)


async def test_local_requests_serialize():
    c = LLMClient(_cfg(is_local=True, tool_call_mode="native"))
    t = _Tracker()
    _wire(c, t)
    await asyncio.gather(*[c.complete([{"role": "user", "content": "hi"}]) for _ in range(5)])
    assert t.calls == 5
    assert t.max_in_flight == 1


async def test_remote_requests_ungated():
    c = LLMClient(_cfg(is_local=False, tool_call_mode="native"))
    t = _Tracker()
    _wire(c, t)
    await asyncio.gather(*[c.complete([{"role": "user", "content": "hi"}]) for _ in range(5)])
    assert t.max_in_flight == 5


async def test_xml_fallback_does_not_deadlock_gate():
    bad = openai.BadRequestError(
        "no tools",
        response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
        body=None,
    )
    c = LLMClient(_cfg(is_local=True, tool_call_mode="xml"))
    t = _Tracker(fail_first=bad)
    _wire(c, t)
    msg, _usage = await asyncio.wait_for(
        c.complete([{"role": "user", "content": "hi"}]), timeout=5
    )
    assert t.calls == 2  # rejected create, then fallback create — gate released between them
    assert msg.content == "ok"
