"""The two primitives an in-process session restart needs (52-RULINGS R1, Option B).

`localharness start` rebuilds its session after `/plugins enable` by running `_start_async` again
under a second `asyncio.run()` in the same process. Two things have to cross that boundary:

1. The inference gate. `provider/client.py` holds a module-level `asyncio.Semaphore`, and an
   asyncio primitive binds to the event loop it is first CONTENDED in: the next loop's first
   contended acquire raises "bound to a different event loop". `reset_inference_gate()` makes a
   fresh one between loops. The hazard itself is pinned too, so a Python that stops binding is
   noticed rather than silently relied on.
2. The conversation. The exact model-side messages live only in `AgentLoop._conversation`, the
   prior-session context only in `_prior_session_context`; `resume_state()` / `resume()` hand
   both to the next AgentLoop as plain, copied data.

The gate tests call `asyncio.run` themselves, so they are plain `def`s: asyncio_mode is "auto",
and an `async def` test would already be running inside a loop.
"""
from __future__ import annotations

import asyncio

import pytest

from localharness.provider import client
from tests.unit.test_compact_md_disable_sentinel import _loop


async def _contend() -> None:
    """Hold the gate's permit and park a second acquire behind it (the parked acquire makes a
    future on THIS loop, which is what binds the semaphore), then release both."""
    sem = client._inference_sem
    await sem.acquire()
    waiter = asyncio.ensure_future(sem.acquire())  # parks on a future of THIS loop
    await asyncio.sleep(0)
    sem.release()
    await waiter
    sem.release()


def test_without_a_reset_a_second_event_loop_cannot_use_a_contended_gate(monkeypatch):
    monkeypatch.setattr(client, "_inference_sem", asyncio.Semaphore(1))
    asyncio.run(_contend())
    with pytest.raises(RuntimeError, match="different event loop"):
        asyncio.run(_contend())


def test_reset_inference_gate_makes_a_fresh_gate_for_the_next_loop(monkeypatch):
    monkeypatch.setattr(client, "_inference_sem", asyncio.Semaphore(1))
    # One permit whatever LOCALHARNESS_MAX_CONCURRENT_INFERENCE says, so the fresh gate is
    # contended in the second loop too.
    monkeypatch.setattr(client, "_MAX_CONCURRENT_INFERENCE", 1)
    asyncio.run(_contend())
    first = client._inference_sem
    client.reset_inference_gate()
    asyncio.run(_contend())
    assert client._inference_sem is not first
    assert client._inference_sem._value == client._MAX_CONCURRENT_INFERENCE
    # The fresh gate keeps the configured concurrency, not a hard-coded one.
    monkeypatch.setattr(client, "_MAX_CONCURRENT_INFERENCE", 3)
    client.reset_inference_gate()
    assert client._inference_sem._value == 3


def test_resume_round_trips_the_conversation_and_prior_context_as_copies(tmp_path):
    conv = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    loop = _loop(tmp_path)
    loop.resume(conv, "PRIOR")
    assert loop.resume_state() == (conv, "PRIOR")
    conv.append({"role": "user", "content": "typed after the hand-over"})
    handed, _ = loop.resume_state()
    handed.append({"role": "user", "content": "the next loop's own"})
    assert loop.resume_state() == (conv[:2], "PRIOR")


def test_a_fresh_loop_has_nothing_to_hand_over(tmp_path):
    assert _loop(tmp_path).resume_state() == ([], "")
