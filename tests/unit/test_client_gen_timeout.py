"""#92 — LLMClient.complete gains a per-call `gen_timeout` bounding ONLY the generation (the work
after the inference permit is acquired). Tier-2 classify uses it so a one-word verdict can be
bounded at 5s of GENERATION, independent of however long the permit-wait took.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace as NS

import pytest

from localharness.provider.client import LLMClient, LLMConfig


def _cfg(mode: str = "native") -> LLMConfig:
    # is_local=False skips the inference gate + local-timeout floor, so gen_timeout is the only bound.
    return LLMConfig(base_url="http://127.0.0.1:9/v1", model="m", timeout_seconds=300.0,
                     tool_call_mode=mode, is_local=False)


def _ok_completion(text: str) -> NS:
    return NS(choices=[NS(message=NS(content=text, tool_calls=None), finish_reason="stop")],
              usage=None)


async def test_gen_timeout_cuts_a_hung_generation():
    client = LLMClient(_cfg())

    async def slow_create(**kwargs):
        await asyncio.sleep(5.0)  # generation that never returns
        return _ok_completion("late")

    client._client = NS(chat=NS(completions=NS(create=slow_create)))
    t0 = time.monotonic()
    with pytest.raises(Exception) as ei:
        await client.complete([{"role": "user", "content": "hi"}], gen_timeout=0.05)
    assert not isinstance(ei.value, TypeError)  # a real timeout, not "unexpected kwarg gen_timeout"
    assert time.monotonic() - t0 < 1.0  # cut at ~0.05s, not after the full 5s


async def test_gen_timeout_none_is_unbounded_passthrough():
    client = LLMClient(_cfg())

    async def fast_create(**kwargs):
        return _ok_completion("hello")

    client._client = NS(chat=NS(completions=NS(create=fast_create)))
    msg, _usage = await client.complete([{"role": "user", "content": "hi"}])  # gen_timeout default None
    assert msg.content == "hello"
