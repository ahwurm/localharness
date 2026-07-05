"""Phase-36 idle-window LLM safety path (idle_llm.py) — the ONE cancellable, context-
bounded, grounded route every model-look shares.

Task 1 locks the machine-safety spine: a SET cancel event returns None fast and truly
cancels the generation (freeze regression), the real-client->text adapter, and the
char-cap context bound. (test_inference_gate_cancel.py locks the gate-level fix; this
locks the idle-window race that rides on top of it.)
"""
import asyncio
import time
from types import SimpleNamespace

import pytest

from localharness.memory.idle_llm import (
    LLMTextAdapter,
    complete_cancellable,
)


class _SlowLLM:
    """A generation that never finishes on its own; records true cancellation."""

    def __init__(self, delay: float = 10.0):
        self.delay = delay
        self.cancelled = False

    async def complete(self, prompt: str) -> str:
        try:
            await asyncio.sleep(self.delay)
            return "slow result"  # pragma: no cover — must be cancelled first
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class _FakeLLM:
    def __init__(self, text: str):
        self.text = text

    async def complete(self, prompt: str) -> str:
        return self.text


class _RecordingLLM:
    """Records the prompt it actually saw — proves char_cap truncates BEFORE the call."""

    def __init__(self, text: str = "ok"):
        self.text = text
        self.seen: str | None = None

    async def complete(self, prompt: str) -> str:
        self.seen = prompt
        return self.text


class _StubClient:
    """Mimics the real LLMClient.complete(messages, tools, stream) -> (message, usage)."""

    def __init__(self, ret):
        self.ret = ret
        self.seen_messages = None

    async def complete(self, messages, tools=None, stream=False):
        self.seen_messages = messages
        return self.ret


@pytest.mark.asyncio
async def test_set_cancel_event_returns_none_and_cancels_generation():
    """A pre-SET cancel event (a user turn already waiting) returns None promptly and
    truly cancels the in-flight generation — never the 10s slow result, no hang."""
    llm = _SlowLLM(delay=10.0)
    cancel = asyncio.Event()
    cancel.set()

    t0 = time.monotonic()
    result = await asyncio.wait_for(complete_cancellable(llm, "prompt", cancel), timeout=3.0)

    assert result is None
    assert time.monotonic() - t0 < 3.0  # nobody waited behind the generation
    assert llm.cancelled  # the gen task was truly cancelled → inference gate released


@pytest.mark.asyncio
async def test_uncancelled_completion_returns_value():
    """An unset cancel event lets the generation win the race and return its text."""
    llm = _FakeLLM("hello")
    result = await complete_cancellable(llm, "prompt", asyncio.Event())
    assert result == "hello"


@pytest.mark.asyncio
async def test_adapter_maps_message_usage_to_content_string():
    """LLMTextAdapter turns the real (message, usage) tuple into message.content, and
    wraps the prompt as a single user-role message dict (Message = dict[str, Any])."""
    client = _StubClient((SimpleNamespace(content="chapter text"), None))
    adapter = LLMTextAdapter(client)

    assert await adapter.complete("p") == "chapter text"
    assert client.seen_messages == [{"role": "user", "content": "p"}]


@pytest.mark.asyncio
async def test_char_cap_truncates_before_call():
    """char_cap bounds the prompt BEFORE the call — the machine-safety context bound so
    an idle look can never launch an unattended long-context prefill."""
    llm = _RecordingLLM("ok")
    await complete_cancellable(llm, "x" * 10000, asyncio.Event(), char_cap=100)
    assert llm.seen is not None and len(llm.seen) <= 100
