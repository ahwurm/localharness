"""Phase 10 telemetry tests — failing stubs created by 10-00, made to pass by 10-01 / 10-02."""
from __future__ import annotations

import pytest

from localharness.agent.context import ContextManager, TokenBudget
from localharness.core.bus import EventBus
from localharness.core.events import Heartbeat, TurnCompleted


# ---------- TELEM-02 stubs ----------

@pytest.mark.asyncio
async def test_complete_native_returns_usage():
    # TELEM-02
    from unittest.mock import AsyncMock, MagicMock
    from localharness.provider.client import LLMClient, LLMConfig

    config = LLMConfig(
        base_url="http://localhost:11434/v1",
        model="test-model",
        api_key="x",
        tool_call_mode="native",
        is_local=True,
        timeout_seconds=300.0,
    )
    client = LLMClient(config)
    fake_message = MagicMock(content="hi", tool_calls=None)
    fake_usage = MagicMock(prompt_tokens=42, completion_tokens=7, total_tokens=49)
    fake_response = MagicMock(choices=[MagicMock(message=fake_message)], usage=fake_usage)
    client._client = MagicMock()
    client._client.chat.completions.create = AsyncMock(return_value=fake_response)

    result = await client._complete_native(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        stream=False,
    )

    assert isinstance(result, tuple) and len(result) == 2
    message, usage = result
    assert usage.prompt_tokens == 42
    assert usage.completion_tokens == 7


@pytest.mark.xfail(strict=True, reason="TELEM-02: TurnCompleted.elapsed_tokens must equal sum of provider usage")
@pytest.mark.asyncio
async def test_turn_completed_elapsed_tokens_matches_tiktoken(mock_llm_client, bus):
    # TELEM-02
    # Wave 1 (10-01-02) will wire Session counters + emit elapsed_tokens
    assert False, "Wave 1 must accumulate usage into Session and emit it on TurnCompleted"


# ---------- TELEM-01 stubs ----------

@pytest.mark.xfail(strict=True, reason="TELEM-01: build_messages must return (messages, TokenBudget|None)")
@pytest.mark.asyncio
async def test_build_messages_returns_budget():
    # TELEM-01
    # Wave 2 (10-02-01) will change build_messages return type to tuple
    assert False, "Wave 2 must change build_messages to return (messages, TokenBudget|None)"


@pytest.mark.xfail(strict=True, reason="TELEM-01: Heartbeat must emit non-zero context_utilization_pct after build_messages")
@pytest.mark.asyncio
async def test_heartbeat_emits_post_build_messages(mock_llm_client, bus):
    # TELEM-01
    # Wave 2 (10-02-02) will move heartbeat emission to AFTER build_messages and compute pct from budget
    assert False, "Wave 2 must reorder heartbeat emission and compute pct from TokenBudget.usage_fraction"


@pytest.mark.xfail(strict=True, reason="TELEM-01: context_utilization_pct must drop after compaction")
@pytest.mark.asyncio
async def test_utilization_drops_after_compaction(mock_llm_client, bus):
    # TELEM-01
    # Wave 2 (10-02-03) success criterion 4
    assert False, "Wave 2 must verify post-compaction heartbeat reflects shrunken context"
