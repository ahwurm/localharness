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


@pytest.mark.asyncio
async def test_turn_completed_elapsed_tokens_matches_tiktoken(mock_llm_client, bus):
    # TELEM-02
    from localharness.agent.loop import AgentLoop
    from localharness.agent.context import ContextManager
    from localharness.agent.permissions import PermissionEvaluator
    from localharness.config.models import AgentConfig

    Resp = mock_llm_client.Response
    Usage = mock_llm_client.Usage
    responses = [
        Resp(
            content="all done",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        ),
    ]
    llm = mock_llm_client(responses)  # return_tuple=True by default
    cfg = AgentConfig(name="test-agent", role="Test agent.")
    ctx = ContextManager()
    perm = PermissionEvaluator()
    loop = AgentLoop(
        config=cfg,
        llm=llm,
        bus=bus,
        context_manager=ctx,
        tool_registry=None,
        permission_evaluator=perm,
    )

    await loop.run_turn("hello")

    captured = bus.history(event_types=[TurnCompleted])
    assert len(captured) == 1
    assert captured[0].elapsed_tokens == 15
    assert captured[0].input_tokens == 10
    assert captured[0].output_tokens == 5
    assert captured[0].tokens_estimated is False


# ---------- TELEM-01 stubs ----------

@pytest.mark.asyncio
async def test_build_messages_returns_budget():
    # TELEM-01
    from localharness.agent.context import ContextManager, TokenBudget

    cm = ContextManager(max_context_tokens=128_000)
    messages = [
        {"role": "system", "content": "you are a helpful agent"},
        {"role": "user", "content": "hi"},
    ]
    result, budget = await cm.build_messages(messages)

    assert isinstance(result, list)
    assert isinstance(budget, TokenBudget)
    assert budget.total_limit == 128_000
    assert budget.current_usage > 0
    assert 0.0 <= budget.usage_fraction <= 1.0


@pytest.mark.asyncio
async def test_heartbeat_emits_post_build_messages(mock_llm_client, bus):
    # TELEM-01
    from localharness.agent.loop import AgentLoop
    from localharness.agent.context import ContextManager
    from localharness.agent.permissions import PermissionEvaluator
    from localharness.config.models import AgentConfig
    from localharness.core.events import Heartbeat

    Resp = mock_llm_client.Response
    Usage = mock_llm_client.Usage
    responses = [Resp(content="done", usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12))]
    llm = mock_llm_client(responses)
    cfg = AgentConfig(name="test-agent", role="Test agent.")
    ctx = ContextManager(max_context_tokens=128_000)
    perm = PermissionEvaluator()
    loop = AgentLoop(
        config=cfg,
        llm=llm,
        bus=bus,
        context_manager=ctx,
        tool_registry=None,
        permission_evaluator=perm,
    )

    await loop.run_turn("hello")

    heartbeats = bus.history(event_types=[Heartbeat])
    assert len(heartbeats) >= 1
    for hb in heartbeats:
        assert hb.context_utilization_pct > 0.0
        assert hb.context_utilization_pct <= 100.0


@pytest.mark.asyncio
async def test_utilization_drops_after_compaction():
    # TELEM-01 — success criterion 4
    from localharness.agent.context import ContextManager
    ctx = ContextManager(max_context_tokens=10_000)
    pre, pre_budget = await ctx.build_messages([
        {"role": "system", "content": "x" * 2000},
        {"role": "user", "content": "y" * 2000},
    ])
    post, post_budget = await ctx.build_messages([
        {"role": "system", "content": "x"},
        {"role": "user", "content": "y"},
    ])
    assert post_budget.usage_fraction < pre_budget.usage_fraction
