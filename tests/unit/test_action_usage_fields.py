"""Every model response names its model, tokens and latency, and history rows copy them.

An `Action` row says which model answered, what the provider reported the response cost and how
long the completion call took. The `llm_response` row holds the usage; the `tool_call` rows of
that response carry the model and the latency but no tokens, so summing a history file counts
each response once. `history.jsonl` copies the values into its existing keys and stays mode 0600.
"""
import asyncio
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from localharness.agent.context import ContextManager
from localharness.agent.loop import AgentLoop
from localharness.agent.permissions import PermissionEvaluator
from localharness.config.models import AgentConfig, PermissionConfig
from localharness.core.events import Action
from localharness.memory.sqlite import MemoryStore
from localharness.provider.client import ProviderDegenerateError, ProviderReasoningBudgetError
from localharness.tools import ToolResult
from tests.conftest import FakeCompletionUsage, FakeLLMResponse, FakeToolCall, MockLLMClient


class _Registry:
    """One tool, `bash`, that always succeeds."""

    def get_tools_for_agent(self, agent_id, division_id, tool_config, **kw):
        return {}

    async def dispatch(self, name, arguments, agent_id, division_id, tool_config, **kw):
        return ToolResult(output="ok", success=True)


def _loop(llm, bus) -> AgentLoop:
    return AgentLoop(
        config=AgentConfig(name="usage-agent", role="Test agent.",
                           permissions=PermissionConfig(mode="unattended")),
        llm=llm,
        bus=bus,
        context_manager=ContextManager(),
        tool_registry=_Registry(),
        permission_evaluator=PermissionEvaluator(),
    )


def _actions(bus) -> list[Action]:
    return list(bus.history(event_types=[Action]))


def _tool_then_done(first_usage, second_usage) -> list[FakeLLMResponse]:
    return [
        FakeLLMResponse(
            content=None,
            tool_calls=[FakeToolCall(id="tc-1", name="bash", arguments={"cmd": "ls"})],
            usage=first_usage,
        ),
        FakeLLMResponse(content="Done.", usage=second_usage),
    ]


# ---------------------------------------------------------------------------
# The event: four optional fields
# ---------------------------------------------------------------------------

def test_the_four_fields_default_to_none_and_old_lines_still_parse():
    a = Action(agent_id="a", session_id="s", action_type="llm_response")
    assert (a.model, a.prompt_tokens, a.completion_tokens, a.latency_ms) == (None,) * 4

    old_line = a.model_dump_json(exclude={"model", "prompt_tokens", "completion_tokens", "latency_ms"})
    assert "latency_ms" not in json.loads(old_line)
    assert Action.model_validate_json(old_line).latency_ms is None


# ---------------------------------------------------------------------------
# The loop: every response-bearing publish
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_two_response_turn_carries_model_tokens_and_latency(bus):
    llm = MockLLMClient(_tool_then_done(
        FakeCompletionUsage(prompt_tokens=120, completion_tokens=7),
        FakeCompletionUsage(prompt_tokens=150, completion_tokens=3),
    ))
    llm.config.model = "qwen-test"
    await _loop(llm, bus).run_turn("task")

    rows = _actions(bus)
    first, call, second = [r for r in rows if r.action_type in ("llm_response", "tool_call")]
    assert (first.action_type, call.action_type, second.action_type) == (
        "llm_response", "tool_call", "llm_response"
    )
    assert (first.model, first.prompt_tokens, first.completion_tokens) == ("qwen-test", 120, 7)
    assert isinstance(first.latency_ms, int) and first.latency_ms >= 0
    # The tool call names the response that produced it, but holds none of its usage.
    assert call.model == "qwen-test"
    assert call.prompt_tokens is None and call.completion_tokens is None
    assert call.latency_ms == first.latency_ms
    assert (second.model, second.prompt_tokens, second.completion_tokens) == ("qwen-test", 150, 3)


@pytest.mark.asyncio
async def test_a_response_with_no_usage_has_no_tokens_but_still_a_model(bus):
    llm = MockLLMClient([FakeLLMResponse(content="Done.", usage=None)])
    llm.config.model = "qwen-test"
    await _loop(llm, bus).run_turn("task")

    (row,) = [r for r in _actions(bus) if r.action_type == "llm_response"]
    assert row.model == "qwen-test"
    assert row.prompt_tokens is None and row.completion_tokens is None


@pytest.mark.asyncio
async def test_latency_is_the_wall_time_of_the_completion_call(bus):
    class Slow(MockLLMClient):
        async def stream_complete(self, messages=None, tools=None, on_token=None, **kw):
            await asyncio.sleep(0.05)
            return await super().stream_complete(messages=messages, tools=tools,
                                                 on_token=on_token, **kw)

    llm = Slow([FakeLLMResponse(content="Done.")])
    await _loop(llm, bus).run_turn("task")

    (row,) = [r for r in _actions(bus) if r.action_type == "llm_response"]
    assert row.latency_ms >= 40


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error, finish_reason",
    [
        (ProviderDegenerateError("looped", unit="1, ", repeats=80, chars=1_500), "degenerate"),
        (ProviderReasoningBudgetError(100, 450), "reasoning_cut"),
    ],
)
async def test_the_abort_branches_carry_model_and_latency_but_no_tokens(bus, error, finish_reason):
    class FailsOnce(MockLLMClient):
        failed = False

        async def stream_complete(self, messages=None, tools=None, on_token=None, **kw):
            if not self.failed:
                self.failed = True
                await asyncio.sleep(0.02)
                raise error
            return await super().stream_complete(messages=messages, tools=tools,
                                                 on_token=on_token, **kw)

    llm = FailsOnce([FakeLLMResponse(content="Done.",
                                     usage=FakeCompletionUsage(prompt_tokens=9, completion_tokens=2))])
    llm.config.model = "qwen-test"
    await _loop(llm, bus).run_turn("task")

    aborted = [r for r in _actions(bus) if r.finish_reason == finish_reason]
    assert len(aborted) == 1
    assert aborted[0].model == "qwen-test"
    assert isinstance(aborted[0].latency_ms, int) and aborted[0].latency_ms >= 15
    assert aborted[0].prompt_tokens is None and aborted[0].completion_tokens is None


@pytest.mark.asyncio
async def test_non_str_model_and_non_int_usage_become_none_and_the_turn_completes(bus):
    llm = MockLLMClient([FakeLLMResponse(
        content="Done.",
        usage=SimpleNamespace(prompt_tokens=MagicMock(), completion_tokens=MagicMock(),
                              total_tokens=0),
    )])
    llm.config.model = MagicMock()
    summary = await _loop(llm, bus).run_turn("task")

    assert summary == "Done."
    (row,) = [r for r in _actions(bus) if r.action_type == "llm_response"]
    assert row.model is None
    assert row.prompt_tokens is None and row.completion_tokens is None


@pytest.mark.asyncio
async def test_a_bool_usage_value_is_not_a_token_count(bus):
    llm = MockLLMClient([FakeLLMResponse(
        content="Done.",
        usage=SimpleNamespace(prompt_tokens=True, completion_tokens=5, total_tokens=0),
    )])
    await _loop(llm, bus).run_turn("task")

    (row,) = [r for r in _actions(bus) if r.action_type == "llm_response"]
    assert row.prompt_tokens is None and row.completion_tokens == 5


# ---------------------------------------------------------------------------
# History rows copy the event's values
# ---------------------------------------------------------------------------

@pytest.fixture
async def store(tmp_path: Path):
    s = MemoryStore(agent_id="usage-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


def _history_records(store: MemoryStore) -> list[dict[str, Any]]:
    path = Path(store._history_path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_history_rows_copy_model_tokens_and_latency(store: MemoryStore):
    base = dict(agent_id="usage-agent", session_id="s1")
    await store._on_action(Action(**base, action_type="llm_response", content="hi", model="m",
                                  prompt_tokens=120, completion_tokens=7, latency_ms=85))
    await store._on_action(Action(**base, action_type="tool_call", tool_call_id="tc-1",
                                  tool_name="bash", model="m", latency_ms=85))
    await store._on_action(Action(**base, action_type="llm_response", content="bare"))

    answered, called, bare = _history_records(store)
    assert (answered["model"], answered["tokens_in"], answered["tokens_out"],
            answered["latency_ms"], answered["v"]) == ("m", 120, 7, 85, 1)
    assert (called["model"], called["tokens_in"], called["tokens_out"],
            called["latency_ms"], called["v"]) == ("m", 0, 0, 85, 1)
    assert (bare["model"], bare["tokens_in"], bare["tokens_out"], bare["latency_ms"]) == ("", 0, 0, 0)
    assert stat.S_IMODE(os.stat(store._history_path).st_mode) == 0o600


@pytest.mark.asyncio
async def test_a_foreign_event_object_still_writes_the_placeholders(store: MemoryStore):
    from datetime import datetime, timezone

    foreign = SimpleNamespace(action_type="llm_response", session_id="s1", agent_id="usage-agent",
                              timestamp=datetime.now(timezone.utc), content="x")
    await store._on_action(foreign)

    (rec,) = _history_records(store)
    assert (rec["model"], rec["tokens_in"], rec["tokens_out"], rec["latency_ms"]) == ("", 0, 0, 0)
