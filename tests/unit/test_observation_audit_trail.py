"""#133 — the tool-result audit trail carries the FULL result, never a display slice.

The Observation event is the audit spine: bus-events.jsonl, history.jsonl and the terminal's
result badge all read it. It used to publish `result_content[:200]` with truncated=False, so a
692-char result was recorded as a complete 200-char one and history.jsonl's original_length
re-measured the slice. These tests pin the honest contract: the event carries exactly what the
model saw, and truncation metadata comes from the registry cap that actually bounded it.
"""
import json
from pathlib import Path
from typing import Any

import pytest

from localharness.agent.context import ContextManager
from localharness.agent.loop import AgentLoop
from localharness.agent.permissions import PermissionEvaluator
from localharness.config.models import AgentConfig
from localharness.core.events import Observation
from localharness.memory.sqlite import MemoryStore
from localharness.tools import Tool, ToolRegistry, ToolResult, ToolSchema
from tests.conftest import FakeLLMResponse, FakeToolCall, MockLLMClient


def _loop(llm, bus, registry) -> AgentLoop:
    return AgentLoop(
        config=AgentConfig(name="test-agent", role="Test agent."),
        llm=llm,
        bus=bus,
        context_manager=ContextManager(),
        tool_registry=registry,
        permission_evaluator=PermissionEvaluator(),
    )


def _one_tool_call_then_done(payload_calls: list[FakeToolCall]) -> list[FakeLLMResponse]:
    return [FakeLLMResponse(content=None, tool_calls=payload_calls), FakeLLMResponse(content="Done.")]


class _RecordingLLM(MockLLMClient):
    """MockLLMClient that keeps every message list it was asked to complete."""

    def __init__(self, responses: list[FakeLLMResponse]) -> None:
        super().__init__(responses)
        self.seen: list[list[dict[str, Any]]] = []

    async def stream_complete(self, messages=None, tools=None, on_token=None):
        self.seen.append([dict(m) for m in (messages or [])])
        return await super().stream_complete(messages=messages, tools=tools, on_token=on_token)


def _observations(bus) -> list[Observation]:
    return [e for e in bus.history(event_types=[Observation]) if e.observation_type == "tool_result"]


# ---------------------------------------------------------------------------
# The event: full output + honest metadata
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_observation_carries_full_uncapped_output(bus):
    """A 1,000-char result reaches the audit trail whole, flagged as NOT truncated."""
    body = "A" * 1000

    class FakeRegistry:
        def get_tools_for_agent(self, agent_id, division_id, tool_config):
            return {}

        async def dispatch(self, name, arguments, agent_id, division_id, tool_config):
            return ToolResult(output=body, success=True)

    llm = _RecordingLLM(_one_tool_call_then_done(
        [FakeToolCall(id="tc-1", name="bash", arguments={"cmd": "ls"})]
    ))
    await _loop(llm, bus, FakeRegistry()).run_turn("task")

    obs = _observations(bus)
    assert len(obs) == 1
    assert obs[0].output == body
    assert len(obs[0].output) == 1000
    assert obs[0].truncated is False
    assert obs[0].original_length == 1000


@pytest.mark.asyncio
async def test_observation_records_real_upstream_cap_truncation(bus):
    """Over the registry's result-size cap: the event stores the capped text and reports the
    TRUE pre-cap length — the cap that actually bounded the result, not an invented one."""
    class _BigTool(Tool):
        def info(self) -> ToolSchema:
            return ToolSchema(
                name="big",
                description="Returns huge output.",
                parameters={"type": "object", "properties": {}, "required": []},
            )

        async def _execute(self, **kwargs: Any) -> ToolResult:
            return self.ok("x" * 2000)

    registry = ToolRegistry(result_size_cap_chars=500)
    await registry.register(_BigTool(), scope="global")

    llm = _RecordingLLM(_one_tool_call_then_done(
        [FakeToolCall(id="tc-big", name="big", arguments={})]
    ))
    await _loop(llm, bus, registry).run_turn("task")

    obs = _observations(bus)
    assert len(obs) == 1
    assert len(obs[0].output) == 500
    assert obs[0].truncated is True
    assert obs[0].original_length == 2000


@pytest.mark.asyncio
async def test_model_still_sees_the_full_result(bus):
    """Guard on the half that was already right: session.push feeds the model the whole
    result. The audit fix must not change what the model reads."""
    body = "B" * 1000

    class FakeRegistry:
        def get_tools_for_agent(self, agent_id, division_id, tool_config):
            return {}

        async def dispatch(self, name, arguments, agent_id, division_id, tool_config):
            return ToolResult(output=body, success=True)

    llm = _RecordingLLM(_one_tool_call_then_done(
        [FakeToolCall(id="tc-1", name="bash", arguments={"cmd": "ls"})]
    ))
    await _loop(llm, bus, FakeRegistry()).run_turn("task")

    tool_msgs = [m for msgs in llm.seen for m in msgs if m.get("role") == "tool"]
    assert tool_msgs, "the second completion must carry the tool result"
    assert body in tool_msgs[-1]["content"]


@pytest.mark.asyncio
async def test_error_result_reports_its_own_length(bus):
    """A failed dispatch stores the forwarded '[tool error] …' text: nothing was dropped,
    so truncated stays False and original_length measures what is stored."""
    class FakeRegistry:
        def get_tools_for_agent(self, agent_id, division_id, tool_config):
            return {}

        async def dispatch(self, name, arguments, agent_id, division_id, tool_config):
            return ToolResult(output="", success=False, error="E" * 400, error_type="execution_error")

    llm = _RecordingLLM(_one_tool_call_then_done(
        [FakeToolCall(id="tc-err", name="bash", arguments={"cmd": "boom"})]
    ))
    await _loop(llm, bus, FakeRegistry()).run_turn("task")

    obs = _observations(bus)
    assert len(obs) == 1
    assert obs[0].output.startswith("[tool error] ")
    assert obs[0].truncated is False
    assert obs[0].original_length == len(obs[0].output)


# ---------------------------------------------------------------------------
# history.jsonl: lengths measured against reality, not the stored slice
# ---------------------------------------------------------------------------

@pytest.fixture
async def store(tmp_path: Path):
    s = MemoryStore(agent_id="obs-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


def _history_records(store: MemoryStore) -> list[dict[str, Any]]:
    path = Path(store._history_path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _obs(**kw: Any) -> Observation:
    base = dict(agent_id="obs-agent", session_id="s1", observation_type="tool_result",
                tool_call_id="tc-1", tool_name="bash")
    base.update(kw)
    return Observation(**base)


@pytest.mark.asyncio
async def test_history_record_lengths_are_real_when_uncapped(store: MemoryStore):
    await store._on_observation(_obs(output="C" * 1000, original_length=1000))

    rec = _history_records(store)[-1]
    assert rec["type"] == "tool_result"
    assert len(rec["content"]) == 1000
    assert rec["truncated"] is False
    assert rec["original_length"] == 1000
    assert rec["stored_length"] == 1000


@pytest.mark.asyncio
async def test_history_record_keeps_true_original_length_when_capped(store: MemoryStore):
    await store._on_observation(_obs(output="x" * 500, truncated=True, original_length=2000))

    rec = _history_records(store)[-1]
    assert rec["truncated"] is True
    assert rec["original_length"] == 2000  # real pre-cap size, never re-measured from storage
    assert rec["stored_length"] == 500


@pytest.mark.asyncio
async def test_history_record_falls_back_to_stored_length(store: MemoryStore):
    """Events without original_length (older producers, replayed ledgers) still record a
    length rather than null — the added field is compatible, not required."""
    await store._on_observation(_obs(output="D" * 42))

    rec = _history_records(store)[-1]
    assert rec["original_length"] == 42
    assert rec["stored_length"] == 42
