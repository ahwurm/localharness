"""End-to-end integration tests for the agent loop with mock LLM.

Tests: natural completion, validation error recovery, stuck escalation, budget enforcement.
"""
from __future__ import annotations

import pytest

from localharness.agent.context import ContextManager
from localharness.agent.loop import AgentLoop
from localharness.agent.permissions import PermissionEvaluator
from localharness.config.models import AgentConfig, BudgetConfig, PermissionConfig, ToolConfig
from localharness.tools.builtin.web_tool import WEB_INGEST_TOOLS
from localharness.core.bus import EventBus
from localharness.core.events import Escalation, TurnFailed
from localharness.tools.builtin import register_builtin_tools
from localharness.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# MockLLMClient + helpers
# ---------------------------------------------------------------------------

class _FakeResponse:
    """Mimics the response object from LLMClient.stream_complete()."""
    def __init__(self, content: str | None, tool_calls: list | None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeToolCall:
    """Mimics a native tool call object (has .function attribute for _extract_tool_calls)."""
    def __init__(self, name: str, arguments: dict, id: str) -> None:
        self._name = name
        self._arguments = arguments
        self.id = id

    @property
    def function(self):
        import json

        class _Fn:
            pass

        fn = _Fn()
        fn.name = self._name
        fn.arguments = json.dumps(self._arguments)
        return fn


class MockLLMClient:
    """Returns scripted responses in order. Once exhausted, returns content="Done."."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = responses
        self._index = 0

        class _Config:
            tool_call_mode = "native"

        self.config = _Config()

    async def stream_complete(self, messages=None, tools=None, on_token=None):
        """Returns (message, usage) tuple to match production shape post-10-01."""
        if self._index >= len(self._responses):
            return _FakeResponse(content="Done.", tool_calls=None), None
        resp = self._responses[self._index]
        self._index += 1
        raw_tool_calls = resp.get("tool_calls")
        tool_call_objs = None
        if raw_tool_calls:
            tool_call_objs = [
                _FakeToolCall(
                    name=tc["name"],
                    arguments=tc["arguments"],
                    id=tc["id"],
                )
                for tc in raw_tool_calls
            ]
        return _FakeResponse(content=resp.get("content"), tool_calls=tool_call_objs), None


def _make_config(max_actions: int = 100) -> AgentConfig:
    return AgentConfig(
        name="test-agent",
        role="Test integration agent.",
        # P-A capability floor: deny web ingestion so the host-tool agent resolves clean (mirrors the
        # real root topology — host tools, web stripped). These tests exercise glob/read, not web.
        tools=ToolConfig(deny=sorted(WEB_INGEST_TOOLS)),
        permissions=PermissionConfig(
            deny_patterns=[],
            budget=BudgetConfig(max_actions=max_actions, max_duration_minutes=30.0),
        ),
    )


async def _make_loop(
    responses: list[dict],
    config: AgentConfig,
    registry: ToolRegistry,
    bus: EventBus | None = None,
) -> AgentLoop:
    bus = bus if bus is not None else EventBus()
    llm = MockLLMClient(responses)
    ctx = ContextManager()
    perm = PermissionEvaluator()
    return AgentLoop(
        config=config,
        llm=llm,
        bus=bus,
        context_manager=ctx,
        tool_registry=registry,
        permission_evaluator=perm,
        kill_file_path=None,
    )


# ---------------------------------------------------------------------------
# Test 1: Natural completion with tool call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_natural_completion_with_tool_call(tmp_path):
    """Full run: mock LLM requests glob tool, receives result, then completes."""
    registry = ToolRegistry()
    await register_builtin_tools(registry)
    config = _make_config()

    responses = [
        {
            "content": None,
            "tool_calls": [{"name": "glob", "arguments": {"pattern": "*.py", "path": str(tmp_path)}, "id": "tc1"}],
        },
        {
            "content": "Found the Python files you asked about.",
            "tool_calls": None,
        },
    ]

    loop = await _make_loop(responses, config, registry)
    summary = await loop.run_turn("Find all Python files in the current directory")

    assert summary, "Summary should be non-empty"
    assert "Found the Python files" in summary


# ---------------------------------------------------------------------------
# Test 2: Validation error recovery
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_validation_error_recovery(tmp_path):
    """First tool call has missing required arg (gets validation error); second succeeds."""
    # Create a test file to read
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")

    registry = ToolRegistry()
    await register_builtin_tools(registry)
    config = _make_config()

    responses = [
        # Response 1: missing required 'path' arg for 'read' — validation error
        {
            "content": None,
            "tool_calls": [{"name": "read", "arguments": {}, "id": "tc1"}],
        },
        # Response 2: retry with valid args
        {
            "content": None,
            "tool_calls": [{"name": "read", "arguments": {"path": str(test_file)}, "id": "tc2"}],
        },
        # Response 3: natural completion
        {
            "content": "File contents retrieved successfully.",
            "tool_calls": None,
        },
    ]

    loop = await _make_loop(responses, config, registry)
    summary = await loop.run_turn("Read the test file")

    assert summary, "Summary should be non-empty"
    assert "File contents retrieved" in summary


# ---------------------------------------------------------------------------
# Test 3: Stuck escalation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stuck_escalation(tmp_path):
    """Mock LLM repeats same tool call 3+ times → stuck detector escalates.

    The stuck notice is the Escalation event's, delivered once (D-04/D-40/D-67): the turn's reply is
    the wrap-up summary, else the last assistant content, else empty — never stuck text. The turn
    still ends as TurnFailed(stuck_detected) whose detail is the reply, and stops well before the
    scripted responses run out.
    """
    registry = ToolRegistry()
    await register_builtin_tools(registry)
    config = _make_config(max_actions=50)

    # Same glob call repeated indefinitely
    responses = [
        {
            "content": None,
            "tool_calls": [{"name": "glob", "arguments": {"pattern": "*.py", "path": str(tmp_path)}, "id": f"tc{i}"}],
        }
        for i in range(10)
    ]

    bus = EventBus()
    loop = await _make_loop(responses, config, registry, bus=bus)
    summary = await loop.run_turn("Find Python files over and over")

    # Exactly one Escalation carries the stuck notice
    escalations = bus.history(event_types=[Escalation])
    assert len(escalations) == 1
    assert escalations[0].reason == "stuck_detected"
    assert "repeated" in escalations[0].detail.lower() or "stuck" in escalations[0].detail.lower()

    # The turn terminated as stuck, and its failure detail is exactly the returned reply
    failed = [e for e in bus.history(event_types=[TurnFailed]) if e.reason == "stuck_detected"]
    assert len(failed) == 1
    assert failed[0].detail == summary

    # The reply carries no stuck notice
    assert "[Agent stuck" not in summary
    assert "stuck" not in summary.lower() and "escalat" not in summary.lower()

    # The loop stopped before exhausting the scripted responses
    assert loop._llm._index < len(responses)


# ---------------------------------------------------------------------------
# Test 4: Budget enforcement
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_budget_enforcement(tmp_path):
    """Agent with max_actions=2 stops after 2 tool calls even with more responses available."""
    registry = ToolRegistry()
    await register_builtin_tools(registry)
    config = _make_config(max_actions=2)

    # Provide many tool-calling responses — agent should stop at budget
    responses = [
        {
            "content": None,
            "tool_calls": [{"name": "glob", "arguments": {"pattern": "*.py", "path": str(tmp_path)}, "id": f"tc{i}"}],
        }
        for i in range(10)
    ]

    loop = await _make_loop(responses, config, registry)
    summary = await loop.run_turn("Keep calling tools")

    assert summary, "Summary should be non-empty"
    lower = summary.lower()
    assert "budget" in lower or "maximum" in lower or "limit" in lower or "2 tool" in lower
