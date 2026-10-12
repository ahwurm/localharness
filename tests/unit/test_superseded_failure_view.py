"""A superseded failure leaves the model's view, never the record.

The agent loop stamps every tool message it pushes with a private outcome; the context manager
drops a failed-then-succeeded exchange from each request it builds (never a denial, never a
staged call, never a failure of a tool that can touch the host, never a message with other
content), and the session, the bus and ``history.jsonl`` keep both halves.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from localharness.agent.context import ContextManager
from localharness.agent.gate import DENIED_OBSERVATION_PREFIX
from localharness.agent.gate_types import GateOutcome
from localharness.agent.loop import (
    AgentLoop,
    Session,
    StuckDetector,
    _tool_message,
    compute_signature,
)
from localharness.agent.permissions import PermissionEvaluator
from localharness.config.models import AgentConfig, BudgetConfig, PermissionConfig
from localharness.core.events import Observation
from localharness.core.types import is_harness_message, provider_messages
from localharness.memory.sqlite import MemoryStore
from localharness.tools import Tool, ToolRegistry, ToolResult, ToolSchema
from tests.conftest import FakeLLMResponse, FakeToolCall, MockLLMClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FlakyProbe(Tool):
    """A safe-host tool that fails on its first call and succeeds from then on."""

    def __init__(self, name: str = "probe", host: str = "safe") -> None:
        super().__init__()
        self._name = name
        self._host = host
        self.calls = 0

    def info(self) -> ToolSchema:
        return ToolSchema(
            name=self._name,
            description="Probe.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": [],
            },
            ingest="none", host=self._host,
        )

    async def _execute(self, **kwargs: Any) -> ToolResult:
        self.calls += 1
        if self.calls == 1:
            return self.err("no such file")
        return self.ok("contents")


class _RecordingLLM(MockLLMClient):
    """MockLLMClient that keeps every message list it was asked to complete."""

    def __init__(self, responses: list[FakeLLMResponse]) -> None:
        super().__init__(responses)
        self.seen: list[list[dict[str, Any]]] = []

    async def stream_complete(self, messages=None, tools=None, on_token=None, **kw):
        self.seen.append([dict(m) for m in (messages or [])])
        return await super().stream_complete(messages=messages, tools=tools, on_token=on_token, **kw)


def _loop(llm, bus, registry, *, ctx: ContextManager | None = None, budget: BudgetConfig | None = None,
          gate=None) -> AgentLoop:
    permissions = PermissionConfig(mode="unattended")
    if budget is not None:
        permissions = PermissionConfig(mode="unattended", budget=budget)
    return AgentLoop(
        config=AgentConfig(name="test-agent", role="Test agent.", permissions=permissions),
        llm=llm,
        bus=bus,
        context_manager=ctx or ContextManager(),
        tool_registry=registry,
        permission_evaluator=PermissionEvaluator(),
        gate=gate,
    )


async def _registry(*tools: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        await registry.register(tool, scope="global")
    return registry


def _tool_messages(loop: AgentLoop) -> list[dict[str, Any]]:
    conversation, _ = loop.resume_state()
    return [m for m in conversation if m.get("role") == "tool"]


def _outcome(call_name: str, args: dict, *, ok: bool, denied: bool = False, staged: bool = False) -> dict:
    return {"outcome": {"call": compute_signature(call_name, args), "ok": ok,
                        "denied": denied, "staged": staged}}


# ---------------------------------------------------------------------------
# Task 1: the stamp
# ---------------------------------------------------------------------------

def test_module_level_signature_matches_the_stuck_detector_method():
    sig = compute_signature("read", {"path": "a"})
    assert sig == StuckDetector().compute_signature("read", {"path": "a"})
    assert len(sig) == 16 and all(c in "0123456789abcdef" for c in sig)
    assert compute_signature("t", {"a": 1, "b": 2}) == compute_signature("t", {"b": 2, "a": 1})


@pytest.mark.asyncio
async def test_a_failed_then_successful_dispatch_is_stamped_with_the_call_signature(bus):
    args = {"path": "a"}
    llm = _RecordingLLM([
        FakeLLMResponse(content=None, tool_calls=[FakeToolCall(id="tc-1", name="probe", arguments=args)]),
        FakeLLMResponse(content=None, tool_calls=[FakeToolCall(id="tc-2", name="probe", arguments=args)]),
        FakeLLMResponse(content="Done."),
    ])
    loop = _loop(llm, bus, await _registry(_FlakyProbe()))
    await loop.run_turn("task")

    first, second = _tool_messages(loop)
    sig = compute_signature("probe", args)
    assert first["_lh"] == {"outcome": {"call": sig, "ok": False, "denied": False, "staged": False}}
    assert second["_lh"] == {"outcome": {"call": sig, "ok": True, "denied": False, "staged": False}}
    for message in (first, second):
        assert "origin" not in message["_lh"]
        assert "_lh" not in provider_messages([message])[0]
        assert is_harness_message(message) is False


@pytest.mark.asyncio
async def test_a_gate_denial_is_stamped_denied(bus):
    class _Deny:
        mode = "guarded"
        asker = None

        async def check(self, *a, **kw):
            return GateOutcome(allowed=False, reason="not today")

    llm = _RecordingLLM([
        FakeLLMResponse(content=None, tool_calls=[FakeToolCall(id="tc-1", name="probe", arguments={"path": "a"})]),
        FakeLLMResponse(content="Done."),
    ])
    loop = _loop(llm, bus, await _registry(_FlakyProbe()), gate=_Deny())
    await loop.run_turn("task")

    (msg,) = _tool_messages(loop)
    assert msg["content"].startswith(DENIED_OBSERVATION_PREFIX)
    assert msg["_lh"]["outcome"]["ok"] is False
    assert msg["_lh"]["outcome"]["denied"] is True
    assert msg["_lh"]["outcome"]["staged"] is False


@pytest.mark.asyncio
async def test_a_staged_refusal_is_stamped_denied_and_staged(bus):
    class _Stage:
        mode = "auto"
        asker = None

        async def check(self, *a, **kw):
            return GateOutcome(allowed=False, reason="needs a human (pending #1)", pending=object())  # type: ignore[arg-type]

    llm = _RecordingLLM([
        FakeLLMResponse(content=None, tool_calls=[FakeToolCall(id="tc-1", name="probe", arguments={})]),
        FakeLLMResponse(content="Done."),
    ])
    loop = _loop(llm, bus, await _registry(_FlakyProbe()), gate=_Stage())
    await loop.run_turn("task")

    (msg,) = _tool_messages(loop)
    assert msg["_lh"]["outcome"]["denied"] is True
    assert msg["_lh"]["outcome"]["staged"] is True


def test_the_helper_stamps_staged_and_leaves_an_unstampable_call_unstamped():
    call = SimpleNamespace(id="tc", name="probe", arguments={"path": "a"})
    staged = _tool_message(call, "x", ok=False, denied=True, staged=True)
    assert staged["_lh"]["outcome"] == {
        "call": compute_signature("probe", {"path": "a"}), "ok": False, "denied": True, "staged": True,
    }
    assert staged["role"] == "tool" and staged["tool_call_id"] == "tc" and staged["content"] == "x"

    class _Unhashable:
        pass

    broken = SimpleNamespace(id="tc", name="probe", arguments={"x": _Unhashable()})
    message = _tool_message(broken, "x", ok=True)
    assert "_lh" not in message and message["content"] == "x"


@pytest.mark.asyncio
async def test_truncated_call_feedback_is_stamped_failed(bus):
    llm = _RecordingLLM([
        FakeLLMResponse(content=None, finish_reason="length",
                        tool_calls=[FakeToolCall(id="tc-1", name="probe", arguments={"path": "a"})]),
        FakeLLMResponse(content="Done."),
    ])
    loop = _loop(llm, bus, await _registry(_FlakyProbe()))
    await loop.run_turn("task")

    (msg,) = _tool_messages(loop)
    assert msg["_lh"]["outcome"] == {
        "call": compute_signature("probe", {"path": "a"}), "ok": False, "denied": False, "staged": False,
    }


@pytest.mark.asyncio
async def test_tool_call_budget_refusal_is_stamped_failed_not_denied(bus):
    llm = _RecordingLLM([
        FakeLLMResponse(content=None, tool_calls=[FakeToolCall(id="tc-1", name="probe", arguments={"path": "a"})]),
        FakeLLMResponse(content="Done."),
    ])
    loop = _loop(llm, bus, await _registry(_FlakyProbe()), budget=BudgetConfig(max_tool_calls=0))
    await loop.run_turn("task")

    (msg,) = _tool_messages(loop)
    assert msg["content"].startswith("Tool call not executed")
    assert msg["_lh"]["outcome"]["ok"] is False
    assert msg["_lh"]["outcome"]["denied"] is False
    assert msg["_lh"]["outcome"]["staged"] is False


@pytest.mark.asyncio
async def test_the_budget_note_lands_in_content_and_the_stamp_survives(bus):
    llm = _RecordingLLM([
        FakeLLMResponse(content=None, tool_calls=[FakeToolCall(id="tc-1", name="probe", arguments={"path": "a"})]),
        FakeLLMResponse(content="Done."),
    ])
    loop = _loop(llm, bus, await _registry(_FlakyProbe()), budget=BudgetConfig(max_actions=50))
    await loop.run_turn("task")

    (msg,) = _tool_messages(loop)
    assert msg["_lh"]["outcome"]["call"] == compute_signature("probe", {"path": "a"})
    assert msg["_lh"]["outcome"]["ok"] is False  # the probe's first call failed


@pytest.mark.asyncio
async def test_step_path_stamps_denial_result_and_truncation(bus):
    # result
    llm = MockLLMClient([
        FakeLLMResponse(content=None, tool_calls=[FakeToolCall(id="tc-1", name="probe", arguments={"path": "a"})]),
    ])
    loop = _loop(llm, bus, await _registry(_FlakyProbe()))
    session = Session(agent_id="test-agent", session_id="s", messages=[])
    session.push({"role": "user", "content": "t"})
    await loop.step(session)
    msg = [m for m in session.messages if m["role"] == "tool"][-1]
    assert msg["_lh"]["outcome"] == {
        "call": compute_signature("probe", {"path": "a"}), "ok": False, "denied": False, "staged": False,
    }

    # denial
    class _Deny:
        mode = "guarded"
        asker = None

        async def check(self, *a, **kw):
            return GateOutcome(allowed=False, reason="no")

    loop = _loop(MockLLMClient([
        FakeLLMResponse(content=None, tool_calls=[FakeToolCall(id="tc-2", name="probe", arguments={})]),
    ]), bus, await _registry(_FlakyProbe()), gate=_Deny())
    session = Session(agent_id="test-agent", session_id="s", messages=[])
    session.push({"role": "user", "content": "t"})
    await loop.step(session)
    msg = [m for m in session.messages if m["role"] == "tool"][-1]
    assert msg["_lh"]["outcome"]["denied"] is True and msg["_lh"]["outcome"]["ok"] is False

    # truncation
    loop = _loop(MockLLMClient([
        FakeLLMResponse(content=None, finish_reason="length",
                        tool_calls=[FakeToolCall(id="tc-3", name="probe", arguments={})]),
    ]), bus, await _registry(_FlakyProbe()))
    session = Session(agent_id="test-agent", session_id="s", messages=[])
    session.push({"role": "user", "content": "t"})
    await loop.step(session)
    msg = [m for m in session.messages if m["role"] == "tool"][-1]
    assert msg["_lh"]["outcome"]["ok"] is False and msg["_lh"]["outcome"]["denied"] is False
