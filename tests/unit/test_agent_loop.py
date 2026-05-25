"""Tests for Session, StuckDetector, BudgetTracker, KillWatcher, and AgentLoop."""
import time
import pytest
from pathlib import Path

from localharness.agent.loop import (
    Session,
    StuckDetector,
    StuckState,
    BudgetTracker,
    BudgetViolation,
    KillWatcher,
    StepResult,
)


# ---------------------------------------------------------------------------
# Session tests
# ---------------------------------------------------------------------------

def test_session_initializes():
    s = Session(agent_id="a", session_id="s", messages=[])
    assert s.agent_id == "a"
    assert s.session_id == "s"
    assert s.messages == []
    assert s.iteration == 0
    assert s.actions_taken == 0
    assert s.summary == ""
    assert s.terminated_reason is None


def test_session_push_appends():
    s = Session(agent_id="a", session_id="s", messages=[])
    msg = {"role": "user", "content": "hello"}
    s.push(msg)
    assert len(s.messages) == 1
    assert s.messages[0] is msg


def test_session_elapsed_seconds_positive():
    s = Session(agent_id="a", session_id="s", messages=[])
    time.sleep(0.01)
    assert s.elapsed_seconds() > 0
    assert s.elapsed_minutes() > 0


def test_session_messages_append_only_via_push():
    """Direct modification of messages list is not via push — push is the correct interface."""
    s = Session(agent_id="a", session_id="s", messages=[])
    s.push({"role": "user", "content": "a"})
    s.push({"role": "user", "content": "b"})
    assert len(s.messages) == 2


# ---------------------------------------------------------------------------
# StuckDetector tests
# ---------------------------------------------------------------------------

def test_stuck_compute_signature_returns_16_chars():
    sd = StuckDetector()
    sig = sd.compute_signature("bash", {"cmd": "ls"})
    assert len(sig) == 16
    assert sig.isalnum()


def test_stuck_compute_signature_order_independent():
    sd = StuckDetector()
    sig1 = sd.compute_signature("tool", {"a": 1, "b": 2})
    sig2 = sd.compute_signature("tool", {"b": 2, "a": 1})
    assert sig1 == sig2


def test_stuck_clear_when_different_calls():
    sd = StuckDetector(window_size=5, recovery_threshold=2, escalation_threshold=3)
    sd.record("bash", {"cmd": "ls"})
    sd.record("bash", {"cmd": "pwd"})
    sd.record("bash", {"cmd": "echo"})
    assert sd.check() == StuckState.CLEAR


def test_stuck_recovering_at_two_identical():
    sd = StuckDetector(window_size=5, recovery_threshold=2, escalation_threshold=3)
    sd.record("bash", {"cmd": "ls"})
    sd.record("bash", {"cmd": "ls"})
    assert sd.check() == StuckState.RECOVERING


def test_stuck_escalate_at_three_identical():
    sd = StuckDetector(window_size=5, recovery_threshold=2, escalation_threshold=3)
    sd.record("bash", {"cmd": "ls"})
    sd.record("bash", {"cmd": "ls"})
    sd.record("bash", {"cmd": "ls"})
    assert sd.check() == StuckState.ESCALATE


def test_stuck_recovery_message_nonempty():
    sd = StuckDetector()
    msg = sd.recovery_message("abcdef1234567890")
    assert isinstance(msg, str) and len(msg) > 0


def test_stuck_most_repeated_signature():
    sd = StuckDetector(window_size=5)
    sd.record("bash", {"cmd": "ls"})
    sd.record("bash", {"cmd": "ls"})
    sd.record("bash", {"cmd": "pwd"})
    sig_ls = sd.compute_signature("bash", {"cmd": "ls"})
    assert sd.most_repeated_signature() == sig_ls


def test_stuck_clear_when_window_too_small():
    sd = StuckDetector(window_size=5, recovery_threshold=2, escalation_threshold=3)
    sd.record("bash", {"cmd": "ls"})
    assert sd.check() == StuckState.CLEAR


# ---------------------------------------------------------------------------
# BudgetTracker tests
# ---------------------------------------------------------------------------

def test_budget_actions_exceeded():
    s = Session(agent_id="a", session_id="s", messages=[])
    s.actions_taken = 5
    tracker = BudgetTracker(max_actions=5, max_duration_minutes=30.0)
    v = tracker.check(s)
    assert isinstance(v, BudgetViolation)
    assert v.reason == "actions"


def test_budget_actions_not_exceeded():
    s = Session(agent_id="a", session_id="s", messages=[])
    s.actions_taken = 4
    tracker = BudgetTracker(max_actions=5, max_duration_minutes=30.0)
    assert tracker.check(s) is None


def test_budget_unlimited_actions():
    """max_actions=0 means unlimited — never trips."""
    s = Session(agent_id="a", session_id="s", messages=[])
    s.actions_taken = 999
    tracker = BudgetTracker(max_actions=0, max_duration_minutes=30.0)
    # Should not trip on actions, only check time (30 min, not elapsed)
    result = tracker.check(s)
    assert result is None or result.reason == "time"


def test_budget_time_exceeded():
    # Use very short duration to trigger time violation
    s = Session(agent_id="a", session_id="s", messages=[])
    time.sleep(0.05)  # short sleep; duration_minutes = 0.0001 (very small)
    tracker = BudgetTracker(max_actions=100, max_duration_minutes=0.0001)
    v = tracker.check(s)
    assert isinstance(v, BudgetViolation)
    assert v.reason == "time"


# ---------------------------------------------------------------------------
# KillWatcher tests
# ---------------------------------------------------------------------------

def test_kill_watcher_false_when_no_file(tmp_path):
    kw = KillWatcher(kill_file_path=tmp_path / "KILL")
    assert kw.is_killed() is False


def test_kill_watcher_true_when_file_exists(tmp_path):
    kill_path = tmp_path / "KILL"
    kill_path.touch()
    kw = KillWatcher(kill_file_path=kill_path)
    assert kw.is_killed() is True


# ---------------------------------------------------------------------------
# Task 2: AgentLoop tests
# ---------------------------------------------------------------------------

import pytest
from localharness.agent.loop import AgentLoop
from localharness.agent.context import ContextManager
from localharness.agent.permissions import PermissionEvaluator
from localharness.core.events import TurnStarted, TurnCompleted, TurnFailed, Action, Observation


def _make_agent_loop(mock_llm_client_factory, responses, bus, config=None, tool_registry=None):
    """Helper to construct an AgentLoop with mock dependencies."""
    from localharness.config.models import AgentConfig
    cfg = config or AgentConfig(name="test-agent", role="Test agent.")
    llm = mock_llm_client_factory(responses)
    ctx = ContextManager()
    perm = PermissionEvaluator()
    return AgentLoop(
        config=cfg,
        llm=llm,
        bus=bus,
        context_manager=ctx,
        tool_registry=tool_registry,
        permission_evaluator=perm,
    )


@pytest.mark.asyncio
async def test_run_turn_publishes_turn_started(mock_llm_client, bus):
    Response = mock_llm_client.Response
    loop = _make_agent_loop(mock_llm_client, [Response(content="Done.")], bus)
    await loop.run_turn("Do something")
    events = bus.history(event_types=[TurnStarted])
    assert len(events) == 1
    assert events[0].task_summary == "Do something"


@pytest.mark.asyncio
async def test_run_turn_completes_naturally_no_tool_calls(mock_llm_client, bus):
    Response = mock_llm_client.Response
    loop = _make_agent_loop(mock_llm_client, [Response(content="All done!")], bus)
    summary = await loop.run_turn("task")
    assert "All done!" in summary


@pytest.mark.asyncio
async def test_run_turn_publishes_turn_completed(mock_llm_client, bus):
    Response = mock_llm_client.Response
    loop = _make_agent_loop(mock_llm_client, [Response(content="Done.")], bus)
    await loop.run_turn("task")
    events = bus.history(event_types=[TurnCompleted])
    assert len(events) == 1


@pytest.mark.asyncio
async def test_run_turn_executes_tool_calls(mock_llm_client, bus):
    """AgentLoop dispatches tool calls and pushes tool results to session."""
    Response = mock_llm_client.Response
    ToolCallObj = mock_llm_client.ToolCall

    # Mock tool registry that returns "ok" for any dispatch
    class FakeRegistry:
        def get_tools_for_agent(self, config):
            return []

        async def dispatch(self, tool_call):
            class Result:
                output = "tool-result"
                is_error = False
            return Result()

    tc = ToolCallObj(id="tc-1", name="bash", arguments={"cmd": "ls"})
    responses = [
        Response(content=None, tool_calls=[tc]),
        Response(content="Finished after tool."),
    ]
    loop = _make_agent_loop(mock_llm_client, responses, bus, tool_registry=FakeRegistry())
    summary = await loop.run_turn("task")
    assert "Finished after tool." in summary


@pytest.mark.asyncio
async def test_run_turn_stops_on_budget_exceeded(mock_llm_client, bus):
    from localharness.config.models import AgentConfig, PermissionConfig, BudgetConfig
    Response = mock_llm_client.Response
    ToolCallObj = mock_llm_client.ToolCall

    class FakeRegistry:
        def get_tools_for_agent(self, config):
            return []
        async def dispatch(self, tc):
            class R:
                output = "x"
                is_error = False
            return R()

    cfg = AgentConfig(
        name="budget-agent",
        role="Test.",
        permissions=PermissionConfig(
            deny_patterns=[],
            budget=BudgetConfig(max_actions=1, max_duration_minutes=30.0),
        ),
    )
    tc = ToolCallObj(id="tc-1", name="bash", arguments={"cmd": "ls"})
    responses = [Response(content=None, tool_calls=[tc])] * 10
    loop = _make_agent_loop(mock_llm_client, responses, bus, config=cfg, tool_registry=FakeRegistry())
    summary = await loop.run_turn("task")
    # Budget exceeded: should publish TurnFailed
    failed = bus.history(event_types=[TurnFailed])
    assert len(failed) >= 1
    assert any(e.reason == "budget_exceeded" for e in failed)
    assert "Budget limit" in summary


@pytest.mark.asyncio
async def test_run_turn_stops_on_kill_file(mock_llm_client, bus, tmp_path):
    Response = mock_llm_client.Response
    kill_path = tmp_path / "KILL"
    kill_path.touch()

    loop = _make_agent_loop(mock_llm_client, [Response(content="unreachable")], bus)
    loop._kill = KillWatcher(kill_file_path=kill_path)
    summary = await loop.run_turn("task")
    failed = bus.history(event_types=[TurnFailed])
    assert any(e.reason == "kill_file" for e in failed)
    assert "kill signal" in summary


@pytest.mark.asyncio
async def test_run_turn_stops_on_stuck_escalation(mock_llm_client, bus):
    Response = mock_llm_client.Response
    ToolCallObj = mock_llm_client.ToolCall

    class FakeRegistry:
        def get_tools_for_agent(self, config):
            return []
        async def dispatch(self, tc):
            class R:
                output = "same"
                is_error = False
            return R()

    # Same tool call repeated 4 times will trigger ESCALATE at iteration 3
    tc = ToolCallObj(id="tc-1", name="bash", arguments={"cmd": "ls"})
    responses = [Response(content=None, tool_calls=[tc])] * 10
    loop = _make_agent_loop(mock_llm_client, responses, bus, tool_registry=FakeRegistry())
    summary = await loop.run_turn("task")
    failed = bus.history(event_types=[TurnFailed])
    assert any(e.reason == "stuck_detected" for e in failed)
    assert "stuck" in summary.lower() or "escalat" in summary.lower()


@pytest.mark.asyncio
async def test_run_turn_injects_recovery_on_recovering(mock_llm_client, bus):
    """At 2 identical calls, recovery message is set; agent continues."""
    Response = mock_llm_client.Response
    ToolCallObj = mock_llm_client.ToolCall

    class FakeRegistry:
        def get_tools_for_agent(self, config):
            return []
        async def dispatch(self, tc):
            class R:
                output = "same"
                is_error = False
            return R()

    tc_same = ToolCallObj(id="tc-1", name="bash", arguments={"cmd": "ls"})
    tc_diff = ToolCallObj(id="tc-2", name="write", arguments={"path": "/tmp/x"})
    # 2 same → triggers recovery; then different → clears; then finish
    responses = [
        Response(content=None, tool_calls=[tc_same]),
        Response(content=None, tool_calls=[tc_same]),
        Response(content=None, tool_calls=[tc_diff]),
        Response(content="Recovery worked."),
    ]
    loop = _make_agent_loop(mock_llm_client, responses, bus, tool_registry=FakeRegistry())
    summary = await loop.run_turn("task")
    # Agent should complete normally (not stuck)
    completed = bus.history(event_types=[TurnCompleted])
    assert len(completed) == 1


@pytest.mark.asyncio
async def test_run_turn_handles_provider_connection_error_with_retry(mock_llm_client, bus):
    """ProviderConnectionError triggers one retry; if second also fails, returns summary."""
    from localharness.provider.client import ProviderConnectionError

    class FailTwiceLLM:
        class config:
            tool_call_mode = "native"

        _count = 0

        async def stream_complete(self, messages=None, tools=None, on_token=None):
            self._count += 1
            raise ProviderConnectionError("connection refused")

    loop = _make_agent_loop(mock_llm_client, [], bus)
    loop._llm = FailTwiceLLM()
    summary = await loop.run_turn("task")
    # Should not raise — returns error summary
    assert isinstance(summary, str) and len(summary) > 0


@pytest.mark.asyncio
async def test_run_turn_never_raises(mock_llm_client, bus):
    """run_turn must return a string even if LLM raises completely unexpected error."""
    class BrokenLLM:
        class config:
            tool_call_mode = "native"

        async def stream_complete(self, **kwargs):
            raise RuntimeError("catastrophic failure")

    loop = _make_agent_loop(mock_llm_client, [], bus)
    loop._llm = BrokenLLM()
    summary = await loop.run_turn("task")
    assert isinstance(summary, str) and len(summary) > 0


@pytest.mark.asyncio
async def test_step_returns_correct_action_type(mock_llm_client, bus):
    Response = mock_llm_client.Response
    loop = _make_agent_loop(mock_llm_client, [Response(content="step done")], bus)
    session = Session(agent_id="test-agent", session_id="sess", messages=[
        {"role": "user", "content": "hi"},
    ])
    result = await loop.step(session)
    assert result.action == "complete"
    assert "step done" in result.llm_response_preview


@pytest.mark.asyncio
async def test_request_messages_go_through_context_manager(mock_llm_client, bus):
    """Verify build_messages is called (orphaned tool result removed before LLM call)."""
    Response = mock_llm_client.Response

    build_called = []

    class TrackingContextManager(ContextManager):
        def build_messages(self, messages, tool_schemas=None):
            build_called.append(True)
            return super().build_messages(messages, tool_schemas)

    loop = _make_agent_loop(mock_llm_client, [Response(content="ok")], bus)
    loop._ctx = TrackingContextManager()
    await loop.run_turn("task")
    assert len(build_called) >= 1
