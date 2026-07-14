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
from localharness.core.bus import EventBus


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
# Permission pattern matching tests
# ---------------------------------------------------------------------------

def test_permission_denies_relative_path():
    """Relative paths must match deny patterns with */ prefix (bug fix)."""
    from localharness.agent.permissions import PermissionEvaluator, PermissionResult
    from localharness.core.types import ToolCall

    class FakePerms:
        deny_patterns = ["write(*/agents/*.yaml)"]

    ev = PermissionEvaluator()
    tc = ToolCall(name="write", arguments={"path": "agents/brave-search.yaml"}, id="t1")
    result = ev.evaluate(tc, FakePerms())
    assert result.denied is True


def test_permission_denies_absolute_path():
    from localharness.agent.permissions import PermissionEvaluator
    from localharness.core.types import ToolCall

    class FakePerms:
        deny_patterns = ["write(*/agents/*.yaml)"]

    ev = PermissionEvaluator()
    tc = ToolCall(name="write", arguments={"path": "/home/user/agents/foo.yaml"}, id="t1")
    result = ev.evaluate(tc, FakePerms())
    assert result.denied is True


def test_permission_allows_non_matching():
    from localharness.agent.permissions import PermissionEvaluator
    from localharness.core.types import ToolCall

    class FakePerms:
        deny_patterns = ["write(*/agents/*.yaml)"]

    ev = PermissionEvaluator()
    tc = ToolCall(name="write", arguments={"path": "src/main.py"}, id="t1")
    result = ev.evaluate(tc, FakePerms())
    assert result.denied is False


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


def _make_agent_loop(mock_llm_client_factory, responses, bus, config=None, tool_registry=None,
                     session_id=None, memory_loader=None, context_manager=None):
    """Helper to construct an AgentLoop with mock dependencies."""
    from localharness.config.models import AgentConfig
    cfg = config or AgentConfig(name="test-agent", role="Test agent.")
    llm = mock_llm_client_factory(responses)
    ctx = context_manager or ContextManager()
    perm = PermissionEvaluator()
    # session_id is additive: most callers (bench/subagent) construct without it and
    # keep per-turn uuid semantics, so only pass it through when the test supplies one.
    extra = {"session_id": session_id} if session_id is not None else {}
    return AgentLoop(
        config=cfg,
        llm=llm,
        bus=bus,
        context_manager=ctx,
        tool_registry=tool_registry,
        permission_evaluator=perm,
        memory_loader=memory_loader,
        **extra,
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
async def test_sitting_session_id_stable_across_turns(mock_llm_client, bus):
    """SESS-01: a loop built with session_id keeps ONE sitting id across every turn, and
    current_session_id is valid BEFORE the first run_turn (kills the repl.py:94 off-by-one
    where turn 1's UserMessage carried a None session)."""
    Response = mock_llm_client.Response
    loop = _make_agent_loop(
        mock_llm_client, [Response(content="a"), Response(content="b")], bus,
        session_id="sit-1",
    )
    assert loop.current_session_id == "sit-1"  # valid at construction, pre-first-turn
    await loop.run_turn("first")
    await loop.run_turn("second")
    started = bus.history(event_types=[TurnStarted])
    assert [e.session_id for e in started] == ["sit-1", "sit-1"]
    assert loop.current_session_id == "sit-1"


@pytest.mark.asyncio
async def test_no_kwarg_keeps_per_turn_uuid(mock_llm_client, bus):
    """Legacy fallback intact: without session_id, each run_turn mints a fresh uuid so
    bench/subagent callers (which never pass the kwarg) keep per-run session semantics."""
    Response = mock_llm_client.Response
    loop = _make_agent_loop(
        mock_llm_client, [Response(content="a"), Response(content="b")], bus,
    )
    await loop.run_turn("first")
    sid1 = loop.current_session_id
    await loop.run_turn("second")
    sid2 = loop.current_session_id
    assert sid1 and sid2 and sid1 != sid2
    assert [e.session_id for e in bus.history(event_types=[TurnStarted])] == [sid1, sid2]


@pytest.mark.asyncio
async def test_run_turn_executes_tool_calls(mock_llm_client, bus):
    """AgentLoop dispatches tool calls and pushes tool results to session."""
    Response = mock_llm_client.Response
    ToolCallObj = mock_llm_client.ToolCall

    # Mock tool registry that returns "ok" for any dispatch
    class FakeRegistry:
        def get_tools_for_agent(self, agent_id, division_id, tool_config):
            return {}

        async def dispatch(self, name, arguments, agent_id, division_id, tool_config):
            from localharness.tools.base import ToolResult
            return ToolResult(output="tool-result", success=True)

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
        def get_tools_for_agent(self, agent_id, division_id, tool_config):
            return {}
        async def dispatch(self, name, arguments, agent_id, division_id, tool_config):
            from localharness.tools.base import ToolResult
            return ToolResult(output="x", success=True)

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
        def get_tools_for_agent(self, agent_id, division_id, tool_config):
            return {}
        async def dispatch(self, name, arguments, agent_id, division_id, tool_config):
            from localharness.tools.base import ToolResult
            return ToolResult(output="same", success=True)

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
        def get_tools_for_agent(self, agent_id, division_id, tool_config):
            return {}
        async def dispatch(self, name, arguments, agent_id, division_id, tool_config):
            from localharness.tools.base import ToolResult
            return ToolResult(output="same", success=True)

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


# ---------------------------------------------------------------------------
# Event publication tests (06-01: Heartbeat, Action(tool_call), TaskComplete)
# ---------------------------------------------------------------------------

from localharness.core.events import Heartbeat, TaskComplete


@pytest.mark.asyncio
async def test_execute_loop_publishes_heartbeat(mock_llm_client, bus):
    """AgentLoop._execute_loop publishes at least one Heartbeat per iteration."""
    Response = mock_llm_client.Response
    loop = _make_agent_loop(mock_llm_client, [Response(content="Done.")], bus)
    await loop.run_turn("task")
    heartbeats = bus.history(event_types=[Heartbeat])
    assert len(heartbeats) >= 1
    assert heartbeats[0].iteration >= 1
    assert heartbeats[0].agent_id == "test-agent"


@pytest.mark.asyncio
async def test_execute_loop_publishes_tool_call_action(mock_llm_client, bus):
    """AgentLoop publishes Action(action_type='tool_call') before each tool dispatch."""
    Response = mock_llm_client.Response
    ToolCallObj = mock_llm_client.ToolCall

    class FakeRegistry:
        def get_tools_for_agent(self, agent_id, division_id, tool_config):
            return {}
        async def dispatch(self, name, arguments, agent_id, division_id, tool_config):
            from localharness.tools.base import ToolResult
            return ToolResult(output="ok", success=True)

    tc = ToolCallObj(id="tc-1", name="glob_files", arguments={"pattern": "*.py"})
    responses = [
        Response(content=None, tool_calls=[tc]),
        Response(content="Finished."),
    ]
    loop = _make_agent_loop(mock_llm_client, responses, bus, tool_registry=FakeRegistry())
    await loop.run_turn("task")

    tool_call_actions = [
        e for e in bus.history(event_types=[Action])
        if e.action_type == "tool_call"
    ]
    assert len(tool_call_actions) >= 1
    assert tool_call_actions[0].tool_name == "glob_files"
    assert tool_call_actions[0].tool_call_id == "tc-1"
    assert tool_call_actions[0].tool_params == {"pattern": "*.py"}


@pytest.mark.asyncio
async def test_llm_response_action_flags_has_tool_calls(mock_llm_client, bus):
    """Each llm_response Action records whether THIS iteration also made tool calls —
    the terminal's discriminator between interstitial narration (True) and the final
    answer (False, rendered via TaskComplete — must never double-print as narration)."""
    Response = mock_llm_client.Response
    ToolCallObj = mock_llm_client.ToolCall

    class FakeRegistry:
        def get_tools_for_agent(self, agent_id, division_id, tool_config):
            return {}
        async def dispatch(self, name, arguments, agent_id, division_id, tool_config):
            from localharness.tools.base import ToolResult
            return ToolResult(output="ok", success=True)

    tc = ToolCallObj(id="tc-1", name="glob_files", arguments={"pattern": "*.py"})
    responses = [
        Response(content="Pulling the data…", tool_calls=[tc]),  # interstitial: narration
        Response(content="Here is the answer."),                 # final: no tool calls
    ]
    loop = _make_agent_loop(mock_llm_client, responses, bus, tool_registry=FakeRegistry())
    await loop.run_turn("task")

    llm_actions = [
        e for e in bus.history(event_types=[Action]) if e.action_type == "llm_response"
    ]
    assert len(llm_actions) == 2
    assert llm_actions[0].has_tool_calls is True   # tool-call iteration → narration
    assert llm_actions[0].content == "Pulling the data…"
    assert llm_actions[1].has_tool_calls is False  # final iteration → the answer, not narration


async def _capture_system_prompt(mock_llm_client, bus) -> str:
    """Run one trivial turn and return the built leading system message content."""
    Response = mock_llm_client.Response
    loop = _make_agent_loop(mock_llm_client, [Response(content="done")], bus)
    captured: list = []
    original = loop._llm.stream_complete

    async def capturing(messages=None, tools=None, on_token=None):
        captured.extend(messages or [])
        return await original(messages=messages, tools=tools, on_token=on_token)

    loop._llm.stream_complete = capturing
    await loop.run_turn("task")
    sys_msgs = [m for m in captured if m.get("role") == "system"]
    assert sys_msgs
    return sys_msgs[0]["content"]


@pytest.mark.asyncio
async def test_system_prompt_injects_working_directory(mock_llm_client, bus):
    """#75: the prompt states the cwd + a placement rule, next to the date line — so the
    model puts files under the launched project dir, not an invented path under $HOME."""
    from pathlib import Path
    sys_content = await _capture_system_prompt(mock_llm_client, bus)
    assert f"Working directory: {Path.cwd()}" in sys_content
    assert "under" in sys_content and "unless the user names another location" in sys_content.lower()


@pytest.mark.asyncio
async def test_system_prompt_has_narration_nudge(mock_llm_client, bus):
    """Belt-and-suspenders nudge: one sentence encouraging a short one-line stage
    announcement when starting a distinct phase of a multi-step task."""
    sys_content = await _capture_system_prompt(mock_llm_client, bus)
    low = sys_content.lower()
    assert "distinct phase" in low
    assert "multi-step task" in low


@pytest.mark.asyncio
async def test_execute_loop_publishes_task_complete(mock_llm_client, bus):
    """AgentLoop publishes TaskComplete(success=True) on natural completion."""
    Response = mock_llm_client.Response
    loop = _make_agent_loop(mock_llm_client, [Response(content="All done!")], bus)
    await loop.run_turn("task")
    completions = bus.history(event_types=[TaskComplete])
    assert len(completions) == 1
    assert completions[0].success is True
    assert completions[0].duration_seconds > 0
    assert completions[0].iterations >= 1
    assert "All done!" in completions[0].summary


@pytest.mark.asyncio
async def test_heartbeat_contains_correct_iteration_and_agent(mock_llm_client, bus):
    """Heartbeat event has correct session.iteration count and agent_id."""
    Response = mock_llm_client.Response
    ToolCallObj = mock_llm_client.ToolCall

    class FakeRegistry:
        def get_tools_for_agent(self, agent_id, division_id, tool_config):
            return {}
        async def dispatch(self, name, arguments, agent_id, division_id, tool_config):
            from localharness.tools.base import ToolResult
            return ToolResult(output="ok", success=True)

    tc = ToolCallObj(id="tc-1", name="bash", arguments={"cmd": "ls"})
    responses = [
        Response(content=None, tool_calls=[tc]),
        Response(content="Done."),
    ]
    loop = _make_agent_loop(mock_llm_client, responses, bus, tool_registry=FakeRegistry())
    await loop.run_turn("task")

    heartbeats = bus.history(event_types=[Heartbeat])
    # Two iterations: one tool call, one completion
    assert len(heartbeats) == 2
    assert heartbeats[0].iteration == 1
    assert heartbeats[1].iteration == 2
    assert all(h.agent_id == "test-agent" for h in heartbeats)


@pytest.mark.asyncio
async def test_tool_call_action_published_before_observation(mock_llm_client, bus):
    """Action(tool_call) is published BEFORE Observation(tool_result) in bus history."""
    Response = mock_llm_client.Response
    ToolCallObj = mock_llm_client.ToolCall

    class FakeRegistry:
        def get_tools_for_agent(self, agent_id, division_id, tool_config):
            return {}
        async def dispatch(self, name, arguments, agent_id, division_id, tool_config):
            from localharness.tools.base import ToolResult
            return ToolResult(output="result", success=True)

    tc = ToolCallObj(id="tc-1", name="bash", arguments={"cmd": "ls"})
    responses = [
        Response(content=None, tool_calls=[tc]),
        Response(content="Done."),
    ]
    loop = _make_agent_loop(mock_llm_client, responses, bus, tool_registry=FakeRegistry())
    await loop.run_turn("task")

    # Find the Action(tool_call) and the Observation for the same tool_call_id
    all_events = bus.history()
    action_idx = None
    observation_idx = None
    for i, e in enumerate(all_events):
        if isinstance(e, Action) and e.action_type == "tool_call" and e.tool_call_id == "tc-1":
            action_idx = i
        if isinstance(e, Observation) and e.tool_call_id == "tc-1":
            observation_idx = i
    assert action_idx is not None, "Action(tool_call) not found"
    assert observation_idx is not None, "Observation(tool_result) not found"
    assert action_idx < observation_idx, "Action must come before Observation"


# ---------------------------------------------------------------------------
# Task 2: Memory loader tiered prompt tests
# ---------------------------------------------------------------------------

from unittest.mock import AsyncMock as _AsyncMock
from dataclasses import dataclass as _dataclass


@_dataclass(frozen=True)
class _MockMemoryContext:
    agent_memory_md: str
    division_md: str
    guardrails_md: str
    fact_count: int
    token_estimate: int


def _make_memory_agent_loop(memory_loader=None):
    """Create an AgentLoop with mocked dependencies for memory testing."""
    from localharness.config.models import AgentConfig
    from tests.conftest import MockLLMClient, FakeLLMResponse

    cfg = AgentConfig(name="test-agent", role="You are a test assistant.")
    llm = MockLLMClient([FakeLLMResponse(content="Done.")])
    ctx = ContextManager()
    perm = PermissionEvaluator()
    mock_bus = EventBus()

    return AgentLoop(
        config=cfg,
        llm=llm,
        bus=mock_bus,
        context_manager=ctx,
        tool_registry=None,
        permission_evaluator=perm,
        memory_loader=memory_loader,
    ), llm


@pytest.mark.asyncio
async def test_memory_loader_tiered_prompt():
    """MEM-03: load_context() result builds tiered system prompt with all 3 sections."""
    memory = _AsyncMock()
    memory.load_context = _AsyncMock(return_value=_MockMemoryContext(
        agent_memory_md="my notes",
        division_md="div context",
        guardrails_md="safety rules",
        fact_count=5,
        token_estimate=100,
    ))

    loop, llm = _make_memory_agent_loop(memory_loader=memory)

    # Capture messages sent to LLM
    captured_messages = []
    original_stream = llm.stream_complete

    async def capturing_stream(messages=None, tools=None, on_token=None):
        captured_messages.extend(messages or [])
        return await original_stream(messages=messages, tools=tools, on_token=on_token)

    llm.stream_complete = capturing_stream
    await loop.run_turn("hello")

    memory.load_context.assert_awaited_once()
    sys_msgs = [m for m in captured_messages if m.get("role") == "system"]
    assert len(sys_msgs) >= 1
    sys_content = sys_msgs[0]["content"]
    assert "## Guardrails\nsafety rules" in sys_content
    assert "## Division Context\ndiv context" in sys_content
    assert "## Agent Memory\nmy notes" in sys_content
    # Verify order: guardrails before division before agent memory
    g_idx = sys_content.index("## Guardrails")
    d_idx = sys_content.index("## Division Context")
    a_idx = sys_content.index("## Agent Memory")
    assert g_idx < d_idx < a_idx


@pytest.mark.asyncio
async def test_memory_loader_empty_tiers_omitted():
    """MEM-03: Empty tiers silently omitted -- no empty ## headings."""
    memory = _AsyncMock()
    memory.load_context = _AsyncMock(return_value=_MockMemoryContext(
        agent_memory_md="notes only",
        division_md="",
        guardrails_md="",
        fact_count=0,
        token_estimate=10,
    ))

    loop, llm = _make_memory_agent_loop(memory_loader=memory)

    captured_messages = []
    original_stream = llm.stream_complete

    async def capturing_stream(messages=None, tools=None, on_token=None):
        captured_messages.extend(messages or [])
        return await original_stream(messages=messages, tools=tools, on_token=on_token)

    llm.stream_complete = capturing_stream
    await loop.run_turn("hello")

    sys_msgs = [m for m in captured_messages if m.get("role") == "system"]
    assert len(sys_msgs) >= 1
    sys_content = sys_msgs[0]["content"]
    assert "## Agent Memory\nnotes only" in sys_content
    assert "## Guardrails" not in sys_content
    assert "## Division Context" not in sys_content


@pytest.mark.asyncio
async def test_memory_loader_failure_nonfatal():
    """MEM-03: Memory load failure is non-fatal -- agent runs with base role only."""
    memory = _AsyncMock()
    memory.load_context = _AsyncMock(side_effect=RuntimeError("db gone"))

    loop, llm = _make_memory_agent_loop(memory_loader=memory)

    captured_messages = []
    original_stream = llm.stream_complete

    async def capturing_stream(messages=None, tools=None, on_token=None):
        captured_messages.extend(messages or [])
        return await original_stream(messages=messages, tools=tools, on_token=on_token)

    llm.stream_complete = capturing_stream

    # Should not raise
    result = await loop.run_turn("hello")
    assert isinstance(result, str)

    sys_msgs = [m for m in captured_messages if m.get("role") == "system"]
    if sys_msgs:
        sys_content = sys_msgs[0]["content"]
        assert "You are a test assistant." in sys_content
        assert "## Agent Memory" not in sys_content


# ---------------------------------------------------------------------------
# Task: compact_md_path parameter tests (08-01)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compact_md_path_from_constructor(tmp_path):
    """AgentLoop constructed with compact_md_path uses that path, not hardcoded Path.home()."""
    from localharness.config.models import AgentConfig
    from tests.conftest import MockLLMClient, FakeLLMResponse

    # Write compact.md to tmp_path (definitely not ~/.localharness)
    compact_file = tmp_path / "compact.md"
    compact_file.write_text("Prior context here")

    cfg = AgentConfig(name="test-agent", role="You are a test assistant.")
    llm = MockLLMClient([FakeLLMResponse(content="Done.")])
    ctx = ContextManager()
    perm = PermissionEvaluator()
    mock_bus = EventBus()

    loop = AgentLoop(
        config=cfg,
        llm=llm,
        bus=mock_bus,
        context_manager=ctx,
        tool_registry=None,
        permission_evaluator=perm,
        compact_md_path=compact_file,
    )

    # Capture messages sent to LLM
    captured_messages = []
    original_stream = llm.stream_complete

    async def capturing_stream(messages=None, tools=None, on_token=None):
        captured_messages.extend(messages or [])
        return await original_stream(messages=messages, tools=tools, on_token=on_token)

    llm.stream_complete = capturing_stream
    await loop.run_turn("hello")

    # The session should contain the compact.md content as [Prior Session Context]
    sys_msgs = [m for m in captured_messages if m.get("role") == "system"]
    compact_contents = [m for m in sys_msgs if "Prior Session Context" in (m.get("content") or "")]
    assert len(compact_contents) >= 1, "compact.md content not loaded into session"
    assert "Prior context here" in compact_contents[0]["content"]


@pytest.mark.asyncio
async def test_compact_md_folds_into_single_leading_system_message(tmp_path):
    """MOVE 0a regression: with a compact.md present, the composed request carries EXACTLY ONE
    system message and it is index 0. A SECOND system message is rejected by strict chat
    templates (vLLM/Qwen: 'System message must be at the beginning') — that latent live-harness
    bug killed 13/15 SEMA-05 days (59/59 TurnFailed). Prior-session context must be FOLDED into
    the first system message's content, never appended as its own system message."""
    from localharness.config.models import AgentConfig
    from tests.conftest import MockLLMClient, FakeLLMResponse

    compact_file = tmp_path / "compact.md"
    compact_file.write_text("user is building a research agent")

    cfg = AgentConfig(name="test-agent", role="You are a test assistant.")
    llm = MockLLMClient([FakeLLMResponse(content="Done.")])
    loop = AgentLoop(
        config=cfg, llm=llm, bus=EventBus(), context_manager=ContextManager(),
        tool_registry=None, permission_evaluator=PermissionEvaluator(),
        compact_md_path=compact_file,
    )

    requests: list[list] = []
    original_stream = llm.stream_complete

    async def capturing_stream(messages=None, tools=None, on_token=None):
        requests.append(list(messages or []))
        return await original_stream(messages=messages, tools=tools, on_token=on_token)

    llm.stream_complete = capturing_stream
    await loop.run_turn("hello")

    assert requests, "no request was sent to the LLM"
    # EVERY request in the turn must carry exactly one system message, at index 0.
    for req in requests:
        sys_idx = [i for i, m in enumerate(req) if m.get("role") == "system"]
        assert sys_idx == [0], f"expected one system message at index 0, got indices {sys_idx}"
    first_sys = requests[0][0]["content"]
    assert "[Prior Session Context]" in first_sys, "compact.md not folded into the system message"
    assert "research agent" in first_sys


def test_compact_md_path_default_fallback():
    """AgentLoop constructed without compact_md_path stores None (fallback activates in run_turn)."""
    from localharness.config.models import AgentConfig

    cfg = AgentConfig(name="test-agent", role="Test.")
    mock_bus = EventBus()
    ctx = ContextManager()
    perm = PermissionEvaluator()

    loop = AgentLoop(
        config=cfg,
        llm=None,
        bus=mock_bus,
        context_manager=ctx,
        tool_registry=None,
        permission_evaluator=perm,
    )
    assert loop._compact_md_path is None


# ---------------------------------------------------------------------------
# Phase 11 / SCEN-02: ParseFailed and StuckRecovered publish contracts
# ---------------------------------------------------------------------------

from unittest.mock import AsyncMock
from localharness.core.events import ParseFailed, StuckRecovered


@pytest.mark.asyncio
async def test_parse_failed_published_on_retry_increment():
    """Stub bus captures ParseFailed with the post-increment retry count and iteration.

    Mirrors the publish-site contract from loop.py: ParseFailed is constructed
    with session.parse_retries (post-increment) and session.iteration.
    """
    bus = AsyncMock()
    await bus.publish(ParseFailed(
        agent_id="agent-1",
        session_id="sess-1",
        iteration=5,
        parse_retry_count=2,
        raw_content_preview="<tool_call>broken",
    ))
    bus.publish.assert_awaited_once()
    ev = bus.publish.await_args.args[0]
    assert isinstance(ev, ParseFailed)
    assert ev.parse_retry_count == 2
    assert ev.iteration == 5
    assert ev.raw_content_preview == "<tool_call>broken"


@pytest.mark.asyncio
async def test_stuck_recovered_published_with_signature():
    """Stub bus captures StuckRecovered with signature + iteration from RECOVERING branch."""
    bus = AsyncMock()
    await bus.publish(StuckRecovered(
        agent_id="agent-1",
        session_id="sess-1",
        iteration=10,
        stuck_signature="tool:bash:{cmd:ls}",
    ))
    bus.publish.assert_awaited_once()
    ev = bus.publish.await_args.args[0]
    assert isinstance(ev, StuckRecovered)
    assert ev.stuck_signature == "tool:bash:{cmd:ls}"
    assert ev.iteration == 10


@pytest.mark.asyncio
async def test_parse_failed_truncates_long_content():
    """raw_content_preview accepts the slice `(raw_content or '')[:200]` pattern from loop.py."""
    long_content = "x" * 500
    ev = ParseFailed(
        agent_id="a", session_id="s", iteration=1, parse_retry_count=1,
        raw_content_preview=(long_content or "")[:200],
    )
    assert len(ev.raw_content_preview) == 200


@pytest.mark.asyncio
async def test_parse_failed_none_content_safe():
    """raw_content_preview accepts the (None or '')[:200] = '' fallback."""
    ev = ParseFailed(
        agent_id="a", session_id="s", iteration=1, parse_retry_count=1,
        raw_content_preview=(None or "")[:200],
    )
    assert ev.raw_content_preview == ""


# ---------------------------------------------------------------------------
# Phase 14 config-driven StuckDetector + recovery_injection
# (14-02 extended AgentConfig + 14-03 wired the agent loop — now live)
# ---------------------------------------------------------------------------


def test_stuck_detector_reads_from_config():
    """The agent loop must instantiate StuckDetector with values from
    AgentConfig.stuck_detector, NOT hardcoded defaults (5/2/3)."""
    from localharness.config.models import AgentConfig
    cfg = AgentConfig(
        name="test",
        role="test",
        stuck_detector={"window_size": 7, "recovery_threshold": 3, "escalation_threshold": 4},
    )
    assert cfg.stuck_detector.window_size == 7
    assert cfg.stuck_detector.recovery_threshold == 3
    assert cfg.stuck_detector.escalation_threshold == 4


def test_recovery_message_from_config():
    """AgentConfig.recovery_injection.message must be addressable via the registry."""
    from localharness.config.models import AgentConfig
    cfg = AgentConfig(
        name="test",
        role="test",
        recovery_injection={"message": "custom recovery wording"},
    )
    assert cfg.recovery_injection.message == "custom recovery wording"


# ---------------------------------------------------------------------------
# Phase 14-03: AgentLoop reads stuck_detector + recovery_injection from AgentConfig
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_loop_instantiates_stuck_detector_from_config(mock_llm_client, bus, monkeypatch):
    """AgentLoop._execute_loop must build StuckDetector with values from
    self._config.stuck_detector (not the hardcoded 5/2/3 literals)."""
    from localharness.config.models import AgentConfig
    from localharness.agent import loop as loop_mod

    captured: dict = {}

    real_init = loop_mod.StuckDetector.__init__

    def spy_init(self, window_size=5, recovery_threshold=2, escalation_threshold=3):
        captured["window_size"] = window_size
        captured["recovery_threshold"] = recovery_threshold
        captured["escalation_threshold"] = escalation_threshold
        real_init(self, window_size=window_size, recovery_threshold=recovery_threshold,
                  escalation_threshold=escalation_threshold)

    monkeypatch.setattr(loop_mod.StuckDetector, "__init__", spy_init)

    cfg = AgentConfig(
        name="test-agent",
        role="t",
        stuck_detector={"window_size": 7, "recovery_threshold": 3, "escalation_threshold": 4},
    )
    Response = mock_llm_client.Response
    loop = _make_agent_loop(mock_llm_client, [Response(content="Done.")], bus, config=cfg)
    await loop.run_turn("task")

    assert captured == {"window_size": 7, "recovery_threshold": 3, "escalation_threshold": 4}


@pytest.mark.asyncio
async def test_agent_loop_stuck_detector_defaults_preserved(mock_llm_client, bus, monkeypatch):
    """With default AgentConfig, StuckDetector still gets (5, 2, 3)."""
    from localharness.config.models import AgentConfig
    from localharness.agent import loop as loop_mod

    captured: dict = {}
    real_init = loop_mod.StuckDetector.__init__

    def spy_init(self, window_size=5, recovery_threshold=2, escalation_threshold=3):
        captured["window_size"] = window_size
        captured["recovery_threshold"] = recovery_threshold
        captured["escalation_threshold"] = escalation_threshold
        real_init(self, window_size=window_size, recovery_threshold=recovery_threshold,
                  escalation_threshold=escalation_threshold)

    monkeypatch.setattr(loop_mod.StuckDetector, "__init__", spy_init)

    cfg = AgentConfig(name="test-agent", role="t")
    Response = mock_llm_client.Response
    loop = _make_agent_loop(mock_llm_client, [Response(content="Done.")], bus, config=cfg)
    await loop.run_turn("task")

    assert captured == {"window_size": 5, "recovery_threshold": 2, "escalation_threshold": 3}


def test_agent_loop_uses_config_recovery_message_not_hardcoded():
    """Source-level guarantee: loop reads self._config.recovery_injection.message,
    no longer calls stuck_detector.recovery_message(repeated_sig)."""
    import inspect
    from localharness.agent import loop as loop_mod
    src = inspect.getsource(loop_mod)
    # Hardcoded form is gone
    assert "StuckDetector(window_size=5, recovery_threshold=2, escalation_threshold=3)" not in src
    # Config-driven access present
    assert "self._config.stuck_detector" in src
    assert "self._config.recovery_injection.message" in src
    # Old call-site replaced (we no longer pass repeated_sig into recovery_message)
    assert "stuck_detector.recovery_message(repeated_sig)" not in src


# ---------------------------------------------------------------------------
# Tool-call JSON guard (_repair_json_object / _sanitize_raw_tool_calls)
# ---------------------------------------------------------------------------

import json as _json
from types import SimpleNamespace

from localharness.agent.loop import _repair_json_object, _sanitize_raw_tool_calls


def test_repair_json_valid_passthrough():
    s = '{"a": 1}'
    assert _repair_json_object(s) is s


def test_repair_json_unterminated_string():
    # Tonight's live failure shape: generation cut mid-arguments string.
    s = '{"agent_id": "web-researcher", "task": "Read these pages and summar'
    repaired = _repair_json_object(s)
    assert repaired is not None
    parsed = _json.loads(repaired)
    assert parsed["agent_id"] == "web-researcher"
    assert parsed["task"].startswith("Read these pages")


def test_repair_json_open_structures():
    repaired = _repair_json_object('{"a": [1, 2, {"b": "c"')
    assert repaired is not None
    assert _json.loads(repaired) == {"a": [1, 2, {"b": "c"}]}


def test_repair_json_dangling_escape():
    repaired = _repair_json_object('{"path": "C:\\\\dir\\')
    assert repaired is not None
    _json.loads(repaired)


def test_repair_json_unrepairable():
    assert _repair_json_object('{"a": twelve}') is None


def _dict_call(args: str, name: str = "agent", id: str = "tc-1"):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": args}}


def test_sanitize_keeps_valid_calls():
    calls = [_dict_call('{"x": 1}')]
    out = _sanitize_raw_tool_calls(calls)
    assert out == calls
    assert out[0]["function"]["arguments"] == '{"x": 1}'


def test_sanitize_repairs_truncated_dict_call():
    calls = [_dict_call('{"agent_id": "web-researcher", "task": "do thi')]
    out = _sanitize_raw_tool_calls(calls)
    assert len(out) == 1
    args = _json.loads(out[0]["function"]["arguments"])  # normalized, parseable
    assert args["agent_id"] == "web-researcher"


def test_sanitize_repairs_object_style_call():
    fn = SimpleNamespace(name="agent", arguments='{"task": "x')
    tc = SimpleNamespace(id="tc-2", function=fn)
    out = _sanitize_raw_tool_calls([tc])
    assert len(out) == 1
    assert _json.loads(out[0].function.arguments) == {"task": "x"}


def test_sanitize_drops_unrepairable_and_preserves_none_semantics():
    out = _sanitize_raw_tool_calls([_dict_call('{"a": twelve}')])
    assert out is None
    assert _sanitize_raw_tool_calls(None) is None
    assert _sanitize_raw_tool_calls([]) == []


# ---------------------------------------------------------------------------
# Budget notes on tool results (_budget_note + loop wiring)
# ---------------------------------------------------------------------------

from unittest.mock import patch

from localharness.agent.loop import BudgetTracker, _budget_note


def _session_for_note(actions: int) -> Session:
    s = Session(agent_id="a", session_id="s", messages=[])
    s.actions_taken = actions
    return s


def test_budget_note_reports_usage():
    note = _budget_note(_session_for_note(3), BudgetTracker(max_actions=12, max_duration_minutes=0))
    assert "3/12 tool calls used" in note
    assert "wrap up" not in note


def test_budget_note_warns_near_action_limit():
    note = _budget_note(_session_for_note(10), BudgetTracker(max_actions=12, max_duration_minutes=0))
    assert "wrap up NOW" in note
    assert "final summary" in note


def test_budget_note_warns_near_time_limit():
    s = _session_for_note(1)
    with patch.object(Session, "elapsed_minutes", return_value=2.5):
        note = _budget_note(s, BudgetTracker(max_actions=0, max_duration_minutes=3.0))
    assert "2.5/3 min elapsed" in note
    assert "wrap up NOW" in note


def test_budget_note_empty_when_unlimited():
    assert _budget_note(_session_for_note(5), BudgetTracker(max_actions=0, max_duration_minutes=0)) == ""


@pytest.mark.asyncio
async def test_loop_appends_budget_note_to_tool_result(faithful_fake_llm, bus, tmp_path):
    from localharness.agent.context import ContextManager
    from localharness.agent.permissions import PermissionEvaluator
    from localharness.config.models import AgentConfig
    from localharness.tools import ToolRegistry
    from localharness.tools.builtin import register_builtin_tools

    reg = ToolRegistry()
    await register_builtin_tools(reg)
    cfg = AgentConfig.model_validate({
        "name": "budget-note-agent",
        "role": "Test agent.",
        # P-A floor: host-tool agent, web ingestion stripped so it resolves clean (uses glob).
        "tools": {"deny": ["web_search", "web_fetch", "web_page_query"]},
        "permissions": {"budget": {"max_actions": 5, "max_duration_minutes": 5.0,
                                   "kill_file": str(tmp_path / "KILL")}},
    })
    loop = AgentLoop(
        config=cfg, llm=faithful_fake_llm(tool_plan=[("glob", {"pattern": "*.py"})]),
        bus=bus, context_manager=ContextManager(), tool_registry=reg,
        permission_evaluator=PermissionEvaluator(), memory_loader=None,
    )
    session = Session(agent_id="budget-note-agent", session_id="s-note", messages=[])
    await loop._execute_loop(session, "list python files", None)

    tool_msgs = [m for m in session.messages if m.get("role") == "tool"]
    assert tool_msgs, "expected at least one tool result"
    assert "[budget: 1/5 tool calls used" in tool_msgs[-1]["content"]


# ---------------------------------------------------------------------------
# Final summary on budget exhaustion (_final_summary_on_budget)
# ---------------------------------------------------------------------------


def _budget_loop(faithful_fake_llm, bus, tmp_path, tool_plan, max_actions):
    from localharness.agent.context import ContextManager
    from localharness.agent.permissions import PermissionEvaluator
    from localharness.config.models import AgentConfig
    from localharness.tools import ToolRegistry

    async def _make():
        from localharness.tools.builtin import register_builtin_tools
        reg = ToolRegistry()
        await register_builtin_tools(reg)
        cfg = AgentConfig.model_validate({
            "name": "budget-final-agent",
            "role": "Test agent.",
            # P-A floor: host-tool agent, web ingestion stripped so it resolves clean (uses glob).
            "tools": {"deny": ["web_search", "web_fetch", "web_page_query"]},
            "permissions": {"budget": {"max_actions": max_actions, "max_duration_minutes": 5.0,
                                       "kill_file": str(tmp_path / "KILL")}},
        })
        return AgentLoop(
            config=cfg, llm=faithful_fake_llm(tool_plan=tool_plan),
            bus=bus, context_manager=ContextManager(), tool_registry=reg,
            permission_evaluator=PermissionEvaluator(), memory_loader=None,
        )
    return _make


@pytest.mark.asyncio
async def test_budget_exhaust_returns_model_findings(faithful_fake_llm, bus, tmp_path):
    """Exhausted budget triggers ONE no-tools pass; the summary carries findings
    (here: the fake echoes the last tool result) plus the budget notice."""
    make = _budget_loop(faithful_fake_llm, bus, tmp_path,
                        tool_plan=[("glob", {"pattern": "*.py"})], max_actions=1)
    loop = await make()
    session = Session(agent_id="budget-final-agent", session_id="s-bf", messages=[])
    summary = await loop._execute_loop(session, "list python files", None)

    assert session.terminated_reason == "budget_actions"
    assert "[Budget limit reached:" in summary
    # the echoed tool result (real glob output incl. budget note) — not a bare notice
    assert "[budget: 1/1 tool calls used" in summary
    # the forced pass also lands in history for session continuity
    assert session.messages[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_budget_exhaust_falls_back_when_model_yields_no_text(faithful_fake_llm, bus, tmp_path):
    """If the final pass produces no usable text (fake still wants to emit a tool call,
    content=None), fall back to the plain budget notice — never crash."""
    make = _budget_loop(faithful_fake_llm, bus, tmp_path,
                        tool_plan=[("glob", {"pattern": "*.py"}), ("glob", {"pattern": "*.md"})],
                        max_actions=1)
    loop = await make()
    session = Session(agent_id="budget-final-agent", session_id="s-bf2", messages=[])
    summary = await loop._execute_loop(session, "list files", None)

    assert session.terminated_reason == "budget_actions"
    assert "[Budget limit reached:" in summary


# ---------------------------------------------------------------------------
# Tool error forwarding + agent tool timeout contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_tool_error_text_reaches_model(faithful_fake_llm, bus, tmp_path):
    """Error results carry their message in .error with output='' — the loop must
    forward the message or the model sees an empty result it can't react to
    (observed live: every fetch failure and the 600s agent timeout surfaced as '')."""
    from localharness.agent.context import ContextManager
    from localharness.agent.permissions import PermissionEvaluator
    from localharness.config.models import AgentConfig
    from localharness.tools import ToolRegistry
    from localharness.tools.builtin import register_builtin_tools

    reg = ToolRegistry()
    await register_builtin_tools(reg)
    cfg = AgentConfig.model_validate({
        "name": "err-fwd-agent",
        "role": "Test agent.",
        # P-A floor: this test drives web_fetch, so deny host-dangerous => web-only clean topology.
        "tools": {"deny": ["bash_exec", "write", "edit", "python_exec"]},
        "permissions": {"budget": {"max_actions": 5, "max_duration_minutes": 5.0,
                                   "kill_file": str(tmp_path / "KILL")}},
    })
    # invalid URL -> WebFetchTool returns err("Invalid URL ...") with output ""
    loop = AgentLoop(
        config=cfg, llm=faithful_fake_llm(tool_plan=[("web_fetch", {"url": "notaurl"})]),
        bus=bus, context_manager=ContextManager(), tool_registry=reg,
        permission_evaluator=PermissionEvaluator(), memory_loader=None,
    )
    session = Session(agent_id="err-fwd-agent", session_id="s-errfwd", messages=[])
    await loop._execute_loop(session, "fetch something", None)

    tool_msgs = [m for m in session.messages if m.get("role") == "tool"]
    assert tool_msgs
    assert "[tool error]" in tool_msgs[0]["content"]
    assert "Invalid URL" in tool_msgs[0]["content"]


def test_agent_tool_timeout_exceeds_child_budget_and_summary_headroom():
    """600s cancelled children mid-final-summary (no terminal event, '' to parent).
    The timeout must stay >= child max duration + slow-model summary headroom."""
    from localharness.agent.subagent import WEB_MAX_DURATION_MINUTES
    from localharness.tools.builtin.agent_tool import AgentTool

    assert AgentTool.timeout_s >= 1800.0
    assert AgentTool.timeout_s >= (WEB_MAX_DURATION_MINUTES * 60) * 2


# ---------------------------------------------------------------------------
# Act-guard: announce-then-halt gets one deterministic nudge
# ---------------------------------------------------------------------------


class _AnnounceThenActLLM:
    """First call: pure intent text, no tool calls. After the nudge: emits the tool
    call, then echoes (FaithfulFakeLLM-style) on the final call."""

    def __init__(self):
        self.calls = 0
        class _Cfg: pass
        self.config = _Cfg(); self.config.tool_call_mode = "native"; self.config.context_window = 128000

    async def stream_complete(self, messages=None, tools=None, on_token=None):
        from types import SimpleNamespace as NS
        self.calls += 1
        if self.calls == 1:
            return NS(content="I'll list the files and then summarize.", tool_calls=None), None
        if self.calls == 2:
            return NS(content=None, tool_calls=[
                {"id": "tc-1", "type": "function",
                 "function": {"name": "glob", "arguments": '{"pattern": "*.py"}'}}]), None
        last = next((m.get("content") for m in reversed(messages or []) if m.get("role") == "tool"), "done")
        return NS(content=str(last)[:100], tool_calls=None), None


@pytest.mark.asyncio
async def test_act_guard_nudges_announce_then_halt(bus, tmp_path):
    from localharness.agent.context import ContextManager
    from localharness.agent.permissions import PermissionEvaluator
    from localharness.config.models import AgentConfig
    from localharness.tools import ToolRegistry
    from localharness.tools.builtin import register_builtin_tools

    reg = ToolRegistry()
    await register_builtin_tools(reg)
    cfg = AgentConfig.model_validate({
        "name": "act-guard-agent", "role": "Test.",
        # P-A floor: host-tool agent, web ingestion stripped so it resolves clean (uses glob).
        "tools": {"deny": ["web_search", "web_fetch", "web_page_query"]},
        "permissions": {"budget": {"max_actions": 5, "max_duration_minutes": 5.0,
                                   "kill_file": str(tmp_path / "KILL")}},
        "self_check": {"enabled": False},
    })
    llm = _AnnounceThenActLLM()
    loop = AgentLoop(config=cfg, llm=llm, bus=bus, context_manager=ContextManager(),
                     tool_registry=reg, permission_evaluator=PermissionEvaluator(),
                     memory_loader=None)
    session = Session(agent_id="act-guard-agent", session_id="s-ag", messages=[])
    await loop._execute_loop(session, "list the python files", None)

    assert session.act_nudge_used is True
    assert session.actions_taken == 1          # the glob actually ran after the nudge
    nudges = [m for m in session.messages if m.get("role") == "user"
              and "took no action" in (m.get("content") or "")]
    assert len(nudges) == 1                    # exactly one nudge, persisted in history
    assert session.terminated_reason == "complete"


class _ConfirmOnNudgeLLM:
    """Conversational sign-off, no tool calls; after the act-guard nudge replies with the
    bare CONFIRMED sentinel. Mirrors _AnnounceThenActLLM's NS/call-counting shape (issue #6:
    the prior reply must surface untouched — no restate, no meta-narration)."""

    def __init__(self):
        self.calls = 0
        class _Cfg: pass
        self.config = _Cfg(); self.config.tool_call_mode = "native"; self.config.context_window = 128000

    async def stream_complete(self, messages=None, tools=None, on_token=None):
        from types import SimpleNamespace as NS
        self.calls += 1
        if self.calls == 1:
            return NS(content="Sounds like a great plan — enjoy the food!", tool_calls=None), None
        return NS(content="CONFIRMED", tool_calls=None), None


@pytest.mark.asyncio
async def test_act_guard_confirmed_surfaces_prior_reply(bus, tmp_path):
    """Issue #6: a nudged no-tool turn that replies CONFIRMED surfaces the model's ORIGINAL
    reply as TaskComplete.summary — the sentinel machinery (_format_completion_summary) does
    the job: no duplicate, no leaked 'CONFIRMED', no meta-narration."""
    from localharness.agent.context import ContextManager
    from localharness.agent.permissions import PermissionEvaluator
    from localharness.config.models import AgentConfig
    from localharness.tools import ToolRegistry
    from localharness.tools.builtin import register_builtin_tools

    reg = ToolRegistry()
    await register_builtin_tools(reg)
    cfg = AgentConfig.model_validate({
        "name": "act-guard-agent", "role": "Test.",
        "tools": {"deny": ["web_search", "web_fetch", "web_page_query"]},
        "permissions": {"budget": {"max_actions": 5, "max_duration_minutes": 5.0,
                                   "kill_file": str(tmp_path / "KILL")}},
        "self_check": {"enabled": False},
    })
    llm = _ConfirmOnNudgeLLM()
    loop = AgentLoop(config=cfg, llm=llm, bus=bus, context_manager=ContextManager(),
                     tool_registry=reg, permission_evaluator=PermissionEvaluator(),
                     memory_loader=None)
    session = Session(agent_id="act-guard-agent", session_id="s-ag-confirmed", messages=[])
    await loop._execute_loop(session, "thanks!", None)

    completions = bus.history(event_types=[TaskComplete])
    assert session.act_nudge_used is True                       # (a) the nudge fired
    assert len(completions) == 1                                # (b) exactly one completion
    assert completions[0].summary == "Sounds like a great plan — enjoy the food!"
    assert "CONFIRMED" not in completions[0].summary            # (c) sentinel never leaks
    assert session.terminated_reason == "complete"             # (d) clean finish


# ---------------------------------------------------------------------------
# Reasoning-parser tool turns: content=None must never enter history/payloads
# (C0 sweep: vLLM request validation rejects replayed content:None -> HTTP 400,
# poisoning every later turn in the session)
# ---------------------------------------------------------------------------


class _ReasoningParserToolCallLLM:
    """Reasoning-parser shape (--reasoning-parser qwen3): the tool-call turn returns
    content=None — ALL tokens went to reasoning + tool_calls. Records every outgoing
    request payload so tests can assert exactly what would hit the server. Optional
    first-turn content string covers the passthrough case."""

    def __init__(self, first_content=None):
        self.calls = 0
        self.first_content = first_content
        self.seen_payloads: list[list[dict]] = []
        class _Cfg: pass
        self.config = _Cfg(); self.config.tool_call_mode = "native"; self.config.context_window = 128000

    async def stream_complete(self, messages=None, tools=None, on_token=None):
        from types import SimpleNamespace as NS
        self.seen_payloads.append([dict(m) for m in (messages or [])])
        self.calls += 1
        if self.calls == 1:
            return NS(content=self.first_content, tool_calls=[
                {"id": "tc-none", "type": "function",
                 "function": {"name": "glob", "arguments": '{"pattern": "*.py"}'}}]), None
        return NS(content="all done", tool_calls=None), None


async def _run_reasoning_parser_double(bus, tmp_path, llm):
    from localharness.config.models import AgentConfig
    from localharness.tools import ToolRegistry
    from localharness.tools.builtin import register_builtin_tools

    reg = ToolRegistry()
    await register_builtin_tools(reg)
    cfg = AgentConfig.model_validate({
        "name": "none-content-agent", "role": "Test.",
        "tools": {"deny": ["web_search", "web_fetch", "web_page_query"]},
        "permissions": {"budget": {"max_actions": 5, "max_duration_minutes": 5.0,
                                   "kill_file": str(tmp_path / "KILL")}},
        "self_check": {"enabled": False},
    })
    loop = AgentLoop(config=cfg, llm=llm, bus=bus, context_manager=ContextManager(),
                     tool_registry=reg, permission_evaluator=PermissionEvaluator(),
                     memory_loader=None)
    session = Session(agent_id="none-content-agent", session_id="s-none", messages=[])
    await loop._execute_loop(session, "list the python files", None)
    return session


@pytest.mark.asyncio
async def test_none_content_tool_turn_history_entry_normalized(bus, tmp_path):
    """The assistant tool-call turn with content=None must land in history with
    content:"" (never None) and tool_calls intact — every later request replays
    the entry and vLLM rejects None ('Input should be a valid string')."""
    llm = _ReasoningParserToolCallLLM()
    session = await _run_reasoning_parser_double(bus, tmp_path, llm)

    asst = next(m for m in session.messages
                if m.get("role") == "assistant" and m.get("tool_calls"))
    assert asst["content"] == ""                              # not None
    assert asst["tool_calls"][0]["id"] == "tc-none"           # tool_calls not dropped


@pytest.mark.asyncio
async def test_none_content_tool_turn_second_request_payload_valid(bus, tmp_path):
    """Two-turn double: turn 1 returns content=None + tool_calls; the NEXT request's
    actual outgoing payload must contain no None content in ANY message (the exact
    shape vLLM 400s on) and the turn must complete."""
    llm = _ReasoningParserToolCallLLM()
    session = await _run_reasoning_parser_double(bus, tmp_path, llm)

    assert llm.calls >= 2
    second = llm.seen_payloads[1]
    offenders = [m for m in second if m.get("content") is None]
    assert offenders == []                                    # nothing None on the wire
    asst = next(m for m in second if m.get("role") == "assistant" and m.get("tool_calls"))
    assert asst["content"] == ""
    assert session.terminated_reason == "complete"


@pytest.mark.asyncio
async def test_string_content_tool_turn_passes_through_untouched(bus, tmp_path):
    """Normalization only rewrites None: a tool-call turn WITH real string content
    keeps it verbatim in history."""
    llm = _ReasoningParserToolCallLLM(first_content="Let me check the files.")
    session = await _run_reasoning_parser_double(bus, tmp_path, llm)

    asst = next(m for m in session.messages
                if m.get("role") == "assistant" and m.get("tool_calls"))
    assert asst["content"] == "Let me check the files."


@pytest.mark.asyncio
async def test_act_guard_nudge_text_offers_sentinel(bus, tmp_path):
    """The single act-guard nudge asks for the bare CONFIRMED sentinel and never invites a
    'restate' (issue #6: 'restate the complete final answer' drew meta-narrated duplicates)."""
    from localharness.agent.context import ContextManager
    from localharness.agent.permissions import PermissionEvaluator
    from localharness.config.models import AgentConfig
    from localharness.tools import ToolRegistry
    from localharness.tools.builtin import register_builtin_tools

    reg = ToolRegistry()
    await register_builtin_tools(reg)
    cfg = AgentConfig.model_validate({
        "name": "act-guard-agent", "role": "Test.",
        "tools": {"deny": ["web_search", "web_fetch", "web_page_query"]},
        "permissions": {"budget": {"max_actions": 5, "max_duration_minutes": 5.0,
                                   "kill_file": str(tmp_path / "KILL")}},
        "self_check": {"enabled": False},
    })
    llm = _ConfirmOnNudgeLLM()
    loop = AgentLoop(config=cfg, llm=llm, bus=bus, context_manager=ContextManager(),
                     tool_registry=reg, permission_evaluator=PermissionEvaluator(),
                     memory_loader=None)
    session = Session(agent_id="act-guard-agent", session_id="s-ag-nudgetext", messages=[])
    await loop._execute_loop(session, "thanks!", None)

    nudges = [m for m in session.messages if m.get("role") == "user"
              and "took no action" in (m.get("content") or "")]
    assert len(nudges) == 1
    assert "reply with exactly CONFIRMED" in nudges[0]["content"]
    assert "restate" not in nudges[0]["content"]


# ---------------------------------------------------------------------------
# SESS-03: the loop.py caller — compaction summaries persist as a per-sitting gist
# ---------------------------------------------------------------------------

def _compacting_ctx(summary_text: str = "compacted: A then B"):
    """A real ContextManager whose pipeline fires SummaryCompactionStage on any over-window
    build. NO bus is wired — EXACTLY production's shape (start_cmd passes the ContextManager
    no bus), so CompactionTriggered structurally cannot fire and the '[Context Summary]'
    marker scan is the only observation hook (research Pitfall 2)."""
    from localharness.agent.context import CompactionPipeline, TokenCounter

    tc = TokenCounter()  # tiktoken, offline — no server needed

    async def _fake_summarize(middle):
        return summary_text

    pipeline = CompactionPipeline(
        token_counter=tc, llm_summarize_fn=_fake_summarize,
        preserve_first_n=1, preserve_last_n=1,
    )
    return ContextManager(max_context_tokens=2_000, pipeline=pipeline, token_counter=tc)


def _big_msgs(rounds: int = 6):
    """A conversation far over a 2k window: alternating user/assistant, each ~1k tokens."""
    return [
        m for i in range(rounds)
        for m in (
            {"role": "user", "content": f"u{i} " + "X" * 4000},
            {"role": "assistant", "content": f"a{i} " + "Y" * 4000},
        )
    ]


@pytest.mark.asyncio
async def test_compaction_gist_persisted_once_per_sitting(mock_llm_client, bus, tmp_path):
    """Composed: run_turn → _execute_loop → build_messages compaction → the marker scan →
    persist_compaction_gist. Two compacting turns in one sitting leave EXACTLY ONE active
    gist row at gist/compaction/sit-1 (supersede/corroborate — never duplicate keys)."""
    from localharness.memory.sqlite import MemoryStore

    store = MemoryStore(agent_id="test-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await store.open()
    try:
        Response = mock_llm_client.Response
        loop = _make_agent_loop(
            mock_llm_client, [Response(content="done 1"), Response(content="done 2")], bus,
            session_id="sit-1", memory_loader=store, context_manager=_compacting_ctx(),
        )
        await loop.run_turn("first", initial_messages=_big_msgs())  # compaction #1
        await loop.run_turn("second")                               # #2 (messages never shrink)

        history = await store.get_fact_history("gist/compaction/sit-1")
        active = [f for f in history if f.status == "active"]
        assert len(active) == 1                       # exactly one active after >= 2 compactions
        assert "compacted: A then B" in active[0].value
        assert active[0].node_kind == "gist" and active[0].provenance == "sit-1"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_compaction_gist_staging_discipline(mock_llm_client, bus, tmp_path):
    """The 0.6 gist write must NOT reorder the injected block: _render_memory_index is
    byte-identical before and after a compacting turn (below the 0.7 injection gate). The
    get_fact assert keeps it non-vacuous — proves a gist was actually written."""
    from localharness.memory.sqlite import MemoryStore

    store = MemoryStore(agent_id="test-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await store.open()
    try:
        Response = mock_llm_client.Response
        loop = _make_agent_loop(
            mock_llm_client, [Response(content="done")], bus,
            session_id="sit-1", memory_loader=store, context_manager=_compacting_ctx(),
        )
        index_before = await store._render_memory_index(10)
        await loop.run_turn("first", initial_messages=_big_msgs())

        assert await store.get_fact("gist/compaction/sit-1") is not None  # the write happened
        assert await store._render_memory_index(10) == index_before       # byte-identical
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_no_memory_no_crash(mock_llm_client, bus):
    """The scan is guarded by `self._memory is not None`: a compacting turn on a loop with
    no memory_loader completes normally, writing no gist and raising nothing."""
    Response = mock_llm_client.Response
    loop = _make_agent_loop(
        mock_llm_client, [Response(content="all done")], bus,
        session_id="sit-1", context_manager=_compacting_ctx(),  # memory_loader stays None
    )
    summary = await loop.run_turn("first", initial_messages=_big_msgs())
    assert "all done" in summary  # reached natural completion, no exception path


# ---------------------------------------------------------------------------
# #77 — Output-token-ceiling truncation guard. A completion cut off at max_tokens
# mid-tool-call (finish_reason="length") must NOT execute the truncated call: the
# loop feeds a deterministic tool-role remedy, counts it, and continues so the model
# retries informed (no blind identical retries, no silently-truncated file).
# ---------------------------------------------------------------------------

class _RecordingRegistry:
    """Tool registry that records every dispatch so a test can prove a call did/didn't run."""
    def __init__(self, dispatched: list):
        self._dispatched = dispatched

    def get_tools_for_agent(self, agent_id, division_id, tool_config):
        return {}

    async def dispatch(self, name, arguments, agent_id, division_id, tool_config):
        from localharness.tools.base import ToolResult
        self._dispatched.append((name, arguments))
        return ToolResult(output="ok", success=True)


@pytest.mark.asyncio
async def test_truncated_tool_call_not_executed_native(mock_llm_client, bus):
    """finish_reason='length' + tool calls (native) → suppress execution, push a tool-role
    remedy naming cause + fix, continue to a clean retry."""
    Response = mock_llm_client.Response
    ToolCallObj = mock_llm_client.ToolCall
    dispatched: list = []
    tc = ToolCallObj(id="tc-1", name="write", arguments={"path": "/tmp/x"})
    responses = [
        Response(content=None, tool_calls=[tc], finish_reason="length"),
        Response(content="Recovered with a smaller write.", finish_reason="stop"),
    ]
    loop = _make_agent_loop(mock_llm_client, responses, bus,
                            tool_registry=_RecordingRegistry(dispatched))
    summary = await loop.run_turn("write a big file")

    assert dispatched == []                                   # truncated call NOT executed
    assert "Recovered with a smaller write." in summary       # loop continued to completion
    remedy = [m for m in loop._conversation
              if m.get("role") == "tool" and m.get("tool_call_id") == "tc-1"]
    assert len(remedy) == 1                                    # the call's id was answered
    assert "cut off" in remedy[0]["content"]
    assert "smaller pieces" in remedy[0]["content"]


@pytest.mark.asyncio
async def test_finish_stop_with_tool_calls_executes_unchanged(mock_llm_client, bus):
    """Guard is scoped strictly to finish_reason='length': a normal completion carrying tool
    calls (finish_reason='stop') dispatches exactly as before."""
    Response = mock_llm_client.Response
    ToolCallObj = mock_llm_client.ToolCall
    dispatched: list = []
    tc = ToolCallObj(id="tc-1", name="bash", arguments={"cmd": "ls"})
    responses = [
        Response(content=None, tool_calls=[tc], finish_reason="stop"),
        Response(content="Finished after tool.", finish_reason="stop"),
    ]
    loop = _make_agent_loop(mock_llm_client, responses, bus,
                            tool_registry=_RecordingRegistry(dispatched))
    summary = await loop.run_turn("task")

    assert dispatched == [("bash", {"cmd": "ls"})]            # executed unchanged
    assert "Finished after tool." in summary


@pytest.mark.asyncio
async def test_truncated_tool_call_not_executed_xml(mock_llm_client, bus):
    """xml mode: truncation cuts the embedded <tool_call> text, but finish_reason='length'
    rides the same stream chunk, so the same guard suppresses execution. The xml back-fill
    populates the assistant tool_calls, so the tool-role remedy still pairs cleanly."""
    Response = mock_llm_client.Response
    dispatched: list = []
    xml = ('<tool_call>\n{"name": "write", "arguments": {"path": "/tmp/x", "content": "y"}}\n'
           '</tool_call>')
    responses = [
        Response(content=xml, tool_calls=[], finish_reason="length"),
        Response(content="Recovered.", finish_reason="stop"),
    ]
    loop = _make_agent_loop(mock_llm_client, responses, bus,
                            tool_registry=_RecordingRegistry(dispatched))
    loop._llm.config.tool_call_mode = "xml"
    summary = await loop.run_turn("task")

    assert dispatched == []                                   # truncated xml call NOT executed
    assert "Recovered." in summary
    assert any(m.get("role") == "tool" and "cut off" in (m.get("content") or "")
               for m in loop._conversation)


@pytest.mark.asyncio
async def test_step_truncated_tool_call_counts_and_skips(mock_llm_client, bus):
    """step() (the single-iteration twin) has the same seam: a length-truncated tool call is
    not executed, is counted in session stats, and pushes the remedy."""
    Response = mock_llm_client.Response
    ToolCallObj = mock_llm_client.ToolCall
    dispatched: list = []
    tc = ToolCallObj(id="tc-1", name="write", arguments={"path": "/tmp/x"})
    loop = _make_agent_loop(
        mock_llm_client,
        [Response(content=None, tool_calls=[tc], finish_reason="length")],
        bus, tool_registry=_RecordingRegistry(dispatched),
    )
    session = Session(agent_id="test-agent", session_id="s1",
                      messages=[{"role": "user", "content": "go"}])
    result = await loop.step(session)

    assert dispatched == []                                   # not executed
    assert session.truncated_tool_calls == 1                  # counted in session stats
    assert result.action != "tool_calls"                     # did not report an execution
    assert any(m.get("role") == "tool" and "cut off" in (m.get("content") or "")
               for m in session.messages)
