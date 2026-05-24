"""Tests for localharness.core.events — all 15 event models, BudgetSpec, AnyEvent, EVENT_TYPE_MAP."""
import json
import pytest
from localharness.core.events import (
    Action,
    AgentCreated,
    AgentDeleted,
    AnyEvent,
    BaseEvent,
    BudgetSpec,
    DelegationRequest,
    DelegationResult,
    Escalation,
    EVENT_TYPE_MAP,
    Heartbeat,
    Observation,
    SystemReady,
    TaskComplete,
    TaskRequest,
    TurnCompleted,
    TurnFailed,
    TurnStarted,
    UserMessage,
    deserialize_event,
)
from localharness.core.types import AgentID, DivisionID, OrgID, SessionID


def test_required_event_types():
    """All 6 required types instantiate without error and have distinct event_type strings."""
    action = Action(agent_id=AgentID("a"), session_id=SessionID("s"), action_type="tool_call")
    obs = Observation(agent_id=AgentID("a"), session_id=SessionID("s"), observation_type="tool_result")
    delegation = DelegationRequest(
        requesting_agent_id=AgentID("a"),
        target_agent_id=AgentID("b"),
        session_id=SessionID("s"),
        task_file="/tmp/task.md",
        budget=BudgetSpec(),
    )
    result = DelegationResult(
        requesting_agent_id=AgentID("a"),
        target_agent_id=AgentID("b"),
        session_id=SessionID("s"),
        success=True,
        summary="done",
    )
    esc = Escalation(
        agent_id=AgentID("a"),
        session_id=SessionID("s"),
        reason="stuck",
        detail="repeated 3x",
        iteration_at_escalation=5,
    )
    hb = Heartbeat(
        agent_id=AgentID("a"),
        session_id=SessionID("s"),
        iteration=1,
        context_utilization_pct=10.0,
    )
    types = {action.event_type, obs.event_type, delegation.event_type, result.event_type, esc.event_type, hb.event_type}
    assert len(types) == 6, "All 6 required event types must have distinct event_type strings"


def test_all_15_event_types():
    """All 15 event types instantiate without error."""
    budget = BudgetSpec()
    SystemReady(config_path="/etc/lh.yaml", provider_base_url="http://localhost:11434", provider_type="ollama", detected_models=["qwen2.5"])
    AgentCreated(agent_id=AgentID("x"), config_path="/tmp/x.yaml")
    AgentDeleted(agent_id=AgentID("x"), config_path="/tmp/x.yaml")
    TurnStarted(agent_id=AgentID("a"), session_id=SessionID("s"), task_summary="hello", budget=budget)
    TurnCompleted(agent_id=AgentID("a"), session_id=SessionID("s"), iterations=1, duration_seconds=1.0, elapsed_tokens=100, summary="done")
    TurnFailed(agent_id=AgentID("a"), session_id=SessionID("s"), reason="llm_error", detail="oops", iterations=1, duration_seconds=0.1)
    UserMessage(content="hi", channel="terminal")
    TaskRequest(agent_id=AgentID("a"), session_id=SessionID("s"), task="do it", budget=budget)
    TaskComplete(agent_id=AgentID("a"), session_id=SessionID("s"), success=True, summary="ok", duration_seconds=1.0, iterations=1)
    Action(agent_id=AgentID("a"), session_id=SessionID("s"), action_type="tool_call")
    Observation(agent_id=AgentID("a"), session_id=SessionID("s"), observation_type="tool_result")
    DelegationRequest(requesting_agent_id=AgentID("a"), target_agent_id=AgentID("b"), session_id=SessionID("s"), task_file="/tmp/t.md", budget=budget)
    DelegationResult(requesting_agent_id=AgentID("a"), target_agent_id=AgentID("b"), session_id=SessionID("s"), success=True, summary="done")
    Escalation(agent_id=AgentID("a"), session_id=SessionID("s"), reason="stuck", detail="x", iteration_at_escalation=1)
    Heartbeat(agent_id=AgentID("a"), session_id=SessionID("s"), iteration=1, context_utilization_pct=5.0)


def test_event_type_matches_class_name():
    """Every event class has event_type matching the Python class name (PascalCase)."""
    classes = [
        SystemReady, AgentCreated, AgentDeleted, TurnStarted, TurnCompleted, TurnFailed,
        UserMessage, TaskRequest, TaskComplete, Action, Observation,
        DelegationRequest, DelegationResult, Escalation, Heartbeat,
    ]
    for cls in classes:
        # Get the default value from model_fields
        default = cls.model_fields["event_type"].default
        assert default == cls.__name__, f"{cls.__name__}: event_type default '{default}' != class name '{cls.__name__}'"


def test_base_event_defaults():
    """BaseEvent generates uuid id, timestamp, seq=None, agent_id=None, session_id=None, parent_id=None."""
    # Use Heartbeat as a concrete instantiable subclass
    hb = Heartbeat(agent_id=AgentID("a"), session_id=SessionID("s"), iteration=1, context_utilization_pct=5.0)
    assert hb.id is not None and len(hb.id) == 36  # UUID4
    assert hb.timestamp is not None
    assert hb.parent_id is None

    # Check seq=None default via Action (has optional fields)
    action = Action(agent_id=AgentID("a"), session_id=SessionID("s"), action_type="llm_response")
    assert action.seq is None


def test_event_serialization_roundtrip():
    """model_dump_json() -> model_validate_json() produces equivalent event for Action and Observation."""
    action = Action(agent_id=AgentID("a"), session_id=SessionID("s"), action_type="tool_call", tool_name="bash")
    json_str = action.model_dump_json()
    restored = Action.model_validate_json(json_str)
    assert restored.id == action.id
    assert restored.action_type == action.action_type
    assert restored.tool_name == action.tool_name
    assert restored.agent_id == action.agent_id

    obs = Observation(agent_id=AgentID("a"), session_id=SessionID("s"), observation_type="tool_result", output="hello")
    obs_json = obs.model_dump_json()
    restored_obs = Observation.model_validate_json(obs_json)
    assert restored_obs.id == obs.id
    assert restored_obs.output == "hello"


def test_event_type_map_complete():
    """EVENT_TYPE_MAP has entries for all 15 event types."""
    assert len(EVENT_TYPE_MAP) == 15
    expected_keys = {
        "SystemReady", "AgentCreated", "AgentDeleted", "TurnStarted", "TurnCompleted",
        "TurnFailed", "UserMessage", "TaskRequest", "TaskComplete", "Action",
        "Observation", "DelegationRequest", "DelegationResult", "Escalation", "Heartbeat",
    }
    assert set(EVENT_TYPE_MAP.keys()) == expected_keys


def test_deserialize_event():
    """deserialize_event(json_line) correctly reconstructs Action and Observation from JSON strings."""
    action = Action(agent_id=AgentID("a"), session_id=SessionID("s"), action_type="llm_response", content="hello")
    line = action.model_dump_json()
    restored = deserialize_event(line)
    assert isinstance(restored, Action)
    assert restored.id == action.id
    assert restored.content == "hello"

    obs = Observation(agent_id=AgentID("a"), session_id=SessionID("s"), observation_type="tool_result", output="world")
    obs_line = obs.model_dump_json()
    restored_obs = deserialize_event(obs_line)
    assert isinstance(restored_obs, Observation)
    assert restored_obs.output == "world"


def test_budget_spec_frozen():
    """BudgetSpec(max_actions=50) is immutable (raises on field assignment)."""
    spec = BudgetSpec(max_actions=50)
    assert spec.max_actions == 50
    with pytest.raises(Exception):  # ValidationError or TypeError from frozen model
        spec.max_actions = 100  # type: ignore[misc]


def test_any_event_union():
    """AnyEvent type contains all 15 event classes."""
    # AnyEvent is a Union; check its __args__
    import typing
    args = typing.get_args(AnyEvent)
    assert len(args) == 15
    expected = {
        SystemReady, AgentCreated, AgentDeleted, TurnStarted, TurnCompleted, TurnFailed,
        UserMessage, TaskRequest, TaskComplete, Action, Observation,
        DelegationRequest, DelegationResult, Escalation, Heartbeat,
    }
    assert set(args) == expected


def test_division_id_org_id_exist():
    """DivisionID and OrgID are NewType wrappers over str."""
    div = DivisionID("engineering")
    org = OrgID("acme")
    assert isinstance(div, str)
    assert isinstance(org, str)
    assert div == "engineering"
    assert org == "acme"


def test_model_copy_updates_seq():
    """event.model_copy(update={'seq': 42}) produces new instance with seq=42, original unchanged."""
    from localharness.core.types import EventSeq
    action = Action(agent_id=AgentID("a"), session_id=SessionID("s"), action_type="tool_call")
    assert action.seq is None
    updated = action.model_copy(update={"seq": EventSeq(42)})
    assert updated.seq == 42
    assert action.seq is None  # original unchanged
    assert updated.id == action.id  # same id, different seq
