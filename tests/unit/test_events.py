"""Tests for localharness.core.events — all 21 event models, BudgetSpec, AnyEvent, EVENT_TYPE_MAP."""
import json
import pytest
from localharness.core.events import (
    Action,
    AgentCreated,
    AgentDeleted,
    AnyEvent,
    BaseEvent,
    BudgetSpec,
    CompactionTriggered,
    ComponentMutated,
    DelegationRequest,
    DelegationResult,
    Escalation,
    EVENT_TYPE_MAP,
    Heartbeat,
    MutationArchived,
    Observation,
    ParseFailed,
    ScenarioCompleted,
    StuckRecovered,
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
        ScenarioCompleted, ParseFailed, StuckRecovered,
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
    """EVENT_TYPE_MAP has entries for all 21 event types."""
    assert len(EVENT_TYPE_MAP) == 21
    expected_keys = {
        "SystemReady", "AgentCreated", "AgentDeleted", "TurnStarted", "TurnCompleted",
        "TurnFailed", "UserMessage", "TaskRequest", "TaskComplete", "Action",
        "Observation", "DelegationRequest", "DelegationResult", "Escalation", "Heartbeat",
        "CompactionTriggered", "ScenarioCompleted", "ParseFailed", "StuckRecovered",
        "ComponentMutated", "MutationArchived",
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
    """AnyEvent type contains all 21 event classes."""
    # AnyEvent is a Union; check its __args__
    import typing
    args = typing.get_args(AnyEvent)
    assert len(args) == 21
    expected = {
        SystemReady, AgentCreated, AgentDeleted, TurnStarted, TurnCompleted, TurnFailed,
        UserMessage, TaskRequest, TaskComplete, Action, Observation,
        DelegationRequest, DelegationResult, Escalation, Heartbeat,
        CompactionTriggered, ScenarioCompleted, ParseFailed, StuckRecovered,
        ComponentMutated, MutationArchived,
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


# ---------------------------------------------------------------------------
# Phase 11 / SCEN-02: ScenarioCompleted, ParseFailed, StuckRecovered
# ---------------------------------------------------------------------------

def test_scenario_completed_shape():
    """ScenarioCompleted carries all 11 SCEN-02 fields + 2 optional fields with sane defaults."""
    ev = ScenarioCompleted(
        scenario_name="x",
        model="qwen",
        success=True,
        latency_ttft=0.1,
        latency_total=1.0,
        tokens_in=100,
        tokens_out=50,
        iterations=3,
        parse_failures=0,
        stuck_recoveries=0,
        tool_call_count=2,
    )
    assert ev.event_type == "ScenarioCompleted"
    assert ev.scenario_name == "x"
    assert ev.model == "qwen"
    assert ev.success is True
    assert ev.latency_ttft == 0.1
    assert ev.latency_total == 1.0
    assert ev.tokens_in == 100
    assert ev.tokens_out == 50
    assert ev.iterations == 3
    assert ev.parse_failures == 0
    assert ev.stuck_recoveries == 0
    assert ev.tool_call_count == 2
    # Optional fields with defaults
    assert ev.internal_latencies == {}
    assert ev.tokens_estimated is False


def test_scenario_completed_in_event_type_map():
    """EVENT_TYPE_MAP routes ScenarioCompleted, ParseFailed, StuckRecovered to their classes."""
    assert EVENT_TYPE_MAP["ScenarioCompleted"] is ScenarioCompleted
    assert EVENT_TYPE_MAP["ParseFailed"] is ParseFailed
    assert EVENT_TYPE_MAP["StuckRecovered"] is StuckRecovered


def test_scenario_completed_roundtrip():
    """deserialize_event reconstructs a ScenarioCompleted from its JSON form."""
    ev = ScenarioCompleted(
        scenario_name="round",
        model="m",
        success=False,
        latency_ttft=0.25,
        latency_total=2.5,
        tokens_in=12,
        tokens_out=34,
        iterations=2,
        parse_failures=1,
        stuck_recoveries=1,
        tool_call_count=4,
        internal_latencies={"model_gen": 0.5, "tool_exec": 1.0},
        tokens_estimated=True,
    )
    restored = deserialize_event(ev.model_dump_json())
    assert isinstance(restored, ScenarioCompleted)
    assert restored.scenario_name == "round"
    assert restored.success is False
    assert restored.tokens_in == 12
    assert restored.parse_failures == 1
    assert restored.stuck_recoveries == 1
    assert restored.tool_call_count == 4
    assert restored.internal_latencies == {"model_gen": 0.5, "tool_exec": 1.0}
    assert restored.tokens_estimated is True


def test_parse_failed_shape_and_map():
    """ParseFailed carries agent_id, session_id, iteration, parse_retry_count, raw_content_preview."""
    ev = ParseFailed(
        agent_id=AgentID("a"),
        session_id=SessionID("s"),
        iteration=1,
        parse_retry_count=2,
        raw_content_preview="x",
    )
    assert ev.event_type == "ParseFailed"
    assert ev.agent_id == "a"
    assert ev.session_id == "s"
    assert ev.iteration == 1
    assert ev.parse_retry_count == 2
    assert ev.raw_content_preview == "x"
    # Empty preview is allowed (None or "" fallback at publish site)
    empty = ParseFailed(
        agent_id=AgentID("a"), session_id=SessionID("s"),
        iteration=0, parse_retry_count=1, raw_content_preview="",
    )
    assert empty.raw_content_preview == ""
    # Roundtrip
    restored = deserialize_event(ev.model_dump_json())
    assert isinstance(restored, ParseFailed)
    assert restored.parse_retry_count == 2


def test_stuck_recovered_shape_and_map():
    """StuckRecovered carries agent_id, session_id, iteration, stuck_signature."""
    ev = StuckRecovered(
        agent_id=AgentID("a"),
        session_id=SessionID("s"),
        iteration=3,
        stuck_signature="tool:bash{}",
    )
    assert ev.event_type == "StuckRecovered"
    assert ev.iteration == 3
    assert ev.stuck_signature == "tool:bash{}"
    # Roundtrip
    restored = deserialize_event(ev.model_dump_json())
    assert isinstance(restored, StuckRecovered)
    assert restored.stuck_signature == "tool:bash{}"


# --- Phase 14-02 Task 2: ComponentMutated event ---


def test_component_mutated_constructs_with_all_fields():
    ev = ComponentMutated(
        path="x.y",
        before_value=1,
        after_value=2,
        layer="user",
        actor="cli",
    )
    assert ev.event_type == "ComponentMutated"
    assert ev.path == "x.y"
    assert ev.before_value == 1
    assert ev.after_value == 2
    assert ev.layer == "user"
    assert ev.actor == "cli"
    assert ev.actor_detail is None


def test_component_mutated_frozen():
    ev = ComponentMutated(
        path="a.b", before_value=None, after_value=1, layer="user", actor="cli"
    )
    with pytest.raises(Exception):
        ev.path = "z"


def test_component_mutated_jsonl_round_trip():
    ev = ComponentMutated(
        path="agent.stuck_detector.window_size",
        before_value=5,
        after_value=7,
        layer="experiment",
        actor="experiment",
        actor_detail="exp-42",
    )
    line = ev.model_dump_json()
    restored = deserialize_event(line)
    assert isinstance(restored, ComponentMutated)
    assert restored.path == "agent.stuck_detector.window_size"
    assert restored.before_value == 5
    assert restored.after_value == 7
    assert restored.layer == "experiment"
    assert restored.actor == "experiment"
    assert restored.actor_detail == "exp-42"


def test_component_mutated_in_event_type_map():
    assert EVENT_TYPE_MAP["ComponentMutated"] is ComponentMutated


def test_component_mutated_in_any_event_union():
    from typing import get_args
    assert ComponentMutated in get_args(AnyEvent)


def test_component_mutated_rejects_invalid_layer():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ComponentMutated(
            path="x", before_value=1, after_value=2, layer="bogus", actor="cli"
        )


def test_component_mutated_rejects_invalid_actor():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ComponentMutated(
            path="x", before_value=1, after_value=2, layer="user", actor="bogus"
        )


def test_component_mutated_accepts_arbitrary_actor_detail_string():
    ev = ComponentMutated(
        path="x", before_value=1, after_value=2, layer="user", actor="proposer",
        actor_detail="proposal-abc-123",
    )
    assert ev.actor_detail == "proposal-abc-123"


def test_component_mutated_accepts_primitive_value_types():
    """before/after_value should accept JSON-serializable primitives."""
    for raw in (1, 0.75, "abc", True, [1, 2, 3], {"k": "v"}, None):
        ev = ComponentMutated(
            path="x", before_value=raw, after_value=raw, layer="user", actor="cli"
        )
        # Round trip must preserve value
        restored = deserialize_event(ev.model_dump_json())
        assert restored.before_value == raw
        assert restored.after_value == raw
