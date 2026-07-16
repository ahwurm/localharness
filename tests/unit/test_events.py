"""Tests for localharness.core.events — all 26 event models, BudgetSpec, AnyEvent, EVENT_TYPE_MAP."""
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
    ExpectationAttached,
    Heartbeat,
    MemoryGateFired,
    MutationArchived,
    Observation,
    OutcomeObserved,
    ParseFailed,
    ScenarioCompleted,
    SentinelAlert,
    StuckRecovered,
    SurpriseScored,
    SystemReady,
    TaskComplete,
    TaskRequest,
    TurnCompleted,
    TurnFailed,
    TurnStarted,
    UserMessage,
    deserialize_event,
)
from localharness.core.types import AgentID, DivisionID, OrgID, SessionID, ToolCallID


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
    """EVENT_TYPE_MAP has entries for all 29 event types."""
    assert len(EVENT_TYPE_MAP) == 29
    expected_keys = {
        "SystemReady", "AgentCreated", "AgentDeleted", "TurnStarted", "TurnCompleted",
        "TurnFailed", "UserMessage", "TaskRequest", "TaskComplete", "Action",
        "Observation", "DelegationRequest", "DelegationResult", "Escalation", "Heartbeat",
        "CompactionTriggered", "ScenarioCompleted", "ParseFailed", "StuckRecovered",
        "ComponentMutated", "MutationArchived", "SentinelAlert", "MemoryGateFired",
        "ExpectationAttached", "OutcomeObserved", "SurpriseScored",
        "ConsolidationStarted", "ConsolidationFinished", "InputRouted",
    }
    assert set(EVENT_TYPE_MAP.keys()) == expected_keys


def test_input_routed_event():
    """InputRouted records every type-anytime input-box routing decision (nudge|queue),
    which tier decided, the rule/reason, and a short preview — dogfood tuning data on the
    session ledger. Survives the bus-ledger JSONL roundtrip."""
    from localharness.core.events import InputRouted

    ev = InputRouted(
        agent_id=AgentID("a"), session_id=SessionID("s"),
        decision="nudge", tier="tier1", rule_or_reason="nudge-initial:stop",
        text_preview="stop, wrong file",
    )
    assert ev.event_type == "InputRouted"
    assert ev.decision == "nudge" and ev.tier == "tier1"
    assert EVENT_TYPE_MAP["InputRouted"] is InputRouted
    restored = deserialize_event(ev.model_dump_json())
    assert isinstance(restored, InputRouted)
    assert restored.decision == "nudge" and restored.rule_or_reason == "nudge-initial:stop"


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


def test_action_has_tool_calls_field():
    """Action carries has_tool_calls (additive, default False): True marks an interstitial
    llm_response whose content is narration (tool calls follow); False marks a final answer.
    Default False keeps every existing Action construction + old JSONL line valid."""
    default = Action(agent_id=AgentID("a"), session_id=SessionID("s"), action_type="tool_call")
    assert default.has_tool_calls is False  # additive default — old constructions unaffected

    narration = Action(
        agent_id=AgentID("a"), session_id=SessionID("s"),
        action_type="llm_response", content="Pulling the data…", has_tool_calls=True,
    )
    assert narration.has_tool_calls is True
    # survives the bus-ledger roundtrip (bus-events.jsonl deserialize path)
    restored = deserialize_event(narration.model_dump_json())
    assert isinstance(restored, Action)
    assert restored.has_tool_calls is True
    assert restored.content == "Pulling the data…"

    # old JSONL lines predate the field — deserialize must fall back to False, not error
    line = default.model_dump_json().replace(', "has_tool_calls":false', "")
    legacy = deserialize_event(line)
    assert legacy.has_tool_calls is False


def test_budget_spec_frozen():
    """BudgetSpec(max_actions=50) is immutable (raises on field assignment)."""
    spec = BudgetSpec(max_actions=50)
    assert spec.max_actions == 50
    with pytest.raises(Exception):  # ValidationError or TypeError from frozen model
        spec.max_actions = 100  # type: ignore[misc]


def test_any_event_union():
    """AnyEvent type contains all 28 event classes."""
    # AnyEvent is a Union; check its __args__
    import typing
    # Import-inside-body (the file's 19-02 idiom) so the module still collects before the
    # #20 consolidation-status events land — the assertions below fail RED until then.
    from localharness.core.events import (
        ConsolidationFinished, ConsolidationStarted, InputRouted,
    )
    args = typing.get_args(AnyEvent)
    assert len(args) == 29
    expected = {
        SystemReady, AgentCreated, AgentDeleted, TurnStarted, TurnCompleted, TurnFailed,
        UserMessage, TaskRequest, TaskComplete, Action, Observation,
        DelegationRequest, DelegationResult, Escalation, Heartbeat,
        CompactionTriggered, ScenarioCompleted, ParseFailed, StuckRecovered,
        ComponentMutated, MutationArchived, SentinelAlert, MemoryGateFired,
        ExpectationAttached, OutcomeObserved, SurpriseScored,
        ConsolidationStarted, ConsolidationFinished, InputRouted,
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


# --- Phase 19 Wave-0: SentinelAlert event (the class lands in 19-02) ---
# SentinelAlert is imported INSIDE each test body so this module still collects before 19-02
# ships the class (the 17-01 import-inside-body idiom); the xfail(strict=False) absorbs the
# ImportError until then, then flips to a pass. Mirrors the 5-point event contract above.


def test_sentinel_alert_roundtrip():
    """SentinelAlert constructs, is frozen, and round-trips through deserialize_event (incl. fixtures list)."""
    from localharness.core.events import SentinelAlert, deserialize_event

    ev = SentinelAlert(
        kind="overfit", detail="gap 0.2 > 0.1", mutation_id="abc",
        metric_value=0.2, threshold=0.1,
    )
    assert ev.event_type == "SentinelAlert"

    # frozen: mutating a field raises (the BaseEvent immutability contract)
    with pytest.raises(Exception):
        ev.kind = "x"

    restored = deserialize_event(ev.model_dump_json())
    assert isinstance(restored, SentinelAlert)
    assert restored.kind == "overfit"
    assert restored.metric_value == 0.2

    # a saturation alert carries its saturated-fixtures list through the round-trip
    sat = SentinelAlert(kind="saturation", detail="fx_a saturated", fixtures=["fx_a", "fx_b"])
    sat_restored = deserialize_event(sat.model_dump_json())
    assert sat_restored.fixtures == ["fx_a", "fx_b"]


async def test_sentinel_alert_delivered_to_subscriber(bus):
    """Publishing a SentinelAlert delivers it to a bus subscriber (fire-and-forget emit contract)."""
    from localharness.core.events import SentinelAlert

    received = []

    async def _handler(event):
        received.append(event)

    bus.subscribe(SentinelAlert, _handler)
    await bus.publish(SentinelAlert(kind="near_duplicate", detail="3 dupes"))
    assert len(received) == 1
    assert received[0].kind == "near_duplicate"


# ---------------------------------------------------------------------------
# Phase 34 / COLL-04: ExpectationAttached, OutcomeObserved, SurpriseScored
# The predictive-gate collect-only bus contract. These MUST round-trip through
# deserialize_event AND replay from JSONL — a forgotten EVENT_TYPE_MAP entry
# makes the live system work but leaves a backfill silently empty (Pitfall 2).
# ---------------------------------------------------------------------------


def test_predictive_events_roundtrip():
    """Each COLL-04 event constructs, serializes, and deserialize_event restores the
    exact class with field equality (tool_call_id, score, quadrant, prior fields)."""
    exp = ExpectationAttached(
        agent_id=AgentID("a"), session_id=SessionID("s"),
        tool_call_id=ToolCallID("tc-1"), tool_name="bash",
        prior_n=7, prior_error_rate=0.1, lat_mean_ms=12.5, lat_var_ms=3.0,
        size_mean=100.0, size_var=9.0,
    )
    r = deserialize_event(exp.model_dump_json())
    assert isinstance(r, ExpectationAttached)
    assert r.tool_call_id == "tc-1"
    assert r.tool_name == "bash"
    assert r.source == "l1_priors"  # default
    assert r.prior_n == 7
    assert r.prior_error_rate == 0.1
    assert r.lat_mean_ms == 12.5

    out = OutcomeObserved(
        agent_id=AgentID("a"), session_id=SessionID("s"),
        tool_call_id=ToolCallID("tc-1"), tool_name="bash",
        is_error=True, output_len=42, duration_ms=15,
    )
    ro = deserialize_event(out.model_dump_json())
    assert isinstance(ro, OutcomeObserved)
    assert ro.tool_call_id == "tc-1"
    assert ro.is_error is True
    assert ro.output_len == 42
    assert ro.duration_ms == 15

    sur = SurpriseScored(
        agent_id=AgentID("a"), session_id=SessionID("s"),
        tool_call_id=ToolCallID("tc-1"), tool_name="bash",
        score=0.87, quadrant="surprising_failure",
        error_surprisal=1.2, z_latency=2.0, z_size=0.5,
    )
    rs = deserialize_event(sur.model_dump_json())
    assert isinstance(rs, SurpriseScored)
    assert rs.tool_call_id == "tc-1"
    assert rs.score == 0.87
    assert rs.quadrant == "surprising_failure"
    assert rs.error_surprisal == 1.2
    assert rs.z_latency == 2.0
    assert rs.z_size == 0.5


async def test_predictive_events_replayable(tmp_path):
    """A serialized SurpriseScored line replays through EventBus.replay() — the Pitfall-2
    regression guard at the REPLAY layer (a forgotten EVENT_TYPE_MAP entry = empty backfill)."""
    from localharness.core.bus import EventBus

    jsonl = tmp_path / "events.jsonl"
    ev = SurpriseScored(
        agent_id=AgentID("a"), session_id=SessionID("s"),
        tool_call_id=ToolCallID("tc-1"), tool_name="bash",
        score=0.5, quadrant="routine",
    )
    jsonl.write_text(ev.model_dump_json() + "\n", encoding="utf-8")

    bus = EventBus(persist_path=jsonl)
    replayed = [e async for e in bus.replay()]
    assert len(replayed) == 1
    assert isinstance(replayed[0], SurpriseScored)
    assert replayed[0].score == 0.5
    assert replayed[0].quadrant == "routine"


# ---------------------------------------------------------------------------
# #20: ConsolidationStarted / ConsolidationFinished — the dreaming-dot lifecycle
# signal. Published on the persisted bus, so (Pitfall 2) they MUST round-trip through
# deserialize_event and be registered in EVENT_TYPE_MAP / AnyEvent.
# ---------------------------------------------------------------------------


def test_consolidation_status_events_roundtrip_and_registered():
    """Both events construct, are frozen, carry event_type == class name, live in
    EVENT_TYPE_MAP + AnyEvent, and reconstruct exactly via deserialize_event."""
    from typing import get_args

    from localharness.core.events import ConsolidationFinished, ConsolidationStarted

    for cls in (ConsolidationStarted, ConsolidationFinished):
        ev = cls(agent_id=AgentID("cons-agent"))
        assert ev.event_type == cls.__name__
        with pytest.raises(Exception):        # frozen: BaseEvent immutability contract
            ev.agent_id = "other"
        assert EVENT_TYPE_MAP[cls.__name__] is cls
        assert cls in get_args(AnyEvent)
        restored = deserialize_event(ev.model_dump_json())
        assert isinstance(restored, cls)
        assert restored.agent_id == "cons-agent"
        assert restored.id == ev.id


async def test_consolidation_status_events_deliver_to_subscriber(bus):
    """Publishing each event reaches a bus subscriber (the delivery seam the terminal
    channel rides for the dreaming dot)."""
    from localharness.core.events import ConsolidationFinished, ConsolidationStarted

    seen = []
    bus.subscribe(ConsolidationStarted, lambda e: seen.append(("start", e)))
    bus.subscribe(ConsolidationFinished, lambda e: seen.append(("end", e)))
    await bus.publish(ConsolidationStarted(agent_id=AgentID("a")))
    await bus.publish(ConsolidationFinished(agent_id=AgentID("a")))
    assert [k for k, _ in seen] == ["start", "end"]
