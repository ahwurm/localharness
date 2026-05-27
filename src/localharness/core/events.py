"""All 18 LocalHarness event models, BudgetSpec, AnyEvent union, EVENT_TYPE_MAP, deserialize_event.

event_type field values are PascalCase matching the Python class name — required for bubus routing
(bubus routes by class.__name__; lowercase Literal values break routing silently).

Events are immutable (frozen=True). Use model_copy(update={...}) to create modified instances.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .types import AgentID, DivisionID, EventSeq, OrgID, SessionID, ToolCallID  # noqa: F401


class BaseEvent(BaseModel):
    """Base class for all LocalHarness events. Immutable — use model_copy() to set seq."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    seq: Optional[EventSeq] = Field(default=None)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_id: Optional[AgentID] = Field(default=None)
    session_id: Optional[SessionID] = Field(default=None)
    parent_id: Optional[str] = Field(default=None)


class SystemReady(BaseEvent):
    """Published once when the harness is fully initialized."""

    event_type: str = "SystemReady"
    config_path: str
    provider_base_url: str
    provider_type: Literal["ollama", "vllm", "llama_cpp", "lm_studio", "unknown"]
    detected_models: list[str]


class AgentCreated(BaseEvent):
    """Published when a new agent config is written to disk."""

    event_type: str = "AgentCreated"
    agent_id: AgentID
    config_path: str
    division_id: Optional[DivisionID] = None


class AgentDeleted(BaseEvent):
    """Published when an agent config is removed."""

    event_type: str = "AgentDeleted"
    agent_id: AgentID
    config_path: str


class BudgetSpec(BaseModel):
    """Budget constraints for one agent execution session."""

    model_config = ConfigDict(frozen=True)

    max_actions: int = 100
    max_duration_minutes: float = 30.0
    max_context_tokens: int = 128_000
    kill_file_path: Optional[str] = None


class TurnStarted(BaseEvent):
    """Published at the beginning of each run_turn() call."""

    event_type: str = "TurnStarted"
    agent_id: AgentID
    session_id: SessionID
    task_summary: str
    budget: BudgetSpec


class TurnCompleted(BaseEvent):
    """Published when run_turn() exits normally."""

    event_type: str = "TurnCompleted"
    agent_id: AgentID
    session_id: SessionID
    iterations: int
    duration_seconds: float
    elapsed_tokens: int
    summary: str
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_estimated: bool = False


class TurnFailed(BaseEvent):
    """Published when run_turn() exits due to an error or limit."""

    event_type: str = "TurnFailed"
    agent_id: AgentID
    session_id: SessionID
    reason: Literal["budget_exceeded", "stuck_detected", "kill_file", "llm_error", "internal_error"]
    detail: str
    iterations: int
    duration_seconds: float


class UserMessage(BaseEvent):
    """Published by a channel adapter when the user sends input."""

    event_type: str = "UserMessage"
    content: str
    channel: str
    attachments: list[str] = Field(default_factory=list)


class TaskRequest(BaseEvent):
    """Published by the orchestrator to dispatch a task to a specific agent."""

    event_type: str = "TaskRequest"
    agent_id: AgentID
    session_id: SessionID
    task: str
    budget: BudgetSpec


class TaskComplete(BaseEvent):
    """Published by the agent loop when the task finishes."""

    event_type: str = "TaskComplete"
    agent_id: AgentID
    session_id: SessionID
    success: bool
    summary: str
    duration_seconds: float
    iterations: int


class Action(BaseEvent):
    """Represents agent intent: something the agent wants to do."""

    event_type: str = "Action"
    agent_id: AgentID
    session_id: SessionID
    action_type: str
    content: Optional[str] = None
    tool_call_id: Optional[ToolCallID] = None
    tool_name: Optional[str] = None
    tool_params: Optional[dict[str, Any]] = None
    risk_level: Optional[Literal["low", "medium", "high"]] = None
    signature: Optional[str] = None


class Observation(BaseEvent):
    """Represents the result of executing an action."""

    event_type: str = "Observation"
    agent_id: AgentID
    session_id: SessionID
    observation_type: str
    tool_call_id: Optional[ToolCallID] = None
    tool_name: Optional[str] = None
    output: Optional[str] = None
    truncated: bool = False
    error: Optional[str] = None
    exit_code: Optional[int] = None


class DelegationRequest(BaseEvent):
    """Published by the orchestrator to delegate a sub-task to another agent."""

    event_type: str = "DelegationRequest"
    requesting_agent_id: AgentID
    target_agent_id: AgentID
    session_id: SessionID
    task_file: str
    budget: BudgetSpec


class DelegationResult(BaseEvent):
    """Published by the delegated agent when it completes its sub-task."""

    event_type: str = "DelegationResult"
    requesting_agent_id: AgentID
    target_agent_id: AgentID
    session_id: SessionID
    success: bool
    summary: str


class Escalation(BaseEvent):
    """Published when an agent cannot proceed and requires orchestrator intervention."""

    event_type: str = "Escalation"
    agent_id: AgentID
    session_id: SessionID
    reason: str
    detail: str
    stuck_signature: Optional[str] = None
    iteration_at_escalation: int


class Heartbeat(BaseEvent):
    """Published periodically by each running agent."""

    event_type: str = "Heartbeat"
    agent_id: AgentID
    session_id: SessionID
    iteration: int
    context_utilization_pct: float
    last_tool: Optional[str] = None


class ScenarioCompleted(BaseEvent):
    """Published by the bench runner when a scenario run finishes (success or fail).

    Carries the standardized SCEN-02 metric payload aggregated from event subscriptions
    across one scenario run. Latency semantics:
      - latency_ttft: prompt-submit -> first user-visible token (user-friction, regression-gated).
        NOTE: when provider streaming is non-functional (current native mode), this equals
        latency_total -- see Phase 11 known limitation.
      - latency_total: prompt-submit -> final response (user-friction, regression-gated).
      - internal_latencies: dict of internal latency buckets (model_gen, tool_exec, mcp), info-only.
    """

    event_type: str = "ScenarioCompleted"
    scenario_name: str
    model: str
    success: bool
    latency_ttft: float
    latency_total: float
    tokens_in: int
    tokens_out: int
    iterations: int
    parse_failures: int
    stuck_recoveries: int
    tool_call_count: int
    internal_latencies: dict[str, float] = Field(default_factory=dict)
    tokens_estimated: bool = False


class ParseFailed(BaseEvent):
    """Published by the agent loop when a tool-call parse attempt fails and parse_retries is bumped.

    SCEN-02 derives `parse_failures` per scenario by counting these events for the run's session.
    """

    event_type: str = "ParseFailed"
    agent_id: AgentID
    session_id: SessionID
    iteration: int
    parse_retry_count: int  # session.parse_retries value AT emission (after increment)
    raw_content_preview: str  # first 200 chars of the unparseable assistant content


class StuckRecovered(BaseEvent):
    """Published by the agent loop on every StuckState.RECOVERING transition.

    SCEN-02 derives `stuck_recoveries` per scenario by counting these events. Note:
    emitted once per RECOVERING decision (not once per re-entry into the state); the
    loop calls stuck_detector.check() once per iteration so this naturally throttles.
    """

    event_type: str = "StuckRecovered"
    agent_id: AgentID
    session_id: SessionID
    iteration: int
    stuck_signature: str  # from stuck_detector.most_repeated_signature()


AnyEvent = Union[
    SystemReady,
    AgentCreated,
    AgentDeleted,
    TurnStarted,
    TurnCompleted,
    TurnFailed,
    UserMessage,
    TaskRequest,
    TaskComplete,
    Action,
    Observation,
    DelegationRequest,
    DelegationResult,
    Escalation,
    Heartbeat,
    ScenarioCompleted,
    ParseFailed,
    StuckRecovered,
]

EVENT_TYPE_MAP: dict[str, type[BaseEvent]] = {
    "SystemReady": SystemReady,
    "AgentCreated": AgentCreated,
    "AgentDeleted": AgentDeleted,
    "TurnStarted": TurnStarted,
    "TurnCompleted": TurnCompleted,
    "TurnFailed": TurnFailed,
    "UserMessage": UserMessage,
    "TaskRequest": TaskRequest,
    "TaskComplete": TaskComplete,
    "Action": Action,
    "Observation": Observation,
    "DelegationRequest": DelegationRequest,
    "DelegationResult": DelegationResult,
    "Escalation": Escalation,
    "Heartbeat": Heartbeat,
    "ScenarioCompleted": ScenarioCompleted,
    "ParseFailed": ParseFailed,
    "StuckRecovered": StuckRecovered,
}


def deserialize_event(line: str) -> AnyEvent:
    """Deserialize one JSONL line into the correct event type."""
    data = json.loads(line)
    event_type = data.get("event_type")
    model_class = EVENT_TYPE_MAP.get(event_type)
    if model_class is None:
        raise ValueError(f"Unknown event_type: {event_type!r}")
    return model_class.model_validate(data)
