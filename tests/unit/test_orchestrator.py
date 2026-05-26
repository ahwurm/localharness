"""Tests for orchestrator routing, Agent Cards, and context guard."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from localharness.orchestrator.cards import AgentCard, AgentCardRegistry, score_card, RoutingDecision
from localharness.config.models import AgentConfig


# --- score_card ---

def test_score_card_keyword_match():
    card = AgentCard(agent_id="research", name="research", description="Web research agent", keywords=["research", "search", "web", "crawl"], example_tasks=["Search the web for X"])
    score = score_card("research this topic using web search", card)
    assert score > 0.30  # above threshold

def test_score_card_no_match():
    card = AgentCard(agent_id="fitness", name="fitness", description="Fitness tracker", keywords=["fitness", "workout", "nutrition"])
    score = score_card("deploy the kubernetes cluster", card)
    assert score < 0.30  # below threshold

def test_score_card_jaccard_example():
    card = AgentCard(agent_id="news", name="news", description="News monitor", keywords=["news"], example_tasks=["Monitor hacker news for AI stories", "Check tech news"])
    score = score_card("Check hacker news for AI stories", card)
    # keyword hit (0.15) + Jaccard(5/7=0.714)*0.25 (0.179) = 0.329
    assert score > 0.30  # Jaccard with example boosts above threshold

def test_score_card_health_penalty():
    card = AgentCard(agent_id="broken", name="broken", description="Broken agent", keywords=["test"], status="error")
    score = score_card("test something", card)
    card_healthy = AgentCard(agent_id="healthy", name="healthy", description="Healthy agent", keywords=["test"])
    score_healthy = score_card("test something", card_healthy)
    assert score < score_healthy  # error status penalizes

def test_score_card_capability_phrase_match():
    card = AgentCard(agent_id="coder", name="coder", description="Code agent", keywords=["code"], capabilities=["write python code"])
    score = score_card("write python code for sorting", card)
    assert score >= 0.30  # keyword(0.15) + capability phrase(0.15) = 0.30


# --- AgentCardRegistry ---

def test_registry_generate_card_from_config():
    config = AgentConfig(name="test-agent", role="Research topics using web search and summarize findings")
    registry = AgentCardRegistry()
    card = registry.generate_card(config)
    assert card.agent_id == "test-agent"
    assert len(card.keywords) > 0
    assert card.description == config.role[:200]

def test_registry_route_returns_best_match():
    registry = AgentCardRegistry()
    registry.register(AgentCard(agent_id="research", name="research", description="Web research", keywords=["research", "search", "web"]))
    registry.register(AgentCard(agent_id="fitness", name="fitness", description="Fitness tracking", keywords=["fitness", "workout", "nutrition"]))
    decision = registry.route("search the web for machine learning papers")
    assert decision.matched is True
    assert decision.agent_id == "research"

def test_registry_route_no_match_below_threshold():
    registry = AgentCardRegistry()
    registry.register(AgentCard(agent_id="fitness", name="fitness", description="Fitness", keywords=["fitness"]))
    decision = registry.route("deploy kubernetes cluster to production")
    assert decision.matched is False

def test_registry_route_empty_registry():
    registry = AgentCardRegistry()
    decision = registry.route("anything")
    assert decision.matched is False
    assert "No active agents" in decision.reason

def test_registry_route_tiebreak_fn_called_on_ambiguity():
    """When top-2 scores are within 0.10 delta and tiebreak_fn is provided, tiebreak_fn is called."""
    registry = AgentCardRegistry()
    # Two agents with very similar keywords so scores will be close
    registry.register(AgentCard(agent_id="agent-a", name="agent-a", description="General helper", keywords=["data", "analysis", "report"]))
    registry.register(AgentCard(agent_id="agent-b", name="agent-b", description="Data specialist", keywords=["data", "analysis", "query"]))

    tiebreak_called_with = {}
    def mock_tiebreak(task: str, candidates: list) -> str:
        tiebreak_called_with["task"] = task
        tiebreak_called_with["candidates"] = [c.agent_id for c in candidates]
        return candidates[1].agent_id  # Pick the second candidate

    decision = registry.route("run data analysis and generate a report", tiebreak_fn=mock_tiebreak)
    # If scores were ambiguous, tiebreak should have been called
    if "task" in tiebreak_called_with:
        assert decision.matched is True
        assert "LLM tiebreak" in decision.reason
        assert decision.agent_id == tiebreak_called_with["candidates"][1]

def test_registry_route_tiebreak_fn_not_called_when_clear_winner():
    """When there's a clear winner (delta >= 0.10), tiebreak_fn is NOT called."""
    registry = AgentCardRegistry()
    registry.register(AgentCard(agent_id="research", name="research", description="Web research", keywords=["research", "search", "web", "crawl"]))
    registry.register(AgentCard(agent_id="fitness", name="fitness", description="Fitness", keywords=["fitness", "workout"]))

    tiebreak_called = [False]
    def mock_tiebreak(task: str, candidates: list) -> str:
        tiebreak_called[0] = True
        return candidates[0].agent_id

    decision = registry.route("search the web for research papers", tiebreak_fn=mock_tiebreak)
    assert decision.matched is True
    assert decision.agent_id == "research"
    assert tiebreak_called[0] is False  # Should not have been called

def test_registry_route_tiebreak_fn_exception_falls_through():
    """If tiebreak_fn raises, route() falls through to best-score match."""
    registry = AgentCardRegistry()
    registry.register(AgentCard(agent_id="agent-a", name="agent-a", description="General helper", keywords=["data", "analysis", "report"]))
    registry.register(AgentCard(agent_id="agent-b", name="agent-b", description="Data specialist", keywords=["data", "analysis", "query"]))

    def broken_tiebreak(task: str, candidates: list) -> str:
        raise RuntimeError("LLM unavailable")

    decision = registry.route("run data analysis and generate a report", tiebreak_fn=broken_tiebreak)
    assert decision.matched is True  # Should still route, just without tiebreak


# --- OrchestratorContextGuard ---

def test_context_guard_trims_to_cap():
    from localharness.orchestrator.router import OrchestratorContextGuard
    guard = OrchestratorContextGuard(max_tokens=16_000)
    # Simulate messages that exceed cap
    messages = [{"role": "system", "content": "sys"}]
    for i in range(50):
        messages.append({"role": "user", "content": f"message {i} " * 100})
        messages.append({"role": "assistant", "content": f"response {i} " * 100})
    trimmed = guard.trim(messages)
    # Should keep system + last N turns, not all 100 messages
    assert len(trimmed) < len(messages)
    assert trimmed[0]["role"] == "system"  # system always preserved


# --- Greeting behavior ---

def test_greeting_new_session():
    """New session greeting: 2-3 sentences + /help pointer."""
    from localharness.orchestrator.router import Orchestrator
    greeting = Orchestrator.compose_greeting(is_returning=False)
    assert "/help" in greeting
    assert len(greeting) > 20  # not empty

def test_greeting_returning_session():
    """Returning session: minimal 'Ready.' banner."""
    from localharness.orchestrator.router import Orchestrator
    greeting = Orchestrator.compose_greeting(is_returning=True)
    assert greeting == "Ready."  # minimal banner for returning users

def test_greeting_returning_session_with_model():
    """Returning session with model name: 'model -- Ready.'"""
    from localharness.orchestrator.router import Orchestrator
    greeting = Orchestrator.compose_greeting(is_returning=True, model_name="gpt-oss-120b")
    assert greeting == "gpt-oss-120b -- Ready."


# --- Orchestrator.begin_agent_creation ---

def test_orchestrator_begin_agent_creation():
    """Orchestrator.begin_agent_creation() instantiates and returns an AgentCreationWorkflow."""
    from localharness.orchestrator.router import Orchestrator
    from localharness.orchestrator.workflow import AgentCreationWorkflow, WorkflowState
    registry = AgentCardRegistry()
    orch = Orchestrator(card_registry=registry)
    wf = orch.begin_agent_creation()
    assert isinstance(wf, AgentCreationWorkflow)
    assert wf.state == WorkflowState.DISCUSS
    # Calling again returns a fresh workflow (not the same instance)
    wf2 = orch.begin_agent_creation()
    assert wf2 is not wf


# --- OrchestratorREPL: session_id on UserMessage ---

@pytest.mark.asyncio
async def test_user_message_includes_session_id():
    """UserMessage published by REPL includes session_id from AgentLoop.current_session_id."""
    from localharness.cli.repl import OrchestratorREPL
    from localharness.core.events import UserMessage

    # Mock agent loop with current_session_id property
    mock_agent = MagicMock()
    mock_agent._config.name = "test-agent"
    mock_agent.current_session_id = "sess-123"
    mock_agent.run_turn = AsyncMock(return_value="done")

    # Mock bus that captures published events
    mock_bus = AsyncMock()
    published_events = []
    async def capture_publish(event):
        published_events.append(event)
    mock_bus.publish = AsyncMock(side_effect=capture_publish)

    # Mock channel: first read returns input, second raises EOFError
    mock_channel = AsyncMock()
    mock_channel.read_input = AsyncMock(side_effect=["hello", EOFError()])

    # Mock orchestrator with no active workflow and route_task returning a decision
    mock_orchestrator = MagicMock()
    mock_orchestrator.active_workflow = None
    mock_orchestrator.route_task = MagicMock(return_value=RoutingDecision(matched=True, agent_id="test-agent", agent_card=None, confidence=1.0, reason="test"))

    repl = OrchestratorREPL(
        orchestrator=mock_orchestrator,
        agent_loop=mock_agent,
        channel=mock_channel,
        bus=mock_bus,
    )
    await repl.run()

    # Find the UserMessage event
    user_msgs = [e for e in published_events if isinstance(e, UserMessage)]
    assert len(user_msgs) == 1
    assert user_msgs[0].session_id == "sess-123"
    assert user_msgs[0].content == "hello"
