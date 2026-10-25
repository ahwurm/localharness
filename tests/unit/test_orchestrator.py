"""Tests for Agent Cards and the agent-creation flow."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from localharness.orchestrator.cards import AgentCardRegistry
from localharness.config.models import AgentConfig


# --- AgentCardRegistry ---

def test_registry_generate_card_from_config():
    config = AgentConfig(name="test-agent", role="Research topics using web search and summarize findings")
    registry = AgentCardRegistry()
    card = registry.generate_card(config)
    assert card.agent_id == "test-agent"
    assert card.description == config.role[:200]


# --- AgentCreationFlow.begin_agent_creation ---

def test_begin_agent_creation():
    """AgentCreationFlow.begin_agent_creation() instantiates and returns an AgentCreationWorkflow."""
    from localharness.orchestrator.router import AgentCreationFlow
    from localharness.orchestrator.workflow import AgentCreationWorkflow, WorkflowState
    registry = AgentCardRegistry()
    orch = AgentCreationFlow(card_registry=registry)
    wf = orch.begin_agent_creation()
    assert isinstance(wf, AgentCreationWorkflow)
    assert wf.state == WorkflowState.DISCUSS
    # Calling again returns a fresh workflow (not the same instance)
    wf2 = orch.begin_agent_creation()
    assert wf2 is not wf


def test_begin_agent_creation_forwards_the_loader(tmp_path):
    from localharness.config.loader import ConfigLoader
    from localharness.orchestrator.router import AgentCreationFlow
    from localharness.orchestrator.workflow import WorkflowState
    orch = AgentCreationFlow(card_registry=AgentCardRegistry())
    loader = ConfigLoader(config_dir=tmp_path)
    assert orch.begin_agent_creation(loader=loader)._loader is loader
    assert orch.begin_agent_creation().state == WorkflowState.DISCUSS


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

    # Mock orchestrator with no active workflow
    mock_orchestrator = MagicMock()
    mock_orchestrator.active_workflow = None

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


# --- OrchestratorREPL: no double output during YAML generation ---

@pytest.mark.asyncio
async def test_yaml_generation_no_double_output():
    """YAML gen uses llm.stream_complete() directly (#18) -- no TaskComplete event, send_message called once."""
    from localharness.cli.repl import OrchestratorREPL
    from localharness.core.events import TaskComplete
    from localharness.orchestrator.workflow import AgentCreationWorkflow, WorkflowState

    # Mock agent loop whose _llm.stream_complete mirrors the REAL LLMClient
    # contract: it returns a (message, usage) TUPLE, never a bare message
    # (provider/client.py). Fidelity guarantee: code that reads .content off
    # the tuple instead of the unpacked message sees "" — the exact defect of
    # #19, which a bare-message mock here masked. Do not "simplify" this back
    # to return_value=<message>.
    mock_response = MagicMock()
    mock_response.content = "name: my-agent\nrole: test helper"
    mock_agent = MagicMock()
    mock_agent._config.name = "test-agent"
    mock_agent._llm = AsyncMock()
    mock_agent._llm.stream_complete = AsyncMock(return_value=(mock_response, None))
    mock_agent.run_turn = AsyncMock(return_value="should not be called")

    # Mock bus that captures published events
    mock_bus = AsyncMock()
    published_events = []
    async def capture_publish(event):
        published_events.append(event)
    mock_bus.publish = AsyncMock(side_effect=capture_publish)

    # Mock channel
    mock_channel = AsyncMock()
    # First input triggers DISCUSS->CONFIGURE transition (description > 10 chars)
    # Second input raises EOFError to exit
    mock_channel.read_input = AsyncMock(side_effect=[
        "build me an agent that does web research and summarization",
        EOFError(),
    ])

    # Create real workflow in DISCUSS state with no gathered data
    workflow = AgentCreationWorkflow()

    # Mock orchestrator with active_workflow set to the real workflow
    mock_orchestrator = MagicMock()
    mock_orchestrator.active_workflow = workflow

    repl = OrchestratorREPL(
        orchestrator=mock_orchestrator,
        agent_loop=mock_agent,
        channel=mock_channel,
        bus=mock_bus,
    )
    await repl.run()

    # Assert run_turn was NOT called (llm.stream_complete used instead)
    mock_agent.run_turn.assert_not_called()

    # Assert llm.stream_complete WAS called
    mock_agent._llm.stream_complete.assert_called_once()

    # Assert no TaskComplete events published
    task_completes = [e for e in published_events if isinstance(e, TaskComplete)]
    assert len(task_completes) == 0

    # Assert send_message called exactly once (the YAML display)
    assert mock_channel.send_message.call_count == 1
    call_args = mock_channel.send_message.call_args
    assert "my-agent" in call_args[0][0]  # YAML content in the message
