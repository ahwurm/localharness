"""#19 repro: REPL agent-creation must actually write a config.

These tests drive the REAL user path — OrchestratorREPL.run() with a real
AgentCreationFlow and a real AgentCreationWorkflow — faking only the process
boundaries (scripted terminal, tuple-shaped LLM client, event bus). The LLM
fake is conftest's MockLLMClient, whose stream_complete returns the REAL
LLMClient contract: a (message, usage) TUPLE (provider/client.py), never a
bare message. #19 was masked precisely by a bare-message mock — do not lower
the fidelity of these fakes.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import yaml

from localharness.cli.repl import OrchestratorREPL
from localharness.orchestrator.cards import AgentCardRegistry
from localharness.orchestrator.router import AgentCreationFlow
from localharness.orchestrator.workflow import WorkflowState

ROLE = "Track stock prices and summarize the daily moves"


class ScriptedChannel:
    """Terminal stand-in: yields scripted user inputs, records sent messages."""

    channel_id = "terminal"

    def __init__(self, inputs: list[str]) -> None:
        self._inputs = list(inputs)
        self.sent: list[str] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def read_input(self) -> str:
        if not self._inputs:
            raise EOFError()
        return self._inputs.pop(0)

    async def send_message(self, text: str, metadata: dict | None = None) -> None:
        self.sent.append(text)


def _repl(channel: ScriptedChannel, llm, config_dir=None):
    orch = AgentCreationFlow(AgentCardRegistry())
    agent = MagicMock()
    agent._config.name = "orchestrator"
    agent.current_session_id = "sess-1"
    agent._llm = llm
    agent.run_turn = AsyncMock()
    bus = AsyncMock()
    repl = OrchestratorREPL(
        orchestrator=orch, agent_loop=agent, channel=channel, bus=bus,
        config_dir=config_dir,
    )
    return repl, orch, agent, bus


def test_agent_creation_e2e_writes_real_config(tmp_path, mock_llm_client):
    """Trigger -> description -> yes must end with a validated YAML on disk.

    Asserts on the REAL file the real deploy path writes — the whole point of
    #19 is that nothing short of this catches the break.
    """
    llm = mock_llm_client([mock_llm_client.Response(
        content=f"```yaml\nname: finance-helper\nrole: {ROLE}\n```",
    )])
    channel = ScriptedChannel([
        "create an agent",
        f"it should {ROLE[0].lower()}{ROLE[1:]}",
        "yes",
    ])
    repl, orch, agent, bus = _repl(channel, llm, config_dir=tmp_path)

    asyncio.run(repl.run())

    # Deploy honors the CONFIRMED YAML's own name (no step gathers a name, and
    # the old 'new_agent' fallback failed AgentConfig's hyphens-only rule).
    config_path = tmp_path / "agents" / "finance-helper.yaml"
    assert config_path.exists(), (
        f"no agent config written; messages sent: {channel.sent}"
    )
    data = yaml.safe_load(config_path.read_text())
    assert data["name"] == "finance-helper"  # what the user confirmed
    assert data["role"] == ROLE  # model's YAML survived, fences stripped
    # The confirm prompt showed the ACTUAL generated YAML, not an empty block
    assert any("finance-helper" in m for m in channel.sent)
    assert any("Agent deployed to" in m for m in channel.sent)
    assert orch.active_workflow is None  # workflow completed and cleared
    agent.run_turn.assert_not_called()  # generation bypasses the agent loop
    bus.publish.assert_not_called()  # workflow traffic stays out of memory


def test_configure_reads_message_from_tuple(mock_llm_client):
    """Defect A: stream_complete returns (message, usage); the flow must read
    .content off the unpacked message — reading it off the tuple yields ""."""
    plain_yaml = f"name: finance-helper\nrole: {ROLE}"
    llm = mock_llm_client([mock_llm_client.Response(content=plain_yaml)])
    channel = ScriptedChannel([f"it should {ROLE[0].lower()}{ROLE[1:]}"])
    repl, orch, agent, bus = _repl(channel, llm)
    workflow = orch.begin_agent_creation()  # already in DISCUSS, nothing gathered

    asyncio.run(repl.run())

    assert workflow.generated_yaml == plain_yaml
    assert workflow.state == WorkflowState.CONFIRM
    assert any(plain_yaml in m for m in channel.sent)


def test_trigger_message_not_consumed_as_description(mock_llm_client):
    """Defect B: the trigger message must NOT be fed into the workflow — it was
    stored as the agent DESCRIPTION and silently advanced DISCUSS->CONFIGURE
    (return value discarded), so the CONFIGURE branch that generates YAML never
    ran. The description is the user's NEXT message."""
    llm = mock_llm_client([])
    channel = ScriptedChannel(["i need an agent for my finance stuff"])
    repl, orch, agent, bus = _repl(channel, llm)

    asyncio.run(repl.run())

    workflow = orch.active_workflow
    assert workflow is not None
    assert workflow.state == WorkflowState.DISCUSS
    assert "description" not in workflow.gathered
    assert any("I'd like to help you create an agent" in m for m in channel.sent)


def test_generation_prompt_states_contract():
    """#33: the generation system prompt stated NO schema, so the model guessed
    the shape (agent: nesting, description-not-role, invented keys) and every
    deploy died at AgentConfig validation. The prompt must state the real
    contract — required fields + the allowed top-level keys — DERIVED from
    AgentConfig.model_fields so prompt and schema can't drift."""
    from localharness.cli.repl import _generation_system_prompt
    from localharness.config.models import AgentConfig

    prompt = _generation_system_prompt()
    fields = AgentConfig.model_fields
    required = [n for n, f in fields.items() if f.is_required()]

    assert required == ["name", "role"]  # the only no-default fields
    for r in required:  # required fields are named as required
        assert r in prompt
    for key in fields:  # every allowed top-level key is listed (derived, not typed)
        assert key in prompt
    # A concrete minimal YAML example (name + role) is shown, no agent: nesting.
    assert "name:" in prompt and "role:" in prompt
    assert "agent:" not in prompt
