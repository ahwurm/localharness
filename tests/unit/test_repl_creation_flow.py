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


def test_generation_prompt_demonstrates_nested_tool_permission_shapes():
    """#33 follow-up (live 0-for-4 post-fix): with the top-level contract stated,
    the model got name/role right but emitted BARE shapes (permissions: read,
    tools: []) whenever the user asked to restrict tools — the prompt names
    tools/permissions as allowed keys but never demonstrates their nested
    object shape. The prompt must derive the sub-model keys (ToolConfig /
    PermissionConfig, so it can't drift) and show a worked nested tools block,
    so a "read-only tools" ask lands as tools.deny — not a bare string/list."""
    from localharness.cli.repl import _generation_system_prompt
    from localharness.config.models import PermissionConfig, ToolConfig

    prompt = _generation_system_prompt()
    for key in ToolConfig.model_fields:  # nested tools keys stated (derived)
        assert key in prompt
    for key in PermissionConfig.model_fields:  # nested permissions keys stated
        assert key in prompt
    # The bare-shape failure class is called out as forbidden.
    assert "never a bare" in prompt
    # A worked NESTED example: a top-level tools: line with an indented deny: child.
    lines = prompt.splitlines()
    tools_starts = [i for i, ln in enumerate(lines) if ln.strip() == "tools:"]
    assert tools_starts, "prompt must show a tools: YAML block"
    assert any(
        ln.startswith((" ", "\t")) and "deny:" in ln
        for i in tools_starts
        for ln in lines[i + 1 : i + 3]
    ), "the tools: block must demonstrate an indented deny: child"


def test_deployed_agent_visible_and_advertised_same_session(components_home, mock_llm_client):
    """#58: a deployed agent was invisible (/agents) and unadvertised until restart — the
    card registry + AgentTool list were frozen at startup. With the post-deploy callback
    (the real start_cmd wiring: load the yaml, register its card, append to the shared list
    the AgentTool advertises), /agents lists it AND the agent tool advertises it — no restart.
    """
    from localharness.config.loader import ConfigLoader
    from localharness.tools.builtin.agent_tool import AgentTool

    home = components_home
    registry = AgentCardRegistry()
    available = ["explore", "web-researcher"]              # the list AgentTool advertises
    agent_tool = AgentTool(agent_runner=AsyncMock(), available_agents=available)
    loader = ConfigLoader(config_dir=home)

    def on_deployed(name: str) -> None:  # mirrors start_cmd._register_deployed_agent
        cfg = loader.load_agent(name, bypass_cache=True)
        registry.register_from_config(cfg)
        if name not in available:
            available.append(name)

    gen = f"name: hn-monitor\nrole: {ROLE}"
    channel = ScriptedChannel(["create an agent", f"it should {ROLE}", "yes", "/agents"])
    llm = mock_llm_client([mock_llm_client.Response(content=gen)])
    orch = AgentCreationFlow(registry)
    agent = MagicMock()
    agent._config.name = "orchestrator"
    agent.current_session_id = "sess-1"
    agent._llm = llm
    agent.run_turn = AsyncMock()
    repl = OrchestratorREPL(
        orchestrator=orch, agent_loop=agent, channel=channel, bus=AsyncMock(),
        config_dir=home, on_agent_deployed=on_deployed,
    )

    asyncio.run(repl.run())

    assert (home / "agents" / "hn-monitor.yaml").exists()  # actually deployed
    # /agents (issued AFTER deploy) lists the new agent — live, no restart.
    agents_msg = next((m for m in channel.sent if "Configured agents" in m), None)
    assert agents_msg is not None and "hn-monitor" in agents_msg
    # The card registry the REPL reads now holds it.
    assert any(c.name == "hn-monitor" for c in registry.all_cards())
    # The agent tool advertises it, so the model knows it can delegate to it.
    assert "hn-monitor" in agent_tool.info().description


def test_slash_quit_mid_wizard_cancels_creation_not_session(tmp_path, mock_llm_client):
    """#60: /quit and /exit during an active creation hard-exited the whole session silently
    (slash handled before the workflow check), while bare 'quit' only cancels the wizard. The
    slash must first CANCEL the wizard (session alive) and require a SECOND /quit to exit."""
    channel = ScriptedChannel(["create an agent", "/quit", "/help", "/quit"])
    repl, orch, agent, bus = _repl(channel, mock_llm_client([]), config_dir=tmp_path)

    asyncio.run(repl.run())  # returns via the SECOND /quit, not the first

    assert any("Agent creation cancelled. /quit again to exit." in m for m in channel.sent)
    assert orch.active_workflow is None  # wizard cancelled
    assert any("Available commands" in m for m in channel.sent)  # /help worked -> session was alive
    agent.run_turn.assert_not_called()


def test_slash_exit_mid_wizard_also_cancels(tmp_path, mock_llm_client):
    """#60: /exit behaves like /quit mid-wizard (cancel first, don't hard-exit)."""
    channel = ScriptedChannel(["create an agent", "/exit"])
    repl, orch, agent, bus = _repl(channel, mock_llm_client([]), config_dir=tmp_path)

    asyncio.run(repl.run())  # EOF after /exit cancels + loop reads to exhaustion

    assert any("Agent creation cancelled" in m for m in channel.sent)
    assert orch.active_workflow is None


def test_intent_and_confirm_prompts_advertise_cancel(tmp_path, mock_llm_client):
    """#59(b): the escape word was 4 undocumented exact-matches. Advertise it — the
    ask-description (intent) prompt AND the confirm prompt must mention 'cancel'."""
    llm = mock_llm_client([mock_llm_client.Response(content=f"name: hn\nrole: {ROLE}")])
    channel = ScriptedChannel(["create an agent", f"it should {ROLE}"])
    repl, orch, agent, bus = _repl(channel, llm, config_dir=tmp_path)

    asyncio.run(repl.run())

    intent = next(m for m in channel.sent if "help you create an agent" in m)
    assert "cancel" in intent.lower()  # the ask-description prompt advertises the escape
    confirm = next(m for m in channel.sent if "look good" in m.lower())
    assert "cancel" in confirm.lower()  # (yes/no/change, or 'cancel')


def test_discuss_reask_prompt_advertises_cancel(tmp_path):
    """#59(b): the DISCUSS re-ask ('Tell me more') must also advertise 'cancel'."""
    channel = ScriptedChannel(["create an agent", "hi"])  # 'hi' too short -> re-ask
    repl, orch, agent, bus = _repl(channel, RaisingLLM(), config_dir=tmp_path)

    asyncio.run(repl.run())

    reask = next(m for m in channel.sent if "Tell me more" in m)
    assert "cancel" in reask.lower()


_INVALID_YAML = "name: good-agent\nrole: monitor stuff\npermissions:\n  mode: read_only"
_INVALID_YAML_2 = "name: good-agent\nrole: monitor stuff\ntemperature: 999"


def test_invalid_then_valid_shows_valid_config(tmp_path, mock_llm_client):
    """#57: invalid generated YAML must NOT be shown for approval. It regenerates ONCE
    (error fed back) and the CONFIRM prompt shows the VALID config."""
    valid = f"name: good-agent\nrole: {ROLE}"
    llm = mock_llm_client([
        mock_llm_client.Response(content=_INVALID_YAML),  # attempt 1 — invalid enum
        mock_llm_client.Response(content=valid),           # attempt 2 — valid
    ])
    channel = ScriptedChannel(["create an agent", f"it should {ROLE}"])
    repl, orch, agent, bus = _repl(channel, llm, config_dir=tmp_path)

    asyncio.run(repl.run())

    confirm = next((m for m in channel.sent if "look good" in m.lower()), None)
    assert confirm is not None, f"no confirm prompt; sent={channel.sent}"
    assert ROLE in confirm  # the VALID config is shown
    assert "read_only" not in confirm  # the invalid one was never presented
    assert orch.active_workflow is not None  # still mid-flow, awaiting confirm


def test_invalid_twice_aborts_without_pydantic_dump(tmp_path, mock_llm_client):
    """#57: two invalid configs in a row -> truthful abort. NO raw Pydantic wall
    (pydantic.dev URL) ever reaches the channel; the config is never shown for approval."""
    llm = mock_llm_client([
        mock_llm_client.Response(content=_INVALID_YAML),
        mock_llm_client.Response(content=_INVALID_YAML_2),
    ])
    channel = ScriptedChannel(["create an agent", f"it should {ROLE}"])
    repl, orch, agent, bus = _repl(channel, llm, config_dir=tmp_path)

    asyncio.run(repl.run())

    assert not any("look good" in m.lower() for m in channel.sent)  # never asked to approve
    assert not any("pydantic" in m.lower() for m in channel.sent)   # no raw ValidationError wall
    assert any("NOT created" in m for m in channel.sent)            # truthful abort
    assert orch.active_workflow is None                             # abandoned cleanly
    assert not (tmp_path / "agents").exists() or not list((tmp_path / "agents").glob("*.yaml"))


def test_generation_prompt_states_enum_legal_values():
    """#57(b): the prompt must state the legal enum values for permissions.mode, DERIVED
    from the Pydantic Literal so it can't drift from AgentConfig."""
    from localharness.cli.repl import _generation_system_prompt
    prompt = _generation_system_prompt()
    assert "auto" in prompt and "manual" in prompt  # permissions.mode legal values


class RaisingLLM:
    """LLM double whose stream_complete RAISES — models a provider timeout.

    conftest's MockLLMClient structurally cannot raise (it only returns scripted
    responses), so #29 (an unhandled generation error tearing down the whole
    session) needs a double that actually throws.
    """

    def __init__(self) -> None:
        class _Config:
            tool_call_mode = "native"
            context_window = 128_000

        self.config = _Config()

    async def stream_complete(self, messages=None, tools=None, on_token=None):
        raise RuntimeError("provider timed out")


def test_deploy_failure_never_claims_success(tmp_path, mock_llm_client):
    """#27: on deploy failure the REPL sent 'Deploy failed' AND THEN, on the same
    (both) path, 'Agent created.' plus reset the workflow as if it succeeded. The
    failure path must never claim success; it must be truthful and abandon cleanly
    with the session still alive.

    (#57 moved schema-invalid configs to a pre-confirm generation abort, so the deploy
    STAGE is now exercised with a genuine deploy-time failure: a VALID config whose
    target already exists — deploy_config refuses to overwrite it (#28).)"""
    (tmp_path / "agents").mkdir(parents=True)
    existing = tmp_path / "agents" / "finance-helper.yaml"
    existing.write_text("name: finance-helper\nrole: the original agent\n")
    original = existing.read_bytes()

    llm = mock_llm_client([mock_llm_client.Response(content=f"name: finance-helper\nrole: {ROLE}")])
    channel = ScriptedChannel(
        ["create an agent", f"it should {ROLE[0].lower()}{ROLE[1:]}", "yes"]
    )
    repl, orch, agent, bus = _repl(channel, llm, config_dir=tmp_path)

    asyncio.run(repl.run())  # returns normally: session survives a failed deploy

    assert not any("Agent created" in m for m in channel.sent)  # #27: the lie
    assert any("Deploy failed" in m for m in channel.sent)  # truthful failure kept
    assert any("NOT created" in m for m in channel.sent)  # explicit non-success
    assert orch.active_workflow is None  # abandoned cleanly, back to normal convo
    assert existing.read_bytes() == original  # refuse-overwrite left it byte-identical


def test_generation_error_keeps_session_alive(tmp_path):
    """#29: the generation stream_complete call had no try/except and the REPL loop
    catches only EOFError, so a provider error during 'create an agent' propagated
    out of run() and tore down the whole session. It must be caught: truthful
    message, session alive, workflow in a sane (cleared) state."""
    channel = ScriptedChannel(["create an agent", f"it should {ROLE[0].lower()}{ROLE[1:]}"])
    repl, orch, agent, bus = _repl(channel, RaisingLLM(), config_dir=tmp_path)

    asyncio.run(repl.run())  # must NOT raise — the session survives the provider error

    assert any("generation failed" in m.lower() for m in channel.sent)
    assert not any("Agent created" in m for m in channel.sent)
    assert orch.active_workflow is None
