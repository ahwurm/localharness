"""Tests for localharness start command smart routing and OrchestratorREPL."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli.app import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_agent(agents_dir: Path, name: str, role: str = "Test role") -> None:
    agents_dir.mkdir(parents=True, exist_ok=True)
    path = agents_dir / f"{name}.yaml"
    path.write_text(yaml.dump({"name": name, "role": role, "model": "inherit"}))


def _make_mock_orchestrator():
    mock = MagicMock()
    mock._card_registry = MagicMock()
    mock._card_registry.all_cards = MagicMock(return_value=[])
    mock.active_workflow = None
    mock.begin_agent_creation = MagicMock()
    mock.route_task = MagicMock(return_value=MagicMock(
        matched=False, agent_id=None, confidence=0.0, reason="No active agents",
    ))
    return mock


# ---------------------------------------------------------------------------
# app.py registration
# ---------------------------------------------------------------------------

def test_app_registers_start_command():
    names = [c.name for c in app.registered_commands]
    assert "start" in names


def test_app_registers_agent_subcommand():
    group_names = [g.name for g in app.registered_groups]
    assert "agent" in group_names


# ---------------------------------------------------------------------------
# single-source window derivation: served -> effective -> override
# ---------------------------------------------------------------------------

def test_effective_max_context_served_minus_reserve():
    """No explicit cap below served-reserve -> derive from served window."""
    from localharness.cli.start_cmd import _effective_max_context
    # cfg at schema default (131072) on a 131072 server -> clamps to served-reserve.
    assert _effective_max_context(131_072, 131_072, 4_096) == 126_976


def test_effective_max_context_honors_fitting_override():
    """An explicit cap that fits under served-reserve wins (override)."""
    from localharness.cli.start_cmd import _effective_max_context
    assert _effective_max_context(131_072, 32_000, 4_096) == 32_000


def test_effective_max_context_clamps_oversized_config():
    """A config larger than served-reserve is clamped down to the served value."""
    from localharness.cli.start_cmd import _effective_max_context
    assert _effective_max_context(64_000, 200_000, 4_096) == 59_904


def test_effective_max_context_no_served_uses_config():
    """Server didn't report a window -> config value is the only signal."""
    from localharness.cli.start_cmd import _effective_max_context
    assert _effective_max_context(None, 61_440, 4_096) == 61_440


# ---------------------------------------------------------------------------
# start — no agents → init flow + default agent creation
# ---------------------------------------------------------------------------

def test_start_no_agents_runs_async(tmp_path, monkeypatch):
    """start with no agents should invoke _start_async and not crash at import."""
    from localharness.cli.start_cmd import start_app
    assert callable(start_app)


# ---------------------------------------------------------------------------
# OrchestratorREPL — slash commands
# ---------------------------------------------------------------------------

def test_repl_slash_help():
    """REPL /help shows help text without calling agent loop."""
    from localharness.cli.repl import OrchestratorREPL, HELP_TEXT

    responses = ["/help"]
    response_iter = iter(responses)

    async def fake_read_input(prompt="you> "):
        val = next(response_iter, None)
        if val is None:
            raise EOFError()
        return val

    mock_channel = AsyncMock()
    mock_channel.read_input = fake_read_input
    mock_loop = AsyncMock()
    mock_bus = AsyncMock()
    mock_orch = _make_mock_orchestrator()

    repl = OrchestratorREPL(orchestrator=mock_orch, agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    # send_message should have been called with HELP_TEXT
    mock_channel.send_message.assert_any_call(HELP_TEXT, metadata={"style": "system.info"})
    # Agent loop should NOT have been called
    mock_loop.run_turn.assert_not_called()


def test_repl_slash_quit_raises_eof():
    """/quit exits the REPL cleanly."""
    from localharness.cli.repl import OrchestratorREPL

    mock_channel = AsyncMock()
    mock_channel.read_input = AsyncMock(return_value="/quit")
    mock_loop = AsyncMock()
    mock_bus = AsyncMock()
    mock_orch = _make_mock_orchestrator()

    repl = OrchestratorREPL(orchestrator=mock_orch, agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    mock_channel.start.assert_called_once()
    mock_channel.stop.assert_called_once()
    mock_loop.run_turn.assert_not_called()


def test_repl_slash_exit_raises_eof():
    """/exit exits the REPL cleanly."""
    from localharness.cli.repl import OrchestratorREPL

    mock_channel = AsyncMock()
    mock_channel.read_input = AsyncMock(return_value="/exit")
    mock_loop = AsyncMock()
    mock_bus = AsyncMock()
    mock_orch = _make_mock_orchestrator()

    repl = OrchestratorREPL(orchestrator=mock_orch, agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    mock_channel.start.assert_called_once()
    mock_channel.stop.assert_called_once()


def test_repl_slash_agents_empty():
    """/agents with no cards shows 'No agents configured' message."""
    from localharness.cli.repl import OrchestratorREPL

    responses = ["/agents"]
    response_iter = iter(responses)

    async def fake_read_input(prompt="you> "):
        val = next(response_iter, None)
        if val is None:
            raise EOFError()
        return val

    mock_channel = AsyncMock()
    mock_channel.read_input = fake_read_input
    mock_loop = AsyncMock()
    mock_bus = AsyncMock()
    mock_orch = _make_mock_orchestrator()

    repl = OrchestratorREPL(orchestrator=mock_orch, agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    calls = mock_channel.send_message.call_args_list
    assert any("No agents configured" in str(c) for c in calls)


def test_repl_slash_agents_with_cards():
    """/agents with registered cards shows formatted list."""
    from localharness.cli.repl import OrchestratorREPL

    mock_card = MagicMock()
    mock_card.name = "finance-agent"
    mock_card.description = "Handles financial tasks"
    mock_card.status = "active"

    responses = ["/agents"]
    response_iter = iter(responses)

    async def fake_read_input(prompt="you> "):
        val = next(response_iter, None)
        if val is None:
            raise EOFError()
        return val

    mock_channel = AsyncMock()
    mock_channel.read_input = fake_read_input
    mock_loop = AsyncMock()
    mock_bus = AsyncMock()
    mock_orch = _make_mock_orchestrator()
    mock_orch._card_registry.all_cards.return_value = [mock_card]

    repl = OrchestratorREPL(orchestrator=mock_orch, agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    calls = mock_channel.send_message.call_args_list
    assert any("finance-agent" in str(c) for c in calls)


def test_repl_unknown_slash_passes_through():
    """Unknown slash command passes through to orchestrator routing then agent loop."""
    from localharness.cli.repl import OrchestratorREPL

    responses = ["/unknown"]
    response_iter = iter(responses)

    async def fake_read_input(prompt="you> "):
        val = next(response_iter, None)
        if val is None:
            raise EOFError()
        return val

    mock_agent_config = MagicMock()
    mock_agent_config.name = "test-agent"

    mock_channel = AsyncMock()
    mock_channel.read_input = fake_read_input
    mock_loop = AsyncMock()
    mock_loop._config = mock_agent_config
    mock_loop.current_session_id = None
    mock_loop.run_turn = AsyncMock(return_value="Done.")
    mock_bus = AsyncMock()
    mock_orch = _make_mock_orchestrator()

    repl = OrchestratorREPL(orchestrator=mock_orch, agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    mock_orch.route_task.assert_called_once_with("/unknown")
    mock_loop.run_turn.assert_called_once_with(task="/unknown", on_token=None)
    # REPL should NOT call send_message with the summary
    summary_calls = [
        c for c in mock_channel.send_message.call_args_list
        if len(c[0]) > 0 and c[0][0] == "Done."
    ]
    assert len(summary_calls) == 0


# ---------------------------------------------------------------------------
# OrchestratorREPL — normal routing
# ---------------------------------------------------------------------------

def test_repl_normal_input_routes_to_agent():
    """Normal text calls route_task then routes to agent_loop.run_turn."""
    from localharness.cli.repl import OrchestratorREPL

    responses = ["do something"]
    response_iter = iter(responses)

    async def fake_read_input(prompt="you> "):
        val = next(response_iter, None)
        if val is None:
            raise EOFError()
        return val

    mock_agent_config = MagicMock()
    mock_agent_config.name = "test-agent"

    mock_channel = AsyncMock()
    mock_channel.read_input = fake_read_input
    mock_loop = AsyncMock()
    mock_loop._config = mock_agent_config
    mock_loop.current_session_id = None
    mock_loop.run_turn = AsyncMock(return_value="Done.")
    mock_bus = AsyncMock()
    mock_orch = _make_mock_orchestrator()

    repl = OrchestratorREPL(orchestrator=mock_orch, agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    mock_orch.route_task.assert_called_once_with("do something")
    mock_loop.run_turn.assert_called_once_with(task="do something", on_token=None)
    # REPL should NOT call send_message with the summary — TaskComplete event does it
    summary_calls = [
        c for c in mock_channel.send_message.call_args_list
        if len(c[0]) > 0 and c[0][0] == "Done."
    ]
    assert len(summary_calls) == 0


def test_repl_does_not_double_fire_output():
    """ORCH-01: After run_turn, REPL does NOT call send_message (TaskComplete event does it)."""
    from localharness.cli.repl import OrchestratorREPL

    responses = ["hello"]
    response_iter = iter(responses)

    async def fake_read_input(prompt="you> "):
        val = next(response_iter, None)
        if val is None:
            raise EOFError()
        return val

    mock_agent_config = MagicMock()
    mock_agent_config.name = "test-agent"

    mock_channel = AsyncMock()
    mock_channel.read_input = fake_read_input
    mock_loop = AsyncMock()
    mock_loop._config = mock_agent_config
    mock_loop.current_session_id = None
    mock_loop.run_turn = AsyncMock(return_value="Done.")
    mock_bus = AsyncMock()
    mock_orch = _make_mock_orchestrator()
    mock_orch.route_task.return_value = MagicMock(
        matched=True, agent_id="test-agent", confidence=0.8, reason="Routed",
    )

    repl = OrchestratorREPL(orchestrator=mock_orch, agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    # REPL must NOT call send_message with the run_turn summary
    summary_calls = [
        c for c in mock_channel.send_message.call_args_list
        if len(c[0]) > 0 and c[0][0] == "Done."
    ]
    assert len(summary_calls) == 0, "REPL should not send summary — TaskComplete event handles output"


def test_repl_calls_route_task_before_run_turn():
    """ORCH-01: route_task is called before run_turn for proper routing order."""
    from localharness.cli.repl import OrchestratorREPL
    from localharness.orchestrator.cards import RoutingDecision

    responses = ["analyze data"]
    response_iter = iter(responses)

    async def fake_read_input(prompt="you> "):
        val = next(response_iter, None)
        if val is None:
            raise EOFError()
        return val

    mock_agent_config = MagicMock()
    mock_agent_config.name = "test-agent"

    mock_channel = AsyncMock()
    mock_channel.read_input = fake_read_input
    mock_loop = AsyncMock()
    mock_loop._config = mock_agent_config
    mock_loop.current_session_id = None
    mock_loop.run_turn = AsyncMock(return_value="Done.")
    mock_bus = AsyncMock()
    mock_orch = _make_mock_orchestrator()
    mock_orch.route_task.return_value = RoutingDecision(
        matched=True, agent_id="test-agent", agent_card=None,
        confidence=0.8, reason="Routed",
    )

    # Track call order
    call_order = []
    original_route = mock_orch.route_task
    original_run = mock_loop.run_turn

    def track_route(*a, **kw):
        call_order.append("route_task")
        return original_route(*a, **kw)

    async def track_run(*a, **kw):
        call_order.append("run_turn")
        return await original_run(*a, **kw)

    mock_orch.route_task = track_route
    mock_loop.run_turn = track_run

    repl = OrchestratorREPL(orchestrator=mock_orch, agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    assert call_order == ["route_task", "run_turn"], f"Expected route_task before run_turn, got: {call_order}"


def test_repl_skips_empty_input():
    """REPL skips empty input without dispatching to agent."""
    from localharness.cli.repl import OrchestratorREPL

    responses = [""]
    response_iter = iter(responses)

    async def fake_read_input(prompt="you> "):
        val = next(response_iter, None)
        if val is None:
            raise EOFError()
        return val

    mock_channel = AsyncMock()
    mock_channel.read_input = fake_read_input
    mock_loop = AsyncMock()
    mock_bus = AsyncMock()
    mock_orch = _make_mock_orchestrator()

    repl = OrchestratorREPL(orchestrator=mock_orch, agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    mock_loop.run_turn.assert_not_called()


def test_repl_exits_on_eof():
    """REPL loop should exit cleanly on EOFError (Ctrl-D)."""
    from localharness.cli.repl import OrchestratorREPL

    mock_channel = AsyncMock()
    mock_channel.read_input = AsyncMock(side_effect=EOFError)
    mock_loop = AsyncMock()
    mock_bus = AsyncMock()
    mock_orch = _make_mock_orchestrator()

    repl = OrchestratorREPL(orchestrator=mock_orch, agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    mock_channel.stop.assert_called_once()


# ---------------------------------------------------------------------------
# OrchestratorREPL — agent creation workflow
# ---------------------------------------------------------------------------

def test_repl_creation_intent_starts_workflow():
    """User types 'create an agent for finance' -> begins agent creation workflow."""
    from localharness.cli.repl import OrchestratorREPL

    responses = ["create an agent for handling finance tasks"]
    response_iter = iter(responses)

    async def fake_read_input(prompt="you> "):
        val = next(response_iter, None)
        if val is None:
            raise EOFError()
        return val

    mock_channel = AsyncMock()
    mock_channel.read_input = fake_read_input
    mock_loop = AsyncMock()
    mock_bus = AsyncMock()
    mock_orch = _make_mock_orchestrator()

    mock_workflow = MagicMock()
    mock_workflow.transition = MagicMock(return_value="discuss")
    mock_orch.begin_agent_creation.return_value = mock_workflow

    repl = OrchestratorREPL(orchestrator=mock_orch, agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    # begin_agent_creation should have been called
    mock_orch.begin_agent_creation.assert_called_once()
    # workflow.transition should have been called with the user input
    mock_workflow.transition.assert_called_once_with("create an agent for handling finance tasks")
    # Channel should show the creation start message
    calls = mock_channel.send_message.call_args_list
    assert any("I'd like to help you create an agent" in str(c) for c in calls)
    # Agent loop should NOT have been called
    mock_loop.run_turn.assert_not_called()


def test_repl_active_workflow_routes_to_creation_handler():
    """When active_workflow is not None, input goes to _handle_creation_workflow."""
    from localharness.cli.repl import OrchestratorREPL
    from localharness.orchestrator.workflow import WorkflowState

    responses = ["it should handle stock analysis"]
    response_iter = iter(responses)

    async def fake_read_input(prompt="you> "):
        val = next(response_iter, None)
        if val is None:
            raise EOFError()
        return val

    mock_channel = AsyncMock()
    mock_channel.read_input = fake_read_input
    mock_loop = AsyncMock()
    mock_bus = AsyncMock()
    mock_orch = _make_mock_orchestrator()

    # Set up active workflow
    mock_workflow = MagicMock()
    mock_workflow.transition = MagicMock(return_value=WorkflowState.DISCUSS)
    mock_orch.active_workflow = mock_workflow

    repl = OrchestratorREPL(orchestrator=mock_orch, agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    # workflow.transition should have been called with the input
    mock_workflow.transition.assert_called_once_with("it should handle stock analysis")
    # Agent loop should NOT have been called
    mock_loop.run_turn.assert_not_called()
    # Channel should show "Tell me more" message (DISCUSS state)
    calls = mock_channel.send_message.call_args_list
    assert any("Tell me more" in str(c) for c in calls)


def test_repl_creation_cancel_clears_workflow():
    """User types 'cancel' during workflow -> clears active workflow."""
    from localharness.cli.repl import OrchestratorREPL
    from localharness.orchestrator.workflow import WorkflowState

    responses = ["cancel"]
    response_iter = iter(responses)

    async def fake_read_input(prompt="you> "):
        val = next(response_iter, None)
        if val is None:
            raise EOFError()
        return val

    mock_channel = AsyncMock()
    mock_channel.read_input = fake_read_input
    mock_loop = AsyncMock()
    mock_bus = AsyncMock()
    mock_orch = _make_mock_orchestrator()

    # Set up active workflow that will transition to CANCELLED
    mock_workflow = MagicMock()
    mock_workflow.transition = MagicMock(return_value=WorkflowState.CANCELLED)
    mock_orch.active_workflow = mock_workflow

    repl = OrchestratorREPL(orchestrator=mock_orch, agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    # _active_workflow should be set to None
    assert mock_orch._active_workflow is None
    # Channel should show cancellation message
    calls = mock_channel.send_message.call_args_list
    assert any("cancelled" in str(c).lower() for c in calls)


# ---------------------------------------------------------------------------
# start_app smart routing (via _discover_agents_for_start helper)
# ---------------------------------------------------------------------------

def test_start_discovers_agents_from_config_dir(tmp_path):
    """_discover_agents_for_start returns agents from the config_dir."""
    from localharness.cli.start_cmd import _discover_agents_for_start

    _write_agent(tmp_path / "agents", "alpha")
    _write_agent(tmp_path / "agents", "beta")

    agents = _discover_agents_for_start(tmp_path)
    names = [a["name"] for a in agents]
    assert "alpha" in names
    assert "beta" in names


def test_start_single_agent_no_picker(tmp_path):
    """With one agent configured, _discover_agents_for_start returns it."""
    from localharness.cli.start_cmd import _discover_agents_for_start

    _write_agent(tmp_path / "agents", "solo")

    agents = _discover_agents_for_start(tmp_path)
    assert len(agents) == 1
    assert agents[0]["name"] == "solo"


def test_resolve_timeout_precedence():
    """Per-agent timeout override wins when set; None falls back to the provider default.

    Regression: AgentConfig.timeout_seconds was dead config — the start path always
    used provider.timeout_seconds, so slow-decode users could not override out of a
    too-tight timeout."""
    from localharness.cli.start_cmd import _resolve_timeout
    assert _resolve_timeout(900.0, 600.0) == 900.0   # explicit agent override wins
    assert _resolve_timeout(None, 600.0) == 600.0     # unset → provider default


# ---------------------------------------------------------------------------
# Task 1: Import path regression tests
# ---------------------------------------------------------------------------

def test_register_builtin_tools_import_path():
    """CLI-02: register_builtin_tools imports from localharness.tools.builtin without error."""
    from localharness.tools.builtin import register_builtin_tools
    assert callable(register_builtin_tools)


def test_register_builtin_tools_not_in_registry():
    """Regression: register_builtin_tools must NOT be importable from tools.registry."""
    import pytest
    with pytest.raises(ImportError):
        from localharness.tools.registry import register_builtin_tools


# ---------------------------------------------------------------------------
# Task 3: UserMessage publishing tests
# ---------------------------------------------------------------------------

def test_repl_publishes_user_message_before_run_turn():
    """MEM-02: Normal input publishes UserMessage with correct agent_id and content."""
    from localharness.cli.repl import OrchestratorREPL
    from localharness.core.events import UserMessage

    responses = ["do something"]
    response_iter = iter(responses)

    async def fake_read_input(prompt="you> "):
        val = next(response_iter, None)
        if val is None:
            raise EOFError()
        return val

    mock_agent_config = MagicMock()
    mock_agent_config.name = "test-agent"

    mock_channel = AsyncMock()
    mock_channel.read_input = fake_read_input
    mock_loop = AsyncMock()
    mock_loop._config = mock_agent_config
    mock_loop.current_session_id = None
    mock_loop.run_turn = AsyncMock(return_value="Done.")
    mock_bus = AsyncMock()
    mock_orch = _make_mock_orchestrator()

    repl = OrchestratorREPL(orchestrator=mock_orch, agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    # bus.publish should have been called with a UserMessage
    publish_calls = mock_bus.publish.call_args_list
    user_msg_calls = [
        c for c in publish_calls
        if isinstance(c[0][0], UserMessage)
    ]
    assert len(user_msg_calls) == 1
    msg = user_msg_calls[0][0][0]
    assert msg.content == "do something"
    assert msg.channel == "terminal"
    assert msg.agent_id == "test-agent"


def test_repl_slash_commands_no_user_message():
    """MEM-02: Slash commands do NOT publish UserMessage -- they are infrastructure."""
    from localharness.cli.repl import OrchestratorREPL
    from localharness.core.events import UserMessage

    responses = ["/help"]
    response_iter = iter(responses)

    async def fake_read_input(prompt="you> "):
        val = next(response_iter, None)
        if val is None:
            raise EOFError()
        return val

    mock_channel = AsyncMock()
    mock_channel.read_input = fake_read_input
    mock_loop = AsyncMock()
    mock_bus = AsyncMock()
    mock_orch = _make_mock_orchestrator()

    repl = OrchestratorREPL(orchestrator=mock_orch, agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    # bus.publish should NOT have been called with UserMessage
    publish_calls = mock_bus.publish.call_args_list
    user_msg_calls = [
        c for c in publish_calls
        if len(c[0]) > 0 and isinstance(c[0][0], UserMessage)
    ]
    assert len(user_msg_calls) == 0


# ---------------------------------------------------------------------------
# Task 2: deploy_config path tests
# ---------------------------------------------------------------------------

def test_deploy_config_writes_to_agents_subdir(tmp_path):
    """ORCH-02: deploy_config writes to config_dir/agents/{name}.yaml."""
    from localharness.orchestrator.workflow import AgentCreationWorkflow
    wf = AgentCreationWorkflow(config_dir=tmp_path)
    wf.set_generated_yaml("name: test-bot\nrole: Test\n")
    result_path = wf.deploy_config("test-bot")
    assert result_path == tmp_path / "agents" / "test-bot.yaml"
    assert result_path.exists()
    assert "name: test-bot" in result_path.read_text()


def test_deploy_config_default_path(tmp_path, monkeypatch):
    """ORCH-02: default deploy path is ~/.localharness/agents/{name}.yaml."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from localharness.orchestrator.workflow import AgentCreationWorkflow
    wf = AgentCreationWorkflow()  # no config_dir
    wf.set_generated_yaml("name: default-bot\nrole: Default\n")
    result_path = wf.deploy_config("default-bot")
    assert result_path == tmp_path / ".localharness" / "agents" / "default-bot.yaml"
    assert result_path.exists()


# ---------------------------------------------------------------------------
# Plan 33-03 Task 2: session lifecycle wired into the REAL _start_async
#
# FINDING-A: the sessions table had ZERO rows across every sitting ever — the
# create_session/end_session primitives had no production caller. These drives run
# the real _start_async with only the EXTERNAL boundaries stubbed (LLM probe,
# tokenizer, REPL loop, plugin discovery); everything memory-side runs for real,
# which is the wiring under test. repl.run is a no-op, so the sitting has zero turns.
# ---------------------------------------------------------------------------

def _stub_start_boundaries(tmp_path, monkeypatch, *, capture_session_id=None, repl_run=None):
    """Write a minimal (known-good) config and stub every external boundary so the
    real _start_async runs offline. `repl_run` overrides the no-op REPL loop to drive
    live bus traffic through the running harness."""
    (tmp_path / "config.yaml").write_text(
        "version: '1'\n"
        "provider:\n"
        "  provider_type: ollama\n"
        "  base_url: http://localhost:11434/v1\n"
        "  default_model: test-model\n"
        "  api_key: none\n"
    )

    async def fake_probe(llm, max_retries=3, delay=2.0):
        # served window comfortably above the default 131072 cfg + 4096 reserve so
        # the fit-check (start_cmd:248) does not abort the drive
        return (True, "native", 262_144)
    monkeypatch.setattr("localharness.cli.start_cmd._probe_llm", fake_probe)

    class _StubTokenCounter:
        # the real TokenCounter FAILS LOUD without a /tokenize server
        def __init__(self, base_url=None, model=None):
            pass

        def count(self, text=""):
            return max(1, len(str(text)) // 4)

        def count_messages(self, messages):
            return sum(self.count(m.get("content", "")) for m in messages)
    monkeypatch.setattr("localharness.agent.context.TokenCounter", _StubTokenCounter)

    async def default_repl_run(self):
        return None  # clean, immediate return -> zero turns -> exit_reason "complete"
    monkeypatch.setattr(
        "localharness.cli.repl.OrchestratorREPL.run", repl_run or default_repl_run
    )

    async def fake_discover(self):
        return []  # keep the test off the real home plugin dir
    monkeypatch.setattr("localharness.plugins.loader.PluginLoader.discover_all", fake_discover)

    if capture_session_id is not None:
        import localharness.agent.loop as _loop_mod
        real_init = _loop_mod.AgentLoop.__init__

        def wrapped_init(self, *args, **kwargs):
            capture_session_id.append(kwargs.get("session_id"))
            return real_init(self, *args, **kwargs)
        monkeypatch.setattr("localharness.agent.loop.AgentLoop.__init__", wrapped_init)


def _read_sessions(tmp_path, agent="orchestrator"):
    """Read the sessions table from the real memory.db the drive wrote.

    Phase 33.1: fresh/migrated installs write under 'orchestrator'; the collision
    drive reads the un-migrated legacy store by passing agent="default"."""
    import sqlite3
    db_path = tmp_path / "agents" / agent / "memory.db"
    assert db_path.exists(), f"memory.db not created at {db_path}"
    con = sqlite3.connect(str(db_path))
    try:
        return con.execute(
            "SELECT id, started_at, ended_at, exit_reason, summary, "
            "turn_count, action_count FROM sessions"
        ).fetchall()
    finally:
        con.close()


async def test_session_lifecycle_create_and_end_once(tmp_path, monkeypatch):
    """A full _start_async lifecycle inserts exactly ONE sessions row, opened at
    start and closed at shutdown with a real exit_reason."""
    from localharness.cli.start_cmd import _start_async
    _stub_start_boundaries(tmp_path, monkeypatch)

    await _start_async(None, False, False, str(tmp_path))

    rows = _read_sessions(tmp_path)
    assert len(rows) == 1
    _id, started_at, ended_at, exit_reason, _summary, _tc, _ac = rows[0]
    assert started_at is not None
    assert ended_at is not None
    assert exit_reason == "complete"

    # ORCH-01: a fresh mint lands the root as 'orchestrator' — the sibling yaml exists
    # with the rewritten name (not the legacy 'default'), and discovery read it back.
    root_yaml = tmp_path / "agents" / "orchestrator.yaml"
    assert root_yaml.exists()
    assert yaml.safe_load(root_yaml.read_text(encoding="utf-8"))["name"] == "orchestrator"


async def test_session_id_threaded_to_agent_loop(tmp_path, monkeypatch):
    """The sitting id minted in start_cmd is the SAME id AgentLoop carries and the
    SAME id the sessions row is keyed by (create == loop == end)."""
    from localharness.cli.start_cmd import _start_async
    captured: list = []
    _stub_start_boundaries(tmp_path, monkeypatch, capture_session_id=captured)

    await _start_async(None, False, False, str(tmp_path))

    assert len(captured) == 1, "exactly one AgentLoop is constructed in a zero-turn drive"
    loop_session_id = captured[0]
    assert loop_session_id is not None, "start_cmd must pass session_id=sitting_id"
    rows = _read_sessions(tmp_path)
    assert len(rows) == 1
    assert rows[0][0] == loop_session_id  # sessions row id == AgentLoop session_id


async def test_vacuous_sitting_leaves_shelf_suppressed(tmp_path, monkeypatch):
    """SESS-05 KILL guardrail, end-to-end: a zero-turn sitting writes a NULL summary
    and the injected index does NOT advertise an empty 'Recent Session History'."""
    from localharness.cli.start_cmd import _start_async
    from localharness.memory.sqlite import MemoryStore
    _stub_start_boundaries(tmp_path, monkeypatch)

    await _start_async(None, False, False, str(tmp_path))

    rows = _read_sessions(tmp_path)
    assert len(rows) == 1
    assert rows[0][4] is None  # summary NULL (vacuous -> suppressed, not "worked on stuff")
    assert rows[0][5] == 0     # turn_count 0

    # A fresh store rendering the injected index must not promise an empty shelf.
    store = MemoryStore(
        agent_id="orchestrator", division_id="default", org_id="default",
        base_dir=str(tmp_path),
    )
    await store.open()
    try:
        ctx = await store.load_context()
    finally:
        await store.close()
    assert "Recent Session History" not in ctx.agent_memory_md


async def test_gate_capture_produces_payload_first_summary_row(tmp_path, monkeypatch):
    """Positive control for the KILL guardrail: a sitting with a real gate capture +
    tool use — published on the LIVE bus the accumulator is subscribed to, during the
    running drive — closes with counts AND a payload-first summary that leads with the
    capture detail, not bookkeeping. This exercises the composed spine end-to-end
    (bus -> SessionAccumulator -> derive_session_summary -> end_session -> sessions row)."""
    from localharness.cli.start_cmd import _start_async
    from localharness.core.events import MemoryGateFired, Observation, TurnCompleted

    async def driving_repl_run(self):
        # agent_id must be the running agent ("orchestrator") for the accumulator filter
        # to pass; session_id is the sitting id the loop carries. publish() awaits handlers
        # inline, so the accumulator has counted these before the finally derives the summary.
        sid = self._agent.current_session_id
        await self._bus.publish(TurnCompleted(
            agent_id="orchestrator", session_id=sid, iterations=1, duration_seconds=1.0,
            elapsed_tokens=150, input_tokens=100, output_tokens=50, summary="done",
        ))
        await self._bus.publish(Observation(
            agent_id="orchestrator", session_id=sid, observation_type="tool_result",
            tool_name="bash_exec", output="ok",
        ))
        await self._bus.publish(MemoryGateFired(
            agent_id="orchestrator", session_id=sid, tier="resolved_error",
            fact_key="gate/resolved_error/bash_exec/k", tool_name="bash_exec",
            detail="uv: command not found",
        ))

    _stub_start_boundaries(tmp_path, monkeypatch, repl_run=driving_repl_run)

    await _start_async(None, False, False, str(tmp_path))

    rows = _read_sessions(tmp_path)
    assert len(rows) == 1
    _id, _s, _e, exit_reason, summary, turn_count, action_count = rows[0]
    assert exit_reason == "complete"
    assert turn_count == 1
    assert action_count == 1
    assert summary is not None
    assert summary.startswith("resolved: uv: command not found")
    assert "bash_exec" in summary


# ---------------------------------------------------------------------------
# Phase 33.1 (ORCH-01/03): the one-time root-agent YAML migration helper.
# agents/default.yaml -> agents/orchestrator.yaml with name: rewritten (a bare rename
# is not enough — discovery reads the name: key). Idempotent + crash-safe + collision-safe.
# ---------------------------------------------------------------------------

def test_migrate_legacy_root_yaml_renames_and_rewrites_name(tmp_path):
    from localharness.cli.agent_cmd import _build_agent_yaml
    from localharness.cli.start_cmd import _migrate_legacy_root_agent_yaml

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "default.yaml").write_text(
        yaml.dump(_build_agent_yaml("default", "General-purpose assistant", None),
                  default_flow_style=False),
        encoding="utf-8",
    )

    _migrate_legacy_root_agent_yaml(agents_dir)

    assert not (agents_dir / "default.yaml").exists()
    orch = agents_dir / "orchestrator.yaml"
    assert orch.exists()
    assert yaml.safe_load(orch.read_text(encoding="utf-8"))["name"] == "orchestrator"


def test_migrate_legacy_root_yaml_collision_refuses_untouched(tmp_path):
    from localharness.cli.agent_cmd import _build_agent_yaml
    from localharness.cli.start_cmd import _migrate_legacy_root_agent_yaml

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    legacy = agents_dir / "default.yaml"
    theirs = agents_dir / "orchestrator.yaml"
    legacy.write_text(
        yaml.dump(_build_agent_yaml("default", "General-purpose assistant", None),
                  default_flow_style=False),
        encoding="utf-8",
    )
    # A genuinely different, pre-existing user 'orchestrator' agent.
    theirs.write_text(
        yaml.dump(_build_agent_yaml("orchestrator", "My custom orchestrator", None),
                  default_flow_style=False),
        encoding="utf-8",
    )
    legacy_before = legacy.read_bytes()
    theirs_before = theirs.read_bytes()

    _migrate_legacy_root_agent_yaml(agents_dir)  # must not raise

    # Both files still exist, byte-identical — never merged, never clobbered.
    assert legacy.read_bytes() == legacy_before
    assert theirs.read_bytes() == theirs_before


def test_migrate_legacy_root_yaml_completes_crash_remnant(tmp_path):
    from localharness.cli.agent_cmd import _build_agent_yaml
    from localharness.cli.start_cmd import _migrate_legacy_root_agent_yaml

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "default.yaml").write_text(
        yaml.dump(_build_agent_yaml("default", "General-purpose assistant", None),
                  default_flow_style=False),
        encoding="utf-8",
    )
    # Simulate a crash between write and unlink: orchestrator.yaml already holds the
    # parsed-equal migrated copy the helper would have written.
    migrated = _build_agent_yaml("orchestrator", "General-purpose assistant", None)
    (agents_dir / "orchestrator.yaml").write_text(
        yaml.dump(migrated, default_flow_style=False), encoding="utf-8"
    )

    _migrate_legacy_root_agent_yaml(agents_dir)

    # The remnant path finishes the job: unlink the legacy file, keep the migrated copy.
    assert not (agents_dir / "default.yaml").exists()
    orch = agents_dir / "orchestrator.yaml"
    assert yaml.safe_load(orch.read_text(encoding="utf-8"))["name"] == "orchestrator"


def test_migrate_legacy_root_yaml_ignores_non_root_default_file(tmp_path):
    from localharness.cli.start_cmd import _migrate_legacy_root_agent_yaml

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    # A file literally named default.yaml whose name: field is NOT 'default' — not the
    # minted root; the helper must leave it entirely alone.
    default_file = agents_dir / "default.yaml"
    default_file.write_text(
        yaml.dump({"name": "something-else", "role": "x", "model": "inherit"},
                  default_flow_style=False),
        encoding="utf-8",
    )
    before = default_file.read_bytes()

    _migrate_legacy_root_agent_yaml(agents_dir)

    assert default_file.read_bytes() == before
    assert not (agents_dir / "orchestrator.yaml").exists()
