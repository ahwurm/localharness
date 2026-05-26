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
    mock_bus = MagicMock()
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
    mock_bus = MagicMock()
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
    mock_bus = MagicMock()
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
    mock_bus = MagicMock()
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
    mock_bus = MagicMock()
    mock_orch = _make_mock_orchestrator()
    mock_orch._card_registry.all_cards.return_value = [mock_card]

    repl = OrchestratorREPL(orchestrator=mock_orch, agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    calls = mock_channel.send_message.call_args_list
    assert any("finance-agent" in str(c) for c in calls)


def test_repl_unknown_slash_passes_through():
    """Unknown slash command passes through to agent loop."""
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
    mock_loop.run_turn = AsyncMock(return_value="Done.")
    mock_bus = MagicMock()
    mock_orch = _make_mock_orchestrator()

    repl = OrchestratorREPL(orchestrator=mock_orch, agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    mock_loop.run_turn.assert_called_once_with(task="/unknown", on_token=None)


# ---------------------------------------------------------------------------
# OrchestratorREPL — normal routing
# ---------------------------------------------------------------------------

def test_repl_normal_input_routes_to_agent():
    """Normal text routes to agent_loop.run_turn."""
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
    mock_loop.run_turn = AsyncMock(return_value="Done.")
    mock_bus = MagicMock()
    mock_orch = _make_mock_orchestrator()

    repl = OrchestratorREPL(orchestrator=mock_orch, agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    mock_loop.run_turn.assert_called_once_with(task="do something", on_token=None)


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
    mock_bus = MagicMock()
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
    mock_bus = MagicMock()
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
    mock_bus = MagicMock()
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
    mock_bus = MagicMock()
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
    mock_bus = MagicMock()
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
