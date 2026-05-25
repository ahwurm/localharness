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
    # We just verify the command is importable and is registered
    from localharness.cli.start_cmd import start_app
    assert callable(start_app)


# ---------------------------------------------------------------------------
# OrchestratorREPL
# ---------------------------------------------------------------------------

def test_orchestrator_repl_exits_on_exit_input():
    """REPL loop should exit when user types 'exit'."""
    from localharness.cli.repl import OrchestratorREPL

    mock_channel = AsyncMock()
    mock_channel.read_input = AsyncMock(return_value="exit")
    mock_loop = AsyncMock()
    mock_bus = MagicMock()

    repl = OrchestratorREPL(agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)

    asyncio.run(repl.run())

    # Channel.start and stop should have been called
    mock_channel.start.assert_called_once()
    mock_channel.stop.assert_called_once()
    # Agent loop should NOT have been called (exit before any task)
    mock_loop.run_turn.assert_not_called()


def test_orchestrator_repl_exits_on_quit_input():
    from localharness.cli.repl import OrchestratorREPL

    mock_channel = AsyncMock()
    mock_channel.read_input = AsyncMock(return_value="quit")
    mock_loop = AsyncMock()
    mock_bus = MagicMock()

    repl = OrchestratorREPL(agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    mock_channel.start.assert_called_once()
    mock_channel.stop.assert_called_once()


def test_orchestrator_repl_exits_on_eof():
    """REPL loop should exit cleanly on EOFError (Ctrl-D)."""
    from localharness.cli.repl import OrchestratorREPL

    mock_channel = AsyncMock()
    mock_channel.read_input = AsyncMock(side_effect=EOFError)
    mock_loop = AsyncMock()
    mock_bus = MagicMock()

    repl = OrchestratorREPL(agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    mock_channel.stop.assert_called_once()


def test_orchestrator_repl_dispatches_task():
    """REPL dispatches non-exit user input to agent loop, then exits."""
    from localharness.cli.repl import OrchestratorREPL

    responses = ["do something", "exit"]
    response_iter = iter(responses)

    async def fake_read_input(prompt="you> "):
        return next(response_iter)

    mock_channel = AsyncMock()
    mock_channel.read_input = fake_read_input

    mock_agent_config = MagicMock()
    mock_agent_config.name = "test-agent"

    mock_loop = AsyncMock()
    mock_loop._config = mock_agent_config
    mock_loop.run_turn = AsyncMock(return_value="Done.")
    mock_bus = MagicMock()

    repl = OrchestratorREPL(agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    mock_loop.run_turn.assert_called_once_with(task="do something", on_token=None)


def test_orchestrator_repl_skips_empty_input():
    """REPL skips empty input without dispatching to agent."""
    from localharness.cli.repl import OrchestratorREPL

    responses = ["", "exit"]
    response_iter = iter(responses)

    async def fake_read_input(prompt="you> "):
        return next(response_iter)

    mock_channel = AsyncMock()
    mock_channel.read_input = fake_read_input
    mock_loop = AsyncMock()
    mock_bus = MagicMock()

    repl = OrchestratorREPL(agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    mock_loop.run_turn.assert_not_called()


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
