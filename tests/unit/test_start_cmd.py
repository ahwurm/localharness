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
# #20: interactive REPL routes memory-subsystem logs to a file, not the terminal
# ---------------------------------------------------------------------------

def test_memory_logs_routed_to_file_not_repl_console(tmp_path):
    """The real emitter behind #20: consolidation/mining use stdlib logging, and the
    interactive start path configures no handler, so their log.warning/exception records
    reach the REPL via logging.lastResort (stderr) over the input prompt.
    _route_memory_logs_to_file sends the memory logger to a file AND stops propagation, so
    records never reach a terminal-bound root handler — while the detail is kept in the file."""
    import logging
    from io import StringIO

    from localharness.cli.start_cmd import _route_memory_logs_to_file

    mem = logging.getLogger("localharness.memory")
    root = logging.getLogger()
    saved = (mem.handlers[:], mem.level, mem.propagate)
    sink = StringIO()  # stand-in for the REPL's terminal-bound root handler (the leak target)
    root_h = logging.StreamHandler(sink)
    root_h.setLevel(logging.DEBUG)
    root.addHandler(root_h)
    try:
        log_path = _route_memory_logs_to_file(tmp_path)
        # exact lines the mining + consolidation paths actually emit today
        logging.getLogger("localharness.memory.mining").warning(
            "mining B4(ii): retracted resurrected stale value"
        )
        logging.getLogger("localharness.memory.consolidation").info(
            "consolidation: folded=3 promoted=1 decayed=0 demoted=0 churn=0.10"
        )
        for h in mem.handlers:
            h.flush()
        terminal_out = sink.getvalue()
        assert "retracted resurrected stale value" not in terminal_out  # not on the terminal
        assert "folded=3" not in terminal_out
        assert mem.propagate is False                                   # bubble stopped
        file_text = Path(log_path).read_text(encoding="utf-8")
        assert "retracted resurrected stale value" in file_text        # detail kept in the file
        assert "folded=3" in file_text
    finally:
        root.removeHandler(root_h)
        mem.handlers[:] = saved[0]
        mem.setLevel(saved[1])
        mem.propagate = saved[2]


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
    """Unknown slash command passes through to the agent loop."""
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
    mock_loop.current_session_id = None
    mock_loop.run_turn = AsyncMock(return_value="Done.")
    mock_bus = AsyncMock()
    mock_orch = _make_mock_orchestrator()

    repl = OrchestratorREPL(orchestrator=mock_orch, agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

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

    repl = OrchestratorREPL(orchestrator=mock_orch, agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    # REPL must NOT call send_message with the run_turn summary
    summary_calls = [
        c for c in mock_channel.send_message.call_args_list
        if len(c[0]) > 0 and c[0][0] == "Done."
    ]
    assert len(summary_calls) == 0, "REPL should not send summary — TaskComplete event handles output"


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
    mock_orch.begin_agent_creation.return_value = mock_workflow

    repl = OrchestratorREPL(orchestrator=mock_orch, agent_loop=mock_loop, channel=mock_channel, bus=mock_bus)
    asyncio.run(repl.run())

    # begin_agent_creation should have been called
    mock_orch.begin_agent_creation.assert_called_once()
    # #19: the trigger message must NOT be fed into the workflow — it was
    # consumed as the agent DESCRIPTION and silently advanced DISCUSS->CONFIGURE
    # (return value discarded), so YAML generation never ran. The description
    # is the user's NEXT message.
    mock_workflow.transition.assert_not_called()
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


def test_repl_agent_creation_uses_streaming_path():
    """#18 RED: REPL agent-creation YAML generation must go through the client's STREAMING
    path (it was a plain whole-response .complete). Drives the CONFIGURE branch directly."""
    from types import SimpleNamespace

    from localharness.cli.repl import OrchestratorREPL
    from localharness.orchestrator.workflow import WorkflowState

    calls = {"stream": 0, "complete": 0}

    class _LLMSpy:
        async def stream_complete(self, messages, tools=None, on_token=None):
            calls["stream"] += 1
            return SimpleNamespace(content="name: x"), None

        async def complete(self, messages, tools=None, stream=False):
            calls["complete"] += 1
            return SimpleNamespace(content="name: x"), None

    class _Workflow:
        gathered = {"description": "an agent that reads files", "name": "reader"}

        def transition(self, _x):
            return WorkflowState.CONFIGURE  # enter the LLM-generation branch

        def set_generated_yaml(self, _y):
            pass

    class _Channel:
        def __init__(self):
            self.messages = []

        async def send_message(self, text, metadata=None):
            self.messages.append(text)

    repl = OrchestratorREPL(
        orchestrator=SimpleNamespace(active_workflow=_Workflow()),
        agent_loop=SimpleNamespace(_llm=_LLMSpy()),
        channel=_Channel(),
        bus=SimpleNamespace(),
    )
    asyncio.run(repl._handle_creation_workflow("make me an agent that reads files"))
    assert calls["stream"] == 1
    assert calls["complete"] == 0


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
        # the fit-check (start_cmd:248) does not abort the drive; 4th slot = probe_error (#44)
        return (True, "native", 262_144, None)
    monkeypatch.setattr("localharness.cli.start_cmd._probe_llm", fake_probe)

    class _StubTokenCounter:
        # the real TokenCounter FAILS LOUD without a /tokenize server
        approximate = False

        def __init__(self, base_url=None, model=None, provider_type=None):
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


async def test_user_message_produces_topical_summary_row(tmp_path, monkeypatch):
    """TIME-01 composed spine: a repl-shaped UserMessage published on the LIVE bus during
    a real _start_async drive flows SessionAccumulator (UserMessage subscription) ->
    derive_session_summary -> end_session -> sessions row. A pure-chat delegation sitting
    (NO gate capture) closes with a zero-model topical slice leading the line — the exact
    owner UAT-2 anchor, pinned by string EQUALITY (a substring assert would miss the
    em-dash separator and the fixed pluralization)."""
    from localharness.cli.start_cmd import _start_async
    from localharness.core.events import Observation, TurnCompleted, UserMessage

    async def driving_repl_run(self):
        # agent_id must be the running agent ("orchestrator") for the accumulator filter to
        # pass; session_id is the sitting id the loop carries. publish() awaits handlers
        # inline, so the accumulator has counted these before the finally derives the summary.
        sid = self._agent.current_session_id
        await self._bus.publish(UserMessage(
            agent_id="orchestrator", session_id=sid,
            content="any fun 4th of July events near the boardwalk?", channel="terminal",
        ))
        for _ in range(3):
            await self._bus.publish(TurnCompleted(
                agent_id="orchestrator", session_id=sid, iterations=1, duration_seconds=1.0,
                elapsed_tokens=150, input_tokens=100, output_tokens=50, summary="done",
            ))
        await self._bus.publish(Observation(
            agent_id="orchestrator", session_id=sid, observation_type="tool_result",
            tool_name="agent", output="ok",
        ))

    _stub_start_boundaries(tmp_path, monkeypatch, repl_run=driving_repl_run)

    await _start_async(None, False, False, str(tmp_path))

    rows = _read_sessions(tmp_path)
    assert len(rows) == 1
    _id, _s, _e, exit_reason, summary, turn_count, action_count = rows[0]
    assert exit_reason == "complete"
    assert turn_count == 3
    assert action_count == 1
    assert summary == (
        'asked: "any fun 4th of July events near the boardwalk?" — 3 turns, 1 delegation'
    )


# ---------------------------------------------------------------------------
# Phase 33.1 (ORCH-01/02/03): upgrade drives — the rename must not cost a single memory.
# Composed proof that plan 01's store migration + Task 1's YAML migration + selection +
# gate wiring cooperate through the REAL production entry point (_start_async), memory
# side fully real (no new stubbing seams beyond _stub_start_boundaries).
# ---------------------------------------------------------------------------

async def test_upgrade_drive_migrates_legacy_root_and_gate_captures(tmp_path, monkeypatch):
    """THE end-to-end upgrade proof: a real pre-rename install — legacy default.yaml (the
    exact bytes an old mint wrote) + a default-keyed store holding a fact and an ended
    sitting — driven through the REAL _start_async comes out MIGRATED: orchestrator.yaml
    with name rewritten, default.yaml gone, the OLD fact + OLD session reachable under
    'orchestrator', AND a live gate capture during the SAME sitting lands in the new
    sitting's payload-first summary row. The (2)+(4) split is Pitfall 1's exact test —
    old memory renders AND live capture lands under the new name = no split-brain."""
    from localharness.cli.agent_cmd import _build_agent_yaml
    from localharness.cli.start_cmd import _start_async
    from localharness.core.events import MemoryGateFired, Observation, TurnCompleted
    from localharness.memory.sqlite import MemoryStore

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / "default.yaml").write_text(
        yaml.dump(_build_agent_yaml("default", "General-purpose assistant", None),
                  default_flow_style=False),
        encoding="utf-8",
    )
    legacy = MemoryStore(agent_id="default", division_id="default", org_id="default",
                         base_dir=str(tmp_path))
    await legacy.open()
    await legacy.store_fact(
        "learned/bash_exec/resolved_error", "uv fix: use .venv/bin/python", confidence=0.9,
    )
    await legacy.create_session("old-sit", {}, "dogfood", 8192)
    await legacy.end_session(
        "old-sit", exit_reason="complete",
        summary="resolved: old-sitting marker; 3 turns, 4 tool calls (bash_exec)",
        turn_count=3, action_count=4, tokens_in=100, tokens_out=50,
    )
    await legacy.close()

    async def driving_repl_run(self):
        # live capture during the migrated sitting — agent_id is the NEW root name; a stale
        # 'default' here would make the accumulator filter drop it (Pitfall 1 signature).
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

    # (1) YAML migrated: name rewritten, legacy file gone
    orch_yaml = agents_dir / "orchestrator.yaml"
    assert orch_yaml.exists()
    assert yaml.safe_load(orch_yaml.read_text(encoding="utf-8"))["name"] == "orchestrator"
    assert not (agents_dir / "default.yaml").exists()

    # (2) two rows in the orchestrator db: the migrated OLD sitting AND the new LIVE one
    rows = _read_sessions(tmp_path)
    assert len(rows) == 2
    summaries = [r[4] for r in rows]
    assert any(s and "old-sitting marker" in s for s in summaries), \
        "the migrated old sitting must be reachable under the new name"
    assert any(s and s.startswith("resolved: uv: command not found") for s in summaries), \
        "the live capture must land under the NEW name (no split-brain — Pitfall 1)"

    # (3) directory adopted: legacy data dir gone
    assert not (tmp_path / "agents" / "default").exists()

    # (4) the old fact renders under the new root (re-keyed, not just dir-aliased)
    store = MemoryStore(agent_id="orchestrator", division_id="default", org_id="default",
                        base_dir=str(tmp_path))
    await store.open()
    try:
        agent_md = (await store.load_context(index_mode=True)).agent_memory_md
    finally:
        await store.close()
    assert "uv fix: use .venv/bin/python" in agent_md


async def test_upgrade_drive_collision_keeps_legacy_root_working(tmp_path, monkeypatch):
    """ORCH-03 collision: a user's OWN pre-existing orchestrator.yaml (different content)
    blocks the rename — the migration refuses, BOTH yaml files stay byte-untouched, and the
    sitting runs under the un-migrated legacy root 'default' (selection prefers it), so the
    user keeps full memory continuity under the old name (nothing merged, nothing clobbered)."""
    from localharness.cli.agent_cmd import _build_agent_yaml
    from localharness.cli.start_cmd import _start_async
    from localharness.memory.sqlite import MemoryStore

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / "default.yaml").write_text(
        yaml.dump(_build_agent_yaml("default", "General-purpose assistant", None),
                  default_flow_style=False),
        encoding="utf-8",
    )
    (agents_dir / "orchestrator.yaml").write_text(
        yaml.dump(_build_agent_yaml("orchestrator", "My custom orchestrator", None),
                  default_flow_style=False),
        encoding="utf-8",
    )
    legacy = MemoryStore(agent_id="default", division_id="default", org_id="default",
                         base_dir=str(tmp_path))
    await legacy.open()
    await legacy.store_fact(
        "learned/bash_exec/resolved_error", "uv fix: use .venv/bin/python", confidence=0.9,
    )
    await legacy.close()

    _stub_start_boundaries(tmp_path, monkeypatch)
    await _start_async(None, False, False, str(tmp_path))

    # both yaml files untouched — never merged, never clobbered
    assert yaml.safe_load(
        (agents_dir / "default.yaml").read_text(encoding="utf-8"))["name"] == "default"
    assert yaml.safe_load(
        (agents_dir / "orchestrator.yaml").read_text(encoding="utf-8"))["role"] == \
        "My custom orchestrator"

    # the sitting ran under the LEGACY root: exactly one new row in the default db
    rows = _read_sessions(tmp_path, agent="default")
    assert len(rows) == 1

    # the fact stays reachable under the un-migrated 'default' (never re-keyed)
    store = MemoryStore(agent_id="default", division_id="default", org_id="default",
                        base_dir=str(tmp_path))
    await store.open()
    try:
        fact = await store.get_fact("learned/bash_exec/resolved_error")
    finally:
        await store.close()
    assert fact is not None and fact.agent_id == "default"


async def test_upgrade_drive_collision_never_grafts_legacy_dir_into_user_orchestrator(
    tmp_path, monkeypatch
):
    """BLOCKER 2 data-loss guard: when the YAML rename REFUSED because the user has their
    own 'orchestrator' agent, the legacy root's DATA DIR must never be adopted into that
    unrelated agent. Pre-rename install — default.yaml + a default-keyed store holding a
    fact — PLUS the user's own orchestrator.yaml and NO orchestrator data dir yet. Driving
    the real _start_async with --agent orchestrator must leave agents/default/ intact (the
    fact still under 'default'), must NOT graft it into the orchestrator store, and must
    NOT delete default.yaml — the released "nothing is merged or overwritten" guarantee."""
    from localharness.cli.agent_cmd import _build_agent_yaml
    from localharness.cli.start_cmd import _start_async
    from localharness.memory.sqlite import MemoryStore

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / "default.yaml").write_text(
        yaml.dump(_build_agent_yaml("default", "General-purpose assistant", None),
                  default_flow_style=False),
        encoding="utf-8",
    )
    # the user's OWN orchestrator agent (different content) — genuine ORCH-03 collision,
    # and it has NO data dir yet (never run), so the destination is free for a bad rename.
    (agents_dir / "orchestrator.yaml").write_text(
        yaml.dump(_build_agent_yaml("orchestrator", "My custom orchestrator", None),
                  default_flow_style=False),
        encoding="utf-8",
    )
    legacy = MemoryStore(agent_id="default", division_id="default", org_id="default",
                         base_dir=str(tmp_path))
    await legacy.open()
    await legacy.store_fact(
        "learned/bash_exec/resolved_error", "uv fix: use .venv/bin/python", confidence=0.9,
    )
    await legacy.close()

    _stub_start_boundaries(tmp_path, monkeypatch)
    await _start_async("orchestrator", False, False, str(tmp_path))

    # (1) the legacy data dir is UNTOUCHED — never adopted into the unrelated orchestrator
    assert (agents_dir / "default").is_dir(), \
        "legacy 'default' data dir was adopted into the user's orchestrator — guarantee broken"

    # (2) the old fact stays reachable under 'default', never re-keyed away
    dstore = MemoryStore(agent_id="default", division_id="default", org_id="default",
                         base_dir=str(tmp_path))
    await dstore.open()
    try:
        fact = await dstore.get_fact("learned/bash_exec/resolved_error")
    finally:
        await dstore.close()
    assert fact is not None and fact.agent_id == "default"

    # (3) the user's orchestrator store never received the legacy fact (nothing merged)
    ostore = MemoryStore(agent_id="orchestrator", division_id="default", org_id="default",
                         base_dir=str(tmp_path))
    await ostore.open()
    try:
        grafted = await ostore.get_fact("learned/bash_exec/resolved_error")
    finally:
        await ostore.close()
    assert grafted is None, \
        "legacy fact was grafted into the user's orchestrator store — nothing should be merged"

    # (4) collision-refusal state preserved: default.yaml never deleted
    assert (agents_dir / "default.yaml").exists()


async def test_agent_flag_default_redirects_to_orchestrator(tmp_path, monkeypatch):
    """ORCH-03 never a hard break: on a migrated/fresh install (only orchestrator.yaml),
    `--agent default` does NOT typer.Exit — it redirects to 'orchestrator' with a note and
    the sitting runs, landing one row in the orchestrator db."""
    from localharness.cli.agent_cmd import _build_agent_yaml
    from localharness.cli.start_cmd import _start_async

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / "orchestrator.yaml").write_text(
        yaml.dump(_build_agent_yaml("orchestrator", "General-purpose assistant", None),
                  default_flow_style=False),
        encoding="utf-8",
    )

    _stub_start_boundaries(tmp_path, monkeypatch)
    # must NOT raise typer.Exit — the redirect keeps old muscle memory / scripts working
    await _start_async("default", False, False, str(tmp_path))

    rows = _read_sessions(tmp_path)  # orchestrator db
    assert len(rows) == 1


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


# ---------------------------------------------------------------------------
# Phase 34-06 (COLL-01/02/04): the collect-only predictive gate wired into the REAL
# _start_async. PredictiveGate + UserSignalDetector open beside WriteGate at startup
# (config-gated on agent.memory.predictive_gate.enabled, soft-degrading independently)
# and close in ordered shutdown. These drives prove the composed spine end-to-end — a
# live tool-call pair lands surprise rows and a correction-worded user turn lands a
# labeled signal — the production wiring, not the unit islands 34-03/34-04 already proved.
# ---------------------------------------------------------------------------

def _read_predictive_counts(tmp_path, agent="orchestrator"):
    """(tool_observations, surprise_scores, correction-labeled user_signals) counts from
    the real memory.db the drive wrote — the three Phase-34 collect-only tables."""
    import sqlite3
    db_path = tmp_path / "agents" / agent / "memory.db"
    assert db_path.exists(), f"memory.db not created at {db_path}"
    con = sqlite3.connect(str(db_path))
    try:
        obs = con.execute("SELECT COUNT(*) FROM tool_observations").fetchone()[0]
        scores = con.execute("SELECT COUNT(*) FROM surprise_scores").fetchone()[0]
        corrections = con.execute(
            "SELECT COUNT(*) FROM user_signals WHERE signal_type = 'correction'"
        ).fetchone()[0]
        return obs, scores, corrections
    finally:
        con.close()


def _capture_start_console(monkeypatch):
    """Capture start_cmd's console output — the summary line carries the warnings list
    (`... [predictive-gate: ...]`). Returns the growing list of printed strings."""
    import localharness.cli.start_cmd as _sc
    printed: list[str] = []
    monkeypatch.setattr(
        _sc.console, "print", lambda *a, **k: printed.append(" ".join(str(x) for x in a))
    )
    return printed


async def _drive_one_tool_call_and_correction(self):
    """A scripted turn on the LIVE bus: one tool call (Action + matching tool_result
    Observation) then the correction-worded user message. agent_id is the running root so
    the collectors' agent filter passes; publish() awaits handlers inline, so every row is
    written before the drive returns and shutdown closes the store."""
    from localharness.core.events import Action, Observation, UserMessage
    sid = self._agent.current_session_id
    await self._bus.publish(Action(
        agent_id="orchestrator", session_id=sid, action_type="tool_call",
        tool_call_id="tc-1", tool_name="bash_exec",
    ))
    await self._bus.publish(Observation(
        agent_id="orchestrator", session_id=sid, observation_type="tool_result",
        tool_call_id="tc-1", tool_name="bash_exec", output="ok",
    ))
    await self._bus.publish(UserMessage(
        agent_id="orchestrator", session_id=sid,
        content="no, i meant the other file", channel="terminal",
    ))


async def test_predictive_collectors_wired(tmp_path, monkeypatch):
    """The composed spine, default-on: a live tool-call turn lands >=1 tool_observations
    row AND >=1 surprise_scores row (PredictiveGate), and the correction-worded user message
    lands >=1 user_signals row labeled 'correction' (UserSignalDetector) — proven through the
    REAL _start_async production entry point, not the unit islands."""
    from localharness.cli.start_cmd import _start_async
    _stub_start_boundaries(
        tmp_path, monkeypatch, repl_run=_drive_one_tool_call_and_correction
    )

    await _start_async(None, False, False, str(tmp_path))

    obs, scores, corrections = _read_predictive_counts(tmp_path)
    assert obs >= 1, "PredictiveGate must persist a tool_observations row for the live tool call"
    assert scores >= 1, "PredictiveGate must persist a surprise_scores row for the live tool call"
    assert corrections >= 1, "UserSignalDetector must label 'no, i meant...' as a correction"


async def test_predictive_gate_config_off(tmp_path, monkeypatch):
    """The off-switch silences everything: with agent.memory.predictive_gate.enabled=False,
    the same drive lands ZERO rows in all three tables, startup emits no predictive-gate /
    user-signals warning (the block is skipped, never caught), and the sitting still closes
    one clean sessions row — REPL behavior identical."""
    from localharness.cli.agent_cmd import _build_agent_yaml
    from localharness.cli.start_cmd import _start_async

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    data = _build_agent_yaml("orchestrator", "General-purpose assistant", None)
    data["memory"] = {"predictive_gate": {"enabled": False}}
    (agents_dir / "orchestrator.yaml").write_text(
        yaml.dump(data, default_flow_style=False), encoding="utf-8"
    )

    printed = _capture_start_console(monkeypatch)
    _stub_start_boundaries(
        tmp_path, monkeypatch, repl_run=_drive_one_tool_call_and_correction
    )

    await _start_async(None, False, False, str(tmp_path))

    obs, scores, corrections = _read_predictive_counts(tmp_path)
    assert (obs, scores, corrections) == (0, 0, 0), "the off-switch must silence all collection"

    out = "\n".join(printed)
    assert "predictive-gate" not in out and "user-signals" not in out, \
        "a disabled gate must emit no soft-degrade warning (the block is skipped, not caught)"

    # REPL behavior identical: the sitting still opens + closes one clean sessions row.
    rows = _read_sessions(tmp_path)
    assert len(rows) == 1 and rows[0][3] == "complete"


async def test_predictive_gate_soft_degrade(tmp_path, monkeypatch):
    """Independent soft-degrade (WriteGate discipline): if PredictiveGate.open() raises at
    startup, the sitting still runs — a 'predictive-gate' warning is recorded, the scorer
    subscribes nothing (zero surprise rows), but the UserSignalDetector still opens on its
    OWN try/except so the correction still lands, and shutdown is clean."""
    from localharness.cli.start_cmd import _start_async

    async def boom(self):
        raise RuntimeError("scorer wiring blew up")
    monkeypatch.setattr("localharness.memory.predictive_gate.PredictiveGate.open", boom)

    printed = _capture_start_console(monkeypatch)
    _stub_start_boundaries(
        tmp_path, monkeypatch, repl_run=_drive_one_tool_call_and_correction
    )

    await _start_async(None, False, False, str(tmp_path))

    out = "\n".join(printed)
    assert "predictive-gate" in out, "a scorer open() failure must soft-degrade with a warning"

    # the loop survives measurement failure: clean completion, one sessions row.
    rows = _read_sessions(tmp_path)
    assert len(rows) == 1 and rows[0][3] == "complete"

    # the two try/excepts are independent — the failed scorer subscribed nothing (zero
    # surprise rows) yet the signal channel opened and labeled the correction.
    obs, scores, corrections = _read_predictive_counts(tmp_path)
    assert (obs, scores) == (0, 0), "the failed scorer must subscribe nothing — no surprise rows"
    assert corrections >= 1, "user-signal detection must survive a predictive-gate open() failure"


# ---------------------------------------------------------------------------
# Phase 35-02 (PGATE-01/02/03): PredictiveWriteGate wired LIVE into the REAL
# _start_async. The gate is constructed + opened beside PredictiveGate/UserSignals
# (config-gated on predictive_gate.enabled AND write_live), soft-degrades on its own
# try/except, and closes in ordered shutdown after user_signal_detector. These drives
# prove the LIVE write path end-to-end: a reliable tool failing surprisingly on the live
# bus produces a persisted sub-0.7 fact — reachability, not object existence.
# ---------------------------------------------------------------------------


async def _drive_reliable_tool_then_surprising_failure(self):
    """Six clean bash_exec calls build a reliable prior (error_rate 0, n=6 >= min_prior_n 5),
    then ONE failure of the SAME tool. PredictiveGate scores that as quadrant
    'surprising_failure' and publishes SurpriseScored onto the live bus; the wired
    PredictiveWriteGate consumes it and writes a gated fact. publish() awaits handlers inline,
    so every write lands before shutdown closes the store."""
    from localharness.core.events import Action, Observation
    sid = self._agent.current_session_id
    for i in range(6):
        await self._bus.publish(Action(
            agent_id="orchestrator", session_id=sid, action_type="tool_call",
            tool_call_id=f"ok-{i}", tool_name="bash_exec",
        ))
        await self._bus.publish(Observation(
            agent_id="orchestrator", session_id=sid, observation_type="tool_result",
            tool_call_id=f"ok-{i}", tool_name="bash_exec", output="ok",
        ))
    # the surprising failure: a normally-reliable tool errors (is_error=1 on a ~0 prior rate)
    await self._bus.publish(Action(
        agent_id="orchestrator", session_id=sid, action_type="tool_call",
        tool_call_id="boom", tool_name="bash_exec",
    ))
    await self._bus.publish(Observation(
        agent_id="orchestrator", session_id=sid, observation_type="tool_result",
        tool_call_id="boom", tool_name="bash_exec", error="unexpected failure", exit_code=1,
    ))


def _read_predgate_facts(tmp_path, agent="orchestrator"):
    """(key, confidence, source) of the facts the PredictiveWriteGate stat channel wrote
    (key predgate/surprising_failure/...) — read straight from the real memory.db."""
    import sqlite3
    db_path = tmp_path / "agents" / agent / "memory.db"
    assert db_path.exists(), f"memory.db not created at {db_path}"
    con = sqlite3.connect(str(db_path))
    try:
        return con.execute(
            "SELECT key, confidence, source FROM facts "
            "WHERE key LIKE 'predgate/surprising_failure/%'"
        ).fetchall()
    finally:
        con.close()


async def test_predictive_write_gate_wired_and_fires(tmp_path, monkeypatch):
    """PGATE-01 end-to-end through the REAL start path: raw tool events -> PredictiveGate ->
    SurpriseScored(surprising_failure) -> the WIRED PredictiveWriteGate -> a persisted sub-0.7
    fact. Proves the gate is CONSTRUCTED, OPENED and REACHABLE on the live bus with write_live
    defaulting True — a green unit on an unwired gate would be a checkmark on a lie."""
    from localharness.cli.start_cmd import _start_async
    _stub_start_boundaries(
        tmp_path, monkeypatch, repl_run=_drive_reliable_tool_then_surprising_failure
    )

    await _start_async(None, False, False, str(tmp_path))

    facts = _read_predgate_facts(tmp_path)
    assert len(facts) >= 1, "the wired PredictiveWriteGate must write a surprising_failure fact"
    _key, confidence, source = facts[0]
    assert source == "predictive_write_gate"
    assert confidence < 0.7, "stat facts stay below the 0.7 injection gate (CLS fast-capture)"


async def test_predictive_write_gate_kill_lever_reverts_writes_keeps_telemetry(tmp_path, monkeypatch):
    """The pre-committed KILL-revert lever, end-to-end: with agent.memory.predictive_gate.
    write_live=False the SAME surprising-failure drive writes ZERO predgate facts (reverted to
    motif-only) while the collect-only scorer STILL persists surprise_scores (scores stay as
    telemetry) — the exact 'revert to motifs, keep the scores' shape the ROADMAP pre-committed."""
    from localharness.cli.agent_cmd import _build_agent_yaml
    from localharness.cli.start_cmd import _start_async

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    data = _build_agent_yaml("orchestrator", "General-purpose assistant", None)
    data["memory"] = {"predictive_gate": {"write_live": False}}
    (agents_dir / "orchestrator.yaml").write_text(
        yaml.dump(data, default_flow_style=False), encoding="utf-8"
    )

    _stub_start_boundaries(
        tmp_path, monkeypatch, repl_run=_drive_reliable_tool_then_surprising_failure
    )

    await _start_async(None, False, False, str(tmp_path))

    # writes OFF: the gate is not even constructed (guard: enabled AND write_live)
    assert _read_predgate_facts(tmp_path) == [], "write_live=False must write zero gated facts"
    # telemetry ON: the collect-only scorer still persisted surprise scores
    _obs, scores, _corr = _read_predictive_counts(tmp_path)
    assert scores >= 1, "the collect-only scorer keeps persisting scores as telemetry (KILL-revert shape)"


# ===========================================================================================
# #44: the capability probe never raises — it reports failures via CapabilityResult.probe_error,
# which _probe_llm ignored, so probe_ok was ALWAYS True. The "Cannot reach model" hard-fail, its
# retry and fallback were DEAD CODE: start proceeded against unserved models / dead endpoints and
# then MISATTRIBUTED the cause at the TokenCounter step ("exposes an exact tokenizer..."). These
# tests pin the fix: a real reachability probe_error aborts start BEFORE the memory store opens,
# with a cause-naming message; an "HTTP 400" probe_error (server rejects the tools param — reachable
# AND serving the model, detect_capabilities forces xml) still proceeds in xml mode.
# ===========================================================================================

def _cap_result(probe_error, *, tool_call_mode="xml", context_window=262_144):
    from localharness.provider.client import CapabilityResult
    return CapabilityResult(
        tool_call_mode=tool_call_mode,
        context_window=context_window,
        supports_streaming=True,
        probe_duration_ms=0.0,
        probe_error=probe_error,
    )


def _capture_err_console(monkeypatch):
    """Capture start_cmd's err_console output — the reachability failure message lands there."""
    import localharness.cli.start_cmd as _sc
    printed: list[str] = []
    monkeypatch.setattr(
        _sc.err_console, "print", lambda *a, **k: printed.append(" ".join(str(x) for x in a))
    )
    return printed


def _stub_start_realprobe(tmp_path, monkeypatch, *, probe_error, available_models=None):
    """Stub every boundary EXCEPT the probe: LLMClient.detect_capabilities returns a chosen
    CapabilityResult so the REAL _probe_llm logic runs (the code under test). Cold-start retries
    run instantly (asyncio.sleep no-op). TokenCounter/REPL/plugins are stubbed so a probe that
    PROCEEDS runs offline."""
    lines = [
        "version: '1'",
        "provider:",
        "  provider_type: vllm",
        "  base_url: http://localhost:8000/v1",
        "  default_model: test-model",
        "  api_key: none",
    ]
    if available_models:
        lines.append("  available_models: [" + ", ".join(available_models) + "]")
    (tmp_path / "config.yaml").write_text("\n".join(lines) + "\n")

    async def fake_detect(self):
        return _cap_result(probe_error)
    monkeypatch.setattr("localharness.provider.client.LLMClient.detect_capabilities", fake_detect)

    async def fast_sleep(*a, **k):
        return None
    monkeypatch.setattr("asyncio.sleep", fast_sleep)  # collapse the cold-start retry backoff

    class _StubTokenCounter:
        approximate = False

        def __init__(self, base_url=None, model=None, provider_type=None):
            pass

        def count(self, text=""):
            return max(1, len(str(text)) // 4)

        def count_messages(self, messages):
            return sum(self.count(m.get("content", "")) for m in messages)
    monkeypatch.setattr("localharness.agent.context.TokenCounter", _StubTokenCounter)

    async def default_repl_run(self):
        return None
    monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", default_repl_run)

    async def fake_discover(self):
        return []
    monkeypatch.setattr("localharness.plugins.loader.PluginLoader.discover_all", fake_discover)


def _spy_store_open(monkeypatch):
    """Record every MemoryStore.open() (call-through). Empty list == the store never opened."""
    from localharness.memory.sqlite import MemoryStore
    calls: list = []
    real_open = MemoryStore.open

    async def spy_open(self):
        calls.append(self)
        return await real_open(self)
    monkeypatch.setattr(MemoryStore, "open", spy_open)
    return calls


# --- direct _probe_llm unit tests (the fix's core) -----------------------------------------

async def test_probe_llm_reports_reachability_failure(monkeypatch):
    """_probe_llm inspects CapabilityResult.probe_error: a connection error -> (False, None, None,
    err), surfacing the concrete cause for the caller's message (was dead — the except only ever
    saw a raise, and detect_capabilities never raises)."""
    from localharness.cli.start_cmd import _probe_llm

    async def fast_sleep(*a, **k):
        return None
    monkeypatch.setattr("asyncio.sleep", fast_sleep)

    class _LLM:
        async def detect_capabilities(self):
            return _cap_result("Connection error.")

    reachable, mode, window, err = await _probe_llm(_LLM(), max_retries=2, delay=0.0)
    assert reachable is False
    assert mode is None and window is None
    assert err == "Connection error."


async def test_probe_llm_http_400_is_reachable_xml():
    """An 'HTTP 400' probe_error is NOT a reachability failure: the server rejected the tools param
    but IS reachable and serving the model. _probe_llm returns reachable=True in xml mode — else
    every function-calling-less server would be wrongly blocked from starting."""
    from localharness.cli.start_cmd import _probe_llm

    class _LLM:
        async def detect_capabilities(self):
            return _cap_result("HTTP 400: tools not supported", tool_call_mode="xml", context_window=99_000)

    reachable, mode, window, err = await _probe_llm(_LLM())
    assert reachable is True
    assert mode == "xml"
    assert window == 99_000
    assert err is None


async def test_probe_llm_clean_probe_succeeds():
    """A clean probe (probe_error=None) returns reachable=True with the probed mode + window."""
    from localharness.cli.start_cmd import _probe_llm

    class _LLM:
        async def detect_capabilities(self):
            return _cap_result(None, tool_call_mode="native", context_window=131_072)

    reachable, mode, window, err = await _probe_llm(_LLM())
    assert (reachable, mode, window, err) == (True, "native", 131_072, None)


# --- through the real _start_async: the hard-fail must fire BEFORE the store opens ----------

async def test_probe_connection_error_aborts_before_store(tmp_path, monkeypatch):
    """A connection-level probe_error (endpoint down) now aborts start BEFORE the memory store is
    constructed — the pre-resource hard-fail the dead code was supposed to be. The message names
    'unreachable' and points at doctor; it does NOT blame the tokenizer."""
    import typer
    from localharness.cli.start_cmd import _start_async

    _stub_start_realprobe(tmp_path, monkeypatch, probe_error="Connection error.")
    open_calls = _spy_store_open(monkeypatch)
    errs = _capture_err_console(monkeypatch)

    with pytest.raises(typer.Exit) as ei:
        await _start_async(None, False, False, str(tmp_path))
    assert ei.value.exit_code != 0
    assert open_calls == [], "start must abort at the probe, BEFORE opening the memory store (#44)"

    out = "\n".join(errs).lower()
    assert "unreachable" in out
    assert "doctor" in out
    assert "tokenizer" not in out, "a reachability failure must not be misattributed to the tokenizer"


async def test_probe_model_not_served_aborts_and_names_model(tmp_path, monkeypatch):
    """A 404 'model does not exist' probe_error aborts before the store and names the concrete
    cause (the unserved model) + points at `localharness model` / `localharness doctor`, with a
    best-effort list of the configured models."""
    import typer
    from localharness.cli.start_cmd import _start_async

    _stub_start_realprobe(
        tmp_path, monkeypatch,
        probe_error="Error code: 404 - {'error': {'message': 'The model `test-model` does not exist.'}}",
        available_models=["served-a", "served-b"],
    )
    open_calls = _spy_store_open(monkeypatch)
    errs = _capture_err_console(monkeypatch)

    with pytest.raises(typer.Exit) as ei:
        await _start_async(None, False, False, str(tmp_path))
    assert ei.value.exit_code != 0
    assert open_calls == [], "an unserved model must abort BEFORE the store opens (#44)"

    out = "\n".join(errs)
    assert "test-model" in out
    assert "not served" in out.lower()
    assert "localharness model" in out and "localharness doctor" in out
    assert "served-a" in out, "the message should hint the configured/served models"


async def test_probe_http_400_proceeds_in_xml_mode(tmp_path, monkeypatch):
    """The critical regression guard: an 'HTTP 400' probe_error means the server rejected the tools
    param but IS reachable and serving the model (detect_capabilities forces xml). Start must PROCEED
    — open the store, write a session row — not hard-fail. Only real reachability errors abort."""
    from localharness.cli.start_cmd import _start_async

    _stub_start_realprobe(tmp_path, monkeypatch, probe_error="HTTP 400: tools param not supported")

    await _start_async(None, False, False, str(tmp_path))  # must NOT raise

    rows = _read_sessions(tmp_path)
    assert len(rows) == 1 and rows[0][3] == "complete", \
        "a 400 tools-rejection is reachable + served — start proceeds in xml, does not hard-fail"
