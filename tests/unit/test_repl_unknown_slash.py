"""#48: an unknown slash command must be rejected client-side, deterministically, with
ZERO LLM calls — never fall through to the orchestrator as ordinary chat.

Drives the real OrchestratorREPL.run() loop, faking only the process boundaries
(scripted channel that records (text, metadata), MagicMock agent, AsyncMock bus).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from localharness.cli.repl import OrchestratorREPL
from localharness.orchestrator.cards import AgentCardRegistry
from localharness.orchestrator.router import AgentCreationFlow


class RecordingChannel:
    """Terminal stand-in: scripted inputs, records every (text, metadata) sent."""

    channel_id = "terminal"

    def __init__(self, inputs: list) -> None:
        self._inputs = list(inputs)
        self.sent: list[tuple[str, dict]] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def read_input(self, prompt: str = ">") -> str:
        if not self._inputs:
            raise EOFError()
        return self._inputs.pop(0)

    async def send_message(self, text: str, agent_id=None, metadata=None) -> None:
        self.sent.append((text, metadata or {}))


def _build_repl(channel: RecordingChannel, run_turn=None):
    orch = AgentCreationFlow(AgentCardRegistry())
    agent = MagicMock()
    agent._config.name = "orchestrator"
    agent.current_session_id = "s1"
    agent._llm = MagicMock()
    agent.run_turn = run_turn or AsyncMock()
    bus = AsyncMock()
    repl = OrchestratorREPL(
        orchestrator=orch, agent_loop=agent, channel=channel, bus=bus,
    )
    return repl, agent, bus


def test_unknown_slash_rejected_deterministically_no_llm():
    channel = RecordingChannel(["/frobnicate"])
    repl, agent, bus = _build_repl(channel)

    asyncio.run(repl.run())

    # Deterministic rejection naming the command + pointing at /help.
    hits = [(t, m) for t, m in channel.sent if "Unknown command: /frobnicate" in t]
    assert hits, f"no deterministic rejection; sent={channel.sent}"
    text, meta = hits[0]
    assert "/help" in text
    assert meta.get("style") == "system.error"
    # The whole point of #48: NO generation turn, and not published as a user message.
    agent.run_turn.assert_not_called()
    bus.publish.assert_not_called()


def test_known_slash_commands_unaffected():
    channel = RecordingChannel(["/help"])
    repl, agent, bus = _build_repl(channel)

    asyncio.run(repl.run())

    assert any("Available commands" in t for t, _ in channel.sent)
    assert not any("Unknown command" in t for t, _ in channel.sent)
    agent.run_turn.assert_not_called()


def test_path_like_slash_still_reaches_the_agent():
    # A legit message that happens to start with '/': more than a single token (it has
    # spaces and extra slashes) so it is NOT claimed as a command — it reaches the agent.
    channel = RecordingChannel(["/tmp/foo whats this file"])
    repl, agent, bus = _build_repl(channel)

    asyncio.run(repl.run())

    assert not any("Unknown command" in t for t, _ in channel.sent)
    agent.run_turn.assert_called_once()
    assert agent.run_turn.call_args.kwargs.get("task") == "/tmp/foo whats this file"


def test_bare_slash_and_multislash_path_fall_through():
    # Bare '/' and a pure path token ('/tmp/foo') both fall through (not a single /word).
    for raw in ("/", "/tmp/foo"):
        channel = RecordingChannel([raw])
        repl, agent, bus = _build_repl(channel)
        asyncio.run(repl.run())
        assert not any("Unknown command" in t for t, _ in channel.sent), raw
        agent.run_turn.assert_called_once()
        assert agent.run_turn.call_args.kwargs.get("task") == raw
