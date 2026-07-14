"""#47: Ctrl+C DURING a turn must cancel the turn and return to the prompt — the SESSION
survives. (Before the fix a mid-turn SIGINT raised KeyboardInterrupt out of the loop and
the whole session exited with 'Goodbye.'.)

We can't deliver a real SIGINT deterministically in a unit test, so we capture the
loop-level SIGINT handler the REPL installs for the turn's duration and invoke it while a
(hanging) turn is in flight — exactly what the real signal callback does.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from localharness.cli.repl import OrchestratorREPL
from localharness.orchestrator.cards import AgentCardRegistry
from localharness.orchestrator.router import AgentCreationFlow


class RecordingChannel:
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


def _build_repl(channel, run_turn):
    orch = AgentCreationFlow(AgentCardRegistry())
    agent = MagicMock()
    agent._config.name = "orchestrator"
    agent.current_session_id = "s1"
    agent._llm = MagicMock()
    agent.run_turn = run_turn
    bus = AsyncMock()
    repl = OrchestratorREPL(
        orchestrator=orch, agent_loop=agent, channel=channel, bus=bus,
    )
    return repl, agent, bus


def test_ctrl_c_during_turn_cancels_turn_not_session():
    turn_started = asyncio.Event()
    cancelled_inside = False

    async def hanging_turn(task, on_token=None):
        turn_started.set()
        try:
            await asyncio.sleep(100)  # in-flight; hangs until cancelled
        except asyncio.CancelledError:
            nonlocal cancelled_inside
            cancelled_inside = True
            raise

    channel = RecordingChannel(["do a long task"])  # then EOFError ends the loop
    repl, agent, bus = _build_repl(channel, hanging_turn)

    async def scenario():
        loop = asyncio.get_running_loop()
        captured: dict = {}
        loop.add_signal_handler = lambda sig, cb, *a: captured.__setitem__("cb", cb)
        loop.remove_signal_handler = lambda sig: None

        run_task = asyncio.ensure_future(repl.run())
        await asyncio.wait_for(turn_started.wait(), timeout=2.0)
        assert "cb" in captured, "REPL must install a loop SIGINT handler for the turn"
        captured["cb"]()  # simulate the mid-turn Ctrl+C
        await asyncio.wait_for(run_task, timeout=2.0)  # loop survives → EOF → returns

    asyncio.run(scenario())

    assert cancelled_inside, "the in-flight turn task must actually be cancelled"
    assert any("Turn cancelled" in t for t, _ in channel.sent), channel.sent
    # Truthful, non-error styling and the session did NOT exit via KeyboardInterrupt.
    assert any(
        "Turn cancelled" in t and m.get("style") == "system.info"
        for t, m in channel.sent
    )


def test_second_ctrl_c_restores_default_sigint_escape_hatch():
    turn_started = asyncio.Event()

    async def hanging_turn(task, on_token=None):
        turn_started.set()
        await asyncio.sleep(100)

    channel = RecordingChannel(["do a long task"])
    repl, agent, bus = _build_repl(channel, hanging_turn)

    async def scenario():
        loop = asyncio.get_running_loop()
        captured: dict = {}
        removed: list = []
        loop.add_signal_handler = lambda sig, cb, *a: captured.__setitem__("cb", cb)
        loop.remove_signal_handler = lambda sig: removed.append(sig)

        run_task = asyncio.ensure_future(repl.run())
        await asyncio.wait_for(turn_started.wait(), timeout=2.0)
        cb = captured["cb"]
        cb()  # 1st Ctrl+C — cancels the turn
        cb()  # 2nd Ctrl+C — escape hatch: restore default SIGINT so a further one hard-exits
        await asyncio.wait_for(run_task, timeout=2.0)
        return removed

    removed = asyncio.run(scenario())
    import signal
    assert signal.SIGINT in removed  # the handler was removed (default restored)


@pytest.mark.asyncio
async def test_normal_turn_completes_without_cancel_message():
    # Sanity: with no interrupt the turn completes and no 'Turn cancelled.' is emitted.
    channel = RecordingChannel(["hello"])
    repl, agent, bus = _build_repl(channel, AsyncMock(return_value="done"))
    await repl.run()
    assert not any("Turn cancelled" in t for t, _ in channel.sent)
    agent.run_turn.assert_awaited_once()
