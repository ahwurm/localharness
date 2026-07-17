"""#93 — REPL exit must not kill a turn mid-finalization.

Exiting box mode (Ctrl+D / double-Ctrl+C) right after an answer could cancel the in-flight turn
between its TaskComplete render and its TurnCompleted publish, leaving a turn with heartbeats but
no completion in the ledger (and no turn-end micro-pass). The coordinator's exit path must give an
in-flight turn a BOUNDED grace to reach its own finalization before cancelling — and never hang.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from localharness.cli.repl import OrchestratorREPL
from localharness.orchestrator.cards import AgentCardRegistry
from localharness.orchestrator.router import AgentCreationFlow


class FakeBoxChannel:
    channel_id = "terminal"

    def __init__(self) -> None:
        self.sent: list = []
        self.ctrl_q: asyncio.Queue | None = None
        self.on_interrupt = None
        self.box_started = False
        self.box_stopped = False
        self.last_activity_summary = ""

    def can_run_input_box(self) -> bool:
        return True

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def start_input_box(self, ctrl_queue, on_interrupt) -> None:
        self.ctrl_q = ctrl_queue
        self.on_interrupt = on_interrupt
        self.box_started = True

    async def stop_input_box(self) -> None:
        self.box_stopped = True

    def box_set_queued(self, n: int) -> None: ...

    def box_flash_decision(self, text: str, seconds: float = 2.0) -> None: ...

    async def box_echo_prompt(self, text: str, annotation: str = "") -> None: ...

    def box_notify_working(self, working: bool) -> None: ...

    async def send_message(self, text: str, agent_id=None, metadata=None) -> None:
        self.sent.append((text, metadata or {}))


def _repl():
    orch = AgentCreationFlow(AgentCardRegistry())
    agent = MagicMock()
    agent._config.name = "orchestrator"
    agent.current_session_id = "s1"
    agent._llm = MagicMock(spec=[])
    agent.push_user_nudge = MagicMock()
    bus = AsyncMock()
    ch = FakeBoxChannel()
    repl = OrchestratorREPL(orchestrator=orch, agent_loop=agent, channel=ch, bus=bus)
    return repl, ch, agent


async def _drive_until_box(ch):
    for _ in range(400):
        if ch.ctrl_q is not None:
            return
        await asyncio.sleep(0.005)
    raise AssertionError("input box never started")


@pytest.mark.asyncio
async def test_exit_grace_finalizes_inflight_turn():
    """A turn still finalizing when the user exits reaches completion (grace-waited, not cancelled)."""
    repl, ch, agent = _repl()
    repl._box_ctrl_q = None
    repl._exit_grace_seconds = 1.0
    completed = asyncio.Event()

    async def finishing(task=None, on_token=None):
        await asyncio.sleep(0.15)  # finalization window (answer rendered, completion pending)
        completed.set()
        return "done"

    agent.run_turn = AsyncMock(side_effect=finishing)
    run_task = asyncio.ensure_future(repl._run_with_box())
    await _drive_until_box(ch)

    ch.ctrl_q.put_nowait(("submit", "do it"))
    await asyncio.sleep(0.02)  # turn starts, not yet finished
    ch.ctrl_q.put_nowait(("eof", None))  # exit while the turn is finalizing

    await asyncio.wait_for(run_task, timeout=3.0)
    assert completed.is_set(), "exit must grace-wait the finalizing turn to completion"
    assert ch.box_stopped


@pytest.mark.asyncio
async def test_exit_does_not_hang_on_stuck_turn():
    """The grace is bounded — a wedged turn is cancelled, exit never hangs indefinitely."""
    repl, ch, agent = _repl()
    repl._box_ctrl_q = None
    repl._exit_grace_seconds = 0.1

    async def hanging(task=None, on_token=None):
        await asyncio.sleep(3600)

    agent.run_turn = AsyncMock(side_effect=hanging)
    run_task = asyncio.ensure_future(repl._run_with_box())
    await _drive_until_box(ch)

    ch.ctrl_q.put_nowait(("submit", "do it"))
    await asyncio.sleep(0.02)
    ch.ctrl_q.put_nowait(("eof", None))

    await asyncio.wait_for(run_task, timeout=3.0)  # must return within grace + margin
    assert ch.box_stopped
    assert repl._turn_task is None or repl._turn_task.cancelled() or repl._turn_task.done()
