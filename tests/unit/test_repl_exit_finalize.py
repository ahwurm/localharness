"""#93 — REPL exit must not kill a turn mid-finalization.

Exiting box mode (Ctrl+D / double-Ctrl+C) right after an answer could cancel the in-flight turn
between its TaskComplete render and its TurnCompleted publish, leaving a turn with heartbeats but
no completion in the ledger (and no turn-end micro-pass). The coordinator's exit path must give an
in-flight turn a BOUNDED grace to reach its own finalization before cancelling — and never hang.
"""
from __future__ import annotations

import asyncio
import sys
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
        self.calls: list[str] = []

    def can_run_input_box(self) -> bool:
        return True

    async def start(self) -> None: ...

    async def stop(self) -> None:
        self.calls.append("stop")

    def begin_stop(self) -> None:
        self.calls.append("begin_stop")

    async def start_input_box(self, ctrl_queue, on_interrupt) -> None:
        self.ctrl_q = ctrl_queue
        self.on_interrupt = on_interrupt
        self.box_started = True

    async def stop_input_box(self) -> None:
        self.calls.append("stop_input_box")
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


@pytest.mark.asyncio
async def test_begin_stop_comes_first_in_the_box_exit():
    repl, ch, _agent = _repl()
    drain = repl._drain_turn_on_exit

    async def _recording_drain():
        ch.calls.append("drain")
        await drain()

    repl._drain_turn_on_exit = _recording_drain
    run_task = asyncio.ensure_future(repl._run_with_box())
    await _drive_until_box(ch)
    ch.ctrl_q.put_nowait(("eof", None))
    await asyncio.wait_for(run_task, timeout=3.0)
    assert ch.calls == ["begin_stop", "drain", "stop_input_box", "stop"]


@pytest.mark.asyncio
async def test_begin_stop_is_per_instance():
    from tests.unit.test_picker import _fake_box
    from tests.unit.test_terminal_permission_prompt import _channel

    stopped, other = _channel(), _channel()
    calls: list[str] = []
    _fake_box(other, calls)
    stopped.begin_stop()
    async with other.suspended_box():
        pass
    assert (stopped._stopping, other._stopping) == (True, False)
    assert calls == ["stop", "start('queue', 'interrupt')"]


async def _box_session_ended_mid_legend(tmp_path, *, answer, grace, end, observed=None):
    """A real box session whose turn is parked on a permission legend when `end(run_task)` ends it."""
    from io import StringIO
    from types import SimpleNamespace

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from rich.console import Console

    from localharness.channels.terminal import TERMINAL_THEME, TerminalChannel
    from localharness.core.bus import EventBus
    from tests.unit.test_terminal_permission_prompt import _request

    def _console():
        return Console(file=StringIO(), force_terminal=True, width=120, theme=TERMINAL_THEME, highlight=False)

    ch = TerminalChannel(EventBus(), {}, history_file=str(tmp_path / "hist"))
    ch._console, ch._err_console = _console(), _console()
    starts: list[int] = []
    real_start = ch.start_input_box

    async def _counting_start(*args):
        starts.append(1)
        await real_start(*args)

    ch.start_input_box = _counting_start
    seen: dict = {"decision": None, "turn": None, "cancelled": False}

    async def run_turn(task=None, on_token=None):
        seen["turn"] = asyncio.current_task()
        try:
            seen["decision"] = (await ch.ask_permission(_request())).kind
        except asyncio.CancelledError:
            seen["cancelled"] = True
            raise

    orch = AgentCreationFlow(AgentCardRegistry())
    agent = MagicMock()
    agent._config.name = "orchestrator"
    agent.current_session_id = "s1"
    agent._llm = MagicMock(spec=[])
    agent.run_turn = run_turn
    repl = OrchestratorREPL(
        orchestrator=orch, agent_loop=agent, channel=ch, bus=AsyncMock(), gate=SimpleNamespace(pending={})
    )
    repl._exit_grace_seconds = grace
    before = sys.stdout
    observed = {} if observed is None else observed
    try:
        with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
            run_task = asyncio.ensure_future(repl._run_with_box())
            for _ in range(400):
                if ch._box_active:
                    break
                await asyncio.sleep(0.005)
            inp.send_text("go\r")
            for _ in range(400):
                if ch._suspended:
                    break
                await asyncio.sleep(0.005)
            end(run_task)
            if answer:
                asyncio.get_running_loop().call_later(0.05, inp.send_text, answer)
            await asyncio.wait_for(asyncio.gather(run_task, return_exceptions=True), timeout=10.0)
    finally:
        if seen["turn"] is not None:
            await asyncio.wait_for(asyncio.gather(seen["turn"], return_exceptions=True), timeout=10.0)
        observed.update(
            seen, starts=len(starts), box_task=ch._box_task, box_patch=ch._box_patch,
            stdout_restored=sys.stdout is before,
        )
        await ch.stop_input_box()
        sys.stdout = before
    return observed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer,grace,decision",
    [("y", 2.0, "allow_once"), ("n", 2.0, "reject_once"), (None, 0.2, None)],
    ids=["yes-inside-the-grace", "no-inside-the-grace", "nothing-until-the-grace-cancels"],
)
async def test_grace_never_restarts_the_box_into_a_stopping_session(tmp_path, answer, grace, decision):
    seen = await _box_session_ended_mid_legend(
        tmp_path, answer=answer, grace=grace, end=lambda run_task: run_task.cancel()
    )
    assert seen["decision"] == decision
    assert seen["cancelled"] is (decision is None)
    assert seen["starts"] == 1
    assert seen["box_task"] is None and seen["box_patch"] is None
    assert seen["stdout_restored"]


@pytest.mark.usefixtures("default_sigint")
@pytest.mark.parametrize(
    "answer,grace,decision",
    [("y", 2.0, "allow_once"), ("n", 2.0, "reject_once"), (None, 0.2, None)],
    ids=["yes-inside-the-grace", "no-inside-the-grace", "nothing-until-the-grace-cancels"],
)
def test_an_external_sigint_mid_legend_ends_the_session_and_leaves_no_box(tmp_path, answer, grace, decision):
    import os
    import signal

    seen: dict = {}
    before = sys.stdout
    try:
        with pytest.raises(KeyboardInterrupt):
            asyncio.run(_box_session_ended_mid_legend(
                tmp_path, answer=answer, grace=grace, observed=seen,
                end=lambda _run_task: os.kill(os.getpid(), signal.SIGINT),
            ))
    finally:
        sys.stdout = before
    assert seen["decision"] == decision
    assert seen["cancelled"] is (decision is None)
    assert seen["starts"] == 1
    assert seen["box_task"] is None and seen["box_patch"] is None
    assert seen["stdout_restored"]
