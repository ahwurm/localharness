"""OrchestratorREPL box-mode coordinator — the type-anytime input box restructure.

The persistent box runs as a sibling task; submissions flow through a control queue. The
policy (submit-idle→start turn, submit-during-turn→route nudge/queue, turn_done→play FIFO,
Ctrl+C→cancel turn / arm-exit, Ctrl+D→exit) is unit-tested by driving _handle_box_event
directly, plus one end-to-end _run_with_box run against a fake box channel. No real
prompt_toolkit app and no live model here (the real box is proven live in tmux)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from localharness.cli.repl import OrchestratorREPL
from localharness.orchestrator.cards import AgentCardRegistry
from localharness.orchestrator.router import AgentCreationFlow


class FakeBoxChannel:
    """Terminal stand-in exposing the box surface without a real prompt_toolkit app."""
    channel_id = "terminal"

    def __init__(self) -> None:
        self.sent: list[tuple[str, dict]] = []
        self.ctrl_q: asyncio.Queue | None = None
        self.on_interrupt = None
        self.queued: int | None = None
        self.flashes: list[str] = []
        self.echoes: list[tuple[str, str]] = []
        self.working: list[bool] = []
        self.box_started = False
        self.box_stopped = False
        self.last_activity_summary = ""

    def can_run_input_box(self) -> bool:
        return True

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def start_input_box(self, ctrl_queue, on_interrupt) -> None:
        self.ctrl_q = ctrl_queue
        self.on_interrupt = on_interrupt
        self.box_started = True

    async def stop_input_box(self) -> None:
        self.box_stopped = True

    def box_set_queued(self, n: int) -> None:
        self.queued = n

    def box_flash_decision(self, text: str, seconds: float = 2.0) -> None:
        self.flashes.append(text)

    async def box_echo_prompt(self, text: str, annotation: str = "") -> None:
        self.echoes.append((text, annotation))

    def box_notify_working(self, working: bool) -> None:
        self.working.append(working)

    async def send_message(self, text: str, agent_id=None, metadata=None) -> None:
        self.sent.append((text, metadata or {}))


def _repl(channel=None, harness=None):
    channel = channel or FakeBoxChannel()
    orch = AgentCreationFlow(AgentCardRegistry())
    agent = MagicMock()
    agent._config.name = "orchestrator"
    agent.current_session_id = "s1"
    agent._llm = MagicMock(spec=[])  # no .complete -> tier-2 unavailable (unit tests use tier-1)
    agent.run_turn = AsyncMock(return_value="done")
    agent.push_user_nudge = MagicMock()
    bus = AsyncMock()
    repl = OrchestratorREPL(
        orchestrator=orch, agent_loop=agent, channel=channel, bus=bus,
        harness_config=harness,
    )
    repl._box_ctrl_q = asyncio.Queue()
    return repl, channel, agent


async def _pending_turn(repl):
    repl._turn_task = asyncio.ensure_future(asyncio.sleep(3600))
    repl._current_task = "the running task"


class TestSubmitPolicy:
    async def test_idle_submit_starts_turn(self):
        repl, ch, agent = _repl()
        await repl._handle_box_event("submit", "index the repo")
        assert repl._turn_task is not None
        agent.run_turn.assert_called_once()
        assert agent.run_turn.call_args.kwargs.get("task") == "index the repo"
        repl._turn_task.cancel()

    async def test_submit_during_turn_tier1_nudge(self):
        repl, ch, agent = _repl()
        await _pending_turn(repl)
        try:
            await repl._handle_box_event("submit", "stop, wrong file")
            agent.push_user_nudge.assert_called_once_with("stop, wrong file")
            assert any("nudging" in f for f in ch.flashes)
            assert not repl._fifo
        finally:
            repl._turn_task.cancel()

    async def test_submit_during_turn_tier1_queue(self):
        repl, ch, agent = _repl()
        await _pending_turn(repl)
        try:
            await repl._handle_box_event("submit", "also update the changelog")
            agent.push_user_nudge.assert_not_called()
            assert list(repl._fifo) == ["also update the changelog"]
            assert ch.queued == 1
            assert any("queued (1)" in f for f in ch.flashes)
        finally:
            repl._turn_task.cancel()

    async def test_force_bang_nudges_and_strips(self):
        repl, ch, agent = _repl()
        await _pending_turn(repl)
        try:
            await repl._handle_box_event("submit", "!keep going anyway")
            agent.push_user_nudge.assert_called_once_with("keep going anyway")
        finally:
            repl._turn_task.cancel()

    async def test_slash_during_turn_is_queued_not_routed(self):
        repl, ch, agent = _repl()
        await _pending_turn(repl)
        try:
            await repl._handle_box_event("submit", "/help")
            agent.push_user_nudge.assert_not_called()
            assert list(repl._fifo) == ["/help"]
        finally:
            repl._turn_task.cancel()


class TestPromptEcho:
    """FIX 1: the coordinator echoes every submission into the scrollback the moment it is
    routed — plain at turn start, annotated (queued (N) / → nudge) mid-turn — so the typed
    prompt persists in the transcript instead of vanishing when the box resets its buffer."""

    async def test_idle_submit_echoes_plain_prompt(self):
        repl, ch, agent = _repl()
        await repl._handle_box_event("submit", "index the repo")
        assert ("index the repo", "") in ch.echoes
        repl._turn_task.cancel()

    async def test_mid_turn_queue_echoes_with_queued_annotation(self):
        repl, ch, agent = _repl()
        await _pending_turn(repl)
        try:
            await repl._handle_box_event("submit", "also update the changelog")
            assert ("also update the changelog", "queued (1)") in ch.echoes
        finally:
            repl._turn_task.cancel()

    async def test_mid_turn_nudge_echoes_with_nudge_annotation(self):
        repl, ch, agent = _repl()
        await _pending_turn(repl)
        try:
            await repl._handle_box_event("submit", "stop, wrong file")
            assert ("stop, wrong file", "→ nudge") in ch.echoes
        finally:
            repl._turn_task.cancel()

    async def test_force_bang_echoes_clean_text_not_the_bang(self):
        repl, ch, agent = _repl()
        await _pending_turn(repl)
        try:
            await repl._handle_box_event("submit", "!keep going anyway")
            assert ("keep going anyway", "→ nudge") in ch.echoes
        finally:
            repl._turn_task.cancel()

    async def test_queue_playback_re_echoes_prompt_at_turn_start(self):
        repl, ch, agent = _repl()
        repl._turn_task = None
        repl._fifo.extend(["first queued"])
        done = asyncio.ensure_future(asyncio.sleep(0))
        await done
        await repl._handle_box_event("turn_done", done)
        # the queued message re-appears (plain) as its own turn begins → chronological transcript
        assert ("first queued", "") in ch.echoes
        assert repl._turn_task is not None
        repl._turn_task.cancel()


class TestQueuePlayback:
    async def test_turn_done_plays_fifo_in_order(self):
        repl, ch, agent = _repl()
        repl._turn_task = None
        repl._fifo.extend(["first queued", "second queued"])
        done = asyncio.ensure_future(asyncio.sleep(0))
        await done
        await repl._handle_box_event("turn_done", done)
        agent.run_turn.assert_called_once()
        assert agent.run_turn.call_args.kwargs.get("task") == "first queued"
        assert list(repl._fifo) == ["second queued"]
        assert repl._turn_task is not None
        repl._turn_task.cancel()

    async def test_turn_done_with_empty_fifo_returns_to_idle(self):
        repl, ch, agent = _repl()
        repl._turn_task = None
        done = asyncio.ensure_future(asyncio.sleep(0))
        await done
        cont = await repl._handle_box_event("turn_done", done)
        assert cont is True
        assert repl._turn_task is None
        agent.run_turn.assert_not_called()


class TestInterruptAndEof:
    async def test_idle_ctrl_c_arms_then_exits(self):
        repl, ch, agent = _repl()
        repl._turn_task = None
        cont = await repl._handle_box_event("interrupt", None)
        assert cont is True and repl._sigint_armed is True
        assert any("Ctrl+C again" in t for t, _ in ch.sent)
        cont2 = await repl._handle_box_event("interrupt", None)
        assert cont2 is False

    async def test_eof_exits(self):
        repl, ch, agent = _repl()
        assert await repl._handle_box_event("eof", None) is False

    async def test_on_box_interrupt_cancels_active_turn(self):
        repl, ch, agent = _repl()
        await _pending_turn(repl)
        repl._on_box_interrupt()
        assert repl._cancelled_by_user is True
        await asyncio.sleep(0)
        assert repl._turn_task.cancelled()

    async def test_on_box_interrupt_idle_enqueues_interrupt(self):
        repl, ch, agent = _repl()
        repl._turn_task = None
        repl._on_box_interrupt()
        kind, _payload = repl._box_ctrl_q.get_nowait()
        assert kind == "interrupt"

    async def test_cancelled_turn_prints_turn_cancelled(self):
        repl, ch, agent = _repl()

        async def hanging(task=None, on_token=None):
            await asyncio.sleep(3600)

        agent.run_turn = AsyncMock(side_effect=hanging)
        await repl._handle_box_event("submit", "long running task")
        task = repl._turn_task
        repl._on_box_interrupt()  # user Ctrl+C
        # deliver the turn_done the done-callback would post
        await asyncio.sleep(0)
        await repl._handle_box_event("turn_done", task)
        assert any("Turn cancelled" in t for t, _ in ch.sent)


class TestRunWithBoxIntegration:
    async def test_end_to_end_start_turn_stop(self):
        repl, ch, agent = _repl()
        repl._box_ctrl_q = None  # _run_with_box creates its own
        run_task = asyncio.ensure_future(repl._run_with_box())
        for _ in range(100):
            if ch.ctrl_q is not None:
                break
            await asyncio.sleep(0.01)
        assert ch.box_started, "box must be started"

        ch.ctrl_q.put_nowait(("submit", "hello world"))
        await asyncio.sleep(0.05)
        agent.run_turn.assert_called()
        assert agent.run_turn.call_args.kwargs.get("task") == "hello world"

        ch.ctrl_q.put_nowait(("eof", None))
        await asyncio.wait_for(run_task, timeout=2.0)
        assert ch.box_stopped, "box must be stopped on exit"
