"""#92c — mid-turn tier-2 routing is NON-BLOCKING with a late nudge-upgrade.

Because a contended tier-2 classify can now legitimately take up to ~30s (permit-wait) it must NOT
block the single-threaded box coordinator. So a tier-1-abstaining mid-turn message is optimistically
QUEUED immediately (owner principle: uncertain/late -> queue) and classified in the BACKGROUND. When
the verdict resolves it funnels back through the ONE control queue: a NUDGE upgrades the message to
a live nudge ONLY if it is still queued (not yet dispatched); otherwise the queue stands.

Also pins the two internal-call fixes on the tier-2 completion seam: disable_thinking + gen_timeout.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest

from localharness.channels import input_router
from localharness.cli.repl import OrchestratorREPL
from localharness.orchestrator.cards import AgentCardRegistry
from localharness.orchestrator.router import AgentCreationFlow


class FakeBoxChannel:
    channel_id = "terminal"

    def __init__(self) -> None:
        self.sent: list = []
        self.queued: int | None = None
        self.flashes: list[str] = []
        self.echoes: list = []
        self.last_activity_summary = ""

    async def start(self): ...
    async def stop(self): ...
    def box_set_queued(self, n): self.queued = n
    def box_flash_decision(self, text, seconds: float = 2.0): self.flashes.append(text)
    async def box_echo_prompt(self, text, annotation: str = ""): self.echoes.append((text, annotation))
    def box_notify_working(self, working): ...
    async def send_message(self, text, agent_id=None, metadata=None): self.sent.append((text, metadata or {}))


def _repl(complete_content: str | None = None):
    orch = AgentCreationFlow(AgentCardRegistry())
    agent = MagicMock()
    agent._config.name = "orchestrator"
    agent.current_session_id = "s1"
    agent.push_user_nudge = MagicMock()
    if complete_content is not None:
        async def _complete(messages, tools=None, stream=False, disable_thinking=False, gen_timeout=None):
            return NS(content=complete_content), None
        agent._llm = NS(complete=_complete)
    else:
        agent._llm = MagicMock(spec=[])
    bus = AsyncMock()
    ch = FakeBoxChannel()
    repl = OrchestratorREPL(orchestrator=orch, agent_loop=agent, channel=ch, bus=bus)
    repl._box_ctrl_q = asyncio.Queue()
    return repl, ch, agent


async def _pending_turn(repl):
    repl._turn_task = asyncio.ensure_future(asyncio.sleep(3600))
    repl._current_task = "the running task"


async def test_tier2_abstain_optimistically_queues_immediately():
    """A tier-1-abstaining mid-turn message queues at once (no blocking on the classify)."""
    repl, ch, agent = _repl(complete_content="NUDGE")
    await _pending_turn(repl)
    try:
        await repl._handle_box_event("submit", "index the tests directory")
        assert list(repl._fifo) == ["index the tests directory"]  # optimistic queue, right now
        agent.push_user_nudge.assert_not_called()
        assert ch.queued == 1
    finally:
        repl._turn_task.cancel()


async def test_tier2_nudge_verdict_upgrades_still_queued_message():
    """The background NUDGE verdict pulls the still-queued message out and delivers it as a nudge."""
    repl, ch, agent = _repl(complete_content="NUDGE")
    await _pending_turn(repl)
    try:
        await repl._handle_box_event("submit", "index the tests directory")
        kind, payload = await asyncio.wait_for(repl._box_ctrl_q.get(), timeout=2.0)
        assert kind == "tier2_result"
        await repl._handle_box_event(kind, payload)
        assert not repl._fifo  # pulled from the queue …
        agent.push_user_nudge.assert_called_once_with("index the tests directory")  # … and nudged
    finally:
        repl._turn_task.cancel()


async def test_tier2_queue_verdict_leaves_it_queued():
    """A QUEUE verdict is a no-op upgrade — the optimistic queue simply stands."""
    repl, ch, agent = _repl(complete_content="QUEUE")
    await _pending_turn(repl)
    try:
        await repl._handle_box_event("submit", "index the tests directory")
        kind, payload = await asyncio.wait_for(repl._box_ctrl_q.get(), timeout=2.0)
        await repl._handle_box_event(kind, payload)
        assert list(repl._fifo) == ["index the tests directory"]
        agent.push_user_nudge.assert_not_called()
    finally:
        repl._turn_task.cancel()


async def test_tier2_nudge_not_applied_after_dispatch():
    """If the message was already dispatched (drained from the queue) before the verdict lands, the
    late NUDGE does NOT retroactively steer — the queue/dispatch decision stands."""
    repl, ch, agent = _repl(complete_content="NUDGE")
    await _pending_turn(repl)
    try:
        await repl._handle_box_event("submit", "index the tests directory")
        repl._fifo.clear()  # simulate: the turn ended and this message was played as its own turn
        kind, payload = await asyncio.wait_for(repl._box_ctrl_q.get(), timeout=2.0)
        await repl._handle_box_event(kind, payload)
        agent.push_user_nudge.assert_not_called()
    finally:
        repl._turn_task.cancel()


async def test_tier2_complete_fn_disables_thinking_and_bounds_generation():
    """The tier-2 completion seam is an INTERNAL call: disable_thinking=True and a 5s gen_timeout."""
    repl, ch, agent = _repl()
    seen = {}

    async def _complete(messages, tools=None, stream=False, disable_thinking=False, gen_timeout=None):
        seen.update(disable_thinking=disable_thinking, gen_timeout=gen_timeout)
        return NS(content="NUDGE"), None

    agent._llm = NS(complete=_complete)
    fn = repl._tier2_complete_fn()
    out = await fn([{"role": "user", "content": "x"}])
    assert out == "NUDGE"
    assert seen["disable_thinking"] is True
    assert seen["gen_timeout"] == 5.0
