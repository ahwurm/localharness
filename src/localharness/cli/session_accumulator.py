"""Sitting-scoped counters + the payload-first session-summary line (SESS-02/05).

Mirrors bench.runner.MetricAccumulator (bus-subscribed counters) and the
WriteGate open/close subscription lifecycle. Zero model calls by construction:
the summary is DERIVED from signals the bus already carries — the gate's capture
details are already payload-first (gate.py composes them at capture time), so
the hard problem is solved upstream; this module just surfaces it.

KILL guardrail (SESS-05, pre-committed): a sitting with nothing discriminating
(no tool use, no gate capture) yields None — the history shelf stays suppressed
rather than gaining a "worked on stuff" line.
"""
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from localharness.core.bus import EventBus, SubscriptionHandle

_LINE_BUDGET = 180   # matches the index-line budget (5192f27) and the flush cap (33-04)
_DETAIL_BUDGET = 120  # leave room for the counts tail

_TIER_LEAD = {"resolved_error": "resolved: ", "stuck_recovered": "unstuck: "}


class SessionAccumulator:
    """Bus-subscribed sitting counters. Agent-id filtered; zero model calls."""

    def __init__(self, bus: "EventBus", agent_id: str) -> None:
        self._bus = bus
        self._agent_id = agent_id
        self._handles: list["SubscriptionHandle"] = []
        self.turn_count = 0
        self.action_count = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.tools_used: Counter[str] = Counter()
        self.captures: list[tuple[str, str]] = []  # (tier, detail)

    async def open(self) -> None:
        from localharness.core.events import (
            MemoryGateFired,
            Observation,
            TurnCompleted,
            TurnFailed,
        )
        sub = self._bus.subscribe
        self._handles += [
            sub(TurnCompleted, self.on_turn_completed, agent_id=self._agent_id),
            sub(TurnFailed, self.on_turn_failed, agent_id=self._agent_id),
            sub(Observation, self.on_observation, agent_id=self._agent_id),
            sub(MemoryGateFired, self.on_gate_fired, agent_id=self._agent_id),
        ]

    async def close(self) -> None:
        for h in self._handles:
            self._bus.unsubscribe(h)
        self._handles.clear()

    async def on_turn_completed(self, event) -> None:
        self.turn_count += 1
        self.tokens_in += int(getattr(event, "input_tokens", 0) or 0)
        self.tokens_out += int(getattr(event, "output_tokens", 0) or 0)

    async def on_turn_failed(self, event) -> None:
        await self.on_turn_completed(event)  # failed turns still count + spend tokens

    async def on_observation(self, event) -> None:
        if event.observation_type == "tool_result" and event.tool_name:
            self.action_count += 1
            self.tools_used[event.tool_name] += 1

    async def on_gate_fired(self, event) -> None:
        self.captures.append((event.tier, event.detail))


def derive_session_summary(acc: Optional[SessionAccumulator]) -> str | None:
    """Payload-first or nothing. Lead with the highest-warrant capture's detail
    (resolved_error > stuck_recovered; novelty never leads — telemetry tier),
    then the counts tail. No capture and no tool use -> None (suppressed)."""
    if acc is None:
        return None
    lead = ""
    for tier in ("resolved_error", "stuck_recovered"):
        detail = next((d for t, d in acc.captures if t == tier and d), None)
        if detail:
            lead = _TIER_LEAD[tier] + detail[:_DETAIL_BUDGET]
            break
    top = ", ".join(name for name, _ in acc.tools_used.most_common(3))
    tail = (f"{acc.turn_count} turns, {acc.action_count} tool calls"
            + (f" ({top})" if top else ""))
    if lead:
        return f"{lead}; {tail}"[:_LINE_BUDGET]
    if acc.tools_used:
        return tail[:_LINE_BUDGET]
    return None
