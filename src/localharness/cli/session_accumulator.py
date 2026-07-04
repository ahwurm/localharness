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
        self.first_ask: str | None = None  # TIME-01: sitting's opening user ask (zero-model topic)

    async def open(self) -> None:
        from localharness.core.events import (
            MemoryGateFired,
            Observation,
            TurnCompleted,
            TurnFailed,
            UserMessage,
        )
        sub = self._bus.subscribe
        self._handles += [
            sub(TurnCompleted, self.on_turn_completed, agent_id=self._agent_id),
            sub(TurnFailed, self.on_turn_failed, agent_id=self._agent_id),
            sub(Observation, self.on_observation, agent_id=self._agent_id),
            sub(MemoryGateFired, self.on_gate_fired, agent_id=self._agent_id),
            sub(UserMessage, self.on_user_message, agent_id=self._agent_id),
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

    async def on_user_message(self, event) -> None:
        # Capture-FIRST: the sitting's opening ask is the topic a human recognizes.
        if self.first_ask is None and (event.content or "").strip():
            self.first_ask = event.content


def _clean_ask(text: str) -> str:
    # Collapse whitespace (gate.py _preview precedent) + neutralize double quotes that
    # would break the single-line `asked: "..."` markdown/quoting contract (Pitfall 3).
    return " ".join((text or "").split()).replace('"', "'")


def derive_session_summary(acc: Optional[SessionAccumulator]) -> str | None:
    """Payload-first or nothing. Lead priority: resolved_error > stuck_recovered
    > asked-slice (TIME-01 — a pure-chat sitting is no longer invisible); novelty
    never leads. No lead and no tool use -> None (SESS-05 suppressed)."""
    if acc is None:
        return None
    lead, sep = "", "; "
    for tier in ("resolved_error", "stuck_recovered"):
        detail = next((d for t, d in acc.captures if t == tier and d), None)
        if detail:
            lead = _TIER_LEAD[tier] + detail[:_DETAIL_BUDGET]
            break
    if not lead and acc.first_ask:
        ask = _clean_ask(acc.first_ask)
        if len(ask) > _DETAIL_BUDGET:
            ask = ask[: _DETAIL_BUDGET - 1].rstrip() + "…"
        lead = f'asked: "{ask}"'
        sep = " — "  # owner specimen: ask lead uses em dash; capture tiers keep "; "
    delegations = acc.tools_used.get("agent", 0)
    tool_calls = acc.action_count - delegations
    top = ", ".join(
        n for n, _ in Counter(
            {k: v for k, v in acc.tools_used.items() if k != "agent"}
        ).most_common(3)
    )
    parts = [f"{acc.turn_count} turn{'s' if acc.turn_count != 1 else ''}"]
    if tool_calls:
        parts.append(
            f"{tool_calls} tool call{'s' if tool_calls != 1 else ''}"
            + (f" ({top})" if top else "")
        )
    if delegations:
        parts.append(f"{delegations} delegation{'s' if delegations != 1 else ''}")
    tail = ", ".join(parts)
    if lead:
        return f"{lead}{sep}{tail}"[:_LINE_BUDGET]
    if acc.tools_used:
        return tail[:_LINE_BUDGET]
    return None
