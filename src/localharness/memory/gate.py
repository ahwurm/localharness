"""Prediction-error write gate (WRITE-03/06) — the harness-initiated memory write path.

Cognitive frame: brains encode preferentially what violates expectations (prediction
error gates encoding); routine is not stored. Translated to this substrate: the bus
already carries cheap, discrete signals that the agent's model of the world was WRONG
and then corrected — a tool call that errored and later succeeded (a resolved mistake),
a stuck-then-recovered loop. The gate turns those into fact-candidate writes with ZERO
added model calls and zero inline latency; the token-costly record *composition* is
deferred to the idle consolidation pass (Phase 31), which promotes or discards them.

Candidates are written BELOW the injection confidence threshold (0.7,
sqlite._render_memory_index) on purpose: they live in the store but never pollute the
injected index until consolidation confirms them — the CLS fast-capture / slow-integrate
split. Tier `resolved_error` > `stuck_recovered` > `novelty` (discrete tiers, not a
continuous surprise score — our signals are event-shaped).

The `self_check`-changed signal was CUT from v1 by the 2026-07-02 critic pass: no
answer-comparison logic exists and none can be computed cheaply inline.

The gate must NEVER break the loop: every handler swallows its own exceptions (logged),
and every decision is published as a MemoryGateFired event — the fork-(b) live
observability for gate density/precision.
"""
from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from localharness.core.bus import EventBus, SubscriptionHandle
    from localharness.core.events import Observation, StuckRecovered
    from localharness.memory.sqlite import MemoryStore

log = logging.getLogger(__name__)

# Candidate confidences sit BELOW the 0.7 injection threshold until consolidation promotes.
_TIER_CONFIDENCE = {
    "resolved_error": 0.65,
    "stuck_recovered": 0.6,
    "novelty": 0.5,
}
_PREVIEW_CHARS = 160


def _h8(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8", "replace")).hexdigest()[:8]


def _preview(text: str | None) -> str:
    one = " ".join((text or "").split())
    return one[:_PREVIEW_CHARS - 1] + "…" if len(one) > _PREVIEW_CHARS else one


class WriteGate:
    """Subscribes to the agent's bus events and writes gated fact candidates.

    Wired beside MemoryStore at startup (start_cmd) when
    `agent.memory.write_gate_enabled` (default True — config-off like the cruncher).
    """

    def __init__(self, store: "MemoryStore", bus: "EventBus", agent_id: str) -> None:
        self._store = store
        self._bus = bus
        self._agent_id = agent_id
        self._handles: list["SubscriptionHandle"] = []
        # (session_id, tool_name) -> last error preview; resolved on next success.
        self._pending_errors: dict[tuple[str, str], str] = {}
        # First-use novelty (per process lifetime; store-level dedup makes repeats no-ops).
        self._seen_tools: set[str] = set()

    async def open(self) -> None:
        from localharness.core.events import Observation, StuckRecovered
        self._handles.append(
            self._bus.subscribe(Observation, self._on_observation, agent_id=self._agent_id)
        )
        self._handles.append(
            self._bus.subscribe(StuckRecovered, self._on_stuck_recovered, agent_id=self._agent_id)
        )

    async def close(self) -> None:
        for h in self._handles:
            self._bus.unsubscribe(h)
        self._handles.clear()

    # ------------------------------------------------------------------
    # Signal handlers (inline-free: dict ops + one SQLite write on fire)
    # ------------------------------------------------------------------

    async def _on_observation(self, event: "Observation") -> None:
        try:
            if event.observation_type != "tool_result" or not event.tool_name:
                return
            key = (event.session_id, event.tool_name)
            if event.error is not None:
                self._pending_errors[key] = _preview(event.error)
                return
            prior_error = self._pending_errors.pop(key, None)
            if prior_error is not None:
                # Resolved mistake — the highest-warrant learning signal.
                await self._capture(
                    tier="resolved_error",
                    session_id=event.session_id,
                    tool_name=event.tool_name,
                    fact_key=f"gate/resolved_error/{event.tool_name}/{_h8(event.session_id, event.tool_name, prior_error)}",
                    value=(
                        f"Tool `{event.tool_name}` failed then later succeeded in the same session "
                        f"(a resolved mistake). Error was: {prior_error} "
                        f"Success followed: {_preview(event.output)} "
                        "(auto-captured by the prediction-error gate; pending consolidation)"
                    ),
                    detail=prior_error,
                )
            elif event.tool_name not in self._seen_tools:
                self._seen_tools.add(event.tool_name)
                await self._capture(
                    tier="novelty",
                    session_id=event.session_id,
                    tool_name=event.tool_name,
                    fact_key=f"gate/novelty/{event.tool_name}",
                    value=(
                        f"First successful use of tool `{event.tool_name}` observed. "
                        f"Result preview: {_preview(event.output)} "
                        "(auto-captured by the novelty gate; pending consolidation)"
                    ),
                    detail="first-of-its-kind tool use",
                )
        except Exception:  # the gate must never break the loop
            log.exception("write-gate observation handler failed (non-fatal)")

    async def _on_stuck_recovered(self, event: "StuckRecovered") -> None:
        try:
            await self._capture(
                tier="stuck_recovered",
                session_id=event.session_id,
                tool_name=None,
                fact_key=f"gate/stuck_recovered/{_h8(event.session_id, event.stuck_signature)}",
                value=(
                    f"Agent got stuck (repeated `{_preview(event.stuck_signature)}`) and recovered "
                    f"at iteration {event.iteration}. "
                    "(auto-captured by the prediction-error gate; pending consolidation)"
                ),
                detail=_preview(event.stuck_signature),
            )
        except Exception:
            log.exception("write-gate stuck handler failed (non-fatal)")

    async def _capture(
        self,
        *,
        tier: str,
        session_id: str,
        tool_name: Optional[str],
        fact_key: str,
        value: str,
        detail: str,
    ) -> None:
        await self._store.store_fact(
            key=fact_key,
            value=value,
            tags=["gate", f"tier:{tier}", "pending_consolidation"],
            confidence=_TIER_CONFIDENCE[tier],
            source="write_gate",
            provenance=session_id,
        )
        from localharness.core.events import MemoryGateFired
        await self._bus.publish(MemoryGateFired(
            agent_id=self._agent_id,
            session_id=session_id,
            tier=tier,  # type: ignore[arg-type]
            fact_key=fact_key,
            tool_name=tool_name,
            detail=detail,
        ))
