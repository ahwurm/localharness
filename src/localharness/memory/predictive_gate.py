"""Collect-only predictive gate (Phase 34, COLL-01/04) — the SENSORY layer.

Measures surprise on every tool outcome, gates NOTHING: per-tool statistical priors
(pure SQL, zero tokens) grade each outcome against the tool's own history; scores and
the three-event contract (ExpectationAttached -> OutcomeObserved -> SurpriseScored)
are persisted for Phase 35 to derive thresholds from. No fact write, no loop behavior,
no injected-block byte keys off anything here.

Shape: WriteGate's exact subscribe/react/swallow/close lifecycle (memory/gate.py) —
additive bus subscriber, zero changes to agent/loop.py. Handlers are awaited inline by
EventBus._deliver, so each handler is ONE cheap indexed SELECT or INSERT (the same cost
class the loop already pays for WriteGate + MemoryStore's own handlers). A scorer
failure logs and drops — the loop can never be broken by measurement.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from typing import TYPE_CHECKING

from localharness.memory.sqlite import _band_z, _tool_error_surprisal, compute_quadrant

if TYPE_CHECKING:
    from datetime import datetime

    from localharness.config.models import PredictiveGateConfig
    from localharness.core.bus import EventBus, SubscriptionHandle
    from localharness.core.events import Action, Observation
    from localharness.memory.sqlite import MemoryStore, ToolPrior

log = logging.getLogger(__name__)


class PredictiveGate:
    """Subscribes to the agent's Action/Observation stream and scores every tool
    outcome against that tool's own history (COLL-01). Wired beside MemoryStore at
    startup when `agent.memory.predictive_gate.enabled` (default True); collect-only,
    it gates nothing — the persisted scores feed the Phase-35 threshold calibration.
    """

    def __init__(
        self, store: "MemoryStore", bus: "EventBus", agent_id: str,
        cfg: "PredictiveGateConfig",
    ) -> None:
        self._store = store
        self._bus = bus
        self._agent_id = agent_id
        self._cfg = cfg
        self._handles: list["SubscriptionHandle"] = []
        # tool_call_id -> (ToolPrior, action_timestamp). Insertion-ordered; capped at
        # cfg.pending_cap by evicting oldest (skip-under-load: collect-only can afford
        # to drop a correlation, it must never queue unboundedly).
        self._pending: dict[str, tuple["ToolPrior", "datetime"]] = {}

    async def open(self) -> None:
        from localharness.core.events import Action, Observation
        self._handles.append(
            self._bus.subscribe(Action, self._on_action, agent_id=self._agent_id)
        )
        self._handles.append(
            self._bus.subscribe(Observation, self._on_observation, agent_id=self._agent_id)
        )

    async def close(self) -> None:
        for h in self._handles:
            self._bus.unsubscribe(h)
        self._handles.clear()
        self._pending.clear()

    # ------------------------------------------------------------------
    # Signal handlers (inline-free: one indexed SELECT / two INSERTs on fire).
    # Every body is wrapped so a failure logs and drops — measurement never
    # surfaces to the publisher (WriteGate discipline).
    # ------------------------------------------------------------------

    async def _on_action(self, event: "Action") -> None:
        try:
            if event.action_type != "tool_call" or not event.tool_call_id or not event.tool_name:
                return
            prior = await self._store.get_tool_prior(event.tool_name)
            while len(self._pending) >= self._cfg.pending_cap:
                self._pending.pop(next(iter(self._pending)))
            self._pending[event.tool_call_id] = (prior, event.timestamp)
            from localharness.core.events import ExpectationAttached
            await self._bus.publish(ExpectationAttached(
                agent_id=self._agent_id,
                session_id=event.session_id,
                tool_call_id=event.tool_call_id,
                tool_name=event.tool_name,
                prior_n=prior.n,
                prior_error_rate=prior.error_rate,
                lat_mean_ms=prior.lat_mean_ms,
                lat_var_ms=prior.lat_var_ms,
                size_mean=prior.size_mean,
                size_var=prior.size_var,
            ))
        except Exception:  # measurement must never break the loop
            log.exception("predictive-gate action handler failed (non-fatal)")

    async def _on_observation(self, event: "Observation") -> None:
        try:
            if event.observation_type != "tool_result" or not event.tool_call_id:
                return
            pending = self._pending.pop(event.tool_call_id, None)
            if pending is None:  # unmatched: pre-subscribe, cap-evicted, or foreign
                return
            prior, action_ts = pending
            duration_ms = max(0, int((event.timestamp - action_ts).total_seconds() * 1000))
            is_error = 1 if event.error is not None else 0
            output_len = len(event.output or "")  # capped at 200 upstream (34-01 schema note)
            ts = int(event.timestamp.timestamp())
            obs_row_id = await self._store.record_tool_observation(
                session_id=event.session_id,
                tool_call_id=event.tool_call_id,
                tool_name=event.tool_name,
                ts=ts,
                is_error=is_error,
                output_len=output_len,
                duration_ms=duration_ms,
                event_id=event.id,
                source="live",
            )
            from localharness.core.events import OutcomeObserved, SurpriseScored
            await self._bus.publish(OutcomeObserved(
                agent_id=self._agent_id,
                session_id=event.session_id,
                tool_call_id=event.tool_call_id,
                tool_name=event.tool_name,
                is_error=bool(is_error),
                output_len=output_len,
                duration_ms=duration_ms,
            ))
            # Compute the three components explicitly so SurpriseScored carries them
            # (raw signed z in the event; abs() only in the composite, matching
            # compute_surprise_score). min_n comes from config, not the module default.
            min_n = self._cfg.min_prior_n
            err_s = _tool_error_surprisal(is_error, prior.error_rate, prior.n, min_n)
            z_lat = _band_z(duration_ms, prior.lat_mean_ms, prior.lat_var_ms, prior.lat_n, min_n)
            z_size = _band_z(output_len, prior.size_mean, prior.size_var, prior.size_n, min_n)
            score = err_s + self._cfg.latency_weight * abs(z_lat) + self._cfg.size_weight * abs(z_size)
            quadrant = compute_quadrant(is_error, prior.error_rate, prior.n, min_n)
            await self._store.record_surprise_score(
                session_id=event.session_id,
                observation_id=obs_row_id,
                expectation_json=json.dumps(dataclasses.asdict(prior)),
                score=score,
                quadrant=quadrant,
                scored_at=ts,
            )
            await self._bus.publish(SurpriseScored(
                agent_id=self._agent_id,
                session_id=event.session_id,
                tool_call_id=event.tool_call_id,
                tool_name=event.tool_name,
                score=score,
                quadrant=quadrant,
                error_surprisal=err_s,
                z_latency=z_lat,
                z_size=z_size,
            ))
        except Exception:  # measurement must never break the loop
            log.exception("predictive-gate observation handler failed (non-fatal)")
