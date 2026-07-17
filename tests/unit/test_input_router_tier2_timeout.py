"""#92 — tier-2 classify: separate permit-wait from generation clock, and log the fallback.

The old classify_tier2 wrapped EVERYTHING (permit-wait + generation) in one 5s wait_for. Under the
capacity-1 inference gate, the agent's own turn holds the permit, so the 5s budget was consumed
WAITING for the permit and tier-2 always timed out -> always QUEUE (tier-2 effectively dead). It
also swallowed every exception with zero logging.

Fix: a permit-contended call gets a bounded permit-wait (~30s) PLUS its generation timeout, so
contention no longer starves the decision; and the swallowed exception is logged (type + message).
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from localharness.channels.input_router import Route, classify_tier2


async def test_permit_contention_no_longer_starves_the_decision():
    """A call whose FIRST 0.2s is permit contention (exceeding the generation timeout) but whose
    generation is then instant resolves to the real verdict — the permit-wait budget covers it."""
    async def contended(_msgs):
        await asyncio.sleep(0.2)  # simulate waiting behind the agent's held inference permit
        return "NUDGE"

    d = await classify_tier2("x", "ctx", contended, timeout=0.1, permit_wait=0.5)
    assert d.route is Route.NUDGE  # old single-5s clock would have cut this at 0.1 -> QUEUE


async def test_total_budget_exceeded_defaults_queue():
    """Past permit_wait + timeout the call is abandoned to the harmless QUEUE default."""
    async def wedged(_msgs):
        await asyncio.sleep(1.0)
        return "NUDGE"

    d = await classify_tier2("x", "ctx", wedged, timeout=0.03, permit_wait=0.03)
    assert d.route is Route.QUEUE and "default-queue" in d.reason


async def test_swallowed_exception_is_logged(caplog):
    """The fallback logs the swallowed error (type + message) at WARNING — no more silent QUEUE."""
    async def boom(_msgs):
        raise RuntimeError("provider exploded")

    with caplog.at_level(logging.WARNING, logger="localharness.channels.input_router"):
        d = await classify_tier2("x", "ctx", boom)

    assert d.route is Route.QUEUE
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "provider exploded" in joined and "RuntimeError" in joined
