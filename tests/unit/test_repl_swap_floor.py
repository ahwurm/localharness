"""The REPL's /model hands the swap the live root's floor, and a swap that ended short never refits."""
from __future__ import annotations

import asyncio

import pytest

from localharness.provider import lifecycle
from tests.unit.test_repl_swap_wrapper import _build, _spy_swap, rig  # noqa: F401 — the fixture


@pytest.mark.asyncio
async def test_the_cold_peer_swap_carries_the_live_floor(tmp_path, rig, monkeypatch):
    _, repl, _, agent, _, _ = _build(tmp_path)
    agent.floor = lambda: 15_360
    seen = _spy_swap(monkeypatch, repl)

    await asyncio.wait_for(repl._handle_slash("/model flash-next"), 5)

    assert seen.calls[0][2]["floor"] == 15_360


@pytest.mark.asyncio
async def test_an_agent_without_a_floor_swaps_with_none(tmp_path, rig, monkeypatch):
    _, repl, _, _, _, _ = _build(tmp_path)
    seen = _spy_swap(monkeypatch, repl)

    await asyncio.wait_for(repl._handle_slash("/model flash-next"), 5)

    assert seen.calls[0][2]["floor"] is None


@pytest.mark.asyncio
async def test_a_swap_that_ended_short_never_refits_the_counter(tmp_path, rig, monkeypatch):
    _, repl, _, agent, _, primary = _build(tmp_path)
    agent.floor = lambda: 15_360
    refits = []

    async def refit(*a, **kw):
        refits.append(a)
        return ""

    repl._refresh_token_counter = refit

    async def below(incumbent, target, **kw):
        return lifecycle.SwapOutcome(None, incumbent, incumbent, RuntimeError("below"), "launch")

    monkeypatch.setattr(lifecycle, "swap", below)

    await asyncio.wait_for(repl._handle_slash("/model flash-next"), 5)

    assert refits == []
