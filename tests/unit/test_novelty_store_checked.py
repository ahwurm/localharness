"""#89 — the novelty gate's "first successful use of tool X" must be checked against the DURABLE
store, not an in-process set.

Audit 2026-07-17: gate/novelty/agent re-fired 14x across 9 days, each claiming "first successful
use". The decision was gated only by `self._seen_tools` (a per-process set that resets every
restart), so every restart re-fired an already-recorded tool — re-narrating it AND silently
bumping updated_at (the recency spine) via store_fact's corroboration touch. The fix consults the
durable store (fact_key_exists, ANY status) before minting; the genuine first-use path is byte-
identical. No live model needed.
"""
import time

import pytest

from localharness.core.events import Observation
from localharness.memory.gate import WriteGate
from localharness.memory.sqlite import MemoryStore


@pytest.fixture
async def store(tmp_path):
    s = MemoryStore(agent_id="nov-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


class _CapturingBus:
    def __init__(self):
        self.published = []

    def subscribe(self, *a, **k):
        return object()

    def unsubscribe(self, h):
        pass

    async def publish(self, e):
        self.published.append(e)


async def _first_use(gate, tool="bash_exec", sess="s1"):
    """Drive one FIRST successful use of `tool` (a tool_result success with no pending error)."""
    await gate._on_observation(Observation(
        agent_id="nov-agent", session_id=sess, observation_type="tool_result",
        tool_name=tool, output="ok", error=None))


def _novelty_fires(bus, tool="bash_exec"):
    from localharness.core.events import MemoryGateFired
    return [e for e in bus.published
            if isinstance(e, MemoryGateFired) and e.fact_key == f"gate/novelty/{tool}"]


async def _seed_novelty(store, tool="bash_exec", *, expires_at=None, sess="old-sess"):
    await store.store_fact(
        key=f"gate/novelty/{tool}",
        value=f"First successful use of tool `{tool}` observed. "
              "(auto-captured by the novelty gate; pending consolidation)",
        tags=["gate", "tier:novelty", "pending_consolidation"],
        confidence=0.5, source="write_gate", provenance=sess, expires_at=expires_at)


@pytest.mark.asyncio
async def test_genuine_first_use_still_mints_and_narrates(store):
    """Positive control: a truly first use (nothing in the store) mints gate/novelty/<tool> AND
    narrates it (MemoryGateFired) — byte-identical to today."""
    bus = _CapturingBus()
    gate = WriteGate(store, bus, "nov-agent")
    await _first_use(gate)
    assert (await store.get_fact("gate/novelty/bash_exec")) is not None
    assert len(_novelty_fires(bus)) == 1                       # narrated exactly once


@pytest.mark.asyncio
async def test_existing_novelty_never_refires_or_bumps(store):
    """A restart (fresh WriteGate = empty _seen_tools) with the fact already in the store must NOT
    re-fire: no narration, and NO silent updated_at bump."""
    await _seed_novelty(store)
    # Pin updated_at to a distinct past value so a corroboration touch would be detectable.
    await store._db.execute(
        "UPDATE facts SET updated_at = 1000 WHERE agent_id = ? AND key = 'gate/novelty/bash_exec'",
        ("nov-agent",))
    await store._db.commit()

    bus = _CapturingBus()
    gate = WriteGate(store, bus, "nov-agent")                  # fresh process: _seen_tools empty
    await _first_use(gate)

    assert _novelty_fires(bus) == []                          # no re-fire / no narration
    assert (await store.get_fact("gate/novelty/bash_exec")).updated_at == 1000  # no silent bump


@pytest.mark.asyncio
async def test_any_status_existing_novelty_blocks_refire(store):
    """"Any status" — an EXPIRED novelty fact (get_fact returns None) still blocks a re-fire, so the
    check cannot be get_fact (active-only); it is fact_key_exists (existence in any status)."""
    await _seed_novelty(store, expires_at=int(time.time()) - 3600)   # already expired
    assert (await store.get_fact("gate/novelty/bash_exec")) is None  # invisible to get_fact
    assert await store.fact_key_exists("gate/novelty/bash_exec") is True

    bus = _CapturingBus()
    gate = WriteGate(store, bus, "nov-agent")
    await _first_use(gate)
    assert _novelty_fires(bus) == []                          # still no re-fire
