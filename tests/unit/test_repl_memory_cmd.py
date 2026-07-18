"""REPL routing for `/memory`: the slash command is CLAIMED (before the unknown-slash reject),
runs model-free (no run_turn, no bus.publish), and threads through to cli.memory_cmd — including the
bare `/memory` (a single /word that would otherwise be rejected) and the two-step forget confirm.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from localharness.cli.repl import OrchestratorREPL
from localharness.memory.sqlite import USER_FORGET_PROVENANCE_PREFIX, MemoryStore


class FakeChannel:
    def __init__(self):
        self.messages: list[str] = []

    async def send_message(self, text, agent_id=None, metadata=None):
        self.messages.append(text)


async def _seeded_store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(agent_id="test-agent", division_id="d", org_id="default",
                        base_dir=str(tmp_path))
    await store.open()
    f = await store.store_fact(key="port", value="vLLM serves on port 8081", confidence=0.9,
                               source="remember")
    proj = await store.get_tag("project")
    ops = await store.get_tag("ops")
    await store.add_bucket_tag(f.id, proj.id)
    await store.add_atom_tag(f.id, ops.id)
    return store


def _repl(channel, store):
    agent = MagicMock()
    agent._config.name = "orchestrator"
    agent.current_session_id = "s1"
    agent._llm = MagicMock()
    agent.run_turn = AsyncMock()
    bus = AsyncMock()
    repl = OrchestratorREPL(
        orchestrator=MagicMock(), agent_loop=agent, channel=channel, bus=bus,
        memory_store=store,
    )
    return repl, agent, bus


@pytest.mark.asyncio
async def test_bare_memory_is_claimed_not_rejected_as_unknown(tmp_path):
    store = await _seeded_store(tmp_path)
    try:
        channel = FakeChannel()
        repl, agent, bus = _repl(channel, store)
        handled = await repl._handle_slash("/memory")
        assert handled is True
        out = channel.messages[-1]
        assert "project/ops" in out           # overview rendered
        assert "Unknown command" not in out   # NOT the unknown-slash reject path
        agent.run_turn.assert_not_called()    # deterministic, no LLM turn
        bus.publish.assert_not_called()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_memory_unavailable_when_no_store(tmp_path):
    channel = FakeChannel()
    repl, _, _ = _repl(channel, None)
    handled = await repl._handle_slash("/memory")
    assert handled is True
    assert "available" in channel.messages[-1].lower()


@pytest.mark.asyncio
async def test_memory_show_and_search_thread_through(tmp_path):
    store = await _seeded_store(tmp_path)
    try:
        channel = FakeChannel()
        repl, _, _ = _repl(channel, store)
        f = await store.get_fact("port")
        await repl._handle_slash(f"/memory show {f.id}")
        assert "vLLM serves on port 8081" in channel.messages[-1]
        assert "ambient-eligible: yes" in channel.messages[-1]
        # Case is preserved from the ORIGINAL string (sliced case-sensitively, like /model).
        await repl._handle_slash("/memory search vLLM")
        assert "8081" in channel.messages[-1]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_memory_forget_confirm_through_repl(tmp_path):
    store = await _seeded_store(tmp_path)
    try:
        channel = FakeChannel()
        repl, _, _ = _repl(channel, store)
        f = await store.get_fact("port")
        await repl._handle_slash(f"/memory forget {f.id}")
        assert "Confirm with" in channel.messages[-1]
        assert await store.get_fact("port") is not None  # preview only, not retired
        await repl._handle_slash(f"/memory forget {f.id} confirm")
        assert "Forgotten" in channel.messages[-1]
        row = await store.get_fact_by_id(f.id)
        assert row.status == "superseded" and row.provenance.startswith(USER_FORGET_PROVENANCE_PREFIX)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_unknown_tag_path_not_confused_with_unknown_command(tmp_path):
    store = await _seeded_store(tmp_path)
    try:
        channel = FakeChannel()
        repl, _, _ = _repl(channel, store)
        await repl._handle_slash("/memory nope/zilch")
        assert "Unknown tag path" in channel.messages[-1]
        assert "Unknown command" not in channel.messages[-1]
    finally:
        await store.close()
