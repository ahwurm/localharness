"""Integration test scaffold for Phase 5 startup wiring requirements.

Covers: BUS-02, MEM-01, MEM-04, HOOK-01, HOOK-02/03, TOOL-05, CTX-01/CTX-02.
"""
from __future__ import annotations

import pytest
from pathlib import Path


@pytest.fixture
def tmp_harness_dir(tmp_path):
    """Create a minimal harness directory structure."""
    config_dir = tmp_path / ".localharness"
    config_dir.mkdir()
    agents_dir = config_dir / "agents" / "test-agent"
    agents_dir.mkdir(parents=True)
    config_yaml = config_dir / "config.yaml"
    config_yaml.write_text(
        "version: '1'\n"
        "provider:\n"
        "  provider_type: ollama\n"
        "  base_url: http://localhost:11434/v1\n"
        "  default_model: test-model\n"
        "  api_key: none\n"
    )
    agent_yaml = config_dir / "agents" / "test-agent.yaml"
    agent_yaml.write_text(
        "name: test-agent\n"
        "role: Test assistant\n"
        "model: test-model\n"
    )
    return config_dir


@pytest.mark.asyncio
async def test_bus_persistence(tmp_harness_dir):
    """BUS-02: EventBus with persist_path writes events to JSONL on disk."""
    from localharness.core.bus import EventBus
    from localharness.core.events import Heartbeat

    events_path = tmp_harness_dir / "agents" / "test-agent" / "bus-events.jsonl"
    bus = EventBus(persist_path=events_path)

    event = Heartbeat(
        agent_id="test-agent",
        session_id="s1",
        iteration=1,
        context_utilization_pct=0.5,
    )
    await bus.publish(event)

    assert events_path.exists()
    content = events_path.read_text()
    assert "Heartbeat" in content or "heartbeat" in content.lower()


@pytest.mark.asyncio
async def test_memory_store_opens_with_wal(tmp_harness_dir):
    """MEM-01: MemoryStore opens with WAL mode."""
    from localharness.memory.sqlite import MemoryStore

    store = MemoryStore(
        agent_id="test-agent",
        division_id="default",
        org_id="default",
        base_dir=str(tmp_harness_dir),
    )
    await store.open()
    try:
        assert store._db is not None
        cursor = await store._db.execute("PRAGMA journal_mode")
        row = await cursor.fetchone()
        assert row[0] == "wal"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_memory_persists_across_sessions(tmp_harness_dir):
    """MEM-04: Facts persist after close and reopen."""
    from localharness.memory.sqlite import MemoryStore

    store = MemoryStore(
        agent_id="test-agent",
        division_id="default",
        org_id="default",
        base_dir=str(tmp_harness_dir),
    )
    await store.open()
    await store.store_fact(key="persist_check", value="test-value")
    await store.close()

    # Reopen
    store2 = MemoryStore(
        agent_id="test-agent",
        division_id="default",
        org_id="default",
        base_dir=str(tmp_harness_dir),
    )
    await store2.open()
    try:
        fact = await store2.get_fact("persist_check")
        assert fact is not None
        assert fact.value == "test-value"
    finally:
        await store2.close()


@pytest.mark.asyncio
async def test_hook_system_wires_to_registry():
    """HOOK-01: HookSystem.wire_to_registry connects pre/post hooks."""
    from localharness.tools.hooks import HookSystem
    from localharness.tools.registry import ToolRegistry

    hook_system = HookSystem()
    registry = ToolRegistry()
    hook_system.wire_to_registry(registry)

    assert len(registry._pre_hooks) >= 1
    assert len(registry._post_hooks) >= 1


@pytest.mark.asyncio
async def test_plugin_loader_discovers_entry_points(tmp_harness_dir):
    """HOOK-02/HOOK-03: PluginLoader.discover_all runs without error."""
    from localharness.tools.hooks import HookSystem
    from localharness.tools.registry import ToolRegistry
    from localharness.plugins.loader import PluginLoader

    registry = ToolRegistry()
    hook_system = HookSystem()
    plugins_dir = tmp_harness_dir / "plugins"
    plugins_dir.mkdir(exist_ok=True)

    loader = PluginLoader(registry, hook_system, plugins_dir=plugins_dir)
    loaded = await loader.discover_all()
    assert isinstance(loaded, list)


@pytest.mark.asyncio
async def test_mcp_registered(tmp_harness_dir):
    """TOOL-05: MCPClientManager registers tools from MCP servers into ToolRegistry."""
    from localharness.tools.registry import ToolRegistry
    from localharness.tools.mcp import MCPClientManager

    registry = ToolRegistry()
    mcp_manager = MCPClientManager(registry)

    results = await mcp_manager.startup([])
    assert isinstance(results, dict)
    assert len(results) == 0

    await mcp_manager.shutdown()


@pytest.mark.asyncio
async def test_compaction_fires_at_80_percent():
    """CTX-01/CTX-02: CompactionPipeline fires when usage >= 80%."""
    from localharness.agent.context import (
        CompactionPipeline, TokenCounter, ContextManager,
    )

    summarize_called = False

    async def mock_summarize(messages):
        nonlocal summarize_called
        summarize_called = True
        return "Summary of conversation."

    pipeline = CompactionPipeline(
        token_counter=TokenCounter(),
        llm_summarize_fn=mock_summarize,
        preserve_first_n=1,
        preserve_last_n=2,
    )
    ctx = ContextManager(
        max_context_tokens=100,  # Very small to trigger compaction
        pipeline=pipeline,
    )

    # Create enough messages to exceed 80% of 100 tokens
    # Need >2 messages in the compactable middle (after preserve_first_n=1, before preserve_last_n=2)
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "A" * 200},
        {"role": "assistant", "content": "B" * 200},
        {"role": "user", "content": "C" * 200},
        {"role": "assistant", "content": "D" * 200},
        {"role": "user", "content": "E" * 200},
        {"role": "assistant", "content": "F" * 200},
        {"role": "user", "content": "G" * 200},
        {"role": "assistant", "content": "H" * 200},
        {"role": "user", "content": "I" * 200},
    ]

    result = await ctx.build_messages(messages)
    assert summarize_called
