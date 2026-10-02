"""REPL routing for `/memory`: the memory plugin's own slash row (PAPI-07), bound by the lifecycle
to the running MemoryPlugin. The slash command is CLAIMED (before the unknown-slash reject), runs
model-free (no run_turn, no bus.publish), and threads through to cli.memory_cmd over the plugin's
StoreBrowse — including the bare `/memory` (a single /word that would otherwise be rejected) and the
two-step forget confirm. With no memory plugin running there is no row: the standard reject.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from localharness.cli.repl import OrchestratorREPL
from localharness.cli.slash_commands import SlashCommand, set_plugin_rows
from localharness.core.bus import EventBus
from localharness.memory.config import MemoryConfig
from localharness.memory.plugin import MemoryPlugin
from localharness.memory.sqlite import USER_FORGET_PROVENANCE_PREFIX
from localharness.plugins.api import PluginContext, PluginPaths, SessionInfo
from localharness.plugins.lifecycle import RunningPlugin, _slash_handler
from localharness.provider.idle_llm import LLMTextAdapter
from localharness.tools.registry import ToolRegistry
from tests.conftest import MockLLMClient

TARGET = "localharness.memory.plugin:MemoryPlugin.slash_memory"


class FakeChannel:
    def __init__(self):
        self.messages: list[str] = []

    async def send_message(self, text, agent_id=None, metadata=None):
        self.messages.append(text)

    async def send_renderable(self, renderable, agent_id=None, metadata=None):
        # Mirror the terminal: render the rich object (overview/show tree) to plain text.
        import io

        from rich.console import Console
        console = Console(file=io.StringIO(), width=200)
        console.print(renderable)
        self.messages.append(console.file.getvalue())


def _ctx(tmp_path: Path) -> PluginContext:
    g, s = tmp_path / "global", tmp_path / "state"
    g.mkdir(parents=True, exist_ok=True)
    s.mkdir(parents=True, exist_ok=True)
    llm = MockLLMClient([])
    return PluginContext(
        bus=EventBus(), tools=ToolRegistry(), hooks=None, config=None,
        agent_config=MemoryConfig(consolidation={"enabled": False}),
        paths=PluginPaths(global_config_dir=g, workspace=None, state_dir=s),
        llm=llm, idle_llm=LLMTextAdapter(llm),
        session=SessionInfo(agent_id="test-agent", division_id="d", sitting_id="s1", model="m",
                            context_tokens=131072, budget={"max_tokens": 1}))


@pytest.fixture
async def plugin(tmp_path):
    """A real MemoryPlugin through tools()/start() on a tmp store, seeded through its own store,
    with ITS slash row installed exactly as the lifecycle builds it (bound to this instance)."""
    ctx = _ctx(tmp_path)
    p = MemoryPlugin()
    await p.tools(ctx)
    await p.start(ctx)
    await p._store.store_fact(key="port", value="vLLM serves on port 8081", confidence=0.9,
                              source="remember")
    (desc,) = MemoryPlugin.manifest.slash
    assert desc.target == TARGET
    rp = RunningPlugin(name="memory", plugin=p, ctx=ctx, bundled=True)
    assert set_plugin_rows([SlashCommand(desc.name, desc.help, _slash_handler(desc.target, rp),
                                         takes_args=True, plugin="memory")]) == []
    try:
        yield p
    finally:
        set_plugin_rows(())
        await p.stop(ctx)


def _repl(channel):
    agent = MagicMock()
    agent._config.name = "orchestrator"
    agent.current_session_id = "s1"
    agent._llm = MagicMock()
    agent.run_turn = AsyncMock()
    bus = AsyncMock()
    repl = OrchestratorREPL(orchestrator=MagicMock(), agent_loop=agent, channel=channel, bus=bus)
    return repl, agent, bus


async def test_bare_memory_is_claimed_not_rejected_as_unknown(plugin):
    channel = FakeChannel()
    repl, agent, bus = _repl(channel)
    handled = await repl._handle_slash("/memory")
    assert handled is True
    out = channel.messages[-1]
    assert "8081" in out                  # recent-memory row rendered
    assert "Unknown command" not in out   # NOT the unknown-slash reject path
    agent.run_turn.assert_not_called()    # deterministic, no LLM turn
    bus.publish.assert_not_called()


async def test_memory_off_is_the_standard_unknown_command():
    """No running memory plugin, no row: the old 'isn't available' text is unreachable from the REPL."""
    channel = FakeChannel()
    repl, agent, _ = _repl(channel)
    assert await repl._handle_slash("/memory") is True
    assert channel.messages == ["Unknown command: /memory — /help lists commands."]
    agent.run_turn.assert_not_called()


async def test_memory_unavailable_when_the_plugin_has_no_browse(plugin):
    """Defensive: a row whose plugin holds no StoreBrowse answers dispatch's own unavailable text."""
    plugin._browse = None
    channel = FakeChannel()
    repl, _, _ = _repl(channel)
    handled = await repl._handle_slash("/memory")
    assert handled is True
    assert "available" in channel.messages[-1].lower()


async def test_memory_show_and_search_thread_through(plugin):
    channel = FakeChannel()
    repl, _, _ = _repl(channel)
    f = await plugin._store.get_fact("port")
    await repl._handle_slash(f"/memory show {f.id}")
    assert "vLLM serves on port 8081" in channel.messages[-1]
    assert "salience" in channel.messages[-1]
    # Case is preserved from the ORIGINAL string (sliced case-sensitively, like /model).
    await repl._handle_slash("/memory search vLLM")
    assert "8081" in channel.messages[-1]


async def test_memory_forget_confirm_through_repl(plugin):
    store = plugin._store
    channel = FakeChannel()
    repl, _, _ = _repl(channel)
    f = await store.get_fact("port")
    await repl._handle_slash(f"/memory forget {f.id}")
    assert "Confirm with" in channel.messages[-1]
    assert await store.get_fact("port") is not None  # preview only, not retired
    await repl._handle_slash(f"/memory forget {f.id} confirm")
    assert "Forgotten" in channel.messages[-1]
    row = await store.get_fact_by_id(f.id)
    assert row.status == "superseded" and row.provenance.startswith(USER_FORGET_PROVENANCE_PREFIX)


async def test_unknown_tag_path_not_confused_with_unknown_command(plugin):
    channel = FakeChannel()
    repl, _, _ = _repl(channel)
    await repl._handle_slash("/memory nope/zilch")
    assert "/memory show" in channel.messages[-1]      # the usage line, claimed
    assert "Unknown command" not in channel.messages[-1]


async def test_a_render_slip_is_contained_with_the_core_rows_text(plugin, monkeypatch):
    """The `/memory failed: X` containment the REPL's core row used now lives in the plugin."""
    from localharness.cli import memory_cmd

    async def boom(*a, **kw): raise RuntimeError("db gone")
    monkeypatch.setattr(memory_cmd, "dispatch", boom)
    channel = FakeChannel()
    repl, _, _ = _repl(channel)
    assert await repl._handle_slash("/memory") is True
    assert channel.messages == ["/memory failed: db gone"]
