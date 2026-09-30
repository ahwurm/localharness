"""`start` runs plugins through the plan and the lifecycle (44-14), driven through the real `_start_async`.

Each test bundles its own plugin by patching `localharness.plugins.builtin.BUILTIN_PLUGINS`, the
one list `bundled_plugins()` reads at call time, and drives the real session build offline with
only the external boundaries stubbed (`tests/unit/test_start_cmd.py::_stub_start_boundaries`: the
probe, the tokenizer, the REPL loop and third-party discovery). The plan, the lifecycle, the
registry, the slash table and the capability floor all run for real: what is asserted is what a
real `localharness start` does with a plugin, not what a helper would do if someone called it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localharness.cli.slash_commands import all_rows, find_row, set_plugin_rows
from localharness.plugins.api import Plugin, PluginManifest, SlashDescriptor
from localharness.plugins.slot import MemorySlot
from localharness.tools.base import Tool, ToolResult, ToolSchema
from tests.unit.test_start_cmd import _capture_start_console, _read_sessions, _stub_start_boundaries

EVENTS: list[str] = []


class _EchoTool(Tool):
    """Declares all four axes (ingest none, host safe), so the root agent keeps it."""

    def info(self) -> ToolSchema:
        return ToolSchema(name="probe_echo", description="Echo.", parameters={},
                          ingest="none", host="safe", result_origin="trusted")

    async def _execute(self, **_: Any) -> ToolResult:
        return self.ok("echo")


class _UndeclaredTool(Tool):
    """Declares nothing, so it counts as ingest untrusted AND host dangerous (fails closed)."""

    def info(self) -> ToolSchema:
        return ToolSchema(name="probe_fetch", description="Fetch.", parameters={})

    async def _execute(self, **_: Any) -> ToolResult:
        return self.ok("")


async def _probe_slash(ctx: Any, args: str) -> str:
    """The /probe row's target, imported by the lifecycle's handler the first time it runs."""
    return f"probe says {args or 'hi'}"


class _Probe(Plugin):
    """a bundled test plugin: one declared tool, one slash row, an artifact root"""

    manifest = PluginManifest(name="probe", version="1", kind="tools", slash=(SlashDescriptor(
        name="/probe", help="Ask the probe", target="tests.unit.test_start_plugins:_probe_slash"),))
    wants_artifacts = True

    async def tools(self, ctx: Any) -> list:
        return [_EchoTool()]

    async def start(self, ctx: Any) -> None:
        EVENTS.append("start")

    async def stop(self, ctx: Any) -> None:
        EVENTS.append("stop")


class _Crashes(Plugin):
    """a bundled test plugin whose start() raises"""

    manifest = PluginManifest(name="crashes", version="1", kind="tools")

    async def start(self, ctx: Any) -> None:
        raise RuntimeError("boom")


class _Ingests(Plugin):
    """a bundled test plugin whose one tool declares nothing"""

    manifest = PluginManifest(name="ingests", version="1", kind="tools")

    async def tools(self, ctx: Any) -> list:
        return [_UndeclaredTool()]


@pytest.fixture(autouse=True)
def _clean_rows_and_events():
    EVENTS.clear()
    yield
    set_plugin_rows(())  # a test that fails mid-drive must not leave rows for the next one


def _bundle(monkeypatch, *classes: type[Plugin]) -> None:
    monkeypatch.setattr("localharness.plugins.builtin.BUILTIN_PLUGINS", classes)


def _record_loop(monkeypatch) -> list[dict]:
    """Every AgentLoop construction's kwargs; the real __init__ still runs (a wrapper, not a double)."""
    import localharness.agent.loop as _loop_mod

    real = _loop_mod.AgentLoop.__init__
    seen: list[dict] = []

    def _rec(self, *args, **kwargs):
        seen.append(kwargs)
        return real(self, *args, **kwargs)

    monkeypatch.setattr("localharness.agent.loop.AgentLoop.__init__", _rec)
    return seen


def _summary(printed: list[str]) -> str:
    return next(line for line in printed if "startup)" in line)


# ------------------------------------------------------------------ Task 1: the wiring


async def test_start_runs_a_bundled_plugin_through_the_lifecycle(tmp_path, monkeypatch):
    """PAPI-02/PAPI-07: the plugin's tool is registered bare with source_plugin set, its slash row is
    in the one table while the session runs (and answers, importing its target on first use) and is
    gone afterwards, start() runs before the REPL and stop() after it, the loop holds the (empty)
    memory slot, and the session closes as it always did."""
    from localharness.cli.start_cmd import _start_async

    during: list[Any] = []

    async def _repl(self):
        EVENTS.append("repl")
        row = find_row("/probe")
        during.extend([row, await row.handler("there") if row is not None else None])

    _stub_start_boundaries(tmp_path, monkeypatch, repl_run=_repl)
    _bundle(monkeypatch, _Probe)
    loops = _record_loop(monkeypatch)

    await _start_async(None, False, False, str(tmp_path))

    (loop,) = loops  # a zero-turn drive builds exactly the root's loop
    registry = loop["tool_registry"]
    assert "probe_echo" in {s.name for s in registry.global_schemas()}, "bare name, global scope"
    assert registry.schema_of("probe_echo").source_plugin == "probe"
    row, reply = during
    assert row is not None and row.plugin == "probe", "the row was not in the table during the session"
    assert reply == "probe says there"
    assert find_row("/probe") is None and [r for r in all_rows() if r.plugin] == [], \
        "the plugin's rows must leave the table when the session ends"
    assert EVENTS == ["start", "repl", "stop"]
    slot = loop["memory_slot"]
    assert isinstance(slot, MemorySlot) and not slot.occupied
    rows = _read_sessions(tmp_path)
    assert len(rows) == 1 and rows[0][3] == "complete"


async def test_the_web_channel_gets_the_artifact_roots_core_accepted(tmp_path, monkeypatch):
    """PAPI-10: the root core computed for a plugin that wants artifacts, `<state dir>/artifacts/
    <name>`, is handed to the web channel, which then serves from it and nowhere else."""
    from localharness.channels.web.channel import WebChannel
    from localharness.cli.start_cmd import _start_async
    from localharness.core.bus import EventBus

    _stub_start_boundaries(tmp_path, monkeypatch)
    _bundle(monkeypatch, _Probe)
    channel = WebChannel(bus=EventBus(), config={})
    bound: list[dict] = []
    real_bind = channel.bind_runtime

    def _rec_bind(**kwargs):
        bound.append(kwargs)
        return real_bind(**kwargs)

    channel.bind_runtime = _rec_bind  # type: ignore[method-assign]

    await _start_async(None, False, False, str(tmp_path), channel_mode="web", web_channel=channel)

    root = tmp_path / "artifacts" / "probe"
    assert [b["artifact_roots"] for b in bound] == [{"probe": root}]
    assert channel.artifact_root("probe") == root


async def test_a_plugin_whose_start_raises_is_named_and_the_session_goes_on(tmp_path, monkeypatch):
    """PAPI-11, criterion 2: the failure is named in the startup summary's warnings, the other
    plugin still starts and stops, and the session starts and ends normally."""
    from localharness.cli.start_cmd import _start_async

    printed = _capture_start_console(monkeypatch)
    _stub_start_boundaries(tmp_path, monkeypatch)
    _bundle(monkeypatch, _Crashes, _Probe)

    await _start_async(None, False, False, str(tmp_path))

    assert "plugin crashes: start() raised RuntimeError: boom" in _summary(printed)
    assert EVENTS == ["start", "stop"]
    rows = _read_sessions(tmp_path)
    assert len(rows) == 1 and rows[0][3] == "complete"


async def test_the_root_floor_runs_after_plugin_tools_register(tmp_path, monkeypatch):
    """SAFE-02 / 44-06: a plugin tool that declares nothing is registered by the lifecycle and then
    stripped from the root by the floor. If the floor ran first, the root would hold it beside
    bash_exec and the loop's own toolset call would raise CoResidenceError."""
    from localharness.cli.start_cmd import _start_async

    _stub_start_boundaries(tmp_path, monkeypatch)
    _bundle(monkeypatch, _Ingests)
    loops = _record_loop(monkeypatch)

    await _start_async(None, False, False, str(tmp_path))

    (loop,) = loops
    registry, cfg = loop["tool_registry"], loop["config"]
    assert registry.schema_of("probe_fetch").source_plugin == "ingests", "premise: it registered"
    assert "probe_fetch" in cfg.tools.deny
    resolved = registry.get_tools_for_agent(cfg.name, cfg.division or "", cfg.tools)  # the loop's call
    assert "probe_fetch" not in resolved and "bash_exec" in resolved


async def test_start_config_dir_discovers_folder_plugins_under_it(tmp_path, monkeypatch):
    """The Phase 38 guard on the new path, end to end with real discovery: an enabled folder plugin
    in `<config-dir>/plugins/` is imported, and its tool is in the session's registry."""
    from localharness.cli.start_cmd import _start_async

    _stub_start_boundaries(tmp_path, monkeypatch, real_plugins=True)
    folder = tmp_path / "plugins" / "fold"
    folder.mkdir(parents=True)
    (folder / "__init__.py").write_text(
        "from localharness.plugins.api import Plugin, PluginManifest\n"
        "from localharness.tools.base import Tool, ToolSchema\n"
        "class _Tool(Tool):\n"
        "    def info(self):\n"
        "        return ToolSchema(name='fold_echo', description='d', parameters={},\n"
        "                          ingest='none', host='safe', result_origin='trusted')\n"
        "    async def _execute(self, **kw):\n"
        "        return self.ok('folded')\n"
        "class FoldPlugin(Plugin):\n"
        "    '''a folder plugin'''\n"
        "    manifest = PluginManifest(name='fold', version='1', kind='tools')\n"
        "    async def tools(self, ctx):\n"
        "        return [_Tool()]\n"
        "plugin = FoldPlugin\n")
    with (tmp_path / "config.yaml").open("a") as f:
        f.write("fold:\n  enabled: true\n")  # a plugin you installed is on only when the machine says so
    loops = _record_loop(monkeypatch)

    await _start_async(None, False, False, str(tmp_path))

    (loop,) = loops
    assert loop["tool_registry"].schema_of("fold_echo").source_plugin == "fold"


def test_the_start_stub_turns_discovery_off_unless_real_plugins(tmp_path):
    """The harness every start drive shares: discovery finds nothing by default, so no drive depends
    on what this venv has installed; `real_plugins=True` leaves the real function in place."""
    from localharness.plugins import discovery

    real = discovery.discover
    (tmp_path / "plugins" / "one").mkdir(parents=True)
    (tmp_path / "plugins" / "one" / "__init__.py").write_text("")
    assert [d.name for d in real(tmp_path) if d.source == "folder"] == ["one"], "premise"
    with pytest.MonkeyPatch.context() as mp:
        _stub_start_boundaries(tmp_path, mp)
        assert discovery.discover(tmp_path) == []
    with pytest.MonkeyPatch.context() as mp:
        _stub_start_boundaries(tmp_path, mp, real_plugins=True)
        assert discovery.discover is real
