"""`start` runs plugins through the plan and the lifecycle (44-14), driven through the real `_start_async`.

Each test bundles its own plugin by patching `localharness.plugins.builtin.BUILTIN_PLUGINS`, the
one list `bundled_plugins()` reads at call time, and drives the real session build offline with
only the external boundaries stubbed (`tests/unit/test_start_cmd.py::_stub_start_boundaries`: the
probe, the tokenizer, the REPL loop and third-party discovery). The plan, the lifecycle, the
registry, the slash table and the capability floor all run for real: what is asserted is what a
real `localharness start` does with a plugin, not what a helper would do if someone called it.
"""
from __future__ import annotations

import re
import sys
from typing import Any

import pytest
from pydantic import BaseModel, Field

from localharness.cli.slash_commands import all_rows, find_row, set_plugin_rows
from localharness.cli.theme import entity
from localharness.plugins.api import Plugin, PluginManifest, PluginPaths, SlashDescriptor
from localharness.plugins.discovery import DiscoveredPlugin
from localharness.plugins.slot import MemorySlot
from localharness.tools.base import Tool, ToolResult, ToolSchema
from localharness.tools.hooks import HARNESS_HOOKIMPL, HookSystem
from tests.unit.test_start_cmd import _capture_start_console, _read_sessions, _stub_start_boundaries

EVENTS: list[str] = []
CONTEXTS: list[Any] = []


class _EchoTool(Tool):
    """Declares ingest none and host safe, so the root agent keeps it."""

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


class _ProbeHooks:
    @HARNESS_HOOKIMPL
    def pre_tool(self, name: str, arguments: dict, agent_id: str, division_id: str) -> None:
        EVENTS.append(f"pre_tool {name}")


class _Probe(Plugin):
    """a bundled test plugin: one declared tool, one slash row, an artifact root"""

    manifest = PluginManifest(name="probe", version="1", kind="tools", slash=(SlashDescriptor(
        name="/probe", help="Ask the probe", target="tests.unit.test_start_plugins:_probe_slash"),))
    wants_artifacts = True

    async def tools(self, ctx: Any) -> list:
        return [_EchoTool()]

    async def start(self, ctx: Any) -> None:
        ctx.hooks.register_plugin(_ProbeHooks(), name="probe")  # the session's own hook system
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


class _ReaderTool(Tool):
    """Declares ingest untrusted (it reads outside content) and host safe."""

    def info(self) -> ToolSchema:
        return ToolSchema(name="probe_read", description="Read.", parameters={},
                          ingest="untrusted", host="safe", result_origin="untrusted")

    async def _execute(self, **_: Any) -> ToolResult:
        return self.ok("")


class _Reads(Plugin):
    """a bundled test plugin whose one tool declares ingest: untrusted"""

    manifest = PluginManifest(name="reads", version="1", kind="tools")

    async def tools(self, ctx: Any) -> list:
        return [_ReaderTool()]


class _BadConfigure(Plugin):
    """a bundled test plugin whose configure() raises"""

    manifest = PluginManifest(name="badconf", version="1", kind="tools")

    async def configure(self, ctx: Any) -> Any:
        raise ValueError("no endpoint")


class _BadTools(Plugin):
    """a bundled test plugin whose tools() raises"""

    manifest = PluginManifest(name="badtools", version="1", kind="tools")

    async def tools(self, ctx: Any) -> list:
        raise KeyError("registry")


class _Hex(BaseModel):
    color: str = Field("#000000", pattern=r"^#[0-9a-f]{6}$")


class _Strict(Plugin):
    """a bundled test plugin with settings"""

    manifest = PluginManifest(name="strict", version="1", kind="tools")
    ConfigModel = _Hex


class _Size(BaseModel):
    size: int = 8


class _Settled(Plugin):
    """a bundled test plugin that keeps the context core built for it"""

    manifest = PluginManifest(name="settled", version="1", kind="tools")
    AgentConfigModel = _Size
    wants_artifacts = True

    async def start(self, ctx: Any) -> None:
        CONTEXTS.append(ctx)


@pytest.fixture(autouse=True)
def _clean_rows_and_events():
    EVENTS.clear()
    CONTEXTS.clear()
    yield
    set_plugin_rows(())  # a test that fails mid-drive must not leave rows for the next one
    for name in [m for m in sys.modules if m.startswith("localharness_folder_plugins")]:
        del sys.modules[name]  # the folder plugin one test imports is not another test's


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
    """PAPI-02/PAPI-07: the plugin's tool is registered bare with source_plugin set and dispatches
    through the session's registry, firing the hook the plugin put on the session's hook system;
    its slash row is in the one table while the session runs (and answers, importing its target on
    first use) and is gone afterwards; start() runs before the REPL and stop() after it; the loop
    holds the memory slot (the transitional occupant, 46-06); and the session closes as it always did."""
    from localharness.cli.start_cmd import _start_async

    during: list[Any] = []

    async def _repl(self):
        EVENTS.append("repl")
        (loop,) = loops  # a zero-turn drive builds exactly the root's loop
        cfg = loop["config"]
        result = await loop["tool_registry"].dispatch("probe_echo", {}, cfg.name, cfg.division or "",
                                                      cfg.tools)
        row = find_row("/probe")
        during.extend([result.output, row, await row.handler("there") if row is not None else None])

    _stub_start_boundaries(tmp_path, monkeypatch, repl_run=_repl)
    _bundle(monkeypatch, _Probe)
    loops = _record_loop(monkeypatch)

    await _start_async(None, False, False, str(tmp_path))

    (loop,) = loops
    registry = loop["tool_registry"]
    assert "probe_echo" in {s.name for s in registry.global_schemas()}, "bare name, global scope"
    assert registry.schema_of("probe_echo").source_plugin == "probe"
    output, row, reply = during
    assert output == "echo"
    assert row is not None and row.plugin == "probe", "the row was not in the table during the session"
    assert reply == "probe says there"
    assert find_row("/probe") is None and [r for r in all_rows() if r.plugin] == [], \
        "the plugin's rows must leave the table when the session ends"
    assert EVENTS == ["start", "repl", "pre_tool probe_echo", "stop"]
    slot = loop["memory_slot"]
    # 46-06 (D3): memory is on, so the slot holds the transitional browse occupant, not a plugin
    assert isinstance(slot, MemorySlot) and slot.occupant_name == "memory"
    rows = _read_sessions(tmp_path)
    assert len(rows) == 1 and rows[0][3] == "complete"


async def test_the_web_channel_gets_the_artifact_roots_core_accepted(tmp_path, monkeypatch):
    """PAPI-10: the root core computed for a plugin that wants artifacts, `<state dir>/artifacts/
    <name>`, is handed to the web channel, which then serves from it and nowhere else."""
    from localharness.channels.web.channel import WebChannel
    from localharness.cli.start_cmd import _start_async
    from localharness.core.bus import EventBus

    from localharness.cli.web_plugin import WebPlugin

    _stub_start_boundaries(tmp_path, monkeypatch)
    _bundle(monkeypatch, _Probe, WebPlugin)  # `web` is a --channel name only while bundled (46-04)
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
    assert channel.artifact_roots() == {"probe": root}


async def test_a_plugin_whose_start_raises_is_named_and_the_session_goes_on(tmp_path, monkeypatch):
    """PAPI-11, criterion 2: the failure is named in the startup summary's warnings, the other
    plugin still starts and stops, and the session starts and ends normally."""
    from localharness.cli.start_cmd import _start_async

    printed = _capture_start_console(monkeypatch)
    _stub_start_boundaries(tmp_path, monkeypatch)
    _bundle(monkeypatch, _Crashes, _Probe)

    await _start_async(None, False, False, str(tmp_path))

    assert "plugin crashes: start() raised RuntimeError: boom" in _summary(printed)
    assert EVENTS == ["start", "stop"]  # the other plugin; a failed start() is not stopped (44-12)
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
    resolved = registry.get_tools_for_agent(cfg.name, cfg.division or "", cfg.tools)  # the loop's call
    assert "probe_fetch" not in resolved and "bash_exec" in resolved
    assert "probe_fetch" in cfg.tools.deny


async def test_the_root_agents_plugin_settings_reach_the_plugin(tmp_path, monkeypatch):
    """44-09: resolve() runs on the session's own loader AFTER it loaded the root agent, so the
    `settled:` section of the root agent's file is the plugin's agent-level settings."""
    from localharness.cli.start_cmd import _start_async

    _stub_start_boundaries(tmp_path, monkeypatch)
    _bundle(monkeypatch, _Settled)
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "orchestrator.yaml").write_text(
        "name: orchestrator\nrole: General-purpose assistant\nmodel: inherit\nsettled:\n  size: 3\n")

    await _start_async(None, False, False, str(tmp_path))

    (ctx,) = CONTEXTS
    assert ctx.agent_config.size == 3


async def test_a_plugins_paths_are_the_ones_core_computed_for_the_session(tmp_path, monkeypatch,
                                                                         fake_home):
    """PAPI-03/PAPI-10 in a workspace session: the machine's config dir, the project's workspace,
    the state dir the work lands in (the workspace), and the artifact root under that state dir."""
    from tests.unit.test_workspace_state_landing import _drive, _workspace_start

    _home, global_dir, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    _bundle(monkeypatch, _Settled)

    await _drive()

    (ctx,) = CONTEXTS
    ws = ws.resolve()
    assert ctx.paths == PluginPaths(global_config_dir=global_dir, workspace=ws, state_dir=ws,
                                    artifact_dir=ws / "artifacts" / "settled")


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


# ------------------------------------------------------------------ Task 2: what the operator sees


def _discovers(monkeypatch, *names: str) -> None:
    """Third-party plugins installed but not enabled: metadata only, each naming a module that does
    not exist, so any attempt to import one fails loudly."""
    found = [DiscoveredPlugin(n, "entry_point", f"{n.replace('-', '_')}:Plugin", n, "0.3.1") for n in names]
    monkeypatch.setattr("localharness.plugins.discovery.discover", lambda global_config_dir: found)


async def test_the_banner_names_the_loaded_plugins_right_after_the_summary(tmp_path, monkeypatch):
    """ENAB-04: one line naming what loaded, directly under the startup summary line."""
    from localharness.cli.start_cmd import _start_async

    printed = _capture_start_console(monkeypatch)
    _stub_start_boundaries(tmp_path, monkeypatch)
    _bundle(monkeypatch, _Probe, _Settled)

    await _start_async(None, False, False, str(tmp_path))

    i = printed.index(_summary(printed))
    assert printed[i + 1] == "  " + entity("tool", "Plugins: probe, settled")


class _ImportSpy:
    """First on sys.meta_path: records every module name an import ASKS for, found or not. An import
    of a module that does not exist leaves nothing in sys.modules, so only the asking shows it."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
        self.asked.append(name)
        return None


async def test_one_available_plugin_gets_the_exact_enable_command(tmp_path, monkeypatch):
    """ENAB-04 / PRD §4: an installed plugin that is not enabled is named once, with the command that
    turns it on, and nothing tries to import it."""
    from localharness.cli.start_cmd import _start_async

    printed = _capture_start_console(monkeypatch)
    _stub_start_boundaries(tmp_path, monkeypatch)
    _discovers(monkeypatch, "lh-exa")
    spy = _ImportSpy()
    monkeypatch.setattr(sys, "meta_path", [spy, *sys.meta_path])

    await _start_async(None, False, False, str(tmp_path))

    assert ("i 1 plugin available, not enabled: lh-exa — run `localharness plugins enable lh-exa` "
            "to turn it on") in printed
    assert "lh_exa" not in spy.asked, "an available plugin must never be imported"
    with pytest.raises(ModuleNotFoundError):
        __import__("lh_exa_premise")
    assert spy.asked[-1] == "lh_exa_premise", "premise: the spy sees an import that is attempted"
    assert "lh_exa" not in _summary(printed) and "lh-exa" not in _summary(printed)
    rows = _read_sessions(tmp_path)
    assert len(rows) == 1 and rows[0][3] == "complete"


async def test_several_available_plugins_share_one_line(tmp_path, monkeypatch):
    from localharness.cli.start_cmd import _start_async

    printed = _capture_start_console(monkeypatch)
    _stub_start_boundaries(tmp_path, monkeypatch)
    _discovers(monkeypatch, "a", "b")

    await _start_async(None, False, False, str(tmp_path))

    assert ("i 2 plugins available, not enabled: a, b — run `localharness plugins enable <name>` "
            "to turn one on") in printed


async def test_a_plugin_tool_the_floor_keeps_from_the_root_is_named(tmp_path, monkeypatch):
    """A plugin tool the root capability floor strips is named in a startup warning, with its plugin
    and why, instead of silently missing. The web verbs (core's) are stripped as always, unnamed."""
    from localharness.cli.start_cmd import _start_async

    printed = _capture_start_console(monkeypatch)
    _stub_start_boundaries(tmp_path, monkeypatch)
    _bundle(monkeypatch, _Ingests, _Reads)

    await _start_async(None, False, False, str(tmp_path))

    summary = _summary(printed)
    assert ("capability floor: the root agent does not hold probe_fetch (plugin ingests) — it "
            "declares no ingest, which counts as ingest: untrusted") in summary
    assert ("capability floor: the root agent does not hold probe_read (plugin reads) — it declares "
            "ingest: untrusted") in summary
    assert summary.count("capability floor:") == 2


async def test_with_no_plugins_the_banner_is_what_it_was(tmp_path, monkeypatch):
    """Criterion 1's other half: nothing bundled and nothing installed prints no plugins line, no hint
    and an unchanged counts segment, and nothing after the summary line."""
    from localharness.cli.start_cmd import _start_async

    printed = _capture_start_console(monkeypatch)
    _stub_start_boundaries(tmp_path, monkeypatch)
    monkeypatch.setattr("localharness.plugins.builtin.BUILTIN_PLUGINS", ())  # web is bundled and on by default (46-02)

    await _start_async(None, False, False, str(tmp_path))

    i = printed.index(_summary(printed))
    assert re.fullmatch(r"\[dim\]\(\d+\.\ds startup\)\[/dim\] -- " + re.escape(entity("agent", "1 agent")),
                        printed[i])
    assert printed[i + 1:] == []


def test_the_hook_system_keeps_no_plugin_bookkeeping():
    """The legacy loader's name list and its registration verb are gone with it: HookSystem's public
    surface is registering a hook object and wiring to a registry, and an instance holds only pluggy."""
    assert {n for n in vars(HookSystem) if not n.startswith("_")} == {"register_plugin", "wire_to_registry"}
    assert set(vars(HookSystem())) == {"pm"}


async def test_a_failure_at_every_stage_is_named_and_the_session_goes_on(tmp_path, monkeypatch):
    """PAPI-11, criterion 2, one plugin per stage: an import that fails, settings that do not
    validate, a configure() and a tools() that raise are each named in the startup summary's warnings
    — as is a warning about a plugin that is off — while the healthy plugin runs, and the session
    starts and ends normally."""
    from localharness.cli.start_cmd import _start_async

    printed = _capture_start_console(monkeypatch)
    _stub_start_boundaries(tmp_path, monkeypatch)
    _bundle(monkeypatch, _BadConfigure, _BadTools, _Strict, _Probe)
    _discovers(monkeypatch, "lh-broken")
    with (tmp_path / "config.yaml").open("a") as f:
        f.write("lh-broken:\n  enabled: true\nstrict:\n  color: red\nbadtools:\n  enabled: 'yes'\n")

    await _start_async(None, False, False, str(tmp_path))

    summary = _summary(printed)
    for named in ("plugin lh-broken: ", "ModuleNotFoundError",                    # import
                  "plugin strict: invalid settings — strict.color",               # settings
                  "plugin badconf: configure() raised ValueError: no endpoint",   # configure
                  "plugin badtools: tools() raised KeyError",                     # tools
                  "`badtools.enabled` must be true or false"):                    # a resolve warning
        assert named in summary, f"{named!r} is not in the startup summary: {summary}"
    assert EVENTS == ["start", "stop"]
    rows = _read_sessions(tmp_path)
    assert len(rows) == 1 and rows[0][3] == "complete"
