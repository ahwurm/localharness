"""The plugin lifecycle (44-12): PAPI-02 order, PAPI-03 context, PAPI-05 registration, PAPI-11
containment, SAFE-06 clamp, PAPI-10 artifact roots, PAPI-07 slash rows, PAPI-08 doctor rows.

Every test builds a Resolution the way resolve() does — the PURE build_load_plan over test plugin
classes, plus `classes` and `settings` — and runs the real lifecycle against a real ToolRegistry.
`bundled` is toggled per test: a class passed as bundled ships with LocalHarness, one passed as
installed came from an entry point (its plan entry is not bundled)."""
from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from dataclasses import fields

import pytest
from pydantic import BaseModel

from localharness.core.bus import EventBus
from localharness.plugins.api import (
    Check, Plugin, PluginContext, PluginManifest, PluginPaths, SlashDescriptor,
)
from localharness.plugins.discovery import DiscoveredPlugin
from localharness.plugins.lifecycle import (
    DoctorRow, LifecycleResult, doctor_rows, start_plugins, stop_plugins,
)
from localharness.plugins.plan import build_load_plan
from localharness.plugins.resolve import PluginSettings, Resolution
from localharness.tools.base import ToolResult, ToolSchema
from localharness.tools.builtin import register_builtin_tools
from localharness.tools.hooks import HARNESS_HOOKIMPL, HookSystem
from localharness.tools.registry import ToolRegistry


class _Tool:
    """A declared test tool (ToolProtocol); `gate_family` is what it declares to the gate."""

    def __init__(self, name: str, gate_family: str | None = None) -> None:
        self._schema = ToolSchema(
            name=name, description="a test tool", parameters={"type": "object", "properties": {}},
            ingest="none", host="safe", result_origin="trusted", gate_family=gate_family)

    def info(self) -> ToolSchema:
        return self._schema

    async def run(self, **kwargs) -> ToolResult:
        return ToolResult(output="ok")


def _plugin(name: str, *, log: list | None = None, requires: tuple[str, ...] = (),
            contributes=(), slash: tuple[SlashDescriptor, ...] = (), wants_artifacts: bool = False,
            requires_localharness: str = ">=0.15,<1", **methods) -> type[Plugin]:
    """A test plugin class. Its default methods record (stage, name) in `log`; `contributes` are
    tool names or tool objects; `methods` replaces any method (a raising configure, say)."""
    calls = log if log is not None else []

    async def configure(self, ctx):
        calls.append(("configure", name))
        return "ready"

    async def contribute(self, ctx):
        calls.append(("tools", name))
        return [_Tool(t) if isinstance(t, str) else t for t in contributes]

    async def start(self, ctx):
        calls.append(("start", name))

    async def stop(self, ctx):
        calls.append(("stop", name))

    manifest = PluginManifest(name=name, version="1.0", kind="tools", requires=requires,
                              slash=slash, requires_localharness=requires_localharness)
    namespace = {"__doc__": f"test plugin {name}", "manifest": manifest,
                 "wants_artifacts": wants_artifacts, "configure": configure, "tools": contribute,
                 "start": start, "stop": stop, **methods}
    return type(f"Plugin_{name}", (Plugin,), namespace)


def _resolution(*, bundled=(), off=(), installed=(), available=(), settings=None) -> Resolution:
    """What resolve() would return: `bundled` and `installed` classes enabled, `off` classes bundled
    but not enabled, `available` names discovered but not enabled (never imported: no class)."""
    classes = {c.manifest.name: c for c in (*bundled, *off, *installed)}
    enabled = {**{n: True for n in classes}, **{c.manifest.name: False for c in off},
               **{n: False for n in available}}
    load_plan = build_load_plan(
        bundled=(*bundled, *off),
        discovered=[DiscoveredPlugin(c.manifest.name, "entry_point", f"tests.fake:{c.manifest.name}")
                    for c in installed]
        + [DiscoveredPlugin(n, "entry_point", f"tests.fake:{n}") for n in available],
        enabled=enabled, imported={c.manifest.name: c for c in installed},
        version="0.15.0", core_keys=frozenset({"org"}), extra_installed=lambda extra: True)
    settings = settings or {}
    return Resolution(load_plan, classes,
                      {n: settings.get(n, PluginSettings(None, None)) for n in classes}, enabled, ())


@pytest.fixture
def paths(tmp_path) -> PluginPaths:
    return PluginPaths(global_config_dir=tmp_path / "global", workspace=None,
                       state_dir=tmp_path / "state")


async def _start(resolution: Resolution, paths: PluginPaths, registry: ToolRegistry | None = None,
                 hooks: HookSystem | None = None) -> tuple[LifecycleResult, ToolRegistry]:
    registry = registry if registry is not None else ToolRegistry()
    result = await start_plugins(resolution, bus=EventBus(), registry=registry, hooks=hooks,
                                 llm=None, paths=paths)
    return result, registry


def _raises(exc: BaseException):
    """A plugin method (any stage) that raises `exc`."""
    async def method(self, *args):
        raise exc
    return method


def _returns(value):
    """A plugin method (any stage) that returns `value`."""
    async def method(self, *args):
        return value
    return method


def _init_raises(self):
    raise RuntimeError("no init")


# --- PAPI-02: the fixed order ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_each_stage_runs_in_dependency_order_and_stop_runs_in_reverse(paths):
    """`a` requires `b`: every stage runs b before a, stage by stage; stop runs a before b."""
    log: list = []
    a = _plugin("a", log=log, requires=("b",))
    b = _plugin("b", log=log)
    resolution = _resolution(bundled=(a, b))
    assert resolution.plan.order == ("b", "a")  # premise: the plan put the dependency first

    result, _ = await _start(resolution, paths)

    assert log == [("configure", "b"), ("configure", "a"), ("tools", "b"), ("tools", "a"),
                   ("start", "b"), ("start", "a")]
    assert result.loaded_names == ["b", "a"] and not result.failed and not result.warnings
    log.clear()
    await stop_plugins(result)
    assert log == [("stop", "a"), ("stop", "b")]


# --- PAPI-03: exactly the v1 context ---------------------------------------------------------------


class _Colors(BaseModel):
    color: str = "#4a90d9"


class _Size(BaseModel):
    size: int = 8


@pytest.mark.asyncio
async def test_each_plugin_gets_exactly_the_v1_context(paths):
    settings = {"art": PluginSettings(_Colors(), _Size()), "plain": PluginSettings(None, None)}
    resolution = _resolution(bundled=(_plugin("art", wants_artifacts=True), _plugin("plain")),
                             settings=settings)
    bus, registry, hooks, llm = EventBus(), ToolRegistry(), HookSystem(), object()

    result = await start_plugins(resolution, bus=bus, registry=registry, hooks=hooks, llm=llm,
                                 paths=paths)

    ctx = {r.name: r.ctx for r in result.running}
    for name in ("art", "plain"):
        assert [f.name for f in fields(ctx[name])] == [
            "bus", "tools", "hooks", "config", "agent_config", "paths", "llm", "idle_llm",
            "session"]
        assert (ctx[name].bus, ctx[name].tools, ctx[name].hooks, ctx[name].llm) == (
            bus, registry, hooks, llm)
        assert ctx[name].config is settings[name].config
        assert ctx[name].agent_config is settings[name].agent_config
    assert ctx["art"].paths == PluginPaths(paths.global_config_dir, None, paths.state_dir,
                                           paths.state_dir / "artifacts" / "art")
    assert ctx["plain"].paths == paths and ctx["plain"].paths.artifact_dir is None


# --- PAPI-05: bare, global, stamped -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_plugin_tool_registers_bare_at_global_scope_with_its_source_plugin(paths):
    result, registry = await _start(
        _resolution(bundled=(_plugin("palette", contributes=("palette_pick",)),)), paths)

    assert [s.name for s in registry.global_schemas()] == ["palette_pick"]
    assert registry.schema_of("palette_pick").source_plugin == "palette"
    assert result.running[0].tool_names == ("palette_pick",)


# --- PAPI-11: containment, one stage at a time --------------------------------------------------------


def _bad(name: str, case: str) -> type[Plugin]:
    """`name` failing the way `case` says; it would contribute `<name>_tool` and a second tool."""
    two = (f"{name}_tool", f"{name}_other")
    return {
        "init": lambda: _plugin(name, contributes=two, __init__=_init_raises),
        "configure": lambda: _plugin(name, contributes=two, configure=_raises(RuntimeError("bad cfg"))),
        "configure-returns-none": lambda: _plugin(name, contributes=two, configure=_returns(None)),
        "tools": lambda: _plugin(name, tools=_raises(RuntimeError("no tools"))),
        "tools-returns-none": lambda: _plugin(name, tools=_returns(None)),
        "not-a-tool": lambda: _plugin(name, tools=_returns([_Tool(f"{name}_tool"), 42])),
        "builtin-name": lambda: _plugin(name, contributes=(f"{name}_tool", "read")),
        "name-classified": lambda: _plugin(name, contributes=(f"{name}_tool", "agent")),
        "repeated-name": lambda: _plugin(name, contributes=(f"{name}_tool", f"{name}_tool")),
        "start": lambda: _plugin(name, contributes=two, start=_raises(RuntimeError("no start"))),
        "start-exits": lambda: _plugin(name, contributes=two, start=_raises(SystemExit(3))),
    }[case]()


_REASONS = {
    "init": "__init__() raised RuntimeError: no init",
    "configure": "configure() raised RuntimeError: bad cfg",
    "configure-returns-none": "configure() returned None",
    "tools": "tools() raised RuntimeError: no tools",
    "tools-returns-none": "tools() returned NoneType, not a list of tools",
    "not-a-tool": "tools() returned 42, which is not a tool",
    "builtin-name": "its tool 'read' has the name of a tool already registered",
    "name-classified": "its tool 'agent' has a name the permission gate classifies by name",
    "repeated-name": "it contributes two tools named 'bad_tool'",
    "start": "start() raised RuntimeError: no start",
    "start-exits": "start() raised SystemExit: 3",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", list(_REASONS))
async def test_a_failing_stage_disables_only_that_plugin(paths, case, caplog):
    """Whatever stage fails — by raising, by sys.exit(), or by returning something unusable — the
    plugin is failed and named, NOTHING of it stays registered, and the others still run."""
    log: list = []
    registry = ToolRegistry()
    await register_builtin_tools(registry)
    builtins = {s.name for s in registry.global_schemas()}
    resolution = _resolution(bundled=(_plugin("good", log=log, contributes=("good_tool",)), _bad("bad", case)))

    with caplog.at_level(logging.WARNING, logger="localharness.plugins.lifecycle"):
        result, _ = await _start(resolution, paths, registry)

    assert list(result.failed) == ["bad"] and _REASONS[case] in result.failed["bad"], result.failed
    assert f"plugin bad: {result.failed['bad']}" in result.warnings
    assert [r for r in caplog.records if "bad" in r.getMessage()], "the log names the plugin"
    assert result.loaded_names == ["good"]
    assert {s.name for s in registry.global_schemas()} == builtins | {"good_tool"}
    assert [s.source_plugin for s in registry.global_schemas() if s.name in builtins] == [
        None] * len(builtins), "a builtin that shares a plugin's tool name is untouched"
    assert ("start", "good") in log and not result.unconfigured


@pytest.mark.asyncio
async def test_an_unconfigured_plugin_registers_nothing_and_names_the_missing_key(paths):
    log: list = []
    resolution = _resolution(bundled=(
        _plugin("x", log=log, contributes=("x_tool",), configure=_returns(("unconfigured", "x.url"))),
        _plugin("y", log=log, contributes=("y_tool",))))

    result, registry = await _start(resolution, paths)

    assert result.unconfigured == {"x": "x.url"} and not result.failed
    assert [w for w in result.warnings if w.startswith("plugin x:") and "x.url" in w]
    assert registry.schema_of("x_tool") is None and result.loaded_names == ["y"]
    assert ("tools", "x") not in log and ("start", "x") not in log


@pytest.mark.asyncio
async def test_a_failed_start_unregisters_the_tools_it_registered(paths):
    """The tools stage registered them; start() failing takes them back out."""
    resolution = _resolution(bundled=(
        _plugin("bad", contributes=("bad_tool",), start=_raises(RuntimeError("no start"))),))
    registered: list = []
    registry = ToolRegistry()
    real_register = registry.register

    async def spy(tool, *args, **kwargs):
        registered.append(tool.info().name)
        await real_register(tool, *args, **kwargs)

    registry.register = spy
    result, _ = await _start(resolution, paths, registry)

    assert registered == ["bad_tool"], "premise: the tool really was registered before start"
    assert registry.schema_of("bad_tool") is None and "bad" in result.failed


@pytest.mark.asyncio
async def test_a_stop_that_raises_is_logged_and_the_next_plugin_still_stops(paths, caplog):
    log: list = []
    resolution = _resolution(bundled=(
        _plugin("first", log=log),
        _plugin("second", log=log, stop=_raises(SystemExit(4)))))
    result, _ = await _start(resolution, paths)
    log.clear()

    with caplog.at_level(logging.WARNING, logger="localharness.plugins.lifecycle"):
        await stop_plugins(result)

    assert log == [("stop", "first")]  # second (started last) raised first, then first stopped
    assert [r for r in caplog.records if "second" in r.getMessage() and "stop" in r.getMessage()]


class _Veto:
    @HARNESS_HOOKIMPL
    def pre_tool(self, name, arguments, agent_id, division_id):
        raise AssertionError("a disabled plugin's hook ran")


async def _hook_then_fail(self, ctx):
    ctx.hooks.register_plugin(_Veto(), name="hooky")
    raise RuntimeError("after hooking")


async def _hook_then_unconfigured(self, ctx):
    ctx.hooks.register_plugin(_Veto(), name="hooky")
    return ("unconfigured", "hooky.url")


@pytest.mark.asyncio
@pytest.mark.parametrize("methods", [{"start": _hook_then_fail},
                                     {"configure": _hook_then_unconfigured}])
async def test_a_disabled_plugins_hooks_are_unregistered(paths, methods):
    """A hook object a plugin put on ctx.hooks goes with it when it fails or turns out unconfigured
    — 'disabled for the session' covers its pre_tool hooks (which could veto) too."""
    hooks = HookSystem()
    kept = _Veto()
    hooks.register_plugin(kept, name="core")  # registered before any plugin ran: never touched
    result, _ = await _start(_resolution(bundled=(_plugin("hooky", **methods),)), paths,
                             hooks=hooks)

    assert "hooky" in result.failed or "hooky" in result.unconfigured
    assert hooks.pm.get_plugins() == {kept}, "only the disabled plugin's hook object is gone"


# --- PAPI-11: the requires cascade ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_plugin_whose_requirement_fails_is_failed_too(paths):
    log: list = []
    resolution = _resolution(bundled=(
        _plugin("a", log=log, requires=("b",), contributes=("a_tool",)),
        _plugin("b", log=log, configure=_raises(RuntimeError("bad cfg")))))

    result, registry = await _start(resolution, paths)

    assert result.failed["a"] == "requires b, which failed this session"
    assert "plugin a: requires b, which failed this session" in result.warnings
    assert not [c for c in log if c[1] == "a"], "a never ran a stage"
    assert registry.schema_of("a_tool") is None and result.loaded_names == []


@pytest.mark.asyncio
async def test_a_dependent_loses_its_tools_when_its_requirement_fails_to_start(paths):
    """Stage by stage, `a` has registered its tools by the time `b` fails at start: they go too."""
    resolution = _resolution(bundled=(
        _plugin("a", requires=("b",), contributes=("a_tool",)),
        _plugin("b", contributes=("b_tool",), start=_raises(RuntimeError("no start")))))

    result, registry = await _start(resolution, paths)

    assert result.failed == {"b": "start() raised RuntimeError: no start",
                             "a": "requires b, which failed this session"}
    assert registry.global_schemas() == []


@pytest.mark.asyncio
async def test_a_plugin_whose_requirement_is_unconfigured_does_not_run(paths):
    resolution = _resolution(bundled=(
        _plugin("a", requires=("b",)),
        _plugin("b", configure=_returns(("unconfigured", "b.token")))))

    result, _ = await _start(resolution, paths)

    assert result.unconfigured == {"b": "b.token"}
    assert result.failed == {"a": "requires b, which is unconfigured this session"}


# --- SAFE-06: the third-party clamp, applied at registration -------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("bundled, declared, kept", [
    (False, "allow", None), (False, "code", "code"), (True, "allow", "allow")])
async def test_only_a_plugin_you_installed_has_its_gate_family_clamped(paths, bundled, declared, kept):
    cls = _plugin("third", contributes=(_Tool("third_tool", gate_family=declared),))
    resolution = _resolution(**({"bundled": (cls,)} if bundled else {"installed": (cls,)}))
    assert resolution.plan.entry("third").bundled is bundled  # premise

    result, registry = await _start(resolution, paths)

    assert registry.schema_of("third_tool").gate_family == kept
    assert registry.schema_of("third_tool").source_plugin == "third"
    clamp = [w for w in result.warnings if w.startswith("plugin third:")]
    if kept is None:
        assert len(clamp) == 1 and "'third_tool'" in clamp[0] and "'allow'" in clamp[0]
    else:
        assert clamp == []


# --- PAPI-10: core computes the artifact root; anything else is refused ---------------------------------


@pytest.mark.asyncio
async def test_a_plugin_that_names_its_own_artifact_root_gets_no_artifact_serving(paths, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    resolution = _resolution(bundled=(
        _plugin("rogue", wants_artifacts=True, artifact_root=lambda self, ctx: elsewhere),
        _plugin("art", wants_artifacts=True)))

    result, _ = await _start(resolution, paths)

    expected = paths.state_dir / "artifacts" / "rogue"
    assert result.artifact_roots == {"art": paths.state_dir / "artifacts" / "art"}
    assert result.loaded_names == ["rogue", "art"], "only its artifact serving is disabled"
    assert [w for w in result.warnings
            if w.startswith("plugin rogue:") and str(elsewhere) in w and str(expected) in w]


@pytest.mark.asyncio
async def test_an_artifact_root_that_raises_costs_only_artifact_serving(paths):
    resolution = _resolution(bundled=(
        _plugin("art", wants_artifacts=True, artifact_root=lambda self, ctx: 1 / 0),))

    result, _ = await _start(resolution, paths)

    assert result.artifact_roots == {} and result.loaded_names == ["art"]
    assert result.warnings == ["plugin art: artifact serving disabled this session — its "
                               "artifact_root() raised ZeroDivisionError: division by zero"]


@pytest.mark.asyncio
async def test_the_core_root_is_accepted_however_it_is_spelled(paths):
    """The comparison is on resolved paths: `.../artifacts/x/../art` is the core root."""
    spelled = paths.state_dir / "artifacts" / "x" / ".." / "art"
    resolution = _resolution(bundled=(
        _plugin("art", wants_artifacts=True, artifact_root=lambda self, ctx: spelled),))

    result, _ = await _start(resolution, paths)

    assert result.artifact_roots == {"art": paths.state_dir / "artifacts" / "art"}
    assert result.warnings == []


# --- PAPI-07: slash rows, imported on first use -----------------------------------------------------------


@pytest.mark.asyncio
async def test_a_slash_row_imports_its_target_only_when_it_runs(paths, tmp_path, monkeypatch):
    module = f"lh_slash_probe_{uuid.uuid4().hex}"
    (tmp_path / f"{module}.py").write_text(
        "CALLS = []\n"
        "async def run(ctx, args):\n"
        "    CALLS.append((ctx, args))\n"
        "    return 'got ' + args\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    resolution = _resolution(bundled=(
        _plugin("probe", slash=(SlashDescriptor(name="/probe", help="Probe it",
                                                target=f"{module}:run"),)),
        _plugin("dead", slash=(SlashDescriptor(name="/dead", help="Never",
                                               target=f"{module}:run"),),
                start=_raises(RuntimeError("no start")))))
    try:
        result, _ = await _start(resolution, paths)
        [row] = result.slash_rows  # a failed plugin has no rows
        assert (row.name, row.description, row.takes_args, row.plugin) == (
            "/probe", "Probe it", True, "probe")
        assert module not in sys.modules, "the target is not imported at session start"

        assert await row.handler("a b") == "got a b"

        assert sys.modules[module].CALLS == [(result.running[0].ctx, "a b")]
    finally:
        sys.modules.pop(module, None)


# --- PAPI-08's data: doctor rows without a session ---------------------------------------------------------


@pytest.mark.asyncio
async def test_doctor_rows_cover_every_plugin_state_without_a_session(paths):
    seen: list[PluginContext] = []

    def doctor(self, ctx):
        seen.append(ctx)
        return [Check(name="checked", status="pass", detail="fine")]

    resolution = _resolution(
        bundled=(_plugin("checked", doctor=doctor, wants_artifacts=True),
                 _plugin("future", requires_localharness=">=9"),
                 _plugin("raising", doctor=lambda self, ctx: 1 / 0),
                 _plugin("junk", doctor=lambda self, ctx: ["not a check"]),
                 _plugin("waiting", configure=_returns(("unconfigured", "waiting.url"))),
                 _plugin("broken", configure=_raises(SystemExit(2)))),
        off=(_plugin("off"),), available=("shiny",))

    rows = {row.name: row for row in await doctor_rows(resolution, paths=paths)}

    assert rows["shiny"] == DoctorRow(
        "shiny", "available", "available — turn on: localharness plugins enable shiny")
    assert rows["off"] == DoctorRow("off", "off", "off — turn on: localharness plugins enable off",
                                    bundled=True)
    assert rows["future"] == DoctorRow(
        "future", "skipped", "skipped — requires localharness >=9, this is 0.15.0", bundled=True)
    assert rows["checked"] == DoctorRow(
        "checked", "on", "", (Check(name="checked", status="pass", detail="fine"),), bundled=True)
    [ctx] = seen
    assert ctx.llm is None and ctx.paths.artifact_dir == paths.state_dir / "artifacts" / "checked"
    [check] = rows["raising"].checks
    assert (check.name, check.status) == ("raising", "fail")
    assert check.detail == "its doctor check raised ZeroDivisionError: division by zero"
    [check] = rows["junk"].checks
    assert check.status == "fail" and "not a list of Check" in check.detail
    assert rows["waiting"] == DoctorRow("waiting", "unconfigured", "unconfigured — set waiting.url",
                                        bundled=True)
    assert rows["broken"] == DoctorRow("broken", "failed", "failed — configure() raised SystemExit: 2",
                                       bundled=True)
    assert list(rows) == [e.name for e in resolution.plan.entries]


# --- ctx.idle_llm / ctx.session (47-02, MEMP-07 / G4) ------------------------------------------


def _recorder(name: str, seen: dict, stage: str = "start"):
    """A plugin class whose `stage` method records the ctx it was handed."""
    async def record(self, ctx):
        seen[name] = ctx
        return [] if stage == "doctor" else None
    return _plugin(name, **{stage: record})


@pytest.mark.asyncio
async def test_a_started_plugin_gets_idle_llm_and_session(paths):
    from localharness.plugins.api import SessionInfo
    from localharness.provider.idle_llm import LLMTextAdapter
    seen: dict = {}
    client = object()
    info = SessionInfo("agent", "div", "sit", "model", 4096, {"max_turns": 3})
    result = await start_plugins(_resolution(bundled=(_recorder("p", seen),)), bus=EventBus(),
                                 registry=ToolRegistry(), hooks=None, llm=client, paths=paths,
                                 session=info)
    assert result.loaded_names == ["p"]
    assert isinstance(seen["p"].idle_llm, LLMTextAdapter) and seen["p"].idle_llm._client is client
    assert seen["p"].session is info


@pytest.mark.asyncio
async def test_no_llm_means_no_idle_llm_and_no_session_by_default(paths):
    seen: dict = {}
    await _start(_resolution(bundled=(_recorder("p", seen),)), paths)
    assert seen["p"].idle_llm is None and seen["p"].session is None


@pytest.mark.asyncio
async def test_doctor_ctx_has_no_idle_llm_or_session(paths):
    seen: dict = {}
    await doctor_rows(_resolution(bundled=(_recorder("p", seen, stage="doctor"),)), paths=paths)
    assert seen["p"].idle_llm is None and seen["p"].session is None


# --- startup_warnings channel + interrupt-safe start (47-02, gaps P1/P3) --------------------------


def _warns(name: str, warnings, *, then: BaseException | None = None, log: list | None = None):
    """A plugin that sets `startup_warnings` in __init__ and appends to it during start()."""
    def __init__(self):
        self.startup_warnings = [] if isinstance(warnings, list) else warnings

    async def start(self, ctx):
        if isinstance(self.startup_warnings, list):
            self.startup_warnings.extend(warnings)
        if then is not None:
            raise then
    return _plugin(name, log=log, __init__=__init__, start=start)


@pytest.mark.asyncio
async def test_startup_warnings_reach_the_result_verbatim(paths):
    result, _ = await _start(_resolution(bundled=(_warns("mem", ["session-start: boom"]),
                                                  _plugin("plain"))), paths)
    assert result.warnings == ["session-start: boom"]
    assert set(result.loaded_names) == {"mem", "plain"}  # "plain" has no attribute: contributes nothing


@pytest.mark.asyncio
async def test_startup_warnings_of_a_failed_start_are_still_reported(paths):
    result, _ = await _start(
        _resolution(bundled=(_warns("mem", ["session-start: boom"], then=RuntimeError("x")),)), paths)
    assert "session-start: boom" in result.warnings
    assert "mem" in result.failed


@pytest.mark.asyncio
async def test_a_non_list_startup_warnings_is_ignored(paths):
    result, _ = await _start(_resolution(bundled=(_warns("s", "oops"),)), paths)
    assert result.warnings == []  # a str is not iterated character by character

    def __init__(self):
        self.startup_warnings = []

    async def start(self, ctx):
        self.startup_warnings.extend(["kept", 7, None, "also kept"])
    result, _ = await _start(_resolution(bundled=(_plugin("m", __init__=__init__, start=start),)),
                             paths)
    assert result.warnings == ["kept", "also kept"]


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), asyncio.CancelledError()],
                         ids=["keyboard_interrupt", "cancelled_error"])
@pytest.mark.asyncio
async def test_keyboard_interrupt_in_a_start_stops_the_running_plugins_then_reraises(paths, interrupt):
    """A then B in plan order; B.start is interrupted -> A (already running) is stopped once, B
    (not running) is not stopped — it cleans up its own half-open state — and the interrupt
    propagates."""
    log: list = []
    a = _plugin("a", log=log)
    b = _plugin("b", log=log, requires=("a",), start=_raises(interrupt))
    resolution = _resolution(bundled=(a, b))
    assert resolution.plan.order == ("a", "b")
    with pytest.raises(type(interrupt)):
        await _start(resolution, paths)
    assert log.count(("stop", "a")) == 1
    assert ("stop", "b") not in log


@pytest.mark.asyncio
async def test_cancelled_error_in_a_start_is_handled_the_same(paths):
    log: list = []
    resolution = _resolution(bundled=(_plugin("a", log=log),
                                      _plugin("b", log=log, requires=("a",),
                                              start=_raises(asyncio.CancelledError()))))
    with pytest.raises(asyncio.CancelledError):
        await _start(resolution, paths)
    assert log.count(("stop", "a")) == 1 and ("stop", "b") not in log


# --- 48 M2: a "Class.method" target is called on the running instance ------------------------------------


@pytest.fixture
def bound_module(tmp_path, monkeypatch):
    """A throwaway module holding a plugin class with a slash method, another class with one, and a
    plain function — each says which it was, so a test sees which one the handler reached."""
    module = f"lh_slash_bound_{uuid.uuid4().hex}"
    (tmp_path / f"{module}.py").write_text(
        "class P:\n"
        "    def __init__(self, marker):\n"
        "        self.marker = marker\n"
        "    async def slash(self, ctx, args):\n"
        "        return ('bound', self.marker, ctx, args)\n"
        "class Other:\n"
        "    @staticmethod\n"
        "    async def slash(ctx, args):\n"
        "        return ('other', ctx, args)\n"
        "async def fn(ctx, args):\n"
        "    return ('fn', ctx, args)\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield module
    sys.modules.pop(module, None)


def _rp(plugin, ctx):
    from localharness.plugins.lifecycle import RunningPlugin
    return RunningPlugin(name="p", plugin=plugin, ctx=ctx, bundled=True)


@pytest.mark.asyncio
async def test_a_method_of_the_running_plugins_class_is_called_on_that_instance(bound_module):
    from localharness.plugins.lifecycle import _slash_handler
    import importlib
    live = importlib.import_module(bound_module).P("live-state")
    ctx = object()
    handler = _slash_handler(f"{bound_module}:P.slash", _rp(live, ctx))
    assert await handler("a b") == ("bound", "live-state", ctx, "a b")


@pytest.mark.asyncio
async def test_a_plain_function_target_is_unchanged(bound_module):
    from localharness.plugins.lifecycle import _slash_handler
    import importlib
    ctx = object()
    handler = _slash_handler(f"{bound_module}:fn", _rp(importlib.import_module(bound_module).P("x"), ctx))
    assert await handler("hi") == ("fn", ctx, "hi")


@pytest.mark.asyncio
async def test_a_method_of_some_other_class_is_imported_and_called_as_before(bound_module):
    from localharness.plugins.lifecycle import _slash_handler
    import importlib
    ctx = object()
    handler = _slash_handler(f"{bound_module}:Other.slash",
                             _rp(importlib.import_module(bound_module).P("x"), ctx))
    assert await handler("hi") == ("other", ctx, "hi")


@pytest.mark.asyncio
async def test_a_non_class_owner_falls_back_to_the_plain_import(bound_module):
    from localharness.plugins.lifecycle import _slash_handler
    import importlib
    ctx = object()
    # "fn.__call__" — owner "fn" is a function, not a class: no binding attempted, plain import.
    handler = _slash_handler(f"{bound_module}:fn.__call__",
                             _rp(importlib.import_module(bound_module).P("x"), ctx))
    assert await handler("hi") == ("fn", ctx, "hi")
