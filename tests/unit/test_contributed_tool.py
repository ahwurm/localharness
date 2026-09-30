"""PAPI-05 / PAPI-11: a plugin's tool is held exactly like a builtin, knows where it came from, and
cannot take a turn down.

`ToolRegistry.register(tool, source_plugin=...)` registers a plugin tool under its bare name at
global scope, no prefix and no separate bucket, so every safety reader judges it by its
declaration alone (CORE-04). Its provenance rides on the schema every reader gets from
`tool.info()`, and a loader override (SAFE-06's gate-family clamp for a plugin you installed) is
applied there too, on EVERY call: the gate reads `tool.info()` per call (agent/loop.py
`_tool_facts`), not the registry's stored copy.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import logging
import sys
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, get_args

import pytest

from localharness.agent import verdict
from localharness.agent.gate import tool_meta_from_schema
from localharness.agent.gate_types import GateSettings, Mode, Verdict
from localharness.config.models import ToolConfig
from localharness.plugins.api import PluginContext, PluginPaths
from localharness.tools.base import Tool, ToolResult, ToolSchema
from localharness.tools.capabilities import ingests_untrusted, is_exec, is_host_dangerous
from localharness.tools.registry import ToolRegistry

CFG = ToolConfig(inherit=["global"])
ROOT = ("root", "")  # agent_id, division_id


def _declared(name: str, family: str | None) -> ToolSchema:
    """A schema as a plugin author writes one: all four safety declarations explicit."""
    return ToolSchema(name=name, description=f"{name} (test plugin tool)",
                      parameters={"type": "object", "properties": {}},
                      ingest="none", host="safe", result_origin="trusted", gate_family=family)


class _Swatch(Tool):
    """A well-behaved plugin tool, written the usual way (a Tool subclass)."""

    timeout_s = 7.0

    def __init__(self, name: str = "swatch", family: str | None = "code") -> None:
        super().__init__()
        self._schema = _declared(name, family)

    def info(self) -> ToolSchema:
        return self._schema

    async def _execute(self, **kwargs: Any) -> ToolResult:
        return self.ok("painted")


class _Raises:
    """A plugin tool that satisfies ToolProtocol directly and raises from run(): nothing between it
    and the loop catches that except the registry's wrapper (the loop catches Exception, not
    SystemExit, and never says which plugin failed)."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def info(self) -> ToolSchema:
        return _declared("boom", "code")

    async def run(self, **kwargs: Any) -> ToolResult:
        raise self._exc


async def test_a_plugin_tool_registers_bare_at_global_scope_and_every_reader_sees_its_plugin():
    reg = ToolRegistry()
    await reg.register(_Swatch(), source_plugin="palette")
    await reg.register(_Swatch("core_like", "allow"))  # core's own tool: no provenance

    assert set(reg._tools["global"]) == {"swatch", "core_like"}  # bare and global, like a builtin
    readers = {
        "schema_of": reg.schema_of("swatch"),
        "schema_of(plugin:P.T)": reg.schema_of("plugin:palette.swatch"),
        "get_tools_for_agent": reg.get_tools_for_agent(*ROOT, CFG)["swatch"],
        "lookup_tool().info()": reg.lookup_tool("swatch", *ROOT, CFG).info(),
        "global_schemas": next(s for s in reg.global_schemas() if s.name == "swatch"),
        "a child built by from_allowed":
            ToolRegistry.from_allowed(["plugin:palette.swatch"], reg).schema_of("swatch"),
    }
    assert {k: s.source_plugin for k, s in readers.items()} == dict.fromkeys(readers, "palette")
    assert reg.schema_of("core_like").source_plugin is None
    assert "source_plugin" not in reg.schema_of("swatch").model_dump()  # off the wire (44-02)

    await reg.unregister("swatch")  # what the lifecycle does to a plugin that fails after tools()
    assert reg.schema_of("swatch") is None


async def test_a_loader_override_is_what_every_reader_and_the_gate_see(tmp_path: Path):
    tool = _Swatch("sprayer", "allow")
    reg = ToolRegistry()
    await reg.register(tool, source_plugin="palette", overrides={"gate_family": None})

    held = reg.lookup_tool("sprayer", *ROOT, CFG)
    seen = [reg.schema_of("sprayer"), held.info(), held.info(),  # every call, not only the first
            reg.get_tools_for_agent(*ROOT, CFG)["sprayer"], *reg.global_schemas()]
    assert [(s.gate_family, s.source_plugin) for s in seen] == [(None, "palette")] * len(seen)
    assert tool.info().gate_family == "allow"  # applied on read; the plugin's object is untouched

    meta = tool_meta_from_schema(held.info())  # exactly what agent/loop.py _tool_facts hands the gate
    assert meta.gate_family is None
    ctx = verdict.GateContext(boundary=tmp_path, workspace=tmp_path, grants=lambda *_: None,
                              mode="guarded")
    assert verdict.evaluate("sprayer", {}, meta, ctx, GateSettings()).verdict is Verdict.ASK


async def test_an_override_without_a_plugin_is_refused():
    reg = ToolRegistry()
    with pytest.raises(ValueError, match="source_plugin"):
        await reg.register(_Swatch(), overrides={"gate_family": None})
    assert reg.schema_of("swatch") is None


@pytest.mark.parametrize("exc", [RuntimeError("boom"), SystemExit(3)], ids=lambda e: type(e).__name__)
async def test_a_plugin_tool_that_raises_is_an_attributed_error_and_the_turn_goes_on(exc, caplog):
    reg = ToolRegistry()
    await reg.register(_Raises(exc), source_plugin="palette")
    with caplog.at_level(logging.WARNING, logger="localharness.tools.registry"):
        result = await reg.dispatch("boom", {}, *ROOT, CFG)

    assert (result.success, result.error_type) == (False, "execution_error")
    assert "'palette'" in result.error and f"{type(exc).__name__}: {exc}" in result.error
    assert [r.levelno for r in caplog.records if "'palette'" in r.getMessage()] == [logging.WARNING]


class _Fails(Tool):
    """The usual plugin tool shape — a Tool subclass — whose own code raises `outcome`, or returns it
    as the error it chose to report."""

    def __init__(self, name: str, outcome: BaseException | str) -> None:
        super().__init__()
        self._name, self._outcome = name, outcome

    def info(self) -> ToolSchema:
        return _declared(self._name, "code")

    async def _execute(self, **kwargs: Any) -> ToolResult:
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self.err(self._outcome)


async def test_a_tool_subclass_that_raises_is_attributed_and_its_own_refusal_is_not(caplog):
    """PAPI-11 for the common plugin tool: Tool.run catches what _execute raises, so the error names
    the plugin only if the wrapper can tell a crash from a refusal — a result the tool chose to
    return as an error is its own words, left alone and not logged."""
    reg = ToolRegistry()
    await reg.register(_Fails("kaboom", RuntimeError("kaboom")), source_plugin="palette")
    await reg.register(_Fails("refuses", "no artifact directory was assigned"), source_plugin="palette")
    with caplog.at_level(logging.WARNING, logger="localharness.tools.registry"):
        raised = await reg.dispatch("kaboom", {}, *ROOT, CFG)
        refused = await reg.dispatch("refuses", {}, *ROOT, CFG)

    assert (raised.success, raised.error_type) == (False, "execution_error")
    assert raised.error == "tool 'kaboom' from plugin 'palette' failed: RuntimeError: kaboom"
    assert (refused.success, refused.error) == (False, "no artifact directory was assigned")
    logged = [r for r in caplog.records if "'palette'" in r.getMessage()]
    assert [(r.levelno, "kaboom" in r.getMessage()) for r in logged] == [(logging.WARNING, True)]


async def test_cancelling_a_turn_still_cancels_a_plugin_tool():
    reg = ToolRegistry()
    await reg.register(_Raises(asyncio.CancelledError()), source_plugin="palette")
    with pytest.raises(asyncio.CancelledError):
        await reg.dispatch("boom", {}, *ROOT, CFG)


async def test_attribute_reads_fall_through_to_the_plugins_own_tool():
    reg = ToolRegistry()
    await reg.register(_Swatch(), source_plugin="palette")
    held = reg.lookup_tool("swatch", *ROOT, CFG)

    assert held.timeout_s == 7.0  # the gate's human-wait budget (agent/loop.py _tool_facts)
    assert (await reg.dispatch("swatch", {}, *ROOT, CFG)).output == "painted"
    with pytest.raises(AttributeError):
        held.no_such_attribute  # noqa: B018


def test_the_name_classified_set_is_every_table_kind_reads_before_a_declaration():
    """Derived from `_kind`'s own source: a sixth name table added there without joining the set
    would let a plugin tool with that name be judged by its name instead of its declaration."""
    tree = ast.parse(inspect.getsource(verdict._kind))
    tables = {node.comparators[0].id for node in ast.walk(tree)
              if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
              and node.left.id == "tool_name" and isinstance(node.ops[0], ast.In)}
    assert len(tables) == 5, tables
    assert verdict.NAME_CLASSIFIED_TOOLS == frozenset().union(*(getattr(verdict, t) for t in tables))
    assert verdict.NAME_CLASSIFIED_TOOLS == {
        "write", "edit", "bash_exec", "python_exec", "cruncher_exec", "agent", "web_fetch"}


async def test_provenance_changes_no_safety_reading_of_a_real_plugin_tool(tmp_path: Path):
    """CORE-04 on the real registration path, with the example plugin's own tool (44-08, loaded
    through its real entry point): registered with source_plugin or without, the gate, the
    capability floor and the context store judge it identically, because they read its declaration
    and nothing else. The only difference a reader can find is the provenance itself."""
    before = set(sys.modules)
    try:
        (ep,) = [e for e in entry_points(group="localharness.plugins") if e.name == "example"]
        plugin_cls = ep.load()
        ctx = PluginContext(
            bus=None, tools=None, hooks=None, config=plugin_cls.ConfigModel(),
            agent_config=plugin_cls.AgentConfigModel(), llm=None,
            paths=PluginPaths(global_config_dir=tmp_path / "cfg", workspace=None,
                              state_dir=tmp_path / "state", artifact_dir=tmp_path / "art"))
        judged = {}
        for plugin in (None, "example"):
            (tool,) = await plugin_cls().tools(ctx)
            reg = ToolRegistry()
            await reg.register(tool, source_plugin=plugin)
            schema = reg.lookup_tool("example_swatch", *ROOT, CFG).info()
            meta = tool_meta_from_schema(schema)
            judged[plugin] = (schema.source_plugin, (
                ingests_untrusted(schema), is_host_dangerous(schema), is_exec(schema),
                reg.result_origin("example_swatch"), sorted(reg.get_tools_for_agent(*ROOT, CFG)),
                [verdict.evaluate("example_swatch", {}, meta, verdict.GateContext(
                    boundary=tmp_path, workspace=tmp_path, grants=lambda *_: None, mode=mode),
                    GateSettings()).verdict for mode in get_args(Mode)],
                (await reg.dispatch("example_swatch", {}, *ROOT, CFG)).success))
    finally:
        for name in set(sys.modules) - before:
            if name.startswith("localharness_plugin_example"):
                del sys.modules[name]

    assert (judged[None][0], judged["example"][0]) == (None, "example")
    assert judged[None][1][-1] is True  # the real tool ran (its PNG written), so the tuples mean it
    assert judged["example"][1] == judged[None][1]
