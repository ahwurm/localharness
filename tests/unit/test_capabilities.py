"""P-A capability floor: no agent may co-resident untrusted-ingest + host-dangerous tools.

The floor reads each tool's DECLARATION (`ingest`, `host`) — never its name, never which plugin it
came from (SAFE-02, CORE-04). Covers the pure predicate (assert_no_coresidence), the memory-clean
invariant (memory declares ingest: none), fail-closed defaults (a tool that declares nothing
co-resides with itself), both resolution chokepoints (from_allowed + get_tools_for_agent), the root
floor (strips every global tool that declares — or defaults to — ingest: untrusted), and a sanity
sweep that every built-in subagent toolset stays clean.
"""
from __future__ import annotations

import pytest

from localharness.agent.context import ContentStore
from localharness.config.models import ToolConfig
from localharness.tools.base import ToolResult, ToolSchema
from localharness.tools.builtin import register_builtin_tools
from localharness.tools.capabilities import (
    CoResidenceError,
    apply_root_capability_floor,
    assert_no_coresidence,
)
from localharness.tools.registry import ToolRegistry

WEB = ["web_fetch", "web_page_query", "web_search"]


class _Bare:
    """A tool that declares nothing — a plugin author's omission. Every axis fails closed."""

    def __init__(self, name: str = "undeclared") -> None:
        self._name = name

    def info(self) -> ToolSchema:
        return ToolSchema(name=self._name, description="d", parameters={})

    async def run(self, **kwargs) -> ToolResult:  # pragma: no cover - never dispatched here
        return ToolResult(output="")


def _decl(name: str, *, ingest: str = "none", host: str = "safe") -> ToolSchema:
    """A synthetic tool with explicit declarations (a plugin's, say): judged by what it says."""
    return ToolSchema(name=name, description="d", parameters={}, ingest=ingest, host=host,
                      result_origin="trusted")


async def _production_registry() -> ToolRegistry:
    """The root's global tools as start registers them: builtins, the memory verbs, tool_result_get."""
    reg = ToolRegistry()
    await register_builtin_tools(reg, eviction_store=ContentStore())
    # the memory verbs reach a real root from the memory plugin; registered directly here
    from localharness.tools.builtin.memory_tools import MemoryGetTool, MemoryRememberTool, MemorySearchTool
    _mem = object()
    for _mt in (MemorySearchTool(_mem), MemoryGetTool(_mem), MemoryRememberTool(_mem)):
        await reg.register(_mt, scope="global")
    return reg


async def _builtin_registry() -> ToolRegistry:
    """Every builtin registered: the production set plus the three the start/subagent paths build
    directly (python_exec, cruncher_exec, agent) — so every builtin NAME resolves to its schema."""
    from localharness.tools.builtin.agent_tool import AgentTool
    from localharness.tools.builtin.cruncher_exec import CruncherExecTool
    from localharness.tools.builtin.python_tool import PythonExecTool

    async def _runner(*_a, **_k):  # pragma: no cover - never invoked
        return ""

    reg = await _production_registry()
    for tool in (PythonExecTool(), CruncherExecTool(seed={}), AgentTool(agent_runner=_runner)):
        await reg.register(tool, scope="global")
    return reg


@pytest.fixture
async def schemas() -> dict[str, ToolSchema]:
    from localharness.tools.mcp import MCPToolWrapper

    reg = await _builtin_registry()
    out = {t.info().name: t.info() for t in reg._tools["global"].values()}
    out["mcp:fetch"] = MCPToolWrapper("fetch", "d", {}, session=None, server_name="srv").info()
    out["bare"] = _Bare("bare").info()
    out["plugin_search"] = _decl("plugin_search", ingest="untrusted")   # declares it ingests
    out["plugin_fmt"] = _decl("plugin_fmt", host="dangerous")           # declares it touches host
    return out


# --- Pure predicate -------------------------------------------------------

@pytest.mark.parametrize(
    "names",
    [
        ("web_fetch", "bash_exec"),
        ("web_search", "write"),
        ("web_page_query", "python_exec"),
        ("mcp:fetch", "bash_exec"),        # MCP ingestion (declared in wrapper code) + host-dangerous
        ("plugin_search", "write"),        # a plugin that declares ingest + host-dangerous
        ("bare",),                         # undeclared: ingest AND host-dangerous — co-resides with itself
    ],
)
def test_coresidence_raises(schemas, names):
    with pytest.raises(CoResidenceError):
        assert_no_coresidence([schemas[n] for n in names])


@pytest.mark.parametrize(
    "names",
    [
        ("web_fetch", "web_page_query"),               # web-only
        ("bash_exec", "write", "edit"),                # danger-only
        ("memory_search", "memory_get", "bash_exec"),  # memory declares ingest: none (the clean invariant)
        ("mcp:fetch", "web_page_query"),               # ingest-only (mcp + web), no host-dangerous
        ("plugin_fmt", "bash_exec"),                   # a plugin is judged by its declaration, not its source
    ],
)
def test_no_coresidence_passes(schemas, names):
    assert_no_coresidence([schemas[n] for n in names])  # must not raise


def test_the_message_names_what_each_declaration_put_on_each_side(schemas):
    with pytest.raises(CoResidenceError) as exc_info:
        assert_no_coresidence([schemas["web_fetch"], schemas["bash_exec"]], agent_id="root")
    msg = str(exc_info.value)
    assert "for agent 'root'" in msg and "['web_fetch']" in msg and "['bash_exec']" in msg


# --- Chokepoint: from_allowed --------------------------------------------

@pytest.mark.asyncio
async def test_from_allowed_rejects_coresident():
    full = ToolRegistry()
    await register_builtin_tools(full)
    with pytest.raises(CoResidenceError):
        ToolRegistry.from_allowed(["web_fetch", "bash_exec"], base_registry=full)


@pytest.mark.asyncio
async def test_from_allowed_rejects_mcp_ingest_plus_bash():
    """An mcp:/plugin: entry the base registry cannot resolve is judged by its declared INTENT —
    the MCP wrapper's posture (ingest untrusted) — so it is rejected beside a host-dangerous tool
    even when no live MCP server or plugin backs it."""
    full = ToolRegistry()
    await register_builtin_tools(full)
    with pytest.raises(CoResidenceError):
        ToolRegistry.from_allowed(["mcp:fetch", "bash_exec"], base_registry=full)
    # plugin ingestion + host-dangerous likewise
    with pytest.raises(CoResidenceError):
        ToolRegistry.from_allowed(["plugin:research_tools.exa_search", "write"], base_registry=full)


@pytest.mark.asyncio
async def test_from_allowed_judges_a_resolved_plugin_tool_by_its_declaration():
    """Resolved, the plugin-form entry is read off the tool itself: undeclared fails closed (it
    co-resides with itself), whatever its prefix says."""
    base = ToolRegistry()
    await base.register(_Bare("exa_search"), scope="global")
    with pytest.raises(CoResidenceError):
        ToolRegistry.from_allowed(["plugin:research_tools.exa_search"], base_registry=base)


@pytest.mark.asyncio
async def test_from_allowed_allows_web_only():
    full = ToolRegistry()
    await register_builtin_tools(full)
    out = ToolRegistry.from_allowed(["web_fetch", "web_page_query", "web_search"], base_registry=full)
    assert out.has("web_fetch")


# --- Chokepoint: get_tools_for_agent -------------------------------------

@pytest.mark.asyncio
async def test_get_tools_for_agent_rejects_coresident():
    reg = ToolRegistry()
    await register_builtin_tools(reg)
    # Default ToolConfig inherits 'global' -> resolves web_* AND bash_exec/write/edit => co-resident.
    cfg = ToolConfig()
    with pytest.raises(CoResidenceError):
        reg.get_tools_for_agent("root", "", cfg)


@pytest.mark.asyncio
async def test_an_undeclared_tool_cannot_be_held_even_alone():
    reg = ToolRegistry()
    await reg.register(_Bare(), scope="global")
    with pytest.raises(CoResidenceError):
        reg.get_tools_for_agent("anyone", "", ToolConfig())


# --- Root floor (§3): strip by declaration, after registration ------------

@pytest.mark.asyncio
async def test_root_floor_denies_exactly_the_declared_ingest_tools():
    reg = await _production_registry()
    cfg = ToolConfig()
    assert apply_root_capability_floor(cfg, reg.global_schemas(), enabled=True) == WEB
    assert sorted(cfg.deny) == WEB
    apply_root_capability_floor(cfg, reg.global_schemas(), enabled=True)
    assert sorted(cfg.deny) == WEB, "a second pass must not duplicate deny entries"


@pytest.mark.asyncio
async def test_root_floor_strips_an_undeclared_global_tool():
    """A global tool nobody described defaults to ingest: untrusted, so the root floor denies it —
    otherwise one undeclared plugin tool would make every root turn raise CoResidenceError."""
    reg = await _production_registry()
    await reg.register(_Bare(), scope="global")
    cfg = ToolConfig()
    assert apply_root_capability_floor(cfg, reg.global_schemas(), enabled=True) == sorted([*WEB, "undeclared"])
    resolved = reg.get_tools_for_agent("root", "", cfg)  # must not raise
    assert "undeclared" not in resolved and "bash_exec" in resolved


def test_root_floor_disabled_denies_nothing():
    cfg = ToolConfig()
    assert apply_root_capability_floor(cfg, [_Bare().info()], enabled=False) == []
    assert cfg.deny == []


@pytest.mark.asyncio
async def test_root_strip_resolves_clean_from_default_config():
    """Exercise the ACTUAL wiring: apply_root_capability_floor (the same call start_cmd makes, over
    the registry's global schemas) turns a DEFAULT root ToolConfig (which inherits global => would
    be co-resident web+bash) into one that resolves with no ingest tool but still keeps
    tool_result_get (declares ingest: none) and the three memory tools."""
    reg = await _production_registry()
    root_cfg = ToolConfig()  # default: inherits global => WOULD be co-resident (web_* + bash/write/edit)
    apply_root_capability_floor(root_cfg, reg.global_schemas(), enabled=True)
    resolved = reg.get_tools_for_agent("root", "", root_cfg)  # must not raise

    assert not (set(resolved) & set(WEB)), f"root still has web ingestion: {set(resolved) & set(WEB)}"
    assert "tool_result_get" in resolved, "root lost tool_result_get (it declares ingest: none)"
    assert {"bash_exec", "memory_search", "memory_get", "remember"} <= set(resolved), \
        "root unexpectedly lost bash or a memory tool (only ingestion should be stripped)"


@pytest.mark.asyncio
async def test_root_without_strip_is_caught_by_chokepoint_fail_closed():
    """Fail-closed safety net: if the root strip is NOT applied (e.g. wiring regressed), the default
    root config is co-resident and the resolution chokepoint REJECTS it — a loud crash, never a
    silent injection->bash hole. (floor is on by module default; we do not touch the global flag.)"""
    reg = await _production_registry()
    root_cfg = ToolConfig()
    apply_root_capability_floor(root_cfg, reg.global_schemas(), enabled=False)  # strip skipped
    with pytest.raises(CoResidenceError):
        reg.get_tools_for_agent("root", "", root_cfg)


# --- Sanity: every built-in subagent toolset stays clean ------------------

@pytest.mark.asyncio
async def test_builtin_subagent_toolsets_clean():
    # Iterate the dispatch table itself so every default builtin — current and future —
    # is checked; v0.5.3 demoted the bash-holding specialists to examples/agents/.
    from localharness.agent.subagent import _BUILTIN_TOOLSETS

    reg = await _builtin_registry()
    for name, toolset in _BUILTIN_TOOLSETS.items():
        schemas = [reg.schema_of(n) for n in toolset]
        assert None not in schemas, f"'{name}' names a tool no builtin declares: {toolset}"
        assert_no_coresidence(schemas, agent_id=name)  # must not raise


# --- Display is pinned to the declarations, and only display ---------------

@pytest.mark.asyncio
async def test_the_display_families_label_exactly_the_tools_that_declare_ingest():
    """The terminal's 'UNTRUSTED, treated as data' group and the phone's `untrusted_ingest` label
    name exactly the builtins that DECLARE ingest: untrusted. They are display sets: the floor never
    reads them. (A PLUGIN tool that ingests is floor-protected but not grouped or labelled — a named
    display residual.)"""
    from types import SimpleNamespace

    from localharness.channels.terminal import _BURST_GROUPS, _UNTRUSTED_NOTE
    from localharness.channels.mobile.channel import MobileChannel
    from localharness.core.bus import EventBus

    schemas = (await _builtin_registry()).global_schemas()
    declared = {s.name for s in schemas if s.ingest == "untrusted"}
    assert declared == set(WEB)
    grouped = {n for group, note, _style in _BURST_GROUPS if note == _UNTRUSTED_NOTE for n in group}
    assert grouped == declared
    channel = MobileChannel(bus=EventBus(), config={})
    labelled = {s.name for s in schemas
                if channel._ask_frame("r1", SimpleNamespace(tool_name=s.name, klass="shell")).untrusted_ingest}
    assert labelled == declared
