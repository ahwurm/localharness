"""P-A capability floor: no agent may co-resident untrusted-ingest + host-dangerous tools.

Covers the pure predicate (assert_no_coresidence), the memory-clean invariant (memory is NOT
ingest), both resolution chokepoints (from_allowed + get_tools_for_agent), the root-no-web
topology, and a sanity sweep that every built-in subagent toolset stays clean.
"""
from __future__ import annotations

import pytest

from localharness.config.models import ToolConfig
from localharness.tools.builtin import register_builtin_tools
from localharness.tools.capabilities import (
    UNTRUSTED_INGEST,
    CoResidenceError,
    apply_root_capability_floor,
    assert_no_coresidence,
)
from localharness.tools.registry import ToolRegistry


# --- Pure predicate -------------------------------------------------------

@pytest.mark.parametrize(
    "names",
    [
        {"web_fetch", "bash_exec"},
        {"web_search", "write"},
        {"web_page_query", "python_exec"},
        {"mcp:fetch", "bash_exec"},                          # MCP ingestion + host-dangerous
        {"plugin:research_tools.exa_search", "write"},       # plugin ingestion + host-dangerous
    ],
)
def test_coresidence_raises(names):
    with pytest.raises(CoResidenceError):
        assert_no_coresidence(names)


@pytest.mark.parametrize(
    "names",
    [
        {"web_fetch", "web_page_query"},               # web-only
        {"bash_exec", "write", "edit"},                # danger-only
        {"memory_search", "memory_get", "bash_exec"},  # memory is NOT ingest (the clean invariant)
        {"mcp:fetch", "web_page_query"},               # ingest-only (mcp + web), no host-dangerous
    ],
)
def test_no_coresidence_passes(names):
    assert_no_coresidence(names)  # must not raise


# --- Chokepoint: from_allowed --------------------------------------------

@pytest.mark.asyncio
async def test_from_allowed_rejects_coresident():
    full = ToolRegistry()
    await register_builtin_tools(full)
    with pytest.raises(CoResidenceError):
        ToolRegistry.from_allowed(["web_fetch", "bash_exec"], base_registry=full)


@pytest.mark.asyncio
async def test_from_allowed_rejects_mcp_ingest_plus_bash():
    """The MCP/plugin ingestion bypass the floor previously MISSED: an mcp: ingestion tool
    co-resident with a host-dangerous tool must be REJECTED (declared-intent check by source),
    not just the 3 built-in web verbs. mcp:fetch need not resolve to a live MCP server."""
    full = ToolRegistry()
    await register_builtin_tools(full)
    with pytest.raises(CoResidenceError):
        ToolRegistry.from_allowed(["mcp:fetch", "bash_exec"], base_registry=full)
    # plugin ingestion + host-dangerous likewise
    with pytest.raises(CoResidenceError):
        ToolRegistry.from_allowed(["plugin:research_tools.exa_search", "write"], base_registry=full)


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


# --- Root-no-web topology (§3) -------------------------------------------

@pytest.mark.asyncio
async def test_root_strip_resolves_clean_from_default_config():
    """Exercise the ACTUAL wiring: apply_root_capability_floor (the same call start_cmd makes) turns
    a DEFAULT root ToolConfig (which inherits global => would be co-resident web+bash) into one that
    resolves with no ingest tool but still keeps tool_result_get (NOT untrusted-ingest)."""
    reg = ToolRegistry()
    await register_builtin_tools(reg)
    # tool_result_get is registered only when a store exists — register it explicitly here.
    from localharness.agent.context import ContentStore
    from localharness.tools.builtin.tool_result_get_tool import ToolResultGetTool
    await reg.register(ToolResultGetTool(ContentStore()), scope="global")

    root_cfg = ToolConfig()  # default: inherits global => WOULD be co-resident (web_* + bash/write/edit)
    apply_root_capability_floor(root_cfg, enabled=True)  # the exact call cli/start_cmd.py makes
    resolved = reg.get_tools_for_agent("root", "", root_cfg)  # must not raise

    assert not (set(resolved) & UNTRUSTED_INGEST), f"root still has web ingestion: {set(resolved) & UNTRUSTED_INGEST}"
    assert "tool_result_get" in resolved, "root lost tool_result_get (it is NOT untrusted-ingest)"
    assert "bash_exec" in resolved, "root unexpectedly lost bash (only web ingestion should be stripped)"


@pytest.mark.asyncio
async def test_root_without_strip_is_caught_by_chokepoint_fail_closed():
    """Fail-closed safety net: if the root strip is NOT applied (e.g. wiring regressed), the default
    root config is co-resident and the resolution chokepoint REJECTS it — a loud crash, never a
    silent injection->bash hole. (floor is on by module default; we do not touch the global flag.)"""
    reg = ToolRegistry()
    await register_builtin_tools(reg)
    root_cfg = ToolConfig()
    apply_root_capability_floor(root_cfg, enabled=False)  # strip skipped — simulate regressed wiring
    with pytest.raises(CoResidenceError):
        reg.get_tools_for_agent("root", "", root_cfg)


# --- Sanity: every built-in subagent toolset stays clean ------------------

def test_builtin_subagent_toolsets_clean():
    # Iterate the dispatch table itself so every default builtin — current and future —
    # is checked; v0.5.3 demoted the bash-holding specialists to examples/agents/.
    from localharness.agent.subagent import _BUILTIN_TOOLSETS

    for name, toolset in _BUILTIN_TOOLSETS.items():
        assert_no_coresidence(toolset)  # must not raise
