"""Roadmap Phase 44, success criterion 4, composed (SAFE-01, SAFE-02, SAFE-03, CORE-04).

Criterion 4, verbatim: "**Declarations fail closed; one declaration, three readers** (§8
"Declarations fail closed"): a bare `ToolSchema` classifies `host: dangerous`, `ingest: untrusted`,
`result_origin: untrusted`, gate "ask"; every builtin declares all four and today's
`UNTRUSTED_INGEST`/`HOST_DANGEROUS` name sets match as a test oracle; a test proves the gate's
per-call family and the capability floor never disagree on any registered tool; MCP wrappers keep
`ingest: untrusted`, `result_origin: untrusted`, gate "ask" from wrapper code; `agent/context.py`
marks memory-tool results untrusted from `result_origin`, with `_MEMORY_TOOLS` deleted."

The three readers are the permission gate (`gate_family`), the capability floor (`ingest`, `host`)
and the context store (`result_origin`). The first test walks, in order, so the section that fails
first names what broke:
  1. a bare schema fails closed through all three readers (and the root floor strips it);
  2. the MCP wrapper declares its posture in its own code, and the gate asks about it;
  3. over EVERY tool a production-shaped root registry holds (the 15 `register_builtin_tools`
     wires, the `agent` tool, one MCP tool): no host-dangerous tool declares or lands in a gate
     tier that runs without a rule set (allow, network); the gate's kind is the declared family;
     the floor's functions classify it by its declaration; stamping `source_plugin` changes no
     classification (CORE-04); the context store stores its evicted body with its declared
     origin — and #140's outcome holds (recall bodies untrusted);
  4. the oracle: the 18 builtins' declarations reproduce today's name sets (imported, one copy);
  5. after the root floor the root holds bash_exec with the three memory tools and no web verb —
     and an MCP tool beside bash is still REJECTED, not stripped;
  6. the name sets are gone from src.
Section 4 comes after 3 on purpose: a reader-versus-declaration check cannot see a declaration that
changed, so the oracle is the backstop for drift, and a changed declaration reddens the reader
section that observes its OUTCOME first. The second test drives the real `start` (zero turns) to
prove the wiring: the root floor runs after the plugin loader registered a tool, and the root's
context store reads origins from the root's own registry.

What this does NOT prove: a third-party plugin tool's gate clamp (SAFE-06) — the contributed-tools
plan proves that; the gate's modes other than `guarded` — test_verdict.py and test_auto_mode.py own
those; that a declaration is TRUE — the floor believes what a tool says.
"""
from __future__ import annotations

import pytest

import localharness.agent.context as context_mod
import localharness.tools.capabilities as capabilities
from localharness.agent.context import ContentStore, _content_handle, _evict_large_tool_results
from localharness.agent.gate import tool_meta_from_schema
from localharness.agent.gate_types import GateSettings, Verdict
from localharness.agent.verdict import UNFAMILIAR_TOOL_KIND, GateContext, _kind, evaluate
from localharness.config.models import ToolConfig
from localharness.plugins.api import Plugin, PluginManifest
from localharness.tools.base import ToolSchema
from localharness.tools.builtin import register_builtin_tools
from localharness.tools.builtin.agent_tool import AgentTool
from localharness.tools.capabilities import (
    CoResidenceError,
    GrantTargetError,
    apply_root_capability_floor,
    assert_grant_target_safe,
    assert_no_coresidence,
    ingests_untrusted,
    is_exec,
    is_host_dangerous,
)
from localharness.tools.mcp import MCPToolWrapper
from localharness.tools.registry import ToolRegistry
from tests.unit.test_capabilities import _Bare
from tests.unit.test_tool_declarations import (
    OLD_HOST_DANGEROUS,
    OLD_MEMORY_TOOLS,
    OLD_UNTRUSTED_INGEST,
    _builtin_schemas,
)

WEB = sorted(OLD_UNTRUSTED_INGEST)
MEMORY = {"memory_search", "memory_get", "remember"}
UNGATED_TIERS = {"allow", "network"}  # tiers the gate runs without a rule-set check on the call


def _raises(check, *args) -> bool:
    try:
        check(*args)
    except (CoResidenceError, GrantTargetError):
        return True
    return False


def _exchanges(bodies: dict[str, str]) -> list[dict]:
    """One assistant call + its bulky result per tool, so eviction can name each producer."""
    out: list[dict] = []
    for i, (tool, body) in enumerate(bodies.items()):
        out.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": tool, "arguments": "{}"}}]})
        out.append({"role": "tool", "tool_call_id": f"c{i}", "content": body})
    return out


async def test_declarations_fail_closed_one_declaration_three_readers(tmp_path):
    guarded = GateContext(workspace=tmp_path, boundary=tmp_path, grants=lambda *a: None, mode="guarded")

    # --- 1. A bare schema fails closed through all three readers -----------------------------
    bare = ToolSchema(name="bare", description="d", parameters={})
    assert (bare.ingest, bare.host, bare.result_origin, bare.gate_family) == (
        "untrusted", "dangerous", "untrusted", None)
    assert ingests_untrusted(bare) and is_host_dangerous(bare), "floor: both sides"
    with pytest.raises(CoResidenceError):
        assert_no_coresidence([bare])  # it co-resides with itself: no agent may hold it
    assert apply_root_capability_floor(ToolConfig(), [bare], enabled=True) == ["bare"]
    bare_meta = tool_meta_from_schema(bare)
    assert _kind("bare", bare_meta) == UNFAMILIAR_TOOL_KIND, "gate: unfamiliar"
    assert evaluate("bare", {}, bare_meta, guarded, GateSettings()).verdict is Verdict.ASK
    holder = ToolRegistry()
    await holder.register(_Bare("bare"), scope="global")
    assert holder.result_origin("bare") == "untrusted", "context: the declared (default) origin"
    store, body = ContentStore(), "B" * 12_000
    _evict_large_tool_results(_exchanges({"bare": body}), store, threshold_chars=8_000,
                              keep_last=0, result_origin=holder.result_origin)
    assert store.origin(_content_handle(body)) == "untrusted"

    # --- 2. MCP keeps its posture, declared in wrapper code ----------------------------------
    mcp_tool = MCPToolWrapper("fetch", "d", {}, session=None, server_name="srv")
    mcp = mcp_tool.info()
    assert {"ingest", "host", "result_origin", "gate_family"} <= mcp.model_fields_set, "declared"
    assert (mcp.ingest, mcp.host, mcp.result_origin, mcp.gate_family) == (
        "untrusted", "safe", "untrusted", None)
    mcp_meta = tool_meta_from_schema(mcp)
    assert _kind(mcp.name, mcp_meta) == "mcp"
    assert evaluate(mcp.name, {}, mcp_meta, guarded, GateSettings()).verdict is Verdict.ASK

    # --- 3. One declaration, three readers, over every registered tool -----------------------
    async def _runner(*_a, **_k):  # pragma: no cover - never invoked
        return ""

    root = ToolRegistry()
    await register_builtin_tools(root, eviction_store=ContentStore())
    # the memory verbs reach a real root from the memory plugin; registered directly here
    from localharness.tools.builtin.memory_tools import MemoryGetTool, MemoryRememberTool, MemorySearchTool
    _mem = object()
    for _mt in (MemorySearchTool(_mem), MemoryGetTool(_mem), MemoryRememberTool(_mem)):
        await root.register(_mt, scope="global")
    await root.register(AgentTool(agent_runner=_runner), scope="global")
    await root.register(mcp_tool, scope="mcp")
    registered = {**root._tools["global"], **root._tools["mcp"]}
    assert len(registered) == 17, sorted(registered)
    bash = root.schema_of("bash_exec")
    bodies = {name: f"{name} " * 3_000 for name in registered if name not in context_mod._WEB_TOOLS}
    store = ContentStore()  # web_fetch/web_search results take the URL-restorable web path instead
    _evict_large_tool_results(_exchanges(bodies), store, threshold_chars=8_000, keep_last=0,
                              result_origin=root.result_origin)
    for name, tool in registered.items():
        schema = tool.info()
        kind = _kind(name, tool_meta_from_schema(schema))
        if is_host_dangerous(schema):  # never an ungated tier — neither declared nor classified
            assert schema.gate_family not in UNGATED_TIERS and kind in {"write", "shell", "code"}, name
        assert kind == ("mcp" if schema.group.startswith("mcp/") else schema.gate_family), name
        assert _raises(assert_no_coresidence, [schema, bash]) == (schema.ingest == "untrusted"), name
        assert _raises(assert_grant_target_safe, [schema]) == (schema.host == "dangerous"), name
        denied = apply_root_capability_floor(ToolConfig(), [schema], enabled=True)
        assert denied == ([name] if schema.ingest == "untrusted" else []), name
        assert root.result_origin(name) == schema.result_origin, name
        # CORE-04: which plugin contributed a tool is provenance, never a safety input.
        stamped = schema.model_copy(update={"source_plugin": "some_plugin"})
        assert (_kind(name, tool_meta_from_schema(stamped)), ingests_untrusted(stamped),
                is_host_dangerous(stamped), is_exec(stamped)) == (
            kind, ingests_untrusted(schema), is_host_dangerous(schema), is_exec(schema)), name
        if name in bodies:
            assert store.origin(_content_handle(bodies[name])) == schema.result_origin, name
    recall = {n: store.origin(_content_handle(bodies[n])) for n in ("memory_search", "memory_get", "bash_exec")}
    assert recall == {"memory_search": "untrusted", "memory_get": "untrusted", "bash_exec": "trusted"}, \
        "#140: recall bodies are untrusted-origin, a generic body is trusted"

    # --- 4. The oracle: the builtins' declarations reproduce today's name sets ---------------
    schemas = await _builtin_schemas()
    assert {n for n, s in schemas.items() if ingests_untrusted(s)} == OLD_UNTRUSTED_INGEST
    assert {n for n, s in schemas.items() if is_host_dangerous(s)} == OLD_HOST_DANGEROUS
    assert {n for n, s in schemas.items() if s.result_origin == "untrusted"} == (
        OLD_MEMORY_TOOLS | OLD_UNTRUSTED_INGEST)

    # --- 5. After the root floor: bash beside the memory tools, no web verb ------------------
    cfg = ToolConfig()
    assert apply_root_capability_floor(cfg, root.global_schemas(), enabled=True) == WEB
    with pytest.raises(CoResidenceError):
        root.get_tools_for_agent("root", "", cfg)  # MCP beside bash: rejected, never stripped
    await root.unregister(mcp.name, scope="mcp")
    resolved = root.get_tools_for_agent("root", "", cfg)  # must not raise
    assert {"bash_exec", *MEMORY} <= set(resolved) and not set(resolved) & set(WEB)

    # --- 6. The name sets are gone from src --------------------------------------------------
    assert not any(hasattr(capabilities, n) for n in ("UNTRUSTED_INGEST", "HOST_DANGEROUS", "EXEC_TOOLS"))
    assert not hasattr(context_mod, "_MEMORY_TOOLS")


async def test_start_reads_declarations_after_every_global_tool_is_registered(
    tmp_path, monkeypatch, fake_home
):
    """The real `start`, zero turns: a tool that declares nothing, registered by the step-5 plugin
    lifecycle, is denied to the root — possible only if the floor runs AFTER registration — the root
    resolves its toolset exactly as the loop does without raising, and the root's context store
    reads origins from the root's own registry."""
    from tests.unit.test_workspace_state_landing import (
        AGENT,
        _drive,
        _global_only_start,
        _install_recorders,
    )

    _global_only_start(tmp_path, monkeypatch, fake_home)

    class _Probe(Plugin):
        """a bundled plugin whose one tool declares nothing"""

        manifest = PluginManifest(name="probe", version="1", kind="tools")

        async def tools(self, ctx):
            return [_Bare("plugin_probe")]

    # Step 5 is the plugin lifecycle (44-14); a bundled plugin reaches it through BUILTIN_PLUGINS.
    # The memory verbs are the memory plugin's, so it stays on the list beside the probe.
    from localharness.memory.plugin import MemoryPlugin
    monkeypatch.setattr("localharness.plugins.builtin.BUILTIN_PLUGINS", (_Probe, MemoryPlugin))
    rec = _install_recorders(monkeypatch)
    await _drive()

    root = next(kw for kw in rec["loop"] if kw["config"].name == AGENT)
    reg, cfg, ctx = root["tool_registry"], root["config"], root["context_manager"]
    assert reg.schema_of("plugin_probe") is not None, "premise: the plugin lifecycle registered it"
    assert {"plugin_probe", *WEB} <= set(cfg.tools.deny)
    resolved = reg.get_tools_for_agent(cfg.name, cfg.division or "", cfg.tools)  # the loop's call
    assert not set(resolved) & {"plugin_probe", *WEB}
    assert {"bash_exec", *MEMORY} <= set(resolved)
    assert ctx._result_origin == reg.result_origin, "the store must read THIS registry's declarations"
