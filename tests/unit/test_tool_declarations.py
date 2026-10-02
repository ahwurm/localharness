"""One declaration per tool, and saying nothing fails closed (SAFE-01, SAFE-02).

`ToolSchema` carries four fields the safety model reads instead of a tool's NAME:

* `ingest` — does the tool bring attacker-controllable content into the context?
* `host` — can it change the machine it runs on?
* `result_origin` — is what it returns trusted?
* `gate_family` — which of the permission gate's rule sets judges its calls?

Every default is the worst case, so a tool that declares nothing is treated as untrusted,
host-dangerous and unfamiliar (asked about): a plugin's omission can never widen what it may do.

The builtins and the MCP wrapper declare today's classes EXACTLY. Today's name sets are copied
below and kept ONLY here, as the oracle that no builtin changed class when the name sets became
declarations.
"""
from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from localharness.agent.gate import tool_meta_from_schema
from localharness.agent.gate_types import ToolMeta
from localharness.agent.verdict import UNFAMILIAR_TOOL_KIND, _kind
from localharness.tools.base import GATE_FAMILIES, ToolSchema

# Today's name sets (tools/capabilities.py:23-24, agent/context.py:141 on main @ 75ad038), kept
# ONLY here as the oracle that no builtin changed class — SAFE-02.
OLD_UNTRUSTED_INGEST = frozenset({"web_search", "web_fetch", "web_page_query"})
OLD_HOST_DANGEROUS = frozenset({"bash_exec", "write", "edit", "python_exec"})
OLD_MEMORY_TOOLS = frozenset({"memory_search", "memory_get"})

DECLARATION = ("gate_family", "ingest", "host", "result_origin")

EXPECTED: dict[str, tuple[str, str, str, str]] = {  # tool name -> DECLARATION values
    "glob": ("allow", "none", "safe", "trusted"),
    "grep": ("allow", "none", "safe", "trusted"),
    "read": ("allow", "none", "safe", "trusted"),
    "chunk": ("allow", "none", "safe", "trusted"),
    "load_document": ("allow", "none", "safe", "trusted"),
    "tool_result_get": ("allow", "none", "safe", "trusted"),
    "memory_search": ("allow", "none", "safe", "untrusted"),
    "memory_get": ("allow", "none", "safe", "untrusted"),
    "remember": ("allow", "none", "safe", "trusted"),
    "write": ("write", "none", "dangerous", "trusted"),
    "edit": ("write", "none", "dangerous", "trusted"),
    "bash_exec": ("shell", "none", "dangerous", "trusted"),
    "python_exec": ("code", "none", "dangerous", "trusted"),
    "cruncher_exec": ("code", "none", "safe", "trusted"),
    "agent": ("delegate", "none", "safe", "trusted"),
    "web_search": ("network", "untrusted", "safe", "untrusted"),
    "web_fetch": ("network", "untrusted", "safe", "untrusted"),
    "web_page_query": ("network", "untrusted", "safe", "untrusted"),
}


async def _builtin_schemas() -> dict[str, ToolSchema]:
    """All 18 builtin schemas, offline. `register_builtin_tools` wires 15 — the memory verbs and
    tool_result_get only when handed a store, and their constructors merely keep it — and the
    three the start/subagent paths wire are built directly with the smallest arguments."""
    from localharness.agent.context import ContentStore
    from localharness.tools.builtin import register_builtin_tools
    from localharness.tools.builtin.agent_tool import AgentTool
    from localharness.tools.builtin.cruncher_exec import CruncherExecTool
    from localharness.tools.builtin.python_tool import PythonExecTool
    from localharness.tools.registry import ToolRegistry

    async def _runner(*args, **kwargs):  # pragma: no cover - never invoked
        return ""

    registry = ToolRegistry()
    await register_builtin_tools(registry, eviction_store=ContentStore())
    # the memory verbs reach a real root from the memory plugin; registered directly here
    from localharness.tools.builtin.memory_tools import MemoryGetTool, MemoryRememberTool, MemorySearchTool
    _mem = object()
    for _mt in (MemorySearchTool(_mem), MemoryGetTool(_mem), MemoryRememberTool(_mem)):
        await registry.register(_mt, scope="global")
    tools = [*registry._tools["global"].values(),
             PythonExecTool(), CruncherExecTool(seed={}), AgentTool(agent_runner=_runner)]
    schemas = {tool.info().name: tool.info() for tool in tools}
    assert len(tools) == len(schemas) == 18 and set(schemas) == set(EXPECTED), sorted(schemas)
    return schemas


def _mcp_schema() -> ToolSchema:
    from localharness.tools.mcp import MCPToolWrapper

    return MCPToolWrapper("fetch", "d", {}, session=None, server_name="srv").info()


# --------------------------------------------------------------- the fields fail closed

def test_a_bare_schema_fails_closed_on_every_axis():
    schema = ToolSchema(name="x", description="d", parameters={})

    assert schema.ingest == "untrusted"
    assert schema.host == "dangerous"
    assert schema.result_origin == "untrusted"
    assert schema.gate_family is None, "undeclared family = the gate's tool-unfamiliar ask"
    assert schema.source_plugin is None


def test_an_unrecognised_family_is_unset_and_warned_about_once(caplog):
    """A plugin's typo must fail closed (asked about), never crash the plugin — and info() runs
    every turn, so the warning is once per tool and value, not once per call."""
    with caplog.at_level(logging.WARNING, logger="localharness.tools.base"):
        first = ToolSchema(name="typo_probe", description="d", parameters={}, gate_family="read_only")
        again = ToolSchema(name="typo_probe", description="d", parameters={}, gate_family="read_only")

    assert first.gate_family is None and again.gate_family is None
    warned = [r.getMessage() for r in caplog.records if r.name == "localharness.tools.base"]
    assert len(warned) == 1, warned
    assert "typo_probe" in warned[0] and "read_only" in warned[0]


@pytest.mark.parametrize("value", [["allow"], 3])
def test_a_non_string_family_is_unset_too(value):
    assert ToolSchema(name="odd_probe", description="d", parameters={}, gate_family=value).gate_family is None


@pytest.mark.parametrize("field", ["ingest", "host", "result_origin"])
def test_the_three_literal_axes_fail_loud(field):
    """Only gate_family is coerced. A bad value on these axes is a bug in the tool, not a typo a
    safe default can absorb — each has exactly two values."""
    with pytest.raises(ValidationError):
        ToolSchema(name="x", description="d", parameters={}, **{field: "sorta"})


async def test_the_declarations_never_reach_the_wire():
    """`_tools_to_api_format` sends `model_dump()` as the function object and the chat template
    renders it into the prompt (measured on the served model: +32 tokens per tool for these five
    keys). Safety metadata is for the harness, not the model: a default session's prompt and its
    tool-token budget must not change because tools learned to describe themselves."""
    from localharness.provider.client import _tools_to_api_format

    wire, _ = _tools_to_api_format([*(await _builtin_schemas()).values(), _mcp_schema()])
    leaked = {k for entry in wire for k in entry["function"]} & {*DECLARATION, "source_plugin"}
    assert leaked == set()


# ------------------------------------------------------ every builtin declares, as ruled

async def test_every_builtin_declares_all_four_explicitly_as_ruled():
    """Explicit even where the value equals a default: a declaration that is only a default is
    indistinguishable from forgetting, so `model_fields_set` is graded, not just the value."""
    for name, schema in (await _builtin_schemas()).items():
        missing = set(DECLARATION) - schema.model_fields_set
        assert not missing, f"{name} leaves {sorted(missing)} at the fail-closed default"
        assert tuple(getattr(schema, f) for f in DECLARATION) == EXPECTED[name], name


async def test_no_builtin_changed_class_against_the_old_name_sets():
    schemas = await _builtin_schemas()

    def declaring(field: str, value: str) -> set[str]:
        return {name for name, schema in schemas.items() if getattr(schema, field) == value}

    assert declaring("ingest", "untrusted") == OLD_UNTRUSTED_INGEST
    assert declaring("host", "dangerous") == OLD_HOST_DANGEROUS
    assert declaring("result_origin", "untrusted") == OLD_MEMORY_TOOLS | OLD_UNTRUSTED_INGEST


def test_an_mcp_tool_keeps_todays_posture_in_wrapper_code():
    """External content is attacker-controllable; the tool was never host-dangerous by the floor's
    definition (an unset `host` would make it co-reside with itself); its gate class stays `mcp`."""
    schema = _mcp_schema()

    assert set(DECLARATION) <= schema.model_fields_set
    assert (schema.ingest, schema.host, schema.result_origin) == ("untrusted", "safe", "untrusted")
    assert schema.gate_family is None
    assert schema.group == "mcp/srv"


async def test_the_display_web_set_is_exactly_the_declared_ingest_set():
    """WEB_INGEST_TOOLS is display-only (the terminal's 'treated as data' note, the phone's label);
    it must name exactly the builtins that DECLARE ingest untrusted."""
    from localharness.tools.builtin.web_tool import WEB_INGEST_TOOLS

    schemas = await _builtin_schemas()
    assert WEB_INGEST_TOOLS == {name for name, s in schemas.items() if s.ingest == "untrusted"}


# ------------------------------------------- the gate reads the declaration (reader one of three)

@pytest.mark.parametrize("family", sorted(GATE_FAMILIES))
def test_every_declared_family_is_the_branch_the_gate_takes(family):
    assert _kind("some_plugin_tool", ToolMeta(gate_family=family)) == family


@pytest.mark.parametrize("group", ["other", "fs.read", "memory", "fs.write", "shell", "code", "delegate", "web"])
def test_an_undeclared_tool_is_unfamiliar_whatever_its_group(group):
    """`group` is exposure taxonomy, not a gate input: a tool the gate does not know by name and
    that declares no family asks — even when its group names a read tier."""
    assert _kind("some_plugin_tool", ToolMeta(group=group)) == UNFAMILIAR_TOOL_KIND


def test_mcp_and_the_name_tables_still_come_first():
    assert _kind("srv__x", ToolMeta(is_mcp=True, gate_family="allow")) == "mcp"
    assert _kind("srv__x", ToolMeta(group="mcp/srv", gate_family="allow")) == "mcp"
    assert _kind("bash_exec", ToolMeta(gate_family="allow")) == "shell"


def test_kind_by_group_is_gone():
    import localharness.agent.verdict as verdict

    assert not hasattr(verdict, "KIND_BY_GROUP")


def test_the_declarable_families_are_exactly_the_gates_branches(tmp_path):
    """GateFamily and `evaluate`'s dispatch cannot drift apart: the literals `evaluate` tests `kind`
    against, minus `mcp` (not declarable), plus `allow` — the fall-through, which really ALLOWs."""
    import ast
    import inspect

    from localharness.agent import verdict
    from localharness.agent.gate_types import GateSettings, Verdict

    tested = {
        node.comparators[0].value
        for node in ast.walk(ast.parse(inspect.getsource(verdict.evaluate)))
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "kind"
        and isinstance(node.comparators[0], ast.Constant)
    }
    assert tested - {"mcp"} | {"allow"} == GATE_FAMILIES

    ctx = verdict.GateContext(workspace=tmp_path, boundary=tmp_path, grants=lambda *a: None, mode="guarded")
    result = verdict.evaluate("some_plugin_tool", {}, ToolMeta(gate_family="allow"), ctx, GateSettings())
    assert result.verdict is Verdict.ALLOW


def test_a_duck_typed_schema_cannot_smuggle_an_unknown_family():
    """`tool.info()` is whatever a tool returns; one that bypasses ToolSchema's validator still
    lands on the fail-closed side at the gate's own schema read."""
    from types import SimpleNamespace

    fake = SimpleNamespace(name="x", group="other", destructive=False, gate_family="read_only")
    assert tool_meta_from_schema(fake).gate_family is None


async def test_the_gate_and_the_declaration_agree_on_every_builtin():
    """Through the gate's real schema read (`tool_meta_from_schema`, what the loop calls on
    `tool.info()`): every builtin lands in exactly the branch it declares."""
    for name, schema in (await _builtin_schemas()).items():
        assert _kind(name, tool_meta_from_schema(schema)) == schema.gate_family, name
    mcp = _mcp_schema()
    assert _kind(mcp.name, tool_meta_from_schema(mcp)) == "mcp"


async def test_each_name_table_agrees_with_that_builtins_declaration():
    """The name tables bind the parameter each rule reads; they must never contradict a
    declaration, or a builtin would be judged by a rule set other than the one it declares."""
    from localharness.agent import verdict as v

    tables = {"write": v.WRITE_TOOL_PATH_PARAMS, "shell": v.SHELL_COMMAND_PARAMS,
              "code": v.CODE_EXEC_TOOLS, "delegate": v.DELEGATE_TOOLS, "network": v.NETWORK_URL_PARAMS}
    schemas = await _builtin_schemas()
    for branch, names in tables.items():
        for name in names:
            assert schemas[name].gate_family == branch, name


async def test_the_askrate_replay_reads_the_same_families():
    """A trace records a NAME, so the replay rebuilds the gate's view from a map — which must be
    the live declarations, or the ask-rate report measures a gate that does not exist."""
    from localharness.bench.askrate import BUILTIN_TOOL_FAMILIES, tool_meta_for

    schemas = await _builtin_schemas()
    assert dict(BUILTIN_TOOL_FAMILIES) == {name: s.gate_family for name, s in schemas.items()}
    for name, schema in schemas.items():
        assert _kind(name, tool_meta_for(name)) == _kind(name, tool_meta_from_schema(schema)), name
