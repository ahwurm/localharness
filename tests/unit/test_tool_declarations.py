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

from localharness.tools.base import ToolSchema

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
    await register_builtin_tools(registry, memory_store=object(), eviction_store=ContentStore())
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
