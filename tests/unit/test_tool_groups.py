"""`ToolSchema.group` — the exposure taxonomy seed (PRD §6).

One dotted name per tool saying what KIND of thing it does. The permission gate keys its ask
classes on the group rather than on tool names (PRD §3.1), and v0.14's hierarchical tools make a
GROUP — not a tool — the unit an agent is granted. Both readers break the same way if a tool
ships unclassified: it silently lands in "other", which no rule mentions.

So the load-bearing assertion here is not "read is fs.read". It is that NOTHING registered is
left at the default.
"""
from __future__ import annotations

import pytest

from localharness.tools.registry import ToolRegistry

# The expected map, as PRD §6 / the phase contract fixes it. Names, not instances, so a tool that
# is renamed fails here rather than quietly losing its group.
EXPECTED_GROUPS = {
    "read": "fs.read",
    "glob": "fs.read",
    "grep": "fs.read",
    "load_document": "fs.read",
    "chunk": "fs.read",
    "tool_result_get": "fs.read",
    "write": "fs.write",
    "edit": "fs.write",
    "bash_exec": "shell",
    "python_exec": "code",
    "cruncher_exec": "code",
    "agent": "delegate",
    "web_search": "web",
    "web_fetch": "web",
    "web_page_query": "web",
    "memory_search": "memory",
    "memory_get": "memory",
    "remember": "memory",
}


class _Store:
    """Stand-in for the memory / eviction stores: `register_builtin_tools` only needs them to be
    non-None to wire the memory verbs and tool_result_get, which is exactly the registration this
    test has to cover — those two families are conditional and would otherwise never be checked."""


async def _registered_builtins() -> dict[str, object]:
    from localharness.tools.builtin import register_builtin_tools

    registry = ToolRegistry()
    await register_builtin_tools(registry, memory_store=_Store(), eviction_store=_Store())
    return {name: tool.info() for name, tool in registry._tools["global"].items()}


async def test_every_registered_builtin_names_a_group():
    """The one that matters: an unclassified tool is invisible to every rule keyed on groups."""
    schemas = await _registered_builtins()
    assert schemas, "no builtins registered — this test would pass vacuously"

    unclassified = sorted(name for name, schema in schemas.items() if schema.group == "other")
    assert not unclassified, f"registered builtins with no group: {unclassified}"


async def test_registered_builtins_match_the_ruled_taxonomy():
    schemas = await _registered_builtins()
    for name, schema in schemas.items():
        assert name in EXPECTED_GROUPS, f"{name} is registered but not in the ruled taxonomy"
        assert schema.group == EXPECTED_GROUPS[name], f"{name}: {schema.group}"


@pytest.mark.parametrize("name", ["python_exec", "cruncher_exec", "agent"])
def test_the_conditionally_registered_tools_name_a_group(name):
    """python_exec, cruncher_exec and agent are wired by the subagent/start paths rather than by
    `register_builtin_tools`, so the sweep above cannot see them. They carry the same obligation."""
    from localharness.tools.builtin.agent_tool import AgentTool
    from localharness.tools.builtin.cruncher_exec import CruncherExecTool
    from localharness.tools.builtin.python_tool import PythonExecTool

    async def _runner(*args, **kwargs):  # pragma: no cover - never invoked
        return ""

    tools = {
        "python_exec": PythonExecTool(),
        "cruncher_exec": CruncherExecTool(seed={}),
        "agent": AgentTool(agent_runner=_runner),
    }
    schema = tools[name].info()
    assert schema.name == name
    assert schema.group == EXPECTED_GROUPS[name]


def test_an_mcp_tool_is_grouped_by_its_server():
    """PRD §6: one group per MCP server. What a discovered tool DOES is unknowable from here, so
    the server it came from is the honest unit of both the ask and the exposure grant."""
    from localharness.tools.mcp import MCPToolWrapper

    schema = MCPToolWrapper(
        name="search", description="d", input_schema={"type": "object"},
        session=None, server_name="exa",
    ).info()

    assert schema.group == "mcp/exa"
    assert schema.name == "exa__search", "the prefixed name is unchanged — only the group is new"


def test_group_defaults_to_other_for_a_tool_that_says_nothing():
    """The default is deliberately a value that reads as unclassified, not a plausible-looking
    group: a third-party tool silently landing in `fs.read` would be a permission hole."""
    from localharness.tools.base import ToolSchema

    assert ToolSchema(name="x", description="d", parameters={}).group == "other"
