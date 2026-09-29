"""Third-party gate trust (SAFE-06, PRD decision 10).

Turning on a plugin you brought yourself is the operator's trust grant for its FLOOR declarations:
ingest, host and result_origin are honoured as written from then on. What its tools declare to the
permission GATE is honoured only when the gate treats it at least as strictly as an undeclared
tool, so a tool can never declare its way into being allowed. Bundled plugins are reviewed
first-party code and are exempt. The loader applies this when it registers a non-bundled plugin's
tools, by rewriting the declaration; the gate, the floor and the context store still read only the
declaration (CORE-04)."""
from __future__ import annotations

from typing import Any

from localharness.agent.gate import tool_meta_from_schema
from localharness.tools.base import ToolSchema

THIRD_PARTY_CLAMPED_FAMILIES: frozenset[str] = frozenset({"allow", "network", "shell", "write"})
"""The families a non-bundled plugin's tool may not keep, MEASURED rather than argued:
tests/unit/test_third_party_gate_clamp.py runs every GateFamily member through agent/verdict.evaluate
and asserts this set equals exactly the members that are ever more permissive than an undeclared
tool. `allow` and `network` are allowed by default in every mode; `shell` allows a call with no
command and any read-only command; `write` allows an in-project write on a channel that shows a diff.
`code` and `delegate` are treated exactly as an undeclared tool is, keyed by name (and read-only mode
denies them)."""


def third_party_overrides(schema: ToolSchema) -> tuple[dict[str, Any], str | None]:
    """The registration override for a NON-bundled plugin's tool, and the warning to show, or
    ({}, None) when what it declares to the gate is honoured as written.

    It reads the schema exactly as the gate will (tool_meta_from_schema). Two declarations reach
    the gate: `gate_family`, rewritten to undeclared when it is in THIRD_PARTY_CLAMPED_FAMILIES,
    and `group`, whose `mcp/` prefix makes the gate judge the tool as that MCP server's and extend
    it any trust the operator gave the server (`mcp_trusted_servers`). A plugin's tool is not an
    MCP server's, so it loses the prefix: its group becomes `other`, the unclassified default."""
    meta = tool_meta_from_schema(schema)
    overrides: dict[str, Any] = {}
    notes: list[str] = []
    if meta.gate_family in THIRD_PARTY_CLAMPED_FAMILIES:
        overrides["gate_family"] = None
        notes.append(f"tool {schema.name!r} declares gate family {meta.gate_family!r}; a plugin you "
                     "installed may only declare families that ask at least as often as an "
                     "undeclared tool — it is treated as undeclared (asked about once per "
                     "workspace in guarded mode)")
    if meta.is_mcp:
        overrides["group"] = "other"
        notes.append(f"tool {schema.name!r} declares group {schema.group!r}, which the permission "
                     "gate reads as an MCP server's tool and would extend that server's trust; a "
                     "plugin's tool is not one — its group is treated as 'other'")
    return overrides, "; ".join(notes) or None
