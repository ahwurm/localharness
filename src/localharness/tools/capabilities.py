"""Capability floor: no single agent may co-resident untrusted-ingest + host-dangerous tools.

Closes the live prompt-injection->bash hole. An agent that ingests attacker-controllable bytes
(web) must not also hold bash/write/edit/exec — otherwise injected page text becomes a host action.
Split into delegated roles: an ingestion agent (no host-dangerous) hands results to a host-acting
agent. Enforced at both toolset-resolution chokepoints (registry.get_tools_for_agent + from_allowed),
gated by a default-on flag (enforce_capability_floor; module-level mirror set from config at startup).

COVERAGE (stated honestly, not overclaimed): untrusted-ingest = the built-in web verbs below PLUS
any mcp:/plugin: tool (external content is attacker-controllable). MCP is detected on BOTH paths
(it lives in the registry's mcp bucket). PLUGIN tools are detected on the from_allowed/dispatch path
(prefix visible) but NOT when resolved via inherited 'global' scope (registered bare) — a NAMED
RESIDUAL; closing it fully needs a per-tool `ingest` config tag. The floor does not claim to cover
an arbitrary tool that ingests attacker text without an mcp:/plugin: source or a web-verb name.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

UNTRUSTED_INGEST = frozenset({"web_search", "web_fetch", "web_page_query"})
HOST_DANGEROUS = frozenset({"bash_exec", "write", "edit", "python_exec"})
# NOTE: memory tools are intentionally NOT untrusted-ingest — verified: tool output goes to
# history.jsonl, memory_get/search read only the facts table, nothing bridges them.

# Module-level mirror of config's enforce_capability_floor (default-on). Synced at startup from
# HarnessConfig.org by set_floor_enabled() — registry chokepoints have no config handle, so they
# read this. The spec sanctions a module-level default when threading the flag is invasive.
_FLOOR_ENABLED = True


def set_floor_enabled(enabled: bool) -> None:
    """Sync the module-level floor flag from config (called once at harness startup)."""
    global _FLOOR_ENABLED
    _FLOOR_ENABLED = bool(enabled)
    if not enabled:
        warnings.warn(
            "enforce_capability_floor=False — the capability floor is DISABLED. An agent may now "
            "co-resident untrusted-ingest (web) with host-dangerous (bash/write/edit/exec) tools, "
            "reopening the prompt-injection->host hole. Migration escape hatch only.",
            stacklevel=2,
        )


def floor_enabled() -> bool:
    return _FLOOR_ENABLED


class CoResidenceError(ValueError):
    pass


def assert_no_coresidence(tool_names: Iterable[str], *, agent_id: str = "") -> None:
    # A tool counts as untrusted-ingest if it is a built-in web verb OR any mcp:/plugin: tool
    # (external/3rd-party content is attacker-controllable). Callers pass mcp tools with an "mcp:"
    # marker and plugin tools with a "plugin:" marker where the source is known.
    # RESIDUAL (named, not hidden): a PLUGIN tool resolved via inherited 'global' scope is registered
    # under a bare name and is NOT prefix-detectable on the get_tools_for_agent path — full coverage
    # needs a per-tool `ingest` config tag (see module docstring). MCP is covered on both paths.
    names = set(tool_names)
    ingest = {n for n in names if n.startswith("mcp:") or n.startswith("plugin:") or n in UNTRUSTED_INGEST}
    danger = {n for n in names if n in HOST_DANGEROUS}
    if ingest and danger:
        who = f" for agent '{agent_id}'" if agent_id else ""
        raise CoResidenceError(
            f"Toolset{who} combines untrusted-ingest {sorted(ingest)} with host-dangerous "
            f"{sorted(danger)}. An agent that ingests attacker-controllable bytes must not also "
            f"hold bash/write/edit/exec (prompt-injection→host hole). Split into delegated roles: "
            f"an ingestion agent (no host-dangerous) that hands results to a host-acting agent."
        )


def apply_root_capability_floor(tool_config: Any, *, enabled: bool | None = None) -> None:
    """Strip untrusted-ingest (web_*) from a host-acting agent's toolset by denying it.

    Called for the ROOT agent at startup (cli/start_cmd.py) so it cannot co-reside web ingestion
    with bash/write/edit (the prompt-injection->host hole) — root delegates ingestion to the
    web-researcher subagent. Keeps tool_result_get (NOT untrusted-ingest). No-op when the floor is
    disabled. Extracted as a function so the wiring is unit-testable, not an untested inline block.
    """
    if enabled is None:
        enabled = floor_enabled()
    if not enabled:
        return
    for t in sorted(UNTRUSTED_INGEST):
        if t not in tool_config.deny:
            tool_config.deny.append(t)
