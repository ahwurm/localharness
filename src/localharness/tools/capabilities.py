"""Capability floor: no single agent may co-resident untrusted-ingest + host-dangerous tools.

Closes the live prompt-injection->bash hole. An agent that ingests attacker-controllable bytes
(web) must not also hold bash/write/edit/exec — otherwise injected page text becomes a host action.
Split into delegated roles: an ingestion agent (no host-dangerous) hands results to a host-acting
agent. Enforced at both toolset-resolution chokepoints (registry.get_tools_for_agent + from_allowed),
gated by a default-on flag (enforce_capability_floor; module-level mirror set from config at startup).

COVERAGE: the floor reads each tool's DECLARATION (tools/base.py ToolSchema) — `ingest` for the
untrusted-ingest side, `host` for the host-dangerous side, `gate_family` for the exec surfaces —
never its name, never which plugin contributed it (CORE-04). Builtins, plugin tools and MCP tools
are judged by the same three reads. The defaults fail closed: a tool that declares nothing both
ingests and reaches the host, so it co-resides with itself and no agent may hold it while the floor
is on (the root floor strips it). MCP tools declare ingest: untrusted in the wrapper's own code
(tools/mcp.py), so a server cannot describe its way out. The residual this module used to name — a
plugin tool inherited through 'global' scope under a bare name, invisible to a prefix check — is
CLOSED: whatever scope or name a tool arrives under, the floor reads what it declares. What the
floor cannot know is whether a declaration is TRUE: a tool that ingests but declares ingest: none is
believed (enabling a third-party plugin is the operator's trust grant, SAFE-06). The memory tools
declare ingest: none — their output is recall of the facts table, not content fetched from outside.
"""
from __future__ import annotations

import re
import warnings
from collections.abc import Iterable
from typing import Any

from localharness.tools.base import ToolSchema


def ingests_untrusted(schema: ToolSchema) -> bool:
    """The untrusted-ingest side: the tool brings attacker-controllable content into the context."""
    return schema.ingest == "untrusted"


def is_host_dangerous(schema: ToolSchema) -> bool:
    """The host-dangerous side: the tool can change the machine it runs on."""
    return schema.host == "dangerous"


def is_exec(schema: ToolSchema) -> bool:
    """An exec surface an agent could fetch remote content through: host-dangerous AND a shell or
    code family — or no declared family, which fails closed (bash_exec, python_exec; not
    cruncher_exec, which declares host: safe; not write/edit, which declare the write family)."""
    return is_host_dangerous(schema) and schema.gate_family in ("shell", "code", None)


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


def assert_no_coresidence(schemas: Iterable[ToolSchema], *, agent_id: str = "") -> None:
    """Refuse a toolset holding a tool that declares ingest: untrusted beside one that declares
    host: dangerous. A tool on both sides — every tool that declares nothing — is its own violation."""
    schemas = list(schemas)
    ingest = sorted({s.name for s in schemas if ingests_untrusted(s)})
    danger = sorted({s.name for s in schemas if is_host_dangerous(s)})
    if ingest and danger:
        who = f" for agent '{agent_id}'" if agent_id else ""
        both = sorted(set(ingest) & set(danger))
        hint = (f" {both} sit on both sides: a tool that declares nothing is assumed to do both — "
                f"declare `ingest`/`host` on its ToolSchema." if both else "")
        raise CoResidenceError(
            f"Toolset{who} combines untrusted-ingest {ingest} with host-dangerous "
            f"{danger}. An agent that ingests attacker-controllable bytes must not also "
            f"hold bash/write/edit/exec (prompt-injection→host hole). Split into delegated roles: "
            f"an ingestion agent (no host-dangerous) that hands results to a host-acting agent.{hint}"
        )


class GrantTargetError(ValueError):
    pass


def assert_grant_target_safe(schemas: Iterable[ToolSchema], *, agent_id: str = "") -> None:
    """Refuse a cross-agent content-handle grant to a HOST-DANGEROUS target (fail closed).

    A granted handle resolves via tool_result_get / chunk, which are NOT untrusted-ingest, so
    assert_no_coresidence would NOT catch a bash/write/edit/exec holder handed an *untrusted*
    granted handle — that would put attacker-controllable bytes one tool_result_get away from a
    host action. So grants may target ONLY no-host-dangerous agents (the cruncher/summarizer
    pattern). This makes "grants flow down into no-danger agents only" a CHECKED invariant, not a
    convention. Judged by each tool's declared `host`. The caller gates on floor_enabled()
    (mirrors assert_no_coresidence)."""
    danger = sorted({s.name for s in schemas if is_host_dangerous(s)})
    if danger:
        who = f" '{agent_id}'" if agent_id else ""
        raise GrantTargetError(
            f"refusing to grant content handle(s) to subagent{who}: its toolset holds host-dangerous "
            f"{danger}. A granted handle is readable (tool_result_get/chunk — not untrusted-"
            f"ingest), so granting to a bash/write/edit/exec holder would put attacker-controllable "
            f"bytes one call from a host action. Grants may target only no-host-dangerous agents (the "
            f"cruncher). Split the work: a no-danger processor reads the handle and returns a summary."
        )


class IngestViaExecError(ValueError):
    pass


# The exec tools an agent can smuggle ingestion through are the ones is_exec() selects:
# host-dangerous with a shell/code family. write/edit touch the host but fetch nothing, so they
# declare the write family and are not gated here.

# Commands whose PURPOSE is pulling REMOTE CONTENT — the ingest capability an exec tool hands an
# agent the floor just denied the web verbs to. Package/VCS/registry ops (pip, uv, git, apt, npm)
# are deliberately ABSENT: they are not content ingestion, and denying them breaks real work.
# Matched on the command string, so this is a REDIRECT (defense-in-depth), NOT a sandbox — a
# determined agent can still obfuscate (base64, a helper script, an env-var'd URL). Stated plainly
# rather than overclaimed: the airtight version is network isolation for the exec tool, which the
# owner ruled out for now (2026-09-17: enforce at the harness level, not a netns).
_INGEST_VIA_EXEC: tuple[tuple[Any, str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), label)
    for pattern, label in (
        # Binary names take a SUFFIX guard too (separator or end), so `wget-log`/`curl-config`
        # as filenames don't trip. `links` (the terminal browser) is deliberately dropped: it is
        # an everyday English word and the false-positive cost dwarfs its redirect value — a
        # named residual, consistent with the redirect-not-sandbox stance above.
        (r"(?:^|[\s;&|(`])(?:curl|wget|aria2c|httpie|lynx|w3m|elinks)(?=$|[\s;&|()'\"`<>])",
         "http client"),
        (r"(?:^|[\s;&|(`])(?:nc|ncat|netcat|telnet|socat)(?=$|[\s;&|()'\"`<>])",
         "raw socket tool"),
        (r"/dev/tcp/", "bash tcp redirect"),
        (r"openssl\s+s_client", "openssl socket"),
        (r"(?:^|[\s;&|(`])(?:ddgs|duckduckgo_search|googlesearch)(?=$|[\s;&|()'\"`<>])",
         "search cli"),
        (r"\b(?:import|from)\s+(?:requests|httpx|aiohttp|urllib|urllib2|urllib3|ddgs"
         r"|socket|selenium|playwright|mechanize)\b", "python network import"),
        (r"\b(?:requests|httpx)\.(?:get|post|head|put|patch|delete|request|Client|AsyncClient)\s*\(",
         "python http call"),
        (r"\burlopen\s*\(|\burllib\.request\b", "python urlopen"),
        (r"\baiohttp\.ClientSession\s*\(", "python aiohttp"),
        (r"\bwebbrowser\.open\s*\(", "webbrowser"),
    )
)


def assert_no_ingest_via_exec(
    schema: ToolSchema,
    arguments: Any,
    *,
    agent_id: str = "",
    has_ingest: bool = False,
) -> None:
    """An agent DENIED the web verbs must not fetch remote content through an exec tool instead.

    apply_root_capability_floor strips web_* from a host-acting agent on the theory that it
    DELEGATES ingestion. But bash is itself an ingest tool — `curl`, `python -c 'import requests'`
    and friends reach the same attacker-controllable bytes — so the deny list alone was advisory
    and the floor's invariant was, for an exec holder, a fiction.

    Observed live 2026-09-17: the root agent, whose own role said "never try to reach the web
    yourself with bash/curl", ran a DuckDuckGo search through bash_exec and fed the result into a
    subagent brief — after its delegation came back empty.

    Enforced at the dispatch chokepoint, so EVERY agent inherits it from its DESIGNATION with no
    per-agent config: an agent holding an ingest verb is untouched; one without it is redirected
    to delegation. `has_ingest` is the designation, resolved by the caller from the agent's own
    toolset. Applies to a tool is_exec() selects, read off its declaration. No-op when the floor
    is disabled.
    """
    if has_ingest or not floor_enabled() or not is_exec(schema):
        return
    if isinstance(arguments, dict):
        text = " ".join(str(v) for v in arguments.values() if isinstance(v, str))
    else:
        text = str(arguments or "")
    for rx, label in _INGEST_VIA_EXEC:
        hit = rx.search(text)
        if hit is None:
            continue
        who = f" '{agent_id}'" if agent_id else ""
        raise IngestViaExecError(
            f"BLOCKED: agent{who} holds no web-ingest tool, so it may not reach the web through "
            f"`{schema.name}` instead (matched {label}: {hit.group(0).strip()!r}). Reading remote "
            f"content here would put attacker-controllable bytes one call from a host action — "
            f"the capability floor denies you the web verbs for exactly that reason, and routing "
            f"around it defeats the split.\n"
            f"DELEGATE instead: agent(agent_id='web-researcher', task='<what you need, including "
            f"the exact queries or URLs>').\n"
            f"If you believe the web-researcher itself is broken, SAY SO TO THE USER and STOP — "
            f"report the failure, do not work around it. A silent workaround hides a real bug."
        )


def apply_root_capability_floor(
    tool_config: Any, schemas: Iterable[ToolSchema], *, enabled: bool | None = None
) -> list[str]:
    """Deny a host-acting agent every tool that declares — or, undeclared, defaults to — ingest:
    untrusted, and return the sorted names denied.

    Called for the ROOT agent by cli/start_cmd.py over the registry's GLOBAL schemas, after every
    global tool is registered (a declaration is only readable off a registered tool), so the root
    cannot co-reside ingestion with bash/write/edit (the prompt-injection->host hole) — it delegates
    ingestion to the web-researcher subagent. With the builtins that is the three web verbs; a
    plugin tool that ingests, or declares nothing, is denied the same way. tool_result_get and the
    memory tools declare ingest: none and stay. MCP tools are not global scope and are not stripped:
    the chokepoint rejects one beside bash. No-op (returns []) when the floor is disabled.
    Extracted as a function so the wiring is unit-testable, not an untested inline block.
    """
    if enabled is None:
        enabled = floor_enabled()
    if not enabled:
        return []
    denied = sorted({s.name for s in schemas if ingests_untrusted(s)})
    tool_config.deny.extend([n for n in denied if n not in tool_config.deny])
    return denied
