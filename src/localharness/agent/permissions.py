"""PermissionEvaluator: deny pattern matching for tool calls."""
from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass

from localharness.core.types import ToolCall


# A denial that has a legitimate ALTERNATIVE should say so. A bare "matches deny pattern"
# tells the model it is blocked but not what to do instead, and a model that does not know
# the supported route retries the blocked one (observed as the retry-the-refused-command
# loop). These tokens appear inside the shipped memory-store patterns themselves, so a new
# pattern naming the same artifacts inherits the guidance with no second list to maintain.
MEMORY_STORE_PATTERN_TOKENS: tuple[str, ...] = (
    "memory.db", "facts_archive", "memory-archive", "localharness memory",
)
MEMORY_STORE_DENY_GUIDANCE = (
    "the memory store is not reachable from a shell. Read and write memory with your memory "
    "tools instead — memory_search to find facts, memory_get to read one, remember to store "
    "one. They are the only supported way in, they read the LIVE store, and archived or "
    "left-over store files are deliberately invisible to them: stale facts must not come back "
    "wearing the authority of current ones."
)


def deny_reason(pattern: str) -> str:
    """The agent-facing sentence for a matched deny pattern — the redirect, where one exists."""
    base = f"Matches deny pattern: {pattern}"
    if any(token in pattern for token in MEMORY_STORE_PATTERN_TOKENS):
        return f"{base} — {MEMORY_STORE_DENY_GUIDANCE}"
    return base


@dataclass
class PermissionResult:
    denied: bool
    reason: str = ""


class PermissionEvaluator:
    """Evaluates tool calls against deny patterns from PermissionConfig."""

    def evaluate(self, tool_call: ToolCall, permissions: object) -> PermissionResult:
        """Return PermissionResult(denied=True) if any deny pattern matches.

        Pattern format: 'tool_name' or 'tool_name(arg_glob)'
        Matching: tool_name must match exactly; if arg_glob present, any string
        value in tool_call.arguments must match via fnmatch.
        """
        deny_patterns: list[str] = getattr(permissions, "deny_patterns", [])
        for pattern in deny_patterns:
            match = re.match(r"^([a-z_][a-z0-9_]*)(?:\((.+)\))?$", pattern)
            if not match:
                continue
            tool_name_pattern, arg_glob = match.group(1), match.group(2)
            if tool_call.name != tool_name_pattern:
                continue
            if arg_glob is None:
                # Bare tool name pattern — any call to this tool is denied
                return PermissionResult(denied=True, reason=deny_reason(pattern))
            # Check arg_glob against all string values in arguments
            # Also try with "./" prefix so relative paths match patterns like "*/agents/*.yaml"
            for v in _iter_string_values(tool_call.arguments):
                if fnmatch.fnmatch(v, arg_glob) or fnmatch.fnmatch("./" + v, arg_glob):
                    return PermissionResult(denied=True, reason=deny_reason(pattern))
        return PermissionResult(denied=False)


def _iter_string_values(obj: object) -> list[str]:
    """Recursively collect all string values from a dict or list."""
    result = []
    if isinstance(obj, str):
        result.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            result.extend(_iter_string_values(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            result.extend(_iter_string_values(item))
    return result
