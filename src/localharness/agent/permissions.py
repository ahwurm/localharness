"""PermissionEvaluator: deny pattern matching for tool calls."""
from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass

from localharness.core.types import ToolCall


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
                return PermissionResult(denied=True, reason=f"Matches deny pattern: {pattern}")
            # Check arg_glob against all string values in arguments
            # Also try with "./" prefix so relative paths match patterns like "*/agents/*.yaml"
            for v in _iter_string_values(tool_call.arguments):
                if fnmatch.fnmatch(v, arg_glob) or fnmatch.fnmatch("./" + v, arg_glob):
                    return PermissionResult(denied=True, reason=f"Matches deny pattern: {pattern}")
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
