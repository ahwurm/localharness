"""Which addresses a URL the model hands to a web tool may reach.

This module holds the session's private-address allowlist: the machine's
`org.web_fetch_allow_private` (machine-level only, so a project value never reaches it), set once per
session by `localharness start`. Module-level because the web tools are rebuilt per agent
(tools/builtin bind_agent_store_tools) and every subagent's copy must read the same list. Empty
before any start — tests and the bench fetch public addresses only."""
from __future__ import annotations

from collections.abc import Iterable

_ALLOW_PRIVATE: tuple[str, ...] = ()


def set_private_allowlist(entries: Iterable[str]) -> None:
    """Set this session's allowlist (`localharness start`, after the config loads)."""
    global _ALLOW_PRIVATE
    _ALLOW_PRIVATE = tuple(entries)


def private_allowlist() -> tuple[str, ...]:
    """The allowlist the running session set; () when none did."""
    return _ALLOW_PRIVATE
