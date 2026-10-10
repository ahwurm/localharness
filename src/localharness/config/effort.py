"""The reasoning-effort ladder, spelled once (spec 16 §7.2).

`resolve_effort` is the only place the order session > agent file > serving endpoint > org default
> "high" is written down; every caller (the loop's constructor, `/effort`, the subagent runner)
asks it. It is pure: no I/O, no logging, no import beyond `typing` and `config.models`, and it
reads the harness with `getattr` so a plain object answers as well as a `HarnessConfig`.
"""
from __future__ import annotations

from typing import Any

from localharness.config.models import LEVELS, EffortLevel, Rung

__all__ = [
    "EFFORT_UNENFORCED_LINE", "EffortLevel", "LEVELS", "Rung", "endpoint_effort", "resolve_effort",
]

EFFORT_UNENFORCED_LINE = "effort unenforced on this server: {key} rejected"
"""What `start`'s summary, `/effort show` and the channel say for a key the server refused: one
text, so the three places cannot drift."""


def _norm(url: str | None) -> str:
    return (url or "").rstrip("/")


def _peer_effort(harness: Any, base_url: str) -> EffortLevel | None:
    """The lifecycle effort of the first peer whose base_url is `base_url`, else None."""
    for peer in getattr(harness, "extra_endpoints", None) or ():
        if _norm(getattr(peer, "base_url", None)) == base_url:
            return getattr(getattr(peer, "lifecycle", None), "effort", None)
    return None


def endpoint_effort(harness: Any, base_url: str) -> EffortLevel | None:
    """The effort of the endpoint serving `base_url`, or None when that endpoint says nothing.

    Keyed on the LIVE base_url, never on whichever record is on disk: the primary's
    `server.effort` answers only for the primary's own URL, so a peer's effort never reaches it.
    """
    url = _norm(base_url)
    if url == _norm(getattr(getattr(harness, "provider", None), "base_url", None)):
        # The primary: its own record answers, or nobody does. A hand-run primary has no `server`
        # and must not borrow a peer's or the active selection's effort.
        return getattr(getattr(harness, "server", None), "effort", None)
    active = getattr(harness, "active_endpoint", None)
    if active is not None and _norm(getattr(active, "base_url", None)) == url:
        own = getattr(active, "effort", None)
        return own if own is not None else _peer_effort(harness, url)
    return _peer_effort(harness, url)


def resolve_effort(
    session: EffortLevel | None,
    agent_file: str | None,
    harness: Any,
    base_url: str,
) -> tuple[EffortLevel, Rung]:
    """The effective level and the rung that supplied it."""
    if session is not None:
        return session, "session"
    if agent_file not in (None, "inherit"):
        return agent_file, "agent"  # type: ignore[return-value]
    ep = endpoint_effort(harness, base_url)
    if ep is not None:
        return ep, "endpoint"
    org = getattr(getattr(harness, "org", None), "default_effort", None)
    if org is not None:
        return org, "org"
    return "high", "default"
