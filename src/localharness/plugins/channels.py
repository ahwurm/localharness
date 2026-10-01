"""The names `start --channel` accepts (PAPI-09), resolved from STATIC manifests before any plugin
loads: core channels, every bundled manifest of kind "channel", and the legacy entry. Third-party
channel kinds are not accepted in v0.16 (bundled only): a channel sees every event, tool results
included, and can inject user messages — a larger trust grant than a tool."""
from __future__ import annotations

CORE_CHANNELS = frozenset({"terminal", "acp"})
LEGACY_CHANNELS = frozenset({"discord"})  # Phase 49 deletes (dispatch converts)
OWN_COMMAND = frozenset({"web", "acp"})   # served by their own command, never built by start


def plugin_channel_names() -> frozenset[str]:
    """Bundled manifests of kind "channel" — class-level manifests only; nothing is instantiated."""
    from localharness.plugins.builtin import bundled_plugins
    return frozenset(c.manifest.name for c in bundled_plugins() if c.manifest.kind == "channel")


def channel_names() -> frozenset[str]:
    return CORE_CHANNELS | LEGACY_CHANNELS | plugin_channel_names()
