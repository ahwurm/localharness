"""The names `start --channel` accepts (PAPI-09), resolved from STATIC manifests before any plugin
loads: core channels, every bundled manifest of kind "channel", and the legacy entry. A channel
manifest names its channels in `manifest.channels`; empty means the plugin's own name. Third-party
channel kinds are not accepted in v0.16 (bundled only): a channel sees every event, tool results
included, and can inject user messages — a larger trust grant than a tool.

`accepted_channels(resolution)` is the enablement-aware set: core plus the channels of channel
plugins the resolved plan has on."""
from __future__ import annotations

CORE_CHANNELS = frozenset({"terminal", "acp"})
LEGACY_CHANNELS = frozenset({"discord"})  # Phase 49 deletes (dispatch converts)
OWN_COMMAND = frozenset({"web", "acp"})   # served by their own command, never built by start


def _names(cls) -> tuple[str, ...]:
    return cls.manifest.channels or (cls.manifest.name,)


def plugin_channel_names() -> frozenset[str]:
    """Bundled manifests of kind "channel" — class-level manifests only; nothing is instantiated."""
    from localharness.plugins.builtin import bundled_plugins
    return frozenset(n for c in bundled_plugins() if c.manifest.kind == "channel" for n in _names(c))


def channel_names() -> frozenset[str]:
    return CORE_CHANNELS | LEGACY_CHANNELS | plugin_channel_names()


def accepted_channels(resolution) -> frozenset[str]:
    """Core channels plus the channels of every channel plugin whose plan state is "on"."""
    on = {e.name for e in resolution.plan.entries if e.state == "on"}
    return CORE_CHANNELS | frozenset(n for name, c in resolution.classes.items()
                                     if name in on and c.manifest.kind == "channel" for n in _names(c))
