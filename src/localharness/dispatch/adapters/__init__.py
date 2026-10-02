"""The chat platforms the dispatch plugin can drive: platform name -> "module:Class" (imported lazily).

A second platform is one file here plus one line in ADAPTERS plus its name in the manifest's
`channels` tuple — nothing outside dispatch/ changes."""
from __future__ import annotations

import importlib

ADAPTERS: dict[str, str] = {"discord": "localharness.dispatch.adapters.discord:DiscordAdapter"}


def load_adapter(name: str) -> type:
    """The adapter class for `name`; importing it imports nothing platform-specific (the SDK is
    imported inside the adapter's connect)."""
    module, _, attr = ADAPTERS[name].partition(":")
    return getattr(importlib.import_module(module), attr)
