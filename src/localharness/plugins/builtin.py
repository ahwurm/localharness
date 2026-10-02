"""BUILTIN_PLUGINS — the ONE list of what ships (CORE-03).

The loader, the startup banner, `plugins list`, `doctor` and `components` all reach it through
bundled_plugins() — never by importing the name — so there is no second list anywhere and a test can
swap this one and watch every reader follow. The image plugin is the first entry, web the second,
memory the third, dispatch the fourth and autoresearch the fifth; each bundled feature that converts appends exactly one class. This is the one module in core allowed to import a plugin
module (CORE-02)."""
from __future__ import annotations

from localharness.plugins.api import Plugin
from localharness.autoresearch.plugin import AutoresearchPlugin
from localharness.cli.web_plugin import WebPlugin
from localharness.dispatch.plugin import DispatchPlugin
from localharness.memory.plugin import MemoryPlugin
from localharness.tools.builtin.image_plugin import ImagePlugin

BUILTIN_PLUGINS: tuple[type[Plugin], ...] = (ImagePlugin, WebPlugin, MemoryPlugin, DispatchPlugin,
                                              AutoresearchPlugin)


def bundled_plugins() -> tuple[type[Plugin], ...]:
    """The shipped plugin classes, read at CALL time (monkeypatchable)."""
    return BUILTIN_PLUGINS
