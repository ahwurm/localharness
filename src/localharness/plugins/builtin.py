"""BUILTIN_PLUGINS — the ONE list of what ships (CORE-03).

The loader, the startup banner, `plugins list`, `doctor` and `components` all reach it through
bundled_plugins() — never by importing the name — so there is no second list anywhere and a test can
swap this one and watch every reader follow. It is empty until the bundled features convert; each
conversion appends exactly one class. This is the one module in core allowed to import a plugin
module (CORE-02)."""
from __future__ import annotations

from localharness.plugins.api import Plugin

BUILTIN_PLUGINS: tuple[type[Plugin], ...] = ()


def bundled_plugins() -> tuple[type[Plugin], ...]:
    """The shipped plugin classes, read at CALL time (monkeypatchable)."""
    return BUILTIN_PLUGINS
