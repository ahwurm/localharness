"""The memory plugin: the memory slot's occupant — facts recalled into each turn, the three memory
tools, the session row and background consolidation.

On by default. plugins/builtin.py imports this module for every `--help`, `doctor` and `plugins list`,
so it imports only the plugin API at module level; the store (aiosqlite), the router, the resonance
engine (numpy), consolidation, the tools and rich are imported inside the methods that use them.
It lives in memory/ because memory/__init__.py re-exports lazily (PEP 562): importing this module
runs the package __init__, which imports nothing. It never imports cli/start_cmd.py (PAPI-03)."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from localharness.plugins.api import (
    Availability, Check, ContextBudget, ContextContribution, MemorySlotPlugin, PluginContext,
    PluginManifest,
)

if TYPE_CHECKING:
    from localharness.tools.base import ToolProtocol

log = logging.getLogger(__name__)


class MemoryPlugin(MemorySlotPlugin):
    """persistent memory: facts recalled into each turn, memory tools, background consolidation"""

    manifest = PluginManifest(name="memory", version="0.1.0", kind="memory", enabled_by_default=True)
    ConfigModel = None  # resolve() strips enabled; nothing else is harness-level
    AgentConfigModel = None  # set to MemoryConfig when the model moves here (same commit as the registration)
    wants_artifacts = False

    def __init__(self) -> None:
        self.startup_warnings: list[str] = []
        self._store: Any = None
        self._twin: Any = None
        self._router: Any = None
        self._engine: Any = None
        self._acc: Any = None
        self._sched: Any = None
        self._session_started = False
        self._browse: Any = None

    async def configure(self, ctx: PluginContext) -> Availability:
        """Opens nothing — doctor calls it outside a session, where ctx.llm is None."""
        return "ready"
