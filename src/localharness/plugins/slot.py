"""Core's memory slot (PAPI-04, ROADMAP D1): at most one occupant of kind "memory", and it may be
empty. Prompt assembly asks it for the per-turn section; the browse API and a subagent's write handle
go through it. Every call into the occupant is contained — a failing memory plugin costs that turn
its memory section, never the turn. Empty in every real session until memory converts."""
from __future__ import annotations

from localharness.plugins.api import MemorySlotPlugin, PluginContext


class MemorySlot:
    """The slot and its occupant: a running MemorySlotPlugin with its context and name, or nobody.
    The lifecycle (plugins/lifecycle.py) seats the plan's memory occupant once it has started."""

    def __init__(self, occupant: MemorySlotPlugin | None = None, ctx: PluginContext | None = None,
                 name: str | None = None) -> None:
        self._occupant, self._ctx, self._name = occupant, ctx, name

    @property
    def occupied(self) -> bool:
        return self._occupant is not None

    @property
    def occupant_name(self) -> str | None:
        return self._name
