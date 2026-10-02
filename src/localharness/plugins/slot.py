"""Core's memory slot (PAPI-04, ROADMAP D1): at most one occupant of kind "memory", and it may be
empty. Prompt assembly asks it for the per-turn section; the browse API and a subagent's write handle
go through it. Every call into the occupant is contained — a failing memory plugin costs that turn
its memory section, never the turn. The bundled memory plugin occupies it by default; it is empty with
memory off or when the memory plugin failed to start."""
from __future__ import annotations

import logging
from localharness.plugins.api import (
    ContextBudget, ContextContribution, MemoryBrowse, MemorySlotPlugin, MemoryWriteHandle,
    PluginContext,
)

log = logging.getLogger(__name__)


class MemorySlot:
    """The slot and its occupant: a running MemorySlotPlugin with its context and name, or nobody.
    The lifecycle (plugins/lifecycle.py) seats the plan's memory occupant once it has started."""

    def __init__(self, occupant: MemorySlotPlugin | None = None, ctx: PluginContext | None = None,
                 name: str | None = None) -> None:
        self._occupant, self._ctx, self._name = occupant, ctx, name

    def seat(self, occupant: MemorySlotPlugin, ctx: PluginContext | None = None,
             name: str | None = None) -> None:
        """Seat an occupant on THIS slot, in place, so every holder of the slot sees it."""
        self._occupant, self._ctx, self._name = occupant, ctx, name

    @property
    def occupied(self) -> bool:
        return self._occupant is not None

    @property
    def occupant_name(self) -> str | None:
        return self._name

    async def context(self, turn: str, budget: ContextBudget) -> ContextContribution:
        """The occupant's section(s) for this turn, within `budget`. Empty when nobody occupies the
        slot, or when the occupant raises or answers with the wrong type (named in a warning): the
        turn goes on without a memory section."""
        if self._occupant is None:
            return ContextContribution()
        error: BaseException | None = None
        try:
            got = await self._occupant.context(self._ctx, turn, budget)
            if isinstance(got, ContextContribution):
                return got
            why = f"returned {type(got).__name__}, not a ContextContribution"
        except (Exception, SystemExit) as exc:  # noqa: BLE001 — a memory failure never takes the turn
            error, why = exc, f"raised {type(exc).__name__}: {exc}"
        log.warning("memory plugin %s: context() %s — no memory section this turn", self._name, why,
                    exc_info=error)
        return ContextContribution()

    def browse(self) -> MemoryBrowse | None:
        """The occupant's browse API; None with no occupant, none offered, or a failure (named)."""
        return self._ask("browse", MemoryBrowse)

    def bind_subagent(self) -> MemoryWriteHandle | None:
        """A write handle for one subagent from the occupant; None — that subagent then persists
        nothing — with no occupant, none offered, or a failure (named)."""
        return self._ask("bind_subagent", MemoryWriteHandle, self._ctx)

    def _ask(self, verb: str, kind: type, *args: object) -> object | None:
        if self._occupant is None:
            return None
        error: BaseException | None = None
        try:
            got = getattr(self._occupant, verb)(*args)
            if got is None or isinstance(got, kind):
                return got
            why = f"returned {type(got).__name__}, not a {kind.__name__}"
        except (Exception, SystemExit) as exc:  # noqa: BLE001 — never fatal
            error, why = exc, f"raised {type(exc).__name__}: {exc}"
        log.warning("memory plugin %s: %s() %s — treated as None", self._name, verb, why,
                    exc_info=error)
        return None
