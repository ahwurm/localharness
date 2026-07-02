"""Hierarchical char-budget reduce — the cruncher's reduce, factored out (v2.0 HIER-01).

Pure algorithm, no I/O: greedy char-budget batching + level-by-level partial combines
until the items fit the budget. The cruncher composes it with its LLM combine turn
(`agent/subagent.py`); the memory hierarchy (HIER-02, `memory/hierarchy.py`) persists
the returned trace as a durable gist tree. Behavior is byte-identical to the while-loop
it replaced (same batching, same sequential combines, same termination conditions) —
the factor-out exists so the intermediate gists stop being throwaway Python strings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Optional


@dataclass(frozen=True)
class ReduceLevel:
    """One reduce level's full trace: batches[i] (the inputs) → outputs[i] (the gist)."""
    level: int
    batches: list[list[str]]
    outputs: list[str]


def batch_by_budget(items: list[str], budget: int) -> list[list[str]]:
    """Greedy char-budget batching (verbatim the cruncher's inline algorithm): a batch
    closes when adding the next item would exceed `budget`; an oversized single item
    still gets its own batch (never dropped)."""
    batches: list[list[str]] = []
    cur: list[str] = []
    clen = 0
    for e in items:
        if cur and clen + len(e) > budget:
            batches.append(cur)
            cur, clen = [], 0
        cur.append(e)
        clen += len(e)
    if cur:
        batches.append(cur)
    return batches


async def hierarchical_reduce(
    items: list[str],
    *,
    budget: int,
    combine_partial: Callable[[list[str]], Awaitable[str]],
    max_levels: int = 4,
    on_level: Optional[Callable[[int, int, int], None]] = None,
) -> tuple[list[str], int, list[ReduceLevel]]:
    """Reduce `items` level-by-level until they fit `budget` (or a single item remains,
    or `max_levels` is hit — the cruncher's proven termination triple).

    Returns (final_items, levels_used, trace). The trace records every batch→gist pair,
    which is exactly what HIER-02 persists (gist nodes + derived_from edges).
    `on_level(level, n_items, n_batches)` is the logging seam.
    """
    level = 0
    trace: list[ReduceLevel] = []
    while len("\n\n".join(items)) > budget and len(items) > 1 and level < max_levels:
        level += 1
        batches = batch_by_budget(items, budget)
        if on_level is not None:
            on_level(level, len(items), len(batches))
        outputs: list[str] = []
        for b in batches:
            outputs.append(await combine_partial(b))
        trace.append(ReduceLevel(level=level, batches=batches, outputs=outputs))
        items = outputs
    return items, level, trace
