"""ParentSampler: ε-greedy parent selection over the GEPA per-fixture Pareto front (AUTO-02).

The archive ships ``pareto_front_per_fixture()`` — the dumb, holdout-blind non-dominated
SET (no weights, no sampling). This module adds the search on top per 15-CONTEXT: coverage-
proportional EXPLOIT over that front + ε EXPLORE across the whole archive (DGM: even rejected
branches are reachable parents) + a cold-start that always seeds from the baseline root.

Sealed-slice (Pitfall 4): the sampler reads ONLY ``pareto_front_per_fixture()`` and each
candidate's ``train_scores_per_fixture``. The sealed holdout column is NEVER consulted — the
absence of any holdout read IS the seal at the sampler. The returned parent is a SIGNAL for
lineage + bias ("go broad, not deep"); it is never composed into state — the proposer targets
live failures, not the parent's component.

Pure module: no Typer, no CLI. The ``rng`` is injectable so the loop driver (18-05) and the
tests get deterministic draws.
"""
from __future__ import annotations

import random

from localharness.autoresearch.archive import ArchiveEntry, ArchiveQuery, ArchiveStore

# Cold-start sentinel: "mutate from HEAD, favor an untouched component". Returned whenever
# there is no front to bias from (empty archive) or a degenerate/empty explore pool.
BASELINE_ROOT = object()


class ParentSampler:
    """ε-greedy parent selection over the GEPA per-fixture Pareto front (AUTO-02).

    Holdout-blind: reads ONLY pareto_front_per_fixture() + train_scores_per_fixture.
    NEVER reads the sealed holdout column (Pitfall 4 — the sealed-slice rationale).
    The returned parent is a SIGNAL for lineage + bias, not composed state.
    """

    def __init__(
        self,
        store: ArchiveStore,
        *,
        epsilon: float = 0.2,
        rng: random.Random | None = None,
    ):
        self._store = store
        self._eps = epsilon
        self._rng = rng or random.Random()

    async def sample(self) -> ArchiveEntry | object:
        """Pick a parent: cold-start → root; else ε EXPLORE / (1-ε) coverage-weighted EXPLOIT."""
        front = await self._store.pareto_front_per_fixture()
        if not front:                       # COLD START — always explore from root, ε is moot
            return BASELINE_ROOT
        if self._rng.random() < self._eps:  # EXPLORE
            return await self._explore()
        return self._exploit(front)         # EXPLOIT (front already fetched; sync)

    async def _explore(self) -> ArchiveEntry | object:
        # ALL statuses, no filter (DGM: overlooked/rejected branches remain reachable parents).
        rows = await self._store.query(ArchiveQuery(limit=10_000))
        return self._rng.choice(rows) if rows else BASELINE_ROOT

    def _exploit(self, front: list[ArchiveEntry]) -> ArchiveEntry | object:
        coverage = self._coverage_counts(front)   # {id: #fixtures-best-on}
        members = list(coverage)
        weights = [coverage[i] for i in members]
        if not members:                            # defensive
            return BASELINE_ROOT
        if sum(weights) == 0:                      # degenerate -> uniform over the front
            return self._rng.choice(front)
        chosen_id = self._rng.choices(members, weights=weights, k=1)[0]
        return next(e for e in front if e.id == chosen_id)

    @staticmethod
    def _coverage_counts(front: list[ArchiveEntry]) -> dict[str, int]:
        """#fixtures each front member ties-or-beats the per-fixture max on (train ONLY).

        Mirrors ArchiveStore.pareto_front_per_fixture's inner loop (archive.py:362-381) —
        reads ONLY train_scores_per_fixture; the sealed holdout column is NEVER consulted.
        Accumulates across the WHOLE fixture set (no early return — coverage is the sum over
        all fixtures).
        """
        fixtures: set[str] = set()
        for e in front:
            if e.train_scores_per_fixture:
                fixtures.update(e.train_scores_per_fixture.keys())
        counts = {e.id: 0 for e in front}
        for fx in fixtures:
            scored = [
                (e, e.train_scores_per_fixture[fx])
                for e in front
                if e.train_scores_per_fixture and fx in e.train_scores_per_fixture
            ]
            if not scored:
                continue
            best = max(s for _, s in scored)
            for e, s in scored:
                if s == best:                      # ties included (mirror the front)
                    counts[e.id] += 1
        return counts
