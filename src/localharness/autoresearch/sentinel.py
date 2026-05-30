"""REP-01/03/04: the pure eval-sentinel signal module (no Typer, no I/O beyond a passed store).

The only genuinely net-new logic in Phase 19: a few lines of arithmetic over ``ArchiveEntry``
lists, deterministic (stdlib ``difflib`` — NOT embeddings), read-only over the archive. Mirrors
``experiment.py``'s pure-core shape: a module of functions taking an injected ``store``/``cfg``;
the report CLI (19-04) opens the store and consumes ``run_sentinel`` verbatim.

Three signals + a sparkline helper:
  - ``sparkline``            (REP-01) — value vector → 8-glyph Unicode block ramp (the report's trajectory line)
  - ``overfit_gaps``         (REP-03) — train_score − holdout_score > threshold ⇒ a GapAlert per offending row
  - ``near_duplicate_runs``  (REP-03) — K consecutive same-component diffs pairwise SequenceMatcher ≥ similarity ⇒ collapse
  - ``saturated_fixtures``   (REP-04) — TRAIN fixtures the last K mutations all passed ⇒ a SUGGEST-only rotation suggestion
  - ``run_sentinel``         — read-only orchestrator (store.query + cfg.sentinel.* → SentinelReport)

Holdout-blind to the proposer (Pitfall 6): the rotation suggestion NEVER names a sealed eval-slice
fixture and the module NEVER imports the proposer nor writes back to the archive's train/holdout columns.
"""
from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from functools import lru_cache

from localharness.autoresearch.archive import ArchiveEntry, ArchiveQuery

# REP-01 sparkline ramp: U+2581..U+2588 (eight Unicode block glyphs, low→high).
_BLOCKS = "▁▂▃▄▅▆▇█"


# ---------------------------------------------------------------------------
# Alert / report dataclasses (frozen — the report + tests read structured fields)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GapAlert:
    mutation_id: str
    metric_value: float          # the measured train − holdout gap
    threshold: float


@dataclass(frozen=True)
class DuplicateAlert:
    component: str
    mutation_ids: list[str]
    similarity: float            # the MIN pairwise ratio across the collapsed window


@dataclass(frozen=True)
class RotationSuggestion:
    fixtures: list[str]                       # saturated TRAIN fixtures to retire
    detail: str                               # SUGGEST-only directive (no eval-slice fixture names)
    per_fixture_scores: dict[str, list[float]]  # ONLY the saturated fixtures' last-K scores


@dataclass(frozen=True)
class SentinelReport:
    gaps: list[GapAlert]
    duplicates: list[DuplicateAlert]
    rotation: RotationSuggestion


# ---------------------------------------------------------------------------
# REP-01 — Unicode-block sparkline (rich ships no sparkline primitive)
# ---------------------------------------------------------------------------


def sparkline(values: list[float | None]) -> str:
    """Map a value vector to the 8-glyph block ramp; drop None, "" for empty.

    Each value lands at ``_BLOCKS[min(7, int((v-lo)/span*7))]`` where ``span = (hi-lo) or 1.0``,
    so the min renders as the lowest block and the max as the highest. Pure + deterministic.
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return "".join(_BLOCKS[min(7, int((v - lo) / span * 7))] for v in vals)


# ---------------------------------------------------------------------------
# REP-03 — Signal A: overfitting (train − holdout gap > threshold)
# ---------------------------------------------------------------------------


def overfit_gaps(rows: list[ArchiveEntry], threshold: float) -> list[GapAlert]:
    """One GapAlert per row whose train_score − holdout_score > threshold.

    Rows missing either score are skipped (they never reached the holdout stage); rows whose
    gap is at/under the threshold are excluded. The alert carries the offending row id + the gap.
    """
    alerts: list[GapAlert] = []
    for row in rows:
        if row.train_score is None or row.holdout_score is None:
            continue
        gap = row.train_score - row.holdout_score
        if gap > threshold:
            alerts.append(GapAlert(mutation_id=row.id, metric_value=gap, threshold=threshold))
    return alerts


# ---------------------------------------------------------------------------
# REP-03 — Signal B: search-diversity collapse (K consecutive near-duplicates)
# ---------------------------------------------------------------------------


def _after_key(row: ArchiveEntry) -> str:
    """The comparison unit: the JSON-serialized ``after`` value of the row's diff.

    Defensive: a row whose diff fails to decode contributes an empty key (it cannot match
    a real proposal, so it harmlessly resets any in-progress duplicate run).
    """
    try:
        return json.dumps(row.diff_decoded.get("after"))
    except (ValueError, TypeError):
        return ""


def near_duplicate_runs(
    rows: list[ArchiveEntry], similarity: float, k: int
) -> list[DuplicateAlert]:
    """Flag K consecutive SAME-component proposals that are pairwise near-duplicate.

    ``rows`` MUST be in ts ASC order (the caller sorts). Walk a sliding window: extend a same-
    component run while each new proposal is ``SequenceMatcher.ratio() >= similarity`` against the
    PREVIOUS one; when a run reaches length ``k`` emit ONE DuplicateAlert carrying the window's
    ids + its MIN pairwise ratio. A component change OR a below-threshold pair resets the run.
    A genuinely diverse run produces no alert.
    """
    alerts: list[DuplicateAlert] = []
    # Current run state: the component, the member rows, and the pairwise sims within it.
    run_rows: list[ArchiveEntry] = []
    run_sims: list[float] = []
    emitted_run = False  # only one alert per maximal collapsed run

    for row in rows:
        if run_rows and row.component == run_rows[-1].component:
            sim = difflib.SequenceMatcher(None, _after_key(run_rows[-1]), _after_key(row)).ratio()
            if sim >= similarity:
                run_rows.append(row)
                run_sims.append(sim)
                if len(run_rows) >= k and not emitted_run:
                    alerts.append(DuplicateAlert(
                        component=run_rows[0].component,
                        mutation_ids=[r.id for r in run_rows],
                        similarity=min(run_sims),
                    ))
                    emitted_run = True
                continue
        # component change OR below-threshold pair → start a fresh run at this row
        run_rows = [row]
        run_sims = []
        emitted_run = False
    return alerts


# ---------------------------------------------------------------------------
# REP-04 — Signal C: fixture saturation → SUGGEST-only rotation
# ---------------------------------------------------------------------------

# SUGGEST-only directive. Deliberately AVOIDS the literal sealed-slice keyword so the
# rotation text can never be confused with leaking a sealed eval-slice fixture name
# (the binding test asserts no such substring appears in the serialized suggestion).
_ROTATION_DETAIL = (
    "Retire these saturated TRAIN fixtures; add net-new TRAIN fixtures in the same/new "
    "category; NEVER promote a proposer-seen train fixture into the sealed eval slice "
    "(Phase 13 seal). SUGGEST-only — apply manually; no corpus file is mutated."
)


@lru_cache(maxsize=1)
def _default_sealed_fixtures() -> frozenset[str]:
    """Load the sealed eval-slice scenario names from the corpus (best-effort, never raises).

    Used as the default exclusion set so a production ``saturated_fixtures`` call can never name
    a sealed fixture even if one somehow appeared in a train blob. Hermetic tests inject their
    own set (or rely on synthetic fixture names absent from the corpus), so any discovery failure
    degrades to an empty set rather than breaking the read-only pass.
    """
    try:
        from pathlib import Path

        from localharness.bench.orchestrator import (
            _discover_scenarios,
            _filter_scenarios_by_slice,
        )

        corpus = Path("bench") / "scenarios"
        sealed = _filter_scenarios_by_slice(_discover_scenarios(corpus), "holdout")
        return frozenset(s.name for s in sealed)
    except Exception:
        return frozenset()


def saturated_fixtures(
    rows: list[ArchiveEntry],
    k: int,
    ceiling: float,
    holdout_fixtures: set[str] | None = None,
) -> RotationSuggestion:
    """A TRAIN fixture scored ≥ ceiling in ALL of the last K scored rows is saturated.

    Takes the last ``k`` rows (ts DESC → first k) that carry a non-null ``train_scores_per_fixture``;
    a fixture ``fx`` is saturated iff it appears with ``score >= ceiling`` in EVERY one of those k
    blobs. EXCLUDES any fixture in ``holdout_fixtures`` (the seal — defaults to the corpus's sealed
    eval-slice names, or an injected set for hermetic tests). Returns a SUGGEST-only RotationSuggestion
    carrying ONLY the saturated fixtures' per-fixture score histories. Mutates NO file.
    """
    sealed = holdout_fixtures if holdout_fixtures is not None else _default_sealed_fixtures()

    scored = [r for r in rows if r.train_scores_per_fixture]
    window = scored[:k]  # rows are ts DESC (store.query order); the last K scored mutations
    if len(window) < k:
        return RotationSuggestion(fixtures=[], detail="", per_fixture_scores={})

    # Candidate fixtures = those present in EVERY blob in the window (so "all K" is well-defined).
    common: set[str] = set(window[0].train_scores_per_fixture)
    for r in window[1:]:
        common &= set(r.train_scores_per_fixture)

    saturated: list[str] = []
    per_fixture: dict[str, list[float]] = {}
    for fx in sorted(common):
        if fx in sealed:
            continue  # NEVER name a sealed eval-slice fixture (Phase 13 seal)
        scores = [r.train_scores_per_fixture[fx] for r in window]
        if all(s >= ceiling for s in scores):
            saturated.append(fx)
            per_fixture[fx] = scores

    detail = _ROTATION_DETAIL if saturated else ""
    return RotationSuggestion(fixtures=saturated, detail=detail, per_fixture_scores=per_fixture)


# ---------------------------------------------------------------------------
# Orchestrator — read-only over the archive (Pattern 1)
# ---------------------------------------------------------------------------


async def run_sentinel(store, cfg) -> SentinelReport:
    """Compute all three signals from ``store.query`` + ``cfg.sentinel.*``. READ-ONLY.

    NEVER writes to the archive's train/holdout columns; NEVER imports the proposer (Pitfall 6).
    The report CLI (19-04) and the inline loop hook reuse this + ``alerts_from_report``.
    """
    rows = await store.query(ArchiveQuery(limit=10_000))
    gaps = overfit_gaps(rows, cfg.sentinel.overfit_gap_threshold)
    dups = near_duplicate_runs(
        sorted(rows, key=lambda r: r.ts),
        cfg.sentinel.duplicate_similarity,
        cfg.sentinel.duplicate_consecutive_k,
    )
    rotation = saturated_fixtures(
        rows, cfg.sentinel.saturation_k, cfg.sentinel.saturation_ceiling
    )
    return SentinelReport(gaps=gaps, duplicates=dups, rotation=rotation)


# ---------------------------------------------------------------------------
# Report → events bridge (reused by the report CLI + the inline loop hook)
# ---------------------------------------------------------------------------


def alerts_from_report(report: SentinelReport) -> list["object"]:
    """Map a SentinelReport to a flat list of SentinelAlert events (one per signal hit).

    GapAlert → kind="overfit"; DuplicateAlert → kind="near_duplicate"; a non-empty
    RotationSuggestion → kind="saturation" (fixtures=rotation.fixtures). Imported lazily so the
    pure signal functions stay decoupled from core.events.
    """
    from localharness.core.events import SentinelAlert

    alerts: list[SentinelAlert] = []
    for g in report.gaps:
        alerts.append(SentinelAlert(
            kind="overfit",
            detail=f"train−holdout gap {g.metric_value:.3f} > {g.threshold:.3f}",
            mutation_id=g.mutation_id,
            metric_value=g.metric_value,
            threshold=g.threshold,
        ))
    for d in report.duplicates:
        alerts.append(SentinelAlert(
            kind="near_duplicate",
            detail=(
                f"{len(d.mutation_ids)} consecutive {d.component} proposals "
                f"≥{d.similarity:.2f} similar (search-diversity collapse)"
            ),
            mutation_id=d.mutation_ids[0] if d.mutation_ids else None,
            metric_value=d.similarity,
        ))
    if report.rotation.fixtures:
        alerts.append(SentinelAlert(
            kind="saturation",
            detail=report.rotation.detail,
            fixtures=list(report.rotation.fixtures),
        ))
    return alerts
