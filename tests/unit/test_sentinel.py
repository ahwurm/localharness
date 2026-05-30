"""Phase 19 Wave-0 — eval-sentinel signals (REP-03 overfit/near-duplicate, REP-04 saturation, REP-01 sparkline).

RED stubs in the project's established Wave-0 cadence (14-04/15-04/16-03/17-04/18-06): every
19-VALIDATION.md sentinel row exists here as an ``@pytest.mark.xfail(strict=False)`` stub with
REAL assertions BEFORE ``src/localharness/autoresearch/sentinel.py`` ships in 19-03. A guarded
module-top import skips the whole module cleanly until then (mirrors the 17-01/18-01 idiom in
test_autoresearch_loop.py), so collection never breaks. Later waves flip xfail→pass as impl lands.

The locked defaults the assertions encode (19-RESEARCH §"Sentinel Signals"):
  overfit_gap_threshold 0.10 · duplicate_similarity 0.90 · duplicate_consecutive_k 3 ·
  saturation_k 5 · saturation_ceiling 0.99. Near-duplicate metric = difflib.SequenceMatcher
  (syntactic, deterministic, hermetic — NOT embeddings).
"""
import json

import pytest

try:
    from localharness.autoresearch.sentinel import (
        sparkline,
        overfit_gaps,
        near_duplicate_runs,
        saturated_fixtures,
        run_sentinel,  # noqa: F401  (orchestrator; consumed by the on-demand report/sentinel command)
    )
except ImportError:
    pytest.skip("sentinel.py lands in 19-03", allow_module_level=True)


def _as_jsonable(obj):
    """Best-effort dict/str view of a sentinel alert/suggestion (shape pinned in 19-03).

    Works whether the impl ships dataclasses, pydantic models, or plain objects — the
    stub only needs to grep the rendered text for a row id / fixture name, not pin a schema.
    """
    for attr in ("model_dump", "_asdict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            return fn()
    if hasattr(obj, "__dict__") and vars(obj):
        return vars(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# REP-01 — pure Unicode-block sparkline mapping
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False)  # impl-pending-19
def test_sparkline_mapping():
    """sparkline maps a [0,1] ramp onto the 8-block glyph string; empty→"" ; None skipped."""
    _RAMP = "▁▂▃▄▅▆▇█"
    out = sparkline([0.0, 0.5, 1.0])
    assert isinstance(out, str)
    assert len(out) == 3
    assert all(ch in _RAMP for ch in out)
    assert out[0] == "▁"  # the min maps to the lowest block
    assert out[-1] == "█"  # the max maps to the highest block

    assert sparkline([]) == ""  # no data → empty line (not a crash)

    skipped = sparkline([None, 0.5])
    assert len(skipped) == 1  # the None is skipped; only the one real value renders


# ---------------------------------------------------------------------------
# REP-03 — overfitting (train − holdout gap > threshold)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False)  # impl-pending-19
async def test_overfit_gap_fires(archive_store, seeded_archive, bus):
    """A row with train 0.9 / holdout 0.7 (gap 0.20 > 0.10) fires; a 0.02-gap row does not."""
    rows = await seeded_archive(
        archive_store,
        [
            dict(id="overfit-row", component="agent.role", status="promoted",
                 train_score=0.9, holdout_score=0.7),
            dict(id="tight-row", component="agent.role", status="promoted",
                 train_score=0.72, holdout_score=0.70),
        ],
    )
    assert rows  # seeding actually wrote (guards against a silent empty archive)

    gaps = overfit_gaps(rows, threshold=0.10)
    assert gaps, "the 0.20-gap promoted row must produce an overfit alert"
    flagged = gaps[0]
    # the alert carries the measured gap and references the offending row
    assert flagged.metric_value == pytest.approx(0.20, abs=1e-9)
    blob = json.dumps(_as_jsonable(flagged))
    assert "overfit-row" in blob  # the alert points at the row that overfit

    # the tight 0.02-gap row must NOT appear in the alert set
    assert "tight-row" not in json.dumps([_as_jsonable(g) for g in gaps])


# ---------------------------------------------------------------------------
# REP-03 — search-diversity collapse (K consecutive near-duplicates)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False)  # impl-pending-19
async def test_near_duplicate_collapse(archive_store, seeded_archive):
    """K=3 consecutive same-component near-identical diffs collapse-alert; a diverse run does not."""
    import difflib

    def _diff(after: str) -> str:
        return json.dumps({"before": "You are an assistant.", "after": after})

    near = [
        "You are a helpful assistant.",
        "You are a helpful assistant!",
        "You are a helpful  assistant.",
    ]
    # sanity: the seeded trio really is pairwise ≥ 0.90 similar (the metric the impl uses)
    for a, b in zip(near, near[1:]):
        ratio = difflib.SequenceMatcher(None, _diff(a), _diff(b)).ratio()
        assert ratio >= 0.90, f"fixture not near-duplicate enough: {ratio}"

    dup_rows = await seeded_archive(
        archive_store,
        [dict(id=f"dup-{n}", component="agent.role", diff=_diff(a), ts=100 + n)
         for n, a in enumerate(near)],
    )
    alerts = near_duplicate_runs(dup_rows, similarity=0.90, k=3)
    assert alerts, "3 consecutive ≥0.90-similar same-component proposals must collapse-alert"

    diverse = [
        "Always cite your sources in full.",
        "Refuse any request that touches the deny list.",
        "Summarize the financial filing in three bullets.",
    ]
    div_rows = await seeded_archive(
        archive_store,
        [dict(id=f"div-{n}", component="agent.role", diff=_diff(a), ts=200 + n)
         for n, a in enumerate(diverse)],
    )
    assert near_duplicate_runs(div_rows, similarity=0.90, k=3) == []  # genuine diversity → no alert


# ---------------------------------------------------------------------------
# REP-04 — fixture saturation → SUGGEST-only rotation (never names a holdout fixture)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False)  # impl-pending-19
async def test_saturation_rotation_suggestion(archive_store, seeded_archive):
    """Last K=5 rows all ceiling fx_a (1.0) but not fx_b (0.4) ⇒ suggest fx_a; never a holdout name."""
    rows = await seeded_archive(
        archive_store,
        [dict(id=f"sat-{n}", component="agent.role", status="promoted", ts=300 + n,
              train_scores_per_fixture={"fx_a": 1.0, "fx_b": 0.4})
         for n in range(5)],
    )
    assert len(rows) == 5

    sug = saturated_fixtures(rows, k=5, ceiling=0.99)
    text = json.dumps(_as_jsonable(sug)).lower()
    assert "fx_a" in text  # the ceiling'd fixture is a rotation candidate
    assert "fx_b" not in text  # the discriminating fixture stays (0.4 is well below ceiling)

    # Seal: the SUGGEST-only rotation NEVER names a sealed holdout fixture (Phase 13).
    # holdout scenarios in this corpus are prefixed prop_holdout_* / h*; assert none leak in.
    assert "holdout" not in text
    assert "prop_holdout" not in text
