"""TAGG-04 (Plan 36.2-03, wave 2) — the no-grouping-regression comparator, unit-proven on
synthetic verdict pairs.

The pre-committed RULING-D KILL reads "no grouping regression vs the run-17 baseline". `compare`
grades a fresh designed-month verdict against the committed run-17 baseline on the pinned
B1/B2/B3/B4/A1 tolerances (formed-topics + ARI floor DERIVED from the baseline; a1_recall and
the b3 distractor bar fixed), and `write_verdict` emits exactly one grep-stable
`regression: none|detected` line. These tests lock a no-regression pair to `none` and each of
the five failing axes to `detected` (+ the axis name), and lock the artifact's grep contract."""
import re
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import tagg_regression_compare as trc  # noqa: E402

_REGRESSION_LINE = re.compile(r"^regression: (none|detected)$", re.MULTILINE)


def _baseline():
    """A dict shaped like run-17's verdict.json (only the fields compare() reads)."""
    return {
        "verdict": "HOLDS",
        "stage_a": {"a1_recall": 0.909},
        "stage_b": {
            "b1_ok": True, "b1_chapters_or_null": 5,
            "formed_topics": ["gpu_ops", "kyoto_trip", "markets", "race_training", "subagents"],
            "b2_ok": True, "ari": 0.759, "b3_ok": True, "noise_in_chapter": 0, "b4_ok": True,
        },
    }


def _candidate_ok():
    """A fresh verdict inside every tolerance: HOLDS, superset topics, ARI 0.74, b3 0, b4 ok."""
    return {
        "verdict": "HOLDS",
        "stage_a": {"a1_recall": 0.90},
        "stage_b": {
            "b1_chapters_or_null": 5,
            "formed_topics": ["gpu_ops", "kyoto_trip", "markets", "race_training", "subagents"],
            "b2_ok": True, "ari": 0.74, "noise_in_chapter": 0, "b4_ok": True,
        },
    }


def _one_regression_line(body: str) -> str:
    matches = _REGRESSION_LINE.findall(body)
    assert len(matches) == 1, f"expected exactly one column-0 regression line, got {matches}"
    return _REGRESSION_LINE.search(body).group(0)


def test_no_regression_pair_writes_none(tmp_path):
    reg, notes = trc.compare(_candidate_ok(), _baseline())
    assert reg is False and notes == []
    out = tmp_path / "verdict.md"
    trc.write_verdict(out, reg, notes, candidate_path="c.json", baseline_path="b.json")
    assert _one_regression_line(out.read_text()) == "regression: none"


def _assert_detected(tmp_path, reg, notes):
    out = tmp_path / "verdict.md"
    trc.write_verdict(out, reg, notes)
    assert _one_regression_line(out.read_text()) == "regression: detected"


def test_regression_detected_missing_chapter_b1(tmp_path):
    """B1: a formed_topics set missing a baseline topic ('markets' dropped) is a regression."""
    c = _candidate_ok()
    c["stage_b"]["formed_topics"] = ["gpu_ops", "kyoto_trip", "race_training", "subagents"]
    reg, notes = trc.compare(c, _baseline())
    assert reg is True and any("markets" in n for n in notes)
    _assert_detected(tmp_path, reg, notes)


def test_regression_detected_low_ari_b2(tmp_path):
    """B2: ARI 0.61 is below the disclosed 0.70 floor (baseline 0.759 - 0.06 band)."""
    c = _candidate_ok()
    c["stage_b"]["ari"] = 0.61
    reg, notes = trc.compare(c, _baseline())
    assert reg is True and any("b2_ari" in n for n in notes)
    _assert_detected(tmp_path, reg, notes)


def test_regression_detected_broken_arc_b4(tmp_path):
    """B4: a broken correction arc (b4_ok false) is a regression."""
    c = _candidate_ok()
    c["stage_b"]["b4_ok"] = False
    reg, notes = trc.compare(c, _baseline())
    assert reg is True and any("b4" in n for n in notes)
    _assert_detected(tmp_path, reg, notes)


def test_regression_detected_verdict_not_holds(tmp_path):
    """A candidate whose own verdict is not HOLDS is a regression."""
    c = _candidate_ok()
    c["verdict"] = "KILL"
    reg, notes = trc.compare(c, _baseline())
    assert reg is True and any("verdict" in n for n in notes)
    _assert_detected(tmp_path, reg, notes)


def test_regression_detected_low_a1_recall(tmp_path):
    """A1: recall 0.70 is below the fixed 0.80 floor."""
    c = _candidate_ok()
    c["stage_a"]["a1_recall"] = 0.70
    reg, notes = trc.compare(c, _baseline())
    assert reg is True and any("a1_recall" in n for n in notes)
    _assert_detected(tmp_path, reg, notes)


def test_no_regression_when_b4_raw_false_but_excused(tmp_path):
    """#41: an EXCUSED B4 run (raw b4_ok False, b4_excused True, verdict HOLDS — the grader's own
    b4_effective path) is NOT a regression. The comparator must derive the same effective value the
    grader folds into its verdict; reading raw b4_ok alone wrongly flagged these excused runs."""
    c = _candidate_ok()
    c["stage_b"]["b4_ok"] = False
    c["stage_b"]["b4_excused"] = True
    reg, notes = trc.compare(c, _baseline())
    assert reg is False and notes == []
    out = tmp_path / "verdict.md"
    trc.write_verdict(out, reg, notes, candidate_path="c.json", baseline_path="b.json")
    assert _one_regression_line(out.read_text()) == "regression: none"


def test_regression_when_b4_raw_false_and_not_excused(tmp_path):
    """#41: a genuinely-failing arc (raw b4_ok False, b4_excused False) is still a regression — the
    effective value is false, so tightening the excusal spares only truly-excused runs."""
    c = _candidate_ok()
    c["stage_b"]["b4_ok"] = False
    c["stage_b"]["b4_excused"] = False
    reg, notes = trc.compare(c, _baseline())
    assert reg is True and any("b4" in n for n in notes)
    _assert_detected(tmp_path, reg, notes)


def test_b4_old_shape_verdict_without_excused_key_unchanged(tmp_path):
    """#41: an old verdict.json with no b4_excused key falls back to raw b4_ok exactly as before —
    b4_ok True -> no note; b4_ok False -> regression. No behavior change for pre-excusal runs."""
    ok = _candidate_ok()                       # b4_ok True, no b4_excused key
    assert "b4_excused" not in ok["stage_b"]
    reg, notes = trc.compare(ok, _baseline())
    assert reg is False and not any("b4" in n for n in notes)
    bad = _candidate_ok()
    bad["stage_b"]["b4_ok"] = False            # raw false, no b4_excused key
    reg2, notes2 = trc.compare(bad, _baseline())
    assert reg2 is True and any("b4" in n for n in notes2)


def test_run17_self_compare_is_none(tmp_path):
    """The committed run-17 baseline compared to ITSELF is by definition no-regression — the
    reproducibility anchor Plan 04's verify greps."""
    import json
    baseline_path = Path(__file__).resolve().parents[2] / ".planning/runs/2026-07-11-run17/results/verdict.json"
    if not baseline_path.exists():
        import pytest
        pytest.skip("run-17 baseline not present (git-ignored .planning)")
    v = json.loads(baseline_path.read_text())
    reg, notes = trc.compare(v, v)
    assert reg is False and notes == []
