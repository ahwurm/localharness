"""Corpus-shape tests for bench/scenarios/*.yaml.

Wave 1: parse-clean + failure-mode-shape checks. Wave 2/3 extend
test_scenario_class_coverage to assert all 12 canonical class names appear.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from localharness.bench.schema import ScenarioSpec, load_scenario

CORPUS_DIR = Path(__file__).resolve().parents[2] / "bench" / "scenarios"


def _list_yaml_fixtures() -> list[Path]:
    if not CORPUS_DIR.exists():
        return []
    return sorted(p for p in CORPUS_DIR.iterdir() if p.suffix == ".yaml")


def test_corpus_loads_all_fixtures():
    """Every YAML under bench/scenarios/ must parse as ScenarioSpec without error."""
    fixtures = _list_yaml_fixtures()
    if not fixtures:
        pytest.skip("bench/scenarios/ is empty — corpus not yet authored")
    for path in fixtures:
        scen = load_scenario(path)
        assert isinstance(scen, ScenarioSpec)


FAILURE_MODE_FIXTURES = {
    "09_stuck_recovery.yaml": "stuck_recoveries",
    "11_deny_pattern_hit.yaml": "deny_events",
    "12_near_compaction.yaml": "compaction_triggered",
}


@pytest.mark.parametrize("filename,event_key", list(FAILURE_MODE_FIXTURES.items()))
def test_failure_mode_fixtures_use_event_counts(filename: str, event_key: str):
    """Failure-mode fixtures decide pass/fail via event_counts only — no LLM-text golden."""
    path = CORPUS_DIR / filename
    if not path.exists():
        pytest.skip(f"fixture {filename} not yet authored")
    scen = load_scenario(path)
    assert scen.success_criteria.event_counts, (
        f"{filename}: failure-mode fixture must define non-empty event_counts"
    )
    assert event_key in scen.success_criteria.event_counts, (
        f"{filename}: expected event_counts to reference {event_key!r}"
    )
    assert scen.success_criteria.golden_output is None, (
        f"{filename}: failure-mode fixtures must not pin golden_output "
        f"(LLM output is incidental; pass/fail is decided by events)"
    )


def test_scenario_class_coverage():
    """Wave 1 skeleton — assert the 3 failure-mode fixtures are present.
    Wave 2/3 will extend this to require all 12 canonical names.
    """
    fixtures = _list_yaml_fixtures()
    if not fixtures:
        pytest.skip("bench/scenarios/ is empty")
    names = {load_scenario(p).name for p in fixtures}
    assert "stuck_recovery" in names
    assert "deny_pattern_hit" in names
    assert "near_compaction" in names
