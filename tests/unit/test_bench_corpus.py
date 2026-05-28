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
    # rglob covers post-Phase-13 train/ + holdout/ subdir layout
    return sorted(p for p in CORPUS_DIR.rglob("*.yaml") if p.is_file())


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
    # rglob covers post-Phase-13 train/ + holdout/ subdir layout
    matches = [p for p in CORPUS_DIR.rglob(filename) if p.is_file()]
    if not matches:
        pytest.skip(f"fixture {filename} not yet authored")
    path = matches[0]
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
    """Assert the corpus contains all 12 canonical scenario names from SCEN-03."""
    fixtures = _list_yaml_fixtures()
    if not fixtures:
        pytest.skip("bench/scenarios/ is empty")
    names = {load_scenario(p).name for p in fixtures}
    all_12 = {
        "pure_qa", "single_read", "write_execute", "fibonacci_sort",
        "file_exploration", "agent_creation", "brave_search_subagent",
        "plugin_mcp_tool", "stuck_recovery", "memory_recall",
        "deny_pattern_hit", "near_compaction",
    }
    missing = all_12 - names
    extra = names - all_12
    assert not missing, f"corpus missing canonical fixtures: {missing}"
    # Extra fixtures are permitted but flagged for sanity
    if extra:
        print(f"NOTE: corpus has extra fixtures beyond canonical 12: {extra}")


# ---------------------------------------------------------------------------
# Plan 12-04 Task 4: known-good + known-bad evaluate() coverage
# ---------------------------------------------------------------------------

# Known-good final messages + counts per fixture. Each entry is what a
# successful run looks like — used to verify success_criteria.evaluate()
# returns True. This is parse-time + evaluate-time coverage; live LLM
# behavior is validated manually per VALIDATION.md "Manual-Only Verifications".
KNOWN_GOOD: dict[str, tuple[str, dict[str, int]]] = {
    # rubric fixtures — final_message satisfies the regex/contains anchor
    "pure_qa": ("The answer is 42.", {}),
    "single_read": ("The file mentions an apricot.", {}),
    "write_execute": ("Script output was: HELLO_BENCH_OK", {}),
    "fibonacci_sort": ("Sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34", {}),
    "file_exploration": ("Found MAGIC_VALUE_777 inside values.txt", {}),
    "agent_creation": ("Subagent returned: STUB_SUBAGENT_OK ...", {}),
    "brave_search_subagent": ("Subagent reported STUB_SUBAGENT_OK summary", {}),
    "plugin_mcp_tool": ("Looked it up - Internet Engineering Task Force", {}),
    "memory_recall": ("The codename was STARFRUIT_42.", {}),
    # event-count fixtures — final_message can be anything; counts satisfy assertions
    "stuck_recovery": ("Gave up.", {"stuck_recoveries": 1}),
    "deny_pattern_hit": ("Operation denied.", {"deny_events": 1}),
    "near_compaction": ("Summary: ...", {"compaction_triggered": 1}),
}


@pytest.mark.parametrize("fixture_name", sorted(KNOWN_GOOD.keys()))
def test_known_good_pass(fixture_name: str):
    """Each fixture's success_criteria.evaluate(known_good_msg, counts=...) returns True."""
    fixtures = {load_scenario(p).name: load_scenario(p) for p in _list_yaml_fixtures()}
    if fixture_name not in fixtures:
        pytest.skip(f"fixture {fixture_name} not on disk")
    scen = fixtures[fixture_name]
    msg, counts = KNOWN_GOOD[fixture_name]
    assert scen.success_criteria.evaluate(msg, counts=counts) is True, (
        f"{fixture_name}: evaluate returned False — rubric anchor or "
        f"event_counts assertion does not match the hand-curated known-good case"
    )


# Known-BAD final messages + counts per fixture. Each entry is a deliberately
# mismatched input that MUST make success_criteria.evaluate(...) return False.
# This proves "broken harness = clean assertion mismatch" — ROADMAP Phase 12 SC4.
KNOWN_BAD: dict[str, tuple[str, dict[str, int]]] = {
    # rubric fixtures — final_message fails the rubric anchor
    "pure_qa": ("The answer is 41.", {}),                          # rubric expects 42
    "single_read": ("The file mentions a banana.", {}),            # rubric expects apricot
    "write_execute": ("Script output was: WRONG_TOKEN", {}),       # rubric expects HELLO_BENCH_OK
    "fibonacci_sort": ("Sequence: 1, 2, 3, 4, 5", {}),             # rubric expects fibonacci
    "file_exploration": ("Found NOTHING in values.txt", {}),       # rubric expects MAGIC_VALUE_777
    "agent_creation": ("Subagent failed silently", {}),            # rubric expects STUB_SUBAGENT_OK
    "brave_search_subagent": ("Search returned no results", {}),   # rubric expects STUB_SUBAGENT_OK
    "plugin_mcp_tool": ("Could not look it up", {}),               # rubric expects "Internet Engineering Task Force"
    "memory_recall": ("I do not remember.", {}),                   # rubric expects STARFRUIT_42
    # event-count fixtures — counts fail the assertion (empty counts when min: 1 asserted)
    "stuck_recovery": ("Gave up.", {}),                            # asserts stuck_recoveries min: 1
    "deny_pattern_hit": ("Operation denied.", {}),                 # asserts deny_events min: 1
    "near_compaction": ("Summary: ...", {}),                       # asserts compaction_triggered min: 1
}


@pytest.mark.parametrize("fixture_name", sorted(KNOWN_BAD.keys()))
def test_known_bad_fail(fixture_name: str):
    """Each fixture's success_criteria.evaluate(known_bad_msg, counts=...) returns False.

    ROADMAP Phase 12 SC4: "User intentionally breaking the harness sees the
    affected scenarios fail with a clear assertion mismatch rather than a hang
    or false pass." This test proves the broken-harness contract end-to-end at
    the SuccessCriteria layer — for every fixture, a deliberately mismatched
    input makes evaluate() return False (no hang, no exception, no false pass).
    """
    fixtures = {load_scenario(p).name: load_scenario(p) for p in _list_yaml_fixtures()}
    if fixture_name not in fixtures:
        pytest.skip(f"fixture {fixture_name} not on disk")
    scen = fixtures[fixture_name]
    msg, counts = KNOWN_BAD[fixture_name]
    assert scen.success_criteria.evaluate(msg, counts=counts) is False, (
        f"{fixture_name}: evaluate returned True on a deliberately mismatched "
        f"input — rubric anchor or event_counts assertion is too permissive "
        f"(broken harness would falsely report success)"
    )
