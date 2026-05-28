"""SCEN-01: ScenarioSpec, LimitsSpec, SuccessCriteria Pydantic validation."""
from __future__ import annotations
from pathlib import Path
import textwrap
import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Phase 13 Wave 1: slice/category/tags additive schema fields
# ---------------------------------------------------------------------------


def _minimal_kwargs(**overrides):
    """Build kwargs for a minimal valid ScenarioSpec including new required fields."""
    from localharness.bench.schema import SuccessCriteria
    from localharness.core.events import BudgetSpec
    base = dict(
        name="t",
        prompt="x",
        success_criteria=SuccessCriteria(golden_output="x"),
        budget=BudgetSpec(),
        slice="train",
        category="tool_basics",
    )
    base.update(overrides)
    return base


@pytest.fixture
def _override_categories(monkeypatch, tmp_path):
    """Point _load_allowed_categories at a tmp categories file with only 'tool_basics'."""
    from localharness.bench import schema as schema_mod
    cats_path = tmp_path / "categories.yaml"
    cats_path.write_text(textwrap.dedent("""\
        categories:
          tool_basics: "test only"
    """))
    monkeypatch.setenv("LOCALHARNESS_CATEGORIES_PATH", str(cats_path))
    schema_mod._load_allowed_categories.cache_clear()
    yield cats_path
    schema_mod._load_allowed_categories.cache_clear()


def test_scenario_spec_rejects_missing_slice():
    """ScenarioSpec rejects YAML missing `slice` field with ValidationError."""
    from localharness.bench.schema import ScenarioSpec
    kw = _minimal_kwargs()
    kw.pop("slice")
    with pytest.raises(ValidationError):
        ScenarioSpec(**kw)


def test_scenario_spec_rejects_missing_category():
    """ScenarioSpec rejects YAML missing `category` field with ValidationError."""
    from localharness.bench.schema import ScenarioSpec
    kw = _minimal_kwargs()
    kw.pop("category")
    with pytest.raises(ValidationError):
        ScenarioSpec(**kw)


def test_scenario_spec_rejects_bad_slice_value():
    """ScenarioSpec rejects slice='training' (not in Literal allowed-set)."""
    from localharness.bench.schema import ScenarioSpec
    with pytest.raises(ValidationError):
        ScenarioSpec(**_minimal_kwargs(slice="training"))


def test_scenario_spec_rejects_unknown_category(_override_categories):
    """ScenarioSpec rejects category not in bench/categories.yaml allowed-set."""
    from localharness.bench.schema import ScenarioSpec
    with pytest.raises(ValidationError):
        ScenarioSpec(**_minimal_kwargs(category="made_up_name"))


def test_scenario_spec_accepts_valid_slice_category_tags():
    """ScenarioSpec accepts valid slice/category/tags combo."""
    from localharness.bench.schema import ScenarioSpec
    spec = ScenarioSpec(**_minimal_kwargs(tags=["uses_mcp"]))
    assert spec.slice == "train"
    assert spec.category == "tool_basics"
    assert spec.tags == ["uses_mcp"]


def test_scenario_spec_tags_defaults_to_empty_list():
    """ScenarioSpec.tags defaults to empty list when omitted."""
    from localharness.bench.schema import ScenarioSpec
    spec = ScenarioSpec(**_minimal_kwargs())
    assert spec.tags == []


def test_scenario_spec_valid(fixture_scenario_path: Path):
    """ScenarioSpec loads a valid YAML file via parse_yaml_raw_as."""
    from localharness.bench.schema import ScenarioSpec, load_scenario
    spec = load_scenario(fixture_scenario_path)
    assert spec.name == "minimal_golden"
    assert spec.prompt.startswith("What is 2 + 2")
    assert spec.success_criteria.golden_output == "4"
    assert spec.budget.max_actions == 5
    assert spec.limits.max_latency_s == 30.0
    assert spec.tools_allowed == []


def test_scenario_spec_rejects_invalid(fixture_invalid_scenario_path: Path):
    """ScenarioSpec rejects YAML missing required `prompt` field."""
    from localharness.bench.schema import load_scenario
    with pytest.raises(ValidationError):
        load_scenario(fixture_invalid_scenario_path)


def test_success_criteria_evaluate_golden():
    """SuccessCriteria.evaluate returns True iff golden_output matches stripped final_message."""
    from localharness.bench.schema import SuccessCriteria
    sc = SuccessCriteria(golden_output="4")
    assert sc.evaluate("4") is True
    assert sc.evaluate("  4  ") is True
    assert sc.evaluate("5") is False
    assert sc.evaluate("4 and more") is False


def test_success_criteria_evaluate_rubric():
    """SuccessCriteria.evaluate ANDs all rubric assertions. contains: and regex: prefixes supported."""
    from localharness.bench.schema import SuccessCriteria
    sc = SuccessCriteria(rubric=["contains:hello", "regex:[Ww]orld"])
    assert sc.evaluate("hello world") is True
    assert sc.evaluate("HELLO World") is False   # contains: is case-sensitive
    assert sc.evaluate("hello") is False           # missing world


def test_success_criteria_evaluate_both_anded():
    """When golden_output AND rubric both present, both must pass."""
    from localharness.bench.schema import SuccessCriteria
    sc = SuccessCriteria(golden_output="4", rubric=["contains:4"])
    assert sc.evaluate("4") is True
    assert sc.evaluate("five") is False


def test_success_criteria_requires_either():
    """SuccessCriteria with no golden_output and empty rubric must raise ValidationError."""
    from localharness.bench.schema import SuccessCriteria
    with pytest.raises(ValidationError):
        SuccessCriteria()


def test_tool_name_prefix_builtin_mcp_plugin():
    """Tool names in tools_allowed support bare (builtin), 'mcp:X', 'plugin:P.X' prefixes."""
    from localharness.bench.schema import parse_tool_name
    assert parse_tool_name("read_file") == ("builtin", "read_file", None)
    assert parse_tool_name("mcp:exa_search") == ("mcp", "exa_search", None)
    assert parse_tool_name("plugin:research_tools.exa_search") == ("plugin", "exa_search", "research_tools")


def test_scenario_spec_name_pattern():
    """name must match ^[a-z][a-z0-9_-]*$."""
    from localharness.bench.schema import ScenarioSpec
    from localharness.core.events import BudgetSpec
    from localharness.bench.schema import SuccessCriteria
    with pytest.raises(ValidationError):
        ScenarioSpec(
            name="BadName",  # uppercase not allowed
            prompt="x",
            success_criteria=SuccessCriteria(golden_output="x"),
            budget=BudgetSpec(),
        )


def test_limits_spec_defaults():
    """LimitsSpec has max_latency_s=300.0 and max_tool_calls=200 defaults."""
    from localharness.bench.schema import LimitsSpec
    lim = LimitsSpec()
    assert lim.max_latency_s == 300.0
    assert lim.max_tool_calls == 200


def test_limits_spec_validation():
    """max_latency_s must be >0; max_tool_calls must be >=0."""
    from localharness.bench.schema import LimitsSpec
    with pytest.raises(ValidationError):
        LimitsSpec(max_latency_s=0)
    with pytest.raises(ValidationError):
        LimitsSpec(max_tool_calls=-1)


# ---------------------------------------------------------------------------
# SCEN-04 plumbing: SuccessCriteria.event_counts (Plan 12-01 Task 1)
# ---------------------------------------------------------------------------

def test_event_counts_operators():
    """event_counts={'deny_events': {'min': 1}} requires observed >= 1."""
    from localharness.bench.schema import SuccessCriteria
    sc = SuccessCriteria(event_counts={"deny_events": {"min": 1}})
    assert sc.evaluate("any", counts={"deny_events": 1}) is True
    assert sc.evaluate("any", counts={"deny_events": 0}) is False
    assert sc.evaluate("any", counts={"deny_events": 5}) is True


def test_event_counts_operators_max():
    """event_counts={'parse_failures': {'max': 0}} requires observed <= 0."""
    from localharness.bench.schema import SuccessCriteria
    sc = SuccessCriteria(event_counts={"parse_failures": {"max": 0}})
    assert sc.evaluate("x", counts={"parse_failures": 0}) is True
    assert sc.evaluate("x", counts={"parse_failures": 1}) is False


def test_event_counts_operators_exact():
    """event_counts={'tool_call_count': {'exact': 2}} requires observed == 2."""
    from localharness.bench.schema import SuccessCriteria
    sc = SuccessCriteria(event_counts={"tool_call_count": {"exact": 2}})
    assert sc.evaluate("x", counts={"tool_call_count": 2}) is True
    assert sc.evaluate("x", counts={"tool_call_count": 3}) is False
    assert sc.evaluate("x", counts={"tool_call_count": 1}) is False


def test_evaluate_combines_all_dimensions():
    """golden_output AND rubric AND event_counts must all pass."""
    from localharness.bench.schema import SuccessCriteria
    sc = SuccessCriteria(
        golden_output="hello hi",
        rubric=["contains:hi"],
        event_counts={"deny_events": {"min": 1}},
    )
    assert sc.evaluate("hello hi", counts={"deny_events": 1}) is True
    assert sc.evaluate("hello hi", counts={"deny_events": 0}) is False
    assert sc.evaluate("bye hi", counts={"deny_events": 1}) is False  # golden mismatch
    # Rubric mismatch — change rubric to a missing token
    sc2 = SuccessCriteria(
        golden_output="hello",
        rubric=["contains:xyzzy"],
        event_counts={"deny_events": {"min": 1}},
    )
    assert sc2.evaluate("hello", counts={"deny_events": 1}) is False


def test_event_counts_alone_satisfies_at_least_one():
    """event_counts alone (no golden_output, no rubric) must construct successfully."""
    from localharness.bench.schema import SuccessCriteria
    sc = SuccessCriteria(event_counts={"stuck_recoveries": {"min": 1}})
    assert sc.event_counts == {"stuck_recoveries": {"min": 1}}


def test_event_counts_missing_key_treated_as_zero():
    """A missing key in counts dict is treated as observed=0."""
    from localharness.bench.schema import SuccessCriteria
    sc = SuccessCriteria(event_counts={"deny_events": {"min": 1}})
    assert sc.evaluate("x", counts={}) is False


def test_event_counts_none_supplied_fails_when_asserted():
    """If event_counts is non-empty but no counts arg is supplied, evaluation fails."""
    from localharness.bench.schema import SuccessCriteria
    sc = SuccessCriteria(event_counts={"deny_events": {"min": 1}})
    assert sc.evaluate("x") is False


def test_evaluate_signature_back_compat():
    """Legacy single-arg evaluate() calls still work when event_counts is unset."""
    from localharness.bench.schema import SuccessCriteria
    sc = SuccessCriteria(rubric=["contains:hi"])
    assert sc.evaluate("hi there") is True
    assert sc.evaluate("nope") is False


# ---------------------------------------------------------------------------
# Wave 1 corpus / SCEN-03: every committed fixture parses as ScenarioSpec
# (schema-side mirror of test_bench_corpus.py::test_corpus_loads_all_fixtures —
# VALIDATION.md maps this test name under both modules)
# ---------------------------------------------------------------------------

def test_corpus_loads_all_fixtures():
    """Schema-side check: every YAML in bench/scenarios/ parses as ScenarioSpec.

    Same logic as test_bench_corpus.py::test_corpus_loads_all_fixtures —
    duplicated here because VALIDATION.md maps this test name under both
    modules. test_bench_corpus.py owns the broader corpus-shape suite;
    this test_bench_schema.py copy guards against schema-level regression
    when corpus content evolves.
    """
    from localharness.bench.schema import load_scenario

    corpus_dir = Path(__file__).resolve().parents[2] / "bench" / "scenarios"
    if not corpus_dir.exists():
        pytest.skip("bench/scenarios/ does not exist yet")
    # rglob covers post-Phase-13 train/ + holdout/ subdir layout
    fixtures = sorted(p for p in corpus_dir.rglob("*.yaml") if p.is_file())
    if not fixtures:
        pytest.skip("bench/scenarios/ is empty")
    for path in fixtures:
        load_scenario(path)   # raises ValidationError on bad input
