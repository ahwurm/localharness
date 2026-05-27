"""SCEN-01: ScenarioSpec, LimitsSpec, SuccessCriteria Pydantic validation."""
from __future__ import annotations
from pathlib import Path
import pytest
from pydantic import ValidationError


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
