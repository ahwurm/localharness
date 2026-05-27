"""SCEN-01: Scenario YAML schema — Pydantic models for ScenarioSpec, LimitsSpec, SuccessCriteria.

Reuses core.events.BudgetSpec verbatim. Tool-name namespacing uses source prefixes:
builtin = bare name, mcp:NAME, plugin:PLUGIN.NAME (locked in 11-CONTEXT.md).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_yaml import parse_yaml_raw_as

from localharness.core.events import BudgetSpec


# -------------------------------------------------------------------------
# Rubric matcher: supports "contains:TEXT", "regex:PATTERN" prefixes
# -------------------------------------------------------------------------

def _match_rubric(assertion: str, text: str) -> bool:
    """Evaluate one rubric assertion against a text. Case-sensitive 'contains' and full regex."""
    if assertion.startswith("contains:"):
        needle = assertion[len("contains:"):]
        return needle in text
    if assertion.startswith("regex:"):
        pattern = assertion[len("regex:"):]
        return re.search(pattern, text) is not None
    # Bare strings treated as case-sensitive substring
    return assertion in text


# -------------------------------------------------------------------------
# Tool name parsing (locked source-prefix convention)
# -------------------------------------------------------------------------

def parse_tool_name(s: str) -> tuple[Literal["builtin", "mcp", "plugin"], str, Optional[str]]:
    """Parse tools_allowed entry into (source, tool_name, plugin_name).

    - 'read_file' -> ('builtin', 'read_file', None)
    - 'mcp:exa_search' -> ('mcp', 'exa_search', None)
    - 'plugin:research_tools.exa_search' -> ('plugin', 'exa_search', 'research_tools')
    """
    if s.startswith("mcp:"):
        return ("mcp", s[len("mcp:"):], None)
    if s.startswith("plugin:"):
        rest = s[len("plugin:"):]
        if "." not in rest:
            raise ValueError(f"plugin tool name must be 'plugin:PLUGIN_NAME.TOOL_NAME', got {s!r}")
        plugin_name, _, tool_name = rest.partition(".")
        return ("plugin", tool_name, plugin_name)
    return ("builtin", s, None)


# -------------------------------------------------------------------------
# LimitsSpec — bench-only hard fail-fast ceilings (distinct from BudgetSpec)
# -------------------------------------------------------------------------

class LimitsSpec(BaseModel):
    """Bench-only hard fail-fast ceilings, distinct from BudgetSpec soft loop budget."""

    model_config = ConfigDict(frozen=True)
    max_latency_s: float = Field(gt=0, default=300.0)
    max_tool_calls: int = Field(ge=0, default=200)


# -------------------------------------------------------------------------
# SuccessCriteria — golden_output + rubric (AND if both set)
# -------------------------------------------------------------------------

class SuccessCriteria(BaseModel):
    """golden_output (exact-text-match) AND/OR rubric (list of assertions) AND/OR event_counts.

    All configured dimensions are ANDed in evaluate(). event_counts maps event-count names
    (e.g., 'deny_events', 'stuck_recoveries', 'compaction_triggered', 'tool_call_count',
    'parse_failures') to operator dicts with keys 'min', 'max', 'exact'.
    """

    model_config = ConfigDict(frozen=True)
    golden_output: Optional[str] = None
    rubric: list[str] = Field(default_factory=list)
    event_counts: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description=(
            "Per-event-type assertions. Keys are event-count names "
            "(e.g., 'deny_events', 'stuck_recoveries', 'compaction_triggered', "
            "'tool_call_count', 'parse_failures'). Values are operator dicts "
            "with keys 'min', 'max', 'exact'. ANDed with golden_output and rubric."
        ),
    )

    @model_validator(mode="after")
    def at_least_one(self) -> "SuccessCriteria":
        if self.golden_output is None and not self.rubric and not self.event_counts:
            raise ValueError(
                "success_criteria must have golden_output, non-empty rubric, or non-empty event_counts"
            )
        return self

    def evaluate(self, final_message: str, counts: dict[str, int] | None = None) -> bool:
        """Return True iff all configured assertions match. counts maps event-count
        name (e.g., 'deny_events') to observed integer count from MetricAccumulator."""
        if self.golden_output is not None:
            if final_message.strip() != self.golden_output.strip():
                return False
        for assertion in self.rubric:
            if not _match_rubric(assertion, final_message):
                return False
        if self.event_counts:
            if counts is None:
                return False
            for key, ops in self.event_counts.items():
                observed = counts.get(key, 0)
                if "exact" in ops and observed != ops["exact"]:
                    return False
                if "min" in ops and observed < ops["min"]:
                    return False
                if "max" in ops and observed > ops["max"]:
                    return False
        return True


# -------------------------------------------------------------------------
# ScenarioSpec — top-level scenario YAML model
# -------------------------------------------------------------------------

class ScenarioSpec(BaseModel):
    """One scenario YAML fixture. Reused BudgetSpec for the soft loop budget."""

    model_config = ConfigDict(frozen=True)
    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    prompt: str
    expected_outcome: str = ""
    success_criteria: SuccessCriteria
    budget: BudgetSpec
    limits: LimitsSpec = Field(default_factory=LimitsSpec)
    tools_allowed: list[str] = Field(
        default_factory=list,
        description=(
            "Whitelist. Empty list = no tools allowed (pure-LLM scenario), not 'all allowed'. "
            "Tool names: builtin = bare ('bash'), MCP = 'mcp:tool_name', "
            "plugin = 'plugin:plugin_name.tool_name'."
        ),
    )
    tolerance: float = 0.10
    min_runs: int = 3
    max_runs: int = 20


# -------------------------------------------------------------------------
# Loader
# -------------------------------------------------------------------------

def load_scenario(path: Path) -> ScenarioSpec:
    """Load + validate a scenario YAML file. Raises pydantic.ValidationError on bad input."""
    path = Path(path)
    return parse_yaml_raw_as(ScenarioSpec, path.read_text(encoding="utf-8"))
