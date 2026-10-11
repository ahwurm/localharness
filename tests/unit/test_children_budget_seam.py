"""The five built-in child budgets live in `org.children_budget` and ride into every builder (08-06).

Task 1 pins the builders: each takes its entry as the keyword-only `budget=`, folds `kill_file` in,
never sets `max_tool_calls`, and with no budget builds on `ChildrenBudgetConfig().for_name(<its own
entry>)` so the builder alone owns the org rung (D-13, D-14).
"""
from __future__ import annotations

import inspect

import pytest

from localharness.agent import subagent as sub
from localharness.config.models import ChildBudget, ChildrenBudgetConfig

_BUILDERS = [
    ("explore", sub.build_explore_config, (8, 3.0)),
    ("web-researcher", sub.build_web_researcher_config, (56, 20.0)),
    ("search-verifier", sub.build_search_verifier_config, (12, 6.0)),
    ("cruncher", sub.build_cruncher_config, (6, 20.0)),
    ("chunk-summarizer", sub.build_chunk_summarizer_config, (3, 6.0)),
]


@pytest.mark.parametrize("name,builder,expected", _BUILDERS)
def test_a_builder_with_no_budget_builds_todays_numbers(name, builder, expected):
    budget = builder().permissions.budget
    assert (budget.max_actions, budget.max_duration_minutes) == expected
    assert budget.max_tool_calls is None
    assert budget.kill_file is None


@pytest.mark.parametrize("name,builder,expected", _BUILDERS)
def test_a_builder_with_no_budget_equals_the_schema_entry(name, builder, expected):
    entry = ChildrenBudgetConfig().for_name(name)
    budget = builder().permissions.budget
    assert (budget.max_actions, budget.max_duration_minutes) == (entry.actions, entry.minutes)


@pytest.mark.parametrize("name,builder,expected", _BUILDERS)
def test_a_given_budget_replaces_the_default_and_kill_file_still_folds_in(name, builder, expected):
    budget = builder(kill_file="/k", budget=ChildBudget(actions=5, minutes=2.0)).permissions.budget
    assert (budget.max_actions, budget.max_duration_minutes) == (5, 2.0)
    assert budget.kill_file == "/k"
    assert budget.max_tool_calls is None


@pytest.mark.parametrize("name,builder,expected", _BUILDERS)
def test_budget_is_keyword_only_with_default_none(name, builder, expected):
    param = inspect.signature(builder).parameters["budget"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is None


def test_a_renamed_build_keeps_its_builtin_entry():
    cfg = sub.build_web_researcher_config(name="web_researcher_2")
    assert cfg.permissions.budget.max_actions == ChildrenBudgetConfig().web_researcher.actions


def test_the_budget_constants_are_gone():
    gone = [n for n in vars(sub)
            if n.endswith(("_MAX_ACTIONS", "_MAX_DURATION_MINUTES", "_MAX_TOOL_CALLS"))]
    assert gone == []
