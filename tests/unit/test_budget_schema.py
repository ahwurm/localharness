"""The budget schema surfaces: the children block, the knob, the stamp, the context sizes, the
card, the gate property and the loop's reading of an unset action ceiling."""
from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from localharness.config.defaults import (
    CURRENT_DEFAULTS_REVISION,
    CURRENT_INIT_NULLS_REVISION,
)
from localharness.config.models import (
    AgentConfig,
    BudgetConfig,
    ChannelsConfig,
    ChildBudget,
    ChildrenBudgetConfig,
    OrgConfig,
    PermissionConfig,
)
from localharness.config.overlay import deep_merge


# --------------------------------------------------------------------------- BudgetConfig


def test_max_actions_accepts_none_and_keeps_its_bounds():
    assert BudgetConfig(max_actions=None).max_actions is None
    assert BudgetConfig(max_actions=1).max_actions == 1
    assert BudgetConfig(max_actions=10_000).max_actions == 10_000
    with pytest.raises(ValidationError):
        BudgetConfig(max_actions=0)
    with pytest.raises(ValidationError):
        BudgetConfig(max_actions=10_001)


# --------------------------------------------------------------------------- children block


def _pair(entry: ChildBudget) -> tuple[int, float]:
    return (entry.actions, entry.minutes)


def test_children_budget_defaults_are_the_locked_numbers():
    cb = OrgConfig().children_budget
    assert _pair(cb.explore) == (8, 3.0)
    assert _pair(cb.web_researcher) == (56, 20.0)
    assert _pair(cb.search_verifier) == (12, 6.0)
    assert _pair(cb.cruncher) == (6, 20.0)
    assert _pair(cb.chunk_summarizer) == (3, 6.0)
    assert _pair(cb.default) == (8, 3.0)
    assert cb.lookup_share == 0.25
    assert cb.review_share == 0.5


def test_children_budget_refuses_unknown_keys_and_out_of_range_values():
    with pytest.raises(ValidationError):
        ChildrenBudgetConfig(bogus=1)
    with pytest.raises(ValidationError):
        ChildBudget(actions=0, minutes=1.0)
    with pytest.raises(ValidationError):
        ChildBudget(actions=1, minutes=0)
    with pytest.raises(ValidationError):
        ChildrenBudgetConfig(lookup_share=0)
    with pytest.raises(ValidationError):
        ChildrenBudgetConfig(review_share=1.5)


@pytest.mark.parametrize("name,entry", [
    ("web-researcher", "web_researcher"),
    ("search-verifier", "search_verifier"),
    ("chunk-summarizer", "chunk_summarizer"),
    ("explore", "explore"),
    ("cruncher", "cruncher"),
    ("data-analyst", "default"),
    ("orchestrator", "default"),
])
def test_for_name_and_entry_name_pick_the_matching_entry(name, entry):
    cb = ChildrenBudgetConfig()
    assert cb.entry_name(name) == entry
    assert cb.for_name(name) is getattr(cb, entry)


@pytest.mark.parametrize("raw,entry,expected", [
    ({"explore": {"actions": 4}}, "explore", (4, 3.0)),
    ({"chunk_summarizer": {"minutes": 2.5}}, "chunk_summarizer", (3, 2.5)),
    ({"default": {"actions": 5}}, "default", (5, 3.0)),
    ({"web_researcher": {"actions": 30}}, "web_researcher", (30, 20.0)),
])
def test_a_one_key_entry_is_completed_from_that_entrys_own_default(raw, entry, expected):
    original = copy.deepcopy(raw)
    got = ChildrenBudgetConfig.model_validate(raw)
    assert _pair(getattr(got, entry)) == expected
    assert raw == original, "the caller's dict must not be mutated"


def test_a_one_key_entry_validates_through_the_org_block():
    org = OrgConfig.model_validate({"children_budget": {"web_researcher": {"actions": 30}}})
    assert _pair(org.children_budget.web_researcher) == (30, 20.0)


def test_a_sibling_another_layer_wrote_survives_the_merge():
    merged = deep_merge({"explore": {"minutes": 5.0}}, {"explore": {"actions": 4}})
    assert _pair(ChildrenBudgetConfig.model_validate(merged).explore) == (4, 5.0)


def test_child_budget_itself_stays_required():
    with pytest.raises(ValidationError):
        ChildBudget(actions=4)
    with pytest.raises(ValidationError):
        ChildBudget(minutes=4.0)


@pytest.mark.parametrize("raw", [
    {"explore": 5},
    {"explore": None},
    {"explore": {"actions": 0}},
    {"explore": {"minutes": 0}},
    {"explore": {"actions": 10_001}},
    {"explore": {"actions": 4, "bogus": 1}},
])
def test_a_bad_entry_is_still_refused(raw):
    with pytest.raises(ValidationError):
        ChildrenBudgetConfig.model_validate(raw)


def test_a_ready_made_entry_instance_passes_through():
    got = ChildrenBudgetConfig.model_validate({"explore": ChildBudget(actions=2, minutes=1.0)})
    assert _pair(got.explore) == (2, 1.0)


# --------------------------------------------------------------------------- knob and stamp


def test_unattended_max_actions_default_and_bounds():
    assert ChannelsConfig().unattended_max_actions == 100
    assert ChannelsConfig(unattended_max_actions=5000).unattended_max_actions == 5000
    with pytest.raises(ValidationError):
        ChannelsConfig(unattended_max_actions=0)


def test_unattended_max_actions_is_declared_below_remote_unattended():
    names = list(ChannelsConfig.model_fields)
    assert names.index("unattended_max_actions") == names.index("remote_unattended") + 1


def test_the_init_nulls_stamp_and_its_revision():
    assert PermissionConfig().init_nulls_revision == 0
    with pytest.raises(ValidationError):
        PermissionConfig(init_nulls_revision=-1)
    assert CURRENT_INIT_NULLS_REVISION == 1
    assert CURRENT_DEFAULTS_REVISION == 3


# --------------------------------------------------------------------------- workspace mark


def test_the_workspace_mark_defaults_empty_and_survives_a_deep_copy():
    cfg = AgentConfig(name="a", role="r")
    assert cfg.budget_kept_from_workspace == frozenset()
    cfg._budget_kept_from_workspace = frozenset({"max_actions"})
    assert cfg.model_copy(deep=True).budget_kept_from_workspace == frozenset({"max_actions"})


def test_the_workspace_mark_is_not_a_yaml_key():
    try:
        cfg = AgentConfig.model_validate(
            {"name": "a", "role": "r", "_budget_kept_from_workspace": ["max_actions"]})
    except ValidationError:
        return
    assert cfg.budget_kept_from_workspace == frozenset()
