"""The budget schema surfaces: the children block, the knob, the stamp, the context sizes, the
card, the gate property and the loop's reading of an unset action ceiling."""
from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from localharness.agent.gate import fail_closed_gate
from localharness.config.defaults import (
    CURRENT_DEFAULTS_REVISION,
    CURRENT_INIT_NULLS_REVISION,
    DEFAULT_TOOL_EVICT_THRESHOLD_CHARS,
    DEFAULT_TOOL_OUTPUT_CHARS,
)
from localharness.config.models import (
    AgentConfig,
    BudgetConfig,
    ChannelsConfig,
    ChildBudget,
    ChildrenBudgetConfig,
    ContextConfig,
    OrgConfig,
    PermissionConfig,
)
from localharness.orchestrator.cards import AgentCardRegistry
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


# --------------------------------------------------------------------------- context sizes


def test_context_sizes_default_to_none_and_keep_their_bounds():
    ctx = ContextConfig()
    assert ctx.max_tool_output_chars is None
    assert ctx.tool_result_evict_threshold_chars is None
    with pytest.raises(ValidationError):
        ContextConfig(max_tool_output_chars=99)
    with pytest.raises(ValidationError):
        ContextConfig(tool_result_evict_threshold_chars=499)
    assert ContextConfig(max_tool_output_chars=100).max_tool_output_chars == 100


def test_the_context_size_constants_are_todays_numbers():
    assert DEFAULT_TOOL_OUTPUT_CHARS == 32_000
    assert DEFAULT_TOOL_EVICT_THRESHOLD_CHARS == 8_000


@pytest.mark.asyncio
async def test_the_bench_pipeline_caps_tool_output_at_todays_size_when_unset():
    from localharness.bench import runner as bench_runner
    from localharness.bench.schema import LimitsSpec, ScenarioSpec, SuccessCriteria
    from localharness.core.events import BudgetSpec

    scen = ScenarioSpec(
        name="cap-default", prompt="x",
        success_criteria=SuccessCriteria(rubric=["contains:X"]),
        budget=BudgetSpec(), limits=LimitsSpec(),
        tools_allowed=[], slice="train", category="tool_basics",
    )
    loop = await bench_runner._build_agent_loop(bus=None, llm_client=None, scenario=scen)
    cap_stage = loop._ctx._pipeline._deterministic_stages[0]
    assert cap_stage.max_chars == DEFAULT_TOOL_OUTPUT_CHARS == 32_000


# --------------------------------------------------------------------------- the card


def _no_limit_config() -> AgentConfig:
    return AgentConfig(name="n", role="r", permissions={"budget": {"max_actions": None}})


def test_a_card_keeps_an_unset_action_limit():
    card = AgentCardRegistry().generate_card(_no_limit_config())
    assert card.budget_max_actions is None


def test_an_agent_without_an_action_limit_registers():
    registry = AgentCardRegistry()
    card = registry.register_from_config(_no_limit_config())
    assert registry.get("n") is card
    assert registry.get("n").budget_max_actions is None


# --------------------------------------------------------------------------- the gate property


class _Channel:
    def __init__(self, local_operator=None):
        if local_operator is not None:
            self.local_operator = local_operator


def test_the_gate_exposes_the_attached_channels_operator_flag(tmp_path):
    gate = fail_closed_gate(permissions=PermissionConfig(), workspace=tmp_path)
    assert gate.local_operator is False, "a fail-closed gate has no channel"
    gate.attach_channel(_Channel(local_operator=True))
    assert gate.local_operator is True
    gate.attach_channel(_Channel())
    assert gate.local_operator is False, "a channel without the attribute is not local"


def test_the_gate_operator_flag_is_read_only(tmp_path):
    gate = fail_closed_gate(permissions=PermissionConfig(), workspace=tmp_path)
    with pytest.raises(AttributeError):
        gate.local_operator = True


# --------------------------------------------------------------------------- the loop


@pytest.mark.asyncio
async def test_a_loop_with_no_action_limit_runs_a_turn_unbounded(faithful_fake_llm, bus, tmp_path):
    from localharness.agent.context import ContextManager
    from localharness.agent.loop import AgentLoop
    from localharness.agent.permissions import PermissionEvaluator
    from localharness.core.events import TurnFailed, TurnStarted
    from localharness.tools import ToolRegistry
    from localharness.tools.builtin import register_builtin_tools

    reg = ToolRegistry()
    await register_builtin_tools(reg)
    cfg = AgentConfig.model_validate({
        "name": "no-limit-agent",
        "role": "Test agent.",
        "tools": {"deny": ["web_search", "web_fetch", "web_page_query"]},
        "permissions": {"mode": "unattended",
                        "budget": {"max_actions": None, "kill_file": str(tmp_path / "KILL")}},
    })
    loop = AgentLoop(
        config=cfg, llm=faithful_fake_llm(tool_plan=[("glob", {"pattern": "*.py"})]),
        bus=bus, context_manager=ContextManager(), tool_registry=reg,
        permission_evaluator=PermissionEvaluator(),
    )
    await loop.run_turn("list python files")

    started = bus.history(event_types=[TurnStarted])
    assert len(started) == 1
    assert started[0].budget.max_actions == 0
    assert not bus.history(event_types=[TurnFailed])
