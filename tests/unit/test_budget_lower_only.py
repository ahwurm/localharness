"""`channels.unattended_max_actions` is machine-level only, and `org.children_budget` and the
shares are lower-only on a workspace's raw config sources.

A cloned repository's `.localharness/` loads with no prompt, so it may make a child's ceiling or
share smaller and never larger; the ceiling a remote unattended session runs under is the
machine's alone. Narrowing happens on the raw sources before they merge, where a key left out and
a `null` can still be told apart."""
from __future__ import annotations

import math

import pytest

from localharness.config import loader as L
from localharness.config.models import ChildrenBudgetConfig
from tests.unit.test_docs_convergence import machine_only_code_set
from tests.unit.test_floor_machine_level import _layers, _warnings

KNOB = "channels.unattended_max_actions"
ENTRIES = ("explore", "web_researcher", "search_verifier", "cruncher", "chunk_summarizer", "default")


def _knob(value) -> dict:
    return {"channels": {"unattended_max_actions": value}}


def _budget(**block) -> dict:
    return {"org": {"children_budget": block}}


def _only_lower(file, path: str) -> str:
    return f"ignoring {path} in {file}: a project may only lower it"


# ---------------------------------------------------------------------------
# channels.unattended_max_actions: machine-level only
# ---------------------------------------------------------------------------


def test_the_knob_is_a_machine_level_only_member():
    assert KNOB in L.HARNESS_GLOBAL_ONLY_FIELDS
    assert KNOB in machine_only_code_set()


def test_a_workspace_config_cannot_raise_the_knob(tmp_path):
    loader, ws = _layers(tmp_path, ws_cfg=_knob(5000))
    assert loader.load_harness().channels.unattended_max_actions == 100
    assert _warnings(loader) == [
        f"ignoring {KNOB} in {ws / 'config.yaml'}: only the global config may set it"]


def test_a_workspace_overrides_file_cannot_raise_the_knob(tmp_path):
    loader, ws = _layers(tmp_path, ws_over=_knob(5000))
    assert loader.load_harness().channels.unattended_max_actions == 100
    assert _warnings(loader) == [
        f"ignoring {KNOB} in {ws / 'overrides.yaml'}: only the global config may set it"]


def test_a_workspace_cannot_lower_the_knob_either(tmp_path):
    # Machine-only means dropped, never narrowed: the global 40 stands over a workspace 10.
    loader, ws = _layers(tmp_path, g_cfg=_knob(40), ws_cfg=_knob(10))
    assert loader.load_harness().channels.unattended_max_actions == 40
    assert _warnings(loader) == [
        f"ignoring {KNOB} in {ws / 'config.yaml'}: only the global config may set it"]


def test_the_global_layers_set_the_knob_freely(tmp_path):
    loader, _ = _layers(tmp_path, g_cfg=_knob(40), g_over=_knob(7))
    assert loader.load_harness().channels.unattended_max_actions == 7
    assert _warnings(loader) == []


def test_a_workspace_restating_the_default_draws_no_warning(tmp_path):
    loader, _ = _layers(tmp_path, ws_cfg=_knob(100))
    assert loader.load_harness().channels.unattended_max_actions == 100
    assert _warnings(loader) == []


# ---------------------------------------------------------------------------
# org.children_budget entries: lower-only
# ---------------------------------------------------------------------------


def test_the_lower_only_set_names_every_entry_and_both_shares():
    # `default` is listed on purpose, not found by accident.
    assert set(ENTRIES) == {n for n, f in ChildrenBudgetConfig.model_fields.items()
                            if f.annotation.__name__ == "ChildBudget"}
    want = {f"org.children_budget.{e}.{k}" for e in ENTRIES for k in ("actions", "minutes")}
    want |= {"org.children_budget.lookup_share", "org.children_budget.review_share"}
    assert L.HARNESS_LOWER_ONLY_FIELDS == want
    assert not L.HARNESS_LOWER_ONLY_FIELDS & L.HARNESS_GLOBAL_ONLY_FIELDS


def _explore(loader):
    b = loader.load_harness().org.children_budget.explore
    return b.actions, b.minutes


def test_a_lower_value_is_taken_and_the_rest_of_the_entry_is_completed(tmp_path):
    loader, _ = _layers(tmp_path, ws_cfg=_budget(explore={"actions": 4}))
    assert _explore(loader) == (4, 3.0)
    assert _warnings(loader) == []


def test_a_higher_value_is_restored_with_a_note(tmp_path):
    loader, ws = _layers(tmp_path, ws_cfg=_budget(explore={"actions": 99}))
    assert _explore(loader) == (8, 3.0)
    assert _warnings(loader) == [_only_lower(ws / "config.yaml", "org.children_budget.explore.actions")]


def test_a_null_is_restored_silently(tmp_path):
    loader, _ = _layers(tmp_path, ws_cfg=_budget(explore={"actions": None}))
    assert _explore(loader) == (8, 3.0)
    assert _warnings(loader) == []


@pytest.mark.parametrize("bad", ["lots", True, False, 2.5, [3], {"a": 1}, 0, -3, 10_001])
def test_a_non_finite_or_out_of_range_actions_value_is_restored_with_a_note(tmp_path, bad):
    loader, ws = _layers(tmp_path, ws_cfg=_budget(explore={"actions": bad}))
    assert _explore(loader) == (8, 3.0)
    assert _warnings(loader) == [_only_lower(ws / "config.yaml", "org.children_budget.explore.actions")]


@pytest.mark.parametrize("bad", ["lots", True, math.nan, math.inf, 0, -1.0, 1441.0])
def test_a_non_finite_or_out_of_range_minutes_value_is_restored_with_a_note(tmp_path, bad):
    loader, ws = _layers(tmp_path, ws_cfg=_budget(explore={"minutes": bad}))
    assert _explore(loader) == (8, 3.0)
    assert _warnings(loader) == [_only_lower(ws / "config.yaml", "org.children_budget.explore.minutes")]


def test_a_global_ceiling_is_the_baseline(tmp_path):
    loader, ws = _layers(tmp_path, g_cfg=_budget(explore={"actions": 20}),
                         ws_cfg=_budget(explore={"actions": 25}))
    assert _explore(loader) == (20, 3.0)
    assert _warnings(loader) == [_only_lower(ws / "config.yaml", "org.children_budget.explore.actions")]
    loader, _ = _layers(tmp_path / "b", g_cfg=_budget(explore={"actions": 20}),
                        ws_cfg=_budget(explore={"actions": 15}))
    assert _explore(loader) == (15, 3.0)
    assert _warnings(loader) == []


def test_the_global_layers_set_a_ceiling_freely(tmp_path):
    loader, _ = _layers(tmp_path, g_cfg=_budget(explore={"actions": 500, "minutes": 90.0}))
    assert _explore(loader) == (500, 90.0)
    assert _warnings(loader) == []


def test_a_global_sibling_survives_the_walk_and_the_merge(tmp_path):
    loader, _ = _layers(tmp_path, g_cfg=_budget(explore={"minutes": 5.0}),
                        ws_cfg=_budget(explore={"actions": 4}))
    assert _explore(loader) == (4, 5.0)
    assert _warnings(loader) == []


def test_a_rejected_workspace_value_leaves_a_global_sibling_alone(tmp_path):
    loader, ws = _layers(tmp_path, g_cfg=_budget(explore={"minutes": 5.0}),
                         ws_cfg=_budget(explore={"actions": 99}))
    assert _explore(loader) == (8, 5.0)
    assert _warnings(loader) == [_only_lower(ws / "config.yaml", "org.children_budget.explore.actions")]


def test_default_is_walked(tmp_path):
    loader, ws = _layers(tmp_path, ws_cfg=_budget(default={"minutes": 9.0}))
    cfg = loader.load_harness().org.children_budget.default
    assert (cfg.actions, cfg.minutes) == (8, 3.0)
    assert _warnings(loader) == [_only_lower(ws / "config.yaml", "org.children_budget.default.minutes")]


def test_a_one_key_entry_is_completed_from_that_entrys_default(tmp_path):
    loader, _ = _layers(tmp_path, ws_cfg=_budget(chunk_summarizer={"minutes": 2.5}))
    b = loader.load_harness().org.children_budget.chunk_summarizer
    assert (b.actions, b.minutes) == (3, 2.5)
    assert _warnings(loader) == []


@pytest.mark.parametrize("entry", ENTRIES)
def test_the_baseline_under_a_silent_global_is_the_entrys_own_default(tmp_path, entry):
    # Fails if the baseline were "unset": 10_000 would then pass for any entry.
    want = getattr(ChildrenBudgetConfig(), entry)
    loader, ws = _layers(tmp_path, ws_cfg=_budget(**{entry: {"actions": 10_000, "minutes": 1440.0}}))
    got = getattr(loader.load_harness().org.children_budget, entry)
    assert (got.actions, got.minutes) == (want.actions, want.minutes)
    assert _warnings(loader) == [
        _only_lower(ws / "config.yaml", f"org.children_budget.{entry}.actions"),
        _only_lower(ws / "config.yaml", f"org.children_budget.{entry}.minutes")]


def test_a_workspace_overrides_file_is_judged_the_same_way(tmp_path):
    loader, ws = _layers(tmp_path, ws_over=_budget(explore={"actions": 99}))
    assert _explore(loader) == (8, 3.0)
    assert _warnings(loader) == [_only_lower(ws / "overrides.yaml", "org.children_budget.explore.actions")]


def test_both_workspace_files_are_each_judged(tmp_path):
    loader, ws = _layers(tmp_path, ws_cfg=_budget(explore={"actions": 99}),
                         ws_over=_budget(explore={"actions": 98}))
    assert _explore(loader) == (8, 3.0)
    assert _warnings(loader) == [
        _only_lower(ws / "config.yaml", "org.children_budget.explore.actions"),
        _only_lower(ws / "overrides.yaml", "org.children_budget.explore.actions")]


def test_a_workspace_less_session_and_the_global_layers_set_any_value(tmp_path):
    from localharness.config.loader import ConfigLoader
    from tests.unit.test_floor_machine_level import PROVIDER, _write
    g = tmp_path / "global"
    _write(g / "config.yaml", {"version": "1", "provider": PROVIDER,
                               **_budget(explore={"actions": 9000}, lookup_share=0.9)})
    cfg = ConfigLoader(config_dir=g).load_harness()
    assert cfg.org.children_budget.explore.actions == 9000
    assert cfg.org.children_budget.lookup_share == 0.9


# ---------------------------------------------------------------------------
# The shares
# ---------------------------------------------------------------------------


def _shares(loader):
    b = loader.load_harness().org.children_budget
    return b.lookup_share, b.review_share


def test_a_lower_share_is_taken(tmp_path):
    loader, _ = _layers(tmp_path, ws_cfg=_budget(review_share=0.2))
    assert _shares(loader) == (0.25, 0.2)
    assert _warnings(loader) == []


@pytest.mark.parametrize("bad", [0.9, 1, 0, -0.1, math.nan, math.inf, "half", True])
def test_a_higher_or_invalid_share_is_restored_with_a_note(tmp_path, bad):
    loader, ws = _layers(tmp_path, ws_cfg=_budget(review_share=bad))
    assert _shares(loader) == (0.25, 0.5)
    assert _warnings(loader) == [_only_lower(ws / "config.yaml", "org.children_budget.review_share")]


def test_a_null_share_is_restored_silently(tmp_path):
    loader, _ = _layers(tmp_path, ws_cfg=_budget(lookup_share=None))
    assert _shares(loader) == (0.25, 0.5)
    assert _warnings(loader) == []


def test_a_global_share_is_the_baseline(tmp_path):
    loader, ws = _layers(tmp_path, g_cfg=_budget(lookup_share=0.4), ws_cfg=_budget(lookup_share=0.45))
    assert _shares(loader) == (0.4, 0.5)
    assert _warnings(loader) == [_only_lower(ws / "config.yaml", "org.children_budget.lookup_share")]


# ---------------------------------------------------------------------------
# The section guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [5, "x", None, [1]])
def test_a_workspace_block_that_is_not_a_mapping_gives_way(tmp_path, bad):
    loader, ws = _layers(tmp_path, ws_cfg={"org": {"children_budget": bad}})
    cfg = loader.load_harness()
    assert cfg.org.children_budget == ChildrenBudgetConfig()
    assert _warnings(loader) == [_only_lower(ws / "config.yaml", "org.children_budget")]


def test_a_non_mapping_block_gives_way_to_the_global_layers(tmp_path):
    loader, ws = _layers(tmp_path, g_cfg=_budget(explore={"actions": 20}),
                         ws_cfg={"org": {"children_budget": 5}})
    assert _explore(loader) == (20, 3.0)
    assert _warnings(loader) == [_only_lower(ws / "config.yaml", "org.children_budget")]


def test_a_null_entry_is_restored_with_a_note(tmp_path):
    loader, ws = _layers(tmp_path, ws_cfg=_budget(web_researcher=None))
    b = loader.load_harness().org.children_budget.web_researcher
    assert (b.actions, b.minutes) == (56, 20.0)
    assert _warnings(loader) == [_only_lower(ws / "config.yaml", "org.children_budget.web_researcher")]


def test_a_null_entry_gives_way_to_a_global_entry(tmp_path):
    loader, _ = _layers(tmp_path, g_cfg=_budget(web_researcher={"actions": 30, "minutes": 10.0}),
                        ws_cfg=_budget(web_researcher=7))
    b = loader.load_harness().org.children_budget.web_researcher
    assert (b.actions, b.minutes) == (30, 10.0)


def test_an_unrelated_workspace_block_of_other_org_keys_is_untouched(tmp_path):
    loader, _ = _layers(tmp_path, ws_cfg={"org": {"name": "proj"}})
    assert loader.load_harness().org.name == "proj"
    assert _warnings(loader) == []


# ---------------------------------------------------------------------------
# permissions.budget.max_actions / max_duration_minutes in an agent file: lower-only against the
# global layer's own cascade; kept and marked where the global layer is silent
# ---------------------------------------------------------------------------

import yaml  # noqa: E402

from localharness.config.loader import AGENT_CORE_LOWER_ONLY_KEYS, ConfigLoader  # noqa: E402

_PROVIDER = {"provider_type": "vllm", "base_url": "http://localhost:8000/v1",
             "default_model": "global-model"}


def _put(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data), encoding="utf-8")


def _agent(name: str = "worker", budget=None, **extra) -> dict:
    data = {"name": name, "role": "Work", **extra}
    if budget is not None:
        data["permissions"] = {"budget": budget}
    return data


def _world(tmp_path, *, g_agents=None, g_divs=None, ws_agents=None, ws_divs=None, g_org=None,
           g_over=None):
    """A global dir and a workspace dir, each with the agent and division files given."""
    g, ws = tmp_path / "global", tmp_path / "proj" / ".localharness"
    ws.mkdir(parents=True)
    _put(g / "config.yaml", {"version": "1", "provider": _PROVIDER})
    if g_org is not None:
        _put(g / "org.yaml", g_org)
    if g_over is not None:
        _put(g / "overrides.yaml", g_over)
    for base, agents, divs in ((g, g_agents, g_divs), (ws, ws_agents, ws_divs)):
        for stem, data in (agents or {}).items():
            _put(base / "agents" / f"{stem}.yaml", data)
        for stem, data in (divs or {}).items():
            _put(base / "divisions" / f"{stem}.yaml", data)
    return ConfigLoader(config_dir=g, local_config_dir=ws), ws


def _notes(loader) -> list[str]:
    return [n for n in loader.agent_warnings if "a project may only lower it" in n]


def _only(path, field: str) -> str:
    return f"ignoring permissions.budget.{field} in {str(path)!r}: a project may only lower it"


def _loaded(loader, name: str = "worker"):
    cfg = loader.load_agent(name)
    return cfg.permissions.budget.max_actions, cfg.permissions.budget.max_duration_minutes, \
        cfg.budget_kept_from_workspace


def test_the_agent_lower_only_set_names_both_budget_keys():
    assert AGENT_CORE_LOWER_ONLY_KEYS == frozenset({
        "permissions.budget.max_actions", "permissions.budget.max_duration_minutes"})


def test_a_shadowing_file_cannot_raise_a_global_budget(tmp_path):
    loader, ws = _world(tmp_path,
                        g_agents={"worker": _agent(budget={"max_actions": 10})},
                        ws_agents={"worker": _agent(budget={"max_actions": 16})})
    assert _loaded(loader)[0] == 10
    assert _notes(loader) == [_only(ws / "agents" / "worker.yaml", "max_actions")]


def test_a_shadowing_file_may_lower_a_global_budget(tmp_path):
    loader, _ = _world(tmp_path,
                       g_agents={"worker": _agent(budget={"max_actions": 10})},
                       ws_agents={"worker": _agent(budget={"max_actions": 5})})
    assert _loaded(loader) == (5, None, frozenset())
    assert _notes(loader) == []


def test_an_equal_value_is_kept_without_a_note(tmp_path):
    loader, _ = _world(tmp_path,
                       g_agents={"worker": _agent(budget={"max_actions": 10})},
                       ws_agents={"worker": _agent(budget={"max_actions": 10})})
    assert _loaded(loader)[0] == 10
    assert _notes(loader) == []


def test_a_null_in_a_shadowing_file_keeps_the_global_value_silently(tmp_path):
    loader, _ = _world(tmp_path,
                       g_agents={"worker": _agent(budget={"max_actions": 10})},
                       ws_agents={"worker": _agent(budget={"max_actions": None})})
    assert _loaded(loader) == (10, None, frozenset())
    assert _notes(loader) == []


def test_an_omitted_budget_in_a_shadowing_file_keeps_the_global_value_silently(tmp_path):
    loader, _ = _world(tmp_path,
                       g_agents={"worker": _agent(budget={"max_actions": 10})},
                       ws_agents={"worker": _agent()})
    assert _loaded(loader) == (10, None, frozenset())
    assert _notes(loader) == []


def test_an_omitted_key_of_a_declared_budget_keeps_the_global_value_silently(tmp_path):
    loader, _ = _world(tmp_path,
                       g_agents={"worker": _agent(budget={"max_actions": 10})},
                       ws_agents={"worker": _agent(budget={"max_duration_minutes": 4.0})})
    assert _loaded(loader) == (10, 4.0, frozenset({"max_duration_minutes"}))
    assert _notes(loader) == []


@pytest.mark.parametrize("bad", ["lots", True, 2.5, [3], 0, -3, 10_001])
def test_a_non_finite_workspace_value_keeps_the_global_value_with_a_note(tmp_path, bad):
    loader, ws = _world(tmp_path,
                        g_agents={"worker": _agent(budget={"max_actions": 10})},
                        ws_agents={"worker": _agent(budget={"max_actions": bad})})
    assert _loaded(loader) == (10, None, frozenset())
    assert _notes(loader) == [_only(ws / "agents" / "worker.yaml", "max_actions")]


def test_a_higher_workspace_value_over_a_global_fifty_gives_fifty(tmp_path):
    loader, ws = _world(tmp_path,
                        g_agents={"worker": _agent(budget={"max_actions": 50})},
                        ws_agents={"worker": _agent(budget={"max_actions": 100})})
    assert _loaded(loader)[0] == 50
    assert _notes(loader) == [_only(ws / "agents" / "worker.yaml", "max_actions")]


def test_minutes_follow_the_same_rule(tmp_path):
    loader, ws = _world(tmp_path,
                        g_agents={"worker": _agent(budget={"max_duration_minutes": 5.0})},
                        ws_agents={"worker": _agent(budget={"max_duration_minutes": 30.0})})
    assert _loaded(loader)[1] == 5.0
    assert _notes(loader) == [_only(ws / "agents" / "worker.yaml", "max_duration_minutes")]


@pytest.mark.parametrize("bad", ["soon", True, 0, 0.05, 1441.0, math.inf])
def test_a_non_finite_minutes_value_keeps_the_global_value_with_a_note(tmp_path, bad):
    loader, ws = _world(tmp_path,
                        g_agents={"worker": _agent(budget={"max_duration_minutes": 5.0})},
                        ws_agents={"worker": _agent(budget={"max_duration_minutes": bad})})
    assert _loaded(loader)[1] == 5.0
    assert _notes(loader) == [_only(ws / "agents" / "worker.yaml", "max_duration_minutes")]


def test_a_workspace_only_agent_is_kept_as_written_and_marked(tmp_path):
    loader, _ = _world(tmp_path, ws_agents={"analyst": _agent(
        "analyst", budget={"max_actions": 16, "max_duration_minutes": 12})})
    assert _loaded(loader, "analyst") == (
        16, 12.0, frozenset({"max_actions", "max_duration_minutes"}))
    assert _notes(loader) == []


def test_only_the_declared_field_is_marked(tmp_path):
    loader, _ = _world(tmp_path, ws_agents={"analyst": _agent(
        "analyst", budget={"max_actions": 16})})
    assert _loaded(loader, "analyst")[0::2] == (16, frozenset({"max_actions"}))


def test_a_workspace_agent_with_no_budget_is_not_marked(tmp_path):
    loader, _ = _world(tmp_path, ws_agents={"analyst": _agent("analyst")})
    assert _loaded(loader, "analyst")[2] == frozenset()
    assert _notes(loader) == []


def test_a_null_budget_value_in_a_workspace_only_agent_is_not_marked(tmp_path):
    loader, _ = _world(tmp_path, ws_agents={"analyst": _agent(
        "analyst", budget={"max_actions": None})})
    assert _loaded(loader, "analyst")[2] == frozenset()


def test_a_non_finite_value_under_a_silent_global_is_removed_with_a_note(tmp_path):
    loader, ws = _world(tmp_path, ws_agents={"analyst": _agent(
        "analyst", budget={"max_actions": "lots"})})
    assert _loaded(loader, "analyst") == (None, None, frozenset())
    assert _notes(loader) == [_only(ws / "agents" / "analyst.yaml", "max_actions")]


def test_a_workspace_file_cannot_set_max_tool_calls(tmp_path):
    loader, _ = _world(tmp_path, ws_agents={"analyst": _agent(
        "analyst", budget={"max_actions": 16, "max_tool_calls": 50})})
    assert loader.load_agent("analyst").permissions.budget.max_tool_calls is None


# -- the division door -------------------------------------------------------------------------

def test_a_workspace_division_cannot_raise_a_global_divisions_budget(tmp_path):
    loader, ws = _world(
        tmp_path,
        g_divs={"research": {"name": "research", "permissions": {"budget": {"max_actions": 20}}}},
        g_agents={"worker": _agent(division="research")},
        ws_divs={"research": {"name": "research",
                              "permissions": {"budget": {"max_actions": 200}}}})
    assert _loaded(loader) == (20, None, frozenset())
    assert _notes(loader) == [_only(ws / "divisions" / "research.yaml", "max_actions")]


def test_a_workspace_division_may_lower_a_global_divisions_budget(tmp_path):
    loader, _ = _world(
        tmp_path,
        g_divs={"research": {"name": "research", "permissions": {"budget": {"max_actions": 20}}}},
        g_agents={"worker": _agent(division="research")},
        ws_divs={"research": {"name": "research",
                              "permissions": {"budget": {"max_actions": 10}}}})
    assert _loaded(loader) == (10, None, frozenset())
    assert _notes(loader) == []


def test_a_shadowing_division_that_omits_the_budget_cannot_lift_it(tmp_path):
    loader, _ = _world(
        tmp_path,
        g_divs={"research": {"name": "research", "permissions": {"budget": {"max_actions": 20}}}},
        g_agents={"worker": _agent(division="research")},
        ws_divs={"research": {"name": "research"}})
    assert _loaded(loader) == (20, None, frozenset())
    assert _notes(loader) == []


def test_a_global_agents_own_budget_outranks_a_workspace_division(tmp_path):
    loader, _ = _world(
        tmp_path,
        g_divs={"research": {"name": "research"}},
        g_agents={"worker": _agent(division="research", budget={"max_actions": 7})},
        ws_divs={"research": {"name": "research",
                              "permissions": {"budget": {"max_actions": 200}}}})
    assert _loaded(loader) == (7, None, frozenset())
    assert _notes(loader) == []


def test_a_workspace_division_under_a_silent_global_is_kept_and_marked(tmp_path):
    loader, _ = _world(
        tmp_path,
        g_agents={"worker": _agent(division="research")},
        ws_divs={"research": {"name": "research",
                              "permissions": {"budget": {"max_actions": 200}}}})
    assert _loaded(loader) == (200, None, frozenset({"max_actions"}))
    assert _notes(loader) == []


# -- the baseline rungs ------------------------------------------------------------------------

def test_a_legacy_global_org_file_is_a_baseline(tmp_path):
    loader, ws = _world(tmp_path, g_org={"name": "o", "permissions": {"budget": {"max_actions": 30}}},
                        ws_agents={"analyst": _agent("analyst", budget={"max_actions": 40})})
    assert _loaded(loader, "analyst") == (30, None, frozenset())
    assert _notes(loader) == [_only(ws / "agents" / "analyst.yaml", "max_actions")]


def test_the_global_overrides_agent_section_is_a_baseline(tmp_path):
    loader, ws = _world(
        tmp_path, g_over={"agent": {"permissions": {"budget": {"max_actions": 25}}}},
        ws_agents={"analyst": _agent("analyst", budget={"max_actions": 40})})
    assert _loaded(loader, "analyst")[0] == 25
    assert _notes(loader) == [_only(ws / "agents" / "analyst.yaml", "max_actions")]


def test_the_global_agent_file_outranks_the_global_division_as_a_baseline(tmp_path):
    loader, ws = _world(
        tmp_path,
        g_divs={"research": {"name": "research", "permissions": {"budget": {"max_actions": 90}}}},
        g_agents={"worker": _agent(division="research", budget={"max_actions": 12})},
        ws_agents={"worker": _agent(division="research", budget={"max_actions": 50})})
    assert _loaded(loader)[0] == 12
    assert _notes(loader) == [_only(ws / "agents" / "worker.yaml", "max_actions")]


# -- no workspace on the cascade; repeated loads ------------------------------------------------

def test_a_global_agent_with_a_global_division_is_loaded_as_before(tmp_path):
    loader, _ = _world(
        tmp_path,
        g_divs={"research": {"name": "research", "permissions": {"budget": {"max_actions": 20}}}},
        g_agents={"worker": _agent(division="research", budget={"max_duration_minutes": 9.0})})
    assert _loaded(loader) == (20, 9.0, frozenset())
    assert _notes(loader) == []


def test_a_workspace_less_session_takes_any_value(tmp_path):
    g = tmp_path / "global"
    _put(g / "config.yaml", {"version": "1", "provider": _PROVIDER})
    _put(g / "agents" / "worker.yaml", _agent(budget={"max_actions": 5000}))
    loader = ConfigLoader(config_dir=g)
    assert _loaded(loader) == (5000, None, frozenset())
    assert _notes(loader) == []


def test_loading_a_file_twice_notes_once_and_agrees_with_itself(tmp_path):
    loader, ws = _world(tmp_path,
                        g_agents={"worker": _agent(budget={"max_actions": 10})},
                        ws_agents={"worker": _agent(budget={"max_actions": 16}),
                                   "analyst": _agent("analyst", budget={"max_actions": 16})})
    path = ws / "agents" / "worker.yaml"
    first = loader.load_agent_file(path)
    second = loader.load_agent_file(path)
    assert first.permissions.budget == second.permissions.budget
    assert _notes(loader) == [_only(path, "max_actions")]
    again = loader.load_agent_file(ws / "agents" / "analyst.yaml")
    twice = loader.load_agent_file(ws / "agents" / "analyst.yaml")
    assert again.budget_kept_from_workspace == twice.budget_kept_from_workspace == {"max_actions"}
