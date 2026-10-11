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
