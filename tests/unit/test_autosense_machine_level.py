"""`autosense`, `effort_budget` and `org.default_effort` are machine-level only.

They decide how the harness sizes and bounds what it serves on this machine's hardware (memory
margins, the folders searched for model files, the reasoning share of every reply), so a cloned
repository's `.localharness/` may not move them. As for `server` and `org.enforce_capability_floor`,
a workspace value is dropped with one startup warning naming the key and the file, and the global
value (or the default) stands. Every ratio is bounded (0, 1]."""
from __future__ import annotations

from pathlib import Path

import pytest

from localharness.config import models as M
from localharness.config.models import HarnessConfig, OrgConfig, ProviderConfig
from tests.unit.test_docs_convergence import machine_only_code_set
from tests.unit.test_floor_machine_level import _layers, _warnings

AUTOSENSE_DEFAULTS = {
    "compute_margin_ratio": 0.10,
    "template_margin_ratio": 1 / 64,
    "unified_reserve_ratio": 0.15,
    "cache_ram_ratio": 0.40,
    "cache_ram_ceiling_ratio": 0.50,
    "notify_ratio": 0.25,
}
EFFORT_BUDGET_DEFAULTS = {"low_ratio": 0.125, "high_ratio": 0.5}

# (model class name, field): resolved on `models` at call time, so a tree without the class fails
# the test, not the collection.
RATIO_FIELDS = (
    [("AutosenseConfig", name) for name in AUTOSENSE_DEFAULTS]
    + [("EffortBudgetConfig", name) for name in EFFORT_BUDGET_DEFAULTS]
)
RATIO_IDS = [f"{model}.{name}" for model, name in RATIO_FIELDS]

_PROVIDER = ProviderConfig(
    provider_type="vllm", base_url="http://127.0.0.1:9/v1", default_model="m")


def _dropped(path: str, file: Path) -> str:
    return f"ignoring {path} in {file}: only the global config may set it"


# ---------------------------------------------------------------------------
# Defaults and bounds
# ---------------------------------------------------------------------------


def test_autosense_defaults_are_the_specs():
    cfg = M.AutosenseConfig()
    for name, want in AUTOSENSE_DEFAULTS.items():
        assert getattr(cfg, name) == want, name
    assert cfg.model_roots == ["~/models", "~/.cache/huggingface"]


def test_effort_budget_defaults_are_the_specs():
    cfg = M.EffortBudgetConfig()
    for name, want in EFFORT_BUDGET_DEFAULTS.items():
        assert getattr(cfg, name) == want, name


def test_org_default_effort_is_unset_by_default():
    assert OrgConfig().default_effort is None


def test_harness_config_carries_both_sections_at_their_defaults():
    cfg = HarnessConfig(provider=_PROVIDER)
    assert cfg.autosense == M.AutosenseConfig()
    assert cfg.effort_budget == M.EffortBudgetConfig()
    assert cfg.org.default_effort is None


def test_model_roots_default_is_not_shared_between_instances():
    a, b = M.AutosenseConfig(), M.AutosenseConfig()
    a.model_roots.append("~/x")
    assert b.model_roots == ["~/models", "~/.cache/huggingface"]


def test_empty_model_roots_validates():
    assert M.AutosenseConfig(model_roots=[]).model_roots == []


@pytest.mark.parametrize("model,name", RATIO_FIELDS, ids=RATIO_IDS)
@pytest.mark.parametrize("bad", [0.0, -0.1, 1.0001])
def test_ratio_outside_zero_one_is_refused(model, name, bad):
    with pytest.raises(ValueError):
        getattr(M, model)(**{name: bad})


@pytest.mark.parametrize("model,name", RATIO_FIELDS, ids=RATIO_IDS)
def test_ratio_of_exactly_one_validates(model, name):
    assert getattr(getattr(M, model)(**{name: 1.0}), name) == 1.0


@pytest.mark.parametrize("model,name", RATIO_FIELDS, ids=RATIO_IDS)
def test_a_tiny_positive_ratio_validates(model, name):
    assert getattr(getattr(M, model)(**{name: 0.0001}), name) == 0.0001


def test_unknown_keys_are_refused():
    with pytest.raises(ValueError):
        M.AutosenseConfig(notify_ratio_typo=0.5)
    with pytest.raises(ValueError):
        M.EffortBudgetConfig(medium_ratio=0.3)


def test_org_default_effort_takes_only_the_four_levels():
    for level in ("off", "low", "high", "max"):
        assert OrgConfig(default_effort=level).default_effort == level
    with pytest.raises(ValueError):
        OrgConfig(default_effort="medium")


# ---------------------------------------------------------------------------
# A workspace file cannot set them
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("file", ["config.yaml", "overrides.yaml"])
@pytest.mark.parametrize("section,key,evil,default", [
    ("autosense", "notify_ratio", 0.5, 0.25),
    ("effort_budget", "high_ratio", 0.9, 0.5),
])
def test_a_workspace_file_cannot_set_a_section(tmp_path, file, section, key, evil, default):
    layer = {("ws_cfg" if file == "config.yaml" else "ws_over"): {section: {key: evil}}}
    loader, ws = _layers(tmp_path, **layer)
    cfg = loader.load_harness()
    assert getattr(getattr(cfg, section), key) == default
    assert _warnings(loader) == [_dropped(section, ws / file)]


@pytest.mark.parametrize("file", ["config.yaml", "overrides.yaml"])
def test_a_workspace_file_cannot_set_org_default_effort(tmp_path, file):
    layer = {("ws_cfg" if file == "config.yaml" else "ws_over"): {"org": {"default_effort": "max"}}}
    loader, ws = _layers(tmp_path, **layer)
    assert loader.load_harness().org.default_effort is None
    assert _warnings(loader) == [_dropped("org.default_effort", ws / file)]


def test_the_global_value_stands_against_a_workspace_value(tmp_path):
    loader, ws = _layers(
        tmp_path,
        g_cfg={"autosense": {"notify_ratio": 0.5}},
        ws_cfg={"autosense": {"notify_ratio": 0.9}})
    assert loader.load_harness().autosense.notify_ratio == 0.5
    assert _warnings(loader) == [_dropped("autosense", ws / "config.yaml")]


def test_the_global_org_default_effort_stands_against_a_workspace_value(tmp_path):
    loader, ws = _layers(
        tmp_path,
        g_cfg={"org": {"default_effort": "low"}},
        ws_over={"org": {"default_effort": "max"}})
    assert loader.load_harness().org.default_effort == "low"
    assert _warnings(loader) == [_dropped("org.default_effort", ws / "overrides.yaml")]


def test_the_global_layers_still_set_all_three(tmp_path):
    loader, _ = _layers(
        tmp_path,
        g_cfg={"autosense": {"notify_ratio": 0.5, "model_roots": ["/m"]},
               "org": {"default_effort": "high"}},
        g_over={"effort_budget": {"low_ratio": 0.2}})
    cfg = loader.load_harness()
    assert cfg.autosense.notify_ratio == 0.5
    assert cfg.autosense.model_roots == ["/m"]
    assert cfg.effort_budget.low_ratio == 0.2
    assert cfg.org.default_effort == "high"
    assert _warnings(loader) == []


def test_a_workspace_org_default_effort_equal_to_the_global_one_is_no_warning(tmp_path):
    loader, _ = _layers(
        tmp_path,
        g_cfg={"org": {"default_effort": "high"}},
        ws_cfg={"org": {"default_effort": "high"}})
    assert loader.load_harness().org.default_effort == "high"
    assert _warnings(loader) == []


def test_an_empty_workspace_autosense_section_still_warns(tmp_path):
    """Pinned deliberately: the comparison is against the section's default object, so a whole
    section is warned even when it restates (or says nothing beyond) the defaults."""
    loader, ws = _layers(tmp_path, ws_cfg={"autosense": {}})
    assert loader.load_harness().autosense == M.AutosenseConfig()
    assert _warnings(loader) == [_dropped("autosense", ws / "config.yaml")]


def test_a_workspace_section_restating_the_defaults_still_warns(tmp_path):
    loader, ws = _layers(tmp_path, ws_cfg={"autosense": dict(AUTOSENSE_DEFAULTS)})
    assert loader.load_harness().autosense == M.AutosenseConfig()
    assert _warnings(loader) == [_dropped("autosense", ws / "config.yaml")]


def test_no_section_in_any_layer_yields_the_defaults_and_no_warning(tmp_path):
    loader, _ = _layers(tmp_path)
    cfg = loader.load_harness()
    assert cfg.autosense == M.AutosenseConfig()
    assert cfg.effort_budget == M.EffortBudgetConfig()
    assert cfg.org.default_effort is None
    assert _warnings(loader) == []


def test_global_empty_model_roots_validates(tmp_path):
    loader, _ = _layers(tmp_path, g_cfg={"autosense": {"model_roots": []}})
    assert loader.load_harness().autosense.model_roots == []


def test_three_warnings_from_one_file_come_out_in_sorted_path_order(tmp_path):
    loader, ws = _layers(
        tmp_path,
        ws_cfg={"org": {"default_effort": "max"},
                "effort_budget": {"low_ratio": 0.2},
                "autosense": {"notify_ratio": 0.9}})
    cfg = loader.load_harness()
    assert (cfg.autosense, cfg.effort_budget, cfg.org.default_effort) == (
        M.AutosenseConfig(), M.EffortBudgetConfig(), None)
    file = ws / "config.yaml"
    assert _warnings(loader) == [
        _dropped("autosense", file),
        _dropped("effort_budget", file),
        _dropped("org.default_effort", file),
    ]


# ---------------------------------------------------------------------------
# The machine-only list names them
# ---------------------------------------------------------------------------


def test_the_three_are_in_the_code_set_of_machine_only_settings():
    assert {"autosense", "effort_budget", "org.default_effort"} <= machine_only_code_set()
