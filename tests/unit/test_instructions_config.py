"""`instructions:` — the two machine-level knobs on the project-instructions file.

`max_ratio` is the largest share of the context window the file may take; `children` says whether
delegated children receive it too. Both are the machine's: a project's `.localharness/` loads with
no prompt, so it may not widen what a file it ships can take from the window or push its text into
every child. The enforcement itself is the loader's generic global-only walk; this module pins the
model's defaults and bounds and that both leaves are listed.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from localharness.config.loader import HARNESS_GLOBAL_ONLY_FIELDS, is_harness_global_only
from localharness.config.models import HarnessConfig, InstructionsConfig
from tests.unit.test_floor_machine_level import _layers, _warnings


def test_the_defaults_are_a_quarter_of_the_window_and_children_on():
    cfg = InstructionsConfig()
    assert (cfg.max_ratio, cfg.children) == (0.25, True)


@pytest.mark.parametrize("ratio", [0, 0.0, -0.1, 0.6, 1.0])
def test_a_ratio_outside_zero_to_one_half_is_rejected(ratio):
    with pytest.raises(ValidationError):
        InstructionsConfig(max_ratio=ratio)


@pytest.mark.parametrize("ratio", [0.01, 0.25, 0.5])
def test_a_ratio_inside_the_bounds_is_accepted(ratio):
    assert InstructionsConfig(max_ratio=ratio).max_ratio == ratio


def test_an_unknown_key_is_rejected():
    with pytest.raises(ValidationError):
        InstructionsConfig(max_size=3)  # type: ignore[call-arg]


def test_the_harness_config_carries_the_section_with_its_defaults(tmp_path):
    from tests.unit.test_registry_catalogue import _make_minimal_harness_cfg

    assert _make_minimal_harness_cfg().instructions == InstructionsConfig()
    assert HarnessConfig.model_fields["instructions"].default_factory is InstructionsConfig


def test_both_leaves_are_machine_level_only_and_the_bare_section_is_not():
    assert {"instructions.max_ratio", "instructions.children"} <= HARNESS_GLOBAL_ONLY_FIELDS
    assert "instructions" not in HARNESS_GLOBAL_ONLY_FIELDS
    assert is_harness_global_only("instructions.max_ratio")
    assert is_harness_global_only("instructions.children")


def test_the_machines_config_sets_both_with_no_warning(tmp_path):
    loader, _ = _layers(tmp_path, g_cfg={"instructions": {"max_ratio": 0.4, "children": False}})
    h = loader.load_harness()
    assert (h.instructions.max_ratio, h.instructions.children) == (0.4, False)
    assert _warnings(loader) == []


def test_the_machines_overrides_file_sets_both_with_no_warning(tmp_path):
    loader, _ = _layers(tmp_path, g_over={"instructions": {"max_ratio": 0.5, "children": False}})
    h = loader.load_harness()
    assert (h.instructions.max_ratio, h.instructions.children) == (0.5, False)
    assert _warnings(loader) == []


def test_a_project_value_gives_way_to_the_machines_own(tmp_path):
    loader, ws = _layers(
        tmp_path,
        g_cfg={"instructions": {"max_ratio": 0.1, "children": False}},
        ws_cfg={"instructions": {"max_ratio": 0.5, "children": True}},
    )
    h = loader.load_harness()
    assert (h.instructions.max_ratio, h.instructions.children) == (0.1, False)
    assert sorted(_warnings(loader)) == sorted([
        f"ignoring instructions.max_ratio in {ws / 'config.yaml'}: only the global config may set it",
        f"ignoring instructions.children in {ws / 'config.yaml'}: only the global config may set it",
    ])
