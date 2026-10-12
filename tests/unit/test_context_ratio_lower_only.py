"""A project may only lower the window ratios, at every door.

`org.context.tool_output_ratio`, `evict_ratio` and `result_cap_ratio` are the share of the live
context window one tool result may take, the size above which a result is evicted, and what one
result may put into the conversation. A cloned repository's `.localharness/` may make them smaller
and never larger: a larger, zero or non-numeric value is dropped with a warning naming the key and
the file, a `null` is dropped quietly. The same rule holds on the harness view (the raw config
files), on an agent's own `context` (its file, its division, the workspace `org.context`) and on a
workspace override of a built-in agent.
"""
from __future__ import annotations

import math

import pytest

from localharness.config.models import ContextConfig
from tests.unit.test_floor_machine_level import _layers, _warnings

NAMES = ("tool_output_ratio", "evict_ratio", "result_cap_ratio")
DEFAULTS = {n: ContextConfig.model_fields[n].default for n in NAMES}
TOR = "tool_output_ratio"


def _ctx(**kw) -> dict:
    return {"org": {"context": kw}}


def _got(loader, name: str = TOR) -> float:
    return getattr(loader.load_harness().org.context, name)


def _note(name: str, file) -> str:
    return f"ignoring org.context.{name} in {file}: a project may only lower it"


# ---------------------------------------------------------------------------
# the harness view
# ---------------------------------------------------------------------------


def test_the_harness_lower_only_set_names_the_three_ratios():
    from localharness.config.loader import HARNESS_LOWER_ONLY_FIELDS
    assert {f"org.context.{n}" for n in NAMES} <= HARNESS_LOWER_ONLY_FIELDS


def test_a_workspace_may_lower_a_ratio_under_a_silent_global(tmp_path):
    loader, _ = _layers(tmp_path, ws_cfg=_ctx(tool_output_ratio=0.03))
    assert _got(loader) == 0.03
    assert _warnings(loader) == []


@pytest.mark.parametrize("bad", [0.2, 0, -0.1, math.nan, "lots", True, 1.5, [0.03]])
def test_a_higher_or_junk_workspace_ratio_gives_the_default_with_a_note(tmp_path, bad):
    loader, ws = _layers(tmp_path, ws_cfg=_ctx(tool_output_ratio=bad))
    assert _got(loader) == DEFAULTS[TOR]
    assert _warnings(loader) == [_note(TOR, ws / "config.yaml")]


def test_a_workspace_null_ratio_is_restored_silently(tmp_path):
    loader, _ = _layers(tmp_path, ws_cfg=_ctx(tool_output_ratio=None))
    assert _got(loader) == DEFAULTS[TOR]
    assert _warnings(loader) == []


def test_a_workspace_may_equal_the_default(tmp_path):
    loader, _ = _layers(tmp_path, ws_cfg=_ctx(tool_output_ratio=DEFAULTS[TOR]))
    assert _got(loader) == DEFAULTS[TOR]
    assert _warnings(loader) == []


def test_a_global_ratio_is_the_ceiling_for_the_workspace(tmp_path):
    loader, ws = _layers(tmp_path, g_cfg=_ctx(evict_ratio=0.01), ws_cfg=_ctx(evict_ratio=0.02))
    assert _got(loader, "evict_ratio") == 0.01
    assert _warnings(loader) == [_note("evict_ratio", ws / "config.yaml")]


def test_a_workspace_below_the_global_ratio_is_taken(tmp_path):
    loader, _ = _layers(tmp_path, g_cfg=_ctx(evict_ratio=0.01), ws_cfg=_ctx(evict_ratio=0.005))
    assert _got(loader, "evict_ratio") == 0.005
    assert _warnings(loader) == []


def test_a_workspace_overrides_file_is_judged_too(tmp_path):
    loader, ws = _layers(tmp_path, ws_over=_ctx(result_cap_ratio=0.5))
    assert _got(loader, "result_cap_ratio") == DEFAULTS["result_cap_ratio"]
    assert _warnings(loader) == [_note("result_cap_ratio", ws / "overrides.yaml")]


def test_a_refused_overrides_value_does_not_discard_a_valid_config_lowering(tmp_path):
    loader, ws = _layers(tmp_path, ws_cfg=_ctx(tool_output_ratio=0.03),
                         ws_over=_ctx(tool_output_ratio=0.9))
    assert _got(loader) == 0.03
    assert _warnings(loader) == [_note(TOR, ws / "overrides.yaml")]


def test_with_no_workspace_a_global_ratio_loads_as_written(tmp_path):
    loader, _ = _layers(tmp_path, g_cfg=_ctx(tool_output_ratio=0.5))
    assert _got(loader) == 0.5
    assert _warnings(loader) == []


def test_a_global_ratio_above_the_default_is_the_ceiling(tmp_path):
    loader, _ = _layers(tmp_path, g_cfg=_ctx(tool_output_ratio=0.5), ws_cfg=_ctx(tool_output_ratio=0.4))
    assert _got(loader) == 0.4


def test_a_zero_ratio_is_still_refused_by_the_schema():
    with pytest.raises(ValueError):
        ContextConfig(tool_output_ratio=0)
