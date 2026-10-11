"""`channels.unattended_max_actions` is machine-level only, and `org.children_budget` and the
shares are lower-only on a workspace's raw config sources.

A cloned repository's `.localharness/` loads with no prompt, so it may make a child's ceiling or
share smaller and never larger; the ceiling a remote unattended session runs under is the
machine's alone. Narrowing happens on the raw sources before they merge, where a key left out and
a `null` can still be told apart."""
from __future__ import annotations

from localharness.config import loader as L
from tests.unit.test_docs_convergence import machine_only_code_set
from tests.unit.test_floor_machine_level import _layers, _warnings

KNOB = "channels.unattended_max_actions"


def _knob(value) -> dict:
    return {"channels": {"unattended_max_actions": value}}


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
