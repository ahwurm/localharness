"""`model_ops.persist_landed` is the one writer of a landing's outcome (D-07, D-08, D-15).

`last_served` always; `active_endpoint` only for a launch or swap; `pending_preset` removed after a
launch or swap and after an attach that resolved on the `pending_preset` rung. An attach, including
an attach to another record, never persists a selection.
"""
from __future__ import annotations

import asyncio

import pytest
import yaml

from localharness.cli import model_ops
from localharness.config.loader import ConfigLoader
from localharness.config.overlay import load_overlay
from localharness.provider import speed_stats
from tests.unit.test_model_cmd_presets import PRIMARY_SERVED, _seed

PEER_ACTIVE = {"name": "flash-next", "base_url": "http://127.0.0.1:8082/v1",
               "provider_type": "llamacpp", "model": "flash-next"}


def _setup(home, *, overlay=None):
    _seed(home)
    if overlay is not None:
        (home / "overrides.yaml").write_text(yaml.safe_dump(overlay), encoding="utf-8")
    harness = ConfigLoader(config_dir=str(home)).load_harness()
    records = {r.name: r for r in model_ops.serving_records(harness)}
    return harness, records


def _ledger(home):
    return speed_stats.default_speed_stats_path(home)


def _land(harness, record, home, **kw):
    asyncio.run(model_ops.persist_landed(harness, record, config_dir=home, **kw))


def _overlay_bytes(home) -> bytes:
    return (home / "overrides.yaml").read_bytes()


@pytest.mark.parametrize("outcome", ["launch", "swap"])
def test_a_peer_launch_or_swap_selects_it_and_drops_the_pending_preset(components_home, outcome):
    harness, records = _setup(components_home, overlay={"pending_preset": "gemma"})
    assert harness.pending_preset == "gemma"

    _land(harness, records["flash-next"], components_home, outcome=outcome)

    overlay = load_overlay(components_home / "overrides.yaml")
    assert overlay["active_endpoint"] == PEER_ACTIVE
    assert "pending_preset" not in overlay
    assert harness.pending_preset is None
    assert speed_stats.last_served(_ledger(components_home)) == "flash-next"


def test_a_primary_swap_clears_the_selection(components_home):
    harness, records = _setup(
        components_home, overlay={"active_endpoint": PEER_ACTIVE, "pending_preset": "gemma"})

    _land(harness, records[PRIMARY_SERVED], components_home, outcome="swap")

    overlay = load_overlay(components_home / "overrides.yaml")
    assert "active_endpoint" not in overlay and "pending_preset" not in overlay
    assert harness.active_endpoint is None
    assert speed_stats.last_served(_ledger(components_home)) == PRIMARY_SERVED


def test_a_launch_uses_the_served_name_it_is_given(components_home):
    harness, records = _setup(components_home)

    _land(harness, records["flash-next"], components_home, outcome="launch", served="alias-x")

    assert load_overlay(components_home / "overrides.yaml")["active_endpoint"]["model"] == "alias-x"


def test_an_attach_on_the_active_rung_changes_only_last_served(components_home):
    harness, records = _setup(
        components_home, overlay={"active_endpoint": PEER_ACTIVE, "pending_preset": "gemma"})
    before = _overlay_bytes(components_home)

    _land(harness, records["flash-next"], components_home, outcome="attach", rung="active_endpoint")

    assert _overlay_bytes(components_home) == before
    assert speed_stats.last_served(_ledger(components_home)) == "flash-next"


def test_an_attach_to_another_record_persists_no_selection(components_home):
    """D-15: the record found serving is not the one a selection names, and it selects nothing."""
    harness, records = _setup(
        components_home, overlay={"active_endpoint": PEER_ACTIVE, "pending_preset": "gemma"})
    before = _overlay_bytes(components_home)

    _land(harness, records[PRIMARY_SERVED], components_home, outcome="attach", rung=None)

    assert _overlay_bytes(components_home) == before
    assert speed_stats.last_served(_ledger(components_home)) == PRIMARY_SERVED


def test_an_attach_on_the_pending_rung_removes_pending_and_keeps_the_endpoint(components_home):
    harness, records = _setup(
        components_home, overlay={"active_endpoint": PEER_ACTIVE, "pending_preset": "flash-next"})

    _land(harness, records["flash-next"], components_home, outcome="attach", rung="pending_preset")

    overlay = load_overlay(components_home / "overrides.yaml")
    assert "pending_preset" not in overlay
    assert overlay["active_endpoint"] == PEER_ACTIVE
    assert harness.pending_preset is None


def test_a_ledger_write_that_raises_oserror_does_not_fail_the_landing(components_home, monkeypatch):
    harness, records = _setup(components_home, overlay={"pending_preset": "gemma"})

    def boom(*a, **kw):
        raise OSError("read-only ledger")

    monkeypatch.setattr(speed_stats, "record_last_served", boom)

    _land(harness, records["flash-next"], components_home, outcome="launch")

    overlay = load_overlay(components_home / "overrides.yaml")
    assert overlay["active_endpoint"]["name"] == "flash-next" and "pending_preset" not in overlay


def test_with_no_pending_preset_nothing_extra_is_written(components_home):
    harness, records = _setup(components_home, overlay={"org": {"default_model": PRIMARY_SERVED}})
    before = (_overlay_bytes(components_home), (components_home / "overrides.yaml").stat().st_mtime_ns)

    _land(harness, records[PRIMARY_SERVED], components_home, outcome="attach", rung="pending_preset")

    path = components_home / "overrides.yaml"
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_an_unknown_outcome_is_refused_before_anything_is_written(components_home):
    harness, records = _setup(components_home, overlay={"pending_preset": "gemma"})
    before = _overlay_bytes(components_home)

    with pytest.raises(ValueError):
        _land(harness, records["flash-next"], components_home, outcome="attached")

    assert _overlay_bytes(components_home) == before
    assert speed_stats.last_served(_ledger(components_home)) is None
    assert model_ops.LANDING_OUTCOMES == frozenset({"launch", "swap", "attach"})


@pytest.mark.parametrize("outcome", ["launch", "swap"])
def test_landing_on_the_peer_the_record_names_keeps_its_effort(components_home, outcome):
    """A resumed peer's level is the person's: landing on the peer `active_endpoint` already names
    is not a switch, so the stored effort survives the landing on disk and in the live harness."""
    stored = {**PEER_ACTIVE, "effort": "low"}
    harness, records = _setup(components_home, overlay={"active_endpoint": stored})
    assert harness.active_endpoint.effort == "low"

    _land(harness, records["flash-next"], components_home, outcome=outcome)

    overlay = load_overlay(components_home / "overrides.yaml")
    assert overlay["active_endpoint"] == stored
    assert harness.active_endpoint.effort == "low"


def test_landing_on_a_different_peer_still_drops_the_old_level(components_home):
    """A real switch (spec 16 §7.2): the level belongs to the endpoint it was chosen on."""
    other = {**PEER_ACTIVE, "name": "elsewhere", "base_url": "http://127.0.0.1:9999/v1", "effort": "low"}
    harness, records = _setup(components_home, overlay={"active_endpoint": other})

    _land(harness, records["flash-next"], components_home, outcome="launch")

    overlay = load_overlay(components_home / "overrides.yaml")
    assert overlay["active_endpoint"] == PEER_ACTIVE
    assert harness.active_endpoint.effort is None


@pytest.mark.parametrize("outcome", ["launch", "swap"])
def test_keep_selection_leaves_the_stored_endpoint_alone_on_a_landing(components_home, outcome):
    harness, records = _setup(
        components_home, overlay={"active_endpoint": PEER_ACTIVE, "pending_preset": "gemma"})

    _land(harness, records[PRIMARY_SERVED], components_home, outcome=outcome, keep_selection=True)

    overlay = load_overlay(components_home / "overrides.yaml")
    assert overlay["active_endpoint"] == PEER_ACTIVE
    assert "pending_preset" not in overlay  # the landing's other effects stand
    assert speed_stats.last_served(_ledger(components_home)) == PRIMARY_SERVED
