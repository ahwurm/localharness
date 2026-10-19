"""A Strata record is priced against its RAM headroom, the GPU and the floor before anything stops."""
from __future__ import annotations

from dataclasses import replace

from localharness.config.models import AutosenseConfig
from localharness.provider import lifecycle, server
from localharness.provider.autosense import window_lines
from tests.unit.strata_fixture import flash_next, strata_record, strata_tree
from tests.unit.test_plan_launch_floor import _world

AUTO = AutosenseConfig()


def _spec(tmp_path, config=None, **over):
    shard, projector = flash_next(tmp_path)
    engine = strata_tree(tmp_path, config=config(shard) if config else None, model=shard)
    return strata_record(engine, shard, projector, **over)


def _plan(spec, config_dir, floor=None):
    sensed = lifecycle.sense_launch(spec, config_dir, AUTO)
    return lifecycle.plan_launch(
        spec, sensed, server.launch_decision(spec, False), autosense=AUTO, api_key_held=False, floor=floor)


def test_the_flash_next_record_plans_its_window_with_the_experts_pinned_in_ram(tmp_path, monkeypatch):
    _world(monkeypatch)
    plan = _plan(_spec(tmp_path), tmp_path)
    report = replace(plan.report, served=65536, verified=True)
    assert window_lines(report, AUTO) == [
        "window: 65,536 · Strata · experts pinned in RAM (40.0 GiB) · one request at a time"]
    assert plan.report.mode == "fit" and plan.full_fit == 65536 and plan.serve_kwargs["context"] == 65536
