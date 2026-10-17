"""The launch line names the window and where the weights live; fit-by-server names its floor."""
from __future__ import annotations

from dataclasses import replace

from localharness.config.models import AutosenseConfig
from localharness.provider.autosense import window_lines
from tests.unit.test_plan_launch_floor import _flash, _plan, _world

GIB = 2**30
RATIOS = AutosenseConfig()


def test_flash_next_launch_line_says_experts_in_ram_and_the_gpu_share(tmp_path, monkeypatch):
    _world(monkeypatch)
    plan = _plan(_flash(tmp_path), tmp_path)
    report = replace(plan.report, served=262144, verified=True)
    assert window_lines(report, RATIOS) == [
        "window: 262,144 · experts in RAM (40.0 GiB) · dense, projector and KV on the GPU (8.5 of 30.2 GiB free)"]
