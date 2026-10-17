"""The launch line names the window and where the weights live; fit-by-server names its floor."""
from __future__ import annotations

from dataclasses import replace

from localharness.config.models import AutosenseConfig
from localharness.provider.autosense import Placement, WindowReport, window_lines
from tests.unit.test_plan_launch_floor import _flash, _plan, _world

GIB = 2**30
RATIOS = AutosenseConfig()


def test_flash_next_launch_line_says_experts_in_ram_and_the_gpu_share(tmp_path, monkeypatch):
    _world(monkeypatch)
    plan = _plan(_flash(tmp_path), tmp_path)
    report = replace(plan.report, served=262144, verified=True)
    assert window_lines(report, RATIOS) == [
        "window: 262,144 · experts in RAM (40.0 GiB) · dense, projector and KV on the GPU (8.5 of 30.2 GiB free)"]


def _launch(**over) -> WindowReport:
    base = WindowReport(served=262144, trained=262144, fits=262144, free_bytes=int(30.2 * GIB),
                        total_bytes=int(31.8 * GIB), weights_bytes=None, asked=262144, mode="fit")
    return replace(base, **over)


def _placement(**over) -> Placement:
    base = Placement(262144, "gpu", int(27.2 * GIB) - 262144 * 1000, 0, 0, 1000.0, int(30.2 * GIB),
                     int(30.2 * GIB), None, None)
    return replace(base, **over)


def test_a_model_all_on_the_gpu():
    assert window_lines(_launch(placement=_placement()), RATIOS) == [
        "window: 262,144 · all on the GPU (27.2 of 30.2 GiB free)"]


def test_a_vllm_checkpoint_names_vllm_and_its_budget():
    placed = Placement(100864, "vllm", int(20.42 * GIB), 0, 0, 65536, int(31.8 * GIB), int(28.6 * GIB), None, None)
    report = _launch(runtime="vllm", served=100864, asked=100864, placement=placed)
    assert window_lines(report, RATIOS) == ["window: 100,864 · vLLM · all on the GPU (26.6 of vLLM's 28.6 GiB)"]


def test_experts_in_ram_without_a_projector_says_dense_and_kv():
    placed = _placement(where="experts in RAM", experts_ram=int(40 * GIB), mmproj=0)
    assert window_lines(_launch(placement=placed), RATIOS) == [
        "window: 262,144 · experts in RAM (40.0 GiB) · dense and KV on the GPU (27.2 of 30.2 GiB free)"]


def test_fit_by_server_names_its_floor():
    report = _launch(mode="fit-by-server", served=4096, floor=15360)
    assert window_lines(report, RATIOS) == ["window: served by llama-server's --fit (at least 15,360) · served 4,096"]


def test_fit_by_server_without_a_floor_is_the_old_line():
    report = _launch(mode="fit-by-server", served=4096)
    assert window_lines(report, RATIOS) == ["window: served by llama-server's --fit · served 4,096"]


def test_an_unverified_window_is_the_asked_one_and_says_so():
    report = _launch(placement=_placement(), served=None, verified=False)
    assert window_lines(report, RATIOS) == ["window: 262,144 · all on the GPU (27.2 of 30.2 GiB free) (not verified)"]


def test_a_server_that_gave_less_than_asked_says_so():
    report = _launch(placement=_placement(), served=131072, asked=262144)
    assert window_lines(report, RATIOS) == [
        "window: 131,072 · all on the GPU (27.1 of 30.2 GiB free) (asked 262,144)"]


def test_a_death_retry_says_so():
    report = _launch(placement=_placement(), served=131072, asked=131072, retried=True)
    assert window_lines(report, RATIOS)[0].endswith(" (after one retry at -c 131,072)")


def test_no_placement_and_attach_reports_keep_the_old_line():
    old = "window: served 262,144 · trained 262,144 · fits 262,144"
    assert window_lines(_launch(mode="record"), RATIOS) == [old]
    assert window_lines(_launch(mode="extra_args", placement=_placement()), RATIOS) == [old]
    attach = _launch(attached=True, placement=_placement(), fits=100000)
    assert window_lines(attach, RATIOS)[0].startswith("window: served 262,144 · trained 262,144 · fits 100,000")
