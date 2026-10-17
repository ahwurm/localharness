# ruff: noqa: F811  (the `popen` fixture is imported by name, then taken as a parameter)
"""The window line says when llama-server put dense layers on the CPU: the process's GPU residency read against the priced share."""
from __future__ import annotations

import dataclasses

import pytest

from localharness.config.models import AutosenseConfig
from localharness.provider import autosense, lifecycle
from localharness.provider.autosense import Placement, WindowReport, spilled_bytes, window_lines
from tests.unit.launch_support import popen  # noqa: F401
from tests.unit.test_launch_fit_retry import BASE, RATIOS, Box, _spec

GIB = 2**30
MIB = 2**20
KV = 36992.0
ENDING = " — {} GiB of layers on the CPU, decode will be slow"
NVIDIA_HELD = 28098 * MIB
SWIFT_HELD = 25960 * MIB


def placed(gpu, *, where="gpu", experts=0, swa=0, kv=KV):
    return Placement(fits=None, where=where, gpu_bytes=gpu, mmproj=0, experts_ram=experts, kv_per_token=kv,
                     free_bytes=30 * GIB, budget_bytes=30 * GIB, cache_ram=None, refusal=None, swa_bytes=swa)


def report(gpu=18_913_411_136, window=262_144, held=NVIDIA_HELD, **over):
    fields = dict(served=window, trained=262_144, fits=window, free_bytes=30 * GIB, total_bytes=32 * GIB,
                  weights_bytes=gpu, asked=window, mode="fit", placement=placed(gpu), held=held)
    return WindowReport(**{**fields, **over})


def priced(r):
    p = r.placement
    return int(p.gpu_bytes + p.kv_per_token * r.served + p.swa_bytes)


def short_by(r, n):
    return dataclasses.replace(r, held=priced(r) + autosense.SPILL_OVERHEAD - n)


def line(r, ratios=None):
    return window_lines(r, ratios or AutosenseConfig())[0]


SWIFT = report(gpu=18_562_465_856 + 927_606_912, window=131_072, held=SWIFT_HELD)


def test_the_nvidia_launch_that_spilled_says_1_9_gib_of_layers_are_on_the_cpu():
    assert priced(report()) == 28_610_641_984
    assert line(report()).endswith(ENDING.format("1.9"))


def test_the_healthy_swift_launch_carries_no_clause():
    assert priced(SWIFT) == 24_338_688_192
    assert spilled_bytes(SWIFT) is None
    assert "layers on the CPU" not in line(SWIFT)


@pytest.mark.parametrize("reserve", [0.0, 3.0, 10.0])
def test_no_configured_value_moves_the_line(reserve):
    assert line(report(), AutosenseConfig(compute_reserve=reserve)) == line(report())
    assert line(report(), AutosenseConfig(compute_reserve=reserve)).endswith(ENDING.format("1.9"))


async def test_activate_reads_the_launched_pids_residency_into_the_window_report(tmp_path, popen, monkeypatch):
    box = Box(monkeypatch)
    asked = []
    held = 23_000 * MIB
    monkeypatch.setattr(autosense, "_smi", lambda query: asked.append(query) or f"1, 5\n4242, {held // MIB}\n")
    live = await lifecycle.SpawnedProcessStrategy().activate(_spec(tmp_path), tmp_path, BASE, autosense=RATIOS)
    assert asked == ["--query-compute-apps=pid,used_memory"]
    assert live.window.held == held
    assert window_lines(live.window, RATIOS)[0].endswith(ENDING.format("2.4"))
    assert "wait_ready" in box.events


async def _activate(tmp_path, monkeypatch, held_by, **over):
    Box(monkeypatch)
    monkeypatch.setattr(autosense, "held_by", held_by)
    return await lifecycle.SpawnedProcessStrategy().activate(_spec(tmp_path, **over), tmp_path, BASE, autosense=RATIOS)


async def test_a_launch_with_ngl_reads_no_residency_and_gets_no_clause(tmp_path, popen, monkeypatch):
    def boom(pid):
        raise AssertionError("held_by read for a launch that sets -ngl")

    live = await _activate(tmp_path, monkeypatch, boom, extra_args=["-ngl", "99"])
    assert live.window.held is None
    assert "layers on the CPU" not in window_lines(live.window, RATIOS)[0]


async def test_nothing_listed_for_the_pid_gives_no_clause(tmp_path, popen, monkeypatch):
    live = await _activate(tmp_path, monkeypatch, lambda pid: None)
    assert live.window.held is None
    assert "layers on the CPU" not in window_lines(live.window, RATIOS)[0]


@pytest.mark.parametrize("over", [
    {"placement": placed(18 * GIB, where="vllm")},
    {"attached": True},
    {"mode": "fit-by-server"},
    {"mode": "record"},
    {"mode": "extra_args"},
    {"verified": False, "served": None, "asked": None},
])
def test_no_clause_where_the_window_was_not_priced_by_the_harness_for_a_llamacpp_fit(over):
    r = dataclasses.replace(report(), **over)
    assert spilled_bytes(dataclasses.replace(r, held=0)) is None
    assert "layers on the CPU" not in "".join(window_lines(dataclasses.replace(r, held=0), AutosenseConfig()))


def test_a_shortfall_one_byte_under_a_gib_says_nothing_and_exactly_a_gib_says_1_0():
    assert spilled_bytes(short_by(report(), GIB - 1)) is None
    assert spilled_bytes(short_by(report(), GIB)) == GIB
    assert line(short_by(report(), GIB)).endswith(ENDING.format("1.0"))


def test_an_experts_in_ram_placement_is_outside_the_calibrated_class():
    r = report()
    r = dataclasses.replace(r, placement=placed(r.placement.gpu_bytes, where="experts in RAM", experts=40 * GIB))
    assert spilled_bytes(short_by(r, 2 * GIB)) is None


def test_a_small_model_at_a_small_window_cannot_read_as_layers_on_the_cpu():
    r = report(gpu=4 * GIB, window=8192)
    assert spilled_bytes(short_by(r, 2 * GIB)) is None
    assert "layers on the CPU" not in line(short_by(r, 2 * GIB))


def test_the_clause_starts_at_ten_gib_of_priced_gpu_bytes():
    ten = report(gpu=10 * GIB, window=8192)
    assert line(short_by(ten, 2 * GIB)).endswith(ENDING.format("2.0"))
    under = report(gpu=10 * GIB - 1, window=8192)
    assert spilled_bytes(short_by(under, 2 * GIB)) is None
