"""A Strata record is priced against its RAM headroom, the GPU and the floor before anything stops."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from localharness.config.models import AutosenseConfig
from localharness.provider import autosense, lifecycle, server, strata
from localharness.provider.autosense import window_lines
from tests.unit.strata_fixture import flash_next, strata_config, strata_record, strata_tree
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


GIB = 2**30
NAME = "swift-1.5-iq3_xxs"
RAM_SENTENCE = (NAME + " cannot run here: its experts need 40.0 GiB of RAM and only {free} GiB is free beside the "
                "system's reserve — close something that holds RAM, or pick a smaller quant.")


def _meminfo(monkeypatch, *, available, total=int(61.17 * GIB)):
    monkeypatch.setattr(autosense, "_read_meminfo", lambda: {"MemTotal": total, "MemAvailable": available})


def _experts(spec, config_dir) -> int:
    return lifecycle.sense_launch(spec, config_dir, AUTO).split.experts


def _refusal(spec, config_dir, floor=None) -> str:
    with pytest.raises(server.LaunchRefused) as refused:
        _plan(spec, config_dir, floor)
    return str(refused.value)


def test_a_box_with_31_gib_available_refuses_with_the_ram_sentence_and_the_box_with_52_gib_does_not(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    _world(monkeypatch, available=52.9)
    assert _plan(spec, tmp_path).report.placement.experts_ram == 42_912_972_800
    _world(monkeypatch, available=31.4)
    assert _refusal(spec, tmp_path) == RAM_SENTENCE.format(free="22.1")


def test_unreadable_ram_refuses_with_the_unreadable_sentence(tmp_path, monkeypatch):
    _world(monkeypatch)
    monkeypatch.setattr(autosense, "_read_meminfo", lambda: None)
    assert _refusal(_spec(tmp_path), tmp_path) == (
        NAME + " cannot run here: its experts need 40.0 GiB of RAM and the free RAM cannot be read (/proc/meminfo) "
        "— pick a quant whose experts fit the GPU.")


def test_experts_plus_ten_gb_exactly_equal_to_the_available_ram_fits_and_one_byte_less_refuses(tmp_path, monkeypatch):
    _world(monkeypatch)
    spec = _spec(tmp_path)
    need = _experts(spec, tmp_path) + 10 * 10**9
    _meminfo(monkeypatch, available=need)
    assert _plan(spec, tmp_path).report.mode == "fit"
    _meminfo(monkeypatch, available=need - 1)
    assert _refusal(spec, tmp_path).startswith(NAME + " cannot run here: its experts need 40.0 GiB of RAM")


def test_ten_gb_is_9_3_gib_and_the_free_figure_prints_with_one_decimal(tmp_path, monkeypatch):
    _world(monkeypatch)
    spec = _spec(tmp_path)
    _meminfo(monkeypatch, available=_experts(spec, tmp_path) + 10**10 - 1)
    assert "only 40.0 GiB is free" in _refusal(spec, tmp_path)
    assert autosense.gib(10**10) == "9.3"


def test_the_machine_reserve_replaces_the_ten_gb_when_larger_and_is_not_added_to_it(tmp_path, monkeypatch):
    _world(monkeypatch)
    spec = _spec(tmp_path)
    experts, total = _experts(spec, tmp_path), 10**11
    _meminfo(monkeypatch, available=experts + 12 * 10**9, total=total)
    assert "its experts need 40.0 GiB of RAM" in _refusal(spec, tmp_path)
    _meminfo(monkeypatch, available=experts + 16 * 10**9, total=total)
    assert _plan(spec, tmp_path).report.mode == "fit"


def test_the_launch_prices_its_own_reading_of_the_ram_not_an_earlier_one(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    _world(monkeypatch, available=52.9)
    _plan(spec, tmp_path)
    _world(monkeypatch, available=31.4)
    assert "its experts need" in _refusal(spec, tmp_path)


def test_dense_layers_the_projector_and_the_windows_kv_above_the_free_gpu_are_the_vram_sentence(tmp_path, monkeypatch):
    _world(monkeypatch, free=3 * GIB)
    assert _refusal(_spec(tmp_path), tmp_path) == (
        NAME + " cannot run here: it needs 5.5 GiB of GPU memory for a 65,536-token window and 3.0 GiB is free — "
        "stop what holds the GPU, or pick a smaller quant.")


def test_unknown_free_gpu_memory_is_not_computed_and_does_not_refuse(tmp_path, monkeypatch):
    _world(monkeypatch, memory=False)
    report = _plan(_spec(tmp_path), tmp_path).report
    assert report.mode == "fit" and report.placement.free_bytes == 0


FLOOR = 13_824


def test_a_record_context_under_the_floor_names_the_record_field(tmp_path, monkeypatch):
    _world(monkeypatch)
    assert _refusal(_spec(tmp_path, context=4096), tmp_path, FLOOR) == (
        NAME + " is set to a 4,096-token window (context:) and the harness needs at least 13,824 — raise it, "
        "or unset it so the harness sizes the window.")


def test_a_configs_window_under_the_floor_names_the_config_file(tmp_path, monkeypatch):
    _world(monkeypatch)
    spec = _spec(tmp_path, lambda shard: strata_config(shard, args=["--native", str(shard), "--max-context", "8192"]))
    assert _refusal(spec, tmp_path, FLOOR) == (
        NAME + " is set to an 8,192-token window (--max-context in strata-swift-IQ3_XXS.json) and the harness "
        "needs at least 13,824 — raise it, or unset it so the harness sizes the window.")


def test_a_window_exactly_at_the_floor_launches(tmp_path, monkeypatch):
    _world(monkeypatch)
    assert _plan(_spec(tmp_path, context=FLOOR), tmp_path, FLOOR).report.mode == "record"


def test_a_config_with_no_window_and_no_record_context_is_refused_and_a_record_context_covers_it(tmp_path, monkeypatch):
    _world(monkeypatch)
    spec = _spec(tmp_path, lambda shard: strata_config(shard, args=["--native", str(shard)]))
    assert _refusal(spec, tmp_path) == (
        NAME + "'s Strata config sets no --max-context — re-run Strata's setup with --context, "
        "or set context: on the record.")
    assert _plan(spec.model_copy(update={"context": 32768}), tmp_path).report.asked == 32768


def test_a_record_window_is_never_a_fit_so_a_retry_cannot_halve_it(tmp_path, monkeypatch):
    _world(monkeypatch)
    plan = _plan(_spec(tmp_path, context=32768), tmp_path)
    assert (plan.report.mode, plan.full_fit, plan.serve_kwargs["context"], plan.retry()) == ("record", None, None, None)
    assert strata.launch_context(plan, plan.inputs[0]) == 32768


def strata_tree_args(spec) -> list[str]:
    return json.loads((Path(spec.engine_dir) / spec.engine_config).read_text("utf-8"))["args"]


def test_the_argv_and_the_copy_gain_no_flag_and_differ_from_the_config_only_in_the_window(tmp_path, monkeypatch):
    _world(monkeypatch)
    spec = _spec(tmp_path, context=32768)
    plan = _plan(spec, tmp_path)
    config = strata_tree_args(spec)
    copy = strata.copy_of(
        plan.inputs[1].engine, spec, context=strata.launch_context(plan, spec), port=spec.port, config_dir=tmp_path)
    assert list(plan.argv) == strata.argv(spec, config_copy=str(strata.copy_path(tmp_path, spec.port)))
    at = config.index("--max-context")
    assert copy["args"] == [*config[:at + 1], "32768", *config[at + 2:]]
