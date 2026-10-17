"""The launch plan prices its window (or refuses) before anything starts: Flash-Next, the modes, the floor and the retry."""
from __future__ import annotations

import pytest

from localharness.config.models import AutosenseConfig, ManagedServerConfig
from localharness.provider import autosense, lifecycle, server
from localharness.provider.autosense import FreeMemory
from tests.unit.gguf_support import load_fixture, write_recorded, write_sparse

GIB = 2**30
AUTO = AutosenseConfig()
MEM_TOTAL = 64_139_128 * 1024
FLASH_MMPROJ = 907_543_520
BASE = "http://127.0.0.1:8080/v1"


def _world(monkeypatch, *, available=52.9, free=int(30.2 * GIB), memory=True):
    mem = FreeMemory(free, int(31.8 * GIB), "nvidia-smi") if memory else None
    monkeypatch.setattr(autosense, "free_memory", lambda ratios: mem)
    monkeypatch.setattr(autosense, "_read_meminfo", lambda: {"MemTotal": MEM_TOTAL, "MemAvailable": int(available * GIB)})


def _flash(tmp_path, **over):
    model = write_recorded(tmp_path, load_fixture("flash-next-iq3-xxs"))[0]
    mmproj = write_sparse(tmp_path / "mmproj.gguf", FLASH_MMPROJ)
    return ManagedServerConfig(**{"runtime": "llamacpp", "binary": "/x/llama-server", "model": str(model),
                                  "mmproj": str(mmproj), "served_name": "Flash-Next", "port": 8080, **over})


def _plan(spec, config_dir, floor=None):
    sensed = lifecycle.sense_launch(spec, config_dir, AUTO)
    return lifecycle.plan_launch(
        spec, sensed, server.launch_decision(spec, False), autosense=AUTO, api_key_held=False, floor=floor)


def _pair(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def test_flash_next_plans_its_window_with_the_experts_in_ram(tmp_path, monkeypatch):
    _world(monkeypatch)
    plan = _plan(_flash(tmp_path), tmp_path)
    argv = list(plan.argv)
    assert _pair(argv, "-c") == "262144" and _pair(argv, "--cache-ram") == "3849" and "--fit-ctx" not in argv
    assert plan.report.mode == "fit" and plan.report.placement.where == "experts in RAM"


def test_flash_next_without_the_ram_for_its_experts_is_one_refusal_sentence(tmp_path, monkeypatch):
    _world(monkeypatch, available=31.4)
    with pytest.raises(server.LaunchRefused) as refused:
        _plan(_flash(tmp_path), tmp_path)
    assert str(refused.value) == (
        "Flash-Next cannot run here: its experts need 40.0 GiB of RAM and only 22.2 GiB is free beside the "
        "system's reserve — close something that holds RAM, or pick a smaller quant.")


async def test_a_refused_activate_starts_and_writes_nothing(tmp_path, monkeypatch):
    _world(monkeypatch, available=31.4)
    touched: list[str] = []
    monkeypatch.setattr(server, "start_server", lambda *a, **kw: touched.append("start_server"))
    monkeypatch.setattr(server, "prepare_config_ini_home", lambda *a, **kw: touched.append("prepare_config_ini_home"))
    monkeypatch.setattr(server, "stop_server", lambda *a, **kw: touched.append("stop_server"))
    config = tmp_path / "cfg"
    config.mkdir()
    with pytest.raises(server.LaunchRefused, match="Flash-Next cannot run here"):
        await lifecycle.SpawnedProcessStrategy().activate(_flash(tmp_path), config, BASE, autosense=AUTO)
    assert touched == [] and list(config.iterdir()) == []


FLASH_GPU = 3_968_785_920
FLOOR = 15_360
KV = 16_320


def _free_for(window):
    gpu = FLASH_GPU + FLASH_MMPROJ
    return gpu + int(AUTO.compute_margin_ratio * gpu) + window * KV


def test_a_priced_window_exactly_at_the_floor_launches_and_one_step_below_is_the_floor_sentence(tmp_path, monkeypatch):
    _world(monkeypatch, free=_free_for(FLOOR))
    assert _pair(list(_plan(_flash(tmp_path), tmp_path, FLOOR).argv), "-c") == "15360"
    _world(monkeypatch, free=_free_for(FLOOR) - 1)
    with pytest.raises(server.LaunchRefused) as refused:
        _plan(_flash(tmp_path), tmp_path, FLOOR)
    assert str(refused.value) == (
        "Flash-Next fits only a 15,104-token window here and the harness needs at least 15,360 — free GPU memory, "
        "or pick a smaller model.")


def test_a_record_context_below_the_floor_is_the_record_context_sentence(tmp_path, monkeypatch):
    _world(monkeypatch)
    with pytest.raises(server.LaunchRefused) as refused:
        _plan(_flash(tmp_path, context=4096), tmp_path, FLOOR)
    assert str(refused.value) == (
        "Flash-Next is set to a 4,096-token window (context:) and the harness needs at least 15,360 — raise it, "
        "or unset it so the harness sizes the window.")
    assert _pair(list(_plan(_flash(tmp_path, context=4096), tmp_path).argv), "-c") == "4096"


def test_a_name_with_a_control_character_reaches_a_sentence_without_it(tmp_path, monkeypatch):
    _world(monkeypatch)
    with pytest.raises(server.LaunchRefused) as refused:
        _plan(_flash(tmp_path, context=4096, served_name="Flash\x1b-Next"), tmp_path, FLOOR)
    assert str(refused.value).startswith("Flash-Next is set to")


def test_a_model_that_cannot_be_read_launches_by_the_server_with_the_floor_as_its_fit_ctx(tmp_path, monkeypatch):
    _world(monkeypatch)
    spec = _flash(tmp_path)
    open(spec.model, "wb").write(b"not a gguf")
    argv = list(_plan(spec, tmp_path, FLOOR).argv)
    assert _pair(argv, "--fit-ctx") == "15360" and "-c" not in argv
    assert _plan(spec, tmp_path).report.mode == "fit-by-server"
    assert "--fit-ctx" not in _plan(spec, tmp_path).argv


@pytest.mark.parametrize("extra", [["-ot", "exps=CPU"], ["--n-cpu-moe", "48"], ["-ngl", "10"]])
def test_a_record_placing_tensors_by_hand_is_not_computed(tmp_path, monkeypatch, extra):
    _world(monkeypatch)
    plan = _plan(_flash(tmp_path, extra_args=extra), tmp_path, FLOOR)
    assert plan.report.mode == "not-computed" and "-c" not in plan.argv and "--fit-ctx" not in plan.argv


def test_an_unknown_free_gpu_memory_is_not_computed(tmp_path, monkeypatch):
    _world(monkeypatch, memory=False)
    plan = _plan(_flash(tmp_path), tmp_path, FLOOR)
    assert plan.report.mode == "not-computed" and "-c" not in plan.argv and "--fit-ctx" not in plan.argv


def test_a_hand_ngl_beyond_the_gpu_is_the_ngl_sentence(tmp_path, monkeypatch):
    _world(monkeypatch)
    with pytest.raises(server.LaunchRefused) as refused:
        _plan(_flash(tmp_path, extra_args=["-ngl", "99"]), tmp_path)
    assert str(refused.value) == (
        "Flash-Next sets -ngl in server.extra_args, which turns llama-server's --fit off, and its experts do not "
        "fit the GPU — drop the -ngl, or add --n-cpu-moe to extra_args.")


def test_the_halve_once_retry_halves_to_the_floor_and_not_below_it(tmp_path, monkeypatch):
    _world(monkeypatch, free=_free_for(30_720))
    retry = _plan(_flash(tmp_path), tmp_path, FLOOR).retry()
    assert retry is not None and _pair(list(retry.argv), "-c") == "15360"
    _world(monkeypatch, free=_free_for(30_464))
    assert _plan(_flash(tmp_path), tmp_path, FLOOR).retry() is None
    _world(monkeypatch, free=_free_for(30_720))
    assert _pair(list(_plan(_flash(tmp_path), tmp_path).retry().argv), "-c") == "15360"


def test_no_floor_is_no_plan_time_refusal_and_no_fit_ctx(tmp_path, monkeypatch):
    _world(monkeypatch, free=_free_for(2_048))
    plan = _plan(_flash(tmp_path), tmp_path)
    assert _pair(list(plan.argv), "-c") == "2048" and "--fit-ctx" not in plan.argv


def test_the_harness_places_nothing_by_hand(tmp_path, monkeypatch):
    _world(monkeypatch)
    argv = list(_plan(_flash(tmp_path), tmp_path, FLOOR).argv)
    assert not {"-ngl", "--n-cpu-moe", "-ot", "-t"} & set(argv)


def test_planning_twice_gives_equal_plans_or_the_same_refusal(tmp_path, monkeypatch):
    _world(monkeypatch)
    spec = _flash(tmp_path)
    assert _plan(spec, tmp_path, FLOOR) == _plan(spec, tmp_path, FLOOR)
    _world(monkeypatch, available=31.4)
    sentences = []
    for _ in range(2):
        with pytest.raises(server.LaunchRefused) as refused:
            _plan(spec, tmp_path, FLOOR)
        sentences.append(str(refused.value))
    assert sentences[0] == sentences[1]


@pytest.mark.parametrize("strategy", [
    lifecycle.ManagedVllmStrategy, lifecycle.SpawnedProcessStrategy, lifecycle.DaemonStrategy,
    lifecycle.LmsStrategy, lifecycle.UnitStrategy])
def test_every_activate_takes_a_floor(strategy):
    import inspect

    assert inspect.signature(strategy.activate).parameters["floor"].default is None
