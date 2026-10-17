# ruff: noqa: F811  (the `popen` fixture is imported by name, then taken as a parameter)
"""The launch plan prices its window (or refuses) before anything starts: Flash-Next, the modes, the floor and the retry."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from localharness.agent import context as agent_context
from localharness.config.models import AutosenseConfig, ManagedServerConfig
from localharness.provider import autosense, lifecycle, server
from localharness.provider.autosense import FreeMemory
from tests.unit.gguf_support import load_fixture, write_recorded, write_sparse
from tests.unit.launch_support import popen  # noqa: F401

GIB = 2**30
AUTO = AutosenseConfig()
MEM_TOTAL = 64_139_128 * 1024
FLASH_MMPROJ = 907_543_520
SWIFT_MMPROJ = 927_606_912
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


def _swift(tmp_path, **over):
    model = write_recorded(tmp_path, load_fixture("swift-27b-nvfp4-q8mix"))[0]
    mmproj = write_sparse(tmp_path / "mmproj.gguf", SWIFT_MMPROJ)
    return ManagedServerConfig(**{"runtime": "llamacpp", "binary": "/x/llama-server", "model": str(model),
                                  "mmproj": str(mmproj), "served_name": "Swift-27B", "port": 8080, **over})


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


def test_a_dense_27b_at_the_boxs_free_figure_plans_the_window_that_stays_resident(tmp_path, monkeypatch):
    _world(monkeypatch)
    plan = _plan(_swift(tmp_path), tmp_path)
    assert _pair(list(plan.argv), "-c") == "209920" and plan.report.placement.where == "gpu"


def test_no_mmap_in_extra_args_shrinks_the_priced_prompt_cache(tmp_path, monkeypatch):
    _world(monkeypatch)
    plain = _pair(list(_plan(_flash(tmp_path), tmp_path).argv), "--cache-ram")
    unmapped = _pair(list(_plan(_flash(tmp_path, extra_args=["--no-mmap"]), tmp_path).argv), "--cache-ram")
    assert plain == "3849" and int(unmapped) < int(plain)


@pytest.mark.parametrize("over", [{}, {"context": 32_768}, {"extra_args": ["-c", "32768"]}])
def test_flash_next_without_the_ram_for_its_experts_is_one_refusal_sentence(tmp_path, monkeypatch, over):
    _world(monkeypatch, available=31.4)
    with pytest.raises(server.LaunchRefused) as refused:
        _plan(_flash(tmp_path, **over), tmp_path)
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
    return gpu + int(AUTO.compute_margin_ratio * gpu) + int(AUTO.compute_reserve * GIB) + window * KV


def test_a_priced_window_exactly_at_the_floor_launches_and_one_step_below_is_the_floor_sentence(tmp_path, monkeypatch):
    _world(monkeypatch, free=_free_for(FLOOR))
    assert _pair(list(_plan(_flash(tmp_path), tmp_path, FLOOR).argv), "-c") == "15360"
    _world(monkeypatch, free=_free_for(FLOOR) - 1)
    with pytest.raises(server.LaunchRefused) as refused:
        _plan(_flash(tmp_path), tmp_path, FLOOR)
    assert str(refused.value) == (
        "Flash-Next fits only a 15,104-token window here and the harness needs at least 15,360 — a q4_0 KV cache "
        "fits it (cache_type_k and cache_type_v: q4_0).")


def test_a_record_context_below_the_floor_is_the_record_context_sentence(tmp_path, monkeypatch):
    _world(monkeypatch)
    with pytest.raises(server.LaunchRefused) as refused:
        _plan(_flash(tmp_path, context=4096), tmp_path, FLOOR)
    assert str(refused.value) == (
        "Flash-Next is set to a 4,096-token window (context:) and the harness needs at least 15,360 — raise it, "
        "or unset it so the harness sizes the window.")
    assert _pair(list(_plan(_flash(tmp_path, context=4096), tmp_path).argv), "-c") == "4096"


@pytest.mark.parametrize("extra", [["-c", "4096"], ["--ctx-size=4096"]])
def test_a_context_in_extra_args_below_the_floor_is_refused_like_the_record_field(tmp_path, monkeypatch, extra):
    _world(monkeypatch)
    with pytest.raises(server.LaunchRefused, match=r"is set to a 4,096-token window \(-c in extra_args\)"):
        _plan(_flash(tmp_path, extra_args=extra), tmp_path, FLOOR)


@pytest.mark.parametrize("extra", [["-c", "0"], ["-c", "20000"], ["-c", "\u00b2"]])
def test_a_context_in_extra_args_at_or_above_the_floor_or_zero_is_left_alone(tmp_path, monkeypatch, extra):
    _world(monkeypatch)
    assert _plan(_flash(tmp_path, extra_args=extra), tmp_path, FLOOR).report.mode == "extra_args"


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


CHECKPOINT = Path(__file__).resolve().parents[1] / "fixtures" / "vllm_checkpoints" / "nvidia-qwen3.8-27b-nvfp4"
SAFETENSORS = 21_921_697_280
GPU_TOTAL = int(31.8 * GIB)


def _vllm(tmp_path, *, weights=SAFETENSORS, **over):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    shutil.copy(CHECKPOINT / "config.json", ckpt / "config.json")
    for i in range(2):
        write_sparse(ckpt / f"model-0000{i + 1}-of-00002.safetensors", weights // 2)
    return ManagedServerConfig(**{"runtime": "vllm", "launch": "binary", "binary": "/x/vllm", "model": str(ckpt), **over})


def _gpu(monkeypatch, total=GPU_TOTAL):
    monkeypatch.setattr(autosense, "free_memory", lambda ratios: FreeMemory(total, total, "nvidia-smi"))


def test_a_vllm_checkpoint_plans_max_model_len_from_its_config_and_its_safetensors(tmp_path, monkeypatch):
    _gpu(monkeypatch)
    plan = _plan(_vllm(tmp_path), tmp_path)
    argv = list(plan.argv)
    assert _pair(argv, "--max-model-len") == "100864" and "--trust-remote-code" not in argv
    assert plan.report.mode == "fit" and plan.report.runtime == "vllm" and plan.report.placement.where == "vllm"


def test_the_flag_a_record_or_extra_args_set_wins_and_is_never_doubled(tmp_path, monkeypatch):
    _gpu(monkeypatch)
    extra = _plan(_vllm(tmp_path, extra_args=["--max-model-len", "8192"]), tmp_path)
    assert extra.report.mode == "extra_args" and list(extra.argv).count("--max-model-len") == 1
    assert _pair(list(extra.argv), "--max-model-len") == "8192"
    record = _plan(_vllm(tmp_path, context=32_768), tmp_path)
    assert record.report.mode == "record" and _pair(list(record.argv), "--max-model-len") == "32768"
    with pytest.raises(server.LaunchRefused, match=r"is set to a 4,096-token window \(context:\)"):
        _plan(_vllm(tmp_path, context=4096), tmp_path, FLOOR)


@pytest.mark.parametrize("over", [{}, {"context": 32_768}])
def test_weights_beyond_the_gpu_budget_are_refused_before_a_start(tmp_path, monkeypatch, over):
    _gpu(monkeypatch)
    with pytest.raises(server.LaunchRefused, match=r"ckpt cannot run here: 30.0 GiB of weights, and vLLM keeps them all"):
        _plan(_vllm(tmp_path, weights=30 * GIB, **over), tmp_path)


def test_a_vllm_window_under_the_floor_is_the_floor_sentence(tmp_path, monkeypatch):
    _gpu(monkeypatch)
    with pytest.raises(server.LaunchRefused, match="ckpt fits only a 100,864-token window here and the harness needs at least 150,000"):
        _plan(_vllm(tmp_path), tmp_path, 150_000)


def test_the_vllm_retry_halves_under_the_same_floor_rule_and_drops_the_parsers(tmp_path, monkeypatch):
    _gpu(monkeypatch)
    retry = _plan(_vllm(tmp_path), tmp_path).retry()
    assert _pair(list(retry.argv), "--max-model-len") == "50432" and "--reasoning-parser" not in retry.argv
    held = _plan(_vllm(tmp_path), tmp_path, 60_000).retry()
    assert _pair(list(held.argv), "--max-model-len") == "100864" and "--reasoning-parser" not in held.argv


def test_a_checkpoint_without_safetensors_or_a_gpu_reading_is_not_computed(tmp_path, monkeypatch):
    _gpu(monkeypatch)
    spec = _vllm(tmp_path)
    for f in Path(spec.model).glob("*.safetensors"):
        f.unlink()
    assert _plan(spec, tmp_path).report is None
    monkeypatch.setattr(autosense, "free_memory", lambda ratios: None)
    assert "--max-model-len" not in _plan(_vllm(tmp_path), tmp_path).argv


def test_checkpoint_bytes_sums_the_regular_safetensors_and_follows_a_link_for_its_size_only(tmp_path):
    assert autosense.checkpoint_bytes(tmp_path) is None
    (tmp_path / "blobs").mkdir()
    write_sparse(tmp_path / "blobs" / "abc", 1_000)
    os.symlink(tmp_path / "blobs" / "abc", tmp_path / "model.safetensors")
    write_sparse(tmp_path / "other.safetensors", 24)
    write_sparse(tmp_path / "model.bin", 99)
    (tmp_path / "dir.safetensors").mkdir()
    assert autosense.checkpoint_bytes(tmp_path) == 1_024
    assert autosense.checkpoint_bytes(tmp_path / "missing") is None


async def test_a_vllm_launch_reports_the_window_the_server_serves(tmp_path, monkeypatch, popen):
    _gpu(monkeypatch)

    async def wait_ready(base_url, **kw):
        return ["ckpt"]

    probed = []

    def probe_window(base_url, model, provider_type=None, *, api_key=None):
        probed.append((model, provider_type))
        return agent_context.WindowProbe(100_864, 100_864)

    monkeypatch.setattr(server, "wait_ready", wait_ready)
    monkeypatch.setattr(agent_context, "probe_window", probe_window)
    live = await lifecycle.ManagedVllmStrategy().activate(_vllm(tmp_path), tmp_path, BASE, autosense=AUTO)
    assert probed == [("ckpt", "vllm")]
    assert (live.window.served, live.window.fits, live.window.asked, live.window.verified) == (100_864, 100_864, 100_864, True)
    assert _pair(popen[0]["cmd"], "--max-model-len") == "100864"
