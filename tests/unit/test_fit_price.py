"""The fit planner's price: where a model's weights live and the window that leaves, at the box's measured figures."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from localharness.config.models import AutosenseConfig
from localharness.provider import autosense
from localharness.provider.autosense import FIT_KEYS, fit_header, read_split
from localharness.provider.gguf_header import read_header
from tests.unit.gguf_support import load_fixture, write_recorded
from tests.unit.test_floor_machine_level import _layers, _warnings

FLASH, SWIFT, NVIDIA = "flash-next-iq3-xxs", "swift-27b-nvfp4-q8mix", "nvidia-27b-nvfp4"
GIB = 2**30
CHECKPOINTS = Path(__file__).resolve().parents[1] / "fixtures" / "vllm_checkpoints"
AUTO = AutosenseConfig()
MEM_TOTAL = 64_139_128 * 1024
SWIFT_MMPROJ, FLASH_MMPROJ = 927_606_912, 907_543_520


def _model(tmp_path, name):
    path = write_recorded(tmp_path, load_fixture(name))[0]
    header = fit_header(read_header(path, FIT_KEYS))
    return read_split(path, header), header


def _mem(available_gib):
    return {"MemTotal": MEM_TOTAL, "MemAvailable": int(available_gib * GIB)}


def _place_27b(tmp_path, free, *, name=SWIFT, mmproj=SWIFT_MMPROJ, header_over=None, auto=AUTO, cache="q8_0",
               share=1 + 1 / 16, **kw):
    split, header = _model(tmp_path, name)
    return autosense.place_llamacpp(
        split, {**header, **(header_over or {})}, name="Swift-27B", weights=0, mmproj=mmproj, free_bytes=free,
        meminfo=_mem(52.9), cache_type_k=cache, cache_type_v=cache, share=share, autosense=auto, **kw)


def _place_flash(tmp_path, available_gib, free=int(30.2 * GIB), **kw):
    split, header = _model(tmp_path, FLASH)
    return autosense.place_llamacpp(
        split, header, name="Flash-Next", weights=0, mmproj=FLASH_MMPROJ, free_bytes=free,
        meminfo=_mem(available_gib), cache_type_k="q8_0", cache_type_v="q8_0", share=1.0, autosense=AUTO, **kw)


def test_the_swift_27b_prices_209920_with_the_reserve_held_back_and_the_trained_window_without_it(tmp_path):
    free = int(30.2 * GIB)
    placed = _place_27b(tmp_path, free)
    assert (placed.fits, placed.where, placed.experts_ram, placed.refusal) == (209_920, "gpu", 0, None)
    flat = AutosenseConfig(compute_reserve=0.0)
    assert _place_27b(tmp_path, free, header_over={"context_length": 10**9}, auto=flat).fits == 296_960
    assert _place_27b(tmp_path, free, auto=flat).fits == 262_144
    assert placed.mmproj == SWIFT_MMPROJ and placed.gpu_bytes == 18_562_465_856 + SWIFT_MMPROJ
    assert placed.kv_per_token == 34_816 * (1 + 1 / 16)


def test_the_compute_reserve_is_three_gib_bounded_and_machine_level(tmp_path):
    assert AutosenseConfig().compute_reserve == 3.0
    for bad in (-1, 65):
        with pytest.raises(ValueError):
            AutosenseConfig(compute_reserve=bad)
    assert AutosenseConfig(compute_reserve=0).compute_reserve == 0
    loader, ws = _layers(tmp_path, ws_cfg={"autosense": {"compute_reserve": 0.0}})
    assert loader.load_harness().autosense.compute_reserve == 3.0
    assert _warnings(loader) == [f"ignoring autosense in {ws / 'config.yaml'}: only the global config may set it"]


def test_flash_next_puts_its_experts_in_ram_and_gives_the_prompt_cache_what_is_left(tmp_path):
    placed = _place_flash(tmp_path, 52.9)
    assert (placed.fits, placed.where, placed.refusal) == (262_144, "experts in RAM", None)
    assert placed.experts_ram == 42_912_972_800
    assert placed.cache_ram == 3_849 and 3.7 <= placed.cache_ram / 1024 < 3.8


def test_no_mmap_counts_the_whole_file_as_host_resident_in_the_prompt_cache(tmp_path):
    split, header = _model(tmp_path, SWIFT)
    weights = 20 * GIB

    def cache_ram(**kw):
        return autosense.place_llamacpp(
            split, header, name="Swift-27B", weights=weights, mmproj=SWIFT_MMPROJ, free_bytes=int(30.2 * GIB),
            meminfo=_mem(52.9), cache_type_k="q8_0", cache_type_v="q8_0", share=1.0, autosense=AUTO, **kw).cache_ram

    mapped, unmapped = cache_ram(), cache_ram(no_mmap=True)
    assert mapped - unmapped == int(AUTO.cache_ram_ratio * weights) // 2**20


def test_an_unreadable_meminfo_gives_no_cache_ram_on_the_gpu(tmp_path):
    split, header = _model(tmp_path, SWIFT)
    placed = autosense.place_llamacpp(
        split, header, name="Swift-27B", weights=0, mmproj=SWIFT_MMPROJ, free_bytes=int(30.2 * GIB), meminfo=None,
        cache_type_k="q8_0", cache_type_v="q8_0", share=1.0, autosense=AUTO)
    assert placed.where == "gpu" and placed.cache_ram is None


def _free_for(window, *, gpu=18_562_465_856 + SWIFT_MMPROJ, kv=36_992):
    return gpu + int(AUTO.compute_margin_ratio * gpu) + int(AUTO.compute_reserve * GIB) + window * kv


def test_the_ram_refusal_is_one_sentence_at_the_episodes_figures(tmp_path):
    placed = _place_flash(tmp_path, 31.4)
    assert (placed.fits, placed.where) == (None, None)
    assert placed.refusal == (
        "Flash-Next cannot run here: its experts need 40.0 GiB of RAM and only 22.2 GiB is free beside the "
        "system's reserve — close something that holds RAM, or pick a smaller quant.")


def test_the_vram_refusal_is_one_sentence_at_the_episodes_figures(tmp_path):
    split, header = _model(tmp_path, NVIDIA)
    placed = autosense.place_llamacpp(
        split, header, name="Qwen3.8-27B-Nvidia-NVFP4", weights=0, mmproj=0, free_bytes=int(12.0 * GIB),
        meminfo=_mem(52.9), cache_type_k="q8_0", cache_type_v="q8_0", share=1.0, autosense=AUTO, floor=15_360)
    assert placed.fits is None and placed.refusal == (
        "Qwen3.8-27B-Nvidia-NVFP4 cannot run here: it needs 17.6 GiB of GPU memory plus a 15,360-token window "
        "and 12.0 GiB is free — stop what holds the GPU, or pick a smaller quant.")


def test_the_vram_refusal_without_a_floor_names_no_window(tmp_path):
    placed = _place_27b(tmp_path, int(12.0 * GIB))
    assert placed.refusal == (
        "Swift-27B cannot run here: it needs 18.2 GiB of GPU memory and 12.0 GiB is free — stop what holds the "
        "GPU, or pick a smaller quant.")


def test_an_unreadable_meminfo_refuses_experts_in_ram(tmp_path):
    split, header = _model(tmp_path, FLASH)
    placed = autosense.place_llamacpp(
        split, header, name="Flash-Next", weights=0, mmproj=FLASH_MMPROJ, free_bytes=int(30.2 * GIB), meminfo=None,
        cache_type_k="q8_0", cache_type_v="q8_0", share=1.0, autosense=AUTO)
    assert placed.fits is None and placed.cache_ram is None and placed.refusal == (
        "Flash-Next cannot run here: its experts need 40.0 GiB of RAM and the free RAM cannot be read "
        "(/proc/meminfo) — pick a quant whose experts fit the GPU.")


def test_a_hand_set_ngl_with_experts_beyond_the_gpu_is_refused_and_a_fitting_model_is_not(tmp_path):
    placed = _place_flash(tmp_path, 52.9, ngl_hand=True)
    assert placed.fits is None and placed.refusal == (
        "Flash-Next sets -ngl in server.extra_args, which turns llama-server's --fit off, and its experts do not "
        "fit the GPU — drop the -ngl, or add --n-cpu-moe to extra_args.")
    ok = _place_27b(tmp_path, int(30.2 * GIB), ngl_hand=True)
    assert (ok.fits, ok.refusal) == (209_920, None)


def test_a_window_under_the_floor_is_the_floor_refusal(tmp_path):
    placed = _place_27b(tmp_path, _free_for(9_216) + 1, floor=15_360)
    assert placed.fits == 9_216 and placed.refusal == (
        "Swift-27B fits only a 9,216-token window here and the harness needs at least 15,360 — a q4_0 KV cache "
        "fits it (cache_type_k and cache_type_v: q4_0).")


FLOOR_13_8 = 13_824
KV_F16, KV_Q8 = 65_536, 34_816


@pytest.mark.parametrize("cache,share,kv,window,got,remedy", [
    ("f16", 1.0, KV_F16, 10_240, 10_240, "a q8_0 KV cache fits it (cache_type_k and cache_type_v: q8_0)"),
    ("q8_0", 1 + 1 / 16, KV_Q8, FLOOR_13_8, 12_800, "it fits with MTP drafting off (speculative: off)"),
    ("q8_0", 1.0, KV_Q8, 10_240, 10_240, "a q4_0 KV cache fits it (cache_type_k and cache_type_v: q4_0)"),
    ("q4_0", 1.0, 18_432, 5_000, 4_864, "free GPU memory, or pick a smaller model"),
    ("q8_0", 1.0, KV_Q8, 5_000, 4_864, "free GPU memory, or pick a smaller model"),
], ids=["q8-kv", "mtp-off", "q4-kv", "nothing-helps-at-q4", "nothing-helps-at-q8"])
def test_a_floor_refusal_names_the_first_remedy_that_reaches_the_floor(tmp_path, cache, share, kv, window, got, remedy):
    placed = _place_27b(tmp_path, _free_for(window, kv=kv), cache=cache, share=share, floor=FLOOR_13_8)
    assert placed.fits == got and placed.refusal == (
        f"Swift-27B fits only {autosense.window_phrase(got)} here and the harness needs at least 13,824 — {remedy}.")


def test_a_refusal_without_room_for_any_window_names_the_remedy_that_gives_one(tmp_path):
    placed = _place_27b(tmp_path, _free_for(0) + 12_000_000, cache="f16", share=1.0)
    assert placed.fits is None and placed.refusal == (
        "Swift-27B cannot run here: it needs 18.2 GiB of GPU memory and 23.0 GiB is free — a q8_0 KV cache fits "
        "it (cache_type_k and cache_type_v: q8_0).")


def test_a_window_exactly_at_the_floor_is_placed_and_one_step_below_is_refused(tmp_path):
    at = _place_27b(tmp_path, _free_for(15_360), floor=15_360)
    assert (at.fits, at.refusal) == (15_360, None)
    below = _place_27b(tmp_path, _free_for(15_360) - 1, floor=15_360)
    assert below.fits == 15_104 and "fits only a 15,104-token window" in below.refusal


def test_a_gpu_window_under_the_floor_moves_the_experts_to_ram(tmp_path):
    split, _ = _model(tmp_path, FLASH)
    gpu = split.gpu + FLASH_MMPROJ
    free = gpu + int(AUTO.compute_margin_ratio * gpu) + int(AUTO.compute_reserve * GIB) + 20_000 * 16_320
    placed = _place_flash(tmp_path, 52.9, free=free, floor=15_360)
    assert (placed.where, placed.fits) == ("experts in RAM", 19_968) and placed.refusal is None
    on_gpu = _place_flash(tmp_path, 52.9, free=free + split.experts, floor=15_360)
    assert on_gpu.where == "gpu" and on_gpu.experts_ram == 0


def test_experts_equal_to_the_ram_room_fit_and_one_byte_more_is_refused(tmp_path):
    split, header = _model(tmp_path, FLASH)
    total = 20 * 10**9
    reserve = AUTO.unified_reserve_ratio * total
    assert reserve == int(reserve)

    def place(avail):
        return autosense.place_llamacpp(
            split, header, name="Flash-Next", weights=0, mmproj=FLASH_MMPROJ, free_bytes=int(30.2 * GIB),
            meminfo={"MemTotal": total, "MemAvailable": avail}, cache_type_k="q8_0", cache_type_v="q8_0",
            share=1.0, autosense=AUTO)

    edge = split.experts + int(reserve)
    assert (place(edge).where, place(edge).cache_ram) == ("experts in RAM", 0)
    assert "its experts need 40.0 GiB of RAM and only 40.0 GiB is free" in place(edge - 1).refusal


def test_less_available_than_the_reserve_reads_zero_gib_free(tmp_path):
    placed = _place_flash(tmp_path, 5.0)
    assert "and only 0.0 GiB is free beside" in placed.refusal


def test_without_a_split_the_whole_file_is_priced_on_the_gpu_only(tmp_path):
    _, header = _model(tmp_path, FLASH)
    placed = autosense.place_llamacpp(
        None, header, name="Flash-Next", weights=76_873_616_640, mmproj=FLASH_MMPROJ, free_bytes=int(30.2 * GIB),
        meminfo=_mem(52.9), cache_type_k="q8_0", cache_type_v="q8_0", share=1.0, autosense=AUTO)
    assert (placed.gpu_bytes, placed.experts_ram, placed.where) == (76_873_616_640, 0, None)
    assert placed.refusal is not None and "it needs 71.6 GiB of GPU memory" in placed.refusal
    assert placed.kv_per_token == 48 * 2 * 256 * (2 * 34 / 32)


@pytest.mark.parametrize("n,phrase", [
    (4_096, "a 4,096-token window"), (8_448, "an 8,448-token window"), (11_264, "an 11,264-token window"),
    (18_432, "an 18,432-token window"), (110_592, "a 110,592-token window"), (15_360, "a 15,360-token window"),
    (80_000, "an 80,000-token window"), (800, "an 800-token window"), (118_000, "a 118,000-token window"),
])
def test_window_phrase_takes_the_article_the_number_is_read_with(n, phrase):
    assert autosense.window_phrase(n) == phrase


def test_a_control_character_in_a_name_never_reaches_a_sentence(tmp_path):
    split, header = _model(tmp_path, NVIDIA)
    placed = autosense.place_llamacpp(
        split, header, name="Bad\x1b[31m\nName", weights=0, mmproj=0, free_bytes=GIB, meminfo=_mem(52.9),
        cache_type_k=None, cache_type_v=None, share=1.0, autosense=AUTO)
    assert placed.refusal.startswith("Bad[31mName cannot run here: ") and "\x1b" not in placed.refusal
    assert "\n" not in placed.refusal


def test_the_same_inputs_priced_twice_give_equal_placements(tmp_path):
    assert _place_flash(tmp_path, 52.9) == _place_flash(tmp_path, 52.9)


NVFP4 = {"nvidia-qwen3.8-27b-nvfp4": 21_921_697_280, "ukisai-swift-1.5-27b-nvfp4": 21_921_695_568}
GPU_31_8 = int(31.8 * GIB)


def _config(name):
    return json.loads((CHECKPOINTS / name / "config.json").read_text())


def _vllm(config, *, weights=21_921_697_280, total=GPU_31_8, extra_args=(), floor=None, name="NVFP4-27B"):
    return autosense.place_vllm(
        config, name=name, weights=weights, total_bytes=total, extra_args=list(extra_args), autosense=AUTO, floor=floor)


@pytest.mark.parametrize("name,weights", NVFP4.items())
def test_both_nvfp4_checkpoints_price_at_100864(name, weights):
    placed = _vllm(_config(name), weights=weights)
    assert (placed.fits, placed.where, placed.refusal) == (100_864, "vllm", None)
    assert autosense.gib(placed.budget_bytes) == "28.6" and placed.kv_per_token == 65_536
    assert placed.gpu_bytes == weights and placed.mmproj == 0 and placed.experts_ram == 0


def test_two_checkpoints_of_the_same_shape_price_equal():
    a, b = (_vllm(_config(n)) for n in NVFP4)
    assert a == b == _vllm(_config("nvidia-qwen3.8-27b-nvfp4"))


GEMMA_BF16, GEMMA_CARD = 62_546_338_264, 32_607 * 2**20


def test_a_bf16_snapshot_beyond_the_budget_is_the_vllm_refusal_naming_a_4_bit_build():
    placed = _vllm(_config("davidau-gemma-4-31b-it"), weights=GEMMA_BF16, total=GEMMA_CARD, name="gemma")
    assert placed.fits is None and placed.where is None and placed.refusal == (
        "gemma cannot run here: 58.3 GiB of bf16 weights, and vLLM keeps them all on the 31.8 GiB GPU — a 4-bit "
        "build of it (a GGUF Q4_K_M for llama.cpp, or NVFP4 for vLLM) would fit.")


def test_a_bf16_snapshot_no_4_bit_build_would_fit_says_pick_a_smaller_quant():
    placed = _vllm(_config("davidau-gemma-4-31b-it"), weights=140 * GIB, total=GEMMA_CARD, name="gemma")
    assert placed.refusal == (
        "gemma cannot run here: 140.0 GiB of bf16 weights, and vLLM keeps them all on the 31.8 GiB GPU — pick a "
        "smaller quant.")


def test_a_quantized_checkpoint_beyond_the_budget_names_no_dtype_and_no_4_bit_build():
    placed = _vllm(_config("nvidia-qwen3.8-27b-nvfp4"), weights=40 * GIB, name="NVFP4-27B")
    assert placed.refusal == (
        "NVFP4-27B cannot run here: 40.0 GiB of weights, and vLLM keeps them all on the 31.8 GiB GPU — pick a "
        "smaller quant.")


def test_an_fp16_config_names_fp16_and_a_dtype_less_one_names_none():
    base = {"num_hidden_layers": 8, "num_key_value_heads": 1, "head_dim": 16}
    assert "58.3 GiB of fp16 weights" in _vllm({**base, "torch_dtype": "float16"}, weights=GEMMA_BF16, total=GEMMA_CARD).refusal
    assert "58.3 GiB of weights" in _vllm(base, weights=GEMMA_BF16, total=GEMMA_CARD).refusal


def test_weights_plus_the_margin_at_the_budget_is_refused_and_one_byte_under_prices():
    config, total = _config("nvidia-qwen3.8-27b-nvfp4"), 10 * 10**9
    weights = 8_181_818_182
    assert weights + int(AUTO.compute_margin_ratio * weights) == int(0.9 * total)
    assert _vllm(config, weights=weights, total=total).refusal is not None
    under = _vllm(config, weights=weights - 1, total=total)
    assert under.refusal is None and under.fits is None


def test_a_config_with_no_layer_types_counts_num_hidden_layers():
    config = {"num_hidden_layers": 64, "num_key_value_heads": 4, "head_dim": 256, "max_position_embeddings": 262_144}
    assert _vllm(config).kv_per_token == 64 * 2 * 4 * 256 * 2


def test_text_config_values_win_over_top_level_ones():
    config = {"num_hidden_layers": 8, "num_key_value_heads": 1, "head_dim": 16, "max_position_embeddings": 1_000,
              "text_config": {"num_hidden_layers": 64, "num_key_value_heads": 4, "head_dim": 256}}
    placed = _vllm(config)
    assert placed.kv_per_token == 64 * 2 * 4 * 256 * 2 and placed.fits == 1_000


def test_the_head_dim_falls_back_to_hidden_size_over_heads_and_the_window_is_capped_at_the_trained_one():
    config = {"num_hidden_layers": 16, "num_key_value_heads": 4, "hidden_size": 2_048, "num_attention_heads": 8,
              "max_position_embeddings": 65_536}
    placed = _vllm(config)
    assert placed.kv_per_token == 16 * 2 * 4 * 256 * 2 and placed.fits == 65_536


@pytest.mark.parametrize("dtype", ["fp8", "fp8_e4m3", "FP8_E5M2"])
def test_an_fp8_kv_cache_halves_the_kv_bytes(dtype):
    config = _config("nvidia-qwen3.8-27b-nvfp4")
    assert _vllm(config, extra_args=["--kv-cache-dtype", dtype]).kv_per_token == 32_768
    assert _vllm(config, extra_args=[f"--kv-cache-dtype={dtype}"]).kv_per_token == 32_768
    assert _vllm(config, extra_args=["--kv-cache-dtype", "auto"]).kv_per_token == 65_536


def test_the_gpu_memory_utilization_in_extra_args_replaces_the_default():
    config = _config("nvidia-qwen3.8-27b-nvfp4")
    assert _vllm(config, extra_args=["--gpu-memory-utilization=0.8"]).budget_bytes == int(0.8 * GPU_31_8)
    assert _vllm(config, extra_args=["--gpu-memory-utilization", "0.95"]).budget_bytes == int(0.95 * GPU_31_8)
    for bad in ("1.5", "0", "-0.1", "abc", "nan"):
        assert _vllm(config, extra_args=["--gpu-memory-utilization", bad]).budget_bytes == int(0.9 * GPU_31_8)


@pytest.mark.parametrize("hostile", [True, "4", 4.0, 0, -4, None, [4]])
def test_a_value_that_is_not_a_positive_int_prices_nothing(hostile):
    for key in ("num_key_value_heads", "head_dim"):
        config = {"num_hidden_layers": 64, "num_key_value_heads": 4, "head_dim": 256, key: hostile}
        placed = _vllm(config)
        assert (placed.fits, placed.where, placed.refusal) == (None, None, None)


def test_no_heads_or_head_dim_anywhere_prices_nothing_without_a_refusal():
    placed = _vllm({"num_hidden_layers": 64})
    assert (placed.fits, placed.refusal, placed.kv_per_token) == (None, None, None)


def test_a_vllm_window_under_the_floor_is_the_floor_refusal():
    placed = _vllm(_config("nvidia-qwen3.8-27b-nvfp4"), floor=200_000)
    assert placed.fits == 100_864 and placed.refusal == (
        "NVFP4-27B fits only a 100,864-token window here and the harness needs at least 200,000 — free GPU "
        "memory, or pick a smaller model.")
    assert _vllm(_config("nvidia-qwen3.8-27b-nvfp4"), floor=100_864).refusal is None


def test_a_control_character_in_a_vllm_name_never_reaches_the_refusal():
    placed = _vllm(_config("nvidia-qwen3.8-27b-nvfp4"), weights=GPU_31_8, name="x\x1b[0m\ny")
    assert placed.refusal.startswith("x[0my cannot run here: ")

