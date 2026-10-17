"""The fit planner's price: where a model's weights live and the window that leaves, at the box's measured figures."""
from __future__ import annotations

import pytest

from localharness.config.models import AutosenseConfig
from localharness.provider import autosense
from localharness.provider.autosense import FIT_KEYS, fit_header, read_split
from localharness.provider.gguf_header import read_header
from tests.unit.gguf_support import load_fixture, write_recorded

FLASH, SWIFT, NVIDIA = "flash-next-iq3-xxs", "swift-27b-nvfp4-q8mix", "nvidia-27b-nvfp4"
GIB = 2**30
AUTO = AutosenseConfig()
MEM_TOTAL = 64_139_128 * 1024
SWIFT_MMPROJ, FLASH_MMPROJ = 927_606_912, 907_543_520


def _model(tmp_path, name):
    path = write_recorded(tmp_path, load_fixture(name))[0]
    header = fit_header(read_header(path, FIT_KEYS))
    return read_split(path, header), header


def _mem(available_gib):
    return {"MemTotal": MEM_TOTAL, "MemAvailable": int(available_gib * GIB)}


def _place_27b(tmp_path, free, *, name=SWIFT, mmproj=SWIFT_MMPROJ, header_over=None, **kw):
    split, header = _model(tmp_path, name)
    return autosense.place_llamacpp(
        split, {**header, **(header_over or {})}, name="Swift-27B", weights=0, mmproj=mmproj, free_bytes=free,
        meminfo=_mem(52.9), cache_type_k="q8_0", cache_type_v="q8_0", share=1 + 1 / 16, autosense=AUTO, **kw)


def _place_flash(tmp_path, available_gib, free=int(30.2 * GIB), **kw):
    split, header = _model(tmp_path, FLASH)
    return autosense.place_llamacpp(
        split, header, name="Flash-Next", weights=0, mmproj=FLASH_MMPROJ, free_bytes=free,
        meminfo=_mem(available_gib), cache_type_k="q8_0", cache_type_v="q8_0", share=1.0, autosense=AUTO, **kw)


def test_the_swift_27b_prices_296960_before_the_cap_and_fits_the_trained_window_on_the_gpu(tmp_path):
    free = int(30.2 * GIB)
    uncapped = _place_27b(tmp_path, free, header_over={"context_length": 10**9})
    assert uncapped.fits == 296_960
    placed = _place_27b(tmp_path, free)
    assert (placed.fits, placed.where, placed.experts_ram, placed.refusal) == (262_144, "gpu", 0, None)
    assert placed.mmproj == SWIFT_MMPROJ and placed.gpu_bytes == 18_562_465_856 + SWIFT_MMPROJ
    assert placed.kv_per_token == 34_816 * (1 + 1 / 16)


def test_flash_next_puts_its_experts_in_ram_and_gives_the_prompt_cache_what_is_left(tmp_path):
    placed = _place_flash(tmp_path, 52.9)
    assert (placed.fits, placed.where, placed.refusal) == (262_144, "experts in RAM", None)
    assert placed.experts_ram == 42_912_972_800
    assert placed.cache_ram == 3_849 and 3.7 <= placed.cache_ram / 1024 < 3.8


def test_an_unreadable_meminfo_gives_no_cache_ram_on_the_gpu(tmp_path):
    split, header = _model(tmp_path, SWIFT)
    placed = autosense.place_llamacpp(
        split, header, name="Swift-27B", weights=0, mmproj=SWIFT_MMPROJ, free_bytes=int(30.2 * GIB), meminfo=None,
        cache_type_k="q8_0", cache_type_v="q8_0", share=1.0, autosense=AUTO)
    assert placed.where == "gpu" and placed.cache_ram is None


def _free_for(window, *, gpu=18_562_465_856 + SWIFT_MMPROJ, kv=36_992):
    return gpu + int(AUTO.compute_margin_ratio * gpu) + window * kv


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
    assert (ok.fits, ok.refusal) == (262_144, None)


def test_a_window_under_the_floor_is_the_floor_refusal(tmp_path):
    placed = _place_27b(tmp_path, _free_for(9_216) + 1, floor=15_360)
    assert placed.fits == 9_216 and placed.refusal == (
        "Swift-27B fits only a 9,216-token window here and the harness needs at least 15,360 — free GPU memory, "
        "or pick a smaller model.")


def test_a_window_exactly_at_the_floor_is_placed_and_one_step_below_is_refused(tmp_path):
    at = _place_27b(tmp_path, _free_for(15_360), floor=15_360)
    assert (at.fits, at.refusal) == (15_360, None)
    below = _place_27b(tmp_path, _free_for(15_360) - 1, floor=15_360)
    assert below.fits == 15_104 and "fits only a 15,104-token window" in below.refusal


def test_a_gpu_window_under_the_floor_moves_the_experts_to_ram(tmp_path):
    split, _ = _model(tmp_path, FLASH)
    free = split.gpu + FLASH_MMPROJ + int(AUTO.compute_margin_ratio * (split.gpu + FLASH_MMPROJ)) + 20_000 * 16_320
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
