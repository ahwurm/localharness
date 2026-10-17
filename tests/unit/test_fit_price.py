"""The fit planner's price: where a model's weights live and the window that leaves, at the box's measured figures."""
from __future__ import annotations

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
