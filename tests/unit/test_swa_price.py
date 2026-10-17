"""Sliding-window layers priced at the cache llama.cpp and vLLM give them, on the box's recorded gemma4 Q4_K_M."""
from __future__ import annotations

import pytest

from localharness.config.models import ManagedServerConfig
from localharness.provider import autosense, lifecycle, server
from localharness.provider.autosense import (FIT_KEYS, FreeMemory, WindowReport, fit_header, read_split,
                                             swa_bytes_per_cell, swa_cells_cap, swa_pattern)
from localharness.provider.gguf_header import kv_layer_ids, read_header, read_tensor_table
from tests.unit.gguf_support import load_fixture, write_gguf, write_recorded
from tests.unit.test_fit_price import AUTO, GIB, NVIDIA, SWIFT, FLASH, _mem, _model

GEMMA = "gemma4-31b-q4-k-m"
FREE = int(30.2 * GIB)
KV = "gemma4.attention."


def _recorded() -> dict:
    return {key: value for key, _, value in load_fixture(GEMMA)["shards"][0]["kv"]}


def _sums() -> tuple[float, float]:
    """Per-token bytes of the full layers and per-cell bytes of the sliding ones, from the recorded header keys."""
    kv = _recorded()
    heads, pattern = kv[KV + "head_count_kv"][1], kv[KV + "sliding_window_pattern"][1]
    full = sum(heads[i] * (kv[KV + "key_length"] + kv[KV + "value_length"]) * 34 / 32
               for i in range(kv["gemma4.block_count"]) if not pattern[i])
    sliding = sum(heads[i] * (kv[KV + "key_length_swa"] + kv[KV + "value_length_swa"]) * 34 / 32
                  for i in range(kv["gemma4.block_count"]) if pattern[i])
    return full, sliding


def _place(tmp_path, extra_args=(), free=FREE):
    split, header = _model(tmp_path, GEMMA)
    return autosense.place_llamacpp(
        split, header, name="Gemma-4-31B", weights=0, mmproj=0, free_bytes=free, meminfo=_mem(52.9),
        cache_type_k="q8_0", cache_type_v="q8_0", share=1.0, autosense=AUTO, extra_args=list(extra_args))


def _hand_window(gpu: int, cap: int) -> int:
    token, cell = _sums()
    room = FREE - gpu - int(AUTO.compute_margin_ratio * gpu) - int(AUTO.compute_reserve * GIB)
    return int((room - cap * cell) // token) // 256 * 256


def test_the_recorded_gemma4_rebuilds_to_the_boxs_file_and_splits_to_its_bytes(tmp_path):
    path = write_recorded(tmp_path, load_fixture(GEMMA))[0]
    assert path.stat().st_size == 18_687_062_560
    table = read_tensor_table(path)
    assert (len(table.entries), table.data_bytes) == (833, 18_671_230_848)
    split = read_split(path, fit_header(read_header(path, FIT_KEYS)))
    assert (split.gpu, split.input, split.experts) == (18_671_230_848, 1_156_055_040, 0)
    assert (split.kv_main, split.kv_draft, split.swa, split.heads_full, split.heads_swa) == (60, 0, 50, 40, 800)
    assert kv_layer_ids(table.entries, {"block_count": 60}) == tuple(range(60))


def test_the_header_reader_keeps_the_sliding_window_keys(tmp_path):
    path = write_recorded(tmp_path, load_fixture(GEMMA))[0]
    header = fit_header(read_header(path, FIT_KEYS))
    kv = _recorded()
    assert header["sliding_window"] == kv[KV + "sliding_window"] == 1024
    assert header["sliding_window_pattern"] == kv[KV + "sliding_window_pattern"][1]
    assert header["head_count_kv"] == kv[KV + "head_count_kv"][1]
    assert [header[k] for k in ("key_length", "value_length", "key_length_swa", "value_length_swa")] == [
        kv[KV + k] for k in ("key_length", "value_length", "key_length_swa", "value_length_swa")]


def test_a_full_layer_costs_per_token_and_a_sliding_cell_per_cell_from_the_recorded_header(tmp_path):
    split, header = _model(tmp_path, GEMMA)
    token, cell = _sums()
    assert (token, cell) == (43_520, 435_200)
    assert autosense.kv_bytes_per_token(header, "q8_0", "q8_0", split) == token
    assert swa_bytes_per_cell(header, "q8_0", "q8_0", split) == cell


def test_the_gemma4_at_30_2_gib_plans_the_window_llama_cpps_own_swa_cache_leaves(tmp_path):
    placed = _place(tmp_path)
    assert (placed.fits, placed.where, placed.refusal) == (152_832, "gpu", None)
    assert placed.fits == _hand_window(18_671_230_848, 4_608)
    assert placed.kv_per_token == 43_520 and placed.swa_bytes == 4_608 * 435_200
    alone = _place(tmp_path, ["-np", "1"])
    assert (alone.fits, alone.refusal) == (183_552, None) and alone.fits == _hand_window(18_671_230_848, 1_536)
    assert alone.swa_bytes == 1_536 * 435_200


def test_the_old_price_refused_what_the_sliding_rule_fits(tmp_path):
    split, header = _model(tmp_path, GEMMA)
    flat = 60 * 16 * (512 + 512) * 34 / 32
    assert flat == 1_044_480 and autosense.kv_bytes_per_token(header, "q8_0", "q8_0", split) != flat
    assert _place(tmp_path, free=FREE).fits > 13_824


def test_the_window_line_counts_the_sliding_cells_the_plan_chose(tmp_path):
    placed = _place(tmp_path)
    report = WindowReport(None, 262_144, placed.fits, FREE, None, None, placed.fits, "fit", placement=placed)
    share = autosense.gib(placed.gpu_bytes + placed.kv_per_token * placed.fits + placed.swa_bytes)
    assert autosense._placement_line(report) == (
        f"window: {placed.fits:,} · all on the GPU ({share} of {autosense.gib(FREE)} GiB free)")


def test_a_record_of_the_gemma4_plans_the_argv_c_the_sliding_rule_gives(tmp_path, monkeypatch):
    model = write_recorded(tmp_path, load_fixture(GEMMA))[0]
    monkeypatch.setattr(autosense, "free_memory", lambda ratios: FreeMemory(FREE, 32 * GIB, "nvidia-smi"))
    monkeypatch.setattr(autosense, "_read_meminfo", lambda: {"MemTotal": 2**36, "MemAvailable": int(52.9 * GIB)})

    def plan(**over):
        spec = ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server", model=str(model), port=8080,
                                   cache_type_k="q8_0", cache_type_v="q8_0", **over)
        sensed = lifecycle.sense_launch(spec, tmp_path, AUTO)
        return lifecycle.plan_launch(spec, sensed, server.launch_decision(spec, False), autosense=AUTO,
                                     api_key_held=False, floor=13_824)

    argv = plan().argv
    assert argv[argv.index("-c") + 1] == "152832"
    one = plan(extra_args=["-np", "1"]).argv
    assert one[one.index("-c") + 1] == "183552"


@pytest.mark.parametrize("name", [SWIFT, NVIDIA, FLASH])
def test_a_model_without_sliding_window_keys_prices_as_before(tmp_path, name):
    split, header = _model(tmp_path, name)
    assert (split.swa, split.heads_full, split.heads_swa) == (0, 0, 0)
    assert swa_bytes_per_cell(header, "q8_0", "q8_0", split) == 0
    assert swa_pattern(header) is None


def _toy(tmp_path, pattern, layers=12):
    kv = {"general.architecture": "toy", "toy.block_count": layers, "toy.context_length": 32768,
          "toy.embedding_length": 64, "toy.attention.head_count": 4, "toy.attention.head_count_kv": 2,
          "toy.attention.key_length": 16, "toy.attention.sliding_window": 1024,
          "toy.attention.sliding_window_pattern": pattern}
    tensors = [("token_embd.weight", [8], 0, 0)] + [(f"blk.{i}.attn_k.weight", [8], 0, 32 * (i + 1)) for i in range(layers)]
    path = write_gguf(tmp_path / "toy.gguf", kv, tensors=tensors, data_bytes=32 * (layers + 1))
    header = fit_header(read_header(path, FIT_KEYS))
    return header, read_split(path, header)


def _largest(room, kv, swa, cap):
    fits = lambda w: w * kv + min(w, cap) * swa <= room  # noqa: E731
    return max(w for w in range(0, 20_000, 256) if fits(w))


@pytest.mark.parametrize("room", [4_500_000, 3_000_000, 2_000_000, 9_000_000, 20_000_000])
def test_the_window_is_the_largest_multiple_of_256_the_full_and_the_capped_sliding_cells_fit(room):
    kv, swa, cap = 1_000, 500, 4_608
    assert autosense._window(room, kv, 10**9, swa, cap) == _largest(room, kv, swa, cap)


def test_a_window_below_the_cells_pays_for_every_sliding_cell_it_holds():
    kv, swa, cap = 1_000, 500, 4_608
    w = autosense._window(4_500_000, kv, 10**9, swa, cap)
    assert w < cap and w * (kv + swa) <= 4_500_000 < (w + 256) * (kv + swa)


def test_swa_full_prices_every_layer_at_the_window(tmp_path):
    assert swa_cells_cap(1024, ["--swa-full"]) is None
    assert autosense._window(9_000_000, 1_000, 10**9, 500, None) == 9_000_000 // 1_500 // 256 * 256
    placed = _place(tmp_path, ["--swa-full"])
    assert placed.fits == 17_920 and placed.swa_bytes == 17_920 * 435_200


def test_an_int_pattern_leaves_every_nth_layer_full(tmp_path):
    header = {"block_count": 12, "sliding_window": 1024, "sliding_window_pattern": 6}
    assert [i for i, sliding in enumerate(swa_pattern(header)) if not sliding] == [5, 11]
    _, split = _toy(tmp_path, 6)
    assert (split.kv_main, split.swa, split.heads_full, split.heads_swa) == (12, 10, 0, 0)


@pytest.mark.parametrize("extra,cells", [
    ([], 4_608), (["-np", "2"], 3_072), (["-np", "2", "--kv-unified"], 2_560), (["--parallel=2", "-kvu"], 2_560),
    (["-np", "1"], 1_536), (["-np", "2", "-ub", "1024"], 4_096), (["-ub", "256"], 4_352), (["-np", "-1"], 4_608),
    (["-np", "abc"], 4_608),
])
def test_the_sliding_cache_holds_what_llama_cpp_allocates_for_the_slots(extra, cells):
    assert swa_cells_cap(1024, extra) == cells


@pytest.mark.parametrize("pattern", [[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [True, False] * 5, "ssssss", [], True, 0])
def test_a_pattern_it_cannot_read_leaves_every_layer_full_priced_as_before(tmp_path, pattern):
    header, split = _toy(tmp_path, pattern)
    assert swa_pattern({**header, "sliding_window_pattern": pattern}) is None
    assert split.swa == 0 and swa_bytes_per_cell(header, "q8_0", "q8_0", split) == 0
    assert autosense.kv_bytes_per_token(header, "f16", "f16", split) == 12 * 2 * 16 * 4
