"""The tensor table split into gpu, experts and input bytes and the layers that hold KV, on the box's recorded models."""
from __future__ import annotations

import pytest

from localharness.config.models import AutosenseConfig, ManagedServerConfig
from localharness.provider import autosense, gguf_header, lifecycle
from localharness.provider.autosense import (FIT_KEYS, FreeMemory, TensorSplit, compute_fit, fit, fit_header,
                                             kv_bytes_per_token, read_split, weights_bytes)
from localharness.provider.gguf_header import (TensorEntry, classify, cross_checked, indexer_layers, kv_layers,
                                               read_header, read_tensor_table)
from localharness.provider.tuning import Tuned, draft_share, launch_pricing
from tests.unit.gguf_support import load_fixture, write_gguf, write_recorded, write_sparse

FLASH, SWIFT, NVIDIA = "flash-next-iq3-xxs", "swift-27b-nvfp4-q8mix", "nvidia-27b-nvfp4"
SPLITS = {
    FLASH: (3_968_785_920, 42_912_972_800, 29_073_290_240, 12, 0, 12),
    SWIFT: (18_562_465_856, 0, 1_350_860_800, 16, 1, 0),
    NVIDIA: (18_913_411_136, 0, 2_542_796_800, 16, 1, 0),
}


def _recorded(tmp_path, name):
    path = write_recorded(tmp_path, load_fixture(name))[0]
    return path, fit_header(read_header(path, FIT_KEYS))


@pytest.mark.parametrize("name,expected", [
    ("token_embd.weight", "input"), ("per_layer_token_embd.weight", "input"), ("pos_embd.weight", "input"),
    ("token_types.weight", "input"), ("blk.0.ffn_down_exps.weight", "experts"), ("blk.0.ffn_down_shexp.weight", "gpu"),
    ("blk.0.attn_k.weight", "gpu"), ("output.weight", "gpu"), ("", "gpu"), ("token_embd", "input"),
])
def test_classify(name, expected):
    assert classify(name) == expected


@pytest.mark.parametrize("name", SPLITS)
def test_the_recorded_tables_split_to_the_measured_bytes(tmp_path, name):
    path, header = _recorded(tmp_path, name)
    split = read_split(path, header)
    assert split == TensorSplit(*SPLITS[name])


def test_a_table_without_an_output_projection_prices_a_gpu_copy_of_token_embd(tmp_path):
    fixture = load_fixture(SWIFT)
    for tensor in fixture["shards"][0]["tensors"]:
        if tensor[0] == "output.weight":
            tensor[0] = "output_proj.weight"
    path = write_recorded(tmp_path, fixture)[0]
    entries = read_tensor_table(path).entries
    embd = next(e.bytes for e in entries if e.name == "token_embd.weight")
    split = read_split(path, fit_header(read_header(path, FIT_KEYS)))
    assert split.gpu == sum(e.bytes for e in entries if classify(e.name) == "gpu") + embd
    assert split.input == SPLITS[SWIFT][2]


@pytest.mark.parametrize("name", SPLITS)
def test_the_layer_counts_come_from_the_attention_tensors(tmp_path, name):
    path, header = _recorded(tmp_path, name)
    entries = read_tensor_table(path).entries
    assert kv_layers(entries, header) == SPLITS[name][3:5]
    assert indexer_layers(entries) == SPLITS[name][5]


def test_the_indexer_key_length_is_read_under_the_architecture(tmp_path):
    _, header = _recorded(tmp_path, FLASH)
    assert header["indexer.key_length"] == 128 and header["full_attention_interval"] == 4


def _e(*names):
    return tuple(TensorEntry(n, 1) for n in names)


def test_a_qkv_block_with_an_ssm_tensor_holds_no_kv():
    entries = _e("blk.0.attn_qkv.weight", "blk.0.ssm_out.weight", "blk.1.attn_qkv.weight", "blk.2.attn_k_norm.weight")
    assert kv_layers(entries, {"block_count": 4}) == (1, 0)


def test_kv_layers_is_pure():
    entries, header = _e("blk.3.attn_k.weight", "blk.4.attn_k.weight"), {"block_count": 5, "nextn_predict_layers": 1}
    assert kv_layers(entries, header) == kv_layers(entries, header) == (1, 1)


@pytest.fixture(scope="module")
def splits(tmp_path_factory):
    out = {}
    for name in SPLITS:
        path, header = _recorded(tmp_path_factory.mktemp(name), name)
        out[name] = (header, read_split(path, header))
    return out


@pytest.mark.parametrize("name,cache,per_token", [
    (FLASH, "q8_0", 16_320), (FLASH, "f16", 30_720), (SWIFT, "q8_0", 34_816), (NVIDIA, "f16", 65_536),
])
def test_kv_prices_the_attention_layers_and_the_indexer(splits, name, cache, per_token):
    header, split = splits[name]
    assert kv_bytes_per_token(header, cache, cache, split) == per_token


def test_the_draft_share_is_the_splits(splits):
    swift, flash = splits[SWIFT], splits[FLASH]
    assert draft_share(*swift) == 1 + 1 / 16
    assert 34_816 * draft_share(*swift) - 34_816 == 2_176
    assert draft_share(*flash) == 1.0


def test_without_a_split_the_price_is_the_block_count_formula(splits):
    header = splits[SWIFT][0]
    assert kv_bytes_per_token(header, "q8_0", "q8_0") == 65 * 4 * 256 * 2 * 34 / 32
    assert draft_share(header) == 1 + 1 / 65


def test_launch_pricing_hands_the_split_to_a_drafting_launch(splits):
    header, split = splits[SWIFT]
    spec = ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server", model="/m.gguf", port=8080,
                               speculative="draft-mtp")
    assert launch_pricing(spec, header, Tuned(), split)[2] == 1 + 1 / 16
    assert launch_pricing(spec, header, Tuned())[2] == 1 + 1 / 65
    assert launch_pricing(spec.model_copy(update={"speculative": None}), header, Tuned(), split)[2] == 1.0


def test_the_fit_uses_the_split_price(splits):
    header, split = splits[SWIFT]
    ratios, free, weights = AutosenseConfig(), 32 * 2**30, 20 * 2**30
    usable = free - weights - int(ratios.compute_margin_ratio * weights)
    kw = dict(cache_type_k="q8_0", cache_type_v="q8_0", weights=weights)
    assert fit(header, free, ratios, split=split, **kw) == min(262_144, int(usable // 34_816))
    assert fit(header, free, ratios, **kw) < fit(header, free, ratios, split=split, **kw)


HEADER = {"block_count": 8, "full_attention_interval": 4}


@pytest.mark.parametrize("sums,main,header,data,ok", [
    ({"input": 50}, 2, HEADER, 100, True),
    ({"input": 51}, 2, HEADER, 100, False),
    ({}, 0, HEADER, 100, False),
    ({}, 9, {"block_count": 8}, 100, False),
    ({}, 8, {"block_count": 8}, 100, True),
    ({}, 1, HEADER, 100, False),
    ({}, 1, {"block_count": 9, "nextn_predict_layers": 1, "full_attention_interval": 8}, 100, True),
    ({}, 1, {}, 100, False),
    ({"experts": 10}, 2, HEADER, 100, False),
    ({"experts": 10}, 2, {**HEADER, "expert_count": 4}, 100, True),
])
def test_cross_checked(sums, main, header, data, ok):
    assert cross_checked(sums, main, header, data) is ok


def _hostile(tmp_path):
    kv = {"general.architecture": "qwen35", "qwen35.block_count": 8, "qwen35.context_length": 32768,
          "qwen35.embedding_length": 64, "qwen35.attention.head_count": 4, "qwen35.attention.head_count_kv": 2}
    tensors = [("token_embd.weight", [8], 0, 0), ("blk.0.ffn_gate_exps.weight", [8], 0, 32),
               ("blk.1.ffn_down_exps.weight", [8], 0, 64)]
    return write_gguf(tmp_path / "hostile.gguf", kv, tensors=tensors, data_bytes=128)


def _server(model, **over):
    return ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server", model=str(model), port=8080, **over)


MEMORY = FreeMemory(2**30, 2**31, "nvidia-smi")


def test_a_hostile_table_fails_the_checks_and_is_priced_as_the_whole_file(tmp_path):
    model = _hostile(tmp_path)
    ratios = AutosenseConfig()
    fi = compute_fit(_server(model), ratios, memory=MEMORY)
    assert fi.split is None
    whole = fit(fi.header, MEMORY.free_bytes, ratios, weights=weights_bytes(str(model)))
    assert fi.fits == whole is not None


def test_compute_fit_carries_the_split_and_the_projector(tmp_path):
    fixture = load_fixture(SWIFT)
    model = write_recorded(tmp_path, fixture)[0]
    mmproj = write_sparse(tmp_path / "mmproj.gguf", fixture["mmproj_bytes"])
    fi = compute_fit(_server(model, mmproj=str(mmproj)), AutosenseConfig(), memory=MEMORY)
    assert fi.split == TensorSplit(*SPLITS[SWIFT]) and fi.mmproj == fixture["mmproj_bytes"]
    assert compute_fit(_server(model), AutosenseConfig(), memory=MEMORY).mmproj == 0


def test_sense_launch_reads_the_table_once_per_shard(tmp_path, monkeypatch):
    fixture = load_fixture(FLASH)
    model = write_recorded(tmp_path, fixture)[0]
    mmproj = write_sparse(tmp_path / "mmproj.gguf", fixture["mmproj_bytes"])
    reads = []
    real = gguf_header._shard_table
    monkeypatch.setattr(gguf_header, "_shard_table", lambda p: reads.append(p) or real(p))
    monkeypatch.setattr(autosense, "free_memory", lambda ratios: MEMORY)
    monkeypatch.setattr(autosense, "_read_meminfo", lambda: {"MemTotal": 2**35, "MemAvailable": 2**34})
    sensed = lifecycle.sense_launch(_server(model, mmproj=str(mmproj)), tmp_path, AutosenseConfig())
    assert len(reads) == 2
    assert sensed.split == TensorSplit(*SPLITS[FLASH]) and sensed.mmproj == fixture["mmproj_bytes"]


def test_an_unreadable_table_leaves_the_split_unset(tmp_path, monkeypatch):
    monkeypatch.setattr(autosense, "free_memory", lambda ratios: MEMORY)
    sensed = lifecycle.sense_launch(_server(_hostile(tmp_path)), tmp_path, AutosenseConfig())
    assert sensed.split is None and sensed.mmproj == 0
