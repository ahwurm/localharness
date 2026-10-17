"""The tensor table split into gpu, experts and input bytes and the layers that hold KV, on the box's recorded models."""
from __future__ import annotations

import pytest

from localharness.provider.autosense import FIT_KEYS, TensorSplit, fit_header, read_split
from localharness.provider.gguf_header import (TensorEntry, classify, indexer_layers, kv_layers,
                                               read_header, read_tensor_table)
from tests.unit.gguf_support import load_fixture, write_recorded

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
