"""The tensor-table reader against the box's recorded models and against hostile tables."""
from __future__ import annotations

import os
import struct

import pytest

from localharness.provider.autosense import weights_bytes
from localharness.provider.gguf_header import (HEADER_BYTE_BUDGET, MAX_DIMS, MAX_STRING, MAX_TENSOR_NAME,
                                               MAX_TENSORS, GgufHeaderError, NotRegularFile, TensorTable,
                                               read_header, read_tensor_table, shard_paths)
from tests.unit.gguf_support import Raw, load_fixture, write_gguf, write_recorded, write_sparse

MEASURED = {
    "flash-next-iq3-xxs": (1224, 75_955_048_960),
    "swift-27b-nvfp4-q8mix": (1252, 19_913_326_656),
    "nvidia-27b-nvfp4": (1252, 21_456_207_936),
    "gemma4-31b-q4-k-m": (833, 18_671_230_848),
}


@pytest.mark.parametrize("name", MEASURED)
def test_a_recorded_table_reads_to_the_byte(tmp_path, name):
    paths = write_recorded(tmp_path, load_fixture(name))
    table = read_tensor_table(paths[0])
    count, total = MEASURED[name]
    assert (len(table.entries), sum(e.bytes for e in table.entries), table.data_bytes) == (count, total, total)


@pytest.mark.parametrize("name", MEASURED)
def test_the_builder_reproduces_each_recorded_shard(tmp_path, name):
    fixture = load_fixture(name)
    paths = write_recorded(tmp_path, fixture)
    assert [p.stat().st_size for p in paths] == [s["file_bytes"] for s in fixture["shards"]]
    assert read_tensor_table(paths[0]).data_bytes == sum(s["file_bytes"] - s["data_start"] for s in fixture["shards"])
    assert read_header(paths[0], {"architecture", "block_count"})["architecture"] == fixture["shards"][0]["kv"][0][2]


def test_the_recorded_data_starts_are_the_measured_ones():
    starts = [s["data_start"] for n in MEASURED for s in load_fixture(n)["shards"]]
    assert starts == [10_966_240, 57_920, 11_016_416, 11_016_640, 15_831_712]


def test_weights_bytes_and_the_table_share_the_shard_set(tmp_path):
    fixture = load_fixture("flash-next-iq3-xxs")
    paths = write_recorded(tmp_path, fixture)
    write_sparse(tmp_path / "mmproj.gguf", fixture["mmproj_bytes"])
    assert shard_paths(paths[0]) == paths
    assert weights_bytes(str(paths[0]), str(tmp_path / "mmproj.gguf")) == 75_966_073_120 + 907_543_520


def _read(tmp_path, kv=None, **kw):
    return read_tensor_table(write_gguf(tmp_path / "m.gguf", kv or {}, **kw))


def _t(name, offset, dims=(8,)):
    return (name, list(dims), 0, offset)


def test_a_count_above_the_cap_is_refused_before_any_info_is_read(tmp_path):
    with pytest.raises(GgufHeaderError, match="too many tensors"):
        _read(tmp_path, n_tensors=MAX_TENSORS + 1)


def test_a_count_at_the_cap_passes_the_count_check(tmp_path):
    with pytest.raises(GgufHeaderError, match="truncated"):
        _read(tmp_path, n_tensors=MAX_TENSORS)


def test_a_name_is_bounded_by_its_bytes(tmp_path):
    assert _read(tmp_path, tensors=[_t("a" * MAX_TENSOR_NAME, 0)], data_bytes=32).entries[0].bytes == 32
    with pytest.raises(GgufHeaderError, match="too long"):
        _read(tmp_path, tensors=[_t("a" * (MAX_TENSOR_NAME + 1), 0)], data_bytes=32)


def test_a_name_of_many_characters_in_few_bytes_is_bounded_by_bytes(tmp_path):
    with pytest.raises(GgufHeaderError, match="too long"):
        _read(tmp_path, tensors=[_t("é" * 129, 0)], data_bytes=32)


def test_invalid_utf8_in_a_name_decodes_with_replacement(tmp_path):
    table = _read(tmp_path, tensors=[(b"blk.\xff.w", [8], 0, 0)], data_bytes=32)
    assert table.entries[0].name == "blk.�.w"


def test_dims_are_bounded(tmp_path):
    assert len(_read(tmp_path, tensors=[_t("w", 0, (1,) * MAX_DIMS)], data_bytes=32).entries) == 1
    with pytest.raises(GgufHeaderError, match="dimensions"):
        _read(tmp_path, tensors=[_t("w", 0, (1,) * (MAX_DIMS + 1))], data_bytes=32)


def test_adjacent_offsets_give_the_distance(tmp_path):
    table = _read(tmp_path, tensors=[_t("a", 0), _t("b", 32)], data_bytes=96)
    assert [e.bytes for e in table.entries] == [32, 64] and table.data_bytes == 96


@pytest.mark.parametrize("offsets", [(32, 32), (64, 32), (0, 0)])
def test_offsets_must_strictly_increase(tmp_path, offsets):
    with pytest.raises(GgufHeaderError, match="offset"):
        _read(tmp_path, tensors=[_t("a", offsets[0]), _t("b", offsets[1])], data_bytes=128)


def test_an_unaligned_offset_is_refused(tmp_path):
    with pytest.raises(GgufHeaderError, match="aligned"):
        _read(tmp_path, tensors=[_t("a", 0), _t("b", 16)], data_bytes=128)


@pytest.mark.parametrize("offset", [96, 160])
def test_an_offset_at_or_past_the_data_end_is_refused(tmp_path, offset):
    with pytest.raises(GgufHeaderError, match="data"):
        _read(tmp_path, tensors=[_t("a", 0), _t("b", offset)], data_bytes=96)


def test_no_tensors_is_an_empty_table_of_the_data_section(tmp_path):
    table = _read(tmp_path, tensors=[], data_bytes=100)
    assert table == TensorTable((), 100)


def test_alignment_moves_the_data_start(tmp_path):
    table = _read(tmp_path, tensors=[_t("a", 0), _t("b", 64)], alignment=64, data_bytes=128)
    assert [e.bytes for e in table.entries] == [64, 64]
    with pytest.raises(GgufHeaderError, match="aligned"):
        _read(tmp_path, tensors=[_t("a", 0), _t("b", 32)], alignment=64, data_bytes=128)


def test_a_zero_alignment_is_refused(tmp_path):
    with pytest.raises(GgufHeaderError, match="alignment"):
        _read(tmp_path, {"general.alignment": Raw(4, b"\0\0\0\0")}, tensors=[], data_bytes=8)


def test_a_fifo_shard_is_refused_without_blocking(tmp_path):
    write_gguf(tmp_path / "m-00001-of-00002.gguf", {}, tensors=[], data_bytes=8)
    os.mkfifo(tmp_path / "m-00002-of-00002.gguf")
    with pytest.raises(NotRegularFile):
        read_tensor_table(tmp_path / "m-00001-of-00002.gguf")


def test_a_missing_shard_is_an_oserror(tmp_path):
    write_gguf(tmp_path / "m-00001-of-00002.gguf", {}, tensors=[], data_bytes=8)
    with pytest.raises(OSError):
        read_tensor_table(tmp_path / "m-00001-of-00002.gguf")


def test_a_header_over_the_byte_budget_is_refused(tmp_path):
    pad = {f"k{i}": Raw(8, struct.pack("<Q", MAX_STRING) + bytes(MAX_STRING)) for i in range(HEADER_BYTE_BUDGET // MAX_STRING + 1)}
    with pytest.raises(GgufHeaderError, match="budget"):
        _read(tmp_path, pad, tensors=[], data_bytes=8)
