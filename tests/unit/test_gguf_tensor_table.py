"""The tensor-table reader against the box's recorded models and against hostile tables."""
from __future__ import annotations

import pytest

from localharness.provider.autosense import weights_bytes
from localharness.provider.gguf_header import read_header, read_tensor_table, shard_paths
from tests.unit.gguf_support import load_fixture, write_recorded, write_sparse

MEASURED = {
    "flash-next-iq3-xxs": (1224, 75_955_048_960),
    "swift-27b-nvfp4-q8mix": (1252, 19_913_326_656),
    "nvidia-27b-nvfp4": (1252, 21_456_207_936),
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
    assert starts == [10_966_240, 57_920, 11_016_416, 11_016_640]


def test_weights_bytes_and_the_table_share_the_shard_set(tmp_path):
    fixture = load_fixture("flash-next-iq3-xxs")
    paths = write_recorded(tmp_path, fixture)
    write_sparse(tmp_path / "mmproj.gguf", fixture["mmproj_bytes"])
    assert shard_paths(paths[0]) == paths
    assert weights_bytes(str(paths[0]), str(tmp_path / "mmproj.gguf")) == 75_966_073_120 + 907_543_520
