"""A cached Hugging Face snapshot is resolved by its contents: what kind it is, whether it is offerable."""
from __future__ import annotations

import hashlib
import json
import os
import signal
from types import SimpleNamespace

import pytest

from localharness.provider import model_library, tuning

GB = 1 << 30
GENERATES = {"architectures": ["Qwen3_5ForConditionalGeneration"]}


class Cache:
    def __init__(self, tmp_path, monkeypatch):
        self.tmp, self.repos = tmp_path, []
        monkeypatch.setattr(model_library, "_cache_repos", lambda: tuple(self.repos))

    def add(self, repo_id, files, *, snap="abc", refs=("main",)):
        root = self.tmp / "hfcache" / ("models--" + repo_id.replace("/", "--"))
        snapshot, blobs = root / "snapshots" / snap, root / "blobs"
        blobs.mkdir(parents=True, exist_ok=True)
        snapshot.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            blob = blobs / hashlib.sha1(f"{repo_id}/{name}".encode()).hexdigest()
            blob.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())
            link = snapshot / name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(os.path.relpath(blob, link.parent))
        rev = SimpleNamespace(commit_hash=snap, snapshot_path=snapshot, refs=frozenset(refs), last_modified=1.0)
        self.repos.append(SimpleNamespace(repo_id=repo_id, repo_type="model", size_on_disk=GB, revisions=[rev]))
        return snapshot


@pytest.fixture
def cache(tmp_path, monkeypatch):
    return Cache(tmp_path, monkeypatch)


def listed():
    return {c.repo_id: c for c in model_library.snapshots()}


def test_the_embedding_shape_is_never_a_checkpoint(cache):
    cache.add("Qwen/Qwen3-Embedding-0.6B", {"config.json": {"architectures": ["Qwen3ForCausalLM"]},
                                            "modules.json": b"[]", "1_Pooling/config.json": b"{}",
                                            "model.safetensors": b"x"})

    assert listed() == {}
    assert model_library.checkpoints() == ()


def test_safetensors_with_a_text_generation_config_is_vllm(cache):
    cache.add("org/Chat", {"config.json": GENERATES, "model-1.safetensors": b"x", "model-2.safetensors": b"x"})

    got = listed()["org/Chat"]
    assert (got.kind, got.refusal, got.ggufs) == ("vllm", None, ())
    assert [c.repo_id for c in model_library.checkpoints()] == ["org/Chat"]


def test_a_snapshot_holding_a_gguf_is_gguf_and_carries_its_paths(cache):
    snapshot = cache.add("org/Quant", {"model-Q4_K_M.gguf": b"GGUF"})

    got = listed()["org/Quant"]
    assert (got.kind, got.ggufs) == ("gguf", (snapshot / "model-Q4_K_M.gguf",))
    assert model_library.checkpoints() == ()


def test_a_config_without_text_generation_is_not_listed(cache):
    cache.add("org/Encoder", {"config.json": {"architectures": ["BertModel"]}, "model.safetensors": b"x"})

    assert listed() == {}


UNLOADED = "its snapshot holds an entry the harness does not load"
GOOD = {"config.json": GENERATES, "model.safetensors": b"x"}


def plant(snapshot, name, target):
    link = snapshot / name
    link.unlink(missing_ok=True)
    link.symlink_to(target)


def test_the_real_layout_with_a_nested_directory_of_links_passes(cache):
    cache.add("org/Nested", {**GOOD, "1_Tokenizer/tokenizer.json": b"{}"})

    assert "org/Nested" in listed()


def test_a_link_to_a_file_outside_blobs_unlists_the_snapshot(cache, tmp_path):
    snapshot = cache.add("org/Out", GOOD)
    outside = tmp_path / "outside.txt"
    outside.write_text("x")
    plant(snapshot, "tokenizer.json", outside)

    assert listed() == {}


def test_a_link_into_another_repos_blobs_unlists_the_snapshot(cache):
    cache.add("org/Other", GOOD)
    other_blob = next((cache.tmp / "hfcache" / "models--org--Other" / "blobs").iterdir())
    snapshot = cache.add("org/Mine", GOOD)
    plant(snapshot, "tokenizer.json", other_blob)

    assert set(listed()) == {"org/Other"}


def test_a_bad_entry_sorted_last_still_unlists_the_snapshot(cache, tmp_path):
    snapshot = cache.add("org/Last", {**GOOD, "aaa.json": b"{}"})
    plant(snapshot, "zzz.json", tmp_path)

    assert listed() == {}


def test_a_fifo_unlists_the_snapshot_and_the_walk_returns(cache):
    snapshot = cache.add("org/Fifo", GOOD)
    os.mkfifo(snapshot / "pipe")
    signal.signal(signal.SIGALRM, lambda *_: pytest.fail("the walk blocked"))
    signal.alarm(2)
    try:
        assert listed() == {}
    finally:
        signal.alarm(0)


def test_a_dangling_link_unlists_the_snapshot(cache):
    snapshot = cache.add("org/Dangling", GOOD)
    plant(snapshot, "tokenizer.json", "../../blobs/missing")

    assert listed() == {}


@pytest.mark.parametrize("name", ["config.json", "tokenizer_config.json"])
def test_an_auto_map_is_refused_by_name_and_never_offered(cache, name):
    cache.add("org/Remote", {**GOOD, name: {**GENERATES, "auto_map": {"AutoModel": "m.X"}}})

    assert listed()["org/Remote"].refusal == "needs trust_remote_code"
    assert model_library.checkpoints() == ()


BAD_CONFIGS = {
    "empty": None,
    "missing config": {"model.safetensors": b"x"},
    "non-object": {"config.json": b"[]", "model.safetensors": b"x"},
    "over cap": {"config.json": b'{"architectures":["XForCausalLM"]}' + b" " * tuning.MODEL_CONFIG_CAP,
                 "model.safetensors": b"x"},
    "invalid utf-8": {"config.json": b'{"architectures": ["\xc3("]}', "model.safetensors": b"x"},
    "malformed": {"config.json": b'{"architectures": [', "model.safetensors": b"x"},
}


@pytest.mark.parametrize("case", BAD_CONFIGS)
def test_an_unreadable_config_gives_no_entry_and_never_raises(cache, case):
    files = BAD_CONFIGS[case]
    cache.add("org/Bad", {} if files is None else files)

    assert listed() == {}


def test_snapshot_of_refuses_a_cached_repo_whose_walk_fails_and_knows_no_uncached_one(cache, tmp_path):
    snapshot = cache.add("org/Planted", GOOD)
    assert model_library.snapshot_of("org/Planted").refusal is None
    plant(snapshot, "tokenizer.json", tmp_path)

    assert model_library.snapshot_of("org/Planted").refusal == UNLOADED
    assert model_library.snapshot_of("org/Nowhere") is None
