"""A cached Hugging Face snapshot is resolved by its contents: what kind it is, whether it is offerable."""
from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from localharness.provider import model_library

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
