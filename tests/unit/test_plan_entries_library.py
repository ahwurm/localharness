"""`model_ops.plan_entries` over what the box holds that no record names: HF snapshots, unrecorded GGUFs, what cannot run here."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from localharness.cli import model_ops
from localharness.config.models import ManagedServerConfig
from localharness.provider import lifecycle, model_library, server, tuning
from localharness.provider.autosense import FreeMemory
from tests.unit.test_library_vllm_peers import cache_repo
from tests.unit.gguf_support import write_gguf
from tests.unit.test_plan_entries import FLOOR, entries_of, harness_of
from tests.unit.test_serve_cmd import PRIMARY, PRIMARY_URL, llama_peer, serve_world
from tests.unit.test_tuning_derive import FLASH_NEXT

GIB = 2**30
CHECKPOINTS = Path(__file__).resolve().parents[1] / "fixtures" / "vllm_checkpoints"
CARD = int(31.8 * GIB)
MEMORY = FreeMemory(int(30.2 * GIB), CARD, "nvidia-smi")
NVIDIA, UKISAI = "nvidia/Qwen3.8-27B-NVFP4", "ukisai/Swift-1.5-Qwen3.8-27b-NVFP4"
SERVING = {PRIMARY_URL: (PRIMARY,)}
NVFP4_BYTES = (10_000_000_000, 10_000_000_000, 1_921_697_280)


def snapshot(tmp_path, repo_id, fixture, sizes=NVFP4_BYTES, **kw):
    config = json.loads((CHECKPOINTS / fixture / "config.json").read_text())
    repo = cache_repo(tmp_path, repo_id, config, **kw)
    directory = repo.revisions[0].snapshot_path
    for i, size in enumerate(sizes):
        shard = directory / ("model.safetensors" if i == 0 else f"model-{i}.safetensors")
        shard.touch()
        os.truncate(shard, size)
    return repo


def cached(monkeypatch, *repos):
    monkeypatch.setattr(model_library, "_cache_repos", lambda: tuple(repos))
    configs = {r.repo_id: str(r.revisions[0].snapshot_path / "config.json") for r in repos}
    monkeypatch.setattr(tuning, "_hf_cached_config", configs.get)


def with_vllm(tmp_path, monkeypatch):
    exe = tmp_path / "bin" / "vllm"
    monkeypatch.setattr(server, "vllm_binary", lambda config_dir: (str(exe), "on PATH"))


@pytest.fixture
def world(tmp_path, monkeypatch):
    return serve_world(tmp_path, monkeypatch)


def listed(world, **kw):
    return entries_of(world, SERVING, memory=MEMORY, **kw)


def test_an_nvfp4_snapshot_is_a_priced_entry_that_is_not_set_up(world, tmp_path, monkeypatch):
    cached(monkeypatch, snapshot(tmp_path, NVIDIA, "nvidia-qwen3.8-27b-nvfp4"))
    with_vllm(tmp_path, monkeypatch)

    entries = listed(world)

    assert model_ops.entry_lines(entries)[3:] == [
        "▸ this box can launch", f"  2. {NVIDIA}  vLLM · 100,864 (not set up)"]
    picked = entries.pick("2")
    assert (picked.kind, picked.setup.checkpoint.repo_id) == ("setup", NVIDIA)
    assert entries.pick(NVIDIA) is picked


def test_listing_the_box_writes_nothing(world, tmp_path, monkeypatch):
    cached(monkeypatch, snapshot(tmp_path, NVIDIA, "nvidia-qwen3.8-27b-nvfp4"))
    with_vllm(tmp_path, monkeypatch)
    library, overlay = world.g / model_library.LIBRARY_FILE, world.g / "overrides.yaml"
    library.write_text('{"version": 1, "entries": {}}', encoding="utf-8")
    overlay.write_text("{}\n", encoding="utf-8")
    before = [os.stat(p).st_mtime_ns for p in (library, overlay)]

    listed(world)

    assert [os.stat(p).st_mtime_ns for p in (library, overlay)] == before


def unrecorded_gguf(world, name="Flash-Next.gguf"):
    models = world.home / "models"
    models.mkdir(exist_ok=True)
    return write_gguf(models / name, FLASH_NEXT)


def window_of(world, path, name):
    harness = harness_of(world)
    spec = ManagedServerConfig(runtime="llamacpp", model=str(path), served_name=name, port=8082, gpu=True,
                               binary=harness.server.binary)
    sensed = lifecycle.sense_launch(spec, world.g, harness.autosense, memory=MEMORY)
    plan = lifecycle.plan_launch(spec, sensed, server.launch_decision(spec, False), autosense=harness.autosense,
                                 api_key_held=False, floor=FLOOR)
    return plan.report.asked


def test_an_unrecorded_gguf_is_a_llama_cpp_entry_priced_like_a_record(world):
    path = unrecorded_gguf(world)

    entry = listed(world).numbered()[1]

    assert (entry.kind, entry.setup.real) == ("setup", os.path.realpath(path))
    assert entry.window == window_of(world, path, entry.name)
    assert model_ops.entry_lines(listed(world))[4] == f"  2. {entry.name}  llama.cpp · {entry.window:,} (not set up)"


def test_a_snapshot_gguf_a_record_names_is_that_records_entry(tmp_path, monkeypatch):
    repo = cache_repo(tmp_path, "org/flash-GGUF")
    gguf = write_gguf(repo.revisions[0].snapshot_path / "flash.gguf", FLASH_NEXT)
    peer = llama_peer("flash-next", 8082)
    peer["lifecycle"]["model"] = str(gguf)
    world = serve_world(tmp_path, monkeypatch, peers=[peer])
    cached(monkeypatch, repo)

    entries = entries_of(world, SERVING, memory=MEMORY)

    assert [e.name for e in entries.numbered()] == [PRIMARY, "flash-next"]
    assert [e.kind for e in entries.numbered()] == ["live", "cold"]


def test_a_snapshot_holding_a_gguf_is_listed_as_the_gguf(tmp_path, monkeypatch):
    repo = cache_repo(tmp_path, "org/flash-GGUF")
    gguf = write_gguf(repo.revisions[0].snapshot_path / "flash.gguf", FLASH_NEXT)
    world = serve_world(tmp_path, monkeypatch)
    cached(monkeypatch, repo)

    entry = entries_of(world, SERVING, memory=MEMORY).numbered()[1]

    assert (entry.kind, entry.runtime, entry.setup.real) == ("setup", "llamacpp", os.path.realpath(gguf))
    assert entry.window == window_of(world, gguf, entry.name)


def test_the_second_nvfp4_snapshot_is_listed_beside_the_first(world, tmp_path, monkeypatch):
    cached(monkeypatch, snapshot(tmp_path, NVIDIA, "nvidia-qwen3.8-27b-nvfp4"),
           snapshot(tmp_path, UKISAI, "ukisai-swift-1.5-27b-nvfp4", sizes=(21_921_695_568,)))
    with_vllm(tmp_path, monkeypatch)

    assert model_ops.entry_lines(listed(world))[3:] == [
        "▸ this box can launch", f"  2. {NVIDIA}  vLLM · 100,864 (not set up)",
        f"  3. {UKISAI}  vLLM · 100,864 (not set up)"]
