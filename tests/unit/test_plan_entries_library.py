"""`model_ops.plan_entries` over what the box holds that no record names: HF snapshots, unrecorded GGUFs, what cannot run here."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from localharness.cli import model_ops, workspace
from localharness.config.models import ManagedServerConfig
from localharness.provider import lifecycle, model_library, server, speed_stats, tuning
from localharness.provider.autosense import FreeMemory
from tests.unit.test_library_vllm_peers import cache_repo
from tests.unit.gguf_support import write_gguf
from tests.unit.test_plan_entries import FLOOR, LLM_SERVICE, entries_of, flash, harness_of
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


def test_one_projector_name_the_config_refuses_is_its_own_line_not_the_lists(world):
    models = world.home / "models"
    models.mkdir(exist_ok=True)
    write_gguf(models / "Flash-Next.gguf", {**FLASH_NEXT, "qwen4exp.embedding_length": 2048})
    unrecorded_gguf(world, "Other-Model.gguf")
    write_gguf(models / "mmproj-Flash-Next\n.gguf", {
        "general.architecture": "clip", "general.type": "mmproj", "clip.vision.projection_dim": 2048})

    entries = listed(world)

    (bad,) = [e for e in entries.numbered() + entries.not_offered if e.refusal and "could not plan it" in e.refusal]
    assert "\n" in str(bad.setup.model.mmproj)
    assert [e.name for e in entries.numbered() if e.kind == "setup" and e.refusal is None]


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


GEMMA = "DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking"
GEMMA_SHARDS = (31_273_169_132, 31_273_169_132)
GEMMA_REASON = "58.3 GiB of bf16 weights, and vLLM keeps them all on the 31.8 GiB GPU"
GEMMA_FIX = "a 4-bit build of it (a GGUF Q4_K_M for llama.cpp, or NVFP4 for vLLM) would fit"
GEMMA_SENTENCE = f"{GEMMA} cannot run here: {GEMMA_REASON} — {GEMMA_FIX}."


def gemma(tmp_path):
    return snapshot(tmp_path, GEMMA, "davidau-gemma-4-31b-it", sizes=GEMMA_SHARDS)


def not_offered(entries):
    return [line for line in model_ops.entry_lines(entries) if line.startswith("  not offered:")]


def test_not_offered_reason_keeps_the_fix_clause():
    assert model_ops.not_offered_reason(GEMMA_SENTENCE, GEMMA) == f"{GEMMA_REASON} — {GEMMA_FIX}"


def test_the_gemma_bf16_snapshot_is_one_unnumbered_line_with_its_reason(world, tmp_path, monkeypatch):
    cached(monkeypatch, gemma(tmp_path))
    with_vllm(tmp_path, monkeypatch)

    entries = listed(world)

    assert not_offered(entries) == [f"  not offered: {GEMMA} — {GEMMA_REASON} — {GEMMA_FIX}"]
    assert [e.name for e in entries.numbered()] == [PRIMARY]
    assert entries.pick(GEMMA) == GEMMA_SENTENCE


def test_a_snapshot_with_an_auto_map_needs_trust_remote_code(world, tmp_path, monkeypatch):
    repo = cache_repo(tmp_path, "org/remote")
    (repo.revisions[0].snapshot_path / "tokenizer_config.json").write_text('{"auto_map": {}}', encoding="utf-8")
    cached(monkeypatch, repo)
    with_vllm(tmp_path, monkeypatch)

    entries = listed(world)

    assert not_offered(entries) == [
        "  not offered: org/remote — needs trust_remote_code — the harness never passes --trust-remote-code"]
    assert entries.pick("org/remote") == (
        "org/remote cannot run here: needs trust_remote_code — the harness never passes --trust-remote-code.")


def test_with_no_vllm_binary_each_vllm_snapshot_says_so(world, tmp_path, monkeypatch):
    cached(monkeypatch, snapshot(tmp_path, NVIDIA, "nvidia-qwen3.8-27b-nvfp4"), gemma(tmp_path))
    monkeypatch.setattr(server, "vllm_binary", lambda config_dir: None)

    entries = listed(world)

    assert not_offered(entries) == [
        f"  not offered: {GEMMA} — no vllm binary on PATH or in the harness's venv — install vllm or put it on PATH",
        f"  not offered: {NVIDIA} — no vllm binary on PATH or in the harness's venv — install vllm or put it on PATH"]
    assert entries.pick(NVIDIA) == (
        f"{NVIDIA} cannot run here: no vllm binary on PATH or in the harness's venv — install vllm or put it on PATH.")
    assert [e.name for e in entries.numbered()] == [PRIMARY]


def test_an_embedding_snapshot_is_nowhere_in_the_list(world, tmp_path, monkeypatch):
    repo = cache_repo(tmp_path, "Qwen/Qwen3-Embedding-0.6B", {"architectures": ["Qwen3ForCausalLM"]})
    directory = repo.revisions[0].snapshot_path
    (directory / "modules.json").write_text("[]", encoding="utf-8")
    (directory / "1_Pooling").mkdir()
    (directory / "1_Pooling" / "config.json").write_text("{}", encoding="utf-8")
    cached(monkeypatch, repo)
    with_vllm(tmp_path, monkeypatch)

    lines = model_ops.entry_lines(listed(world))

    assert "Embedding" not in "\n".join(lines) and lines[-1] == f"  1. {PRIMARY}  (serving)  ● [active]"


def test_a_box_holding_nothing_unrecorded_has_no_setup_entry_and_no_library_line(world, tmp_path, monkeypatch):
    with_vllm(tmp_path, monkeypatch)

    lines = model_ops.entry_lines(listed(world))

    assert lines == ["Models:", "▸ llama.cpp · 127.0.0.1:8080", f"  1. {PRIMARY}  (serving)  ● [active]"]


def test_not_offered_lines_follow_every_numbered_entry_in_name_order(world, tmp_path, monkeypatch):
    unrecorded_gguf(world)
    cached(monkeypatch, snapshot(tmp_path, "zeta/model", "nvidia-qwen3.8-27b-nvfp4"),
           snapshot(tmp_path, "alpha/model", "nvidia-qwen3.8-27b-nvfp4"))
    monkeypatch.setattr(server, "vllm_binary", lambda config_dir: None)

    lines = model_ops.entry_lines(listed(world))

    assert [line.split(" — ")[0] for line in lines[-2:]] == ["  not offered: alpha/model", "  not offered: zeta/model"]
    assert lines[-3].startswith("  2. ")


UNIT_SERVES = "qwen-27b-swift-1.5"


@pytest.fixture
def unit_world(world, monkeypatch):
    monkeypatch.setattr(workspace, "port_holder", lambda port: LLM_SERVICE)
    return world


def listing_of(world, served=UNIT_SERVES):
    return model_ops.listing(
        harness_of(world), live={PRIMARY_URL: (served,)}, config_dir=world.g,
        trust=model_ops.trust_as_recorded(world.g), floor=FLOOR, memory=MEMORY)


def sentence_for(served=UNIT_SERVES):
    return workspace.foreign_unit_sentence(LLM_SERVICE, served=served, port=8080)


def test_the_entries_a_unit_forbids_answer_with_the_foreign_unit_sentence(unit_world, tmp_path, monkeypatch):
    cached(monkeypatch, snapshot(tmp_path, NVIDIA, "nvidia-qwen3.8-27b-nvfp4"))
    with_vllm(tmp_path, monkeypatch)

    entries, lines = listing_of(unit_world)

    forbidden = entries.groups[1]
    assert forbidden.header == "available once llm.service is stopped" and not forbidden.numbered
    assert {e.name for e in forbidden.entries} == {NVIDIA}
    assert entries.pick(NVIDIA) == sentence_for()
    assert lines[-1] == sentence_for()


def test_the_sentence_is_recorded_and_not_said_twice(unit_world):
    ledger = speed_stats.default_speed_stats_path(unit_world.g)

    listing_of(unit_world)

    assert not speed_stats.foreign_unit_changed(ledger, "llm.service", UNIT_SERVES)
    _, again = listing_of(unit_world)
    assert sentence_for() not in again
    assert again[1] == "▸ llama.cpp · 127.0.0.1:8080 · llm.service (yours)"


def test_the_sentence_comes_back_once_when_the_unit_serves_another_model(unit_world):
    listing_of(unit_world)

    _, changed = listing_of(unit_world, served="flash")
    _, after = listing_of(unit_world, served="flash")

    assert changed[-1] == sentence_for("flash") and sentence_for("flash") not in after


def test_building_the_list_alone_records_nothing(unit_world):
    entries_of(unit_world, {PRIMARY_URL: (UNIT_SERVES,)}, memory=MEMORY)

    assert speed_stats.foreign_unit_changed(speed_stats.default_speed_stats_path(unit_world.g), "llm.service", UNIT_SERVES)


def test_a_foreign_process_that_is_no_unit_keeps_the_imperative_sentence_and_records_nothing(tmp_path, monkeypatch):
    world = serve_world(tmp_path, monkeypatch, peers=[flash(tmp_path)])
    holder = workspace.Holder(None, 77, "llama-server", 1000, None)
    monkeypatch.setattr(workspace, "port_holder", lambda port: holder)

    entries, lines = listing_of(world)

    assert entries.foreign is None and entries.foreign_key is None
    assert not [line for line in lines if "systemctl" in line]
    assert entries.groups[1].entries[0].refusal == workspace.incumbent_sentence(
        holder, port=8080, form="imperative", served=PRIMARY, target="flash-next")
    ledger = speed_stats.default_speed_stats_path(world.g)
    assert speed_stats.foreign_unit_changed(ledger, "llm.service", UNIT_SERVES)
