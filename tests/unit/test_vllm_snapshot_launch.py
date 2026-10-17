"""A vLLM binary launch of a cached checkpoint runs the verified snapshot directory, offline, under its repo id."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from localharness.agent import context as agent_context
from localharness.config.models import AutosenseConfig, ManagedServerConfig
from localharness.provider import autosense, lifecycle, model_library, server
from localharness.provider.autosense import FreeMemory
from tests.unit.gguf_support import write_sparse

AUTO = AutosenseConfig()
BASE = "http://127.0.0.1:8085/v1"
REPO = "nvidia/Qwen3.8-27B-NVFP4"
CARD = int(31.8 * 2**30)
CONFIG = Path(__file__).resolve().parents[1] / "fixtures" / "vllm_checkpoints" / "nvidia-qwen3.8-27b-nvfp4" / "config.json"
SAFETENSORS = 21_921_697_280
REFUSAL_TAIL = "— the harness launches only a verified checkpoint."


class Cache:
    def __init__(self, tmp_path, monkeypatch):
        self.tmp, self.repos = tmp_path, []
        monkeypatch.setattr(model_library, "_cache_repos", lambda: tuple(self.repos))

    def add(self, repo_id, config=None):
        root = self.tmp / "hfcache" / ("models--" + repo_id.replace("/", "--"))
        snapshot, blobs = root / "snapshots" / "abc", root / "blobs"
        blobs.mkdir(parents=True)
        snapshot.mkdir(parents=True)
        files = {"config.json": config or json.loads(CONFIG.read_text("utf-8")), "model.safetensors": SAFETENSORS}
        for name, data in files.items():
            blob = blobs / hashlib.sha1(name.encode()).hexdigest()
            if isinstance(data, int):
                write_sparse(blob, data)
            else:
                blob.write_text(json.dumps(data), encoding="utf-8")
            (snapshot / name).symlink_to(os.path.relpath(blob, snapshot))
        rev = SimpleNamespace(commit_hash="abc", snapshot_path=snapshot, refs=frozenset(("main",)), last_modified=1.0)
        self.repos.append(SimpleNamespace(repo_id=repo_id, repo_type="model", size_on_disk=1 << 30, revisions=[rev]))
        return snapshot


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_context, "probe_window", lambda *a, **kw: agent_context.WindowProbe(100_864, 100_864))
    monkeypatch.setattr(autosense, "free_memory", lambda ratios: FreeMemory(CARD, CARD, "nvidia-smi"))
    return Cache(tmp_path, monkeypatch)


def _spec(model=REPO, **over):
    return ManagedServerConfig(**{"runtime": "vllm", "launch": "binary", "binary": "/x/vllm", "model": model,
                                  "port": 8085, **over})


def _plan(spec, config_dir):
    sensed = lifecycle.sense_launch(spec, config_dir, AUTO)
    return lifecycle.plan_launch(
        spec, sensed, server.launch_decision(spec, False), autosense=AUTO, api_key_held=False)


def test_a_cached_repo_launches_from_its_snapshot_under_its_repo_id(cache, tmp_path):
    snapshot = cache.add(REPO)
    plan = _plan(_spec(), tmp_path)
    assert list(plan.argv) == [
        "/x/vllm", "serve", str(snapshot), "--port", "8085", "--served-model-name", REPO,
        "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder", "--enable-auto-tool-choice",
        "--max-model-len", "100864", "--host", "127.0.0.1"]
    assert "--trust-remote-code" not in plan.argv
    assert "HF_HUB_OFFLINE" in plan.env_names


def test_an_uncached_repo_launches_as_before(cache, tmp_path):
    plan = _plan(_spec(), tmp_path)
    assert plan.argv[2] == REPO and "--served-model-name" not in plan.argv
    assert "HF_HUB_OFFLINE" not in plan.env_names and plan.serve_kwargs.get("snapshot") is None


def test_a_docker_launch_keeps_the_repo_id(cache, tmp_path):
    cache.add(REPO)
    spec = _spec(launch="docker", binary=None, docker_image="vllm/vllm-openai")
    plan = _plan(spec, tmp_path)
    assert plan.argv[plan.argv.index("--model") + 1] == REPO
    assert "HF_HUB_OFFLINE" not in plan.env_names and plan.serve_kwargs.get("snapshot") is None


def test_the_launch_environment_is_offline_over_the_persons_shell():
    env, _dropped = server.child_environment("vllm", {"HF_HUB_OFFLINE": "0"}, hf_offline=True)
    assert env["HF_HUB_OFFLINE"] == "1"
    assert server.child_environment("vllm", {"HF_HUB_OFFLINE": "0"}) == (None, [])


def test_the_record_env_cannot_turn_the_launch_online():
    env, _dropped = server.child_environment("vllm", {}, record_env={"HF_HUB_OFFLINE": "0"}, hf_offline=True)
    assert env["HF_HUB_OFFLINE"] == "1"


async def test_activate_starts_the_snapshot_launch_offline(cache, tmp_path, monkeypatch):
    cache.add(REPO)
    started = []

    async def wait_ready(base_url, **kw):
        return [REPO]

    monkeypatch.setattr(server, "wait_ready", wait_ready)
    monkeypatch.setattr(server, "start_server", lambda cfg, cmd, **kw: started.append((cmd, kw)) or 4321)
    await lifecycle.ManagedVllmStrategy().activate(_spec(), tmp_path, BASE, autosense=AUTO)
    assert started[0][1]["hf_offline"] is True


async def test_activate_of_an_uncached_repo_is_not_offline(cache, tmp_path, monkeypatch):
    started = []

    async def wait_ready(base_url, **kw):
        return [REPO]

    monkeypatch.setattr(server, "wait_ready", wait_ready)
    monkeypatch.setattr(server, "start_server", lambda cfg, cmd, **kw: started.append((cmd, kw)) or 4321)
    await lifecycle.ManagedVllmStrategy().activate(_spec(), tmp_path, BASE, autosense=AUTO)
    assert started[0][1]["hf_offline"] is False
