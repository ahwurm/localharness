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
    assert server.child_environment("vllm", {"HF_HUB_OFFLINE": "0", "MAX_JOBS": "3"}) == (None, [])


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
    assert not started[0][1].get("hf_offline")


def _plant_link(snapshot):
    assert [c.repo_id for c in model_library.snapshots()] == [REPO]
    (snapshot / "tokenizer.json").symlink_to("/etc/hostname")


def test_a_link_planted_after_the_listing_refuses_the_launch(cache, tmp_path):
    _plant_link(cache.add(REPO))
    with pytest.raises(server.LaunchRefused) as refused:
        _plan(_spec(), tmp_path)
    assert str(refused.value) == (
        f"Qwen3.8-27B-NVFP4 cannot run here: its snapshot holds an entry the harness does not load {REFUSAL_TAIL}")


def test_a_cache_that_cannot_be_listed_refuses_the_launch(cache, tmp_path, monkeypatch):
    def unreadable():
        raise OSError("unreadable")

    monkeypatch.setattr(model_library, "_cache_repos", unreadable)
    with pytest.raises(server.LaunchRefused) as refused:
        _plan(_spec(), tmp_path)
    assert str(refused.value) == f"Qwen3.8-27B-NVFP4 cannot run here: its Hugging Face cache cannot be read {REFUSAL_TAIL}"


def test_a_cached_repo_with_no_safetensors_refuses_the_launch(cache, tmp_path):
    (cache.add(REPO) / "model.safetensors").unlink()
    with pytest.raises(server.LaunchRefused) as refused:
        _plan(_spec(), tmp_path)
    assert str(refused.value) == (
        "Qwen3.8-27B-NVFP4 cannot run here: its snapshot holds no safetensors weights of a text-generation model "
        + REFUSAL_TAIL)


def test_an_auto_map_written_after_the_listing_refuses_the_launch(cache, tmp_path):
    snapshot = cache.add(REPO)
    assert [c.repo_id for c in model_library.snapshots()] == [REPO]
    config = json.loads((snapshot / "config.json").read_text("utf-8"))
    (snapshot / "config.json").write_text(json.dumps({**config, "auto_map": {"AutoModel": "x.Y"}}), encoding="utf-8")
    with pytest.raises(server.LaunchRefused, match="needs trust_remote_code") as refused:
        _plan(_spec(), tmp_path)
    assert str(refused.value).endswith(REFUSAL_TAIL)


async def test_a_refused_snapshot_starts_nothing(cache, tmp_path, monkeypatch):
    _plant_link(cache.add(REPO))
    monkeypatch.setattr(server, "start_server", lambda *a, **kw: pytest.fail("a server was started"))
    with pytest.raises(server.LaunchRefused, match=REFUSAL_TAIL):
        await lifecycle.ManagedVllmStrategy().activate(_spec(), tmp_path, BASE, autosense=AUTO)


@pytest.mark.parametrize("floor", [15_360, None])
async def test_a_swap_to_a_refused_snapshot_leaves_the_incumbent_serving(cache, tmp_path, monkeypatch, floor):
    _plant_link(cache.add(REPO))
    calls = []
    monkeypatch.setattr(server, "stop_server", lambda *a, **kw: calls.append("stop") or True)
    monkeypatch.setattr(server, "start_server", lambda *a, **kw: calls.append("start") or 4321)
    incumbent = lifecycle.Heavy(ManagedServerConfig(
        runtime="llamacpp", binary="/x/llama-server", model="/m/a.gguf", served_name="incumbent", port=8080),
        "http://127.0.0.1:8080/v1")
    lines = []

    class Reporter:
        async def line(self, text):
            lines.append(text)

        async def freed(self):
            calls.append("freed")

    outcome = await lifecycle.swap(
        incumbent, lifecycle.Heavy(_spec(), BASE), config_dir=tmp_path, autosense=AUTO, reporter=Reporter(),
        redact=lambda t: t, floor=floor)
    assert outcome.stage == "refused" and calls == []
    assert lines[0].endswith(f"{REFUSAL_TAIL} incumbent keeps serving.")


def test_the_units_exec_environment_is_offline_only_for_a_snapshot_launch(tmp_path):
    from localharness.cli import serve_cmd

    spec = _spec()
    offline = serve_cmd._exec_environment({"PATH": "/b"}, "/h", {}, spec, None, None, hf_offline=True)
    assert offline["HF_HUB_OFFLINE"] == "1" and offline["PATH"] == "/b"
    assert "HF_HUB_OFFLINE" not in serve_cmd._exec_environment({"PATH": "/b"}, "/h", {}, spec, None, None)


def _unit_vllm(tmp_path, monkeypatch):
    from tests.unit.test_serve_cmd import serve_world

    return serve_world(tmp_path, monkeypatch, runtime="vllm", server_extra={"model": REPO, "port": 8085})


def test_the_unit_execs_the_snapshot_argv_offline(tmp_path, monkeypatch):
    from tests.unit.test_serve_cmd import _launched

    world = _unit_vllm(tmp_path, monkeypatch)
    snapshot = Cache(tmp_path / "cache", monkeypatch)
    (tmp_path / "cache").mkdir()
    shot = snapshot.add(REPO)
    path, argv, env = _launched(world, world.run())
    assert argv[:3] == [str(world.binary), "serve", str(shot)]
    assert argv[argv.index("--served-model-name") + 1] == REPO
    assert env["HF_HUB_OFFLINE"] == "1" and path == str(world.binary)


def test_the_unit_execs_an_uncached_repo_as_before(tmp_path, monkeypatch):
    from tests.unit.test_serve_cmd import _launched

    world = _unit_vllm(tmp_path, monkeypatch)
    _path, argv, env = _launched(world, world.run())
    assert argv[2] == REPO and "--served-model-name" not in argv and "HF_HUB_OFFLINE" not in env
