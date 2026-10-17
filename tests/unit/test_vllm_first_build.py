"""A vLLM checkpoint's first launch on this box says it compiles kernels; the launch that answered is remembered."""
from __future__ import annotations

import contextlib

import pytest
import yaml

from localharness.agent import context as agent_context
from localharness.cli import start_serving
from localharness.cli.model_ops import resolve_serving
from localharness.cli.workspace import MachineTrust
from localharness.config.loader import ConfigLoader
from localharness.config.models import AutosenseConfig, ManagedServerConfig
from localharness.provider import autosense, lifecycle, server, speed_stats
from localharness.provider.autosense import FreeMemory
from tests.unit.test_vllm_snapshot_launch import REPO, Cache

AUTO = AutosenseConfig()
PRIMARY = "qwen-27b-swift-1.5"
PEER = "nvfp4"
PEER_URL = "http://127.0.0.1:8081/v1"
CARD = int(31.8 * 2**30)
DERIVED = "reasoning qwen3 · tools qwen3_coder"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_context, "probe_window", lambda *a, **kw: agent_context.WindowProbe(100_864, 100_864))
    monkeypatch.setattr(autosense, "free_memory", lambda ratios: FreeMemory(CARD, CARD, "nvidia-smi"))
    return Cache(tmp_path / "hf", monkeypatch)


def _vllm(**over):
    return ManagedServerConfig(**{"runtime": "vllm", "launch": "binary", "binary": "/x/vllm", "model": REPO,
                                  "port": 8081, **over})


def _llama():
    return ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server", model="/m/a.gguf", port=8081)


def _ledger(home):
    return speed_stats.default_speed_stats_path(home)


class World:
    def __init__(self, home, monkeypatch) -> None:
        self.home = home
        self.started: list[list[str]] = []
        peer = {"name": PEER, "base_url": PEER_URL, "provider_type": "vllm", "gpu": True,
                "lifecycle": {"gpu": True, "runtime": "vllm", "launch": "binary", "binary": "/x/vllm",
                              "model": REPO, "port": 8081}}
        data = {
            "version": "1",
            "provider": {"provider_type": "llamacpp", "base_url": "http://127.0.0.1:8080/v1",
                         "default_model": PRIMARY, "available_models": [PRIMARY]},
            "org": {"default_model": PRIMARY, "audit_log_path": str(home / "audit.jsonl")},
            "extra_endpoints": [peer],
            "server": {"runtime": "llamacpp", "binary": "/x/llama-server", "model": "/m/a.gguf",
                       "served_name": PRIMARY, "port": 8080},
        }
        (home / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
        self.harness = ConfigLoader(config_dir=home).load_harness()

        async def wait_ready(base_url, **kw):
            return [REPO]

        @contextlib.asynccontextmanager
        async def lock(config_dir, *, on_wait=None):
            yield

        monkeypatch.setattr(lifecycle, "launch_lock", lock)
        monkeypatch.setattr(lifecycle, "GPU_FREE_SETTLE_SECONDS", 0.0)
        monkeypatch.setattr(server, "wait_ready", wait_ready)
        monkeypatch.setattr(server, "start_server", lambda cfg, cmd, **kw: self.started.append(list(cmd)) or 4321)

    async def swap(self) -> str:
        """Launch the peer through `run_swap` and return the clause its launch line would carry."""
        target = resolve_serving(self.harness, MachineTrust(), PEER, config_dir=self.home)
        kept: list[str] = []
        lines: list[str] = []

        async def line(text):
            lines.append(text)

        reporter = start_serving.ConsoleSwapReporter(lambda text, **kw: lines.append(text), self.harness)
        outcome = await start_serving.run_swap(
            self.harness, None, target, config_dir=self.home, autosense=AUTO, reporter=reporter,
            redact=reporter.redact, on_plan=lambda plan: kept.append(start_serving.plan_clause(plan)))
        assert outcome.error is None, outcome.error
        return kept[0]


@pytest.fixture
def world(tmp_path, monkeypatch, cache):
    cache.add(REPO)
    return World(tmp_path, monkeypatch)


async def test_the_first_launch_of_a_checkpoint_says_it_compiles_kernels_and_the_second_does_not(world):
    spec = _vllm()
    assert lifecycle.first_build(spec, world.home) is True
    first = await world.swap()
    assert first == f"{DERIVED}; {start_serving.FIRST_BUILD_CLAUSE}"
    assert isinstance(speed_stats._load(_ledger(world.home))[lifecycle.record_key(spec)]["served_here_at"], str)
    assert lifecycle.first_build(spec, world.home) is False
    assert await world.swap() == DERIVED


def test_the_clause_names_the_wait_and_the_busy_machine():
    assert start_serving.FIRST_BUILD_CLAUSE == "first launch of this checkpoint compiles kernels — several minutes, the machine will be busy"


def test_a_llama_cpp_launch_is_never_a_first_build(tmp_path):
    assert lifecycle.first_build(_llama(), tmp_path) is False
