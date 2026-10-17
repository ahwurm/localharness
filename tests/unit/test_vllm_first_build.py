"""A vLLM checkpoint's first launch on this box says it compiles kernels; the launch that answered is remembered."""
from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace

import pytest
import yaml

from localharness.agent import context as agent_context
from localharness.cli import start_serving
from localharness.cli.model_ops import resolve_serving
from localharness.cli.workspace import MachineTrust
from localharness.config.loader import ConfigLoader
from localharness.config.models import AutosenseConfig, EndpointRef, ManagedServerConfig
from localharness.provider import autosense, lifecycle, server, speed_stats, units
from localharness.provider.autosense import FreeMemory
from tests.unit.procfs_support import fake_proc  # noqa: F401 — the fixture
from tests.unit.test_launch_paths_repl import (  # noqa: F401 — the fixtures
    PEER_URL as REPL_PEER_URL,
    PRIMARY_DOWN,
    _launchable_peer,
    _liveness_not_launched,
    _no_machine_probes,
    _session,
    boundaries,
)
from tests.unit.test_vllm_snapshot_launch import REPO, Cache
from tests.unit.unit_strategy_support import unit_world

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


async def _planless(spec, home):
    return await lifecycle.ManagedVllmStrategy().activate(spec, home, PEER_URL, autosense=None)


async def test_a_planless_launch_is_stamped(world):
    spec = _vllm()
    assert lifecycle.first_build(spec, world.home) is True
    await _planless(spec, world.home)
    assert lifecycle.first_build(spec, world.home) is False


async def test_a_llama_cpp_launch_through_its_strategy_leaves_no_stamp(world):
    spec = _llama()
    await lifecycle.SpawnedProcessStrategy().activate(spec, world.home, PEER_URL, autosense=None)
    assert "served_here_at" not in speed_stats._load(_ledger(world.home)).get(lifecycle.record_key(spec), {})


def test_a_window_report_alone_counts_as_a_launch_that_answered(tmp_path):
    spec = _vllm()
    speed_stats.record_window_report(_ledger(tmp_path), "vllm", REPO, 100_864, 100_864, None)
    assert lifecycle.first_build(spec, tmp_path) is False


def test_with_no_config_directory_there_is_no_ledger_to_read(tmp_path):
    assert lifecycle.first_build(_vllm(), None) is False


def test_a_boot_attach_that_closes_a_marker_stamps_the_record(tmp_path):
    spec = _vllm()
    directory = server.server_dir(tmp_path)
    directory.mkdir(parents=True)
    marker = {"record": lifecycle.record_key(spec), "binary": "/x/vllm", "invocation": "inv-1", "cause": "remembered",
              "shape": "full", "planned_context": None, "full_fit": None, "free_bytes": None, "dropped": [],
              "at": "2026-10-13T11:59:30+00:00"}
    (directory / units.BOOT_MARKER_NAME).write_text(json.dumps(marker), encoding="utf-8")
    record = SimpleNamespace(spec=spec, base_url=PEER_URL, prefix="server")
    start_serving.boot_clause(tmp_path, record, autosense=AUTO, unit_state=units.UnitRuntimeState("inv-1", 0),
                              served_window=100_864)
    assert lifecycle.first_build(spec, tmp_path) is False


def test_the_unit_stamps_a_vllm_launch(tmp_path, monkeypatch, fake_proc):  # noqa: F811
    world = unit_world(tmp_path, monkeypatch, fake_proc)
    monkeypatch.setattr(server, "start_server", lambda *a, **kw: pytest.fail("the harness started a server"))
    spec = _vllm(model="/m/ckpt", port=8080)
    world.ready()
    world.http(served="/m/ckpt")
    serving = world.serve_on_restart(spec)

    async def go():
        async with lifecycle.launch_lock(world.g):
            return await world.strategy(spec).activate(spec, world.g, world.url(), timeout_seconds=60, autosense=None)

    assert lifecycle.first_build(spec, world.g) is True
    asyncio.run(go())
    serving.join()
    world.stop_servers()
    assert lifecycle.first_build(spec, world.g) is False


def _repl_peer():
    return EndpointRef(
        name=PEER, base_url=REPL_PEER_URL, provider_type="vllm", gpu=True, api_key="none",
        lifecycle=ManagedServerConfig(runtime="vllm", launch="binary", binary="/x/vllm", model=REPO, port=8080, gpu=True))


async def _cold_heads(tmp_path):
    _, repl, channel, _ = _session(tmp_path, _repl_peer())
    await repl._handle_slash(f"/model {PEER}")
    return [m for m in channel.messages if m.startswith(f"Launching {PEER}")]


async def test_the_repl_cold_head_warns_once_and_the_next_session_does_not(tmp_path, boundaries):  # noqa: F811
    (first,) = await _cold_heads(tmp_path)
    assert first.endswith(f" ({start_serving.FIRST_BUILD_CLAUSE})")
    (second,) = await _cold_heads(tmp_path)
    assert "compiles kernels" not in second


async def test_the_repl_restart_line_carries_the_clause_on_a_first_launch(tmp_path, boundaries):  # noqa: F811
    _, repl, channel, _ = _session(tmp_path, _launchable_peer(), live=PRIMARY_DOWN)
    await repl._handle_slash("/model model-a")
    (line,) = [m for m in channel.messages if m.startswith("Restarting managed vLLM with model-a")]
    assert f" ({start_serving.FIRST_BUILD_CLAUSE})" in line
