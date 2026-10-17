"""A Strata record launches through its row on a fake engine tree; Popen and HTTP are faked and nothing runs."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from localharness.agent import context as agent_context
from localharness.config.models import AutosenseConfig
from localharness.provider import autosense, lifecycle, procfs, server, strata
from localharness.provider.autosense import FreeMemory
from tests.unit.launch_support import popen  # noqa: F401
from tests.unit.procfs_support import fake_proc  # noqa: F401
from tests.unit.strata_fixture import flash_next, strata_config, strata_record, strata_tree

BASE = "http://127.0.0.1:8083/v1"
SERVED = "swift-1.5-iq3_xxs"


class World:
    def __init__(self, tmp_path: Path, monkeypatch, started: list, config: Callable[[Path], dict] | None = None):
        shard, projector = flash_next(tmp_path)
        self.shard, self.projector = shard, projector
        self.engine = strata_tree(tmp_path, config=config(shard) if config else None, model=shard)
        self.config_dir = tmp_path / "cfg"
        self.config_dir.mkdir()
        self.spec = strata_record(self.engine, shard, projector)
        self.health = {"loaded": True, "service": "strata", "max_context": 65536}
        self.models = {"data": [{"id": SERVED, "meta": {"n_ctx": 65536}}]}
        self.requests: list[str] = []

        def answer(request: httpx.Request) -> httpx.Response:
            self.requests.append(request.url.path)
            body = {"/health": self.health, "/v1/models": self.models}.get(request.url.path)
            return httpx.Response(200 if body is not None else 404, json=body)

        monkeypatch.setattr(server, "_TRANSPORT", httpx.MockTransport(answer))
        monkeypatch.setattr(server, "server_pid", lambda config_dir: 4242 if started else None)

    async def activate(self, **kw):
        return await strata.StrataStrategy().activate(self.spec, self.config_dir, BASE, **kw)

    @property
    def copy(self) -> dict:
        return json.loads(strata.copy_path(self.config_dir, 8083).read_text("utf-8"))


@pytest.fixture
def make_world(tmp_path, monkeypatch, popen):  # noqa: F811
    return lambda config=None: World(tmp_path, monkeypatch, popen, config)


@pytest.fixture
def world(make_world):
    return make_world()


async def test_a_strata_record_launches_through_its_row_end_to_end(world, popen, monkeypatch):  # noqa: F811
    monkeypatch.setenv("STRATA_SKIP_SHA256", "1")
    live = await world.activate(autosense=None)
    engine, copy = str(world.engine), str(strata.copy_path(world.config_dir, 8083))
    assert popen[0]["cmd"] == [
        f"{engine}/.venv/bin/python", f"{engine}/serve/server.py", "--engine", "strata", "--config", copy,
        "--port", "8083", "--host", "127.0.0.1"]
    assert popen[0]["cwd"] == engine
    assert "STRATA_SKIP_SHA256" not in popen[0]["env"] and live.dropped_env == ("STRATA_SKIP_SHA256",)
    assert world.copy["model_name"] == SERVED
    assert world.copy["args"][world.copy["args"].index("--native") + 1] == str(world.shard)
    assert live.served_models == [SERVED] and live.handle == 4242 and "/health" in world.requests


GIB = 2**30
SENSED = AutosenseConfig()
MISMATCH = (f"{SERVED}'s Strata config serves another file than its model: path — "
            "re-run Strata's setup with --gguf-dir, or fix the record.")


@pytest.fixture
def sensing(monkeypatch):
    monkeypatch.setattr(autosense, "free_memory", lambda ratios: FreeMemory(24 * GIB, 32 * GIB, "nvidia-smi"))
    monkeypatch.setattr(autosense, "_read_meminfo", lambda: {"MemTotal": 64 * GIB, "MemAvailable": 60 * GIB})


def with_args(*extra: str) -> Callable[[Path], dict]:
    def build(shard: Path) -> dict:
        config = strata_config(shard)
        config["args"] += extra
        return config
    return build


@pytest.mark.parametrize("planned", [False, True])
async def test_a_config_serving_another_file_is_refused_before_anything_starts(
        make_world, popen, sensing, planned):  # noqa: F811
    world = make_world(lambda shard: strata_config("/elsewhere/other.gguf"))
    with pytest.raises(server.LaunchRefused) as refused:
        await world.activate(autosense=SENSED if planned else None)
    assert str(refused.value) == MISMATCH
    assert popen == [] and not strata.copy_path(world.config_dir, 8083).exists()


@pytest.mark.parametrize("config,named", [
    (lambda shard: strata_config(shard, env={"X": "1"}), "sets env"),
    (with_args("--open"), "passes --open"),
    (with_args("-x"), "passes -x"),
    (with_args("--api-key=abc"), "passes --api-key"),
])
async def test_an_unchecked_key_or_flag_refuses_naming_it(make_world, popen, config, named):  # noqa: F811
    world = make_world(config)
    with pytest.raises(server.LaunchRefused, match=named):
        await world.activate(autosense=None)
    assert popen == []


async def test_a_negative_number_after_a_flag_is_a_value_not_a_flag(make_world, popen):  # noqa: F811
    world = make_world(with_args("--prefill", "-1"))
    await world.activate(autosense=None)
    assert popen and world.copy["args"][-2:] == ["--prefill", "-1"]
    assert [strata.is_flag(t) for t in ("-1", "-0.5", "-x", "--kv", "q8_0", "-")] == [False, False, True, True, False, True]


def max_context(copy: dict) -> str:
    return copy["args"][copy["args"].index("--max-context") + 1]


@pytest.mark.parametrize("planned", [False, True])
async def test_a_record_context_over_the_configs_window_is_the_copys_window(world, sensing, planned):
    world.spec = world.spec.model_copy(update={"context": 32768})
    await world.activate(autosense=SENSED if planned else None)
    assert max_context(world.copy) == "32768"


@pytest.mark.parametrize("planned", [False, True])
async def test_without_a_record_context_the_copy_keeps_the_configs_window(world, sensing, planned):
    await world.activate(autosense=SENSED if planned else None)
    assert max_context(world.copy) == "65536"


def test_a_window_the_config_never_set_is_appended_to_its_arguments(tmp_path):
    shard, _ = flash_next(tmp_path)
    config = strata_config(shard)
    at = config["args"].index("--max-context")
    del config["args"][at:at + 2]
    engine = strata.EngineConfig(b"", config, "")
    copy = strata.copy_of(engine, strata_record(tmp_path, shard, shard), context=8192, port=8083, config_dir=tmp_path)
    assert copy["args"][-2:] == ["--max-context", "8192"]


def served_window(monkeypatch, bodies: dict):
    def get(url, **kw):
        if url not in bodies:
            raise httpx.ConnectError("down")
        return SimpleNamespace(json=lambda: bodies[url])

    monkeypatch.setattr(httpx, "get", get)
    return lambda: agent_context._probe_strata("http://h/v1", SERVED, {})


def test_the_served_window_is_the_models_n_ctx_and_the_trained_one_its_n_ctx_train(monkeypatch):
    probe = served_window(monkeypatch, {"http://h/v1/models": {"data": [
        {"id": SERVED, "meta": {"n_ctx": 65536, "n_ctx_train": 262144}}]}})
    assert probe() == agent_context.WindowProbe(65536, 262144)


def test_without_an_n_ctx_the_served_window_is_the_healths_max_context(monkeypatch):
    probe = served_window(monkeypatch, {"http://h/v1/models": {"data": [{"id": SERVED}]},
                                        "http://h/health": {"max_context": 65536}})
    assert probe() == agent_context.WindowProbe(65536, None)
    assert served_window(monkeypatch, {})() == agent_context.WindowProbe(None, None)


def _running(fake_proc, spec, *, exe):  # noqa: F811
    argv = (strata.interpreter(spec), strata.serve_py(spec), "--engine", "strata", "--config", "/c/8083.json",
            "--port", str(spec.port), "--host", "127.0.0.1")
    fake_proc.process(4242, argv=argv, exe=exe)
    record = procfs.read_process(4242)
    assert isinstance(record, procfs.ProcRecord)
    return record


def test_the_venv_python_serving_strata_matches_the_identity_and_the_engine_child_does_not(fake_proc, world):  # noqa: F811
    spec = world.spec
    identity = lifecycle.record_identity(spec)
    assert identity == strata.StrataStrategy().launch_identity(spec)
    assert procfs.identity_mismatch(_running(fake_proc, spec, exe=strata.interpreter(spec)), identity) is None
    assert procfs.identity_mismatch(_running(fake_proc, spec, exe=f"{world.engine}/engine/strata"), identity)
