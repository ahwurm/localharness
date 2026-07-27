"""Provider lifecycle layer (0.11 Phase A): LifecycleStrategy interface + ManagedVllmStrategy.

ADDITIVE scaffolding — the strategy DELEGATES to provider/server.py (zero behavior change) and
no caller routes through it yet (Phase C). These tests monkeypatch server.* on the module
object (the harness's module-attribute pattern) to prove the delegation chain and the NEW
name-based liveness, without launching anything real.
"""
from __future__ import annotations

from localharness.config.models import ManagedServerConfig
from localharness.provider import server
from localharness.provider.lifecycle import (
    LifecycleStrategy,
    LiveEndpoint,
    Liveness,
    ManagedVllmStrategy,
    SpawnedProcessStrategy,
)


def _docker_srv(**over) -> ManagedServerConfig:
    kw = dict(launch="docker", docker_image="img:tag", model="m", port=8000)
    kw.update(over)
    return ManagedServerConfig(**kw)


def _binary_srv(**over) -> ManagedServerConfig:
    kw = dict(launch="binary", binary="/usr/bin/vllm", model="m", port=8001)
    kw.update(over)
    return ManagedServerConfig(**kw)


def _llamacpp_srv(**over) -> ManagedServerConfig:
    kw = dict(
        runtime="llamacpp", launch="binary", binary="/x/llama-server",
        model="/x/model.gguf", port=8080,
        extra_args=["-c", "32768", "-a", "qwen3.6-35b-a3b"],
    )
    kw.update(over)
    return ManagedServerConfig(**kw)


# ---------------------------------------------------------------------------
# Interface shape
# ---------------------------------------------------------------------------


def test_managed_vllm_strategy_satisfies_protocol():
    """The concrete strategy structurally implements the LifecycleStrategy Protocol."""
    assert isinstance(ManagedVllmStrategy(), LifecycleStrategy)


# ---------------------------------------------------------------------------
# activate — serve_command -> start_server -> wait_ready (both launch modes)
# ---------------------------------------------------------------------------


async def test_activate_docker_delegates_and_returns_live_endpoint(tmp_path, monkeypatch):
    """activate runs the real server.py chain and returns a LiveEndpoint whose served_models
    come from wait_ready and whose handle is the CONTAINER NAME in docker mode (its stop
    target). on_poll + timeout are threaded straight through to wait_ready — no behavior added."""
    calls: dict = {}

    def fake_serve_command(spec):
        calls["serve_command"] = spec
        return ["CMD", "--model", "m"]

    def fake_start_server(config_dir, cmd):
        calls["start_server"] = (config_dir, cmd)
        return 4321

    async def fake_wait_ready(base_url, config_dir=None, timeout_seconds=1800.0, poll_seconds=3.0, on_poll=None):
        calls["wait_ready"] = (base_url, config_dir, timeout_seconds)
        if on_poll is not None:
            on_poll(1.0)
        return ["served-m"]

    monkeypatch.setattr(server, "serve_command", fake_serve_command)
    monkeypatch.setattr(server, "start_server", fake_start_server)
    monkeypatch.setattr(server, "wait_ready", fake_wait_ready)

    srv = _docker_srv()
    seen_poll: list[float] = []
    ep = await ManagedVllmStrategy().activate(
        srv, tmp_path, "http://localhost:8000/v1", timeout_seconds=42.0, on_poll=seen_poll.append,
    )

    assert isinstance(ep, LiveEndpoint)
    assert ep.base_url == "http://localhost:8000/v1"
    assert ep.served_models == ["served-m"]
    assert ep.handle == server.DOCKER_CONTAINER_NAME              # docker stop-handle = container name
    assert calls["serve_command"] is srv                          # spec passed through unchanged
    assert calls["start_server"] == (tmp_path, ["CMD", "--model", "m"])
    assert calls["wait_ready"] == ("http://localhost:8000/v1", tmp_path, 42.0)
    assert seen_poll == [1.0]                                     # on_poll delegated to wait_ready


async def test_activate_binary_handle_is_the_launched_pid(tmp_path, monkeypatch):
    """binary mode: the handle is the pid start_server returned (there the pid IS the server)."""
    monkeypatch.setattr(server, "serve_command", lambda spec: ["/usr/bin/vllm", "serve", "m"])
    monkeypatch.setattr(server, "start_server", lambda config_dir, cmd: 9999)

    async def fake_wait_ready(base_url, config_dir=None, **kw):
        return ["served-m"]

    monkeypatch.setattr(server, "wait_ready", fake_wait_ready)

    ep = await ManagedVllmStrategy().activate(_binary_srv(), tmp_path, "http://localhost:8001/v1")
    assert ep.handle == 9999
    assert ep.served_models == ["served-m"]


# ---------------------------------------------------------------------------
# stop — the #100 verified stop, with launch threaded from the spec
# ---------------------------------------------------------------------------


async def test_stop_delegates_to_verified_stop_server(tmp_path, monkeypatch):
    """stop -> server.stop_server(config_dir, launch=spec.launch) — the existing verified stop."""
    seen: dict = {}

    def fake_stop(config_dir, launch="binary", **kw):
        seen["args"] = (config_dir, launch)
        return True

    monkeypatch.setattr(server, "stop_server", fake_stop)
    await ManagedVllmStrategy().stop(_docker_srv(), tmp_path)
    assert seen["args"] == (tmp_path, "docker")


# ---------------------------------------------------------------------------
# liveness — NAME-based for docker (the fix), pid-based for binary
# ---------------------------------------------------------------------------


def test_liveness_docker_is_name_based_never_pidfile(tmp_path, monkeypatch):
    """docker liveness reads the CONTAINER by name (docker_container_running); server_pid — the
    docker-run CLIENT pid — must NEVER be consulted in docker mode (the bug this fixes)."""
    def _boom(*a, **k):
        raise AssertionError("server_pid must not be called for docker liveness")

    monkeypatch.setattr(server, "server_pid", _boom)

    monkeypatch.setattr(server, "docker_container_running", lambda name=server.DOCKER_CONTAINER_NAME: True)
    live = ManagedVllmStrategy().liveness(_docker_srv(), tmp_path)
    assert isinstance(live, Liveness)
    assert live.alive is True
    assert server.DOCKER_CONTAINER_NAME in (live.detail or "")

    monkeypatch.setattr(server, "docker_container_running", lambda name=server.DOCKER_CONTAINER_NAME: False)
    assert ManagedVllmStrategy().liveness(_docker_srv(), tmp_path).alive is False


def test_liveness_binary_uses_server_pid(tmp_path, monkeypatch):
    """binary liveness = server_pid is not None (correct there — the pidfile pid IS vLLM)."""
    monkeypatch.setattr(server, "server_pid", lambda config_dir: 1234)
    assert ManagedVllmStrategy().liveness(_binary_srv(), tmp_path).alive is True
    monkeypatch.setattr(server, "server_pid", lambda config_dir: None)
    assert ManagedVllmStrategy().liveness(_binary_srv(), tmp_path).alive is False


# ---------------------------------------------------------------------------
# SpawnedProcessStrategy (0.11 Phase B): the harness spawns llama.cpp itself.
# Same server.py delegation chain as ManagedVllmStrategy, but the stop-handle is
# always the launched PID, stop is always the pid-group teardown (launch="binary"),
# and liveness is always pid-based (the pidfile pid IS the real llama-server here —
# never the docker orphan-client-pid case #99/#100). Mocked, launches nothing real.
# ---------------------------------------------------------------------------


def test_spawned_process_strategy_satisfies_protocol():
    assert isinstance(SpawnedProcessStrategy(), LifecycleStrategy)


async def test_spawned_activate_delegates_and_handle_is_pid(tmp_path, monkeypatch):
    """activate runs serve_command -> start_server -> wait_ready and returns a LiveEndpoint whose
    handle is the launched PID (a spawned process, never a container name). on_poll + timeout are
    threaded straight through to wait_ready."""
    calls: dict = {}

    def fake_serve_command(spec):
        calls["serve_command"] = spec
        return ["llama-server", "-m", "x"]

    def fake_start_server(config_dir, cmd):
        calls["start_server"] = (config_dir, cmd)
        return 5555

    async def fake_wait_ready(base_url, config_dir=None, timeout_seconds=1800.0, poll_seconds=3.0, on_poll=None):
        calls["wait_ready"] = (base_url, config_dir, timeout_seconds)
        if on_poll is not None:
            on_poll(2.0)
        return ["qwen3.6-35b-a3b"]

    monkeypatch.setattr(server, "serve_command", fake_serve_command)
    monkeypatch.setattr(server, "start_server", fake_start_server)
    monkeypatch.setattr(server, "wait_ready", fake_wait_ready)

    srv = _llamacpp_srv()
    seen_poll: list[float] = []
    ep = await SpawnedProcessStrategy().activate(
        srv, tmp_path, "http://127.0.0.1:8080/v1", timeout_seconds=99.0, on_poll=seen_poll.append,
    )

    assert isinstance(ep, LiveEndpoint)
    assert ep.base_url == "http://127.0.0.1:8080/v1"
    assert ep.served_models == ["qwen3.6-35b-a3b"]
    assert ep.handle == 5555                                     # spawned stop-handle = the real pid
    assert calls["serve_command"] is srv
    assert calls["start_server"] == (tmp_path, ["llama-server", "-m", "x"])
    assert calls["wait_ready"] == ("http://127.0.0.1:8080/v1", tmp_path, 99.0)
    assert seen_poll == [2.0]


async def test_spawned_stop_uses_binary_verified_stop(tmp_path, monkeypatch):
    """stop -> server.stop_server(config_dir, launch="binary"): the pid-group teardown, ALWAYS —
    a spawned server is a single tracked process, never a docker-run client."""
    seen: dict = {}

    def fake_stop(config_dir, launch="binary", **kw):
        seen["args"] = (config_dir, launch)
        return True

    monkeypatch.setattr(server, "stop_server", fake_stop)
    await SpawnedProcessStrategy().stop(_llamacpp_srv(), tmp_path)
    assert seen["args"] == (tmp_path, "binary")


def test_spawned_liveness_is_pid_based_never_docker(tmp_path, monkeypatch):
    """spawned liveness = server_pid is not None (the pidfile pid IS the llama-server process);
    docker_container_running — the docker-only path — must NEVER be consulted here."""
    def _boom(*a, **k):
        raise AssertionError("docker_container_running must not be called for a spawned process")

    monkeypatch.setattr(server, "docker_container_running", _boom)

    monkeypatch.setattr(server, "server_pid", lambda config_dir: 5555)
    live = SpawnedProcessStrategy().liveness(_llamacpp_srv(), tmp_path)
    assert isinstance(live, Liveness)
    assert live.alive is True
    assert "5555" in (live.detail or "")

    monkeypatch.setattr(server, "server_pid", lambda config_dir: None)
    assert SpawnedProcessStrategy().liveness(_llamacpp_srv(), tmp_path).alive is False
