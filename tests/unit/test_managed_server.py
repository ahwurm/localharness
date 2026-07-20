"""Managed vLLM server: refarch profiles, config schema, lifecycle, /model plumbing."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from localharness.config.models import HarnessConfig, ManagedServerConfig, ProviderConfig
from localharness.provider import server
from localharness.provider.refarch import REF_ARCHS

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Reference architectures — data integrity
# ---------------------------------------------------------------------------


def test_refarch_profiles_complete():
    keys = [ra.key for ra in REF_ARCHS]
    assert len(keys) == len(set(keys))
    for ra in REF_ARCHS:
        assert ra.default_model
        assert ra.context_tokens >= 8192
        assert "--max-model-len" in ra.serve_extra_args
        assert (REPO_ROOT / ra.doc).exists(), f"{ra.key}: doc {ra.doc} missing"
        if ra.launch == "docker":
            assert ra.docker_image
        else:
            assert ra.pip_package


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


def _harness(server_cfg: ManagedServerConfig | None) -> HarnessConfig:
    return HarnessConfig(
        provider=ProviderConfig(
            provider_type="vllm",
            base_url="http://localhost:8081/v1",
            default_model="m",
        ),
        server=server_cfg,
    )


def test_server_block_roundtrips_yaml():
    from pydantic_yaml import parse_yaml_raw_as, to_yaml_str

    cfg = _harness(ManagedServerConfig(binary="/x/vllm", model="org/repo", extra_args=["--max-model-len", "65536"]))
    loaded = parse_yaml_raw_as(HarnessConfig, to_yaml_str(cfg))
    assert loaded.server is not None
    assert loaded.server.model == "org/repo"
    assert loaded.server.port == 8081


def test_server_block_optional_backcompat():
    assert _harness(None).server is None


def test_launch_target_validated():
    with pytest.raises(ValueError):
        ManagedServerConfig(launch="binary", model="m")  # no binary path
    with pytest.raises(ValueError):
        ManagedServerConfig(launch="docker", model="m")  # no image


# ---------------------------------------------------------------------------
# Serve command composition
# ---------------------------------------------------------------------------


def test_serve_command_binary():
    srv = ManagedServerConfig(binary="/opt/vllm", model="org/repo", port=8081, extra_args=["--max-model-len", "65536"])
    cmd = server.serve_command(srv)
    assert cmd[:3] == ["/opt/vllm", "serve", "org/repo"]
    assert "--port" in cmd and "8081" in cmd
    assert "--max-model-len" in cmd


def test_serve_command_docker():
    srv = ManagedServerConfig(launch="docker", docker_image="img:tag", model="org/repo", port=8081)
    cmd = server.serve_command(srv)
    assert cmd[:2] == ["docker", "run"]
    assert "img:tag" in cmd
    assert f"{8081}:8000" in " ".join(cmd)  # host port maps onto the container's 8000
    assert "--model" in cmd and "org/repo" in cmd


# ---------------------------------------------------------------------------
# Lifecycle: start / pid / stop / log tail
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.name == "nt",
    reason="managed server lifecycle uses POSIX process groups; Windows support tracked separately",
)
def test_start_stop_lifecycle(tmp_path):
    cmd = [sys.executable, "-c", "import time; print('up', flush=True); time.sleep(60)"]
    pid = server.start_server(tmp_path, cmd)
    try:
        assert server.server_pid(tmp_path) == pid
        assert server.pid_path(tmp_path).exists()
    finally:
        assert server.stop_server(tmp_path) is True
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and server.server_pid(tmp_path) is not None:
        time.sleep(0.1)
    assert server.server_pid(tmp_path) is None
    assert not server.pid_path(tmp_path).exists()
    assert "launch:" in server.log_tail(tmp_path)


@pytest.mark.skipif(
    os.name == "nt",
    reason="managed server lifecycle uses POSIX process groups; Windows support tracked separately",
)
def test_server_pid_stale_file_cleared(tmp_path):
    server.server_dir(tmp_path).mkdir(parents=True)
    server.pid_path(tmp_path).write_text("999999999")
    assert server.server_pid(tmp_path) is None
    assert not server.pid_path(tmp_path).exists()


# ---------------------------------------------------------------------------
# Readiness polling
# ---------------------------------------------------------------------------


class _Resp:
    status_code = 200

    @staticmethod
    def json():
        return {"data": [{"id": "served-model"}]}


class _StubAsyncClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, timeout=None):
        return _Resp()


@pytest.mark.asyncio
async def test_wait_ready_returns_served_models(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _StubAsyncClient)
    models = await server.wait_ready("http://localhost:8081/v1", timeout_seconds=5.0)
    assert models == ["served-model"]


@pytest.mark.asyncio
async def test_wait_ready_fails_fast_when_process_died(tmp_path):
    server.server_dir(tmp_path).mkdir(parents=True)
    server.log_path(tmp_path).write_text("boom: CUDA out of memory\n")
    # No pidfile → managed process is dead → RuntimeError with log tail, no polling.
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        await server.wait_ready("http://localhost:1/v1", config_dir=tmp_path, timeout_seconds=5.0)


# ---------------------------------------------------------------------------
# Local model registry (in-REPL full swap): named local checkpoints with
# per-model args, mounted into the docker server; picker offers them by name
# ---------------------------------------------------------------------------


def _registry_srv(**over) -> ManagedServerConfig:
    kw = dict(
        launch="docker", docker_image="img:tag", model="qwen3.6-35b-a3b", port=8000,
        extra_args=["--kv-cache-dtype", "fp8"],
        local_models=[
            {"name": "qwen3.6-35b-a3b", "path": "/x/Qwen35",
             "extra_args": ["--moe-backend", "marlin"]},
            {"name": "qwen3.6-27b", "path": "/x/Qwen27"},
        ],
    )
    kw.update(over)
    return ManagedServerConfig(**kw)


def test_local_model_registry_roundtrip():
    srv = _registry_srv()
    assert srv.entry_for("qwen3.6-27b").path == "/x/Qwen27"
    assert srv.entry_for("qwen3.6-27b").extra_args == []
    assert srv.entry_for("nope") is None


def test_serve_command_docker_registry_entry_mounts_and_names():
    srv = _registry_srv()
    joined = " ".join(server.serve_command(srv))
    assert "-v /x/Qwen35:/models/serving:ro" in joined      # checkpoint mounted read-only
    assert "--model /models/serving" in joined               # container path, not host path
    assert "--served-model-name qwen3.6-35b-a3b" in joined   # picker name == served id
    assert "--moe-backend marlin" in joined                  # per-model args appended
    assert "--kv-cache-dtype fp8" in joined                  # shared args retained

    srv.model = "qwen3.6-27b"
    joined27 = " ".join(server.serve_command(srv))
    assert "-v /x/Qwen27:/models/serving:ro" in joined27
    assert "--served-model-name qwen3.6-27b" in joined27
    assert "--moe-backend" not in joined27                   # MoE flag never leaks to the dense model


def test_serve_command_binary_registry_entry():
    srv = ManagedServerConfig(
        launch="binary", binary="/usr/bin/vllm", model="m27", port=8001,
        local_models=[{"name": "m27", "path": "/x/Q27", "extra_args": ["--enforce-eager"]}],
    )
    joined = " ".join(server.serve_command(srv))
    assert "serve /x/Q27" in joined
    assert "--served-model-name m27" in joined
    assert "--enforce-eager" in joined


class _SlowStubAsyncClient:
    """Fails twice, then answers — exercises the poll loop and its progress callback."""

    calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, timeout=None):
        import httpx

        type(self).calls += 1
        if type(self).calls < 3:
            raise httpx.ConnectError("not up yet")

        class _Resp:
            status_code = 200

            def json(self):
                return {"data": [{"id": "served-model"}]}

        return _Resp()


@pytest.mark.asyncio
async def test_wait_ready_reports_poll_progress(monkeypatch):
    import httpx

    _SlowStubAsyncClient.calls = 0
    monkeypatch.setattr(httpx, "AsyncClient", _SlowStubAsyncClient)
    seen: list[float] = []
    models = await server.wait_ready(
        "http://localhost:8081/v1", timeout_seconds=30.0, poll_seconds=0.01,
        on_poll=seen.append,
    )
    assert models == ["served-model"]
    assert len(seen) >= 2                      # called on each unready poll
    assert seen == sorted(seen)                # elapsed seconds are nondecreasing


def test_server_pid_zombie_is_dead(tmp_path):
    """#99: a crashed `docker run --rm` client sits as a ZOMBIE until the REPL reaps it —
    os.kill(pid, 0) succeeds on zombies, which defeated wait_ready's fail-fast (observed
    live: a quantization-mismatch crash left the loading line spinning toward the full
    1800s timeout instead of surfacing the crash log tail). A zombie is DEAD for
    lifecycle purposes: server_pid clears the stale pidfile and reports None."""
    server.server_dir(tmp_path).mkdir(parents=True)
    pid = os.fork()
    if pid == 0:
        os._exit(0)                # child dies instantly -> zombie until reaped
    time.sleep(0.2)
    server.pid_path(tmp_path).write_text(str(pid))
    try:
        assert server.server_pid(tmp_path) is None       # zombie == dead
        assert not server.pid_path(tmp_path).exists()    # stale pidfile cleared
    finally:
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass


# ---------------------------------------------------------------------------
# #100: VERIFIED docker stop — the old container must be GONE (name free), not
# just signalled, before a swap may relaunch; --rm removal races `docker run`
# ---------------------------------------------------------------------------


class _FakeDocker:
    """Scripted `docker` CLI: inspect returns rc per call from a queue (0 = name in use,
    1 = gone); every invocation is recorded."""

    def __init__(self, inspect_rcs):
        self.inspect_rcs = list(inspect_rcs)
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kw):
        self.calls.append(list(cmd))
        rc = 0
        if cmd[:2] == ["docker", "inspect"]:
            rc = self.inspect_rcs.pop(0) if self.inspect_rcs else 0

        class _R:
            returncode = rc
            stdout = b""
            stderr = b""
        return _R()


def _stop_docker(monkeypatch, tmp_path, inspect_rcs):
    fake = _FakeDocker(inspect_rcs)
    monkeypatch.setattr(server.subprocess, "run", fake)
    monkeypatch.setattr(server.time, "sleep", lambda s: None)
    server.stop_server(tmp_path, launch="docker")
    return fake


def test_docker_stop_verifies_name_free(monkeypatch, tmp_path):
    """Happy path: stop, then poll inspect until the name is gone. No force-remove."""
    fake = _stop_docker(monkeypatch, tmp_path, inspect_rcs=[0, 0, 1])
    joined = [" ".join(c) for c in fake.calls]
    assert any(c.startswith("docker stop") for c in joined)
    assert sum(c.startswith("docker inspect") for c in joined) == 3
    assert not any(c.startswith("docker rm") for c in joined)


def test_docker_stop_force_removes_when_name_sticks(monkeypatch, tmp_path):
    """--rm removal wedged: after the poll window, rm -f fires and the name frees."""
    stuck = [0] * 200 + [1]           # in use through the first window, freed after rm -f
    fake = _stop_docker(monkeypatch, tmp_path, inspect_rcs=stuck)
    joined = [" ".join(c) for c in fake.calls]
    assert any(c.startswith("docker rm -f") for c in joined)


def test_docker_stop_raises_when_name_never_frees(monkeypatch, tmp_path):
    """Fail explicit (#100): never return with the name still taken — a relaunch would
    race straight into the conflict this exists to prevent."""
    fake = _FakeDocker([0] * 1000)
    monkeypatch.setattr(server.subprocess, "run", fake)
    monkeypatch.setattr(server.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="still exists"):
        server.stop_server(tmp_path, launch="docker")
