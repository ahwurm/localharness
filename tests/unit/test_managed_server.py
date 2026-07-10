"""Managed vLLM server: refarch profiles, config schema, lifecycle, /model plumbing."""
from __future__ import annotations

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
