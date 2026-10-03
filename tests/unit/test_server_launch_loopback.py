"""A model server the harness launches listens on this machine only, and a key it is told to require
reaches it through its environment — never its command line or serve.log.

- vLLM, docker: `-p 127.0.0.1:<port>:8000`. Only the published address changes: inside the container
  vLLM keeps listening on the container's own interfaces, which is what `-p` forwards to.
- vLLM, binary: `--host 127.0.0.1` after `--port` (a machine-level `extra_args` `--host` comes later
  and still wins).
- `server.bind_all: true` gives back 0.16's argv exactly; llama.cpp and ollama were loopback already.
- Requiring a key is opt-in (`server.require_api_key`, default false): `required_key_kwargs` is the
  one place that decides it, so with the default every launch and probe is called as in 0.16.

Every launch here is recorded; no process or container is started.
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr

from localharness.config.models import ManagedServerConfig
from localharness.provider import lifecycle, server

KEY = "sk-K"


def _hf_cache() -> str:
    return str(Path("~/.cache/huggingface").expanduser())


def _docker(**over) -> ManagedServerConfig:
    return ManagedServerConfig(**{"launch": "docker", "docker_image": "img:tag", "model": "org/repo",
                                  "port": 8081, **over})


def _binary(**over) -> ManagedServerConfig:
    return ManagedServerConfig(**{"launch": "binary", "binary": "/opt/vllm", "model": "org/repo",
                                  "port": 8081, "extra_args": ["--max-model-len", "65536"], **over})


# ------------------------------------------------------------------------------- the argv


def test_a_docker_vllm_publishes_on_this_machine_only():
    assert server.serve_command(_docker()) == [
        "docker", "run", "--rm", "--name", server.DOCKER_CONTAINER_NAME, "--gpus", "all", "--ipc=host",
        "-v", f"{_hf_cache()}:/root/.cache/huggingface",
        "-p", "127.0.0.1:8081:8000",
        "img:tag", "--model", "org/repo",
    ]


def test_bind_all_gives_back_the_old_docker_argv_exactly():
    assert server.serve_command(_docker(bind_all=True)) == [
        "docker", "run", "--rm", "--name", server.DOCKER_CONTAINER_NAME, "--gpus", "all", "--ipc=host",
        "-v", f"{_hf_cache()}:/root/.cache/huggingface",
        "-p", "8081:8000",
        "img:tag", "--model", "org/repo",
    ]


def test_a_keyed_docker_launch_names_the_variable_never_its_value():
    cmd = server.serve_command(_docker(), keyed=True)
    assert cmd[cmd.index("-p") - 2:cmd.index("-p") + 2] == ["-e", "VLLM_API_KEY", "-p", "127.0.0.1:8081:8000"]
    assert not any("=" in part and "VLLM_API_KEY" in part for part in cmd)


def test_a_binary_vllm_gets_host_127_0_0_1():
    assert server.serve_command(_binary()) == [
        "/opt/vllm", "serve", "org/repo", "--port", "8081", "--host", "127.0.0.1",
        "--max-model-len", "65536",
    ]


def test_bind_all_gives_back_the_old_binary_argv_exactly():
    assert server.serve_command(_binary(bind_all=True)) == [
        "/opt/vllm", "serve", "org/repo", "--port", "8081", "--max-model-len", "65536",
    ]


def test_a_keyed_binary_launch_has_the_same_argv():
    assert server.serve_command(_binary(), keyed=True) == server.serve_command(_binary())


def test_a_registry_entry_keeps_its_served_name_after_the_host():
    srv = _binary(model="m27", extra_args=[],
                  local_models=[{"name": "m27", "path": "/x/Q27", "extra_args": ["--enforce-eager"]}])
    assert server.serve_command(srv) == [
        "/opt/vllm", "serve", "/x/Q27", "--port", "8081", "--host", "127.0.0.1",
        "--served-model-name", "m27", "--enforce-eager",
    ]


def test_a_machine_level_host_in_extra_args_still_wins():
    cmd = server.serve_command(_binary(extra_args=["--host", "100.64.0.7"]))
    assert cmd[-2:] == ["--host", "100.64.0.7"]  # vLLM takes the last value


@pytest.mark.parametrize("bind_all", [False, True])
def test_llama_cpp_and_ollama_argv_are_unchanged(bind_all):
    llama = ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server", model="/x/m.gguf",
                                port=8080, extra_args=["-c", "4096"], bind_all=bind_all)
    assert server.serve_command(llama) == [
        "/x/llama-server", "-m", "/x/m.gguf", "--host", "127.0.0.1", "--port", "8080", "-c", "4096"]
    ollama = ManagedServerConfig(runtime="ollama", model="qwen2.5:0.5b", port=11434, bind_all=bind_all)
    assert server.serve_command(ollama) == [
        "env", "OLLAMA_HOST=127.0.0.1:11434", "OLLAMA_KEEP_ALIVE=-1", "ollama", "serve"]


# ------------------------------------------------------------------------------- the launch


class _Popen:
    """Records what start_server hands subprocess.Popen; starts nothing."""

    calls: list[dict] = []

    def __init__(self, cmd, **kwargs):
        type(self).calls.append({"cmd": cmd, **kwargs})
        self.pid = 4242


@pytest.fixture
def popen(monkeypatch):
    _Popen.calls = []
    monkeypatch.setattr(server.subprocess, "Popen", _Popen)
    return _Popen.calls


def test_a_required_key_travels_in_the_environment_not_in_argv_or_the_log(tmp_path, popen):
    assert server.start_server(tmp_path, ["true"], api_key=KEY) == 4242
    (call,) = popen
    assert call["cmd"] == ["true"]
    assert call["env"]["VLLM_API_KEY"] == KEY
    assert call["env"]["PATH"] == os.environ["PATH"]  # the rest of the environment is inherited
    log = server.log_path(tmp_path).read_text(encoding="utf-8")
    assert "=== launch: true ===" in log and KEY not in log


def test_without_a_key_the_launch_inherits_the_environment(tmp_path, popen):
    server.start_server(tmp_path, ["true"])
    assert popen[0]["env"] is None


# ------------------------------------------------------------------------------- the probes


def _keyed_transport(seen: list[httpx.Request], key: str = KEY) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.headers.get("Authorization") != f"Bearer {key}":
            return httpx.Response(401, json={"error": "Unauthorized"})
        return httpx.Response(200, json={"data": [{"id": "m"}]})
    return httpx.MockTransport(handler)


async def test_wait_ready_sends_a_required_key_on_every_poll(monkeypatch):
    seen: list[httpx.Request] = []
    monkeypatch.setattr(server, "_TRANSPORT", _keyed_transport(seen))
    assert await server.wait_ready("http://127.0.0.1:8081/v1", api_key=KEY, timeout_seconds=5.0) == ["m"]
    assert [r.headers["Authorization"] for r in seen] == [f"Bearer {KEY}"]


async def test_wait_ready_without_a_key_sends_no_authorization(monkeypatch):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": [{"id": "m"}]})

    monkeypatch.setattr(server, "_TRANSPORT", httpx.MockTransport(handler))
    assert await server.wait_ready("http://127.0.0.1:8081/v1", timeout_seconds=5.0) == ["m"]
    assert len(seen) == 1 and "Authorization" not in seen[0].headers


def test_the_key_helpers():
    assert server.auth_headers(SecretStr(KEY)) == {"Authorization": f"Bearer {KEY}"}
    assert server.auth_headers(KEY) == {"Authorization": f"Bearer {KEY}"}
    assert server.auth_headers("none") == server.auth_headers("") == server.auth_headers(None) == {}
    assert server.key_kwargs(None) == server.key_kwargs("none") == server.key_kwargs(SecretStr("")) == {}
    assert server.key_kwargs(KEY) == server.key_kwargs(SecretStr(KEY)) == {"api_key": KEY}


def test_a_key_is_required_only_when_the_server_is_set_to_require_it():
    assert server.required_key_kwargs(_binary(require_api_key=True), SecretStr(KEY)) == {"api_key": KEY}
    assert server.required_key_kwargs(_binary(), SecretStr(KEY)) == {}  # the default: off
    assert server.required_key_kwargs(_binary(require_api_key=True), SecretStr("none")) == {}
    assert server.required_key_kwargs(_binary(require_api_key=True), "") == {}
    assert server.required_key_kwargs(None, SecretStr(KEY)) == {}
    # only a real `true` counts: a stand-in object whose attribute is merely truthy is not a setting
    assert server.required_key_kwargs(SimpleNamespace(require_api_key="yes"), KEY) == {}


# ------------------------------------------------------------------------------- the strategies


def _record_chain(monkeypatch) -> dict:
    calls: dict = {}

    def serve_command(spec, **kw):
        calls["serve_command"] = kw
        return ["CMD"]

    def start_server(config_dir, cmd, **kw):
        calls["start_server"] = kw
        return 7

    async def wait_ready(base_url, **kw):
        calls["wait_ready"] = {k: v for k, v in kw.items() if k == "api_key"}
        return ["m"]

    monkeypatch.setattr(server, "serve_command", serve_command)
    monkeypatch.setattr(server, "start_server", start_server)
    monkeypatch.setattr(server, "wait_ready", wait_ready)
    return calls


async def test_a_keyed_vllm_activation_threads_the_key_through_launch_and_readiness(tmp_path, monkeypatch):
    calls = _record_chain(monkeypatch)
    await lifecycle.ManagedVllmStrategy().activate(_docker(), tmp_path, "http://127.0.0.1:8081/v1",
                                                   api_key=KEY)
    assert calls == {"serve_command": {"keyed": True}, "start_server": {"api_key": KEY},
                     "wait_ready": {"api_key": KEY}}


async def test_a_keyless_vllm_activation_calls_everything_as_before(tmp_path, monkeypatch):
    calls = _record_chain(monkeypatch)
    await lifecycle.ManagedVllmStrategy().activate(_docker(), tmp_path, "http://127.0.0.1:8081/v1")
    assert calls == {"serve_command": {}, "start_server": {}, "wait_ready": {}}


async def test_a_spawned_server_gets_the_key_on_its_readiness_probe_only(tmp_path, monkeypatch):
    calls = _record_chain(monkeypatch)
    spec = ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server", model="/x/m.gguf", port=8080)
    await lifecycle.SpawnedProcessStrategy().activate(spec, tmp_path, "http://127.0.0.1:8080/v1",
                                                      api_key=KEY)
    assert calls == {"serve_command": {}, "start_server": {}, "wait_ready": {"api_key": KEY}}


@pytest.mark.parametrize("cls", [lifecycle.LifecycleStrategy, lifecycle.ManagedVllmStrategy,
                                 lifecycle.SpawnedProcessStrategy, lifecycle.DaemonStrategy,
                                 lifecycle.LmsStrategy])
def test_every_strategy_accepts_a_key(cls):
    param = inspect.signature(cls.activate).parameters["api_key"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is None


# ------------------------------------------------------------------------------- the other probes


def _record_get(monkeypatch) -> list[dict]:
    """httpx.get as a recorder answering one served model; returns the headers each call sent."""
    sent: list[dict] = []

    def get(url, **kw):
        sent.append(dict(kw.get("headers") or {}))
        return httpx.Response(200, json={"data": [{"id": "m", "max_model_len": 4096}]})

    monkeypatch.setattr(httpx, "get", get)
    return sent


def test_list_live_models_sends_a_key_only_when_given_one(monkeypatch):
    from localharness.cli import model_ops

    sent = _record_get(monkeypatch)
    assert model_ops.list_live_models("http://127.0.0.1:8081/v1", api_key=KEY) == (["m"], True)
    assert model_ops.list_live_models("http://127.0.0.1:8081/v1") == (["m"], True)
    assert sent == [{"Authorization": f"Bearer {KEY}"}, {}]


def test_the_window_probe_sends_a_key_only_when_given_one(monkeypatch):
    from localharness.agent.context import probe_served_window

    sent = _record_get(monkeypatch)
    assert probe_served_window("http://127.0.0.1:8081/v1", "m", "vllm", api_key=KEY) == 4096
    assert probe_served_window("http://127.0.0.1:8081/v1", "m", "vllm") == 4096
    assert sent == [{"Authorization": f"Bearer {KEY}"}, {}]
