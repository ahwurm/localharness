"""A model server the harness launches listens on this machine only, and a key it is told to require
reaches it through its environment — never its command line or serve.log.

- vLLM, docker: `-p 127.0.0.1:<port>:8000`. Only the published address changes: inside the container
  vLLM keeps listening on the container's own interfaces, which is what `-p` forwards to.
- vLLM, binary and llama.cpp: `--host 127.0.0.1` is the LAST argument whenever the launch binds
  loopback, so an `extra_args` `--host` can never widen it (the server keeps the last `--host`).
- `server.bind_all: true` gives back 0.16's argv exactly for vLLM; llama.cpp gains `--host 0.0.0.0`
  (before `extra_args`); ollama was loopback already.
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
from tests.unit.launch_support import PopenRecorder, popen, record_chain, record_opens  # noqa: F401

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
        "/opt/vllm", "serve", "org/repo", "--port", "8081", "--max-model-len", "65536",
        "--host", "127.0.0.1",
    ]


def test_bind_all_gives_back_the_old_binary_argv_exactly():
    assert server.serve_command(_binary(bind_all=True)) == [
        "/opt/vllm", "serve", "org/repo", "--port", "8081", "--max-model-len", "65536",
    ]


def test_a_keyed_binary_launch_has_the_same_argv():
    assert server.serve_command(_binary(), keyed=True) == server.serve_command(_binary())


def test_a_registry_entry_keeps_its_served_name_and_the_host_comes_last():
    srv = _binary(model="m27", extra_args=[],
                  local_models=[{"name": "m27", "path": "/x/Q27", "extra_args": ["--enforce-eager"]}])
    assert server.serve_command(srv) == [
        "/opt/vllm", "serve", "/x/Q27", "--port", "8081",
        "--served-model-name", "m27", "--enforce-eager", "--host", "127.0.0.1",
    ]


def test_an_extra_args_host_never_widens_a_loopback_launch():
    cmd = server.serve_command(_binary(extra_args=["--host", "100.64.0.7"]))
    assert cmd[-2:] == ["--host", "127.0.0.1"]  # the server takes the last value


def test_with_bind_all_an_extra_args_host_still_stands():
    cmd = server.serve_command(_binary(extra_args=["--host", "100.64.0.7"], bind_all=True))
    assert cmd[-2:] == ["--host", "100.64.0.7"]


def _llama(**over) -> ManagedServerConfig:
    return ManagedServerConfig(**{"runtime": "llamacpp", "binary": "/x/llama-server",
                                  "model": "/x/m.gguf", "port": 8080, **over})


def test_a_loopback_llama_cpp_launch_ends_with_the_loopback_host():
    assert server.serve_command(_llama(extra_args=["-c", "4096"])) == [
        "/x/llama-server", "-m", "/x/m.gguf", "--port", "8080", "-c", "4096", "--host", "127.0.0.1"]


def test_a_bind_all_llama_cpp_launch_puts_its_host_before_extra_args():
    assert server.serve_command(_llama(extra_args=["-c", "4096"], bind_all=True)) == [
        "/x/llama-server", "-m", "/x/m.gguf", "--port", "8080", "--host", "0.0.0.0", "-c", "4096"]


@pytest.mark.parametrize("extra", [["--host", "0.0.0.0"], ["--host=0.0.0.0"], []])
def test_no_extra_args_host_outlives_the_narrowing_on_llama_cpp(extra):
    cmd = server.serve_command(_llama(extra_args=extra))
    assert cmd[-2:] == ["--host", "127.0.0.1"]


def test_a_bind_all_launch_narrowed_by_the_decision_ends_with_the_loopback_host():
    cmd = server.serve_command(_llama(bind_all=True), loopback=True)
    assert cmd[-2:] == ["--host", "127.0.0.1"] and "0.0.0.0" not in cmd


def test_a_narrower_extra_args_host_still_beats_the_bind_all_host():
    cmd = server.serve_command(_llama(bind_all=True, extra_args=["--host", "100.64.0.7"]))
    assert cmd[-2:] == ["--host", "100.64.0.7"] and cmd.index("0.0.0.0") < cmd.index("100.64.0.7")


def test_a_docker_launch_narrowed_by_the_decision_publishes_on_loopback():
    cmd = server.serve_command(_docker(bind_all=True), loopback=True)
    assert cmd[cmd.index("-p") + 1] == "127.0.0.1:8081:8000"


def test_a_vllm_binary_narrowed_by_the_decision_ends_with_the_loopback_host():
    assert server.serve_command(_binary(bind_all=True), loopback=True)[-2:] == ["--host", "127.0.0.1"]


@pytest.mark.parametrize("bind_all", [False, True])
def test_ollama_argv_is_unchanged(bind_all):
    ollama = ManagedServerConfig(runtime="ollama", model="qwen2.5:0.5b", port=11434, bind_all=bind_all)
    assert server.serve_command(ollama) == [
        "env", "OLLAMA_HOST=127.0.0.1:11434", "OLLAMA_KEEP_ALIVE=-1", "ollama", "serve"]


# ------------------------------------------------------------------------------- the record reaches argv


def _full_llama(**over) -> ManagedServerConfig:
    return _llama(model="~/m/x.gguf", served_name="qwen", mmproj="~/m/p.gguf", cache_type_k="q8_0",
                  cache_type_v="q8_0", context=65536, cache_ram=0, require_api_key=True,
                  api_key_file="/k/key", extra_args=["-fa", "on"], **over)


def test_every_record_field_reaches_the_llama_cpp_argv_in_order():
    cmd = server.serve_command(_full_llama(), keyed=True, loopback=True, key_file_ok=True)
    assert cmd == [
        "/x/llama-server", "-m", os.path.expanduser("~/m/x.gguf"), "-a", "qwen",
        "--mmproj", os.path.expanduser("~/m/p.gguf"), "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "-c", "65536", "--cache-ram", "0", "--api-key-file", "/k/key", "--port", "8080",
        "-fa", "on", "--host", "127.0.0.1",
    ]


def test_the_key_file_rides_argv_only_when_the_decision_passes_it():
    cmd = server.serve_command(_full_llama(), keyed=True, loopback=True, key_file_ok=False)
    assert "--api-key-file" not in cmd and "/k/key" not in cmd


def test_an_unset_field_adds_nothing():
    assert server.serve_command(_llama()) == [
        "/x/llama-server", "-m", "/x/m.gguf", "--port", "8080", "--host", "127.0.0.1"]


def test_the_fit_keywords_fill_an_unset_record_and_never_override_a_set_one():
    filled = server.serve_command(_llama(), context=32768, cache_ram=512)
    assert filled[filled.index("-c") + 1] == "32768" and filled[filled.index("--cache-ram") + 1] == "512"
    kept = server.serve_command(_llama(context=4096, cache_ram=0), context=32768, cache_ram=512)
    assert kept[kept.index("-c") + 1] == "4096" and kept[kept.index("--cache-ram") + 1] == "0"
    assert kept.count("-c") == 1 and kept.count("--cache-ram") == 1


def test_a_zero_cache_ram_is_passed_not_dropped():
    assert server.serve_command(_llama(cache_ram=0)).count("--cache-ram") == 1


def test_the_fit_keywords_do_nothing_to_a_vllm_record():
    assert server.serve_command(_binary(), context=32768, cache_ram=512) == server.serve_command(_binary())
    assert server.serve_command(_docker(), context=32768, cache_ram=512) == server.serve_command(_docker())


def test_serve_command_does_no_io_and_is_deterministic(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("serve_command touched the filesystem")

    spec = _full_llama()
    with record_opens() as opened:
        monkeypatch.setattr(os, "stat", boom)
        first = server.serve_command(spec, keyed=True, loopback=True, key_file_ok=True)
        second = server.serve_command(spec, keyed=True, loopback=True, key_file_ok=True)
    assert first == second
    assert not any(p.endswith(("x.gguf", "p.gguf", "key")) for p in opened)


def test_a_0_17_vllm_docker_record_keeps_its_argv_byte_for_byte():
    srv = _docker(extra_args=["--max-model-len", "65536"], bind_all=True)
    assert server.serve_command(srv, keyed=True) == [
        "docker", "run", "--rm", "--name", server.DOCKER_CONTAINER_NAME, "--gpus", "all", "--ipc=host",
        "-v", f"{_hf_cache()}:/root/.cache/huggingface", "-e", "VLLM_API_KEY",
        "-p", "8081:8000", "img:tag", "--model", "org/repo", "--max-model-len", "65536",
    ]


def test_a_0_17_vllm_binary_record_keeps_every_token_and_only_the_host_moves():
    old_literal = ["/opt/vllm", "serve", "org/repo", "--port", "8081", "--host", "127.0.0.1",
                   "--max-model-len", "65536"]
    assert sorted(server.serve_command(_binary())) == sorted(old_literal)


def test_one_table_names_each_runtimes_key_variable():
    assert dict(server.KEY_ENV) == {"vllm": "VLLM_API_KEY", "llamacpp": "LLAMA_API_KEY"}
    with pytest.raises(TypeError):
        server.KEY_ENV["x"] = "y"  # type: ignore[index]


def test_the_docker_launch_names_the_variable_from_that_table():
    cmd = server.serve_command(_docker(), keyed=True)
    assert cmd[cmd.index("-e") + 1] == server.KEY_ENV["vllm"]


def test_every_runtime_has_a_label():
    assert dict(server.RUNTIME_LABELS) == {
        "vllm": "vLLM", "llamacpp": "llama.cpp", "ollama": "Ollama", "lmstudio": "LM Studio"}
    with pytest.raises(TypeError):
        server.RUNTIME_LABELS["x"] = "y"  # type: ignore[index]


# ------------------------------------------------------------------------------- the launch


def test_a_required_key_travels_in_the_environment_not_in_argv_or_the_log(tmp_path, popen):  # noqa: F811
    assert server.start_server(tmp_path, ["true"], api_key=KEY) == 4242
    (call,) = popen
    assert call["cmd"] == ["true"]
    assert call["env"]["VLLM_API_KEY"] == KEY
    assert call["env"]["PATH"] == os.environ["PATH"]  # the rest of the environment is inherited
    log = server.log_path(tmp_path).read_text(encoding="utf-8")
    assert "=== launch: true ===" in log and KEY not in log


def test_without_a_key_the_launch_inherits_the_environment(tmp_path, popen, monkeypatch):  # noqa: F811
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
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


async def test_a_keyed_vllm_activation_threads_the_key_through_launch_and_readiness(tmp_path, monkeypatch):
    calls = record_chain(monkeypatch)
    await lifecycle.ManagedVllmStrategy().activate(_docker(), tmp_path, "http://127.0.0.1:8081/v1",
                                                   api_key=KEY)
    assert calls == {"serve_command": {"keyed": True, "loopback": True, "key_file_ok": False,
                                       "context": None, "cache_ram": None},
                     "start_server": {"runtime": "vllm", "record_env": None, "api_key": KEY},
                     "wait_ready": {"api_key": KEY}}


async def test_a_keyless_vllm_activation_calls_everything_as_before(tmp_path, monkeypatch):
    calls = record_chain(monkeypatch)
    await lifecycle.ManagedVllmStrategy().activate(_docker(), tmp_path, "http://127.0.0.1:8081/v1")
    assert calls == {"serve_command": {"keyed": False, "loopback": True, "key_file_ok": False,
                                       "context": None, "cache_ram": None},
                     "start_server": {"runtime": "vllm", "record_env": None}, "wait_ready": {}}


async def test_a_spawned_server_gets_the_key_in_its_launch_environment_and_readiness_probe(tmp_path, monkeypatch):
    calls = record_chain(monkeypatch)
    spec = ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server", model="/x/m.gguf", port=8080)
    await lifecycle.SpawnedProcessStrategy().activate(spec, tmp_path, "http://127.0.0.1:8080/v1",
                                                      api_key=KEY)
    assert calls == {"serve_command": {"keyed": True, "loopback": True, "key_file_ok": False,
                                       "context": None, "cache_ram": None},
                     "start_server": {"runtime": "llamacpp", "record_env": None, "api_key": KEY},
                     "wait_ready": {"api_key": KEY}}


@pytest.mark.parametrize("cls", [lifecycle.LifecycleStrategy, lifecycle.ManagedVllmStrategy,
                                 lifecycle.SpawnedProcessStrategy, lifecycle.DaemonStrategy,
                                 lifecycle.LmsStrategy])
def test_every_strategy_accepts_a_key(cls):
    param = inspect.signature(cls.activate).parameters["api_key"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is None


@pytest.mark.parametrize("cls", [lifecycle.LifecycleStrategy, lifecycle.ManagedVllmStrategy,
                                 lifecycle.SpawnedProcessStrategy, lifecycle.DaemonStrategy,
                                 lifecycle.LmsStrategy])
def test_every_strategy_accepts_a_peer(cls):
    param = inspect.signature(cls.activate).parameters["peer"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is None


@pytest.mark.parametrize("cls", [lifecycle.LifecycleStrategy, lifecycle.ManagedVllmStrategy,
                                 lifecycle.SpawnedProcessStrategy, lifecycle.DaemonStrategy,
                                 lifecycle.LmsStrategy])
def test_every_strategy_accepts_autosense(cls):
    param = inspect.signature(cls.activate).parameters["autosense"]
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


# ------------------------------------------------------------------------------- the recorders


def test_record_opens_sees_an_open_and_not_a_stat(tmp_path):
    statted, opened_file, after = tmp_path / "statted", tmp_path / "opened", tmp_path / "after"
    for p in (statted, opened_file, after):
        p.write_text("x", encoding="utf-8")
    with record_opens() as seen:
        os.stat(statted)
        open(opened_file, encoding="utf-8").close()
    assert str(opened_file) in seen
    assert str(statted) not in seen
    recorded = list(seen)
    open(after, encoding="utf-8").close()  # the hook stays installed, but it records nothing now
    assert seen == recorded and str(after) not in seen
