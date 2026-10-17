"""The environment `serve` execs the server with (spec 20 §9.3, §12; D-15; T-14-17, T-14-21).

The unit's body builds it from an allowlist: the pinned home, the values captured in the verified unit file
(`PATH`, `HF_HOME`, `LD_LIBRARY_PATH` — never the manager's), exactly the eleven `EXEC_INHERITED` names, the
record's own `env:`, the llama.cpp config home and the key. A decoy the service manager (or anything that
called its `set-environment`) put in its own environment must not reach the server.

`serve_world` (tests/unit/test_serve_cmd.py) is the world; the exec is its recorder, so no server starts.
"""
from __future__ import annotations

import os

import pytest

from localharness.cli import serve_cmd
from localharness.config.models import ManagedServerConfig
from tests.unit.test_serve_cmd import _Exec, serve_world

KEY = "sk-exec-env-7d3f9a"
ALLOWED_LLAMA = {"HOME", "PATH", "HF_HOME", "LD_LIBRARY_PATH", "XDG_CONFIG_HOME", *serve_cmd.EXEC_INHERITED}


def _decoys(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/manager/decoy/bin")
    monkeypatch.setenv("HF_HOME", "/manager/hf")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/manager/lib")
    monkeypatch.setenv("CUDA_INJECTION64_PATH", "/evil.so")
    monkeypatch.setenv("LD_PRELOAD", "/evil.so")
    monkeypatch.setenv("SOME_OTHER", "1")
    monkeypatch.setenv("LLAMA_ARG_PORT", "9999")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("LANG", "C.UTF-8")


def _exec_env(w, result) -> dict:
    assert isinstance(result.exception, _Exec), (result.exit_code, w.err, result.exception)
    return w.execs[0][2]


def test_the_allowlist_is_eleven_names_in_the_documented_order_with_no_wildcard():
    assert serve_cmd.EXEC_INHERITED == (
        "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "LC_MESSAGES", "XDG_RUNTIME_DIR",
        "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES")
    assert not [n for n in serve_cmd.EXEC_INHERITED if n.endswith(("*", "_"))]


def test_decoys_in_the_managers_environment_never_reach_the_server(tmp_path, monkeypatch):
    w = serve_world(tmp_path, monkeypatch,
                    captured={"PATH": f"{tmp_path}/bin:/usr/bin", "HF_HOME": "/h/hf"})
    _decoys(monkeypatch)

    env = _exec_env(w, w.run())

    assert env["PATH"] == f"{tmp_path}/bin:/usr/bin" and env["HF_HOME"] == "/h/hf"
    assert env["HOME"] == str(w.home)
    assert env["CUDA_VISIBLE_DEVICES"] == "0" and env["LANG"] == "C.UTF-8"
    assert env["XDG_CONFIG_HOME"] == str(w.g / "xdg" / "8080")
    for absent in ("CUDA_INJECTION64_PATH", "LD_PRELOAD", "SOME_OTHER", "LLAMA_ARG_PORT", "LD_LIBRARY_PATH",
                   "LOCALHARNESS_DIR", "LOCALHARNESS_HOME", "INVOCATION_ID"):
        assert absent not in env
    assert not {"/manager/decoy/bin", "/manager/hf", "/manager/lib", "/evil.so"} & set(env.values())


def test_the_base_holds_only_the_allowlisted_names(tmp_path, monkeypatch):
    w = serve_world(tmp_path, monkeypatch)
    _decoys(monkeypatch)

    env = _exec_env(w, w.run())

    assert set(env) <= ALLOWED_LLAMA


def test_path_and_ld_library_path_come_from_the_unit_file_not_the_environment(tmp_path, monkeypatch):
    w = serve_world(tmp_path, monkeypatch,
                    captured={"PATH": f"{tmp_path}/bin", "LD_LIBRARY_PATH": "/opt/cuda/lib"})
    _decoys(monkeypatch)

    env = _exec_env(w, w.run())

    assert env["PATH"] == f"{tmp_path}/bin" and env["LD_LIBRARY_PATH"] == "/opt/cuda/lib"
    assert "HF_HOME" not in env  # the unit captured none: the manager's is not substituted


def test_a_keyed_record_gets_the_key_in_the_environment_and_never_on_argv(tmp_path, monkeypatch):
    w = serve_world(tmp_path, monkeypatch, provider_key=KEY)
    _decoys(monkeypatch)

    result = w.run()

    env = _exec_env(w, result)
    path, argv, _ = w.execs[0]
    assert env["LLAMA_API_KEY"] == KEY
    assert not [a for a in argv if KEY in a] and "--api-key-file" not in argv
    assert KEY not in w.said() and "LLAMA_ARG_PORT" not in env


def test_the_records_own_env_is_applied_over_the_base(tmp_path, monkeypatch):
    w = serve_world(tmp_path, monkeypatch, server_extra={"env": {"LLAMA_ARG_THREADS": "4"}})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("LLAMA_ARG_PORT", "9999")

    env = _exec_env(w, w.run())

    assert env["LLAMA_ARG_THREADS"] == "4" and env["CUDA_VISIBLE_DEVICES"] == "0"
    assert "LLAMA_ARG_PORT" not in env


def test_an_unkeyed_vllm_binary_record_gets_a_copy_of_the_base_not_the_environment(tmp_path, monkeypatch):
    w = serve_world(tmp_path, monkeypatch, runtime="vllm")
    for name in serve_cmd.EXEC_INHERITED:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("USER", "alice")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    _decoys(monkeypatch)
    seen: list = []
    real = serve_cmd._exec_environment

    def spy(captured, home, inherited, spec, api_key, ini_home, hf_offline=False):
        out = real(captured, home, inherited, spec, api_key, ini_home, hf_offline=hf_offline)
        seen.append((out, inherited, spec.runtime, ini_home))
        return out

    monkeypatch.setattr(serve_cmd, "_exec_environment", spy)

    env = _exec_env(w, w.run())

    out, inherited, runtime, ini_home = seen[0]
    assert runtime == "vllm" and ini_home is None
    assert out is env or out == env
    assert env == {"HOME": str(w.home), "PATH": str(w.bindir), "USER": "alice", "CUDA_VISIBLE_DEVICES": "0",
                   "LANG": "C.UTF-8"}
    assert type(env) is dict and env is not os.environ and inherited is os.environ


def test_exec_environment_returns_a_plain_dict_even_when_child_environment_adds_nothing(monkeypatch):
    spec = ManagedServerConfig(runtime="vllm", launch="binary", binary="/x/vllm", model="m", port=8000)
    inherited = {"USER": "alice", "PATH": "/manager/bin", "SOME_OTHER": "1"}

    env = serve_cmd._exec_environment({"PATH": "/captured"}, "/home/u", inherited, spec, None, None)

    assert env == {"HOME": "/home/u", "PATH": "/captured", "USER": "alice"}
    assert env is not inherited


def test_exec_environment_puts_the_key_and_the_config_home_for_llama_cpp():
    spec = ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server", model="/m.gguf", port=8080)

    env = serve_cmd._exec_environment({"PATH": "/captured"}, "/home/u", {}, spec, KEY, "/g/xdg/8080")

    assert env == {"HOME": "/home/u", "PATH": "/captured", "XDG_CONFIG_HOME": "/g/xdg/8080",
                   "LLAMA_API_KEY": KEY}


@pytest.mark.parametrize("name", ["PATH", "HF_HOME", "LD_LIBRARY_PATH"])
def test_the_captured_names_are_not_taken_from_the_inherited_environment(name):
    spec = ManagedServerConfig(runtime="vllm", launch="binary", binary="/x/vllm", model="m", port=8000)

    env = serve_cmd._exec_environment({}, "/home/u", {name: "/manager/value"}, spec, None, None)

    assert name not in env
