"""LAUNCH-04: the key reaches a launched server in the child environment only, and the launched
runtime's own argv-equivalent namespace is scrubbed from what it inherits (D-14, D-15, D-16, D-29).

The scrub is a blocklist, never an allowlist: CUDA_VISIBLE_DEVICES, LD_LIBRARY_PATH, PATH and HF_HOME
must survive. Dropped names are logged, names only.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from localharness.config.models import ManagedServerConfig
from localharness.provider import lifecycle, server
from tests.unit.launch_support import PopenRecorder, popen  # noqa: F401

KEY = "sk-K-secret"

LLAMA_DECOYS = {
    "LLAMA_ARG_N_GPU_LAYERS": "0",
    "LLAMA_ARG_NO_MMAP": "1",
    "LLAMA_API_KEY": "shell-key",
    "MTMD_BACKEND_DEVICE": "/dev/null",
}
SURVIVORS = {"HF_TOKEN": "hf-decoy", "CUDA_VISIBLE_DEVICES": "1", "HF_HOME": "/h",
             "LD_LIBRARY_PATH": "/lib/x"}


def _set(monkeypatch, env: dict[str, str]) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)


def _llama(**over) -> ManagedServerConfig:
    return ManagedServerConfig(**{"runtime": "llamacpp", "binary": "/x/llama-server",
                                  "model": "/x/m.gguf", "port": 8080, **over})


# ------------------------------------------------------------------------------- the tracer


async def test_a_keyed_llama_cpp_activation_holds_its_key_in_the_child_environment_only(
        tmp_path, monkeypatch, popen):  # noqa: F811
    _set(monkeypatch, {**LLAMA_DECOYS, **SURVIVORS})

    async def wait_ready(base_url, **kw):
        return ["m"]

    monkeypatch.setattr(server, "wait_ready", wait_ready)
    spec = _llama(require_api_key=True, env={"LLAMA_ARG_N_GPU_LAYERS": "99"})
    await lifecycle.SpawnedProcessStrategy().activate(
        spec, tmp_path, "http://127.0.0.1:8080/v1", api_key=KEY)

    (call,) = popen
    env = call["env"]
    assert env["LLAMA_API_KEY"] == KEY
    assert env["LLAMA_ARG_N_GPU_LAYERS"] == "99"  # the record's value beats the scrubbed shell's
    assert "LLAMA_ARG_NO_MMAP" not in env and "MTMD_BACKEND_DEVICE" not in env
    for name, value in SURVIVORS.items():
        assert env[name] == value
    assert "PATH" in env
    assert all(KEY not in str(part) for part in call["cmd"])
    assert KEY not in server.log_path(tmp_path).read_text(encoding="utf-8")


# ------------------------------------------------------------------------------- child_environment


def test_a_vllm_child_drops_the_shell_key_and_keeps_the_rest_of_its_settings():
    env, dropped = server.child_environment(
        "vllm", {"VLLM_API_KEY": "shell", "VLLM_SERVER_DEV_MODE": "1", "PATH": "/b", "MAX_JOBS": "3"})
    assert dropped == ["VLLM_API_KEY"]
    assert env == {"VLLM_SERVER_DEV_MODE": "1", "PATH": "/b", "MAX_JOBS": "3"}


@pytest.mark.parametrize("runtime", ["ollama", "lmstudio"])
def test_a_runtime_with_no_namespace_to_scrub_inherits_everything(runtime):
    base = {"PATH": "/b", "LLAMA_ARG_N_GPU_LAYERS": "0", "VLLM_API_KEY": "shell"}
    assert server.child_environment(runtime, base) == (None, [])


def test_a_llama_cpp_launch_with_nothing_to_drop_inherits_the_environment():
    assert server.child_environment("llamacpp", {"PATH": "/b"}) == (None, [])


def test_a_key_for_a_runtime_with_no_key_variable_is_an_error_naming_it():
    with pytest.raises(ValueError, match="ollama"):
        server.child_environment("ollama", {"PATH": "/b"}, api_key="k")


def test_the_environment_is_built_copy_scrub_record_env_then_the_key_last():
    base = {"PATH": "/b", "LLAMA_ARG_N_GPU_LAYERS": "0", "LLAMA_API_KEY": "shell"}
    env, dropped = server.child_environment(
        "llamacpp", base, record_env={"LLAMA_API_KEY": "from-record", "LLAMA_ARG_CTX_SIZE": "8"},
        api_key="held")
    assert dropped == ["LLAMA_API_KEY", "LLAMA_ARG_N_GPU_LAYERS"]
    assert env == {"PATH": "/b", "LLAMA_ARG_CTX_SIZE": "8", "LLAMA_API_KEY": "held"}


def test_a_name_in_the_shell_and_the_record_ends_with_the_records_value():
    env, _ = server.child_environment(
        "llamacpp", {"LLAMA_ARG_N_GPU_LAYERS": "0"}, record_env={"LLAMA_ARG_N_GPU_LAYERS": "99"})
    assert env == {"LLAMA_ARG_N_GPU_LAYERS": "99"}


def test_child_environment_does_not_touch_its_base():
    base = {"LLAMA_API_KEY": "shell", "PATH": "/b"}
    server.child_environment("llamacpp", base, api_key="k")
    assert base == {"LLAMA_API_KEY": "shell", "PATH": "/b"}


def test_a_blocklist_keeps_the_whole_gpu_and_loader_environment():
    base = {**SURVIVORS, "PATH": "/b", "LLAMA_ARG_THREADS": "2"}
    env, _ = server.child_environment("llamacpp", base)
    assert env == {**SURVIVORS, "PATH": "/b"}
    env, _ = server.child_environment("vllm", {**base, "VLLM_API_KEY": "s", "MAX_JOBS": "3"})
    assert env == {**base, "MAX_JOBS": "3"}


# ------------------------------------------------------------------------------- start_server


def test_without_a_key_or_anything_to_drop_the_child_inherits_exactly(tmp_path, monkeypatch, popen):  # noqa: F811
    for name in list(server.os.environ):
        if name.startswith(("LLAMA_ARG_", "VLLM_API_KEY", "LLAMA_API_KEY", "MTMD_BACKEND_DEVICE")):
            monkeypatch.delenv(name)
    server.start_server(tmp_path, ["true"], runtime="llamacpp")
    assert popen[0]["env"] is None


def test_the_dropped_names_are_logged_without_their_values(tmp_path, monkeypatch, popen):  # noqa: F811
    _set(monkeypatch, {**LLAMA_DECOYS, **SURVIVORS})
    with capture_logs() as logs:
        server.start_server(tmp_path, ["true"], runtime="llamacpp", api_key=KEY)
    scrubbed = [e for e in logs if e["event"] == "launch environment scrubbed"]
    assert len(scrubbed) == 1
    dropped = scrubbed[0]["dropped"]
    assert dropped == sorted(dropped)
    assert set(LLAMA_DECOYS) <= set(dropped)
    text = str(logs)
    for value in ("/dev/null", "shell-key", KEY):
        assert value not in text
    assert KEY not in server.log_path(tmp_path).read_text(encoding="utf-8")


def test_a_vllm_launch_logs_the_inherited_vllm_names_and_the_dropped_key(tmp_path, monkeypatch, popen):  # noqa: F811
    _set(monkeypatch, {"VLLM_API_KEY": "shell", "VLLM_SERVER_DEV_MODE": "1", **SURVIVORS})
    with capture_logs() as logs:
        server.start_server(tmp_path, ["true"], runtime="vllm")
    (call,) = popen
    assert "VLLM_API_KEY" not in call["env"]
    assert call["env"]["VLLM_SERVER_DEV_MODE"] == "1"
    for name, value in SURVIVORS.items():
        assert call["env"][name] == value
    by_event = {e["event"]: e for e in logs}
    assert by_event["launch environment scrubbed"]["dropped"] == ["VLLM_API_KEY"]
    assert "VLLM_SERVER_DEV_MODE" in by_event["launch environment inherits VLLM_ settings"]["names"]
    assert "shell" not in str(logs)


def test_the_launch_line_never_holds_the_key(tmp_path, popen):  # noqa: F811
    server.start_server(tmp_path, ["serve", "--port", "8080"], runtime="llamacpp", api_key=KEY)
    log = server.log_path(tmp_path).read_text(encoding="utf-8")
    assert "=== launch: serve --port 8080 ===" in log and KEY not in log


# ------------------------------------------------------------------------------- every call names its runtime


def test_every_start_server_call_in_src_passes_runtime():
    """D-14, D-30: a call that omits `runtime=` would silently launch with the vLLM scrub, so every
    call in src names its runtime (lifecycle.py's three; init writes records and launches nothing)."""
    src = Path(__file__).resolve().parents[2] / "src" / "localharness"
    calls: list[tuple[str, int, bool]] = []
    for path in sorted(src.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            named = (isinstance(func, ast.Attribute) and func.attr == "start_server") or (
                isinstance(func, ast.Name) and func.id == "start_server")
            if named:
                calls.append((f"{path.relative_to(src)}:{node.lineno}", node.lineno,
                              any(kw.arg == "runtime" for kw in node.keywords)))
    missing = [where for where, _, has in calls if not has]
    assert not missing, f"start_server called without runtime= at {missing}"
    assert len(calls) >= 3, f"expected lifecycle.py's three, found {calls}"
