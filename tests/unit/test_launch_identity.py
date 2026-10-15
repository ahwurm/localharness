"""START-12 (D-25): every lifecycle strategy states what its launch looks like in `/proc` as plain
`LaunchIdentity` data, and `lifecycle.record_identity` adds the record's own alias or model token so two
records on one binary and port can be told apart. Pure data plus a fake `/proc` tree: nothing starts."""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from localharness.config.models import ManagedServerConfig
from localharness.provider import lifecycle, procfs
from tests.unit.procfs_support import FakeProc, fake_proc  # noqa: F401

LLAMA = "/x/llama-server"


def llama(**kw) -> ManagedServerConfig:
    kw.setdefault("binary", LLAMA)
    return ManagedServerConfig(runtime="llamacpp", model="/x/m.gguf", port=8080, **kw)


def identity(spec: ManagedServerConfig) -> procfs.LaunchIdentity | None:
    return lifecycle.strategy_for(spec).launch_identity(spec)


# ------------------------------------------------------------------------------- the four strategies


def test_llamacpp_names_its_binary_port_and_port_variable():
    assert identity(llama()) == procfs.LaunchIdentity(
        frozenset({"llama-server"}), 8080, (), "LLAMA_ARG_PORT")


def test_a_symlinked_llamacpp_binary_matches_under_both_names(tmp_path):
    real = tmp_path / "real" / "llama-server-cuda"
    real.parent.mkdir()
    real.write_text("", encoding="utf-8")
    link = tmp_path / "bin" / "ls-link"
    link.parent.mkdir()
    link.symlink_to(real)
    got = identity(llama(binary=str(link)))
    assert got is not None and got.exe_names == frozenset({"ls-link", "llama-server-cuda"})


def test_llamacpp_without_a_binary_expects_llama_server():
    spec = llama()
    object.__setattr__(spec, "binary", None)
    got = identity(spec)
    assert got is not None and got.exe_names == frozenset({"llama-server"})


def test_vllm_binary_is_the_entry_point_or_its_interpreter_serving():
    spec = ManagedServerConfig(runtime="vllm", launch="binary", binary="/x/vllm", model="org/m", port=8000)
    assert identity(spec) == procfs.LaunchIdentity(frozenset({"vllm", "python*"}), 8000, ("serve",), None)


def test_vllm_docker_is_the_docker_client_under_the_fixed_container_name():
    spec = ManagedServerConfig(runtime="vllm", launch="docker", docker_image="img", model="org/m", port=8000)
    assert identity(spec) == procfs.LaunchIdentity(
        frozenset({"docker"}), None, ("--name", "localharness-vllm"), None)


def test_ollama_is_the_daemon_with_the_port_in_ollama_host():
    spec = ManagedServerConfig(runtime="ollama", model="qwen", port=11434)
    assert identity(spec) == procfs.LaunchIdentity(frozenset({"ollama"}), 11434, (), "OLLAMA_HOST")


def test_lm_studio_has_no_identity():
    spec = ManagedServerConfig(runtime="lmstudio", model="m", port=1234)
    assert identity(spec) is None
    assert lifecycle.record_identity(spec) is None


def test_every_strategy_class_states_a_launch_identity():
    for strategy in (lifecycle.ManagedVllmStrategy, lifecycle.SpawnedProcessStrategy,
                     lifecycle.DaemonStrategy, lifecycle.LmsStrategy):
        assert callable(getattr(strategy, "launch_identity", None))
        assert isinstance(strategy(), lifecycle.LifecycleStrategy)


# ------------------------------------------------------------------------------- record_identity


def test_record_identity_adds_the_alias_when_the_record_has_one():
    got = lifecycle.record_identity(llama(served_name="qwen"))
    assert got is not None and got.argv_contains == ("-a", "qwen")
    assert got.exe_names == frozenset({"llama-server"}) and got.port == 8080


def test_record_identity_adds_the_model_when_there_is_no_alias():
    got = lifecycle.record_identity(llama())
    assert got is not None and got.argv_contains == ("-m", "/x/m.gguf")


def test_record_identity_expands_a_tilde_model_as_the_launch_does():
    spec = ManagedServerConfig(runtime="llamacpp", binary=LLAMA, model="~/m.gguf", port=8080)
    got = lifecycle.record_identity(spec)
    assert got is not None and got.argv_contains == ("-m", os.path.expanduser("~/m.gguf"))


def test_record_identity_adds_the_model_to_a_vllm_binary_and_nothing_to_the_rest():
    vllm = ManagedServerConfig(runtime="vllm", launch="binary", binary="/x/vllm", model="org/m", port=8000)
    got = lifecycle.record_identity(vllm)
    assert got is not None and got.argv_contains == ("serve", "org/m")
    ollama = ManagedServerConfig(runtime="ollama", model="qwen", port=11434)
    assert lifecycle.record_identity(ollama) == identity(ollama)


def docker(model: str, port: int) -> ManagedServerConfig:
    return ManagedServerConfig(runtime="vllm", launch="docker", docker_image="img", model=model, port=port)


def _docker_run(fake_proc: FakeProc, spec: ManagedServerConfig, *, loopback: bool = True) -> procfs.ProcRecord:  # noqa: F811
    """The `docker run` client of `spec` as `serve_command` spells it (the fixed container name)."""
    published = f"127.0.0.1:{spec.port}:8000" if loopback else f"{spec.port}:8000"
    fake_proc.process(4242, argv=("docker", "run", "--rm", "--name", "localharness-vllm", "--gpus", "all",
                                  "-p", published, "img", "--model", spec.model),
                      exe="/usr/bin/docker")
    record = procfs.read_process(4242)
    assert isinstance(record, procfs.ProcRecord)
    return record


def test_two_docker_records_are_told_apart_by_their_model_and_published_port(fake_proc):  # noqa: F811
    """They all run under one container name; the `docker run` line is what differs."""
    a, b = docker("org/a", 8000), docker("org/b", 8001)
    record = _docker_run(fake_proc, a)
    id_a, id_b = lifecycle.record_identity(a), lifecycle.record_identity(b)
    assert procfs.identity_mismatch(record, id_a) is None
    assert procfs.identity_mismatch(record, id_b) == "has no org/b on its command line"
    same_model_other_port = lifecycle.record_identity(docker("org/a", 8001))
    assert "has none of 127.0.0.1:8001:8000 / 8001:8000" in procfs.identity_mismatch(
        record, same_model_other_port)


def test_a_docker_record_matches_its_launch_bound_beyond_loopback_too(fake_proc):  # noqa: F811
    spec = docker("org/a", 8000)
    record = _docker_run(fake_proc, spec, loopback=False)
    assert procfs.identity_mismatch(record, lifecycle.record_identity(spec)) is None


def _launched(fake_proc: FakeProc, spec: ManagedServerConfig) -> procfs.ProcRecord:  # noqa: F811
    argv = (LLAMA, "-m", "/x/m.gguf", "-a", spec.served_name or "qwen", "--port", str(spec.port))
    fake_proc.process(4242, argv=argv, exe=LLAMA)
    record = procfs.read_process(4242)
    assert isinstance(record, procfs.ProcRecord)
    return record


def test_a_launch_for_one_alias_is_not_a_launch_for_another(fake_proc):  # noqa: F811
    qwen = llama(served_name="qwen")
    flash = llama(served_name="flash-next")
    record = _launched(fake_proc, qwen)
    qwen_id, flash_id = lifecycle.record_identity(qwen), lifecycle.record_identity(flash)
    assert qwen_id is not None and flash_id is not None
    assert procfs.identity_mismatch(record, qwen_id) is None
    assert procfs.identity_mismatch(record, flash_id) == "has no flash-next on its command line"


@pytest.mark.parametrize("method", ["activate", "stop", "liveness", "launch_identity"])
def test_the_protocol_names_the_launch_identity_beside_the_other_strategy_methods(method):
    assert hasattr(lifecycle.LifecycleStrategy, method)


def test_every_start_server_call_in_lifecycle_passes_the_record_port():
    source = Path(lifecycle.__file__).read_text(encoding="utf-8")
    calls = [node for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == "start_server"]
    assert len(calls) == 3
    for call in calls:
        assert any(kw.arg == "port" for kw in call.keywords), ast.get_source_segment(source, call)
