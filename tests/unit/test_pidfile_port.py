"""START-12 / START-13 / COMPAT-01: the pidfile names its port and `procfs.read_launch` is the one reader
of "which launch is live and whose is it" (spec 20 §8.3, §12 "a pidfile is a claim, not proof").

Everything runs on a tmp config dir and a fake `/proc` tree (`fake_proc`); `subprocess.Popen` is the
suite's recorder, so no process starts and no signal is sent. Test fakes take `**kw`.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from localharness.config.models import ManagedServerConfig
from localharness.provider import lifecycle, procfs, server
from tests.unit.launch_support import PopenRecorder, popen  # noqa: F401
from tests.unit.procfs_support import FakeProc, fake_proc, stat_bytes  # noqa: F401

LLAMA_EXE = "/opt/llama/llama-server"
LLAMA_ARGV = (LLAMA_EXE, "-m", "/x/m.gguf", "--port", "8080")
UNIT_CGROUP = f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/localharness-llm.service"
SPEC = ManagedServerConfig(runtime="llamacpp", binary=LLAMA_EXE, model="/x/m.gguf", port=8080)


def identity() -> procfs.LaunchIdentity:
    ident = lifecycle.SpawnedProcessStrategy().launch_identity(SPEC)
    assert ident is not None
    return ident


def write_pidfile(config_dir: Path, text: str) -> Path:
    directory = server.server_dir(config_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / procfs.PIDFILE_NAME
    path.write_text(text, encoding="utf-8")
    return path


# ------------------------------------------------------------------------------- the tracer slice


def test_a_launch_writes_pid_start_and_port_privately(tmp_path, monkeypatch, popen):  # noqa: F811
    monkeypatch.setattr(server, "_proc_start_time", lambda pid: 777)
    assert server.start_server(tmp_path, ["true"], runtime="llamacpp", port=8080) == 4242
    path = server.pid_path(tmp_path)
    assert path.read_text(encoding="utf-8") == "4242 777 8080"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [p.name for p in server.server_dir(tmp_path).iterdir() if p.name != "serve.log"] == [procfs.PIDFILE_NAME]


def test_read_launch_verifies_the_process_against_the_strategys_identity(tmp_path, fake_proc):  # noqa: F811
    fake_proc.process(4242, argv=LLAMA_ARGV, exe=LLAMA_EXE, start_time=777)
    write_pidfile(tmp_path, "4242 777 8080")
    assert procfs.read_launch(tmp_path, expect=identity()) == procfs.Launch(
        4242, 777, 8080, "spawned", True, False)


def test_server_pid_is_built_on_the_launch_reader(tmp_path, monkeypatch, fake_proc):  # noqa: F811
    fake_proc.process(4242, argv=LLAMA_ARGV, exe=LLAMA_EXE, start_time=777)
    write_pidfile(tmp_path, "4242 777 8080")
    monkeypatch.setattr(server.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(server, "_read_proc_stat", lambda pid: stat_bytes(pid, 777))
    assert server.server_pid(tmp_path) == 4242
    assert server.pid_path(tmp_path).exists()


def test_the_launch_identity_of_a_llama_server_names_its_binary_and_port():
    assert identity() == procfs.LaunchIdentity(frozenset({"llama-server"}), 8080, (), "LLAMA_ARG_PORT")


def test_the_writer_writes_two_tokens_without_a_port_and_one_without_a_start(
        tmp_path, monkeypatch, popen):  # noqa: F811
    monkeypatch.setattr(server, "_proc_start_time", lambda pid: 777)
    server.start_server(tmp_path, ["true"])
    assert server.pid_path(tmp_path).read_text(encoding="utf-8") == "4242 777"
    monkeypatch.setattr(server, "_proc_start_time", lambda pid: None)
    server.start_server(tmp_path, ["true"], port=8080)
    assert server.pid_path(tmp_path).read_text(encoding="utf-8") == "4242"
