"""START-12 / START-11: a pidfile is a claim, not proof (D-27, D-18).

`stop_server(..., *, expect)` signals nothing unless the pidfile's live process matches `expect`;
`expect=None` never authorises a signal; a refusal raises the one named line and leaves the pidfile.
`wait_ready(expect=)` fails at once when the live launch is not the record's. `/proc` is faked through a
patched `procfs.read_process`; nothing real is read, signalled or started.
"""
from __future__ import annotations

import os
import signal
from pathlib import Path

import httpx
import pytest

from localharness.config.models import ManagedServerConfig
from localharness.provider import lifecycle, procfs, server
from localharness.provider.procfs import LaunchIdentity, ProcRecord, ProcRefusal

PID = 4242
START = 777
LLAMA = LaunchIdentity(frozenset({"llama-server"}), 8080, (), "LLAMA_ARG_PORT")


def stat_line(pid: int, start: int) -> bytes:
    return (f"{pid} (llama-server) S ".encode() + b" ".join([b"0"] * 18) + b" " + str(start).encode()
            + b" " + b" ".join([b"0"] * 30))


def record(*, exe: str = "/x/llama-server", argv: tuple[str, ...] | None = None, sid: int | None = PID,
           uid: int | None = None) -> ProcRecord:
    return ProcRecord(PID, os.getuid() if uid is None else uid, argv or (exe, "--port", "8080"), exe, None,
                      START, None, None, (), sid=sid)


class Signals:
    """Everything the stop under test could do to a process, recorded."""

    def __init__(self) -> None:
        self.killpg: list[tuple[int, int]] = []
        self.dead = False


@pytest.fixture
def proc(monkeypatch):
    """A live pid 4242 whose `/proc` answer is `state['read']`; `os.killpg` records and kills it."""
    signals = Signals()
    state: dict = {"read": record()}
    monkeypatch.setattr(procfs, "read_process", lambda pid, **kw: state["read"])
    monkeypatch.setattr(server, "_read_proc_stat", lambda pid: stat_line(pid, START))
    monkeypatch.setattr(server.os, "kill", lambda pid, sig: None)

    def killpg(pid, sig):
        signals.killpg.append((pid, sig))
        signals.dead = True

    monkeypatch.setattr(server.os, "killpg", killpg)
    monkeypatch.setattr(server, "_alive", lambda pid, start=None: not signals.dead)
    monkeypatch.setattr(server, "_is_zombie", lambda pid: False)
    monkeypatch.setattr(server.time, "sleep", lambda s: None)
    signals.state = state  # type: ignore[attr-defined]
    return signals


def write_pidfile(tmp_path: Path, text: str = f"{PID} {START} 8080") -> Path:
    server.server_dir(tmp_path).mkdir(parents=True, exist_ok=True)
    path = server.pid_path(tmp_path)
    path.write_text(text, encoding="utf-8")
    return path


def llama_spec() -> ManagedServerConfig:
    return ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server", model="/x/m.gguf", port=8080)


def refusal_line(tmp_path: Path, exe: str) -> str:
    return (f"pidfile names pid {PID} ({exe}), which is not a harness launch — not signalled; "
            f"stop it, or remove {server.pid_path(tmp_path)}")


# ------------------------------------------------------------------------------- the tracer


async def test_a_spawned_stop_names_a_planted_pid_and_signals_nothing(tmp_path, proc):
    proc.state["read"] = record(exe="/usr/bin/sleep", argv=("sleep", "60"))
    pidfile = write_pidfile(tmp_path)

    with pytest.raises(RuntimeError) as refused:
        await lifecycle.SpawnedProcessStrategy().stop(llama_spec(), tmp_path)

    assert str(refused.value) == refusal_line(tmp_path, "sleep")
    assert proc.killpg == []
    assert pidfile.exists()


async def test_a_spawned_stop_signals_the_process_that_matches_its_identity(tmp_path, proc):
    pidfile = write_pidfile(tmp_path)

    await lifecycle.SpawnedProcessStrategy().stop(llama_spec(), tmp_path)

    assert proc.killpg == [(PID, signal.SIGTERM)]
    assert not pidfile.exists()


def test_expect_none_never_authorises_a_signal(tmp_path, proc):
    pidfile = write_pidfile(tmp_path)

    with pytest.raises(RuntimeError) as refused:
        server.stop_server(tmp_path, expect=None)

    assert str(refused.value) == refusal_line(tmp_path, "llama-server")
    assert proc.killpg == []
    assert pidfile.exists()


@pytest.mark.parametrize("changed,why", [
    ({"argv": ("/x/llama-server", "--port", "9090")}, "port"),
    ({"sid": 1}, "session"),
    ({"uid": os.getuid() + 1}, "uid"),
    ({"exe": None}, "exe"),
])
def test_a_process_that_fails_any_part_of_the_identity_is_not_signalled(tmp_path, proc, changed, why):
    base = record()
    proc.state["read"] = ProcRecord(**{**base.__dict__, **changed})
    pidfile = write_pidfile(tmp_path)

    with pytest.raises(RuntimeError, match="which is not a harness launch"):
        server.stop_server(tmp_path, expect=LLAMA)

    assert proc.killpg == [] and pidfile.exists()


def test_a_process_proc_will_not_show_is_named_with_a_question_mark(tmp_path, proc):
    proc.state["read"] = ProcRefusal("is gone")
    write_pidfile(tmp_path)

    with pytest.raises(RuntimeError) as refused:
        server.stop_server(tmp_path, expect=LLAMA)

    assert str(refused.value) == refusal_line(tmp_path, "?")
    assert proc.killpg == []


def test_a_refused_docker_stop_does_not_stop_the_container_by_name(tmp_path, proc, monkeypatch):
    proc.state["read"] = record(exe="/usr/bin/sleep", argv=("sleep", "60"))
    write_pidfile(tmp_path)
    monkeypatch.setattr(server.subprocess, "run",
                        lambda *a, **kw: pytest.fail("docker must not run after a refused verdict"))
    docker = LaunchIdentity(frozenset({"docker"}), None, ("--name", server.DOCKER_CONTAINER_NAME), None)

    with pytest.raises(RuntimeError, match="not a harness launch"):
        server.stop_server(tmp_path, launch="docker", expect=docker)


def test_no_live_pid_is_nothing_to_refuse(tmp_path, proc):
    assert server.stop_server(tmp_path, expect=None) is False
    assert proc.killpg == []


# ------------------------------------------------------------------------------- wait_ready(expect=)


def models_transport(monkeypatch) -> list[str]:
    """Answer `/models` with one served id through the `_TRANSPORT` seam; the list records each poll."""
    polls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        polls.append(str(request.url))
        return httpx.Response(200, json={"data": [{"id": "qwen"}]})

    monkeypatch.setattr(server, "_TRANSPORT", httpx.MockTransport(handler))
    return polls


async def test_wait_ready_refuses_a_live_launch_that_is_not_the_records_before_any_poll(
        tmp_path, proc, monkeypatch):
    proc.state["read"] = record(exe="/usr/bin/sleep", argv=("sleep", "60"))
    write_pidfile(tmp_path)
    polls = models_transport(monkeypatch)

    with pytest.raises(RuntimeError) as refused:
        await server.wait_ready("http://127.0.0.1:8080/v1", config_dir=tmp_path, expect=LLAMA,
                                runtime="llamacpp")

    assert str(refused.value) == f"llama.cpp: the live launch (pid {PID}) is not this record's — not waited on"
    assert polls == []


async def test_wait_ready_polls_as_today_when_the_live_launch_is_the_records(tmp_path, proc, monkeypatch):
    write_pidfile(tmp_path)
    polls = models_transport(monkeypatch)

    served = await server.wait_ready("http://127.0.0.1:8080/v1", config_dir=tmp_path, expect=LLAMA,
                                     runtime="llamacpp")

    assert served == ["qwen"] and polls == ["http://127.0.0.1:8080/v1/models"]


async def test_wait_ready_without_expect_reads_no_process(tmp_path, proc, monkeypatch):
    write_pidfile(tmp_path)
    polls = models_transport(monkeypatch)
    monkeypatch.setattr(procfs, "read_process",
                        lambda pid, **kw: pytest.fail("expect=None must not read the process"))

    served = await server.wait_ready("http://127.0.0.1:8080/v1", config_dir=tmp_path, runtime="llamacpp")

    assert served == ["qwen"] and len(polls) == 1


async def test_wait_ready_with_no_launch_keeps_the_exited_during_startup_text(tmp_path, proc, monkeypatch):
    polls = models_transport(monkeypatch)

    with pytest.raises(RuntimeError, match="llama.cpp exited during startup"):
        await server.wait_ready("http://127.0.0.1:8080/v1", config_dir=tmp_path, expect=LLAMA,
                                runtime="llamacpp")

    assert polls == []
