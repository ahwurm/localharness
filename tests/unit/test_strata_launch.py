"""A Strata launch runs only what the machine record confirmed, is recognised by its proven process and stops parent-first."""
from __future__ import annotations

import asyncio
import json
import signal
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from localharness.agent import context as agent_context
from localharness.config import trust
from localharness.provider import lifecycle, procfs, server, strata
from tests.unit.launch_support import popen  # noqa: F401
from tests.unit.strata_fixture import CONFIG_NAME, strata_record
from tests.unit.test_stop_server_expect import PID, START, llama_spec, record, refusal_line, stat_line, write_pidfile
from tests.unit.test_strata_activate import BASE, SENSED, SERVED, make_world, sensing  # noqa: F401

PEER = "strata-peer"
UNCONFIRMED = strata.UNCONFIRMED.format(name=SERVED)


def copy_file(world):
    return strata.copy_path(world.config_dir, 8083)


@pytest.mark.parametrize("peer", [PEER, None])
async def test_a_launch_whose_server_py_changed_after_the_record_is_refused_before_anything_starts(
        make_world, popen, peer):  # noqa: F811
    world = make_world(peer=peer)
    (world.engine / "serve" / "server.py").write_text("print('changed')\n", encoding="utf-8")
    with pytest.raises(server.LaunchRefused) as refused:
        await world.activate(autosense=None)
    assert str(refused.value) == UNCONFIRMED
    assert popen == [] and not copy_file(world).exists()


@pytest.mark.parametrize("peer", [PEER, None])
async def test_a_launch_of_the_recorded_files_runs(make_world, popen, peer):  # noqa: F811
    world = make_world(peer=peer)
    await world.activate(autosense=None)
    assert len(popen) == 1 and copy_file(world).exists()


async def test_config_bytes_changed_between_the_record_and_the_launch_are_refused(make_world, popen):  # noqa: F811
    world = make_world(peer=PEER)
    path = world.engine / CONFIG_NAME
    config = json.loads(path.read_text("utf-8"))
    config["args"] += ["--prefill", "512"]
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(server.LaunchRefused) as refused:
        await world.activate(autosense=None)
    assert str(refused.value) == UNCONFIRMED
    assert popen == [] and not copy_file(world).exists()


async def test_a_file_swapped_after_start_checked_is_refused_at_the_launch(make_world, popen):  # noqa: F811
    world = make_world(peer=PEER)
    assert [e["name"] for e in trust.machine_snapshot(world.config_dir) if e["name"].endswith("strata_code")]
    (world.engine / "engine" / "strata").write_text("#!/bin/sh\n", encoding="utf-8")
    with pytest.raises(server.LaunchRefused):
        await world.activate(autosense=None)
    assert popen == []


async def test_a_launch_with_no_machine_record_is_refused(make_world, popen, monkeypatch):  # noqa: F811
    world = make_world()
    monkeypatch.setattr(trust, "recorded_machine", lambda config_dir: None)
    with pytest.raises(server.LaunchRefused) as refused:
        await world.activate(autosense=None)
    assert str(refused.value) == UNCONFIRMED
    assert popen == []


@pytest.mark.parametrize("path", ["serve", CONFIG_NAME, "engine/strata"])
async def test_an_install_made_group_writable_since_the_set_up_is_refused(make_world, popen, path):  # noqa: F811
    world = make_world(peer=PEER)
    (world.engine / path).chmod(0o777)
    with pytest.raises(server.LaunchRefused) as refused:
        await world.activate(autosense=None)
    assert str(refused.value) == strata.NOT_AN_INSTALL.format(name=SERVED, engine_dir=str(world.engine))
    assert popen == [] and not copy_file(world).exists()


async def test_the_planned_launch_is_checked_the_same_way(make_world, popen, sensing):  # noqa: F811
    world = make_world(peer=PEER)
    (world.engine / "serve" / "server.py").write_text("print('changed')\n", encoding="utf-8")
    with pytest.raises(server.LaunchRefused) as refused:
        await world.activate(autosense=SENSED)
    assert str(refused.value) == UNCONFIRMED and popen == []


# ------------------------------------------------------------------------------- identity and stop

CFG = Path("/cfg")
SPEC = strata_record(Path("/e/Strata"), Path("/m/a.gguf"), Path("/m/p.gguf"))
PYTHON = "/usr/bin/python3.12"
ENGINE = "/e/Strata/engine/strata"


def strata_proc(cfg: Path, port: int = 8083, **over):
    return record(exe=PYTHON, argv=tuple(strata.argv(SPEC, config_copy=str(strata.copy_path(cfg, port)))), **over)


def mismatch(rec) -> str | None:
    return procfs.identity_mismatch(rec, strata.identity(SPEC, CFG))


def test_the_venv_python_launch_with_this_ports_copy_is_the_identity():
    assert mismatch(strata_proc(CFG)) is None


@pytest.mark.parametrize("rec", [
    record(exe=ENGINE, argv=(ENGINE, "--serve"), sid=PID + 1),
    record(exe=ENGINE, argv=(ENGINE, "--serve")),
    record(exe="/usr/bin/bash", argv=("bash",)),
])
def test_the_engine_child_and_other_processes_are_not_the_identity(rec):
    assert mismatch(rec) is not None


def test_another_ports_copy_is_not_this_identity():
    other = SPEC.model_copy(update={"port": 8084})
    rec = record(exe=PYTHON, argv=tuple(strata.argv(other, config_copy=str(strata.copy_path(CFG, 8084)))))
    assert mismatch(rec) is not None


def test_a_strata_identity_adds_no_llamacpp_token_and_a_llamacpp_one_still_does():
    assert not {"-a", "-m"} & set(lifecycle.record_identity(SPEC).argv_contains)
    named = llama_spec().model_copy(update={"served_name": "alias"})
    assert lifecycle.record_identity(named).argv_contains[-2:] == ("-a", "alias")


def test_own_launch_finds_a_verified_strata_launch(tmp_path, monkeypatch):
    write_pidfile(tmp_path, f"{PID} {START} 8083")
    monkeypatch.setattr(procfs, "_stat_bytes", lambda pid: stat_line(pid, START))
    monkeypatch.setattr(procfs, "read_process", lambda pid, **kw: strata_proc(tmp_path))
    found = lifecycle.own_launch(tmp_path, [SPEC])
    assert found is not None and found.pid == PID


class Calls:
    """Everything the stop under test could do, in order: `/proc` reads and signals."""

    def __init__(self) -> None:
        self.log: list[tuple] = []
        self.clock = 0.0
        self.killed = False
        self.read = None
        self.alive = lambda: not self.killed

    @property
    def signals(self) -> list[tuple]:
        return [c for c in self.log if c[0] != "proof"]

    @property
    def proven_before_each_signal(self) -> bool:
        return all(self.log[i - 1] == ("proof",) for i, c in enumerate(self.log) if c[0] != "proof")


@pytest.fixture
def proc(monkeypatch):
    calls = Calls()

    def read(pid, **kw):
        calls.log.append(("proof",))
        return calls.read

    def kill(pid, sig):
        if sig:
            calls.log.append(("kill", pid, sig))

    def killpg(pid, sig):
        calls.log.append(("killpg", pid, sig))
        calls.killed = sig == signal.SIGKILL

    def sleep(seconds):
        calls.clock += seconds

    monkeypatch.setattr(procfs, "read_process", read)
    monkeypatch.setattr(server, "_read_proc_stat", lambda pid: stat_line(pid, START))
    monkeypatch.setattr(server.os, "kill", kill)
    monkeypatch.setattr(server.os, "killpg", killpg)
    monkeypatch.setattr(server, "_alive", lambda pid, start=None: calls.alive())
    monkeypatch.setattr(server, "_is_zombie", lambda pid: False)
    monkeypatch.setattr(server, "time", SimpleNamespace(monotonic=lambda: calls.clock, sleep=sleep))
    return calls


@pytest.fixture
def live(tmp_path, proc):
    """A pidfile naming a verified Strata launch of port 8083 under `tmp_path`."""
    proc.read = strata_proc(tmp_path)
    return write_pidfile(tmp_path, f"{PID} {START} 8083")


async def stop_strata(config_dir: Path) -> None:
    await strata.StrataStrategy().stop(SPEC, config_dir)


async def test_a_parent_that_exits_on_its_sigterm_gets_that_one_signal_and_nothing_else(tmp_path, proc, live):
    polls = iter([True, True])
    proc.alive = lambda: next(polls, False)
    await stop_strata(tmp_path)
    assert proc.signals == [("kill", PID, signal.SIGTERM)] and proc.proven_before_each_signal
    assert not live.exists()


async def test_a_parent_alive_through_the_grace_is_reproven_and_the_group_killed_after_it(tmp_path, proc, live):
    await stop_strata(tmp_path)
    assert proc.signals == [("kill", PID, signal.SIGTERM), ("killpg", PID, signal.SIGKILL)]
    assert proc.proven_before_each_signal and proc.clock >= 30.0


async def test_a_parent_that_became_another_process_in_the_grace_is_not_killed_and_the_stop_raises(
        tmp_path, proc, live, monkeypatch):
    after_term = server.os.kill

    def change_then_kill(pid, sig):
        after_term(pid, sig)
        if sig == signal.SIGTERM:
            proc.read = record(exe=ENGINE, argv=(ENGINE,))

    monkeypatch.setattr(server.os, "kill", change_then_kill)
    proc.alive = lambda: True
    with pytest.raises(RuntimeError, match="which is not a harness launch"):
        await stop_strata(tmp_path)
    assert proc.signals == [("kill", PID, signal.SIGTERM)] and live.exists()


async def test_a_parent_gone_one_poll_before_the_grace_ends_gets_no_sigkill(tmp_path, proc, live):
    proc.alive = lambda: proc.clock < 29.0
    await stop_strata(tmp_path)
    assert proc.signals == [("kill", PID, signal.SIGTERM)]


async def test_no_pidfile_and_a_pid_that_is_not_the_recorded_process_signal_nothing(tmp_path, proc, monkeypatch):
    stop = lambda: asyncio.to_thread(  # noqa: E731
        server.stop_server, tmp_path, expect=strata.identity(SPEC, tmp_path), parent_first=True)
    assert await stop() is False
    write_pidfile(tmp_path, f"{PID} {START} 8083")
    monkeypatch.setattr(server, "_read_proc_stat", lambda pid: stat_line(pid, START + 1))
    assert await stop() is False
    assert proc.log == []


async def test_a_pidfile_planted_to_name_an_unrelated_process_is_never_signalled(tmp_path, proc, live):
    proc.read = record(exe="/usr/bin/bash", argv=("bash",))
    with pytest.raises(RuntimeError) as refused:
        await stop_strata(tmp_path)
    assert str(refused.value) == refusal_line(tmp_path, "bash")
    assert proc.signals == [] and live.exists()


async def test_a_llamacpp_stop_still_signals_the_group_first_and_reproves_before_the_sigkill(tmp_path, proc):
    proc.read = record()
    write_pidfile(tmp_path)
    await lifecycle.SpawnedProcessStrategy().stop(llama_spec(), tmp_path)
    assert proc.signals == [("killpg", PID, signal.SIGTERM), ("killpg", PID, signal.SIGKILL)]
    assert proc.proven_before_each_signal


# ------------------------------------------------------------------------------- readiness, window, logs, environment

KEY = "sk-held-0123456789abcdef"
BLOCKED = ("STRATA_SKIP_SHA256", "STRATA_PREBUILT_URL", "STRATA_SOURCE", "STRATA_EXECV", "HF_ENDPOINT")


def answers(*steps):
    """A transport answering each request from the next step: an exception is raised, a (path, body) pair served."""
    queue = list(steps)

    def handler(request: httpx.Request) -> httpx.Response:
        step = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(step, Exception):
            raise step
        return httpx.Response(200, json=step[request.url.path])

    return httpx.MockTransport(handler)


async def test_readiness_waits_through_a_refused_connection_and_loaded_false(monkeypatch):
    loading = {"/health": {"loaded": False, "service": "strata"}}
    ready = {"/health": {"loaded": True, "service": "strata"}, "/v1/models": {"data": [{"id": SERVED}]}}
    monkeypatch.setattr(server, "_TRANSPORT", answers(httpx.ConnectError("refused"), httpx.ConnectError("refused"), loading, ready))
    polls: list[float] = []
    served = await server.wait_ready(BASE, poll_seconds=0, runtime="strata", ready=strata.health_ready, on_poll=polls.append)
    assert served == [SERVED] and len(polls) == 3


async def test_a_health_answer_from_another_service_keeps_the_wait_going(monkeypatch):
    other = {"/health": {"loaded": True, "service": "other"}}
    monkeypatch.setattr(server, "_TRANSPORT", answers(other))
    with pytest.raises(TimeoutError):
        await server.wait_ready(BASE, poll_seconds=0.01, timeout_seconds=0.05, runtime="strata", ready=strata.health_ready)


@pytest.mark.parametrize("status", [401, 403])
async def test_a_keyed_models_refusal_after_loaded_is_a_death_not_a_wait(monkeypatch, status):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"loaded": True, "service": "strata"})
        return httpx.Response(status)

    monkeypatch.setattr(server, "_TRANSPORT", httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="Strata refused the key"):
        await server.wait_ready(BASE, poll_seconds=0, timeout_seconds=30, runtime="strata", ready=strata.health_ready)


def private_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)


async def test_a_parent_that_died_during_startup_shows_both_logs_with_the_key_and_bearer_masked(tmp_path):
    private_log(server.log_path(tmp_path), f"server line\nAuthorization: Bearer abc.def.ghi\nkey {KEY}\n")
    private_log(strata.log_path(tmp_path, 8083), f"engine line\ntoken {KEY} Bearer xyz.uvw.rst\n")
    with pytest.raises(RuntimeError) as died:
        await server.wait_ready(
            BASE, config_dir=tmp_path, runtime="strata", api_key=KEY, engine_log=strata.log_path(tmp_path, 8083))
    text = str(died.value)
    assert text.startswith("Strata exited during startup. Log tail:") and "server line" in text and "engine line" in text
    assert KEY not in text and "abc.def.ghi" not in text and "xyz.uvw.rst" not in text


def test_the_engine_tail_is_read_by_name_inside_its_own_private_directory(tmp_path):
    assert server.scrubbed_tail(tmp_path, engine_log=strata.log_path(tmp_path, 8083)).endswith("(no log)")
    private_log(strata.log_path(tmp_path, 8083), "engine says\n")
    link = tmp_path / "strata" / "8084.log"
    link.symlink_to(tmp_path / "elsewhere")
    assert "engine says" in server.scrubbed_tail(tmp_path, engine_log=strata.log_path(tmp_path, 8083))
    assert server.scrubbed_tail(tmp_path, engine_log=link).endswith("(no log)")


def test_the_window_falls_back_to_health_max_context_when_the_model_list_has_no_n_ctx(monkeypatch):
    bodies = {"/v1/models": {"data": [{"id": SERVED}]}, "/health": {"max_context": 65536}}
    monkeypatch.setattr(
        httpx, "get", lambda url, **kw: httpx.Response(200, json=bodies["/" + url.split("/", 3)[3]]))
    assert agent_context._probe_strata("http://127.0.0.1:8083/v1", SERVED, {}).served == 65536


async def test_the_child_environment_drops_the_escape_hatches_and_carries_the_held_key_on_neither_argv_nor_the_log(
        make_world, popen, monkeypatch, tmp_path):  # noqa: F811
    for name in BLOCKED:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("STRATA_API_KEY", "old")
    world = make_world()
    live = await world.activate(autosense=None, api_key=KEY)
    env, cmd = popen[0]["env"], popen[0]["cmd"]
    assert not set(BLOCKED) & set(env) and env["STRATA_API_KEY"] == KEY
    assert set(live.dropped_env) == {*BLOCKED, "STRATA_API_KEY"}
    assert KEY not in " ".join(cmd) and "--api-key" not in cmd and "--open" not in cmd
    assert KEY not in server.log_path(world.config_dir).read_text("utf-8")
