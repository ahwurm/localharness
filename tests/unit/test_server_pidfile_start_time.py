"""LAUNCH-08: the pidfile records the process start time, and one `_same_process` gate stands before
every liveness check and every signal, so a reused pid is never signalled (D-17, D-26).

`/proc/<pid>/stat` is read through the `server._read_proc_stat` seam; the tests patch it.
"""
from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

import pytest

from localharness.config.models import ManagedServerConfig
from localharness.provider import lifecycle, procfs, server
from localharness.provider.procfs import LaunchIdentity, ProcRecord
from tests.unit.launch_support import PopenRecorder, popen  # noqa: F401

# What a stop expects here: any python process leading its own session (R-13-15). The real children
# below add the `-c` token their command line carries.
IDENT = LaunchIdentity(frozenset({"python*"}), None, (), None)
REAL_IDENT = LaunchIdentity(frozenset({"python*"}), None, ("-c",), None)


def proc_matches(monkeypatch) -> None:
    """`/proc` says every pid is a python session leader of this user: the identity verdict passes."""
    monkeypatch.setattr(
        procfs, "read_process",
        lambda pid, **kw: ProcRecord(pid, os.getuid(), ("python3",), "/usr/bin/python3", None, 777,
                                     None, None, (), sid=pid))


def stat_line(pid: int, start: int, *, comm: str = "llama-server", state: str = "S") -> bytes:
    """A /proc/<pid>/stat line: state is field 3 (index 0 after the last ")"), start is field 22 (index 19)."""
    return (f"{pid} ({comm}) {state} ".encode() + b" ".join([b"0"] * 18) + b" " + str(start).encode()
            + b" " + b" ".join([b"0"] * 30))


def proc_says(monkeypatch, start: int | None, *, state: str = "S") -> None:
    monkeypatch.setattr(
        server, "_read_proc_stat",
        lambda pid: None if start is None else stat_line(pid, start, state=state))


def pidfile(tmp_path: Path, text: str) -> None:
    server.server_dir(tmp_path).mkdir(parents=True, exist_ok=True)
    server.pid_path(tmp_path).write_text(text, encoding="utf-8")


# ------------------------------------------------------------------------------- parsing


def test_a_comm_with_spaces_and_parentheses_does_not_move_the_start_time(monkeypatch):
    monkeypatch.setattr(server, "_read_proc_stat", lambda pid: stat_line(4242, 777, comm="a b) (c"))
    assert server._proc_start_time(4242) == 777


def test_the_start_time_is_none_when_the_entry_is_missing_or_short(monkeypatch):
    monkeypatch.setattr(server, "_read_proc_stat", lambda pid: None)
    assert server._proc_start_time(4242) is None
    monkeypatch.setattr(server, "_read_proc_stat", lambda pid: b"4242 (x) S 1 2")
    assert server._proc_start_time(4242) is None
    monkeypatch.setattr(server, "_read_proc_stat", lambda pid: b"garbage")
    assert server._proc_start_time(4242) is None


@pytest.mark.parametrize("text,expected", [("4242", (4242, None, None)), ("4242 777", (4242, 777, None)),
                                           ("  4242   777\n", (4242, 777, None)),
                                           ("4242 777 8080", (4242, 777, 8080)),
                                           ("4242 777 0", (4242, 777, None))])
def test_a_pidfile_is_a_bare_pid_a_pid_and_start_or_a_pid_start_and_port(text, expected):
    assert server._parse_pidfile(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "4242 x", "x", "1 2 3", "0", "-1", "1", "0 777", "-5 777", "1 777", "4242 777 8080 9"])
def test_a_malformed_pidfile_does_not_parse(text):
    with pytest.raises(ValueError):
        server._parse_pidfile(text)


# ------------------------------------------------------------------------------- _same_process


def test_a_start_time_equal_to_proc_is_the_same_process_and_one_off_is_not(monkeypatch):
    proc_says(monkeypatch, 777)
    assert server._same_process(4242, 777) is True
    proc_says(monkeypatch, 778)
    assert server._same_process(4242, 777) is False


def test_a_bare_pid_matches_by_pid_only_and_reads_no_proc(monkeypatch):
    def boom(pid):
        raise AssertionError("a bare pid must not read /proc")
    monkeypatch.setattr(server, "_read_proc_stat", boom)
    assert server._same_process(4242, None) is True


def test_a_missing_proc_entry_is_not_the_same_process_and_never_raises(monkeypatch):
    proc_says(monkeypatch, None)
    assert server._same_process(4242, 777) is False


def test_start_kw_splats_only_a_recorded_start():
    assert server._start_kw(None) == {}
    assert server._start_kw(777) == {"start": 777}


# ------------------------------------------------------------------------------- the pidfile write


def test_start_server_records_the_start_time_beside_the_pid(tmp_path, monkeypatch, popen):  # noqa: F811
    proc_says(monkeypatch, 777)
    assert server.start_server(tmp_path, ["true"]) == 4242
    assert server.pid_path(tmp_path).read_text(encoding="utf-8") == "4242 777"


def test_start_server_writes_a_bare_pid_when_proc_cannot_be_read(tmp_path, monkeypatch, popen):  # noqa: F811
    proc_says(monkeypatch, None)
    server.start_server(tmp_path, ["true"])
    assert server.pid_path(tmp_path).read_text(encoding="utf-8") == "4242"


# ------------------------------------------------------------------------------- server_pid


def test_server_pid_trusts_a_matching_start_and_drops_a_reused_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(server.os, "kill", lambda pid, sig: None)
    pidfile(tmp_path, "4242 777")
    proc_says(monkeypatch, 777)
    assert server.server_pid(tmp_path) == 4242
    proc_says(monkeypatch, 778)
    assert server.server_pid(tmp_path) is None
    assert not server.pid_path(tmp_path).exists()


def test_a_bare_pid_pidfile_from_0_17_stays_valid(tmp_path, monkeypatch):
    monkeypatch.setattr(server.os, "kill", lambda pid, sig: None)
    pidfile(tmp_path, "4242")
    proc_says(monkeypatch, 12345)
    assert server.server_pid(tmp_path) == 4242
    assert server.pid_path(tmp_path).exists()


@pytest.mark.parametrize("text", ["", "4242 x", "1 2 3", "0", "-1", "1", "0 777", "x"])
def test_a_malformed_pidfile_is_removed_and_reads_as_no_server(tmp_path, monkeypatch, text):
    monkeypatch.setattr(server.os, "kill", lambda pid, sig: None)
    pidfile(tmp_path, text)
    assert server.server_pid(tmp_path) is None
    assert not server.pid_path(tmp_path).exists()


# ------------------------------------------------------------------------------- the signals


def _signals(monkeypatch) -> list[int]:
    signals: list[int] = []
    monkeypatch.setattr(server.os, "killpg", lambda pid, sig: signals.append(sig))
    monkeypatch.setattr(server.time, "sleep", lambda s: None)
    return signals


def test_stop_server_signals_nothing_when_the_pid_was_reused(tmp_path, monkeypatch):
    signals = _signals(monkeypatch)
    monkeypatch.setattr(server.os, "kill", lambda pid, sig: None)
    pidfile(tmp_path, "4242 777")
    proc_says(monkeypatch, 778)
    assert server.stop_server(tmp_path, expect=IDENT) is False
    assert signals == []


def test_the_sigterm_gate_holds_even_when_server_pid_is_faked(tmp_path, monkeypatch):
    signals = _signals(monkeypatch)
    monkeypatch.setattr(server.os, "kill", lambda pid, sig: None)
    pidfile(tmp_path, "4242 777")
    monkeypatch.setattr(server, "server_pid", lambda _cd: 4242)
    proc_says(monkeypatch, 778)
    assert server.stop_server(tmp_path, expect=IDENT) is False
    assert signals == []


def test_a_pid_reused_during_the_sigterm_grace_gets_no_sigkill(tmp_path, monkeypatch):
    signals: list[int] = []

    def killpg(pid, sig):
        signals.append(sig)
        state["start"] = 778  # the old process exits and an unrelated one takes the pid

    state = {"start": 777}
    monkeypatch.setattr(server.os, "killpg", killpg)
    monkeypatch.setattr(server.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(server.time, "sleep", lambda s: None)
    monkeypatch.setattr(server, "_read_proc_stat", lambda pid: stat_line(pid, state["start"]))
    pidfile(tmp_path, "4242 777")
    proc_matches(monkeypatch)
    assert server.stop_server(tmp_path, timeout_seconds=0.5, expect=IDENT) is True
    assert signals == [signal.SIGTERM]
    assert not server.pid_path(tmp_path).exists()


def test_the_sigkill_gate_holds_when_liveness_is_faked_true(tmp_path, monkeypatch):
    signals: list[int] = []
    state = {"start": 777}

    def killpg(pid, sig):
        signals.append(sig)
        state["start"] = 778

    monkeypatch.setattr(server.os, "killpg", killpg)
    monkeypatch.setattr(server.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(server.time, "sleep", lambda s: None)
    monkeypatch.setattr(server, "_read_proc_stat", lambda pid: stat_line(pid, state["start"]))
    monkeypatch.setattr(server, "_alive", lambda pid, start=None: True)
    monkeypatch.setattr(server, "_wait_dead", lambda pid, attempts, poll_seconds=0.5, *, start=None: True)
    pidfile(tmp_path, "4242 777")
    proc_matches(monkeypatch)
    assert server.stop_server(tmp_path, timeout_seconds=0.2, expect=IDENT) is True
    assert signals == [signal.SIGTERM]


def test_a_matching_process_still_gets_term_then_kill(tmp_path, monkeypatch):
    signals = _signals(monkeypatch)
    monkeypatch.setattr(server.os, "kill", lambda pid, sig: None)
    proc_says(monkeypatch, 777)
    pidfile(tmp_path, "4242 777")
    def wait_dead(pid, attempts, poll_seconds=0.5, *, start=None):
        assert start == 777  # the confirm poll is gated by the recorded start too
        return True

    monkeypatch.setattr(server, "_wait_dead", wait_dead)
    proc_matches(monkeypatch)
    assert server.stop_server(tmp_path, timeout_seconds=0.1, expect=IDENT) is True
    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_alive_is_false_for_a_reused_pid_and_true_for_the_recorded_one(monkeypatch):
    monkeypatch.setattr(server.os, "kill", lambda pid, sig: None)
    proc_says(monkeypatch, 777)
    assert server._alive(4242, 777) is True
    assert server._alive(4242, 778) is False
    assert server._alive(4242) is True


async def test_free_accelerator_signals_no_process_whose_start_time_differs(tmp_path, monkeypatch):
    signals = _signals(monkeypatch)
    monkeypatch.setattr(server.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(lifecycle, "GPU_FREE_SETTLE_SECONDS", 0.0)
    pidfile(tmp_path, "4242 777")
    proc_says(monkeypatch, 778)
    spawned = ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server", model="/x/m.gguf", port=8080)
    other = ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server", model="/x/n.gguf", port=8081)
    stopped = await lifecycle.free_accelerator((spawned, "http://127.0.0.1:8080/v1"), other, tmp_path)
    assert stopped is not None
    assert signals == []


# ------------------------------------------------------------------------------- a real process


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc/<pid>/stat")
def test_a_real_child_records_its_real_start_time_and_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(procfs, "PROC_ROOT", "/proc")  # the suite points it at an empty dir; this child is the test's own
    pid = server.start_server(tmp_path, [sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        tokens = server.pid_path(tmp_path).read_text(encoding="utf-8").split()
        assert len(tokens) == 2 and int(tokens[0]) == pid
        with open(f"/proc/{pid}/stat", "rb") as fh:
            assert tokens[1] == fh.read().rsplit(b")", 1)[1].split()[19].decode()
        assert server.server_pid(tmp_path) == pid
    finally:
        assert server.stop_server(tmp_path, expect=REAL_IDENT) is True
    assert not server._alive(pid)


# ------------------------------------------------------------------------------- a launch that fails


def test_start_server_closes_its_log_handle_and_raises_runtimeerror_when_popen_fails(
        tmp_path, monkeypatch):
    # A binary that is set but missing passes the launch decision and fails only at Popen. The
    # failure must be the RuntimeError every activate caller prints and restores from, and the
    # serve.log handle opened just before must not leak.
    opened: list = []
    real_open = open

    def recording_open(*args, **kwargs):
        handle = real_open(*args, **kwargs)
        opened.append(handle)
        return handle

    def popen_fails(cmd, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", cmd[0])

    monkeypatch.setattr(server, "open", recording_open, raising=False)
    monkeypatch.setattr(server.subprocess, "Popen", popen_fails)

    with pytest.raises(RuntimeError, match=r"could not launch /x/llama-server: No such file or directory"):
        server.start_server(tmp_path, ["/x/llama-server", "-m", "/x/m.gguf"], runtime="llamacpp")

    assert len(opened) == 1 and opened[0].closed
    assert not server.pid_path(tmp_path).exists()


@pytest.mark.parametrize("text", ["0", "-1", "1", "0 777", "x"])
def test_stop_server_signals_nothing_for_a_pidfile_that_is_not_a_process(tmp_path, monkeypatch, text):
    # pid 0 would make killpg signal the harness's own process group; none of these may reach a signal
    monkeypatch.setattr(server.os, "kill", lambda pid, sig: None)
    signalled: list[tuple] = []
    monkeypatch.setattr(server.os, "killpg", lambda pid, sig: signalled.append((pid, sig)))
    pidfile(tmp_path, text)
    assert server.stop_server(tmp_path, expect=IDENT) is False
    assert signalled == []
    assert not server.pid_path(tmp_path).exists()
