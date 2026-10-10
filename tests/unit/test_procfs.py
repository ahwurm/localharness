"""The one `/proc` module (D-15, D-16, D-17): listeners, socket inode to pid, and one frozen
process record read through a single directory fd. Every test points `procfs.PROC_ROOT` at a tmp
tree (`fake_proc`); nothing here reads the real `/proc`."""
from __future__ import annotations

import os

import pytest

from localharness.provider import procfs, server
from tests.unit.procfs_support import FakeProc, fake_proc, real_uid  # noqa: F401

# ----------------------------------------------------------------------- the shared start-time parser


def test_start_time_from_stat_reads_field_22_after_the_last_paren():
    raw = b"42 (a) b) S 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 777 0"
    assert server.start_time_from_stat(raw) == 777


def test_start_time_from_stat_is_none_for_malformed_bytes():
    assert server.start_time_from_stat(b"garbage") is None
    assert server.start_time_from_stat(b"42 (x) S 1 2") is None
    assert server.start_time_from_stat(b"") is None
    assert server.start_time_from_stat(b"42 (x) " + b"S " + b"z " * 30) is None


def test_proc_start_time_still_answers_through_the_read_proc_stat_seam(monkeypatch):
    raw = b"42 (a) b) S 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 777 0"
    monkeypatch.setattr(server, "_read_proc_stat", lambda pid: raw)
    assert server._proc_start_time(42) == 777
    monkeypatch.setattr(server, "_read_proc_stat", lambda pid: None)
    assert server._proc_start_time(42) is None


# ------------------------------------------------------------------------------------- listeners


def test_listeners_returns_the_listen_rows_of_the_port(fake_proc):
    fake_proc.listen(8080, 68389, 1000)
    fake_proc.listen(8080, 70000, 1000, state="01")  # ESTABLISHED
    fake_proc.listen(8080, 0, 1000)  # inode 0 never matches
    fake_proc.listen(9090, 70001, 1000)  # another port
    assert procfs.listeners(8080) == [procfs.Listener(68389, 1000)]


def test_listeners_adds_the_tcp6_rows_after_the_tcp_rows(fake_proc):
    fake_proc.listen(8080, 68389, 1000)
    fake_proc.listen(8080, 68390, 1001, family="tcp6")
    assert procfs.listeners(8080) == [procfs.Listener(68389, 1000), procfs.Listener(68390, 1001)]


def test_listeners_of_an_unheld_port_is_empty(fake_proc):
    fake_proc.listen(8080, 68389, 1000)
    assert procfs.listeners(9) == []


def test_a_missing_tcp6_is_skipped_silently(fake_proc):
    fake_proc.listen(8080, 68389, 1000)
    (fake_proc.root / "net" / "tcp6").unlink()
    assert procfs.listeners(8080) == [procfs.Listener(68389, 1000)]


def test_a_missing_net_directory_is_no_listeners(fake_proc):
    for name in ("tcp", "tcp6"):
        (fake_proc.root / "net" / name).unlink()
    assert procfs.listeners(8080) == []


def test_listeners_ignores_garbage_rows_and_a_partial_last_line(fake_proc):
    fake_proc.listen(8080, 68389, 1000)
    good = (fake_proc.root / "net" / "tcp").read_text()
    fake_proc.raw_net("tcp", good + "   9: nonsense 0A\n" + "  10: 0100007F:1F90 00000000:0000 0A")
    assert procfs.listeners(8080) == [procfs.Listener(68389, 1000)]


def test_proc_root_is_read_at_call_time(fake_proc, tmp_path, monkeypatch):
    fake_proc.listen(8080, 68389, 1000)
    other = tmp_path / "other"
    (other / "net").mkdir(parents=True)
    monkeypatch.setattr(procfs, "PROC_ROOT", str(other))
    assert procfs.listeners(8080) == []


# ---------------------------------------------------------------------------------- pid_for_inode


def test_pid_for_inode_finds_the_pid_holding_the_socket(fake_proc):
    fake_proc.process(4242, fds={3: 68389})
    fake_proc.process(4243, fds={3: 11111})
    assert procfs.pid_for_inode(68389) == 4242


def test_pid_for_inode_is_none_when_nobody_holds_it(fake_proc):
    fake_proc.process(4243, fds={3: 11111})
    assert procfs.pid_for_inode(68389) is None


def test_pid_for_inode_skips_a_pid_of_another_uid(fake_proc, monkeypatch):
    fake_proc.process(4242, fds={3: 68389})
    monkeypatch.setattr(procfs.os, "getuid", lambda: os.stat(fake_proc.root).st_uid + 1)
    assert procfs.pid_for_inode(68389) is None


def test_pid_for_inode_skips_non_numeric_entries_and_unreadable_fd_dirs(fake_proc):
    (fake_proc.root / "self").mkdir()
    (fake_proc.root / "net" / "fd").mkdir()
    fake_proc.process(4241)  # no fd directory at all
    fake_proc.process(4242, fds={3: 68389})
    assert procfs.pid_for_inode(68389) == 4242


def test_pid_for_inode_stops_at_the_pid_cap(fake_proc, monkeypatch):
    # scandir order is arbitrary, so the cap is proved on the boundary of one pid
    fake_proc.process(4242, fds={3: 68389})
    monkeypatch.setattr(procfs, "PID_SCAN_CAP", 1)
    assert procfs.pid_for_inode(68389) == 4242
    monkeypatch.setattr(procfs, "PID_SCAN_CAP", 0)
    assert procfs.pid_for_inode(68389) is None


def test_pid_for_inode_stops_at_the_fd_cap(fake_proc, monkeypatch):
    fake_proc.process(4242, fds={3: 68389})
    monkeypatch.setattr(procfs, "FD_SCAN_CAP", 1)
    assert procfs.pid_for_inode(68389) == 4242
    monkeypatch.setattr(procfs, "FD_SCAN_CAP", 0)
    assert procfs.pid_for_inode(68389) is None


# ----------------------------------------------------------------------------------- read_process


def llama_process(fake_proc: FakeProc, pid: int = 4242, **kw):
    cgroup = f"/user.slice/user-{real_uid()}.slice/user@{real_uid()}.service/app.slice/llama-server.service"
    args = dict(
        argv=("llama-server", "-m", "/m/a.gguf"),
        environ={"LLAMA_ARG_CTX_SIZE": "65536"},
        cgroup=cgroup,
        maps=("/usr/lib/libc.so.6", "/m/a.gguf"),
        exe="/opt/llama/llama-server",
        cwd="/home/u",
        start_time=777,
    )
    args.update(kw)
    return fake_proc.process(pid, **args)


def test_read_process_returns_the_frozen_record(fake_proc):
    llama_process(fake_proc)
    rec = procfs.read_process(4242)
    assert rec == procfs.ProcRecord(
        pid=4242,
        uid=os.getuid(),
        argv=("llama-server", "-m", "/m/a.gguf"),
        exe="/opt/llama/llama-server",
        cwd="/home/u",
        start_time=777,
        unit="llama-server.service",
        environ={"LLAMA_ARG_CTX_SIZE": "65536"},
        mapped_ggufs=("/m/a.gguf",),
    )
    with pytest.raises(AttributeError):
        rec.pid = 1  # frozen
    with pytest.raises(TypeError):
        rec.environ["X"] = "y"  # the environment is read-only too


def test_a_port_to_a_listener_to_a_pid_to_a_record(fake_proc):
    fake_proc.listen(8080, 68389, os.getuid())
    llama_process(fake_proc, 4242, fds={3: 68389})
    (listener,) = procfs.listeners(8080)
    pid = procfs.pid_for_inode(listener.inode)
    assert pid == 4242
    assert procfs.read_process(pid).argv[0] == "llama-server"


def test_the_environment_splits_on_the_first_equals_and_drops_malformed_entries(fake_proc):
    llama_process(fake_proc, environ_raw=b"A=b=c\0NOEQ\0=x\0LLAMA_ARG_X=\0")
    rec = procfs.read_process(4242)
    assert dict(rec.environ) == {"A": "b=c", "LLAMA_ARG_X": ""}


def test_an_unreadable_exe_and_cwd_are_none(fake_proc):
    llama_process(fake_proc, exe=None, cwd=None)
    rec = procfs.read_process(4242)
    assert rec.exe is None and rec.cwd is None
