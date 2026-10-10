"""The one `/proc` module (D-15, D-16, D-17): listeners, socket inode to pid, and one frozen
process record read through a single directory fd. Every test points `procfs.PROC_ROOT` at a tmp
tree (`fake_proc`); nothing here reads the real `/proc`."""
from __future__ import annotations

import os

import pytest

from localharness.provider import procfs, server
from tests.unit.launch_support import record_opens
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


# --------------------------------------------------------------------- caps, refusals, pid reuse (D-16, D-25)


def test_a_cmdline_of_exactly_the_cap_reads_and_one_more_byte_is_refused(fake_proc):
    llama_process(fake_proc, cmdline=b"a" * procfs.CMDLINE_CAP)
    assert isinstance(procfs.read_process(4242), procfs.ProcRecord)
    llama_process(fake_proc, cmdline=b"a" * (procfs.CMDLINE_CAP + 1))
    assert procfs.read_process(4242) == procfs.ProcRefusal(
        "has its command line over 262144 bytes", "llama-server.service")


def test_an_environ_of_exactly_the_cap_reads_and_one_more_byte_is_refused(fake_proc):
    llama_process(fake_proc, environ_raw=b"A=" + b"a" * (procfs.ENVIRON_CAP - 2))
    assert isinstance(procfs.read_process(4242), procfs.ProcRecord)
    llama_process(fake_proc, environ_raw=b"A=" + b"a" * (procfs.ENVIRON_CAP - 1))
    assert procfs.read_process(4242) == procfs.ProcRefusal(
        "has its environment over 262144 bytes", "llama-server.service")


def test_an_unreadable_environ_is_refused_and_carries_the_unit(fake_proc, monkeypatch):
    llama_process(fake_proc)
    real = procfs._read_at

    def read_at(name, dir_fd, cap):
        if name == "environ":
            raise PermissionError(13, "Permission denied")
        return real(name, dir_fd, cap)

    monkeypatch.setattr(procfs, "_read_at", read_at)
    assert procfs.read_process(4242) == procfs.ProcRefusal(
        "has an unreadable environment", "llama-server.service")


def test_an_unreadable_cmdline_or_stat_reads_as_gone(fake_proc, monkeypatch):
    llama_process(fake_proc)
    real = procfs._read_at
    for victim in ("cmdline", "stat"):
        def read_at(name, dir_fd, cap, victim=victim):
            if name == victim:
                raise PermissionError(13, "Permission denied")
            return real(name, dir_fd, cap)

        monkeypatch.setattr(procfs, "_read_at", read_at)
        refusal = procfs.read_process(4242)
        assert isinstance(refusal, procfs.ProcRefusal) and refusal.reason == "is gone"


def test_a_malformed_stat_reads_as_gone(fake_proc):
    llama_process(fake_proc, stat=b"garbage")
    assert procfs.read_process(4242) == procfs.ProcRefusal("is gone")


def test_a_different_start_time_on_the_second_read_is_pid_reuse(fake_proc, monkeypatch):
    llama_process(fake_proc)
    real = procfs._read_at
    stats: list[int] = []

    def read_at(name, dir_fd, cap):
        if name == "stat":
            stats.append(1)
            if len(stats) == 2:
                return b"4242 (x) S " + b"1 " + b"0 " * 17 + b"778 0\n"
        return real(name, dir_fd, cap)

    monkeypatch.setattr(procfs, "_read_at", read_at)
    assert procfs.read_process(4242) == procfs.ProcRefusal(
        "pid reused during the read", "llama-server.service")


def test_a_start_time_that_vanishes_on_the_second_read_is_pid_reuse(fake_proc, monkeypatch):
    llama_process(fake_proc)
    real = procfs._read_at
    stats: list[int] = []

    def read_at(name, dir_fd, cap):
        if name == "stat":
            stats.append(1)
            if len(stats) == 2:
                raise FileNotFoundError(2, "gone")
        return real(name, dir_fd, cap)

    monkeypatch.setattr(procfs, "_read_at", read_at)
    refusal = procfs.read_process(4242)
    assert isinstance(refusal, procfs.ProcRefusal) and refusal.reason == "pid reused during the read"


def test_another_uid_is_refused_before_any_file_is_read(fake_proc, monkeypatch):
    llama_process(fake_proc)
    other_uid = os.getuid() + 1
    monkeypatch.setattr(procfs.os, "getuid", lambda: other_uid)
    reads: list[str] = []
    real = procfs._read_at
    monkeypatch.setattr(procfs, "_read_at", lambda name, fd, cap: (reads.append(name), real(name, fd, cap))[1])
    assert procfs.read_process(4242) == procfs.ProcRefusal("is owned by another uid")
    assert reads == []


def test_a_pid_with_no_directory_is_gone(fake_proc):
    assert procfs.read_process(999999) == procfs.ProcRefusal("is gone")


def test_a_pid_directory_that_is_a_file_is_gone(fake_proc):
    (fake_proc.root / "5").write_text("x")
    assert procfs.read_process(5) == procfs.ProcRefusal("is gone")


def closes_per_open(monkeypatch):
    """Count directory-fd opens and the closes of exactly those fds."""
    opened_dirs: list[int] = []
    closed: list[int] = []
    real_open, real_close = procfs.os.open, procfs.os.close

    def spy_open(path, flags, *a, **kw):
        fd = real_open(path, flags, *a, **kw)
        if "dir_fd" not in kw:
            opened_dirs.append(fd)
        return fd

    def spy_close(fd):
        closed.append(fd)
        return real_close(fd)

    monkeypatch.setattr(procfs.os, "open", spy_open)
    monkeypatch.setattr(procfs.os, "close", spy_close)
    return opened_dirs, closed


@pytest.mark.parametrize("case", ["record", "cmdline", "environ", "environ_perm", "reuse", "uid",
                                  "gone"])
def test_every_return_path_closes_the_directory_fd(fake_proc, monkeypatch, case):
    llama_process(fake_proc)
    real = procfs._read_at
    if case == "cmdline":
        llama_process(fake_proc, cmdline=b"a" * (procfs.CMDLINE_CAP + 1))
    elif case == "environ":
        llama_process(fake_proc, environ_raw=b"a" * (procfs.ENVIRON_CAP + 1))
    elif case == "environ_perm":
        def read_at(name, dir_fd, cap):
            if name == "environ":
                raise PermissionError(13, "no")
            return real(name, dir_fd, cap)

        monkeypatch.setattr(procfs, "_read_at", read_at)
    elif case == "reuse":
        stats: list[int] = []

        def read_at(name, dir_fd, cap):
            if name == "stat":
                stats.append(1)
                if len(stats) == 2:
                    return b"4242 (x) S " + b"1 " + b"0 " * 17 + b"1 0\n"
            return real(name, dir_fd, cap)

        monkeypatch.setattr(procfs, "_read_at", read_at)
    elif case == "uid":
        other_uid = os.getuid() + 1
        monkeypatch.setattr(procfs.os, "getuid", lambda: other_uid)
    pid = 999999 if case == "gone" else 4242
    opened_dirs, closed = closes_per_open(monkeypatch)
    procfs.read_process(pid)
    assert len(opened_dirs) == (0 if case == "gone" else 1)
    assert all(fd in closed for fd in opened_dirs)


def test_exe_and_cwd_are_readlinked_never_opened(fake_proc):
    target_exe = fake_proc.root.parent / "llama-server-binary"
    target_cwd = fake_proc.root.parent / "work"
    target_exe.write_text("#!/bin/sh\n")
    target_cwd.mkdir()
    llama_process(fake_proc, exe=str(target_exe), cwd=str(target_cwd))
    with record_opens() as opened:
        rec = procfs.read_process(4242)
    assert rec.exe == str(target_exe) and rec.cwd == str(target_cwd)
    assert not any(str(p) in (str(target_exe), str(target_cwd)) for p in opened)
    assert any(str(p).endswith("environ") or str(p) == "environ" for p in opened)  # the hook sees opens


# ----------------------------------------------------------------------------------------- the unit


def unit_of(fake_proc, cgroup):
    llama_process(fake_proc, cgroup=cgroup)
    rec = procfs.read_process(4242)
    assert isinstance(rec, procfs.ProcRecord)
    return rec.unit


def under_user(name, uid=None):
    uid = real_uid() if uid is None else uid
    return f"/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{name}"


def test_the_unit_is_the_last_component_under_the_user_manager(fake_proc):
    assert unit_of(fake_proc, under_user("llm.service")) == "llm.service"
    assert unit_of(fake_proc, under_user("llm@2.service")) == "llm@2.service"
    assert unit_of(fake_proc, under_user("a:b_c.d-e.service")) == "a:b_c.d-e.service"


@pytest.mark.parametrize("name", [
    "bad\x1bname.service",
    "bad;rm.service",
    "x" * 248 + ".service",  # 256 bytes
    "llm.scope",
    "llm",
    ".service",
])
def test_a_hostile_or_foreign_unit_name_is_none(fake_proc, name):
    assert unit_of(fake_proc, under_user(name)) is None


def test_a_unit_at_the_charset_edge_of_255_bytes_is_kept(fake_proc):
    name = "x" * 247 + ".service"  # 255 bytes
    assert len(name.encode()) == 255
    assert unit_of(fake_proc, under_user(name)) == name


def test_a_system_unit_path_is_none(fake_proc):
    assert unit_of(fake_proc, "/system.slice/llama.service") is None


def test_another_users_manager_path_is_none(fake_proc):
    assert unit_of(fake_proc, under_user("llm.service", uid=real_uid() + 1)) is None


def test_the_user_manager_itself_is_not_a_unit_below_it(fake_proc):
    assert unit_of(fake_proc, f"/user.slice/user-{real_uid()}.slice/user@{real_uid()}.service") is None


def test_no_cgroup_line_is_no_unit(fake_proc):
    assert unit_of(fake_proc, None) is None


# ----------------------------------------------------------------------------------------- maps


def mapped(fake_proc, **kw):
    llama_process(fake_proc, **kw)
    rec = procfs.read_process(4242)
    assert isinstance(rec, procfs.ProcRecord)
    return rec.mapped_ggufs


def test_gguf_and_blobs_paths_qualify_and_deleted_entries_do_not(fake_proc):
    assert mapped(fake_proc, maps=(
        "/lib/libc.so.6", "/m/x.GGUF", "/hf/blobs/abc123", "/m/gone.gguf (deleted)", "[heap]",
    )) == ("/m/x.GGUF", "/hf/blobs/abc123")


def test_the_same_path_mapped_twice_is_listed_once(fake_proc):
    assert mapped(fake_proc, maps=("/m/a.gguf", "/m/b.gguf", "/m/a.gguf")) == ("/m/a.gguf", "/m/b.gguf")


def test_a_pathname_with_spaces_is_kept_whole(fake_proc):
    assert mapped(fake_proc, maps=("/my models/big model.gguf",)) == ("/my models/big model.gguf",)


def test_a_maps_line_cap_hit_keeps_what_was_found_and_is_not_an_error(fake_proc, monkeypatch):
    monkeypatch.setattr(procfs, "MAPS_LINE_CAP", 5)
    filler = "".join(f"7f00{i:08x}-7f01{i:08x} r--p 00000000 08:01 1 /lib/x{i}.so\n" for i in range(5))
    late = "7f0000000000-7f0100000000 r--s 00000000 08:01 9 /m/late.gguf\n"
    assert mapped(fake_proc, maps_text=filler + late) == ()
    early = "7f0000000000-7f0100000000 r--s 00000000 08:01 9 /m/early.gguf\n"
    assert mapped(fake_proc, maps_text=early + filler + late) == ("/m/early.gguf",)


def test_a_maps_file_past_the_real_line_cap_gives_none_and_no_error(fake_proc):
    lines = "".join(f"7f00{i:08x}-7f01{i:08x} r--p 00000000 08:01 1 /lib/x.so\n"
                    for i in range(procfs.MAPS_LINE_CAP + 10))
    late = "7f0000000000-7f0100000000 r--s 00000000 08:01 9 /m/late.gguf\n"
    assert mapped(fake_proc, maps_text=lines + late) == ()


def test_more_candidates_than_the_cap_keep_the_first_ones(fake_proc):
    paths = tuple(f"/m/{i:03d}.gguf" for i in range(procfs.MAPPED_PATHS_CAP + 8))
    assert mapped(fake_proc, maps=paths) == paths[:procfs.MAPPED_PATHS_CAP]


def test_an_unreadable_maps_is_no_candidates(fake_proc):
    llama_process(fake_proc)
    (fake_proc.root / "4242" / "maps").unlink()
    rec = procfs.read_process(4242)
    assert isinstance(rec, procfs.ProcRecord) and rec.mapped_ggufs == ()


def test_procfs_does_no_gguf_parsing_http_or_subprocess():
    import pathlib

    text = pathlib.Path(procfs.__file__).read_text()
    for banned in ("import subprocess", "import httpx", "gguf_header", "import pwd"):
        assert banned not in text
