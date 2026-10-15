"""`server.refuse_live_launch` and `server.write_pidfile`: the one live-pidfile guard and the one
pidfile writer (the guard `start_server` and `serve` share before either writes a pidfile).

`/proc` is the fake tree of `procfs_support.fake_proc`, the start-time read is the `_read_proc_stat`
seam, `os.kill` answers "alive" or "gone" by patch, and Popen is the suite's recorder: nothing here
signals, launches or reads a real process.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from localharness.provider import procfs, server
from tests.unit.launch_support import popen  # noqa: F401 — the fixture
from tests.unit.procfs_support import fake_proc  # noqa: F401 — the fixture
from tests.unit.test_server_pidfile_start_time import pidfile, proc_says

UNIT_CGROUP = (f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/"
               f"{procfs.HARNESS_UNIT}")


def _alive(monkeypatch) -> None:
    monkeypatch.setattr(server.os, "kill", lambda pid, sig: None)


def _gone(monkeypatch) -> None:
    def kill(pid, sig):
        raise ProcessLookupError(pid)

    monkeypatch.setattr(server.os, "kill", kill)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_no_pidfile_is_nothing_in_the_way_and_nothing_is_written(tmp_path):
    assert server.refuse_live_launch(tmp_path, ["/x/llama-server"]) is None
    assert not server.pid_path(tmp_path).exists()
    assert not server.log_path(tmp_path).exists()


def test_a_live_process_in_the_pidfile_is_refused_and_the_pidfile_is_left_alone(
        tmp_path, monkeypatch, fake_proc):  # noqa: F811
    fake_proc.process(4242, argv=("llama-server",), exe="/x/llama-server")
    _alive(monkeypatch)
    pidfile(tmp_path, "4242 777")
    proc_says(monkeypatch, 777)

    with pytest.raises(RuntimeError) as refused:
        server.refuse_live_launch(tmp_path, ["/x/llama-server"])

    assert str(refused.value) == server.LIVE_LAUNCH_LINE.format(
        pid=4242, exe="llama-server", pidfile=server.pid_path(tmp_path))
    assert _text(server.pid_path(tmp_path)) == "4242 777"


def test_a_live_process_in_the_autostart_units_cgroup_gets_the_unit_line(
        tmp_path, monkeypatch, fake_proc):  # noqa: F811
    fake_proc.process(4242, argv=("llama-server",), exe="/x/llama-server", cgroup=UNIT_CGROUP)
    _alive(monkeypatch)
    pidfile(tmp_path, "4242 777")
    proc_says(monkeypatch, 777)

    with pytest.raises(RuntimeError) as refused:
        server.refuse_live_launch(tmp_path, ["/x/llama-server"])

    text = str(refused.value)
    assert text == server.LIVE_UNIT_LAUNCH_LINE.format(
        pid=4242, exe="llama-server", pidfile=server.pid_path(tmp_path))
    assert "localharness model <preset>" in text and "autostart's server" in text
    assert text != server.LIVE_LAUNCH_LINE.format(
        pid=4242, exe="llama-server", pidfile=server.pid_path(tmp_path))
    assert _text(server.pid_path(tmp_path)) == "4242 777"


def test_a_process_in_some_other_cgroup_unit_keeps_the_generic_line(
        tmp_path, monkeypatch, fake_proc):  # noqa: F811
    other = UNIT_CGROUP.replace(procfs.HARNESS_UNIT, "someone-else.service")
    fake_proc.process(4242, argv=("llama-server",), exe="/x/llama-server", cgroup=other)
    _alive(monkeypatch)
    pidfile(tmp_path, "4242 777")
    proc_says(monkeypatch, 777)

    with pytest.raises(RuntimeError) as refused:
        server.refuse_live_launch(tmp_path, ["/x/llama-server"])

    assert "autostart's server" not in str(refused.value)
    assert "stop that process first" in str(refused.value)


def test_a_bare_pid_naming_a_live_bash_is_dropped_and_the_note_is_returned(
        tmp_path, monkeypatch, fake_proc):  # noqa: F811
    fake_proc.process(4242, argv=("bash",), exe="/usr/bin/bash", comm="bash")
    _alive(monkeypatch)
    pidfile(tmp_path, "4242")
    proc_says(monkeypatch, 777)

    note = server.refuse_live_launch(tmp_path, ["/x/llama-server"])

    assert note == "=== dropped pidfile: pid 4242 runs bash, not a server launch ===\n"
    assert not server.pid_path(tmp_path).exists()


def test_a_bare_pid_naming_a_live_server_launch_is_still_refused(
        tmp_path, monkeypatch, fake_proc):  # noqa: F811
    fake_proc.process(4242, argv=("llama-server",), exe="/x/llama-server")
    _alive(monkeypatch)
    pidfile(tmp_path, "4242")
    proc_says(monkeypatch, 777)

    with pytest.raises(RuntimeError, match="names a live process"):
        server.refuse_live_launch(tmp_path, ["/x/llama-server"])

    assert _text(server.pid_path(tmp_path)) == "4242"


def test_a_stale_pidfile_for_a_dead_pid_is_dropped_and_nothing_is_in_the_way(tmp_path, monkeypatch):
    _gone(monkeypatch)
    pidfile(tmp_path, "4242 777")
    proc_says(monkeypatch, 777)

    assert server.refuse_live_launch(tmp_path, ["/x/llama-server"]) is None
    assert not server.pid_path(tmp_path).exists()


def test_start_server_refuses_through_the_guard_before_anything_starts(
        tmp_path, monkeypatch, popen, fake_proc):  # noqa: F811
    fake_proc.process(4242, argv=("llama-server",), exe="/x/llama-server", cgroup=UNIT_CGROUP)
    _alive(monkeypatch)
    pidfile(tmp_path, "4242 777")
    proc_says(monkeypatch, 777)

    with pytest.raises(RuntimeError, match="autostart's server"):
        server.start_server(tmp_path, ["true"])

    assert popen == []
    assert not server.log_path(tmp_path).exists()
    assert _text(server.pid_path(tmp_path)) == "4242 777"


def test_a_serve_shaped_sequence_guards_then_writes_its_own_pid_and_reads_it_back(tmp_path, monkeypatch):
    proc_says(monkeypatch, 31337)
    assert server.refuse_live_launch(tmp_path, ["/x/llama-server"]) is None

    server.write_pidfile(tmp_path, os.getpid(), port=8080)

    assert _text(server.pid_path(tmp_path)) == f"{os.getpid()} 31337 8080"
    launch = procfs.read_launch(tmp_path, stat_reader=server._read_proc_stat)
    assert launch is not None
    assert (launch.pid, launch.start, launch.port) == (os.getpid(), 31337, 8080)


@pytest.mark.parametrize("port", [0, -1, 65536, 100000])
def test_a_port_outside_the_range_is_not_written(tmp_path, monkeypatch, port):
    proc_says(monkeypatch, 31337)
    server.write_pidfile(tmp_path, 4242, port=port)
    assert _text(server.pid_path(tmp_path)) == "4242 31337"


@pytest.mark.parametrize("port,tail", [(1, " 1"), (65535, " 65535"), (None, "")])
def test_the_port_boundaries_and_an_absent_port(tmp_path, monkeypatch, port, tail):
    proc_says(monkeypatch, 31337)
    server.write_pidfile(tmp_path, 4242, port=port)
    assert _text(server.pid_path(tmp_path)) == f"4242 31337{tail}"


def test_an_unknown_start_time_writes_the_bare_pid_even_with_a_port(tmp_path, monkeypatch):
    proc_says(monkeypatch, None)
    server.write_pidfile(tmp_path, 4242, port=8080)
    assert _text(server.pid_path(tmp_path)) == "4242"


def test_write_pidfile_opens_and_closes_its_own_directory_and_leaves_a_given_one_open(
        tmp_path, monkeypatch):
    from localharness.core import private_files

    proc_says(monkeypatch, 31337)
    server.write_pidfile(tmp_path, 4242, port=8080)  # opened and closed here
    fd = private_files.open_private_dir(server.server_dir(tmp_path))
    try:
        server.write_pidfile(tmp_path, 4243, port=8081, dir_fd=fd)
        os.fstat(fd)  # still open: a descriptor the caller owns is never closed here
    finally:
        os.close(fd)
    assert _text(server.pid_path(tmp_path)) == "4243 31337 8081"


def test_write_pidfile_replaces_a_planted_link_and_propagates_oserror(tmp_path, monkeypatch):
    proc_says(monkeypatch, 31337)
    target = tmp_path / "elsewhere"
    target.write_text("keep", encoding="utf-8")
    server.server_dir(tmp_path).mkdir(parents=True)
    server.pid_path(tmp_path).symlink_to(target)

    server.write_pidfile(tmp_path, 4242, port=None)

    assert not server.pid_path(tmp_path).is_symlink()
    assert _text(server.pid_path(tmp_path)) == "4242 31337"
    assert _text(target) == "keep"

    server.pid_path(tmp_path).unlink()
    server.pid_path(tmp_path).mkdir()  # a directory at the pidfile: the replace cannot succeed
    with pytest.raises(OSError):
        server.write_pidfile(tmp_path, 4242, port=None)
