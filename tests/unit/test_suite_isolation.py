"""D-60: outside test_units every test sees an empty /proc and a systemd that answers "not checked".
WR-09: no test dials a real service socket."""
from __future__ import annotations

import errno
import os
import socket
import threading

import httpx
import pytest

from localharness.provider import procfs, units
from tests.conftest import _REAL_CONNECT


def test_proc_root_is_an_existing_empty_directory():
    assert procfs.PROC_ROOT != "/proc"
    assert os.path.isdir(procfs.PROC_ROOT) and os.listdir(procfs.PROC_ROOT) == []
    assert procfs.listeners(8080) == []


def test_the_systemd_seam_answers_not_checked_without_running_anything(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a test spawned a process")

    monkeypatch.setattr(units.subprocess, "run", boom)
    assert units._run(["systemctl", "--user", "show"], timeout=1, cap=1) is None
    assert units.unit_file_state("llm.service") is None
    assert isinstance(units.read_journal("llm.service"), units.Skip)


def test_xdg_config_home_is_unset(monkeypatch):
    assert "XDG_CONFIG_HOME" not in os.environ


# ------------------------------------------------------------------------- WR-09: no real sockets

REAL_SERVICE_PORTS = (8080, 11434, 1234, 8000)


@pytest.mark.parametrize("port", REAL_SERVICE_PORTS)
def test_a_default_service_port_reads_as_nothing_listening(port):
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("127.0.0.1", port), timeout=1)
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", port)) == errno.ECONNREFUSED
    with pytest.raises(httpx.ConnectError):
        httpx.get(f"http://127.0.0.1:{port}/props", timeout=1)


def test_the_guard_is_what_answers_not_a_machine_with_nothing_up():
    assert socket.socket.connect is not _REAL_CONNECT
    with pytest.raises(ConnectionRefusedError, match="a test tried to dial"):
        socket.create_connection(("127.0.0.1", 8080), timeout=1)


def test_the_machines_network_is_not_dialable():
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("192.0.2.1", 80), timeout=1)
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("10.0.0.5", 11434), timeout=1)


def test_an_in_process_fake_server_on_a_port_the_kernel_chose_is_reachable():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    accepted: list[bytes] = []

    def serve():
        conn, _ = server.accept()
        with conn:
            accepted.append(conn.recv(4))
            conn.sendall(b"pong")

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            client.sendall(b"ping")
            assert client.recv(4) == b"pong"
    finally:
        thread.join(2)
        server.close()
    assert accepted == [b"ping"]


@pytest.mark.real_sockets
def test_a_test_marked_real_sockets_keeps_the_real_connect():
    assert socket.socket.connect is _REAL_CONNECT
