"""The key file at `start` (LAUNCH-05, D-21..D-23): the one guarded open and the byte rule, the read
into the session's in-memory bearer after the trust decision, the peer's read at its cold launch and
the agreement between `doctor` and the launch (WR-01).

Every file here is a tmp file; no server starts and no port is dialled.
"""
from __future__ import annotations

import asyncio
import fcntl
import os
import threading

import pytest

from localharness.core.private_files import NotRegularFile, open_regular
from localharness.provider import server
from localharness.provider.server import KEY_FILE_CAP, KeyFileError, read_key_file
from tests.unit.launch_support import popen, record_opens  # noqa: F401


def _fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


def _write(tmp_path, data: bytes, name: str = "key"):
    path = tmp_path / name
    path.write_bytes(data)
    return path


# --- open_regular: the Phase 2 helper's contract this plan relies on -------------------------------

def test_open_regular_returns_a_close_on_exec_descriptor_of_the_file(tmp_path):
    path = _write(tmp_path, b"K\n")
    fd = open_regular(path)
    try:
        assert os.fstat(fd).st_ino == path.stat().st_ino
        assert fcntl.fcntl(fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
    finally:
        os.close(fd)


@pytest.mark.parametrize("make", ["dir", "fifo", "dev_null"])
def test_open_regular_refuses_anything_but_a_regular_file_and_leaks_no_descriptor(tmp_path, make):
    if make == "dir":
        target = tmp_path / "d"
        target.mkdir()
    elif make == "fifo":
        target = tmp_path / "f"
        os.mkfifo(target)
    else:
        target = "/dev/null"
    before = _fd_count()
    outcome: list[BaseException | None] = []

    def attempt():
        try:
            os.close(open_regular(target))
            outcome.append(None)
        except BaseException as exc:  # noqa: BLE001 - the test records whatever came back
            outcome.append(exc)

    worker = threading.Thread(target=attempt, daemon=True)
    worker.start()
    worker.join(5)
    assert not worker.is_alive(), "open_regular blocked"
    assert isinstance(outcome[0], NotRegularFile)
    assert _fd_count() == before


def test_open_regular_on_a_missing_path_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_regular(tmp_path / "absent")


# --- read_key_file: the byte rule ---------------------------------------------------------------

def test_the_constants_are_what_the_decision_names():
    assert KEY_FILE_CAP == 4096
    assert issubclass(KeyFileError, RuntimeError)


@pytest.mark.parametrize("content", [b"K\n", b"K", b"\nK\n\n", b"\n\n\nK"])
def test_a_single_valid_line_is_the_key_with_empty_lines_dropped(tmp_path, content):
    assert read_key_file(_write(tmp_path, content)) == "K"


@pytest.mark.parametrize("content, reason", [
    (b"#c\nK\n", "comment line"),
    (b"K\n# trailing\n", "comment line"),
    (b"K1\nK2\n", "more than one line"),
    (b"", "no key"),
    (b"\n\n", "no key"),
    (b"K\r\n", "byte outside printable ASCII"),
    (b"K\xc3\xa9", "byte outside printable ASCII"),
    (b"K \n", "byte outside printable ASCII"),
    (b"a b\n", "byte outside printable ASCII"),
    (b"a\tb\n", "byte outside printable ASCII"),
    (b"a,b\n", "comma or quote"),
    (b'a"b\n', "comma or quote"),
])
def test_a_malformed_file_is_refused_with_the_reason_named(tmp_path, content, reason):
    with pytest.raises(KeyFileError) as raised:
        read_key_file(_write(tmp_path, content))
    assert reason in str(raised.value)


def test_an_all_comment_file_is_refused(tmp_path):
    with pytest.raises(KeyFileError, match="comment line"):
        read_key_file(_write(tmp_path, b"#\n"))


def test_exactly_the_cap_is_accepted_and_one_byte_more_is_refused(tmp_path):
    assert read_key_file(_write(tmp_path, b"x" * KEY_FILE_CAP)) == "x" * KEY_FILE_CAP
    with pytest.raises(KeyFileError, match="over 4096 bytes"):
        read_key_file(_write(tmp_path, b"x" * (KEY_FILE_CAP + 1), "big"))


def test_a_file_far_over_the_cap_is_read_no_further_than_the_cap_plus_one(tmp_path, monkeypatch):
    path = _write(tmp_path, b"x" * (KEY_FILE_CAP * 10))
    asked: list[int] = []
    real_fdopen = os.fdopen

    def spy(fd, mode="r", *a, **kw):
        handle = real_fdopen(fd, mode, *a, **kw)
        real_read = handle.read
        handle.read = lambda n=-1: (asked.append(n), real_read(n))[1]
        return handle

    monkeypatch.setattr(server.os, "fdopen", spy)
    with pytest.raises(KeyFileError, match="over"):
        read_key_file(path)
    assert asked == [KEY_FILE_CAP + 1]


def test_a_directory_and_a_missing_file_name_their_reasons(tmp_path):
    with pytest.raises(KeyFileError, match="not a regular file"):
        read_key_file(tmp_path)
    with pytest.raises(KeyFileError, match="no such file"):
        read_key_file(tmp_path / "absent")


def test_a_fifo_key_file_is_refused_within_five_seconds_and_opened_once(tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    outcome: list[BaseException | str] = []

    def attempt():
        try:
            outcome.append(read_key_file(fifo))
        except KeyFileError as exc:
            outcome.append(exc)

    with record_opens() as opened:
        worker = threading.Thread(target=attempt, daemon=True)
        worker.start()
        worker.join(5)
    assert not worker.is_alive(), "read_key_file blocked on a FIFO"
    assert isinstance(outcome[0], KeyFileError) and "not a regular file" in str(outcome[0])
    assert opened.count(str(fifo)) == 1


def test_a_read_opens_the_path_exactly_once(tmp_path):
    path = _write(tmp_path, b"K\n")
    with record_opens() as opened:
        read_key_file(path)
    assert opened.count(str(path)) == 1


def test_reading_twice_returns_the_same_key_and_leaves_the_file_unchanged(tmp_path):
    path = _write(tmp_path, b"K\n")
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    assert read_key_file(path) == read_key_file(path) == "K"
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_a_tilde_path_is_expanded(tmp_path, monkeypatch):
    _write(tmp_path, b"K\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert read_key_file("~/key") == "K"


# --- _read_session_key: the harness's in-memory bearer ----------------------------------------------

def _harness(path, **over):
    from types import SimpleNamespace

    from pydantic import SecretStr

    from localharness.config.models import ManagedServerConfig
    spec = ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server", model="/x/m.gguf", port=8080,
                               require_api_key=True, api_key_file=str(path), **over)
    return SimpleNamespace(server=spec, provider=SimpleNamespace(api_key=SecretStr("none")))


def test_a_valid_file_becomes_the_sessions_secret_and_returns_no_line(tmp_path):
    from pydantic import SecretStr

    from localharness.cli.start_cmd import _read_session_key
    harness = _harness(_write(tmp_path, b"KEY-SENTINEL-1\n"))
    assert _read_session_key(harness) is None
    assert isinstance(harness.provider.api_key, SecretStr)
    assert harness.provider.api_key.get_secret_value() == "KEY-SENTINEL-1"
    assert "KEY-SENTINEL-1" not in repr(harness.provider.api_key)
    assert server.required_key_kwargs(harness.server, harness.provider.api_key) == {"api_key": "KEY-SENTINEL-1"}


def test_a_refused_file_returns_the_one_line_and_keeps_the_session_keyless(tmp_path):
    from localharness.cli.start_cmd import _read_session_key
    path = _write(tmp_path, b"a,b\n")
    harness = _harness(path)
    assert _read_session_key(harness) == (
        f"server.api_key_file {path}: comma or quote — this session sends no key")
    assert harness.provider.api_key.get_secret_value() == "none"
    assert server.required_key_kwargs(harness.server, harness.provider.api_key) == {}


def test_a_refused_file_never_leaks_its_content_into_the_line(tmp_path):
    from localharness.cli.start_cmd import _read_session_key
    path = _write(tmp_path, b"KEY-SENTINEL-2,x\n")
    assert "KEY-SENTINEL-2" not in _read_session_key(_harness(path))


def test_an_environment_key_is_replaced_by_the_files_key_and_kept_when_the_file_fails(tmp_path):
    from pydantic import SecretStr

    from localharness.cli.start_cmd import _read_session_key
    good = _harness(_write(tmp_path, b"FROM-FILE\n"))
    good.provider.api_key = SecretStr("FROM-ENV")
    assert _read_session_key(good) is None
    assert good.provider.api_key.get_secret_value() == "FROM-FILE"
    bad = _harness(_write(tmp_path, b"", "empty"))
    bad.provider.api_key = SecretStr("FROM-ENV")
    assert "no key" in _read_session_key(bad)
    assert bad.provider.api_key.get_secret_value() == "FROM-ENV"


# --- doctor and the launch agree (WR-01) ------------------------------------------------------------

LLAMA_RECORD = {"runtime": "llamacpp", "binary": "/x/llama-server", "model": "/x/m.gguf", "port": 8080,
                "bind_all": True, "require_api_key": True}


async def _launch_argv(tmp_path, monkeypatch, popen, key_path) -> list[str]:
    """The argv `start` would launch a `bind_all: true` llama.cpp record with: the key file read by
    `_read_session_key`, the launch key built from the session's bearer, the real `start_server`
    handing a recorded Popen. No process starts and no port is dialled."""
    from types import SimpleNamespace

    from pydantic import SecretStr

    from localharness.cli import start_cmd
    from localharness.config.models import ManagedServerConfig
    from localharness.provider import autosense

    async def wait_ready(base_url, **kw):
        return ["m"]

    monkeypatch.setattr(server, "wait_ready", wait_ready)
    monkeypatch.setattr(server, "server_pid", lambda _d: None)
    monkeypatch.setattr(server, "docker_container_running", lambda _n: False)
    monkeypatch.setattr(autosense, "_run_nvidia_smi", lambda: None)
    monkeypatch.setattr(start_cmd.console, "print", lambda *a, **k: None)
    monkeypatch.setattr(start_cmd.err_console, "print", lambda *a, **k: None)
    spec = ManagedServerConfig(**LLAMA_RECORD, api_key_file=str(key_path))
    harness = SimpleNamespace(server=spec, provider=SimpleNamespace(api_key=SecretStr("none")))
    start_cmd._read_session_key(harness)
    launch_key = server.required_key_kwargs(harness.server, harness.provider.api_key)
    assert await start_cmd._start_managed_server(
        spec, tmp_path, "http://127.0.0.1:8080/v1", launch_key) is None
    (call,) = popen
    return call["cmd"]


def _doctor_text(tmp_path, monkeypatch, fake_home, key_path) -> str:
    from tests.unit.test_doctor_exposure_rows import _doctor
    return _doctor(tmp_path, monkeypatch, fake_home, key="none",
                   server={**LLAMA_RECORD, "api_key_file": str(key_path)})


def test_a_valid_key_file_reads_keyed_and_exposed_in_doctor_and_launches_so(
        tmp_path, monkeypatch, fake_home, popen):
    from tests.unit.test_doctor_exposure_rows import EVERY_INTERFACE, LLAMA_NETWORK_ROW
    key = _write(tmp_path, b"K\n")
    out = _doctor_text(tmp_path, monkeypatch, fake_home, key)
    assert LLAMA_NETWORK_ROW in out and EVERY_INTERFACE not in out
    argv = asyncio.run(_launch_argv(tmp_path, monkeypatch, popen, key))
    assert "--api-key-file" in argv and "0.0.0.0" in argv


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_a_missing_or_non_regular_key_file_reads_loopback_in_doctor_and_launches_so(
        tmp_path, monkeypatch, fake_home, popen, kind):
    key = tmp_path / "absent"
    if kind == "directory":
        key.mkdir()
    out = _doctor_text(tmp_path, monkeypatch, fake_home, key)
    assert "reachable from other machines" not in out
    argv = asyncio.run(_launch_argv(tmp_path, monkeypatch, popen, key))
    assert "--api-key-file" not in argv and argv[-2:] == ["--host", "127.0.0.1"]


def test_the_wr01_residual_a_refused_regular_file_reads_keyed_in_doctor_but_launches_loopback(
        tmp_path, monkeypatch, fake_home, popen):
    """Recorded, not closed (DOCTOR-03: doctor stats the key file and never opens it): a regular file
    the byte rule refuses still reads keyed in doctor, while `start` prints the one line and
    launches on loopback."""
    from tests.unit.test_doctor_exposure_rows import LLAMA_NETWORK_ROW
    key = _write(tmp_path, b"K\r\n")
    assert LLAMA_NETWORK_ROW in _doctor_text(tmp_path, monkeypatch, fake_home, key)
    argv = asyncio.run(_launch_argv(tmp_path, monkeypatch, popen, key))
    assert "--api-key-file" not in argv and argv[-2:] == ["--host", "127.0.0.1"]
