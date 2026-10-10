"""The key file at `start` (LAUNCH-05, D-21..D-23): the one guarded open and the byte rule, the read
into the session's in-memory bearer after the trust decision, the peer's read at its cold launch and
the agreement between `doctor` and the launch (WR-01).

Every file here is a tmp file; no server starts and no port is dialled.
"""
from __future__ import annotations

import fcntl
import os
import threading

import pytest

from localharness.core.private_files import NotRegularFile, open_regular
from localharness.provider import server
from localharness.provider.server import KEY_FILE_CAP, KeyFileError, read_key_file
from tests.unit.launch_support import record_opens


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
