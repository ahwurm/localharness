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


BASE = "http://127.0.0.1:8080/v1"


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
    return SimpleNamespace(server=spec, provider=SimpleNamespace(api_key=SecretStr("none"), base_url=BASE))


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
    call, _ = await _launch_call(tmp_path, monkeypatch, popen, key_path)
    return call["cmd"]


async def _launch_call(tmp_path, monkeypatch, popen, key_path, provider_key="none"):
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
    harness = SimpleNamespace(server=spec, provider=SimpleNamespace(api_key=SecretStr(provider_key), base_url=BASE))
    line = start_cmd._read_session_key(harness)
    launch_key = server.required_key_kwargs(harness.server, harness.provider.api_key)
    assert await start_cmd._start_managed_server(
        spec, tmp_path, "http://127.0.0.1:8080/v1", launch_key) is None
    (call,) = popen
    return call, line


def _doctor_text(tmp_path, monkeypatch, fake_home, key_path) -> str:
    from tests.unit.test_doctor_exposure_rows import _doctor
    # The `popen` fixture replaces the process-wide subprocess.Popen with a recorder that starts
    # nothing; doctor's git_read row (10-04) would run git through subprocess.run on top of it. This
    # file grades the key-file rows, so the row is switched off (no repository found, no git run).
    from localharness.tools.builtin import git_read_tool
    monkeypatch.setattr(git_read_tool, "git_read_repository_present", lambda root: False)
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


def test_cr01_a_refused_file_with_another_key_held_stays_off_argv_and_launches_loopback(
        tmp_path, monkeypatch, popen):
    """The decision, the argv and the printed line agree: the harness holds a configured key, the byte
    rule refuses the CRLF file, so the file never rides argv, the launch is loopback even though
    `bind_all` is true, the other key still reaches the child, and the line says the file is not used
    (never `sends no key`)."""
    key = _write(tmp_path, b"K\r\n")
    call, line = asyncio.run(_launch_call(tmp_path, monkeypatch, popen, key, provider_key="CONFIGURED"))
    argv = call["cmd"]
    assert "--api-key-file" not in argv and str(key) not in argv
    assert "0.0.0.0" not in argv and argv[-2:] == ["--host", "127.0.0.1"]
    assert call["env"]["LLAMA_API_KEY"] == "CONFIGURED"
    assert line == (f"server.api_key_file {key}: byte outside printable ASCII — the file is not used; "
                    "this session keeps sending the key it already holds")
    assert "sends no key" not in line and "CONFIGURED" not in line


def test_cr01_a_refused_file_with_no_other_key_keeps_the_sends_no_key_line_and_loopback(
        tmp_path, monkeypatch, popen):
    key = _write(tmp_path, b"K\r\n")
    call, line = asyncio.run(_launch_call(tmp_path, monkeypatch, popen, key))
    argv = call["cmd"]
    assert "--api-key-file" not in argv and argv[-2:] == ["--host", "127.0.0.1"]
    assert "LLAMA_API_KEY" not in (call["env"] or {})
    assert line.endswith("— this session sends no key")


def test_cr01_the_decision_follows_the_refusal_even_when_the_harness_is_keyed(tmp_path):
    from localharness.config.models import ManagedServerConfig
    key = _write(tmp_path, b"K\n")
    spec = ManagedServerConfig(**LLAMA_RECORD, api_key_file=str(key))
    ok = server.launch_decision(spec, True)
    assert (ok.loopback, ok.key_file_ok, ok.note) == (False, True, None)
    server.mark_key_file(spec, True)
    refused = server.launch_decision(spec, True)
    assert (refused.loopback, refused.key_file_ok) == (True, False)
    assert refused.note == f"--api-key-file {key} unreadable: refused by the key-file rule — launched on 127.0.0.1 for this run"
    assert "--api-key-file" not in server.serve_command(
        spec, keyed=True, loopback=refused.loopback, key_file_ok=refused.key_file_ok)
    assert server.effective_posture(spec, keyed=True, decision=refused) == (False, None)
    server.mark_key_file(spec, False)  # a later successful read clears it
    assert server.launch_decision(spec, True).key_file_ok is True


def test_cr01_the_refusal_is_session_state_not_configuration(tmp_path):
    from localharness.config.models import ManagedServerConfig
    spec = ManagedServerConfig(**LLAMA_RECORD, api_key_file=str(_write(tmp_path, b"K\n")))
    before = spec.model_dump()
    server.mark_key_file(spec, True)
    assert spec.key_file_refused is True and spec.model_dump() == before


# --- WR-06: the key goes to provider.base_url, so only this machine gets one out of a file --------------


@pytest.mark.parametrize("base_url", [
    "http://127.0.0.1:8080/v1", "http://localhost:8080/v1", "http://[::1]:8080/v1", "http://127.0.0.2/v1"])
def test_wr06_a_loopback_endpoint_gets_the_files_key(tmp_path, base_url):
    from localharness.cli.start_cmd import _read_session_key
    harness = _harness(_write(tmp_path, b"FILE-KEY\n"))
    harness.provider.base_url = base_url
    assert _read_session_key(harness) is None
    assert harness.provider.api_key.get_secret_value() == "FILE-KEY"


@pytest.mark.parametrize("base_url", [
    "http://192.168.1.50:8080/v1", "http://attacker.example/v1", "https://203.0.113.9:9/v1", "not a url"])
def test_wr06_any_other_endpoint_never_has_the_file_opened_or_its_key_sent(tmp_path, base_url):
    from localharness.cli.start_cmd import _read_session_key
    from tests.unit.launch_support import record_opens
    key = _write(tmp_path, b"FILE-KEY\n")
    harness = _harness(key)
    harness.provider.base_url = base_url
    with record_opens() as opened:
        line = _read_session_key(harness)
    assert str(key) not in opened
    assert harness.provider.api_key.get_secret_value() == "none"
    assert line.startswith(f"server.api_key_file {key}: ") and "is not this machine" in line
    assert line.endswith("— this session sends no key") and "FILE-KEY" not in line
    assert harness.server.key_file_refused is True  # the launch keeps the file off argv too


def test_wr06_a_configured_key_is_kept_and_said_so_when_the_endpoint_moved(tmp_path):
    from pydantic import SecretStr

    from localharness.cli.start_cmd import _read_session_key
    harness = _harness(_write(tmp_path, b"FILE-KEY\n"))
    harness.provider.base_url = "http://192.168.1.50:8080/v1"
    harness.provider.api_key = SecretStr("CONFIGURED")
    line = _read_session_key(harness)
    assert "the file is not used; this session keeps sending the key it already holds" in line
    assert harness.provider.api_key.get_secret_value() == "CONFIGURED"


def test_wr06_a_url_with_userinfo_prints_only_host_and_port(tmp_path):
    from localharness.cli.start_cmd import _read_session_key
    harness = _harness(_write(tmp_path, b"FILE-KEY\n"))
    harness.provider.base_url = "http://user:SECRET-PW@192.168.1.50:8080/v1?token=SECRET-TOK"
    line = _read_session_key(harness)
    assert "192.168.1.50:8080" in line and "SECRET" not in line


def test_wr06_the_launch_for_a_moved_endpoint_is_loopback_without_the_file(tmp_path, monkeypatch, popen):
    key = _write(tmp_path, b"K\n")

    async def go():
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
        spec = ManagedServerConfig(**LLAMA_RECORD, api_key_file=str(key))
        harness = SimpleNamespace(server=spec, provider=SimpleNamespace(
            api_key=SecretStr("none"), base_url="http://192.168.1.50:8080/v1"))
        start_cmd._read_session_key(harness)
        launch_key = server.required_key_kwargs(harness.server, harness.provider.api_key)
        assert await start_cmd._start_managed_server(
            spec, tmp_path, "http://192.168.1.50:8080/v1", launch_key) is None

    asyncio.run(go())
    (call,) = popen
    assert "--api-key-file" not in call["cmd"] and call["cmd"][-2:] == ["--host", "127.0.0.1"]


# --- IN-01: file-supplied text in the key-file lines is shown escaped ----------------------------------


def test_in01_an_escape_sequence_in_the_configured_path_prints_escaped(tmp_path):
    from localharness.cli.start_cmd import _read_session_key
    path = tmp_path / "\x1b[31mred\x1b[0m"
    line = _read_session_key(_harness(path))
    assert "\x1b" not in line and "\\x1b[31mred\\x1b[0m" in line
    assert line.endswith("no such file — this session sends no key")
