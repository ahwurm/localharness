"""START-12 / START-13 / COMPAT-01: the pidfile names its port and `procfs.read_launch` is the one reader
of "which launch is live and whose is it" (spec 20 §8.3, §12 "a pidfile is a claim, not proof").

Everything runs on a tmp config dir and a fake `/proc` tree (`fake_proc`); `subprocess.Popen` is the
suite's recorder, so no process starts and no signal is sent. Test fakes take `**kw`.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from localharness.config.models import ManagedServerConfig
from localharness.provider import lifecycle, procfs, server
from tests.unit.launch_support import PopenRecorder, popen  # noqa: F401
from tests.unit.procfs_support import FakeProc, fake_proc, stat_bytes  # noqa: F401

LLAMA_EXE = "/opt/llama/llama-server"
LLAMA_ARGV = (LLAMA_EXE, "-m", "/x/m.gguf", "--port", "8080")
UNIT_CGROUP = f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/localharness-llm.service"
SPEC = ManagedServerConfig(runtime="llamacpp", binary=LLAMA_EXE, model="/x/m.gguf", port=8080)


def identity() -> procfs.LaunchIdentity:
    ident = lifecycle.SpawnedProcessStrategy().launch_identity(SPEC)
    assert ident is not None
    return ident


def write_pidfile(config_dir: Path, text: str) -> Path:
    directory = server.server_dir(config_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / procfs.PIDFILE_NAME
    path.write_text(text, encoding="utf-8")
    return path


# ------------------------------------------------------------------------------- the tracer slice


def test_a_launch_writes_pid_start_and_port_privately(tmp_path, monkeypatch, popen):  # noqa: F811
    monkeypatch.setattr(server, "_proc_start_time", lambda pid: 777)
    assert server.start_server(tmp_path, ["true"], runtime="llamacpp", port=8080) == 4242
    path = server.pid_path(tmp_path)
    assert path.read_text(encoding="utf-8") == "4242 777 8080"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [p.name for p in server.server_dir(tmp_path).iterdir() if p.name != "serve.log"] == [procfs.PIDFILE_NAME]


def test_read_launch_verifies_the_process_against_the_strategys_identity(tmp_path, fake_proc):  # noqa: F811
    fake_proc.process(4242, argv=LLAMA_ARGV, exe=LLAMA_EXE, start_time=777)
    write_pidfile(tmp_path, "4242 777 8080")
    assert procfs.read_launch(tmp_path, expect=identity()) == procfs.Launch(
        4242, 777, 8080, "spawned", True, False)


def test_server_pid_is_built_on_the_launch_reader(tmp_path, monkeypatch, fake_proc):  # noqa: F811
    fake_proc.process(4242, argv=LLAMA_ARGV, exe=LLAMA_EXE, start_time=777)
    write_pidfile(tmp_path, "4242 777 8080")
    monkeypatch.setattr(server.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(server, "_read_proc_stat", lambda pid: stat_bytes(pid, 777))
    assert server.server_pid(tmp_path) == 4242
    assert server.pid_path(tmp_path).exists()


def test_the_launch_identity_of_a_llama_server_names_its_binary_and_port():
    assert identity() == procfs.LaunchIdentity(frozenset({"llama-server"}), 8080, (), "LLAMA_ARG_PORT")


def test_the_writer_writes_two_tokens_without_a_port_and_one_without_a_start(
        tmp_path, monkeypatch, popen):  # noqa: F811
    monkeypatch.setattr(server, "_proc_start_time", lambda pid: 777)
    server.start_server(tmp_path, ["true"])
    assert server.pid_path(tmp_path).read_text(encoding="utf-8") == "4242 777"
    monkeypatch.setattr(server, "_proc_start_time", lambda pid: None)
    server.start_server(tmp_path, ["true"], port=8080)
    assert server.pid_path(tmp_path).read_text(encoding="utf-8") == "4242"


# ------------------------------------------------------------------------------- legacy files (COMPAT-01)


@pytest.mark.parametrize("text, start", [("4242", None), ("4242 777", 777)])
def test_legacy_pidfiles_still_read_with_no_port(tmp_path, fake_proc, text, start):  # noqa: F811
    fake_proc.process(4242, argv=LLAMA_ARGV, exe=LLAMA_EXE, start_time=777)
    write_pidfile(tmp_path, text)
    assert procfs.read_launch(tmp_path) == procfs.Launch(4242, start, None, "spawned", False)


@pytest.mark.parametrize("text", ["4242", "4242 777"])
def test_a_legacy_file_takes_its_port_from_the_verified_process(tmp_path, fake_proc, text):  # noqa: F811
    fake_proc.process(4242, argv=LLAMA_ARGV, exe=LLAMA_EXE, start_time=777)
    write_pidfile(tmp_path, text)
    launch = procfs.read_launch(tmp_path, expect=identity())
    assert launch is not None and launch.verified and launch.port == 8080


def test_a_legacy_file_never_takes_a_port_from_an_unverified_process(tmp_path, fake_proc):  # noqa: F811
    fake_proc.process(4242, argv=("/usr/bin/sleep", "--port", "8080"), exe="/usr/bin/sleep", start_time=777)
    write_pidfile(tmp_path, "4242 777")
    launch = procfs.read_launch(tmp_path, expect=identity())
    assert launch is not None and not launch.verified and launch.port is None


def test_server_pid_keeps_its_stale_file_and_reused_pid_behaviour_on_a_legacy_file(
        tmp_path, monkeypatch, fake_proc):  # noqa: F811
    monkeypatch.setattr(server.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(server, "_read_proc_stat", lambda pid: stat_bytes(pid, 778))
    write_pidfile(tmp_path, "4242 777")
    assert server.server_pid(tmp_path) is None
    assert not server.pid_path(tmp_path).exists()

    def gone(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(server.os, "kill", gone)
    write_pidfile(tmp_path, "4242")
    assert server.server_pid(tmp_path) is None
    assert not server.pid_path(tmp_path).exists()


# ------------------------------------------------------------------------------- planted links


def test_a_symlinked_server_directory_stops_the_launch_before_anything_starts(tmp_path, popen):  # noqa: F811
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / procfs.SERVER_DIR_NAME).symlink_to(elsewhere)
    with pytest.raises(RuntimeError, match=r"could not open the server directory .*vllm"):
        server.start_server(tmp_path, ["true"], runtime="llamacpp", port=8080)
    assert popen == []
    assert list(elsewhere.iterdir()) == []


def test_read_launch_reads_nothing_through_a_symlinked_server_directory(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / procfs.PIDFILE_NAME).write_text("4242 777 8080", encoding="utf-8")
    (tmp_path / procfs.SERVER_DIR_NAME).symlink_to(elsewhere)
    assert procfs.read_launch(tmp_path) is None
    assert procfs.read_pidfile(tmp_path) is None


def test_a_pidfile_that_is_a_link_is_never_read_through(tmp_path):
    target = tmp_path / "x"
    target.write_text("4242 777 8080", encoding="utf-8")
    server.server_dir(tmp_path).mkdir(parents=True)
    server.pid_path(tmp_path).symlink_to(target)
    assert procfs.read_launch(tmp_path) is None


def test_a_launch_replaces_a_planted_link_at_the_pidfile_and_never_follows_it(
        tmp_path, monkeypatch, popen):  # noqa: F811
    monkeypatch.setattr(server, "_proc_start_time", lambda pid: 777)
    target = tmp_path / "x"
    target.write_text("untouched", encoding="utf-8")
    server.server_dir(tmp_path).mkdir(parents=True)
    server.pid_path(tmp_path).symlink_to(target)
    server.start_server(tmp_path, ["true"], runtime="llamacpp", port=8080)
    assert not server.pid_path(tmp_path).is_symlink()
    assert server.pid_path(tmp_path).read_text(encoding="utf-8") == "4242 777 8080"
    assert target.read_text(encoding="utf-8") == "untouched"


# ------------------------------------------------------------------------------- mismatches, caps


def test_a_process_that_is_not_the_launch_is_unverified_and_the_file_stays(tmp_path, fake_proc):  # noqa: F811
    fake_proc.process(4242, argv=("/usr/bin/sleep", "60"), exe="/usr/bin/sleep", start_time=777)
    path = write_pidfile(tmp_path, "4242 777 8080")
    launch = procfs.read_launch(tmp_path, expect=identity())
    assert launch is not None and launch.verified is False
    assert path.exists()


def test_a_reused_pid_reads_none_and_read_launch_never_unlinks(tmp_path, fake_proc):  # noqa: F811
    fake_proc.process(4242, argv=LLAMA_ARGV, exe=LLAMA_EXE, start_time=778)
    path = write_pidfile(tmp_path, "4242 777 8080")
    assert procfs.read_launch(tmp_path, expect=identity()) is None
    assert path.exists()


def test_a_missing_process_is_none_when_a_start_is_recorded_and_unverified_when_not(
        tmp_path, fake_proc):  # noqa: F811
    path = write_pidfile(tmp_path, "4242 777 8080")
    assert procfs.read_launch(tmp_path, expect=identity()) is None
    path.write_text("4242", encoding="utf-8")
    launch = procfs.read_launch(tmp_path, expect=identity())
    assert launch is not None and launch.verified is False
    assert path.exists()


@pytest.mark.parametrize("text", ["4", "4242 777 8080 9", "0 777 8080", "1", "4242 x", "-4242", ""])
def test_a_pidfile_over_the_cap_or_off_the_shape_reads_none(tmp_path, text):
    path = write_pidfile(tmp_path, text if text != "4" else "4" * (procfs.PIDFILE_CAP + 1))
    assert procfs.read_launch(tmp_path) is None
    assert path.exists()


def test_a_port_out_of_range_in_the_file_reads_as_no_port(tmp_path):
    write_pidfile(tmp_path, "4242 777 70000")
    assert procfs.read_pidfile(tmp_path) == (4242, 777, None)


# ------------------------------------------------------------------------------- owner and starting


def test_owner_is_unit_only_for_the_harness_units_main_process(tmp_path, fake_proc):  # noqa: F811
    fake_proc.process(4242, argv=LLAMA_ARGV, exe=LLAMA_EXE, start_time=777, cgroup=UNIT_CGROUP)
    write_pidfile(tmp_path, "4242 777 8080")
    unit = procfs.read_launch(tmp_path, expect=identity(), main_pid=4242)
    assert unit == procfs.Launch(4242, 777, 8080, "unit", True, False)
    assert procfs.read_launch(tmp_path, expect=identity(), main_pid=None).owner == "spawned"
    assert procfs.read_launch(tmp_path, expect=identity(), main_pid=4243).owner == "spawned"


def test_owner_is_spawned_for_a_process_outside_the_units_cgroup_even_as_main_pid(
        tmp_path, fake_proc):  # noqa: F811
    fake_proc.process(4242, argv=LLAMA_ARGV, exe=LLAMA_EXE, start_time=777, cgroup="/user.slice/other.service")
    write_pidfile(tmp_path, "4242 777 8080")
    launch = procfs.read_launch(tmp_path, expect=identity(), main_pid=4242)
    assert launch is not None and launch.owner == "spawned"


def test_the_unit_serving_process_is_starting_and_not_yet_a_verified_server(tmp_path, fake_proc):  # noqa: F811
    argv = ("/usr/bin/python3", "-I", "-m", "localharness", "serve")
    fake_proc.process(4242, argv=argv, exe="/usr/bin/python3", start_time=777, cgroup=UNIT_CGROUP)
    write_pidfile(tmp_path, "4242 777")
    launch = procfs.read_launch(tmp_path, expect=identity(), main_pid=4242)
    assert launch is not None
    assert launch.starting is True and launch.verified is False and launch.owner == "unit"


def test_a_serve_process_that_is_not_the_units_main_pid_is_not_starting(tmp_path, fake_proc):  # noqa: F811
    argv = ("/usr/bin/python3", "-I", "-m", "localharness", "serve")
    fake_proc.process(4242, argv=argv, exe="/usr/bin/python3", start_time=777, cgroup=UNIT_CGROUP)
    write_pidfile(tmp_path, "4242 777")
    for main in (None, 9999):
        launch = procfs.read_launch(tmp_path, expect=identity(), main_pid=main)
        assert launch is not None and launch.starting is False


def test_a_server_that_is_up_is_not_starting(tmp_path, fake_proc):  # noqa: F811
    fake_proc.process(4242, argv=LLAMA_ARGV, exe=LLAMA_EXE, start_time=777, cgroup=UNIT_CGROUP)
    write_pidfile(tmp_path, "4242 777 8080")
    launch = procfs.read_launch(tmp_path, expect=identity(), main_pid=4242)
    assert launch is not None and launch.starting is False and launch.verified is True
