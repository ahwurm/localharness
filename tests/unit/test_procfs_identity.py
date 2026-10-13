"""A pidfile is a claim, not proof (spec 20 §8.3, §12): `procfs.identity_mismatch` is the one pure
verdict on whether a `/proc` record is a harness launch, judged against `LaunchIdentity` data a
strategy supplies. Every record here is read from a fake `/proc` tree (`fake_proc`); nothing reads
the real `/proc`, signals a process or starts a server."""
from __future__ import annotations

import dataclasses

import pytest

from localharness.provider import procfs, server
from tests.unit.procfs_support import FakeProc, fake_proc, real_uid  # noqa: F401

LLAMA_EXE = "/opt/llama/llama-server"
LLAMA_ARGV = (LLAMA_EXE, "-m", "/m.gguf", "--port", "8080")
LLAMA = procfs.LaunchIdentity(frozenset({"llama-server"}), 8080, (), "LLAMA_ARG_PORT")


def _record(fake_proc: FakeProc, pid: int = 4242, **kw) -> procfs.ProcRecord:
    fake_proc.process(pid, **kw)
    record = procfs.read_process(pid)
    assert isinstance(record, procfs.ProcRecord), record
    return record


# ----------------------------------------------------------------------- the tracer slice


def test_the_stat_parser_lives_in_procfs_and_server_re_exports_it():
    assert server.start_time_from_stat is procfs.start_time_from_stat


def test_session_from_stat_reads_field_6_after_the_last_paren():
    raw = b"42 (a) b) c) S 1 0 4242 " + b"0 " * 16 + b"777 0 0\n"
    assert procfs.session_from_stat(raw) == 4242
    assert procfs.session_from_stat(b"garbage") is None
    assert procfs.session_from_stat(b"42 (x) S 1 2") is None


def test_a_process_record_carries_its_session_id(fake_proc):
    assert _record(fake_proc, argv=LLAMA_ARGV, exe=LLAMA_EXE).sid == 4242
    assert _record(fake_proc, 4243, argv=LLAMA_ARGV, exe=LLAMA_EXE, sid=1).sid == 1


def test_the_server_module_names_come_from_procfs(tmp_path):
    assert server.pid_path(tmp_path) == tmp_path / procfs.SERVER_DIR_NAME / procfs.PIDFILE_NAME
    assert procfs.SERVER_DIR_NAME == "vllm" and procfs.PIDFILE_NAME == "server.pid"
    assert procfs.HARNESS_UNIT == "localharness-llm.service"


def test_a_fake_llama_server_launch_passes_identity_mismatch(fake_proc):
    record = _record(fake_proc, argv=LLAMA_ARGV, exe=LLAMA_EXE)
    assert procfs.identity_mismatch(record, LLAMA) is None


# ----------------------------------------------------------------------- parse_pidfile


@pytest.mark.parametrize("text, want", [
    ("4242", (4242, None, None)),
    ("4242\n", (4242, None, None)),
    ("4242 777", (4242, 777, None)),
    ("4242 777 8080", (4242, 777, 8080)),
    ("4242 777 8080\n", (4242, 777, 8080)),
    ("4242 777 1", (4242, 777, 1)),
    ("4242 777 65535", (4242, 777, 65535)),
])
def test_parse_pidfile_reads_one_two_and_three_tokens(text, want):
    assert procfs.parse_pidfile(text) == want


@pytest.mark.parametrize("text", ["4242 777 0", "4242 777 70000", "4242 777 65536", "4242 777 -1"])
def test_parse_pidfile_reads_a_port_outside_1_to_65535_as_absent(text):
    assert procfs.parse_pidfile(text) == (4242, 777, None)


@pytest.mark.parametrize("text", ["1", "0 7", "-5 7 80", "", "   \n", "a b", "1 2 3 4", "4242 x",
                                  "4242 777 http", "4242 777 8080 9"])
def test_parse_pidfile_refuses_pid_le_1_more_than_three_tokens_and_non_integers(text):
    with pytest.raises(ValueError):
        procfs.parse_pidfile(text)


# ----------------------------------------------------------------------- identity_mismatch matrix


VLLM = procfs.LaunchIdentity(frozenset({"vllm", "python*"}), 8000, ("serve",), None)
OLLAMA = procfs.LaunchIdentity(frozenset({"ollama"}), 11434, (), "OLLAMA_HOST")
DOCKER = procfs.LaunchIdentity(frozenset({"docker"}), None, ("--name", "localharness-vllm"), None)
UNIT_CGROUP = (f"/user.slice/user-{real_uid()}.slice/user@{real_uid()}.service/app.slice/"
               f"{procfs.HARNESS_UNIT}")


def test_another_uid_is_a_mismatch(fake_proc, monkeypatch):
    record = _record(fake_proc, argv=LLAMA_ARGV, exe=LLAMA_EXE)
    other = real_uid() + 1  # read before the patch: procfs.os is the os module
    monkeypatch.setattr(procfs.os, "getuid", lambda **kw: other)
    assert procfs.identity_mismatch(record, LLAMA) == "is owned by another uid"


def test_no_readable_executable_is_a_mismatch(fake_proc):
    record = _record(fake_proc, argv=LLAMA_ARGV, exe=None)
    assert procfs.identity_mismatch(record, LLAMA) == "has no readable executable"


def test_an_executable_outside_the_names_is_a_mismatch(fake_proc):
    record = _record(fake_proc, argv=LLAMA_ARGV, exe="/usr/bin/bash")
    assert procfs.identity_mismatch(record, LLAMA) == "runs bash, not llama-server"


def test_an_exe_name_is_an_exact_basename_not_a_prefix(fake_proc):
    record = _record(fake_proc, argv=LLAMA_ARGV, exe="/opt/llama/llama-server-evil")
    assert "runs llama-server-evil" in procfs.identity_mismatch(record, LLAMA)


def test_a_star_entry_matches_a_basename_prefix(fake_proc):
    argv = ("/venv/bin/python3.12", "/venv/bin/vllm", "serve", "m", "--port", "8000")
    record = _record(fake_proc, argv=argv, exe="/venv/bin/python3.12")
    assert procfs.identity_mismatch(record, VLLM) is None
    assert "runs perl, not python*/vllm" in procfs.identity_mismatch(
        _record(fake_proc, 4243, argv=argv, exe="/usr/bin/perl"), VLLM)


def test_a_missing_argv_token_is_a_mismatch(fake_proc):
    argv = ("/venv/bin/python3.12", "/venv/bin/vllm", "--port", "8000")
    record = _record(fake_proc, argv=argv, exe="/venv/bin/python3.12")
    assert procfs.identity_mismatch(record, VLLM) == "has no serve on its command line"


def test_an_argv_token_is_a_whole_element_not_a_substring(fake_proc):
    argv = ("/venv/bin/python3.12", "observe", "--port", "8000")
    record = _record(fake_proc, argv=argv, exe="/venv/bin/python3.12")
    assert procfs.identity_mismatch(record, VLLM) == "has no serve on its command line"


def test_another_port_in_argv_is_a_mismatch(fake_proc):
    argv = (LLAMA_EXE, "-m", "/m.gguf", "--port", "8081")
    record = _record(fake_proc, argv=argv, exe=LLAMA_EXE)
    assert procfs.identity_mismatch(record, LLAMA) == "serves port 8081, not 8080"


def test_the_equals_form_of_the_port_passes(fake_proc):
    record = _record(fake_proc, argv=(LLAMA_EXE, "--port=8080"), exe=LLAMA_EXE)
    assert procfs.identity_mismatch(record, LLAMA) is None
    assert procfs.port_of(record, LLAMA) == 8080


def test_the_port_comes_from_the_environment_when_argv_has_none(fake_proc):
    record = _record(fake_proc, argv=(LLAMA_EXE, "-m", "/m.gguf"), exe=LLAMA_EXE,
                     environ={"LLAMA_ARG_PORT": "8080"})
    assert procfs.identity_mismatch(record, LLAMA) is None
    other = _record(fake_proc, 4243, argv=(LLAMA_EXE,), exe=LLAMA_EXE,
                    environ={"LLAMA_ARG_PORT": "9090"})
    assert procfs.identity_mismatch(other, LLAMA) == "serves port 9090, not 8080"


def test_argv_wins_over_the_environment(fake_proc):
    record = _record(fake_proc, argv=(LLAMA_EXE, "--port", "8080"), exe=LLAMA_EXE,
                     environ={"LLAMA_ARG_PORT": "9090"})
    assert procfs.port_of(record, LLAMA) == 8080


def test_a_host_and_port_environment_value_passes_for_ollama(fake_proc):
    record = _record(fake_proc, argv=("/usr/bin/ollama", "serve"), exe="/usr/bin/ollama",
                     environ={"OLLAMA_HOST": "127.0.0.1:11434"})
    assert procfs.identity_mismatch(record, OLLAMA) is None
    assert procfs.port_of(record, OLLAMA) == 11434


def test_a_bracketed_ipv6_host_and_port_reads_the_port(fake_proc):
    record = _record(fake_proc, argv=("/usr/bin/ollama", "serve"), exe="/usr/bin/ollama",
                     environ={"OLLAMA_HOST": "[::1]:11434"})
    assert procfs.port_of(record, OLLAMA) == 11434


def test_a_record_that_names_no_port_is_a_mismatch_when_the_identity_has_one(fake_proc):
    record = _record(fake_proc, argv=("/usr/bin/ollama", "serve"), exe="/usr/bin/ollama")
    assert procfs.identity_mismatch(record, OLLAMA) == "names no port"


@pytest.mark.parametrize("bad", ["0", "70000", "x", "", "127.0.0.1"])
def test_an_unusable_port_value_names_no_port(fake_proc, bad):
    record = _record(fake_proc, argv=(LLAMA_EXE, "--port", bad), exe=LLAMA_EXE)
    assert procfs.port_of(record, LLAMA) is None
    assert procfs.identity_mismatch(record, LLAMA) == "names no port"


def test_a_dangling_port_flag_names_no_port(fake_proc):
    record = _record(fake_proc, argv=(LLAMA_EXE, "--port"), exe=LLAMA_EXE)
    assert procfs.port_of(record, LLAMA) is None


def test_a_process_that_is_not_its_own_session_leader_is_a_mismatch(fake_proc):
    record = _record(fake_proc, argv=LLAMA_ARGV, exe=LLAMA_EXE, sid=1)
    assert procfs.identity_mismatch(record, LLAMA) == "is not its own session leader"


def test_a_record_whose_session_could_not_be_read_is_a_mismatch(fake_proc):
    base = _record(fake_proc, argv=LLAMA_ARGV, exe=LLAMA_EXE)
    assert procfs.identity_mismatch(dataclasses.replace(base, sid=None), LLAMA) \
        == "is not its own session leader"


def test_a_docker_client_passes_and_the_port_check_is_skipped(fake_proc):
    argv = ("docker", "run", "--name", "localharness-vllm", "-p", "8000:8000", "img")
    record = _record(fake_proc, argv=argv, exe="/usr/bin/docker")
    assert procfs.identity_mismatch(record, DOCKER) is None
    other = _record(fake_proc, 4243, argv=("docker", "run", "--name", "other"), exe="/usr/bin/docker")
    assert procfs.identity_mismatch(other, DOCKER) == "has no localharness-vllm on its command line"


def test_the_checks_run_in_the_order_uid_exe_argv_port_session(fake_proc):
    """A record wrong in every way reports the first check only."""
    record = _record(fake_proc, argv=("bash",), exe="/usr/bin/bash", sid=1)
    assert procfs.identity_mismatch(record, VLLM).startswith("runs bash")
    ok_exe = _record(fake_proc, 4243, argv=("python3",), exe="/usr/bin/python3", sid=1)
    assert procfs.identity_mismatch(ok_exe, VLLM) == "has no serve on its command line"
    ok_argv = _record(fake_proc, 4244, argv=("python3", "serve"), exe="/usr/bin/python3", sid=1)
    assert procfs.identity_mismatch(ok_argv, VLLM) == "names no port"
    ok_port = _record(fake_proc, 4245, argv=("python3", "serve", "--port", "8000"),
                      exe="/usr/bin/python3", sid=1)
    assert procfs.identity_mismatch(ok_port, VLLM) == "is not its own session leader"


# ----------------------------------------------------------------------- owner == "unit"


def _unit_record(fake_proc: FakeProc, *, cgroup: str | None = UNIT_CGROUP, pid: int = 4242):
    return _record(fake_proc, pid, argv=LLAMA_ARGV, exe=LLAMA_EXE, cgroup=cgroup)


def test_a_unit_owned_launch_passes_in_the_harness_unit_as_its_main_pid(fake_proc):
    record = _unit_record(fake_proc)
    assert record.unit == procfs.HARNESS_UNIT
    assert procfs.identity_mismatch(record, LLAMA, owner="unit", main_pid=4242) is None


def test_a_unit_owned_launch_needs_a_main_pid(fake_proc):
    record = _unit_record(fake_proc)
    assert procfs.identity_mismatch(record, LLAMA, owner="unit") \
        == f"is not the main process of {procfs.HARNESS_UNIT}"


def test_a_unit_owned_launch_that_is_not_the_main_pid_is_a_mismatch(fake_proc):
    record = _unit_record(fake_proc)
    assert procfs.identity_mismatch(record, LLAMA, owner="unit", main_pid=999) \
        == f"is not the main process of {procfs.HARNESS_UNIT}"


def test_a_unit_owned_launch_in_another_unit_is_a_mismatch(fake_proc):
    other = UNIT_CGROUP.replace(procfs.HARNESS_UNIT, "llama-server.service")
    record = _unit_record(fake_proc, cgroup=other)
    assert procfs.identity_mismatch(record, LLAMA, owner="unit", main_pid=4242) \
        == f"is not the main process of {procfs.HARNESS_UNIT}"


def test_a_spawned_launch_ignores_the_unit_and_the_main_pid(fake_proc):
    record = _record(fake_proc, argv=LLAMA_ARGV, exe=LLAMA_EXE)
    assert record.unit is None
    assert procfs.identity_mismatch(record, LLAMA, owner="spawned", main_pid=7) is None


# ----------------------------------------------------------------------- serve_starting


SERVE_ARGV = ("/venv/bin/python", "-I", "-m", "localharness", "serve", "--config-dir", "/c")


def _serve_record(fake_proc: FakeProc, *, argv=SERVE_ARGV, cgroup: str | None = UNIT_CGROUP):
    return _record(fake_proc, argv=argv, exe="/venv/bin/python", cgroup=cgroup)


def test_serve_starting_is_true_for_the_harness_unit_main_process(fake_proc):
    assert procfs.serve_starting(_serve_record(fake_proc), main_pid=4242) is True


def test_serve_starting_is_false_with_the_tokens_out_of_order(fake_proc):
    argv = ("/venv/bin/python", "-m", "localharness", "-I", "serve")
    assert procfs.serve_starting(_serve_record(fake_proc, argv=argv), main_pid=4242) is False


def test_serve_starting_is_false_with_the_tokens_not_consecutive(fake_proc):
    argv = ("/venv/bin/python", "-I", "-m", "localharness", "--x", "serve")
    assert procfs.serve_starting(_serve_record(fake_proc, argv=argv), main_pid=4242) is False


def test_serve_starting_is_false_in_another_unit(fake_proc):
    other = UNIT_CGROUP.replace(procfs.HARNESS_UNIT, "other.service")
    assert procfs.serve_starting(_serve_record(fake_proc, cgroup=other), main_pid=4242) is False


def test_serve_starting_is_false_outside_any_unit(fake_proc):
    assert procfs.serve_starting(_serve_record(fake_proc, cgroup=None), main_pid=4242) is False


def test_serve_starting_is_false_without_a_main_pid_or_for_another_pid(fake_proc):
    record = _serve_record(fake_proc)
    assert procfs.serve_starting(record, main_pid=None) is False
    assert procfs.serve_starting(record, main_pid=1) is False
