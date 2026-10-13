"""A pidfile is a claim, not proof (spec 20 §8.3, §12): `procfs.identity_mismatch` is the one pure
verdict on whether a `/proc` record is a harness launch, judged against `LaunchIdentity` data a
strategy supplies. Every record here is read from a fake `/proc` tree (`fake_proc`); nothing reads
the real `/proc`, signals a process or starts a server."""
from __future__ import annotations

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
