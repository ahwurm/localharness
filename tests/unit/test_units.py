"""The systemd side of the init import (D-21, D-22, D-50): the journal of a user unit, its unit
file, `systemctl --user show` and the enumeration of the user's own service units, all through the
one subprocess seam `units._run`. Nothing here runs a real `journalctl` or `systemctl`; this
module opts out of the suite's `_run` stub only so the real seam can be tested against a fake
`subprocess.run`."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from localharness.provider import procfs, units
from localharness.provider.procfs import ProcRecord
from tests.unit.units_support import FakeSystemd, fake_systemd  # noqa: F401

SHOW = "Id,LoadState,FragmentPath,UnitFileState,DropInPaths"


def _model(tmp_path: Path, name: str = "a.gguf") -> str:
    path = tmp_path / name
    path.write_bytes(b"GGUF")
    return str(path)


# ------------------------------------------------------------------- the suite isolation (D-60)


def test_the_suite_isolation_is_off_for_this_module_but_proc_is_empty():
    assert os.path.isdir(procfs.PROC_ROOT) and procfs.PROC_ROOT != "/proc"
    assert os.listdir(procfs.PROC_ROOT) == []
    assert procfs.listeners(8080) == []
    assert "XDG_CONFIG_HOME" not in os.environ


# -------------------------------------------------------------------------- unit_file_state


def test_unit_file_state_parses_systemctl_show(fake_systemd):
    fake_systemd.answer(
        ["systemctl", "--user", "show", "-p", SHOW, "--", "llama-server.service"],
        "Id=llama-server.service\nLoadState=loaded\n"
        "FragmentPath=/home/u/.config/systemd/user/llama-server.service\n"
        "UnitFileState=enabled\nDropInPaths=\n")
    state = units.unit_file_state("llama-server.service")
    assert state == units.UnitState("llama-server.service", "loaded",
                                    "/home/u/.config/systemd/user/llama-server.service",
                                    "enabled", "")
    assert fake_systemd.calls == [("systemctl", "--user", "show", "-p", SHOW, "--",
                                   "llama-server.service")]
    assert fake_systemd.limits == [(units.SYSTEMCTL_TIMEOUT_S, units.SYSTEMCTL_OUTPUT_CAP)]


def test_unit_file_state_is_none_when_not_checked(fake_systemd):
    assert units.unit_file_state("llama-server.service") is None  # _run answers None


def test_unit_file_state_is_none_without_a_load_state(fake_systemd):
    fake_systemd.answer_prefix("systemctl", stdout="garbage\n")
    assert units.unit_file_state("x.service") is None


def test_a_missing_unit_is_an_answer(fake_systemd):
    fake_systemd.answer_prefix("systemctl", stdout="Id=nope.service\nLoadState=not-found\n"
                                                    "FragmentPath=\nUnitFileState=\nDropInPaths=\n")
    state = units.unit_file_state("nope.service")
    assert state is not None and state.load_state == "not-found"


def test_a_value_holding_an_equals_sign_keeps_it(fake_systemd):
    fake_systemd.answer_prefix("systemctl", stdout="Id=a.service\nLoadState=loaded\n"
                                                    "FragmentPath=/h/a=b.service\n")
    assert units.unit_file_state("a.service").fragment_path == "/h/a=b.service"


# ------------------------------------------------------------------------------ read_journal


def _journal_argv(unit: str) -> tuple[str, ...]:
    return ("journalctl", "--user", "-o", "json", "-n", "200", "--all",
            "--output-fields=_EXE,_CMDLINE", "--no-pager", f"_SYSTEMD_USER_UNIT={unit}",
            f"_UID={os.getuid()}")


def test_read_journal_takes_the_last_llama_server_run(fake_systemd, tmp_path):
    model = _model(tmp_path)
    exe = "/home/u/src/llama.cpp/build/bin/llama-server"
    fake_systemd.journal([
        {"_EXE": "/opt/old/llama-server", "_CMDLINE": "llama-server -m /gone.gguf --port 1"},
        {"_EXE": exe, "_CMDLINE": f"llama-server -m {model} --alias qwen-27b --host 0.0.0.0 "
                                  "--port 8080 -c 65536"},
    ])
    rec = units.read_journal("llm.service")
    assert rec == ProcRecord(
        pid=0, uid=os.getuid(),
        argv=("llama-server", "-m", model, "--alias", "qwen-27b", "--host", "0.0.0.0",
              "--port", "8080", "-c", "65536"),
        exe=exe, cwd=None, start_time=0, unit="llm.service", environ=None, mapped_ggufs=())
    assert fake_systemd.calls == [_journal_argv("llm.service")]
    assert "-u" not in fake_systemd.calls[0]
    assert fake_systemd.limits == [(units.JOURNAL_TIMEOUT_S, units.JOURNAL_OUTPUT_CAP)]


def test_read_journal_takes_the_model_from_model_equals(fake_systemd, tmp_path):
    model = _model(tmp_path)
    fake_systemd.journal([{"_EXE": "/b/llama-server", "_CMDLINE": f"llama-server --model={model}"}])
    assert isinstance(units.read_journal("u.service"), ProcRecord)


def test_read_journal_matches_on_argv0_when_the_exe_is_a_wrapper_path(fake_systemd, tmp_path):
    model = _model(tmp_path)
    fake_systemd.journal([{"_EXE": "/usr/bin/bash", "_CMDLINE": f"/x/llama-server -m {model}"}])
    rec = units.read_journal("u.service")
    assert isinstance(rec, ProcRecord) and rec.exe == "/usr/bin/bash"


def test_read_journal_not_checked_is_a_skip_that_names_no_model(fake_systemd):
    skip = units.read_journal("llm.service")
    assert skip == units.Skip("has an unreadable journal", names_model=False)


def test_read_journal_without_a_llama_server_entry(fake_systemd):
    fake_systemd.journal([{"_EXE": "/usr/bin/python3", "_CMDLINE": "python3 -m http.server"}])
    assert units.read_journal("u.service") == units.Skip(
        "has no llama-server in its journal", names_model=False)


def test_read_journal_relative_model_is_a_skip(fake_systemd):
    fake_systemd.journal([{"_EXE": "/b/llama-server", "_CMDLINE": "llama-server -m models/a.gguf"}])
    skip = units.read_journal("u.service")
    assert isinstance(skip, units.Skip) and "relative path (models/a.gguf)" in skip.reason
    assert skip.names_model


def test_read_journal_relative_exe_is_a_skip(fake_systemd):
    fake_systemd.journal([{"_EXE": "bin/llama-server", "_CMDLINE": "llama-server -m /m/a.gguf"}])
    skip = units.read_journal("u.service")
    assert isinstance(skip, units.Skip) and "relative path (bin/llama-server)" in skip.reason


def test_read_journal_a_model_that_is_not_a_regular_file_is_a_skip(fake_systemd, tmp_path):
    fifo = tmp_path / "pipe.gguf"
    os.mkfifo(fifo)
    for cmd in (f"llama-server -m {tmp_path}/missing.gguf", f"llama-server -m {tmp_path}",
                f"llama-server -m {fifo}", "llama-server --port 8080"):
        fake_systemd.journal([{"_EXE": "/b/llama-server", "_CMDLINE": cmd}])
        assert units.read_journal("u.service") == units.Skip("names no regular -m file"), cmd
