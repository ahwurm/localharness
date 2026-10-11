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

pytestmark = pytest.mark.real_units_seam  # tests/conftest.py leaves `units._run` real for this module

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


# --------------------------------------------------------------------------------- read_unit


def _unit_file(tmp_path: Path, body: str, name: str = "llm.service") -> str:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return str(path)


def _service(*lines: str) -> str:
    return "[Unit]\nDescription=llama %n\n\n[Service]\n" + "\n".join(lines) + "\n\n[Install]\nWantedBy=default.target\n"


@pytest.fixture
def home(tmp_path, monkeypatch) -> str:
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    return str(h)


def test_read_unit_joins_continuations_expands_and_collects_environment(tmp_path, home):
    path = _unit_file(tmp_path, _service(
        "WorkingDirectory=%h/srv",
        'Environment="LLAMA_ARG_N_GPU_LAYERS=99" LLAMA_ARG_FLASH_ATTN=on',
        "ExecStart=/opt/llama/llama-server \\",
        "  -m %h/models/a.gguf \\",
        "  --port 8080"))
    rec = units.read_unit(path, unit="llm.service")
    assert rec == ProcRecord(
        pid=0, uid=os.getuid(),
        argv=("/opt/llama/llama-server", "-m", f"{home}/models/a.gguf", "--port", "8080"),
        exe=None, cwd=f"{home}/srv",
        environ={"LLAMA_ARG_N_GPU_LAYERS": "99", "LLAMA_ARG_FLASH_ATTN": "on"},
        start_time=0, unit="llm.service", mapped_ggufs=())


def test_read_unit_defaults_the_working_directory_to_home(tmp_path, home):
    path = _unit_file(tmp_path, _service("ExecStart=/b/llama-server -m /m/a.gguf"))
    rec = units.read_unit(path, unit="u.service")
    assert isinstance(rec, ProcRecord) and rec.cwd == home and dict(rec.environ) == {}


def test_read_unit_a_dash_working_directory_drops_the_dash(tmp_path, home):
    path = _unit_file(tmp_path, _service("WorkingDirectory=-/srv/x", "ExecStart=/b/llama-server"))
    assert units.read_unit(path, unit="u.service").cwd == "/srv/x"


def test_read_unit_at_prefix_shifts_argv0(tmp_path, home):
    path = _unit_file(tmp_path, _service("ExecStart=-@/opt/llama/llama-server llama-server -m /m/a.gguf"))
    rec = units.read_unit(path, unit="u.service")
    assert rec.exe == "/opt/llama/llama-server"
    assert rec.argv == ("llama-server", "-m", "/m/a.gguf")


def test_read_unit_a_pipe_in_the_prefix_run_is_a_shell(tmp_path, home):
    path = _unit_file(tmp_path, _service("ExecStart=|/opt/llama/llama-server -m /m/a.gguf"))
    assert units.read_unit(path, unit="u.service") == units.Skip("runs through a shell")


def test_read_unit_percent_specifiers(tmp_path, home):
    import pwd
    path = _unit_file(tmp_path, _service("ExecStart=/b/llama-server --alias 100%% --api-key-file /k/%u"))
    rec = units.read_unit(path, unit="u.service")
    assert rec.argv[2] == "100%"
    assert rec.argv[-1] == "/k/" + pwd.getpwuid(os.getuid()).pw_name
    bad = units.read_unit(_unit_file(tmp_path, _service("ExecStart=/b/llama-server --alias %n"),
                                     "bad.service"), unit="bad.service")
    assert isinstance(bad, units.Skip) and "%n" in bad.reason
    # a specifier in a directive nobody reads does not skip
    ok = _unit_file(tmp_path, _service("ExecStart=/b/llama-server"), "ok.service")
    assert isinstance(units.read_unit(ok, unit="ok.service"), ProcRecord)


def test_read_unit_a_specifier_in_the_environment_or_working_directory_skips(tmp_path, home):
    for lines in (("Environment=X=%t", "ExecStart=/b/llama-server"),
                  ("WorkingDirectory=%t", "ExecStart=/b/llama-server")):
        skip = units.read_unit(_unit_file(tmp_path, _service(*lines)), unit="u.service")
        assert isinstance(skip, units.Skip) and "%t" in skip.reason


@pytest.mark.parametrize("lines, fragment", [
    (("ExecStart=/b/llama-server -m $HOME/a.gguf",), "$"),
    (("Environment=A=$B", "ExecStart=/b/llama-server"), "$"),
    (("EnvironmentFile=/etc/llama.env", "ExecStart=/b/llama-server -m /m/a.gguf"), "EnvironmentFile"),
    (("ExecStart=/b/llama-server -m /m/a.gguf", "ExecStart=/b/llama-server -m /m/b.gguf"), "ExecStart"),
    (("Description=x",), "ExecStart"),
    (("ExecStart=bin/llama-server -m /m/a.gguf",), "relative"),
    (("ExecStart=/b/llama-server -m '/m/a.gguf",), "unparseable"),
    (("ExecStart=/b/llama-server --alias a\\x20b",), "backslash"),
    (("ExecStart=/b/llama-server", "WorkingDirectory=srv"), "WorkingDirectory"),
])
def test_read_unit_skips_with_a_named_reason(tmp_path, home, lines, fragment):
    skip = units.read_unit(_unit_file(tmp_path, _service(*lines)), unit="u.service")
    assert isinstance(skip, units.Skip) and fragment in skip.reason, skip


def test_read_unit_an_empty_assignment_resets_exec_start(tmp_path, home):
    path = _unit_file(tmp_path, _service("ExecStart=/b/other -x", "ExecStart=",
                                         "ExecStart=/b/llama-server -m /m/a.gguf"))
    rec = units.read_unit(path, unit="u.service")
    assert isinstance(rec, ProcRecord) and rec.argv[0] == "/b/llama-server"


def test_read_unit_only_the_service_section_is_read(tmp_path, home):
    path = _unit_file(tmp_path, "[Unit]\nExecStart=/b/other\n[Service]\nExecStart=/b/llama-server\n"
                                "# ExecStart=/b/commented\n; ExecStart=/b/semi\n[Install]\nExecStart=/x\n")
    assert isinstance(units.read_unit(path, unit="u.service"), ProcRecord)


def test_read_unit_env_wrapper_names_the_model(tmp_path, home):
    path = _unit_file(tmp_path, _service("ExecStart=/usr/bin/env llama-server -m /m/a.gguf"))
    assert units.read_unit(path, unit="u.service") == units.Skip("runs env", names_model=True)


def test_read_unit_a_script_without_a_model_token_names_none(tmp_path, home):
    path = _unit_file(tmp_path, _service("ExecStart=/home/u/bin/run-active.sh"))
    assert units.read_unit(path, unit="u.service") == units.Skip("runs run-active.sh",
                                                                 names_model=False)


def test_read_unit_an_unrelated_dollar_unit_reads_runs_program(tmp_path, home):
    path = _unit_file(tmp_path, _service('ExecStart=/usr/bin/sleep $DELAY'))
    assert units.read_unit(path, unit="u.service") == units.Skip("runs sleep", names_model=False)


def test_read_unit_caps(tmp_path, home):
    long_line = "ExecStart=/b/llama-server " + "-x " * (units.UNIT_LINE_CAP // 3 + 10)
    skip = units.read_unit(_unit_file(tmp_path, _service(long_line)), unit="u.service")
    assert isinstance(skip, units.Skip) and str(units.UNIT_LINE_CAP) in skip.reason
    many = "ExecStart=/b/llama-server " + " ".join(["a"] * (units.UNIT_TOKEN_CAP + 1))
    skip = units.read_unit(_unit_file(tmp_path, _service(many), "tok.service"), unit="u.service")
    assert isinstance(skip, units.Skip) and str(units.UNIT_TOKEN_CAP) in skip.reason
    big = _service("ExecStart=/b/llama-server") + "# " + "x" * units.UNIT_FILE_CAP
    skip = units.read_unit(_unit_file(tmp_path, big, "big.service"), unit="u.service")
    assert isinstance(skip, units.Skip) and str(units.UNIT_FILE_CAP) in skip.reason


def test_read_unit_a_continuation_chain_over_the_line_cap_skips(tmp_path, home):
    body = "ExecStart=/b/llama-server \\\n" + "".join("  -x " + "y" * 100 + " \\\n" for _ in range(200)) + "  -z"
    skip = units.read_unit(_unit_file(tmp_path, _service(body)), unit="u.service")
    assert isinstance(skip, units.Skip) and str(units.UNIT_LINE_CAP) in skip.reason


def test_read_unit_a_fifo_is_a_skip_and_never_blocks(tmp_path, home):
    fifo = tmp_path / "pipe.service"
    os.mkfifo(fifo)
    assert units.read_unit(str(fifo), unit="u.service") == units.Skip("has an unreadable unit file",
                                                                       names_model=False)


def test_read_unit_missing_and_non_utf8_files(tmp_path, home):
    assert isinstance(units.read_unit(str(tmp_path / "nope.service"), unit="u"), units.Skip)
    bad = tmp_path / "bad.service"
    bad.write_bytes(b"[Service]\nExecStart=/b/llama-server \xff\n")
    assert isinstance(units.read_unit(str(bad), unit="u"), units.Skip)


def test_read_unit_never_opens_through_a_plain_open(tmp_path, home, monkeypatch):
    from localharness.provider import units as mod
    seen: list[str] = []
    real = mod.open_regular
    monkeypatch.setattr(mod, "open_regular", lambda p: seen.append(str(p)) or real(p))
    path = _unit_file(tmp_path, _service("ExecStart=/b/llama-server"))
    units.read_unit(path, unit="u.service")
    assert seen == [path]


# -------------------------------------------------------------------------------- unit_record


def _state(unit="llm.service", fragment="", drop_ins="") -> units.UnitState:
    return units.UnitState(unit, "loaded", fragment, "disabled", drop_ins)


def test_unit_record_prefers_the_journal(fake_systemd, tmp_path, home):
    model = _model(tmp_path)
    fake_systemd.journal([{"_EXE": "/b/llama-server", "_CMDLINE": f"llama-server -m {model}"}])
    frag = _unit_file(tmp_path, _service("ExecStart=/b/llama-server -m /m/other.gguf"))
    source, rec = units.unit_record(_state(fragment=frag))
    assert source == "journal" and rec.argv[2] == model


def test_unit_record_falls_back_to_the_unit_file(fake_systemd, tmp_path, home):
    frag = _unit_file(tmp_path, _service("ExecStart=/b/llama-server -m /m/other.gguf"))
    source, rec = units.unit_record(_state(fragment=frag))
    assert source == "unit" and rec.unit == "llm.service" and rec.argv[2] == "/m/other.gguf"


def test_unit_record_drop_ins_skip_the_unit_file(fake_systemd, tmp_path, home):
    frag = _unit_file(tmp_path, _service("ExecStart=/b/llama-server -m /m/other.gguf"))
    skip = units.unit_record(_state(fragment=frag, drop_ins="/h/llm.service.d/o.conf"))
    assert isinstance(skip, units.Skip)
    assert skip.reason == "has drop-ins (/h/llm.service.d/o.conf)" and skip.names_model


def test_unit_record_drop_ins_on_an_unrelated_unit_name_no_model(fake_systemd, tmp_path, home):
    frag = _unit_file(tmp_path, _service("ExecStart=/usr/bin/pipewire"))
    skip = units.unit_record(_state("pw.service", fragment=frag, drop_ins="/h/pw.service.d/o.conf"))
    assert isinstance(skip, units.Skip) and not skip.names_model


def test_unit_record_no_unit_file(fake_systemd):
    assert units.unit_record(_state()) == units.Skip("has no unit file", names_model=False)


def test_unit_record_the_unit_file_skip_when_the_journal_names_nothing(fake_systemd, tmp_path, home):
    frag = _unit_file(tmp_path, _service("ExecStart=/home/u/bin/run-active.sh"))
    assert units.unit_record(_state(fragment=frag)) == units.Skip("runs run-active.sh",
                                                                  names_model=False)


def test_unit_record_the_journal_skip_wins_when_it_names_a_model(fake_systemd, tmp_path, home):
    fake_systemd.journal([{"_EXE": "/b/llama-server", "_CMDLINE": "llama-server -m rel/a.gguf"}])
    frag = _unit_file(tmp_path, _service("ExecStart=/home/u/bin/run-active.sh"))
    skip = units.unit_record(_state(fragment=frag))
    assert isinstance(skip, units.Skip) and "relative path (rel/a.gguf)" in skip.reason


def test_a_journal_record_takes_the_environment_and_cwd_of_a_unit_file_naming_the_same_model(
        fake_systemd, tmp_path, home):
    """WR-04: the journal knows neither; the unit file does."""
    model = _model(tmp_path)
    fake_systemd.journal([{"_EXE": "/b/llama-server", "_CMDLINE": f"llama-server -m {model} -ngl 9"}])
    frag = _unit_file(tmp_path, _service(
        "WorkingDirectory=/srv/llama", "Environment=LLAMA_ARG_PORT=8090 LLAMA_API_KEY=s3cret",
        f"ExecStart=/b/llama-server -m {model} -ngl 9"))
    source, rec = units.unit_record(_state(fragment=frag))
    assert source == "journal"
    assert dict(rec.environ) == {"LLAMA_ARG_PORT": "8090", "LLAMA_API_KEY": "s3cret"}
    assert rec.cwd == "/srv/llama" and rec.exe == "/b/llama-server"
    assert units.journal_environment_gap(_state(fragment=frag), rec) is None


def test_a_unit_file_that_spells_the_same_words_gives_the_exactly_quoted_tokens(fake_systemd, tmp_path, home):
    """WR-02: `--alias "my model"` reads as three words in the journal and as two in the unit file."""
    model = _model(tmp_path)
    fake_systemd.journal([{"_EXE": "/b/llama-server", "_CMDLINE": f"llama-server -m {model} --alias my model"}])
    frag = _unit_file(tmp_path, _service(f'ExecStart=/b/llama-server -m {model} --alias "my model"'))
    _, rec = units.unit_record(_state(fragment=frag))
    assert rec.argv[-2:] == ("--alias", "my model")


def test_a_unit_file_that_names_another_model_leaves_the_journal_record_alone(fake_systemd, tmp_path, home):
    model = _model(tmp_path)
    fake_systemd.journal([{"_EXE": "/b/llama-server", "_CMDLINE": f"llama-server -m {model}"}])
    frag = _unit_file(tmp_path, _service("Environment=LLAMA_ARG_PORT=1", "ExecStart=/b/llama-server -m /m/other.gguf"))
    _, rec = units.unit_record(_state(fragment=frag))
    assert rec.environ is None and rec.cwd is None
    assert units.journal_environment_gap(_state(fragment=frag), rec) == "names another model"


def test_the_environment_a_unit_file_hides_is_named_not_dropped_silently(fake_systemd, tmp_path, home):
    model = _model(tmp_path)
    fake_systemd.journal([{"_EXE": "/b/llama-server", "_CMDLINE": f"llama-server -m {model}"}])
    frag = _unit_file(tmp_path, _service("EnvironmentFile=/etc/llama.env", f"ExecStart=/b/llama-server -m {model}"))
    _, rec = units.unit_record(_state(fragment=frag))
    assert rec.environ is None
    assert units.journal_environment_gap(_state(fragment=frag), rec) == "uses EnvironmentFile="
    drop = _state(fragment=frag, drop_ins="/h/llm.service.d/o.conf")
    assert units.journal_environment_gap(drop, rec) == "has drop-ins (/h/llm.service.d/o.conf)"
    assert units.journal_environment_gap(_state(), rec) is None  # no unit file: nothing to say


# ------------------------------------------------------------------------------- user_units


LIST_ARGV = ("systemctl", "--user", "list-unit-files", "--type=service", "--plain", "--no-legend",
             "--no-pager")


def _block(unit: str, state: str = "loaded", frag: str = "", enabled: str = "disabled",
           drop: str = "") -> str:
    return (f"Id={unit}\nLoadState={state}\nFragmentPath={frag}\nUnitFileState={enabled}\n"
            f"DropInPaths={drop}\n")


def test_user_units_filters_sorts_and_reads_in_one_batched_show(fake_systemd):
    fake_systemd.answer(LIST_ARGV, "\n".join([
        "zz.service enabled enabled", "a@.service static -", "gen.service generated -",
        "t.service transient -", "al.service alias -", "m.service masked enabled",
        "b.service bad -", "llm.service disabled enabled", "pipewire.service static -", ""]))
    names = ("llm.service", "pipewire.service", "zz.service")
    fake_systemd.answer(["systemctl", "--user", "show", "-p", SHOW, "--", *names],
                        "\n".join(_block(n, frag=f"/u/{n}") for n in (names[2], names[0], names[1])))
    states = units.user_units()
    assert [s.unit for s in states] == list(names)
    assert states[0] == units.UnitState("llm.service", "loaded", "/u/llm.service", "disabled", "")
    assert fake_systemd.calls == [LIST_ARGV, ("systemctl", "--user", "show", "-p", SHOW, "--", *names)]


def test_user_units_scans_at_most_the_cap_alphabetically(fake_systemd):
    assert units.UNITS_SCANNED_CAP == 64
    all_names = [f"u{i:03d}.service" for i in range(70)]
    fake_systemd.answer(LIST_ARGV, "".join(f"{n} disabled -\n" for n in reversed(all_names)))
    fake_systemd.answer_prefix("systemctl", "--user", "show", stdout="")
    units.user_units()
    show = fake_systemd.calls[1]
    assert list(show[show.index("--") + 1:]) == all_names[:64]


def test_user_units_is_empty_when_not_checked(fake_systemd):
    assert units.user_units() == []  # list-unit-files answered None
    fake_systemd.answer(LIST_ARGV, "llm.service disabled -\n")
    assert units.user_units() == []  # show answered None
    assert len(fake_systemd.calls_of("systemctl")) == 3


def test_user_units_with_nothing_left_after_filtering_runs_no_show(fake_systemd):
    fake_systemd.answer(LIST_ARGV, "a@.service static -\nm.service masked -\n")
    assert units.user_units() == []
    assert fake_systemd.calls == [LIST_ARGV]


def test_user_units_names_a_leading_dash_only_after_the_separator(fake_systemd):
    fake_systemd.answer(LIST_ARGV, "-evil.service disabled -\n")
    fake_systemd.answer_prefix("systemctl", "--user", "show", stdout="")
    units.user_units()
    show = fake_systemd.calls[1]
    assert show.index("--") < show.index("-evil.service")


def test_the_list_command_is_list_unit_files_never_list_units():
    src = Path(units.__file__).read_text(encoding="utf-8")
    assert '"list-unit-files"' in src and '"list-units"' not in src


# ------------------------------------------------------------------------------ the real _run


class _Done:
    def __init__(self, stdout: bytes, returncode: int = 0) -> None:
        self.stdout, self.returncode = stdout, returncode


@pytest.fixture
def real_run(monkeypatch):
    """The real `_run` over a fake `which_outside_cwd` and `subprocess.run`."""
    calls: list[tuple[list[str], dict]] = []
    state: dict = {"which": "/usr/bin/journalctl", "outcome": _Done(b"ok\n")}

    def which(cmd):
        calls.append(([cmd], {"which": True}))
        return state["which"]

    def run(argv, **kw):
        calls.append((argv, kw))
        outcome = state["outcome"]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(units, "which_outside_cwd", which)
    monkeypatch.setattr(units.subprocess, "run", run)
    return state, calls


def test_run_resolves_the_program_and_closes_stdin(real_run):
    _state, calls = real_run
    assert units._run(["journalctl", "--user", "-n", "1"], timeout=3, cap=100) == "ok\n"
    argv, kw = calls[-1]
    assert argv == ["/usr/bin/journalctl", "--user", "-n", "1"]
    assert kw["stdin"] == subprocess.DEVNULL and kw["timeout"] == 3
    assert kw["capture_output"] is True and not kw.get("shell")


def test_run_without_a_resolved_path_runs_no_subprocess(real_run):
    state, calls = real_run
    state["which"] = None
    assert units._run(["journalctl", "x"], timeout=1, cap=10) is None
    assert all(kw.get("which") for _argv, kw in calls)  # only the lookup happened


@pytest.mark.parametrize("outcome", [subprocess.TimeoutExpired("j", 1), FileNotFoundError("gone"),
                                     PermissionError("no"), _Done(b"x", 1), _Done(b"", 127)])
def test_run_refuses_every_failure_shape(real_run, outcome):
    state, _calls = real_run
    state["outcome"] = outcome
    assert units._run(["journalctl"], timeout=1, cap=10) is None


def test_run_enforces_the_output_cap(real_run):
    state, _calls = real_run
    state["outcome"] = _Done(b"x" * 11)
    assert units._run(["journalctl"], timeout=1, cap=10) is None
    state["outcome"] = _Done(b"x" * 10)
    assert units._run(["journalctl"], timeout=1, cap=10) == "x" * 10


def test_run_decodes_with_replacement(real_run):
    state, _calls = real_run
    state["outcome"] = _Done(b"a\xffb")
    assert units._run(["journalctl"], timeout=1, cap=10) == "a�b"


# ------------------------------------------------------------------ the journal's hostile shapes


def test_journal_entries_with_missing_or_foreign_fields_name_no_run(fake_systemd, tmp_path):
    fake_systemd.journal([
        {"_CMDLINE": "llama-server -m /m/a.gguf"},
        {"_EXE": ["/b/llama-server"], "_CMDLINE": "llama-server -m /m/a.gguf"},
        {"_EXE": "/usr/bin/python3", "_CMDLINE": None},
        "this is not json", "[1, 2]", "",
    ])
    assert units.read_journal("u.service") == units.Skip("has no llama-server in its journal",
                                                         names_model=False)


@pytest.mark.parametrize("cmdline", [None, [108, 108]])
def test_a_newest_run_whose_command_line_the_journal_cannot_show_is_a_skip(fake_systemd, tmp_path, cmdline):
    model = _model(tmp_path)
    fake_systemd.journal([
        {"_EXE": "/b/llama-server", "_CMDLINE": f"llama-server -m {model} --host 0.0.0.0 --api-key-file /k"},
        {"_EXE": "/b/llama-server", "_CMDLINE": cmdline},
    ])
    assert units.read_journal("u.service") == units.Skip("has a command line the journal cannot show")


def test_an_unusable_newest_run_does_not_hand_over_an_older_one(fake_systemd, tmp_path):
    """WR-03: the owner moved the unit back to loopback; the older run bound every interface."""
    model = _model(tmp_path)
    fake_systemd.journal([
        {"_EXE": "/b/llama-server", "_CMDLINE": f"llama-server -m {model} --host 0.0.0.0 --api-key-file /k"},
        "not json",
        {"_EXE": "/b/llama-server", "_CMDLINE": f"llama-server -m {model} --alias it's --host 127.0.0.1"},
    ])
    assert units.read_journal("u.service") == units.Skip("has an argument the journal cannot show exactly")


@pytest.mark.parametrize("tail", ['--chat-template-kwargs {"enable_thinking":false}', "--alias 'a b'",
                                  "--override-kv a=str:C:\\x"])
def test_the_journal_does_not_quote_so_a_quote_or_backslash_is_not_read_as_shell(fake_systemd, tmp_path, tail):
    model = _model(tmp_path)
    fake_systemd.journal([{"_EXE": "/b/llama-server", "_CMDLINE": f"llama-server -m {model} {tail}"}])
    assert units.read_journal("u.service") == units.Skip("has an argument the journal cannot show exactly")


def test_a_non_llama_newer_entry_does_not_hide_the_llama_run(fake_systemd, tmp_path):
    model = _model(tmp_path)
    fake_systemd.journal([
        {"_EXE": "/b/llama-server", "_CMDLINE": f"llama-server -m {model}"},
        {"_EXE": "/usr/bin/python3", "_CMDLINE": None},
    ])
    assert isinstance(units.read_journal("u.service"), ProcRecord)


def test_a_non_llama_last_entry_does_not_replace_the_last_llama_one(fake_systemd, tmp_path):
    model = _model(tmp_path)
    fake_systemd.journal([
        {"_EXE": "/b/llama-server", "_CMDLINE": f"llama-server -m {model}"},
        {"_EXE": "/usr/bin/python3", "_CMDLINE": "python3 x.py"},
    ])
    assert isinstance(units.read_journal("u.service"), ProcRecord)
