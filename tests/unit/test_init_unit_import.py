"""INIT-09 / INIT-08 / INIT-02 (the unit path): nothing answers, and a stopped llama-server user unit
of the person's own is the primary preset (Phase 5, plan 14; D-07, D-21, D-23, D-24, D-29, D-50, D-64).

The unit is read from its journal (the last llama-server run), else its unit file, and imported no
wider than it ran: a non-loopback bind with no key file is imported on loopback and says so. Its
model leads the ranking, discovery's other models are the peers, and a unit that could not be used
is named in the one no-binary line. Everything outside the files is faked: `/proc` is empty,
systemd a `FakeSystemd`, detection an `AsyncMock` that finds nothing, the fit a stub.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import localharness.cli.init_cmd as init_cmd
from localharness.cli import library
from localharness.provider import speed_stats
from localharness.provider.detector import DetectorResult
from tests.unit.interview_support import scripted
from tests.unit.test_init_core_setup import (
    _exited,
    _hermetic,  # noqa: F401  (autouse fixture; real sockets are refused by conftest)
    _init,
    _terminal,
)
from tests.unit.test_init_live_import import (  # noqa: F401  (fixtures and helpers)
    block_of,
    lines_of,
    live,
)
from tests.unit.procfs_support import fake_proc  # noqa: F401  (fixture)
from tests.unit.units_support import fake_systemd  # noqa: F401  (fixture)

UNIT = "llm.service"
LOOPBACK_LINE = (f"{UNIT} binds 0.0.0.0 with no key file — imported on 127.0.0.1 · "
                 "server.bind_all: true beside server.api_key_file restores it (asks once)")
IMPORTED_LINE = (f"{UNIT} is yours and was imported as the server: record; from now on the harness "
                 "launches qwen-27b itself and never touches the unit.")
STOPPED_LINE = f"{UNIT} sets -c 65536; the first start launches -c 114688 · server.context: 65536 keeps it"


def _nothing_answers(live) -> None:  # noqa: F811
    live.detect.return_value = DetectorResult(found=False, provider_type="unknown", base_url="",
                                              models=[], suggested_model="", probe_duration_ms=0.0)


def _systemd(live, *states: tuple[str, str, str, str]) -> None:  # noqa: F811
    """`(unit, enabled, fragment path, drop-in paths)` per unit: what list-unit-files and the
    batched show answer."""
    live.fake_systemd.answer_prefix(
        "systemctl", "--user", "list-unit-files", stdout="".join(f"{u} {e}\n" for u, e, *_ in states))
    blocks = [f"Id={u}\nLoadState=loaded\nFragmentPath={frag}\nUnitFileState={e}\nDropInPaths={drops}\n"
              for u, e, frag, drops in states]
    live.fake_systemd.answer_prefix("systemctl", "--user", "show", stdout="\n".join(blocks))


def _journal(live, unit: str = UNIT, *, extra: str = "", model: Path | None = None,  # noqa: F811
             host: str | None = "0.0.0.0", exe: str | None = None) -> None:
    """The unit's journal holds one llama-server run: A.gguf under the alias qwen-27b on :8080."""
    host_flags = f" --host {host}" if host else ""
    cmdline = f"llama-server -m {model or live.model_a} --alias qwen-27b{host_flags} --port 8080{extra}"
    live.fake_systemd.journal([{"_EXE": exe or str(live.bin), "_CMDLINE": cmdline}], unit=unit)


def _stopped_unit(live, *, extra: str = " -c 65536 -ngl 99", host: str | None = "0.0.0.0") -> None:  # noqa: F811
    _nothing_answers(live)
    _systemd(live, (UNIT, "disabled", str(live.home / ".config/systemd/user" / UNIT), ""))
    _journal(live, extra=extra, host=host)


# ------------------------------------------------------------------ the tracer


def test_a_stopped_units_journal_becomes_the_primary_with_discoverys_other_model_as_a_peer(live):  # noqa: F811
    _stopped_unit(live)

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    cfg = live.written()
    server = cfg["server"]
    assert server["runtime"] == "llamacpp"
    assert server["binary"] == str(live.bin)
    assert server["model"] == str(live.model_a)
    assert server["served_name"] == "qwen-27b" and server["port"] == 8080
    assert server["extra_args"] == ["-ngl", "99"]
    assert server.get("bind_all", False) is False and "context" not in server
    assert None not in server.values()

    [peer] = live.peers()
    assert peer["name"] == "B" and peer["base_url"] == "http://127.0.0.1:8081/v1"
    assert peer["lifecycle"]["model"] == str(live.model_b)

    provider = cfg["provider"]
    assert provider["default_model"] == "qwen-27b"
    assert provider["base_url"] == "http://127.0.0.1:8080/v1"
    assert provider["available_models"] == ["qwen-27b"]


def test_a_stopped_unit_wins_silently_for_a_person_and_the_interview_is_never_asked(live):  # noqa: F811
    _stopped_unit(live)
    _terminal(live.monkeypatch)
    asked = scripted(live.monkeypatch)

    result = _init(live.cfg)

    _exited(result, 0)
    assert asked.asked == []
    assert live.written()["server"]["model"] == str(live.model_a)
    assert [p["lifecycle"]["model"] for p in live.peers()] == [str(live.model_b)]


def test_the_block_reads_the_primary_its_flags_the_loopback_line_the_stopped_line_then_the_peer(live):  # noqa: F811
    _stopped_unit(live)

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert block_of(result) == [
        f"serving qwen-27b (llama.cpp) ← ~/models/gguf/A.gguf · binary: {live.bin} (from {UNIT})",
        "flags: -ngl 99",
        LOOPBACK_LINE,
        IMPORTED_LINE,
        STOPPED_LINE,
        "/model switches to B (llama.cpp) ← ~/models/gguf/B.gguf",
    ]
    assert not any("holds :" in line for line in lines_of(result))  # the unit is stopped: nothing to attach


def test_an_api_key_file_on_the_journals_argv_keeps_the_bind_and_the_key_file(live):  # noqa: F811
    _stopped_unit(live, extra=" -c 65536 -ngl 99 --api-key-file /k")

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    server = live.written()["server"]
    assert server["bind_all"] is True and server["api_key_file"] == "/k"
    assert server["require_api_key"] is True
    assert LOOPBACK_LINE not in lines_of(result)


def test_a_unit_that_binds_loopback_is_imported_as_it_ran(live):  # noqa: F811
    _stopped_unit(live, host="127.0.0.1")

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert LOOPBACK_LINE not in lines_of(result)
    assert live.written()["server"].get("bind_all", False) is False


def test_the_units_model_is_the_primary_though_another_is_newer_and_the_ledger_names_it(live):  # noqa: F811
    _stopped_unit(live)
    os.utime(live.model_a, (1_000_000, 1_000_000))
    os.utime(live.model_b, (2_000_000, 2_000_000))
    live.cfg.mkdir(parents=True, exist_ok=True)
    speed_stats.record_model_path(live.ledger(), "llamacpp", "B", str(live.model_b))

    ranked = library.import_peer_models(live.cfg, exclude="/nothing", unit_model=os.path.realpath(live.model_a))
    assert [m.path.name for m in ranked] == ["A.gguf", "B.gguf"]
    assert [m.path.name for m in library.import_peer_models(live.cfg, exclude="/nothing", unit_model=None)] == [
        "B.gguf", "A.gguf"]

    _exited(_init(live.cfg, "--no-input"), 0)

    assert live.written()["server"]["model"] == str(live.model_a)


def test_the_units_model_reaches_the_ranking_through_the_signature(live, monkeypatch):  # noqa: F811
    _stopped_unit(live)
    asked: list[object] = []
    real = library.import_peer_models

    def spy(config_path, *, exclude, unit_model):
        asked.append(unit_model)
        return real(config_path, exclude=exclude, unit_model=unit_model)

    monkeypatch.setattr(library, "import_peer_models", spy)

    _exited(_init(live.cfg, "--no-input"), 0)

    assert asked == [os.path.realpath(live.model_a)]


def test_the_live_imports_peer_lookup_still_ranks_without_a_unit_model(live, monkeypatch):  # noqa: F811
    asked: list[object] = []
    real = library.import_peer_models

    def spy(config_path, *, exclude, unit_model):
        asked.append(unit_model)
        return real(config_path, exclude=exclude, unit_model=unit_model)

    monkeypatch.setattr(library, "import_peer_models", spy)
    live.detect.return_value = DetectorResult(
        found=True, provider_type="llamacpp", base_url="http://127.0.0.1:8080/v1", models=["qwen-27b"],
        suggested_model="qwen-27b", probe_duration_ms=1.0)
    live.serve("-c", "65536")

    _exited(_init(live.cfg, "--no-input"), 0)

    assert asked == [None]


def test_the_weights_path_and_the_window_triple_are_recorded_under_the_served_name(live):  # noqa: F811
    _stopped_unit(live)

    _exited(_init(live.cfg, "--no-input"), 0)

    assert speed_stats.model_path(live.ledger(), "llamacpp", "qwen-27b") == str(live.model_a)
    report = speed_stats.last_window_report(live.ledger(), "llamacpp", "qwen-27b")
    assert (report.served, report.trained, report.fits) == (65536, 262144, 114688)


def test_without_a_dash_c_nothing_is_printed_about_the_window(live):  # noqa: F811
    _stopped_unit(live, extra=" -ngl 99")

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert not any(" sets -c " in line for line in lines_of(result))
    assert speed_stats.last_window_report(live.ledger(), "llamacpp", "qwen-27b") is None
    assert "context" not in live.written()["server"]


def test_the_unit_path_asks_probes_and_launches_nothing(live):  # noqa: F811
    _stopped_unit(live)

    _exited(_init(live.cfg, "--no-input"), 0)  # Confirm / Prompt / _probe_capabilities / Popen raise


@pytest.mark.parametrize("flags", [(), ("--no-input",)], ids=["no-flag", "no-input"])
def test_a_machine_with_no_unit_keeps_the_discovery_path(flags, live):  # noqa: F811
    _nothing_answers(live)  # list-unit-files answers nothing: "not checked"

    result = _init(live.cfg, *flags)

    _exited(result, 0)
    assert live.written()["server"]["port"] == 8081


# ------------------------------------------------------------------ the unit file, and the skip reasons


def _unit_file(live, *lines: str, name: str = UNIT) -> Path:  # noqa: F811
    path = live.home / ".config" / "systemd" / "user" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[Unit]\nDescription=llama\n\n[Service]\n" + "\n".join(lines)
                    + "\n\n[Install]\nWantedBy=default.target\n", encoding="utf-8")
    return path


def _third_model(live) -> None:  # noqa: F811
    from tests.unit.test_init_live_import import gguf

    gguf(live.home / "models" / "gguf" / "C.gguf")


def test_a_journal_record_takes_the_unit_files_environment_when_both_name_the_same_model(live):  # noqa: F811
    """WR-04: the journal knows no Environment=; the unit file beside it does."""
    _nothing_answers(live)
    path = _unit_file(live, "Environment=LLAMA_ARG_N_PARALLEL=2",
                      f"ExecStart={live.bin} -m {live.model_a} --alias qwen-27b --host 0.0.0.0 --port 8080")
    _systemd(live, (UNIT, "disabled", str(path), ""))
    _journal(live)

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert live.written()["server"]["env"] == {"LLAMA_ARG_N_PARALLEL": "2"}
    assert "env: LLAMA_ARG_N_PARALLEL=2" in block_of(result)
    assert not any("not read" in line for line in lines_of(result))


def test_an_environment_a_unit_file_keeps_out_of_reach_is_named_in_the_block(live):  # noqa: F811
    _nothing_answers(live)
    path = _unit_file(live, "EnvironmentFile=/etc/llama.env",
                      f"ExecStart={live.bin} -m {live.model_a} --alias qwen-27b --host 0.0.0.0 --port 8080")
    _systemd(live, (UNIT, "disabled", str(path), ""))
    _journal(live)

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert f"environment of {UNIT} not read (uses EnvironmentFile=)" in block_of(result)
    assert live.written()["server"]["port"] == 8080


def test_the_unit_file_answers_when_the_journal_does_not_and_a_key_in_it_is_a_key(live):  # noqa: F811
    _nothing_answers(live)
    path = _unit_file(live, "WorkingDirectory=%h", "Environment=LLAMA_API_KEY=K LLAMA_ARG_N_GPU_LAYERS=99",
                      "ExecStart=/opt/llama/llama-server \\", "  -m %h/models/gguf/A.gguf --host 0.0.0.0")
    _systemd(live, (UNIT, "disabled", str(path), ""))  # no journal answer: "has an unreadable journal"

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    cfg = live.written()
    server = cfg["server"]
    assert server["binary"] == "/opt/llama/llama-server" and server["model"] == str(live.model_a)
    assert server["bind_all"] is True and server["require_api_key"] is True
    assert server["env"] == {"LLAMA_ARG_N_GPU_LAYERS": "99"}
    assert server["served_name"] == "A" and server["port"] == 8080
    assert cfg["provider"]["api_key"] == "K"
    assert block_of(result)[0] == (
        f"serving A (llama.cpp) ← ~/models/gguf/A.gguf · binary: /opt/llama/llama-server (from {UNIT})")
    assert LOOPBACK_LINE not in lines_of(result)  # the key is the evidence for the bind
    assert "K" not in [tok for line in lines_of(result) for tok in line.split()]
    assert "LLAMA_API_KEY" not in result.output


def test_a_unit_file_with_no_key_evidence_is_imported_on_loopback(live):  # noqa: F811
    _nothing_answers(live)
    path = _unit_file(live, "ExecStart=/opt/llama/llama-server -m %h/models/gguf/A.gguf --host 0.0.0.0")
    _systemd(live, (UNIT, "disabled", str(path), ""))

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert live.written()["server"].get("bind_all", False) is False
    assert LOOPBACK_LINE in lines_of(result)


def _wrapper(live, *, name: str = UNIT, program: str = "run-active.sh") -> None:  # noqa: F811
    _nothing_answers(live)
    path = _unit_file(live, f"ExecStart=/home/u/bin/{program} -m /m/a.gguf", name=name)
    _systemd(live, (name, "enabled", str(path), ""))
    _third_model(live)


def test_a_wrapper_unit_with_no_llama_server_on_path_is_named_in_the_one_line(live):  # noqa: F811
    _wrapper(live)
    live.bin.unlink()

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    skipped = f"{UNIT} runs run-active.sh; no llama-server found — set server.binary"
    assert lines_of(result).count(skipped) == 1
    assert result.output.count("no llama-server") == 1  # never also the GGUF count line
    assert "found 3 GGUFs" not in result.output
    cfg = live.written()
    assert "binary" not in cfg["server"] and cfg["server"]["port"] == 8081  # discovery's records


def test_the_skipped_unit_leaves_the_first_start_not_ready(live):  # noqa: F811
    _wrapper(live)
    live.bin.unlink()

    out = init_cmd.core_setup(str(live.cfg), endpoint=None, model=None, force=False, interactive=False)

    assert out == init_cmd.SetupResult(live.cfg / "config.yaml", False)


def test_a_wrapper_unit_with_llama_server_on_path_says_nothing_about_the_unit(live):  # noqa: F811
    _wrapper(live)

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert "run-active.sh" not in result.output and "no llama-server" not in result.output
    assert live.written()["server"]["binary"] == str(live.bin)


def test_a_unit_with_an_environment_file_gives_that_reason(live):  # noqa: F811
    _nothing_answers(live)
    path = _unit_file(live, "EnvironmentFile=/etc/llama.env",
                      "ExecStart=/opt/llama/llama-server -m %h/models/gguf/A.gguf")
    _systemd(live, (UNIT, "enabled", str(path), ""))
    live.bin.unlink()

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert lines_of(result).count(
        f"{UNIT} uses EnvironmentFile=; no llama-server found — set server.binary") == 1


def test_a_unit_unrelated_to_llama_server_is_never_named(live):  # noqa: F811
    _nothing_answers(live)
    path = _unit_file(live, "ExecStart=/usr/bin/pipewire", name="pipewire.service")
    _systemd(live, ("pipewire.service", "enabled", str(path), ""))
    _third_model(live)
    live.bin.unlink()

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert "pipewire" not in result.output
    assert lines_of(result).count("found 3 GGUFs under ~/models; no llama-server — set server.binary") == 1


def test_a_second_units_skip_prints_nothing_when_the_journal_supplied_the_record(live):  # noqa: F811
    _nothing_answers(live)
    wrapper = _unit_file(live, "ExecStart=/home/u/bin/run-active.sh -m /m/a.gguf", name="a-wrap.service")
    _systemd(live, ("a-wrap.service", "enabled", str(wrapper), ""),
             (UNIT, "disabled", str(live.home / ".config/systemd/user" / UNIT), ""))
    _journal(live)  # llm.service's journal answers; a-wrap.service's does not

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert "run-active.sh" not in result.output and "no llama-server" not in result.output
    assert live.written()["server"]["served_name"] == "qwen-27b"


def test_a_unit_with_drop_ins_is_skipped_and_named(live):  # noqa: F811
    _nothing_answers(live)
    path = _unit_file(live, "ExecStart=/opt/llama/llama-server -m %h/models/gguf/A.gguf")
    _systemd(live, (UNIT, "enabled", str(path), "/home/u/.config/systemd/user/llm.service.d/x.conf"))
    live.bin.unlink()

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    [line] = [ln for ln in lines_of(result) if "no llama-server found" in ln]
    assert line.startswith(f"{UNIT} has drop-ins (")


def test_no_unit_and_no_gguf_still_reaches_the_not_detected_path(live):  # noqa: F811
    _nothing_answers(live)
    live.model_a.unlink()
    live.model_b.unlink()

    result = _init(live.cfg, "--no-input")

    assert result.exit_code == 1, result.output
    assert "No local LLM detected" in result.output


def test_a_hostile_unit_name_is_shown_escaped(live):  # noqa: F811
    assert init_cmd._unit_skip_line("a\x1b[2Jb.service", "runs x\ny") == (
        "a\\x1b[2Jb.service runs x\\x0ay; no llama-server found — set server.binary")
