"""INIT-04 / INIT-05 / INIT-06 / INIT-07 / INIT-08 / INIT-12 / INIT-14: a fresh `init` finds a
llama-server already answering on the detected port and imports its process (Phase 5, plan 12).

The listener is the trigger (D-26): `procfs.listeners` -> `pid_for_inode` -> `read_process` ->
`init_import.import_record`, before the model list is judged (a keyed server answers 401 with none).
The process becomes `server:` with discovery's other GGUFs as peers; the block says every value it
took, then one line about who holds the port (D-29, D-32). Everything outside the process is faked:
`/proc` is a `FakeProc` tree, systemd a `FakeSystemd`, detection an `AsyncMock`, the fit a stub; no
real server is started and no capability probe runs.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

import localharness.cli.init_cmd as init_cmd
from localharness.cli import init_import
from localharness.cli.workspace import decide_machine_trust
from localharness.config import trust
from localharness.provider import autosense, speed_stats
from localharness.provider.detector import DetectorResult
from tests.unit.gguf_support import write_gguf
from tests.unit.procfs_support import fake_proc  # noqa: F401  (fixture)
from tests.unit.test_init_core_setup import (
    _Asked,
    _exited,
    _hermetic,  # noqa: F401  (autouse fixture: no real HTTP)
    _init,
    _terminal,
)
from tests.unit.test_init_cmd import _make_capability_result
from tests.unit.launch_support import record_opens
from tests.unit.test_init_adoption import _expected
from tests.unit.trust_support import answers, never_asked
from tests.unit.units_support import fake_systemd  # noqa: F401  (fixture)

UNIT = "llama-server.service"
INODE = 5551
PID = 1234
URL = "http://127.0.0.1:8080/v1"


def gguf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_gguf(path, {"general.architecture": "llama", "general.type": "model",
                      "llama.embedding_length": 5120, "llama.context_length": 32768})
    return path


def cgroup(unit: str = UNIT) -> str:
    return f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{unit}"


def show(unit: str = UNIT, enabled: str = "disabled", load: str = "loaded") -> str:
    return (f"Id={unit}\nLoadState={load}\nFragmentPath=/home/u/.config/systemd/user/{unit}\n"
            f"UnitFileState={enabled}\nDropInPaths=\n")


def detected(*, models=("qwen-27b",), raw: dict | None = None, url: str = URL,
             provider_type: str = "llamacpp") -> DetectorResult:
    return DetectorResult(found=True, provider_type=provider_type, base_url=url, models=list(models),
                          suggested_model=models[0] if models else "", probe_duration_ms=1.0,
                          raw_response=raw or {})


KEYED_401 = {"error": {"message": "Invalid API Key", "type": "authentication_error", "code": 401}}


def _boom(name: str):
    def _raise(*_a, **_k):
        raise AssertionError(f"{name} called: an import asks, probes and launches nothing")
    return _raise


class Live:
    """The box: a llama-server process on :8080 as a user unit, two GGUFs on disk, llama-server on
    PATH, every question, probe and spawn booby-trapped."""

    def __init__(self, tmp_path: Path, home: Path, proc, bin_dir: Path, monkeypatch) -> None:
        self.tmp_path = tmp_path
        self.home = home
        self.fake_proc = proc
        self.monkeypatch = monkeypatch
        self.bin = bin_dir / "llama-server"
        self.model_a = home / "models" / "gguf" / "A.gguf"
        self.model_b = home / "models" / "gguf" / "B.gguf"
        self.cfg = tmp_path / "cfg"
        self.detect = AsyncMock(return_value=detected())

    def serve(self, *extra: str, model: Path | None = None, environ: dict | None = None,
              unit: str | None = UNIT, base: tuple[str, ...] | None = None, exe: str | None = None,
              **process_kw) -> None:
        """Replace the listener: argv is `llama-server -m A --alias qwen-27b --host 127.0.0.1 --port
        8080` (unless `base` says otherwise) plus `extra`."""
        argv = base if base is not None else (
            "llama-server", "-m", str(model or self.model_a), "--alias", "qwen-27b",
            "--host", "127.0.0.1", "--port", "8080")
        self.fake_proc.listen(8080, INODE, os.getuid())
        self.fake_proc.process(PID, argv=(*argv, *extra), environ=environ, exe=exe or str(self.bin),
                               cwd=str(self.home), cgroup=cgroup(unit) if unit else None,
                               fds={7: INODE}, **process_kw)

    def written(self) -> dict:
        return yaml.safe_load((self.cfg / "config.yaml").read_text(encoding="utf-8"))

    def ledger(self) -> Path:
        return speed_stats.default_speed_stats_path(self.cfg)


@pytest.fixture
def live(tmp_path, monkeypatch, fake_home, fake_proc, fake_systemd):  # noqa: F811
    home = fake_home(tmp_path / "home", clear_overrides=False)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "llama-server"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.delenv("LOCALHARNESS_COMMIT", raising=False)
    box = Live(tmp_path, home, fake_proc, bin_dir, monkeypatch)
    gguf(box.model_a)
    gguf(box.model_b)
    fake_systemd.answer_prefix("systemctl", "--user", "show", stdout=show())
    monkeypatch.setattr(init_cmd, "detect_provider", box.detect)
    monkeypatch.setattr(autosense, "_run_nvidia_smi", lambda: None)
    monkeypatch.setattr(autosense, "compute_fit", lambda server, ratios, **kw: autosense.FitInputs(
        header={"context_length": 262144}, offload="full", memory=None, weights=None, meminfo=None,
        fits=114688))
    for name in ("Confirm", "IntPrompt", "Prompt"):
        monkeypatch.setattr(init_cmd, name, _Asked)
    monkeypatch.setattr(init_cmd, "_probe_capabilities", _boom("_probe_capabilities"))
    monkeypatch.setattr(subprocess, "Popen", _boom("subprocess.Popen"))
    monkeypatch.setattr(init_cmd, "_guided_setup", _boom("_guided_setup"))
    box.fake_systemd = fake_systemd
    return box


def lines_of(result) -> list[str]:
    return [" ".join(line.split()) for line in (result.output or "").splitlines()]


def block_of(result) -> list[str]:
    """The lines of the shown block: from `serving ...` up to the line before the receipt."""
    lines = lines_of(result)
    start = next(i for i, line in enumerate(lines) if line.startswith("serving "))
    end = next(i for i, line in enumerate(lines) if line.startswith("✓ LocalHarness configured"))
    return [line for line in lines[start:end] if line]


# ------------------------------------------------------------------ the tracer


def test_a_running_llama_server_becomes_the_primary_with_discoverys_other_model_as_a_peer(live):
    live.serve("-c", "65536", "-ngl", "99")

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    cfg = live.written()
    server = cfg["server"]
    assert server["runtime"] == "llamacpp"
    assert server["binary"] == str(live.bin)
    assert server["model"] == str(live.model_a)
    assert server["served_name"] == "qwen-27b" and server["port"] == 8080
    assert server["extra_args"] == ["-ngl", "99"]
    assert "context" not in server and None not in server.values()

    [peer] = cfg["extra_endpoints"]
    assert peer["name"] == "B" and peer["base_url"] == "http://127.0.0.1:8081/v1"
    assert peer["lifecycle"]["model"] == str(live.model_b) and peer["lifecycle"]["port"] == 8081
    assert "binary" not in peer["lifecycle"]

    provider = cfg["provider"]
    assert provider["provider_type"] == "llamacpp"
    assert provider["base_url"] == URL
    assert provider["default_model"] == "qwen-27b"
    assert provider["available_models"] == ["qwen-27b"]
    assert cfg["org"]["default_model"] == "qwen-27b"


def test_the_block_is_the_primary_its_extras_the_second_line_then_the_peer(live):
    live.serve("-c", "65536", "-ngl", "99")

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert block_of(result) == [
        f"serving qwen-27b (llama.cpp) ← ~/models/gguf/A.gguf · binary: {live.bin} (from {UNIT})",
        "flags: -ngl 99",
        f"{UNIT} serves -c 65536; the harness launches -c 114688 once it stops "
        "· server.context: 65536 keeps it",
        "/model switches to B (llama.cpp) ← ~/models/gguf/B.gguf",
    ]
    assert f"{UNIT} holds :8080; the harness attaches" not in " ".join(lines_of(result))


def test_the_import_asks_nothing_probes_nothing_and_launches_nothing(live):
    live.serve("-c", "65536")

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)  # Confirm / Prompt / _probe_capabilities / Popen all raise if touched


def test_the_ledger_holds_the_weights_path_and_the_window_triple_that_was_printed(live):
    live.serve("-c", "65536")

    _exited(_init(live.cfg, "--no-input"), 0)

    assert speed_stats.model_path(live.ledger(), "llamacpp", "qwen-27b") == str(live.model_a)
    report = speed_stats.last_window_report(live.ledger(), "llamacpp", "qwen-27b")
    assert (report.served, report.trained, report.fits) == (65536, 262144, 114688)


def test_without_a_dash_c_the_second_line_is_the_attach_form_and_no_window_is_recorded(live):
    live.serve()

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert block_of(result) == [
        f"serving qwen-27b (llama.cpp) ← ~/models/gguf/A.gguf · binary: {live.bin} (from {UNIT})",
        f"{UNIT} holds :8080; the harness attaches",
        "/model switches to B (llama.cpp) ← ~/models/gguf/B.gguf",
    ]
    assert "context" not in live.written()["server"]
    assert speed_stats.last_window_report(live.ledger(), "llamacpp", "qwen-27b") is None
    assert speed_stats.model_path(live.ledger(), "llamacpp", "qwen-27b") == str(live.model_a)


def test_an_interactive_init_prints_the_same_block(live):
    live.serve("-c", "65536", "-ngl", "99")
    unattended = block_of(_init(live.cfg, "--no-input"))
    (live.cfg / "config.yaml").unlink()
    _terminal(live.monkeypatch)

    result = _init(live.cfg)

    _exited(result, 0)
    assert block_of(result) == unattended


def test_a_pid_in_no_unit_is_named_by_its_pid_and_command(live):
    live.serve("-c", "65536", unit=None)

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    block = block_of(result)
    assert block[0].endswith(f"(from pid {PID} (llama-server))")
    assert f"pid {PID} (llama-server) serves -c 65536; the harness launches -c 114688 once it stops " \
           "· server.context: 65536 keeps it" in block


def test_a_binary_found_on_path_reads_on_path(live):
    live.fake_proc.listen(8080, INODE, os.getuid())
    live.fake_proc.process(PID, argv=("llama-server", "-m", str(live.model_a), "--alias", "qwen-27b",
                                      "--port", "8080"), cgroup=cgroup(), fds={7: INODE})  # no exe link

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert block_of(result)[0].endswith(f"· binary: {live.bin} (on PATH)")
    assert live.written()["server"]["binary"] == str(live.bin)


def test_the_fit_is_sized_with_the_window_the_incumbent_serves_added_back(live):
    live.serve("-c", "65536")
    asked: list[dict] = []

    def _fit(server, ratios, **kw):
        asked.append({"model": server.model, **kw})
        return autosense.FitInputs(header={}, offload="full", memory=None, weights=None, meminfo=None,
                                   fits=114688)

    live.monkeypatch.setattr(autosense, "compute_fit", _fit)

    _exited(_init(live.cfg, "--no-input"), 0)

    assert {"model": str(live.model_a), "attached_served": 65536} in asked


def test_a_fit_that_cannot_be_computed_reads_with_no_dash_c(live):
    live.serve("-c", "65536")
    live.monkeypatch.setattr(autosense, "compute_fit", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert any(line.endswith("the harness launches with no -c once it stops "
                             "· server.context: 65536 keeps it") for line in block_of(result))


def test_the_model_list_is_not_judged_first_a_keyed_server_answering_401_still_imports(live):
    live.detect.return_value = detected(models=(), raw=KEYED_401)
    live.serve("--api-key-file", "/keys/k", "-c", "65536")

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert "No models available" not in result.output
    server = live.written()["server"]
    assert server["api_key_file"] == "/keys/k" and server["require_api_key"] is True
    assert live.written()["provider"]["default_model"] == "qwen-27b"


def test_nothing_listening_keeps_todays_attach_path(live, monkeypatch):
    """No listener on the detected port (an empty /proc): detection's answer is attached as before."""
    monkeypatch.setattr(init_cmd, "_probe_capabilities", lambda result, model: _make_capability_result())

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    cfg = live.written()
    assert cfg.get("server") is None and cfg["provider"]["default_model"] == "qwen-27b"


def test_a_change_never_imports(live, monkeypatch):
    live.serve("-c", "65536")
    live.cfg.mkdir(parents=True, exist_ok=True)
    (live.cfg / "config.yaml").write_text(
        "version: '1'\nprovider:\n  provider_type: ollama\n  base_url: http://original/v1\n"
        "  default_model: old\n", encoding="utf-8")
    monkeypatch.setattr(init_cmd, "_import_listener", _boom("_import_listener"))
    monkeypatch.setattr(init_cmd, "_probe_capabilities", lambda result, model: _make_capability_result())

    result = _init(live.cfg, "--no-input", "--model", "qwen-27b")

    _exited(result, 0)


# ------------------------------------------------------------------ secrets, shown extras, adoption


SECRET = "sk-live-0123456789secret"


def everything_under(*roots: Path) -> dict[Path, str]:
    """The text of every regular file below `roots` (what init may have written anywhere)."""
    found: dict[Path, str] = {}
    for root in roots:
        for path in Path(root).rglob("*"):
            if path.is_file() and not path.is_symlink():
                found[path] = path.read_text(encoding="utf-8", errors="replace")
    return found


def machine_dirs(live) -> list[Path]:
    return [live.cfg, live.home, Path(os.environ["LOCALHARNESS_HOME"])]


def test_an_inline_key_is_written_only_as_provider_api_key_in_the_private_config(live):
    live.detect.return_value = detected(models=(), raw=KEYED_401)
    live.serve("--api-key", SECRET, "--host", "0.0.0.0", "--verbose", "-c", "65536")  # noqa: S104

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    cfg = live.written()
    assert cfg["provider"]["api_key"] == SECRET
    assert (live.cfg / "config.yaml").stat().st_mode & 0o777 == 0o600
    server = cfg["server"]
    assert server["bind_all"] is True and server["require_api_key"] is True
    assert SECRET not in " ".join(server["extra_args"]) and "--verbose" not in server["extra_args"]
    assert init_import.VERBOSE_DROPPED in lines_of(result)  # the one line, verbatim
    assert SECRET not in result.output
    for path, text in everything_under(*machine_dirs(live)).items():
        if path != live.cfg / "config.yaml":
            assert SECRET not in text, path


def test_an_unkeyed_wide_bind_says_every_interface_no_key(live):
    live.serve("--host", "0.0.0.0", "-c", "65536")  # noqa: S104

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert any("every interface, no key" in line for line in block_of(result))
    assert live.written()["server"]["bind_all"] is True


def test_a_key_file_is_recorded_and_never_opened_on_a_terminal_or_without_one(live, tmp_path):
    key = tmp_path / "keys" / "key"
    key.parent.mkdir()
    key.write_text("K-from-the-file\n", encoding="utf-8")
    live.detect.return_value = detected(models=(), raw=KEYED_401)
    live.serve("--api-key-file", str(key), "-c", "65536")

    for index, interactive in enumerate((True, False)):
        live.cfg = tmp_path / f"cfg{index}"
        if interactive:
            _terminal(live.monkeypatch)
        else:
            _terminal(live.monkeypatch, on=False)
        with record_opens() as opened:
            result = _init(live.cfg, *([] if interactive else ["--no-input"]))
        _exited(result, 0)
        assert str(key) not in opened
        server = live.written()["server"]
        assert server["api_key_file"] == str(key) and server["require_api_key"] is True
        assert f"key file: {key} (read at start, never by init)" in lines_of(result)
        assert "K-from-the-file" not in result.output


def test_an_interactive_import_adopts_the_key_file_and_the_next_start_asks_nothing(live, tmp_path):
    key = tmp_path / "key"
    key.write_text("K\n", encoding="utf-8")
    live.detect.return_value = detected(models=(), raw=KEYED_401)
    live.serve("--api-key-file", str(key), "-c", "65536")
    _terminal(live.monkeypatch)

    _exited(_init(live.cfg), 0)

    recorded = trust.recorded_machine(live.cfg)
    assert recorded is not None and recorded["entries"] == _expected(live.cfg)
    assert {"server.api_key_file", "server.model", "server.binary", "server.require_api_key"} \
        <= {e["name"] for e in recorded["entries"]}
    never_asked(live.monkeypatch)
    decided = decide_machine_trust(live.cfg, ask=True, launching=True)
    assert decided.withheld == {} and decided.line == ""


def test_a_no_input_import_adopts_nothing_and_the_next_terminal_start_asks_about_the_key_file(
        live, tmp_path):
    key = tmp_path / "key"
    key.write_text("K\n", encoding="utf-8")
    live.detect.return_value = detected(models=(), raw=KEYED_401)
    live.serve("--api-key-file", str(key), "-c", "65536")

    _exited(_init(live.cfg, "--no-input"), 0)

    assert trust.recorded_machine(live.cfg) is None
    asked = answers(live.monkeypatch, True)
    decide_machine_trust(live.cfg, ask=True, launching=True)
    assert len(asked) == 1 and "server.api_key_file" in asked[0]


def test_the_shell_tools_init_adopts_nothing(live, monkeypatch):
    live.serve("-c", "65536")
    monkeypatch.setenv("LOCALHARNESS_COMMIT", "1")
    _terminal(live.monkeypatch)

    _exited(_init(live.cfg), 0)

    assert trust.recorded_machine(live.cfg) is None


def test_the_adoption_holds_no_key_value(live):
    live.detect.return_value = detected(models=(), raw=KEYED_401)
    live.serve("--api-key", SECRET, "-c", "65536")
    _terminal(live.monkeypatch)

    _exited(_init(live.cfg), 0)

    assert SECRET not in yaml.safe_dump(trust.recorded_machine(live.cfg))


def test_config_ini_is_carried_and_shown_with_its_annotation(live, monkeypatch):
    ini = live.home / ".config" / "llama.cpp" / "config.ini"
    ini.parent.mkdir(parents=True)
    ini.write_text("tools = all\n", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(live.home / ".config"))
    live.serve("-c", "65536")

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert live.written()["server"]["config_ini"] == "tools = all\n"
    assert "config.ini (~/.config/llama.cpp): tools=all (serves a shell-exec tool on the port)" \
        in block_of(result)


def test_a_decoy_hugging_face_token_in_the_environment_appears_nowhere(live):
    live.serve("-c", "65536", environ={"HF_TOKEN": "hf_decoy", "LLAMA_ARG_THREADS": "8"})
    _terminal(live.monkeypatch)

    result = _init(live.cfg)

    _exited(result, 0)
    assert live.written()["server"]["env"] == {"LLAMA_ARG_THREADS": "8"}
    assert "hf_decoy" not in result.output
    for path, text in everything_under(*machine_dirs(live)).items():
        assert "hf_decoy" not in text, path


def test_an_escape_sequence_in_an_argument_prints_escaped_on_every_line(live):
    live.serve(base=("llama-server", "-m", str(live.model_a), "--alias", "\x1b[2Kqwen", "--port", "8080",
                     "-c", "65536", "--note\x0aX"))

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    shown = result.output
    assert "\\x1b[2Kqwen" in shown
    for line in shown.splitlines():
        assert not any(ord(ch) < 0x20 and ch != "\t" for ch in line), repr(line)
