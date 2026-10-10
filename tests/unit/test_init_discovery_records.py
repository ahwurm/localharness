"""INIT-01 / INIT-02 / INIT-03 (discovery): a fresh `init` with GGUFs on the machine and nothing
serving writes presets (Phase 5, D-10, D-11, D-13, D-14, D-24, D-39, D-41, D-49, D-51, D-54, D-55).

The first ranked model is the `server:` record, the rest are launchable peers; one block says which,
nothing is asked and nothing is launched (`Confirm`, `Prompt`, `IntPrompt` and `Popen` raise if
touched), and the red "No local LLM detected" line and the port list are never printed. Every GGUF is
a real header under a faked HOME; nvidia-smi, `/proc` and the daemons are stubbed.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

import localharness.cli.init_cmd as init_cmd
from localharness.config.loader import ConfigLoader
from localharness.provider import autosense, speed_stats
from localharness.provider.detector import DetectorResult
from tests.unit.gguf_support import write_gguf
from tests.unit.procfs_support import fake_proc  # noqa: F401  (fixture)
from tests.unit.test_init_core_setup import (
    _Asked,
    _exited,
    _flat,
    _hermetic,  # noqa: F401  (autouse fixture: no real HTTP)
    _init,
    _terminal,
)


def _model(path: Path, *, embedding: int = 5120, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_gguf(path, {"general.architecture": "llama", "general.type": "model",
                      "llama.embedding_length": embedding, "llama.context_length": 32768})
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _projector(path: Path, *, dim: int = 5120) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_gguf(path, {"general.architecture": "clip", "general.type": "mmproj",
                             "clip.vision.projection_dim": dim})


def _bin_dir(tmp_path: Path, monkeypatch, *, with_server: bool = True) -> Path:
    """PATH is one tmp dir, holding an executable llama-server or nothing at all."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    if with_server:
        exe = bin_dir / "llama-server"
        exe.write_text("#!/bin/sh\n", encoding="utf-8")
        exe.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    return bin_dir


def _not_found() -> AsyncMock:
    return AsyncMock(return_value=DetectorResult(
        found=False, provider_type="unknown", base_url="", models=[], suggested_model="",
        probe_duration_ms=0.0))


def _boom(name: str):
    def _raise(*_a, **_k):
        raise AssertionError(f"{name} called: discovery must ask and launch nothing")
    return _raise


@pytest.fixture
def box(tmp_path, monkeypatch, fake_home):
    """Two models and a projector under the faked HOME, llama-server on PATH, nothing answering,
    no GPU, a terminal on which every question and every spawn is booby-trapped."""
    home = fake_home(tmp_path / "home", clear_overrides=False)
    root = home / "models" / "gguf"
    swift = _model(root / "Swift-27B-Q8.gguf", mtime=1_000_000)
    qwen = _model(root / "Qwen-27B.gguf", mtime=2_000_000)
    proj = _projector(root / "mmproj-Swift-27B-F16.gguf")
    bin_dir = _bin_dir(tmp_path, monkeypatch)
    detect = _not_found()
    monkeypatch.setattr(init_cmd, "detect_provider", detect)
    monkeypatch.setattr(autosense, "_run_nvidia_smi", lambda: None)
    _terminal(monkeypatch)
    for name in ("Confirm", "IntPrompt", "Prompt"):
        monkeypatch.setattr(init_cmd, name, _Asked)
    monkeypatch.setattr(subprocess, "Popen", _boom("subprocess.Popen"))
    monkeypatch.setattr(init_cmd, "_guided_setup", _boom("_guided_setup"))
    cfg = tmp_path / "cfg"
    return SimpleNamespace(home=home, root=root, swift=swift, qwen=qwen, proj=proj,
                           bin=bin_dir / "llama-server", cfg=cfg, detect=detect)


def _written(cfg: Path) -> dict:
    return yaml.safe_load((cfg / "config.yaml").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ the tracer


def test_a_fresh_init_with_ggufs_writes_a_primary_and_a_peer_and_asks_nothing(box):
    result = _init(box.cfg)

    _exited(result, 0)
    cfg = _written(box.cfg)
    server = cfg["server"]
    assert server["runtime"] == "llamacpp"
    assert server["binary"] == str(box.bin)
    assert server["model"] == str(box.qwen)
    assert server["served_name"] == "Qwen-27B" and server["port"] == 8081
    assert "mmproj" not in server and "context" not in server and "effort" not in server
    assert None not in server.values()

    [peer] = cfg["extra_endpoints"]
    assert {k: v for k, v in peer.items() if k != "lifecycle"} == {
        "name": "Swift-27B-Q8", "base_url": "http://127.0.0.1:8082/v1", "provider_type": "llamacpp",
        "api_key": "none", "extra_headers": {}, "gpu": True}
    life = peer["lifecycle"]
    assert life.items() >= {"gpu": True, "runtime": "llamacpp", "model": str(box.swift),
                            "mmproj": str(box.proj), "served_name": "Swift-27B-Q8", "port": 8082,
                            "extra_args": []}.items()
    for absent in ("binary", "env", "config_ini", "api_key_file", "cache_type_k", "cache_type_v",
                   "cache_ram", "context", "effort"):
        assert absent not in life, absent

    assert cfg["provider"]["provider_type"] == "llamacpp"
    assert cfg["provider"]["base_url"] == "http://127.0.0.1:8081/v1"
    assert cfg["provider"]["default_model"] == "Qwen-27B"
    assert cfg["provider"]["available_models"] == ["Qwen-27B"]
    assert cfg["org"]["default_model"] == "Qwen-27B"
    # what was written is a config the next start loads
    loaded = ConfigLoader(config_dir=box.cfg).load_harness()
    assert loaded.server.served_name == "Qwen-27B" and loaded.extra_endpoints[0].name == "Swift-27B-Q8"


def test_one_block_names_the_primary_with_its_binary_and_what_model_switches_to(box):
    result = _init(box.cfg)

    lines = result.output.splitlines()
    primary = f"serving Qwen-27B (llama.cpp) ← ~/models/gguf/Qwen-27B.gguf · binary: {box.bin} (on PATH)"
    peer = ("/model switches to Swift-27B-Q8 (llama.cpp, +mmproj-Swift-27B-F16.gguf) "
            "← ~/models/gguf/Swift-27B-Q8.gguf")
    assert primary in lines, lines
    assert lines.index(peer) == lines.index(primary) + 1, lines
    out = _flat(result)
    for absent in ("No local LLM detected", "Checked:", "Set up vLLM"):
        assert absent not in out, out
    assert "LocalHarness configured at" in out and init_cmd.RECEIPT_ROW in out


def test_discovery_reports_a_server_ready_for_the_first_start(box):
    out = init_cmd.core_setup(str(box.cfg), endpoint=None, model=None, force=False, interactive=True)

    assert out == init_cmd.SetupResult(box.cfg / "config.yaml", True)


def test_discovery_needs_no_terminal(box):
    result = _init(box.cfg, "--no-input")

    _exited(result, 0)
    assert _written(box.cfg)["server"]["served_name"] == "Qwen-27B"


def test_a_change_that_finds_nothing_never_discovers_or_writes_presets(box, monkeypatch):
    box.cfg.mkdir()
    seeded = ("version: '1'\nprovider:\n  provider_type: ollama\n  base_url: http://original/v1\n"
              "  default_model: old\n")
    (box.cfg / "config.yaml").write_text(seeded, encoding="utf-8")
    monkeypatch.setattr(autosense, "free_memory", _boom("free_memory"))

    result = _init(box.cfg, "--model", "x")

    _exited(result, 1)
    assert (box.cfg / "config.yaml").read_text(encoding="utf-8") == seeded
    assert "Nothing changed" in _flat(result)


# ------------------------------------------------------------------ ranking (D-07, D-08, D-09, D-49)


def _primary(cfg: Path) -> str:
    return _written(cfg)["server"]["served_name"]


def test_a_model_the_ledger_recorded_outranks_a_newer_one(box):
    box.cfg.mkdir()
    speed_stats.record_model_path(speed_stats.default_speed_stats_path(box.cfg), "llamacpp", "x",
                                  str(box.swift))

    _exited(_init(box.cfg), 0)

    cfg = _written(box.cfg)
    assert cfg["server"]["served_name"] == "Swift-27B-Q8"
    assert cfg["server"]["mmproj"] == str(box.proj)
    assert [p["name"] for p in cfg["extra_endpoints"]] == ["Qwen-27B"]


def test_a_model_that_fits_outranks_a_newer_one_that_does_not(box, monkeypatch):
    memory = SimpleNamespace(free_bytes=24 << 30)
    seen = []

    def fake_fit(spec, ratios, *, memory):
        seen.append(memory)
        return SimpleNamespace(fits=8192 if spec.model == str(box.swift) else None)

    monkeypatch.setattr(autosense, "free_memory", lambda _ratios: memory)
    monkeypatch.setattr(autosense, "compute_fit", fake_fit)

    _exited(_init(box.cfg), 0)

    assert _primary(box.cfg) == "Swift-27B-Q8"
    assert seen == [memory, memory]  # every model sized against the one probe


def test_otherwise_the_newest_file_wins_and_then_the_path_string(box):
    for stale in (box.swift, box.qwen):
        os.utime(stale, (3_000_000, 3_000_000))

    _exited(_init(box.cfg), 0)

    assert _primary(box.cfg) == "Qwen-27B"  # equal mtimes: Qwen-27B.gguf sorts before Swift-27B-Q8.gguf


def test_free_memory_is_probed_once_for_three_models(box, monkeypatch):
    _model(box.root / "Third-7B.gguf", embedding=4096)
    calls = []
    monkeypatch.setattr(autosense, "free_memory", lambda ratios: calls.append(ratios) or None)

    _exited(_init(box.cfg), 0)

    assert len(calls) == 1


def test_free_memory_is_not_probed_when_no_gguf_exists(tmp_path, monkeypatch, fake_home):
    fake_home(tmp_path / "home", clear_overrides=False)
    _bin_dir(tmp_path, monkeypatch)
    monkeypatch.setattr(init_cmd, "detect_provider", _not_found())
    monkeypatch.setattr(autosense, "free_memory", _boom("free_memory"))

    result = _init(tmp_path / "cfg", "--no-input")

    _exited(result, 1)
    assert "No local LLM detected" in _flat(result) and not (tmp_path / "cfg" / "config.yaml").exists()


def test_a_size_that_cannot_be_computed_leaves_that_model_ranked_without_a_fit(box, monkeypatch):
    def fake_fit(spec, ratios, *, memory):
        if spec.model == str(box.qwen):
            raise OSError("unreadable")
        return SimpleNamespace(fits=4096)

    monkeypatch.setattr(autosense, "compute_fit", fake_fit)

    _exited(_init(box.cfg), 0)

    assert _primary(box.cfg) == "Swift-27B-Q8"


# ------------------------------------------------------------------ ports and names (D-05, D-11)


def test_peers_take_the_ports_above_the_primary_skipping_a_listener(box, fake_proc):  # noqa: F811
    _model(box.root / "Third-7B.gguf", embedding=4096, mtime=500_000)
    fake_proc.listen(8082, inode=77, uid=os.getuid())

    _exited(_init(box.cfg), 0)

    cfg = _written(box.cfg)
    assert cfg["server"]["port"] == 8081
    assert [(p["name"], p["lifecycle"]["port"]) for p in cfg["extra_endpoints"]] == [
        ("Swift-27B-Q8", 8083), ("Third-7B", 8084)]
    assert [p["base_url"] for p in cfg["extra_endpoints"]] == [
        "http://127.0.0.1:8083/v1", "http://127.0.0.1:8084/v1"]


def test_free_ports_skip_the_ports_a_record_uses_and_the_ones_listening(fake_proc):  # noqa: F811
    fake_proc.listen(9002, inode=5, uid=os.getuid())

    assert init_cmd._free_ports(9001, 3, taken={9001, 9004}) == [9003, 9005, 9006]
    assert init_cmd._free_ports(9001, 0, taken=set()) == []
    assert init_cmd._free_ports(65534, 5, taken=set()) == [65534, 65535]


def test_peer_records_name_each_model_once_and_carry_only_what_launches_it(box):
    found = init_cmd._found_models(box.cfg)
    assert [m.name for m in found] == ["Qwen-27B", "Swift-27B-Q8"]

    [peer] = init_cmd._peer_records(found[1:], start=9100, taken_names={"Swift-27B-Q8"},
                                    taken_ports={9100})

    assert peer.name == "Swift-27B-Q8-2" and peer.base_url == "http://127.0.0.1:9101/v1"
    assert peer.provider_type == "llamacpp" and peer.gpu is True
    life = peer.lifecycle
    assert (life.runtime, life.model, life.mmproj, life.served_name, life.port, life.gpu) == (
        "llamacpp", str(box.swift), str(box.proj), "Swift-27B-Q8-2", 9101, True)
    assert not (life.binary or life.extra_args or life.env or life.config_ini or life.api_key_file)


def test_two_files_with_one_preset_name_get_a_numbered_second(tmp_path, monkeypatch, fake_home):
    home = fake_home(tmp_path / "home", clear_overrides=False)
    _model(home / "models" / "a" / "m.gguf", mtime=2_000_000)
    _model(home / "models" / "b" / "m.gguf", embedding=4096, mtime=1_000_000)
    _bin_dir(tmp_path, monkeypatch)
    monkeypatch.setattr(init_cmd, "detect_provider", _not_found())
    monkeypatch.setattr(autosense, "_run_nvidia_smi", lambda: None)

    result = _init(tmp_path / "cfg", "--no-input")

    _exited(result, 0)
    cfg = _written(tmp_path / "cfg")
    assert cfg["server"]["served_name"] == "m"
    assert [p["name"] for p in cfg["extra_endpoints"]] == ["m-2"]
    assert cfg["extra_endpoints"][0]["lifecycle"]["served_name"] == "m-2"


def test_a_filename_with_an_escape_is_shown_escaped_and_named_with_a_dash(tmp_path, monkeypatch,
                                                                         fake_home):
    home = fake_home(tmp_path / "home", clear_overrides=False)
    _model(home / "models" / "q\x1b[2Jx.gguf", mtime=2_000_000)
    _model(home / "models" / "plain\nline.gguf", embedding=4096, mtime=1_000_000)
    _bin_dir(tmp_path, monkeypatch)
    monkeypatch.setattr(init_cmd, "detect_provider", _not_found())
    monkeypatch.setattr(autosense, "_run_nvidia_smi", lambda: None)

    result = _init(tmp_path / "cfg", "--no-input")

    _exited(result, 0)
    assert "\x1b" not in result.output and "\r" not in result.output
    assert "~/models/q\\x1b[2Jx.gguf" in result.output
    assert "~/models/plain\\x0aline.gguf" in result.output
    cfg = _written(tmp_path / "cfg")
    assert cfg["server"]["served_name"] == "q--2Jx"
    assert [p["name"] for p in cfg["extra_endpoints"]] == ["plain-line"]


# ------------------------------------------------------------------ no llama-server (D-06, D-24, D-55)


def test_no_llama_server_on_path_declares_the_records_and_says_so_once(box, monkeypatch):
    _bin_dir(box.bin.parent.parent, monkeypatch, with_server=False)
    box.bin.unlink()

    out = init_cmd.core_setup(str(box.cfg), endpoint=None, model=None, force=False, interactive=True)

    assert out == init_cmd.SetupResult(box.cfg / "config.yaml", False)
    cfg = _written(box.cfg)
    assert "binary" not in cfg["server"] and "binary" not in cfg["extra_endpoints"][0]["lifecycle"]
    assert cfg["server"]["model"] == str(box.qwen)


def test_the_no_binary_line_prints_once_and_the_primary_line_has_no_binary_clause(box, monkeypatch):
    box.bin.unlink()

    result = _init(box.cfg)

    _exited(result, 0)
    lines = result.output.splitlines()
    assert lines.count("found 2 GGUFs under ~/models; no llama-server \u2014 set server.binary") == 1
    assert result.output.count("no llama-server") == 1
    assert "serving Qwen-27B (llama.cpp) \u2190 ~/models/gguf/Qwen-27B.gguf" in lines  # nothing after it
    assert "binary:" not in result.output


def test_a_program_named_llama_server_in_the_cwd_is_not_the_binary(box, monkeypatch):
    box.bin.unlink()
    here = box.cfg.parent / "here"
    here.mkdir()
    (here / "llama-server").write_text("#!/bin/sh\n", encoding="utf-8")
    (here / "llama-server").chmod(0o755)
    monkeypatch.chdir(here)
    monkeypatch.setenv("PATH", f".{os.pathsep}{os.pathsep}{box.bin.parent}")  # relative entries only

    _exited(_init(box.cfg), 0)

    assert "binary" not in _written(box.cfg)["server"]


def test_the_no_binary_line_names_every_root_holding_a_model_and_only_those():
    class M:
        def __init__(self, real):
            self.real = real

    line = init_cmd._no_binary_line([M("/data/a/x.gguf"), M("/data/b/y.gguf")],
                                    ["/data/a", "/data/b", "/data/empty"])

    assert line == "found 2 GGUFs under /data/a, /data/b; no llama-server \u2014 set server.binary"
    assert init_cmd._no_binary_line([M("/data/a/x.gguf")], ["/data/a"]) == (
        "found 1 GGUF under /data/a; no llama-server \u2014 set server.binary")


def test_the_model_roots_a_machine_overrides_file_names_are_the_ones_walked(tmp_path, monkeypatch,
                                                                            fake_home):
    home = fake_home(tmp_path / "home", clear_overrides=False)
    _model(home / "models" / "ignored.gguf")
    _model(home / "weights" / "kept.gguf")
    _bin_dir(tmp_path, monkeypatch)
    monkeypatch.setattr(init_cmd, "detect_provider", _not_found())
    monkeypatch.setattr(autosense, "_run_nvidia_smi", lambda: None)
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "overrides.yaml").write_text(f"autosense:\n  model_roots: ['{home / 'weights'}']\n",
                                            encoding="utf-8")

    _exited(_init(cfg_dir, "--no-input"), 0)

    cfg = _written(cfg_dir)
    assert cfg["server"]["served_name"] == "kept" and cfg["extra_endpoints"] == []


def test_the_lm_studio_and_ollama_stores_are_not_walked(tmp_path, monkeypatch, fake_home):
    home = fake_home(tmp_path / "home", clear_overrides=False)
    _model(home / ".lmstudio" / "models" / "pub" / "repo" / "m.gguf")
    _model(home / ".ollama" / "models" / "blobs" / "b.gguf")
    monkeypatch.setattr(init_cmd, "detect_provider", _not_found())
    monkeypatch.setattr(init_cmd, "_guided_setup", lambda _p: None)
    _bin_dir(tmp_path, monkeypatch)

    result = _init(tmp_path / "cfg", "--no-input")

    _exited(result, 1)
    assert "No local LLM detected" in _flat(result)


# ------------------------------------------------------------------ Ollama and LM Studio (D-41, D-54)


def _daemons(monkeypatch, by_port):
    """detect_provider: nothing on the sweep; on a single-port ask, what `by_port` says."""
    calls = []

    async def fake(timeout_seconds=1.0, ports=None):
        calls.append((timeout_seconds, ports))
        kind = by_port.get(ports[0]) if ports and len(ports) == 1 else None
        if kind is None:
            return DetectorResult(found=False, provider_type="unknown", base_url="", models=[],
                                  suggested_model="", probe_duration_ms=0.0)
        provider, models = kind
        return DetectorResult(found=True, provider_type=provider, base_url=f"http://x:{ports[0]}/v1",
                              models=list(models), suggested_model=models[0], probe_duration_ms=0.0)

    monkeypatch.setattr(init_cmd, "detect_provider", fake)
    return calls


def test_an_ollama_that_answers_is_named_with_three_of_its_models(box, monkeypatch):
    calls = _daemons(monkeypatch, {11434: ("ollama", ["alpha", "beta", "gamma", "delta"])})

    result = _init(box.cfg)

    _exited(result, 0)
    lines = result.output.splitlines()
    peer = next(i for i, line in enumerate(lines) if line.startswith("/model switches to"))
    assert lines[peer + 1] == "Ollama holds 4 models (alpha, beta, gamma, \u2026) \u2014 left in Ollama, no preset written"
    assert sum("holds" in line for line in lines) == 1
    assert (1.0, [11434]) in calls and (1.0, [1234]) in calls
    written = (box.cfg / "config.yaml").read_text(encoding="utf-8")
    assert not any(name in written for name in ("alpha", "beta", "gamma", "delta"))
    assert "ollama" not in {p["provider_type"] for p in _written(box.cfg)["extra_endpoints"]}


def test_two_ollama_models_are_both_listed_and_lm_studio_has_its_own_line(box, monkeypatch):
    _daemons(monkeypatch, {11434: ("ollama", ["a", "b"]), 1234: ("lmstudio", ["qwen\x1b[31m", "z"])})

    result = _init(box.cfg)

    _exited(result, 0)
    lines = result.output.splitlines()
    assert "Ollama holds 2 models (a, b) \u2014 left in Ollama, no preset written" in lines
    assert "LM Studio holds 2 models (qwen\\x1b[31m, z) \u2014 left in LM Studio, no preset written" in lines
    assert "\x1b" not in result.output


def test_a_detector_that_ignores_the_port_it_was_given_adds_no_line(box, monkeypatch):
    _exited(_init(box.cfg), 0)  # `box.detect` answers not found whatever it is asked
    box.detect.assert_any_call(timeout_seconds=1.0, ports=[11434])
    box.detect.assert_any_call(timeout_seconds=1.0, ports=[1234])
    assert "holds" not in _init(box.cfg.parent / "again").output


def test_a_daemon_that_answers_as_some_other_runtime_adds_no_line(box, monkeypatch):
    _daemons(monkeypatch, {11434: ("vllm", ["m"])})

    assert "holds" not in _init(box.cfg).output


def test_the_daemon_probes_only_run_for_a_llama_cpp_record(monkeypatch):
    monkeypatch.setattr(init_cmd, "detect_provider", _boom("detect_provider"))
    recs = init_cmd.InitRecords(
        server=init_cmd.ManagedServerConfig(runtime="vllm", model="m", binary="/v"), served="m",
        lines=["serving m (vLLM) \u2190 m"])

    init_cmd._show_records(recs)  # would raise if it asked a daemon
