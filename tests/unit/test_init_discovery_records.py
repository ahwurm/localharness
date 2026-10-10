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
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

import localharness.cli.init_cmd as init_cmd
from localharness.config.loader import ConfigLoader
from localharness.config.models import ManagedServerConfig
from localharness.provider import autosense, procfs, speed_stats
from localharness.provider.detector import DetectorResult
from tests.unit.gguf_support import write_gguf
from tests.unit.procfs_support import FakeProc
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
    return type("Box", (), dict(home=home, root=root, swift=swift, qwen=qwen, proj=proj,
                                bin=bin_dir / "llama-server", cfg=cfg, detect=detect))


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
