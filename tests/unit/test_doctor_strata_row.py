"""doctor's `Strata:` row: the clone-and-setup command for a known Flash-Next file, the install's configs, and silence otherwise.

The world is `test_serve_cmd.serve_world`; the Flash-Next shards are the recorded fixture written sparse under HOME.
Doctor's httpx is a mock and no program is spawned.
"""
from __future__ import annotations

import re
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from localharness.agent.context import WindowProbe
from localharness.cli import doctor_cmd
from localharness.cli.app import app
from tests.unit.strata_fixture import flash_next
from tests.unit.test_doctor_cmd import (  # noqa: F401  (autouse fixtures, imported for their effect)
    _models_resp,
    _route_the_window_probe,
    _stub_token_counter,
)
from tests.unit.test_serve_cmd import llama_peer, serve_world

runner = CliRunner()
ROWS = re.compile(r"^i\s+(Strata: .*)$", re.MULTILINE)
SHARD1 = "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
COMMAND = ("git clone https://github.com/Niko1221/Strata ~/src/Strata && cd ~/src/Strata && ./setup.sh --yes --no-start "
           "--family swift --model IQ3_XXS --gguf-dir {gguf_dir} --context 65536 --vision no --no-browser")


@pytest.fixture(autouse=True)
def nothing_runs(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("doctor's Strata row ran a program")
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(doctor_cmd, "_llama_server_linked_backends", lambda binary: None)


@pytest.fixture
def mock_httpx(monkeypatch):
    mock = MagicMock()
    mock.get.return_value = _models_resp({"data": [{"id": "swift"}]})
    monkeypatch.setattr(doctor_cmd, "httpx", mock)
    return mock


def _world(tmp_path, monkeypatch, directory: str = "models/gguf/flashnext"):
    home = tmp_path / "home"
    shard = home / directory / SHARD1
    world = serve_world(tmp_path, monkeypatch, peers=[{**llama_peer("flash", 8082), "lifecycle": {
        **llama_peer("flash", 8082)["lifecycle"], "model": str(shard)}}])
    (home / directory).mkdir(parents=True)
    flash_next(home / directory)
    world.rerecord()
    return world, home / directory


def _doctor(world):
    with patch("localharness.agent.context.probe_window", return_value=WindowProbe(None, None)):
        return runner.invoke(app, ["doctor", "--config-dir", str(world.g)])


def _rows(result) -> list[str]:
    return ROWS.findall(result.output)


def test_a_known_flash_next_file_and_no_install_prints_the_clone_and_setup_command(tmp_path, monkeypatch, mock_httpx):
    world, _ = _world(tmp_path, monkeypatch)

    result = _doctor(world)

    assert _rows(result) == ["Strata: not installed — " + COMMAND.format(gguf_dir="~/models/gguf/flashnext")]
    assert "./setup.sh --yes --no-start --family swift --model IQ3_XXS --gguf-dir ~/models/gguf/flashnext " \
           "--context 65536 --vision no --no-browser" in result.output
    out = result.output
    assert out.index("LLM endpoint reachable") < out.index("Strata:") < out.index("Config directory writable")
