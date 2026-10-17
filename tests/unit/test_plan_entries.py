"""`model_ops.plan_entries`: the one list `/model` and `localharness model` render, built over `serve_world` records."""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from localharness.cli import model_ops
from localharness.cli.app import app
from localharness.config.loader import ConfigLoader
from localharness.provider import autosense, lifecycle, server
from localharness.provider.autosense import FreeMemory
from tests.unit.gguf_support import write_gguf
from tests.unit.test_serve_cmd import GIB, PRIMARY, PRIMARY_URL, llama_peer, serve_world
from tests.unit.test_tuning_derive import FLASH_NEXT

runner = CliRunner()
FLOOR = 13_824
PEER_URL = "http://127.0.0.1:8082/v1"


def flash(tmp_path: Path, name: str = "flash-next", port: int = 8082) -> dict:
    peer = llama_peer(name, port)
    peer["lifecycle"]["model"] = str(write_gguf(tmp_path / f"{name}.gguf", FLASH_NEXT))
    return peer


def harness_of(world):
    return ConfigLoader(config_dir=world.g, local_config_dir=None).load_harness()


def entries_of(world, live, *, floor=FLOOR, **kw):
    harness = harness_of(world)
    return model_ops.plan_entries(
        harness, live=live, config_dir=world.g, trust=model_ops.trust_as_recorded(world.g), floor=floor, **kw)


def asked(world, index, *, floor=FLOOR) -> int:
    harness = harness_of(world)
    spec = harness.extra_endpoints[index].lifecycle
    memory = FreeMemory(40 * GIB, 48 * GIB, "nvidia-smi")
    sensed = lifecycle.sense_launch(spec, world.g, harness.autosense, memory=memory)
    plan = lifecycle.plan_launch(spec, sensed, server.launch_decision(spec, False),
                                 autosense=harness.autosense, api_key_held=False, floor=floor)
    return plan.report.asked


@pytest.fixture
def world(tmp_path, monkeypatch):
    return serve_world(tmp_path, monkeypatch, peers=[flash(tmp_path)])


SERVING = {PRIMARY_URL: (PRIMARY,), PEER_URL: None}


def test_a_serving_primary_and_a_cold_peer_are_one_list(world):
    entries = entries_of(world, SERVING)

    window = asked(world, 0)
    assert model_ops.entry_lines(entries) == [
        "Models:",
        "▸ llama.cpp · 127.0.0.1:8080",
        f"  1. {PRIMARY}  (serving)  ● [active]",
        "▸ this box can launch",
        f"  2. flash-next  llama.cpp · {window:,}"]
    assert [e.name for e in entries.numbered()] == [PRIMARY, "flash-next"]


def test_a_cold_peers_window_is_the_plans(world):
    cold = entries_of(world, SERVING).numbered()[1]

    assert (cold.kind, cold.window, cold.runtime) == ("cold", asked(world, 0), "llamacpp")


def test_the_list_reads_free_memory_once(world, monkeypatch):
    reads = []
    real = autosense.free_memory
    monkeypatch.setattr(autosense, "free_memory", lambda ratios: reads.append(1) or real(ratios))

    entries_of(world, SERVING)

    assert len(reads) == 1


def test_a_number_picks_the_entry_it_is_printed_beside(world):
    entries = entries_of(world, SERVING)

    assert [entries.pick(str(n)) for n in (1, 2)] == list(entries.numbered())
    assert entries.pick("2").record.name == "flash-next"
    assert entries.pick("3") is None and entries.pick("0") is None


def test_localharness_model_prints_the_list_from_plan_entries(world, monkeypatch):
    monkeypatch.setattr(model_ops, "list_live_models",
                        lambda url, *a, **kw: ([PRIMARY], True) if url == PRIMARY_URL else ([], False))

    result = runner.invoke(app, ["model", "--config-dir", str(world.g)])

    assert result.exit_code == 0, result.output
    assert "▸ this box can launch" in result.output
    assert f"  2. flash-next  llama.cpp · {asked(world, 0, floor=None):,}" in result.output
    assert "start will launch it" not in result.output


def test_localharness_model_with_a_number_resolves_through_the_list(world, monkeypatch):
    monkeypatch.setattr(model_ops, "list_live_models",
                        lambda url, *a, **kw: ([PRIMARY], True) if url == PRIMARY_URL else ([], False))
    seen = []
    monkeypatch.setattr(model_ops, "resolve_serving", lambda h, t, name, **kw: seen.append(name) or None)

    runner.invoke(app, ["model", "2", "--config-dir", str(world.g)])

    assert seen == ["flash-next"]
