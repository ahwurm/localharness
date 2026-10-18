"""The list represents a sensed Strata install on the Flash-Next file's one entry and offers Strata once when the llama.cpp plan is refused or slow."""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from localharness.cli import model_ops
from localharness.cli.app import app
from tests.unit.strata_fixture import CONFIG_NAME, flash_next, strata_tree
from tests.unit.test_plan_entries import SERVING, entries_of
from tests.unit.test_serve_cmd import PRIMARY, PRIMARY_URL, llama_peer, serve_world

runner = CliRunner()
NAME = "flash-next"


def flash_world(tmp_path, monkeypatch, *, extra=()):
    shard1, mmproj = flash_next(tmp_path)
    peer = llama_peer(NAME, 8082)
    peer["lifecycle"].update(model=str(shard1), mmproj=str(mmproj))
    return serve_world(tmp_path, monkeypatch, peers=[peer, *extra]), shard1


def install(world, shard1: Path, **kw) -> Path:
    return strata_tree(world.home / "src", model=shard1, **kw)


def line_of(entries, name=NAME) -> str:
    return next(line for line in model_ops.entry_lines(entries) if f" {name} " in line or line.endswith(f" {name}"))


def test_a_sensed_install_marks_the_entry_and_makes_the_strata_set_up_pickable(tmp_path, monkeypatch):
    world, shard1 = flash_world(tmp_path, monkeypatch)
    install(world, shard1)

    entries = entries_of(world, SERVING)

    assert line_of(entries).endswith(f"{NAME}  llama.cpp · experts in RAM · 262,144 · Strata available")
    picked = entries.pick(f"{NAME}-strata")
    assert (picked.kind, picked.runtime, picked.name) == ("setup", "strata", f"{NAME}-strata")
    assert (picked.setup.config, picked.setup.real, picked.setup.record.name) == (CONFIG_NAME, str(shard1), NAME)


def test_localharness_model_prints_the_suffix(tmp_path, monkeypatch):
    world, shard1 = flash_world(tmp_path, monkeypatch)
    install(world, shard1)
    monkeypatch.setattr(model_ops, "list_live_models",
                        lambda url, *a, **kw: ([PRIMARY], True) if url == PRIMARY_URL else ([], False))

    result = runner.invoke(app, ["model", "--config-dir", str(world.g)])

    assert result.exit_code == 0, result.output
    assert any(line.endswith(" · Strata available") and NAME in line for line in result.output.splitlines())
