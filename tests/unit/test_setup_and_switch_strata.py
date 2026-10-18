"""`Set it up and switch?` adopts a sensed Strata install as a second preset beside the llama.cpp one, and every record on a file answers to its own name."""
from __future__ import annotations

import yaml

from localharness.cli import workspace
from localharness.config import trust
from localharness.provider import autosense
from localharness.provider.autosense import FreeMemory
from tests.unit.interview_support import scripted
from tests.unit.strata_fixture import CONFIG_NAME, PROJECTOR
from tests.unit.test_launch_paths_repl import _liveness_not_launched, _no_machine_probes  # noqa: F401 — autouse
from tests.unit.test_plan_entries import SERVING, entries_of
from tests.unit.test_plan_entries_strata import NAME, fake_live, flash_world, install, invoke, swapping
from tests.unit.test_serve_cmd import GIB
from tests.unit.test_setup_and_switch import terminal

STRATA = f"{NAME}-strata"


def pin_box(monkeypatch) -> None:
    monkeypatch.setattr(autosense, "free_memory", lambda ratios: FreeMemory(int(30.2 * GIB), int(31.8 * GIB), "nvidia-smi"))
    monkeypatch.setattr(autosense, "_read_meminfo", lambda: {"MemTotal": 61 * GIB, "MemAvailable": int(52.9 * GIB)})


def overrides(world) -> list[dict]:
    path = world.g / "overrides.yaml"
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("extra_endpoints", []) if path.exists() else []


def cold_entries(world):
    return [e for e in entries_of(world, SERVING).numbered() if e.kind == "cold"]


def uncovered(world) -> list[dict]:
    return workspace.uncovered(trust.recorded_machine(world.g)["entries"], trust.machine_snapshot(world.g))


def test_a_person_adopts_the_strata_preset_beside_the_llamacpp_one(tmp_path, monkeypatch):
    world, shard1 = flash_world(tmp_path, monkeypatch)
    pin_box(monkeypatch)
    install(world, shard1)
    fake_live(monkeypatch)
    terminal(monkeypatch)
    seen = swapping(monkeypatch)
    asked = scripted(monkeypatch, "set-up")
    config_before = (world.g / "config.yaml").read_bytes()

    result = invoke(world, STRATA)

    assert result.exit_code == 0, result.output
    [question] = asked.asked
    assert question.question == "Set it up and switch?"
    assert f"• {STRATA} (Strata, +{PROJECTOR}) ← {shard1} · ~/src/Strata · {CONFIG_NAME}" in " ".join(result.output.split())
    llama, strata = overrides(world)
    assert strata["provider_type"] == "strata" and strata["gpu"] is True and strata["name"] == STRATA
    expected = {"runtime": "strata", "engine_dir": "~/src/Strata", "engine_config": CONFIG_NAME, "model": str(shard1),
                "mmproj": str(shard1.parent / PROJECTOR), "served_name": STRATA, "port": 8083, "gpu": True}
    assert {k: strata["lifecycle"][k] for k in expected} == expected
    assert (world.g / "config.yaml").read_bytes() == config_before
    assert llama == yaml.safe_load(config_before)["extra_endpoints"][0]
    assert uncovered(world) == []
    assert seen == [STRATA]
    [shown] = cold_entries(world)
    assert (shown.name, shown.others) == (NAME, (STRATA,))
