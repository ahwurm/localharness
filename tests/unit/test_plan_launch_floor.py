"""The launch plan prices its window (or refuses) before anything starts: Flash-Next, the modes, the floor and the retry."""
from __future__ import annotations

import pytest

from localharness.config.models import AutosenseConfig, ManagedServerConfig
from localharness.provider import autosense, lifecycle, server
from localharness.provider.autosense import FreeMemory
from tests.unit.gguf_support import load_fixture, write_recorded, write_sparse

GIB = 2**30
AUTO = AutosenseConfig()
MEM_TOTAL = 64_139_128 * 1024
FLASH_MMPROJ = 907_543_520
BASE = "http://127.0.0.1:8080/v1"


def _world(monkeypatch, *, available=52.9, free=int(30.2 * GIB), memory=True):
    mem = FreeMemory(free, int(31.8 * GIB), "nvidia-smi") if memory else None
    monkeypatch.setattr(autosense, "free_memory", lambda ratios: mem)
    monkeypatch.setattr(autosense, "_read_meminfo", lambda: {"MemTotal": MEM_TOTAL, "MemAvailable": int(available * GIB)})


def _flash(tmp_path, **over):
    model = write_recorded(tmp_path, load_fixture("flash-next-iq3-xxs"))[0]
    mmproj = write_sparse(tmp_path / "mmproj.gguf", FLASH_MMPROJ)
    return ManagedServerConfig(**{"runtime": "llamacpp", "binary": "/x/llama-server", "model": str(model),
                                  "mmproj": str(mmproj), "served_name": "Flash-Next", "port": 8080, **over})


def _plan(spec, config_dir, floor=None):
    sensed = lifecycle.sense_launch(spec, config_dir, AUTO)
    return lifecycle.plan_launch(
        spec, sensed, server.launch_decision(spec, False), autosense=AUTO, api_key_held=False, floor=floor)


def _pair(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def test_flash_next_plans_its_window_with_the_experts_in_ram(tmp_path, monkeypatch):
    _world(monkeypatch)
    plan = _plan(_flash(tmp_path), tmp_path)
    argv = list(plan.argv)
    assert _pair(argv, "-c") == "262144" and _pair(argv, "--cache-ram") == "3849" and "--fit-ctx" not in argv
    assert plan.report.mode == "fit" and plan.report.placement.where == "experts in RAM"


def test_flash_next_without_the_ram_for_its_experts_is_one_refusal_sentence(tmp_path, monkeypatch):
    _world(monkeypatch, available=31.4)
    with pytest.raises(server.LaunchRefused) as refused:
        _plan(_flash(tmp_path), tmp_path)
    assert str(refused.value) == (
        "Flash-Next cannot run here: its experts need 40.0 GiB of RAM and only 22.2 GiB is free beside the "
        "system's reserve — close something that holds RAM, or pick a smaller quant.")


async def test_a_refused_activate_starts_and_writes_nothing(tmp_path, monkeypatch):
    _world(monkeypatch, available=31.4)
    touched: list[str] = []
    monkeypatch.setattr(server, "start_server", lambda *a, **kw: touched.append("start_server"))
    monkeypatch.setattr(server, "prepare_config_ini_home", lambda *a, **kw: touched.append("prepare_config_ini_home"))
    monkeypatch.setattr(server, "stop_server", lambda *a, **kw: touched.append("stop_server"))
    config = tmp_path / "cfg"
    config.mkdir()
    with pytest.raises(server.LaunchRefused, match="Flash-Next cannot run here"):
        await lifecycle.SpawnedProcessStrategy().activate(_flash(tmp_path), config, BASE, autosense=AUTO)
    assert touched == [] and list(config.iterdir()) == []
