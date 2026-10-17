# ruff: noqa: F811
"""`start` says what happened: the capability probe waits for start's own launch.

Driven as tests/unit/test_start_serving_e2e.py drives it: the sensor's `list_live_models`, the three
process boundaries a launch crosses and the `/proc` side are faked; no socket is opened and no service
manager is asked.
"""
from __future__ import annotations

import logging

import pytest
import yaml

from localharness.cli import model_ops, start_cmd
from localharness.cli.start_cmd import _start_managed_server
from localharness.config.models import AutosenseConfig
from localharness.cli.start_cmd import _probe_llm as real_probe
from localharness.provider.autosense import FreeMemory
from localharness.provider.client import CapabilityResult, LLMClient
from tests.unit.launch_support import stub_liveness
from tests.unit.procfs_support import fake_proc  # noqa: F401
from tests.unit.test_sense_serving import PEER_URL, PRIMARY, PRIMARY_URL, home  # noqa: F401
from tests.unit.test_launch_fit_retry import BASE, Box
from tests.unit.test_launch_line_clause import GIB, _World
from tests.unit.test_launch_plan import SWIFT, _gguf_spec
from tests.unit.test_start_serving_e2e import FLASH_LOW, World, _flash_launch


@pytest.fixture
def world(home, monkeypatch, fake_proc):  # noqa: F811
    w = World(home, monkeypatch)
    w.served_ids = {PEER_URL: ["flash-next"], PRIMARY_URL: [PRIMARY]}
    return w


@pytest.fixture
def trace(world, monkeypatch):
    """The real `_probe_llm` over a fake `detect_capabilities` that writes into `world.events`."""
    client_log = logging.getLogger("localharness.provider.client")

    class Mark(logging.Handler):
        def emit(self, record):
            if "Capability probe inconclusive" in record.getMessage():
                world.events.append("inconclusive")

    async def detect(self):
        world.events.append("detect")
        if self.config.base_url in world.answering:
            return CapabilityResult("native", 262_144, True, 1.0, None, True)
        client_log.warning("Capability probe inconclusive (attempt 1/3): %s — retrying", "unreachable")
        return CapabilityResult("xml", 0, False, 1.0, "Connection error.", False)

    handler = Mark()
    client_log.addHandler(handler)
    monkeypatch.setattr(LLMClient, "detect_capabilities", detect)
    monkeypatch.setattr(start_cmd, "_probe_llm", real_probe)
    yield
    client_log.removeHandler(handler)


def _order(world) -> list[str]:
    return [e.split(":")[0] for e in world.events if not e.startswith("lock")]


async def test_a_cold_box_probes_only_after_its_own_launch_and_logs_no_inconclusive_before_it(
        world, trace):
    world.config(active=FLASH_LOW, last_served=PRIMARY)

    await world.start()

    assert _order(world) == ["start", "wait", "detect"]


async def test_an_attach_probes_at_once_and_launches_nothing(world, trace):
    world.config(active=FLASH_LOW, last_served="flash-next")
    world.answering = {PEER_URL}

    await world.start()

    assert _order(world)[:1] == ["detect"] and world.starts == []
    assert "start" not in _order(world)


async def test_an_attach_only_config_probes_at_once_and_keeps_its_could_not_reach_message(
        world, trace):
    (world.home / "config.yaml").write_text(yaml.safe_dump({
        "version": "1",
        "provider": {"provider_type": "llamacpp", "base_url": PRIMARY_URL, "default_model": PRIMARY,
                     "api_key": "none"}}), encoding="utf-8")
    (world.home / "overrides.yaml").unlink(missing_ok=True)

    assert await world.start_exit() == 1

    assert _order(world) == ["detect", "inconclusive"]
    assert any("model endpoint unreachable at" in line for line in world.err_lines)


async def test_a_loading_server_that_answers_no_model_list_is_not_probed_before_wait_ready(
        world, fake_proc, trace, monkeypatch):
    stub_liveness(monkeypatch, True)
    world.config(active=FLASH_LOW, last_served="flash-next")
    _flash_launch(fake_proc, world.home)

    def loading(base_url, timeout=3.0, **kw):
        if base_url == PEER_URL:
            raise model_ops.MalformedModelListError("503")
        return ([], False)

    monkeypatch.setattr(model_ops, "list_live_models", loading)

    await world.start()

    assert _order(world) == ["wait", "detect"]


RESUME = "your last /model choice"


@pytest.fixture
def launch(monkeypatch):
    box = Box(monkeypatch)
    box.mem = FreeMemory(40 * GIB, 48 * GIB, "nvidia-smi")
    box.weights = 18 * GIB
    return _World(monkeypatch, box)


async def test_a_launch_of_the_last_choice_says_so_beside_the_derived_clause(tmp_path, launch):
    spec = _gguf_spec(tmp_path, SWIFT)
    assert await _start_managed_server(
        spec, tmp_path, BASE, {}, autosense=AutosenseConfig(), resume=True) is None
    assert launch.out == [f"Managed llama.cpp is not running — starting it ({RESUME}; kv q8_0 · mtp (1 layer); "
                          "model load can take several minutes)..."]


async def test_a_launch_that_is_not_the_last_choice_carries_no_resume_clause(tmp_path, launch):
    spec = _gguf_spec(tmp_path, SWIFT)
    assert await _start_managed_server(spec, tmp_path, BASE, {}, autosense=AutosenseConfig()) is None
    assert RESUME not in launch.out[0]


async def test_an_attach_on_the_active_endpoint_rung_prints_no_clause(world):
    world.config(active=FLASH_LOW, last_served=PRIMARY)
    world.answering = {PEER_URL}

    lines = await world.start()

    assert not [x for x in lines if x.startswith("Serving ") or RESUME in x]
    assert world.starts == []


async def test_a_cold_start_on_the_active_endpoint_rung_names_the_last_choice_in_its_launch_line(world):
    world.config(active=FLASH_LOW, last_served=PRIMARY)

    lines = await world.start()

    (launch_line,) = [x for x in lines if "is not running" in x]
    assert launch_line.startswith(f"Managed llama.cpp (flash-next) is not running — starting it ({RESUME}")


async def test_a_cold_start_on_the_server_rung_carries_no_clause(world):
    world.config()

    lines = await world.start()

    (launch_line,) = [x for x in lines if "is not running" in x]
    assert RESUME not in launch_line
