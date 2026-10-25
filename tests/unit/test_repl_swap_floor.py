"""The REPL's /model hands the swap the live root's floor, and a swap that ended short never refits."""
from __future__ import annotations

import asyncio

import pytest

from localharness.provider import lifecycle
from tests.unit.test_repl_swap_wrapper import _build, _spy_swap, rig  # noqa: F401 — the fixture


@pytest.mark.asyncio
async def test_the_cold_peer_swap_carries_the_live_floor(tmp_path, rig, monkeypatch):
    _, repl, _, agent, _, _ = _build(tmp_path)
    agent.floor = lambda: 15_360
    seen = _spy_swap(monkeypatch, repl)

    await asyncio.wait_for(repl._handle_slash("/model flash-next"), 5)

    assert seen.calls[0][2]["floor"] == 15_360


@pytest.mark.asyncio
async def test_an_agent_without_a_floor_swaps_with_none(tmp_path, rig, monkeypatch):
    _, repl, _, _, _, _ = _build(tmp_path)
    seen = _spy_swap(monkeypatch, repl)

    await asyncio.wait_for(repl._handle_slash("/model flash-next"), 5)

    assert seen.calls[0][2]["floor"] is None


@pytest.mark.asyncio
async def test_a_swap_that_ended_short_never_refits_the_counter(tmp_path, rig, monkeypatch):
    _, repl, _, agent, _, primary = _build(tmp_path)
    agent.floor = lambda: 15_360
    refits = []

    async def refit(*a, **kw):
        refits.append(a)
        return ""

    repl._refresh_token_counter = refit

    async def below(incumbent, target, **kw):
        return lifecycle.SwapOutcome(None, incumbent, incumbent, RuntimeError("below"), "launch")

    monkeypatch.setattr(lifecycle, "swap", below)

    await asyncio.wait_for(repl._handle_slash("/model flash-next"), 5)

    assert refits == []


# --- the managed restart of the primary ------------------------------------------------------- #

from localharness.provider import autosense  # noqa: E402
from localharness.provider import server as managed_server  # noqa: E402
from tests.unit.test_launch_paths_repl import boundaries  # noqa: E402, F401 — the fixture
from tests.unit.test_model_swap_back import (  # noqa: E402, F401 — autouse fixtures
    _gguf,
    _liveness_not_launched,
    _llama,
    _no_machine_probes,
    _on_a_peer,
    _ready_as,
)

FLOOR = 15_360


@pytest.fixture
def order(monkeypatch):
    """Every call that decides, frees, stops or launches, in order; the floor each decision and launch saw."""
    events: list[str] = []
    seen: dict = {"refusal": [], "activate": [], "answer": None}
    real_free, real_stop = lifecycle.free_accelerator, lifecycle.SpawnedProcessStrategy.stop

    def refusal(target, incumbent, **kw):
        events.append("refusal")
        seen["refusal"].append((target, incumbent, kw))
        return seen["answer"]

    async def free(*a, **kw):
        events.append("free")
        return await real_free(*a, **kw)

    async def stop(self, spec, config_dir):
        events.append("stop")
        return await real_stop(self, spec, config_dir)

    monkeypatch.setattr(lifecycle, "swap_refusal", refusal)
    monkeypatch.setattr(lifecycle, "free_accelerator", free)
    monkeypatch.setattr(lifecycle.SpawnedProcessStrategy, "stop", stop)
    return events, seen


@pytest.mark.asyncio
async def test_a_refused_restart_target_stops_nothing_and_keeps_the_incumbent(tmp_path, boundaries, order):
    events, seen = order
    seen["answer"] = "Did not start Other-Q4: too big. qwen-27b keeps serving."
    path = _gguf(tmp_path)
    primary = _llama(served_name="qwen-27b")
    _, repl, channel, agent, _ = _on_a_peer(tmp_path, primary)
    agent.floor = lambda: FLOOR
    before = agent._llm.config.base_url

    await repl._handle_slash(f"/model {path}")

    assert events == ["refusal"] and boundaries.stops == [] and boundaries.cmds == []
    target, incumbent, kw = seen["refusal"][0]
    assert target.spec.model == str(path) and kw["floor"] == FLOOR
    assert incumbent is not None
    assert channel.messages[-1] == seen["answer"]
    assert agent._llm.config.base_url == before
    assert primary.model == "/x/primary.gguf"


@pytest.mark.asyncio
async def test_an_accepted_restart_asks_before_it_stops_and_launches_with_the_floor(
        tmp_path, boundaries, order, monkeypatch):
    events, seen = order
    path = _gguf(tmp_path)
    _, repl, _, agent, _ = _on_a_peer(tmp_path, _llama(served_name="qwen-27b"))
    agent.floor = lambda: FLOOR
    _ready_as(monkeypatch, "Other-Q4")
    real = lifecycle.SpawnedProcessStrategy.activate

    async def activate(self, spec, config_dir, base_url, **kw):
        events.append("activate")
        seen["activate"].append(kw["floor"])
        return await real(self, spec, config_dir, base_url, **kw)

    monkeypatch.setattr(lifecycle.SpawnedProcessStrategy, "activate", activate)

    await repl._handle_slash(f"/model {path}")

    assert events.index("refusal") < events.index("free") < events.index("stop") < events.index("activate")
    assert seen["activate"] == [FLOOR]


@pytest.mark.asyncio
async def test_a_restart_that_serves_below_the_floor_is_a_failed_launch(tmp_path, boundaries, order, monkeypatch):
    path = _gguf(tmp_path)
    primary = _llama(served_name="qwen-27b")
    _, repl, channel, agent, _ = _on_a_peer(tmp_path, primary)
    agent.floor = lambda: FLOOR
    served = autosense.WindowReport(
        served=4096, trained=None, fits=None, free_bytes=None, total_bytes=None, weights_bytes=None,
        asked=None, mode="record")

    async def activate(self, spec, config_dir, base_url, **kw):
        return lifecycle.LiveEndpoint(base_url, ["Other-Q4"], None, window=served)

    monkeypatch.setattr(lifecycle.SpawnedProcessStrategy, "activate", activate)

    await repl._handle_slash(f"/model {path}")

    assert any(m.startswith("Model swap failed: Other-Q4 served a 4,096-token window, below the 15,360 the "
                            "harness needs") for m in channel.messages)
    assert not any(m.startswith("Switched to") for m in channel.messages)
    assert boundaries.stops and primary.model == "/x/primary.gguf"


# --- localharness model and start ------------------------------------------------------------ #

from localharness.cli import model_cmd, start_serving  # noqa: E402
from localharness.config.loader import ConfigLoader  # noqa: E402
from tests.unit.test_model_cmd_unit_swap import cli  # noqa: E402, F401 — the fixture
from tests.unit.test_start_serving import _run  # noqa: E402
from tests.unit.test_start_serving import home, swap_rig  # noqa: E402, F401 — the fixtures
from tests.unit.test_start_serving_e2e import world  # noqa: E402, F401 — the fixture
from tests.unit.procfs_support import fake_proc  # noqa: E402, F401 — the fixture
from tests.unit.test_sense_serving import PRIMARY, PRIMARY_URL  # noqa: E402
from tests.unit.test_lifecycle_swap_unit import flash_ready, unit_incumbent  # noqa: E402


def _spy_run_swap(monkeypatch) -> list[dict]:
    seen: list[dict] = []
    real = start_serving.run_swap

    async def spy(*a, **kw):
        seen.append(kw)
        return await real(*a, **kw)

    monkeypatch.setattr(start_serving, "run_swap", spy)
    return seen


@pytest.mark.asyncio
async def test_run_swap_hands_the_floor_to_lifecycle_swap(swap_rig, monkeypatch):
    seen: list = []

    async def fake_swap(incumbent, target, **kw):
        seen.append(kw["floor"])
        return lifecycle.SwapOutcome(None, None, None, None, None)

    monkeypatch.setattr(lifecycle, "swap", fake_swap)

    await _run(swap_rig, None, "flash-next", floor=FLOOR)
    await _run(swap_rig, None, "flash-next")

    assert seen == [FLOOR, None]


@pytest.mark.asyncio
async def test_starts_swap_of_another_launch_passes_the_pre_launch_floor(world, monkeypatch):
    async def floor(agent_config, skills=(), **kw):
        return FLOOR

    monkeypatch.setattr(start_serving, "launch_floor", floor)
    seen = _spy_run_swap(monkeypatch)
    world.config(last_served=PRIMARY)
    world.answering = {PRIMARY_URL}

    await world.start_exit(preset="flash-next")

    assert [kw["floor"] for kw in seen] == [FLOOR]


def test_the_cli_swap_prices_the_root_agent_the_way_start_does(cli, monkeypatch):
    unit_incumbent(cli.world)
    flash = flash_ready(cli.world)
    seen = _spy_run_swap_sync(monkeypatch)

    async def floor(agent_config, skills=(), **kw):
        return FLOOR

    monkeypatch.setattr(start_serving, "launch_floor", floor)
    monkeypatch.setattr(ConfigLoader, "load_agent", lambda self, name: name)

    assert cli.run("flash-next").exit_code == 0
    flash.join()

    assert seen == [FLOOR]


def test_the_cli_swap_without_a_loadable_root_uses_the_cached_floor(cli, monkeypatch):
    unit_incumbent(cli.world)
    flash = flash_ready(cli.world)
    seen = _spy_run_swap_sync(monkeypatch)
    lifecycle.record_floor(cli.world.g, 12_000, {})

    def missing(self, name):
        raise RuntimeError("no such agent")

    monkeypatch.setattr(ConfigLoader, "load_agent", missing)

    assert cli.run("flash-next").exit_code == 0
    flash.join()

    assert seen == [12_000]


def _spy_run_swap_sync(monkeypatch) -> list:
    floors: list = []
    real = start_serving.run_swap

    async def spy(*a, **kw):
        floors.append(kw["floor"])
        return await real(*a, **kw)

    monkeypatch.setattr(start_serving, "run_swap", spy)
    return floors
