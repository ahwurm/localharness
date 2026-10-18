"""`Set it up and switch?` adopts a sensed Strata install as a second preset beside the llama.cpp one, and every record on a file answers to its own name."""
from __future__ import annotations

import shutil

import pytest
import yaml

from localharness.agent.gate_types import Decision
from localharness.cli import interview, model_cmd, workspace
from localharness.config import trust
from localharness.config.loader import ConfigLoader
from localharness.provider import autosense, lifecycle, speed_stats
from localharness.provider.autosense import FreeMemory
from tests.unit.interview_support import scripted
from tests.unit.strata_fixture import CONFIG_NAME, PROJECTOR, flash_next, strata_tree
from tests.unit.test_launch_paths_repl import _liveness_not_launched, _no_machine_probes  # noqa: F401 — autouse
from tests.unit.test_plan_entries import SERVING, entries_of, twin_files
from tests.unit.test_plan_entries_strata import NAME, fake_live, flash_world, install, invoke, swapping
from tests.unit.test_repl_model_cmd import _xrepl
from tests.unit.test_serve_cmd import GIB, PRIMARY, PRIMARY_URL, llama_peer, serve_world
from tests.unit.test_setup_and_switch import terminal

STRATA = f"{NAME}-strata"
PEER_URL = "http://127.0.0.1:8082/v1"
STRATA_URL = "http://127.0.0.1:8083/v1"


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


@pytest.fixture
def box(tmp_path, monkeypatch):
    world, shard1 = flash_world(tmp_path, monkeypatch)
    pin_box(monkeypatch)
    world.engine = install(world, shard1)
    world.shard = shard1
    fake_live(monkeypatch)
    monkeypatch.setattr(model_cmd, "_own_unit", lambda cfg: None)
    monkeypatch.setattr(workspace, "port_holder", lambda port: None)
    terminal(monkeypatch)
    world.before = state(world)
    return world


def state(world) -> tuple[bytes, bytes, dict | None]:
    files = (world.g / "config.yaml", world.g / "overrides.yaml")
    return (*(p.read_bytes() if p.exists() else b"" for p in files), trust.recorded_machine(world.g))


def said(result) -> str:
    return " ".join(result.output.split())


OFF_TERMINAL = f"{STRATA} is not set up — localharness model --setup sets it up"


def test_not_now_writes_nothing_and_switches_nothing(box, monkeypatch):
    seen = swapping(monkeypatch)
    scripted(monkeypatch, "not-now")

    result = invoke(box, STRATA)

    assert result.exit_code == 0, result.output
    assert state(box) == box.before and seen == []


def test_off_a_terminal_nothing_is_set_up(box, monkeypatch):
    terminal(monkeypatch, on=False)
    asked = scripted(monkeypatch)

    result = invoke(box, STRATA)

    assert result.exit_code == 2 and OFF_TERMINAL in said(result)
    assert asked.asked == [] and state(box) == box.before


def test_an_agents_command_on_a_terminal_sets_nothing_up(box, monkeypatch):
    monkeypatch.setenv("LOCALHARNESS_COMMIT", "1")
    asked = scripted(monkeypatch)

    result = invoke(box, STRATA)

    assert result.exit_code == 2 and OFF_TERMINAL in said(result)
    assert asked.asked == [] and state(box) == box.before


def test_an_install_gone_since_the_listing_is_not_set_up(box, monkeypatch):
    def vanish(*a, **k):
        shutil.rmtree(box.engine)
        return interview.Answer(("set-up",), False, True)

    monkeypatch.setattr(interview, "ask", vanish)
    seen = swapping(monkeypatch)

    result = invoke(box, STRATA)

    assert result.exit_code == 2 and f"{box.shard} is gone — nothing set up" in said(result)
    assert state(box) == box.before and seen == []


@pytest.fixture
def session(box):
    harness = ConfigLoader(config_dir=box.g, local_config_dir=None).load_harness()
    repl, channel, agent, _ = _xrepl(box.g, harness, {PRIMARY_URL: [PRIMARY], PEER_URL: "unreachable", STRATA_URL: "unreachable"})
    agent._llm.config.base_url, agent._llm.config.model = PRIMARY_URL, PRIMARY
    agent._llm.config.provider_type = "llamacpp"
    channel.requests = []
    return harness, repl, channel


def answering(channel, *kinds, local_operator=True, can_ask=True):
    channel.local_operator, channel.can_ask = local_operator, can_ask
    kinds = list(kinds)

    async def ask(request):
        channel.requests.append(request)
        return Decision(kinds.pop(0))

    channel.ask_permission = ask


@pytest.fixture
def swaps(monkeypatch):
    targets = []

    async def fake_swap(incumbent, target, **kw):
        targets.append(target)
        return lifecycle.SwapOutcome(None, None, None, RuntimeError("stop here"), "launch")

    monkeypatch.setattr(lifecycle, "swap", fake_swap)
    return targets


@pytest.mark.asyncio
async def test_the_repl_asks_through_the_channel_writes_the_strata_peer_and_launches_it(box, session, swaps):
    harness, repl, channel = session
    answering(channel, "allow_once")

    await repl._handle_slash(f"/model {STRATA}")

    [request] = channel.requests
    assert (request.grantable, request.reason) == (False, "Set it up and switch?")
    assert request.tool_params == {"model": STRATA}
    assert any(m.startswith(f"• {STRATA} (Strata, +{PROJECTOR}) ← {box.shard}") for m in channel.messages)
    assert STRATA in [ep.name for ep in harness.extra_endpoints]
    [target] = swaps
    assert (target.spec.runtime, target.spec.model, target.spec.port) == ("strata", str(box.shard), 8083)
    assert uncovered(box) == []


@pytest.mark.asyncio
async def test_a_rejection_in_the_repl_writes_and_launches_nothing(box, session, swaps):
    _, repl, channel = session
    answering(channel, "reject_once")

    await repl._handle_slash(f"/model {STRATA}")

    assert len(channel.requests) == 1 and swaps == []
    assert state(box) == box.before


@pytest.mark.asyncio
@pytest.mark.parametrize("operator,can_ask", [(False, True), (True, False), (False, False)])
async def test_a_channel_with_no_local_operator_is_told_where_to_set_it_up(box, session, swaps, operator, can_ask):
    _, repl, channel = session
    answering(channel, local_operator=operator, can_ask=can_ask)

    await repl._handle_slash(f"/model {STRATA}")

    assert channel.messages[-1] == OFF_TERMINAL
    assert channel.requests == [] and swaps == [] and state(box) == box.before


@pytest.mark.asyncio
async def test_an_install_gone_since_the_listing_is_not_launched_by_the_repl(box, session, swaps):
    _, repl, channel = session

    async def vanish(request):
        shutil.rmtree(box.engine)
        return Decision("allow_once")

    answering(channel)
    channel.ask_permission = vanish

    await repl._handle_slash(f"/model {STRATA}")

    assert "is gone — nothing set up" in channel.messages[-1] and swaps == []
    assert state(box) == box.before


def pair_world(tmp_path, monkeypatch, kind: str):
    """A file with two records: ``llamacpp`` (two llama.cpp names) or ``strata`` (a llama.cpp record and a Strata one); the two names in config order."""
    pin_box(monkeypatch)
    if kind == "llamacpp":
        world = serve_world(tmp_path, monkeypatch, peers=twin_files(tmp_path))
        return world, ("x", "x-strata")
    shard1, _ = flash_next(tmp_path)
    engine = strata_tree(tmp_path / "engine", model=shard1)
    llama = llama_peer(NAME, 8082)
    llama["lifecycle"].update(model=str(shard1), mmproj=str(shard1.parent / PROJECTOR))
    strata = {"name": STRATA, "base_url": STRATA_URL, "provider_type": "strata", "gpu": True,
              "lifecycle": {"runtime": "strata", "engine_dir": str(engine), "engine_config": CONFIG_NAME,
                            "model": str(shard1), "served_name": STRATA, "port": 8083, "gpu": True}}
    return serve_world(tmp_path, monkeypatch, peers=[llama, strata]), (NAME, STRATA)


PAIRS = pytest.mark.parametrize("kind", ["llamacpp", "strata"])


def cold_of(world, **kw):
    entries = entries_of(world, {PRIMARY_URL: (PRIMARY,)}, **kw)
    return [e for group in entries.groups for e in group.entries if e.kind == "cold"] + [
        e for e in entries.not_offered if e.kind == "cold"]


@PAIRS
def test_every_record_on_a_file_is_picked_by_its_own_name(tmp_path, monkeypatch, kind):
    world, (first, second) = pair_world(tmp_path, monkeypatch, kind)
    entries = entries_of(world, {PRIMARY_URL: (PRIMARY,)})

    picks = {name: entries.pick(name) for name in (first, second)}

    assert {name: (e.record.name, e.name) for name, e in picks.items()} == {first: (first, first), second: (second, second)}
    assert picks[first].others == (second,) and picks[second].others == (first,)


@PAIRS
def test_the_cli_launches_the_record_a_name_names(tmp_path, monkeypatch, kind):
    world, names = pair_world(tmp_path, monkeypatch, kind)
    fake_live(monkeypatch)
    seen = swapping(monkeypatch)

    for name in names:
        assert invoke(world, name).exit_code == 0

    assert seen == list(names)


@PAIRS
@pytest.mark.asyncio
async def test_the_repl_launches_the_record_a_name_names(tmp_path, monkeypatch, kind):
    world, names = pair_world(tmp_path, monkeypatch, kind)
    fake_live(monkeypatch)
    targets = []

    async def fake_swap(incumbent, target, **kw):
        targets.append(target.peer)
        return lifecycle.SwapOutcome(None, None, None, RuntimeError("stop here"), "launch")

    monkeypatch.setattr(lifecycle, "swap", fake_swap)
    harness = ConfigLoader(config_dir=world.g, local_config_dir=None).load_harness()
    repl, channel, agent, _ = _xrepl(world.g, harness, {PRIMARY_URL: [PRIMARY], PEER_URL: "unreachable", STRATA_URL: "unreachable"})
    agent._llm.config.base_url, agent._llm.config.model = PRIMARY_URL, PRIMARY
    agent._llm.config.provider_type = "llamacpp"

    for name in names:
        await repl._handle_slash(f"/model {name}")

    assert targets == list(names)


@PAIRS
def test_the_first_in_config_order_is_shown_until_the_person_lands_on_the_other(tmp_path, monkeypatch, kind):
    world, (first, second) = pair_world(tmp_path, monkeypatch, kind)

    [before] = cold_of(world)
    speed_stats.record_last_served(speed_stats.default_speed_stats_path(world.g), second)
    [after] = cold_of(world)

    assert (before.name, before.others) == (first, (second,))
    assert (after.name, after.others) == (second, (first,))
