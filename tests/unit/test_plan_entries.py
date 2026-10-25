"""`model_ops.plan_entries`: the one list `/model` and `localharness model` render, built over `serve_world` records."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from localharness.cli import model_ops, workspace
from localharness.cli.app import app
from localharness.config.loader import ConfigLoader
from localharness.provider import autosense, lifecycle, procfs, server, speed_stats
from localharness.provider.autosense import FreeMemory
from tests.unit.gguf_support import write_gguf
from tests.unit.test_serve_cmd import GIB, PRIMARY, PRIMARY_URL, attach_peer, docker_peer, llama_peer, serve_world
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


# ====================================================================== owners, the foreign holder, peers once, order

PEER_ONLY = {PRIMARY_URL: None, PEER_URL: None}
LAUNCH = procfs.Launch(pid=4321, start=None, port=8080, owner="spawned", verified=True)
UNIT_LAUNCH = dataclasses.replace(LAUNCH, owner="unit")
LLM_SERVICE = workspace.Holder(
    "llm.service", 77, "llama-server", 1000, SimpleNamespace(unit_file_state="enabled", load_state="loaded"))


def headers(entries) -> list[str]:
    return [g.header for g in entries.groups]


def test_the_group_names_the_harness_as_owner_of_its_own_launch(world, monkeypatch):
    monkeypatch.setattr(lifecycle, "own_launch", lambda config_dir, specs: LAUNCH)

    assert headers(entries_of(world, SERVING))[0] == "llama.cpp · 127.0.0.1:8080 · the harness"


def test_the_group_names_autostart_for_the_units_main_process(world, monkeypatch):
    monkeypatch.setattr(lifecycle, "own_launch", lambda config_dir, specs: LAUNCH)
    monkeypatch.setattr(procfs, "read_launch", lambda config_dir, *, expect=None, main_pid=None: (
        UNIT_LAUNCH if main_pid == 4321 else LAUNCH))

    entries = entries_of(world, SERVING, unit=SimpleNamespace(main_pid=4321))

    assert headers(entries)[0] == "llama.cpp · 127.0.0.1:8080 · autostart"


def test_the_group_names_a_foreign_unit_as_yours(world, monkeypatch):
    monkeypatch.setattr(workspace, "port_holder", lambda port: LLM_SERVICE)

    entries = entries_of(world, SERVING)

    assert headers(entries) == ["llama.cpp · 127.0.0.1:8080 · llm.service (yours)", "available once llm.service is stopped"]
    assert entries.foreign == workspace.foreign_unit_sentence(LLM_SERVICE, served=PRIMARY, port=8080)
    assert entries.foreign_key == ("llm.service", PRIMARY)


def test_a_foreign_holder_leaves_the_launch_group_unnumbered_and_unpickable(world, monkeypatch):
    monkeypatch.setattr(workspace, "port_holder", lambda port: LLM_SERVICE)

    entries = entries_of(world, SERVING)

    sentence = workspace.foreign_unit_sentence(LLM_SERVICE, served=PRIMARY, port=8080)
    assert [e.name for e in entries.numbered()] == [PRIMARY]
    assert model_ops.entry_lines(entries)[3:] == [
        "▸ available once llm.service is stopped",
        f"     flash-next  llama.cpp · {entries.groups[1].entries[0].window:,}"]
    assert entries.pick("flash-next") == sentence
    assert entries.pick("2") is None


def test_a_foreign_holder_prices_against_the_card_less_a_gibibyte(world, monkeypatch):
    monkeypatch.setattr(workspace, "port_holder", lambda port: LLM_SERVICE)
    asks = []
    real = lifecycle.memory_after_launch
    monkeypatch.setattr(lifecycle, "memory_after_launch", lambda auto, launch, *, stopping=False: (
        asks.append((launch, stopping)) or real(auto, launch, stopping=stopping)))

    entries_of(world, SERVING)

    assert asks == [(None, True)]


def test_a_down_lifecycle_peer_is_one_cold_entry_and_no_note(world):
    entries = entries_of(world, PEER_ONLY)

    assert [e.name for e in entries.numbered()] == [PRIMARY, "flash-next"]
    assert entries.notes == ()


def test_a_down_attach_only_peer_is_one_note_and_no_entry(tmp_path, monkeypatch):
    world = serve_world(tmp_path, monkeypatch, peers=[attach_peer("attached", 1234)])

    entries = entries_of(world, {PRIMARY_URL: (PRIMARY,), "http://127.0.0.1:1234/v1": None})

    assert entries.notes == ("skipped attached (http://127.0.0.1:1234/v1): unreachable",)
    assert [e.name for e in entries.numbered()] == [PRIMARY]


def test_a_model_an_earlier_endpoint_serves_is_listed_once(tmp_path, monkeypatch):
    peers = [attach_peer("first", 1234), attach_peer("second", 1235)]
    world = serve_world(tmp_path, monkeypatch, peers=peers)
    urls = {"http://127.0.0.1:1234/v1": ("shared", PRIMARY), "http://127.0.0.1:1235/v1": ("shared", "own")}

    entries = entries_of(world, {PRIMARY_URL: (PRIMARY,), **urls})

    assert [e.name for e in entries.numbered()] == [PRIMARY, "shared", "own"]
    assert entries.notes == (
        "second (http://127.0.0.1:1235/v1) also serves 'shared' — hidden by first (first configured wins)",)


def test_entries_keep_config_order_with_notes_last(tmp_path, monkeypatch):
    peers = [flash(tmp_path, "cold-one", 8082), attach_peer("up", 1234), attach_peer("gone", 1235)]
    world = serve_world(tmp_path, monkeypatch, peers=peers)
    live = {PRIMARY_URL: (PRIMARY,), PEER_URL: None, "http://127.0.0.1:1234/v1": ("peer-model",),
            "http://127.0.0.1:1235/v1": None}

    entries = entries_of(world, live)

    assert headers(entries) == ["llama.cpp · 127.0.0.1:8080", "LM Studio · 127.0.0.1:1234", "this box can launch"]
    assert [e.name for e in entries.numbered()] == [PRIMARY, "peer-model", "cold-one"]
    assert model_ops.entry_lines(entries)[-1] == "  · skipped gone (http://127.0.0.1:1235/v1): unreachable"


def test_the_serving_endpoint_comes_first_even_when_it_is_a_peer(tmp_path, monkeypatch):
    world = serve_world(tmp_path, monkeypatch, peers=[attach_peer("up", 1234)])
    live = {PRIMARY_URL: (PRIMARY,), "http://127.0.0.1:1234/v1": ("peer-model",)}

    entries = entries_of(world, live, current=("http://127.0.0.1:1234/v1", "peer-model"))

    assert [e.name for e in entries.numbered()] == ["peer-model", PRIMARY]
    assert [e.active for e in entries.numbered()] == [True, False]


def test_no_record_and_no_answer_gives_no_numbered_entry(tmp_path, monkeypatch):
    world = serve_world(tmp_path, monkeypatch)
    world.edit_config(lambda d: d.pop("server"))
    world.rerecord()

    entries = entries_of(world, {PRIMARY_URL: None})

    assert entries.numbered() == () and entries.groups == ()


def test_without_peers_the_list_is_the_serving_group_and_the_launch_group_only(world):
    world.edit_config(lambda d: d.update(extra_endpoints=[]))
    world.rerecord()

    serving = entries_of(world, {PRIMARY_URL: (PRIMARY,)})
    down = entries_of(world, {PRIMARY_URL: None})

    assert headers(serving) == ["llama.cpp · 127.0.0.1:8080"] and serving.notes == ()
    assert headers(down) == ["this box can launch"] and down.notes == ()


def test_a_name_with_an_escape_byte_prints_it_as_four_characters(tmp_path, monkeypatch):
    world = serve_world(tmp_path, monkeypatch, peers=[flash(tmp_path, "bad\x1b[2Jname", 8082), attach_peer("a\x1b[31mb", 1234)])
    live = {PRIMARY_URL: (PRIMARY,), PEER_URL: None, "http://127.0.0.1:1234/v1": None}

    out = "\n".join(model_ops.entry_lines(entries_of(world, live)))

    assert "\x1b" not in out
    assert "bad\\x1b[2Jname" in out and "skipped a\\x1b[31mb" in out


# ====================================================================== one entry per file, refusals, every record by name


def twin_files(tmp_path: Path) -> list[dict]:
    first = flash(tmp_path, "x", 8082)
    second = flash(tmp_path, "x-strata", 8083)
    link = tmp_path / "x-strata.gguf"
    link.unlink()
    link.symlink_to(first["lifecycle"]["model"])
    second["lifecycle"]["model"] = str(link)
    return [first, second]


def test_two_records_on_one_file_are_one_entry_shown_as_the_first_in_config_order(tmp_path, monkeypatch):
    world = serve_world(tmp_path, monkeypatch, peers=twin_files(tmp_path))

    entries = entries_of(world, {PRIMARY_URL: (PRIMARY,)})

    cold = entries.groups[1].entries
    assert [(e.name, e.others) for e in cold] == [("x", ("x-strata",))]
    assert entries.pick("x").record.name == "x" and entries.pick("x-strata").record.name == "x-strata"


def test_the_record_last_served_is_the_one_shown(tmp_path, monkeypatch):
    world = serve_world(tmp_path, monkeypatch, peers=twin_files(tmp_path))
    speed_stats.record_last_served(speed_stats.default_speed_stats_path(world.g), "x-strata")

    cold = entries_of(world, {PRIMARY_URL: (PRIMARY,)}).groups[1].entries

    assert [(e.name, e.others) for e in cold] == [("x-strata", ("x",))]


def test_a_withheld_record_is_a_not_offered_line(world):
    world.edit_config(lambda d: d["extra_endpoints"][0]["lifecycle"].update(binary="/other/llama-server"))

    entries = entries_of(world, SERVING)

    assert entries.numbered()[1:] == ()
    assert model_ops.entry_lines(entries)[-1] == "  not offered: flash-next — its record has changes you have not confirmed"


def test_a_plan_the_floor_refuses_is_a_not_offered_line(world, monkeypatch):
    refusal = autosense.FLOOR_REFUSAL.format(name="flash-next", window=autosense.window_phrase(9216), floor=FLOOR,
                                        remedy=autosense.FLOOR_REMEDY)

    def refuse(*args, **kw):
        raise server.LaunchRefused(refusal)

    monkeypatch.setattr(lifecycle, "plan_launch", refuse)

    entries = entries_of(world, SERVING)

    assert model_ops.entry_lines(entries)[-1] == (
        "  not offered: flash-next — fits only a 9,216-token window here and the harness needs at least 13,824 — free GPU memory, or pick a smaller model")
    assert entries.pick("flash-next") == refusal


def test_one_record_that_cannot_be_planned_does_not_fail_the_list(world, monkeypatch):
    def broken(*args, **kw):
        raise OSError("boom")

    monkeypatch.setattr(lifecycle, "sense_launch", broken)

    lines = model_ops.entry_lines(entries_of(world, SERVING))

    assert lines[-1] == "  not offered: flash-next — could not plan it (OSError)"


@pytest.mark.parametrize("serving", [True, False])
@pytest.mark.parametrize("record", [PRIMARY, "flash-next", "dock"])
def test_every_record_is_reachable_by_name(tmp_path, monkeypatch, record, serving):
    world = serve_world(tmp_path, monkeypatch, peers=[flash(tmp_path), docker_peer("dock", 8090), attach_peer("attached")])
    live = {PRIMARY_URL: (PRIMARY,) if serving else None, PEER_URL: None, "http://127.0.0.1:8090/v1": None}

    entries = entries_of(world, live)

    assert entries.pick(record) is not None
    assert {r.name for r in model_ops.serving_records(harness_of(world)) if r.spec is not None} == {PRIMARY, "flash-next", "dock"}


def test_building_the_list_writes_nothing_and_takes_no_lock(world, monkeypatch):
    def forbidden(*args, **kw):
        raise AssertionError("the list started, stopped or locked something")

    for target in ("start_server", "stop_server"):
        monkeypatch.setattr(server, target, forbidden)
    for target in ("launch_lock", "launch_lock_sync"):
        monkeypatch.setattr(lifecycle, target, forbidden)
    before = {str(p): (p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(world.g.rglob("*")) if p.is_file()}

    entries_of(world, SERVING)

    assert {str(p): (p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(world.g.rglob("*")) if p.is_file()} == before
