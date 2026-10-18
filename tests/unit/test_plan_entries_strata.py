"""The list represents a sensed Strata install on the Flash-Next file's one entry and offers Strata once when the llama.cpp plan is refused or slow."""
from __future__ import annotations

from pathlib import Path

import typer
from typer.testing import CliRunner

from localharness.cli import model_cmd, model_ops
from localharness.cli.app import app
from localharness.provider import autosense, speed_stats
from tests.unit.strata_fixture import CONFIG_NAME, flash_next, strata_tree
from tests.unit.test_plan_entries import SERVING, entries_of
from tests.unit.test_serve_cmd import GIB, PRIMARY, PRIMARY_URL, llama_peer, serve_world

runner = CliRunner()
NAME = "flash-next"


def flash_world(tmp_path, monkeypatch, *, extra=()):
    shard1, mmproj = flash_next(tmp_path)
    peer = llama_peer(NAME, 8082)
    peer["lifecycle"].update(model=str(shard1), mmproj=str(mmproj))
    return serve_world(tmp_path, monkeypatch, peers=[peer, *extra]), shard1


def install(world, shard1: Path, **kw) -> Path:
    return strata_tree(world.home / "src", model=shard1, **kw)


def strata_peer(world, shard1: Path) -> dict:
    return {"name": f"{NAME}-strata", "base_url": "http://127.0.0.1:8083/v1", "provider_type": "strata", "gpu": True,
            "lifecycle": {"runtime": "strata", "engine_dir": str(world.home / "src" / "Strata"),
                          "engine_config": CONFIG_NAME, "model": str(shard1), "served_name": f"{NAME}-strata",
                          "port": 8083, "gpu": True}}


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


def memory(monkeypatch, available_gib: float) -> None:
    monkeypatch.setattr(autosense, "_read_meminfo", lambda: {"MemTotal": 64 * GIB, "MemAvailable": int(available_gib * GIB)})


def ledger_of(world) -> Path:
    return speed_stats.default_speed_stats_path(world.g)


def test_experts_in_ram_without_an_install_offers_strata_once_after_the_pick(tmp_path, monkeypatch):
    world, _ = flash_world(tmp_path, monkeypatch)

    entries = entries_of(world, SERVING)

    assert dict(entries.offers) == {NAME: "experts in RAM"}
    assert not any("Strata" in line for line in model_ops.entry_lines(entries))
    assert model_ops.offer_lines(entries, NAME, world.g) == [model_ops.STRATA_OFFER]
    assert model_ops.offer_lines(entries, NAME, world.g) == []
    assert not speed_stats.strata_offer_changed(ledger_of(world), NAME, "experts in RAM")


def test_a_refused_plan_offers_strata_and_a_changed_reason_offers_it_again(tmp_path, monkeypatch):
    world, _ = flash_world(tmp_path, monkeypatch)
    slow = entries_of(world, SERVING)
    memory(monkeypatch, 31.4)
    refused = entries_of(world, SERVING)

    assert [e.name for e in refused.not_offered] == [NAME] and dict(refused.offers) == {NAME: "refused"}
    assert model_ops.offer_lines(slow, "2", world.g) == [model_ops.STRATA_OFFER]
    assert model_ops.offer_lines(slow, "2", world.g) == []
    assert model_ops.offer_lines(refused, NAME, world.g) == [model_ops.STRATA_OFFER]
    assert model_ops.offer_lines(refused, NAME, world.g) == []
    assert model_ops.offer_lines(slow, NAME, world.g) == [model_ops.STRATA_OFFER]


def test_another_entry_offers_again(tmp_path, monkeypatch):
    (tmp_path / "second").mkdir()
    other = llama_peer("other", 8083)
    other["lifecycle"].update(model=str(flash_next(tmp_path / "second")[0]))
    world, _ = flash_world(tmp_path, monkeypatch, extra=[other])
    entries = entries_of(world, SERVING)

    assert model_ops.offer_lines(entries, NAME, world.g) == [model_ops.STRATA_OFFER]
    assert model_ops.offer_lines(entries, "other", world.g) == [model_ops.STRATA_OFFER]
    assert model_ops.offer_lines(entries, "other", world.g) == []
    assert model_ops.offer_lines(entries, NAME, world.g) == [model_ops.STRATA_OFFER]


def test_a_sensed_install_says_available_instead_of_offering(tmp_path, monkeypatch):
    world, shard1 = flash_world(tmp_path, monkeypatch)
    install(world, shard1)

    entries = entries_of(world, SERVING)

    assert dict(entries.offers) == {} and model_ops.offer_lines(entries, NAME, world.g) == []


def test_a_refused_entry_with_an_install_says_available_on_its_not_offered_line(tmp_path, monkeypatch):
    world, shard1 = flash_world(tmp_path, monkeypatch)
    install(world, shard1)
    memory(monkeypatch, 31.4)

    entries = entries_of(world, SERVING)

    assert model_ops.entry_lines(entries)[-1].startswith(f"  not offered: {NAME} — ")
    assert model_ops.entry_lines(entries)[-1].endswith(" · Strata available")
    assert entries.pick(f"{NAME}-strata").runtime == "strata" and dict(entries.offers) == {}


def test_a_file_outside_the_family_offers_nothing(tmp_path, monkeypatch):
    world, shard1 = flash_world(tmp_path, monkeypatch)
    renamed = shard1.rename(shard1.with_name("flash-next.gguf"))
    world.edit_config(lambda c: c["extra_endpoints"][0]["lifecycle"].update(model=str(renamed)))
    world.rerecord()
    install(world, renamed)

    entries = entries_of(world, SERVING)

    assert dict(entries.offers) == {} and entries.strata == ()
    assert not any("Strata" in line for line in model_ops.entry_lines(entries))


def test_a_file_with_a_strata_record_offers_nothing_and_adds_no_second_entry(tmp_path, monkeypatch):
    world, shard1 = flash_world(tmp_path, monkeypatch)
    install(world, shard1)
    world.edit_config(lambda c: c["extra_endpoints"].append(strata_peer(world, shard1)))
    world.rerecord()

    entries = entries_of(world, SERVING)

    assert dict(entries.offers) == {} and entries.strata == ()
    assert not any("Strata available" in line for line in model_ops.entry_lines(entries))


def fake_live(monkeypatch) -> None:
    monkeypatch.setattr(model_ops, "list_live_models",
                        lambda url, *a, **kw: ([PRIMARY], True) if url == PRIMARY_URL else ([], False))


def invoke(world, name: str):
    return runner.invoke(app, ["model", name, "--config-dir", str(world.g)])


def swapping(monkeypatch, *lines: str, exit_code: int | None = None) -> list[str]:
    seen: list[str] = []

    def swap(harness, machine_trust, serving, *, config_dir, unit):
        seen.append(serving.record.name)
        for line in lines:
            model_cmd.console.print(line, markup=False, soft_wrap=True)
        if exit_code is not None:
            raise typer.Exit(exit_code)

    monkeypatch.setattr(model_cmd, "_will_launch", lambda unit, cfg: True)
    monkeypatch.setattr(model_cmd, "_swap_now", swap)
    return seen


def test_a_refused_set_up_pick_prints_the_refusal_then_the_offer_and_exits_2(tmp_path, monkeypatch):
    world = serve_world(tmp_path, monkeypatch)
    (world.home / "models").mkdir()
    flash_next(world.home / "models")
    memory(monkeypatch, 31.4)
    fake_live(monkeypatch)
    (refused,) = entries_of(world, SERVING).not_offered

    result = invoke(world, refused.name)
    again = invoke(world, refused.name)

    lines = result.output.splitlines()
    assert result.exit_code == 2 and lines[-1] == model_ops.STRATA_OFFER and "cannot run here" in lines[-2]
    assert again.exit_code == 2 and model_ops.STRATA_OFFER not in again.output


def test_a_launch_prints_its_line_then_the_offer_once(tmp_path, monkeypatch):
    world, _ = flash_world(tmp_path, monkeypatch)
    fake_live(monkeypatch)
    seen = swapping(monkeypatch, "stopping swift…, starting flash-next")

    first, second = invoke(world, NAME), invoke(world, NAME)

    assert seen == [NAME, NAME] and first.exit_code == 0
    assert first.output.splitlines()[-2:] == ["stopping swift…, starting flash-next", model_ops.STRATA_OFFER]
    assert second.output.splitlines()[-1] == "stopping swift…, starting flash-next"


def test_a_refused_swap_prints_the_refusal_then_the_offer_and_keeps_its_exit(tmp_path, monkeypatch):
    world, _ = flash_world(tmp_path, monkeypatch)
    fake_live(monkeypatch)
    swapping(monkeypatch, "Flash-Next cannot run here: too big. swift keeps serving.", exit_code=1)

    result = invoke(world, NAME)

    assert result.exit_code == 1
    assert result.output.splitlines()[-2:] == ["Flash-Next cannot run here: too big. swift keeps serving.", model_ops.STRATA_OFFER]


def test_a_pick_outside_the_family_plans_no_list_and_offers_nothing(tmp_path, monkeypatch):
    world, shard1 = flash_world(tmp_path, monkeypatch)
    renamed = shard1.rename(shard1.with_name("flash-next.gguf"))
    world.edit_config(lambda c: c["extra_endpoints"][0]["lifecycle"].update(model=str(renamed)))
    world.rerecord()
    fake_live(monkeypatch)
    swapping(monkeypatch, "starting flash-next")
    planned: list[int] = []
    real = model_ops.plan_entries
    monkeypatch.setattr(model_ops, "plan_entries", lambda *a, **kw: planned.append(1) or real(*a, **kw))

    result = invoke(world, NAME)

    assert result.output.splitlines()[-1] == "starting flash-next" and planned == []


def test_an_escape_byte_in_a_record_name_is_shown_escaped_in_the_suffix_line(tmp_path, monkeypatch):
    bad = "bad\x1b[2Jname"
    shard1, mmproj = flash_next(tmp_path)
    peer = llama_peer(bad, 8082)
    peer["lifecycle"].update(model=str(shard1), mmproj=str(mmproj))
    world = serve_world(tmp_path, monkeypatch, peers=[peer])
    install(world, shard1)

    lines = model_ops.entry_lines(entries_of(world, SERVING))

    assert not any("\x1b" in line for line in lines)
    assert any("bad\\x1b[2Jname" in line and line.endswith(" · Strata available") for line in lines)
