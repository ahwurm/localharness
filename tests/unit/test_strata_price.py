# ruff: noqa: F811  (the fixtures are imported by name, then taken as parameters)
"""A Strata record is priced against its RAM headroom, the GPU and the floor before anything stops."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from localharness.agent import context as agent_context
from localharness.agent.context import WindowProbe
from localharness.config.models import AutosenseConfig
from localharness.provider import autosense, lifecycle, server, speed_stats, strata
from localharness.provider.autosense import window_lines
from tests.unit.launch_support import popen  # noqa: F401
from tests.unit.strata_fixture import flash_next, strata_config, strata_record, strata_tree
from tests.unit.test_plan_launch_floor import _world
from tests.unit.test_strata_activate import SENSED, make_world, max_context, sensing, world  # noqa: F401

AUTO = AutosenseConfig()


def _spec(tmp_path, config=None, **over):
    shard, projector = flash_next(tmp_path)
    engine = strata_tree(tmp_path, config=config(shard) if config else None, model=shard)
    return strata_record(engine, shard, projector, **over)


def _plan(spec, config_dir, floor=None):
    sensed = lifecycle.sense_launch(spec, config_dir, AUTO)
    return lifecycle.plan_launch(
        spec, sensed, server.launch_decision(spec, False), autosense=AUTO, api_key_held=False, floor=floor)


def test_the_flash_next_record_plans_its_window_with_the_experts_pinned_in_ram(tmp_path, monkeypatch):
    _world(monkeypatch)
    plan = _plan(_spec(tmp_path), tmp_path)
    report = replace(plan.report, served=65536, verified=True)
    assert window_lines(report, AUTO) == [
        "window: 65,536 · Strata · experts pinned in RAM (40.0 GiB) · one request at a time"]
    assert plan.report.mode == "fit" and plan.full_fit == 65536 and plan.serve_kwargs["context"] == 65536


GIB = 2**30
NAME = "swift-1.5-iq3_xxs"
RAM_SENTENCE = (NAME + " cannot run here: its experts need 40.0 GiB of RAM and only {free} GiB is free beside the "
                "system's reserve — close something that holds RAM, or pick a smaller quant.")


def _meminfo(monkeypatch, *, available, total=int(61.17 * GIB)):
    monkeypatch.setattr(autosense, "_read_meminfo", lambda: {"MemTotal": total, "MemAvailable": available})


def _experts(spec, config_dir) -> int:
    return lifecycle.sense_launch(spec, config_dir, AUTO).split.experts


def _refusal(spec, config_dir, floor=None) -> str:
    with pytest.raises(server.LaunchRefused) as refused:
        _plan(spec, config_dir, floor)
    return str(refused.value)


def test_a_box_with_31_gib_available_refuses_with_the_ram_sentence_and_the_box_with_52_gib_does_not(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    _world(monkeypatch, available=52.9)
    assert _plan(spec, tmp_path).report.placement.experts_ram == 42_912_972_800
    _world(monkeypatch, available=31.4)
    assert _refusal(spec, tmp_path) == RAM_SENTENCE.format(free="22.1")


def test_unreadable_ram_refuses_with_the_unreadable_sentence(tmp_path, monkeypatch):
    _world(monkeypatch)
    monkeypatch.setattr(autosense, "_read_meminfo", lambda: None)
    assert _refusal(_spec(tmp_path), tmp_path) == (
        NAME + " cannot run here: its experts need 40.0 GiB of RAM and the free RAM cannot be read (/proc/meminfo) "
        "— pick a quant whose experts fit the GPU.")


def test_experts_plus_ten_gb_exactly_equal_to_the_available_ram_fits_and_one_byte_less_refuses(tmp_path, monkeypatch):
    _world(monkeypatch)
    spec = _spec(tmp_path)
    need = _experts(spec, tmp_path) + 10 * 10**9
    _meminfo(monkeypatch, available=need)
    assert _plan(spec, tmp_path).report.mode == "fit"
    _meminfo(monkeypatch, available=need - 1)
    assert _refusal(spec, tmp_path).startswith(NAME + " cannot run here: its experts need 40.0 GiB of RAM")


def test_ten_gb_is_9_3_gib_and_the_free_figure_prints_with_one_decimal(tmp_path, monkeypatch):
    _world(monkeypatch)
    spec = _spec(tmp_path)
    _meminfo(monkeypatch, available=_experts(spec, tmp_path) + 10**10 - 1)
    assert "only 40.0 GiB is free" in _refusal(spec, tmp_path)
    assert autosense.gib(10**10) == "9.3"


def test_the_machine_reserve_replaces_the_ten_gb_when_larger_and_is_not_added_to_it(tmp_path, monkeypatch):
    _world(monkeypatch)
    spec = _spec(tmp_path)
    experts, total = _experts(spec, tmp_path), 10**11
    _meminfo(monkeypatch, available=experts + 12 * 10**9, total=total)
    assert "its experts need 40.0 GiB of RAM" in _refusal(spec, tmp_path)
    _meminfo(monkeypatch, available=experts + 16 * 10**9, total=total)
    assert _plan(spec, tmp_path).report.mode == "fit"


def test_the_launch_prices_its_own_reading_of_the_ram_not_an_earlier_one(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    _world(monkeypatch, available=52.9)
    _plan(spec, tmp_path)
    _world(monkeypatch, available=31.4)
    assert "its experts need" in _refusal(spec, tmp_path)


def test_dense_layers_the_projector_and_the_windows_kv_above_the_free_gpu_are_the_vram_sentence(tmp_path, monkeypatch):
    _world(monkeypatch, free=3 * GIB)
    assert _refusal(_spec(tmp_path), tmp_path) == (
        NAME + " cannot run here: it needs 5.5 GiB of GPU memory for a 65,536-token window and 3.0 GiB is free — "
        "stop what holds the GPU, or pick a smaller quant.")


def test_unknown_free_gpu_memory_is_not_computed_and_does_not_refuse(tmp_path, monkeypatch):
    _world(monkeypatch, memory=False)
    report = _plan(_spec(tmp_path), tmp_path).report
    assert report.mode == "fit" and report.placement.free_bytes == 0


FLOOR = 13_824


def test_a_record_context_under_the_floor_names_the_record_field(tmp_path, monkeypatch):
    _world(monkeypatch)
    assert _refusal(_spec(tmp_path, context=4096), tmp_path, FLOOR) == (
        NAME + " is set to a 4,096-token window (context:) and the harness needs at least 13,824 — raise it, "
        "or unset it so the harness sizes the window.")


def test_a_configs_window_under_the_floor_names_the_config_file(tmp_path, monkeypatch):
    _world(monkeypatch)
    spec = _spec(tmp_path, lambda shard: strata_config(shard, args=["--native", str(shard), "--max-context", "8192"]))
    assert _refusal(spec, tmp_path, FLOOR) == (
        NAME + " is set to an 8,192-token window (--max-context in strata-swift-IQ3_XXS.json) and the harness "
        "needs at least 13,824 — raise it, or unset it so the harness sizes the window.")


def test_a_window_exactly_at_the_floor_launches(tmp_path, monkeypatch):
    _world(monkeypatch)
    assert _plan(_spec(tmp_path, context=FLOOR), tmp_path, FLOOR).report.mode == "record"


def test_a_config_with_no_window_and_no_record_context_is_refused_and_a_record_context_covers_it(tmp_path, monkeypatch):
    _world(monkeypatch)
    spec = _spec(tmp_path, lambda shard: strata_config(shard, args=["--native", str(shard)]))
    assert _refusal(spec, tmp_path) == (
        NAME + "'s Strata config sets no --max-context — re-run Strata's setup with --context, "
        "or set context: on the record.")
    assert _plan(spec.model_copy(update={"context": 32768}), tmp_path).report.asked == 32768


def test_a_record_window_is_never_a_fit_so_a_retry_cannot_halve_it(tmp_path, monkeypatch):
    _world(monkeypatch)
    plan = _plan(_spec(tmp_path, context=32768), tmp_path)
    assert (plan.report.mode, plan.full_fit, plan.serve_kwargs["context"], plan.retry()) == ("record", None, None, None)
    assert strata.launch_context(plan, plan.inputs[0]) == 32768


def strata_tree_args(spec) -> list[str]:
    return json.loads((Path(spec.engine_dir) / spec.engine_config).read_text("utf-8"))["args"]


def test_the_argv_and_the_copy_gain_no_flag_and_differ_from_the_config_only_in_the_window(tmp_path, monkeypatch):
    _world(monkeypatch)
    spec = _spec(tmp_path, context=32768)
    plan = _plan(spec, tmp_path)
    config = strata_tree_args(spec)
    copy = strata.copy_of(
        plan.inputs[1].engine, spec, context=strata.launch_context(plan, spec), port=spec.port, config_dir=tmp_path)
    assert list(plan.argv) == strata.argv(spec, config_copy=str(strata.copy_path(tmp_path, spec.port)))
    at = config.index("--max-context")
    assert copy["args"] == [*config[:at + 1], "32768", *config[at + 2:]]


class Deaths:
    """The wait is scripted (an exception is a death or a timeout, a list the served ids), the stop is recorded, the probe answers the copy's window."""

    def __init__(self, world, monkeypatch, *script):
        self.world, self.script, self.copies, self.stops, self.stop_error = world, list(script), [], [], None

        async def wait_ready(base_url, **kw):
            self.copies.append(world.copy)
            step = self.script.pop(0)
            if isinstance(step, BaseException):
                raise step
            return step

        def stop_server(config_dir, **kw):
            self.stops.append(kw)
            if self.stop_error is not None:
                raise self.stop_error

        monkeypatch.setattr(server, "wait_ready", wait_ready)
        monkeypatch.setattr(server, "stop_server", stop_server)
        monkeypatch.setattr(agent_context, "probe_window", lambda *a, **kw: WindowProbe(int(max_context(world.copy)), 262144))

    @property
    def windows(self) -> list[str]:
        return [max_context(copy) for copy in self.copies]

    def ledger(self):
        return speed_stats.last_retry(speed_stats.default_speed_stats_path(self.world.config_dir), lifecycle.record_key(self.world.spec))


def death(text: str = "Strata died during startup") -> RuntimeError:
    return RuntimeError(text)


async def test_a_config_window_that_dies_is_relaunched_once_at_half_and_the_retry_is_remembered(world, sensing, popen, monkeypatch):
    deaths = Deaths(world, monkeypatch, death(), [NAME])
    live = await world.activate(autosense=SENSED)
    assert len(popen) == 2 and deaths.windows == ["65536", "32768"]
    assert popen[0]["cmd"] == popen[1]["cmd"]
    assert [(stop["launch"], stop["parent_first"], stop["expect"].port) for stop in deaths.stops] == [("binary", True, 8083)]
    record = deaths.ledger()
    assert record.full_fit == 65536 and record.dropped == ()
    assert window_lines(live.window, SENSED)[0].endswith(" (after one retry at --max-context 32,768)")


async def test_the_copy_of_the_retry_differs_from_the_first_only_in_the_window(world, sensing, popen, monkeypatch):
    deaths = Deaths(world, monkeypatch, death(), [NAME])
    await world.activate(autosense=SENSED)
    first, second = deaths.copies
    assert {**first, "args": None} == {**second, "args": None}
    assert [(a, b) for a, b in zip(first["args"], second["args"], strict=True) if a != b] == [("65536", "32768")]


@pytest.mark.parametrize("floor,launches", [(32768, 2), (32769, 1)])
async def test_a_window_whose_half_equals_the_floor_retries_at_it_and_one_whose_half_is_below_is_not_retried(
        world, sensing, popen, monkeypatch, floor, launches):
    deaths = Deaths(world, monkeypatch, death("first death"), [NAME])
    if launches == 2:
        await world.activate(autosense=SENSED, floor=floor)
        assert deaths.windows == ["65536", "32768"]
    else:
        with pytest.raises(RuntimeError, match="first death"):
            await world.activate(autosense=SENSED, floor=floor)
        assert deaths.stops == []
    assert len(popen) == launches


async def test_a_launch_with_no_plan_is_never_retried(world, popen, monkeypatch):
    deaths = Deaths(world, monkeypatch, death("first death"), [NAME])
    with pytest.raises(RuntimeError, match="first death"):
        await world.activate(autosense=None)
    assert len(popen) == 1 and deaths.stops == []


async def test_two_deaths_end_with_the_community_remedy_then_the_second_log_tail(world, sensing, popen, monkeypatch):
    deaths = Deaths(world, monkeypatch, death("first\nlog one"), death("second death\nlog tail two\nsee serve.log"))
    with pytest.raises(RuntimeError) as dead:
        await world.activate(autosense=SENSED)
    text = str(dead.value)
    assert text.startswith(strata.STRATA_TWICE.format(name=NAME) + "\n")
    assert text.endswith("second death\nlog tail two\nsee serve.log") and "log one" not in text
    assert len(popen) == 2 and len(deaths.stops) == 2 and deaths.ledger() is None


async def test_the_name_in_the_second_death_sentence_shows_a_control_character_escaped(world, sensing, popen, monkeypatch):
    Deaths(world, monkeypatch, death(), death())
    world.spec = world.spec.model_copy(update={"served_name": "bad\x1b[31mname"})
    with pytest.raises(RuntimeError) as dead:
        await world.activate(autosense=SENSED)
    assert str(dead.value).startswith("Strata could not load bad\\x1b[31mname twice;") and "\x1b" not in str(dead.value)


async def test_a_second_death_whose_stop_fails_still_ends_with_the_twice_sentence(world, sensing, popen, monkeypatch):
    deaths = Deaths(world, monkeypatch, death(), death("second"))
    stops = deaths.stops

    def stop_once(config_dir, **kw):
        stops.append(kw)
        if len(stops) == 2:
            raise RuntimeError("pid survived")

    monkeypatch.setattr(server, "stop_server", stop_once)
    with pytest.raises(RuntimeError, match="twice"):
        await world.activate(autosense=SENSED)


async def test_the_next_launch_after_a_remembered_retry_plans_the_halved_window_directly(world, sensing, popen, monkeypatch):
    Deaths(world, monkeypatch, death(), [NAME], [NAME])
    await world.activate(autosense=SENSED)
    sensed = lifecycle.sense_launch(world.spec, world.config_dir, SENSED)
    plan = lifecycle.plan_launch(world.spec, sensed, server.launch_decision(world.spec, False), autosense=SENSED, api_key_held=False)
    assert (plan.shape, plan.cause, plan.serve_kwargs["context"]) == ("retry", "remembered", 32768)
    await world.activate(autosense=SENSED)
    assert max_context(world.copy) == "32768" and len(popen) == 3


async def test_a_full_shape_that_serves_clears_the_remembered_record_and_the_ledger_is_written_once(world, sensing, popen, monkeypatch):
    deaths = Deaths(world, monkeypatch, death(), [NAME], [NAME])
    written = []
    real = speed_stats.record_retry
    monkeypatch.setattr(speed_stats, "record_retry", lambda *a, **kw: (written.append(1), real(*a, **kw))[1])
    await world.activate(autosense=SENSED)
    assert len(written) == 1 and deaths.ledger() is not None
    world.spec = world.spec.model_copy(update={"context": 65536})
    await world.activate(autosense=SENSED)
    assert deaths.ledger() is None and len(written) == 1


async def test_a_pinned_context_that_dies_is_relaunched_once_at_the_same_window_and_remembers_nothing(world, sensing, popen, monkeypatch):
    world.spec = world.spec.model_copy(update={"context": 32768})
    deaths = Deaths(world, monkeypatch, death(), [NAME])
    await world.activate(autosense=SENSED)
    assert len(popen) == 2 and deaths.windows == ["32768", "32768"] and deaths.ledger() is None


async def test_a_pinned_context_that_dies_twice_ends_with_the_twice_sentence(world, sensing, popen, monkeypatch):
    world.spec = world.spec.model_copy(update={"context": 32768})
    Deaths(world, monkeypatch, death(), death("second"))
    with pytest.raises(RuntimeError, match="^Strata could not load swift-1.5-iq3_xxs twice;"):
        await world.activate(autosense=SENSED)
    assert len(popen) == 2


async def test_a_timeout_is_raised_at_once_with_one_launch(world, sensing, popen, monkeypatch):
    deaths = Deaths(world, monkeypatch, TimeoutError("slow"))
    with pytest.raises(TimeoutError):
        await world.activate(autosense=SENSED)
    assert len(popen) == 1 and deaths.stops == []


async def test_a_parent_that_stays_alive_while_the_wait_times_out_is_a_timeout_not_a_retry(world, sensing, popen):
    world.health["loaded"] = False
    with pytest.raises(TimeoutError):
        await world.activate(autosense=SENSED, timeout_seconds=0.05)
    assert len(popen) == 1


async def test_a_pid_that_survives_the_stop_ends_the_retry_with_the_first_deaths_message(world, sensing, popen, monkeypatch):
    deaths = Deaths(world, monkeypatch, death("first death\nits log"), [NAME])
    deaths.stop_error = RuntimeError("pid 4242 survived SIGKILL")
    with pytest.raises(RuntimeError) as dead:
        await world.activate(autosense=SENSED)
    assert str(dead.value) == "first death\nits log" and len(popen) == 1
