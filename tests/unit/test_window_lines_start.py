"""AUTO-02 / AUTO-05 caller half: where `start` shows the window block (spec 16 6.1-6.3, D-03, D-09,
D-21..D-23, D-25, D-28, D-34..D-36, D-38).

`start` probes ONCE (`probe_window`), computes the attach report against the ledger's last record and
prints the block only when the triple moved (always on a launch it made itself), gates the #127
"Context budget fitted" line on the same verdict, and records the triple every time.

Nothing here touches a real server, `nvidia-smi` or the host: the probe, the free memory, the weights
size and the harness-pidfile liveness are faked through their module seams, and the ledger lives in
`tmp_path`.
"""
from __future__ import annotations

import pytest
import typer
import yaml

from localharness.agent.context import WindowProbe
from localharness.cli import start_cmd
from localharness.cli.start_cmd import _start_async
from localharness.config.models import AutosenseConfig
from localharness.provider import autosense
from localharness.provider.autosense import FreeMemory, WindowReport
from localharness.provider.speed_stats import (
    default_speed_stats_path,
    last_window_report,
    record_window_report,
)
from tests.unit.gguf_support import write_gguf
from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries

GIB = 2**30
URL = "http://127.0.0.1:9/v1"  # the discard port: a live :8080 would rename the model
HEADER_A = {"general.architecture": "qwen3", "qwen3.block_count": 64, "qwen3.attention.head_count_kv": 8,
            "qwen3.attention.key_length": 128, "qwen3.context_length": 262144}
KV = 262144  # bytes per token of Header A at f16
WEIGHTS_A = 19864223744  # 18.5 GiB
LINE1 = ("window: served 65,536 · trained 262,144 · fits 90,112 "
         "(free 6.0 GiB of 32.6 beside the served model; weights 18.5 GiB)")
UNDO_HAND = "  → the next harness launch uses -c 90112"


class World:
    """One start's faked surroundings: the probe (with a call counter), free memory, weights size and
    whether the harness's own pidfile is live."""

    def __init__(self, tmp_path, monkeypatch):
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.probe_calls: list[tuple] = []
        self.probe = WindowProbe(65536, 262144)
        self.free = FreeMemory(6 * GIB, int(32.6 * GIB), "nvidia-smi")
        self.alive = False
        self.budgets: list[int] = []
        self.ledger = default_speed_stats_path(tmp_path)

        def probe_window(base_url, model, provider_type=None, *, api_key=None):
            self.probe_calls.append((base_url, model, provider_type, api_key))
            return self.probe

        def probe_served_window(*a, **k):
            raise AssertionError("start no longer calls probe_served_window (D-03)")

        monkeypatch.setattr("localharness.agent.context.probe_window", probe_window)
        monkeypatch.setattr("localharness.agent.context.probe_served_window", probe_served_window)
        monkeypatch.setattr(autosense, "free_memory", lambda ratios: self.free)
        monkeypatch.setattr(autosense, "weights_bytes", lambda model, mmproj=None: WEIGHTS_A)
        monkeypatch.setattr(start_cmd, "_managed_server_running", lambda strategy, srv, d: self.alive)

        async def capture_budget(repl):
            self.budgets.append(repl._agent._ctx.max_context_tokens)

        _stub_start_boundaries(tmp_path, monkeypatch, repl_run=capture_budget)
        self.out = _capture_start_console(monkeypatch)
        self.gguf = write_gguf(tmp_path / "model.gguf", HEADER_A)

    def config(self, *, provider_type="llamacpp", record: dict | None | bool = True,
               budget: int | None = None, **server_over) -> None:
        data: dict = {"version": "1",
                      "provider": {"provider_type": provider_type, "base_url": URL, "api_key": "none",
                                   "default_model": "qwen"}}
        if record:
            data["server"] = {"runtime": "llamacpp", "binary": "/x/llama-server",
                              "model": str(self.gguf), "port": 8080, **server_over}
        if budget is not None:
            data["org"] = {"context": {"max_context_tokens": budget}}
        (self.tmp_path / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")

    async def start(self) -> list[str]:
        """Run one `start`; the lines it printed (this start only)."""
        before = len(self.out)
        await _start_async(None, False, False, str(self.tmp_path))
        return self.out[before:]

    def capability_probe_sees_no_window(self) -> None:
        async def no_window(llm, max_retries=3, delay=2.0):
            return (True, "native", None, None)
        self.monkeypatch.setattr(start_cmd, "_probe_llm", no_window)

    def window(self, lines: list[str]) -> list[str]:
        return [x for x in lines if x.startswith("window:") or x.startswith("  →")]

    def last(self, provider_type="llamacpp"):
        return last_window_report(self.ledger, provider_type, "qwen")


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.config()
    return w


# ------------------------------------------------------------------------------- the tracer


async def test_an_attach_to_a_foreign_server_prints_the_two_line_block_once_and_records_the_triple(world):
    first = await world.start()
    assert world.window(first) == [LINE1, UNDO_HAND]
    rec = world.last()
    assert (rec.served, rec.trained, rec.fits) == (65536, 262144, 90112)

    second = await world.start()
    assert world.window(second) == []
    assert not [x for x in second if x.startswith("Context budget fitted")]


@pytest.mark.parametrize("seed,why", [
    ((65536, 262144, 60000), "fits moved by more than notify_ratio"),
    ((32768, 262144, 90112), "served changed"),
    ((65536, 131072, 90112), "trained changed"),
    ((65536, 262144, None), "a fit known against unknown"),
])
async def test_a_moved_triple_prints_the_block_again(world, seed, why):
    record_window_report(world.ledger, "llamacpp", "qwen", *seed)
    assert world.window(await world.start()) == [LINE1, UNDO_HAND], why
    assert world.last().fits == 90112


async def test_start_probes_the_window_exactly_once(world):
    await world.start()
    assert len(world.probe_calls) == 1
    base_url, model, provider_type, api_key = world.probe_calls[0]
    assert (model, provider_type) == ("qwen", "llamacpp")


# ------------------------------------------------------------------------------- D-38 the undo


@pytest.mark.parametrize("alive", [False, True])
async def test_a_record_context_names_the_record_whoever_launched_the_incumbent(world, alive):
    world.config(context=65536)
    world.alive = alive
    assert world.window(await world.start()) == [
        LINE1, "  → unset `server.context` to relaunch with -c 90112"]


async def test_a_c_in_extra_args_names_extra_args(world):
    world.config(extra_args=["-c", "65536"])
    assert world.window(await world.start()) == [
        LINE1, "  → drop the -c from `server.extra_args` to relaunch with -c 90112"]


async def test_the_long_and_equals_forms_of_the_flag_count_too(world):
    world.config(extra_args=["--ctx-size=65536"])
    assert world.window(await world.start())[1] == (
        "  → drop the -c from `server.extra_args` to relaunch with -c 90112")


async def test_an_unset_context_on_a_foreign_incumbent_says_the_next_launch_uses_it(world):
    world.alive = False
    assert world.window(await world.start())[1] == UNDO_HAND


async def test_an_unset_context_on_an_incumbent_the_harness_launched_has_nothing_to_undo(world):
    world.alive = True
    assert world.window(await world.start()) == [LINE1]


# ------------------------------------------------------------------------------- D-34 no record


async def test_no_record_prints_no_window_line_and_never_points_at_init_but_records(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.config(record=False)
    lines = await w.start()
    assert w.window(lines) == []
    assert not [x for x in lines if "localharness init" in x or "to let the harness size the window" in x]
    rec = w.last()
    assert (rec.served, rec.trained, rec.fits) == (65536, 262144, None)


# ------------------------------------------------------------------------------- AUTO-02 adjacency


def _free_for(served: int, fits: int) -> FreeMemory:
    return FreeMemory((fits - served) * KV, 32 * GIB, "nvidia-smi")


@pytest.mark.parametrize("fits,printed", [(100000, False), (100001, True)])
async def test_a_fit_wobble_of_exactly_notify_ratio_prints_nothing_and_one_token_more_prints(
        world, fits, printed):
    record_window_report(world.ledger, "llamacpp", "qwen", 65536, 262144, 80000)  # 80000 * 0.25 = 20000
    world.free = _free_for(65536, fits)
    lines = world.window(await world.start())
    assert bool(lines) is printed
    assert world.last().fits == fits  # recorded either way


async def test_a_fit_inside_notify_ratio_of_the_served_window_prints_nothing_but_records(world):
    world.probe = WindowProbe(100000, 262144)
    world.free = _free_for(100000, 110000)  # 10% above served: window_lines has nothing to say
    assert world.window(await world.start()) == []
    rec = world.last()
    assert (rec.served, rec.fits) == (100000, 110000)


# ------------------------------------------------------------------------------- AUTO-05 a silent probe


async def test_a_probe_that_returns_nothing_is_not_a_failed_start_and_records_what_is_known(world):
    world.probe = WindowProbe(None, None)
    world.capability_probe_sees_no_window()
    lines = await world.start()  # must not raise
    assert world.window(lines) == []
    rec = world.last()
    assert rec.served is None and rec.trained == 262144  # the file's own header names the trained window


# ------------------------------------------------------------------------------- D-23 the #127 line


async def test_the_fitted_budget_line_prints_on_the_first_start_not_the_second_and_the_budget_holds(world):
    first = await world.start()
    fitted = [x for x in first if x.startswith("Context budget fitted to 65,536:")]
    assert len(fitted) == 1
    second = await world.start()
    assert not [x for x in second if x.startswith("Context budget fitted")]
    assert world.budgets == [65536, 65536]  # fitted in both, said once


async def test_the_fitted_budget_line_prints_again_when_the_served_window_moved(world):
    await world.start()
    world.probe = WindowProbe(32768, 262144)
    world.free = _free_for(32768, 90112)
    again = await world.start()
    assert [x for x in again if x.startswith("Context budget fitted to 32,768:")]


async def test_a_pinned_budget_still_aborts_the_same_way(world):
    world.config(budget=120000)
    with pytest.raises(typer.Exit):
        await world.start()


# ------------------------------------------------------------------------------- D-21 other runtimes


async def test_ollama_prints_served_and_trained_on_change_only(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.config(provider_type="ollama", record=False)
    w.probe = WindowProbe(4096, 32768)
    first = await w.start()
    assert w.window(first) == ["window: served 4,096 · trained 32,768 · Ollama decides the window"]
    assert w.window(await w.start()) == []


async def test_lmstudio_prints_served_and_trained_without_the_ollama_clause(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.config(provider_type="lmstudio", record=False)
    w.probe = WindowProbe(8192, 32768)
    assert w.window(await w.start()) == ["window: served 8,192 · trained 32,768"]


async def test_an_attach_runtime_that_reported_nothing_prints_nothing(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.config(provider_type="ollama", record=False)
    w.probe = WindowProbe(None, None)
    w.capability_probe_sees_no_window()
    assert w.window(await w.start()) == []


async def test_vllm_prints_no_window_line(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.config(provider_type="vllm", record=False)
    w.probe = WindowProbe(32768, 32768)
    assert w.window(await w.start()) == []
    assert w.last("vllm").served == 32768


def test_start_never_spells_the_no_record_pointer():
    text = open(start_cmd.__file__, encoding="utf-8").read()
    assert "to let the harness size the window" not in text
    assert "probe_served_window" not in text


def test_the_window_report_type_start_builds_is_the_one_the_lines_read():
    report, undo = start_cmd._attach_window_report(
        type("H", (), {"server": None, "autosense": AutosenseConfig()})(), "ollama", 1, 2,
        harness_launched=False)
    assert isinstance(report, WindowReport) and undo is None
    assert report.attached and report.mode == "not-computed" and report.fits is None
