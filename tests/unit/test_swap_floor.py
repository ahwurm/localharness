"""The swap checks the window floor before it stops anything and again after the launch."""
from __future__ import annotations

import ast
import inspect

import pytest

from localharness.config.models import AutosenseConfig, ManagedServerConfig
from localharness.provider import autosense, lifecycle, server
from localharness.provider.autosense import FreeMemory
from localharness.provider.lifecycle import Heavy, swap
from tests.unit.gguf_support import load_fixture, write_recorded, write_sparse

GIB = 2**30
AUTO = AutosenseConfig()
FLOOR = 15_360
MEM_TOTAL = 64_139_128 * 1024
FLASH_MMPROJ = 907_543_520
INCUMBENT = "qwen-27b-swift-1.5"
RAM_SENTENCE = (
    "Flash-Next cannot run here: its experts need 40.0 GiB of RAM and only 22.2 GiB is free beside the "
    "system's reserve — close something that holds RAM, or pick a smaller quant.")


def _world(monkeypatch, *, available=52.9, free=int(30.2 * GIB), total=int(31.8 * GIB)):
    mem = FreeMemory(free, total, "nvidia-smi")
    monkeypatch.setattr(autosense, "free_memory", lambda ratios: mem)
    monkeypatch.setattr(autosense, "_read_meminfo", lambda: {"MemTotal": MEM_TOTAL, "MemAvailable": int(available * GIB)})


def _flash(tmp_path, **over) -> Heavy:
    model = write_recorded(tmp_path, load_fixture("flash-next-iq3-xxs"))[0]
    mmproj = write_sparse(tmp_path / "mmproj.gguf", FLASH_MMPROJ)
    spec = ManagedServerConfig(**{"runtime": "llamacpp", "binary": "/x/llama-server", "model": str(model),
                                  "mmproj": str(mmproj), "served_name": "Flash-Next", "port": 8081, **over})
    return Heavy(spec, "http://127.0.0.1:8081/v1")


def _incumbent() -> Heavy:
    spec = ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server", model="/m/a.gguf",
                               served_name=INCUMBENT, port=8080)
    return Heavy(spec, "http://127.0.0.1:8080/v1")


class Reporter:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.lines: list[str] = []

    async def freed(self) -> None:
        self.calls.append("freed")

    async def launched(self, live) -> None:
        self.calls.append("launched")

    async def restored(self, spec) -> None:
        self.calls.append("restored")

    async def line(self, text: str) -> None:
        self.calls.append("line")
        self.lines.append(text)


@pytest.fixture
def events(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(lifecycle, "GPU_FREE_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(server, "stop_server", lambda *a, **kw: seen.append("stop") or True)
    monkeypatch.setattr(server, "start_server", lambda cfg, cmd, **kw: seen.append("start") or 4321)

    async def wait(base_url, config_dir=None, **kw):
        seen.append("wait")
        return ["Flash-Next"]

    monkeypatch.setattr(server, "wait_ready", wait)
    return seen


async def _swap(tmp_path, incumbent, target, reporter, **kw):
    return await swap(incumbent, target, config_dir=tmp_path, autosense=AUTO, reporter=reporter,
                      redact=lambda t: t, floor=FLOOR, **kw)


async def test_a_refused_target_stops_nothing_and_the_incumbent_keeps_serving(tmp_path, monkeypatch, events):
    _world(monkeypatch, available=31.4)
    reporter = Reporter()
    outcome = await _swap(tmp_path, _incumbent(), _flash(tmp_path), reporter)
    assert reporter.lines == [f"{RAM_SENTENCE} {INCUMBENT} keeps serving."]
    assert outcome.stage == "refused" and outcome.live is None and outcome.stopped is None
    assert str(outcome.error) == f"{RAM_SENTENCE} {INCUMBENT} keeps serving."
    assert events == [] and reporter.calls == ["line"]


def test_swap_plans_the_target_before_it_frees_the_accelerator():
    body = ast.parse(inspect.getsource(lifecycle.swap)).body[0]
    line = {n.id: n.lineno for n in ast.walk(body) if isinstance(n, ast.Name)}
    assert line["swap_refusal"] < line["free_accelerator"]


SMI = "4321, 18432\n999, 512\n"


def _smi(monkeypatch, out):
    monkeypatch.setattr(autosense, "_smi", lambda query: out)


@pytest.mark.parametrize("out, expected", [
    (SMI, 18_432 * 2**20),
    ("4321, 100\n4321, 28\n", 128 * 2**20),
    ("garbage\n4321, x\n4321, 7\n", 7 * 2**20),
    ("999, 512\n", None),
    ("", None),
    (None, None),
])
def test_held_by_sums_the_pids_rows_and_skips_what_it_cannot_read(monkeypatch, out, expected):
    _smi(monkeypatch, out)
    assert autosense.held_by(4321) == expected


def test_held_by_reads_at_most_256_lines(monkeypatch):
    _smi(monkeypatch, "1, 1\n" * 256 + "4321, 18432\n")
    assert autosense.held_by(4321) is None


def test_held_by_asks_nvidia_smi_for_compute_apps_with_the_shared_runner(monkeypatch):
    asked: list[str] = []
    monkeypatch.setattr(autosense, "_smi", lambda query: asked.append(query) or SMI)
    autosense.held_by(4321)
    assert asked == ["--query-compute-apps=pid,used_memory"]


def test_memory_after_stop_adds_what_the_incumbent_holds_and_never_passes_the_total():
    mem = FreeMemory(2 * GIB, 32 * GIB, "nvidia-smi")
    assert autosense.memory_after_stop(mem, 18 * GIB) == FreeMemory(20 * GIB, 32 * GIB, "nvidia-smi")
    assert autosense.memory_after_stop(mem, 40 * GIB).free_bytes == 32 * GIB


def test_memory_after_stop_without_a_share_is_the_card_less_one_gib():
    mem = FreeMemory(2 * GIB, 32 * GIB, "nvidia-smi")
    assert autosense.memory_after_stop(mem, None).free_bytes == 31 * GIB
    assert autosense.memory_after_stop(None, 18 * GIB) is None


def _plan_sees(monkeypatch, tmp_path, *, launch, free=2 * GIB):
    """The memory swap_refusal hands the plan when the incumbent's pidfile reads as `launch`."""
    from localharness.provider import procfs
    _world(monkeypatch, free=free, total=32 * GIB)
    _smi(monkeypatch, SMI)
    monkeypatch.setattr(procfs, "read_launch", lambda *a, **kw: launch)
    seen: list = []
    real = lifecycle.sense_launch
    monkeypatch.setattr(lifecycle, "sense_launch", lambda *a, **kw: seen.append(kw["memory"]) or real(*a, **kw))
    monkeypatch.setattr(lifecycle.os, "kill", lambda *a: pytest.fail("a pid was signalled"))
    lifecycle.swap_refusal(_flash(tmp_path), _incumbent(), config_dir=tmp_path, autosense=AUTO, floor=FLOOR)
    return seen[0]


def test_a_verified_incumbent_launch_lends_the_memory_its_pid_holds(monkeypatch, tmp_path):
    from localharness.provider import procfs
    got = _plan_sees(monkeypatch, tmp_path, launch=procfs.Launch(4321, None, 8080, "spawned", True))
    assert got.free_bytes == 2 * GIB + 18_432 * 2**20


@pytest.mark.parametrize("unverified", [True, False])
def test_an_unverified_or_unreadable_launch_lends_the_card_less_one_gib(monkeypatch, tmp_path, unverified):
    from localharness.provider import procfs
    launch = procfs.Launch(4321, None, 8080, "spawned", False) if unverified else None
    assert _plan_sees(monkeypatch, tmp_path, launch=launch).free_bytes == 31 * GIB


def test_no_incumbent_plans_against_the_probe(monkeypatch, tmp_path):
    _world(monkeypatch)
    seen: list = []
    real = lifecycle.sense_launch
    monkeypatch.setattr(lifecycle, "sense_launch", lambda *a, **kw: seen.append(kw) or real(*a, **kw))
    assert lifecycle.swap_refusal(_flash(tmp_path), None, config_dir=tmp_path, autosense=AUTO, floor=FLOOR) is None
    assert "memory" not in seen[0]
