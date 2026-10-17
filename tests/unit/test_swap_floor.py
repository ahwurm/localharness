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
    names = [n.func.id for n in ast.walk(body) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert names.index("swap_refusal") < names.index("free_accelerator")
