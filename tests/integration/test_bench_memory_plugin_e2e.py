"""MEMP-05 end to end, offline: a named train scenario recalls through the memory PLUGIN.

The REAL `accumulate_runs -> execute_one_run -> _build_agent_loop -> start_plugins -> AgentLoop`
with only the model faked (the 48-01 recording fake, which answers from the system prompt alone).
A `stop_plugins` spy proves every run tears its plugins down, and a before/after listing of the
temp root proves each run's `bench-mem-*` dir is removed. Train-slice files only; never the holdout.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import localharness.plugins.lifecycle as lifecycle
from localharness.bench.runner import accumulate_runs
from localharness.bench.schema import load_scenario
from tests.integration.test_bench_memory_prompt_equality import _RecallFake
from tests.integration.test_memory_compat_baseline_e2e import _FrozenNow

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "bench" / "scenarios" / "train"


def _bench_dirs() -> set[str]:
    return {p.name for p in Path(tempfile.gettempdir()).glob("bench-mem-*")}


async def _run_twice(file: str, tokens: list[str], tmp_path, monkeypatch):
    monkeypatch.setattr("localharness.agent.loop.datetime", _FrozenNow)
    monkeypatch.setenv("LOCALHARNESS_CATEGORIES_PATH", str(REPO / "bench" / "categories.yaml"))
    monkeypatch.chdir(tmp_path)
    stops: list = []
    real_stop = lifecycle.stop_plugins

    async def _spy(result):
        stops.append(result)
        await real_stop(result)
    monkeypatch.setattr(lifecycle, "stop_plugins", _spy)
    fake = _RecallFake(tokens)
    before = _bench_dirs()
    samples, _stop = await accumulate_runs(load_scenario(TRAIN / file), "fake", tmp_path / "results",
                                           llm_client_factory=lambda _s: fake,
                                           min_runs_override=2, max_runs_override=2)
    return samples, fake, stops, _bench_dirs() - before


async def test_memory_recall_completes_through_the_plugin(tmp_path, monkeypatch):
    samples, fake, stops, leaked = await _run_twice("10_memory_recall.yaml", ["STARFRUIT_42"],
                                                    tmp_path, monkeypatch)
    assert [s.success for s in samples] == [True, True]
    assert fake.calls and all("STARFRUIT_42" in c["system"] for c in fake.calls)
    assert len(stops) == 2 and all(r.slot.occupied for r in stops)
    assert leaked == set(), f"bench-mem dirs left behind: {leaked}"


async def test_overwrite_recall_sees_the_latest_value_through_the_plugin(tmp_path, monkeypatch):
    samples, fake, stops, leaked = await _run_twice(
        "24_stateful_behavior_overwrite_recall.yaml", ["amber"], tmp_path, monkeypatch)
    assert [s.success for s in samples] == [True, True]
    for c in fake.calls:  # the 48-01 golden shows one line: "- favorite_color: amber"
        assert "- favorite_color: amber" in c["system"]
        assert "blue" not in c["system"]
    assert len(stops) == 2
    assert leaked == set(), f"bench-mem dirs left behind: {leaked}"


def test_bench_runner_imports_no_memory_module():
    code = ("import localharness.bench.runner, sys; print(sorted(m for m in sys.modules "
            "if m.startswith('localharness.memory') or m.endswith('memory_tools')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout
