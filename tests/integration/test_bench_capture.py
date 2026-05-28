"""BENCH-03: per-run JSONL trace capture + replay round-trip."""
from __future__ import annotations
import json
from pathlib import Path
import pytest


@pytest.mark.asyncio
async def test_jsonl_trace_written(tmp_path: Path):
    """execute_one_run constructs EventBus(persist_path=...) which writes the JSONL trace."""
    from localharness.bench.runner import execute_one_run
    from localharness.bench.schema import ScenarioSpec, SuccessCriteria
    from localharness.core.events import BudgetSpec
    scen = ScenarioSpec(
        name="t", prompt="hi",
        success_criteria=SuccessCriteria(golden_output="hi"),
        budget=BudgetSpec(max_actions=1, max_duration_minutes=0.1),
        slice="train",
        category="tool_basics",
    )
    run_path = tmp_path / "bench" / "results" / "mock" / "t" / "ts.jsonl"
    # Use MockLLMClient via a factory the runner accepts
    result = await execute_one_run(scen, model="mock", run_path=run_path, llm_client=None)
    assert run_path.exists()
    lines = run_path.read_text().strip().splitlines()
    assert len(lines) >= 1


@pytest.mark.asyncio
async def test_jsonl_replayable(tmp_path: Path):
    """JSONL trace lines parse back into BaseEvent subclasses via deserialize_event."""
    from localharness.bench.runner import execute_one_run
    from localharness.bench.schema import ScenarioSpec, SuccessCriteria
    from localharness.core.events import BudgetSpec, deserialize_event
    scen = ScenarioSpec(
        name="t", prompt="hi",
        success_criteria=SuccessCriteria(golden_output="hi"),
        budget=BudgetSpec(max_actions=1, max_duration_minutes=0.1),
        slice="train",
        category="tool_basics",
    )
    run_path = tmp_path / "bench" / "results" / "mock" / "t" / "ts.jsonl"
    await execute_one_run(scen, model="mock", run_path=run_path, llm_client=None)
    for line in run_path.read_text().strip().splitlines():
        ev = deserialize_event(line)
        assert ev.event_type  # event_type is set


@pytest.mark.asyncio
async def test_results_path_pattern(tmp_path: Path):
    """Trace written to bench/results/{model}/{scenario}/{timestamp}.jsonl pattern."""
    from localharness.bench.runner import resolve_run_path
    path = resolve_run_path(results_root=tmp_path / "bench" / "results", model="qwen-3.6", scenario_name="qna", timestamp="20260527T120000Z")
    assert path.parent.parent.parent.name == "results"
    assert path.parent.parent.name == "qwen-3.6"
    assert path.parent.name == "qna"
    assert path.name == "20260527T120000Z.jsonl"
