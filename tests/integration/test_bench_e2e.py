"""End-to-end: one-scenario bench against MockLLMClient produces summary.json + summary.md."""
from __future__ import annotations
from pathlib import Path
import pytest


@pytest.mark.asyncio
async def test_one_scenario_full_pipeline(tmp_path: Path, mock_llm_client, fixture_scenario_path: Path):
    """Single-scenario bench: ScenarioSpec → N runs → summary.json + summary.md written."""
    from localharness.bench.runner import run_bench
    from localharness.bench.schema import load_scenario
    # Copy fixture into a tmp corpus dir
    corpus = tmp_path / "scenarios"; corpus.mkdir()
    (corpus / "minimal_golden.yaml").write_text(fixture_scenario_path.read_text())
    results_root = tmp_path / "results"
    # Stub LLM that returns "4" once (matches golden_output)
    client = mock_llm_client([mock_llm_client.Response(content="4")])
    rc = await run_bench(
        scenario="minimal_golden",
        matrix=False,
        models=[],
        threshold_overrides=[],
        corpus_path=corpus,
        results_path=results_root,
        json_output=True,
        llm_client_factory=lambda _: client,
        min_runs_override=1,
        max_runs_override=1,
    )
    assert rc == 0
    # Expect per-model output (mock model name)
    model_dirs = list(results_root.iterdir())
    assert len(model_dirs) >= 1
    # Each model dir has summary.json + summary.md
    for d in model_dirs:
        assert (d / "summary.json").exists()
        assert (d / "summary.md").exists()


def test_corpus_dry_run():
    """Full-corpus dry-run: every YAML under bench/scenarios/ loads as ScenarioSpec.

    Final check before phase verify — confirms the 12 fixtures parse without
    triggering Pydantic validation errors (no schema drift, no malformed YAML,
    no missing required fields).
    """
    from pathlib import Path
    from localharness.bench.schema import load_scenario

    corpus = Path(__file__).resolve().parents[2] / "bench" / "scenarios"
    fixtures = sorted(corpus.rglob("*.yaml"))
    assert len(fixtures) >= 12, (
        f"expected at least 12 fixtures, found {len(fixtures)}: "
        f"{[p.name for p in fixtures]}"
    )
    loaded = [load_scenario(p) for p in fixtures]
    names = {s.name for s in loaded}
    all_12 = {
        "pure_qa", "single_read", "write_execute", "fibonacci_sort",
        "file_exploration", "agent_creation", "brave_search_subagent",
        "plugin_mcp_tool", "stuck_recovery", "memory_recall",
        "deny_pattern_hit", "near_compaction",
    }
    assert all_12.issubset(names), f"missing canonical names: {all_12 - names}"


def test_bench_slice_holdout_lists_only_holdout_fixtures():
    """Hermetic discovery+filter smoke for `localharness bench --slice holdout`.

    Confirms the exact invariant the user-facing flag promises end-to-end:
    rglob discovery + _filter_scenarios_by_slice yields ONLY holdout-slice
    fixtures, and exactly the 5 holdout categories at 2 each (10 total).

    Patches no LLM — exercises the discovery/filter seam directly, which is
    the same seam the CLI invokes via run_bench. Preferred over a typer.testing
    invocation here because it avoids mocking deep into the runner per the
    13-03 plan guidance ("fallback version is preferred for hermetic CI").
    """
    from pathlib import Path
    from localharness.bench.orchestrator import (
        _discover_scenarios,
        _load_scenarios_from_paths,
        _filter_scenarios_by_slice,
    )

    corpus = Path(__file__).resolve().parents[2] / "bench" / "scenarios"
    all_scens = _load_scenarios_from_paths(_discover_scenarios(corpus))
    holdout = _filter_scenarios_by_slice(all_scens, "holdout")

    assert len(holdout) == 10, (
        f"expected exactly 10 holdout fixtures, got {len(holdout)}: "
        f"{[s.name for s in holdout]}"
    )
    assert all(s.slice == "holdout" for s in holdout), (
        "non-holdout slice leaked through filter: "
        f"{[(s.name, s.slice) for s in holdout if s.slice != 'holdout']}"
    )
    assert {s.category for s in holdout} == {
        "long_horizon_planning", "tool_ambiguity_resolution",
        "graceful_failure", "self_correction", "constraint_satisfaction",
    }, f"unexpected holdout categories: {sorted({s.category for s in holdout})}"

    # Symmetric check: --slice train yields exactly the 25 train fixtures (24 + the J3 scored
    # over-window cruncher faithfulness scenario, 25_agent_orchestration_over_window_cruncher).
    train = _filter_scenarios_by_slice(all_scens, "train")
    assert len(train) == 25, (
        f"expected exactly 25 train fixtures, got {len(train)}: "
        f"{[s.name for s in train]}"
    )
    assert all(s.slice == "train" for s in train)

    # --slice all is the no-op bypass: 35 fixtures total.
    full = _filter_scenarios_by_slice(all_scens, "all")
    assert len(full) == 35, f"expected 35 total fixtures, got {len(full)}"
