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
    fixtures = sorted(corpus.glob("*.yaml"))
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
