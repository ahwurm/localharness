"""End-to-end: one-scenario bench against MockLLMClient produces summary.json + summary.md."""
from __future__ import annotations
from pathlib import Path
import pytest


@pytest.mark.xfail(strict=True, reason="Wave 3: full pipeline not yet wired (11-04)")
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
