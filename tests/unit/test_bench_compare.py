"""BENCH-05: bench compare regression diff, threshold override, exit codes."""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import pytest


@pytest.fixture
def baseline_dir(tmp_path: Path) -> Path:
    d = tmp_path / "baseline" / "qwen-3.6-27b"
    d.mkdir(parents=True)
    src = Path(__file__).parent.parent / "fixtures" / "bench" / "baseline_summary.json"
    shutil.copy(src, d / "summary.json")
    return d.parent

@pytest.fixture
def head_dir(tmp_path: Path) -> Path:
    d = tmp_path / "head" / "qwen-3.6-27b"
    d.mkdir(parents=True)
    src = Path(__file__).parent.parent / "fixtures" / "bench" / "head_summary.json"
    shutil.copy(src, d / "summary.json")
    return d.parent


@pytest.mark.xfail(strict=True, reason="Wave 3: bench.compare not yet created (11-04)")
def test_compare_flags_latency_regression(baseline_dir, head_dir):
    """head latency_total median +20% vs baseline triggers regression flag (threshold +15%)."""
    from localharness.bench.compare import diff_summaries
    baseline = json.loads((baseline_dir / "qwen-3.6-27b" / "summary.json").read_text())
    head = json.loads((head_dir / "qwen-3.6-27b" / "summary.json").read_text())
    thresholds = {"latency_total": {"type": "relative", "value": 0.15}}
    result = diff_summaries(baseline, head, thresholds)
    scen = result.per_scenario["minimal_golden"]
    assert scen.regressions.get("latency_total") is True


@pytest.mark.xfail(strict=True, reason="Wave 3: bench.compare not yet created (11-04)")
def test_compare_flags_success_regression(baseline_dir, head_dir):
    """head success_rate -20pp vs baseline triggers regression flag (threshold -5pp)."""
    from localharness.bench.compare import diff_summaries
    baseline = json.loads((baseline_dir / "qwen-3.6-27b" / "summary.json").read_text())
    head = json.loads((head_dir / "qwen-3.6-27b" / "summary.json").read_text())
    thresholds = {"success_rate": {"type": "absolute_pp", "value": -0.05}}
    result = diff_summaries(baseline, head, thresholds)
    scen = result.per_scenario["minimal_golden"]
    assert scen.regressions.get("success_rate") is True


@pytest.mark.xfail(strict=True, reason="Wave 3: bench.compare not yet created (11-04)")
def test_compare_exit_zero_no_regression(baseline_dir):
    """run_compare with identical baseline+head dirs exits 0."""
    import asyncio
    from localharness.bench.compare import run_compare
    rc = asyncio.run(run_compare(baseline_dir, baseline_dir, threshold_overrides=[], json_output=True))
    assert rc == 0


@pytest.mark.xfail(strict=True, reason="Wave 3: bench.compare not yet created (11-04)")
def test_compare_exit_one_on_regression(baseline_dir, head_dir):
    """run_compare with regressed head exits 1."""
    import asyncio
    from localharness.bench.compare import run_compare
    rc = asyncio.run(run_compare(baseline_dir, head_dir, threshold_overrides=[], json_output=True))
    assert rc == 1


@pytest.mark.xfail(strict=True, reason="Wave 3: bench.compare not yet created (11-04)")
def test_compare_exit_two_on_missing_baseline(tmp_path):
    """run_compare with nonexistent baseline dir exits 2 (infra error)."""
    import asyncio
    from localharness.bench.compare import run_compare
    nonexistent = tmp_path / "nonexistent"
    rc = asyncio.run(run_compare(nonexistent, nonexistent, threshold_overrides=[], json_output=True))
    assert rc == 2


@pytest.mark.xfail(strict=True, reason="Wave 3: bench.compare not yet created (11-04)")
def test_compare_exit_three_on_unstable(tmp_path):
    """run_compare exits 3 when any scenario marked stable=false in head summary."""
    import asyncio, json
    from localharness.bench.compare import run_compare
    b = tmp_path / "b" / "qwen"; b.mkdir(parents=True)
    h = tmp_path / "h" / "qwen"; h.mkdir(parents=True)
    unstable = {"model": "qwen", "scenarios": {"x": {"n_runs": 20, "stable": False,
        "latency_total": {"median": 1.0, "p95": 1.0, "mean": 1.0, "std": 0.0, "n": 20},
        "success_rate": {"rate": 1.0, "successes": 20, "n": 20, "wilson_ci": {"p_hat": 1.0, "lower": 0.8, "upper": 1.0, "half_width_pct": 10.0}},
        "samples_latency_total": [1.0]*20, "samples_success": [True]*20,
    }}}
    (b / "summary.json").write_text(json.dumps(unstable))
    (h / "summary.json").write_text(json.dumps(unstable))
    rc = asyncio.run(run_compare(tmp_path / "b", tmp_path / "h", threshold_overrides=[], json_output=True))
    assert rc == 3


@pytest.mark.xfail(strict=True, reason="Wave 3: bench.compare not yet created (11-04)")
def test_cli_threshold_override(baseline_dir, head_dir):
    """CLI --threshold metric=value overrides default. Loose threshold accepts the regression."""
    from localharness.bench.compare import resolve_thresholds
    resolved = resolve_thresholds(
        cli_overrides=["latency_total=0.50"],   # +50% instead of default +15%
        scenario_yaml_thresholds=None,
        bench_yaml_thresholds=None,
    )
    assert resolved["latency_total"]["value"] == 0.50


@pytest.mark.xfail(strict=True, reason="Wave 3: bench.compare not yet created (11-04)")
def test_threshold_precedence_cli_beats_bench_yaml():
    """Precedence: CLI > scenario YAML > bench.yaml > defaults."""
    from localharness.bench.compare import resolve_thresholds
    resolved = resolve_thresholds(
        cli_overrides=["latency_total=0.30"],
        scenario_yaml_thresholds={"latency_total": {"type": "relative", "value": 0.20}},
        bench_yaml_thresholds={"latency_total": {"type": "relative", "value": 0.10}},
    )
    assert resolved["latency_total"]["value"] == 0.30


def test_welch_ab_test_signature():
    """welch_ab_test returns (t_stat, p_value, regressed) tuple."""
    from localharness.bench.aggregator import welch_ab_test
    result = welch_ab_test([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert len(result) == 3
    t_stat, p_value, regressed = result
    assert isinstance(t_stat, float)
    assert isinstance(p_value, float)
    assert isinstance(regressed, bool)


@pytest.mark.xfail(strict=True, reason="Wave 3: bench.compare not yet created (11-04)")
def test_parse_failures_absolute_threshold(baseline_dir, tmp_path):
    """parse_failures: +1 absolute over baseline triggers regression (any new parse failure)."""
    from localharness.bench.compare import diff_summaries
    baseline = json.loads((baseline_dir / "qwen-3.6-27b" / "summary.json").read_text())
    head = json.loads((baseline_dir / "qwen-3.6-27b" / "summary.json").read_text())
    head["scenarios"]["minimal_golden"]["parse_failures"]["median"] = 1
    thresholds = {"parse_failures": {"type": "absolute", "value": 1}}
    result = diff_summaries(baseline, head, thresholds)
    scen = result.per_scenario["minimal_golden"]
    assert scen.regressions.get("parse_failures") is True
