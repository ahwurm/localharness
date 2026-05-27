"""BENCH-01/02/06: CLI surface — Typer sub-app smoke tests via CliRunner."""
from __future__ import annotations
from pathlib import Path
import pytest
from typer.testing import CliRunner


runner = CliRunner()


@pytest.mark.xfail(strict=True, reason="Wave 3: bench_app not registered on root app yet (11-04)")
def test_bench_command_registered():
    """`localharness bench --help` exits 0 and shows bench help text."""
    from localharness.cli.app import app
    result = runner.invoke(app, ["bench", "--help"])
    assert result.exit_code == 0
    assert "bench" in result.stdout.lower()


@pytest.mark.xfail(strict=True, reason="Wave 3: bench_app not registered on root app yet (11-04)")
def test_default_runs_all_scenarios(monkeypatch, tmp_path):
    """`localharness bench` (no args) calls run_bench with scenario=None."""
    from localharness.cli.app import app
    called = {}
    async def fake_run_bench(**kwargs):
        called.update(kwargs)
        return 0
    monkeypatch.setattr("localharness.bench.runner.run_bench", fake_run_bench)
    result = runner.invoke(app, ["bench", "--corpus", str(tmp_path)])
    assert called.get("scenario") is None


@pytest.mark.xfail(strict=True, reason="Wave 3: bench_app not registered on root app yet (11-04)")
def test_single_scenario_flag(monkeypatch, tmp_path):
    """`localharness bench --scenario qna` calls run_bench with scenario='qna'."""
    from localharness.cli.app import app
    called = {}
    async def fake_run_bench(**kwargs):
        called.update(kwargs)
        return 0
    monkeypatch.setattr("localharness.bench.runner.run_bench", fake_run_bench)
    result = runner.invoke(app, ["bench", "--scenario", "qna", "--corpus", str(tmp_path)])
    assert called.get("scenario") == "qna"


@pytest.mark.xfail(strict=True, reason="Wave 3: bench_app not registered on root app yet (11-04)")
def test_matrix_flag(monkeypatch, tmp_path):
    """`localharness bench --matrix` calls run_bench with matrix=True."""
    from localharness.cli.app import app
    called = {}
    async def fake_run_bench(**kwargs):
        called.update(kwargs)
        return 0
    monkeypatch.setattr("localharness.bench.runner.run_bench", fake_run_bench)
    result = runner.invoke(app, ["bench", "--matrix", "--corpus", str(tmp_path)])
    assert called.get("matrix") is True


@pytest.mark.xfail(strict=True, reason="Wave 3: bench_app not registered on root app yet (11-04)")
def test_ad_hoc_model_subset(monkeypatch, tmp_path):
    """`localharness bench --model A --model B` passes models=['A','B'] to run_bench."""
    from localharness.cli.app import app
    called = {}
    async def fake_run_bench(**kwargs):
        called.update(kwargs)
        return 0
    monkeypatch.setattr("localharness.bench.runner.run_bench", fake_run_bench)
    result = runner.invoke(app, ["bench", "--model", "qwen-3.6-27b", "--model", "gpt-oss-120b", "--corpus", str(tmp_path)])
    assert called.get("models") == ["qwen-3.6-27b", "gpt-oss-120b"]


@pytest.mark.xfail(strict=True, reason="Wave 3: bench_app not registered on root app yet (11-04)")
def test_compare_subcommand_help():
    """`localharness bench compare --help` exits 0 and mentions baseline/head."""
    from localharness.cli.app import app
    result = runner.invoke(app, ["bench", "compare", "--help"])
    assert result.exit_code == 0
    assert "baseline" in result.stdout.lower()
    assert "head" in result.stdout.lower()
