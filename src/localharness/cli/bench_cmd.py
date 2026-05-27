"""Typer sub-app for `localharness bench` and `localharness bench compare`.

Locked CLI surface (CONTEXT.md):
  - Matrix is OPT-IN: default invocation iterates ONLY the current backend.
    `--matrix` flag opts into full matrix iteration.
  - `--json` flag emits machine-readable JSON status to stdout.
  - Exit codes:
      bench: 0=success, 2=infra failure (config missing, empty corpus, total_runs==0)
      bench compare: 0=stable, 1=regressed, 2=infra, 3=unstable (per locked policy)
"""
from __future__ import annotations

import asyncio
import json as _json
import logging
from pathlib import Path
from typing import Optional

import typer

bench_app = typer.Typer(
    name="bench",
    help="Run scenario benchmarks; compare runs for regressions. Matrix is opt-in (--matrix).",
    no_args_is_help=False,
)

log = logging.getLogger(__name__)


@bench_app.callback(invoke_without_command=True)
def bench_default(
    ctx: typer.Context,
    config: Path = typer.Option(
        Path("./bench/bench.yaml"),
        "--config",
        "-c",
        help="Path to bench.yaml (BenchConfig).",
    ),
    corpus: Optional[Path] = typer.Option(
        None,
        "--corpus",
        help="Override corpus_path. Scenarios discovered as *.yaml in this dir.",
    ),
    results: Optional[Path] = typer.Option(
        None,
        "--results",
        help="Override results_path. Per-model subdirs live here.",
    ),
    scenario: Optional[str] = typer.Option(
        None,
        "--scenario",
        "-s",
        help="Scenario NAME to run (filters discovered corpus by ScenarioSpec.name).",
    ),
    model: Optional[list[str]] = typer.Option(
        None,
        "--model",
        "-m",
        help="Subset matrix to these model names (repeatable).",
    ),
    matrix: bool = typer.Option(
        False,
        "--matrix",
        help="Run the FULL configured matrix. Default: only the current backend (matrix[0]).",
    ),
    threshold: Optional[list[str]] = typer.Option(
        None,
        "--threshold",
        "-t",
        help="Threshold override token `metric=value` (repeatable). Passed through to compare.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit machine-readable JSON status to stdout.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable INFO logging."),
) -> None:
    """Run bench scenarios. Default = current backend only. --matrix opts into full iteration."""
    if ctx.invoked_subcommand is not None:
        return  # leave it to the subcommand

    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    # Late import so monkeypatch.setattr("localharness.bench.runner.run_bench", ...)
    # in tests can intercept the call.
    from localharness.bench.runner import run_bench

    config_path: Optional[Path] = config if config is not None and config.exists() else None
    if config is not None and not config.exists() and corpus is None and results is None:
        # No bench.yaml AND no inline overrides -> user error
        msg = f"BenchConfig not found at {config}"
        if json_output:
            typer.echo(_json.dumps({"status": "error", "reason": msg, "exit_code": 2}))
        else:
            typer.echo(msg, err=True)
        raise typer.Exit(code=2)

    exit_code = asyncio.run(
        run_bench(
            scenario=scenario,
            matrix=matrix,
            models=list(model) if model else [],
            threshold_overrides=list(threshold) if threshold else [],
            corpus_path=corpus,
            results_path=results,
            json_output=json_output,
            llm_client_factory=None,
            config_path=config_path,
        )
    )
    raise typer.Exit(code=exit_code)


@bench_app.command(name="compare")
def bench_compare(
    baseline: Path = typer.Option(
        ...,
        "--baseline",
        "-b",
        help="Directory containing baseline summary.json files.",
    ),
    head: Path = typer.Option(
        ...,
        "--head",
        help="Directory containing head summary.json files.",
    ),
    threshold: Optional[list[str]] = typer.Option(
        None,
        "--threshold",
        "-t",
        help="Threshold overrides in `metric=value` form (repeatable). Wins over scenario/bench.yaml.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit CompareResult as JSON to stdout (per CONTEXT.md).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable INFO logging."),
) -> None:
    """Compare two bench result directories.

    Exit codes (locked CONTEXT.md):
      0 = stable (no regressions, no unstable scenarios)
      1 = at least one regression
      2 = infra failure (missing dir, malformed summary, empty corpus)
      3 = at least one head scenario unstable
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    from localharness.bench.compare import run_compare

    exit_code = asyncio.run(
        run_compare(
            baseline=baseline,
            head=head,
            threshold_overrides=threshold or [],
            json_output=json_output,
        )
    )

    if not json_output:
        verdict_label = {0: "stable", 1: "regressed", 2: "infra_failure", 3: "unstable"}.get(exit_code, "unknown")
        typer.echo(f"Compare verdict: {verdict_label} (exit code {exit_code})")
    raise typer.Exit(code=exit_code)
