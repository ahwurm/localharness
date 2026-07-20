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


def _locate_fixture_source() -> Optional[Path]:
    """Resolve tests/fixtures/bench relative to a repo checkout — the source `bench run`
    auto-stages from. Tries importlib.resources first (covers an editable/dev install, where the
    installed package IS the repo's src/localharness) then falls back to a path relative to this
    file. Returns None for a plain installed package — that source tree isn't shipped in the wheel.
    """
    candidates: list[Path] = []
    try:
        import importlib.resources as _resources
        pkg_root = Path(str(_resources.files("localharness"))).resolve()
        candidates.append(pkg_root.parents[1] / "tests" / "fixtures" / "bench")
    except Exception:
        pass
    candidates.append(Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "bench")
    for c in candidates:
        if c.exists():
            return c
    return None


def _auto_stage_bench_fixtures() -> None:
    """Best-effort: stage tests/fixtures/bench/* into /tmp/bench_fixtures (+ %TEMP%/bench_fixtures
    on Windows) before a bench run, so scenarios that hardcode that path (e.g.
    02_single_read.yaml) find their data without running the test suite first. Never blocks the
    bench run — a missing source (installed-package case) or any staging failure only warns.
    """
    source = _locate_fixture_source()
    if source is None:
        log.warning("bench fixture source (tests/fixtures/bench) not found — skipping auto-stage")
        return
    try:
        from localharness.bench.fixtures import stage_bench_fixtures
        stage_bench_fixtures(source)
    except Exception:
        log.warning("bench fixture auto-stage failed", exc_info=True)


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
    slice_: str = typer.Option(
        "train",
        "--slice",
        help=(
            "Corpus slice to run: 'train' (default — matches Phase 17 EXP-03 invariant), "
            "'holdout' (sealed slice, explicit opt-in), or 'all' (full corpus diagnostic). "
            "Overridden by --scenario."
        ),
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

    if slice_ not in ("train", "holdout", "all"):
        msg = f"Invalid --slice value {slice_!r}. Must be one of: train, holdout, all."
        if json_output:
            typer.echo(_json.dumps({"status": "error", "reason": msg, "exit_code": 2}))
        else:
            typer.echo(msg, err=True)
        raise typer.Exit(code=2)

    # Scenario prompts hardcode /tmp/bench_fixtures/... (that text must stay literal), so a
    # standalone `bench run` needs the same staging pytest does before those scenarios can pass.
    _auto_stage_bench_fixtures()

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
            slice=slice_,
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


@bench_app.command(name="pack")
def bench_pack(
    results: Path = typer.Option(..., "--results", help="Bench results tree to pack (isolated dir — never a live agent's session dir)."),
    out: Path = typer.Option(..., "--out", help="Output directory for {manifest.json, trajectories.jsonl}."),
) -> None:
    """Build a versioned trace pack from bench runs: gate-verdicted trajectories in
    chat format, leak-scanned, manifest-stamped with the harness version. Regenerate
    per release — packs supersede as functionality evolves. Files without a
    ScenarioCompleted verdict (live sessions) are skipped, never packed."""
    from localharness.bench.pack import PackLeakError, build_pack

    try:
        manifest = build_pack(results, out)
    except (PackLeakError, RuntimeError) as exc:
        typer.echo(f"pack build failed: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(
        f"pack built: runs_packed={manifest['runs_packed']} "
        f"files_skipped={manifest['files_skipped']} "
        f"harness_version={manifest['harness_version']} -> {out}"
    )
