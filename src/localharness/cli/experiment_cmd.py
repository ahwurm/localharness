"""`localharness experiment run <proposal_id>` command (EXP-01 / EXP-05).

Thin typer wrapper over the 17-03 ``autoresearch.experiment.run_experiment`` two-stage
promotion gate. Mirrors the ``propose_cmd`` / ``autoresearch_cmd`` idioms verbatim:
per-command helper duplication (``_err`` / ``_run`` / ``_archive_db_path`` — no shared
module, matching CLAUDE.md "no speculative abstractions"), default human verdict line +
``--json``, and the worker-thread async bridge (REQUIRED — the CliRunner tests run under
pytest-asyncio, which holds a live loop on the calling thread).

The CLI exit code IS the gate verdict (EXP-05): 0=promote, 1=reject-train,
2=reject-holdout, 3=inconclusive; structural refusals surface as >=4 (distinct from the
0-3 gate band). ``run_experiment`` catches its own ``ExperimentRefusal`` and returns the
>=4 code, so the command receives an int in ALL cases; any UNEXPECTED exception is mapped
to the structural band (exit 4), never a leaked stack trace and never exit 1.
"""
from __future__ import annotations

import asyncio
import json as _json
from pathlib import Path

import typer
from rich.console import Console

from localharness.autoresearch.experiment import run_experiment
from localharness.config.paths import config_dir_env_override

console = Console()
err_console = Console(stderr=True)

# Gate verdict labels (mirror bench_cmd's verdict-label dict). Codes >=4 are structural.
VERDICT = {0: "promote", 1: "reject-train", 2: "reject-holdout", 3: "inconclusive"}


# ------------------------------------------------------------------ #
# Helpers (mirror propose_cmd.py — per-command duplication, no shared module)
# ------------------------------------------------------------------ #


def _archive_db_path() -> Path:
    """Resolve .localharness/archive.db, honoring the config-dir env chain (LOCALHARNESS_DIR >
    LOCALHARNESS_HOME); CWD/.localharness when neither is set (#35)."""
    override = config_dir_env_override()
    base = Path(override).expanduser() if override else Path.cwd() / ".localharness"
    return base / "archive.db"


def _err(json_output: bool, message: str, exit_code: int = 4) -> None:
    """Print error to stderr (or JSON) and exit in the structural band (>=4 by default)."""
    if json_output:
        typer.echo(_json.dumps({"error": message}), err=True)
    else:
        err_console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code=exit_code)


def _run(coro):
    """Async-from-sync bridge.

    In production the CLI has no running loop, so ``asyncio.run`` succeeds directly.
    Under pytest-asyncio the calling thread already holds a live loop, so run the
    coroutine on a fresh loop in a worker thread instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # no running loop (the normal CLI path)

    import threading

    result: dict = {}

    def _worker() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # propagate to caller thread
            result["error"] = exc

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


# ------------------------------------------------------------------ #
# experiment run
# ------------------------------------------------------------------ #

experiment_app = typer.Typer(
    name="experiment",
    help="Run a proposal through the promotion gate (train Welch -> holdout Bonferroni).",
    no_args_is_help=True,
)


@experiment_app.callback()
def _experiment_callback() -> None:
    """Promotion-gate tools.

    No-op group callback: forces Typer's command-group mode so ``run`` stays a named
    subcommand (a single-command Typer app otherwise collapses the command into the
    callback, dropping the ``run`` name — see `localharness experiment run <id>`).
    """


@experiment_app.command("run")
def experiment_run(
    proposal_id: str = typer.Argument(
        ..., help="Archive id (full UUID or 8-char prefix) of an in_flight proposal"
    ),
    trials: int = typer.Option(
        1, "--trials", help="Trial family size for Bonferroni correction (alpha = 0.05 / trials)"
    ),
    keep: bool = typer.Option(
        False, "--keep", help="Keep the git worktree for debugging (default: remove)"
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit the verdict as JSON"),
) -> None:
    """Run <proposal_id> through the two-stage gate. Exit code IS the verdict:
    0=promote, 1=reject-train, 2=reject-holdout, 3=inconclusive; >=4=structural refusal."""
    try:
        exit_code = _run(run_experiment(proposal_id, trials=trials, keep=keep))
    except Exception as exc:
        # A structural-band failure, NOT a gate verdict — never leak a stack trace or exit 1.
        _err(json_output, f"experiment failed: {exc}", exit_code=4)
        return

    label = VERDICT.get(exit_code, "structural-refusal")

    if json_output:
        typer.echo(
            _json.dumps(
                {
                    "proposal_id": proposal_id,
                    "verdict": label,
                    "exit_code": exit_code,
                    "trials": trials,
                }
            )
        )
    else:
        console.print(f"Verdict: {label} (exit {exit_code})")

    raise typer.Exit(code=exit_code)
