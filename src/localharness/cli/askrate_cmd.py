"""`localharness ask-rate` — the internal ask-rate report (PRD §3.6).

Hidden from `--help` on purpose: this is an instrument for the people changing the permission
classifier, not a verb the project markets (PRD §10, "ask-rate report (internal, not a marketed
verb)").
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from localharness.bench.askrate import FIRST_N_SESSIONS_DEFAULT, build_report, render


def ask_rate(
    traces: Path = typer.Option(
        ...,
        "--traces",
        help="Directory of per-session bus-event JSONL files (searched recursively).",
    ),
    first: int = typer.Option(
        FIRST_N_SESSIONS_DEFAULT,
        "--first",
        min=1,
        help="Convergence split: compare the first N sessions against the rest.",
    ),
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        help="Workspace the replay assumes; defaults to the current directory. "
             "Ignored when the traces already carry PermissionAsked events.",
    ),
) -> None:
    """Report how often the permission gate asked, or would have asked, per session."""
    traces_dir = Path(traces).expanduser()
    if not traces_dir.is_dir():
        typer.echo(f"no such traces directory: {traces_dir}", err=True)
        raise typer.Exit(code=2)
    report = build_report(traces_dir, workspace=workspace, first_n=first)
    if not report.sessions:
        typer.echo(f"no session files (*.jsonl) under {traces_dir}", err=True)
        raise typer.Exit(code=2)
    typer.echo(render(report))
