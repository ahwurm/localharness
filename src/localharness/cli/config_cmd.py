"""localharness config commands — post-install configuration maintenance."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from localharness.config import migrate as _migrate
from localharness.config.defaults import CURRENT_DEFAULTS_REVISION

console = Console()
err_console = Console(stderr=True)

config_app = typer.Typer(
    name="config",
    help="Inspect and maintain your LocalHarness configuration.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

_NOTE = (
    "Additive only — if you deliberately removed a default, re-remove it after migrating."
)


@config_app.command("migrate")
def migrate(
    config_dir: Annotated[
        str,
        typer.Option("--config-dir", envvar="LOCALHARNESS_DIR"),
    ] = "~/.localharness",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Report what would change; write nothing."),
    ] = False,
) -> None:
    """Additively sync the shipped default deny patterns into your config.yaml.

    This is the explicit surface for a fold-in that `localharness start` also does
    automatically on the first start after a package upgrade. `init` bakes the fully-resolved
    `org.permissions.deny_patterns` into config.yaml, so a later growth of the shipped default
    deny list never reaches an existing install (the follow-up disclosed in the v0.9.1 release
    notes). This appends any missing shipped defaults and stamps the config's defaults
    revision — additive ONLY: it never removes or reorders your own entries, touches no other
    key, and (because it is revision-gated) never re-adds a default you deliberately deleted. A
    timestamped backup is written before the config is updated.
    """
    config_file = Path(config_dir).expanduser() / "config.yaml"

    try:
        original, plan = _migrate.load_plan(config_file)
    except _migrate.MigrationError as exc:
        err_console.print(f"[bold red]✗[/bold red] {exc}")
        raise typer.Exit(1)

    if plan is None:
        console.print(
            f"[green]✓[/green] Deny patterns already up to date (defaults revision "
            f"{CURRENT_DEFAULTS_REVISION}). Nothing to add."
        )
        raise typer.Exit(0)

    if plan.added:
        console.print(
            f"[bold]{len(plan.added)}[/bold] shipped default deny pattern(s) missing from "
            f"{config_file} (defaults revision {plan.from_revision} → {plan.to_revision}):"
        )
        for p in plan.added:
            console.print(f"  [green]+[/green] {p}")
    else:
        console.print(
            f"No new deny patterns to add — updating defaults revision "
            f"{plan.from_revision} → {plan.to_revision}."
        )
    console.print(f"\n[dim]{_NOTE}[/dim]")

    if dry_run:
        console.print("\n[cyan]i[/cyan] --dry-run: nothing written.")
        raise typer.Exit(0)

    try:
        backup = _migrate.apply(config_file, original, plan)
    except Exception as exc:
        err_console.print(
            f"[bold red]✗[/bold red] Refusing to write — {exc}"
        )
        raise typer.Exit(1)

    console.print(
        f"\n[green]✓[/green] Added {len(plan.added)} pattern(s); stamped defaults revision "
        f"{plan.to_revision}.\n  Backup: {backup}"
    )
