"""localharness config commands — post-install configuration maintenance."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console

from localharness.config.models import HarnessConfig, PermissionConfig

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

    `localharness init` bakes the fully-resolved `org.permissions.deny_patterns` into
    config.yaml, so a later growth of the shipped default deny list never reaches an
    existing install (the follow-up disclosed in the v0.9.1 release notes). This
    appends any missing shipped defaults — additive ONLY: it never removes or reorders
    your own entries, and it touches no other config key. A timestamped backup is
    written before the config is updated.
    """
    config_file = Path(config_dir).expanduser() / "config.yaml"

    if not config_file.exists():
        err_console.print(
            f"[bold red]✗[/bold red] No config found at {config_file} — run "
            "'localharness init' first."
        )
        raise typer.Exit(1)

    original_bytes = config_file.read_bytes()
    try:
        data = yaml.safe_load(original_bytes.decode("utf-8"))
    except yaml.YAMLError as exc:
        err_console.print(
            f"[bold red]✗[/bold red] Could not parse {config_file}:\n       {exc}"
        )
        raise typer.Exit(1)
    if not isinstance(data, dict):
        err_console.print(
            f"[bold red]✗[/bold red] {config_file} is not a valid config mapping."
        )
        raise typer.Exit(1)

    # Read the user's existing deny list (defensive over an absent org/permissions block).
    org = data.get("org") if isinstance(data.get("org"), dict) else {}
    perms = org.get("permissions") if isinstance(org.get("permissions"), dict) else {}
    user_deny = perms.get("deny_patterns")
    user_deny = list(user_deny) if isinstance(user_deny, list) else []

    missing = [p for p in PermissionConfig().deny_patterns if p not in user_deny]

    if not missing:
        console.print(
            "[green]✓[/green] Deny patterns already up to date — every shipped default "
            "is present. Nothing to add."
        )
        raise typer.Exit(0)

    console.print(
        f"[bold]{len(missing)}[/bold] shipped default deny pattern(s) missing from "
        f"{config_file}:"
    )
    for p in missing:
        console.print(f"  [green]+[/green] {p}")
    console.print(f"\n[dim]{_NOTE}[/dim]")

    if dry_run:
        console.print("\n[cyan]i[/cyan] --dry-run: nothing written.")
        raise typer.Exit(0)

    # Build the updated config, mutating ONLY org.permissions.deny_patterns.
    updated = dict(data)
    updated_org = dict(org)
    updated_perms = dict(perms)
    updated_perms["deny_patterns"] = [*user_deny, *missing]
    updated_org["permissions"] = updated_perms
    updated["org"] = updated_org

    # Validate through the real model BEFORE writing — a migrate that writes an invalid
    # config is worse than none.
    try:
        HarnessConfig.model_validate(updated)
    except Exception as exc:
        err_console.print(
            f"[bold red]✗[/bold red] Refusing to write — the migrated config fails "
            f"validation:\n       {exc}"
        )
        raise typer.Exit(1)

    # Backup the pre-migration bytes FIRST, then write the updated config.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = config_file.with_name(f"config.yaml.bak-{stamp}")
    backup.write_bytes(original_bytes)
    config_file.write_text(
        yaml.safe_dump(updated, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    console.print(
        f"\n[green]✓[/green] Added {len(missing)} pattern(s) to "
        f"org.permissions.deny_patterns.\n  Backup: {backup}"
    )
