"""localharness config commands — post-install configuration maintenance and inspection."""
from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from localharness.config import migrate as _migrate
from localharness.config.defaults import CURRENT_DEFAULTS_REVISION
from localharness.config.loader import ConfigError, ConfigNotFoundError
from localharness.config.paths import resolve_config_dir
from localharness.registry.catalogue import (
    LAYER_DEFAULT,
    LAYER_GLOBAL_CONFIG,
    LAYER_GLOBAL_OVERRIDES,
    LAYER_WORKSPACE_CONFIG,
    LAYER_WORKSPACE_OVERRIDES,
    _LAYER_PRIORITY,
)
from localharness.registry.provenance import display_note, layered_catalogue

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
        str | None,
        typer.Option(
            "--config-dir",
            envvar="LOCALHARNESS_DIR",
            show_default=False,
            help="Config directory. Default: $LOCALHARNESS_DIR, else $LOCALHARNESS_HOME, else ~/.localharness.",
        ),
    ] = None,
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
    config_file = resolve_config_dir(config_dir) / "config.yaml"

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
        # escape() around the path and around each pattern: both are data. A deny pattern is a
        # glob — `write(*/[old]*)` is a legal one — and rich would eat `[old]` as a style tag, so
        # this listing would print rules that are not the rules being added (39-05).
        console.print(
            f"[bold]{len(plan.added)}[/bold] shipped default deny pattern(s) missing from "
            + escape(f"{config_file} (defaults revision {plan.from_revision} → "
                     f"{plan.to_revision}):"),
            soft_wrap=True,
        )
        for p in plan.added:
            console.print("  [green]+[/green] " + escape(str(p)), soft_wrap=True)
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


# ------------------------------------------------------------------ #
# show
# ------------------------------------------------------------------ #


def _layer_files(cfg_path: Path, workspace: Optional[Path]) -> list[tuple[str, Path]]:
    """The files in play for this invocation, LOWEST priority first — merge order, as printed.

    Derived from `_LAYER_PRIORITY` reversed rather than re-typed, so a band added to the merge
    cannot fall out of this header silently; the constant is package-private in registry.catalogue
    and imported anyway for exactly that reason (provenance.py takes the same liberty with
    `_load_yaml_file`, for the same "one anchor" reason). `LAYER_EXPERIMENT` has no file of its
    own — it is the in-memory overlay a running gate applies — so it never carries a row.

    With no workspace there are two rows, not four marked absent: a user who never made a
    `.localharness/` must see no trace of the feature (LAYR-03).
    """
    files = {
        LAYER_GLOBAL_CONFIG: cfg_path / "config.yaml",
        LAYER_GLOBAL_OVERRIDES: cfg_path / "overrides.yaml",
    }
    if workspace is not None:
        files[LAYER_WORKSPACE_CONFIG] = workspace / "config.yaml"
        files[LAYER_WORKSPACE_OVERRIDES] = workspace / "overrides.yaml"
    return [(band, files[band]) for band in reversed(_LAYER_PRIORITY) if band in files]


@config_app.command("show")
def show(
    config_dir: Annotated[
        str | None,
        typer.Option(
            "--config-dir",
            envvar="LOCALHARNESS_DIR",
            show_default=False,
            help="Config directory. Default: $LOCALHARNESS_DIR, else $LOCALHARNESS_HOME, else ~/.localharness.",
        ),
    ] = None,
    show_all: Annotated[
        bool,
        typer.Option("--all", help="Include keys nothing has overridden (the compiled-in defaults)."),
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit JSON instead of a table.")
    ] = False,
) -> None:
    """Print the effective config and the file that set each key.

    The merge order, lowest priority first: the global config.yaml, the global overrides.yaml, this
    project's config.yaml, this project's overrides.yaml. The specific beats the general — a
    workspace's word wins wherever the two disagree, and the global layer still governs every key
    the workspace is silent about.

    By default this lists only the keys some file actually sets. `--all` adds the ~180 compiled-in
    defaults; they are the same on every machine, so they answer no question about YOUR config.
    """
    # The RAW flag value, and no prompt on a machine-output run: an explicit --config-dir is a full
    # replacement and must skip discovery entirely (LAYR-02), and a trust prompt on a payload
    # stream is 39-04's rule. Called ONCE — a second call re-prints the gate's notice.
    from localharness.cli.workspace import resolve_workspace_layer

    workspace = resolve_workspace_layer(
        config_dir, interactive=False if json_output else None
    )
    cfg_path = resolve_config_dir(config_dir)

    try:
        # tool_registry=None deliberately: the tools.*.description family is documentation, not
        # config, and building a live registry (asyncio.run + every builtin tool) is real cost to
        # add to a read-only command that would list ~30 rows nothing can set.
        catalogue, _overlays = layered_catalogue(cfg_path, workspace)
    except ConfigNotFoundError as exc:
        err_console.print(
            f"[bold red]Error:[/bold red] Failed to load config: {exc} "
            "Run 'localharness init' to create it."
        )
        raise typer.Exit(2)
    except ConfigError as exc:
        err_console.print(f"[bold red]Error:[/bold red] Failed to load config: {exc}")
        raise typer.Exit(1)

    header_rows = _layer_files(cfg_path, workspace)
    rows = sorted(catalogue.values(), key=lambda e: e.path)
    if not show_all:
        rows = [e for e in rows if e.winning_layer != LAYER_DEFAULT]

    if json_output:
        # typer.echo, never console.print: Rich wraps machine output at the terminal width and
        # eats `[...]` in the DATA as markup (dogfood F1, fixed in 43-02 and pinned here from
        # birth). One serializer for the whole CLI — components_cmd's, imported, not copied.
        from localharness.cli.components_cmd import _serialize_value

        payload = {
            "layers": [
                {"layer": band, "path": str(p), "exists": p.exists()}
                for band, p in header_rows
            ],
            "keys": [
                {
                    "path": e.path,
                    "value": _serialize_value(e.current_value),
                    "type": e.type_name,
                    "layer": e.winning_layer,
                    "default": _serialize_value(e.default_value),
                }
                for e in rows
            ],
        }
        typer.echo(_json.dumps(payload, indent=2))
        return

    console.print(
        "Config layers, lowest priority first — each one wins any key the ones above it set:"
    )
    width = max(len(band) for band, _ in header_rows)
    for band, path in header_rows:
        mark = "present" if path.exists() else "missing"
        # escape() around the path, markup outside it: a folder legally named `[old] proj` printed
        # as ` proj` here would be this command reporting a path that does not exist, in the one
        # command whose whole job is saying where your config comes from (39-05's measured lesson).
        #
        # soft_wrap=True for the same reason, found by the real-binary drive and not by a test:
        # Rich hard-wraps at the console width, so a deep project path arrives with a newline
        # folded into it and the user copies half a path. Soft-wrapping hands the line to the
        # TERMINAL intact — it still looks wrapped on screen, and it is one line in the data.
        console.print(
            f"  [cyan]{band:<{width}}[/cyan]  [dim]{mark}[/dim]  {escape(str(path))}",
            soft_wrap=True,
        )

    if not rows:
        # Not reachable through a valid global config today — `provider:` is required, so at least
        # one catalogued leaf is always attributed to a file — and kept anyway: an empty table with
        # column headings and no rows reads like a failure. Recorded as deliberately ungraded.
        console.print(
            "\nNothing is overridden — every value is the shipped default. "
            "Run with --all to see them."
        )
        return

    table = Table(title="Effective config", show_lines=False)
    table.add_column("key", style="cyan", no_wrap=True)
    table.add_column("value", overflow="fold")
    table.add_column("set by", style="green")
    for e in rows:
        # display_note: `[]` in this column does not mean nothing is enforced (see provenance).
        table.add_row(
            e.path,
            escape(repr(e.current_value) + display_note(e.path, e.current_value)),
            e.winning_layer,
        )
    console.print(table)

    if not show_all:
        console.print(
            f"[dim]Showing the {len(rows)} key(s) some file sets, of {len(catalogue)} total. "
            f"Run with --all to include the {len(catalogue) - len(rows)} compiled-in "
            f"defaults.[/dim]"
        )
