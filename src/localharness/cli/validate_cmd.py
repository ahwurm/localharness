"""localharness validate command — YAML config validation."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.rule import Rule

from localharness.config.loader import ConfigError, ConfigLoader
from localharness.config.paths import resolve_config_dir

console = Console()

_PASS = "[green]✓[/green]"
_FAIL = "[bold red]✗[/bold red]"


def validate(
    path: Annotated[
        str | None,
        typer.Argument(help="Path to specific YAML to validate. If not set, validates all."),
    ] = None,
    config_dir: Annotated[
        str | None,
        typer.Option(
            "--config-dir",
            envvar="LOCALHARNESS_DIR",
            show_default=False,
            help="Config directory. Default: $LOCALHARNESS_DIR, else $LOCALHARNESS_HOME, else ~/.localharness.",
        ),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Reserved. No warning-level checks exist yet; currently identical to the default.",
        ),
    ] = False,
) -> None:
    """Validate agent YAML configuration files.

    Reports parse errors, field validation failures with line numbers.
    Exit code 0 if all valid, 1 if any invalid, 2 if no config files found.
    """
    # #54: --strict is reserved — no warning-level checks exist yet, so it is a no-op today.
    # Don't silently accept it: disclose that this run is identical to the default.
    if strict:
        console.print(
            "[dim]Note: --strict is reserved — no warning-level checks exist yet, so this "
            "run is identical to the default.[/dim]"
        )

    cfg_path = resolve_config_dir(config_dir)
    # LAYR-01: validate checks the workspace's agent and division files too, or it would report
    # "all valid" about a set of files the session will not actually load. The RAW `config_dir`
    # is what the resolver needs — `cfg_path` has already lost whether it was explicit.
    from localharness.cli.workspace import resolve_workspace_layer
    loader = ConfigLoader(config_dir=cfg_path, local_config_dir=resolve_workspace_layer(config_dir))

    results: list[tuple[str, ConfigError | None]] = []

    if path is not None:
        # Validate single file
        target = Path(path)
        if not target.exists():
            console.print(f"[bold red]Error:[/bold red] File not found: {target}")
            raise typer.Exit(1)
        try:
            _validate_single_file(loader, target)
            results.append((str(target), None))
        except ConfigError as exc:
            results.append((str(target), exc))
    else:
        results = loader.validate_all()

    if not results:
        # #119: every command that fails on a missing config ends with the same next step
        # doctor gives — a user who ran `validate` first must not be left without `init`.
        console.print("[yellow]No configuration files found.[/yellow]")
        console.print("Run 'localharness init' to create it.")
        raise typer.Exit(2)

    console.print("\nValidating configs...\n")
    valid_count = 0
    invalid_count = 0

    for file_path, error in results:
        name = Path(file_path).name
        if error is None:
            console.print(f"  {name:<35} {_PASS} valid")
            valid_count += 1
        else:
            console.print(f"  {name:<35} {_FAIL} invalid")
            _print_error_details(error, file_path)
            invalid_count += 1

    console.print()
    console.print(Rule())
    console.print(f"{valid_count} config(s) valid, {invalid_count} invalid.")

    if invalid_count > 0:
        raise typer.Exit(1)


def _validate_single_file(loader: ConfigLoader, path: Path) -> None:
    """Load a single YAML file through the appropriate loader method."""
    from localharness.config.loader import ConfigError, _load_yaml_file
    from localharness.config.models import AgentConfig, DivisionConfig, HarnessConfig

    data = _load_yaml_file(path)
    text = path.read_text(encoding="utf-8")

    # Determine type by filename / location
    name = path.stem
    parent = path.parent.name

    if name == "config":
        loader._validate_dict(HarnessConfig, data, str(path), text)
    elif name == "org":
        from localharness.config.models import OrgConfig
        loader._validate_dict(OrgConfig, data, str(path), text)
    elif parent == "divisions":
        loader._validate_dict(DivisionConfig, data, str(path), text)
    else:
        # Assume agent config
        loader._validate_dict(AgentConfig, data, str(path), text)


def _print_error_details(error: ConfigError, reported_path: str | None = None) -> None:
    """Print structured error details with field path and line number.

    `reported_path` is the file this row is LISTED under. The harness row is always listed under
    the global `config.yaml` (validate_all's key), but with a workspace layer the error can belong
    to one of three other files — and the row prints a bare basename, which is the same
    `config.yaml` in either layer. So when the report's own owning file differs from the row's, it
    is printed: without it a user reading `validate` would be told the right LINE of a file they
    have no way to identify (CLI-02 / dogfood F5). `doctor` gets this for free — it renders the
    whole `str(exc)`, header included.

    Both clauses are inert when no workspace applies: the paths agree and `source_path` is None, so
    the output is byte-identical to pre-43 (LAYR-03).
    """
    from localharness.config.loader import ConfigFieldError, ConfigValidationError, ConfigParseError

    if isinstance(error, ConfigValidationError):
        if reported_path is not None and error.path != reported_path:
            # escape(): a path is data, not markup. A folder named `[old] proj` otherwise prints as
            # a path that does not exist, in the command people run to find out what is wrong.
            console.print(f"    [dim]in {escape(error.path)}[/dim]")
        for field_err in error.errors:
            line_info = f"Line {field_err.yaml_line}: " if field_err.yaml_line else ""
            origin = f"{escape(field_err.source_path)} " if field_err.source_path else ""
            console.print(
                f"    [red]{origin}{line_info}{field_err.field_path}:[/red] {field_err.message}"
            )
            if field_err.value is not None:
                console.print(f"    [dim]  value: {field_err.value!r}[/dim]")
    elif isinstance(error, ConfigParseError):
        console.print(f"    [red]Line {error.line}:{error.column}: YAML parse error — {error.message}[/red]")
    else:
        console.print(f"    [red]{error}[/red]")
