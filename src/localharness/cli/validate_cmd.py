"""localharness validate command — YAML config validation."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.rule import Rule

from localharness.config.loader import ConfigError, ConfigLoader

console = Console()

_PASS = "[green]✓[/green]"
_FAIL = "[bold red]✗[/bold red]"


def validate(
    path: Annotated[
        str | None,
        typer.Argument(help="Path to specific YAML to validate. If not set, validates all."),
    ] = None,
    config_dir: Annotated[
        str,
        typer.Option("--config-dir", envvar="LOCALHARNESS_DIR"),
    ] = "~/.localharness",
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Treat warnings as errors."),
    ] = False,
) -> None:
    """Validate agent YAML configuration files.

    Reports parse errors, field validation failures with line numbers.
    Exit code 0 if all valid, 1 if any invalid, 2 if no config files found.
    """
    cfg_path = Path(config_dir).expanduser()
    loader = ConfigLoader(config_dir=cfg_path)

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
        console.print("[yellow]No configuration files found.[/yellow]")
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
            _print_error_details(error)
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


def _print_error_details(error: ConfigError) -> None:
    """Print structured error details with field path and line number."""
    from localharness.config.loader import ConfigFieldError, ConfigValidationError, ConfigParseError

    if isinstance(error, ConfigValidationError):
        for field_err in error.errors:
            line_info = f"Line {field_err.yaml_line}: " if field_err.yaml_line else ""
            console.print(f"    [red]{line_info}{field_err.field_path}:[/red] {field_err.message}")
            if field_err.value is not None:
                console.print(f"    [dim]  value: {field_err.value!r}[/dim]")
    elif isinstance(error, ConfigParseError):
        console.print(f"    [red]Line {error.line}:{error.column}: YAML parse error — {error.message}[/red]")
    else:
        console.print(f"    [red]{error}[/red]")
