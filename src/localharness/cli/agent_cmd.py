"""localharness agent subcommands: create, list."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

console = Console()
err_console = Console(stderr=True)

_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,62}[a-z0-9]$|^[a-z]$")

agent_app = typer.Typer(name="agent", help="Manage LocalHarness agents.", no_args_is_help=True)


def _validate_name(name: str) -> bool:
    return bool(_NAME_RE.match(name))


def _build_agent_yaml(name: str, role: str, model: str | None) -> dict:
    return {
        "name": name,
        "role": role,
        "model": model or "inherit",
    }


def _discover_agents(config_dir: Path) -> list[dict]:
    """Discover agents from both global and local dirs. Local overrides global by name."""
    global_dir = config_dir / "agents"
    local_dir = Path(".localharness") / "agents"

    agents: dict[str, dict] = {}

    # Load global first (lower priority)
    if global_dir.exists():
        for f in sorted(global_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
                if "name" not in data:
                    data["name"] = f.stem
                agents[f.stem] = data
            except Exception:
                pass

    # Local overrides global
    if local_dir.exists():
        for f in sorted(local_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
                if "name" not in data:
                    data["name"] = f.stem
                agents[f.stem] = data
            except Exception:
                pass

    return list(agents.values())


@agent_app.command("create")
def agent_create(
    name: Annotated[str, typer.Argument(help="Agent name (lowercase alphanumeric + hyphens)")],
    role: Annotated[str, typer.Option("--role", "-r", help="Agent role description")] = "General-purpose agent",
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model name. Inherits org default if not set.")] = None,
    global_scope: Annotated[bool, typer.Option("--global", help="Add agent to global config (~/.localharness/agents/)")] = False,
    project_scope: Annotated[bool, typer.Option("--project", help="Add agent to project config (./.localharness/agents/)")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print YAML without writing")] = False,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing agent with the same name (default refuses)")] = False,
    config_dir: Annotated[str, typer.Option("--config-dir", envvar="LOCALHARNESS_DIR")] = "~/.localharness",
) -> None:
    """Create a new agent YAML config."""
    # Validate name
    if not _validate_name(name):
        err_console.print(
            f"[bold red]Error:[/bold red] Invalid agent name '{name}'. "
            "Must be lowercase alphanumeric with hyphens, start and end with letter or digit."
        )
        raise typer.Exit(code=1)

    # Mutual exclusion check
    if global_scope and project_scope:
        err_console.print("[bold red]Error:[/bold red] Cannot use both --global and --project")
        raise typer.Exit(code=1)

    # Determine scope
    if global_scope:
        use_global = True
    elif project_scope:
        use_global = False
    else:
        # Interactive prompt
        answer = typer.prompt(
            "Add globally or to this project?",
            default="global",
        ).strip().lower()
        if answer == "global":
            use_global = True
        elif answer == "project":
            use_global = False
        else:
            err_console.print(f"[bold red]Error:[/bold red] Invalid answer '{answer}'. Expected 'global' or 'project'.")
            raise typer.Exit(code=1)

    # Build YAML content
    agent_data = _build_agent_yaml(name, role, model)
    yaml_text = yaml.dump(agent_data, default_flow_style=False, sort_keys=False)

    if dry_run:
        console.print(yaml_text)
        return

    # Determine target directory
    if use_global:
        target_dir = Path(config_dir).expanduser() / "agents"
    else:
        target_dir = Path(".localharness") / "agents"

    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{name}.yaml"
    # #55: never silently overwrite an existing agent — a live receipt erased a user's
    # tools.deny restriction under a "✓ created". The chat flow already refuses (#28
    # workflow.deploy_config); enforce the same invariant here. --force is the escape hatch.
    if target_path.exists() and not force:
        err_console.print(
            f"[bold red]Error:[/bold red] Agent '{name}' already exists at {target_path}. "
            "Choose a different name, edit the file directly, or pass --force to overwrite."
        )
        raise typer.Exit(code=1)
    target_path.write_text(yaml_text, encoding="utf-8")

    console.print(f"[green]✓[/green] Agent '{name}' created at {target_path}")
    console.print("  Edit the YAML to customize role, tools, and permissions.")


@agent_app.command("list")
def agent_list(
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON array")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show full details")] = False,
    config_dir: Annotated[str, typer.Option("--config-dir", envvar="LOCALHARNESS_DIR")] = "~/.localharness",
) -> None:
    """List all configured agents."""
    agents = _discover_agents(Path(config_dir).expanduser())

    if not agents:
        console.print("No agents configured. Run: localharness agent create <name>")
        return

    if json_output:
        console.print(json.dumps(agents))
        return

    table = Table(title="Agents")
    table.add_column("Name", style="bold cyan")
    table.add_column("Role")
    if verbose:
        table.add_column("Model")

    for a in agents:
        row = [a.get("name", ""), a.get("role", "")]
        if verbose:
            row.append(a.get("model", "inherit"))
        table.add_row(*row)

    console.print(table)
