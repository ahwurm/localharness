"""localharness agent subcommands: create, list."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Annotated, Literal

import typer
import yaml
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from localharness.cli.workspace import NO_INPUT_HELP
from localharness.config.paths import WORKSPACE_DIR_NAME, resolve_config_dir

console = Console()
err_console = Console(stderr=True)
log = logging.getLogger(__name__)

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


UNREAD_PROJECT_AGENTS = "{n} project agent files in {dir} not read: {why}; drop it to read them"
_UNREAD_WHY = {
    "flag": "--config-dir was given",
    "LOCALHARNESS_DIR": "LOCALHARNESS_DIR is set",
    "LOCALHARNESS_HOME": "LOCALHARNESS_HOME is set",
}


def config_dir_source(config_dir: str | None) -> Literal["flag", "LOCALHARNESS_DIR", "LOCALHARNESS_HOME"] | None:
    """What named the config directory: the flag, LOCALHARNESS_DIR, LOCALHARNESS_HOME, or nothing."""
    if config_dir is not None:
        # typer fills `config_dir` from LOCALHARNESS_DIR too, so it is the flag only when the value differs
        return "LOCALHARNESS_DIR" if os.environ.get("LOCALHARNESS_DIR") == config_dir else "flag"
    if os.environ.get("LOCALHARNESS_DIR"):
        return "LOCALHARNESS_DIR"
    return "LOCALHARNESS_HOME" if os.environ.get("LOCALHARNESS_HOME") else None


def unread_project_agents(cwd: Path | None, config_dir: str | None) -> list[str]:
    """The one line saying project agent files sit unread because an explicit config dir skipped the project."""
    from localharness.config.paths import discover_workspace_dir, resolve_config_dir

    source = config_dir_source(config_dir)
    found = discover_workspace_dir(cwd) if source else None
    files = sorted((found / "agents").glob("*.yaml")) if found else []
    if not files or found.resolve() == resolve_config_dir(config_dir).resolve():
        return []
    return [UNREAD_PROJECT_AGENTS.format(n=len(files), dir=found / "agents", why=_UNREAD_WHY[source])]


def _warn_unreadable_workspace_files(workspace: Path | None) -> None:
    """Name the workspace files this command could not read, the way `config show` does.

    `agent list` reads the workspace layer, so a file in that layer that will not parse is news —
    and the roster was silent about it: a broken workspace config.yaml gave the same
    "No agents configured" as an empty project, exit 0, nothing anywhere. Stderr and exit 0 are
    both deliberate: the roster itself is unaffected (discovery reads agents/, not config.yaml),
    so this is a notice beside a valid answer, and --json's stdout stays parseable.
    """
    from localharness.config.loader import _load_yaml_file

    if workspace is None:
        return
    for path in (workspace / "config.yaml", workspace / "overrides.yaml"):
        if not path.exists():
            continue
        try:
            _load_yaml_file(path)
        except Exception:
            err_console.print(
                "[bold red]✗[/bold red] " + escape(f"unreadable, skipped: {path}"), soft_wrap=True
            )


@agent_app.command("create")
def agent_create(
    name: Annotated[str, typer.Argument(help="Agent name (lowercase alphanumeric + hyphens)")],
    role: Annotated[str, typer.Option("--role", "-r", help="Agent role description")] = "General-purpose agent",
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model name. Inherits org default if not set.")] = None,
    global_scope: Annotated[bool, typer.Option("--global", help="Add agent to global config (~/.localharness/agents/)")] = False,
    project_scope: Annotated[bool, typer.Option("--project", help="Add agent to this project's workspace (nearest .localharness/agents/ up-tree, else ./.localharness/agents/). Not available with an explicit config directory.")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print YAML without writing")] = False,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing agent with the same name (default refuses)")] = False,
    no_input: Annotated[bool, typer.Option("--no-input", help=NO_INPUT_HELP)] = False,
    config_dir: Annotated[
        str | None,
        typer.Option(
            "--config-dir",
            envvar="LOCALHARNESS_DIR",
            show_default=False,
            help="Config directory. Default: $LOCALHARNESS_DIR, else $LOCALHARNESS_HOME, else ~/.localharness.",
        ),
    ] = None,
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
    elif no_input:
        # --no-input means nobody is watching, and this prompt has no safe default: writing to the
        # wrong layer is exactly the mistake the flag exists to prevent (A-B3's whole story).
        err_console.print(
            "[bold red]Error:[/bold red] --no-input needs a scope: pass --global or --project."
        )
        raise typer.Exit(code=2)
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

    from localharness.config.loader import ConfigLoader

    if use_global:
        layer, local = "machine", None
    else:
        layer = "project"
        # --project means "this project's workspace". Route through the SAME resolution the
        # readers use (agent list, start), or creation and discovery disagree from any
        # subdirectory. The relative literal (not `Path.cwd() / ...`) keeps the success/refusal
        # messages byte-identical to before.
        from localharness.cli.workspace import resolve_workspace_layer
        from localharness.config.paths import config_dir_env_override, discover_workspace_dir

        # an explicit config dir skips discovery, so `--project` has no project to write
        source = config_dir_source(config_dir)
        if source is not None:
            chosen = config_dir if config_dir is not None else config_dir_env_override()
            label = {"flag": "--config-dir", "LOCALHARNESS_DIR": "$LOCALHARNESS_DIR", "LOCALHARNESS_HOME": "$LOCALHARNESS_HOME"}[source]
            err_console.print(
                "[bold red]Error:[/bold red] "
                + escape(f"--project cannot be used with an explicit config directory "
                         f"({label}={chosen}).")
                + " An explicit config directory replaces the whole configuration — nothing "
                "layers a project on top of it, so an agent written to this project would be "
                "invisible to every command run the same way. Drop that setting to use this "
                "project's workspace, or pass --global to write into the config directory you "
                "named.",
                soft_wrap=True,
            )
            raise typer.Exit(code=2)

        found_layer = resolve_workspace_layer(config_dir, interactive=False if no_input else None)
        if found_layer is None:
            # B5/B6: None here is one of two very different situations. Nothing found at all is
            # the BOOTSTRAP case — a fresh project's first agent creates the workspace, silently,
            # exactly as v0.12 did (LAYR-03). Found-but-not-in-use (outside the project and
            # untrusted, or undecided with no terminal to ask) is not: falling back silently mints
            # a SECOND dotdir beside the one the user already has, and the receipt below would
            # name a path that looks like the workspace they meant. Announce it and name both.
            skipped = discover_workspace_dir()
            if skipped is not None:
                err_console.print(
                    escape(f"Note: the workspace at {skipped.resolve()} is not in use for this "
                           f"command, so --project is creating a new one at "
                           f"{Path(WORKSPACE_DIR_NAME)} instead."),
                    style="dim",
                    soft_wrap=True,
                )
            local = Path(WORKSPACE_DIR_NAME)
        else:
            local = found_layer

    loader = ConfigLoader(config_dir=resolve_config_dir(config_dir), local_config_dir=local)
    target = loader.agents_dir(layer) / f"{name}.yaml"
    try:
        (local or resolve_config_dir(config_dir)).mkdir(parents=True, exist_ok=True)
        target = loader.write_agent(name, yaml_text, layer=layer, overwrite=force)
    except (ValueError, OSError) as exc:
        reason = getattr(exc, "strerror", None) or str(exc)
        if isinstance(exc, FileExistsError) and not exc.strerror:
            reason += " pass --force to overwrite"
        err_console.print("[bold red]Error:[/bold red] " + escape(f"cannot write {target}: {reason}"), soft_wrap=True)
        raise typer.Exit(code=1)

    # escape() around the path, markup outside it: the target lives under the user's own
    # directory names, and a project folder named `[old] proj` would otherwise print as a path
    # that does not exist — under a green checkmark saying the file is there (39-05).
    console.print("[green]✓[/green] " + escape(f"Agent '{name}' created at {target}"),
                  soft_wrap=True)
    console.print("  Edit the YAML to customize role, tools, and permissions.")


def _read_row(loader, f) -> dict | None:
    from localharness.config.loader import ConfigParseError

    try:
        return loader.read_agent_file(f.path)
    except (OSError, ConfigParseError) as exc:
        log.debug("agent file %s was skipped: %s", f.path, exc)
        return None


@agent_app.command("list")
def agent_list(
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON array")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show full details")] = False,
    config_dir: Annotated[
        str | None,
        typer.Option(
            "--config-dir",
            envvar="LOCALHARNESS_DIR",
            show_default=False,
            help="Config directory. Default: $LOCALHARNESS_DIR, else $LOCALHARNESS_HOME, else ~/.localharness.",
        ),
    ] = None,
) -> None:
    """List all configured agents."""
    from localharness.channels.base import one_line
    from localharness.cli.workspace import resolve_workspace_layer
    from localharness.config.loader import ConfigLoader, problem_lines, shadow_label
    from localharness.orchestrator.cards import AgentCardRegistry

    # `config_dir` here is the RAW flag value on purpose: resolving it first would erase the
    # "was this explicit" signal the workspace gate is built on.
    # --json is machine output: never stop to ask. An undecided workspace is inert here and
    # gets its prompt from the next interactive command in this directory.
    workspace = resolve_workspace_layer(config_dir, interactive=False if json_output else None)
    _warn_unreadable_workspace_files(workspace)
    for line in unread_project_agents(None, config_dir):
        err_console.print(line, style="yellow", markup=False, soft_wrap=True)
    loader = ConfigLoader(config_dir=resolve_config_dir(config_dir), local_config_dir=workspace)
    problems = AgentCardRegistry(loader=loader).refresh()
    for line in problem_lines(problems):
        err_console.print("[yellow]⚠ " + escape(line) + "[/yellow]", soft_wrap=True)
    # dispatch resolves first-wins and never falls back, so a winner that does not load hides its stem
    dropped = {p.path.stem for p in problems if not p.loads}
    winners = loader.agent_winners()
    shadowed = {f.path.stem: f.path for f in loader.discover_agent_files() if f != winners[f.path.stem]}
    rows = [
        (stem, f, data)
        for stem, f in winners.items()
        if stem not in dropped and (data := _read_row(loader, f)) is not None
    ]

    if not rows:
        # An empty roster under --json is `[]`: prose on stdout is a parse error in whatever pipes this.
        if json_output:
            typer.echo(json.dumps([]))
            return
        console.print("No agents configured. Run: localharness agent create <name>")
        return

    if json_output:
        # typer.echo, never console.print: Rich reads `[...]` in the data as markup and hard-wraps mid-JSON.
        typer.echo(json.dumps([
            {**data, "origin": f.origin, "path": str(f.path), **({"shadows": str(shadowed[stem])} if stem in shadowed else {})}
            for stem, f, data in rows
        ]))
        return

    table = Table(title="Agents")
    table.add_column("Name", style="bold cyan")
    table.add_column("Role")
    table.add_column("Origin")
    if verbose:
        table.add_column("Model")

    for stem, f, data in rows:
        # flatten first, escape second: escape() before one_line would let one_line strip the ESC of `[\x1b/red]`
        cells = [escape(stem), escape(one_line(str(data.get("role", "")))), f.origin]
        if verbose:
            cells.append(escape(one_line(str(data.get("model", "inherit")))))
        table.add_row(*cells)

    console.print(table)
    for stem, _, _ in rows:
        if stem in shadowed:
            console.print(shadow_label(shadowed[stem]), style="dim", markup=False, soft_wrap=True)
