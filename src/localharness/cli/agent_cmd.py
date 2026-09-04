"""localharness agent subcommands: create, list."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from localharness.cli.workspace import NO_INPUT_HELP
from localharness.config.paths import WORKSPACE_DIR_NAME, resolve_config_dir

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


# Phase 38 (#150): `agent list` reads ConfigLoader.discover_agents() — the same roster `start`
# reads. Its own copy of discovery lived here and was the only one that dropped a malformed yaml
# silently (a bare except that discarded the error), so a typo'd file looked exactly like an
# absent one. It now reports what it skipped.


def _skipped_agent_file(f: Path, exc: Exception) -> None:
    """discover_agents' user-visible warning. config/ logs; the CLI owns the console.

    escape() around the path and the reason, markup outside: a folder legally named `[old] proj`
    is data, and rich would silently delete `[old]` from it — this line would then name a file
    that does not exist, in the message whose entire job is naming the file that went wrong
    (39-05's measured lesson). soft_wrap so a deep path is handed to the terminal whole.
    """
    err_console.print(
        "[yellow]⚠ " + escape(f"skipping unreadable agent file {f}: {exc}") + "[/yellow]",
        soft_wrap=True,
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

    # Determine target directory
    if use_global:
        target_dir = resolve_config_dir(config_dir) / "agents"
    else:
        # --project means "this project's workspace". Route through the SAME resolution the
        # readers use (agent list, start), or creation and discovery disagree from any
        # subdirectory. The relative literal (not `Path.cwd() / ...`) keeps the success/refusal
        # messages byte-identical to before.
        from localharness.cli.workspace import resolve_workspace_layer
        from localharness.config.paths import config_dir_env_override, discover_workspace_dir

        # A-B3. An explicit config dir SKIPS discovery entirely — that is LAYR-02, and it is not
        # negotiable: an explicit dir is a full replacement, not a base to layer on. So `--project`
        # here has no project to resolve, and the old fallback wrote `./.localharness/agents/x.yaml`
        # under a green checkmark — a file this same invocation's readers cannot see, because
        # `agent list --config-dir X` and `start --config-dir X` skip discovery too. Refuse instead
        # of writing somewhere nothing reads. Exit 2 (usage error), matching `init --workspace`'s
        # conflicting-flags guard rather than this command's name/scope validation.
        chosen = config_dir if config_dir is not None else config_dir_env_override()
        if chosen is not None:
            # Naming the right one matters: a message that says "--config-dir" to somebody who
            # never typed it sends them looking for a flag that is not on their command line.
            # `config_dir` is not the flag — typer fills it from LOCALHARNESS_DIR too (its
            # `envvar=`), so the parameter alone cannot tell the two apart; matching the env value
            # can. Measured on the real binary: the first version of this message got it wrong.
            if config_dir is not None and os.environ.get("LOCALHARNESS_DIR") == config_dir:
                source = "$LOCALHARNESS_DIR"
            elif config_dir is not None:
                source = "--config-dir"
            elif os.environ.get("LOCALHARNESS_DIR"):
                source = "$LOCALHARNESS_DIR"
            else:
                source = "$LOCALHARNESS_HOME"
            err_console.print(
                "[bold red]Error:[/bold red] "
                + escape(f"--project cannot be used with an explicit config directory "
                         f"({source}={chosen}).")
                + " An explicit config directory replaces the whole configuration — nothing "
                "layers a project on top of it, so an agent written to this project would be "
                "invisible to every command run the same way. Drop that setting to use this "
                "project's workspace, or pass --global to write into the config directory you "
                "named.",
                soft_wrap=True,
            )
            raise typer.Exit(code=2)

        layer = resolve_workspace_layer(config_dir, interactive=False if no_input else None)
        if layer is None:
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
            target_dir = Path(WORKSPACE_DIR_NAME) / "agents"
        else:
            target_dir = layer / "agents"

    target_path = target_dir / f"{name}.yaml"
    # #55: never silently overwrite an existing agent — a live receipt erased a user's
    # tools.deny restriction under a "✓ created". The chat flow already refuses (#28
    # workflow.deploy_config); enforce the same invariant here. --force is the escape hatch.
    #
    # Decided BEFORE the mkdir, not after: a command that has already refused must not have
    # created anything on its way to refusing (E's "crash after write").
    if target_path.exists() and not force:
        err_console.print(
            "[bold red]Error:[/bold red] "
            + escape(f"Agent '{name}' already exists at {target_path}. ")
            + "Choose a different name, edit the file directly, or pass --force to overwrite.",
            soft_wrap=True,
        )
        raise typer.Exit(code=1)

    # The target is validated before anything is written, so a failure cannot leave a half-made
    # tree behind an error message. `agents` as a FILE (H4), a dangling symlink (H2), a symlink
    # loop (H3) and an unwritable parent (H5) all land here as OSError subclasses; each one is
    # something the user can fix once they are told which path is wrong.
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path.write_text(yaml_text, encoding="utf-8")
    except OSError as exc:
        err_console.print(
            "[bold red]Error:[/bold red] "
            + escape(f"cannot write {target_path}: {exc.strerror or exc}"),
            soft_wrap=True,
        )
        raise typer.Exit(code=1)

    # escape() around the path, markup outside it: the target lives under the user's own
    # directory names, and a project folder named `[old] proj` would otherwise print as a path
    # that does not exist — under a green checkmark saying the file is there (39-05).
    console.print("[green]✓[/green] " + escape(f"Agent '{name}' created at {target_path}"),
                  soft_wrap=True)
    console.print("  Edit the YAML to customize role, tools, and permissions.")


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
    from localharness.cli.workspace import resolve_workspace_layer
    from localharness.config.loader import ConfigLoader

    agents = ConfigLoader(
        config_dir=resolve_config_dir(config_dir),
        # `config_dir` here is the RAW flag value on purpose: resolving it first would erase the
        # "was this explicit" signal the workspace gate is built on.
        # --json is machine output: never stop to ask. An undecided workspace is inert here and
        # gets its prompt from the next interactive command in this directory.
        local_config_dir=resolve_workspace_layer(
            config_dir, interactive=False if json_output else None
        ),
    ).discover_agents(on_error=_skipped_agent_file)

    if not agents:
        # `--json` is a machine contract, and an empty roster is `[]` — the one answer a caller's
        # `json.loads` can read. Prose on stdout is a parse error in whatever is piping this
        # (39-06's deferred item; the D1 repro hit it on a fresh workspace-only project).
        if json_output:
            typer.echo(json.dumps([]))
            return
        console.print("No agents configured. Run: localharness agent create <name>")
        return

    if json_output:
        # typer.echo, never console.print: Rich interprets `[...]` in the DATA as markup and
        # silently drops it, and hard-wraps at the terminal width, injecting newlines mid-JSON.
        # Both were reproduced on the shipped 0.12.x wheel (v0.13 dogfood F1) at every width
        # tested. Every other machine-output emitter in cli/ already does it this way.
        typer.echo(json.dumps(agents))
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
