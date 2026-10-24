"""localharness skills command — what was found and why anything was not loaded."""
from __future__ import annotations

from typing import Annotated

import typer


def skills(
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
    """List the skills found and why any was skipped."""
    from localharness.cli.workspace import resolve_workspace_layer
    from localharness.config.paths import resolve_config_dir
    from localharness.config.skills import SkillSet

    cfg_path = resolve_config_dir(config_dir)
    workspace = resolve_workspace_layer(config_dir, interactive=False)
    found = SkillSet.discover(cfg_path, workspace.parent if workspace is not None else None)
    for s in found.skills:
        typer.echo(f"{s.name} ({s.origin}, {str(s.path)!r})")
