"""localharness skills command — what was found and why anything was not loaded."""
from __future__ import annotations

from typing import Annotated

import typer

DROPPED = "description dropped from the listing ({length} characters; the listing budget is {budget}); shorten to {fit}"


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
    from localharness.cli.skill_session import skill_row
    from localharness.cli.slash_commands import SLASH_COMMANDS, _refusal
    from localharness.cli.workspace import resolve_workspace_layer
    from localharness.config.paths import resolve_config_dir
    from localharness.config.skills import LISTING_BUDGET_CHARS, SkillSet, shown

    cfg_path = resolve_config_dir(config_dir)
    workspace = resolve_workspace_layer(config_dir, interactive=False)
    found = SkillSet.discover(cfg_path, workspace.parent if workspace is not None else None)
    for s, (_, _, text, fit) in zip(found.skills, shown(found.skills)):
        mark = DROPPED.format(length=len(s.description), budget=LISTING_BUDGET_CHARS, fit=fit) if text is None else None
        typer.echo(f"{s.name} ({s.origin}, {str(s.path)!r})" + (f" — {mark}" if mark else ""))
    core = {row.name for row in SLASH_COMMANDS}
    refused = (_refusal(skill_row(s), core) for s in found.skills)
    lines = [f"  {s.reason}. {s.fix}." for s in found.skipped] + [f"  {w}" for w in refused if w is not None]
    if lines:
        typer.echo("\n".join(["Skipped:", *lines]))
