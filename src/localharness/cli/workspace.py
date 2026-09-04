"""Does a workspace layer apply to THIS invocation, and may we load it?

The one composition point for v0.13 workspace discovery. Four rules, in order:

1. Explicit selection wins and skips discovery entirely (LAYR-02). "Explicit" means the
   --config-dir flag OR either env var — LOCALHARNESS_HOME counts, because a user who pointed the
   harness at a specific dir did not ask for a project layer, and because every existing test runs
   with LOCALHARNESS_HOME set (that is what keeps the suite discovery-inert with no test edits).
2. The nearest `.localharness/` at or above CWD is the candidate (LAYR-01/LAYR-04).
3. A candidate INSIDE the project you are standing in loads silently. "Inside" means its folder is
   your current directory, or it sits at or below the root of the git repository containing your
   current directory — its RESOLVED folder, so a symlinked `.localharness/` counts as the tree it
   points at rather than the one holding the link. Nested directories inherit their project's
   config — that is what every other
   project-scoped tool does and what users already expect, and a prompt that fires on every project
   is a prompt everybody clicks through (owner ruling 2026-09-03).
4. A candidate from OUTSIDE that project is trust-gated (LAYR-05): above your repository root, or
   a parent folder while no repository contains you at all. Agent yamls are executable intent —
   role, model, tools, deny rules — so config reaching in from a tree you did not open must be
   agreed to. Asked once; the answer is permanent. Undecided + no terminal = inert, and NOTHING is
   recorded: a scripted run must not spend the user's one-time answer for them.

Known and accepted, stated plainly in SECURITY.md: cloning a repository and running the harness
inside it loads that repository's agent files with no prompt. That is the exposure the literal
`./.localharness` read has always had; rule 4 covers config coming from OUTSIDE the tree you chose
to open, which is the new reach discovery adds.

Called at the CLI edge with the RAW --config-dir value, before any resolution. That raw value is
the only place "was this explicit" survives: every command pre-resolves to a concrete Path before
constructing ConfigLoader, so the signal cannot be recovered downstream.

NEVER bind the result to a module-level name — the answer is per-invocation (the workspace root
varies with CWD).

| config_dir arg | env override | workspace found | in project | stored trust | tty | result           | recorded |
|----------------|--------------|-----------------|------------|--------------|-----|------------------|----------|
| set            | any          | (not searched)  | —          | (not read)   | any | None             | no       |
| None           | set          | (not searched)  | —          | (not read)   | any | None             | no       |
| None           | None         | no              | —          | (not read)   | any | None             | no       |
| None           | None         | yes             | yes        | (not read)   | any | the workspace    | no       |
| None           | None         | yes             | no         | True         | any | the workspace    | no       |
| None           | None         | yes             | no         | False        | any | None + notice    | no       |
| None           | None         | yes             | no         | None         | no  | None + notice    | NO       |
| None           | None         | yes             | no         | None         | yes | prompt's answer  | yes      |
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

from rich.console import Console

log = logging.getLogger(__name__)

# Notices AND the prompt go to stderr: `agent list --json` writes machine-readable JSON on
# stdout, and a trust banner there would corrupt it.
_notice_console = Console(stderr=True)


def resolve_workspace_layer(
    config_dir: Optional[Union[str, Path]] = None,
    *,
    interactive: Optional[bool] = None,
) -> Optional[Path]:
    """The workspace layer for this invocation, or None. See the module docstring's table."""
    from localharness.config import trust
    from localharness.config.paths import (
        config_dir_env_override,
        discover_workspace_dir,
        workspace_is_within_repo,
    )

    if config_dir is not None or config_dir_env_override() is not None:
        return None

    found = discover_workspace_dir()
    if found is None:
        return None

    here = Path.cwd().resolve()
    # BOTH sides resolved, or `.localharness` being a SYMLINK to another tree would read as "your
    # own directory" and load agent files from anywhere on the machine with no prompt at all. The
    # trust store already keys on the realpath (config/trust.py), so this is also what makes the
    # two halves of the gate agree on what a workspace IS. `real` is identical to `found` for
    # every ordinary directory — the walk builds it from an already-resolved ancestor.
    real = found.resolve()
    if real.parent == here or workspace_is_within_repo(found, here):
        # In project: your own directory, or the same repository you are working in. Nested
        # inherits — no prompt, and nothing is written to the trust store.
        log.info("workspace layer: %s (in project)", found)
        return found

    decision = trust.is_trusted(found)
    if decision is True:
        log.info("workspace layer: %s (trusted)", found)
        return found
    if decision is False:
        _notice(f"Workspace {real} is not trusted — its config layer is ignored.")
        return None

    if interactive is None:
        # `sys.stdin` is None in a detached process; no stdin at all is no terminal either.
        interactive = sys.stdin is not None and sys.stdin.isatty()
    if not interactive:
        # Fail closed (SECURITY.md: deny on doubt) but do NOT record — a later interactive
        # session in this directory still gets asked once.
        _notice(
            f"Found a workspace at {real} from outside this project — ignoring its config "
            "layer (no terminal to ask). Run an interactive localharness command here to decide."
        )
        return None

    trusted = _ask(real)
    trust.record_trust(found, trusted)
    if trusted:
        log.info("workspace layer: %s (trusted just now)", found)
        return found
    _notice(f"Workspace {real} recorded as not trusted — its config layer is ignored.")
    return None


def _notice(message: str) -> None:
    """One line on stderr. `markup=False` because the message carries a filesystem path, and a
    folder named `[old] proj` is legal everywhere while `[old]` is rich markup — parsing it would
    turn a notice into a crashed command."""
    _notice_console.print(message, style="dim", markup=False)


def _ask(found: Path) -> bool:
    """The one-time question. Names what is at stake AND why this workspace is being asked about,
    since the ones inside your own project never are. Takes the RESOLVED workspace so a symlinked
    dotdir names the tree the files actually come from — "the workspace at ./" is not a question
    anyone can answer."""
    from rich.prompt import Confirm
    from rich.text import Text

    question = Text(
        f"The workspace at {found.parent} is outside the project you are in. Load its agent "
        "and config files? They define roles, models and tool permissions — treat them like "
        "code you are about to run."
    )
    return bool(Confirm.ask(question, console=_notice_console, default=False))
