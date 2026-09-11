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
   points at rather than the one holding the link. From a linked worktree that root includes the
   checkout the worktree was cut from (owner ruling R1): a worktree is your own project, and the
   harness must not ask you about your own repository. Nested directories inherit their project's
   config — that is what every other
   project-scoped tool does and what users already expect, and a prompt that fires on every project
   is a prompt everybody clicks through (owner ruling 2026-09-03).
4. A candidate from OUTSIDE that project is trust-gated (LAYR-05): above your repository root, or
   a parent folder while no repository contains you at all. Agent yamls are executable intent —
   role, model, tools, deny rules — so config reaching in from a tree you did not open must be
   agreed to. Asked once; the answer is permanent. Undecided + non-interactive = inert, and
   NOTHING is recorded: a scripted run must not spend the user's one-time answer for them.
   "Non-interactive" is two things, and the notice must not claim only the first: no terminal
   attached, OR a caller that passed `interactive=False` because this run is machine output
   (`--json`) or was told not to ask (`--no-input`). Saying "no terminal to ask" on a tty-attached
   `--json` run was a false reason for a true refusal (F11).

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
from typing import Callable, Optional, Union

from rich.console import Console

log = logging.getLogger(__name__)

TrustAsker = Callable[[str], bool]
"""``(question_text) -> bool`` — how a non-terminal channel puts the one-time trust question
(PRD §3.5). Synchronous; see :func:`resolve_workspace_layer` for why, and for the bridge an
async client uses."""

# F6. `--json` already forces the non-interactive path, but `doctor`, `validate` and `agent
# create` have no machine-output mode — so a hook or CI job running them in a directory with an
# untrusted workspace hit the one-time trust prompt, and answering it (or letting it time out into
# whatever the harness felt like) spends a permanent decision nobody was there to make. One help
# string for all three: three copies is two chances to describe the same flag differently.
NO_INPUT_HELP = (
    "Never ask about an untrusted workspace: skip its config layer, say so, and record nothing. "
    "For hooks, CI, and any run with no one watching."
)

# Notices AND the prompt go to stderr: `agent list --json` writes machine-readable JSON on
# stdout, and a trust banner there would corrupt it.
#
# soft_wrap on the CONSOLE, not per-call (F12): every line this module emits carries a workspace
# path, and rich hard-wraps at the terminal width — a deep project path arrives with a newline
# folded into it and the user copies half of it. Setting it here also covers the trust PROMPT,
# which renders through `Console.input` and takes no soft_wrap argument of its own. The question
# a person is answering must show them the whole path they are answering about.
_notice_console = Console(stderr=True, soft_wrap=True)


def resolve_workspace_layer(
    config_dir: Optional[Union[str, Path]] = None,
    *,
    interactive: Optional[bool] = None,
    asker: Optional[TrustAsker] = None,
) -> Optional[Path]:
    """The workspace layer for this invocation, or None. See the module docstring's table.

    ``asker`` is how a channel that is not a terminal puts the trust question (PRD §3.5: "the
    existing workspace-trust dialog becomes the first client of ask_permission"). It is
    SYNCHRONOUS on purpose — every one of this function's eight callers is ordinary synchronous
    CLI code, and making them async to accommodate one caller would be a large change unrelated
    to permissions. An async client (the ACP adapter, phase B) bridges with
    ``await asyncio.to_thread(resolve_workspace_layer, asker=...)`` and, inside its asker,
    ``asyncio.run_coroutine_threadsafe(self.request_permission(text), loop).result()``.

    Passing an asker implies there IS someone to ask, so it also satisfies ``interactive``:
    Zed is not a TTY, and under the letter of phase 39 every ACP session would otherwise
    silently ignore an outside workspace forever. With no asker the phase-39 semantics are
    exactly as before — a non-interactive run stays inert and records nothing, so a later
    interactive session in the same directory still gets asked once.
    """
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

    # `is_trusted_tree`, not `is_trusted`: v0.14.1 added a second place a trust decision can be
    # recorded — the session question records the workspace ROOT, this one records the
    # `.localharness` directory inside it — and one yes has to answer both (owner ruling
    # 2026-09-11: "it asks to trust the workspace… trusted = load its config AND auto"). The
    # walk goes upward only, so trusting a project never trusts the directory above it.
    decision = trust.is_trusted_tree(found)
    if decision is True:
        log.info("workspace layer: %s (trusted)", found)
        return found
    if decision is False:
        _notice(f"Workspace {real} is not trusted — its config layer is ignored.")
        return None

    if interactive is None:
        # An injected asker IS someone to ask, whether or not stdin is a terminal (PRD §3.5).
        interactive = asker is not None or _stdin_is_a_terminal()
    if not interactive:
        # Fail closed (SECURITY.md: deny on doubt) but do NOT record — a later interactive
        # session in this directory still gets asked once.
        _notice(
            f"Found a workspace at {real} from outside this project — ignoring its config "
            "layer, because this run is non-interactive and the question was never asked. "
            "Run an interactive localharness command here to decide."
        )
        return None

    trusted = _ask(real, asker)
    trust.record_trust(found, trusted)
    if trusted:
        log.info("workspace layer: %s (trusted just now)", found)
        return found
    _notice(f"Workspace {real} recorded as not trusted — its config layer is ignored.")
    return None


def _stdin_is_a_terminal() -> bool:
    """Is there a person to ask? `sys.stdin` is None in a detached process, and no stdin at all is
    no terminal either. One definition for both questions this module asks."""
    return sys.stdin is not None and sys.stdin.isatty()


# The offer `start` makes when a project has no workspace at all (owner ruling 2026-09-04). The
# missing step users hit is not "how do I create a workspace" — it is not knowing they could —
# and the moment they would want one is the moment they start the harness inside a project.
# One question, default no, asked only where a person is there to answer it.
OFFER_PROMPT = "No workspace here — create ./.localharness for this project?"


def offer_workspace_creation(
    config_dir: Optional[Union[str, Path]] = None,
    *,
    interactive: Optional[bool] = None,
) -> Optional[Path]:
    """Offer to scaffold `./.localharness` for this project; return the new layer, or None.

    Every guard below is a case where the offer would be wrong, not merely unhelpful:

    1. An explicit `--config-dir` or either env var is a FULL replacement (LAYR-02) — that run
       asked for one specific directory and a project layer is not what it wants.
    2. A workspace already found up-tree means the answer to "is there a workspace here" is yes,
       whatever the trust gate then decided about loading it. Offering to create a second one
       beside a declined first is how you end up with two.
    3. No terminal, or a caller that said `--no-input`: silence. A prompt that fires in a script,
       a hook or CI is a hang, and this one WRITES — it must never fire where nobody is watching.
       EOF answers no for the same reason.
    4. `$HOME` and the machine's global config dir are not projects. `./.localharness` standing in
       home IS the global layer, and "create a workspace" there means overwrite your machine.
    5. A "no" already recorded for this directory. Asked once per directory, ever — the same
       shape as the trust question, and for the same reason: a prompt that returns every time you
       start is one people learn to dismiss without reading (owner ruling 2026-09-04).

    Refusals cost nothing and say nothing — a user who says no must not be asked to read a
    paragraph about it, and is not asked again. Only an ANSWERED prompt records: EOF and every
    silent path above leave the store untouched, so a session that could not ask has not spent
    the decision. `init --workspace` and `mkdir .localharness` never consult it, so a recorded no
    cannot stand between a user and a workspace they went and asked for.

    The creation itself is `init --workspace`'s scaffolder, not a second implementation of it:
    one directory shape, one set of race and error guarantees.
    """
    import typer

    from localharness.config import trust
    from localharness.config.paths import (
        WORKSPACE_DIR_NAME,
        config_dir_env_override,
        discover_workspace_dir,
    )

    if config_dir is not None or config_dir_env_override() is not None:
        return None
    if discover_workspace_dir() is not None:
        return None
    if interactive is None:
        interactive = _stdin_is_a_terminal()
    if not interactive:
        return None
    try:
        target = Path.cwd().resolve() / WORKSPACE_DIR_NAME
    except OSError:  # the directory was deleted under this process — nowhere to create anything
        return None

    # Both privates deliberately: `_is_the_global_config_dir` is the ONE realpath-keyed answer to
    # "is this the machine's own dir" (init_cmd) and `_home_stop` is the ONE home the walks stop
    # at (paths). Re-deriving either here is how two guards that must agree start disagreeing.
    from localharness.cli.init_cmd import _is_the_global_config_dir, _scaffold_workspace
    from localharness.config.paths import _home_stop

    home = _home_stop()
    if _is_the_global_config_dir(target) or (home is not None and target.parent == home):
        return None
    if trust.offer_was_declined(target):
        return None

    answer = _ask_create()
    if answer is not True:
        if answer is False:  # a person said no; EOF (None) is not an answer and records nothing
            trust.record_offer_decline(target)
        return None
    try:
        _scaffold_workspace(endpoint=None, model=None, config_dir=None, next_steps=False)
    except typer.Exit:
        # The scaffolder already printed what went wrong (it owns every filesystem message on this
        # path). Startup continues on the global layer rather than dying halfway into a session
        # the user asked for — a workspace that could not be created is not a reason not to work.
        return None
    return target


def _ask_create() -> Optional[bool]:
    """The offer itself: yes, no, or None for "there was nobody to answer".

    `Text` so a project path in the rendered line can never be read as rich markup. EOF behaves
    like a no for THIS run — a closed stdin is not consent to write to the filesystem — but is
    None rather than False, because a terminal that went away has not decided anything and must
    not spend the one question this directory ever gets.
    """
    from rich.prompt import Confirm
    from rich.text import Text

    try:
        return bool(Confirm.ask(Text(OFFER_PROMPT), console=_notice_console, default=False))
    except EOFError:
        return None


def _notice(message: str) -> None:
    """One line on stderr. `markup=False` because the message carries a filesystem path, and a
    folder named `[old] proj` is legal everywhere while `[old]` is rich markup — parsing it would
    turn a notice into a crashed command."""
    _notice_console.print(message, style="dim", markup=False)


TRUST_QUESTION = (
    "The workspace at {parent} is outside the project you are in. Load its agent and config "
    "files? They define roles, models and tool permissions — treat them like code you are about "
    "to run."
)
"""The one-time question. Names what is at stake AND why this workspace is being asked about,
since the ones inside your own project never are. A module constant now, because a channel that
is not a terminal renders the same words (PRD §3.5: the trust dialog stops being terminal-only)."""


def _confirm_on_a_tty(question: str) -> bool:
    """The default :data:`TrustAsker`: today's `rich` prompt on the terminal, unchanged."""
    from rich.prompt import Confirm
    from rich.text import Text

    return bool(Confirm.ask(Text(question), console=_notice_console, default=False))


def _ask(found: Path, asker: Optional[TrustAsker] = None) -> bool:
    """Put the trust question to whoever is listening.

    Takes the RESOLVED workspace so a symlinked dotdir names the tree the files actually come
    from — "the workspace at ./" is not a question anyone can answer.
    """
    return bool((asker or _confirm_on_a_tty)(TRUST_QUESTION.format(parent=found.parent)))
