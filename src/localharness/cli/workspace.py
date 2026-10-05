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
import re
import shlex
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Union

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

    # Trust is for what you saw (R18): the question lists the MCP servers the workspace's agent
    # files start, and a Yes approves exactly that list.
    snap = trust.executables_snapshot(real)
    trusted = _ask(TRUST_QUESTION.format(parent=real.parent) + _servers_suffix(snap), asker)
    # Recorded on the workspace ROOT, not on the `.localharness` directory inside it, so that
    # ONE answer settles both trust questions (owner ruling 2026-09-11). `session_trust` asks
    # about the root and looks it up with `is_trusted_tree`, which walks UPWARD — from
    # `<root>/.localharness` it reaches `<root>`, but never the other way. Keying the child, as
    # this did before, meant a session with an outside-the-repo workspace answered here and was
    # then asked the session question all over again a second later.
    trust.record_trust(real.parent, trusted)
    if trusted:
        trust.record_executables(real.parent, snap)
        log.info("workspace layer: %s (trusted just now)", found)
        return found
    _notice(f"Workspace {real} recorded as not trusted — its config layer is ignored.")
    return None


def _stdin_is_a_terminal() -> bool:
    """Is there a person to ask? `sys.stdin` is None in a detached process, and no stdin at all is
    no terminal either. One definition for both questions this module asks."""
    return sys.stdin is not None and sys.stdin.isatty()


# The ONE question `start` asks in a project it has never seen (owner bar, 2026-09-11: a live
# end-to-end run met two prompts in a row — "No workspace here — create ./.localharness for this
# project?" and then "Trust this workspace?" — and the bar is one).
#
# They were always one decision wearing two hats. Creating the state store and trusting the place
# are both "yes, I mean to work here"; nobody says yes to one and no to the other, and asking
# twice teaches a person to hit return without reading, which is the failure the whole release is
# about. So there is one sentence, and it says all three things a yes does.
OFFER_PROMPT = (
    "Trust this workspace? LocalHarness will keep its state in ./.localharness here, load that "
    "config, and run tools without asking except for dangerous actions."
)

TRUST_ONLY_PROMPT = (
    "Trust this workspace? LocalHarness will load its .localharness config and run tools here "
    "without asking, except for dangerous actions."
)
"""The same question where ``.localharness`` already exists, so there is nothing to create. Kept
as its own string rather than composed, because the sentence a person reads at 2am is the design
and a half-sentence about creating a directory that is already there reads as a bug."""


def settle_startup_trust(
    config_dir: Optional[Union[str, Path]] = None,
    *,
    interactive: Optional[bool] = None,
) -> Optional[Path]:
    """The ONE startup question: do you trust this workspace? Returns a CREATED layer, or None.

    The return value is narrow on purpose: it is the workspace this call brought into existence,
    and nothing else. Whether an ALREADY-existing workspace's config should load is
    :func:`resolve_workspace_layer`'s answer and only its — returning one from here would have
    handed the caller a workspace whose config the trust gate had just declined.

    Named for what it decides rather than for what it creates, because as of v0.14.1 it decides
    both. It was `offer_workspace_creation` — "No workspace here — create ./.localharness for
    this project?" — and a live end-to-end run met it and then met the trust question a second
    later. The owner's bar is one prompt (2026-09-11), and they were always one decision: nobody
    says yes to keeping state here and no to working here.

    So one sentence, and a yes does all of it — creates `./.localharness` when there is none,
    records the workspace ROOT as trusted (which is the key `cli/session_trust` reads later, so
    that question finds this answer and never fires), and lets the session stay in `auto`. A no
    creates nothing, records the root as untrusted so the session runs `guarded`, and — when
    there was a directory to offer — keeps the create-offer's own "asked once, ever" memory in
    `declined_workspace_offers.yaml`.

    It runs on the synchronous startup path, BEFORE any session store is opened. Session files
    inside the project are never evidence of anything: they can be committed to a repository, so
    a fresh clone would arrive looking "worked in" (only the machine's own store counts as prior
    use, in `cli/session_trust`). When the project's agent files start MCP servers, the question
    lists them and a yes approves exactly that list (`cli/workspace.decide_project_trust` asks
    again, at a later start, only when the list changes).

    Every guard below is a case where the question would be wrong, not merely unhelpful:

    1. An explicit `--config-dir` or either env var is a FULL replacement (LAYR-02) — that run
       asked for one specific directory and a project layer is not what it wants.
    2. A workspace already found up-tree means there is nothing to CREATE — the question loses
       that clause (`TRUST_ONLY_PROMPT`) and asks only about trust. Offering to create a second
       one beside an existing one is how you end up with two.
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
    existing = discover_workspace_dir()
    if interactive is None:
        interactive = _stdin_is_a_terminal()
    try:
        here = Path.cwd().resolve()
    except OSError:  # the directory was deleted under this process — nowhere to create anything
        return None
    target = here / WORKSPACE_DIR_NAME

    # Both privates deliberately: `_is_the_global_config_dir` is the ONE realpath-keyed answer to
    # "is this the machine's own dir" (init_cmd) and `_home_stop` is the ONE home the walks stop
    # at (paths). Re-deriving either here is how two guards that must agree start disagreeing.
    from localharness.cli.init_cmd import _is_the_global_config_dir, _scaffold_workspace
    from localharness.config.paths import _home_stop

    home = _home_stop()
    if _is_the_global_config_dir(target) or (home is not None and target.parent == home):
        return None

    # The workspace ROOT is what the trust decision is about — the project you are standing in,
    # not the dotdir inside it — and it is the key `session_trust` reads later with
    # `is_trusted_tree`, so one answer here settles that question too.
    root = existing.resolve().parent if existing is not None else here
    decided = trust.is_trusted_tree(root)
    if decided is not None:
        return None
    if not interactive:
        # Nobody to ask. Record nothing: an unasked question has not been answered, and the gate
        # falls back to `guarded` for this run (cli/session_trust).
        return None
    if existing is None and trust.offer_was_declined(target):
        return None

    snap = trust.executables_snapshot(existing) if existing is not None else []
    answer = _ask_create((TRUST_ONLY_PROMPT if existing is not None else OFFER_PROMPT)
                         + _servers_suffix(snap))
    if answer is None:
        return None  # EOF is not an answer; nothing recorded, nothing created
    trust.record_trust(root, answer)
    if not answer:
        if existing is None:
            # The create-offer's own memory, kept: a "no" is not re-asked as an offer either.
            trust.record_offer_decline(target)
        _notice(DECLINED_NOTICE.format(root=root))
        return None
    trust.record_executables(root, snap)  # the yes approved exactly the list it showed
    if existing is not None:
        return None
    try:
        _scaffold_workspace(endpoint=None, model=None, config_dir=None, next_steps=False)
    except typer.Exit:
        # The scaffolder already printed what went wrong (it owns every filesystem message on this
        # path). Startup continues on the global layer rather than dying halfway into a session
        # the user asked for — a workspace that could not be created is not a reason not to work.
        return None
    return target


DECLINED_NOTICE = (
    "Workspace {root} is not trusted — this session asks before boundary-crossing and "
    "destructive calls."
)


def _ask_create(prompt: str = OFFER_PROMPT) -> Optional[bool]:
    """The offer itself: yes, no, or None for "there was nobody to answer".

    `Text` so a project path in the rendered line can never be read as rich markup. EOF behaves
    like a no for THIS run — a closed stdin is not consent to write to the filesystem — but is
    None rather than False, because a terminal that went away has not decided anything and must
    not spend the one question this directory ever gets.
    """
    from rich.prompt import Confirm
    from rich.text import Text

    try:
        return bool(Confirm.ask(Text(prompt), console=_notice_console, default=False))
    except EOFError:
        return None


def _notice(message: str) -> None:
    """One line on stderr. `markup=False` because the message carries a filesystem path, and a
    folder named `[old] proj` is legal everywhere while `[old]` is rich markup — parsing it would
    turn a notice into a crashed command."""
    _notice_console.print(message, style="dim", markup=False)


TRUST_QUESTION = (
    "Trust the workspace at {parent}? It is outside the project you are in. LocalHarness will "
    "load its .localharness config — agents, models and tool permissions, which you should treat "
    "like code you are about to run — and run tools there without asking, except for dangerous "
    "actions."
)
"""The one-time question. Names what is at stake AND why this workspace is being asked about,
since the ones inside your own project never are. A module constant, because a channel that is
not a terminal renders the same words (PRD §3.5: the trust dialog stops being terminal-only).

v0.14.1 made it the SAME question as ``cli/session_trust.TRUST_QUESTION`` — both halves in one
sentence — because as of that release it is the same decision and the same record (owner ruling
2026-09-11: "unify… so there is ONE question and ONE record; trusted = load its config AND
auto"). It stayed a separate string only because this one fires before any channel exists, on
the synchronous path that resolves the config layer, and so has to name the directory it found
rather than the workspace root the session is about."""


def _confirm_on_a_tty(question: str) -> bool:
    """The default :data:`TrustAsker`: today's `rich` prompt on the terminal, unchanged."""
    from rich.prompt import Confirm
    from rich.text import Text

    return bool(Confirm.ask(Text(question), console=_notice_console, default=False))


def _ask(question: str, asker: Optional[TrustAsker] = None) -> bool:
    """Put the trust question to whoever is listening. The caller builds it from the RESOLVED
    workspace, so a symlinked dotdir names the tree the files actually come from — "the workspace
    at ./" is not a question anyone can answer."""
    return bool((asker or _confirm_on_a_tty)(question))


# ------------------------------------------------------------------ what a project may start
#
# "No" means no, and trust is for what you saw (orchestrator rulings R5, R6, R18). A project's own
# agent files can name MCP servers — programs started with your environment, or addresses
# connected to with headers. They start only for a trusted project, and only the set you were
# shown: the trust store keeps that set beside the Yes (config/trust.executables_snapshot), and a
# start whose set differs asks once, on a terminal, before anything loads. Nothing here ever asks
# mid-task; a run that cannot ask starts none of the new or changed set and says how to fix that.

TRUST_PROJECT_ENV = "LOCALHARNESS_TRUST_PROJECT"
"""`=1`: trust the project you are in for this run only (any command that starts a session —
`web` and `acp` have no `--trust-project`). Nothing is recorded and nothing is asked."""

SERVERS_SUFFIX = "\nIts agent files start these programs (MCP servers):\n{diff}"

EXECUTABLES_CHANGED_QUESTION = (
    "The MCP servers this project's agent files start have changed since you trusted it:\n"
    "{diff}\nStart them?")

NOT_STARTED_DECLINED = "you said No to them — start asks again next time"
NOT_STARTED_DECLINED_EARLIER = (
    "you said No to them earlier in this session — start asks again next time")
NOT_STARTED_UNASKED = ("they changed since you trusted this project (or were never shown to you) — "
                       "run `localharness start` on a terminal to review them, or {one_run}")
MCP_NOT_STARTED_LINE = "Not starting the MCP servers in {files}: {why}"
NEXT_START_REVIEW_NOTICE = ("Its MCP servers start after you review them at the next "
                            "`localharness start` on a terminal.")

_DECLINED: set[tuple[str, str]] = set()
"""(scope key, fingerprint) pairs answered No in this process: the /plugins restart is a second
_start_async in the same process and must not ask the same question again (orchestrator ruling R18)."""


def _one_run(channel_mode: str) -> str:
    """The one-run escape for the command that was run: `mobile` and `acp` have no --trust-project."""
    return ("set LOCALHARNESS_TRUST_PROJECT=1 for one run" if channel_mode in ("mobile", "acp")
            else "pass --trust-project for one run")


def untrusted_remedy(channel_mode: str, root: Path, *, recorded_no: bool) -> str:
    """Why an untrusted project's servers did not start, and every way to change that (R18)."""
    from localharness.config import trust

    store = trust.trust_store_path()
    if recorded_no:
        return (f"you said No to trusting this project — {_one_run(channel_mode)}, or change its "
                f"entry in {store} to `trusted: true`")
    return (f"this project is not trusted yet — answer Yes to the trust question at a `localharness "
            f"start` on a terminal, {_one_run(channel_mode)}, or add `{root}: {{trusted: true}}` "
            f"to {store}")


@dataclass(frozen=True)
class ProjectTrust:
    """What start decided about the project's own executable content (its MCP servers)."""
    executables: bool
    why: str = ""


def decide_project_trust(workspace: Optional[Path], *, ask: bool, trust_flag: bool,
                         channel_mode: str = "terminal") -> ProjectTrust:
    """Decided once at start (and again at the /plugins restart, which is a new _start_async in the
    same process). Asks only when `ask` (the caller passes channel_mode == "terminal" and not
    --no-input and a terminal on stdin) and only when the project's server set differs from the
    one recorded at the last Yes."""
    from localharness.config import trust

    if workspace is None or trust_flag:
        return ProjectTrust(True)
    root = Path(workspace).resolve().parent
    decision = trust.is_trusted_tree(workspace)
    if decision is not True:
        return ProjectTrust(False, untrusted_remedy(channel_mode, root, recorded_no=decision is False))
    snap = trust.executables_snapshot(workspace)
    stored = trust.recorded_executables(root)
    if stored is None:
        if trust.is_trusted(root) is True or trust.is_trusted(workspace) is True or not snap:
            # A record made for THIS project before this release (or by hand): its Yes predates
            # the list, and adopting what it holds now is the upgrade's one silent step. Or there
            # is nothing to start.
            trust.record_executables(root, snap)
            return ProjectTrust(True)
        stored = trust.NOTHING_APPROVED  # trusted only through a parent folder: never shown
    if stored["fingerprint"] == trust.fingerprint(snap):
        return ProjectTrust(True)
    # Whole entries, never names: two servers may share a name and MCPClientManager.startup
    # connects both, so `[a: evil, a: benign]` must not collapse onto an approved `a: benign`.
    if all(e in stored["servers"] for e in snap):
        trust.record_executables(root, snap)  # only removals (or a reorder): nothing new runs
        return ProjectTrust(True)
    if (str(root), trust.fingerprint(snap)) in _DECLINED:
        return ProjectTrust(False, NOT_STARTED_DECLINED_EARLIER)
    if not ask:
        return ProjectTrust(False, NOT_STARTED_UNASKED.format(one_run=_one_run(channel_mode)))
    if _confirm_or_no(EXECUTABLES_CHANGED_QUESTION.format(
            diff=executables_diff(stored["servers"], snap))):
        trust.record_executables(root, snap)
        return ProjectTrust(True)
    _DECLINED.add((str(root), trust.fingerprint(snap)))
    return ProjectTrust(False, NOT_STARTED_DECLINED)


def _confirm_or_no(question: str) -> bool:
    """The terminal question; a closed stdin (EOF) answers No."""
    try:
        return _confirm_on_a_tty(question)
    except EOFError:
        return False


_UNSHOWABLE = re.compile(r"[\x00-\x1f\x7f-\x9f\u200e\u200f\u202a-\u202e\u2066-\u2069]")


def _visible(text: object) -> str:
    """A value from a file, safe to print inside one line of a question: every control character
    (escape sequences, carriage returns, newlines, bidi overrides) shown as an escape, so a crafted
    name or argument can neither redraw the line nor fake a second one."""
    return _UNSHOWABLE.sub(lambda m: f"\\x{ord(m.group()):02x}" if ord(m.group()) < 0x100
                           else f"\\u{ord(m.group()):04x}", str(text))


def _uncovered(old: list[dict], new: list[dict]) -> list[dict]:
    """The entries of `new` not covered by `old`, counted as a multiset of whole entries: a second
    verbatim copy of an entry is uncovered too."""
    from localharness.config.trust import _canonical

    left = Counter(_canonical(e) for e in old)
    out = []
    for entry in new:
        if left[_canonical(entry)]:
            left[_canonical(entry)] -= 1
        else:
            out.append(entry)
    return out


def _diff(old: list[dict], new: list[dict], *, pair: Callable[[dict], tuple], target: Callable[[dict], str],
          counted: bool, removed: bool) -> str:
    """One two-space-indented line per entry of `new` not covered by `old` ("+", or "~" when it
    pairs — display only — with a gone entry of the same `pair` key, each pairing once), then one
    "-" line per gone entry left unpaired when `removed`; env and header NAMES on a line under."""
    if counted:
        added, gone = _uncovered(old, new), _uncovered(new, old)
    else:
        added, gone = [e for e in new if e not in old], [e for e in old if e not in new]
    lines: list[str] = []
    for entry in added:
        twin = next((g for g in gone if pair(g) == pair(entry)), None)
        if twin is not None:
            gone.remove(twin)
        lines.append(f"  {'~' if twin is not None else '+'} {_visible(entry.get('name'))} "
                     f"({_visible(entry.get('file'))}): {_visible(target(entry))}")
        lines += [f"      {label}: {', '.join(map(_visible, entry[label]))}"
                  for label in ("env", "headers") if entry.get(label)]
    if removed:
        lines += [f"  - {_visible(g.get('name'))} ({_visible(g.get('file'))})" for g in gone]
    return "\n".join(lines)


def _server_target(entry: dict) -> str:
    if entry.get("transport") == "streamable_http":
        return str(entry.get("url") or "")
    return shlex.join([str(entry.get("command") or ""), *map(str, entry.get("args") or [])])


def executables_diff(old: list[dict], new: list[dict], *, removed: bool = True) -> str:
    """The project's servers, old record against the files now: "+ evil2 (a.yaml): /bin/sh -c id"
    for one that is new, "~ evil (a.yaml): …" for one that changed (paired by file and name for
    display only — what is new is decided by whole entries), "- web (b.yaml)" for one that is gone
    (when `removed`), and "env: A, B" under a server that names env vars. Never a value."""
    return _diff(old, new, pair=lambda e: (e.get("file"), e.get("name")), target=_server_target,
                 counted=False, removed=removed)


def _servers_suffix(snap: list[dict]) -> str:
    """The server list appended to a trust question, or "" when the project starts none."""
    return SERVERS_SUFFIX.format(diff=executables_diff([], snap)) if snap else ""


# ------------------------------------------------------------------ what the machine's files may do
#
# Self-extension stays (owner ruling, Option 1): the agent may write the machine's own agent files —
# a new specialist is the harness being used. What such a file STARTS (an MCP server), LOADS (a
# local embedding model is Python sentence-transformers imports) or LOOSENS (a permission looser
# than the shipped default; in a division file or the legacy org.yaml too, since an agent file may
# name a division) applies only after one Yes at a start on a terminal (orchestrator ruling R12a).
# The first start after the upgrade adopts the files as they are (you wrote them); removals and
# tightenings never ask; without a terminal, or after No, the change is withheld at every read of
# that file this run (ConfigLoader(machine_withheld=…)) and named in one line.

MACHINE_CHANGED_QUESTION = (
    "These changed in {global_dir} since you last confirmed them — each starts a program, loads "
    "code or loosens what the agent may do:\n{diff}\nApply them?")
MACHINE_WITHHELD_LINE = ("Not applying {n} change(s) in {global_dir} that you have not confirmed "
                         "({names}): {why}")
MACHINE_UNASKED = "run `localharness start` on a terminal to review them"
MACHINE_DECLINED = "you said No — start asks again next time"
MACHINE_DECLINED_EARLIER = "you said No earlier in this session — start asks again next time"


@dataclass(frozen=True)
class MachineTrust:
    """What start withheld from the machine's own files: {file: {(kind, name), …}} and one line."""
    withheld: Mapping[str, frozenset[tuple[str, str]]] = field(default_factory=dict)
    line: str = ""


def machine_diff(old: list[dict], new: list[dict], *, removed: bool = True) -> str:
    """One line per entry of `new` not covered verbatim by `old`, counted as decide_machine_trust
    counts: "+ g2 (agents/orchestrator.yaml): /bin/sh -c id" for an added one, "~ …" when it pairs
    with a no-longer-present entry of the same (file, kind, name) — that pairing is display only —
    and "- …" for a removed one (only when `removed`), plus "  env: A, B" under an MCP server that
    names env vars. Never a value of env or headers."""
    return _diff(old, new, pair=lambda e: (e.get("file"), e.get("kind"), e.get("name")),
                 target=lambda e: str(e.get("shown") or ""), counted=True, removed=removed)


def decide_machine_trust(global_dir: Path, *, ask: bool) -> MachineTrust:
    """Decided once at start, before anything loads (orchestrator ruling R12a). The agent may write
    the machine's agent files; what they start, load or loosen applies only after one Yes here."""
    from localharness.config import trust

    snap = trust.machine_snapshot(global_dir)
    rec = trust.recorded_machine(global_dir)
    if rec is None:
        trust.record_machine(global_dir, snap)  # the first start after the upgrade: adopted
        return MachineTrust()
    # Counted, never keyed on (file, kind, name): two MCP servers in one file may share a name. A
    # kind the record predates is adopted once — its entries were there before the rule.
    confirmed = [*rec["entries"], *(e for e in snap if e.get("kind") not in rec["kinds"])]
    changed = _uncovered(confirmed, snap)
    if not changed:
        if rec["fingerprint"] != trust.fingerprint(snap) or set(rec["kinds"]) != trust.MACHINE_KINDS:
            trust.record_machine(global_dir, snap)  # removals, tightenings, a newly known kind
        return MachineTrust()
    declined = (trust.machine_key(global_dir), trust.fingerprint(snap))
    if declined in _DECLINED:
        return _withhold(changed, global_dir, MACHINE_DECLINED_EARLIER)
    if not ask:
        return _withhold(changed, global_dir, MACHINE_UNASKED)
    if _confirm_or_no(MACHINE_CHANGED_QUESTION.format(global_dir=global_dir,
                                                      diff=machine_diff(confirmed, snap))):
        trust.record_machine(global_dir, snap)
        return MachineTrust()
    _DECLINED.add(declined)
    return _withhold(changed, global_dir, MACHINE_DECLINED)


def _withhold(changed: list[dict], global_dir: Path, why: str) -> MachineTrust:
    by_file: dict[str, set[tuple[str, str]]] = {}
    for entry in changed:
        by_file.setdefault(entry["file"], set()).add((entry["kind"], entry["name"]))
    names = ", ".join(f"{_visible(e['name'])} ({_visible(e['file'])})" for e in changed)
    return MachineTrust({f: frozenset(v) for f, v in by_file.items()}, MACHINE_WITHHELD_LINE.format(
        n=len(changed), global_dir=global_dir, names=names, why=why))
