"""Shared vocabulary for the permission gate.

Implements the data half of PRD §3 (``.planning/2026-09-11-zed-acp-and-permission-spine-prd.md``).
Pure data: no I/O and no imports from the loop, tools, or channels, so the shell classifier
(``agent/shell_classify``), the verdict (``agent/verdict``), the grant store (``config/grants``),
the effectful gate (``agent/gate``) and every channel renderer share one vocabulary without
import cycles.

Every default rule set below is a named value with its source. None of them is a bare
constant: the sets are the enumerated policy the PRD ratified, and each is overridable from
``permissions.ask`` in config (A3 maps ``AskConfig`` onto ``GateSettings``).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Literal

# --------------------------------------------------------------------------- modes

Mode = Literal["auto", "guarded", "trusted", "read-only", "unattended"]

MODE_STRICTNESS: dict[str, int] = {
    "unattended": 0, "auto": 1, "trusted": 2, "guarded": 3, "read-only": 4,
}
"""Strictness order used by the loader's narrow-only union (PRD §3.3, §3.4).

A project layer may only raise strictness; ``unattended`` is never a default and is never
settable from a channel command. ``auto`` sits one rung above ``unattended`` because it is the
first mode that still asks about anything at all, and below ``trusted`` because ``trusted``
asks about every destructive command while ``auto`` asks only about the ones that point outside
the project (owner ruling 2026-09-11).
"""

DEFAULT_MODE: Mode = "auto"
"""The default for every channel that can ask.

Owner ruling 2026-09-11, after using the shipped ``guarded`` default for a day: "way too
intrusive, it stopped me multiple times… the default should be an auto mode that almost never
triggers unless genuinely risky / dangerous… essentially the thinnest interaction off of no
interaction". ``guarded`` — which asks once per new shell signature, once per write outside the
project, once per interpreter, and on EVERY write when the session starts in ``$HOME`` — is
still there, one ``/mode guarded`` away, for anyone who wants it. It is no longer what a person
meets on first run.
"""

AUTO_ASK_CLASSES: frozenset[str] = frozenset({"protected-path", "shell-destructive"})
"""The only two ask classes ``auto`` still raises (owner ruling 2026-09-11: "auto = thinnest
interaction; asks only when genuinely dangerous").

Everything else a call can raise — a first-exposure shell signature, a write outside the
project, an inline interpreter, a subagent dispatch, an MCP tool, a fetch from a new host, a
session with no boundary at all — is allowed silently: none of it is irreversible, and asking
about all of it is the prompt fatigue that trains a person to answer "yes" without reading.
What is left is the pair that cannot be undone by the next command: writing a path whose
contents decide what runs later (:data:`PROTECTED_PATHS_HOME_DEFAULT`,
:data:`PROTECTED_PATHS_SYSTEM_DEFAULT`, :data:`PROTECTED_PATHS_WORKSPACE_DEFAULT`), and a
destructive shell command — the latter narrowed further by
:data:`TARGET_SCOPED_DESTRUCTIVE_VERBS`, because ``rm -rf build`` inside your own project is
routine.

``auto`` also asks about an ask that carries NO key, whatever its class: a command the gate
could not read (``{"command": ["rm", "-rf", "/"]}``), a command name computed at runtime
(``$RM -rf build``), a path argument that is not a string. Those are not benign classes, they
are calls the gate could not classify, and the whole rule above rests on having classified the
call. See :func:`localharness.agent.verdict._ask_record`.
"""


# --------------------------------------------------------------------------- verdicts

class Verdict(Enum):
    """PRD §3.1: the three outcomes of the pure verdict, evaluated deny → ask → allow."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


AskClass = Literal[
    "shell-destructive",
    "protected-path",
    "no-boundary",
    "edit-outside",
    "edit-unreviewed",
    "shell-unfamiliar",
    "interpreter-inline",
    "code-exec",
    "delegate",
    "tool-unfamiliar",
    "mcp",
    "network-host",
]

UNGRANTABLE_CLASSES: frozenset[str] = frozenset({"shell-destructive", "protected-path", "no-boundary"})
"""PRD §3.1 ungrantable tier: asks every time in guarded and trusted; allowed in unattended,
which is v0.13's behavior by definition (only the DENY tier holds there, PRD §3.4).

Checked before any grant lookup (critic finding 12) so an old benign grant can never cover a
destructive variant.
"""


@dataclass(frozen=True)
class PermissionRequest:
    """What a channel renders when the verdict is ASK (PRD §3.5)."""

    tool_name: str
    tool_params: dict[str, Any]
    klass: str
    key: str | None
    grantable: bool
    reason: str
    display: str
    """One-line human rendering, e.g. ``bash: rm -rf build/  (destructive, asks every time)``."""

    grant_keys: tuple[tuple[str, str], ...] = ()
    """Every ``(klass, key)`` an ``allow_always`` answer should remember, not just the primary.

    One tool call can raise several asks at once (two unfamiliar commands writing into two new
    directories). They are merged into ONE request — ``klass``/``key`` name the most severe of
    them — and this carries the rest, so a single "always here" ends the whole call's asking
    (PRD §7: "the same command never asks again in that workspace"). Empty means "just
    ``(klass, key)``". Ignored when ``grantable`` is False: one ungrantable ask makes the whole
    request ungrantable and nothing durable is written."""

    agent_id: str | None = None
    """WHO is asking — the id of the agent whose tool call raised this (PRD §3.4: subagents
    share the parent's gate, and therefore its channel).

    One gate serves an orchestrator and every subagent it dispatches, so without this a renderer
    could only say "a tool call wants to run `rm -rf`" and the person answering had no way to
    tell which agent's call it was. Optional because a caller that builds a request by hand (a
    test, a channel replaying one) has no agent to name; the gate fills it in on every real ask.
    ``PermissionGate`` also prefixes ``display`` with it when the asker is not the session's own
    agent, so a channel that renders nothing but the one line still shows who asked."""

    options_legend: str | None = None
    """A replacement for the channel's own key legend, for a question that is not a tool call.

    The workspace-trust question (``cli/session_trust``) is the one user of it: it is rendered
    through this same request shape so it lands where every other permission question lands, but
    the ungrantable legend ends "(asks every time — cannot be remembered)" and this answer is
    precisely the one that IS remembered. None everywhere else, and a channel that does not read
    it simply draws its own two options — the same question with plainer buttons."""

    auto_entry: str | None = None
    """Which :class:`AutoBlacklist` entry raised this, when ``auto`` is the mode.

    Reported by ``localharness ask-rate --mode auto`` so the blacklist can be curated from what
    actually fires over a real corpus. None in every other mode, and on an ask that ``auto``
    would not have raised at all."""

    call_id: str | None = None
    """The tool call's own id (``Action.tool_call_id``), so a channel can pair the question with
    the call it is about.

    ACP is the reason it has to travel on the request: the client renders a permission dialog
    against a `tool_call` it already knows about, and without the id the adapter can only guess
    which one — the last call it saw — which is wrong the moment a model emits two tool calls in
    one response. Optional for the same reason as ``agent_id``."""


@dataclass(frozen=True)
class VerdictResult:
    verdict: Verdict
    reason: str = ""
    request: PermissionRequest | None = None


DecisionKind = Literal["allow_once", "allow_always", "reject_once", "reject_always"]
"""The four ACP ``PermissionOptionKind`` values; every channel maps its UI onto these."""


@dataclass(frozen=True)
class Decision:
    kind: DecisionKind

    @property
    def allowed(self) -> bool:
        return self.kind.startswith("allow")

    @property
    def remembered(self) -> bool:
        return self.kind.endswith("always")


Asker = Callable[[PermissionRequest], Awaitable[Decision]]
"""A channel's rendering of an ASK: awaitable so the loop pauses until a human answers."""


@dataclass(frozen=True)
class GateOutcome:
    """Result of the effectful ``PermissionGate.check`` (A4): the loop only needs allow/deny."""

    allowed: bool
    reason: str = ""


@dataclass(frozen=True)
class ToolMeta:
    """The schema facts the verdict needs, decoupled from ``ToolSchema`` (PRD §3.1 table)."""

    destructive: bool = False
    group: str = "other"
    is_mcp: bool = False
    mcp_server: str | None = None


# --------------------------------------------------------------------------- shell

@dataclass(frozen=True)
class ShellSegment:
    """One command after PRD §3.2 structural classification."""

    signature: str
    argv: tuple[str, ...]
    read_only: bool = False
    destructive: bool = False
    inline_interpreter: bool = False
    write_targets: tuple[str, ...] = ()
    unresolvable_write: bool = False

    target_scoped_destructive: bool = False
    """True when this segment's destructiveness is entirely a question of WHERE it points —
    its verb is in :data:`TARGET_SCOPED_DESTRUCTIVE_VERBS` (owner ruling 2026-09-11).

    False on every other destructive segment, including every non-destructive one: ``sudo``,
    ``curl … | sh``, ``git push --force`` and ``dd`` are irreversible wherever they run, so
    ``auto`` asks about them regardless of target."""

    destructive_targets: tuple[str, ...] = ()
    """The paths a target-scoped destructive segment operates ON, joined onto the directory the
    segment runs in exactly as :attr:`write_targets` are.

    Kept apart from ``write_targets`` on purpose: ``rm`` does not WRITE its operands, it deletes
    them, and folding the two together would change what ``guarded`` asks about. Empty on every
    segment whose ``target_scoped_destructive`` is False."""

    pipe_to_shell: bool = False
    """True when this segment is the SINK of a ``curl … | sh`` (:data:`PIPE_TO_SHELL_SINKS_DEFAULT`).

    Carried separately from ``destructive`` because the sink's signature — a bare ``sh``, a
    ``python3`` — says nothing about why it is dangerous, and ``auto``'s blacklist is read off
    signatures for everything else."""

    unresolvable_destructive: bool = False
    """True when a target-scoped destructive segment names a target the classifier cannot place
    — a variable, a glob, a substitution, a ``cd`` it could not follow — or names none at all.

    ``auto`` reads it as "ask": the whole narrowing rests on knowing where the command points,
    so not knowing is the one answer that cannot be allowed."""


@dataclass(frozen=True)
class ShellClassification:
    segments: tuple[ShellSegment, ...]
    dropped: tuple[str, ...] = ()

    @property
    def signatures(self) -> tuple[str, ...]:
        return tuple(s.signature for s in self.segments)

    @property
    def destructive(self) -> bool:
        return any(s.destructive for s in self.segments)

    @property
    def inline_interpreter(self) -> bool:
        return any(s.inline_interpreter for s in self.segments)

    @property
    def write_targets(self) -> tuple[str, ...]:
        return tuple(t for s in self.segments for t in s.write_targets)

    @property
    def unresolvable_write(self) -> bool:
        return any(s.unresolvable_write for s in self.segments)


# --------------------------------------------------------------------------- grants

@dataclass(frozen=True)
class Grant:
    """PRD §3.3: a remembered "allow always". Provenance is mandatory; a record without it is invalid."""

    key: str
    klass: str
    granted_at: str
    """ISO-8601 UTC."""
    channel: str
    session_id: str
    workspace: str
    """Realpath of the workspace the grant belongs to; nested workspaces inherit."""


@dataclass(frozen=True)
class Refusal:
    """PRD §3.3: a remembered "never here" — a grant's negative twin, same key space.

    A "never" is STRUCTURAL, not a text pattern: it is filed under the very key the prompt
    offered ("this shell signature", "this directory", "this MCP tool"), so it denies exactly
    the calls that key names and nothing else. The earlier shape turned a refusal into an
    fnmatch DENY pattern (``bash_exec(*cp*)``), which also banned ``scp``, ``cpio`` and any
    command mentioning a path containing "cp" — far more than the human answered.

    Provenance is mandatory, exactly as for :class:`Grant`. A refusal wins over any grant on
    the same key, in any mode, and cannot be undone from a prompt (edit ``grants.yaml``).
    """

    key: str
    klass: str
    refused_at: str
    """ISO-8601 UTC."""
    channel: str
    session_id: str
    workspace: str
    """Realpath of the workspace the refusal belongs to; nested workspaces inherit."""


# --------------------------------------------------------------------------- settings

READ_ONLY_SIGNATURES_DEFAULT: frozenset[str] = frozenset({
    "ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "tree", "which", "env", "printenv",
    "stat", "file", "du", "df", "date", "uname", "pwd", "echo", "true", "false", "type",
    "git status", "git diff", "git log", "git show", "git branch", "git rev-parse", "git remote",
    "git remote show", "git remote get-url", "git stash list", "git stash show",
    "sed -n", "sort", "uniq", "cut", "tr", "basename", "dirname", "realpath", "readlink",
    "id", "whoami", "hostname", "nproc", "free", "uptime", "ps", "ss", "lsof",
    "docker ps", "docker logs", "docker images", "docker inspect", "docker version", "docker info",
})
"""PRD §3.1 ALLOW tier for shell. Union of Claude Code's built-in read-only Bash set
(code.claude.com/docs/en/permissions) and the read-only tokens observed in the 384-session
dogfood corpus (PRD §5). ``find`` is read-only only without ``-exec``/``-delete`` (the classifier
lifts those payloads, PRD §3.2 step 5).

The docker row is the other half of narrowing the destructive set below: asking about
``docker ps`` is the fatigue the ASK tier exists to avoid (docker-ps(1), docker-logs(1),
docker-images(1), docker-inspect(1) — all report, none of them start, stop or remove anything)."""

WINDOWS_DESTRUCTIVE_FLAG_VERBS: dict[str, tuple[str, ...]] = {
    "Remove-Item": ("-Recurse", "-Force"),
    "ri": ("-Recurse", "-Force"),
    "del": ("/s", "/q"),
    "erase": ("/s", "/q"),
    "rd": ("/s", "/q"),
    "rmdir": ("/s", "/q"),
}
"""The Windows half of :data:`DESTRUCTIVE_FLAG_VERBS_DEFAULT`, spelled the way Windows spells it.

``bash_exec`` on Windows runs git-bash, and git-bash will happily start ``powershell`` or ``cmd``
(v0.14 critic A4), so the delete verbs on the other side of that door need the same treatment
``rm`` gets: the flag that makes them recursive is in the signature, and the plain verb is a
different key. ``Remove-Item`` and its alias ``ri`` are the PowerShell cmdlets; ``del``/``erase``
and ``rd``/``rmdir`` are the cmd builtins (about_Remove-Item, del(1)/rmdir(1) in the Windows
Commands reference).

The flags carry their own lead character because the two shells disagree about it — PowerShell
uses one dash and a whole word, cmd uses a slash — and the classifier matches them the way each
shell does: a dash flag case-insensitively by prefix (``-Recurse``, ``-recurse``, ``-rec``,
``-r`` are one flag), a slash flag case-insensitively but whole. The third delete verb,
``rm``, needs no entry: it is PowerShell's alias for ``Remove-Item`` too, and ``rm -Recurse``
already canonicalizes to ``rm -r`` through the POSIX cluster rule."""

WINDOWS_DESTRUCTIVE_SIGNATURES: frozenset[str] = frozenset(
    " ".join((verb, *combination))
    for verb, flags in WINDOWS_DESTRUCTIVE_FLAG_VERBS.items()
    for size in range(1, len(flags) + 1)
    for combination in itertools.combinations(flags, size)
)
"""Every flag combination of :data:`WINDOWS_DESTRUCTIVE_FLAG_VERBS`, derived rather than typed.

The classifier emits the flags it found in the verb's own canonical order, so ``del /s /q`` and
``del /q`` and ``del /s`` are three different signatures and all three have to be in the
destructive set — exactly the shape ``rm -r`` / ``rm -f`` / ``rm -rf`` has above. Deriving them
from the one flag table is what keeps the two in step: a flag added there can never go missing
here."""

DESTRUCTIVE_SIGNATURES_DEFAULT: frozenset[str] = WINDOWS_DESTRUCTIVE_SIGNATURES | frozenset({
    "rm -r", "rm -f", "rm -rf",
    "format", "diskpart",
    "git push --force", "git reset --hard", "git clean -f",
    "git branch -D", "git branch -d", "git branch -M",
    "git remote set-url", "git remote remove", "git remote rm", "git remote prune",
    "git stash drop", "git stash clear",
    "git checkout --", "git restore", "git restore --worktree",
    "git reflog expire", "git gc --prune", "git filter-branch", "git push --delete",
    "git worktree remove --force", "git submodule deinit --force",
    "chmod -R", "chown", "chgrp",
    "sudo", "su", "doas",
    "docker exec", "docker run", "docker start", "docker restart", "docker stop", "docker kill",
    "docker rm", "docker rmi", "docker system prune",
    "docker container exec", "docker container run", "docker container start",
    "docker container restart", "docker container stop", "docker container kill",
    "docker container rm", "docker container prune", "docker image rm", "docker image prune",
    "docker volume rm", "docker volume prune", "docker network rm",
    "docker compose up", "docker compose down", "docker compose exec", "docker compose run",
    "docker compose rm",
    "docker-compose up", "docker-compose down", "docker-compose exec", "docker-compose run",
    "docker-compose rm",
    "dd", "mkfs", "shred", "truncate",
    "find -delete",
})
"""PRD §3.1 ungrantable ``shell-destructive`` class: matched on the canonical signature
(flags included, PRD §3.2 step 7), before any grant lookup. Sources: PRD §3.1 table; Claude Code
auto-mode block list (force push, ``git reset --hard``, recursive delete); the project's own
shipped deny defaults (sudo, ``rm -rf``, docker). Pipe-to-shell is a separate rule below.

A BARE name here condemns every spelling of it — right for ``sudo``, ``su``, ``doas``, ``dd``,
``mkfs`` and ``shred``, which do only one thing. Docker is not like that, so it is enumerated by
subcommand instead (owner ruling on review finding R9): the entries above are the ones that run
code on the host (``exec``, ``run``, ``compose up/run/exec``) or destroy state (``stop``,
``kill``, ``rm``, ``rmi``, ``prune``, ``compose down``), the ``docker container …`` /
``docker image …`` management spellings of the same operations, and both spellings of compose
carry their own subcommand. Everything else docker does is left where it belongs — ``ps``, ``logs``,
``images``, ``inspect``, ``version`` and ``info`` are reads in the ALLOW tier above, and
``build``, ``pull``, ``push``, ``tag``, ``login`` are ordinary unfamiliar commands the human can
grant once. The bare ``docker`` entry this replaces made every one of those ungrantable, which is
ask-fatigue on commands nobody needs protection from. The shipped deny defaults still hard-deny
the worst of these first (stop, kill, rm, ``compose down``): DENY is a tier above ASK, not a
substitute for these entries.

The git block is the same judgement applied to git's own operations (v0.14 critic A2). ``git
branch`` and ``git remote`` are READS in the ALLOW tier above, and every operation they carry as a
third word or a flag used to collapse onto those two keys — so the read-only verdict covered
``git branch -D``, and ``git remote set-url origin http://attacker/`` (every later push and pull
redirected) asked nothing at all. The entries here are the ones that destroy work that is not
recoverable from the repository itself: a deleted branch or stash, a discarded worktree change
(``git checkout --``, ``git restore`` with no ``--staged``), an expired reflog or pruned object
(``git reflog expire``, ``git gc --prune`` — the two commands that throw away the copies the other
deletes are recoverable FROM), a rewritten history (``git filter-branch``), a deleted remote
branch (``git push --delete``), and a forced removal of a worktree or submodule with whatever was
in it. Everything else git does stays grantable on its own key: ``git remote add``, ``git stash``
and its push/pop/apply, ``git checkout BRANCH``, ``git switch``, ``git worktree add``,
``git submodule update``.

``format`` and ``diskpart`` are bare names for the same reason ``mkfs`` is: each does exactly one
thing, and it is not recoverable (Windows Commands reference).

Docker's management-command spellings ARE in this set (``370513d``): ``docker container rm`` and
``docker container exec/run/start/restart/stop/kill``, ``docker image rm``, and the ``prune`` of
each group are the same operations as the short spellings beside them, so they are ungrantable for
the same reason. The residual is narrower than it was, and named: a management group holds reads
too — ``docker container ls``, ``docker image ls``, ``docker network inspect`` — and those are
neither in this set nor in the read-only tier above, so they classify as unfamiliar and can be
granted, one operation at a time, which is what :data:`NESTED_SUBCOMMAND_GROUPS` in the classifier
buys."""

DESTRUCTIVE_FLAG_VERBS_DEFAULT: dict[str, tuple[str, ...]] = {
    "rm": ("r", "f"),
    "chmod": ("R",),
    "git push": ("force",),
    "git reset": ("hard",),
    "git clean": ("f",),
    "git worktree remove": ("force",),
    "git submodule deinit": ("force",),
    **WINDOWS_DESTRUCTIVE_FLAG_VERBS,
}
"""Verbs whose destructive variant is a flag. The classifier canonicalizes only these flags
into the signature (``rm -r -f x`` → ``rm -rf``; ``git push -f`` → ``git push --force``) so a grant on
``rm`` never covers ``rm -rf`` (critic finding 12). Long forms map: ``--recursive``→r,
``--force``→f/force, ``--recursive`` for chmod→R.

The two three-word git keys are the operations that are only destructive when forced:
``git worktree remove`` and ``git submodule deinit`` both refuse to throw away uncommitted work
until ``--force`` is passed, so the unforced spelling stays grantable and the forced one is in the
ungrantable set above (git-worktree(1), git-submodule(1))."""

PIPE_TO_SHELL_SOURCES_DEFAULT: frozenset[str] = frozenset({
    "curl", "wget", "iwr", "Invoke-WebRequest",
})
"""The fetch side of the pipe-to-shell rule: a segment with one of these signatures piped into a
sink below makes the SINK destructive (PRD §3.1; Claude Code block list). ``curl`` and ``wget``
are the two fetchers the dogfood corpus (PRD §5) actually used; the rule is overridable from
``permissions.ask.pipe_to_shell_sources`` for anyone whose install ships another.

``iwr`` and its full spelling ``Invoke-WebRequest`` are the PowerShell fetcher, added with the
rest of the Windows row (v0.14 critic A4) — ``iwr http://x | iex`` is the download-and-run idiom
every PowerShell install instruction uses, and it is the same rule, on the other shell. (On
Windows, ``curl`` and ``wget`` are aliases for this cmdlet, so those two already carried.)"""

PIPE_TO_SHELL_SINKS_DEFAULT: frozenset[str] = frozenset({
    "sh", "bash", "zsh", "python", "python3",
    "powershell", "pwsh", "cmd", "iex", "Invoke-Expression",
})
"""The execute side of the same rule: ``curl … | sh`` and friends, where the sink segment is
destructive (PRD §3.1; Claude Code block list). Read with the sources above — neither half means
anything alone, and a bare ``sh`` is an interpreter, not a destructive command.

The second row is PowerShell's and cmd's half of the same idiom (v0.14 critic A4). ``iex`` is the
alias for ``Invoke-Expression``, which runs a string as code — the exact sink ``sh`` is, spelled
in the shell the owner's Windows machine actually runs."""

INTERPRETER_COMMANDS_DEFAULT: frozenset[str] = frozenset({
    "python", "python3", "bash", "sh", "zsh", "node", "uv run", "ruby", "perl",
    "fish", "ksh", "dash", "script",
    "powershell", "pwsh", "cmd",
})
"""Commands whose SIGNATURE carries an interpreter mode (PRD §3.2 step 7, critic finding 5).

``python3 -c``, ``python3 -m MOD``, ``python3 <script>`` and bare ``python3`` are four different
grant keys, so answering "always" about an inline one-liner is never an answer about a script.
The list is the interpreters present in the dogfood corpus (PRD §5), where ``python3`` is the
second signature nearly every workspace is asked about.

``fish``, ``ksh`` and ``dash`` are here because a shell the list forgets is a shell the gate
never opens — their ``-c`` payload is recursed into like ``bash -c``'s (review finding R9).
``script`` is in the same row for the same reason: ``script -c CMD FILE`` runs CMD through a
shell to record the session (script(1)).

``powershell``, ``pwsh`` and ``cmd`` are the Windows row (v0.14 critic A4). ``bash_exec`` on
Windows runs git-bash, which can start either of them, and
``powershell -Command "Remove-Item -Recurse -Force X"`` signed as a bare, grantable
``powershell`` — one "always" on which covered every future command line the model cared to put
behind it. They are interpreters exactly as ``bash`` is, so the mode goes in the key and the
payload is opened."""

INLINE_CODE_FLAGS_DEFAULT: dict[str, tuple[str, ...]] = {
    "python": ("-c",), "python3": ("-c",), "uv run": ("-c",),
    "bash": ("-c",), "sh": ("-c",), "zsh": ("-c",),
    "fish": ("-c",), "ksh": ("-c",), "dash": ("-c",), "script": ("-c",),
    "node": ("-e", "--eval", "-p", "--print"),
    "ruby": ("-e",), "perl": ("-e", "-E"),
    "powershell": ("-Command", "-EncodedCommand", "-File"),
    "pwsh": ("-Command", "-EncodedCommand", "-File"),
    "cmd": ("/c", "/k"),
}
"""Per-interpreter flags that mean "the code is on the command line" (PRD §3.2 step 7).

The table that turns a command into the ``interpreter-inline`` class and puts the mode in the
grant key. Each row is that interpreter's own documented inline flags — ``node`` carries four
because ``-p``/``--print`` evaluate too. Deliberately NOT overridable from config (see
``AskConfig``): it is a canonicalization table, and a wrong row silently changes what a grant
key means.

The Windows rows carry only the CANONICAL spelling of each mode on purpose (v0.14 critic A4).
PowerShell and cmd match their options case-insensitively and accept any unambiguous prefix, so
the classifier matches these rows that way too and returns the canonical spelling — ``-c``,
``-command``, ``-Comm`` and ``-Command`` are one mode and therefore one grant key, and
``/C`` is ``/c``. Listing the abbreviations here would do the opposite: an exact hit on ``-c``
would file the same command under a second key (powershell(1) / pwsh(1) "-Command",
"-EncodedCommand", "-File"; cmd(1) "/c", "/k")."""

INLINE_BY_NATURE_DEFAULT: frozenset[str] = frozenset({"eval", "awk", "gawk", "mawk"})
"""Commands that are inline interpreters with no flag at all (PRD §3.1 ``interpreter-inline``,
grantable per owner ruling 2026-09-11 §9.3). ``eval`` takes its code as plain arguments, so
there is no mode flag to key on — the command itself is the signature.

The ``awk`` family is here for the same reason and with one difference (v0.14 critic A5): its
first positional is a PROGRAM, in awk's own language, and that program can call ``system()`` or
write through ``print > "file"``. So ``awk`` is an inline interpreter — the segment carries
``inline_interpreter`` and the human is told this command runs code — but the program text is NOT
recursed into as shell, because it is not shell (the classifier's ``OPAQUE_INLINE_PROGRAMS`` names
that half). ``perl -e`` and ``ruby -e`` reach the same class through their flags instead, being
interpreters with a mode. Source: awk(1) ``system()`` and output redirection."""

WRAPPER_COMMANDS_DEFAULT: frozenset[str] = frozenset({
    "env", "nohup", "time", "nice", "ionice", "command", "builtin", "exec", "timeout", "stdbuf",
    "watch", "flock", "setsid", "chroot", "busybox", "caffeinate", "unbuffer", "systemd-run",
    "poetry", "pipx", "uvx", "wsl",
})
"""PRD §3.2 step 4: peeled so the signature is the wrapped command's. ``sudo`` is NOT a wrapper
(destructive before peeling).

The second row is the review's finding R9: every one of these runs its argument as a command, so
before they were peeled ``watch cp x ~/.ssh/authorized_keys`` signed as an unfamiliar ``watch``
with no write target at all. Their per-command argument shapes (``watch -n N``, ``flock`` and
``chroot``'s leading operand, ``poetry run`` / ``pipx run``'s required subcommand) live with the
peeling logic in ``shell_classify``, which is where the other canonicalization tables are.

``wsl`` is the Windows door in the other direction (v0.14 critic A4): from git-bash it starts a
command inside the Linux VM, and ``wsl rm -rf x`` / ``wsl -e rm -rf x`` / ``wsl -- rm -rf x``
all signed as a bare, grantable ``wsl``. It is a wrapper rather than an interpreter because its
payload is argv and not a string — peeling reaches the real ``rm -rf`` key, which is a better
answer than an opaque ``wsl -e``. The Linux side of that VM is not this workspace; the boundary
check on the peeled command's targets is what decides, exactly as it does for ``chroot``."""

DROPPED_COMMANDS_DEFAULT: frozenset[str] = frozenset({
    "cd", "pushd", "popd", "export", "pwd", "true", ":",
})
"""PRD §3.2 step 6: dropped only when the segment is exactly this command with no substitution
inside. ``cd`` was the most frequent first token in the corpus, always as ``cd X && …``.

Dropped does not mean ignored: navigation moves the shell, so the classifier tracks the
directory across the sequence and joins relative write targets onto it (``cd ~/.ssh && echo x >>
authorized_keys`` is a write to ``~/.ssh/authorized_keys``, review finding R2a). ``pushd`` and
``popd`` are here for the same reason ``cd`` is — they are navigation, and leaving them out made
them unfamiliar commands that asked."""

SUBCOMMAND_TOOLS_DEFAULT: frozenset[str] = frozenset({
    "git", "uv", "pip", "pip3", "npm", "npx", "pnpm", "yarn", "docker", "docker-compose",
    "cargo", "make", "gh", "kubectl",
})
"""PRD §3.2 step 7: the signature includes the first subcommand (``git status`` ≠ ``git push``).

``docker-compose`` is the old standalone spelling of ``docker compose`` and needs the same
treatment, or ``docker-compose up`` and ``docker-compose ps`` would share one key."""

PAYLOAD_COMMANDS_DEFAULT: frozenset[str] = frozenset({"find", "xargs", "parallel"})
"""PRD §3.2 step 5: ``find -exec/-execdir/-ok CMD``, ``xargs CMD`` and ``parallel CMD ::: args``
lift CMD into its own segment (critic finding 4).

The set is a real knob (review finding R9b): any command listed here has its payload read as
"the first non-option token onward", so an install whose runner is not in this row can add it in
config without a code change. GNU ``parallel`` is in the defaults because it is the runner the
review used as its example — ``parallel rm -rf {} ::: a b`` signed as an unfamiliar
``parallel``."""

WRITE_SHAPED_COMMANDS_DEFAULT: frozenset[str] = frozenset({
    "tee", "cp", "mv", "install", "rsync", "dd", "curl", "wget", "ln", "touch", "mkdir", "unzip",
    "tar", "git",
})
"""PRD §3.2 step 8 (critic finding 2): commands whose arguments name a write target, checked
against the boundary and the protected set on every call. Redirections ``>``/``>>`` apply to any
segment.

``git`` is here for the four subcommands that choose where a repository goes — ``clone``, ``init``,
``worktree add``, ``submodule add`` (v0.14 critic A3). ``git clone https://x ~/.ssh`` reported no
write target at all, so nothing checked the destination against the protected set, and a clone
writes a ``.git/hooks`` directory git will later execute. Every other git signature names no
target: git writes inside the repository it is already in, which the boundary already covers. The
per-subcommand destination table lives with the other canonicalization tables in
``shell_classify``."""

GIT_CONFIG_DANGEROUS_KEYS_DEFAULT: tuple[str, ...] = (
    "core.hooksPath", "core.sshCommand", "core.fsmonitor", "core.pager", "core.editor",
    "core.askPass", "credential.helper",
    "include.path", "includeIf.*",
    "alias.*", "url.*.insteadOf",
    "diff.*.command", "diff.external", "filter.*.clean", "filter.*.smudge", "merge.*.driver",
    "sequence.editor", "gpg.program",
)
"""Git config keys whose VALUE is a command git will run later (PRD §3.1 ``shell-destructive``).

Writing one of these repoints future execution: ``core.hooksPath`` makes an ordinary ``git commit``
run a script of the attacker's choosing, ``core.pager`` and ``diff.external`` turn a *read*
(``git log``, ``git diff``) into an exec, ``credential.helper`` and ``core.askPass`` hand over
secrets, ``url.*.insteadOf`` silently redirects a fetch to another host, and ``include.path`` /
``includeIf.*`` pull in a whole config file that can set any of the others. That is why they are
ungrantable rather than merely unfamiliar: the damage happens on some *later* command, so an
"always" answer here would be an answer about a command the human never sees.

Derivation: every git-config(1) key whose documented value is a command line or a path git later
executes or includes, restricted to the ones reachable from a single ``git config`` write (the
hook, pager, editor, credential, alias, external-diff/filter/merge-driver and include families).
Source: git-config(1) and githooks(5); raised by the v0.14 critic as F3(a), because the shipped
classifier filed all of these under one grantable ``git config`` key.

``*`` is a glob over the rest of the key (``alias.*`` covers every alias; ``url.*.insteadOf``
covers every URL base, dots and slashes included), matched case-insensitively because git treats
section and variable names that way. Over-matching is the safe direction: it asks.
"""

SOURCE_COMMANDS_DEFAULT: frozenset[str] = frozenset({"source", "."})
"""Shell builtins that run a FILE's contents in the current shell (PRD §3.2 step 7).

``source ~/.bashrc`` and ``. ./env.sh`` are the same command and are an interpreter by nature:
nothing on the command line says what will run. They get the ``<script>`` placeholder key that
``python3 <script>`` gets, and carry ``inline_interpreter`` so the gate files them under
``interpreter-inline`` — grantable per owner ruling §9.3, with the interpreter gap named in
SECURITY.md rather than hidden behind a signature that looks like a plain command."""

PROTECTED_PATHS_HOME_DEFAULT: tuple[str, ...] = (
    "~/.ssh", "~/.aws", "~/.gnupg", "~/.config/gh", "~/.kube", "~/.docker",
    "~/.bashrc", "~/.zshrc", "~/.profile", "~/.bash_profile", "~/.zprofile",
    "~/.localharness",
    "~/.git-credentials", "~/.netrc", "~/.npmrc", "~/.pypirc",
    "~/.config/gcloud", "~/.azure",
    "~/AppData/Roaming/GitHub CLI", "~/AppData/Local/Microsoft/Credentials",
    "~/Documents/PowerShell", "~/Documents/WindowsPowerShell",
)
"""PRD §3.1 ``protected-path`` (ungrantable). ``~/.localharness`` is protected because writing it
changes what the harness does next; the harness's own runtime store under it is exempted by the
verdict (A2). Source: Claude Code "protected paths" tier + PRD critic finding 1.

The second block is the credential files the first one missed (v0.14 critic A6), each one a
plaintext token store for a service that can publish or deploy: ``~/.git-credentials`` (git's own
``store`` helper, git-credential-store(1)), ``~/.netrc`` (curl, wget and git fall back to it,
netrc(5)), ``~/.npmrc`` and ``~/.pypirc`` (the npm and PyPI upload tokens — a write here
redirects a publish to another registry, npmrc(5), distutils "The .pypirc file"), and
``~/.config/gcloud`` / ``~/.azure`` (the two cloud SDK credential stores that were missing beside
``~/.aws``). ``~/.docker`` already covers ``~/.docker/config.json``: a directory entry protects
its subtree.

The third block is the same tier on Windows, where the owner also dogfoods: GitHub CLI's token
store, the Windows Credential Manager's file backing, and the two PowerShell profile directories —
whose ``profile.ps1`` runs on every new session, the way ``~/.bashrc`` does (about_Profiles).
They are written under ``~`` like every other entry, which is platform-agnostic and harmless
where the directory does not exist."""

PROTECTED_PATHS_WORKSPACE_DEFAULT: tuple[str, ...] = (
    ".git", ".localharness", ".env", ".env.*", "*.pem", "*.key", "id_rsa*", "id_ed25519*",
)
"""PRD §3.1: names matched at any depth inside the workspace; a directory entry protects its
subtree (``.git/hooks/pre-commit`` is protected via ``.git``)."""

PROTECTED_PATHS_SYSTEM_DEFAULT: tuple[str, ...] = (
    "/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64", "/boot", "/var", "/opt", "/root", "/srv",
    "/System", "/Library", "/Applications",
    "C:/Windows", "C:/Program Files", "C:/Program Files (x86)", "C:/ProgramData",
)
"""The machine's own directories, ungrantable like the two sets above (owner ruling 2026-09-11:
"auto = thinnest interaction; asks only when genuinely dangerous").

``auto`` allows a write outside the project silently — that is most of what made ``guarded``
intrusive — and this set is what keeps "outside the project" from meaning ``/etc/hosts``,
``/usr/bin``, a systemd unit or a startup item. Without it the new default would have been a
strict loosening of the shipped one on exactly the paths a person cannot undo from the next
prompt. The three rows are the Linux/BSD system tree (FHS: ``/etc`` configuration, ``/usr`` and
``/bin``/``/sbin``/``/lib`` the installed system, ``/boot`` the kernel, ``/var`` service state,
``/opt`` and ``/srv`` add-on software and served data, ``/root`` the superuser's home), macOS's
three (``/System``, ``/Library``, ``/Applications``), and Windows's, where the owner also
dogfoods.

Matched AFTER realpath, against both the entry as written and its own realpath, so macOS's
``/etc`` → ``/private/etc`` and Linux's merged-usr ``/lib`` → ``/usr/lib`` are the same entry
rather than a hole. The Windows rows are matched case-insensitively on the posix spelling of the
resolved path, because Windows paths are case-insensitive and a drive-letter path is not
absolute on a POSIX host (see :func:`localharness.agent.verdict._system_root_matches`).

:data:`PROTECTED_PATHS_SYSTEM_EXEMPT_DEFAULT` carves the scratch directories back out."""

PROTECTED_PATHS_SYSTEM_EXEMPT_DEFAULT: tuple[str, ...] = ("/tmp", "/var/tmp")
"""Scratch directories that are NOT protected, even though one of them sits inside ``/var``.

``/tmp`` and ``/var/tmp`` are where every build, every test run and every ``mktemp`` writes
(FHS: both are for temporary files, and ``/var/tmp`` is the one that survives a reboot). Leaving
``/var/tmp`` inside the ``/var`` entry above would make an ordinary scratch write an ungrantable
prompt, which is the fatigue this release exists to remove. ``/tmp`` needs no carve-out on a
stock Linux — it is not under any entry above — and is listed anyway because on a host where it
is a symlink into one, the honest answer is still "scratch".

Deliberately NOT config-settable, in either layer: every other rule set here can be TIGHTENED
from a project layer, and this is the one list where adding an entry removes protection."""

TARGET_SCOPED_DESTRUCTIVE_VERBS: frozenset[str] = frozenset({
    "rm", "rmdir", "chmod", "chown", "chgrp", "truncate",
    "find",
    "Remove-Item", "ri", "del", "erase", "rd",
})
"""Destructive verbs whose danger is ENTIRELY where they point (owner ruling 2026-09-11).

``rm -rf build`` inside your own project is routine — it is what a build script does — and
``rm -rf ~/Documents`` is not. Nothing about the verb separates them; only the target does. So
``auto`` resolves these verbs' operands (:attr:`ShellSegment.destructive_targets`) and allows
the command when every one of them lands inside the workspace boundary and none is protected,
and asks otherwise: outside the boundary, protected, or unresolvable.

Every OTHER member of :data:`DESTRUCTIVE_SIGNATURES_DEFAULT` is irreversible regardless of
target and keeps asking in ``auto``: ``sudo``/``su``/``doas`` (the target is the whole machine),
pipe-to-shell (the target is code nobody has read), ``git push --force``/``--delete``,
``git reset --hard``, ``git clean -f``, ``git checkout --``/``git restore``, ``git stash
drop``/``clear``, ``git branch -D``, ``git filter-branch``, ``git reflog expire``,
``git gc --prune`` (each destroys work that is inside the project and still unrecoverable),
``dd``, ``mkfs``, ``format``/``diskpart`` (a device, not a path), and the docker verbs that run
or destroy containers.

``find`` is in the set for its ``-delete`` spelling only — a plain ``find`` is read-only and
never reaches the destructive branch. ``shred`` is in it despite overwriting irrecoverably: so
does ``rm``, and the same operand rule tells them apart. The Windows delete verbs are the
spellings of the same operation on the other shell (:data:`WINDOWS_DESTRUCTIVE_FLAG_VERBS`).

Deliberately NOT config-settable: adding a verb here LOOSENS ``auto``."""

DOTTED_VARIANT_SEPARATOR = "."
"""How a command family spells its variants: ``mkfs`` is a dispatcher and the thing anybody
actually runs is ``mkfs.ext4``, ``mkfs.xfs``, ``mkfs.vfat`` (mkfs(8): "mkfs.<fstype>").

The destructive sets name the FAMILY, and without this the entry matched only the bare
dispatcher — which nobody types — so every real invocation of the one command in the set that
makes a filesystem was classified as an unfamiliar, grantable command. Restricted to a dotted
suffix on an entry that is already a bare name, so it can never widen a flagged entry like
``rm -rf`` or a subcommand entry like ``git push --force``."""

AUTO_IRREVERSIBLE_SIGNATURES: frozenset[str] = frozenset({
    "sudo", "su", "doas",
    "dd", "mkfs", "shred", "format", "diskpart",
    "git push --force", "git push --delete", "git reset --hard", "git clean -f",
})
"""The commands ``auto`` asks about wherever they point (owner ruling 2026-09-11: "only hard
blacklists for git and rm and shit like that, and even then very minimal").

Matched on the canonical signature, so ``git push -f`` and ``git push --force-with-lease`` are
the one ``git push --force`` entry and ``git push origin :branch`` is the one
``git push --delete`` entry (the classifier canonicalizes both).

Why each is here and not merely "destructive": ``sudo``/``su``/``doas`` put the command outside
the boundary by definition — the target is the machine. ``dd``, ``mkfs``, ``format`` and
``diskpart`` write DEVICES, which no path check covers, and ``shred`` overwrites so that the
file is gone even from a backup of the block. The four git entries destroy work that no later
command can recover: a force-push and a delete-push rewrite what other people have already
pulled, ``git reset --hard`` and ``git clean -f`` throw away uncommitted work with no reflog
entry to walk back to.

What is deliberately NOT here, though the fuller
:data:`DESTRUCTIVE_SIGNATURES_DEFAULT` that ``guarded`` uses still carries it: every docker verb
(the shipped ``permissions.deny_patterns`` already hard-denies ``docker stop``/``kill``/``rm``/
``compose down``, which is a tier above asking), ``git branch -D``/``-d``, ``git stash
drop``/``clear``, ``git checkout --``/``git restore``, ``git reflog expire``,
``git gc --prune``, ``git filter-branch``, ``git worktree remove --force``,
``git submodule deinit --force``, ``git remote set-url``, and ``git config`` writes to a key git
later executes. Each of those is recoverable, rare, or already covered — and each of them
stopped the owner mid-task in the v0.14.0 dogfood. They keep their classification so ``guarded``
is unchanged; ``auto`` does not ask about them."""

AUTO_PROTECTED_PATHS_WORKSPACE: tuple[str, ...] = (".git", ".localharness")
"""The in-project protected names ``auto`` keeps, out of
:data:`PROTECTED_PATHS_WORKSPACE_DEFAULT` (owner ruling 2026-09-11).

``.git`` because a write under it re-points what the next ordinary git command executes
(``.git/hooks/*``, ``.git/config``), and ``.localharness`` because a write under it changes what
the harness itself does next. ``.env``/``.env.*``, ``*.pem``, ``*.key``, ``id_rsa*`` and
``id_ed25519*`` are dropped: writing your own project's ``.env`` is ordinary work, the real key
material lives under ``~/.ssh`` and the other home entries, and those stay protected in every
mode. ``guarded`` keeps the full set."""


@dataclass(frozen=True)
class AutoBlacklist:
    """The ONE curated structure that decides what ``auto`` still asks about.

    Owner ruling 2026-09-11: "default to auto mode so that it asks to trust the workspace, and
    allows anything except a dangerous blacklist that we can explore rather than building out a
    whitelist", refined to "only hard blacklists for git and rm and shit like that, and even then
    very minimal". Kept apart from the fuller rule sets ``guarded`` uses so the two can be curated
    independently: ``guarded``'s job is to ask about anything it has not seen, ``auto``'s is to
    ask about the handful of things that cannot be undone.

    ``localharness ask-rate --mode auto`` reports which of these entries actually fired over a
    corpus, which is how the list gets shorter over time rather than longer.

    Not in it, and therefore silent in ``auto``: every interpreter, ``python_exec``, ``agent``,
    every MCP and plugin tool, every network call, and every write to a path this does not name.
    Two tiers still run ahead of it and are unaffected: the config ``deny_patterns`` (a DENY, the
    owner's own list) and a recorded :class:`Refusal`.
    """

    target_scoped_verbs: frozenset[str] = TARGET_SCOPED_DESTRUCTIVE_VERBS
    """Asks only when a target is outside the boundary, protected, or unresolvable."""

    irreversible_signatures: frozenset[str] = AUTO_IRREVERSIBLE_SIGNATURES
    """Asks wherever it points."""

    protected_paths_workspace: tuple[str, ...] = AUTO_PROTECTED_PATHS_WORKSPACE
    """The in-project protected names; the home and system sets apply in full."""

    pipe_to_shell: bool = True
    """``curl … | sh`` and its PowerShell twin: the code being run has not been read by anyone,
    so there is no target to check and no signature that means anything
    (:data:`PIPE_TO_SHELL_SOURCES_DEFAULT`, :data:`PIPE_TO_SHELL_SINKS_DEFAULT`)."""


AUTO_BLACKLIST = AutoBlacklist()
"""The shipped blacklist. One value, so a change is one diff and one review."""


@dataclass(frozen=True)
class GateSettings:
    """Every tunable of the gate, with the PRD defaults. Built from ``permissions.ask`` (A3)."""

    read_only_signatures: frozenset[str] = READ_ONLY_SIGNATURES_DEFAULT
    destructive_signatures: frozenset[str] = DESTRUCTIVE_SIGNATURES_DEFAULT
    destructive_flag_verbs: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DESTRUCTIVE_FLAG_VERBS_DEFAULT)
    )
    pipe_to_shell_sources: frozenset[str] = PIPE_TO_SHELL_SOURCES_DEFAULT
    pipe_to_shell_sinks: frozenset[str] = PIPE_TO_SHELL_SINKS_DEFAULT
    interpreter_commands: frozenset[str] = INTERPRETER_COMMANDS_DEFAULT
    inline_code_flags: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(INLINE_CODE_FLAGS_DEFAULT)
    )
    inline_by_nature: frozenset[str] = INLINE_BY_NATURE_DEFAULT
    source_commands: frozenset[str] = SOURCE_COMMANDS_DEFAULT
    git_config_dangerous_keys: tuple[str, ...] = GIT_CONFIG_DANGEROUS_KEYS_DEFAULT
    wrapper_commands: frozenset[str] = WRAPPER_COMMANDS_DEFAULT
    dropped_commands: frozenset[str] = DROPPED_COMMANDS_DEFAULT
    subcommand_tools: frozenset[str] = SUBCOMMAND_TOOLS_DEFAULT
    payload_commands: frozenset[str] = PAYLOAD_COMMANDS_DEFAULT
    write_shaped_commands: frozenset[str] = WRITE_SHAPED_COMMANDS_DEFAULT
    protected_paths_home: tuple[str, ...] = PROTECTED_PATHS_HOME_DEFAULT
    protected_paths_workspace: tuple[str, ...] = PROTECTED_PATHS_WORKSPACE_DEFAULT
    protected_paths_system: tuple[str, ...] = PROTECTED_PATHS_SYSTEM_DEFAULT
    protected_paths_system_exempt: tuple[str, ...] = PROTECTED_PATHS_SYSTEM_EXEMPT_DEFAULT
    """Not reachable from ``AskConfig``: see the constant's docstring."""
    target_scoped_destructive_verbs: frozenset[str] = TARGET_SCOPED_DESTRUCTIVE_VERBS
    """Not reachable from ``AskConfig``: see the constant's docstring."""
    auto_blacklist: AutoBlacklist = AUTO_BLACKLIST
    """The ``auto`` blacklist. Not reachable from ``AskConfig``: every field of it either
    LOOSENS the mode when extended (the two signature sets) or is the one list that decides
    whether the default mode is safe at all — and config travels with a repo (PRD §3.3)."""
    mcp_trusted_servers: frozenset[str] = frozenset()
    """PRD §3.1: a whole MCP server whose tools skip the once-per-tool ask."""
    ask_network_hosts: bool = False
    """PRD §3.1 choice 1 / owner ruling §9.4: network reads are silent by default."""
    ask_timeout_s: float | None = None
    """PRD §3.5: None derives the wait from the tool timeout at the call site."""
