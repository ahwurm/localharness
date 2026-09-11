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

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Literal

# --------------------------------------------------------------------------- modes

Mode = Literal["guarded", "trusted", "read-only", "unattended"]

MODE_STRICTNESS: dict[str, int] = {"unattended": 0, "trusted": 1, "guarded": 2, "read-only": 3}
"""Strictness order used by the loader's narrow-only union (PRD §3.3, §3.4).

A project layer may only raise strictness; ``unattended`` is never a default and is never
settable from a channel command.
"""

DEFAULT_MODE: Mode = "guarded"
"""PRD §3.4: the default for every channel that can ask."""


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

DESTRUCTIVE_SIGNATURES_DEFAULT: frozenset[str] = frozenset({
    "rm -r", "rm -f", "rm -rf",
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

Residual, named rather than hidden: docker's management-command spellings of the same operations
(``docker container rm``, ``docker container exec/run``, ``docker image rm``) are NOT in this set,
so they classify as unfamiliar and can be granted. They are aliases for entries that are here."""

DESTRUCTIVE_FLAG_VERBS_DEFAULT: dict[str, tuple[str, ...]] = {
    "rm": ("r", "f"),
    "chmod": ("R",),
    "git push": ("force",),
    "git reset": ("hard",),
    "git clean": ("f",),
    "git worktree remove": ("force",),
    "git submodule deinit": ("force",),
}
"""Verbs whose destructive variant is a flag. The classifier canonicalizes only these flags
into the signature (``rm -r -f x`` → ``rm -rf``; ``git push -f`` → ``git push --force``) so a grant on
``rm`` never covers ``rm -rf`` (critic finding 12). Long forms map: ``--recursive``→r,
``--force``→f/force, ``--recursive`` for chmod→R.

The two three-word git keys are the operations that are only destructive when forced:
``git worktree remove`` and ``git submodule deinit`` both refuse to throw away uncommitted work
until ``--force`` is passed, so the unforced spelling stays grantable and the forced one is in the
ungrantable set above (git-worktree(1), git-submodule(1))."""

PIPE_TO_SHELL_SOURCES_DEFAULT: frozenset[str] = frozenset({"curl", "wget"})
"""The fetch side of the pipe-to-shell rule: a segment with one of these signatures piped into a
sink below makes the SINK destructive (PRD §3.1; Claude Code block list). Two entries because
these are the two fetchers the dogfood corpus (PRD §5) actually used; the rule is overridable
from ``permissions.ask.pipe_to_shell_sources`` for anyone whose install ships another."""

PIPE_TO_SHELL_SINKS_DEFAULT: frozenset[str] = frozenset({"sh", "bash", "zsh", "python", "python3"})
"""The execute side of the same rule: ``curl … | sh`` and friends, where the sink segment is
destructive (PRD §3.1; Claude Code block list). Read with the sources above — neither half means
anything alone, and a bare ``sh`` is an interpreter, not a destructive command."""

INTERPRETER_COMMANDS_DEFAULT: frozenset[str] = frozenset({
    "python", "python3", "bash", "sh", "zsh", "node", "uv run", "ruby", "perl",
    "fish", "ksh", "dash", "script",
})
"""Commands whose SIGNATURE carries an interpreter mode (PRD §3.2 step 7, critic finding 5).

``python3 -c``, ``python3 -m MOD``, ``python3 <script>`` and bare ``python3`` are four different
grant keys, so answering "always" about an inline one-liner is never an answer about a script.
The list is the interpreters present in the dogfood corpus (PRD §5), where ``python3`` is the
second signature nearly every workspace is asked about.

``fish``, ``ksh`` and ``dash`` are here because a shell the list forgets is a shell the gate
never opens — their ``-c`` payload is recursed into like ``bash -c``'s (review finding R9).
``script`` is in the same row for the same reason: ``script -c CMD FILE`` runs CMD through a
shell to record the session (script(1))."""

INLINE_CODE_FLAGS_DEFAULT: dict[str, tuple[str, ...]] = {
    "python": ("-c",), "python3": ("-c",), "uv run": ("-c",),
    "bash": ("-c",), "sh": ("-c",), "zsh": ("-c",),
    "fish": ("-c",), "ksh": ("-c",), "dash": ("-c",), "script": ("-c",),
    "node": ("-e", "--eval", "-p", "--print"),
    "ruby": ("-e",), "perl": ("-e", "-E"),
}
"""Per-interpreter flags that mean "the code is on the command line" (PRD §3.2 step 7).

The table that turns a command into the ``interpreter-inline`` class and puts the mode in the
grant key. Each row is that interpreter's own documented inline flags — ``node`` carries four
because ``-p``/``--print`` evaluate too. Deliberately NOT overridable from config (see
``AskConfig``): it is a canonicalization table, and a wrong row silently changes what a grant
key means."""

INLINE_BY_NATURE_DEFAULT: frozenset[str] = frozenset({"eval"})
"""Commands that are inline interpreters with no flag at all (PRD §3.1 ``interpreter-inline``,
grantable per owner ruling 2026-09-11 §9.3). ``eval`` takes its code as plain arguments, so
there is no mode flag to key on — the command itself is the signature."""

WRAPPER_COMMANDS_DEFAULT: frozenset[str] = frozenset({
    "env", "nohup", "time", "nice", "ionice", "command", "builtin", "exec", "timeout", "stdbuf",
    "watch", "flock", "setsid", "chroot", "busybox", "caffeinate", "unbuffer", "systemd-run",
    "poetry", "pipx", "uvx",
})
"""PRD §3.2 step 4: peeled so the signature is the wrapped command's. ``sudo`` is NOT a wrapper
(destructive before peeling).

The second row is the review's finding R9: every one of these runs its argument as a command, so
before they were peeled ``watch cp x ~/.ssh/authorized_keys`` signed as an unfamiliar ``watch``
with no write target at all. Their per-command argument shapes (``watch -n N``, ``flock`` and
``chroot``'s leading operand, ``poetry run`` / ``pipx run``'s required subcommand) live with the
peeling logic in ``shell_classify``, which is where the other canonicalization tables are."""

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
    "tee", "cp", "mv", "install", "rsync", "dd", "curl", "wget", "ln", "touch", "mkdir", "unzip", "tar",
})
"""PRD §3.2 step 8 (critic finding 2): commands whose arguments name a write target, checked
against the boundary and the protected set on every call. Redirections ``>``/``>>`` apply to any
segment."""

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
)
"""PRD §3.1 ``protected-path`` (ungrantable). ``~/.localharness`` is protected because writing it
changes what the harness does next; the harness's own runtime store under it is exempted by the
verdict (A2). Source: Claude Code "protected paths" tier + PRD critic finding 1."""

PROTECTED_PATHS_WORKSPACE_DEFAULT: tuple[str, ...] = (
    ".git", ".localharness", ".env", ".env.*", "*.pem", "*.key", "id_rsa*", "id_ed25519*",
)
"""PRD §3.1: names matched at any depth inside the workspace; a directory entry protects its
subtree (``.git/hooks/pre-commit`` is protected via ``.git``)."""


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
    mcp_trusted_servers: frozenset[str] = frozenset()
    """PRD §3.1: a whole MCP server whose tools skip the once-per-tool ask."""
    ask_network_hosts: bool = False
    """PRD §3.1 choice 1 / owner ruling §9.4: network reads are silent by default."""
    ask_timeout_s: float | None = None
    """PRD §3.5: None derives the wait from the tool timeout at the call site."""
