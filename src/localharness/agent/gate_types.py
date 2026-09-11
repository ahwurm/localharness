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
    "mcp",
    "network-host",
]

UNGRANTABLE_CLASSES: frozenset[str] = frozenset({"shell-destructive", "protected-path", "no-boundary"})
"""PRD §3.1 ungrantable tier: asks every time in guarded and trusted, DENY in unattended.

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


# --------------------------------------------------------------------------- settings

READ_ONLY_SIGNATURES_DEFAULT: frozenset[str] = frozenset({
    "ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "tree", "which", "env", "printenv",
    "stat", "file", "du", "df", "date", "uname", "pwd", "echo", "true", "false", "type",
    "git status", "git diff", "git log", "git show", "git branch", "git rev-parse", "git remote",
    "sed -n", "sort", "uniq", "cut", "tr", "basename", "dirname", "realpath", "readlink",
    "id", "whoami", "hostname", "nproc", "free", "uptime", "ps", "ss", "lsof",
})
"""PRD §3.1 ALLOW tier for shell. Union of Claude Code's built-in read-only Bash set
(code.claude.com/docs/en/permissions) and the read-only tokens observed in the 384-session
dogfood corpus (PRD §5). ``find`` is read-only only without ``-exec``/``-delete`` (the classifier
lifts those payloads, PRD §3.2 step 5)."""

DESTRUCTIVE_SIGNATURES_DEFAULT: frozenset[str] = frozenset({
    "rm -r", "rm -f", "rm -rf",
    "git push --force", "git reset --hard", "git clean -f",
    "chmod -R", "chown", "chgrp",
    "sudo", "su", "doas",
    "docker", "docker compose", "docker-compose",
    "dd", "mkfs", "shred", "truncate",
    "find -delete",
})
"""PRD §3.1 ungrantable ``shell-destructive`` class: matched on the canonical signature
(flags included, PRD §3.2 step 7), before any grant lookup. Sources: PRD §3.1 table; Claude Code
auto-mode block list (force push, ``git reset --hard``, recursive delete); the project's own
shipped deny defaults (sudo, ``rm -rf``, docker). Pipe-to-shell is a separate rule below."""

DESTRUCTIVE_FLAG_VERBS_DEFAULT: dict[str, tuple[str, ...]] = {
    "rm": ("r", "f"),
    "chmod": ("R",),
    "git push": ("force",),
    "git reset": ("hard",),
    "git clean": ("f",),
}
"""Verbs whose destructive variant is a flag. The classifier canonicalizes only these flags
into the signature (``rm -r -f x`` → ``rm -rf``; ``git push -f`` → ``git push --force``) so a grant on
``rm`` never covers ``rm -rf`` (critic finding 12). Long forms map: ``--recursive``→r,
``--force``→f/force, ``--recursive`` for chmod→R."""

PIPE_TO_SHELL_SOURCES_DEFAULT: frozenset[str] = frozenset({"curl", "wget"})
PIPE_TO_SHELL_SINKS_DEFAULT: frozenset[str] = frozenset({"sh", "bash", "zsh", "python", "python3"})
"""``curl … | sh`` and friends: the sink segment is destructive (PRD §3.1; Claude Code block list)."""

INTERPRETER_COMMANDS_DEFAULT: frozenset[str] = frozenset({
    "python", "python3", "bash", "sh", "zsh", "node", "uv run", "ruby", "perl",
})
INLINE_CODE_FLAGS_DEFAULT: dict[str, tuple[str, ...]] = {
    "python": ("-c",), "python3": ("-c",), "uv run": ("-c",),
    "bash": ("-c",), "sh": ("-c",), "zsh": ("-c",),
    "node": ("-e", "--eval", "-p", "--print"),
    "ruby": ("-e",), "perl": ("-e", "-E"),
}
INLINE_BY_NATURE_DEFAULT: frozenset[str] = frozenset({"eval"})
"""PRD §3.1 ``interpreter-inline`` class (grantable per owner ruling 2026-09-11 §9.3). The
signature carries the mode so ``python3 -c``, ``python3 -m MOD``, ``python3 <script>`` and bare
``python3`` are four different grant keys (critic finding 5)."""

WRAPPER_COMMANDS_DEFAULT: frozenset[str] = frozenset({
    "env", "nohup", "time", "nice", "ionice", "command", "builtin", "exec", "timeout", "stdbuf",
})
"""PRD §3.2 step 4: peeled so the signature is the wrapped command's. ``sudo`` is NOT a wrapper
(destructive before peeling)."""

DROPPED_COMMANDS_DEFAULT: frozenset[str] = frozenset({"cd", "export", "pwd", "true", ":"})
"""PRD §3.2 step 6: dropped only when the segment is exactly this command with no substitution
inside. ``cd`` was the most frequent first token in the corpus, always as ``cd X && …``."""

SUBCOMMAND_TOOLS_DEFAULT: frozenset[str] = frozenset({
    "git", "uv", "pip", "pip3", "npm", "npx", "pnpm", "yarn", "docker", "cargo", "make", "gh", "kubectl",
})
"""PRD §3.2 step 7: the signature includes the first subcommand (``git status`` ≠ ``git push``)."""

PAYLOAD_COMMANDS_DEFAULT: frozenset[str] = frozenset({"find", "xargs"})
"""PRD §3.2 step 5: ``find -exec/-execdir/-ok CMD`` and ``xargs CMD`` lift CMD into its own
segment (critic finding 4)."""

WRITE_SHAPED_COMMANDS_DEFAULT: frozenset[str] = frozenset({
    "tee", "cp", "mv", "install", "rsync", "dd", "curl", "wget", "ln", "touch", "mkdir", "unzip", "tar",
})
"""PRD §3.2 step 8 (critic finding 2): commands whose arguments name a write target, checked
against the boundary and the protected set on every call. Redirections ``>``/``>>`` apply to any
segment."""

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
