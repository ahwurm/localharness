"""The pure permission verdict: DENY → ungrantable ASK → grant → grantable ASK → ALLOW.

Implements PRD §3.1 (``.planning/2026-09-11-zed-acp-and-permission-spine-prd.md``) with the
mode effects of §3.4 and the shell classification of §3.2 (delegated to
``agent/shell_classify``). Nothing here prompts, writes, or awaits: :func:`evaluate` maps a
tool call onto a verdict and, when the verdict is ASK, the :class:`PermissionRequest` a channel
renders. The effectful half — asking a human, writing the grant, publishing the bus events —
is ``agent/gate.PermissionGate`` (A4).

Three properties this file is responsible for, each closing a critic finding from PRD §11:

* **Boundary first, prompt second.** The boundary is DERIVED from where you stand
  (:func:`derive_boundary`), never read from config; config may only narrow it
  (:func:`narrow_boundary`). A repo cannot move it (finding 6).
* **$HOME collapses.** When the derived root is ``$HOME`` or an ancestor of it there is no
  boundary at all, and every write-shaped call asks, ungrantably (finding 1).
* **Ungrantable before grants.** The destructive / protected / no-boundary checks run BEFORE
  any grant lookup, so an old benign grant can never cover a destructive variant (finding 12),
  and write-shaped shell targets are re-checked on every call (finding 2).

``read-only`` mode is decided ONCE, before the branches (:func:`_read_only_denies`), over the
kinds PRD §3.4 enumerates (write-shaped, non-read-only shell, code, delegate) plus anything a
tool schema marks ``destructive``. Known gap, named rather than hidden: a tool that mutates
without setting that flag — an MCP tool, a plugin's — is gated only by its once-per-tool ask,
not by read-only mode. The flag is the only thing a tool says about itself, so a tool that
misdescribes itself is believed.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Callable, Optional
from urllib.parse import urlparse

from localharness.agent.gate_types import (
    AUTO_ASK_CLASSES,
    DEFAULT_MODE,
    UNGRANTABLE_CLASSES,
    GateSettings,
    Grant,
    Mode,
    PermissionRequest,
    Refusal,
    ToolMeta,
    Verdict,
    VerdictResult,
)
from localharness.agent.permissions import PermissionResult
from localharness.config.paths import ARCHIVE_DB_NAME, global_config_dir

GrantLookup = Callable[[Path, str, str], Optional[Grant]]
"""``(workspace, klass, key) -> Grant | None`` — ``config.grants.GrantStore.lookup`` in
production.

The CLASS is part of the question, not decoration. Every ask class has its own key space and the
same string lives in several of them: ``python_exec`` is a shell signature AND the ``code-exec``
tool name; a shell signature can look exactly like a directory. Asking with the key alone let an
"always" on the shell command ``python_exec`` satisfy the ``python_exec`` TOOL, and let a shell
signature that happened to read as a path pre-approve a whole directory subtree through
:func:`_granted_directory`'s ancestor walk. A grant answers one question; it is consulted for
that question only."""

RefusalLookup = Callable[[Path, str, str], Optional[Refusal]]
"""``(workspace, klass, key) -> Refusal | None`` — ``config.grants.GrantStore.refused`` in
production.

A refusal is a grant's negative twin in the same key space (PRD §3.3), so "never here" is
consulted with the same call shape as "always here" and denies exactly the (class, key) it
names."""

REFUSAL_DENY_REASON = "refused by a human in this workspace (never here)"
"""PRD §3.3: what the model and the person watching are told when a stored "never" decides the
call. A refusal denies without prompting — that is the "asks no more" half of the answer."""

PATH_KEYED_CLASSES: frozenset[str] = frozenset(
    {"edit-outside", "edit-unreviewed", "protected-path", "no-boundary"}
)
"""Ask classes whose grant key is a filesystem path (PRD §3.1 table). A refusal on one of these
covers the subtree beneath it, exactly as :func:`_granted_directory` makes a directory grant
cover its subtree — "never write there again" means the place, not one path segment."""

DenyFn = Callable[[str, dict], PermissionResult]
"""``(tool_name, tool_params) -> PermissionResult`` — wraps the existing
``agent/permissions.PermissionEvaluator.evaluate`` (the DENY tier, unchanged)."""

AUTO_PIPE_TO_SHELL_ENTRY = "pipe-to-shell"
"""The name :func:`_auto_blacklisted` reports for a ``curl … | sh``.

The sink's signature is a bare ``sh`` or ``python3``, which is not the reason the call is on the
blacklist, so the report would have been unreadable ("sh: 4 prompts") without a name for the
RULE. Every other entry reports as itself, because every other entry is a signature."""

MODES_WITHOUT_GRANTS: frozenset[str] = frozenset({"auto"})
"""Modes in which the grant store is never READ and never WRITTEN (owner ruling 2026-09-11:
``auto`` is "blacklist-only… allows anything except a dangerous blacklist").

A grant is a memory of an answer to a question. ``auto`` does not ask those questions — a first
-exposure command, a write outside the project, an unfamiliar tool all simply run — so there is
nothing to remember and nothing to consult, and consulting the store anyway would make the mode
depend on state a person cannot see. The one thing that IS consulted is the negative twin: a
recorded :class:`Refusal` still denies, in every mode, because "never here" is a DENY and DENY
outranks modes (PRD §3.3, §3.4)."""


# --------------------------------------------------------------- tool vocabulary

WRITE_TOOL_PATH_PARAMS: dict[str, str] = {"write": "path", "edit": "path"}
"""The builtin ``fs.write`` tools and the parameter naming their target, read off the schemas:
``tools/builtin/write_tool.py:44`` and ``edit_tool.py:48``. Not guessed — the exact keys."""

WRITE_PATH_PARAM_CANDIDATES: tuple[str, ...] = ("path", "file_path")
"""Fallback order for a NON-builtin tool that declares ``group="fs.write"`` (A3's taxonomy):
``path`` is this project's convention, ``file_path`` the common one elsewhere. A plugin whose
write tool names its target anything else is classified by its group but has no resolvable
target, which lands it in the unresolvable branch — treated as outside (PRD §3.2 step 8)."""

SHELL_COMMAND_PARAMS: dict[str, str] = {"bash_exec": "command"}
"""``tools/builtin/bash_tool.py:218`` — the one opaque string PRD §3.2 parses."""

SHELL_WORKING_DIR_PARAMS: dict[str, str] = {"bash_exec": "working_dir"}
"""The parameter naming the directory a shell call actually RUNS IN
(``tools/builtin/bash_tool.py:229``, default ``"."``).

PRD §3.2 step 8 resolves every write target, and a relative target resolves against the process's
cwd — which for ``bash_exec`` is this parameter, not the workspace. Anchoring relative targets at
the workspace regardless made ``bash_exec(command="echo k >> authorized_keys",
working_dir="~/.ssh")`` read as an in-workspace write and ALLOW: the boundary check was answering
a question about a file that was never going to be written."""

NETWORK_URL_PARAMS: dict[str, str] = {"web_fetch": "url"}
"""``tools/builtin/web_tool.py:164``. ``web_search`` (a query) and ``web_page_query`` (a query
over an already-fetched page held in this process) name no host, so a per-host ask cannot apply
to them."""

CODE_EXEC_TOOLS: frozenset[str] = frozenset({"python_exec", "cruncher_exec"})
"""PRD §3.1 ``code-exec`` class; the two builtin interpreters
(``tools/builtin/python_tool.py:31``, ``cruncher_exec.py:115``). Grant key = the tool name."""

DELEGATE_TOOLS: frozenset[str] = frozenset({"agent"})
"""PRD §3.1 ``delegate`` class (``tools/builtin/agent_tool.py:36``). A subagent's own calls pass
this same gate, so the dispatch asks once and the child's boundary crossings still surface."""

KIND_BY_GROUP: dict[str, str] = {
    "fs.write": "write",
    "shell": "shell",
    "code": "code",
    "delegate": "delegate",
    "web": "network",
    "fs.read": "allow",
    "memory": "allow",
}
"""Fallback classification by ``ToolSchema.group`` (A3, PRD §6) for tools this module does not
know by name — a plugin's or a future builtin's. Names win when known, because a name pins the
exact parameter to read; groups are the open extension point.

The map is CLOSED: the read tiers (``fs.read``, ``memory``) are listed explicitly, so a group
that is not here — ``other``, the ``ToolSchema.group`` default, or anything a plugin invents — is
a family the gate has no rules for and lands in :data:`UNFAMILIAR_TOOL_KIND` rather than in ALLOW.
The old ``.get(group, "allow")`` default meant every plugin tool, and every tool whose schema
lookup failed, ran unasked even with ``destructive=True`` on its schema."""

MCP_GROUP_PREFIX = "mcp/"
"""``tools/mcp.py:81`` names every MCP tool's group ``mcp/<server>``. A call that carries one is
judged on the MCP path even if ``ToolMeta.is_mcp`` did not survive the trip."""

UNFAMILIAR_TOOL_KIND = "tool-unfamiliar"
"""The branch for a tool in no known family: a grantable ask keyed on the tool NAME, so it is
answered once per workspace and then remembered — the same shape as ``code-exec`` and
``delegate``, which are also keyed by name."""

UNFAMILIAR_TOOL_REASON = "tool not in a known family; asks once per workspace"
"""What the human reads. It says what the gate knows (nothing about this tool) rather than
naming a rule, because the answer it wants is "do you want this tool to run here at all"."""

DYNAMIC_COMMAND_NAME_PREFIXES: tuple[str, ...] = ("$", "`")
"""A shell segment whose command NAME is itself a substitution or a variable — ``$(echo rm) -rf
build``, ``"$RM" -rf build``. The classifier lifts the substitution into its own segment but
cannot say what the outer segment will actually RUN, so its signature is not a stable identity.

PRD §3.2 "named residual gaps": a grant is a memory of a decision, and there is nothing here to
remember — one "always" on ``$RM`` would cover every future dynamically-built command. So the
call asks every time, ungrantably (``grantable=False``, ``key=None``), and the grant store is
never consulted for it. Found by replaying the real corpus through the shipped rules."""

DYNAMIC_COMMAND_NAME_REASON = "command name is computed at runtime; cannot be remembered"
"""The reason string for :data:`DYNAMIC_COMMAND_NAME_PREFIXES` — it is what the human reads in
the prompt, so it says why there is no "always" option rather than naming a rule."""

UNRESOLVABLE_TARGET_CHARS: tuple[str, ...] = ("$", "*", "?", "`")
"""PRD §3.2 step 8: a write target carrying a variable, a glob or a substitution cannot be
resolved at classification time, and "unresolvable targets are treated as outside"."""

HARNESS_RUNTIME_STORE_NAMES: tuple[str, ...] = (
    ARCHIVE_DB_NAME,
    "KILL",
    "audit.jsonl",
    "speed_stats.json",
    ".repl_history",
)
"""PRD §3.1: ``~/.localharness`` is a protected path "except the harness's own runtime store".

These are the entries under the GLOBAL config dir the harness WRITES while it runs, rather than
the config it READS to decide what to do next — so touching one cannot change the harness's
behavior and gating them would only add prompts to the harness's own bookkeeping. Sources, in
order: ``config/paths.ARCHIVE_DB_NAME``; ``config/models.py:166`` (``kill_file`` default
``KILL``); ``config/models.py:1719`` (``audit_log_path`` default ``audit.jsonl``);
``provider/speed_stats.py:44``; ``cli/start_cmd.py:1305`` (the REPL history file). Everything
else under the config dir — ``config.yaml``, ``overrides.yaml``, ``agents/``, ``plugins/``,
``trusted_workspaces.yaml``, ``grants.yaml`` — is behavior-changing and stays protected.
A config-relocated kill file or audit log (an absolute path elsewhere) is outside this dir and
is judged by the ordinary boundary rules."""

READ_ONLY_DENIED_KINDS: frozenset[str] = frozenset({"write", "code", "delegate"})
"""PRD §3.4's read-only list, as branches of :func:`_kind`: write/edit (and anything in the
``fs.write`` group), ``python_exec``/``cruncher_exec`` (and the ``code`` group), and subagent
dispatch (the ``delegate`` group). Shell is judged per segment, not by kind
(:func:`_read_only_denies`), because most shell calls are reads."""

READ_ONLY_DENY_REASON = "not permitted in read-only mode"
"""PRD §3.4: the soft-deny text. It is returned to the model AS the tool observation, so the
model can re-plan rather than retry — hence a sentence, not an error code."""

MISSING_TARGET_DISPLAY = "<no path argument>"
"""A write-shaped call that names no target at all. It cannot be resolved, so PRD §3.2 step 8's
"unresolvable targets are treated as outside" applies — the call asks rather than passing."""

NON_STRING_SHELL_COMMAND_REASON = "shell command is not a string; cannot classify"
"""A ``command`` argument that is PRESENT but not a string — a list, a dict, a number.

A model that emits ``{"command": ["rm", "-rf", "/"]}`` used to collapse into the same branch as
"no command at all" and ALLOW, because both were tested with one ``isinstance(...) or not
strip()`` guard. Absent is a call that does nothing; present-and-unreadable is a call whose
content the classifier cannot see, and PRD §3.2 step 8's rule for anything it cannot see is
"treat it as outside" — so it asks, UNGRANTABLY (``grantable=False``, ``key=None``): there is
no stable identity here to remember, exactly as for a runtime-computed command name
(:data:`DYNAMIC_COMMAND_NAME_PREFIXES`).

Under ``unattended`` this ask becomes an ALLOW like every other ask (PRD §3.4) — the mode is
"nobody is watching, run it", and carving out an exception here would make bench and cron
behave differently from every other ungrantable class. Under ``read-only`` it DENIES: a command
that cannot be read cannot be shown to be a read."""

NON_STRING_URL_REASON = "url argument {param} is not a string; cannot name a host"
"""The third instance of the same split, in the network branch. It only binds when
``permissions.ask.network_hosts`` is on — and that is exactly when it matters, because that knob
exists to raise a per-host ask, and an unreadable url used to skip it by taking the "names no
host" exit. Ungrantable: there is no host to key a grant on."""

NON_STRING_PATH_REASON = "path argument {param} is not a string; cannot classify"
"""The same shape for a write-shaped call whose path parameter is present but not a string.

Filed under ``edit-outside`` because that is what PRD §3.2 step 8 already says about a target
that cannot be resolved — it is treated as outside the boundary — but with ``grantable=False``
and no key: an unreadable argument is not an identity a human can answer "always" about. The
old path let it through :func:`_write_target`'s ``isinstance`` filter as "no target named" and
then asked GRANTABLY on the literal key :data:`MISSING_TARGET_DISPLAY`, so one "always here"
pre-approved every future write whose target the gate could not read."""

ASK_SEVERITY_ORDER: tuple[str, ...] = (
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
)
"""PRD §3.1's two ASK tables read top to bottom: the ungrantable tier first, then the grantable
one, each in the order the PRD lists it.

One call can raise several asks at once (``mkdir -p /tmp/x/y && touch /tmp/x/y/f`` raises two
first-exposure commands and two outside-the-boundary directories). They are asked TOGETHER, in
one request, and this order decides which one is the request's primary ``klass``/``key`` — the
most severe thing the call does is what the question is named after."""

GROUPED_REASON_BY_CLASS: dict[str, str] = {
    "shell-destructive": "destructive shell commands: {details}",
    "protected-path": "writes protected paths: {details}",
    "no-boundary": "no workspace boundary here, so every write asks: {details}",
    "edit-outside": "writes outside the workspace boundary: {details}",
    "shell-unfamiliar": "commands not seen in this workspace before: {details}",
    "interpreter-inline": "runs code inline through interpreters: {details}",
}
"""How several asks of the SAME class read on one line (PRD §3.5: the request renders as one
line). Used only when a call raises more than one ask of a class — a single ask keeps its own
sentence — so the human reads "commands not seen in this workspace before: mkdir, touch"
instead of the same sentence twice. Any class not listed falls back to ``<class>: <details>``."""

GROUPED_REASON_FALLBACK = "{klass}: {details}"
"""The shape for a class with no entry in :data:`GROUPED_REASON_BY_CLASS`; every class that can
plausibly repeat within one call has one, so this is the honest default rather than a stub."""

DISPLAY_ARG_MAX_CHARS = 80
"""PRD §3.5: the request renders as ONE line in a prompt_toolkit prompt, a Discord message and
a Zed dialog. 80 is the conventional terminal width, and the full arguments travel on
``PermissionRequest.tool_params`` for any channel that wants to expand them."""

_ELLIPSIS = "…"


# --------------------------------------------------------------------- boundary

def derive_boundary(
    cwd: Path, local_dir: Optional[Path], git_toplevel: Optional[Path], home: Path
) -> Optional[Path]:
    """The workspace boundary, derived from where you stand (PRD §3.1).

    Project root = the directory holding the nearest in-project ``.localharness/``
    (``local_dir.parent``), else the git toplevel, else the CWD; resolved to a realpath so a
    symlinked checkout and its real path are one boundary.

    Returns **None** when the root is ``$HOME`` or any ancestor of it (``/`` included). That is
    critic finding 1: a boundary that contains your whole home directory is not a boundary, and
    the honest answer is "there is none" — every write-shaped call then asks, ungrantably, and
    the ACP adapter tells the user to open a project folder.
    """
    root = local_dir.parent if local_dir is not None else (git_toplevel if git_toplevel is not None else cwd)
    root = Path(root).expanduser().resolve()
    home = Path(home).expanduser().resolve()
    if root == home or root in home.parents:
        return None
    return root


def narrow_boundary(
    boundary: Optional[Path], configured_root: Optional[str]
) -> tuple[Optional[Path], Optional[str]]:
    """Apply ``permissions.workspace_root`` as a NARROWING of the derived boundary (PRD §3.1).

    Returns ``(effective_boundary, warning)``. A configured root inside the derived boundary
    narrows it. A root outside it — or any root at all when there is no derived boundary — is
    ignored with a warning: config may only tighten, never move or invent the boundary
    (critic finding 6), and inventing one where ``$HOME`` collapsed would turn ungrantable
    no-boundary asks into silent allows.
    """
    if configured_root is None:
        return boundary, None
    configured = Path(configured_root).expanduser().resolve()
    if boundary is None:
        return None, (
            f"permissions.workspace_root={configured_root!r} ignored: there is no workspace "
            "boundary here (the project root is your home directory or above)"
        )
    if configured == boundary or boundary in configured.parents:
        return configured, None
    return boundary, (
        f"permissions.workspace_root={configured_root!r} ignored: it is outside the derived "
        f"workspace boundary {boundary}"
    )


# ---------------------------------------------------------------------- context

@dataclass(frozen=True)
class GateContext:
    """Everything the verdict needs about the session, gathered once by A4's gate (PRD §3.1).

    ``can_ask`` is carried for the channel renderers and for A4's fail-closed path (no asker →
    DENY); :func:`evaluate` itself never reads it, because "can this channel ask" is a property
    of the channel, not of the call.
    """

    boundary: Optional[Path]
    workspace: Path
    grants: GrantLookup
    refusals: Optional[RefusalLookup] = None
    """"Never here" answers (PRD §3.3), consulted ahead of ``grants``. None means none are
    recorded — a caller that cannot read them simply asks, it never silently allows."""

    mode: Mode = DEFAULT_MODE
    can_ask: bool = True
    has_review_surface: bool = False
    deny: Optional[DenyFn] = None


# ------------------------------------------------------------------- internals

def _kind(tool_name: str, meta: ToolMeta) -> str:
    """Which branch of PRD §3.1's tables this call belongs to."""
    if meta.is_mcp:
        return "mcp"
    if tool_name in WRITE_TOOL_PATH_PARAMS:
        return "write"
    if tool_name in SHELL_COMMAND_PARAMS:
        return "shell"
    if tool_name in CODE_EXEC_TOOLS:
        return "code"
    if tool_name in DELEGATE_TOOLS:
        return "delegate"
    if tool_name in NETWORK_URL_PARAMS:
        return "network"
    group = meta.group or ""
    if group.startswith(MCP_GROUP_PREFIX):
        return "mcp"
    return KIND_BY_GROUP.get(group, UNFAMILIAR_TOOL_KIND)


def _write_target(tool_name: str, params: dict) -> Optional[str]:
    """The raw target string of a write-shaped tool call, or None when it names none."""
    param = WRITE_TOOL_PATH_PARAMS.get(tool_name)
    names = (param,) if param else WRITE_PATH_PARAM_CANDIDATES
    for name in names:
        value = params.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _bad_string_param(params: dict, names: tuple[str, ...]) -> Optional[str]:
    """The first of ``names`` that is PRESENT in ``params`` but is not a string.

    The distinction the guards below rest on: a parameter that is MISSING says the call names
    nothing, a parameter that is present and unreadable says the call names something the gate
    cannot see. The two used to share one ``isinstance`` test, and the unreadable case inherited
    the harmless answer (:data:`NON_STRING_SHELL_COMMAND_REASON`).

    An explicit ``None`` counts as MISSING, not as unreadable: JSON's null is how a model spells
    "no argument", and the tool it is handed to will reject it for the same reason the gate
    would have — there is nothing there to run.
    """
    for name in names:
        value = params.get(name)
        if name in params and value is not None and not isinstance(value, str):
            return name
    return None


def _resolve(target: str, anchor: Optional[Path]) -> Optional[Path]:
    """Realpath a write target, or None when it cannot be resolved (PRD §3.2 step 8).

    ``anchor`` is the directory a RELATIVE target resolves against — the directory the call will
    actually run in (:func:`_shell_anchor` for shell, the workspace otherwise). ``None`` means
    even that is unknown, so a relative target cannot be placed and is treated as outside.

    ``expanduser`` runs BEFORE the absoluteness test, so a target the classifier already joined
    onto a ``cd`` (``~/.ssh/authorized_keys``) is recognised as absolute and never gets the anchor
    prepended. A target holding a variable, glob or substitution resolves to None — outside.
    """
    if any(ch in target for ch in UNRESOLVABLE_TARGET_CHARS):
        return None
    try:
        path = Path(target).expanduser()
        if not path.is_absolute():
            if anchor is None:
                return None
            path = Path(anchor) / path
        return path.resolve()
    except (OSError, ValueError):
        return None


def _shell_anchor(tool_name: str, params: dict, ctx: GateContext) -> Optional[Path]:
    """Where this shell call's relative write targets land (:data:`SHELL_WORKING_DIR_PARAMS`).

    The tool's own working-directory argument wins; it is expanded and realpathed exactly as the
    tool does it (``tools/builtin/paths.resolve_user_path``), so ``~/.ssh`` anchors relative
    targets in ``~/.ssh`` — protected — and ``/tmp/x`` anchors them outside the boundary. With no
    argument (or the default ``"."``) the anchor is the workspace, which is where the tool
    anchors a relative ``working_dir`` when it is confined (``bash_tool.py:252``).

    Returns None when the working directory itself is unresolvable (a variable or a glob): its
    relative targets then resolve to None and are treated as outside, per PRD §3.2 step 8. A
    ``working_dir`` that is PRESENT but not a string is unresolvable for the same reason and
    gets the same answer — anchoring its relative targets at the workspace instead would have
    let a call the gate cannot place read as an in-workspace write.
    """
    name = SHELL_WORKING_DIR_PARAMS.get(tool_name, "")
    if _bad_string_param(params, (name,)) is not None:
        return None
    raw = params.get(name)
    if not isinstance(raw, str) or not raw.strip():
        return Path(ctx.workspace)
    return _resolve(raw, Path(ctx.workspace))


def _within(base: Optional[Path], path: Path) -> bool:
    """Is ``path`` at or below ``base``? (Both already realpaths.)"""
    if base is None:
        return False
    return path == base or base in path.parents


def _exempt_runtime_store(path: Path) -> bool:
    """Is this one of the harness's own runtime files under the global config dir?

    PRD §3.1's "except the harness's own runtime store" exemption; see
    :data:`HARNESS_RUNTIME_STORE_NAMES` for what counts and why.
    """
    try:
        config_dir = global_config_dir().resolve()
    except (OSError, ValueError):
        return False
    if not _within(config_dir, path) or path == config_dir:
        return False
    return path.relative_to(config_dir).parts[0] in HARNESS_RUNTIME_STORE_NAMES


def _system_root_matches(path: Path, raw: str) -> bool:
    """Is ``path`` at or below the system directory ``raw``? (:data:`PROTECTED_PATHS_SYSTEM_DEFAULT`.)

    Two spellings, because the set covers three operating systems:

    * A drive-letter entry (``C:/Windows``) is not an absolute path on a POSIX host — ``Path``
      would anchor it at the CWD — and Windows compares paths case-insensitively. So it is
      matched as a case-insensitive prefix of the resolved path's posix spelling, which is what
      a real Windows realpath renders as (``C:/Windows/System32``).
    * A POSIX entry is matched both as written AND as its own realpath, so macOS's
      ``/etc`` → ``/private/etc`` and Linux's merged-usr ``/lib`` → ``/usr/lib`` are one entry
      rather than a hole: the target is realpathed before it gets here, so an entry that is a
      symlink would otherwise never match anything.
    """
    if PureWindowsPath(raw).drive:
        base = PureWindowsPath(raw).as_posix().rstrip("/").lower()
        here = path.as_posix().lower()
        return here == base or here.startswith(base + "/")
    literal = Path(raw)
    candidates = {literal}
    try:
        candidates.add(literal.resolve())
    except (OSError, ValueError):
        pass
    return any(path == base or _within(base, path) for base in candidates)


def _protected_system(path: Path, settings: GateSettings) -> bool:
    """Does this resolved target land in one of the machine's own directories?

    The scratch carve-out wins over the roots: ``/var/tmp`` is inside ``/var`` and is where
    ordinary work happens (:data:`PROTECTED_PATHS_SYSTEM_EXEMPT_DEFAULT`).
    """
    for raw in settings.protected_paths_system_exempt:
        if _system_root_matches(path, raw):
            return False
    return any(_system_root_matches(path, raw) for raw in settings.protected_paths_system)


def _protected(path: Path, ctx: GateContext, settings: GateSettings) -> bool:
    """Does this resolved target land on a protected path? (PRD §3.1 ``protected-path``.)

    Three sets: absolute home paths (``~/.ssh``, shell rc files, the harness's own config dir),
    the machine's own directories (``/etc``, ``/usr``, ``C:/Windows`` — :func:`_protected_system`),
    and names matched at any depth inside the workspace (``.git``, ``.env*``, ``*.pem``). A
    directory entry protects its subtree, so ``.git/hooks/pre-commit`` is protected via
    ``.git``. The harness's runtime store is exempted (:func:`_exempt_runtime_store`).

    The system set is what makes ``auto`` safe to ship as the default (owner ruling 2026-09-11:
    "auto = thinnest interaction; asks only when genuinely dangerous"): ``auto`` allows a write
    outside the project silently, and without this set "outside the project" would include
    ``/etc/hosts``.
    """
    if _exempt_runtime_store(path):
        return False
    if _protected_system(path, settings):
        return True
    for raw in settings.protected_paths_home:
        try:
            base = Path(raw).expanduser().resolve()
        except (OSError, ValueError):
            continue
        if path == base or _within(base, path):
            return True
    try:
        config_dir = global_config_dir().resolve()
    except (OSError, ValueError):
        config_dir = None
    if config_dir is not None and (path == config_dir or _within(config_dir, path)):
        return True
    # `auto` keeps only the in-project names whose CONTENTS decide what runs next (owner ruling
    # 2026-09-11): `.git` — hooks and config — and `.localharness`. Writing your own project's
    # `.env` is ordinary work, and the key material the other patterns guard lives under the home
    # set, which applies in every mode. `guarded` keeps the full set.
    patterns = (
        settings.auto_blacklist.protected_paths_workspace if ctx.mode == "auto"
        else settings.protected_paths_workspace
    )
    for container in (ctx.workspace, ctx.boundary):
        if container is None:
            continue
        container = Path(container).expanduser().resolve()
        if not _within(container, path):
            continue
        for part in path.relative_to(container).parts:
            if any(fnmatch.fnmatch(part, pattern) for pattern in patterns):
                return True
    return False


def _truncate(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= DISPLAY_ARG_MAX_CHARS:
        return flat
    return flat[: DISPLAY_ARG_MAX_CHARS - 1] + _ELLIPSIS


@dataclass(frozen=True)
class _Ask:
    """One thing a call would have to ask about, before the asks are merged into a request.

    A call raises a LIST of these — every unsatisfied class it touches — and :func:`_decide`
    turns the list into the single :class:`PermissionRequest` a human answers once (PRD §7:
    "'always' → the same command never asks again in that workspace"). ``detail`` is the short
    identity of this one ask (a shell signature, a directory) used when several asks of the
    same class are collapsed onto one line.
    """

    klass: str
    key: Optional[str]
    grantable: bool
    reason: str
    detail: str = ""
    allowed_in_auto: bool = True
    """Whether ``auto`` lets this one through (owner ruling 2026-09-11). See
    :func:`_ask_record` for the rule and :data:`~localharness.agent.gate_types.AUTO_ASK_CLASSES`
    for why the blacklist is these two classes."""
    auto_entry: Optional[str] = None
    """WHICH blacklist entry fired, when one did — reported by ``ask-rate --mode auto`` so the
    list can be curated from evidence (:func:`_auto_blacklisted`)."""


def _ask_record(
    klass: str, key: Optional[str], reason: str, *, grantable: Optional[bool] = None,
    detail: Optional[str] = None, allowed_in_auto: Optional[bool] = None,
    auto_entry: Optional[str] = None,
) -> _Ask:
    """One ask, with ``grantable`` defaulting to the class's tier (``UNGRANTABLE_CLASSES``).

    It is overridden only where a normally-grantable class has nothing rememberable to key on —
    a shell segment whose command NAME is computed at runtime
    (:data:`DYNAMIC_COMMAND_NAME_PREFIXES`).

    ``allowed_in_auto`` defaults to the blacklist rule of owner ruling 2026-09-11 ("auto =
    thinnest interaction; asks only when genuinely dangerous"): ``auto`` asks when the class is
    in :data:`~localharness.agent.gate_types.AUTO_ASK_CLASSES`, and otherwise when the ask
    carries NO key — which is exactly the set of calls the gate could not read well enough to
    check against the blacklist at all (an unreadable ``command``, a command name computed at
    runtime, a path argument that is not a string). Allowing those would be allowing a call
    BECAUSE it could not be classified, which inverts the rule. It is passed explicitly only by
    the shell branch, where a destructive command that points entirely inside the project is
    allowed (:func:`_destructive_stays_inside`).
    """
    return _Ask(
        klass=klass,
        key=key,
        grantable=(klass not in UNGRANTABLE_CLASSES) if grantable is None else grantable,
        reason=reason,
        detail=detail if detail is not None else (key or ""),
        allowed_in_auto=(
            (klass not in AUTO_ASK_CLASSES and key is not None)
            if allowed_in_auto is None else allowed_in_auto
        ),
        auto_entry=auto_entry,
    )


def _granted(ctx: GateContext, klass: str, key: str) -> bool:
    """Is there a stored "always here" for this ``(class, key)``? (PRD §3.3.)

    The one place the grant store is read, so :data:`MODES_WITHOUT_GRANTS` can hold for the
    whole verdict rather than in each branch that remembered to check.
    """
    if ctx.mode in MODES_WITHOUT_GRANTS:
        return False
    return ctx.grants(ctx.workspace, klass, key) is not None


def _ordered_unique(asks: list[_Ask]) -> list[_Ask]:
    """Deduplicate on ``(klass, key)`` and sort by :data:`ASK_SEVERITY_ORDER` (stable)."""
    seen: dict[tuple[str, Optional[str]], _Ask] = {}
    for ask in asks:
        seen.setdefault((ask.klass, ask.key), ask)
    order = {klass: i for i, klass in enumerate(ASK_SEVERITY_ORDER)}
    return sorted(seen.values(), key=lambda a: order.get(a.klass, len(order)))


def _grouped_reasons(asks: list[_Ask]) -> list[tuple[str, str]]:
    """``[(klass, one readable phrase)]`` — one entry per class, in the order given."""
    out: list[tuple[str, str]] = []
    for klass in dict.fromkeys(a.klass for a in asks):
        members = [a for a in asks if a.klass == klass]
        if len(members) == 1:
            out.append((klass, members[0].reason))
            continue
        details = ", ".join(dict.fromkeys(a.detail for a in members if a.detail))
        template = GROUPED_REASON_BY_CLASS.get(klass, GROUPED_REASON_FALLBACK)
        out.append((klass, template.format(klass=klass, details=details)))
    return out


def _decide(
    ctx: GateContext,
    tool_name: str,
    params: dict,
    asks: list[_Ask],
    *,
    salient: str,
) -> VerdictResult:
    """Merge every ask one call raised into ONE verdict, applying PRD §3.4's mode effects.

    This is the shape the milestone is for: a human answers a tool call once, not once per
    class it touches. The most severe ask (:data:`ASK_SEVERITY_ORDER`) names the request; every
    grantable ask's key travels on ``grant_keys`` so one "always here" remembers all of them;
    and ONE ungrantable ask makes the whole request ungrantable — the call asks every time and
    no grant is written, because a single "always" must never quietly remember a destructive or
    protected-path exposure that was bundled with a benign one.

    A stored "never here" (:data:`REFUSAL_DENY_REASON`) is checked over every collected ask
    first and denies the whole call without prompting: the human answered the call, so one
    refused key is enough, and no mode may override it — a refusal is a DENY, and DENY ignores
    modes (PRD §3.3, §3.4).

    ``unattended`` turns every remaining ASK into ALLOW (today's behavior named honestly — bench
    and scheduled jobs pin it, critic finding 7). ``trusted`` allows a request only when every
    ask in it is grantable.

    ``auto`` — the default since v0.14.1 (owner ruling 2026-09-11: "auto = thinnest interaction;
    asks only when genuinely dangerous") — drops every ask the blacklist does not name and asks
    about what is left. When nothing is left the call ALLOWS; when something is, the request
    names only the dangerous part, so the question a person reads is "this deletes outside your
    project", never "this deletes outside your project, and also runs a command I have not seen
    before". A refusal is still checked over EVERY ask first, including the ones auto would have
    allowed: "never here" is a DENY and outranks every mode.
    """
    ordered = _ordered_unique(asks)
    for ask in ordered:
        refusal = _refused(ctx, ask.klass, ask.key)
        if refusal is not None:
            return VerdictResult(Verdict.DENY, f"{REFUSAL_DENY_REASON}: {refusal.key}")
    if ctx.mode == "unattended":
        return VerdictResult(
            Verdict.ALLOW, f"unattended mode: {ordered[0].klass} allowed without asking"
        )
    if ctx.mode == "auto":
        blacklisted = [ask for ask in ordered if not ask.allowed_in_auto]
        if not blacklisted:
            return VerdictResult(
                Verdict.ALLOW, f"auto mode: {ordered[0].klass} is not on the blacklist"
            )
        ordered = blacklisted
    primary = ordered[0]
    grantable = all(a.grantable for a in ordered)
    if grantable and ctx.mode == "trusted":
        return VerdictResult(Verdict.ALLOW, f"trusted mode: {primary.klass} allowed without asking")
    groups = _grouped_reasons(ordered)
    reason = "; ".join(text for _, text in groups)
    body = "; ".join(f"{klass} — {text}" for klass, text in groups)
    display = f"{tool_name}: {_truncate(salient)}  ({body})" if salient else f"{tool_name}  ({body})"
    return VerdictResult(
        verdict=Verdict.ASK,
        reason=reason,
        request=PermissionRequest(
            tool_name=tool_name,
            tool_params=params,
            klass=primary.klass,
            key=primary.key,
            grantable=grantable,
            reason=reason,
            display=display,
            grant_keys=tuple((a.klass, a.key) for a in ordered if a.grantable and a.key),
            auto_entry=next((a.auto_entry for a in ordered if a.auto_entry), None),
        ),
    )


def _refused(ctx: GateContext, klass: str, key: Optional[str]) -> Optional[Refusal]:
    """The stored "never here" covering this ask, if a human wrote one (PRD §3.3).

    Consulted at every point the grant store is (and BEFORE it, so a refusal beats a later
    "always" on the same key), and again over every collected ask in :func:`_decide` — one
    refused key denies the whole call, because the human answered the call, not the class.

    For a path-keyed class the walk goes upward from the key, so a refusal on ``/tmp/x`` covers
    ``/tmp/x/y/f``; for every other class the key matches exactly. Nothing here is a text
    pattern: a refusal on the signature ``cp`` cannot touch ``scp`` or ``cpio``.
    """
    if ctx.refusals is None or not key:
        return None
    if klass in PATH_KEYED_CLASSES:
        here = Path(key)
        for candidate in (here, *here.parents):
            refusal = ctx.refusals(ctx.workspace, klass, str(candidate))
            if refusal is not None:
                return refusal
        return None
    return ctx.refusals(ctx.workspace, klass, key)


def _granted_directory(ctx: GateContext, klass: str, directory: Path) -> bool:
    """Is this directory — or any directory above it — already granted? (PRD §3.1 edit-outside.)

    An ``edit-outside`` grant is keyed on the target's parent directory, and a human who
    answered "always here" for ``/tmp/x`` meant the place, not that one path segment.
    Verification A defect D4: the lookup checked only the immediate parent, so a project
    writing into fresh subdirectories under an already-approved root paid a first-exposure
    prompt forever — PRD §8's "growing vocabularies" risk arriving through paths.

    The walk goes from the parent UPWARD to the filesystem root and stops at the first hit, so
    it can only ever find a grant a human gave on a wider directory; it never invents one. It
    cannot widen a grant into a protected path either: :func:`_target_asks` classifies a
    protected target before it ever reaches here (PRD §3.1's fixed order), so a grant on ``~``
    does not cover ``~/.ssh``.

    ``klass`` is always a PATH-KEYED class (:data:`PATH_KEYED_CLASSES`; ``edit-outside`` is the
    only one that consults grants) and is passed through to the lookup, so the walk can only
    ever find grants a human gave about directories. Without it, a shell-signature grant whose
    key happened to read as a path — ``/usr/local/bin/deploy``, or simply ``/tmp`` — pre-approved
    every write under that directory.
    """
    for candidate in (directory, *directory.parents):
        if _granted(ctx, klass, str(candidate)):
            return True
    return False


def _target_asks(
    ctx: GateContext,
    settings: GateSettings,
    targets: tuple[tuple[str, Optional[Path]], ...],
) -> list[_Ask]:
    """The write-shaped checks of PRD §3.1, run over EVERY target of one call.

    Each target is classified once, in class order — protected-path, then no-boundary (both
    ungrantable), then edit-outside — and every unsatisfied target contributes its own ask, so
    a command that writes two new places outside the boundary asks about both at once instead
    of once per turn. A target already covered by a grant contributes nothing.
    """
    asks: list[_Ask] = []
    for raw, resolved in targets:
        if resolved is not None and _protected(resolved, ctx, settings):
            asks.append(_ask_record(
                "protected-path", str(resolved), f"writes a protected path ({resolved})",
            ))
            continue
        if ctx.boundary is None:
            asks.append(_ask_record(
                "no-boundary", str(resolved or raw),
                "no workspace boundary here (the project root is your home directory or above)",
            ))
            continue
        if resolved is not None and _within(ctx.boundary, resolved):
            continue
        if resolved is not None:
            key, where = str(resolved.parent), str(resolved)
            granted = _granted_directory(ctx, "edit-outside", resolved.parent)
        else:
            key, where = raw, f"{raw} (unresolvable)"
            granted = _granted(ctx, "edit-outside", key)
        if granted and _refused(ctx, "edit-outside", key) is None:
            continue
        asks.append(_ask_record(
            "edit-outside", key, f"writes outside the workspace boundary ({where})",
        ))
    return asks


def _evaluate_write(
    tool_name: str, params: dict, ctx: GateContext, settings: GateSettings
) -> VerdictResult:
    """``write`` / ``edit`` and any tool in the ``fs.write`` group (PRD §3.1).

    Read-only mode is handled up front in :func:`evaluate`, not here."""
    param = WRITE_TOOL_PATH_PARAMS.get(tool_name)
    bad = _bad_string_param(params, (param,) if param else WRITE_PATH_PARAM_CANDIDATES)
    if bad is not None:
        return _decide(ctx, tool_name, params, [_ask_record(
            "edit-outside", None, NON_STRING_PATH_REASON.format(param=bad),
            grantable=False, detail=bad,
        )], salient="")
    raw = _write_target(tool_name, params)
    target = ((raw, _resolve(raw, ctx.workspace)),) if raw else ((MISSING_TARGET_DISPLAY, None),)
    salient = raw or ""
    asks = _target_asks(ctx, settings, target)
    if not asks and not ctx.has_review_surface:
        key = str(Path(ctx.workspace).expanduser().resolve())
        if not _granted(ctx, "edit-unreviewed", key) or _refused(ctx, "edit-unreviewed", key):
            asks.append(_ask_record(
                "edit-unreviewed", key, "this channel shows no diff to review the edit in",
            ))
    if not asks:
        return VerdictResult(Verdict.ALLOW, "in-workspace edit")
    return _decide(ctx, tool_name, params, asks, salient=salient)


def _auto_blacklisted(
    segment, anchor: Optional[Path], ctx: GateContext, settings: GateSettings
) -> Optional[str]:
    """Which :class:`~localharness.agent.gate_types.AutoBlacklist` entry this segment hits, if any.

    The whole of ``auto``'s shell rule, in one function (owner ruling 2026-09-11: "only hard
    blacklists for git and rm and shit like that, and even then very minimal"). Returns the
    entry's own name — a signature, or the target-scoped verb, or ``pipe-to-shell`` — so the
    ask-rate report can say WHICH blacklist entry fired over a corpus and the list can get
    shorter with evidence rather than longer with fear. ``None`` means the segment runs silently,
    destructive or not: the fuller set ``guarded`` uses is much longer, and every entry outside
    this one stopped the owner mid-task in the v0.14.0 dogfood.

    A target-scoped verb (``rm``, ``chmod``, ``find -delete``, the Windows deletes) fires only
    when the command points somewhere it should not. Three ways that happens, all of them "I
    cannot say this is safe": the classifier could not place a target (a variable, a glob, a
    ``cd`` it could not follow, or no target at all); a target is outside the boundary; a target
    is protected. ``rm -rf build`` inside your own checkout is what a build script does, and it
    runs.

    When there is no boundary (the session started in ``$HOME``) the effective boundary is the
    directory the command actually runs in. ``auto`` does not raise the ``no-boundary`` ask at
    all, so without this the mode would have no answer for ``rm -rf`` in a home session; "inside
    the directory you are standing in" is the honest fallback, and everything above ``$HOME``
    still lands outside it.
    """
    blacklist = settings.auto_blacklist
    if blacklist.pipe_to_shell and segment.pipe_to_shell:
        return AUTO_PIPE_TO_SHELL_ENTRY
    signature = segment.signature
    if signature in blacklist.irreversible_signatures:
        return signature
    verb = signature.split(" ", 1)[0]
    if verb in blacklist.irreversible_signatures:
        return verb
    if verb not in blacklist.target_scoped_verbs:
        return None
    if segment.unresolvable_destructive:
        return signature
    boundary = ctx.boundary if ctx.boundary is not None else anchor
    if boundary is None:
        return signature
    for raw in segment.destructive_targets:
        resolved = _resolve(raw, anchor)
        if resolved is None or _protected(resolved, ctx, settings):
            return signature
        if not _within(Path(boundary), resolved):
            return signature
    return None


def _evaluate_shell(
    tool_name: str, params: dict, ctx: GateContext, settings: GateSettings
) -> VerdictResult:
    """``bash_exec`` (PRD §3.1 shell rows, classified by PRD §3.2).

    The classifier is imported lazily so this module stays importable — and every non-shell
    verdict testable — independently of it.
    """
    name = SHELL_COMMAND_PARAMS.get(tool_name, "command")
    command = params.get(name)
    if _bad_string_param(params, (name,)) is not None:
        return _decide(ctx, tool_name, params, [_ask_record(
            "shell-unfamiliar", None, NON_STRING_SHELL_COMMAND_REASON,
            grantable=False, detail=type(command).__name__,
        )], salient="")
    if not isinstance(command, str) or not command.strip():
        return VerdictResult(Verdict.ALLOW, "no shell command to classify")

    from localharness.agent.shell_classify import classify_shell

    classified = classify_shell(command, settings)
    non_read_only = tuple(s for s in classified.segments if not s.read_only)

    anchor = _shell_anchor(tool_name, params, ctx)

    asks: list[_Ask] = []
    destructive = {s.signature for s in classified.segments if s.destructive}
    for signature in sorted(destructive):
        entries = [
            _auto_blacklisted(segment, anchor, ctx, settings)
            for segment in classified.segments
            if segment.destructive and segment.signature == signature
        ]
        hit = next((entry for entry in entries if entry is not None), None)
        asks.append(_ask_record(
            "shell-destructive", signature, f"destructive shell command ({signature})",
            allowed_in_auto=hit is None, auto_entry=hit,
        ))

    asks += _target_asks(ctx, settings, tuple(
        (target, _resolve(target, anchor))
        for segment in classified.segments
        for target in segment.write_targets
    ))

    for segment in non_read_only:
        if segment.signature in destructive:
            continue  # already asked about, under the stricter class
        if segment.signature.startswith(DYNAMIC_COMMAND_NAME_PREFIXES):
            asks.append(_ask_record(
                "shell-unfamiliar", None, DYNAMIC_COMMAND_NAME_REASON,
                grantable=False, detail=segment.signature,
            ))
            continue
        klass, reason = (
            ("interpreter-inline", f"runs code inline through an interpreter ({segment.signature})")
            if segment.inline_interpreter
            else ("shell-unfamiliar",
                  f"shell command not seen in this workspace before ({segment.signature})")
        )
        granted = _granted(ctx, klass, segment.signature)
        if granted and _refused(ctx, klass, segment.signature) is None:
            continue
        asks.append(_ask_record(klass, segment.signature, reason))
    if not asks:
        return VerdictResult(Verdict.ALLOW, "read-only shell")
    return _decide(ctx, tool_name, params, asks, salient=command)


def _evaluate_named(
    tool_name: str, params: dict, ctx: GateContext, *, klass: str, reason: str
) -> VerdictResult:
    """The classes keyed by tool NAME: ``code-exec``, ``delegate`` and ``tool-unfamiliar``.

    PRD §3.1 names the first two. ``tool-unfamiliar`` is the same shape for a tool in no known
    family — a plugin's, or one whose schema could not be read — so an unknown tool is answered
    once per workspace and then remembered, instead of running unasked.

    Read-only mode is handled up front in :func:`evaluate`, not here.
    """
    if _granted(ctx, klass, tool_name) and not _refused(ctx, klass, tool_name):
        return VerdictResult(Verdict.ALLOW, f"granted in this workspace: {tool_name}")
    return _decide(
        ctx, tool_name, params, [_ask_record(klass, tool_name, reason)],
        salient=_first_string(params),
    )


def _evaluate_network(
    tool_name: str, params: dict, ctx: GateContext, settings: GateSettings
) -> VerdictResult:
    """Network reads: ALLOW unless ``permissions.ask.network_hosts`` is on (PRD §3.1 choice 1).

    70 % of real tool calls are web fetches (PRD §5), so per-host prompts are off by default;
    with the knob on, the grant key is the host.
    """
    if not settings.ask_network_hosts:
        return VerdictResult(Verdict.ALLOW, "network read")
    name = NETWORK_URL_PARAMS.get(tool_name, "url")
    url = params.get(name)
    if _bad_string_param(params, (name,)) is not None:
        # Same split as the shell and write branches: a url the gate cannot read is not a call
        # that names no host, it is a call whose host it cannot see. With the per-host knob on,
        # collapsing the two let an unreadable url skip the ask the knob exists to raise.
        return _decide(ctx, tool_name, params, [_ask_record(
            "network-host", None, NON_STRING_URL_REASON.format(param=name),
            grantable=False, detail=type(url).__name__,
        )], salient="")
    if not isinstance(url, str) or not url.strip():
        return VerdictResult(Verdict.ALLOW, "network call names no host")
    host = urlparse(url).hostname
    if not host:
        return VerdictResult(Verdict.ALLOW, "network call names no host")
    if _granted(ctx, "network-host", host) and not _refused(ctx, "network-host", host):
        return VerdictResult(Verdict.ALLOW, f"granted in this workspace: {host}")
    return _decide(
        ctx, tool_name, params,
        [_ask_record("network-host", host, f"first fetch from {host} in this workspace")],
        salient=url,
    )


def _evaluate_mcp(
    tool_name: str, params: dict, meta: ToolMeta, ctx: GateContext, settings: GateSettings
) -> VerdictResult:
    """Any MCP tool: ask once per ``(server, tool)`` unless the server is trusted (PRD §3.1).

    A repo that registers an MCP server has added things that ASK, not things that run
    (PRD §3.3, critic finding 6).
    """
    server = meta.mcp_server or ""
    bare = tool_name.split("__", 1)[1] if server and tool_name.startswith(f"{server}__") else tool_name
    key = f"mcp/{server}/{bare}"
    if server and server in settings.mcp_trusted_servers:
        return VerdictResult(Verdict.ALLOW, f"trusted MCP server: {server}")
    if _granted(ctx, "mcp", key) and not _refused(ctx, "mcp", key):
        return VerdictResult(Verdict.ALLOW, f"granted in this workspace: {key}")
    return _decide(
        ctx, tool_name, params,
        [_ask_record("mcp", key, f"first use of {key} in this workspace")],
        salient=_first_string(params),
    )


def _read_only_denies(
    kind: str, tool_name: str, params: dict, meta: ToolMeta, settings: GateSettings
) -> bool:
    """Does ``read-only`` mode refuse this call? (PRD §3.4, one place.)

    A SHELL call is judged on its segments and nothing else: ``bash_exec``'s schema is marked
    ``destructive`` as a whole (``bash_tool.py:240``), and most shell calls are reads, so the
    flag would refuse ``ls`` — PRD §3.4 refuses the *non-read-only* shell. Any other call is
    refused when

    * its branch is one of :data:`READ_ONLY_DENIED_KINDS` — write, code, delegate — which covers
      both the builtins by name and any tool declaring those groups; or
    * the tool's own schema says ``destructive``. That flag is the only thing a tool can tell the
      gate about itself, and read-only is where it has to count: ``remember`` mutates the memory
      store, so it carries the flag and is refused here.

    This used to be re-checked by hand inside each branch, and the branches disagreed: the
    network branch never checked, so a plugin ``web`` tool with ``destructive=True`` ran; the
    read-tier tail checked only the flag, so ``remember`` (group ``memory``,
    ``destructive=False`` at the time) wrote memory in read-only mode. One check, consulted
    before the branches, is what makes the mode a property of the MODE.

    The shell branch classifies the command again after this returns False. That is one extra
    parse, only in read-only mode, only for a command that turned out to be all reads.
    """
    if kind == "shell":
        name = SHELL_COMMAND_PARAMS.get(tool_name, "command")
        command = params.get(name)
        if _bad_string_param(params, (name,)) is not None:
            return True  # unreadable command: it cannot be SHOWN to be a read, so read-only refuses
        if not isinstance(command, str) or not command.strip():
            return False

        from localharness.agent.shell_classify import classify_shell

        classified = classify_shell(command, settings)
        return bool(any(not s.read_only for s in classified.segments) or classified.write_targets)
    return meta.destructive or kind in READ_ONLY_DENIED_KINDS


def _first_string(params: dict) -> str:
    """The first string argument, for the one-line display (PRD §3.5)."""
    for value in params.values():
        if isinstance(value, str) and value.strip():
            return value
    return ""


# ------------------------------------------------------------------- the verdict

def evaluate(
    tool_name: str,
    tool_params: dict,
    tool_meta: ToolMeta,
    ctx: GateContext,
    settings: GateSettings,
) -> VerdictResult:
    """The whole verdict for one tool call (PRD §3.1), pure, and asked ONCE.

    Order, fixed: **DENY → ungrantable ASK → refusals → grant lookup → grantable ASK → ALLOW**,
    where a refusal is a human's "never here" recorded as a negative grant (PRD §3.3) and denies
    the call outright without prompting. With the
    mode effects of PRD §3.4 applied when the asks are merged (:func:`_decide`). Deny is never
    overridden by a grant or a mode. The ungrantable checks (shell-destructive, protected-path,
    no-boundary) run before any grant is consulted — critic finding 12.

    The tiers decide precedence, not how many questions a human answers: every unsatisfied ask
    a call raises is COLLECTED and returned as one :class:`PermissionRequest` carrying all of
    their keys (``grant_keys``), so ``mkdir -p /tmp/x/y && touch /tmp/x/y/f`` prompts once and
    one "always here" remembers the lot. Asking per class was the shipped behaviour and it cost
    a whole turn per class (verification A, defect D1); PRD §7's acceptance line and §3.6's
    median-zero SLO are properties of the CALL, not of the class.

    ``unattended`` mode turns every ASK into ALLOW, including the ungrantable ones: that is
    today's shipped behavior named honestly, and it is what bench and scheduled jobs pin
    (PRD §3.4, critic findings 7 and 8). PRD §3.1's table says "DENY in unattended" for the
    ungrantable tier; §3.4 and the bench requirement say ALLOW, and §3.4 is the one implemented
    here — bench scenarios use ``rm`` and ``chmod`` and must not start failing.
    """
    params = tool_params if isinstance(tool_params, dict) else {}
    if ctx.deny is not None:
        denied = ctx.deny(tool_name, params)
        if denied.denied:
            return VerdictResult(Verdict.DENY, denied.reason or "matches a deny pattern")

    kind = _kind(tool_name, tool_meta)
    if ctx.mode == "read-only" and _read_only_denies(kind, tool_name, params, tool_meta, settings):
        return VerdictResult(Verdict.DENY, READ_ONLY_DENY_REASON)
    if kind == "mcp":
        return _evaluate_mcp(tool_name, params, tool_meta, ctx, settings)
    if kind == "write":
        return _evaluate_write(tool_name, params, ctx, settings)
    if kind == "shell":
        return _evaluate_shell(tool_name, params, ctx, settings)
    if kind == "code":
        return _evaluate_named(
            tool_name, params, ctx,
            klass="code-exec", reason=f"runs code in this session ({tool_name})",
        )
    if kind == "delegate":
        return _evaluate_named(
            tool_name, params, ctx,
            klass="delegate", reason=f"dispatches a subagent ({tool_name})",
        )
    if kind == "network":
        return _evaluate_network(tool_name, params, ctx, settings)
    if kind == UNFAMILIAR_TOOL_KIND:
        return _evaluate_named(
            tool_name, params, ctx, klass=UNFAMILIAR_TOOL_KIND, reason=UNFAMILIAR_TOOL_REASON,
        )
    return VerdictResult(Verdict.ALLOW, "read-tier tool")
