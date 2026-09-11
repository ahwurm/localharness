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

Known gap, named rather than hidden: ``read-only`` mode denies the kinds PRD §3.4 enumerates
(write-shaped, non-read-only shell, code, delegate) plus anything a tool schema marks
``destructive``; an MCP tool that mutates without that flag is only gated by its once-per-tool
ask, not by read-only mode.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

from localharness.agent.gate_types import (
    DEFAULT_MODE,
    UNGRANTABLE_CLASSES,
    GateSettings,
    Grant,
    Mode,
    PermissionRequest,
    ToolMeta,
    Verdict,
    VerdictResult,
)
from localharness.agent.permissions import PermissionResult
from localharness.config.paths import ARCHIVE_DB_NAME, global_config_dir

GrantLookup = Callable[[Path, str], Optional[Grant]]
"""``(workspace, key) -> Grant | None`` — ``config.grants.GrantStore.lookup`` in production."""

DenyFn = Callable[[str, dict], PermissionResult]
"""``(tool_name, tool_params) -> PermissionResult`` — wraps the existing
``agent/permissions.PermissionEvaluator.evaluate`` (the DENY tier, unchanged)."""


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
}
"""Fallback classification by ``ToolSchema.group`` (A3, PRD §6) for tools this module does not
know by name — a plugin's or a future builtin's. Names win when known, because a name pins the
exact parameter to read; groups are the open extension point."""

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

READ_ONLY_DENY_REASON = "not permitted in read-only mode"
"""PRD §3.4: the soft-deny text. It is returned to the model AS the tool observation, so the
model can re-plan rather than retry — hence a sentence, not an error code."""

MISSING_TARGET_DISPLAY = "<no path argument>"
"""A write-shaped call that names no target at all. It cannot be resolved, so PRD §3.2 step 8's
"unresolvable targets are treated as outside" applies — the call asks rather than passing."""

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
    return KIND_BY_GROUP.get(meta.group, "allow")


def _write_target(tool_name: str, params: dict) -> Optional[str]:
    """The raw target string of a write-shaped tool call, or None when it names none."""
    param = WRITE_TOOL_PATH_PARAMS.get(tool_name)
    names = (param,) if param else WRITE_PATH_PARAM_CANDIDATES
    for name in names:
        value = params.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _resolve(target: str, workspace: Path) -> Optional[Path]:
    """Realpath a write target, or None when it cannot be resolved (PRD §3.2 step 8).

    Relative targets resolve against the workspace — the directory the tool runs in. A target
    holding a variable, glob or substitution resolves to None and is treated as outside.
    """
    if any(ch in target for ch in UNRESOLVABLE_TARGET_CHARS):
        return None
    try:
        path = Path(target).expanduser()
        if not path.is_absolute():
            path = Path(workspace) / path
        return path.resolve()
    except OSError:
        return None


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
    except OSError:
        return False
    if not _within(config_dir, path) or path == config_dir:
        return False
    return path.relative_to(config_dir).parts[0] in HARNESS_RUNTIME_STORE_NAMES


def _protected(path: Path, ctx: GateContext, settings: GateSettings) -> bool:
    """Does this resolved target land on a protected path? (PRD §3.1 ``protected-path``.)

    Two sets: absolute home paths (``~/.ssh``, shell rc files, the harness's own config dir)
    and names matched at any depth inside the workspace (``.git``, ``.env*``, ``*.pem``). A
    directory entry protects its subtree, so ``.git/hooks/pre-commit`` is protected via
    ``.git``. The harness's runtime store is exempted (:func:`_exempt_runtime_store`).
    """
    if _exempt_runtime_store(path):
        return False
    for raw in settings.protected_paths_home:
        try:
            base = Path(raw).expanduser().resolve()
        except OSError:
            continue
        if path == base or _within(base, path):
            return True
    try:
        config_dir = global_config_dir().resolve()
    except OSError:
        config_dir = None
    if config_dir is not None and (path == config_dir or _within(config_dir, path)):
        return True
    for container in (ctx.workspace, ctx.boundary):
        if container is None:
            continue
        container = Path(container).expanduser().resolve()
        if not _within(container, path):
            continue
        for part in path.relative_to(container).parts:
            if any(fnmatch.fnmatch(part, pattern) for pattern in settings.protected_paths_workspace):
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


def _ask_record(
    klass: str, key: Optional[str], reason: str, *, grantable: Optional[bool] = None,
    detail: Optional[str] = None,
) -> _Ask:
    """One ask, with ``grantable`` defaulting to the class's tier (``UNGRANTABLE_CLASSES``).

    It is overridden only where a normally-grantable class has nothing rememberable to key on —
    a shell segment whose command NAME is computed at runtime
    (:data:`DYNAMIC_COMMAND_NAME_PREFIXES`).
    """
    return _Ask(
        klass=klass,
        key=key,
        grantable=(klass not in UNGRANTABLE_CLASSES) if grantable is None else grantable,
        reason=reason,
        detail=detail if detail is not None else (key or ""),
    )


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

    ``unattended`` turns every ASK into ALLOW (today's behavior named honestly — bench and
    scheduled jobs pin it, critic finding 7). ``trusted`` allows a request only when every ask
    in it is grantable. DENY is never reached from here.
    """
    ordered = _ordered_unique(asks)
    primary = ordered[0]
    grantable = all(a.grantable for a in ordered)
    if ctx.mode == "unattended":
        return VerdictResult(Verdict.ALLOW, f"unattended mode: {primary.klass} allowed without asking")
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
        ),
    )


def _granted_directory(ctx: GateContext, directory: Path) -> bool:
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
    """
    for candidate in (directory, *directory.parents):
        if ctx.grants(ctx.workspace, str(candidate)) is not None:
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
            granted = _granted_directory(ctx, resolved.parent)
        else:
            key, where = raw, f"{raw} (unresolvable)"
            granted = ctx.grants(ctx.workspace, key) is not None
        if granted:
            continue
        asks.append(_ask_record(
            "edit-outside", key, f"writes outside the workspace boundary ({where})",
        ))
    return asks


def _evaluate_write(
    tool_name: str, params: dict, ctx: GateContext, settings: GateSettings
) -> VerdictResult:
    """``write`` / ``edit`` and any tool in the ``fs.write`` group (PRD §3.1)."""
    if ctx.mode == "read-only":
        return VerdictResult(Verdict.DENY, READ_ONLY_DENY_REASON)
    raw = _write_target(tool_name, params)
    target = ((raw, _resolve(raw, ctx.workspace)),) if raw else ((MISSING_TARGET_DISPLAY, None),)
    salient = raw or ""
    asks = _target_asks(ctx, settings, target)
    if not asks and not ctx.has_review_surface:
        key = str(Path(ctx.workspace).expanduser().resolve())
        if ctx.grants(ctx.workspace, key) is None:
            asks.append(_ask_record(
                "edit-unreviewed", key, "this channel shows no diff to review the edit in",
            ))
    if not asks:
        return VerdictResult(Verdict.ALLOW, "in-workspace edit")
    return _decide(ctx, tool_name, params, asks, salient=salient)


def _evaluate_shell(
    tool_name: str, params: dict, ctx: GateContext, settings: GateSettings
) -> VerdictResult:
    """``bash_exec`` (PRD §3.1 shell rows, classified by PRD §3.2).

    The classifier is imported lazily so this module stays importable — and every non-shell
    verdict testable — independently of it.
    """
    command = params.get(SHELL_COMMAND_PARAMS.get(tool_name, "command"))
    if not isinstance(command, str) or not command.strip():
        return VerdictResult(Verdict.ALLOW, "no shell command to classify")

    from localharness.agent.shell_classify import classify_shell

    classified = classify_shell(command, settings)
    non_read_only = tuple(s for s in classified.segments if not s.read_only)
    if ctx.mode == "read-only" and (non_read_only or classified.write_targets):
        return VerdictResult(Verdict.DENY, READ_ONLY_DENY_REASON)

    asks: list[_Ask] = []
    destructive = {s.signature for s in classified.segments if s.destructive}
    for signature in sorted(destructive):
        asks.append(_ask_record(
            "shell-destructive", signature, f"destructive shell command ({signature})",
        ))

    asks += _target_asks(ctx, settings, tuple(
        (target, _resolve(target, ctx.workspace))
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
        if ctx.grants(ctx.workspace, segment.signature) is not None:
            continue
        if segment.inline_interpreter:
            asks.append(_ask_record(
                "interpreter-inline", segment.signature,
                f"runs code inline through an interpreter ({segment.signature})",
            ))
        else:
            asks.append(_ask_record(
                "shell-unfamiliar", segment.signature,
                f"shell command not seen in this workspace before ({segment.signature})",
            ))
    if not asks:
        return VerdictResult(Verdict.ALLOW, "read-only shell")
    return _decide(ctx, tool_name, params, asks, salient=command)


def _evaluate_named(
    tool_name: str, params: dict, ctx: GateContext, *, klass: str, reason: str
) -> VerdictResult:
    """The classes keyed by tool name: ``code-exec`` and ``delegate`` (PRD §3.1)."""
    if ctx.mode == "read-only":
        return VerdictResult(Verdict.DENY, READ_ONLY_DENY_REASON)
    if ctx.grants(ctx.workspace, tool_name) is not None:
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
    url = params.get(NETWORK_URL_PARAMS.get(tool_name, "url"))
    if not isinstance(url, str) or not url.strip():
        return VerdictResult(Verdict.ALLOW, "network call names no host")
    host = urlparse(url).hostname
    if not host:
        return VerdictResult(Verdict.ALLOW, "network call names no host")
    if ctx.grants(ctx.workspace, host) is not None:
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
    if ctx.mode == "read-only" and meta.destructive:
        return VerdictResult(Verdict.DENY, READ_ONLY_DENY_REASON)
    if server and server in settings.mcp_trusted_servers:
        return VerdictResult(Verdict.ALLOW, f"trusted MCP server: {server}")
    if ctx.grants(ctx.workspace, key) is not None:
        return VerdictResult(Verdict.ALLOW, f"granted in this workspace: {key}")
    return _decide(
        ctx, tool_name, params,
        [_ask_record("mcp", key, f"first use of {key} in this workspace")],
        salient=_first_string(params),
    )


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

    Order, fixed: **DENY → ungrantable ASK → grant lookup → grantable ASK → ALLOW**, with the
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
    if ctx.mode == "read-only" and tool_meta.destructive:
        return VerdictResult(Verdict.DENY, READ_ONLY_DENY_REASON)
    return VerdictResult(Verdict.ALLOW, "read-tier tool")
