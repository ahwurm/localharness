"""The effectful half of the permission gate: ask a human, remember the answer.

Implements PRD §3.5 (rendering and fail-closed), §3.6 (the two bus events) and the write side
of §3.3 (``.planning/2026-09-11-zed-acp-and-permission-spine-prd.md``). The *decision* is pure
and lives in ``agent/verdict.evaluate``; everything with a side effect lives here, so there is
exactly one place that awaits a human, one place that writes a grant, and one place that can
fail closed.

The gate is per session and shared by reference with every subagent (PRD §3.4: "subagents
inherit the parent session's channel and mode"), which is why :attr:`PermissionGate.mode` is a
plain mutable attribute rather than a constructor-frozen value — a ``/mode`` switch in the
terminal reaches a running subagent through the same object.

Fail-closed is the whole safety story of §3.5's last row: a channel that cannot ask
(``asker is None`` — bench, cron, a non-tty pipe) turns every ASK into a DENY with a reason the
model can re-plan against, plus ONE loud warning naming the fix. Silence there would be a
regression nobody notices, so the warning is not optional.
"""
from __future__ import annotations

import asyncio
import fnmatch
import logging
import time
from pathlib import Path
from typing import Any, Optional

from localharness.agent.gate_types import (
    DEFAULT_MODE,
    MODE_STRICTNESS,
    Asker,
    Decision,
    GateOutcome,
    GateSettings,
    Mode,
    PermissionRequest,
    ToolMeta,
    Verdict,
)
from localharness.agent.permissions import PermissionResult
from localharness.agent.verdict import DenyFn, GateContext, derive_boundary, evaluate
from localharness.config.grants import GrantStore, new_grant

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ constants

NO_ASKER_REASON = "needs human approval; this channel cannot ask"
"""PRD §3.5, last row: the observation the model receives when the verdict is ASK and no
channel can render it. A sentence rather than an error code, because it is fed back as the tool
result so the model can choose a different route."""

NO_ASKER_WARNING = (
    "%s: a tool call needed human approval but this channel cannot ask, so it was denied "
    "(%s). For runs with nobody watching set `permissions.mode: unattended` in config; for an "
    "interactive session use a channel that can ask (terminal, Discord, Zed). This warning is "
    "logged once per session."
)
"""PRD §3.5: "a loud startup warning naming the fix". Logged at WARNING once per gate instance
— once per run, not once per call — so a bench or cron run that silently lost a capability is
visible in the log without drowning it."""

ASK_TIMEOUT_TOOL_MULTIPLE = 1.0
"""Derivation for the ask timeout when ``permissions.ask.timeout_s`` is unset (PRD §3.5: the
Discord default is "derived from the tool timeout").

The human is given exactly the wall time the tool call itself was allowed. Rationale, stated so
it is not a bare number: the gate must not make a turn take longer than it already could, and
one tool timeout is the only budget at the call site that is already tuned to this workspace's
tools. A multiple above 1.0 would let one unanswered prompt outlast the work it guards; below
1.0 would deny answers that arrive while the tool would still have been running.

It binds only on channels that cannot hold the question open (``ChannelAdapter.
ask_holds_dialog`` False — Discord and anything message-shaped). A terminal and a Zed dialog
have a person in front of them and PRD §3.5 gives both "Timeout: none", so the gate awaits
those with no deadline at all — verification A defect D3, where an unanswered terminal prompt
auto-denied after the tool's own timeout while three separate docs promised it would not.
Nothing is awaited under a deadline either when both the config value and the tool timeout are
None."""

TIMEOUT_DECISION = Decision(kind="reject_once")
"""PRD §3.5: "deny on timeout", and §3.6: a timeout resolves as ``reject_once`` so
timeout-denies are countable as their own guardrail. Never ``reject_always`` — nobody answered,
so nothing durable may be written."""

TIMEOUT_REASON = "no answer within {seconds:.0f}s; denied"

MS_PER_SECOND = 1000
"""Unit conversion for ``PermissionResolved.latency_ms`` — ``time.monotonic()`` returns seconds
and the event, like its siblings on the bus, reports milliseconds."""

MCP_GROUP_PREFIX = "mcp/"
"""``tools/mcp.py:81`` gives every MCP tool the group ``mcp/<server>``. That group IS how the
registry knows the server name, so the ``ToolMeta`` builder reads it rather than taking a
second, drift-prone path through the registry."""

SHELL_DENY_TOOLS: frozenset[str] = frozenset({"bash_exec"})
"""Tools whose one argument is an opaque command string, so a "never here" answer has to match
as a SUBSTRING of it. Mirrors ``verdict.SHELL_COMMAND_PARAMS``."""

SHELL_DENY_CLASSES: frozenset[str] = frozenset(
    {"shell-destructive", "shell-unfamiliar", "interpreter-inline"}
)
"""Ask classes whose grant key is a shell signature (PRD §3.1 table)."""

PATH_DENY_CLASSES: frozenset[str] = frozenset(
    {"edit-outside", "protected-path", "no-boundary", "edit-unreviewed"}
)
"""Ask classes whose grant key is a filesystem path — a directory for ``edit-outside`` and
``edit-unreviewed``, a file for ``protected-path``."""

SUBSTRING_DENY_CLASSES: frozenset[str] = frozenset({"network-host"})
"""Ask classes whose key appears INSIDE an argument (a host inside a URL)."""


def deny_pattern_for(tool_name: str, klass: str, key: Optional[str]) -> str:
    """Turn a "never here" answer into a deny pattern that matches this call from now on.

    PRD §3.3: ``"Never" answers write a deny pattern into the same file (deny tier, so it wins
    forever)``. The pattern has to match the same call through the DENY tier's fnmatch over raw
    argument strings, so the shape follows the shape of the key:

    * **shell** (key is a signature, or the tool is ``bash_exec``) → ``bash_exec(*<key>*)`` —
      the embedded form the shipped deny defaults already use for ``sudo`` and ``rm -rf``,
      because the signature sits anywhere inside the command string.
    * **path** (key is a directory or file) → ``<tool>(<key>*)`` for a filesystem tool: the
      prefix glob covers the subtree, which is what "never write there again" means when the
      key is the target's parent directory.
    * **substring** (a host inside a URL) → ``<tool>(*<key>*)``.
    * **anything else** (``code-exec``, ``delegate``, ``mcp``, or no key at all) → the bare
      tool name, which the DENY tier reads as "every call to this tool".
    """
    if not key:
        return tool_name
    if klass in SHELL_DENY_CLASSES or tool_name in SHELL_DENY_TOOLS:
        return f"{tool_name}(*{key}*)"
    if klass in PATH_DENY_CLASSES:
        return f"{tool_name}({key}*)"
    if klass in SUBSTRING_DENY_CLASSES:
        return f"{tool_name}(*{key}*)"
    return tool_name


def derive_session_boundary(
    cwd: Optional[Path] = None, local_dir: Optional[Path] = None
) -> Optional[Path]:
    """The workspace boundary for a session standing in ``cwd`` (PRD §3.1).

    The three inputs of :func:`verdict.derive_boundary` gathered from the running process:
    where you stand, the workspace layer v0.13 discovery applied (``ConfigLoader._local_dir``,
    when one applied), and the nearest git checkout. The git walk is
    ``config/paths._nearest_repo_root`` — the repo's existing marker-file walk, which handles a
    linked worktree's ``.git`` FILE and stops at ``$HOME``, and which runs no subprocess. It is
    imported rather than re-implemented so "what counts as your project" has one definition.
    """
    from localharness.config.paths import _nearest_repo_root

    here = Path(cwd) if cwd is not None else Path.cwd()
    home = Path.home()
    try:
        git_toplevel = _nearest_repo_root(here, home)
    except OSError:
        git_toplevel = None
    return derive_boundary(cwd=here, local_dir=local_dir, git_toplevel=git_toplevel, home=home)


def settings_from(permissions: Any) -> GateSettings:
    """``permissions.ask.to_gate_settings()``, tolerant of a config object that has no ask
    block (a hand-built stub in a test, or a plugin's own permission object)."""
    ask = getattr(permissions, "ask", None)
    to_settings = getattr(ask, "to_gate_settings", None)
    return to_settings() if callable(to_settings) else GateSettings()


def fail_closed_gate(
    *,
    permissions: Any,
    deny: Optional[DenyFn] = None,
    bus: Any = None,
    workspace: Optional[Path] = None,
    local_dir: Optional[Path] = None,
) -> PermissionGate:
    """The gate an :class:`~localharness.agent.loop.AgentLoop` builds when its caller passed none.

    CONTRACTS A4: "if a caller passes none, construct a fail-closed gate (no asker, mode from
    config) so nothing silently runs ungated". No asker means every ASK denies with
    :data:`NO_ASKER_REASON` and logs the fix once — a call site that forgot the gate degrades
    loudly instead of running unguarded, which is the inverse of the failure this phase exists
    to remove.
    """
    ws = Path(workspace) if workspace is not None else Path.cwd()
    return PermissionGate(
        boundary=derive_session_boundary(cwd=ws, local_dir=local_dir),
        workspace=ws,
        grants=GrantStore(),
        mode=getattr(permissions, "mode", DEFAULT_MODE),
        asker=None,
        channel_name="none",
        has_review_surface=False,
        deny=deny,
        settings=settings_from(permissions),
        bus=bus,
    )


def deny_fn_from(evaluator: Any, permissions: Any) -> DenyFn:
    """Adapt the shipped ``PermissionEvaluator`` to the gate's :data:`DenyFn` shape.

    The DENY tier must stay byte-identical to today's behaviour (CONTRACTS "keep
    ``permission_evaluator`` as the gate's DENY tier"), so nothing is reimplemented here: the
    call is simply re-packed into the ``ToolCall`` that ``evaluate`` expects. One adapter, used
    by the loop's fail-closed default gate and by ``cli/start_cmd``, so the two cannot drift.
    """

    def _deny(tool_name: str, params: dict) -> PermissionResult:
        from localharness.core.types import ToolCall

        return evaluator.evaluate(ToolCall(name=tool_name, arguments=params or {}), permissions)

    return _deny


def tool_meta_from_schema(schema: Any, *, mcp_server: Optional[str] = None) -> ToolMeta:
    """The verdict's view of a tool, read off its ``ToolSchema`` (A3's taxonomy, PRD §6).

    ``destructive`` and ``group`` come straight from the schema. MCP-ness is derived from the
    group prefix (:data:`MCP_GROUP_PREFIX`); ``mcp_server`` overrides it for a caller that
    already holds the server name.
    """
    group = getattr(schema, "group", None) or "other"
    server = mcp_server
    is_mcp = bool(server) or group.startswith(MCP_GROUP_PREFIX)
    if is_mcp and not server:
        server = group[len(MCP_GROUP_PREFIX):] or None
    return ToolMeta(
        destructive=bool(getattr(schema, "destructive", False)),
        group=group,
        is_mcp=is_mcp,
        mcp_server=server,
    )


def _iter_strings(obj: Any) -> list[str]:
    """Every string value inside a tool-call argument tree (same walk as ``permissions``)."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _iter_strings(v)]
    if isinstance(obj, (list, tuple)):
        return [s for v in obj for s in _iter_strings(v)]
    return []


def _matches_stored_deny(pattern: str, tool_name: str, params: dict) -> bool:
    """Does a stored "never here" pattern match this call?

    Same fnmatch semantics as ``agent/permissions.PermissionEvaluator`` — exact tool name, then
    the argument glob against every string argument — with one deliberate difference: the tool
    name is compared literally instead of through that module's ``[a-z_][a-z0-9_]*`` regex.
    The regex exists to parse patterns a human typed into config; these patterns are generated
    by :func:`deny_pattern_for`, and MCP tool names routinely carry capitals, so running them
    through the regex would silently drop a human's "never here" answer.
    """
    name, _, rest = pattern.partition("(")
    if name != tool_name:
        return False
    if not rest:
        return True
    glob = rest[:-1] if rest.endswith(")") else rest
    return any(fnmatch.fnmatch(v, glob) or fnmatch.fnmatch("./" + v, glob) for v in _iter_strings(params))


# ------------------------------------------------------------------- the gate

class PermissionGate:
    """One session's human-approval gate (PRD §3.5, §3.6).

    Holds the session-scoped state the pure verdict needs (boundary, workspace, grant store,
    mode) plus the channel's rendering of an ASK, and is passed BY REFERENCE to the loop, the
    REPL and every subagent so a mode switch or a fresh grant is seen everywhere at once.
    """

    def __init__(
        self,
        *,
        boundary: Optional[Path],
        workspace: Path,
        grants: GrantStore,
        mode: Mode = DEFAULT_MODE,
        asker: Optional[Asker] = None,
        channel_name: str = "none",
        ask_holds_dialog: bool = False,
        has_review_surface: bool = False,
        deny: Optional[DenyFn] = None,
        settings: Optional[GateSettings] = None,
        bus: Any = None,
    ) -> None:
        self.boundary = boundary
        self.workspace = Path(workspace)
        self.grants = grants
        self.mode: Mode = mode
        """Plain mutable attribute, shared with subagents (PRD §3.4). Set it through
        :meth:`set_mode` from anything a human drives."""
        self.asker = asker
        self.channel_name = channel_name
        self.ask_holds_dialog = ask_holds_dialog
        """PRD §3.5: True when the channel keeps the question open itself, so :meth:`check`
        awaits the answer with no deadline. Set from the channel by :meth:`attach_channel`."""
        self.has_review_surface = has_review_surface
        self.settings = settings or GateSettings()
        self.bus = bus
        self._config_deny = deny
        self._warned_cannot_ask = False

    def attach_channel(self, channel: Any) -> None:
        """Point the gate at the channel that will render its questions (PRD §3.5).

        Separate from ``__init__`` because the session's boundary, grant store and mode are known
        before the channel is built, and because the ACP adapter (phase B) attaches itself the
        same way once the client has told it whether it can show a diff. A channel that leaves
        ``can_ask`` False contributes no asker, which is exactly the fail-closed path — its
        ``ask_permission`` is never called, so a mismatch between the flag and the method cannot
        hang a turn. ``ask_holds_dialog`` comes from the channel for the same reason the other
        two do: whether a question can sit open until somebody answers is a fact about the
        surface it is drawn on, not about the call (PRD §3.5's Timeout column).
        """
        self.channel_name = getattr(channel, "channel_id", "none")
        self.has_review_surface = bool(getattr(channel, "has_review_surface", False))
        self.ask_holds_dialog = bool(getattr(channel, "ask_holds_dialog", False))
        self.asker = channel.ask_permission if getattr(channel, "can_ask", False) else None

    # ---------------------------------------------------------------- modes

    def set_mode(self, name: str, *, from_channel: bool = False) -> Mode:
        """Switch the session mode (PRD §3.4), validating the name.

        ``unattended`` is never settable from a channel command: it turns every ASK into an
        ALLOW, so a chat message must not be able to reach it — it is set explicitly in config
        by bench and scheduled jobs, and nowhere else. Raises ``ValueError`` with a message the
        channel shows verbatim.
        """
        if name not in MODE_STRICTNESS:
            known = ", ".join(sorted(MODE_STRICTNESS, key=lambda m: -MODE_STRICTNESS[m]))
            raise ValueError(f"unknown mode {name!r}; choose one of: {known}")
        if from_channel and name == "unattended":
            raise ValueError(
                "unattended mode allows every call without asking, so it cannot be set from a "
                "channel command — set `permissions.mode: unattended` in config instead"
            )
        self.mode = name  # type: ignore[assignment]
        return self.mode

    # ----------------------------------------------------------------- deny

    def _deny(
        self, tool_name: str, params: dict, config_deny: Optional[DenyFn] = None
    ) -> PermissionResult:
        """The DENY tier: the config deny patterns, then the workspace's "never here" answers.

        The config half is the existing evaluator, untouched — and it belongs to the CALLING
        agent, not to the session: a subagent may tighten its own deny list, so
        :meth:`check` passes that agent's deny function in and it wins over the gate's own.
        The stored half is re-read on every call (``GrantStore`` reloads), so a
        ``reject_always`` answered one minute ago wins the next minute without a restart.
        """
        deny = config_deny if config_deny is not None else self._config_deny
        if deny is not None:
            result = deny(tool_name, params)
            if result.denied:
                return result
        for pattern in self.grants.deny_patterns_for(self.workspace):
            if _matches_stored_deny(pattern, tool_name, params):
                return PermissionResult(
                    denied=True, reason=f"you answered 'never here' for this: {pattern}"
                )
        return PermissionResult(denied=False)

    def context(self, deny: Optional[DenyFn] = None) -> GateContext:
        """The :class:`GateContext` for one call, gathered from this session's state."""
        return GateContext(
            boundary=self.boundary,
            workspace=self.workspace,
            grants=self.grants.lookup,
            mode=self.mode,
            can_ask=self.asker is not None,
            has_review_surface=self.has_review_surface,
            deny=lambda name, params: self._deny(name, params, deny),
        )

    # ---------------------------------------------------------------- check

    async def check(
        self,
        tool_name: str,
        tool_params: dict,
        tool_meta: ToolMeta,
        *,
        agent_id: str,
        session_id: str,
        tool_timeout_s: Optional[float] = None,
        deny: Optional[DenyFn] = None,
    ) -> GateOutcome:
        """Decide one tool call, asking a human when the verdict says to (PRD §3.1, §3.5).

        ALLOW and DENY pass straight through. ASK either reaches a channel — publishing
        :class:`PermissionAsked`, awaiting the answer under a deadline, publishing
        :class:`PermissionResolved`, and making an "always" answer durable — or, when no channel
        can ask, fails closed with :data:`NO_ASKER_REASON`.

        ``deny`` is the CALLING agent's own deny tier (``AgentLoop._deny_fn``). It is per call
        rather than per gate because one shared gate serves an orchestrator and its subagents,
        and each of those resolved its own deny-pattern union from its own config layers.
        """
        result = evaluate(tool_name, tool_params, tool_meta, self.context(deny), self.settings)
        if result.verdict is Verdict.ALLOW:
            return GateOutcome(allowed=True, reason=result.reason)
        if result.verdict is Verdict.DENY:
            return GateOutcome(allowed=False, reason=result.reason)

        request = result.request
        assert request is not None  # evaluate() always attaches one to an ASK
        if self.asker is None:
            self._warn_cannot_ask(tool_name)
            return GateOutcome(allowed=False, reason=NO_ASKER_REASON)
        return await self._ask(
            request, agent_id=agent_id, session_id=session_id, tool_timeout_s=tool_timeout_s
        )

    def _warn_cannot_ask(self, tool_name: str) -> None:
        if self._warned_cannot_ask:
            return
        self._warned_cannot_ask = True
        log.warning(NO_ASKER_WARNING, tool_name, NO_ASKER_REASON)

    def _timeout_s(self, tool_timeout_s: Optional[float]) -> Optional[float]:
        """How long to wait for a human (PRD §3.5; see :data:`ASK_TIMEOUT_TOOL_MULTIPLE`).

        None means "no deadline": :meth:`_ask` then awaits the answer without
        ``asyncio.wait_for``. That is what a channel holding the dialog open gets — a terminal
        or Zed, per PRD §3.5's "Timeout: none" — because the deadline exists for a question
        posted where nobody may be looking, not for one a person is staring at.
        """
        if self.ask_holds_dialog:
            return None
        if self.settings.ask_timeout_s is not None:
            return self.settings.ask_timeout_s
        if tool_timeout_s is None:
            return None
        return tool_timeout_s * ASK_TIMEOUT_TOOL_MULTIPLE

    async def _publish(self, event: Any) -> None:
        """PRD §3.6: the two events ride the same bus — and so the same trace files — as
        ``Action``/``Observation``, which is what makes the ask rate measurable with no extra
        instrumentation. A gate built without a bus (a unit test) simply records nothing."""
        if self.bus is None:
            return
        await self.bus.publish(event)

    async def _ask(
        self,
        request: PermissionRequest,
        *,
        agent_id: str,
        session_id: str,
        tool_timeout_s: Optional[float],
    ) -> GateOutcome:
        # Imported here, not at module scope: `core/events` imports `agent/gate_types`, which
        # pulls in the `agent` package, which imports the loop, which imports this module. A
        # module-level import would close that cycle and break `import localharness.channels`.
        from localharness.core.events import PermissionAsked, PermissionResolved

        await self._publish(
            PermissionAsked(
                agent_id=agent_id,
                session_id=session_id,
                tool_name=request.tool_name,
                klass=request.klass,
                key=request.key if request.grantable else None,
                channel=self.channel_name,
            )
        )
        timeout = self._timeout_s(tool_timeout_s)
        started = time.monotonic()
        timed_out = False
        try:
            if timeout is None:
                decision = await self.asker(request)  # type: ignore[misc]
            else:
                decision = await asyncio.wait_for(self.asker(request), timeout)  # type: ignore[misc]
        except asyncio.TimeoutError:
            decision, timed_out = TIMEOUT_DECISION, True
        latency_ms = int((time.monotonic() - started) * MS_PER_SECOND)

        wrote_grant = self._remember(decision, request, session_id)
        await self._publish(
            PermissionResolved(
                agent_id=agent_id,
                session_id=session_id,
                tool_name=request.tool_name,
                klass=request.klass,
                key=request.key if request.grantable else None,
                decision=decision.kind,
                latency_ms=latency_ms,
                wrote_grant=wrote_grant,
            )
        )
        if decision.allowed:
            return GateOutcome(allowed=True, reason=f"approved by a human ({decision.kind})")
        if timed_out:
            return GateOutcome(allowed=False, reason=TIMEOUT_REASON.format(seconds=timeout or 0))
        return GateOutcome(allowed=False, reason=f"refused by a human ({decision.kind})")

    def _remember(
        self, decision: Decision, request: PermissionRequest, session_id: str
    ) -> bool:
        """Make an "always" answer durable (PRD §3.3). Returns whether anything was written.

        One request can carry several keys (``PermissionRequest.grant_keys``): a call that
        raised two first-exposure commands and an outside-the-boundary directory asked once, so
        an "always here" has to remember all three or the identical call asks again next turn
        (verification A, defect D1). ``grant_keys`` empty means the one primary key.

        An ungrantable request never writes a grant even if a channel hands back
        ``allow_always``: those requests ask every time by construction, so the answer is
        downgraded to ``allow_once`` here rather than trusted to every channel's UI to get
        right. Grantability is read off ``request.grantable`` and never off the class name —
        a normally-grantable class can arrive ungrantable when it has nothing rememberable to
        key on (``verdict.DYNAMIC_COMMAND_NAME_PREFIXES``: a shell segment whose command name is
        computed at runtime).

        ``reject_always`` is not downgraded for being ungrantable — a "never" answer is a
        tightening, and tightening is always allowed — but it IS downgraded when the request
        carries no key at all. With nothing to identify the call by, the only pattern that could
        be written is the bare tool name, which would ban every shell command forever; that is
        far more than the human answered, so nothing durable is written and the answer stands as
        a ``reject_once``.
        """
        if not decision.remembered or not request.key:
            return False
        if decision.kind == "allow_always":
            if not request.grantable:
                return False
            for klass, key in request.grant_keys or ((request.klass, request.key),):
                self.grants.add(
                    new_grant(
                        key=key,
                        klass=klass,
                        workspace=self.workspace,
                        channel=self.channel_name,
                        session_id=session_id,
                    )
                )
            return True
        self.grants.add_deny(
            self.workspace,
            deny_pattern_for(request.tool_name, request.klass, request.key),
            channel=self.channel_name,
            session_id=session_id,
        )
        return True
