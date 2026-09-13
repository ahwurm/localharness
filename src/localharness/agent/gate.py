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
import json
import logging
import time
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from localharness.agent.gate_types import (
    DEFAULT_MODE,
    MODE_STRICTNESS,
    PENDING_OBSERVATION,
    PENDING_REPEAT_OBSERVATION,
    UNGRANTABLE_OBSERVATION_SUFFIX,
    Asker,
    Decision,
    GateOutcome,
    GateSettings,
    Mode,
    PendingCall,
    PermissionRequest,
    ToolMeta,
    Verdict,
)
from localharness.agent.permissions import PermissionResult
from localharness.agent.verdict import DenyFn, GateContext, derive_boundary, evaluate
from localharness.config.grants import GrantStore, new_grant, new_refusal

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

STAGING_MODES: frozenset[str] = frozenset({"auto"})
"""The modes in which an ASK is PARKED for a human instead of put to one (owner ruling
2026-09-12: a blacklisted call must never block the agent loop in the default mode).

``auto`` only. It is the mode a person gets without choosing it, it asks about a handful of
irreversible things (:data:`~localharness.agent.gate_types.AUTO_BLACKLIST`), and its whole design
goal is "the thinnest interaction off of no interaction" — a question that stops the loop until
somebody walks back to the keyboard is the thickest interaction there is. ``guarded`` and
``trusted`` keep the blocking ask because being asked is what those modes ARE; ``read-only``
denies and ``unattended`` allows, so neither reaches an ask at all.

A frozenset rather than ``self.mode == "auto"`` at the call site so the rule is one named thing
to read, and widening it later is one diff."""

PENDING_STAGED_LOG = "pending #%d staged for a human on the %s channel: %s"
"""Logged at INFO when a call is parked. Same reason :data:`MODE_SET_FROM_CHANNEL_LOG` is
logged: a session that skipped a step because nobody answered should say so in its own log, not
only in a channel's scrollback that scrolls away."""

PENDING_ANSWERED_LOG = "pending #%d %s by a human on the %s channel: %s"
"""Logged at INFO on every :meth:`PermissionGate.approve` / :meth:`PermissionGate.deny`. The
second field is the verb ("approved"/"denied") so one grep finds both halves of the audit trail
the staging queue replaces the blocking prompt with."""

PENDING_APPROVED_VERB = "approved"
PENDING_DENIED_VERB = "denied"
"""The two verbs :data:`PENDING_ANSWERED_LOG` takes, named so the log and any renderer that
echoes them cannot drift."""

APPROVED_ONCE_REASON = "approved by a human from the channel (pending #{id})"
"""The allow reason for the ONE retry a :meth:`PermissionGate.approve` buys.

An approval is deliberately not a grant and not a mode change: the human said yes to the call
they read, so the key is consumed by the next matching call and the one after that stages
again. Nothing durable is written — ``/approve`` is the ungrantable tier's answer, and the
ungrantable tier asks every time by construction (:data:`~localharness.agent.gate_types.
UNGRANTABLE_CLASSES`)."""

NO_PENDING_ERROR = "nothing is pending"
PENDING_UNKNOWN_ERROR = "no pending call #{id}"
"""The two ``KeyError`` payloads of :meth:`PermissionGate.approve` / :meth:`PermissionGate.deny`.
A channel catches the KeyError and renders its own one-liner; these exist so a caller that
prints the exception still prints a sentence."""

MODE_SET_FROM_CHANNEL_LOG = "permission mode %s -> %s, set by a human on the %s channel"
"""Logged at INFO every time :meth:`PermissionGate.set_mode` is driven from a channel.

The audit trail that replaces the old refusal: a human may now switch their own session into
``unattended`` mid-turn (owner, 2026-09-11: "I can't swap my active localharness session to
unattended without exiting"), and a session that spent part of its life with the gate off should
say so in its own log rather than only in someone's memory."""

TIMEOUT_DECISION = Decision(kind="reject_once")
"""PRD §3.5: "deny on timeout", and §3.6: a timeout resolves as ``reject_once`` so
timeout-denies are countable as their own guardrail. Never ``reject_always`` — nobody answered,
so nothing durable may be written."""

TIMEOUT_REASON = "no answer within {seconds:g}s; denied"
"""What the model is told when nobody answered in time.

``:g`` rather than ``:.0f``: a sub-second deadline rendered as "no answer within 0s", which
reads as a bug in the gate rather than as a short timeout, and is the first thing a person
debugging a too-eager auto-deny would see. ``:g`` prints 0.2 as "0.2" and 30.0 as "30"."""

DENIED_OBSERVATION_PREFIX = "Permission denied: "
"""How the loop labels a gated call in the observation it hands back (``agent/loop.py``).

It is also what a channel matches on to show the human WHY a call was refused: the reason used
to reach only the model, so a person watching saw ``✗ write (exit 1): [DENIED]`` and nothing
else (verification A, defect D7). One definition, imported by both sides, so the label and the
matcher cannot drift apart."""

GATE_ERROR_REASON = "the permission gate could not decide this call; denied"
"""What the model is told when :func:`verdict.evaluate` itself raises (PRD §3.5's fail-closed
row, applied to the gate's own failure rather than to a channel that cannot ask).

`evaluate` is pure, but it parses model-supplied strings and realpaths model-supplied paths, so
"it cannot raise" is a claim about code that changes every time the classifier does. A null byte
in a write path (``Path("/tmp/x\\0y").resolve()`` raises ValueError, not OSError) escaped it and
crashed the whole turn — the gate failing OPEN in the worst possible way: not by allowing the
call, but by taking the agent down with it. A crash here is now a denial with a sentence the
model can re-plan against, and the traceback goes to the log where it can be fixed."""

GATE_ERROR_LOG = (
    "the permission gate raised while deciding %r; denying the call. This is a bug in the "
    "verdict — the call itself was not run."
)
"""Logged with the traceback (``log.exception``) every time :data:`GATE_ERROR_REASON` is
returned. Not once-per-session like :data:`NO_ASKER_WARNING`: each occurrence is a distinct
argument the verdict could not handle, and the arguments are the evidence."""

MS_PER_SECOND = 1000
"""Unit conversion for ``PermissionResolved.latency_ms`` — ``time.monotonic()`` returns seconds
and the event, like its siblings on the bus, reports milliseconds."""

SUBAGENT_DISPLAY_PREFIX = "[{agent_id}] "
"""How a subagent's ask is labelled in the one line every channel renders (PRD §3.5).

One gate serves the orchestrator and every subagent it dispatches (PRD §3.4), so the person
answering sees prompts from several agents on one surface with nothing to tell them apart — and
"approve `rm -rf build`" is a different question depending on which agent asked it. The
orchestrator's own asks are NOT prefixed: they are the common case, and a label on every line
would be noise that trains the eye to skip it.

It goes on ``display`` rather than being left to each renderer because ``display`` is the one
thing every channel is guaranteed to show; ``PermissionRequest.agent_id`` carries the same fact
structurally for a channel that wants to render it its own way."""

CALL_IDENTITY = "{tool_name}\x00{params}"
"""How :func:`call_identity` spells "the same call again" for the staging queue.

Deliberately NOT the grant key. A grant key is a CLASS of calls — the signature ``rm -rf`` is one
key however many directories it is pointed at — which is exactly right for "never ask about this
kind of command again" and exactly wrong here: a pending call is a specific command a human is
being shown and asked to answer, and folding ``rm -rf ~/notes`` and ``rm -rf ~/photos`` into one
queue entry would have them approve a command they never read. The whole tool call is the
identity, so a retry of the same call finds its own pending number and nothing else does.

It is also the only identity available where it is needed: the one-shot approval is consulted at
the TOP of :meth:`PermissionGate.check`, before ``evaluate`` has run and therefore before any
request, class or key exists."""

MCP_GROUP_PREFIX = "mcp/"
"""``tools/mcp.py:81`` gives every MCP tool the group ``mcp/<server>``. That group IS how the
registry knows the server name, so the ``ToolMeta`` builder reads it rather than taking a
second, drift-prone path through the registry."""


def call_identity(tool_name: str, params: dict) -> str:
    """One tool call's identity for the staging queue (:data:`CALL_IDENTITY`).

    ``sort_keys`` so two dicts that differ only in insertion order are one call; ``default=repr``
    so a parameter the model smuggled past JSON (a path object from a plugin tool, say) degrades
    to a stable string instead of raising inside the gate. A value that defeats even that falls
    back to ``repr`` of the whole mapping, which is still stable within one process — the cost of
    a wrong answer here is a duplicate queue entry, never a wrong permission.
    """
    try:
        rendered = json.dumps(params or {}, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        rendered = repr(params)
    return CALL_IDENTITY.format(tool_name=tool_name, params=rendered)


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
    except (OSError, ValueError):
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
        owner_agent_id: Optional[str] = None,
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
        self.owner_agent_id = owner_agent_id
        """Whose prompts are the UNLABELLED ones: the session's own agent, everything else being
        a subagent sharing this gate (PRD §3.4).

        Passed in by the session's entry point when it knows the orchestrator's id. When it does
        not, the FIRST agent to call :meth:`check` becomes the owner — the orchestrator runs
        before it can dispatch anything, so the first caller is the orchestrator by construction,
        and the alternative (labelling every prompt, the orchestrator's included) is noise that
        trains the eye to skip the label."""

        self._config_deny = deny
        self._warned_cannot_ask = False

        self.pending: dict[int, PendingCall] = {}
        """Calls parked for a human, oldest first (owner ruling 2026-09-12).

        Insertion-ordered, which is what makes "the oldest" — the default target of
        :meth:`approve` and :meth:`deny` — a plain ``next(iter(...))`` rather than a sort over a
        timestamp. Public because every surface that shows the queue (``/pending``, a channel's
        notice, a footer) reads it directly; nothing outside the gate WRITES it."""

        self._pending_seq = 0
        """The last number handed out. Never reset and never reused inside a session: a person
        who typed ``/approve 2`` must not find that 2 is now a different command."""

        self._pending_by_call: dict[str, int] = {}
        """:func:`call_identity` → pending id, so a model re-trying a staged call gets told its
        own number back instead of filling the queue with copies of one command."""

        self._approved_once: dict[str, int] = {}
        """:func:`call_identity` → the pending number it was approved as, for every call a human
        approved that has not re-run yet.

        One-shot tickets, not a grant store: see :data:`APPROVED_ONCE_REASON`. Consulted before
        ``evaluate``, and the ticket is spent whether or not the call then survives the deny
        tier, so an approval can never be stockpiled. It keeps the pending number because that
        number is the only handle the human, the model and the log share for this one call."""

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
        """Switch the session mode, validating the name.

        EVERY mode is settable from a channel, ``unattended`` included. It was refused there
        until v0.14.1, on the reasoning that a chat message must not be able to switch the gate
        off — but the person typing ``/mode`` in their own terminal IS the person the gate
        protects, and the owner met the rule the only way anyone does: "I can't swap my active
        localharness session to unattended without exiting" (2026-09-11). A decision a human
        makes about their own session is not an escalation; making them restart to make it was
        the bug. The switch is logged at INFO so a session that ran unattended says so in its
        log.

        The rule that stays is the one about a layer that is NOT a human: a project's config may
        only raise strictness, never lower it (``config/loader.MODE_STRICTNESS``), so a cloned
        repo still cannot put itself in ``auto`` or ``unattended``.

        Raises ``ValueError`` with a message the channel shows verbatim when the name is unknown.
        """
        if name not in MODE_STRICTNESS:
            known = ", ".join(sorted(MODE_STRICTNESS, key=lambda m: MODE_STRICTNESS[m]))
            raise ValueError(f"unknown mode {name!r}; choose one of: {known}")
        if from_channel and name != self.mode:
            log.info(MODE_SET_FROM_CHANNEL_LOG, self.mode, name, self.channel_name)
        self.mode = name  # type: ignore[assignment]
        return self.mode

    # ----------------------------------------------------------------- deny

    def _deny(
        self, tool_name: str, params: dict, config_deny: Optional[DenyFn] = None
    ) -> PermissionResult:
        """The DENY tier: the config deny patterns, unchanged.

        This tier belongs to the CALLING agent, not to the session: a subagent may tighten its
        own deny list, so :meth:`check` passes that agent's deny function in and it wins over
        the gate's own.

        A human's "never here" is NOT here. It is a structural refusal in the grant key space
        (``GrantStore.add_refusal``) consulted by ``verdict.evaluate``, not an fnmatch pattern
        over raw arguments — refusing the signature ``cp`` must not also ban ``scp``, ``cpio``
        and every command whose arguments merely contain "cp".

        The refusal reason carries :data:`~localharness.agent.gate_types.
        UNGRANTABLE_OBSERVATION_SUFFIX`, because this tier is the one nobody can lift from a
        channel: no ``/approve`` reaches it and no mode switch does either. The model could not
        tell that from the word "denied" and stopped to compose prose asking the human to run
        the command by hand — the stall the staging queue exists to end — so the sentence that
        ends it is attached here, at the un-approvable refusal itself.
        """
        deny = config_deny if config_deny is not None else self._config_deny
        if deny is not None:
            result = deny(tool_name, params)
            if result.denied:
                return PermissionResult(
                    denied=True, reason=result.reason + UNGRANTABLE_OBSERVATION_SUFFIX
                )
        return PermissionResult(denied=False)

    def context(self, deny: Optional[DenyFn] = None) -> GateContext:
        """The :class:`GateContext` for one call, gathered from this session's state."""
        return GateContext(
            boundary=self.boundary,
            workspace=self.workspace,
            grants=self.grants.lookup,
            refusals=self.grants.refused,
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
        call_id: Optional[str] = None,
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

        ``call_id`` is the tool call's own id, carried through to
        :attr:`PermissionRequest.call_id` so a channel can pair the question with the call it is
        about (ACP renders its dialog against a ``tool_call`` it already knows).
        """
        if self.owner_agent_id is None:
            self.owner_agent_id = agent_id  # see the attribute's docstring: first caller owns
        identity = call_identity(tool_name, tool_params)
        if identity in self._approved_once:
            # A human answered `/approve` for exactly this call. The ticket is spent here, before
            # the deny tier is consulted, so it can never be stockpiled — but the deny tier still
            # wins, because `permissions.deny_patterns` is the owner's own never-run list and no
            # channel answer is above it (it may also have CHANGED since the call was staged).
            pending_id = self._approved_once.pop(identity)
            if not self._deny(tool_name, tool_params, deny).denied:
                return GateOutcome(
                    allowed=True, reason=APPROVED_ONCE_REASON.format(id=pending_id)
                )
        try:
            result = evaluate(tool_name, tool_params, tool_meta, self.context(deny), self.settings)
        except Exception:  # noqa: BLE001 — a verdict that crashes must deny, never escape
            log.exception(GATE_ERROR_LOG, tool_name)
            return GateOutcome(allowed=False, reason=GATE_ERROR_REASON)
        if result.verdict is Verdict.ALLOW:
            return GateOutcome(allowed=True, reason=result.reason)
        if result.verdict is Verdict.DENY:
            return GateOutcome(allowed=False, reason=result.reason)

        request = result.request
        assert request is not None  # evaluate() always attaches one to an ASK
        if self.asker is None:
            # Checked BEFORE staging on purpose: staging is a promise that a human can be
            # reached, and a channel that cannot ask cannot keep it. A bench or cron run would
            # otherwise pile up a queue nobody will ever answer, silently, in place of the one
            # loud warning that names the fix.
            self._warn_cannot_ask(tool_name)
            return GateOutcome(allowed=False, reason=NO_ASKER_REASON)
        if self.mode in STAGING_MODES:
            return await self._stage(
                request, agent_id=agent_id, session_id=session_id, call_id=call_id,
                identity=identity,
            )
        return await self._ask(
            request, agent_id=agent_id, session_id=session_id, call_id=call_id,
            tool_timeout_s=tool_timeout_s,
        )

    # -------------------------------------------------------------- staging

    async def _stage(
        self,
        request: PermissionRequest,
        *,
        agent_id: str,
        session_id: str,
        call_id: Optional[str],
        identity: str,
    ) -> GateOutcome:
        """Park a blocked call for a human and let the turn carry on (owner ruling 2026-09-12).

        The asker is NOT called: nobody is asked, so nothing is awaited and the loop never
        stops. What the model gets back is a refusal it can route around
        (:data:`~localharness.agent.gate_types.PENDING_OBSERVATION`); what the human gets is a
        :class:`~localharness.core.events.PermissionStaged` on the bus, which is the only way a
        channel hears about this — the loop holds no channel handle and surfaces ordinary
        denials by writing a prefix onto the Observation, which is a fact about ONE tool result
        and cannot carry a queue.

        A repeat of a call already in the queue returns the SAME number and the shorter
        :data:`~localharness.agent.gate_types.PENDING_REPEAT_OBSERVATION`, so a model that
        retries fills the log rather than the queue.
        """
        from localharness.core.events import PermissionStaged

        staged = self.pending.get(self._pending_by_call.get(identity, 0))
        if staged is not None:
            return GateOutcome(
                allowed=False,
                reason=PENDING_REPEAT_OBSERVATION.format(id=staged.id),
                pending=staged,
            )
        self._pending_seq += 1
        staged = PendingCall(
            id=self._pending_seq,
            request=self._attributed(request, agent_id=agent_id, call_id=call_id),
            rendering=request.display,
            agent_label=self._agent_label(agent_id),
            session_id=session_id,
            created_at=time.time(),
        )
        self.pending[staged.id] = staged
        self._pending_by_call[identity] = staged.id
        log.info(PENDING_STAGED_LOG, staged.id, self.channel_name, staged.rendering)
        await self._publish(
            PermissionStaged(
                agent_id=agent_id,
                session_id=session_id,
                pending=staged,
                total=len(self.pending),
                channel=self.channel_name,
            )
        )
        return GateOutcome(
            allowed=False,
            reason=PENDING_OBSERVATION.format(id=staged.id, rendering=staged.rendering),
            pending=staged,
        )

    async def approve(self, n: Optional[int] = None) -> PendingCall:
        """A human says yes to a parked call. ``None`` answers the oldest one.

        The approval buys exactly ONE run of that same call (:data:`APPROVED_ONCE_REASON`): the
        human said yes to the command they read, not to its class, and the ungrantable tier this
        queue is made of asks every time by construction. Nothing durable is written, so a
        second identical call stages again as a new number.

        It does not run anything by itself. The call is re-issued by the MODEL, nudged by
        whatever surface took the approval (``cli/repl`` pushes a user nudge into the running
        turn, or submits one as a fresh turn when the session is idle) — which keeps the agent
        loop the only thing that ever dispatches a tool.

        Raises ``KeyError`` when ``n`` names no parked call, or when nothing is parked at all.
        """
        return await self._answer(n, approved=True)

    async def deny(self, n: Optional[int] = None) -> PendingCall:
        """A human says no to a parked call. ``None`` answers the oldest one.

        Nothing durable is written here either: this is a ``reject_once``, not the
        ``reject_always`` that a blocking prompt can turn into a stored refusal. A queue entry
        is a single command somebody looked at and declined; turning that into a permanent
        "never here" is a bigger answer than the one they gave, and ``grants.yaml`` is where a
        permanent one belongs.

        Raises ``KeyError`` on an unknown id, exactly as :meth:`approve` does.
        """
        return await self._answer(n, approved=False)

    async def _answer(self, n: Optional[int], *, approved: bool) -> PendingCall:
        """The shared half of :meth:`approve` and :meth:`deny`: pop, log, publish."""
        from localharness.core.events import PermissionResolved

        if n is None:
            if not self.pending:
                raise KeyError(NO_PENDING_ERROR)
            n = next(iter(self.pending))  # insertion order: the oldest still waiting
        if n not in self.pending:
            raise KeyError(PENDING_UNKNOWN_ERROR.format(id=n))
        staged = self.pending.pop(n)
        # The identity index is scanned rather than mirrored in a second dict: the queue is
        # human-scale (a handful of entries a person is expected to read), and one index that
        # cannot fall out of step with `pending` is worth more here than the lookup.
        for identity, pending_id in list(self._pending_by_call.items()):
            if pending_id == n:
                del self._pending_by_call[identity]
                if approved:
                    self._approved_once[identity] = n
        log.info(
            PENDING_ANSWERED_LOG,
            staged.id,
            PENDING_APPROVED_VERB if approved else PENDING_DENIED_VERB,
            self.channel_name,
            staged.rendering,
        )
        request = staged.request
        await self._publish(
            PermissionResolved(
                agent_id=request.agent_id or "",
                session_id=staged.session_id,
                tool_name=request.tool_name,
                klass=request.klass,
                key=request.key if request.grantable else None,
                decision="allow_once" if approved else "reject_once",
                latency_ms=int((time.time() - staged.created_at) * MS_PER_SECOND),
                wrote_grant=False,
            )
        )
        return staged

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
        call_id: Optional[str] = None,
        tool_timeout_s: Optional[float],
    ) -> GateOutcome:
        # Imported here, not at module scope: `core/events` imports `agent/gate_types`, which
        # pulls in the `agent` package, which imports the loop, which imports this module. A
        # module-level import would close that cycle and break `import localharness.channels`.
        from localharness.core.events import CANCELLED_RESOLUTION, PermissionAsked

        request = self._attributed(request, agent_id=agent_id, call_id=call_id)
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
        except asyncio.CancelledError:
            # The turn went away mid-prompt (Ctrl-C, a channel closing, a turn timeout). The ask
            # is still on the bus, so it gets its answer — nobody chose it, hence `cancelled`,
            # not one of the four DecisionKinds. Without this, a cancelled prompt left the
            # Asked/Resolved pair open forever and read in a trace like a prompt still waiting.
            # The publish is best-effort: nothing may stand between a cancellation and its
            # re-raise.
            with suppress(Exception):
                await self._resolved(
                    request, agent_id=agent_id, session_id=session_id,
                    decision=CANCELLED_RESOLUTION, started=started, wrote_grant=False,
                )
            raise

        wrote_grant = self._remember(decision, request, session_id)
        await self._resolved(
            request, agent_id=agent_id, session_id=session_id,
            decision=decision.kind, started=started, wrote_grant=wrote_grant,
        )
        if decision.allowed:
            return GateOutcome(allowed=True, reason=f"approved by a human ({decision.kind})")
        if timed_out:
            return GateOutcome(allowed=False, reason=TIMEOUT_REASON.format(seconds=timeout or 0))
        return GateOutcome(allowed=False, reason=f"refused by a human ({decision.kind})")

    def _attributed(
        self, request: PermissionRequest, *, agent_id: str, call_id: Optional[str]
    ) -> PermissionRequest:
        """Stamp WHO is asking and WHICH call onto the request the channels will see.

        The verdict is pure and knows nothing about sessions, so it builds the request without
        either; they are facts about the call site, and this is the call site. A subagent's ask
        also gets :data:`SUBAGENT_DISPLAY_PREFIX` on ``display``, because one gate serves the
        orchestrator and every child it dispatches (PRD §3.4) and the person answering sees them
        all on one surface.
        """
        return replace(
            request,
            agent_id=agent_id,
            call_id=call_id,
            display=self._agent_label(agent_id) + request.display,
        )

    def _agent_label(self, agent_id: str) -> str:
        """:data:`SUBAGENT_DISPLAY_PREFIX` for a subagent's call, ``""`` for the session's own.

        Read by :meth:`_attributed`, which glues it onto ``display``, and carried separately on
        :class:`~localharness.agent.gate_types.PendingCall` so a queue renderer can lay the two
        out its own way instead of splitting a string back apart.
        """
        if agent_id and agent_id != self.owner_agent_id:
            return SUBAGENT_DISPLAY_PREFIX.format(agent_id=agent_id)
        return ""

    async def _resolved(
        self,
        request: PermissionRequest,
        *,
        agent_id: str,
        session_id: str,
        decision: str,
        started: float,
        wrote_grant: bool,
    ) -> None:
        """Publish the answer half of the pair (PRD §3.6). One builder, three exits.

        Every path out of :meth:`_ask` that published a :class:`PermissionAsked` has to publish
        this, so the two events are built in exactly one place and a new exit cannot forget a
        field.
        """
        from localharness.core.events import PermissionResolved

        await self._publish(
            PermissionResolved(
                agent_id=agent_id,
                session_id=session_id,
                tool_name=request.tool_name,
                klass=request.klass,
                key=request.key if request.grantable else None,
                decision=decision,
                latency_ms=int((time.monotonic() - started) * MS_PER_SECOND),
                wrote_grant=wrote_grant,
            )
        )

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

        ``reject_always`` writes a REFUSAL — a negative grant in the same key space, filed under
        the very keys the prompt offered (PRD §3.3). A grantable request refuses every key it
        asked about, the mirror of an "always here"; an ungrantable one refuses only its primary
        key, because the rest were bundled into an ask the human answered about that one thing.
        It is not downgraded for being ungrantable — a "never" answer is a tightening, and
        tightening is always allowed — but it IS downgraded when the request carries no key at
        all. With nothing to identify the call by, the only thing that could be written is the
        bare tool name, which would ban every shell command forever; that is far more than the
        human answered, so nothing durable is written and the answer stands as a
        ``reject_once``.

        The earlier shape wrote an fnmatch DENY pattern derived from the key
        (``bash_exec(*cp*)``), which from then on also blocked ``scp``, ``cpio`` and any command
        whose arguments contained "cp", with nothing in the UI able to undo it. The refusal keeps
        the spirit of §3.3 — a "never" wins over any later grant and asks no more — while
        denying exactly what the human refused.
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
        refused = (
            request.grant_keys or ((request.klass, request.key),)
            if request.grantable
            else ((request.klass, request.key),)
        )
        for klass, key in refused:
            self.grants.add_refusal(
                new_refusal(
                    key=key,
                    klass=klass,
                    workspace=self.workspace,
                    channel=self.channel_name,
                    session_id=session_id,
                )
            )
        return True
