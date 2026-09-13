"""The Agent Client Protocol adapter — localharness inside Zed's agent panel (PRD §4).

`.planning/2026-09-11-zed-acp-and-permission-spine-prd.md` §4 is the design; §3.5's Zed row is
the rendering contract. ACP is JSON-RPC over stdin/stdout and is the ONE mechanism Zed
recognizes an outside agent through, so this file is the whole surface a Zed user touches.

One object plays two roles on purpose. `AcpChannel` is a `ChannelAdapter` — the same seam
Discord uses, so tool calls, failures and the permission ask reach Zed through the mechanisms
that already exist — and it is simultaneously the ACP `Agent` (the structural Protocol in
`acp/interfaces.py`: `initialize`, `new_session`, `prompt`, `cancel`, `set_session_mode`,
`on_connect`). Splitting them would have meant two objects holding references to each other's
private state — the connection, the session id, the gate, the running turn — with no boundary
worth defending between them.

Lifecycle, and why it is shaped this way (PRD §4, critic finding 10):

1. `initialize` must return immediately, so nothing is built here beyond remembering what the
   client can do. Whether the client offers BOTH `fs/read_text_file` and `fs/write_text_file` is
   what decides `has_review_surface`, and therefore whether an in-workspace edit asks at all
   (§3.1 choice 2). The pair is indivisible: an editor-backed write is a read-modify-write.
2. `new_session(cwd)` must also return immediately. It derives the workspace boundary and runs
   v0.13 workspace discovery — including the one-time trust question, put to the human through
   `session/request_permission` because Zed is not a terminal and the phase-39 rule would
   otherwise leave an outside `.localharness/` silently ignored forever.
3. The model server and the `AgentLoop` come up on the FIRST prompt, where a session id exists
   and progress can stream as `agent_message_chunk`s. Bringing them up in `new_session` would
   mean a Zed user staring at a spinner with no channel to explain it.

Bring-up itself is `cli/start_cmd._start_async` with `channel_mode="acp"` — the SAME function
that builds a terminal or a Discord session, not a third copy of it. Everything a real session
has (memory, MCP servers, plugins, the subagent fleet, the context manager, the gate) is there
because that one path built it.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import acp
from acp.helpers import (
    start_tool_call,
    text_block,
    tool_content,
    update_agent_message_text,
    update_current_mode,
    update_tool_call,
)
from acp.schema import (
    AgentCapabilities,
    Implementation,
    InitializeResponse,
    NewSessionResponse,
    PermissionOption,
    PromptResponse,
    SessionMode,
    SessionModeState,
    SetSessionModeResponse,
    ToolCallLocation,
    ToolCallUpdate,
)

from localharness.channels.base import ChannelAdapter, sanitize_for_display
from localharness.channels.errors import NotInteractiveError
from localharness.core.events import (
    PENDING_APPROVED_NUDGE,
    PENDING_DENIED_NUDGE,
    Action,
    Escalation,
    Observation,
    ParseFailed,
    PermissionResolved,
    PermissionStaged,
    TaskComplete,
    TurnFailed,
)

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ constants

AGENT_NAME = "localharness"
"""What Zed shows as the agent's implementation name in the handshake (`agent_info`)."""

ACP_MODES: tuple[SessionMode, ...] = (
    SessionMode(
        id="auto",
        name="Auto",
        description=(
            "Trust this project once, then stay out of the way. A dialog appears only for the "
            "blacklist: a delete aimed outside the project, a force push, reset --hard, "
            "clean -f, sudo, curl | sh, dd/mkfs/shred, a write to your secret stores or a system "
            "directory, and a write to .git or .localharness. Nothing is remembered."
        ),
    ),
    SessionMode(
        id="guarded",
        name="Guarded",
        description=(
            "Ask once about each new thing and remember the answer: anything that leaves this "
            "project folder, touches a protected path, or runs a command this workspace has "
            "never allowed. Reads never ask."
        ),
    ),
    SessionMode(
        id="trusted",
        name="Trusted",
        description=(
            "Like Auto, but a destructive command whose target is inside this project asks too."
        ),
    ),
    SessionMode(
        id="unattended",
        name="Unattended",
        description=(
            "Never ask: every question becomes a yes and only your deny patterns stop anything. "
            "How the harness behaved before v0.14 — pick it for a session you are watching."
        ),
    ),
    SessionMode(
        id="read-only",
        name="Read only",
        description=(
            "Look, don't touch: writes, edits, code execution and non-read-only shell commands "
            "are refused with an explanation the model can re-plan against."
        ),
    ),
)
"""The modes Zed's mode picker offers, with the human names and descriptions from PRD §3.4.

`auto` is first because it is the default a session starts in (v0.14.1): the v0.14.0 default,
`guarded`, asked during ordinary work often enough that people stopped reading the questions, so
the ask-once classes are silent in `auto` and only the genuinely dangerous ones still stop you.
`guarded` stays on the picker for anyone who wants the ask-once-and-remember behaviour back.

`unattended` is on the picker from v0.14.1 (owner ruling: `/mode unattended` and `/mode auto` are
accepted from the terminal, Discord and Zed alike). It turns every ASK into an ALLOW, which a
person watching a session may decide out loud; a scheduled job with nobody watching still writes
`permissions.mode: unattended` in config. `PermissionGate.set_mode(from_channel=True)` is the layer
that has to agree — an entry advertised here and refused there would fail on the first prompt."""

ACP_MODE_IDS: frozenset[str] = frozenset(m.id for m in ACP_MODES)
"""The settable set, derived from :data:`ACP_MODES` so the advertised list and the accepted list
cannot drift. Used to validate `session/set_mode` BEFORE the gate exists (a mode can be picked
before the first prompt); once it exists, the gate's own validation is what answers."""

TOOL_GROUP_TO_ACP_KIND: dict[str, str] = {
    "fs.read": "read",
    "fs.write": "edit",
    "shell": "execute",
    "code": "execute",
    "web": "fetch",
    "delegate": "other",
    "memory": "think",
}
"""`ToolSchema.group` (the v0.14 exposure taxonomy, PRD §6) → ACP's fixed `ToolCallKind` set.

The mapping is stated once, here, rather than tool-by-tool: groups are how the permission gate
already classifies a tool it does not know by name, so the icon Zed draws and the verdict the
gate reaches are answering the same question from the same fact. `delegate` maps to `other`
because a subagent is not one of ACP's kinds and `think` would misread a child that writes
files; `mcp/<server>` and anything unclassified fall through to :data:`ACP_KIND_DEFAULT`."""

ACP_KIND_DEFAULT = "other"
"""Every group ACP has no word for — `mcp/<server>`, and the `other` default itself."""

PATH_PARAM_NAMES: tuple[str, ...] = ("path", "file_path", "filepath")
"""Parameter names the builtin filesystem tools use for their target (`tools/builtin/*.py`).
A `tool_call`'s `locations` is what lets Zed jump to the file the agent is working on, so the
lookup is by the names the schemas actually declare rather than a guess per tool."""

EDITOR_FILE_IO_TOOLS: tuple[str, ...] = ("read", "write", "edit")
"""The builtins whose file I/O is routed through the editor when the client advertises `fs`
(PRD §4, "Edits through the editor"). Named rather than discovered by group because the routing
is only correct for tools whose whole job is one text file: `glob`/`grep` walk trees, and
`bash_exec` runs a process that ACP cannot intercept."""

PROVIDER_BRINGUP_NOTICE_AFTER_S = 2.0
"""How long the first prompt may sit silent before the adapter says what it is doing.

PRD §4 / critic finding 10: bring-up streams status "if the daemon is already up, the normal
case on the Spark, this costs nothing". Two seconds is the honest divider between those two
worlds — an attached provider answers its probe well inside it, so a warm session shows no
status line at all, while a cold model server never finishes inside it, so a user who is about
to wait is told inside one breath rather than after one."""

PROVIDER_BRINGUP_NOTICE_EVERY_S = 15.0
"""The repeat interval for the status line once bring-up is known to be slow. Long enough that a
two-minute vLLM load produces a handful of lines rather than a wall, short enough that the panel
never looks frozen."""

BRINGUP_STATUS = "Starting the session — {seconds:.0f}s so far…"
"""The status line the first prompt streams while the session is being built.

It says "the session", not "the model server", because this wait covers the WHOLE of
`_start_async` — config load, memory open, MCP servers, the subagent fleet — and only
sometimes a cold model server. Naming the model server was wrong on a warm box, where the
provider answers its probe instantly and the seconds being counted are somebody else's."""

BRINGUP_FAILED = "The session could not be started: {error}"
"""What the user sees when `_start_async` refuses (no config, an unreachable provider, a model
the server does not serve). The real reason, verbatim — a Zed panel has no stderr."""

BRINGUP_EXITED = (
    "The session could not be started — localharness stopped during startup (exit {code}). "
    "The reason was printed to the agent server's log: in Zed, open the agent panel's menu and "
    "choose 'View Server Logs'. Common causes: no config yet (`localharness init`), a config the "
    "current version rejects, or the model server not answering."
)
"""`typer.Exit` carries only a number, and startup's real explanation goes to stderr through the
rich consoles `start_cmd` prints with. Repeating "1" into the panel would be a message that says
nothing, so this names where the sentence actually is and what usually causes it."""

PERMISSION_OPTION_NAMES: dict[str, str] = {
    "allow_once": "Allow once",
    "allow_always": "Always allow in this workspace",
    "reject_once": "No",
    "reject_always": "Never allow in this workspace",
}
"""The four ACP `PermissionOptionKind` values as button labels (PRD §3.5, Zed row). The option
id IS the kind, so the answer maps back to a `Decision` with no second table to keep in sync."""

GRANTABLE_OPTION_KINDS: tuple[str, ...] = (
    "allow_once", "allow_always", "reject_once", "reject_always",
)
UNGRANTABLE_OPTION_KINDS: tuple[str, ...] = ("allow_once", "reject_once")
"""PRD §3.5: a request that cannot be remembered offers only the `_once` pair — an "always"
button on a class that asks every time by construction would be a lie."""

WORKSPACE_TRUST_KLASS = "workspace-trust"
"""The one ask whose answer is permanent (`cli/session_trust`), and the one that must not be
drawn with the ungrantable buttons."""

TRUST_OPTION_NAMES: dict[str, str] = {
    "allow_always": "Trust this workspace",
    "reject_once": "Not now",
}
TRUST_OPTION_KINDS: tuple[str, ...] = ("allow_always", "reject_once")
"""The workspace-trust question's own two buttons.

It arrives `grantable=False` — there is no "once" and no "always" to tell apart, because yes IS
always — and the generic ungrantable pair would then label a permanent decision "Allow once",
which is the opposite of what it does. `allow_always` is the kind that reads as permanent in
Zed's UI; the gate writes no grant for an ungrantable request whatever kind comes back, and
`session_trust` records the trust itself, so the label is the only thing this changes."""

PERMISSION_FALLBACK_DECISION = "reject_once"
"""Fail closed (SECURITY.md "deny on doubt") when there is nowhere to put the question, and what
a dismissed dialog means: ACP's `DeniedOutcome{outcome:"cancelled"}` is the user hitting Escape,
which is a refusal of this call and nothing more."""

TRUST_TOOL_CALL_TITLE = "Trust this workspace?"
"""The title on the trust question's `request_permission` payload. The question itself
(`cli/workspace.TRUST_QUESTION`) is the body; ACP has no dialog primitive other than a permission
request, so the one-time trust dialog rides on the same mechanism as every other ask (PRD §3.5:
"the existing workspace-trust dialog becomes the first client of ask_permission").

It said "Load workspace configuration?" until v0.14.1, when the config-layer question and the
session-permission question became one decision and one record (owner ruling 2026-09-11). A
title naming only the config half would be describing half of what a yes now does."""

PENDING_NOTICE = (
    "⏸ needs you  #{id}  {rendering}   ({total} pending) — send /approve {id} to run it or "
    "/deny {id} to skip"
)
"""The one line a Zed user gets when `auto` PARKS a call instead of asking (owner ruling
2026-09-12).

It rides as ordinary agent text, not as a `session/request_permission`: a dialog is exactly what
staging exists to remove, and ACP has no notice primitive between the two. So the panel shows one
inline line, the turn carries on underneath it, and nothing blocks.

Worded differently from `channels/base.PENDING_NOTICE_LINE` for one reason: Zed gives this agent
no command surface of its own, so the only way a person can answer is to TYPE into the same
message box the turn came from. "send /approve 3" says that; a bare "/approve 3" in parentheses
reads, in an editor panel, like a button somebody is meant to find."""

PENDING_LIST_HEADER = "{total} waiting on you:"
PENDING_LIST_LINE = "  #{id}  {label}{rendering}   (/approve {id} · /deny {id})"
NOTHING_PENDING = "Nothing pending."
"""`/pending`: the queue `auto` parked, one line each — the same one-line rendering every other
surface shows, so the number a person types is the number they read."""

PENDING_NEEDS_A_NUMBER = (
    "{verb} takes a pending number, e.g. {verb} 1 — or {verb} on its own for the oldest."
)
PENDING_UNKNOWN = "No pending #{id}. Send /pending to see what is waiting."
"""A number nobody parked. One line and no turn: a typo must not become a prompt the model then
answers as if it were a request."""

PENDING_RESOLVED_LINES: dict[bool, str] = {
    True: "✅ approved #{id}  {rendering} — it runs when the model re-issues it",
    False: "❌ skipped #{id}  {rendering}",
}
"""The outcome of one parked call, written into the transcript where the notice was.

Keyed by "was it approved" so the two halves of the pair cannot drift apart. The approval line
says "it runs when the model re-issues it" rather than "ran": `PermissionGate.approve` records an
answer and dispatches nothing — the call is re-issued by the MODEL, which may have finished the
task another way by the time a human gets round to the queue."""


PENDING_COMMAND_VERBS: frozenset[str] = frozenset({"/pending", "/approve", "/deny"})
"""The three commands a Zed prompt can be instead of a turn (:func:`_pending_command`)."""

ONE_THREAD_PER_PROCESS = (
    "This localharness process is already serving a thread in {current}. Start a second "
    "LocalHarness agent server for {requested} — one agent process serves one project folder and "
    "one thread."
)
"""v1 limitation, stated as an error rather than silently answering for the wrong thread.

ACP itself allows many sessions on one connection ("Each connection can support several
concurrent sessions" — agentclientprotocol.com/overview/architecture), and this adapter does not:
the harness session derives its boundary, config layer, memory and state directory from ONE
directory that the process changes into, and everything downstream of `serve()` — the loop, the
gate, the running turn — is that one session. Accepting a second `session/new` would tag one
thread's `fs/*` calls, permission dialogs and `session/update`s with the other thread's id, and
let `session/cancel` stop the wrong turn. Refusing is the honest version of what is built.
Named in docs/zed.md's "not yet" list."""

UNKNOWN_SESSION = (
    "Unknown session {requested}. This localharness process is serving {current} and no other "
    "session — start a second LocalHarness agent server for another thread."
)
"""A `session/prompt` or `session/set_mode` for an id this process never issued, or issued and
replaced. It can only be a client that believes it has two threads here; answering it would run
the turn on the one session this process does have, under another thread's id."""

SESSION_TEARDOWN_TIMEOUT_S = 10.0
"""How long `aclose` waits for the harness session's ordered shutdown (MCP servers, memory
consolidation, the LLM client) once the editor has gone away. Bounded because the user is no
longer watching: a wedged MCP server must not keep the process alive, and the shutdown itself is
best-effort in `start_cmd`'s `finally`. Ten seconds is the same order as the MCP client's own
shutdown budget and long enough for a WAL checkpoint."""

TOOL_TITLE_ARG_CHARS = 80
"""How much of a tool call's leading argument goes into the `tool_call` title Zed renders on one
row. Long enough for a path or a short command, short enough not to wrap the panel."""

MCP_SERVERS_NOT_CONNECTED_LOG = "acp session/new: %d editor-passed MCP server(s) not connected: %s"
"""Logged to stderr (the agent-server log Zed shows under 'View Server Logs') the moment the
editor hands over servers this version will not start — named one by one, so the line answers
"which of my servers is missing?" without a second round trip."""

MCP_SERVERS_NOT_CONNECTED_NOTICE = (
    "Note: this editor passed {count} MCP server(s) with the thread ({servers}), and this "
    "version of LocalHarness does not connect them — none of their tools are available here. "
    "LocalHarness starts the MCP servers declared in its own config (`tools.mcp_servers` in your "
    "agent or org YAML); a server listed there is available in Zed, the terminal and Discord "
    "alike."
)
"""The same fact in the panel, once, on the first prompt of the session.

The stderr line alone would be invisible: a Zed user watches the panel, not the server log, and
a model that never sees the tools cannot explain their absence either. So the person is told in
the one place they are looking, and told where MCP servers actually come from — the drop is a
v1 limitation (docs/zed.md's "not yet" list), not a failure to report."""

ATTACHMENT_PLACEHOLDER = "[attachment: {label} — not read by this agent]"
"""What a non-text prompt block becomes in the task text (D2).

Zed's @-file-mentions, images and pasted resources arrive as blocks with no `.text`, and this
adapter reads text only. Dropping them silently is the worst of the three options: the user
believes the file was read, and the model answers about a mention it never saw. A placeholder
line makes the gap visible to BOTH — the model can say "I cannot see that file, paste it or tell
me the path" instead of inventing its contents."""

ATTACHMENT_UNNAMED_LABEL = "unnamed"
"""The label when a block carries no name, uri or type at all — nothing is still worth a line."""


def split_display(display: str) -> tuple[str, Optional[str]]:
    """A `PermissionRequest.display` split into a dialog title and its body.

    One command can now carry several reasons at once, and the gate renders them as a short
    multi-line `display`. ACP's `ToolCallUpdate.title` is a one-row string — a client showing
    embedded newlines in it would either clip everything after the first line or blow up the
    dialog's header — so the first line becomes the title and the rest becomes a text content
    block underneath it, which is where the reasons belong anyway. A single-line display is
    unchanged and gets no body.
    """
    head, _, rest = (display or "").partition("\n")
    tail = rest.strip("\n")
    return head, tail or None


def acp_kind_for_group(group: str) -> str:
    """`ToolSchema.group` → ACP `ToolCallKind` (:data:`TOOL_GROUP_TO_ACP_KIND`, PRD §4)."""
    return TOOL_GROUP_TO_ACP_KIND.get(group or "", ACP_KIND_DEFAULT)


def _first_path(params: dict[str, Any]) -> Optional[str]:
    """The filesystem target of a tool call, if it declares one (:data:`PATH_PARAM_NAMES`)."""
    for name in PATH_PARAM_NAMES:
        value = (params or {}).get(name)
        if isinstance(value, str) and value:
            return value
    return None


def _tool_title(tool_name: str, params: dict[str, Any]) -> str:
    """The one-line title on Zed's tool-call row: the tool and what it is pointed at."""
    params = params or {}
    lead = _first_path(params)
    if lead is None:
        for value in params.values():
            if isinstance(value, str) and value.strip():
                lead = value.strip()
                break
    if not lead:
        return tool_name
    flat = " ".join(lead.split())
    if len(flat) > TOOL_TITLE_ARG_CHARS:
        flat = flat[: TOOL_TITLE_ARG_CHARS - 1] + "…"
    return f"{tool_name}: {flat}"


def _pending_command(text: str) -> tuple[str, str]:
    """A prompt that is a queue command → (verb, argument); anything else → ("", "").

    WHOLE-message and nothing else: "can you /approve 1 while you're there" is a sentence for the
    model, not a command, and reading a verb out of the middle of a prompt would silently swallow
    a turn. Case-insensitive because the box a Zed user types into capitalizes nothing for them.
    """
    parts = (text or "").strip().split()
    if not parts or len(parts) > 2 or parts[0].lower() not in PENDING_COMMAND_VERBS:
        return "", ""
    return parts[0].lower(), parts[1] if len(parts) > 1 else ""


class AcpChannel(ChannelAdapter):
    """localharness as an ACP agent, and as the channel that renders its session (PRD §4).

    See the module docstring for why one object is both. Everything with a wire format lives
    here; nothing here decides a permission — `ask_permission` renders the question the
    `PermissionGate` already decided to ask, and the answer goes straight back as a `Decision`.
    """

    channel_id = "acp"

    can_ask = True
    """PRD §3.5: Zed renders an ASK as its own permission dialog, held open with no timeout."""

    ask_holds_dialog = True
    """The client holds the question open, so the gate must not put a deadline on it (PRD §3.5,
    Zed row: "Timeout: none"). A timeout here would turn a user who stepped away from their
    editor into a `reject_once` that also looks, in the ask-rate report, like a channel that
    could not reach anybody. Discord is the opposite case — a message nobody reacts to has to
    expire — which is why this is a per-channel flag rather than a rule in the gate."""

    def __init__(self, *, config_dir: Optional[str] = None) -> None:
        # bus=None on purpose: the bus does not exist until `_start_async` builds the session on
        # the first prompt, and `initialize` has to answer long before that. `serve()` is the one
        # place it is set, so there is no window where a half-wired channel is subscribed.
        super().__init__(bus=None, config={})  # type: ignore[arg-type]
        self._config_dir = config_dir

        self._conn: Any = None
        self._client_can_read = False
        self._client_can_write = False

        self._session_id: Optional[str] = None
        self._cwd: Optional[Path] = None
        self._boundary: Optional[Path] = None
        self._workspace: Optional[Path] = None

        self._agent_loop: Any = None
        self._gate: Any = None
        self._registry: Any = None
        self._pending_mode: Optional[str] = None

        self._session_task: Optional[asyncio.Task] = None
        self._session_error: Optional[str] = None
        self._ready = asyncio.Event()
        self._closed = asyncio.Event()

        self._turn_task: Optional[asyncio.Task] = None
        self._streamed_this_turn = False
        # call id -> (tool_name, params) for every tool call Zed has a row for and no result yet.
        # A map rather than one slot: subagents run concurrently on this bus, so two identical
        # calls can be in flight at once (D4).
        self._pending_calls: dict[str, tuple[str, dict]] = {}
        # Every call `auto` parked this session, by pending id. Kept here and not read off the
        # gate because a resolution REMOVES it from `gate.pending` before the event is published,
        # and the outcome line still has to name the command that was answered.
        self._staged: dict[int, Any] = {}
        self._announced: set[int] = set()
        """Pending ids whose outcome line is already in the transcript. The bus handler and the
        slash command both answer the same call — one from `PermissionResolved`, one from the
        return value of `gate.approve` — so the line is written by whichever arrives first and
        never twice."""
        self._pending_notice: Optional[str] = None
        self._handles: list[Any] = []

    # ------------------------------------------------------------ ACP: agent side

    def on_connect(self, conn: Any) -> None:
        """Store the connection the SDK hands back (sync, called once inside the connection's
        constructor — `acp/agent/connection.py`). It is how every `session/update`,
        `session/request_permission` and `fs/*` call reaches the client."""
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any = None,
        client_info: Any = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        """The handshake — fast, and nothing but the handshake (PRD §4).

        `auth_methods=[]` says there is nothing to log in to; the ACP registry's CI requires the
        field to be present, and a local harness has no account. `load_session=False` is honest:
        session ids are fresh per start (`start_cmd.py`), so we cannot resume one.

        The client's `fs` capabilities are the load-bearing part, and they count only as a PAIR:
        `read_text_file` AND `write_text_file` together give this session a review surface (Zed's
        diff pane, accept/reject per hunk), which is what lets an in-workspace edit run without
        asking at all (PRD §3.1 choice 2, critic finding 11). Write alone is not a review surface
        but a data-loss hazard — an append or an edit has to READ the buffer before rewriting the
        whole file, so a write-only client would append onto (or diff against) the stale disk
        copy and silently drop the user's unsaved work (review finding R5). Without the pair,
        edits go to disk exactly as in a terminal and the gate asks once per workspace.
        """
        fs = getattr(client_capabilities, "fs", None)
        self._client_can_read = bool(getattr(fs, "read_text_file", False))
        self._client_can_write = bool(getattr(fs, "write_text_file", False))
        self.has_review_surface = self._client_can_read and self._client_can_write
        log.info(
            "acp initialize: protocol=%s client_fs_read=%s client_fs_write=%s",
            protocol_version, self._client_can_read, self._client_can_write,
        )
        from localharness import resolved_version

        return InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_capabilities=AgentCapabilities(load_session=False),
            auth_methods=[],
            agent_info=Implementation(name=AGENT_NAME, version=resolved_version()),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: Optional[list[str]] = None,
        mcp_servers: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        """A session id, the boundary, and the mode picker — immediately (PRD §4).

        The `AgentLoop` and the model server are NOT built here; that happens on the first
        prompt, where status can stream. What does happen here is everything pure or cheap:
        v0.13 workspace discovery (with the one-time trust question rendered as a permission
        dialog), then the boundary derived from where the session stands, by the same functions
        `cli/start_cmd` uses, so a Zed session and a terminal session agree on what "this
        project" means.

        The process changes into `cwd`: the harness derives its boundary, its config layer and
        its state directory from the working directory, so one process serves one project folder
        and one thread (:data:`ONE_THREAD_PER_PROCESS`). A second `session/new` — another folder
        or the same one — is `invalid_params`, not a second session sharing the first one's loop.

        `mcp_servers` is the editor's own MCP list, and this version does not connect it. That is
        said out loud twice — once to the server log here, once into the panel on the first
        prompt (:data:`MCP_SERVERS_NOT_CONNECTED_NOTICE`) — because dropping it silently leaves a
        user waiting for tools that will never appear, with nothing anywhere to explain it.
        """
        from acp.core import RequestError

        from localharness.agent.gate import derive_session_boundary
        from localharness.agent.verdict import narrow_boundary
        from localharness.cli.workspace import resolve_workspace_layer

        requested = Path(cwd).expanduser()
        if self._session_id is not None:
            raise RequestError.invalid_params(
                {"reason": ONE_THREAD_PER_PROCESS.format(current=self._cwd, requested=requested)}
            )

        # Claimed before the work below, because the trust question is put through
        # `session/request_permission`, which needs a session id to address. Released again if
        # that work fails, so a folder that could not be opened does not lock the process out.
        session_id = uuid.uuid4().hex
        self._session_id = session_id
        try:
            os.chdir(requested)
            self._cwd = requested.resolve()
            self._workspace = await asyncio.to_thread(
                resolve_workspace_layer, self._config_dir, asker=self._trust_asker()
            )
            derived = derive_session_boundary(cwd=self._cwd, local_dir=self._workspace)
            # `permissions.workspace_root` may only NARROW the derived boundary and is read from
            # the agent config, which does not exist yet — the session build applies it again on
            # the real config. Here it is derived without narrowing so `new_session` can answer
            # the only question it needs to: is there a project folder at all?
            self._boundary, _ = narrow_boundary(derived, None)
        except BaseException:
            self._session_id = None
            self._cwd = None
            raise
        self._note_unconnected_mcp_servers(mcp_servers)
        log.info("acp session %s: cwd=%s boundary=%s", session_id, self._cwd, self._boundary)

        return NewSessionResponse(
            session_id=session_id,
            modes=SessionModeState(
                current_mode_id=self._current_mode_id(), available_modes=list(ACP_MODES)
            ),
        )

    def _note_unconnected_mcp_servers(self, mcp_servers: Optional[list[Any]]) -> None:
        """Say — twice — that the editor's MCP servers are not connected (D1).

        `session/new` carries the servers the EDITOR manages, and connecting them would mean
        starting processes on the user's machine that the harness's own config never authorized,
        outside the tool policy that governs everything else. Not doing it is the defensible
        v1 answer; doing it silently is not. The log line reaches the server log now, the notice
        is queued for the first prompt, where there is a panel to print it in.
        """
        servers = list(mcp_servers or [])
        if not servers:
            return
        names = [
            (getattr(s, "name", "") or "").strip() or ATTACHMENT_UNNAMED_LABEL for s in servers
        ]
        joined = ", ".join(names)
        log.warning(MCP_SERVERS_NOT_CONNECTED_LOG, len(names), joined)
        self._pending_notice = MCP_SERVERS_NOT_CONNECTED_NOTICE.format(
            count=len(names), servers=joined
        )

    async def set_session_mode(
        self, session_id: str, mode_id: str, **kwargs: Any
    ) -> SetSessionModeResponse | None:
        """Zed's mode picker (PRD §3.4).

        Before the first prompt there is no gate, so the mode is validated against the advertised
        list and remembered; once the gate exists it IS the validator, and its `ValueError`
        message is what the user sees. Either way an unknown or forbidden mode is
        `invalid_params`, never a silent no-op that leaves the picker showing a mode the session
        is not in. A mode set on any id but the live one is refused for the same reason a prompt
        is: this process has one session, and switching its mode on another thread's behalf is a
        change nobody asked for.
        """
        from acp.core import RequestError

        self._require_live_session(session_id)
        try:
            if self._gate is not None:
                self._gate.set_mode(mode_id, from_channel=True)
            elif mode_id in ACP_MODE_IDS:
                self._pending_mode = mode_id
            else:
                raise ValueError(
                    f"unknown mode {mode_id!r}; choose one of: "
                    + ", ".join(m.id for m in ACP_MODES)
                )
        except ValueError as exc:
            raise RequestError.invalid_params({"reason": str(exc)}) from exc

        await self._update(update_current_mode(mode_id))
        return SetSessionModeResponse()

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        """One user turn (PRD §4).

        Order matters and is the design: no boundary means no turn at all (a session standing in
        `$HOME` would ask about every single write, so the honest answer is to say so and stop);
        otherwise the session is brought up if this is the first prompt, streaming status while
        it takes; then `run_turn` runs with `on_token` wired to `agent_message_chunk`, as its own
        task so `session/cancel` has something to cancel.

        A prompt for any id but the live one is refused (:data:`UNKNOWN_SESSION`). It used to
        overwrite the live id, so a second thread's prompt silently re-tagged the first thread's
        file reads, permission dialogs and updates with its own session id.

        `/pending`, `/approve N` and `/deny N` are intercepted ahead of the boundary check and
        the bring-up: Zed gives this agent no command surface, so the queue is answered by typing
        into the same box, and those three answers must cost neither a model turn nor a session
        bring-up (nothing can be parked before a session exists). The one exception
        comes back as text — an approval on an idle session is a sentence the model has to run a
        turn on, so it falls through the ordinary path below and Zed sees the retry stream.
        """
        self._require_live_session(session_id)
        text = _prompt_text(prompt)

        # First prompt only: whatever `session/new` could not say because there was no panel yet.
        if self._pending_notice is not None:
            notice, self._pending_notice = self._pending_notice, None
            await self.send_message(notice)

        verb, arg = _pending_command(text)
        if verb:
            followup = await self._run_pending_command(verb, arg)
            if followup is None:
                return PromptResponse(stop_reason="end_turn")
            text = followup

        if self._boundary is None:
            from localharness.cli.start_cmd import NO_BOUNDARY_NOTICE

            await self.send_message(NO_BOUNDARY_NOTICE)
            return PromptResponse(stop_reason="end_turn")

        if not await self._ensure_session():
            await self.send_message(self._session_error or BRINGUP_FAILED.format(error="unknown"))
            return PromptResponse(stop_reason="end_turn")

        self._streamed_this_turn = False
        task = asyncio.create_task(self._agent_loop.run_turn(task=text, on_token=self._on_token))
        self._turn_task = task
        try:
            # `asyncio.wait` rather than `await task`: a cancelled turn must be reported as
            # `cancelled`, and awaiting a cancelled task would raise CancelledError into THIS
            # coroutine, where it reads as the request handler itself being torn down.
            await asyncio.wait({task})
        finally:
            self._turn_task = None
        if task.cancelled():
            return PromptResponse(stop_reason="cancelled")
        exc = task.exception()
        if exc is not None:
            # `run_turn` is documented never to raise; if it does, the user gets the reason
            # rather than an empty panel. Not a refusal — the model refused nothing.
            log.exception("acp turn raised", exc_info=exc)
            await self.send_message(BRINGUP_FAILED.format(error=exc))
        return PromptResponse(stop_reason="end_turn")

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        """`session/cancel` — the same path SIGINT takes in the REPL (PRD §4). A notification,
        so it returns at once; the in-flight `prompt` answers `cancelled`.

        Only for the live session: a cancel carrying another id is a client cancelling a thread
        this process does not serve, and stopping the running turn on its behalf would kill
        somebody else's work. A notification cannot answer, so it is dropped and logged.
        """
        if session_id != self._session_id:
            log.warning("acp cancel for an unknown session: %s", session_id)
            return
        task = self._turn_task
        if task is not None and not task.done():
            task.cancel()

    # ------------------------------------------------------------ session bring-up

    async def _ensure_session(self) -> bool:
        """Build the harness session on the first prompt, streaming status (PRD §4).

        Returns whether there is a loop to run a turn on. The build is
        `cli/start_cmd._start_async` with `channel_mode="acp"` — one path for terminal, Discord
        and Zed — running as its own task because that function OWNS the session for its
        lifetime (it ends by serving this channel and tears the session down in its `finally`).
        """
        if self._agent_loop is not None:
            return True
        if self._session_task is None:
            self._session_task = asyncio.create_task(self._run_session())

        started = time.monotonic()
        wait_s = PROVIDER_BRINGUP_NOTICE_AFTER_S
        while not self._ready.is_set():
            try:
                await asyncio.wait_for(asyncio.shield(self._ready.wait()), wait_s)
            except asyncio.TimeoutError:
                wait_s = PROVIDER_BRINGUP_NOTICE_EVERY_S
                await self.send_message(
                    BRINGUP_STATUS.format(seconds=time.monotonic() - started)
                )
        return self._agent_loop is not None

    async def _run_session(self) -> None:
        """Own the harness session for the life of the connection.

        `_start_async` returns only when `serve()` below returns, i.e. when the ACP connection
        closes — so this task IS the session, and its `finally` (MCP shutdown, memory
        consolidation, the LLM client) runs on the way out exactly as it does for a REPL.
        """
        from localharness.cli.start_cmd import _start_async

        try:
            await _start_async(
                agent_name=None,
                verbose=False,
                debug=False,
                config_dir=self._config_dir,
                channel_mode="acp",
                no_input=True,
                acp_channel=self,
            )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 — typer.Exit included; the panel gets the reason
            code = getattr(exc, "exit_code", None)
            self._session_error = (
                BRINGUP_EXITED.format(code=code)
                if code is not None
                else BRINGUP_FAILED.format(error=exc)
            )
            log.exception("acp session bring-up failed")
        finally:
            # Set unconditionally: a failed bring-up must release the waiting prompt, or the
            # user watches a status line forever.
            self._ready.set()

    async def serve(self, *, bus: Any, agent_loop: Any, gate: Any, tool_registry: Any) -> None:
        """Run this channel in place of the REPL (called by `start_cmd`'s acp branch).

        The terminal and Discord let `OrchestratorREPL` pull a line and run a turn; ACP turns
        arrive as JSON-RPC requests instead, so this serves the bus subscriptions and then simply
        waits for the connection to end. Everything a turn needs is handed over here, which is
        also the moment a mode picked before the first prompt takes effect.
        """
        self.bus = bus
        self._agent_loop = agent_loop
        self._gate = gate
        self._registry = tool_registry
        self._wire_editor_file_io(tool_registry)
        if self._pending_mode is not None:
            gate.set_mode(self._pending_mode, from_channel=True)
            self._pending_mode = None
        await self.start()
        self._ready.set()
        try:
            await self._closed.wait()
        finally:
            await self.stop()

    async def aclose(self, timeout_s: float = SESSION_TEARDOWN_TIMEOUT_S) -> None:
        """End the session: the ACP connection closed, or stdin hit EOF.

        Releases `serve`, which returns through `_start_async`'s ordered teardown (MCP servers,
        memory consolidation, the LLM client). Bounded, because a wedged MCP server must not stop
        the process from exiting after the editor has already gone away.
        """
        self._closed.set()
        task = self._session_task
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout_s)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001 — teardown never fails the exit
            task.cancel()

    def _wire_editor_file_io(self, registry: Any) -> None:
        """Point `read`/`write`/`edit` at the editor instead of the disk (PRD §4).

        ALL-OR-NOTHING, and only on the three builtins whose whole job is one text file
        (:data:`EDITOR_FILE_IO_TOOLS`). The seam needs BOTH `fs` capabilities because every
        editor-backed write is a read-modify-write: ACP has no append, so `write(mode='append')`
        rewrites the whole file, and `edit` matches `old_string` against the buffer. Wiring the
        write half alone would have appended onto the stale DISK copy and thrown away every
        unsaved line (review finding R5) — so a client advertising only one of the two gets
        neither hook, `has_review_surface` stays False, and the gate asks once for an
        in-workspace edit instead (PRD §3.1).

        The hooks are set on the registered instances rather than passed at construction because
        capabilities are known at `initialize`, long after `register_builtin_tools()` ran — see
        `tools/base.Tool.file_read_hook`.
        """
        if not (self._client_can_read and self._client_can_write):
            return
        for name in EDITOR_FILE_IO_TOOLS:
            tool = registry._find_tool_by_name(name)
            if tool is None:
                continue
            tool.file_read_hook = self._read_text_file
            if name != "read":
                tool.file_write_hook = self._write_text_file

    async def _read_text_file(self, path: Path) -> str:
        """`fs/read_text_file` — the buffer the user is looking at, unsaved edits included."""
        resp = await self._conn.read_text_file(self._session_id, str(path))
        return resp.content

    async def _write_text_file(self, path: Path, content: str) -> None:
        """`fs/write_text_file` — the change lands in Zed's review pane, not silently on disk."""
        await self._conn.write_text_file(self._session_id, str(path), content)

    # ------------------------------------------------------------ the permission ask

    async def ask_permission(self, request: Any) -> Any:
        """Render one ASK verdict as Zed's permission dialog (PRD §3.5, Zed row).

        No timeout: Zed holds the dialog open, and the gate passes none for this channel. The
        options are the four ACP kinds, or the `_once` pair when the request cannot be
        remembered — and the workspace-trust question's own pair, because its answer is the one
        ungrantable answer that IS kept (`WORKSPACE_TRUST_KLASS`). A dismissed dialog
        (`DeniedOutcome`) is a refusal of this call only — never a durable "never", which the
        user did not say.
        """
        from localharness.agent.gate_types import Decision

        if self._conn is None or self._session_id is None:
            log.warning("acp permission with no connection: %s", getattr(request, "display", ""))
            return Decision(kind=PERMISSION_FALLBACK_DECISION)

        if getattr(request, "klass", "") == WORKSPACE_TRUST_KLASS:
            kinds, names = TRUST_OPTION_KINDS, TRUST_OPTION_NAMES
        elif request.grantable:
            kinds, names = GRANTABLE_OPTION_KINDS, PERMISSION_OPTION_NAMES
        else:
            kinds, names = UNGRANTABLE_OPTION_KINDS, PERMISSION_OPTION_NAMES
        params = dict(getattr(request, "tool_params", {}) or {})
        tool_name = getattr(request, "tool_name", "")
        title, body = split_display(sanitize_for_display(request.display))
        response = await self._conn.request_permission(
            self._session_id,
            ToolCallUpdate(
                tool_call_id=self._pending_call_id(request, tool_name, params),
                title=title,
                kind=acp_kind_for_group(self._group_for(tool_name)),
                status="pending",
                content=[tool_content(text_block(body))] if body else None,
                raw_input=params,
            ),
            options=[PermissionOption(option_id=k, name=names[k], kind=k) for k in kinds],
        )
        chosen = getattr(response.outcome, "option_id", None)
        if chosen in kinds:
            return Decision(kind=chosen)  # type: ignore[arg-type]
        return Decision(kind=PERMISSION_FALLBACK_DECISION)

    def _trust_asker(self):
        """Bridge the SYNCHRONOUS workspace-trust question onto this async connection.

        `cli/workspace.resolve_workspace_layer` takes a plain `Callable[[str], bool]` and runs on
        a worker thread (`asyncio.to_thread` above), so its answer has to come back across the
        thread boundary — `run_coroutine_threadsafe` onto the loop that owns the connection, per
        that function's own docstring. Without this the phase-39 rule would treat a Zed session
        as non-interactive and silently ignore an outside `.localharness/` forever (PRD §3.5).
        """
        loop = asyncio.get_running_loop()

        def _ask(question: str) -> bool:
            return asyncio.run_coroutine_threadsafe(self._ask_trust(question), loop).result()

        return _ask

    async def _ask_trust(self, question: str) -> bool:
        """The trust question as a permission dialog — ACP's only dialog primitive."""
        if self._conn is None or self._session_id is None:
            return False
        response = await self._conn.request_permission(
            self._session_id,
            ToolCallUpdate(
                tool_call_id=f"trust-{uuid.uuid4().hex}",
                title=TRUST_TOOL_CALL_TITLE,
                kind="switch_mode",
                status="pending",
                content=[tool_content(text_block(question))],
            ),
            options=[
                PermissionOption(option_id=k, name=PERMISSION_OPTION_NAMES[k], kind=k)
                for k in UNGRANTABLE_OPTION_KINDS
            ],
        )
        return getattr(response.outcome, "option_id", None) == "allow_once"

    # ------------------------------------------------------------ the pending queue

    async def on_permission_staged(self, event: Any) -> None:
        """`PermissionStaged` → the one-line notice, and a local copy of the parked call.

        The copy is what lets the outcome line name a command: `PermissionGate._answer` pops the
        call out of `gate.pending` before publishing its resolution, so by the time this channel
        hears the answer the gate no longer holds the rendering it has to print.
        """
        self._staged[event.pending.id] = event.pending
        await self.send_pending_notice(event.pending, event.total)

    async def send_pending_notice(self, pending: Any, total: int) -> None:
        """One inline line (:data:`PENDING_NOTICE`), not a dialog — see that constant."""
        await self.send_message(
            PENDING_NOTICE.format(
                id=pending.id, rendering=sanitize_for_display(pending.rendering), total=total
            )
        )

    async def on_permission_resolved(self, event: Any) -> None:
        """Write the outcome of a PARKED call into the transcript; ignore every other answer.

        An ordinary ASK resolves through this event too, and Zed already drew that dialog and its
        outcome — repeating it in the panel would report a decision the user just made by hand.
        A staged call is the one whose answer carries `pending_id` (set by `gate.approve` /
        `gate.deny`); a dialog answer carries none and is left alone.
        """
        pid = getattr(event, "pending_id", None)
        answered = self._staged.get(pid) if pid is not None else None
        if answered is None:
            return
        await self._announce_resolution(answered, str(event.decision).startswith("allow"))

    async def _announce_resolution(self, pending: Any, approved: bool) -> None:
        """The outcome line for one parked call, written once however the answer arrived."""
        if pending.id in self._announced:
            return
        self._announced.add(pending.id)
        await self.send_message(
            PENDING_RESOLVED_LINES[approved].format(
                id=pending.id, rendering=sanitize_for_display(pending.rendering)
            )
        )

    async def _run_pending_command(self, verb: str, arg: str) -> Optional[str]:
        """Answer the queue in place of a turn. Returns the sentence the MODEL still has to hear.

        `None` means the command is finished — `/pending`, any error, a denial, and an approval
        that reached a running turn as a nudge. A string means the approval had no turn to nudge,
        so the caller runs it as an ordinary user turn; that is the only way an approval reaches
        the model on an idle session, and it keeps the agent loop the one thing that dispatches a
        tool.

        Honest about the nudge path: a Zed client sends one `session/prompt` at a time, so
        `/approve` typed while a turn runs is normally only possible for a client that issues
        concurrent requests. The branch exists because JSON-RPC permits exactly that and a nudge
        into the live turn is the right answer when it happens.
        """
        queue = list((getattr(self._gate, "pending", None) or {}).values())
        if verb == "/pending":
            lines = [PENDING_LIST_HEADER.format(total=len(queue))] + [
                PENDING_LIST_LINE.format(
                    id=p.id, label=p.agent_label, rendering=sanitize_for_display(p.rendering)
                )
                for p in queue
            ]
            await self.send_message("\n".join(lines) if queue else NOTHING_PENDING)
            return None

        approve = verb == "/approve"
        if arg and not arg.isdigit():
            await self.send_message(PENDING_NEEDS_A_NUMBER.format(verb=verb))
            return None
        # No gate yet means no session yet, and nothing can have been parked before one exists —
        # so the empty queue is the true answer, not a reason to start a model server.
        if self._gate is None:
            await self.send_message(NOTHING_PENDING)
            return None
        answer = self._gate.approve if approve else self._gate.deny
        try:
            answered = await answer(int(arg) if arg else None)
        except KeyError:
            await self.send_message(
                PENDING_UNKNOWN.format(id=arg) if arg else NOTHING_PENDING
            )
            return None

        # Usually already written by `on_permission_resolved` (the gate publishes inside the
        # await above); this is what keeps the confirmation from going missing on a gate wired
        # without a bus.
        await self._announce_resolution(answered, approve)
        nudge = (PENDING_APPROVED_NUDGE if approve else PENDING_DENIED_NUDGE).format(
            id=answered.id, rendering=answered.rendering
        )
        if self._turn_running():
            self._agent_loop.push_user_nudge(nudge)
            return None
        # A denial needs no turn of its own: there is nothing for an idle session to do about it.
        return nudge if approve else None

    def _turn_running(self) -> bool:
        """Is a turn in flight right now? `prompt` owns the task and clears it in its `finally`,
        so this is the same slot `session/cancel` cancels."""
        task = self._turn_task
        return task is not None and not task.done()

    # ------------------------------------------------------------ bus → session/update

    async def start(self) -> None:
        """Subscribe to the events Zed renders. Tool lifecycle is on the bus (PRD §1), so this is
        the same seam Discord and the terminal use — no second notion of what a tool call is."""
        self._handles = [
            self.bus.subscribe(Action, self.on_action),
            self.bus.subscribe(Observation, self.on_observation),
            self.bus.subscribe(TaskComplete, self.on_task_complete),
            self.bus.subscribe(TurnFailed, self.on_turn_failed),
            self.bus.subscribe(ParseFailed, self.on_parse_failed),
            self.bus.subscribe(Escalation, self.on_escalation),
            # The queue: the only way this channel hears that `auto` parked a call, and the only
            # way it hears the answer once somebody gives one.
            self.bus.subscribe(PermissionStaged, self.on_permission_staged),
            self.bus.subscribe(PermissionResolved, self.on_permission_resolved),
        ]

    async def stop(self) -> None:
        for handle in self._handles:
            if handle is not None:
                self.bus.unsubscribe(handle)
        self._handles = []

    async def on_action(self, event: Action) -> None:
        """`Action` → ACP `tool_call` (PRD §4). Kind comes from the tool's group, and a
        filesystem tool also reports its target so Zed can link to the file."""
        if event.action_type != "tool_call" or self._conn is None or self._session_id is None:
            return
        name = event.tool_name or ""
        params = event.tool_params or {}
        call_id = event.tool_call_id or uuid.uuid4().hex
        # The gate asks AFTER this event is published (`agent/loop.py`), so remembering the call
        # here is what lets `ask_permission` attach its dialog to the row Zed already drew.
        # Keyed by call id and dropped in `on_observation`, so concurrent calls coexist and the
        # map cannot grow for the life of the session.
        self._pending_calls[call_id] = (name, params)
        group = self._group_for(name)
        target = _first_path(params) if group.startswith("fs.") else None
        await self._update(
            start_tool_call(
                call_id,
                _tool_title(name, params),
                kind=acp_kind_for_group(group),
                status="in_progress",
                raw_input=params,
                locations=[ToolCallLocation(path=target)] if target else None,
            )
        )

    async def on_task_complete(self, event: TaskComplete) -> None:
        """The answer, but only when nothing streamed it already.

        The normal path is `on_token`: a streaming provider types the answer into the panel as it
        generates, and posting the completion summary on top of that would print it twice. But
        `on_token` is the PROVIDER's promise, not the loop's — a runtime that does not stream, or
        a turn whose last completion produced no deltas, would otherwise leave a panel showing
        tool rows and no answer at all. So the summary is the fallback, gated on whether anything
        actually reached the user this turn. Child turns stay internal, as everywhere else
        (`channels/base.on_task_complete`): a subagent's summary belongs to its parent.
        """
        if getattr(event, "parent_id", None) or self._streamed_this_turn:
            return
        await self.send_message(event.summary)

    async def on_observation(self, event: Observation) -> None:
        """`Observation` → ACP `tool_call_update`, completed or failed with its output."""
        if self._conn is None or self._session_id is None:
            return
        call_id = event.tool_call_id
        if not call_id:
            return
        self._pending_calls.pop(call_id, None)  # the call is over; it can pair with nothing now
        output = event.output or event.error or ""
        await self._update(
            update_tool_call(
                call_id,
                status="failed" if event.error else "completed",
                content=[tool_content(text_block(output))] if output else None,
                raw_output=output or None,
            )
        )

    # ------------------------------------------------------------ ChannelAdapter output

    async def _update(self, update: Any) -> None:
        if self._conn is None or self._session_id is None:
            return
        await self._conn.session_update(self._session_id, update)

    async def _on_token(self, token: str) -> None:
        """`on_token` → `agent_message_chunk` (PRD §4).

        Everything is a message chunk today, thinking included: `on_token` carries no phase, so
        routing reasoning to `agent_thought_chunk` waits on exposing the phase the terminal
        status row already tracks (PRD §4, "Thinking"). Said plainly in docs/zed.md rather than
        guessed at here.
        """
        if token:
            self._streamed_this_turn = True
            await self._update(update_agent_message_text(token))

    async def send_message(
        self,
        content: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if content:
            await self._update(update_agent_message_text(content))

    async def send_streaming(
        self, token_stream: AsyncIterator[str], agent_id: str | None = None
    ) -> str:
        full = ""
        async for token in token_stream:
            full += token
            await self._on_token(token)
        return full

    async def send_tool_call(
        self, tool_name: str, arguments: dict[str, Any], agent_id: str | None = None
    ) -> None:
        """Unused: `on_action` is overridden to keep the bus event's `tool_call_id`, which ACP
        needs to pair a call with its result. Kept because the base class declares it."""
        return

    async def send_tool_result(
        self, tool_name: str, result: str, is_error: bool, agent_id: str | None = None
    ) -> None:
        """Unused for the same reason as `send_tool_call` — see `on_observation`."""
        return

    async def send_error(
        self, error: str, detail: str | None = None, agent_id: str | None = None
    ) -> None:
        """A failed turn reaches the panel as text (PRD §4: `TurnFailed` streams its reason).

        Deliberately NOT `stop_reason="refusal"`: the model refused nothing, the turn broke, and
        claiming a refusal would put words in its mouth."""
        text = error if not detail else f"{error}\n{detail}"
        await self.send_message(text)

    async def read_input(self, prompt: str = "> ") -> str:
        """ACP turns arrive as `session/prompt` requests, never by pulling a line: this channel
        does not run under the REPL (see `serve`)."""
        raise NotInteractiveError("the ACP channel receives turns via session/prompt")

    # ------------------------------------------------------------ helpers

    def _require_live_session(self, session_id: str) -> None:
        """Refuse a request addressed to any session but the one this process is serving.

        One process, one session (:data:`ONE_THREAD_PER_PROCESS`); everything session-scoped —
        the loop, the gate, the turn task, the `fs/*` calls, the permission dialog — belongs to
        it. Raising here is what keeps a stale or foreign id from being answered by it.
        """
        if session_id == self._session_id:
            return
        from acp.core import RequestError

        raise RequestError.invalid_params(
            {"reason": UNKNOWN_SESSION.format(requested=session_id, current=self._cwd)}
        )

    def _current_mode_id(self) -> str:
        from localharness.agent.gate_types import DEFAULT_MODE

        if self._gate is not None:
            return str(self._gate.mode)
        return self._pending_mode or DEFAULT_MODE

    def _group_for(self, tool_name: str) -> str:
        """The tool's `ToolSchema.group`, or `other` when the registry has not been built yet
        (a permission ask can only happen once it has, but `initialize`-time callers cannot know
        that)."""
        registry = self._registry
        if registry is None or not tool_name:
            return ACP_KIND_DEFAULT
        tool = registry._find_tool_by_name(tool_name)
        if tool is None:
            return ACP_KIND_DEFAULT
        try:
            return tool.info().group or ACP_KIND_DEFAULT
        except Exception:  # noqa: BLE001 — a broken schema must not block the dialog
            return ACP_KIND_DEFAULT

    def _pending_call_id(self, request: Any, tool_name: str, params: dict) -> str:
        """The id of the `tool_call` this ask belongs to (D4).

        The request's own `call_id` is the answer whenever the gate carries one: it is the id the
        loop published the `Action` under, so the dialog lands on exactly the row Zed drew, with
        no inference at all. `getattr` because that field is arriving separately — this works
        before and after it lands.

        Without it, the fallback is the old heuristic, now over a MAP rather than one slot: the
        newest pending call whose name and arguments match. One slot was wrong the moment two
        calls were in flight at once (subagents share this bus), because the second `Action`
        overwrote the first and the first call's dialog then attached to the second call's row —
        a user approving one command while reading another. Matching still cannot separate two
        identical calls, so a fresh id (Zed draws its own row) remains the honest last resort.
        """
        call_id = getattr(request, "call_id", None)
        if isinstance(call_id, str) and call_id:
            return call_id
        for known_id, (name, known_params) in reversed(list(self._pending_calls.items())):
            if name == tool_name and known_params == params:
                return known_id
        return f"ask-{uuid.uuid4().hex}"


def _block_label(block: Any) -> str:
    """What to call a prompt block this adapter cannot read — its name, else its uri, else its
    type. `ResourceContentBlock` (a Zed @-mention) carries `name` and `uri`; an embedded
    resource carries them one level down under `resource`; an image block has only its type."""
    for attr in ("name", "uri"):
        value = getattr(block, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    resource = getattr(block, "resource", None)
    if resource is not None:
        for attr in ("name", "uri"):
            value = getattr(resource, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
    kind = getattr(block, "type", None)
    return kind.strip() if isinstance(kind, str) and kind.strip() else ATTACHMENT_UNNAMED_LABEL


def _prompt_text(blocks: list[Any]) -> str:
    """The user's turn as text: every text block, and a placeholder line for everything else.

    ACP prompts can carry images, audio, resource links and embedded resources — a Zed
    @-file-mention is a `ResourceContentBlock`, part of the BASELINE protocol, so it arrives in
    ordinary use rather than as an exotic case. The harness's `run_turn` takes a string and this
    adapter reads none of those, but a block that vanishes silently is the failure that costs
    most: the user watched the editor attach a file and assumes it was read, and the model
    answers about text it never received.

    So every non-text block becomes one :data:`ATTACHMENT_PLACEHOLDER` line, in the position it
    was sent. The model sees exactly what was dropped and can ask for it; the user sees the same
    sentence quoted back. Reading attachments is a later phase (docs/zed.md's "not yet" list).
    """
    parts: list[str] = []
    for block in blocks or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            if text:
                parts.append(text)
            continue
        parts.append(ATTACHMENT_PLACEHOLDER.format(label=_block_label(block)))
    return "\n".join(parts)


LocalHarnessAcpAgent = AcpChannel
"""The ACP `Agent` this process runs (`cli/acp_cmd.py`). Same object as the channel — see the
module docstring — named for the role it plays at the protocol edge."""
