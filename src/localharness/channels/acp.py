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
   client can do. Whether the client offers `fs/write_text_file` is what decides
   `has_review_surface`, and therefore whether an in-workspace edit asks at all (§3.1 choice 2).
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
    Action,
    Escalation,
    Observation,
    ParseFailed,
    TaskComplete,
    TurnFailed,
)

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ constants

AGENT_NAME = "localharness"
"""What Zed shows as the agent's implementation name in the handshake (`agent_info`)."""

ACP_MODES: tuple[SessionMode, ...] = (
    SessionMode(
        id="guarded",
        name="Guarded",
        description=(
            "Ask before anything leaves this project folder, touches a protected path, or runs "
            "a command this workspace has never allowed. Reads never ask."
        ),
    ),
    SessionMode(
        id="trusted",
        name="Trusted",
        description=(
            "Allow anything that could have been remembered with 'always'. Destructive shell "
            "commands and protected paths still ask every time."
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

`unattended` is deliberately absent. It turns every ASK into an ALLOW, so it is set in config by
bench and scheduled jobs and is never reachable from a picker — the same rule
`PermissionGate.set_mode(from_channel=True)` enforces one layer down (PRD §3.4)."""

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

BRINGUP_STATUS = "Starting the model server — {seconds:.0f}s so far…"
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

PERMISSION_FALLBACK_DECISION = "reject_once"
"""Fail closed (SECURITY.md "deny on doubt") when there is nowhere to put the question, and what
a dismissed dialog means: ACP's `DeniedOutcome{outcome:"cancelled"}` is the user hitting Escape,
which is a refusal of this call and nothing more."""

TRUST_TOOL_CALL_TITLE = "Load workspace configuration?"
"""The title on the trust question's `request_permission` payload. The question itself
(`cli/workspace.TRUST_QUESTION`) is the body; ACP has no dialog primitive other than a permission
request, so the one-time trust dialog rides on the same mechanism as every other ask (PRD §3.5:
"the existing workspace-trust dialog becomes the first client of ask_permission")."""

ONE_PROJECT_PER_PROCESS = (
    "This localharness process is already serving {current}. Open {requested} in its own Zed "
    "thread, or restart the agent server for that folder — one agent process serves one project."
)
"""v1 limitation, stated as an error rather than silently answering for the wrong folder: the
harness session derives its boundary, config layer and memory from ONE directory, and the
process changes into it. A second `session/new` for a different folder would be served by a
session pointed somewhere else, which is exactly the kind of quiet wrong answer the boundary
exists to prevent. Named in docs/zed.md's "not yet" list."""

SESSION_TEARDOWN_TIMEOUT_S = 10.0
"""How long `aclose` waits for the harness session's ordered shutdown (MCP servers, memory
consolidation, the LLM client) once the editor has gone away. Bounded because the user is no
longer watching: a wedged MCP server must not keep the process alive, and the shutdown itself is
best-effort in `start_cmd`'s `finally`. Ten seconds is the same order as the MCP client's own
shutdown budget and long enough for a WAL checkpoint."""

TOOL_TITLE_ARG_CHARS = 80
"""How much of a tool call's leading argument goes into the `tool_call` title Zed renders on one
row. Long enough for a path or a short command, short enough not to wrap the panel."""


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
        self._session_ids: set[str] = set()
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
        self._pending_call: tuple[str, str, dict] | None = None
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

        The client's `fs` capabilities are the load-bearing part: `write_text_file` is what gives
        this session a review surface (Zed's diff pane, accept/reject per hunk), which is what
        lets an in-workspace edit run without asking at all (PRD §3.1 choice 2, critic finding
        11). Without it, edits go to disk and the gate asks once per workspace.
        """
        fs = getattr(client_capabilities, "fs", None)
        self._client_can_read = bool(getattr(fs, "read_text_file", False))
        self._client_can_write = bool(getattr(fs, "write_text_file", False))
        self.has_review_surface = self._client_can_write
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
        its state directory from the working directory, so one process serves one project
        folder (:data:`ONE_PROJECT_PER_PROCESS`).
        """
        from acp.core import RequestError

        from localharness.agent.gate import derive_session_boundary
        from localharness.agent.verdict import narrow_boundary
        from localharness.cli.workspace import resolve_workspace_layer

        requested = Path(cwd).expanduser()
        if self._cwd is not None and requested.resolve() != self._cwd:
            raise RequestError.invalid_params(
                {"reason": ONE_PROJECT_PER_PROCESS.format(current=self._cwd, requested=requested)}
            )

        session_id = uuid.uuid4().hex
        self._session_id = session_id
        self._session_ids.add(session_id)

        if self._cwd is None:
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
            log.info("acp session %s: cwd=%s boundary=%s", session_id, self._cwd, self._boundary)

        return NewSessionResponse(
            session_id=session_id,
            modes=SessionModeState(
                current_mode_id=self._current_mode_id(), available_modes=list(ACP_MODES)
            ),
        )

    async def set_session_mode(
        self, session_id: str, mode_id: str, **kwargs: Any
    ) -> SetSessionModeResponse | None:
        """Zed's mode picker (PRD §3.4). `unattended` is never reachable from here.

        Before the first prompt there is no gate, so the mode is validated against the advertised
        list and remembered; once the gate exists it IS the validator, and its `ValueError`
        message is what the user sees. Either way an unknown or forbidden mode is
        `invalid_params`, never a silent no-op that leaves the picker showing a mode the session
        is not in.
        """
        from acp.core import RequestError

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
        """
        self._session_id = session_id
        text = _prompt_text(prompt)

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
        so it returns at once; the in-flight `prompt` answers `cancelled`."""
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

        Only when the client advertises the matching `fs` capability, and only on the three
        builtins whose whole job is one text file (:data:`EDITOR_FILE_IO_TOOLS`). The hooks are
        set on the registered instances rather than passed at construction because capabilities
        are known at `initialize`, long after `register_builtin_tools()` ran — see
        `tools/base.Tool.file_read_hook`.
        """
        if not (self._client_can_read or self._client_can_write):
            return
        for name in EDITOR_FILE_IO_TOOLS:
            tool = registry._find_tool_by_name(name)
            if tool is None:
                continue
            if self._client_can_read:
                tool.file_read_hook = self._read_text_file
            if self._client_can_write and name != "read":
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
        remembered. A dismissed dialog (`DeniedOutcome`) is a refusal of this call only — never
        a durable "never", which the user did not say.
        """
        from localharness.agent.gate_types import Decision

        if self._conn is None or self._session_id is None:
            log.warning("acp permission with no connection: %s", getattr(request, "display", ""))
            return Decision(kind=PERMISSION_FALLBACK_DECISION)

        kinds = GRANTABLE_OPTION_KINDS if request.grantable else UNGRANTABLE_OPTION_KINDS
        params = dict(getattr(request, "tool_params", {}) or {})
        tool_name = getattr(request, "tool_name", "")
        title, body = split_display(sanitize_for_display(request.display))
        response = await self._conn.request_permission(
            self._session_id,
            ToolCallUpdate(
                tool_call_id=self._pending_call_id(tool_name, params),
                title=title,
                kind=acp_kind_for_group(self._group_for(tool_name)),
                status="pending",
                content=[tool_content(text_block(body))] if body else None,
                raw_input=params,
            ),
            options=[
                PermissionOption(option_id=k, name=PERMISSION_OPTION_NAMES[k], kind=k)
                for k in kinds
            ],
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
        self._pending_call = (call_id, name, params)
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

    def _pending_call_id(self, tool_name: str, params: dict) -> str:
        """The id of the `tool_call` this ask belongs to.

        The loop publishes `Action` before it consults the gate and executes tool calls one at a
        time, so the last one seen IS the pending call — matched on name and arguments so a
        mismatch (a subagent sharing this bus) falls back to a fresh id and Zed draws its own
        row rather than attaching the dialog to the wrong one.
        """
        pending = self._pending_call
        if pending is not None and pending[1] == tool_name and pending[2] == params:
            return pending[0]
        return f"ask-{uuid.uuid4().hex}"


def _prompt_text(blocks: list[Any]) -> str:
    """The user's turn as text: every text block in the prompt, joined.

    ACP prompts can carry images, audio and resource links; the harness's `run_turn` takes a
    string, so v1 reads the text and says so in docs/zed.md rather than pretending to have seen
    a screenshot.
    """
    parts = [b.text for b in (blocks or []) if isinstance(getattr(b, "text", None), str)]
    return "\n".join(p for p in parts if p)


LocalHarnessAcpAgent = AcpChannel
"""The ACP `Agent` this process runs (`cli/acp_cmd.py`). Same object as the channel — see the
module docstring — named for the role it plays at the protocol edge."""
