"""OrchestratorREPL -- interactive prompt_toolkit loop for LocalHarness."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import re
import shlex
import signal
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

from rich.protocol import is_renderable

from localharness.agent.gate_types import MODE_STRICTNESS
from localharness.channels import input_router
from localharness.cli.model_ops import _hold_peer_session_key, _launch_key, _peer_session_key
from localharness.cli.slash_commands import SlashCommand, find_row, help_text
from localharness.config.effort import EFFORT_UNENFORCED_LINE, LEVELS, resolve_effort
from localharness.core.content import ImageError, describe, fit_all, fit_to_tokens
from localharness.core.events import (
    PENDING_APPROVED_NUDGE,
    PENDING_DENIED_NUDGE,
    InputRouted,
    PermissionResolved,
    PermissionStaged,
    UserMessage,
)
from localharness.provider.runtimes import RUNTIMES, row_of

log = logging.getLogger(__name__)

RELAUNCH_REPORT_DRAIN_SECONDS = 2.0
"""How long the REPL's exit waits for a relaunch report still being sent before cancelling it."""

MODE_EFFECTS: dict[str, str] = {
    "auto": "runs everything except a small blacklist — deletes pointing outside this project, "
            "sudo, pipe-to-shell, force-push and hard-reset, and writes to protected paths",
    "guarded": "asks before a call crosses the workspace boundary or looks destructive, and "
               "remembers your answer",
    "trusted": "allows the rememberable asks outright; destructive and protected-path calls "
               "still ask every time",
    "read-only": "refuses writes, non-read-only shell and code execution with an explanation "
                 "the model can work around",
    "unattended": "allows every ask; only the deny list still holds",
}
"""Each mode in the words a person reads after typing `/mode`. Keyed by mode name so the text and
the modes cannot drift apart."""

MODE_SETTABLE_NAMES = ", ".join(sorted(MODE_STRICTNESS, key=lambda name: MODE_STRICTNESS[name]))
"""What `/mode` and Discord's `mode` will set: every mode, loosest first.

`unattended` used to be left out and refused by the gate, on the reasoning that a chat message
must not switch the gate off. The owner met that rule the only way anyone does — "I can't swap
my active localharness session to unattended without exiting" (2026-09-11) — and it is gone: the
person typing at their own prompt is the person the gate protects. Derived from the strictness
table rather than typed out, so a mode added there can never go missing here."""

MODE_STATUS_TEMPLATE = "Permission mode: {mode}. Switch with /mode <name> — {settable}."
MODE_CHANGED_TEMPLATE = "Permission mode: {mode} — {effect}."

# /effort (spec 16 §7.1): every line a person reads is a constant, so tests pin the text by import.
EFFORT_SET_TEMPLATE = "effort: {level} ({rung})"
EFFORT_NONE_LINE = "effort: none (no effort keys are sent)"
EFFORT_LIST_TEMPLATE = "effort: {levels} — /effort <level> sets it, /effort show names the rung"
EFFORT_UNKNOWN_TEMPLATE = (
    "Unknown effort '{arg}'. Levels: off, low, high, max — /effort show reports the current level."
)
EFFORT_AGENT_LINE = "  {name}: {level}"
# One text with start's summary line: defined in config/effort.py, named here for this module's callers.
EFFORT_UNENFORCED_TEMPLATE = EFFORT_UNENFORCED_LINE
EFFORT_PERSIST_FAILED_TEMPLATE = "Effort set for this session, but persisting it failed: {exc}"
EFFORT_PICKER_PROMPT = "Reasoning effort for this session (/effort). This question holds no tool call."

NOTHING_PENDING = "Nothing pending."
PENDING_HEADER = "{total} waiting on you — /approve N or /deny N:"
PENDING_LINE = "  #{id}  {label}{rendering}"
"""`/pending`: the queue of calls `auto` parked instead of asking about (owner ruling
2026-09-12). One line per call, the same one-line rendering every other surface shows, so the
number a person types is the number they read."""

PENDING_COMMANDS: dict[bool, tuple[str, str]] = {
    True: ("approve", "approved"),
    False: ("deny", "denied"),
}
"""(command word, past tense) for the two answers, keyed by "is this an approval". One table so
the word in the error line ("/approve takes a number") and the word in the confirmation
("Pending #1 approved") cannot drift from each other or from the command that was typed."""

PENDING_NO_SUCH = "No pending call {id}. /pending lists what is waiting."
PENDING_NEEDS_A_NUMBER = "/{verb} takes a pending number, e.g. /{verb} 1 (or /{verb} for the oldest)."
PENDING_ANSWERED_LINE = "Pending #{id} {verb}: {rendering}"
"""What the person who typed `/approve` or `/deny` sees back. The command is echoed with the
command it answered, because a bare "done" after a number typed from memory is how the wrong
call gets approved quietly."""


PENDING_HOTKEYS: dict[str, bool] = {"approve_pending": True, "deny_pending": False}
"""The box's two hotkey control events (ctrl+y / ctrl+n), mapped to "is this an approval".

They travel the CONTROL queue, not the slash path, and that is the whole reason they exist: a
slash command typed while a turn runs is queued and replayed after it (`_route_during_turn`), so
a person sitting at the keyboard watching the turn could not unblock the step it had just
skipped until the turn was over — the very wait staging removes for the model, still there for
the human. The box cannot answer the gate itself; the REPL owns it, and one serialized loop
answering it is what keeps the queue race-free."""

BARE_MODE_COMMAND = "mode"

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp")


def _route_preview(text: str) -> str:
    """The audit preview of a routed message: whitespace runs collapsed to one space, then the first
    80 characters, so a multi-line message stays one audit row (spec 21 §6.5)."""
    return " ".join(text.split())[:80]


def _dropped_image_path(line: str) -> str | None:
    """The path when the whole line is ONE token naming an existing picture — what a terminal
    pastes for a file dragged onto it (shell-quoted when the name has spaces). Anything else,
    including a sentence that merely mentions a .png, is None."""
    try:
        tokens = shlex.split(line.strip())
    except ValueError:
        return None
    if len(tokens) != 1 or not tokens[0].lower().endswith(_IMAGE_SUFFIXES):
        return None
    return tokens[0] if Path(tokens[0]).expanduser().is_file() else None
"""On a channel whose class sets `bare_mode_command` (PRD §3.4), this first word IS the command."""


# #129: a /model swap that writes a new default to the user overlay must SAY so — the write
# outlives the session, and a silent one leaves the user running an experiment forever.
SAVED_DEFAULT_NOTE = " (saved as default — persists across restarts)"
# A typed .gguf is served for this session only: the record keeps its own weights and name, so the
# saved default is not rewritten for a name the record never serves (IN-57).
TYPED_SWAP_NOTE = " (this session only — the saved default is unchanged)"
# A llama.cpp primary restarts on a weights file: anything else a person typed (a local_models name,
# an HF-cache repo id) is answered here, before anything is stopped (D-58).
NOT_A_GGUF_LINE = "{target} is not a GGUF file — /model <path.gguf> or a preset name"
# The launch line of a swap that stopped an incumbent says which one held the GPU (D-21): carried
# right after "the swap can take several minutes..." so every check that the line STARTS with
# "Launching <peer>" still holds.
GPU_HOLDER_CLAUSE = " {holder} was stopped to free the GPU."
# A landing's selection (active_endpoint / last_served / pending_preset) that could not be written:
# the in-session switch already happened (D-07).
PERSIST_SELECTION_FAILED = "Switched for this session, but persisting the selection failed: {exc}"

PLUGINS_USAGE = "Usage: /plugins, /plugins enable <name> or /plugins disable <name>"
PLUGINS_TERMINAL_ONLY = ("/plugins works only in a terminal session. From a shell: "
                         "localharness plugins enable <name>")
PLUGINS_NO_SESSION = "/plugins needs a session started with `localharness start`."
PLUGINS_PARKED = "Answer the parked calls first (/pending lists them): a restart would drop them."
PLUGINS_RESTARTING = "Restarting with {name} {state} — your conversation is kept."
"""What `/plugins` answers. A restart drops the gate's parked calls with the session, so it is
refused while any wait; the other channels cannot restart a session at all (R10)."""


# Keywords that signal the user wants to create an agent via conversation.
# Checked case-insensitively against user input when no workflow is active.
_CREATION_TRIGGERS = ("create an agent", "create agent", "make an agent",
                      "new agent", "build an agent", "i want an agent",
                      "i need an agent", "set up an agent", "setup an agent")


def _literal_values(annotation: Any) -> list[str]:
    """Extract the string values of a Pydantic Literal field, drilling through list[Literal[...]]
    (#57: derive permissions.mode / tools.inherit legal values off the schema, no hardcoding)."""
    import typing

    args = typing.get_args(annotation)
    if args and typing.get_origin(annotation) in (list, set, tuple, frozenset):
        return _literal_values(args[0])
    return [a for a in args if isinstance(a, str)]


# Pretty framework labels for the /model tree headers + picker-menu meta (0.10.0 model tree):
# an endpoint's provider_type mapped to a human name. Unknown/blank falls back to a neutral label.
_FRAMEWORK_LABELS = {
    "vllm": "vLLM",
    "ollama": "Ollama",
    "llamacpp": "llama.cpp",
    "lmstudio": "LM Studio",
    "unknown": "server",
}


def _framework_label(provider_type: str | None) -> str:
    """Human framework name for an endpoint's provider_type (menu meta)."""
    return _FRAMEWORK_LABELS.get(provider_type or "", provider_type or "server")


def _generation_system_prompt() -> str:
    """System prompt for agent-YAML generation, DERIVED from AgentConfig (#33).

    Reading required fields + the allowed top-level keys off model_fields keeps
    the stated contract and the Pydantic schema (extra='forbid') from drifting:
    the model is told exactly the shape validation will accept, so it stops
    guessing (agent: nesting, description-not-role, invented keys). The nested
    tools/permissions shapes are stated the same derived way (live 0-for-4: a
    "read-only tools" ask produced bare `permissions: read` / `tools: []`).
    Enforcement still lives in AgentConfig; this just states the contract.
    """
    from localharness.config.models import AgentConfig, PermissionConfig, ToolConfig

    fields = AgentConfig.model_fields
    required = ", ".join(n for n, f in fields.items() if f.is_required())
    allowed = ", ".join(fields)
    tool_keys = ", ".join(ToolConfig.model_fields)
    perm_keys = ", ".join(PermissionConfig.model_fields)
    # #57(b): state the LEGAL enum values for the fields the prompt names, DERIVED from the
    # Pydantic Literals so the stated contract and the schema (extra='forbid' + Literal) can't
    # drift — the model stops guessing `permissions.mode: read_only` (live) or bad inherit scopes.
    mode_values = ", ".join(_literal_values(PermissionConfig.model_fields["mode"].annotation))
    inherit_values = ", ".join(_literal_values(ToolConfig.model_fields["inherit"].annotation))
    return (
        "Generate a LocalHarness agent YAML config. Return ONLY the YAML, no prose.\n"
        f"Required top-level keys (no defaults): {required}.\n"
        "  - name: lowercase letters, digits and hyphens only, e.g. hn-monitor.\n"
        "  - role: one sentence saying what the agent does.\n"
        "Every other key has a default — omit it unless the user asked for it.\n"
        f"Allowed top-level keys (no others; unknown keys are rejected): {allowed}.\n"
        "Do not wrap the keys under any parent key; every key is top-level.\n"
        "tools and permissions are nested objects — never a bare string or list:\n"
        f"  - tools object keys: {tool_keys} (tools.inherit is a list of: {inherit_values}). "
        "To restrict tools, deny names:\n"
        "      tools:\n"
        "        deny: [bash, write]\n"
        f"  - permissions object keys: {perm_keys} (permissions.mode is one of: {mode_values}).\n\n"
        "Example (a read-only agent):\n"
        "name: hn-monitor\n"
        "role: monitor Hacker News and summarize the top stories each morning\n"
        "tools:\n"
        "  deny: [bash, write]"
    )


class _ReplSwapReporter:
    """What the REPL says and remembers while `lifecycle.swap` runs (spec 20 §8.2, D-19, D-21).

    `freed` — after the accelerator is free, and again if a cancellation stopped the target — forgets
    the heavy the session tracked; the first call also sends the launch line (today's text plus the
    holds-the-GPU clause when an incumbent was stopped, R-13-24), after the session's first-launch
    lines and before the launch, and the decision's note. `launched` / `restored` move `_active_heavy`
    to what is now running. `line` shows a failure sentence, masked here again (R-13-26) on top of the
    swap's own masking."""

    def __init__(self, repl: "OrchestratorREPL", *, target: Any, incumbent: Any, launch_head: str,
                 tradeoff: str, note: str | None, redact: Any) -> None:
        self._repl = repl
        self._target = target
        self._incumbent = incumbent
        self._launch_head = launch_head
        self._tradeoff = tradeoff
        self._note = note
        self._redact = redact
        self._said = False

    async def freed(self) -> None:
        repl = self._repl
        repl._active_heavy = None
        if self._said:
            return
        self._said = True
        from localharness.provider import lifecycle as _lifecycle
        holder = ""
        if self._incumbent is not None and self._incumbent.spec is not self._target.spec:
            holder = GPU_HOLDER_CLAUSE.format(holder=_lifecycle.heavy_label(self._incumbent))
        await repl._flush_launch_lines()
        await repl._send_info(f"{self._launch_head}{holder}{self._tradeoff}", colorize=True)
        if self._note:
            await repl._send_info(self._note)

    async def launched(self, live: Any) -> None:
        self._repl._active_heavy = (self._target.spec, self._target.base_url)

    async def restored(self, spec: Any) -> None:
        self._repl._active_heavy = (spec, self._incumbent.base_url)

    async def line(self, text: str) -> None:
        await self._repl._channel.send_message(self._redact(text), metadata={"style": "system.error"})


class OrchestratorREPL:
    """Interactive REPL for the orchestrator layer.

    Slash commands are deterministic (no LLM). When agent-creation intent is
    detected, drives the AgentCreationWorkflow state machine through conversation.
    All other input is dispatched to the agent loop.
    """

    def __init__(
        self,
        orchestrator: Any,
        agent_loop: Any,
        channel: Any,
        bus: Any,
        config_dir: Path | None = None,
        workspace: Path | None = None,
        harness_config: Any = None,
        on_agent_deployed: Any = None,
        gate: Any = None,
        instructions: Any = None,
        on_plugin_switch: Any = None,
        mcp_manager: Any = None,
        queued: Sequence[str] = (),
        launch_lines: Sequence[str] = (),
        machine_trust: Any = None,
        serving_heavy: tuple[Any, str] | None = None,
        unit_state: Any = None,
    ) -> None:
        from localharness.cli.workspace import MachineTrust
        self._orchestrator = orchestrator
        self._agent = agent_loop
        # The session's PermissionGate (PRD §3.4) — what `/mode` switches. Defaults to the
        # loop's own, which is the same object the loop hands its subagents, so a mode change
        # here reaches a running child. Passed explicitly by start_cmd; None in tests without a
        # loop, where /mode says so rather than raising.
        self._gate = gate
        # The session's per-root project-instructions handle from start, as `gate=` is the session's
        # gate (D-60). None when the workspace has no instructions file: /instructions then says so.
        self._instructions = instructions
        self._channel = channel
        self._bus = bus
        self._config_dir = config_dir
        # The harness's own autostart unit state `start` sensed ONCE for this invocation (D-19), or None:
        # every strategy, swap, free, stop and restore the session builds takes it, so a launch goes
        # through the unit exactly as `start`'s does. The session never senses the unit itself;
        # `UnitStrategy` re-reads the live state before each restart and stop.
        self._unit_state = unit_state
        # The workspace layer that applies to this session, or None. Resolved ONCE by the caller at
        # session start and carried for the session LIFETIME — never re-walked mid-session, even if
        # the user cd's elsewhere in a tool call (39-04's contract). Design D: the REPL never
        # discovers anything itself; the answer arrives here as an argument.
        self._workspace = workspace
        self._harness = harness_config  # HarnessConfig — needed by /model to persist swaps
        # /effort: the session rung of the ladder (None until the person picks one). The REPL sets the
        # LOOP's level only — AgentConfig.effort is the agent file's own rung and is never written here.
        self._session_effort: Optional[str] = None
        # Rejected effort keys already said on the channel this session (D-29): a failed ledger write
        # must not re-announce every turn. Cleared when the session moves to a different server.
        self._announced_effort_keys: set[str] = set()
        # The normalized base_url of the endpoint the ledger entry is being tracked for; None until a
        # live client answers (a test double without one stays None).
        self._effort_endpoint: Optional[str] = self._llm_base_url()
        # #58: called with the deployed agent's name after a successful in-session creation,
        # to register it into the LIVE session (card registry + AgentTool advertisement) so
        # /agents lists it and the model can delegate to it without a restart. None in tests
        # / non-interactive paths that don't wire it.
        self._on_agent_deployed = on_agent_deployed
        # /plugins enable|disable NAME: the session's answer, `await hook(verb, name)` — a line to
        # show, or None to restart (`hook("list", "")` is bare /plugins). start_cmd passes it for a
        # terminal session only; None elsewhere.
        self._on_plugin_switch = on_plugin_switch
        # /mcp: the live MCPClientManager for hot-connect/hot-remove. None in tests.
        self._mcp_manager = mcp_manager
        # Lines typed ahead of a /plugins restart, carried into the rebuilt session: played first.
        self._resume_queue: list[str] = list(queued)
        # The withheld-posture / conversion lines `start` composed but did not print itself (D-15,
        # D-17, D-26): the session's FIRST launch announcement prints them once and clears the list.
        self._pending_launch_lines: list[str] = list(launch_lines)
        # What `start` withheld from the machine's own files (D-11): a swap target is resolved
        # against it, so the swap path cannot run without a trust in hand. A session built without
        # one (a test, a stand-in) has nothing withheld.
        self._machine_trust = machine_trust if machine_trust is not None else MachineTrust()
        # (verb, plugin name) once /plugins asked for a restart; start_cmd reads it after run().
        self.restart_request: tuple[str, str] | None = None
        # --- Type-anytime input box (box mode); inert on the classic path ---
        # A sentence a slash command left for the MODEL, to be started as an ordinary user turn
        # once the command has been handled (`/approve` on an idle session). Consumed and
        # cleared by _dispatch_input; None the rest of the time.
        self._slash_followup: Optional[str] = None
        # Pictures attached for the NEXT turn (/image, Ctrl+V, a dropped path), consumed by
        # _start_user_turn together with whatever the channel's own transport carried.
        self._staged_images: list[dict] = []
        if hasattr(self._channel, "on_paste_image"):
            self._channel.on_paste_image = self._paste_image
        self._turn_task: Optional[asyncio.Task] = None      # the in-flight turn, or None (idle)
        self._current_task: str = ""                          # its originating request (tier-2 context)
        self._fifo: deque[str] = deque()                      # queued messages → future turns (FIFO)
        self._sigint_armed: bool = False                      # idle double-Ctrl+C to exit
        self._cancelled_by_user: bool = False                 # this turn was cancelled by Ctrl+C
        self._box_ctrl_q: Optional[asyncio.Queue] = None      # box → coordinator control events
        # Bus handles for the staging queue (PermissionStaged / PermissionResolved) — the REPL
        # is the only object holding all three pieces this needs: the bus, the gate whose queue
        # is the truth, and the channel that draws it.
        self._pending_handles: list[Any] = []
        self._ctrl_ready_at: float = 0.0                      # when the coordinator last went idle
        # #93: bounded grace on exit for an in-flight turn to reach its own finalization
        # (TurnCompleted publish + ledger flush) before it is cancelled — never hangs exit.
        self._exit_grace_seconds: float = 2.0
        # /model picker (item 9): session cache of the server's live model list, served to the
        # input-menu completer via the channel hook. Warmed by a best-effort prefetch in run()
        # and refreshed on every /model; the menu itself NEVER fetches.
        self._model_cache: list[str] = []
        self._model_prefetch: Optional[asyncio.Task] = None
        if hasattr(self._channel, "model_names_fn"):
            self._channel.model_names_fn = lambda: self._model_cache
        # Phase C2 GPU-lock: the heavy (GPU) server currently occupying the single accelerator, as
        # (spec, base_url), or None. Seeded from the harness-managed primary when it's GPU-bound; a
        # cold-peer heavy-swap moves it, and a swap BACK to the primary moves it back. Tracks the
        # accelerator occupant INDEPENDENT of which endpoint the conversation is bound to (switching
        # to a CPU-light peer leaves the heavy primary up), so the enforceable invariant — the harness
        # never LAUNCHES a heavy server while a heavy it MANAGES is running — holds before a cold
        # heavy launch. BOUNDARY: only a harness-managed primary (server set) is tracked/stoppable;
        # an unmanaged primary (server=None) or an attach-only peer is the operator's to manage — the
        # lock cannot stop what it did not launch.
        self._active_heavy: Optional[tuple[Any, str]] = None
        # The loop's reconnect supplier (`server_recovery`) keeps its last pair here: (key, pair, the
        # active spec held so its id cannot be reused while the memo stands). None until first asked.
        self._recovery_memo: Optional[tuple[tuple, Any, Any]] = None
        # Reports of relaunches that already returned (window lines, the decision's note), sent in the
        # background so a slow channel never holds the recovery flight past its bound. Held here: asyncio
        # keeps only weak references to tasks.
        self._relaunch_reports: set[asyncio.Task] = set()
        # The spec object the managed restart actually launched, paired with the very `_active_heavy`
        # tuple it was launched under: a typed `.gguf` restart launches a COPY of the shared record
        # (the record keeps its own weights), and the relaunch must bring back that copy. Any later
        # assignment of `_active_heavy` makes a new tuple, so a stale pairing is never used.
        self._active_launch: Optional[tuple[Any, Any]] = None
        _srv = getattr(harness_config, "server", None)
        _prov = getattr(harness_config, "provider", None)
        # The lock tracks only what the harness LAUNCHED (D-33): a `server:` record alone proves
        # nothing is up, and a foreign process on its port is not ours to stop. So the seed asks the
        # strategy's own liveness (a pidfile, a container, or — LM Studio — a 2 s HTTP probe, P6, at
        # construction). Any error is "not up" (D-58); no config dir means nothing to ask about.
        # `start` hands over what it attached to or launched (`serving_heavy`, D-19): when that
        # server's own liveness says it is up it IS the accelerator occupant, whichever record it is
        # (a peer chosen by --preset, say); otherwise today's seed from the primary runs.
        # Only a GPU record is the accelerator's occupant (D-33): a CPU-only server that happens to be
        # serving is not ours to stop for the next cold `/model`.
        if (serving_heavy is not None and getattr(serving_heavy[0], "gpu", False)
                and config_dir is not None):
            _held_spec = serving_heavy[0]
            try:
                from localharness.provider.lifecycle import strategy_for
                _held_alive = bool(strategy_for(_held_spec, unit=self._unit_state).liveness(
                    _held_spec, self._server_config_dir).alive)
            except Exception:  # noqa: BLE001 — a liveness that cannot be asked is "not launched"
                _held_alive = False
            if _held_alive:
                self._active_heavy = (serving_heavy[0], serving_heavy[1])
        if (self._active_heavy is None and _srv is not None and getattr(_srv, "gpu", False)
                and _prov is not None and config_dir is not None):
            try:
                from localharness.provider.lifecycle import strategy_for
                _alive = bool(strategy_for(_srv, unit=self._unit_state).liveness(_srv, self._server_config_dir).alive)
            except Exception:  # noqa: BLE001 — a liveness that cannot be asked is "not launched"
                _alive = False
            if _alive:
                self._active_heavy = (_srv, _prov.base_url)

    @property
    def _server_config_dir(self) -> Path | None:
        """Where the vLLM pidfile/log/venv live: ALWAYS the global layer.

        There is one physical GPU daemon per machine (C2 single-pidfile invariant), so server
        lifecycle must never follow a workspace layer even once /model or a peer swap runs
        inside one. None stays None — the callers below already handle an unset config dir.
        """
        from localharness.config.paths import global_config_dir
        return global_config_dir(self._config_dir) if self._config_dir is not None else None

    @property
    def _audit_base_dir(self) -> Path | None:
        """Where this session's audit records land: the WORKSPACE when one applies, else the
        session's config dir. The log follows the work (MEMS-04).

        Deliberately NOT `_server_config_dir`'s sibling — the GPU daemon's state is machine-wide
        and never follows a workspace (C2), so the two properties answer opposite questions from
        the same session and must stay textually distinct.
        """
        return self._workspace if self._workspace is not None else self._config_dir

    @property
    def queued(self) -> tuple[str, ...]:
        """Lines typed ahead and not played yet — what a /plugins restart carries over."""
        return tuple(self._fifo)

    async def run(self) -> None:
        """Entry point. Route to the persistent-input-box loop on a real interactive terminal
        (kill-switch: terminal.inputbox_enabled), else today's classic read_input sequencing."""
        self._model_prefetch = asyncio.create_task(self._prefetch_model_cache())
        self._subscribe_pending()
        try:
            if self._use_input_box():
                await self._run_with_box()
            else:
                await self._run_classic()
        finally:
            self._unsubscribe_pending()
            if self._model_prefetch is not None and not self._model_prefetch.done():
                self._model_prefetch.cancel()
            # A relaunch's window lines and ledger entry still being sent get a short, bounded chance
            # to finish before the event loop tears them down; whatever is left is cancelled.
            try:
                await self._drain_relaunch_reports(timeout=RELAUNCH_REPORT_DRAIN_SECONDS)
            except Exception:  # noqa: BLE001 - leaving the REPL never fails on a best-effort report
                log.debug("relaunch reports could not be drained at exit", exc_info=True)

    def _subscribe_pending(self) -> None:
        """Watch the staging queue change, so the row under the input box can follow it.

        The ROW is the REPL's to draw and nothing else is: it shows the whole queue, oldest
        first, and the gate is the only thing that knows what is left after an answer — an object
        `channels` deliberately cannot import. The transcript LINE stays the channel's own,
        drawn from the channel's own PermissionStaged subscription like every other event a
        person reads, so a channel that already says it (Discord, ACP) never says it twice
        because the REPL is hosting it.
        """
        self._pending_handles = [
            self._bus.subscribe(PermissionStaged, self._on_pending_changed),
            self._bus.subscribe(PermissionResolved, self._on_pending_changed),
        ]
        # A channel that answers a parked call out of band (a Discord reaction) gets the same
        # path the hotkeys take, so the gate records it and the model hears it as words.
        if getattr(self._channel, "_pending_resolver", "absent") is None:
            self._channel._pending_resolver = self._resolve_pending_from_channel
        # The same handshake for the two things a channel with its own out-of-band surface (the
        # mobile channel's POST verbs) needs and cannot reach on its own: steering the RUNNING turn,
        # and cancelling it. Both live on the REPL because the REPL owns the loop handle and the
        # turn task; a channel that started holding either would be a second owner of the one
        # thing that must have exactly one. Installed only when the channel DECLARES the
        # attribute as None — a channel that does not is skipped with no error, which is why
        # `MobileChannel` declares both as class attributes.
        if getattr(self._channel, "_nudge_resolver", "absent") is None:
            self._channel._nudge_resolver = self._nudge_from_channel
        if getattr(self._channel, "_cancel_resolver", "absent") is None:
            self._channel._cancel_resolver = self._cancel_from_channel

    async def _nudge_from_channel(self, text: str, intent: str = "nudge") -> bool:
        """Steer the running turn with a line a channel already classified.

        No classifier is spent: the human pressed the nudge button, so tier-2's up-to-35-second
        permit-wait + timeout budget (bug #92) buys nothing here but the delay it was measured
        causing. Returns False when there is no turn to steer, so the caller can say so.
        """
        clean = (text or "").strip()
        if not clean or not self._turn_running():
            return False
        self._agent.push_user_nudge(clean)
        return True

    async def _cancel_from_channel(self) -> bool:
        """Cancel the in-flight turn from a channel's own surface.

        The same cancel `_await_turn_with_sigint` performs for Ctrl+C: cancelling the TURN, not
        the session. Returns False when nothing is running, so a mis-tap on an idle session is a
        no-op with an honest answer rather than a silent one.
        """
        task = self._turn_task
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def _resolve_pending_from_channel(self, action: str, pending_id: int) -> None:
        await self._answer_pending(str(pending_id), approve=(action == "approve"))

    def _unsubscribe_pending(self) -> None:
        for handle in self._pending_handles:
            try:
                self._bus.unsubscribe(handle)
            except Exception:  # noqa: BLE001 — a stale handle must never fail the exit path
                pass
        self._pending_handles = []

    async def _on_pending_changed(self, event: Any) -> None:
        """A call was parked, or one was answered — redraw the row from the gate's queue.

        Both events land here because the row asks the same question of both: what is waiting
        now? The event carries the one call it is about; the gate is re-read rather than the
        event counted, so a hotkey, a `/approve`, and a blocking prompt in another mode all
        leave the row saying the same true thing.
        """
        self._refresh_pending_row()

    def _refresh_pending_row(self) -> None:
        """Push the gate's queue (oldest first) at the channel's input box, if it has one."""
        setter = getattr(self._channel, "box_set_pending", None)
        if setter is None:
            return  # a channel with no input box: Discord, ACP, a test double
        gate = self._session_gate()
        setter(list(getattr(gate, "pending", {}).values()) if gate is not None else [])

    def _use_input_box(self) -> bool:
        """Box mode only for the real TerminalChannel on an interactive TTY, with the config
        kill-switch on. Mock/scripted channels and non-TTY (pipes, CI) → classic path."""
        from localharness.channels.terminal import TerminalChannel

        if not isinstance(self._channel, TerminalChannel):
            return False
        try:
            if not self._channel.can_run_input_box():
                return False
        except Exception:
            return False
        term = getattr(self._harness, "terminal", None)
        return bool(getattr(term, "inputbox_enabled", True))

    async def _run_classic(self) -> None:
        """Main REPL loop: slash commands, agent-creation workflows, then the agent loop."""
        await self._channel.start()
        await self._establish_workspace_trust()
        # Lines typed ahead of a /plugins restart play first; the ones a replayed /plugins leaves
        # unplayed stay in the queue (`queued`) for the next rebuild.
        self._fifo.extend(self._resume_queue)
        self._resume_queue = []
        try:
            while True:
                try:
                    user_input = self._fifo.popleft() if self._fifo else await self._channel.read_input()
                except EOFError:
                    break
                if not user_input:
                    continue

                try:
                    if self._turn_running():
                        # A turn started outside this loop (a Discord reaction approving a
                        # parked call, `_answer_pending`) must finish before the next line is
                        # dispatched: one AgentLoop, one turn at a time (critic, 2026-09-12).
                        await self._await_turn_with_sigint(self._turn_task)
                    task = await self._dispatch_input(user_input)
                    # v1: the single agent loop handles every turn directly. Multi-agent
                    # routing (AgentCardRegistry.route) will be wired in for dispatch in
                    # MULTI-02 (v2). Run it as a cancellable task so a mid-turn Ctrl+C
                    # cancels the TURN, not the session (#47).
                    if task is not None:
                        self._turn_task = task  # so `_turn_running()` is true here too
                        await self._await_turn_with_sigint(task)
                    # Co-author consent (spec 14): if the git hook flagged a first
                    # commit for an undecided project, ask now (after the turn).
                    await self._settle_pending_coauthor()
                    await self._announce_rejected_effort_keys()
                    # NOTE: Do NOT send_message here. The TaskComplete event handler
                    # in TerminalChannel.on_task_complete() handles output.
                    # Sending here would produce duplicate output.
                except EOFError:
                    break
        finally:
            await self._channel.stop()

    async def _dispatch_input(self, user_input: str) -> Optional[asyncio.Task]:
        """Handle ONE line of user input: slash command, active creation workflow, or creation
        intent — OR publish a UserMessage and START a turn task. Returns the started turn task,
        or None when the line was fully handled without a turn. Shared by classic + box paths;
        may raise EOFError (e.g. /quit) which the caller treats as 'exit the REPL'."""
        # A dropped file: terminals paste the path of a file dragged onto them. One token naming
        # an existing picture is an attachment, not a sentence for the model.
        dropped = _dropped_image_path(user_input)
        if dropped is not None:
            await self._slash_image(dropped, dropped.lower())
            return None

        # Slash commands — deterministic, no LLM
        if user_input.startswith("/"):
            if await self._handle_slash(user_input):
                # A command may leave a sentence the MODEL has to hear: `/approve` on an idle
                # session has nothing to nudge, so its approval becomes an ordinary user turn.
                # It is started here rather than inside the handler because this is the one
                # place that owns "a line of input became a turn", and each caller adopts the
                # returned task its own way (box coordinator, classic loop).
                follow, self._slash_followup = self._slash_followup, None
                return None if follow is None else await self._start_user_turn(follow)

        # PRD §3.4: a chat platform with no slash convention of its own sets `bare_mode_command`,
        # so `mode <name>` as the first word is the same command there. Terminal users type
        # `/mode`; a terminal line starting with the bare word stays a message, because "mode" is
        # an ordinary English word and the terminal already has an unambiguous spelling for it.
        if getattr(self._channel, "bare_mode_command", False):
            head, _, rest = user_input.strip().partition(" ")
            if head.lower() == BARE_MODE_COMMAND:
                await self._handle_mode_cmd(rest.strip())
                return None

        # If a creation workflow is active, drive it
        if self._orchestrator.active_workflow is not None:
            await self._handle_creation_workflow(user_input)
            return None

        # Check for creation intent in natural language
        if self._detect_creation_intent(user_input):
            self._orchestrator.begin_agent_creation(config_dir=self._config_dir)
            # #19: do NOT transition with the trigger message. The workflow already starts in
            # DISCUSS; feeding the trigger here consumed it as the agent DESCRIPTION and silently
            # advanced to CONFIGURE, skipping the CONFIGURE branch — the user's NEXT message is
            # the description.
            await self._channel.send_message(
                "I'd like to help you create an agent. "
                "Tell me more about what you need it to do "
                "(or say 'cancel' to stop).",  # #59: advertise the escape
                metadata={"style": "system.info"},
            )
            return None

        return await self._start_user_turn(user_input)

    async def _start_user_turn(self, text: str) -> Optional[asyncio.Task]:
        """Publish the UserMessage and start the turn task for one line of ordinary input.

        The tail of `_dispatch_input`, named so `/approve` can reach it too: an approval typed at
        an idle session is a sentence the model has to run a turn on, and it must enter the
        session by exactly the path every other typed line does (memory pipeline included), not
        by a private shortcut into `run_turn`."""
        # Publish user message for memory pipeline. channel_id is the adapter's class
        # attribute ("terminal", a chat platform's id, ...) — history rows carry the REAL channel.
        ch_id = getattr(self._channel, "channel_id", None)
        # Pictures: the ones staged here (/image, Ctrl+V, a dropped path — already fitted to the
        # cap when staged) plus the ones the channel's transport carried with this very line (a
        # phone upload, a chat attachment), fitted now. A picture that cannot fit stops the line:
        # half a message, sent silently, is the failure this feature exists to avoid.
        images, self._staged_images = list(self._staged_images), []
        carried = getattr(self._channel, "take_images", list)()
        if not isinstance(carried, list):   # a stand-in without the seam carries nothing
            getattr(carried, "close", lambda: None)()  # an un-awaited coroutine, closed quietly
            carried = []
        if carried:
            try:
                images += fit_all(carried, self._agent._config.context.max_image_tokens)
            except ImageError as exc:
                await self._channel.send_message(f"Not sent: {exc}", metadata={"style": "system.error"})
                return None
        await self._bus.publish(
            UserMessage(
                agent_id=self._agent._config.name,
                session_id=self._agent.current_session_id,
                content=text,
                channel=ch_id if isinstance(ch_id, str) else "terminal",
                attachments=[describe(p) for p in images],
            )
        )
        # Streaming is opt-in BY THE CHANNEL, through a declared flag rather than the presence of
        # a method: a channel that says `streams_tokens` gets the model's answer as it generates,
        # and one that says nothing is driven exactly as before. The terminal says nothing and
        # keeps passing None — it has never streamed answer text — so this adds live text to the
        # mobile channel without touching what any existing surface does.
        on_token = self._channel.on_token if getattr(self._channel, "streams_tokens", False) else None
        # `images=` only when there are any: every agent-loop stand-in that takes (task, on_token)
        # keeps working, and a text-only line is the call it always was.
        extra = {"images": images} if images else {}
        return asyncio.ensure_future(self._agent.run_turn(task=text, on_token=on_token, **extra))

    # ------------------------------------------------------------------ #
    # Persistent type-anytime input box coordinator (box mode)
    # ------------------------------------------------------------------ #

    async def _run_with_box(self) -> None:
        """Box-mode main loop. The persistent input box runs as a sibling task feeding a control
        queue; a single serialized event loop here owns all policy — start turns, route mid-turn
        submissions (nudge now / queue for later), play the FIFO after each turn, handle Ctrl+C
        (cancel the turn) and Ctrl+D (exit). Turn completion posts a 'turn_done' event via the
        task's done-callback, so everything funnels through one queue — no races."""
        await self._channel.start()
        await self._establish_workspace_trust()
        self._box_ctrl_q = asyncio.Queue()
        self._turn_task = None
        self._fifo.clear()
        self._fifo.extend(self._resume_queue)
        self._resume_queue = []
        self._sigint_armed = False
        try:
            await self._channel.start_input_box(self._box_ctrl_q, self._on_box_interrupt)
            self._ctrl_ready_at = time.monotonic()
            if self._fifo:  # lines typed ahead of a /plugins restart: play them first
                self._channel.box_set_queued(len(self._fifo))
                try:
                    await self._play_next_from_fifo()
                except EOFError:
                    return  # a replayed /quit or /plugins ends this REPL too
            while True:
                kind, payload = await self._box_ctrl_q.get()
                if not await self._handle_box_event(kind, payload):
                    break
                # Reference point for interrupt staleness: everything queued while the event
                # above was being handled was typed BEFORE the loop could answer it.
                self._ctrl_ready_at = time.monotonic()
        finally:
            await self._drain_turn_on_exit()
            await self._channel.stop_input_box()
            await self._channel.stop()

    async def _drain_turn_on_exit(self) -> None:
        """#93: on REPL exit (Ctrl+D / double-Ctrl+C), give an in-flight turn a BOUNDED grace to
        reach its OWN finalization before cancelling it.

        An exit typed right after the answer otherwise cancels run_turn in the window between its
        TaskComplete render and its TurnCompleted publish — the turn then has heartbeats but no
        completion in the ledger (#93). run_turn's TurnCompleted publish also writes the durable
        JSONL ledger row at publish-start (before subscribers), so reaching that publish is what
        flushes the turn record. The turn-end micro-pass (a TurnCompleted subscriber) may be cut
        by the grace — acceptable on exit. A user-cancelled turn (_cancelled_by_user) is already
        being torn down; the same bounded await reaps it. Never hangs: past the grace, cancel."""
        task = self._turn_task
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=self._exit_grace_seconds)
        except asyncio.TimeoutError:
            task.cancel()  # grace elapsed — stop it (shield kept wait_for from cancelling)
        except (asyncio.CancelledError, Exception):
            pass  # run_turn owns its errors; an already-cancelled turn resolves here too

    def _on_box_interrupt(self) -> None:
        """Ctrl+C on an empty buffer (called synchronously from the box keybinding). During a
        turn → cancel it instantly (never queued behind routing latency); when idle → hand an
        'interrupt' event to the loop for arm-then-exit."""
        if self._turn_task is not None and not self._turn_task.done():
            self._cancelled_by_user = True
            self._turn_task.cancel()
        elif self._box_ctrl_q is not None:
            # Timestamped: a slash command holds this single coordinator for as long as it runs
            # (a managed /model swap: minutes), so a queued interrupt must be datable to tell
            # "the user is pressing Ctrl+C now" from "the user pressed it while we were blocked".
            self._box_ctrl_q.put_nowait(("interrupt", time.monotonic()))

    async def _handle_box_event(self, kind: str, payload: Any) -> bool:
        """Apply one control event. Returns False to end the REPL."""
        try:
            if kind == "eof":
                return False
            if kind == "interrupt":
                if payload is not None and payload < self._ctrl_ready_at:
                    # Pressed while the loop was blocked (a /model swap can block it for
                    # minutes) → stale by the time it drains. It never reached the user as an
                    # arming message, so counting it toward the ladder would end the session
                    # the instant the wait they were sitting through finished.
                    return True
                # idle Ctrl+C: arm once, exit on the second (mirrors the classic prompt).
                if self._sigint_armed:
                    return False
                self._sigint_armed = True
                await self._channel.send_message(
                    "(Press Ctrl+C again to exit)", metadata={"style": "system.info"}
                )
                return True
            if kind == "turn_done":
                await self._finish_turn(payload)
                self._turn_task = None
                self._channel.box_notify_working(False)
                await self._settle_pending_coauthor()
                await self._announce_rejected_effort_keys()
                await self._play_next_from_fifo()
                return True
            if kind in PENDING_HOTKEYS:
                await self._answer_pending("", approve=PENDING_HOTKEYS[kind])
                return True
            if kind == "tier2_result":
                # #92c: a background tier-2 verdict landed — apply the late upgrade (or let the
                # optimistic queue stand). Serialized here in the one coordinator loop → race-free.
                clean, decision = payload
                await self._apply_tier2_result(clean, decision)
                return True
            if kind == "submit":
                self._sigint_armed = False
                clean, forced = input_router.strip_force(payload)
                if not clean:
                    return True
                if self._turn_task is not None and not self._turn_task.done():
                    await self._route_during_turn(clean, forced)
                else:
                    # FIX 1: persist the prompt in the scrollback the moment it's submitted,
                    # so the turn (rendered above the box) opens with the user's line visible.
                    await self._channel.box_echo_prompt(clean)
                    task = await self._dispatch_input(clean)
                    if task is not None:
                        self._start_turn_task(task, clean)
                return True
        except EOFError:
            return False  # /quit (or a queued /quit) ends the REPL
        return True

    async def _answer_pending(self, arg: str, *, approve: bool) -> None:
        """ctrl+y / ctrl+n from the box (``arg`` empty: the OLDEST parked call) and a Discord
        reaction (``arg`` = the notice's pending id): answer one parked call, right now.

        The same two steps `/approve` takes — the gate records the answer, the model hears it as
        words, nothing here ever dispatches a tool — reached from the control queue instead of
        the slash queue (see :data:`PENDING_HOTKEYS`). An approval at an IDLE session becomes an
        ordinary user turn here, adopted exactly as a typed line's turn is, because
        `_dispatch_input` (which normally starts the one `/approve` leaves behind) is not on this
        path.
        """
        await self._handle_pending_answer(arg, approve=approve)
        follow, self._slash_followup = self._slash_followup, None
        if follow is None:
            return
        task = await self._start_user_turn(follow)
        if task is not None:
            self._start_turn_task(task, follow)

    async def _route_during_turn(self, clean: str, forced: bool) -> None:
        """Decide nudge vs queue for a message typed while a turn runs, deliver it, and reflect
        the decision in the box frame + the InputRouted ledger event.

        #92c: the fast, deterministic decisions (force / slash / tier-1) resolve INLINE. A
        tier-1-abstaining message needs the tier-2 LLM classify, which shares the capacity-1
        inference gate and may wait ~30s for a slot — so it must NOT block this single-threaded
        coordinator. It is optimistically QUEUED at once (owner: uncertain/late → queue) and
        classified in the BACKGROUND; the verdict funnels back through the one control queue and
        may upgrade the message to a live nudge iff it is still queued (see _apply_tier2_result)."""
        # A line with a picture staged is a NEW turn, never a nudge: the running turn's nudge
        # path is text-only, and the picture would otherwise wait for some later line.
        if self._staged_images and not forced:
            await self._deliver_route(
                clean, input_router.Decision(input_router.Route.QUEUE, "tier1", "image-attached"))
            return
        # Slash commands mid-turn are session-level actions — queue them deterministically
        # (run between turns), never spend an LLM classification call on a command.
        if not forced and clean.startswith("/"):
            await self._deliver_route(
                clean, input_router.Decision(input_router.Route.QUEUE, "tier1", "slash-command"))
            return
        if forced:
            await self._deliver_route(
                clean, input_router.Decision(input_router.Route.NUDGE, "force", "force-bang"))
            return
        t1 = input_router.classify_tier1(clean)
        if t1 is not None:
            await self._deliver_route(clean, t1)
            return
        complete_fn = self._tier2_complete_fn()
        if not (self._tier2_enabled() and complete_fn is not None):
            await self._deliver_route(
                clean, input_router.Decision(input_router.Route.QUEUE, "tier1", "abstain-default-queue"))
            return
        # Optimistic queue NOW; the resolved tier-2 verdict is published by _apply_tier2_result.
        await self._queue_message(clean, publish=False)
        asyncio.ensure_future(self._tier2_classify_bg(clean, complete_fn))

    async def _deliver_route(self, clean: str, decision) -> None:
        """Publish a routing decision to the ledger and apply it: NUDGE steers the running turn;
        QUEUE appends to the FIFO. Shared by the inline decisions and the background tier-2 late
        upgrade. FIX 1: the echo into the scrollback is the permanent transcript record; the
        transient border flash is the ephemeral confirmation at the input locus."""
        await self._bus.publish(InputRouted(
            agent_id=self._agent._config.name,
            session_id=self._agent.current_session_id,
            decision=decision.route.value, tier=decision.tier,
            rule_or_reason=decision.reason, text_preview=_route_preview(clean),
        ))
        if decision.route is input_router.Route.NUDGE:
            self._agent.push_user_nudge(clean)
            await self._channel.box_echo_prompt(clean, annotation="→ nudge")
            self._channel.box_flash_decision("→ nudging current turn")
        else:
            await self._queue_message(clean, publish=False)

    async def _queue_message(self, clean: str, *, publish: bool) -> None:
        """Append a message to the FIFO and reflect it in the box frame (queued count + echo)."""
        self._fifo.append(clean)
        self._channel.box_set_queued(len(self._fifo))
        await self._channel.box_echo_prompt(clean, annotation=f"queued ({len(self._fifo)})")
        self._channel.box_flash_decision(f"queued ({len(self._fifo)})")

    async def _tier2_classify_bg(self, clean: str, complete_fn) -> None:
        """#92c: run the tier-2 classify OFF the coordinator, then post the verdict back through
        the one control queue so the late upgrade is applied race-free (single-serialized loop)."""
        try:
            decision = await input_router.classify_tier2(clean, self._turn_context(), complete_fn)
        except Exception:  # classify_tier2 already defaults to QUEUE on error — belt-and-suspenders
            decision = input_router.Decision(input_router.Route.QUEUE, "tier2", "tier2-bg-error-queue")
        q = self._box_ctrl_q
        if q is not None:
            q.put_nowait(("tier2_result", (clean, decision)))

    async def _apply_tier2_result(self, clean: str, decision) -> None:
        """#92c: apply a background tier-2 verdict. A NUDGE upgrades the message to a live nudge
        ONLY if it is still queued AND a turn is running (not yet dispatched); otherwise the
        optimistic queue / dispatch stands (owner: uncertain/late → queue). Either way the
        resolved verdict is recorded to the InputRouted ledger exactly once."""
        upgradable = (
            decision.route is input_router.Route.NUDGE
            and clean in self._fifo
            and self._turn_task is not None
            and not self._turn_task.done()
        )
        if upgradable:
            self._fifo.remove(clean)
            self._channel.box_set_queued(len(self._fifo))
            await self._deliver_route(clean, decision)  # publishes NUDGE + steers the running turn
        else:
            await self._bus.publish(InputRouted(
                agent_id=self._agent._config.name,
                session_id=self._agent.current_session_id,
                decision=decision.route.value, tier=decision.tier,
                rule_or_reason=decision.reason, text_preview=_route_preview(clean),
            ))

    async def _play_next_from_fifo(self) -> None:
        """After a turn ends, start the next queued message as a turn. Commands/workflow lines
        are handled inline (no turn) and we keep draining until a real turn starts or the FIFO
        empties."""
        while self._fifo:
            text = self._fifo.popleft()
            self._channel.box_set_queued(len(self._fifo))
            # FIX 1: re-echo the queued prompt (plain) as its own turn begins, so the transcript
            # reads chronologically — [queued echo] … [turn starts here with the prompt again].
            await self._channel.box_echo_prompt(text)
            task = await self._dispatch_input(text)
            if task is not None:
                self._start_turn_task(task, text)
                return

    def _start_turn_task(self, task: asyncio.Task, text: str) -> None:
        """Adopt a started turn task in box mode: track it, arm the working glyph, and route its
        completion back through the control queue. No loop SIGINT handler here — the box owns raw
        mode, so Ctrl+C arrives via the pt keybinding (_on_box_interrupt), not a signal."""
        self._turn_task = task
        self._current_task = text
        self._cancelled_by_user = False
        self._channel.box_notify_working(True)
        q = self._box_ctrl_q
        task.add_done_callback(lambda t: q.put_nowait(("turn_done", t)) if q is not None else None)

    async def _finish_turn(self, task: asyncio.Task) -> None:
        """Reap a finished turn: surface a user-cancel truthfully; a normal finish already
        rendered via the TaskComplete handler (never re-print here)."""
        try:
            await task
        except asyncio.CancelledError:
            if self._cancelled_by_user:
                await self._channel.send_message(
                    "Turn cancelled.", metadata={"style": "system.info"}
                )
        except Exception:  # noqa: BLE001 — run_turn owns its errors (TurnFailed → channel);
            log.warning("box turn task raised", exc_info=True)  # a stray one must not kill the REPL
        finally:
            self._cancelled_by_user = False

    def _turn_context(self) -> str:
        """Compact running-turn summary for tier-2: current request + latest step/tool."""
        parts = []
        if self._current_task:
            parts.append(f"request: {self._current_task[:200]}")
        last = getattr(self._channel, "last_activity_summary", "")
        if last:
            parts.append(f"latest step: {last[:120]}")
        return " | ".join(parts)

    def _tier2_enabled(self) -> bool:
        term = getattr(self._harness, "terminal", None)
        return bool(getattr(term, "input_router_tier2_enabled", True))

    def _tier2_complete_fn(self):
        """Bounded one-shot classifier seam over the harness's configured LLM, or None when
        unavailable (tier-1 + queue-default then). Reuses the shared client — a fresh local
        client can't take a 5s timeout (300s floor); input_router bounds the call itself."""
        llm = getattr(self._agent, "_llm", None)
        if llm is None or not hasattr(llm, "complete"):
            return None

        async def _complete(messages: list[dict]) -> str:
            # #92: an INTERNAL classification call — disable_thinking (its bounded budget must not
            # be spent on hidden CoT under a reasoning parser) and gen_timeout=5s bounds GENERATION
            # only (the permit-wait is bounded separately by classify_tier2).
            msg, _usage = await llm.complete(
                messages, tools=None, stream=False, disable_thinking=True, gen_timeout=5.0,
            )
            return getattr(msg, "content", None) or ""

        return _complete

    async def _await_turn_with_sigint(self, turn_task: asyncio.Task) -> None:
        """Await ONE agent turn (classic path) as a cancellable task so a mid-turn Ctrl+C cancels
        the turn, not the session (#47).

        While a turn is in flight the prompt_toolkit input app is NOT holding the terminal
        (it released raw mode when read_input returned), so Ctrl+C arrives as a real SIGINT.
        The default handler raises KeyboardInterrupt out of the event loop, tearing the whole
        session down ('Goodbye.') — the exact inversion users hit. We install a loop-level
        SIGINT handler for the turn's duration that CANCELS the turn task instead. Cancelling
        propagates asyncio.CancelledError through the loop (never caught by run_turn's
        `except Exception`) and closes the in-flight streaming HTTP call, so vLLM aborts
        generation engine-side (the #18 disconnect behavior) — no ghost request.

        Idle Ctrl+C is unchanged: at the prompt the input app's own c-c key binding
        (TerminalChannel.read_input) absorbs it. A second Ctrl+C while a turn is already
        cancelling puts back the handler that was in place before the turn, so a further
        Ctrl+C exits the way the host does (escape hatch). `turn_task` is the already-started
        turn (from _dispatch_input).

        The handler in place before the turn goes back afterwards — not Python's default, which
        is what `remove_signal_handler` alone leaves. When `localharness mobile` hosts this REPL
        that handler is uvicorn's graceful shutdown; with the default in its place, Ctrl-C after
        the first turn raised KeyboardInterrupt out of the loop and killed the process instead
        of stopping it (seen on the real surface, 2026-10-04)."""
        loop = asyncio.get_running_loop()
        interrupts = 0
        cancelled_by_user = False
        previous = signal.getsignal(signal.SIGINT)

        def _restore() -> None:
            try:
                loop.remove_signal_handler(signal.SIGINT)
                if previous is not None:
                    signal.signal(signal.SIGINT, previous)
            except (NotImplementedError, RuntimeError, ValueError, TypeError):
                pass

        def _on_sigint() -> None:
            nonlocal interrupts, cancelled_by_user
            interrupts += 1
            if interrupts == 1:
                cancelled_by_user = True
                turn_task.cancel()
            else:
                _restore()  # escape hatch: a further Ctrl+C reaches the host's handler

        handler_installed = False
        try:
            loop.add_signal_handler(signal.SIGINT, _on_sigint)
            handler_installed = True
        except (NotImplementedError, RuntimeError, ValueError):
            # No loop signal support (non-main thread / some event loops): run the turn
            # anyway; Ctrl+C keeps its prior whole-loop behavior. Never block the turn.
            pass

        try:
            await turn_task
        except asyncio.CancelledError:
            if cancelled_by_user:
                # The turn was cancelled by Ctrl+C — the SESSION survives. Truthful line;
                # send_message stops the thinking spinner / any burst first, so the prompt
                # returns with sane state (no half-rendered spinner).
                await self._channel.send_message(
                    "Turn cancelled.", metadata={"style": "system.info"}
                )
            else:
                # The REPL task itself was cancelled (e.g. shutdown): don't leak the turn.
                turn_task.cancel()
                raise
        finally:
            if handler_installed:
                _restore()

    async def _handle_reasoning_cmd(self, arg: str) -> None:
        """/reasoning [on|off] — toggle the live reasoning stream on a channel that has one.

        Gated on the capability, not on the class: this refused on every non-terminal channel
        until the mobile channel could stream reasoning too, at which point an `isinstance` check
        was refusing a command the surface could perfectly well honour (WEBCH-19).
        """
        if not getattr(self._channel, "has_display_toggles", False):
            await self._channel.send_message(
                "/reasoning needs a channel that renders a reasoning stream; this one does not.",
                metadata={"style": "system.info"},
            )
            return
        if arg in ("on", "off"):
            state = arg == "on"
        elif not arg:
            state = not self._channel.show_reasoning
        else:
            await self._channel.send_message(
                "Usage: /reasoning [on|off]", metadata={"style": "system.info"},
            )
            return
        self._channel.show_reasoning = state
        note = (
            "Reasoning stream: on — the model's thinking prints as dim ⋯ lines while it "
            "generates (needs the server's reasoning parser; persist with "
            "terminal.show_reasoning: true)."
            if state else "Reasoning stream: off."
        )
        await self._channel.send_message(note, metadata={"style": "system.info"})

    async def _handle_verbose_cmd(self, arg: str) -> None:
        """/verbose [on|off] — the detailed view: every tool call itemized with its arguments
        and its own result line (the default groups read/memory/web families into one counter
        line each), plus the reasoning stream. Same as `localharness start --verbose`.

        Capability-gated for the same reason `/reasoning` is (WEBCH-19)."""
        if not getattr(self._channel, "has_display_toggles", False):
            await self._channel.send_message(
                "/verbose needs a channel with a detailed view; this one does not have one.",
                metadata={"style": "system.info"},
            )
            return
        if arg in ("on", "off"):
            state = arg == "on"
        elif not arg:
            state = not self._channel.verbose
        else:
            await self._channel.send_message(
                "Usage: /verbose [on|off]", metadata={"style": "system.info"},
            )
            return
        self._channel.verbose = state
        note = (
            "Verbose: on — every tool call prints its own line with arguments, and the model's "
            "reasoning streams as dim ⋯ lines (start with --verbose to begin a session this way)."
            if state else
            "Verbose: off — read, memory and web calls group into one counter line per family; "
            "reasoning follows /reasoning."
        )
        await self._channel.send_message(note, metadata={"style": "system.info"})

    async def _establish_workspace_trust(self) -> None:
        """Settle this session's workspace trust before the first turn (owner ruling 2026-09-11).

        Called once, right after the channel starts and before anything can be typed, because
        that is the first moment the question can be DRAWN — the gate is attached long before
        the channel is live. A recognized or already-recorded workspace says one quiet line or
        nothing at all; only a root nobody has worked in asks.

        Failures are swallowed on purpose, loudly in the log: a trust store that cannot be read
        must not be the reason a session refuses to start. The mode it leaves behind in that
        case is whatever config asked for, which is the same place a pre-v0.14.1 session began.
        """
        gate = self._session_gate()
        if gate is None:
            return
        from localharness.cli.session_trust import establish_session_trust

        async def _notice_later(text: str) -> None:
            await self._channel.send_message(text, metadata={"style": "system.info"})

        pending: list[str] = []
        try:
            await establish_session_trust(gate, pending.append, instructions=self._instructions)
        except Exception:  # noqa: BLE001 — a broken trust store costs a notice, never the session
            log.warning("could not settle workspace trust; leaving the configured mode alone",
                        exc_info=True)
            return
        for text in pending:
            await _notice_later(text)

        # Co-author consent (spec 14): settle once per project + install the hook.
        # Best-effort — a failure here costs a log line, never the session.
        from localharness.cli.coauthor import settle_coauthor_startup
        try:
            await settle_coauthor_startup(gate, self._workspace, pending.append)
        except Exception:  # noqa: BLE001
            log.warning("could not settle co-author consent", exc_info=True)
        for text in pending:
            await _notice_later(text)

    async def _settle_pending_coauthor(self) -> None:
        """After a turn: if the git hook flagged a first commit for an undecided
        project, ask the consent question now. Best-effort — a failure costs a log
        line, never the session."""
        from localharness.cli.coauthor import settle_pending_coauthor
        gate = self._session_gate()
        if gate is None:
            return
        try:
            await settle_pending_coauthor(gate, lambda t: None)
        except Exception:  # noqa: BLE001
            log.warning("could not settle pending co-author consent", exc_info=True)

    def _session_gate(self) -> Any:
        """The gate `/mode` acts on — the one the running loop and its subagents share."""
        return self._gate if self._gate is not None else getattr(self._agent, "gate", None)

    async def _handle_mode_cmd(self, arg: str) -> None:
        """`/mode [name]` — read or switch the session permission mode.

        With no argument it reports the current mode and every mode that can be set, `auto` and
        `unattended` included. Nothing is refused here any more: the person at this prompt is
        the person the gate protects, and being unable to reach `unattended` without restarting
        the session was the bug, not the guardrail (owner, 2026-09-11). The gate logs the switch.
        """
        gate = self._session_gate()
        if gate is None:
            await self._channel.send_message(
                "No permission gate on this session, so there is no mode to change.",
                metadata={"style": "system.info"},
            )
            return
        name = (arg or "").strip().lower()
        if not name:
            await self._channel.send_message(
                MODE_STATUS_TEMPLATE.format(mode=gate.mode, settable=MODE_SETTABLE_NAMES),
                metadata={"style": "system.info"},
            )
            return
        try:
            gate.set_mode(name, from_channel=True)
        except ValueError as exc:
            await self._channel.send_message(str(exc), metadata={"style": "system.error"})
            return
        await self._channel.send_message(
            MODE_CHANGED_TEMPLATE.format(mode=gate.mode, effect=MODE_EFFECTS[gate.mode]),
            metadata={"style": "system.info"},
        )

    def _llm_base_url(self) -> Optional[str]:
        """The normalized base_url of the agent's live client, or None when the agent has none (a
        test double) or it is not a string."""
        url = getattr(getattr(getattr(self._agent, "_llm", None), "config", None), "base_url", None)
        return url.rstrip("/") if isinstance(url, str) else None

    def _current_effort(self) -> Optional[str]:
        """The level the loop is sending right now, or None when it sends none (or is a double)."""
        level = getattr(self._agent, "effective_effort", None)
        return level if level in LEVELS else None

    def _resolve_effort_now(self, *, base_url: Optional[str] = None) -> Optional[tuple[str, str]]:
        """`resolve_effort` over this session's inputs: the session rung, the agent FILE's rung (read,
        never written), the harness, and `base_url` (the live client's when omitted). None when there
        is no harness, agent or answerable level — a test double — so callers skip silently."""
        agent = self._agent
        if self._harness is None or agent is None:
            return None
        url = base_url if base_url is not None else self._llm_base_url()
        if not isinstance(url, str):
            return None
        file_rung = getattr(getattr(agent, "_config", None), "effort", None)
        level, rung = resolve_effort(
            self._session_effort, file_rung if isinstance(file_rung, str) else None,
            self._harness, url,
        )
        return (level, rung) if level in LEVELS else None

    def _effort_show_lines(self) -> list[str]:
        """`/effort show` (D-05): the loop's level and rung, each agent file that sets its own level,
        and every effort key this server rejected, named verbatim."""
        from localharness.cli import model_ops

        level = getattr(self._agent, "effective_effort", None)
        if level in LEVELS:
            lines = [EFFORT_SET_TEMPLATE.format(
                level=level, rung=getattr(self._agent, "effort_rung", None) or "default")]
        else:
            lines = [EFFORT_NONE_LINE]
        for name, own in model_ops.agents_with_effort(self._config_dir, local_config_dir=self._workspace):
            lines.append(EFFORT_AGENT_LINE.format(name=name, level=own))
        rejected = getattr(getattr(self._agent, "_llm", None), "rejected_effort_keys", None) or ()
        lines += [EFFORT_UNENFORCED_TEMPLATE.format(key=key) for key in sorted(rejected)]
        return lines

    async def _handle_effort_cmd(self, arg: str) -> None:
        """`/effort [off|low|high|max|show]` — set or read the session's reasoning effort.

        Never starts a model turn and never raises. Bare `/effort` opens a picker where one can be
        drawn (the terminal's input box, the phone with a client attached) and otherwise prints the
        levels on one line. An answer from a picker is applied exactly like a typed level.
        """
        if arg == "show":
            await self._send_info("\n".join(self._effort_show_lines()))
            return
        if not arg:
            current = self._current_effort()
            channel = self._channel
            can_pick = getattr(channel, "can_run_picker", None)
            if self._use_input_box() and hasattr(channel, "run_picker"):
                choice = await channel.run_picker(list(LEVELS), current)
            elif callable(can_pick) and can_pick():
                choice = await channel.run_picker(list(LEVELS), current, prompt=EFFORT_PICKER_PROMPT)
            else:
                shown = "  ".join(f"[{lv}]" if lv == current else lv for lv in LEVELS)
                await self._send_info(EFFORT_LIST_TEMPLATE.format(levels=shown))
                return
            if choice is None:
                return  # Esc, expiry or nobody attached: nothing changes and nothing is said
            arg = str(choice)
        if arg not in LEVELS:
            await self._channel.send_message(
                EFFORT_UNKNOWN_TEMPLATE.format(arg=arg), metadata={"style": "system.error"},
            )
            return
        await self._apply_effort(arg)

    async def _apply_effort(self, level: str) -> None:
        """Make `level` the session rung: re-resolve, set the loop, say so, then persist. The session
        change comes first and stands whatever persisting does (D-14)."""
        self._session_effort = level
        resolved = self._resolve_effort_now()
        if resolved is not None and hasattr(self._agent, "set_effort"):
            self._agent.set_effort(*resolved)
            await self._send_info(EFFORT_SET_TEMPLATE.format(level=resolved[0], rung=resolved[1]))
        else:
            await self._send_info(EFFORT_SET_TEMPLATE.format(level=level, rung="session"))
        if self._harness is None:
            return  # nothing to persist into (a session built without a harness config)
        from localharness.cli import model_ops
        try:
            note = await model_ops.persist_effort(
                self._harness, level, base_url=self._agent._llm.config.base_url,
                channel_id=self._channel.channel_id,
                config_dir=self._config_dir, audit_base_dir=self._audit_base_dir,
            )
        except Exception as exc:  # noqa: BLE001 — the session change already happened
            await self._channel.send_message(
                EFFORT_PERSIST_FAILED_TEMPLATE.format(exc=exc), metadata={"style": "system.error"},
            )
            return
        if note:
            await self._send_info(note)

    def _effort_ledger_key(self) -> Optional[tuple[str, str]]:
        """`(provider_type, model)` of the live client — the one ledger key the after-turn hook and
        the rebind clear share. None when the client is a double without them."""
        config = getattr(getattr(self._agent, "_llm", None), "config", None)
        ptype, model = getattr(config, "provider_type", None), getattr(config, "model", None)
        return (ptype, model) if isinstance(ptype, str) and isinstance(model, str) else None

    def _rebind_effort_ledger(self, base_url: str) -> None:
        """A switch to a DIFFERENT endpoint clears the target's rejected-key entry and this session's
        announced set (D-29): a rejection belongs to the server that made it, and two servers can share
        a `(provider_type, model)` key. A same-endpoint refresh leaves both alone. Best-effort."""
        target = base_url.rstrip("/")
        previous = self._effort_endpoint
        self._effort_endpoint = target
        if previous is None or previous == target:
            return
        self._announced_effort_keys.clear()
        key = self._effort_ledger_key()
        if key is None:
            return
        from localharness.provider.speed_stats import default_speed_stats_path, record_rejected_effort_keys
        try:
            record_rejected_effort_keys(default_speed_stats_path(self._config_dir), *key, [])
        except Exception:  # noqa: BLE001 — a ledger error never breaks a swap
            log.warning("could not clear the rejected-effort-key ledger entry", exc_info=True)

    async def _announce_rejected_effort_keys(self) -> None:
        """After a turn: say once, on the channel, each effort key this server refused, and keep the
        ledger's record of them honest (D-29). Never raises.

        The entry GROWS by `live - recorded` (each such key is announced, then the union written) and
        SHRINKS by `(recorded & accepted) - live` — a recorded key this session re-sent and the server
        took. When neither changes nothing is written, so a turn that never re-sent a key (`max`,
        `off`, a first turn with an empty live set) leaves the record as it was.
        """
        try:
            key = self._effort_ledger_key()
            if key is None:
                return
            llm = self._agent._llm
            live = frozenset(getattr(llm, "rejected_effort_keys", None) or ())
            accepted = frozenset(getattr(llm, "accepted_effort_keys", None) or ())
            if self._effort_endpoint is None:
                self._effort_endpoint = self._llm_base_url()
            from localharness.provider.speed_stats import (
                default_speed_stats_path,
                last_rejected_effort_keys,
                record_rejected_effort_keys,
            )
            ledger = default_speed_stats_path(self._config_dir)
            recorded = set(last_rejected_effort_keys(ledger, *key) or [])
            grown = live - recorded
            shrunk = (recorded & accepted) - live
            for k in sorted(grown - self._announced_effort_keys):
                await self._send_info(EFFORT_UNENFORCED_TEMPLATE.format(key=k))
                self._announced_effort_keys.add(k)
            self._announced_effort_keys -= shrunk  # an accepted key is news again if refused later
            if grown or shrunk:
                try:
                    record_rejected_effort_keys(ledger, *key, sorted((recorded | grown) - shrunk))
                except OSError:
                    log.warning("could not record the rejected effort keys", exc_info=True)
        except Exception:  # noqa: BLE001 — a notice must never cost the session
            log.warning("could not announce rejected effort keys", exc_info=True)

    def _turn_running(self) -> bool:
        """Is a turn in flight right now? (The box coordinator's own test, in one place.)

        Read by `/approve` and `/deny` to choose between nudging the running turn and starting a
        fresh one. It is honestly usually False when those commands run: a slash command typed
        mid-turn is QUEUED by `_route_during_turn` and played back after the turn ends, so the
        nudge path is the one a programmatic caller or a classic-mode session takes, not the one
        a person typing into the box normally hits.
        """
        return self._turn_task is not None and not self._turn_task.done()

    async def _handle_pending_cmd(self) -> None:
        """`/pending` — the calls `auto` parked for a human instead of asking about."""
        gate = self._session_gate()
        queue = list(getattr(gate, "pending", {}).values()) if gate is not None else []
        if not queue:
            await self._channel.send_message(NOTHING_PENDING, metadata={"style": "system.info"})
            return
        lines = [PENDING_HEADER.format(total=len(queue))]
        lines += [
            PENDING_LINE.format(id=p.id, label=p.agent_label, rendering=p.rendering)
            for p in queue
        ]
        await self._channel.send_message("\n".join(lines), metadata={"style": "system.info"})

    async def _handle_pending_answer(self, arg: str, *, approve: bool) -> None:
        """`/approve [N]` and `/deny [N]` — answer one parked call; no argument takes the oldest.

        The gate only records the answer; nothing is dispatched here. An approval reaches the
        MODEL as words — a nudge into the running turn, or a fresh user turn when the session is
        idle — so the agent loop stays the only thing that ever runs a tool. A denial needs no
        turn of its own: there is nothing for an idle session to do about it.
        """
        command, verb = PENDING_COMMANDS[approve]
        gate = self._session_gate()
        if gate is None:
            await self._channel.send_message(NOTHING_PENDING, metadata={"style": "system.info"})
            return
        arg = (arg or "").strip()
        if arg and not arg.isdigit():
            await self._channel.send_message(
                PENDING_NEEDS_A_NUMBER.format(verb=command),
                metadata={"style": "system.error"},
            )
            return
        answer = gate.approve if approve else gate.deny
        try:
            answered = await answer(int(arg) if arg else None)
        except KeyError:
            await self._channel.send_message(
                PENDING_NO_SUCH.format(id=f"#{arg}" if arg else "— nothing is waiting"),
                metadata={"style": "system.error"},
            )
            return
        await self._channel.send_message(
            PENDING_ANSWERED_LINE.format(id=answered.id, verb=verb, rendering=answered.rendering),
            metadata={"style": "system.info"},
        )
        template = PENDING_APPROVED_NUDGE if approve else PENDING_DENIED_NUDGE
        nudge = template.format(id=answered.id, rendering=answered.rendering)
        if self._turn_running():
            if self._agent is not None:
                self._agent.push_user_nudge(nudge)
        elif approve:
            self._slash_followup = nudge

    async def _handle_slash(self, cmd: str) -> bool:
        """Handle a slash command from the ONE table (cli/slash_commands.py). Returns True if
        handled, False to pass the line through to the agent."""
        raw = cmd.strip()
        lowered = cmd.lower().strip()
        row = find_row(lowered)
        if row is not None:
            if callable(row.handler):
                await self._run_plugin_slash(row, raw[len(row.name):].strip())
            else:
                # A core row's adapter gets the text after the name as typed and lower-cased;
                # each passes on exactly the slice its command always took.
                await getattr(self, row.handler)(raw[len(row.name):], lowered[len(row.name):])
            return True

        # #48: a single-token "/word" is the COMMAND namespace — reject unknown ones
        # deterministically (no LLM turn) instead of letting them fall through to the
        # orchestrator as chat. Rule: ^/[a-zA-Z0-9_-]+$ — a lone leading-slash token and
        # nothing else. A bare "/", a path ("/tmp/foo", extra slashes), or "/word ..."
        # with more text is NOT claimed and falls through to the agent exactly as before.
        if re.fullmatch(r"/[a-zA-Z0-9_-]+", raw):
            await self._channel.send_message(
                f"Unknown command: {raw} — /help lists commands.",
                metadata={"style": "system.error"},
            )
            return True

        # Not a command — pass through to the orchestrator (natural language / paths).
        return False

    async def _run_plugin_slash(self, row: SlashCommand, args: str) -> None:
        """A plugin's slash row (PAPI-11). Its failure — an exception, a sys.exit(), or a reply
        that is neither text nor a rich renderable — is logged and shown naming the plugin, and the
        session goes on. Text it returns is shown as info; a renderable goes to the channel's
        renderable path (as the core /memory row's trees always did); None or "" shows nothing."""
        try:
            reply = await row.handler(args)
            if reply is not None and not isinstance(reply, str) and not is_renderable(reply):
                raise TypeError(f"returned {type(reply).__name__}, not text or a renderable")
        except (Exception, SystemExit) as exc:  # noqa: BLE001 — a plugin never ends the session
            log.warning("slash command %s from plugin %r failed", row.name, row.plugin, exc_info=True)
            await self._channel.send_message(
                f"{row.name} (plugin {row.plugin}) failed: {type(exc).__name__}: {exc}",
                metadata={"style": "system.error"},
            )
            return
        if reply is None:
            return
        if isinstance(reply, str):
            if reply:
                await self._channel.send_message(reply, metadata={"style": "system.info"})
            return
        await self._channel.send_renderable(reply)

    # One adapter per core row of the slash table (named by its `handler`): `args` is the text
    # after the command name as typed, `args_lower` the same text lower-cased.

    async def _slash_quit(self, args: str, args_lower: str) -> None:
        # #60: mid-wizard, /quit and /exit are handled BEFORE the run-loop's workflow
        # branch, so they used to hard-exit the whole SESSION silently (while bare 'quit'
        # only cancels the wizard). Cancel the CREATION first and stay alive; a repeat
        # /quit (no active workflow now) exits normally.
        if self._orchestrator.active_workflow is not None:
            self._orchestrator._active_workflow = None
            await self._channel.send_message(
                "Agent creation cancelled. /quit again to exit.",
                metadata={"style": "system.info"},
            )
            return
        raise EOFError()

    async def _slash_help(self, args: str, args_lower: str) -> None:
        # Rendered when typed, so plugin rows added after import are listed. The Keys block is
        # added only while the terminal's persistent box is live: a channel without the accessor
        # (the phone, Discord, Zed, test doubles) or with the box off reads as it always did.
        from localharness.channels.keys import detect_terminal, help_block

        block = None
        state_fn = getattr(self._channel, "keys_help_state", None)
        state = state_fn() if callable(state_fn) else None
        if state is not None:
            keys, seen = state
            block = help_block(keys, terminal=detect_terminal(os.environ), shift_enter_seen=seen)
        await self._channel.send_message(help_text(keys_block=block), metadata={"style": "system.info"})

    async def _slash_model(self, args: str, args_lower: str) -> None:
        # The ORIGINAL case — model ids are case-sensitive.
        await self._handle_model_cmd(args.strip())

    async def _slash_reasoning(self, args: str, args_lower: str) -> None:
        await self._handle_reasoning_cmd(args_lower.strip())

    async def _slash_effort(self, args: str, args_lower: str) -> None:
        await self._handle_effort_cmd(args_lower.strip())

    async def _slash_verbose(self, args: str, args_lower: str) -> None:
        await self._handle_verbose_cmd(args_lower.strip())

    async def _slash_mode(self, args: str, args_lower: str) -> None:
        await self._handle_mode_cmd(args_lower.strip())

    async def _slash_plugins(self, args: str, args_lower: str) -> None:
        await self._handle_plugins_cmd(args_lower.strip())

    async def _slash_mcp(self, args: str, args_lower: str) -> None:
        # /mcp — in-session MCP server management (spec 15). Session state is reached
        # via `self`; the handler has no ctx parameter.
        from localharness.cli.mcp_cmd import run_mcp_slash
        await run_mcp_slash(self, args)

    async def _handle_plugins_cmd(self, arg: str) -> None:
        """/plugins lists the plugins; /plugins enable|disable <name> switches one in THIS session.
        The session restarts with the flag changed and the conversation carried over: this REPL ends
        with restart_request set, start_cmd tears the session down in its usual order, runs the
        plugin's setup step on the plain terminal — never in the input box — and rebuilds the
        session from the handle. Only a channel that can_switch_plugins (the terminal) does this."""
        info = {"style": "system.info"}
        if not getattr(self._channel, "can_switch_plugins", False):
            await self._channel.send_message(PLUGINS_TERMINAL_ONLY, metadata=info)
            return
        if self._on_plugin_switch is None:
            await self._channel.send_message(PLUGINS_NO_SESSION, metadata=info)
            return
        words = arg.split()
        if not words:
            await self._channel.send_message(await self._on_plugin_switch("list", ""), metadata=info)
            return
        if len(words) != 2 or words[0] not in ("enable", "disable"):
            await self._channel.send_message(PLUGINS_USAGE, metadata=info)
            return
        verb, name = words
        gate = self._session_gate()
        if gate is not None and getattr(gate, "pending", None):
            await self._channel.send_message(PLUGINS_PARKED, metadata=info)
            return
        line = await self._on_plugin_switch(verb, name)
        if line is not None:
            await self._channel.send_message(line, metadata=info)
            return
        self.restart_request = (verb, name)
        await self._channel.send_message(
            PLUGINS_RESTARTING.format(name=name, state="on" if verb == "enable" else "off"), metadata=info)
        raise EOFError()  # ends the REPL as /quit does; start_cmd runs the step and resumes

    async def _slash_pending(self, args: str, args_lower: str) -> None:
        await self._handle_pending_cmd()

    async def _slash_approve(self, args: str, args_lower: str) -> None:
        await self._handle_pending_answer(args_lower, approve=True)

    async def _slash_deny(self, args: str, args_lower: str) -> None:
        await self._handle_pending_answer(args_lower, approve=False)

    async def _slash_image(self, args: str, args_lower: str) -> None:
        """Attach a picture to the next message: `/image <path>`, `/image` (the clipboard),
        `/image clear`. Staged here; _start_user_turn sends it with the next line."""
        from localharness.core.content import image_part, load_image_file
        arg = args.strip()
        if args_lower.strip() == "clear":
            n, self._staged_images = len(self._staged_images), []
            await self._channel.send_message(f"Dropped {n} staged image(s)." if n else "No images staged.",
                                             metadata={"style": "system.info"})
            return
        try:
            if arg:
                tokens = shlex.split(arg)
                part = load_image_file(tokens[0] if len(tokens) == 1 else arg)
            else:
                from localharness.channels.clipboard import read_clipboard_image
                part = image_part(read_clipboard_image(), name="clipboard")
        except (ImageError, ValueError) as exc:
            await self._channel.send_message(f"Not attached: {exc}", metadata={"style": "system.error"})
            return
        await self._stage_image(part)

    async def _stage_image(self, part: dict) -> None:
        """Fit one picture to `context.max_image_tokens` and stage it, printing the receipt
        (size, cost) or the refusal — the user learns the cost before the line is sent."""
        try:
            part = fit_to_tokens(part, self._agent._config.context.max_image_tokens)
        except ImageError as exc:
            await self._channel.send_message(f"Not attached: {exc}", metadata={"style": "system.error"})
            return
        self._staged_images.append(part)
        n = len(self._staged_images)
        await self._channel.send_message(
            f"📎 {describe(part)} — attached to your next message" + (f" ({n} staged)" if n > 1 else ""),
            metadata={"style": "system.info"})

    async def _paste_image(self) -> None:
        """Ctrl+V in the input box: the clipboard's picture, staged exactly like `/image`."""
        await self._slash_image("", "")

    async def _slash_task(self, args: str, args_lower: str) -> None:
        from localharness.agent.task_record import TaskState
        state = getattr(self._agent, "_task_context", None)
        if not isinstance(state, TaskState):
            text = "No task record."
        elif args_lower.strip() == "clear":
            state.clear()
            text = "Task record cleared; task.json deleted."
        else:
            text = state.show()
        await self._channel.send_message(text, metadata={"style": "system.info"})

    async def _slash_instructions(self, args: str, args_lower: str) -> None:
        """`/instructions off | on | show` — the one shared handler's text, verbatim (D-59, D-60)."""
        from localharness.config.project_records import run_instructions_command

        arg = args_lower.strip()
        text = run_instructions_command(self._instructions, arg)
        await self._channel.send_message(text, metadata={"style": "system.info"})
        if arg == "on":
            # The terminal keeps a change line for its next input bubble; `on` answers it (D-60).
            # A channel without the method (Discord, the phone, a test double) has nothing to clear.
            clear = getattr(self._channel, "clear_instructions_status", None)
            if callable(clear):
                clear()

    async def _slash_agents(self, args: str, args_lower: str) -> None:
        cards = self._orchestrator._card_registry.all_cards()
        if not cards:
            await self._channel.send_message(
                "No agents configured. Describe what you need and I'll create one.",
                metadata={"style": "system.info"},
            )
        else:
            lines = ["Configured agents:"]
            for card in cards:
                status_mark = f"[{card.status}]" if hasattr(card, "status") else ""
                lines.append(f"  {card.name} -- {card.description[:80]} {status_mark}")
            await self._channel.send_message(
                "\n".join(lines),
                metadata={"style": "system.info"},
            )

    # ------------------------------------------------------------------ #
    # /model — list and swap models
    # ------------------------------------------------------------------ #

    def _foreign_incumbent(self, managed: Any, target: str) -> str | None:
        """The imperative sentence when a process the harness did not launch holds the primary's
        port, else None (D-32, D-33, D-58, D-69).

        The port is `server.port`, else `provider.base_url`'s (when that names this machine). The
        listener is looked up FIRST: a free port costs no docker, pidfile or HTTP probe. Only with a
        listener is the strategy's liveness asked — a listener it claims is the harness's own.
        Reads and decides only: nothing here stops, launches or signals anything."""
        from urllib.parse import urlparse

        from localharness.cli import model_ops, workspace
        from localharness.provider import procfs

        provider = getattr(self._harness, "provider", None)
        base_url = getattr(provider, "base_url", None)
        if managed is not None:
            port = getattr(managed, "port", None)
        else:
            try:
                parsed = urlparse(base_url or "")
                port = parsed.port
                local = (parsed.hostname or "") in (
                    "localhost", "127.0.0.1", "::1", "0.0.0.0", "::")
            except ValueError:
                return None
            if not local:
                return None  # a remote primary: a listener here says nothing about it
        if not isinstance(port, int) or not procfs.listeners(port):
            return None
        claimed = False
        if managed is not None:
            try:
                from localharness.provider.lifecycle import strategy_for
                claimed = bool(
                    strategy_for(managed, unit=self._unit_state).liveness(managed, self._server_config_dir).alive)
            except Exception:  # noqa: BLE001 — a liveness that cannot be asked claims nothing
                claimed = False
        if claimed and getattr(row_of(getattr(managed, "runtime", None)), "gguf", False):
            claimed = self._pidfile_process_holds(port)
        if claimed:
            return None
        holder = workspace.port_holder(port)
        if holder is None:
            return None
        llm = getattr(self._agent, "_llm", None)
        served = None
        if llm is not None and base_url and \
                str(llm.config.base_url).rstrip("/") == str(base_url).rstrip("/"):
            served = llm.config.model
        if not served:
            served = model_ops.primary_served_name(managed, provider) or getattr(
                provider, "default_model", None)
        return workspace.incumbent_sentence(
            holder, port=port, form="imperative", served=served, target=target)

    def _pidfile_process_holds(self, port: int) -> bool:
        """False only when the pidfile's live process is known and every listener of ``port`` whose owner
        can be read is neither it nor one of its descendants: the pidfile is shared by the primary and
        every peer, so a live peer must not make a foreign process on the primary's port "ours"
        (IN-54). Anything that cannot be established (no pidfile, a listener of another uid, a capped
        scan) keeps the strategy's answer."""
        from localharness.provider import procfs
        from localharness.provider import server as managed_server
        try:
            pid = managed_server.server_pid(self._server_config_dir)
            if pid is None:
                return True
            owners = {procfs.pid_for_inode(row.inode)
                      for row in procfs.listeners(port) if row.reachable_on_loopback}
            owners.discard(None)
            if not owners:
                return True
            return any(procfs.is_self_or_descendant(owner, pid) for owner in owners)
        except Exception:  # noqa: BLE001 — what cannot be read claims nothing new
            return True

    async def _plan_model_entries(self, llm: Any) -> tuple[Any, dict, set]:
        """`(entries, live, malformed)`: every distinct endpoint probed at once, then the list built off the loop."""
        from localharness.cli import model_ops

        provider = getattr(self._harness, "provider", None)
        peers = getattr(self._harness, "extra_endpoints", None) or []
        urls = {model_ops._norm_url(u): u for u in
                (llm.config.base_url, getattr(provider, "base_url", None), *(ep.base_url for ep in peers)) if u}
        malformed: set[str] = set()

        async def probe(key: str, url: str) -> "tuple[str, ...] | None":
            try:
                ids, reachable = await self._live_models(url)
            except model_ops.MalformedModelListError:
                malformed.add(key)
                return ()
            except Exception:  # noqa: BLE001 — a probe that fails answers nothing, never raises into /model
                return None
            return tuple(ids) if reachable else None

        live = dict(zip(urls, await asyncio.gather(*(probe(k, u) for k, u in urls.items()))))
        entries = await asyncio.to_thread(model_ops.plan_entries, self._harness, live=live,
                                          config_dir=self._server_config_dir, trust=self._machine_trust,
                                          floor=self._live_floor(), unit=self._unit_state,
                                          current=(llm.config.base_url, llm.config.model))
        return entries, live, malformed

    async def _set_up_and_switch(self, entry: Any) -> None:
        """Ask the person at this machine once; on yes write and adopt the entry's peer, then switch to it as to any cold peer."""
        from localharness.agent.gate_types import PermissionRequest
        from localharness.cli import library
        from localharness.cli.workspace import visible_text
        from localharness.config.loader import ConfigLoader

        if not (getattr(self._channel, "local_operator", False) and getattr(self._channel, "can_ask", False)):
            await self._send_info(library.SETUP_ELSEWHERE_LINE.format(name=visible_text(entry.name)))
            return
        cfg = self._server_config_dir
        bullet = await asyncio.to_thread(library.setup_bullet, cfg, entry.setup)
        await self._send_info(bullet)
        answer = await self._channel.ask_permission(PermissionRequest(
            tool_name="model", tool_params={"model": entry.name}, klass="model-setup", key=None, grantable=False,
            reason=library.SETUP_SWITCH_QUESTION, display=f"{bullet}\n{library.SETUP_SWITCH_QUESTION}"))
        if not answer.allowed:
            return
        done = await asyncio.to_thread(library.set_up, cfg, entry.setup.real)
        for line in done.lines:
            await self._send_info(line)
        if done.name is None:
            return
        loader = ConfigLoader(config_dir=cfg, local_config_dir=self._workspace,
                              machine_withheld=self._machine_trust.withheld)
        written = await asyncio.to_thread(loader.load_harness)
        peer = next((ep for ep in written.extra_endpoints if ep.name == done.name), None)
        if peer is not None:
            self._harness.extra_endpoints.append(peer)
            await self._handle_model_cmd(done.name)

    async def _handle_model_cmd(self, arg: str) -> None:
        """List the models `model_ops.plan_entries` offers, or switch to one by number or name."""
        if arg.strip().lower() == "list":
            # /help says "/model — List available models"; typing "/model list" is the natural
            # reading and must list, not error with "Unknown model 'list'" (new-user papercut).
            arg = ""
        llm = getattr(self._agent, "_llm", None)
        if llm is None or self._harness is None or self._config_dir is None:
            await self._send_info("Model switching is unavailable in this session.")
            return

        current = llm.config.model
        from localharness.cli import model_ops
        entries, live, malformed = await self._plan_model_entries(llm)
        cur_key = model_ops._norm_url(llm.config.base_url)
        if cur_key in malformed:
            await self._send_info(
                f"The server at {llm.config.base_url} responded, but the response wasn't "
                "understood — is base_url pointing at an OpenAI-compatible API?"
            )
            return
        managed = self._harness.server
        provider = getattr(self._harness, "provider", None)
        medians = self._measured_medians()
        cur_ptype = self._provider_type_for_base_url(llm.config.base_url)
        prim_ptype = getattr(provider, "provider_type", None)
        prim_served = model_ops.primary_served_name(managed, provider) if managed is not None else None
        self._model_cache[:] = self._menu_items(entries)

        if not arg:
            lines = model_ops.entry_lines(entries)
            if lines == ["Models:"]:
                await self._send_info(
                    f"Could not reach the model server at {llm.config.base_url}. Is it running?"
                    if live.get(cur_key) is None else "No models visible at the endpoint."
                )
                return
            lines.append("Switch with /model <name|number>, or scroll the menu and press Enter.")
            await self._send_info("\n".join(lines), colorize=True)
            open_menu = getattr(self._channel, "box_open_model_menu", None)
            if open_menu is not None:
                open_menu()  # one-Enter picker: menu pops with the first model highlighted
            return

        entry = entries.pick(arg)
        if isinstance(entry, str):
            await self._send_info(entry)
            return
        if entry is None:
            typed = Path(arg).expanduser()
            if managed is None or not typed.exists():
                await self._send_info(f"Unknown model '{arg}'. /model lists what's available.")
                return
            if not (RUNTIMES[managed.runtime].gguf and typed.is_file() and arg.lower().endswith(".gguf")):
                await self._send_info(NOT_A_GGUF_LINE.format(target=arg))
                return
            target = arg
        elif entry.active:
            await self._send_info(f"{entry.name} is already active.")
            return
        elif entry.kind == "setup" and entry.setup is not None:
            await self._set_up_and_switch(entry)
            return
        elif entry.kind not in ("live", "cold"):
            from localharness.cli.workspace import visible_text
            await self._send_info(next(
                (line for line in model_ops.entry_lines(entries) if visible_text(entry.name) in line),
                entry.name))
            return
        else:
            target = entry.name if entry.kind == "live" else entry.record.served

        if entry is not None and entry.kind == "live":
            ep = entry.endpoint
            if model_ops._norm_url(ep.base_url if ep is not None else provider.base_url) == cur_key:
                # Same (current) endpoint → hot-swap, mechanics unchanged.
                llm.config.model = target
                cap = await llm.detect_capabilities()
                note = await self._refresh_token_counter(target)
                saved = await self._persist_landed(llm.config.base_url, target)
                rate = self._measured_note(medians, cur_ptype, target)
                await self._send_info(
                    f"Switched to {target} (tool calling: {cap.tool_call_mode}).{note}"
                    + (f" {rate}." if rate else "")
                    + (SAVED_DEFAULT_NOTE if saved else ""), colorize=True,
                )
                return
            if ep is None:
                from localharness.config.models import EndpointRef
                ep = EndpointRef(
                    name="primary", base_url=provider.base_url, provider_type=provider.provider_type,
                    api_key=provider.api_key)
            # Cross-endpoint → re-point the client at the peer, then reuse the SAME re-probe /
            # refit / counter-rebind machinery as a hot-swap. The token counter is rebound with the
            # PEER's provider_type, so an Ollama target is labeled approximate, not the old vLLM's
            # exact /tokenize contract.
            # #3: pass ep.extra_headers directly — NOT `ep.extra_headers or None`. `None` means
            # "leave unchanged" in rebind_endpoint, so an endpoint with the default {} headers would
            # otherwise silently INHERIT the previous endpoint's headers. An endpoint's identity is
            # exactly its own headers.
            llm.rebind_endpoint(
                ep.base_url, api_key=ep.api_key, extra_headers=ep.extra_headers,
                provider_type=ep.provider_type,
            )
            llm.config.model = target
            cap = await llm.detect_capabilities()
            note = await self._refresh_token_counter(
                target, base_url=ep.base_url, provider_type=ep.provider_type
            )
            saved = await self._persist_landed(ep.base_url, target, endpoint=ep)
            rate = self._measured_note(medians, ep.provider_type, target)
            await self._send_info(
                f"Switched to {target} on {ep.name} — {ep.base_url} "
                f"(tool calling: {cap.tool_call_mode}).{note}"
                + (f" {rate}." if rate else "")
                + (SAVED_DEFAULT_NOTE if saved else ""), colorize=True,
            )
            return

        if entry is not None and entry.kind == "cold" and entry.record.peer is not None:
            # A process the harness did not launch holds the primary's port: say so in one sentence
            # and touch nothing — before any key read, launch decision, stop or announcement (D-33).
            # Off the loop: it reads /proc and may ask systemd or an LM Studio endpoint.
            incumbent = await asyncio.to_thread(self._foreign_incumbent, managed, target)
            if incumbent is not None:
                await self._send_info(incumbent)
                return
            # Cold lifecycle peer (always GPU-heavy — EndpointRef validator) → free the accelerator of
            # any DIFFERENT heavy incumbent the harness MANAGES (the GPU-lock — at most one launched
            # heavy up, so the single per-config pidfile always tracks the sole live launched server),
            # LAUNCH the peer, then rebind like a cross-endpoint switch. A failed launch tears the
            # half-started peer down (shared pidfile → else it orphans) and RESTORES the incumbent, so
            # the box is never left with two heavies or nothing serving. The lock governs servers the
            # harness LAUNCHED; it can't stop an unmanaged/attach-only one (see __init__ seeding note).
            from localharness.provider import lifecycle as _lifecycle
            from localharness.provider.server import (
                effective_posture,
                launch_decision,
                required_key_kwargs,
            )
            ep = entry.record.peer
            # The peer's key file is read here, at its cold launch and from a lifecycle the loader
            # kept (D-23). On success the key is held in memory on the endpoint, so the launch, the
            # rebind and every later probe of this peer carry it; a refused file leaves the peer as
            # Phase 1 would launch it (keyless: loopback, no file on argv) after one line.
            peer_key, key_line = _peer_session_key(ep)
            _hold_peer_session_key(ep, peer_key)
            # The one launch decision, asked BEFORE anything is stopped or announced (D-08): a launch
            # that cannot happen answers in one line and leaves the incumbent exactly as it was.
            launch_kw = required_key_kwargs(ep.lifecycle, ep.api_key)
            keyed = bool(launch_kw)
            decision = launch_decision(ep.lifecycle, keyed, peer=ep.name)
            if decision.refusal is not None:
                await self._channel.send_message(decision.refusal, metadata={"style": "system.error"})
                return
            if key_line:
                await self._send_info(key_line)
            # The swap's target is the record the trust-resolved ladder chose for this peer (D-11,
            # SEC-02): a record the machine withholds is refused here, with spec 16's line, and
            # nothing is stopped or launched. The incumbent is what the session tracks as running.
            serving = model_ops.resolve_serving(
                self._harness, self._machine_trust, ep.name, config_dir=self._config_dir)
            if isinstance(serving, model_ops.ServingRefusal):
                await self._channel.send_message(serving.line, metadata={"style": "system.error"})
                return
            swap_target = model_ops.heavy_for(serving, self._harness)
            swap_incumbent = (model_ops.incumbent_heavy(self._active_heavy, self._harness)
                              if self._active_heavy is not None else None)
            box_note = getattr(self._channel, "box_activity", None)

            def _cold_progress(elapsed: float) -> None:
                if box_note is not None:
                    box_note(f"launching {ep.name} · {int(elapsed)}s")

            tradeoff = self._swap_tradeoff_note(cur_ptype, current, ep.provider_type, target)
            _, posture = effective_posture(ep.lifecycle, keyed=keyed, decision=decision)
            redact = self._channel_redactor(ep.api_key)
            reporter = _ReplSwapReporter(
                self, target=swap_target, incumbent=swap_incumbent,
                launch_head=(f"Launching {ep.name}{f' ({posture})' if posture else ''} — "
                             "the swap can take several minutes..."
                             + (f" {_lifecycle.RUNS_OUTSIDE_AUTOSTART}"
                                if _lifecycle.outside_autostart(ep.lifecycle, self._unit_state) else "")
                             + await self._first_build_note(ep.lifecycle)),
                tradeoff=tradeoff, note=decision.note, redact=redact)
            async with self._launch_lock():
                try:
                    # free the accelerator, launch the peer with its own key (only when its lifecycle
                    # is set to require one), restore the incumbent on a failed launch (D-19, D-20)
                    outcome = await _lifecycle.swap(
                        swap_incumbent, swap_target, config_dir=self._server_config_dir,
                        autosense=getattr(self._harness, "autosense", None), reporter=reporter,
                        redact=redact, on_poll=_cold_progress, unit=self._unit_state,
                        floor=self._live_floor(),
                    )
                finally:
                    if box_note is not None:
                        box_note(None)
            if outcome.error is not None:
                return  # the reporter already said why, redacted
            live_ep = outcome.live
            served = live_ep.served_models[0] if live_ep.served_models else target
            # the launch's window line, on the success path and OUTSIDE the try above: a reporting
            # fault must never read as a failed launch and tear a healthy server down (D-22, D-36)
            await self._announce_window(live_ep, ep.provider_type, served)
            # #3: pass ep.extra_headers directly (see the cross-endpoint branch above) — never None,
            # which would leave the previous endpoint's headers on the client.
            llm.rebind_endpoint(ep.base_url, api_key=ep.api_key, extra_headers=ep.extra_headers,
                                provider_type=ep.provider_type)
            llm.config.model = served
            cap = await llm.detect_capabilities()
            note = await self._refresh_token_counter(
                served, base_url=ep.base_url, provider_type=ep.provider_type
            )
            saved = await self._persist_landed(ep.base_url, served, endpoint=ep)
            rate = self._measured_note(medians, ep.provider_type, served)
            await self._send_info(
                f"Switched to {served} on {ep.name} — {ep.base_url} "
                f"(tool calling: {cap.tool_call_mode}).{note}"
                + (f" {rate}." if rate else "")
                + (SAVED_DEFAULT_NOTE if saved else ""), colorize=True,
            )
            return

        gguf = RUNTIMES[managed.runtime].gguf
        typed_gguf = Path(target).expanduser().is_file() and target.lower().endswith(".gguf")
        # A process the harness did not launch holding the primary's port ends the restart in the same
        # one sentence, before anything is launched or stopped: a launch beside it would be answered by
        # the foreign server and reported as ready (WR-52). Off the loop, like the cold-peer guard.
        incumbent = await asyncio.to_thread(self._foreign_incumbent, managed, target)
        if incumbent is not None:
            await self._send_info(incumbent)
            return
        # Downloaded-but-not-served → managed restart. The managed vLLM serves at the PRIMARY
        # provider base_url; if we're currently on a peer (an in-session cross-endpoint switch),
        # re-point the client back FIRST so the restart's probe + wait_ready target the vLLM, not
        # the peer. A no-op in the common case (current already == primary).
        _prim = getattr(self._harness, "provider", None)
        from localharness.provider.server import (
            RUNTIME_LABELS,
            effective_posture,
            launch_decision,
            required_key_kwargs,
        )
        # The one launch decision, asked BEFORE the rebind-back, the stop of any peer holding the
        # accelerator and the announcement (D-08): a refused restart leaves the client bound where it
        # was and the incumbent serving.
        launch_kw = required_key_kwargs(managed, getattr(_prim, "api_key", None))
        decision = launch_decision(managed, bool(launch_kw))
        if decision.refusal is not None:
            await self._channel.send_message(decision.refusal, metadata={"style": "system.error"})
            return
        # Under the harness's own unit the restart goes through the unit, which serves the record the
        # machine fingerprinted and nothing else (D-20): a typed .gguf, or a name other than the record's
        # on a non-llama.cpp primary, is refused here, BEFORE the rebind-back and before anything is
        # stopped. `UnitStrategy.activate` keeps its own refusal as the backstop.
        from localharness.provider import lifecycle as _lifecycle
        from localharness.provider.lifecycle import free_accelerator
        strategy = _lifecycle.strategy_for(managed, unit=self._unit_state)
        typed_swap = gguf and typed_gguf and target != prim_served
        if isinstance(strategy, _lifecycle.UnitStrategy) and (
                typed_swap or (not gguf and target != prim_served)):
            await self._channel.send_message(
                _lifecycle.UNIT_TYPED_LINE.format(path=target), metadata={"style": "system.error"})
            return
        # And what `serve` would refuse at its start (an `autostart.env` the machine record does not cover),
        # still before anything is stopped: the cold branch's `lifecycle.swap` asks the same question.
        refusal = await asyncio.to_thread(
            _lifecycle.unit_launch_refusal, managed, None, self._unit_state, self._server_config_dir)
        if refusal is not None:
            await self._channel.send_message(
                self._channel_redactor()(_lifecycle.SWAP_REFUSED.format(name=target, why=refusal)),
                metadata={"style": "system.error"})
            return
        # The plan for what the restart will launch, asked at the live floor before the rebind-back and
        # before anything is stopped: a target it refuses leaves the primary serving.
        planned = managed
        if typed_swap:
            from localharness.provider import gguf_header
            planned = managed.model_copy(update={
                "model": str(Path(target).expanduser()), "served_name": gguf_header.preset_name(target)})
        elif not gguf and target != prim_served:
            planned = managed.model_copy(update={"model": target})
        primary_url = _prim.base_url if _prim is not None else llm.config.base_url
        incumbent_heavy = model_ops.incumbent_heavy(self._active_heavy or (managed, primary_url), self._harness)
        target_heavy = model_ops.incumbent_heavy((planned, primary_url), self._harness)
        floor = self._live_floor()
        unplanned = await asyncio.to_thread(
            _lifecycle.swap_refusal, target_heavy, incumbent_heavy, config_dir=self._server_config_dir,
            autosense=getattr(self._harness, "autosense", None), floor=floor,
            unit=self._unit_state)
        if unplanned is not None:
            await self._channel.send_message(
                self._channel_redactor()(unplanned), metadata={"style": "system.error"})
            return
        # Tradeoff BEFORE the rebind-back below — cur_ptype must key where the current model
        # actually ran (possibly a peer), not the primary we are about to re-point at.
        tradeoff = self._swap_tradeoff_note(cur_ptype, current, prim_ptype, target)
        if _prim is not None and llm.config.base_url != _prim.base_url:
            # #3: restore the primary identity FULLY — clear any peer extra_headers with {} (not
            # None, which would leave the peer's headers on the client). ProviderConfig has no
            # headers field, so the primary's identity carries none.
            llm.rebind_endpoint(_prim.base_url, api_key=_prim.api_key, extra_headers={},
                                provider_type=_prim.provider_type)
        # a failure line carries a launch's tail: it is masked before any channel shows it (D-22)
        redact_failure = self._channel_redactor()
        label = RUNTIME_LABELS.get(managed.runtime, managed.runtime)
        _, posture = effective_posture(managed, keyed=bool(launch_kw), decision=decision)
        await self._flush_launch_lines()
        await self._send_info(
            f"Restarting managed {label} with {target}{f' ({posture})' if posture else ''}"
            f" — model load can take several minutes...{await self._first_build_note(planned)}{tradeoff}",
            colorize=True,
        )
        if decision.note:
            await self._send_info(decision.note)
        box_note = getattr(self._channel, "box_activity", None)

        def _swap_progress(elapsed: float) -> None:
            # Rendered in the box status row; a minutes-long load must never look frozen.
            if box_note is not None:
                box_note(f"loading {target} · {int(elapsed)}s")

        # GPU-lock (Phase C2): if a DIFFERENT heavy peer holds the accelerator — we're swapping BACK
        # to the primary from a cold-launched peer — verified-stop it FIRST, in its OWN try so a
        # stop-failure is reported honestly and NOTHING is torn down (the peer is still up). A no-op
        # when the primary is itself the incumbent (the common restart case), whose own stop below
        # reloads the model.
        async with self._launch_lock():
            try:
                freed = await free_accelerator(
                    self._active_heavy, managed, self._server_config_dir, unit=self._unit_state)
            except (RuntimeError, TimeoutError) as exc:
                if box_note is not None:
                    box_note(None)
                await self._channel.send_message(
                    redact_failure(f"Could not free the GPU: {exc}. No change made — the previous "
                                   "model should still be serving."), metadata={"style": "system.error"}
                )
                return
            if freed is not None:
                self._active_heavy = None
            # The shared record's own weights and name, put back if the restart fails (WR-53).
            before = (managed.model, managed.served_name)
            spec = managed
            try:
                # The verified stop can block ~90s worst-case (#100); the strategy runs it OFF the event
                # loop (asyncio.to_thread inside strategy.stop) so heartbeats/the channel stay live while
                # the old server drains. The model mutation MUST land strictly BETWEEN stop and activate:
                # activate's serve_command reads spec.model to build the launch command.
                await strategy.stop(managed, self._server_config_dir)
                # The record already names its weights: the primary's served name never replaces them
                # (a llama.cpp alias never reaches -m; a vLLM --served-model-name alias never becomes
                # the vLLM model, D-70). Only a name other than the primary's (non-llama.cpp), or a typed
                # regular .gguf (llama.cpp, named in memory only, never persisted), changes the model.
                if not gguf and target != prim_served:
                    managed.model = target
                elif typed_swap:
                    # A copy for the launch: the shared record (the primary's served name, the listing,
                    # the swap-back target) keeps its own weights, so a typed file is never "the primary"
                    # (WR-53, IN-57).
                    spec = planned
                # the provider's key, only when the managed server is set to require one (default off)
                ep = await strategy.activate(
                    spec, self._server_config_dir, llm.config.base_url, on_poll=_swap_progress,
                    autosense=getattr(self._harness, "autosense", None), floor=floor, **launch_kw,
                )
                served_window = ep.window.served if ep.window is not None else None
                if floor is not None and served_window is not None and served_window < floor:
                    from localharness.provider.autosense import window_phrase
                    raise RuntimeError(_lifecycle.SWAP_BELOW_FLOOR.format(
                        name=_lifecycle.heavy_label(target_heavy),
                        window=window_phrase(served_window), floor=f"{floor:,}"))
                models = ep.served_models
            except (RuntimeError, TimeoutError) as exc:
                # The primary half-started (still alive on a TimeoutError) — tear it down before restoring
                # a peer incumbent (shared pidfile → else it orphans). It is stopped as LAUNCHED: `spec`
                # (the copy for a typed .gguf, else `managed` still carrying the new model), because the
                # stop proves the record's own launch from its model / served name, and the reset below
                # would hand it the old ones. If we freed a peer (b2 swap-back), restore it; otherwise
                # the primary is simply down (the pre-existing restart-failure mode).
                await self._safe_stop(spec)
                managed.model, managed.served_name = before
                restored, why = await self._restore_incumbent(freed)
                tail = (f" Restored {restored}." + (f" {why}" if why else "") if restored
                        else f" Could not restore it: {why}" if why
                        else " The box now has NO model serving — manual attention needed."
                        if freed
                        else " The managed server is down — retry /model to relaunch.")
                await self._channel.send_message(
                    redact_failure(f"Model swap failed: {exc}{tail}"), metadata={"style": "system.error"}
                )
                return
            except BaseException:
                managed.model, managed.served_name = before
                raise
            finally:
                if box_note is not None:
                    box_note(None)
            self._active_heavy = (managed, llm.config.base_url)
            self._active_launch = (self._active_heavy, spec)  # the copy for a typed .gguf, else managed
        served = models[0] if models else target
        # the restart's window line, after the try/except above (D-22, D-36)
        await self._announce_window(ep, prim_ptype or managed.runtime, served)
        llm.config.model = served
        cap = await llm.detect_capabilities()
        note = await self._refresh_token_counter(served)
        # Swapping back to a non-llama.cpp primary by a served-model-name alias relaunched the record
        # as written; persisting the alias would write it over server.model (D-70), so nothing is.
        alias_swap_back = (not gguf and target == prim_served
                           and managed.model != target)
        saved = (False if alias_swap_back or typed_swap
                 else await self._persist_landed(llm.config.base_url, served))
        rate = self._measured_note(medians, prim_ptype, served)
        await self._send_info(
            f"Switched to {served} (tool calling: {cap.tool_call_mode}).{note}"
            + (f" {rate}." if rate else "")
            + (SAVED_DEFAULT_NOTE if saved else TYPED_SWAP_NOTE if typed_swap else ""), colorize=True,
        )

    def _peer_by_base_url(self, base_url: str):
        """The configured peer ``EndpointRef`` serving ``base_url``, or None (e.g. the primary)."""
        for ep in getattr(self._harness, "extra_endpoints", None) or []:
            if ep.base_url == base_url:
                return ep
        return None

    async def _safe_stop(self, spec) -> None:
        """Best-effort verified-stop of a server, swallowing errors: tears a HALF-STARTED target down
        before an incumbent is restored (the pidfile is one-per-config_dir, so a still-alive orphan
        would be clobbered-and-lost by the restore's own launch). The body is `lifecycle.safe_stop`,
        the one every surface's swap shares."""
        from localharness.provider import lifecycle as _lifecycle
        await _lifecycle.safe_stop(spec, self._server_config_dir, unit=self._unit_state)

    async def _announce_window(self, live: Any, provider_type: str, model_id: str) -> None:
        """The swap report names the window a launch got — what ``activate`` asked and the probe
        verified (``LiveEndpoint.window``), never a recomputation — always (no change gate), as one
        line with no undo because a launch report never carries line 2, and records the triple under
        ``(provider_type, model_id)`` in the ledger (D-22, D-36). A launch with no window (a stand-in
        harness without ``autosense``, vLLM) says and records nothing; a probe that found nothing or
        a smaller window than asked is a completed swap with a longer line, never a failure (D-19).
        A ledger write failure never breaks a swap."""
        report = getattr(live, "window", None)
        if report is None:
            return
        from localharness.config.models import AutosenseConfig
        from localharness.provider.autosense import window_lines
        from localharness.provider.speed_stats import default_speed_stats_path, record_window_report
        ratios = getattr(self._harness, "autosense", None) or AutosenseConfig()
        for line in window_lines(report, ratios):
            await self._send_info(line)
        if report.served is None and report.trained is None and report.fits is None:
            return  # a launch that learned nothing must not overwrite a good recorded baseline with Nones
        try:
            record_window_report(default_speed_stats_path(self._config_dir), provider_type, model_id,
                                 report.served, report.trained, report.fits)
        except OSError:
            log.debug("window ledger write failed", exc_info=True)

    async def _restore_incumbent(self, stopped) -> tuple[str | None, str | None]:
        """Best-effort re-launch of a heavy incumbent stopped for a swap that then FAILED, so the box
        is never left with nothing serving. ``stopped`` is the ``(spec, base_url)`` free_accelerator
        returned, or None. Returns ``(label, note)`` on success (and restores ``_active_heavy``) —
        ``note`` is the launch decision's one line when it withheld a widening — and
        ``(None, why)`` when the decision refuses the relaunch or the launch itself fails, so the caller
        reports the cause instead of "nothing serving" (D-08). ``(None, None)`` is nothing to restore:
        the caller then reports the box needs manual attention.

        The relaunch is `lifecycle.restore_incumbent`; this wrapper keeps the REPL's part: the
        session's first-launch lines go out first, and the launch lock is held around the relaunch
        (re-entrant for a caller that already holds it). A cold swap does not come through here: its
        restore runs inside `lifecycle.swap`, under the one lock the swap's caller holds."""
        if stopped is None:
            return None, None
        spec, base_url = stopped
        from localharness.cli import model_ops
        from localharness.provider import lifecycle as _lifecycle
        from localharness.provider.server import launch_decision
        heavy = model_ops.incumbent_heavy(stopped, self._harness)
        try:
            if self._pending_launch_lines and launch_decision(
                    spec, bool(heavy.launch_kw), peer=heavy.peer).refusal is None:
                await self._flush_launch_lines()  # a restore that is the session's first launch
            async with self._launch_lock():
                label, note = await _lifecycle.restore_incumbent(
                    heavy, config_dir=self._server_config_dir,
                    autosense=getattr(self._harness, "autosense", None), unit=self._unit_state)
                if label is not None:
                    self._active_heavy = (spec, base_url)
            return label, note
        except Exception as exc:  # noqa: BLE001 — restore is best-effort; the caller surfaces the failure
            # a fault in the REPL's own part (the line flush, the lock): the same (None, why) shape
            text = str(exc)
            return None, f"{type(exc).__name__}: {text}" if text else type(exc).__name__

    def server_recovery(self):
        """What the loop does when the model server it talks to drops (D-19): the supplier ``start``
        late-binds into the root loop and the runner. Never None. One ``ServerRecovery`` per
        ``(client address, active spec, active address)``, so callers of one pair share one flight;
        any change of the three builds a new pair.

        The ADDRESS rule: the pair is relaunchable only while ``_active_heavy`` names the very server
        the client is using, by exact string equality of ``_active_heavy[1]`` and the client's
        ``base_url``. A switch to an attach-only peer leaves ``_active_heavy`` at the primary, so this
        test is what stops the primary's spec being relaunched against the peer's address. The relaunch
        is the spec object the harness launched (never a re-read of disk or a workspace layer; after a
        typed ``.gguf`` restart, the copy that restart launched, not the shared record) and its
        key is that endpoint's own, fixed here (``_launch_key``): a keyed peer is relaunched with its
        key, a keyless primary lends a keyed peer nothing, and a ``require_api_key`` spec whose key
        resolves to nothing is polled, never relaunched (D-14). Its config dir is its strategy's own
        (``recovery_config_dir``: None for LM Studio). A ``gpu: false`` managed primary is not in
        ``_active_heavy`` but is the harness's: polled with its dir, never relaunched. A hand-run
        primary and an attach-only peer get a plain poll."""
        from localharness.provider.lifecycle import ServerRecovery, strategy_for
        client_base_url = self._agent._llm.config.base_url
        active = self._active_heavy
        # what was launched under this very `_active_heavy`: a typed .gguf restart's copy, else the record
        launched = self._active_launch
        launch_spec = (launched[1] if active is not None and launched is not None and launched[0] is active
                       else active[0] if active else None)
        cfg_dir = self._server_config_dir
        relaunchable = active is not None and active[1] == client_base_url and cfg_dir is not None
        # the endpoint's own key as it is NOW (a cold peer launch re-reads its key file): a changed key
        # builds a new pair. Only a digest enters the memo key, never the key itself.
        launch_key = _launch_key(self._harness, client_base_url, launch_spec) if relaunchable else {}
        key_digest = (hashlib.sha256(repr(sorted(launch_key.items())).encode("utf-8")).hexdigest()
                      if launch_key else None)
        key = (client_base_url, id(active[0]) if active else None, active[1] if active else None,
               id(launch_spec) if launch_spec is not None else None, key_digest)
        memo = self._recovery_memo
        if memo is not None and memo[0] == key:
            return memo[1]
        relaunch = None
        config_dir = None
        runtime = None
        if active is not None and active[1] == client_base_url:
            spec = launch_spec
            runtime = getattr(spec, "runtime", None)
            if cfg_dir is not None:
                config_dir = strategy_for(spec, unit=self._unit_state).recovery_config_dir(cfg_dir)
                if not (getattr(spec, "require_api_key", False) is True and not launch_key):
                    relaunch = self._build_relaunch(
                        spec, client_base_url, cfg_dir, launch_key,
                        self._peer_name_for(spec, client_base_url),
                        self._provider_type_for_base_url(client_base_url))
        else:
            managed = getattr(self._harness, "server", None)
            provider = getattr(self._harness, "provider", None)
            if (managed is not None and not getattr(managed, "gpu", False) and provider is not None
                    and provider.base_url == client_base_url):
                runtime = getattr(managed, "runtime", None)
                if cfg_dir is not None:
                    config_dir = strategy_for(managed, unit=self._unit_state).recovery_config_dir(cfg_dir)
        pair = ServerRecovery(client_base_url, config_dir, relaunch, runtime)
        # the active spec and the launched one are held so their ids cannot be reused while the memo stands
        self._recovery_memo = (key, pair, (active[0] if active else None, launch_spec))
        return pair

    def _launch_lock(self):
        """The launch lock for this session's global config dir (D-47): every REPL launch takes it, and
        it is re-entrant for the task that holds it, so a swap that fails and restores its incumbent
        nests without waiting on itself. When another LocalHarness holds it past the queue-visibility
        threshold the person is told once (in words that say whether the holder is this session's own
        launch or another process's). A REPL built with no config dir has nothing to lock."""
        cfg = self._server_config_dir
        if cfg is None:
            return contextlib.nullcontext()
        from localharness.provider.lifecycle import launch_lock, launch_lock_wait_line
        # the line is chosen when it is said: a holder in this very process is not "another LocalHarness"
        return launch_lock(cfg, on_wait=lambda: self._send_info(launch_lock_wait_line(cfg)))

    async def _first_build_note(self, spec: Any) -> str:
        from localharness.cli.start_serving import FIRST_BUILD_CLAUSE
        from localharness.provider import lifecycle as _lifecycle

        building = await asyncio.to_thread(_lifecycle.first_build, spec, self._server_config_dir)
        return f" ({FIRST_BUILD_CLAUSE})" if building else ""

    def _live_floor(self) -> int | None:
        floor = getattr(self._agent, "floor", None)
        return floor() if callable(floor) else None

    def _channel_redactor(self, *keys: Any):
        """The one-argument masking function a failure line goes through before any channel shows it
        (D-22, SEC-04): `model_ops.redact_for_channel` with the session's keys — the ones given
        (a peer's held session key) and the harness provider's (R-13-26). The literal "none" is a
        proposer's "no key", never a secret: masking it would hide every "none" around it."""
        from localharness.cli import model_ops
        provider = getattr(self._harness, "provider", None)
        held = []
        for key in (*keys, getattr(provider, "api_key", None)):
            raw = key.get_secret_value() if hasattr(key, "get_secret_value") else key
            if isinstance(raw, str) and raw and raw != "none":
                held.append(raw)

        def redact(text: str) -> str:
            for key in held or [None]:
                text = model_ops.redact_for_channel(text, self._harness, key)
            return text

        return redact

    def _peer_name_for(self, spec: Any, base_url: str) -> str | None:
        """The configured peer's name for ``base_url``, only when ``spec`` IS that peer's own lifecycle
        (identity, not equality); None for the primary or a spec that merely equals a peer's."""
        peer_ep = self._peer_by_base_url(base_url)
        if peer_ep is not None and getattr(peer_ep, "lifecycle", None) is spec:
            return peer_ep.name
        return None

    def _build_relaunch(self, spec: Any, base_url: str, cfg_dir: Path | None,
                        launch_key: dict[str, str], peer: str | None, provider_type: str | None):
        """The zero-argument relaunch of the server the harness launched at ``base_url`` (D-22): the
        REPL's sixth launch path. It activates ``spec`` itself (the object it was built from, never a
        re-read of disk) through the same strategy, with the same ``peer=`` and ``autosense=`` the cold
        branch passes and the key fixed when the pair was built. The launch decision, the loopback-last
        argv and the key in the child's environment all come from ``activate``; a refusal or failure
        propagates to the recovery flight (D-24). It takes the launch lock and, under it, asks again
        whether a server is already up (another process may have brought one up while this waited): if
        so it returns None and launches nothing."""

        async def relaunch():
            from localharness.provider import lifecycle as _lifecycle
            from localharness.provider import server as managed_server
            strategy = _lifecycle.strategy_for(spec, unit=self._unit_state)

            def _already_up() -> bool:
                try:
                    if strategy.liveness(spec, cfg_dir).alive:
                        return True
                except Exception:  # noqa: BLE001 — a liveness that cannot be asked is "not up"
                    pass
                return managed_server.server_pid(cfg_dir) is not None

            async with self._launch_lock():
                if await asyncio.to_thread(_already_up):
                    return None
                if self._pending_launch_lines and managed_server.launch_decision(
                        spec, bool(launch_key), peer=peer).refusal is None:
                    try:
                        await self._flush_launch_lines()  # a relaunch that is the session's first launch
                    except Exception:  # noqa: BLE001 - a report never stops the launch it describes
                        log.debug("relaunch: the first-launch lines could not be sent", exc_info=True)
                live = await strategy.activate(
                    spec, cfg_dir, base_url, timeout_seconds=_lifecycle.relaunch_budget(), peer=peer,
                    autosense=getattr(self._harness, "autosense", None), **launch_key)
            # The report runs in the background, outside the lock, the launch and the flight's bound:
            # this closure is what the recovery flight awaits under its window, so a slow send (one
            # Discord or phone line per window line) crossing the deadline would otherwise end the turn
            # as "did not come back" while the relaunched server is serving. Returning right after
            # activate leaves no await between a live endpoint and the flight.
            task = asyncio.get_running_loop().create_task(
                self._report_relaunch(live, spec, provider_type))
            reports = self.__dict__.setdefault("_relaunch_reports", set())
            reports.add(task)
            task.add_done_callback(reports.discard)
            return live

        return relaunch

    async def _report_relaunch(self, live: Any, spec: Any, provider_type: str | None) -> None:
        """The relaunch's report: its window lines and ledger, then the decision's note. Best-effort and
        guarded: a channel fault (a Discord send error, a closed phone stream) costs the lines only."""
        try:
            await self._announce_window(
                live, provider_type or spec.runtime,
                live.served_models[0] if live.served_models else spec.model)
            note = getattr(live, "note", None)
            if note:
                await self._send_info(note)
        except Exception:  # noqa: BLE001 - a report never fails a launch that succeeded
            log.debug("relaunch: the launch report could not be sent", exc_info=True)

    async def _drain_relaunch_reports(self, timeout: float | None = None) -> None:
        """Wait for the background relaunch reports still being sent. `run` calls it at exit with a
        bound (`RELAUNCH_REPORT_DRAIN_SECONDS`): a report not sent by then is cancelled, costing only
        its lines. Without a timeout (tests) it waits for all of them. During a turn the report runs
        beside the loop's re-send, so its lines may land among the re-sent answer's; that costs
        ordering only (each line is whole) and is accepted."""
        reports = [task for task in self.__dict__.get("_relaunch_reports", ()) if not task.done()]
        if not reports:
            return
        _, pending = await asyncio.wait(reports, timeout=timeout)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def _provider_type_for_base_url(self, base_url: str) -> str | None:
        """Resolve the runtime ``provider_type`` for ``base_url`` against the CURRENT session config
        — the primary provider first, then any configured peer. #2: used as the token-counter
        default so a same-endpoint hot-swap performed while the session SITS ON A PEER refits with
        the peer's tokenize contract, not the primary's. Trusting ``provider.provider_type`` here
        stranded a SECOND same-peer hop on the primary's stale ptype (an Ollama peer counted as
        vLLM → the counter rebind hard-raised, mislabeling a clean switch as 'couldn't rebind' and
        suggesting a retry that no-ops on the already-active short-circuit)."""
        provider = getattr(self._harness, "provider", None)
        if provider is not None and base_url == provider.base_url:
            return provider.provider_type
        ep = self._peer_by_base_url(base_url)
        return ep.provider_type if ep is not None else None

    async def _persist_landed(self, base_url: str, model: str, *, endpoint: Any = None) -> bool:
        """Persist a completed switch by WHERE it landed, through `model_ops.persist_landed` — the
        one writer of a landing's outcome (D-07): `active_endpoint` (a peer's, or cleared for the
        primary), `last_served`, and the removal of `pending_preset`. Every REPL landing is a "swap":
        a `/model` switch always changes the selection. The primary keeps today's default-model
        overlay path too (issue #22); a PEER records the additive active-endpoint selection that
        never mutates provider/server.

        Returns True only when a new DEFAULT MODEL actually landed on disk — #129: the caller
        announces persistence, so the flag must never be optimistic."""
        from localharness.cli import model_ops
        provider = getattr(self._harness, "provider", None)
        if provider is None:
            return await self._persist_default_model(model)
        # A peer that shares the primary's address is told apart by being the very endpoint switched to.
        record = next((r for r in model_ops.serving_records(self._harness)
                       if endpoint is not None and r.peer is endpoint), None)
        if record is None:
            record = model_ops.record_for_base_url(self._harness, base_url)
        if record is None:
            # #6: an unknown base_url is NEITHER the primary NOR a configured peer. Persisting it via
            # persist_default_model is the EXACT corruption persist_active_endpoint exists to prevent
            # — it would rewrite provider/server.model, so the next `start` builds `vllm serve
            # <peer-model>` on the managed GPU. Persist NOTHING; log loudly. Currently unreachable
            # (callers pass the live client base_url, always the primary or a known peer) — this is a
            # guard against a future caller, not a live path.
            log.warning(
                "/model landed on an unrecognized endpoint %r (model %r) — not persisting: it "
                "matches neither the primary provider nor any configured extra_endpoints.",
                base_url, model,
            )
            return False
        if record.peer is not None:
            await self._persist_active_endpoint(record.peer, model)
            return False   # a peer selection is not a persisted default model
        try:
            await model_ops.persist_landed(
                self._harness, record, outcome="swap", served=model, config_dir=self._config_dir)
        except Exception as exc:  # noqa: BLE001 — the in-session switch already succeeded
            await self._channel.send_message(
                PERSIST_SELECTION_FAILED.format(exc=self._channel_redactor()(str(exc))),
                metadata={"style": "system.error"},
            )
        return await self._persist_default_model(model)

    async def _persist_active_endpoint(self, endpoint, model: str) -> None:
        """Persist a cross-endpoint (peer) switch to the atomic user overlay as an additive
        active_endpoint record — never corrupts the primary provider/server — through
        `model_ops.persist_landed` with the landing's other effects (`last_served`, the removal of
        `pending_preset`). Best-effort: a failure is surfaced but never crashes the live session,
        which has already switched."""
        from localharness.cli import model_ops
        try:
            record = next((r for r in model_ops.serving_records(self._harness) if r.peer is endpoint), None)
            if record is None:
                record = model_ops.record_for_base_url(self._harness, endpoint.base_url)
            if record is None:
                raise ValueError(f"{endpoint.base_url} is not a configured endpoint")
            await model_ops.persist_landed(
                self._harness, record, outcome="swap", served=model, config_dir=self._config_dir
            )
        except Exception as exc:  # noqa: BLE001 — the in-session switch already succeeded
            await self._channel.send_message(
                "Switched for this session, but persisting the active endpoint failed: "
                f"{self._channel_redactor(getattr(endpoint, 'api_key', None))(str(exc))}",
                metadata={"style": "system.error"},
            )

    async def _refresh_token_counter(
        self, model: str, *, base_url: str | None = None, provider_type: str | None = None
    ) -> str:
        """After a swap, refit the context-window budget (#31) and rebind the shared TokenCounter
        (#25/#30) to the new served model. The counter is ONE object shared by the context manager,
        compaction pipeline and subagent runner, so an in-place rebind/refit updates them all. Both
        probes BLOCK (urllib/httpx; worst case ~40s: two content shapes plus the message-level
        capability probe, which is one call for vLLM and two for llama.cpp, 10s timeout each),
        so they run OFF the event loop (#32)
        — the Discord adapter and idle consolidation share it. Never aborts a completed swap; returns
        a disclosure string (leading space; '' when clean) to append to the switch message so the
        user is told on the CHANNEL — not a swallowed log line — when counting/budget can't track."""
        import asyncio

        from localharness.agent import context as context_mod

        ctx = getattr(self._agent, "_ctx", None)
        # Cross-endpoint swap passes the TARGET endpoint's base_url + provider_type so the window
        # refit and counter rebind use the RIGHT runtime's tokenize contract (an Ollama target must
        # rebind ptype="ollama" → labeled approximate, not the old vLLM). Default to the current
        # primary (existing same-endpoint behavior) when a caller omits them.
        base_url = base_url if base_url is not None else self._agent._llm.config.base_url
        # #2: default the provider_type by RESOLVING the (possibly-peer) base_url the session is on,
        # NOT by unconditionally trusting the primary provider — else a same-endpoint hot-swap while
        # sitting on a peer would rebind the counter with the primary's stale ptype.
        ptype = (
            provider_type
            if provider_type is not None
            else self._provider_type_for_base_url(base_url)
        )
        notes: list[str] = []

        # #31: refit the budget to the new model's served window, or disclose when unknowable — a
        # stale 128K budget on a 32K model passes over-window requests that 400 mid-session. The
        # ContextManager reads max_context_tokens live, so the in-place mutation updates every
        # consumer (TokenBudget gates, compaction thresholds, emergency floor).
        # #132: an explicit per-model pin OUTRANKS the probe — it is configuration, not discovery.
        # It is also the only answer when a runtime's window is undiscoverable (the probe path below
        # can only disclose and keep a budget that belongs to the model we just left).
        #
        # Resolved through ContextConfig.resolve_budget — the SAME call `start` and `doctor` make
        # (#137) — rather than by reading `model_context_overrides` here. A second hand-rolled copy
        # of the pin lookup is exactly how doctor once blessed a 61,440 scalar for a session running
        # on a 40,000 pin.
        #
        # HONEST SCOPE: this binds the PINNED half. When resolve_budget says UNPINNED, the branch
        # below still refits to the served window, so the live budget can differ from the scalar in
        # config.yaml that start_cmd calls the single source of truth. That is deliberate — a stale
        # budget on a newly-swapped model 400s mid-session and the probe is the only evidence of the
        # new model's real window — and it is disclosed in the note the user gets.
        _cfg_ctx = getattr(getattr(self._agent, "_config", None), "context", None)
        pin, pinned = _cfg_ctx.resolve_budget(model) if _cfg_ctx is not None else (None, False)
        if getattr(ctx, "max_context_tokens", None) and pinned:
            if pin != ctx.max_context_tokens:
                ctx.max_context_tokens = pin
                notes.append(f"context budget pinned to {pin:,} tokens for {model}.")
        elif getattr(ctx, "max_context_tokens", None):
            try:
                window = await asyncio.to_thread(
                    context_mod.probe_served_window, base_url, model, ptype,
                    **_launch_key(getattr(self, "_harness", None), base_url),
                )
            except Exception:  # noqa: BLE001 — a probe error must never brick a done swap
                window = None
            if window:
                # The budget IS the served window — the reply reserve is taken at runtime
                # (agent.context.response_reserve), never a second time here.
                if window != ctx.max_context_tokens:
                    ctx.max_context_tokens = window
                    notes.append(f"context budget refit to the served window ({window:,} tokens).")
            else:
                notes.append(
                    "context budget unchanged — couldn't read this model's served window; "
                    "re-run `localharness init` if its window differs."
                )

        # #145: a CONFIGURED output cap must fit the reserve inside the (possibly new) budget, or
        # prompt + max_tokens overruns the served window and a strict server 400s mid-session.
        # Re-fitted from the agent's configured max_tokens rather than the live value, so a swap
        # DOWN to a small window and back UP restores the full allotment instead of ratcheting
        # the cap down for the session. (Must match start_cmd's — see its LLMConfig construction.)
        _budget = getattr(ctx, "max_context_tokens", None)
        _llm_cfg = getattr(self._agent._llm, "config", None)
        _configured_cap = getattr(getattr(self._agent, "_config", None), "max_tokens", None)
        if _budget and _llm_cfg is not None:
            # Both halves of the ONE shared reserve move together. The context manager sizes the
            # reserve from this number and lets history fill the rest; leaving it at the old
            # window's cap would let history grow past what the new request actually asks for.
            # An unset cap stays unset across the swap — no number is invented for the new
            # window, and the new window's end is the only bound, exactly as before the swap.
            ctx.max_response_tokens = _configured_cap
            _cap = context_mod.clamp_response_tokens(_budget, _configured_cap)
            if _cap != getattr(_llm_cfg, "max_tokens", None):
                _llm_cfg.max_tokens = _cap
                if _cap is not None:
                    notes.append(f"per-reply output cap set to {_cap:,} tokens to fit the window.")

        # #30: rebind the counter off-loop. rebind() is exception-safe (restores the prior binding
        # on a failed re-probe), so a failure leaves an exact, usable counter — but bound to the OLD
        # model, so DISCLOSE it on the channel and tell the user to retry (never a silent swap).
        rebind = getattr(getattr(ctx, "_token_counter", None), "rebind", None)
        if rebind is not None:
            try:
                await asyncio.to_thread(rebind, base_url, model, ptype)
            except Exception as exc:  # noqa: BLE001 — never let a counter refresh strand a done swap
                log.warning("TokenCounter rebind after /model swap failed: %s", exc)
                notes.append(
                    f"token counting could not rebind to {model} and still uses the previous "
                    "model, so counts may not match — re-run /model to retry."
                )

        # The one post-switch point all four /model branches reach (spec §7.2): the level is re-resolved
        # for the TARGET endpoint, so a peer whose lifecycle says `effort: max` moves the loop while
        # the agent file's own rung (AgentConfig.effort) stays what the file says.
        try:
            resolved = self._resolve_effort_now(base_url=base_url)
            if resolved is not None and hasattr(self._agent, "set_effort"):
                self._agent.set_effort(*resolved)
        except Exception:  # noqa: BLE001 — never let an effort refresh strand a completed swap
            log.warning("effort re-resolve after /model swap failed", exc_info=True)
        self._rebind_effort_ledger(base_url)
        return (" " + " ".join(notes)) if notes else ""

    async def _live_models(self, base_url: str) -> tuple[list[str], bool]:
        """Delegate to the shared probe so the REPL and the `localharness model` CLI share ONE
        failure taxonomy (#38 — this kills the diverged duplicate that had no reachable flag).
        Returns ``(model_ids, reachable)``; raises MalformedModelListError on a reached-but-wrong
        body. Off the event loop: the probe blocks (httpx.get, up to ~3s)."""
        import asyncio

        from localharness.cli import model_ops
        return await asyncio.to_thread(model_ops.list_live_models, base_url,
                                       **_launch_key(getattr(self, "_harness", None), base_url))

    @staticmethod
    def _menu_items(entries: Any) -> list:
        """`(name, meta)` for every numbered entry in listing order; a measured rate is a color-banded fragment."""
        from localharness.provider.speed_stats import tps_band

        out = []
        for e in entries.numbered():
            if e.kind == "live":
                head = "serving now" if e.active else "serving"
            else:
                head = _framework_label(e.runtime) + (" · experts in RAM" if e.where == "experts in RAM" else "")
                head += f" · {e.window:,}" if e.window is not None else ""
            if e.rate is None:
                out.append((e.name, head))
            else:
                band = tps_band(float(e.rate.split()[0]))
                out.append((e.name, [("", head + " · "), (f"class:tps-{band}", e.rate.removesuffix(" measured"))]))
        return out

    def _measured_medians(self) -> dict[str, float]:
        """speed_key → median VERIFIED tok/s from the auto-recorded ledger (the client measures
        every cleanly finished stream). Empty on any miss — surfaces fall back to the
        hand-authored registry tps, or show nothing. Never estimates."""
        from localharness.provider.speed_stats import all_median_tps, default_speed_stats_path
        try:
            return all_median_tps(default_speed_stats_path(self._config_dir))
        except Exception:
            return {}

    @staticmethod
    def _measured_note(medians: dict, ptype: str | None, name: str, entry=None) -> str | None:
        """Plain-text 'N t/s measured' for listings/swap notes: ledger median first, the
        registry's hand-measured value (`~`-marked) as fallback, None when neither exists."""
        from localharness.provider.speed_stats import speed_key
        v = medians.get(speed_key(ptype, name)) if ptype else None
        if v is not None:
            return f"{v:.1f} t/s measured"
        tps = getattr(entry, "tps", None) if entry is not None else None
        if tps:
            return f"~{tps:g} t/s measured"
        return None

    def _swap_tradeoff_note(self, cur_ptype: str | None, cur_model: str,
                            tgt_ptype: str | None, tgt_model: str) -> str:
        """One sentence naming the speed tradeoff a HEAVY swap is about to make, from verified
        ledger medians only ('' when neither side has one). The colored per-row view lives in
        the picker; this is the last-chance line where the swap costs minutes of model load.

        The 'N t/s measured' phrasing is LOAD-BEARING, not prose: both callers send this line
        with colorize=True and terminal._RATE_NOTE only bands numbers written that way. Shapes
        that put the number before a bare 't/s' rendered with no band at all."""
        from localharness.provider.speed_stats import speed_key
        m = self._measured_medians()
        cur = m.get(speed_key(cur_ptype, cur_model)) if cur_ptype else None
        tgt = m.get(speed_key(tgt_ptype, tgt_model)) if tgt_ptype else None
        if cur is not None and tgt is not None:
            return f" Speed: {cur:.1f} t/s measured → {tgt:.1f} t/s measured."
        if tgt is not None:
            return f" Target: {tgt:.1f} t/s measured."
        if cur is not None:
            return f" Current model runs {cur:.1f} t/s measured; target unmeasured."
        return ""

    async def _prefetch_model_cache(self) -> None:
        """Warm the /model picker menu without ever blocking the UI: best-effort and silent, so a server
        that cannot be listed leaves the cache empty and /model itself reports why when run."""
        llm = getattr(self._agent, "_llm", None)
        if llm is None or self._harness is None:
            return
        try:
            entries, _, _ = await self._plan_model_entries(llm)
            self._model_cache[:] = self._menu_items(entries)
        except Exception:  # noqa: BLE001 — a prefetch that fails leaves no menu
            log.debug("model menu prefetch failed", exc_info=True)

    async def _persist_default_model(self, model: str) -> bool:
        """Persist the swap to the atomic, audited USER OVERLAY (issue #22 pattern) so the next
        start uses it — replaces the prior full, non-atomic config.yaml rewrite. Best-effort:
        a persistence failure (e.g. the merged config no longer validates) is surfaced but never
        crashes the live session, which has already switched."""
        from localharness.cli import model_ops
        try:
            audit_warning = await model_ops.persist_default_model(
                self._harness, model,
                config_dir=self._config_dir, audit_base_dir=self._audit_base_dir,
                local_operator=getattr(self._channel, "local_operator", False),
            )
        except Exception as exc:  # noqa: BLE001 — the in-session swap already succeeded
            await self._channel.send_message(
                f"Switched for this session, but persisting the new default failed: {exc}",
                metadata={"style": "system.error"},
            )
            return False
        # #37: the overlay write succeeded; a post-write audit-emit failure is a secondary note,
        # not a persist failure — surface it without contradicting the successful switch.
        if audit_warning:
            await self._send_info(audit_warning)
        # Pin trap: name any agent whose yaml pins a concrete model — the persisted default
        # won't reach it next start (per-agent pin wins by design; this only warns).
        pinned = model_ops.pinned_agents(self._config_dir, local_config_dir=self._workspace)
        if pinned:
            lines = ["Note: this new default won't reach these agents until their yaml model pin changes:"]
            lines += [f"  - {name} (pinned to {pin!r})" for name, pin in pinned]
            await self._send_info("\n".join(lines))
        return True

    async def _flush_launch_lines(self) -> None:
        """Print the pending launch lines (D-15, D-17) as one info message each, then clear them, so the
        session's first launch says what it narrowed or converted and no later launch repeats it. Called
        once the launch decision has no refusal and before the announcement."""
        pending, self._pending_launch_lines = self._pending_launch_lines, []
        for line in pending:
            await self._send_info(line)

    async def _send_info(self, text: str, colorize: bool = False) -> None:
        meta: dict = {"style": "system.info"}
        if colorize:
            meta["colorize"] = "rates"  # terminal styles 'N t/s measured' notes by band
        await self._channel.send_message(text, metadata=meta)

    def _detect_creation_intent(self, user_input: str) -> bool:
        """Check if user input signals agent creation intent."""
        lower = user_input.lower()
        return any(trigger in lower for trigger in _CREATION_TRIGGERS)

    async def _handle_creation_workflow(self, user_input: str) -> None:
        """Drive AgentCreationWorkflow state machine with user input.

        Called when self._orchestrator.active_workflow is not None.
        Transitions the workflow, sends appropriate prompts, and handles
        terminal states (DEPLOY, COMPLETE, CANCELLED).
        """
        from localharness.orchestrator.workflow import WorkflowState

        workflow = self._orchestrator.active_workflow
        new_state = workflow.transition(user_input)

        if new_state == WorkflowState.CANCELLED:
            await self._channel.send_message(
                "Agent creation cancelled. Back to normal conversation.",
                metadata={"style": "system.info"},
            )
            self._orchestrator._active_workflow = None
            return

        if new_state == WorkflowState.CONFIGURE:
            # Workflow gathered enough info — use the LLM directly to generate YAML.
            # stream_complete (not run_turn) avoids publishing TaskComplete events (double output).
            import re

            from localharness.orchestrator.workflow import validate_agent_yaml

            gathered = workflow.gathered
            description = gathered.get("description", user_input)
            base_messages = [
                {"role": "system", "content": _generation_system_prompt()},
                {"role": "user", "content": description},
            ]

            async def _generate(msgs: list[dict]) -> str:
                # #18 stream at the transport level; #19 unpack the (message, usage) tuple —
                # reading .content off the tuple itself always yielded "".
                message, _usage = await self._agent._llm.stream_complete(msgs, tools=None)
                raw = getattr(message, "content", "") or ""
                # Strip markdown code fences (LLMs often wrap in ```yaml...```).
                raw = re.sub(r"^```(?:yaml)?\s*\n?", "", raw.strip())
                return re.sub(r"\n?```\s*$", "", raw.strip())

            try:
                yaml_str = await _generate(base_messages)
                err = validate_agent_yaml(yaml_str)
                if err is not None:
                    # #57: NEVER show YAML that will explode at deploy. Regenerate ONCE, feeding
                    # the exact validation error back so the model corrects that field.
                    retry_messages = base_messages + [
                        {"role": "assistant", "content": yaml_str},
                        {"role": "user", "content": (
                            f"That config is invalid: {err}. Fix ONLY that and return the "
                            "corrected YAML, nothing else."
                        )},
                    ]
                    yaml_str = await _generate(retry_messages)
                    err = validate_agent_yaml(yaml_str)
            except Exception as exc:
                # #29: a provider error here must not tear down the whole session
                # (the REPL loop catches only EOFError). Abandon creation truthfully.
                self._orchestrator._active_workflow = None
                await self._channel.send_message(
                    f"Agent generation failed: {exc}\nAgent was NOT created. "
                    "Say 'create an agent' to try again.",
                    metadata={"style": "system.error"},
                )
                return

            if err is not None:
                # #57: still invalid after one correction — abort truthfully. Never present
                # invalid YAML for approval; the short, URL-free error says what went wrong
                # (no raw Pydantic wall / pydantic.dev URL reaches the channel).
                self._orchestrator._active_workflow = None
                await self._channel.send_message(
                    f"Agent generation produced an invalid config twice ({err}). "
                    "Agent was NOT created — creation abandoned. "
                    "Say 'create an agent' to try again.",
                    metadata={"style": "system.error"},
                )
                return

            workflow.set_generated_yaml(yaml_str)
            workflow.transition("configure_done")  # advance to CONFIRM
            await self._channel.send_message(
                f"Here's the generated config:\n\n```yaml\n{yaml_str}\n```\n\n"
                "Does this look good? (yes/no/change, or 'cancel')",  # #59: advertise the escape
                metadata={"style": "system.info"},
            )
            return

        if new_state == WorkflowState.DEPLOY:
            # User confirmed — deploy the config. #19: no step ever gathers a
            # name, and the old 'new_agent' fallback failed AgentConfig's
            # hyphens-only name rule — the default path could never deploy.
            # Pass None so deploy_config honors the confirmed YAML's own name.
            name = workflow.gathered.get("name")
            try:
                config_path = workflow.deploy_config(name)
            except Exception as exc:
                # #27: never claim success on failure. Abandon creation with a
                # truthful message; the session stays alive in normal conversation.
                self._orchestrator._active_workflow = None
                await self._channel.send_message(
                    f"Deploy failed: {exc}\nAgent was NOT created. "
                    "Say 'create an agent' to try again.",
                    metadata={"style": "system.error"},
                )
                return
            # Success path only: advance through aftercare, then confirm creation.
            workflow.transition("deployed")
            # Per user decision: after creation, back to prompt (no auto-handoff)
            workflow.transition("done")
            self._orchestrator._active_workflow = None
            # #58: register the new agent into the LIVE session so /agents lists it and the
            # model can delegate to it without a restart. config_path.stem is the deployed
            # name (deploy_config honors the confirmed YAML's own name). Best-effort: a
            # registration hiccup must not turn a successful deploy into a failure.
            if self._on_agent_deployed is not None:
                try:
                    self._on_agent_deployed(config_path.stem)
                except Exception as exc:  # noqa: BLE001
                    log.warning("post-deploy live registration failed for %s: %s", config_path.stem, exc)
            await self._channel.send_message(
                f"Agent deployed to {config_path}",
                metadata={"style": "system.info"},
            )
            await self._channel.send_message(
                "Agent created. Back to normal conversation.",
                metadata={"style": "system.info"},
            )
            return

        if new_state == WorkflowState.COMPLETE:
            self._orchestrator._active_workflow = None
            return

        # Still in DISCUSS state — ask for more info
        if new_state == WorkflowState.DISCUSS:
            await self._channel.send_message(
                "Tell me more. What tasks should this agent handle? "
                "What tools does it need? (or say 'cancel' to stop)",  # #59: advertise the escape
                metadata={"style": "system.info"},
            )
