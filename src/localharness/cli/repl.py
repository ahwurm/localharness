"""OrchestratorREPL -- interactive prompt_toolkit loop for LocalHarness."""
from __future__ import annotations

import asyncio
import logging
import re
import signal
from collections import deque
from pathlib import Path
from typing import Any, Optional

from localharness.channels import input_router
from localharness.cli.slash_commands import help_text
from localharness.core.events import InputRouted, UserMessage

log = logging.getLogger(__name__)


# Derived from the single-source SLASH_COMMANDS table (shared with the input completion menu).
HELP_TEXT = help_text()


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
        harness_config: Any = None,
        on_agent_deployed: Any = None,
        memory_store: Any = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._agent = agent_loop
        self._channel = channel
        self._bus = bus
        self._config_dir = config_dir
        self._harness = harness_config  # HarnessConfig — needed by /model to persist swaps
        # The agent's opened MemoryStore — the /memory window reads/retires through it. None in
        # tests / in-memory sessions (no persistence) → /memory reports it's unavailable.
        self._store = memory_store
        # #58: called with the deployed agent's name after a successful in-session creation,
        # to register it into the LIVE session (card registry + AgentTool advertisement) so
        # /agents lists it and the model can delegate to it without a restart. None in tests
        # / non-interactive paths that don't wire it.
        self._on_agent_deployed = on_agent_deployed
        # --- Type-anytime input box (box mode); inert on the classic path ---
        self._turn_task: Optional[asyncio.Task] = None      # the in-flight turn, or None (idle)
        self._current_task: str = ""                          # its originating request (tier-2 context)
        self._fifo: deque[str] = deque()                      # queued messages → future turns (FIFO)
        self._sigint_armed: bool = False                      # idle double-Ctrl+C to exit
        self._cancelled_by_user: bool = False                 # this turn was cancelled by Ctrl+C
        self._box_ctrl_q: Optional[asyncio.Queue] = None      # box → coordinator control events
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

    async def run(self) -> None:
        """Entry point. Route to the persistent-input-box loop on a real interactive terminal
        (kill-switch: terminal.inputbox_enabled), else today's classic read_input sequencing."""
        self._model_prefetch = asyncio.create_task(self._prefetch_model_cache())
        try:
            if self._use_input_box():
                await self._run_with_box()
            else:
                await self._run_classic()
        finally:
            if self._model_prefetch is not None and not self._model_prefetch.done():
                self._model_prefetch.cancel()

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
        try:
            while True:
                try:
                    user_input = await self._channel.read_input()
                except EOFError:
                    break
                if not user_input:
                    continue

                try:
                    task = await self._dispatch_input(user_input)
                    # v1: the single agent loop handles every turn directly. Multi-agent
                    # routing (AgentCardRegistry.route) will be wired in for dispatch in
                    # MULTI-02 (v2). Run it as a cancellable task so a mid-turn Ctrl+C
                    # cancels the TURN, not the session (#47).
                    if task is not None:
                        await self._await_turn_with_sigint(task)
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
        # Slash commands — deterministic, no LLM
        if user_input.startswith("/"):
            if await self._handle_slash(user_input):
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

        # Publish user message for memory pipeline. channel_id is the adapter's class
        # attribute ("terminal", "discord", ...) — history rows carry the REAL channel.
        ch_id = getattr(self._channel, "channel_id", None)
        await self._bus.publish(
            UserMessage(
                agent_id=self._agent._config.name,
                session_id=self._agent.current_session_id,
                content=user_input,
                channel=ch_id if isinstance(ch_id, str) else "terminal",
            )
        )
        return asyncio.ensure_future(self._agent.run_turn(task=user_input, on_token=None))

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
        self._box_ctrl_q = asyncio.Queue()
        self._turn_task = None
        self._fifo.clear()
        self._sigint_armed = False
        try:
            await self._channel.start_input_box(self._box_ctrl_q, self._on_box_interrupt)
            while True:
                kind, payload = await self._box_ctrl_q.get()
                if not await self._handle_box_event(kind, payload):
                    break
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
            self._box_ctrl_q.put_nowait(("interrupt", None))

    async def _handle_box_event(self, kind: str, payload: Any) -> bool:
        """Apply one control event. Returns False to end the REPL."""
        try:
            if kind == "eof":
                return False
            if kind == "interrupt":
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
                await self._play_next_from_fifo()
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

    async def _route_during_turn(self, clean: str, forced: bool) -> None:
        """Decide nudge vs queue for a message typed while a turn runs, deliver it, and reflect
        the decision in the box frame + the InputRouted ledger event.

        #92c: the fast, deterministic decisions (force / slash / tier-1) resolve INLINE. A
        tier-1-abstaining message needs the tier-2 LLM classify, which shares the capacity-1
        inference gate and may wait ~30s for a slot — so it must NOT block this single-threaded
        coordinator. It is optimistically QUEUED at once (owner: uncertain/late → queue) and
        classified in the BACKGROUND; the verdict funnels back through the one control queue and
        may upgrade the message to a live nudge iff it is still queued (see _apply_tier2_result)."""
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
            rule_or_reason=decision.reason, text_preview=clean[:80],
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
                rule_or_reason=decision.reason, text_preview=clean[:80],
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
        cancelling restores the default SIGINT handler, so a further Ctrl+C hard-exits
        (escape hatch). `turn_task` is the already-started turn (from _dispatch_input)."""
        loop = asyncio.get_running_loop()
        interrupts = 0
        cancelled_by_user = False

        def _on_sigint() -> None:
            nonlocal interrupts, cancelled_by_user
            interrupts += 1
            if interrupts == 1:
                cancelled_by_user = True
                turn_task.cancel()
            else:
                # Escape hatch: restore the default SIGINT so a further Ctrl+C hard-exits.
                try:
                    loop.remove_signal_handler(signal.SIGINT)
                except (NotImplementedError, RuntimeError, ValueError):
                    pass

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
                try:
                    loop.remove_signal_handler(signal.SIGINT)
                except (NotImplementedError, RuntimeError, ValueError):
                    pass

    async def _handle_slash(self, cmd: str) -> bool:
        """Handle slash commands. Returns True if handled, False to pass through."""
        cmd_lower = cmd.lower().strip()

        if cmd_lower in ("/quit", "/exit"):
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
                return True
            raise EOFError()

        if cmd_lower == "/help":
            await self._channel.send_message(
                HELP_TEXT,
                metadata={"style": "system.info"},
            )
            return True

        if cmd_lower == "/model" or cmd_lower.startswith("/model "):
            # Slice the ORIGINAL string — model ids are case-sensitive.
            await self._handle_model_cmd(cmd.strip()[len("/model"):].strip())
            return True

        if cmd_lower == "/memory" or cmd_lower.startswith("/memory "):
            # Slice the ORIGINAL string — ids and search words are case-sensitive. Claimed here,
            # BEFORE the unknown-/word reject below, so bare "/memory" isn't refused as unknown.
            await self._handle_memory_cmd(cmd.strip()[len("/memory"):].strip())
            return True

        if cmd_lower == "/agents":
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
            return True

        # #48: a single-token "/word" is the COMMAND namespace — reject unknown ones
        # deterministically (no LLM turn) instead of letting them fall through to the
        # orchestrator as chat. Rule: ^/[a-zA-Z0-9_-]+$ — a lone leading-slash token and
        # nothing else. A bare "/", a path ("/tmp/foo", extra slashes), or "/word ..."
        # with more text is NOT claimed and falls through to the agent exactly as before.
        stripped = cmd.strip()
        if re.fullmatch(r"/[a-zA-Z0-9_-]+", stripped):
            await self._channel.send_message(
                f"Unknown command: {stripped} — /help lists commands.",
                metadata={"style": "system.error"},
            )
            return True

        # Not a command — pass through to the orchestrator (natural language / paths).
        return False

    # ------------------------------------------------------------------ #
    # /model — list and swap models
    # ------------------------------------------------------------------ #

    async def _handle_model_cmd(self, arg: str) -> None:
        """List models or switch. A model already served by the endpoint hot-swaps
        (Ollama serves many); a different downloaded model on a harness-managed
        vLLM triggers a server restart (vLLM serves one at a time)."""
        llm = getattr(self._agent, "_llm", None)
        if llm is None or self._harness is None or self._config_dir is None:
            await self._send_info("Model switching is unavailable in this session.")
            return

        current = llm.config.model
        from localharness.cli import model_ops
        try:
            live, reachable = await self._live_models(llm.config.base_url)
        except model_ops.MalformedModelListError:
            # #38: reached but the reply isn't a model list — its OWN message, not "no models".
            await self._send_info(
                f"The server at {llm.config.base_url} responded, but the response wasn't "
                "understood — is base_url pointing at an OpenAI-compatible API?"
            )
            return
        managed = self._harness.server
        downloaded: list[str] = []
        if managed is not None:
            from localharness.provider import server as managed_server
            registry = [e.name for e in managed.local_models if e.name not in live]
            hf_cached = [m for m in managed_server.list_cached_models()
                         if m not in live and m not in registry]
            downloaded = registry + hf_cached
        choices = live + downloaded
        if reachable:
            self._model_cache[:] = choices  # picker menu offers swappable models too

        if not arg:
            if not choices:
                # #38: distinguish an unreachable runtime from a reached-but-empty one.
                if not reachable:
                    await self._send_info(
                        f"Could not reach the model server at {llm.config.base_url}. Is it running?"
                    )
                else:
                    await self._send_info(
                        "No models visible at the endpoint or in the local download cache."
                    )
                return
            lines = ["Models:"]
            for i, m in enumerate(live, start=1):
                mark = "  [active]" if m == current else ""
                lines.append(f"  {i}. {m}  (serving){mark}")
            for i, m in enumerate(downloaded, start=len(live) + 1):
                lines.append(f"  {i}. {m}  (downloaded — switching restarts the managed server)")
            lines.append("Switch with /model <name|number>.")
            await self._send_info("\n".join(lines))
            return

        # Resolve target: number, exact name, or (managed only) a local checkpoint path.
        if arg.isdigit() and 1 <= int(arg) <= len(choices):
            target = choices[int(arg) - 1]
        elif arg in choices:
            target = arg
        elif managed is not None and Path(arg).expanduser().exists():
            target = arg
        else:
            await self._send_info(
                f"Unknown model '{arg}'. /model lists what's available."
            )
            return

        if target == current:
            await self._send_info(f"{target} is already active.")
            return

        if target in live:
            llm.config.model = target
            cap = await llm.detect_capabilities()
            note = await self._refresh_token_counter(target)
            await self._persist_default_model(target)
            await self._send_info(
                f"Switched to {target} (tool calling: {cap.tool_call_mode}).{note}"
            )
            return

        # Downloaded-but-not-served → managed restart
        from localharness.provider import server as managed_server
        await self._send_info(
            f"Restarting managed vLLM with {target} — model load can take several minutes..."
        )
        box_note = getattr(self._channel, "box_activity", None)

        def _swap_progress(elapsed: float) -> None:
            # Rendered in the box status row; a minutes-long load must never look frozen.
            if box_note is not None:
                box_note(f"loading {target} · {int(elapsed)}s")

        try:
            managed_server.stop_server(self._config_dir, launch=managed.launch)
            managed.model = target
            managed_server.start_server(self._config_dir, managed_server.serve_command(managed))
            models = await managed_server.wait_ready(
                llm.config.base_url, config_dir=self._config_dir, on_poll=_swap_progress
            )
        except (RuntimeError, TimeoutError) as exc:
            await self._channel.send_message(
                f"Model swap failed: {exc}", metadata={"style": "system.error"}
            )
            return
        finally:
            if box_note is not None:
                box_note(None)
        served = models[0] if models else target
        llm.config.model = served
        cap = await llm.detect_capabilities()
        note = await self._refresh_token_counter(served)
        await self._persist_default_model(served)
        await self._send_info(
            f"Switched to {served} (tool calling: {cap.tool_call_mode}).{note}"
        )

    async def _refresh_token_counter(self, model: str) -> str:
        """After a swap, refit the context-window budget (#31) and rebind the shared TokenCounter
        (#25/#30) to the new served model. The counter is ONE object shared by the context manager,
        compaction pipeline and subagent runner, so an in-place rebind/refit updates them all. Both
        probes BLOCK (urllib/httpx, up to ~20s for two shapes), so they run OFF the event loop (#32)
        — the Discord adapter and idle consolidation share it. Never aborts a completed swap; returns
        a disclosure string (leading space; '' when clean) to append to the switch message so the
        user is told on the CHANNEL — not a swallowed log line — when counting/budget can't track."""
        import asyncio

        from localharness.agent import context as context_mod
        from localharness.cli.init_cmd import _fit_context_tokens

        ctx = getattr(self._agent, "_ctx", None)
        base_url = self._agent._llm.config.base_url
        ptype = getattr(getattr(self._harness, "provider", None), "provider_type", None)
        notes: list[str] = []

        # #31: refit the budget to the new model's served window, or disclose when unknowable — a
        # stale 128K budget on a 32K model passes over-window requests that 400 mid-session. The
        # ContextManager reads max_context_tokens live, so the in-place mutation updates every
        # consumer (TokenBudget gates, compaction thresholds, emergency floor).
        if getattr(ctx, "max_context_tokens", None):
            try:
                window = await asyncio.to_thread(
                    context_mod.probe_served_window, base_url, model, ptype
                )
            except Exception:  # noqa: BLE001 — a probe error must never brick a done swap
                window = None
            if window:
                fitted = _fit_context_tokens(window)
                if fitted != ctx.max_context_tokens:
                    ctx.max_context_tokens = fitted
                    notes.append(
                        f"context budget refit to {fitted:,} tokens (served window {window:,})."
                    )
            else:
                notes.append(
                    "context budget unchanged — couldn't read this model's served window; "
                    "re-run `localharness init` if its window differs."
                )

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
        return (" " + " ".join(notes)) if notes else ""

    async def _live_models(self, base_url: str) -> tuple[list[str], bool]:
        """Delegate to the shared probe so the REPL and the `localharness model` CLI share ONE
        failure taxonomy (#38 — this kills the diverged duplicate that had no reachable flag).
        Returns ``(model_ids, reachable)``; raises MalformedModelListError on a reached-but-wrong
        body. Off the event loop: the probe blocks (httpx.get, up to ~3s)."""
        import asyncio

        from localharness.cli import model_ops
        return await asyncio.to_thread(model_ops.list_live_models, base_url)

    async def _prefetch_model_cache(self) -> None:
        """Warm the /model picker menu without ever blocking the UI — best-effort and silent:
        an unreachable or malformed server leaves the cache empty (no menu), and /model itself
        still reports those states properly when actually run."""
        llm = getattr(self._agent, "_llm", None)
        if llm is None:
            return
        try:
            live, reachable = await self._live_models(llm.config.base_url)
        except Exception:
            return
        if reachable:
            names = list(live)
            managed = getattr(self._harness, "server", None) if self._harness is not None else None
            if managed is not None:
                names += [e.name for e in managed.local_models if e.name not in names]
            self._model_cache[:] = names

    async def _persist_default_model(self, model: str) -> None:
        """Persist the swap to the atomic, audited USER OVERLAY (issue #22 pattern) so the next
        start uses it — replaces the prior full, non-atomic config.yaml rewrite. Best-effort:
        a persistence failure (e.g. the new default collides with a configured proposer.model)
        is surfaced but never crashes the live session, which has already switched."""
        from localharness.cli import model_ops
        try:
            audit_warning = await model_ops.persist_default_model(
                self._harness, model, config_dir=self._config_dir
            )
        except Exception as exc:  # noqa: BLE001 — the in-session swap already succeeded
            await self._channel.send_message(
                f"Switched for this session, but persisting the new default failed: {exc}",
                metadata={"style": "system.error"},
            )
            return
        # #37: the overlay write succeeded; a post-write audit-emit failure is a secondary note,
        # not a persist failure — surface it without contradicting the successful switch.
        if audit_warning:
            await self._send_info(audit_warning)
        # Pin trap: name any agent whose yaml pins a concrete model — the persisted default
        # won't reach it next start (per-agent pin wins by design; this only warns).
        pinned = model_ops.pinned_agents(self._config_dir)
        if pinned:
            lines = ["Note: this new default won't reach these agents until their yaml model pin changes:"]
            lines += [f"  - {name} (pinned to {pin!r})" for name, pin in pinned]
            await self._send_info("\n".join(lines))

    async def _send_info(self, text: str) -> None:
        await self._channel.send_message(text, metadata={"style": "system.info"})

    # ------------------------------------------------------------------ #
    # /memory — the tag-hierarchy window into persistent memory
    # ------------------------------------------------------------------ #

    async def _handle_memory_cmd(self, arg: str) -> None:
        """`/memory` — browse/inspect/retire the agent's persistent memory, navigated by the tag
        hierarchy (overview / list / show / forget / search). Model-free; delegates to
        cli.memory_cmd.dispatch and prints its plain text. Works unchanged in classic + box mode:
        a slash command queues mid-turn and runs between turns. Reads are WAL-safe under a live
        turn's writes; a render slip is contained so it can never tear down the REPL."""
        from localharness.cli import memory_cmd

        try:
            result = await memory_cmd.dispatch(self._store, arg)
        except Exception as exc:  # noqa: BLE001 — a read/render slip must never kill the session
            log.warning("/memory failed", exc_info=True)
            result = f"/memory failed: {exc}"
        # overview + show render as rich trees (renderables); listings/search/forget stay text.
        if isinstance(result, str):
            await self._send_info(result)
        else:
            await self._channel.send_renderable(result)

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
