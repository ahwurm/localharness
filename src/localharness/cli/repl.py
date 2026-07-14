"""OrchestratorREPL -- interactive prompt_toolkit loop for LocalHarness."""
from __future__ import annotations

import asyncio
import logging
import re
import signal
from pathlib import Path
from typing import Any

from localharness.core.events import UserMessage

log = logging.getLogger(__name__)


HELP_TEXT = """\
Available commands:
  /help     Show this help message
  /agents   List configured agents
  /model    List available models; /model <name|number> to switch
  /quit     Exit LocalHarness
  /exit     Exit LocalHarness

Everything else is handled by the orchestrator through natural language."""


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
    ) -> None:
        self._orchestrator = orchestrator
        self._agent = agent_loop
        self._channel = channel
        self._bus = bus
        self._config_dir = config_dir
        self._harness = harness_config  # HarnessConfig — needed by /model to persist swaps

    async def run(self) -> None:
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
                    # Slash commands — deterministic, no LLM
                    if user_input.startswith("/"):
                        if await self._handle_slash(user_input):
                            continue

                    # If a creation workflow is active, drive it
                    if self._orchestrator.active_workflow is not None:
                        await self._handle_creation_workflow(user_input)
                        continue

                    # Check for creation intent in natural language
                    if self._detect_creation_intent(user_input):
                        self._orchestrator.begin_agent_creation(
                            config_dir=self._config_dir,
                        )
                        # #19: do NOT transition with the trigger message. The workflow
                        # already starts in DISCUSS; feeding the trigger here consumed
                        # it as the agent DESCRIPTION and silently advanced to CONFIGURE
                        # (returned state discarded), skipping the CONFIGURE branch in
                        # _handle_creation_workflow — the only place YAML generation
                        # runs. The user's NEXT message is the description.
                        await self._channel.send_message(
                            "I'd like to help you create an agent. "
                            "Tell me more about what you need it to do "
                            "(or say 'cancel' to stop).",  # #59: advertise the escape
                            metadata={"style": "system.info"},
                        )
                        continue

                    # Publish user message for memory pipeline. channel_id is the
                    # adapter's class attribute ("terminal", "discord", ...) — history
                    # rows must carry the REAL channel, not a hardcoded "terminal".
                    ch_id = getattr(self._channel, "channel_id", None)
                    await self._bus.publish(
                        UserMessage(
                            agent_id=self._agent._config.name,
                            session_id=self._agent.current_session_id,
                            content=user_input,
                            channel=ch_id if isinstance(ch_id, str) else "terminal",
                        )
                    )
                    # v1: the single agent loop handles every turn directly. Multi-agent
                    # routing (AgentCardRegistry.route) will be wired in for dispatch in
                    # MULTI-02 (v2). Run it as a cancellable task so a mid-turn Ctrl+C
                    # cancels the TURN, not the session (#47).
                    await self._run_turn_interruptible(user_input)
                    # NOTE: Do NOT send_message here. The TaskComplete event handler
                    # in TerminalChannel.on_task_complete() handles output.
                    # Sending here would produce duplicate output.
                except EOFError:
                    break
        finally:
            await self._channel.stop()

    async def _run_turn_interruptible(self, user_input: str) -> None:
        """Run ONE agent turn as a cancellable task so a mid-turn Ctrl+C cancels the turn,
        not the session (#47).

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
        (escape hatch)."""
        turn_task = asyncio.ensure_future(
            self._agent.run_turn(task=user_input, on_token=None)
        )
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
            downloaded = [m for m in managed_server.list_cached_models() if m not in live]
        choices = live + downloaded

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
        try:
            managed_server.stop_server(self._config_dir, launch=managed.launch)
            managed.model = target
            managed_server.start_server(self._config_dir, managed_server.serve_command(managed))
            models = await managed_server.wait_ready(
                llm.config.base_url, config_dir=self._config_dir
            )
        except (RuntimeError, TimeoutError) as exc:
            await self._channel.send_message(
                f"Model swap failed: {exc}", metadata={"style": "system.error"}
            )
            return
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
