"""OrchestratorREPL -- interactive prompt_toolkit loop for LocalHarness."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from localharness.core.events import UserMessage


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
                        workflow = self._orchestrator.begin_agent_creation(
                            config_dir=self._config_dir,
                        )
                        # Feed the initial input into the workflow to start DISCUSS
                        workflow.transition(user_input)
                        await self._channel.send_message(
                            "I'd like to help you create an agent. "
                            "Tell me more about what you need it to do.",
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
                    # MULTI-02 (v2).
                    await self._agent.run_turn(
                        task=user_input,
                        on_token=None,
                    )
                    # NOTE: Do NOT send_message here. The TaskComplete event handler
                    # in TerminalChannel.on_task_complete() handles output.
                    # Sending here would produce duplicate output.
                except EOFError:
                    break
        finally:
            await self._channel.stop()

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

        # Unknown slash command — pass through to orchestrator
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
        live = await self._live_models(llm.config.base_url)
        managed = self._harness.server
        downloaded: list[str] = []
        if managed is not None:
            from localharness.provider import server as managed_server
            downloaded = [m for m in managed_server.list_cached_models() if m not in live]
        choices = live + downloaded

        if not arg:
            if not choices:
                await self._send_info("No models visible at the endpoint or in the local download cache.")
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
            self._persist_default_model(target)
            await self._send_info(
                f"Switched to {target} (tool calling: {cap.tool_call_mode})."
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
        self._persist_default_model(served)
        await self._send_info(
            f"Switched to {served} (tool calling: {cap.tool_call_mode}). "
            "If this model serves a different context window, re-run `localharness init` to refit the budget."
        )

    async def _live_models(self, base_url: str) -> list[str]:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{base_url.rstrip('/')}/models", timeout=3.0)
                return [m["id"] for m in resp.json().get("data", [])]
        except Exception:
            return []

    def _persist_default_model(self, model: str) -> None:
        """Write the swap into config.yaml so the next start uses it."""
        self._harness.provider.default_model = model
        self._harness.org.default_model = model
        if model not in self._harness.provider.available_models:
            self._harness.provider.available_models.append(model)
        from pydantic_yaml import to_yaml_str
        (self._config_dir / "config.yaml").write_text(
            to_yaml_str(self._harness), encoding="utf-8"
        )

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
            # Workflow gathered enough info — use LLM directly to generate YAML.
            # Uses llm.complete() instead of run_turn() to avoid publishing
            # TaskComplete events which cause double terminal output.
            import re
            gathered = workflow.gathered
            messages = [
                {"role": "system", "content": "Generate a LocalHarness agent YAML config. Return only the YAML, no explanation."},
                {"role": "user", "content": gathered.get("description", user_input)},
            ]
            # #18: stream at the transport level. Return-value shape is unchanged
            # (stream_complete returns the same (message, usage) as complete).
            response = await self._agent._llm.stream_complete(messages, tools=None)
            yaml_str = getattr(response, "content", "") or ""
            # Strip markdown code fences if present (LLMs often wrap in ```yaml...```)
            yaml_str = re.sub(r'^```(?:yaml)?\s*\n?', '', yaml_str.strip())
            yaml_str = re.sub(r'\n?```\s*$', '', yaml_str.strip())
            workflow.set_generated_yaml(yaml_str)
            workflow.transition("configure_done")  # advance to CONFIRM
            await self._channel.send_message(
                f"Here's the generated config:\n\n```yaml\n{yaml_str}\n```\n\n"
                "Does this look good? (yes/no/change)",
                metadata={"style": "system.info"},
            )
            return

        if new_state == WorkflowState.DEPLOY:
            # User confirmed — deploy the config
            name = workflow.gathered.get("name", "new_agent")
            try:
                config_path = workflow.deploy_config(name)
                await self._channel.send_message(
                    f"Agent deployed to {config_path}",
                    metadata={"style": "system.info"},
                )
            except Exception as exc:
                await self._channel.send_message(
                    f"Deploy failed: {exc}",
                    metadata={"style": "system.error"},
                )
            # Advance through aftercare
            workflow.transition("deployed")
            # Per user decision: after creation, back to prompt (no auto-handoff)
            workflow.transition("done")
            self._orchestrator._active_workflow = None
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
                "What tools does it need?",
                metadata={"style": "system.info"},
            )
