"""OrchestratorREPL -- interactive prompt_toolkit loop for LocalHarness."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from localharness.core.events import UserMessage


HELP_TEXT = """\
Available commands:
  /help     Show this help message
  /agents   List configured agents
  /quit     Exit LocalHarness
  /exit     Exit LocalHarness

Everything else is handled by the orchestrator through natural language."""


# Keywords that signal the user wants to create an agent via conversation.
# Checked case-insensitively against user input when no workflow is active.
_CREATION_TRIGGERS = ("create an agent", "create agent", "make an agent",
                      "new agent", "build an agent", "i want an agent",
                      "i need an agent", "set up an agent", "setup an agent")


class OrchestratorREPL:
    """Interactive REPL routing through Orchestrator.

    Slash commands are deterministic (no LLM). All other input
    routes through the Orchestrator for LLM-driven handling.
    When agent creation intent is detected, drives the
    AgentCreationWorkflow state machine through conversation.
    """

    def __init__(
        self,
        orchestrator: Any,
        agent_loop: Any,
        channel: Any,
        bus: Any,
        config_dir: Path | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._agent = agent_loop
        self._channel = channel
        self._bus = bus
        self._config_dir = config_dir

    async def run(self) -> None:
        """Main REPL loop. Slash commands first, then Orchestrator routing."""
        await self._channel.start()
        try:
            while True:
                try:
                    user_input = await self._channel.read_input("you> ")
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

                    # Route through orchestrator — all other input
                    await self._bus.publish(
                        UserMessage(
                            agent_id=self._agent._config.name,
                            content=user_input,
                            channel="terminal",
                        )
                    )
                    summary = await self._agent.run_turn(
                        task=user_input,
                        on_token=None,
                    )
                    await self._channel.send_message(
                        summary,
                        agent_id=self._agent._config.name,
                    )
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
            # Workflow gathered enough info — use LLM to generate YAML
            gathered = workflow.gathered
            prompt = (
                f"Generate a LocalHarness agent YAML config for: "
                f"{gathered.get('description', user_input)}. "
                f"Return only the YAML, no explanation."
            )
            yaml_str = await self._agent.run_turn(task=prompt, on_token=None)
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
