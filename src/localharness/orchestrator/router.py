"""Orchestrator layer: the agent-creation flow and the no-config welcome message."""
from __future__ import annotations

from typing import TYPE_CHECKING

from localharness.orchestrator.cards import AgentCardRegistry

if TYPE_CHECKING:
    from localharness.config.loader import ConfigLoader
    from localharness.orchestrator.workflow import AgentCreationWorkflow


class AgentCreationFlow:
    """Drives the conversational agent-creation workflow.

    Holds the AgentCardRegistry (the REPL reads it to list configured agents) and
    begins/tracks the active AgentCreationWorkflow. Also exposes the static no-config
    welcome message shown at startup before any config exists.
    """

    def __init__(self, card_registry: AgentCardRegistry) -> None:
        self._card_registry = card_registry
        self._active_workflow: AgentCreationWorkflow | None = None

    @staticmethod
    def no_config_message() -> str:
        """The first words a newcomer sees: `localharness start` first, no model server needed."""
        return (
            "Welcome to LocalHarness.\n\n"
            "To begin, run: localharness start\n"
            "It finds a model on this machine or offers one to fetch, and sets everything up;\n"
            "a model server is not required.\n\n"
            "For model and inference provider recommendations for your hardware,\n"
            "see: https://localharness.dev"
        )

    def begin_agent_creation(self, loader: "ConfigLoader | None" = None) -> "AgentCreationWorkflow":
        """Start a new agent creation workflow and store it as the active one."""
        from localharness.orchestrator.workflow import AgentCreationWorkflow
        self._active_workflow = AgentCreationWorkflow(loader=loader)
        return self._active_workflow

    @property
    def active_workflow(self) -> "AgentCreationWorkflow | None":
        return self._active_workflow
