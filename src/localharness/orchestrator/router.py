"""Orchestrator layer: the agent-creation flow and the no-config welcome message."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from localharness.orchestrator.cards import AgentCardRegistry

if TYPE_CHECKING:
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
        # #51: localharness.dev/resources is a 404 — point at the live site root, which exists.
        return (
            "Welcome to LocalHarness.\n\n"
            "To configure, run: localharness init\n\n"
            "For model and inference provider recommendations for your hardware,\n"
            "see: https://localharness.dev"
        )

    def begin_agent_creation(self, config_dir: Path | None = None) -> "AgentCreationWorkflow":
        """Start a new agent creation workflow.

        Instantiates AgentCreationWorkflow and stores it as the active workflow.
        ORCH-02 requires the orchestrator layer to drive the creation flow through
        this delegation.
        """
        from localharness.orchestrator.workflow import AgentCreationWorkflow
        self._active_workflow = AgentCreationWorkflow(config_dir=config_dir)
        return self._active_workflow

    @property
    def active_workflow(self) -> "AgentCreationWorkflow | None":
        return self._active_workflow
