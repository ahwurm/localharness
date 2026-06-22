"""Orchestrator: thin router, synthesizer, and conversation manager."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from localharness.orchestrator.cards import AgentCard, AgentCardRegistry, RoutingDecision

if TYPE_CHECKING:
    from localharness.orchestrator.workflow import AgentCreationWorkflow


class OrchestratorContextGuard:
    """Enforces the 10-15% context budget for the orchestrator.

    Trims orchestrator conversation to stay within max_tokens cap.
    Strategy: always preserve system message (index 0), trim-to-last-N-turns.
    """

    def __init__(self, max_tokens: int = 16_000, token_counter: "Any | None" = None) -> None:
        self._max_tokens = max_tokens
        if token_counter is None:
            from localharness.agent.context import TokenCounter
            token_counter = TokenCounter()
        self._counter = token_counter

    def trim(self, messages: list[dict]) -> list[dict]:
        usage = self._counter.count_messages(messages)
        if usage <= self._max_tokens:
            return messages

        # Preserve system message + trim from oldest user/assistant turns
        system = [messages[0]] if messages and messages[0].get("role") == "system" else []
        rest = messages[len(system):]

        while rest and self._counter.count_messages(system + rest) > self._max_tokens:
            # Remove oldest 2 messages (one turn = user + assistant)
            rest = rest[2:] if len(rest) >= 2 else rest[1:]

        return system + rest


class Orchestrator:
    """Thin router, synthesizer, and conversation manager.

    Instantiated with config; manages greeting, routing, and workflow delegation.
    Full event-bus integration (start/stop lifecycle) will wire into
    existing CLI start command.
    """

    def __init__(
        self,
        card_registry: AgentCardRegistry,
        context_guard: OrchestratorContextGuard | None = None,
    ) -> None:
        self._card_registry = card_registry
        self._context_guard = context_guard or OrchestratorContextGuard()
        self._messages: list[dict] = []
        self._active_workflow: AgentCreationWorkflow | None = None

    @staticmethod
    def compose_greeting(is_returning: bool, model_name: str = "") -> str:
        if is_returning:
            if model_name:
                return f"{model_name} -- Ready."
            return "Ready."
        return (
            "Hello. I'm the LocalHarness orchestrator. "
            "Tell me what you'd like to build or ask /help for available commands."
        )

    @staticmethod
    def no_config_message() -> str:
        return (
            "Welcome to LocalHarness.\n\n"
            "To configure, run: localharness init\n\n"
            "For model and inference provider recommendations for your hardware,\n"
            "see: localharness.dev/resources"
        )

    def route_task(self, task: str) -> RoutingDecision:
        return self._card_registry.route(task)

    def begin_agent_creation(self, config_dir: Path | None = None) -> "AgentCreationWorkflow":
        """Start a new agent creation workflow.

        Instantiates AgentCreationWorkflow and stores it as the active workflow.
        ORCH-02 requires orchestrator to drive the creation flow through this delegation.
        """
        from localharness.orchestrator.workflow import AgentCreationWorkflow
        self._active_workflow = AgentCreationWorkflow(config_dir=config_dir)
        return self._active_workflow

    @property
    def active_workflow(self) -> "AgentCreationWorkflow | None":
        return self._active_workflow

    def trim_context(self) -> list[dict]:
        return self._context_guard.trim(self._messages)
