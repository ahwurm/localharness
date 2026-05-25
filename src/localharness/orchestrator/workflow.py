"""Agent creation workflow state machine."""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any


class WorkflowState(str, Enum):
    DISCUSS = "discuss"
    CONFIGURE = "configure"
    CONFIRM = "confirm"
    DEPLOY = "deploy"
    AFTERCARE = "aftercare"
    CANCELLED = "cancelled"
    COMPLETE = "complete"


class AgentCreationWorkflow:
    """3-stage conversational state machine for agent creation.

    States: discuss -> configure -> confirm -> deploy -> aftercare -> complete
    Cancel from any state. Back from confirm -> discuss.
    """

    def __init__(self, config_dir: Path | None = None) -> None:
        self._state = WorkflowState.DISCUSS
        self._config_dir = config_dir or Path.home() / ".localharness" / "agents"
        self._gathered: dict[str, Any] = {}
        self._generated_yaml: str = ""
        self._agent_name: str = ""

    @property
    def state(self) -> WorkflowState:
        return self._state

    @property
    def gathered(self) -> dict[str, Any]:
        return dict(self._gathered)

    def transition(self, user_input: str) -> WorkflowState:
        """Process user input and transition state.

        Returns the new state after processing input.
        """
        lower = user_input.lower().strip()

        # Cancel from any state
        if lower in ("cancel", "quit", "exit", "nevermind"):
            self._state = WorkflowState.CANCELLED
            return self._state

        if self._state == WorkflowState.DISCUSS:
            return self._handle_discuss(user_input)
        elif self._state == WorkflowState.CONFIGURE:
            return self._handle_configure(user_input)
        elif self._state == WorkflowState.CONFIRM:
            return self._handle_confirm(user_input)
        elif self._state == WorkflowState.DEPLOY:
            return self._handle_deploy(user_input)
        elif self._state == WorkflowState.AFTERCARE:
            return self._handle_aftercare(user_input)
        return self._state

    def _handle_discuss(self, user_input: str) -> WorkflowState:
        if "description" not in self._gathered:
            self._gathered["description"] = user_input
        if self._has_minimum_fields():
            self._state = WorkflowState.CONFIGURE
        return self._state

    def _handle_configure(self, user_input: str) -> WorkflowState:
        # Workflow tracks state; actual generation is done by caller.
        self._state = WorkflowState.CONFIRM
        return self._state

    def _handle_confirm(self, user_input: str) -> WorkflowState:
        lower = user_input.lower().strip()
        if lower in ("yes", "y", "ok", "looks good", "deploy", "lgtm"):
            self._state = WorkflowState.DEPLOY
        elif lower in ("no", "n", "change", "update", "edit", "redo"):
            self._state = WorkflowState.DISCUSS
        return self._state

    def _handle_deploy(self, user_input: str) -> WorkflowState:
        self._state = WorkflowState.AFTERCARE
        return self._state

    def _handle_aftercare(self, user_input: str) -> WorkflowState:
        lower = user_input.lower().strip()
        if lower in ("no", "n", "done", "skip", "nothing"):
            self._state = WorkflowState.COMPLETE
        return self._state

    def _has_minimum_fields(self) -> bool:
        return "description" in self._gathered and len(self._gathered["description"]) > 10

    def set_gathered(self, key: str, value: Any) -> None:
        """Set a gathered field (used by orchestrator after LLM extraction)."""
        self._gathered[key] = value

    def set_generated_yaml(self, yaml_str: str) -> None:
        """Store the generated YAML for confirmation display."""
        self._generated_yaml = yaml_str

    @property
    def generated_yaml(self) -> str:
        return self._generated_yaml

    def deploy_config(self, agent_name: str) -> Path:
        """Write the generated YAML to the config directory.

        Returns the path to the written config file.
        """
        config_path = self._config_dir / f"{agent_name}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = config_path.with_suffix(".yaml.tmp")
        tmp_path.write_text(self._generated_yaml)
        os.replace(str(tmp_path), str(config_path))
        self._agent_name = agent_name
        return config_path
