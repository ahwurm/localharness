"""Agent creation workflow state machine."""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any


# A description shorter than this can't advance DISCUSS -> CONFIGURE. It is also the
# boundary #56 uses to decide replace-vs-append: at/below it the stored description is a
# too-short stub to be REPLACED; above it, a valid description a 'change' follow-up APPENDS to.
_MIN_DESCRIPTION_LEN = 10


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
        self._config_dir = config_dir or Path.home() / ".localharness"
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
        # #56: the description must stay MUTABLE across DISCUSS turns. The old code stored
        # it once (`if "description" not in gathered`), so (a) a 'change' at confirm sent
        # the flow back here but the follow-up never replaced the description — regeneration
        # ran on the stale original (identical config); and (b) a too-short first reply set
        # a description that could never be replaced — a permanent DISCUSS wedge.
        text = user_input.strip()
        existing = self._gathered.get("description", "")
        if len(existing) <= _MIN_DESCRIPTION_LEN:
            # No usable description yet (empty, or a too-short stub that never advanced):
            # (re)place it — kills the short-then-long wedge.
            self._gathered["description"] = text
        else:
            # A valid description already advanced once and the user returned via 'change':
            # APPEND the correction so the original intent AND the correction both reach the
            # generator (regeneration reruns on the combined text, not the stale original).
            self._gathered["description"] = f"{existing}\n{text}"
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
        return "description" in self._gathered and len(self._gathered["description"]) > _MIN_DESCRIPTION_LEN

    def set_gathered(self, key: str, value: Any) -> None:
        """Set a gathered field (used by orchestrator after LLM extraction)."""
        self._gathered[key] = value

    def set_generated_yaml(self, yaml_str: str) -> None:
        """Store the generated YAML for confirmation display."""
        self._generated_yaml = yaml_str

    @property
    def generated_yaml(self) -> str:
        return self._generated_yaml

    def deploy_config(self, agent_name: str | None = None) -> Path:
        """Write the generated YAML to the config directory.

        Validates the YAML parses correctly and satisfies AgentConfig schema
        before writing. agent_name overrides the YAML's name field; when None
        (#19), the confirmed YAML's own name is honored — deploying what the
        user actually confirmed. A config with no name anywhere fails explicitly
        (#28: never placeholder). Refuses to overwrite an existing agent config
        (#28: os.replace silently clobbered it). Raises ValueError if the YAML is
        invalid, unnamed, fails schema validation, or the target already exists.
        Returns the path to the written config file.
        """
        import yaml as _yaml
        from localharness.config.models import AgentConfig

        # Parse YAML
        try:
            data = _yaml.safe_load(self._generated_yaml)
        except _yaml.YAMLError as exc:
            raise ValueError(f"Generated YAML is not valid: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"Generated YAML is not a mapping (got {type(data).__name__})")

        # Determine the deployment name. #28: no placeholder — a generated config
        # with no top-level name fails explicitly instead of silently becoming
        # 'new-agent' and clobbering some other agent's slot.
        if agent_name is None:
            agent_name = data.get("name")
        if not agent_name:
            raise ValueError(
                "Generated config has no 'name' field; cannot create the agent. "
                "Regenerate with a top-level 'name'."
            )
        data["name"] = agent_name

        # Validate against AgentConfig schema
        AgentConfig(**data)  # raises ValidationError (subclass of ValueError) if malformed

        # Re-serialize with corrected name
        validated_yaml = _yaml.dump(data, default_flow_style=False, sort_keys=False)

        config_path = self._config_dir / "agents" / f"{agent_name}.yaml"
        # #28: never silently overwrite an existing agent — os.replace clobbers.
        if config_path.exists():
            raise ValueError(
                f"Agent {agent_name!r} already exists at {config_path}; refusing to "
                "overwrite. Choose a different name or remove the existing config."
            )
        config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = config_path.with_suffix(".yaml.tmp")
        tmp_path.write_text(validated_yaml)
        os.replace(str(tmp_path), str(config_path))
        self._agent_name = agent_name
        return config_path
