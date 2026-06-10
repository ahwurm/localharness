"""AgentTool: delegate tasks to subagents (Claude Code agent-as-tool pattern)."""
from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from localharness.tools.base import Tool, ToolResult, ToolSchema


class AgentTool(Tool):
    """Delegates a task to a named subagent and returns the summary.

    The orchestrator's LLM calls this tool when it decides a subagent
    should handle a task. This is the runtime delegation path for ORCH-04.
    """

    # Must exceed the child's time budget PLUS a worst-case final-summary generation
    # on a slow local model (observed: 600s cancelled children 7 min into generating
    # their summary, returning "" with no terminal event). The parent's own turn
    # budget is the real backstop.
    timeout_s: float | None = 1800.0

    def __init__(
        self,
        agent_runner: Callable[[str, str], Coroutine[Any, Any, str]],
        available_agents: list[str] | None = None,
    ) -> None:
        self._agent_runner = agent_runner
        self._available_agents = available_agents or []

    def info(self) -> ToolSchema:
        agent_list = ", ".join(self._available_agents) if self._available_agents else "none configured"
        return ToolSchema(
            name="agent",
            description=(
                f"Delegate a task to a subagent. Available agents: {agent_list}. "
                "Use this when a specialized agent would handle the task better than you. "
                "Returns the agent's summary response."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "Name/ID of the agent to delegate to.",
                    },
                    "task": {
                        "type": "string",
                        "description": "The task description to give the agent.",
                    },
                },
                "required": ["agent_id", "task"],
            },
            scope="agent",
            estimated_tokens=500,
            destructive=False,
        )

    async def _execute(self, agent_id: str, task: str) -> ToolResult:
        try:
            summary = await self._agent_runner(agent_id, task)
            return self.ok(summary, delegated_to=agent_id)
        except (KeyError, ValueError):
            return self.err(
                f"Agent '{agent_id}' not found. "
                f"Available: {', '.join(self._available_agents) or 'none'}",
                error_type="not_found",
            )
        except Exception as exc:
            return self.err(
                f"Agent '{agent_id}' failed: {exc}",
                error_type="execution_error",
            )
