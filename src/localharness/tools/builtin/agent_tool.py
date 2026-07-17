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
    # budget is the real backstop. Kept at >= 2x the web-researcher's max duration
    # (now 20 min → 2400s) so the parent never times out before the child's own budget
    # binds (invariant guarded by test_agent_tool_timeout_exceeds_child_budget_and_summary_headroom).
    timeout_s: float | None = 2400.0

    def __init__(
        self,
        agent_runner: Callable[..., Coroutine[Any, Any, str]],
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
                "Returns the agent's summary response. You can also BUILD a new specialist: "
                "write ~/.localharness/agents/<name>.yaml (fields: name, role, "
                "tools: {add: [tool names]}, permissions: {budget: {max_actions, "
                "max_duration_minutes}}), then delegate to that name immediately. "
                "If you only hold a LARGE document as a handle (you saw a stub like "
                "\"[tool result evicted — call tool_result_get('<id>')]\" or a 'pg-N' page), pass "
                "that id in grant_handles to let the subagent read the full body by handle — the "
                "bytes never enter your or its prompt. Delegate over-window analysis to 'cruncher'."
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
                        "description": (
                            "A SELF-CONTAINED instruction the subagent can act on with no other "
                            "context — distill the user's request into one concrete directive. "
                            "NEVER paste the user's verbatim sentence: the subagent can't see this "
                            "conversation and doesn't know who the user is, so a relayed 'ask X "
                            "for Y' makes it go hunting for X instead of doing Y. "
                            "GOOD: 'Write three puns about databases.' "
                            "BAD: 'ask the joke-writer for a database pun'. "
                            "GOOD: 'Summarize the retry logic in src/agent/loop.py in 5 bullets.' "
                            "BAD: 'look into that loop thing I mentioned'."
                        ),
                    },
                    "grant_handles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional handle id(s) (from an eviction stub or a 'pg-N' page alias) to "
                            "hand the subagent, so it can read that large content by reference without "
                            "the bytes entering any prompt. Grants are refused for host-dangerous "
                            "agents; use a no-danger processor like 'cruncher'."
                        ),
                    },
                },
                "required": ["agent_id", "task"],
            },
            scope="agent",
            estimated_tokens=500,
            destructive=False,
        )

    async def _execute(self, agent_id: str, task: str, grant_handles: list[str] | None = None) -> ToolResult:
        try:
            summary = await self._agent_runner(agent_id, task, grant_handles)
            return self.ok(summary, delegated_to=agent_id)
        except ValueError as exc:
            # The runner's ValueErrors are actionable by design ("dispatch not wired
            # (available: ...) — you can CREATE one..."); rebuilding a generic not-found from
            # self._available_agents self-contradicts whenever the advertised list drifts from
            # what the runner can dispatch (live receipt 2026-07-17: "'data-analyst' not
            # found. Available: ... data-analyst ...").
            return self.err(
                str(exc)
                or f"Agent '{agent_id}' not found. "
                   f"Available: {', '.join(self._available_agents) or 'none'}",
                error_type="not_found",
            )
        except KeyError:
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
