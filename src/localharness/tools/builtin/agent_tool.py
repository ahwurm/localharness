"""AgentTool: delegate tasks to subagents (Claude Code agent-as-tool pattern)."""
from __future__ import annotations

from collections.abc import Callable, Coroutine, Iterable
from pathlib import Path
from typing import Any

from localharness.tools.base import Tool, ToolResult, ToolSchema


HANDOFF_INSTRUCTION = (
    "Finish with a HANDOFF block: lines `status: completed|partial|blocked`, `artifacts: <paths>`, "
    "`findings: ...`, `evidence: ...`, `uncertainties: ...`, `remaining: ...`."
)

# The two budget shares a delegation can ask for (`org.children_budget.lookup_share` / `review_share`).
# A shape picks only the share; a child's ceiling never moves with it.
SHAPES = ("lookup", "review")

BUILTIN_AGENT_NAMES = ("explore", "web-researcher", "cruncher", "search-verifier")
INVALID_AGENT_NAME = "the file name is the agent name: lowercase letters, digits, hyphens"
ROSTER_NAME_CAP = 24


def _roster(names: Iterable[str]) -> str:
    """The first ROSTER_NAME_CAP names in order, then ' and N more'; '' for none."""
    names = list(names)
    tail = f" and {len(names) - ROSTER_NAME_CAP} more" if len(names) > ROSTER_NAME_CAP else ""
    return ", ".join(names[:ROSTER_NAME_CAP]) + tail


def compose_brief(
    task: str, purpose: str | None = None, inputs: list[str] | None = None,
    constraints: str | None = None, expected_output: str | None = None,
    checks: list[str] | None = None, stop_condition: str | None = None,
) -> str:
    """The child's brief: `task` unchanged for a plain call; otherwise a structured assignment
    built only from these fields (never the parent transcript) plus the HANDOFF instruction.
    Inputs are listed, not granted — grant_handles stays the only grant path."""
    if not any((purpose, inputs, constraints, expected_output, checks, stop_condition)):
        return task
    lines = [f"Purpose: {purpose}"] if purpose else []
    lines.append(f"Assignment: {task}")
    lines += [f"Inputs: {', '.join(inputs)}"] if inputs else []
    lines += [f"Constraints: {constraints}"] if constraints else []
    lines += [f"Expected output: {expected_output}"] if expected_output else []
    lines += [f"Checks: {'; '.join(checks)}"] if checks else []
    lines += [f"Stop when: {stop_condition}"] if stop_condition else []
    return "\n".join(lines) + "\n\n" + HANDOFF_INSTRUCTION


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
        agent_runner: Callable[..., Coroutine[Any, Any, Any]],
        available_agents: Iterable[str] | None = None,
        machine_agents_dir: Path | None = None,
        project_agents_dir: Path | None = None,
    ) -> None:
        self._agent_runner = agent_runner
        self._available_agents = available_agents or []
        self._machine_agents_dir = machine_agents_dir
        self._project_agents_dir = project_agents_dir

    def _build_text(self) -> str:
        machine = self._machine_agents_dir
        agents = str(machine) if machine else "~/.localharness/agents"
        tools = f"{machine.parent / 'tools'}/" if machine else "~/.localharness/tools/"
        where = (
            f"A project specialist goes at {self._project_agents_dir}/<name>.yaml and is read by "
            f"every session started in this project; a machine specialist goes at {agents}/<name>.yaml "
            "and serves every project. "
            if self._project_agents_dir is not None else
            f"There is no project layer in this session, so write it at {agents}/<name>.yaml; "
            "a project gets a layer with `localharness init --workspace` in its folder if it has "
            "none yet; it is read from the next start made from that folder without `--config-dir` "
            "or LOCALHARNESS_DIR/LOCALHARNESS_HOME. "
        )
        tighten = (
            "A project specialist's permissions can only tighten the machine's. "
            if self._project_agents_dir is not None else ""
        )
        return (
            "You can also BUILD a new specialist as a YAML file (fields: name, role, "
            "tools: {add: [tool names]}, permissions: {budget: {max_actions, "
            "max_duration_minutes}}), then delegate to that name immediately. "
            + where
            + f"{INVALID_AGENT_NAME.capitalize()}. "
            "Compose specialists from the existing tools; "
            "never write your own tool for something the harness already provides (web search, fetch, files, delegation). "
            "A specialist gets no MCP servers of its own: compose it from the harness tools (`tools.add`). "
            + tighten
            + f"A script you add under {tools} runs only as an unconfirmed shell command "
            "(asked about in guarded mode) until they confirm it at their next `localharness start`. "
        )

    def info(self) -> ToolSchema:
        agent_list = _roster(self._available_agents) or "none configured"
        return ToolSchema(
            name="agent",
            group="delegate",
            gate_family="delegate", ingest="none", host="safe", result_origin="trusted",
            description=(
                f"Delegate a task to a subagent. Available agents: {agent_list}. "
                "Use this when a specialized agent would handle the task better than you. "
                "Returns the agent's summary response. "
                f"{self._build_text()}"
                "If you only hold a LARGE document as a handle (you saw a stub like "
                "\"[tool result evicted — call tool_result_get('<id>')]\" or a 'pg-N' page), pass "
                "that id in grant_handles to let the subagent read the full body by handle — the "
                "bytes never enter your or its prompt. Delegate over-window analysis to 'cruncher'. "
                "For substantial work, add purpose/inputs/constraints/expected_output/checks/"
                "stop_condition: the subagent gets a structured assignment and ends with a "
                "HANDOFF block."
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
                    "purpose": {"type": "string", "description": "Why this work is needed."},
                    "inputs": {
                        "type": "array", "items": {"type": "string"},
                        "description": "workspace paths or handle ids the subagent should use",
                    },
                    "constraints": {"type": "string", "description": "Limits the work must respect."},
                    "expected_output": {"type": "string", "description": "The deliverable to produce."},
                    "checks": {
                        "type": "array", "items": {"type": "string"},
                        "description": "What the result must satisfy.",
                    },
                    "stop_condition": {"type": "string", "description": "When the subagent should stop."},
                    "shape": {
                        "type": "string",
                        "enum": ["lookup", "review"],
                        "description": (
                            "lookup (default): a smaller share of your remaining budget; "
                            "review: a larger one."
                        ),
                    },
                },
                "required": ["agent_id", "task"],
            },
            scope="agent",
            estimated_tokens=500,
            destructive=False,
        )

    async def _execute(
        self, agent_id: str, task: str, grant_handles: list[str] | None = None,
        purpose: str | None = None, inputs: list[str] | None = None,
        constraints: str | None = None, expected_output: str | None = None,
        checks: list[str] | None = None, stop_condition: str | None = None,
        shape: str | None = None,
    ) -> ToolResult:
        # The registry's validator maps JSON types only and ignores `enum`, so the two values are
        # checked here, before anything is dispatched.
        if shape is not None and shape not in SHAPES:
            return self.err(f"shape must be 'lookup' or 'review', not {shape!r}",
                            error_type="validation_error")
        brief = compose_brief(task, purpose, inputs, constraints, expected_output, checks,
                              stop_condition)
        try:
            # `shape` rides only when the model supplied one: the runner contract stays three
            # positional arguments, so every existing runner (and test fake) keeps working.
            if shape is None:
                result = await self._agent_runner(agent_id, brief, grant_handles)
            else:
                result = await self._agent_runner(agent_id, brief, grant_handles, shape=shape)
            if isinstance(result, str):
                return self.ok(result, delegated_to=agent_id, status="completed",
                               terminated_reason=None, tool_calls=0, artifacts=[],
                               child_session_id=None, handoff=None)
            # A DelegationResult (duck-typed: importing subagent here would be a cycle).
            return self.ok(result.text, delegated_to=agent_id, status=result.status,
                           terminated_reason=result.terminated_reason,
                           tool_calls=result.tool_calls, artifacts=list(result.artifacts),
                           child_session_id=result.child_session_id, handoff=result.handoff)
        except ValueError as exc:
            # The runner's ValueErrors are actionable by design ("dispatch not wired
            # (available: ...) — you can CREATE one..."); rebuilding a generic not-found from
            # self._available_agents self-contradicts whenever the advertised list drifts from
            # what the runner can dispatch (live receipt 2026-07-17: "'data-analyst' not
            # found. Available: ... data-analyst ...").
            return self.err(
                str(exc)
                or f"Agent '{agent_id}' not found. "
                   f"Available: {_roster(self._available_agents) or 'none'}",
                error_type="not_found",
            )
        except KeyError:
            return self.err(
                f"Agent '{agent_id}' not found. "
                f"Available: {_roster(self._available_agents) or 'none'}",
                error_type="not_found",
            )
        except Exception as exc:
            return self.err(
                f"Agent '{agent_id}' failed: {exc}",
                error_type="execution_error",
            )
