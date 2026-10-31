"""TaskTool: the model's one path into the stage pipeline; the person's verbs have no tool path."""
from __future__ import annotations

from typing import Any

from localharness.core.caps import Caps
from localharness.orchestrator.task_flow import TaskFlow
from localharness.orchestrator.task_flow.record import STAGES, TOOL_STAGES
from localharness.tools.base import Tool, ToolResult, ToolSchema

BOUNDARY = {row.stage: boundary for boundary, row in STAGES.items() if row.stage}
DESCRIPTION = "Run one unit of the open task per call; only the person opens a task."


class TaskTool(Tool):
    def __init__(self, flow: TaskFlow) -> None:
        self._flow = flow

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="task", group="task",
            gate_family="allow", ingest="none", host="safe", result_origin="untrusted",
            description=DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "stage": {"type": "string", "enum": list(TOOL_STAGES)},
                    "doc": {"type": "string", "description": f"new, plan: the doc to file, at most {Caps.DOC} bytes"},
                    "question": {"type": "string", "maxLength": Caps.LINE, "description": "research: one question"},
                },
                "required": ["stage"],
            },
            destructive=False,
            estimated_tokens=500,
        )

    async def _execute(self, stage: Any = None, doc: Any = None, question: Any = None, **extra: Any) -> ToolResult:
        if stage not in TOOL_STAGES:
            return self.err(f"stage must be one of {', '.join(TOOL_STAGES)}", error_type="validation_error")
        carried = doc if doc is not None else question
        return ToolResult(output=await self._flow.run(BOUNDARY[stage], carried, by="model"))
