"""TaskTool: the model's one path into the stage pipeline; the person's verbs have no tool path."""
from __future__ import annotations

from typing import Any

from localharness.core.caps import Caps
from localharness.orchestrator.task_flow import TaskFlow
from localharness.orchestrator.task_flow.record import STAGES, TOOL_STAGES, valid_question
from localharness.tools.base import Tool, ToolResult, ToolSchema

BOUNDARY = {row.stage: boundary for boundary, row in STAGES.items() if row.stage}
TAKES = {"new": "doc", "plan": "doc", "research": "question"}
TAKEN_BY = {"doc": "new and plan", "question": "research"}
DESCRIPTION = "\n".join([
    f"Each call runs one unit of the open task and names the next; only the person opens a task. A doc is at most {Caps.DOC} bytes.",
    *(f"{stage}: reads {', '.join(STAGES[BOUNDARY[stage]].reads)}, writes {', '.join(STAGES[BOUNDARY[stage]].writes)}; "
      f"takes {TAKES.get(stage, 'nothing')}" for stage in TOOL_STAGES),
])


def _bad(stage: str, doc: Any, question: Any, extra: dict) -> str | None:
    if extra:
        return f"{', '.join(sorted(extra))} not valid for task; the arguments are stage, doc, question"
    for name, value in (("doc", doc), ("question", question)):
        if value is not None and TAKES.get(stage) != name:
            return f"{name} is taken by stage {TAKEN_BY[name]}, not {stage}"
    if doc is not None and not (isinstance(doc, str) and doc.strip() and len(doc.encode()) <= Caps.DOC):
        return f"a doc is non-empty text of at most {Caps.DOC} bytes; file it shorter"
    if question is not None and not valid_question(question):
        return f"a question is one plain line of at most {Caps.LINE} characters"
    return None


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
        if bad := _bad(stage, doc, question, extra):
            return self.err(bad, error_type="validation_error")
        carried = doc if doc is not None else question
        return ToolResult(output=await self._flow.run(BOUNDARY[stage], carried, by="model"))
