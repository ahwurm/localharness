"""TaskTool: the model's one bounded way to keep the working task record.

The runtime owns receipts, artifact hashes, human turns and timestamps; nothing here writes them.
A human decision, waiver, checkpoint or budget raise counts only when `human_quote` is found in
a human turn the runtime recorded (TaskState.substantiated)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from localharness.agent.task_context import Requirement, TaskContext
from localharness.agent.task_record import (
    MAX_DECISIONS, MAX_OBJECTIVE, MAX_QUESTIONS, MAX_TEXT, Decision, TaskRecord, TaskRecordTooLarge,
    TaskState,
)
from localharness.tools.base import Tool, ToolResult, ToolSchema

ALLOWED: dict[str, set[str]] = {
    "start": {"objective", "assignment", "stop_boundary", "requested_status", "instructions_path",
              "decisions", "human_quote"},
    "update": {"assignment", "stop_boundary", "requested_status", "next_action", "question",
               "resolve_question", "revision_budget", "delegation_budget", "human_quote"},
    "decide": {"text", "human_quote"},
    "artifact": {"key", "path"},
    "check": {"key", "description", "tool", "arguments", "result_field", "expected", "depends_on",
              "human_quote"},
    "waive": {"key", "human_quote"},
    "close": {"status", "note", "human_quote"},
    "show": set(),
}
CHECKPOINT_RULE = ("A checkpoint is a human-requested boundary; quote the human's words in human_quote, "
                   "or use partial for your own stop")
MAX_ITEMS = 16
MAX_ARGUMENTS_CHARS = 2000


class _Refused(Exception):
    pass


def _text(name: str, value: Any, cap: int = MAX_TEXT) -> str:
    if not isinstance(value, str):
        raise _Refused(f"{name} must be text")
    if len(value) > cap:
        raise _Refused(f"{name} exceeds {cap} characters; narrow it")
    return value


def _str(desc: str, cap: int = MAX_TEXT) -> dict[str, Any]:
    return {"type": "string", "maxLength": cap, "description": desc}


class TaskTool(Tool):
    def __init__(self, state: TaskState) -> None:
        self._state = state

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="task", group="task",
            gate_family="allow", ingest="none", host="safe", result_origin="trusted",
            description=(
                "Keep the working record for substantive multi-step work (artifacts, several steps, "
                "delegation). Only for that — never for questions or answer-only discussion. start a "
                "task, record the human's corrections as decisions (quote their words in human_quote), "
                "register artifacts and machine-checkable checks, and close at the requested boundary. "
                "Receipts, hashes, and human turns are recorded by the runtime; you cannot write them."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": list(ALLOWED)},
                    "objective": _str("start: the requested outcome", MAX_OBJECTIVE),
                    "assignment": _str("The current bounded piece of work"),
                    "stop_boundary": _str("Where to stop and hand back"),
                    "requested_status": {"type": "string",
                                         "enum": ["complete", "checkpoint", "partial", "blocked", "unknown"]},
                    "instructions_path": _str("Workspace file with the user's instructions"),
                    "decisions": {"type": "array", "maxItems": MAX_DECISIONS,
                                  "items": {"type": "string", "maxLength": MAX_TEXT}},
                    "next_action": _str("update: the next step"),
                    "question": _str("update: an open question for the human"),
                    "resolve_question": _str("update: exact text of a question now answered"),
                    "revision_budget": {"type": "integer"},
                    "delegation_budget": {"type": "integer"},
                    "text": _str("decide: the decision"),
                    "human_quote": _str("The human's own words that support this"),
                    "key": _str("artifact/check/waive key"),
                    "path": _str("artifact: workspace file path"),
                    "description": _str("check: what it proves"),
                    "tool": _str("check: tool name of the exact call that proves it"),
                    "arguments": {"type": "object", "description": "check: that call's arguments"},
                    "result_field": _str("check: result metadata field to compare"),
                    "expected": {"type": ["string", "integer", "number", "boolean"],
                                 "description": "check: expected value of result_field (e.g. 0 for exit_code)"},
                    "depends_on": {"type": "array", "items": {"type": "string", "maxLength": MAX_TEXT},
                                   "description": "check: artifact keys it depends on"},
                    "status": {"type": "string", "enum": ["complete", "checkpoint", "blocked"]},
                    "note": _str("close: what remains"),
                },
                "required": ["action"],
            },
            destructive=False,
            estimated_tokens=500,
        )

    async def _execute(self, action: str, **fields: Any) -> ToolResult:
        state = self._state
        fields = {k: v for k, v in fields.items() if v is not None}
        if action not in ALLOWED:
            return self.err(f"Unknown action {action!r}; use one of {', '.join(ALLOWED)}",
                            error_type="validation_error")
        extra = sorted(set(fields) - ALLOWED[action])
        if extra:
            return self.err(f"{', '.join(extra)} not valid for {action}; allowed: "
                            f"{', '.join(sorted(ALLOWED[action])) or 'none'}", error_type="validation_error")
        if action == "show":
            return self.ok(state.show())
        rec = state.current
        if action != "start" and (rec is None or rec.closed or rec.context.status == "complete"):
            return self.err("No active task; call task start for substantive work",
                            error_type="validation_error")
        prev = rec.to_dict() if rec is not None else None
        try:
            message = getattr(self, f"_{action}")(state, fields)
            state.save()
        except (_Refused, TaskRecordTooLarge, ValueError) as exc:
            state.current = TaskRecord.from_dict(prev) if prev else None
            text = str(exc)
            if isinstance(exc, TaskRecordTooLarge):
                text = "Task record exceeds 64 KiB; narrow the record"
            return self.err(text, error_type="validation_error")
        return self.ok(message)

    # --- helpers: each mutates state.current or raises _Refused ---

    def _inside(self, raw: str) -> Path:
        root = Path(self._state.workspace or os.getcwd()).resolve()
        target = (root / Path(raw).expanduser()).resolve()
        if not target.is_relative_to(root):
            raise _Refused(f"Path must stay inside the workspace {root}")
        return target

    def _quoted(self, fields: dict[str, Any]) -> bool:
        return self._state.substantiated(fields.get("human_quote"))

    def _requested(self, fields: dict[str, Any]) -> str | None:
        status = fields.get("requested_status")
        if status == "checkpoint" and not self._quoted(fields):
            raise _Refused(CHECKPOINT_RULE)
        return status

    def _start(self, state: TaskState, f: dict[str, Any]) -> str:
        for name in ("objective", "assignment"):
            if not f.get(name):
                raise _Refused(f"start requires {name}")
        decisions = f.get("decisions", [])
        if not isinstance(decisions, list) or len(decisions) > MAX_DECISIONS:
            raise _Refused(f"decisions is limited to {MAX_DECISIONS}")
        ctx = TaskContext(_text("objective", f["objective"], MAX_OBJECTIVE),
                          stop_boundary=_text("stop_boundary", f.get("stop_boundary", "")),
                          requested_status=self._requested(f) or "complete")  # type: ignore[arg-type]
        raw = f.get("instructions_path")
        record = TaskRecord(
            context=ctx, workspace=state.workspace or os.getcwd(),
            assignment=_text("assignment", f["assignment"]),
            instructions_path=str(self._inside(_text("instructions_path", raw))) if raw else "",
            decisions=[Decision(_text("decision", d), "human" if state.substantiated(d) else "model",
                                len(state.recent_turns)) for d in decisions],
        )
        replaced = state.begin(record)
        lines = [f"Started task {record.id}."]
        if replaced is not None:
            lines.append(f"Replaced unfinished task {replaced.id}.")
        return " ".join(lines)

    def _update(self, state: TaskState, f: dict[str, Any]) -> str:
        rec = state.current
        assert rec is not None
        for name in ("revision_budget", "delegation_budget"):
            if name in f:
                if not isinstance(f[name], int) or f[name] < 0:
                    raise _Refused(f"{name} must be a non-negative integer")
                if f[name] > getattr(rec, name) and not self._quoted(f):
                    raise _Refused(f"Raising {name} needs the human's words in human_quote")
                setattr(rec, name, f[name])
        status = self._requested(f)
        if status:
            rec.context.requested_status = status  # type: ignore[assignment]
        if "assignment" in f:
            rec.assignment = _text("assignment", f["assignment"])
        if "stop_boundary" in f:
            rec.context.stop_boundary = _text("stop_boundary", f["stop_boundary"])
        if "next_action" in f:
            rec.next_action = _text("next_action", f["next_action"])
        if "resolve_question" in f:
            if f["resolve_question"] not in rec.questions:
                raise _Refused("resolve_question must match an open question exactly")
            rec.questions.remove(f["resolve_question"])
        if "question" in f:
            if len(rec.questions) >= MAX_QUESTIONS:
                raise _Refused(f"questions is limited to {MAX_QUESTIONS}")
            rec.questions.append(_text("question", f["question"]))
        return f"Updated task {rec.id}."

    def _decide(self, state: TaskState, f: dict[str, Any]) -> str:
        rec = state.current
        assert rec is not None
        if "text" not in f:
            raise _Refused("decide requires text")
        if len(rec.decisions) >= MAX_DECISIONS:
            raise _Refused(f"decisions is limited to {MAX_DECISIONS}")
        origin = "human" if self._quoted(f) else "model"
        rec.decisions.append(Decision(_text("text", f["text"]), origin, len(state.recent_turns)))
        label = "human decision" if origin == "human" else "assumption (no matching human words)"
        return f"Recorded {label}."

    def _artifact(self, state: TaskState, f: dict[str, Any]) -> str:
        ctx = state.current.context  # type: ignore[union-attr]
        if "key" not in f or "path" not in f:
            raise _Refused("artifact requires key and path")
        key = _text("key", f["key"])
        if key not in ctx.artifacts and len(ctx.artifacts) >= MAX_ITEMS:
            raise _Refused(f"artifacts is limited to {MAX_ITEMS}")
        ctx.artifacts[key] = self._inside(_text("path", f["path"]))
        return f"Artifact {key}: {ctx.artifacts[key]}."

    def _check(self, state: TaskState, f: dict[str, Any]) -> str:
        ctx = state.current.context  # type: ignore[union-attr]
        for name in ("key", "description", "tool"):
            if name not in f:
                raise _Refused(f"check requires {name}")
        arguments = f.get("arguments", {})
        if not isinstance(arguments, dict):
            raise _Refused("arguments must be an object")
        if len(json.dumps(arguments, default=str)) > MAX_ARGUMENTS_CHARS:
            raise _Refused(f"arguments exceeds {MAX_ARGUMENTS_CHARS} characters of JSON; narrow it")
        depends = f.get("depends_on", [])
        if not isinstance(depends, list) or any(d not in ctx.artifacts for d in depends):
            raise _Refused("depends_on must name declared artifact keys")
        key = _text("key", f["key"])
        if key not in ctx.requirements and len(ctx.requirements) >= MAX_ITEMS:
            raise _Refused(f"requirements is limited to {MAX_ITEMS}")
        old = ctx.requirements.get(key)
        ctx.revise(Requirement(
            key, _text("description", f["description"]), _text("tool", f["tool"]), arguments,
            _text("result_field", f["result_field"]) if "result_field" in f else None,
            f.get("expected"), tuple(depends),
            str(int(old.revision) + 1) if old else "1", "human" if self._quoted(f) else "model",
        ))
        return f"Check {key} declared (revision {ctx.requirements[key].revision})."

    def _waive(self, state: TaskState, f: dict[str, Any]) -> str:
        ctx = state.current.context  # type: ignore[union-attr]
        key = f.get("key")
        if key not in ctx.requirements:
            raise _Refused("waive requires the key of a declared check")
        if not self._quoted(f):
            raise _Refused("A waiver is a human decision; quote the human's words in human_quote")
        ctx.waive(key, human_decision=f["human_quote"])
        return f"Check {key} waived by the human."

    def _close(self, state: TaskState, f: dict[str, Any]) -> str:
        rec = state.current
        assert rec is not None
        status = f.get("status")
        if status not in {"complete", "checkpoint", "blocked"}:
            raise _Refused("close requires status complete, checkpoint, or blocked")
        if status == "checkpoint" and not self._quoted(f):
            raise _Refused(CHECKPOINT_RULE)
        rec.context.requested_status = status  # type: ignore[assignment]
        rec.closed = status == "complete"
        if "note" in f:
            rec.next_action = _text("note", f["note"])
        return (f"Closing task {rec.id} as {status}. The final status is decided by the runtime at "
                "the end of this turn from actual evidence.")
