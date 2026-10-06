"""TaskTool: the model's one bounded way to keep the working task record.

The runtime owns receipts, artifact hashes, human turns and timestamps; nothing here writes them.
Human decisions, waivers and checkpoints cite a stored human turn by number (human_turn); the
runtime checks only that the turn exists and copies its verbatim text."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from localharness.agent.task_context import Requirement, TaskContext
from localharness.agent.task_record import (
    MAX_DECISIONS, MAX_JUDGMENTS, MAX_OBJECTIVE, MAX_QUESTIONS, MAX_REFERENCES, MAX_TEXT, Decision,
    Judgment, Reference, TaskRecord, TaskRecordTooLarge, TaskState, _turn_label,
)
from localharness.tools.base import Tool, ToolResult, ToolSchema

ALLOWED: dict[str, set[str]] = {
    "start": {"objective", "assignment", "stop_boundary", "requested_status", "instructions_path",
              "decisions", "human_turn", "references", "artifacts"},
    "update": {"assignment", "stop_boundary", "requested_status", "next_action", "question",
               "resolve_question", "human_turn", "references", "artifacts"},
    "decide": {"text", "human_turn"},
    "artifact": {"key", "path"},
    "check": {"key", "description", "tool", "arguments", "result_field", "expected", "depends_on",
              "human_turn"},
    "reference": {"source", "path", "handle"},
    "judge": {"key", "criterion", "assessment", "passages", "fix"},
    "waive": {"key", "human_turn"},
    "integrate": {"delegation_id", "note"},
    "close": {"status", "note", "human_turn"},
    "show": set(),
}
CHECKPOINT_NEEDS_TURN = ("checkpoint requires human_turn: the human turn that asked to stop; "
                         "use partial for your own stop")
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
                "task, record the human's corrections as decisions (cite the human turn number in human_turn; turns are numbered in arrival order), "
                "register artifacts and machine-checkable checks, and close at the requested boundary. "
                "Receipts, hashes, and human turns are recorded by the runtime; you cannot write them. "
                "Declare files the work depends on (instructions, voice samples, sources) as references "
                "so they stay in view; record editorial criteria with judge — judgments are opinion, "
                "never evidence. integrate delegated results: say how each subagent's result was used. "
                "Declare a check before you run its command so the run binds "
                "as evidence."
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
                                  "items": {"anyOf": [
                                      {"type": "string", "maxLength": MAX_TEXT},
                                      {"type": "object",
                                       "properties": {"text": {"type": "string", "maxLength": MAX_TEXT},
                                                      "human_turn": {"type": "integer"}},
                                       "required": ["text", "human_turn"]}]}},
                    "references": {"type": "array", "maxItems": MAX_REFERENCES,
                                   "description": "start/update: files to keep in view",
                                   "items": {"type": "object",
                                             "properties": {"source": {"type": "string", "maxLength": 200},
                                                            "path": {"type": "string", "maxLength": MAX_TEXT}},
                                             "required": ["source", "path"]}},
                    "artifacts": {"type": "object", "description": "start/update: artifact key -> workspace path",
                                  "additionalProperties": {"type": "string", "maxLength": MAX_TEXT}},
                    "next_action": _str("update: the next step"),
                    "question": _str("update: an open question for the human"),
                    "resolve_question": _str("update: exact text of a question now answered"),
                    "text": _str("decide: the decision"),
                    "human_turn": {"type": "integer", "minimum": 1,
                                   "description": "Number of the human turn this rests on "
                                                  "(the packet lists the stored turns)"},
                    "key": _str("artifact/check/judge/waive key"),
                    "path": _str("artifact/reference: workspace file path"),
                    "source": _str("reference: a short label, e.g. voice sample", 200),
                    "handle": _str("reference: a tool_result_get handle instead of a path", 64),
                    "criterion": _str("judge: the editorial criterion"),
                    "assessment": _str("judge: your assessment against the criterion"),
                    "passages": _str("judge: the passages the assessment concerns"),
                    "fix": _str("judge: the proposed fix"),
                    "description": _str("check: what it proves"),
                    "tool": _str("check: tool name of the call that proves it"),
                    "arguments": {"type": "object",
                                  "description": "check: the arguments that identify that call "
                                                 "(extra actual arguments such as a timeout are allowed)"},
                    "result_field": _str("check: result metadata field to compare"),
                    "expected": {"type": ["string", "integer", "number", "boolean"],
                                 "description": "check: expected value of result_field (e.g. 0 for exit_code)"},
                    "depends_on": {"type": "array", "items": {"type": "string", "maxLength": MAX_TEXT},
                                   "description": "check: artifact keys it depends on"},
                    "status": {"type": "string", "enum": ["complete", "checkpoint", "partial", "blocked"]},
                    "note": _str("close: what remains; integrate: how the delegated result was used"),
                    "delegation_id": _str("integrate: the delegation id, e.g. d1", 16),
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
        if action != "start" and state._live() is None:
            return self.err("No active task; call task start for substantive work",
                            error_type="validation_error")
        prev, prev_notes = (rec.to_dict() if rec is not None else None), list(state.notes)
        try:
            message = getattr(self, f"_{action}")(state, fields)
            state.save()
        except (_Refused, TaskRecordTooLarge, ValueError) as exc:
            state.current = TaskRecord.from_dict(prev) if prev else None
            state.notes = prev_notes
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

    def _cited(self, f: dict[str, Any], required: str | None = None) -> dict[str, Any] | None:
        """The stored human turn `f["human_turn"]` names; the only check is that it is stored."""
        if "human_turn" not in f:
            if required:
                raise _Refused(required)
            return None
        turn = self._state.cite(f["human_turn"])
        if turn is None or isinstance(f["human_turn"], bool):
            span = self._state.turn_range()
            raise _Refused(f"human turn {f['human_turn']} is not on record; "
                           + (f"stored turns: {span}" if span else "no human turns are stored"))
        return turn

    def _checkpoint(self, rec: TaskRecord, f: dict[str, Any]) -> None:
        """start/update: a requested checkpoint cites the human turn that asked for it."""
        status = f.get("requested_status")
        if status == "checkpoint":
            t = self._cited(f, CHECKPOINT_NEEDS_TURN)
            rec.checkpoint_turn, rec.checkpoint_text = t["n"], t["text"]  # type: ignore[index]
        elif "human_turn" in f:
            raise _Refused("on start/update, human_turn cites a checkpoint request")
        elif status:
            rec.checkpoint_turn, rec.checkpoint_text = None, ""
        if status:
            rec.context.requested_status = status

    def _start(self, state: TaskState, f: dict[str, Any]) -> str:
        for name in ("objective", "assignment"):
            if not f.get(name):
                raise _Refused(f"start requires {name}")
        decisions = f.get("decisions", [])
        if not isinstance(decisions, list) or len(decisions) > MAX_DECISIONS:
            raise _Refused(f"decisions is limited to {MAX_DECISIONS}")
        ctx = TaskContext(_text("objective", f["objective"], MAX_OBJECTIVE),
                          stop_boundary=_text("stop_boundary", f.get("stop_boundary", "")))
        raw = f.get("instructions_path")
        record = TaskRecord(
            context=ctx, workspace=state.workspace or os.getcwd(),
            assignment=_text("assignment", f["assignment"]),
            instructions_path=str(self._inside(_text("instructions_path", raw))) if raw else "",
        )
        replaced = state.begin(record)
        for d in decisions:
            if isinstance(d, dict):
                if set(d) != {"text", "human_turn"}:
                    raise _Refused("a cited decision is {text, human_turn}")
                t = self._cited(d)
                record.decisions.append(Decision(_text("decision", d["text"]), "human",
                                                 t["n"], t["text"]))  # type: ignore[index]
            else:
                record.decisions.append(Decision(_text("decision", d), "model"))
        self._checkpoint(record, f)
        lines = [f"Started task {record.id}."]
        if replaced is not None:
            lines.append(f"Replaced unfinished task {replaced.id}.")
        if record.human_turns:
            lines.append("Human turns on record: " + "; ".join(
                f"{t['n']}: {json.dumps(t['text'][:80], ensure_ascii=False)}" for t in record.human_turns) + ".")
        return " ".join(lines) + self._apply_batch(state, f)

    def _apply_batch(self, state: TaskState, f: dict[str, Any]) -> str:
        """Declare artifacts/references given with start/update through the per-item helpers; a
        refusal part-way is undone by _execute's rollback, so the batch is all or nothing."""
        artifacts, references = f.get("artifacts", {}), f.get("references", [])
        if not isinstance(artifacts, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in artifacts.items()):
            raise _Refused("artifacts must map keys to paths")
        if not isinstance(references, list):
            raise _Refused("references must be a list of {source, path}")
        if any(not isinstance(r, dict) or set(r) != {"source", "path"} for r in references):
            raise _Refused("each reference needs source and path")
        for key, path in artifacts.items():
            self._artifact(state, {"key": key, "path": path})
        for r in references:
            self._reference(state, {"source": r["source"], "path": r["path"]})
        if not artifacts and not references:
            return ""
        a, n = len(artifacts), len(references)
        return (f" Declared {a} artifact{'s' * (a != 1)} and {n} reference{'s' * (n != 1)}.")

    def _update(self, state: TaskState, f: dict[str, Any]) -> str:
        rec = state.current
        assert rec is not None
        self._checkpoint(rec, f)
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
        return f"Updated task {rec.id}." + self._apply_batch(state, f)

    def _decide(self, state: TaskState, f: dict[str, Any]) -> str:
        rec = state.current
        assert rec is not None
        if "text" not in f:
            raise _Refused("decide requires text")
        if len(rec.decisions) >= MAX_DECISIONS:
            raise _Refused(f"decisions is limited to {MAX_DECISIONS}")
        t = self._cited(f)
        text = _text("text", f["text"])
        if t is None:
            rec.decisions.append(Decision(text, "model"))
            return "Recorded assumption (no human turn cited)."
        rec.decisions.append(Decision(text, "human", t["n"], t["text"]))
        return f"Recorded human decision (turn {t['n']})."

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
        if not arguments:
            raise _Refused("declare the exact arguments of the call that proves this check")
        key = _text("key", f["key"])
        if any(j.key == key for j in state.current.judgments):  # type: ignore[union-attr]
            raise _Refused("check keys must differ from judgment keys")
        if key not in ctx.requirements and len(ctx.requirements) >= MAX_ITEMS:
            raise _Refused(f"requirements is limited to {MAX_ITEMS}")
        old = ctx.requirements.get(key)
        t = self._cited(f)
        ctx.revise(Requirement(
            key, _text("description", f["description"]), _text("tool", f["tool"]), arguments,
            _text("result_field", f["result_field"]) if "result_field" in f else None,
            f.get("expected"), tuple(depends),
            str(int(old.revision) + 1) if old else "1", "human" if t else "model",
            t["n"] if t else None, t["text"] if t else "",
        ))
        return f"Check {key} declared (revision {ctx.requirements[key].revision})."

    def _reference(self, state: TaskState, f: dict[str, Any]) -> str:
        rec = state.current
        assert rec is not None
        if "source" not in f:
            raise _Refused("reference requires source")
        source = _text("source", f["source"], 200)
        if ("path" in f) == ("handle" in f):
            raise _Refused("reference requires exactly one of path or handle")
        if "path" in f:
            target = self._inside(_text("path", f["path"]))
            if not target.is_file():
                raise _Refused("reference path must be an existing file")
            ref = Reference(source, path=str(target.relative_to(
                Path(state.workspace or os.getcwd()).resolve())))
        else:
            ref = Reference(source, handle=_text("handle", f["handle"], 64))
        same = [i for i, r in enumerate(rec.references) if r.source == source]
        if same:
            rec.references[same[0]] = ref
        elif len(rec.references) >= MAX_REFERENCES:
            raise _Refused("references is limited to four; split or narrow the step")
        else:
            rec.references.append(ref)
        return f"Reference {source} declared; it is kept in view from the next request."

    def _judge(self, state: TaskState, f: dict[str, Any]) -> str:
        rec = state.current
        assert rec is not None
        if "key" not in f:
            raise _Refused("judge requires key")
        key = _text("key", f["key"])
        if key in rec.context.requirements:
            raise _Refused("judgment keys must differ from check keys")
        judgment = next((j for j in rec.judgments if j.key == key), None)
        if judgment is None:
            if "criterion" not in f:
                raise _Refused("a new judgment requires criterion")
            if len(rec.judgments) >= MAX_JUDGMENTS:
                raise _Refused(f"judgments is limited to {MAX_JUDGMENTS}")
            judgment = Judgment(key, _text("criterion", f["criterion"]))
            rec.judgments.append(judgment)
        for name in ("criterion", "assessment", "passages", "fix"):
            if name in f:
                setattr(judgment, name, _text(name, f[name]))
        if "assessment" in f and judgment.status != "waived":
            judgment.status = "assessed"  # the waiver stands; the assessment is recorded beside it
        return f"Judgment {key}: {judgment.status} (editorial opinion, not evidence)."

    def _waive(self, state: TaskState, f: dict[str, Any]) -> str:
        rec = state.current
        assert rec is not None
        ctx, key = rec.context, f.get("key")
        judgment = next((j for j in rec.judgments if j.key == key), None)
        if key not in ctx.requirements and judgment is None:
            raise _Refused("waive requires the key of a declared check or judgment")
        t = self._cited(f, "A waiver cites the human turn that waives it: set human_turn")
        label = _turn_label(t["n"], t["text"])  # type: ignore[index]
        if judgment is not None:
            judgment.status, judgment.waived_by = "waived", label
            return f"Judgment {key} waived (human turn {t['n']})."  # type: ignore[index]
        ctx.waive(key, human_decision=label)
        return f"Check {key} waived (human turn {t['n']})."  # type: ignore[index]

    def _integrate(self, state: TaskState, f: dict[str, Any]) -> str:
        rec = state.current
        assert rec is not None
        if not f.get("delegation_id") or not f.get("note"):
            raise _Refused("integrate requires delegation_id and note")
        d = next((x for x in rec.delegations if x.id == f["delegation_id"]), None)
        if d is None:
            raise _Refused(f"Unknown delegation {f['delegation_id']!r}; known: "
                           f"{', '.join(x.id for x in rec.delegations) or 'none'}")
        if d.status == "running":
            raise _Refused(f"delegation {d.id} is still running")
        d.integrated = _text("note", f["note"])
        return f"Delegation {d.id} integrated."

    def _close(self, state: TaskState, f: dict[str, Any]) -> str:
        rec = state.current
        assert rec is not None
        status = f.get("status")
        if status not in {"complete", "checkpoint", "partial", "blocked"}:
            raise _Refused("close requires status complete, checkpoint, partial, or blocked")
        if status == "checkpoint":
            t = self._cited(f, CHECKPOINT_NEEDS_TURN)
            rec.checkpoint_turn, rec.checkpoint_text = t["n"], t["text"]  # type: ignore[index]
        elif "human_turn" in f:
            raise _Refused("on close, human_turn cites a checkpoint request")
        else:
            rec.checkpoint_turn, rec.checkpoint_text = None, ""
        rec.context.requested_status = status  # type: ignore[assignment]
        rec.closed = status == "complete"
        if "note" in f:
            rec.next_action = _text("note", f["note"])
        line = state.evidence_line() if status == "complete" else ""
        return (f"Closing task {rec.id} as {status}."
                + (f" The record stays live until its declared evidence is settled: {line}" if line else "")
                + " The final status is decided by the runtime at the end of this turn from actual evidence.")
