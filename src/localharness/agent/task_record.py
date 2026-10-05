"""One persisted working record per agent, composed over the TaskContext evidence core.

The record keeps what a coworker keeps on substantive work: the objective, the current bounded
assignment, accepted decisions, artifacts, and machine-checkable checks. The runtime owns
receipts, artifact hashes, human turns, and timestamps; the model changes the record only
through the bounded `task` tool, and a human decision counts only when its quoted words are
found in a turn the runtime itself recorded (`substantiated`). `TaskState` is the holder the
loop and the tool share: with no record it is inert (no packet, no file).
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from localharness.agent.task_context import Receipt, Requirement, TaskContext
from localharness.core.private_files import write_private_bytes

log = logging.getLogger(__name__)

FORMAT = 1
MAX_TEXT = 400
MAX_OBJECTIVE = 600
MAX_DECISIONS = 8
MAX_QUESTIONS = 8
MAX_HUMAN_TURNS = 24
MAX_TURN_CHARS = 2000
MAX_RECORD_BYTES = 64 * 1024
CORRUPT_NOTICE = "Task state could not be read; starting without it"
_QUOTES = str.maketrans("", "", "\"'“”‘’`")


class TaskRecordTooLarge(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_quote(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_QUOTES).lower()).strip()


@dataclass(frozen=True)
class Decision:
    text: str
    origin: Literal["human", "model"]
    turn: int


@dataclass
class TaskRecord:
    context: TaskContext
    workspace: str
    id: str = field(default_factory=lambda: secrets.token_hex(4))
    format: int = FORMAT
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)
    assignment: str = ""
    instructions_path: str = ""
    decisions: list[Decision] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    next_action: str = ""
    revision_budget: int = 1
    revisions_used: int = 0
    delegation_budget: int = 2
    human_turns: list[str] = field(default_factory=list)
    artifact_revisions: dict[str, str | None] = field(default_factory=dict)
    closed: bool = False

    def to_dict(self) -> dict[str, Any]:
        ctx = self.context
        return {
            "format": self.format, "id": self.id, "created": self.created, "updated": self.updated,
            "workspace": self.workspace, "assignment": self.assignment,
            "instructions_path": self.instructions_path,
            "decisions": [vars(d) for d in self.decisions], "questions": self.questions,
            "next_action": self.next_action, "revision_budget": self.revision_budget,
            "revisions_used": self.revisions_used, "delegation_budget": self.delegation_budget,
            "human_turns": self.human_turns, "artifact_revisions": self.artifact_revisions,
            "closed": self.closed,
            "context": {
                "objective": ctx.objective, "stop_boundary": ctx.stop_boundary,
                "requested_status": ctx.requested_status, "status": ctx.status,
                "latest_human": ctx.latest_human,
                "requirements": {k: {**vars(r), "dependencies": list(r.dependencies)}
                                 for k, r in ctx.requirements.items()},
                "artifacts": {k: str(p) for k, p in ctx.artifacts.items()},
                "receipts": {k: vars(r) for k, r in ctx.receipts.items()},
                "waivers": {k: list(w) for k, w in ctx.waivers.items()},
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskRecord:
        """Rebuild a saved record. Every non-core field has a default so later slices can add
        fields without bumping the format. A malformed shape raises ValueError/TypeError/KeyError."""
        if not isinstance(d, dict) or d.get("format") != FORMAT:
            raise ValueError("unsupported task record format")
        c = d["context"]
        ctx = TaskContext(
            str(c["objective"]),
            {k: Requirement(**{**r, "dependencies": tuple(r.get("dependencies", ()))})
             for k, r in c.get("requirements", {}).items()},
            {k: Path(p) for k, p in c.get("artifacts", {}).items()},
            stop_boundary=c.get("stop_boundary", ""),
            requested_status=c.get("requested_status", "complete"),
            status=c.get("status", "active"), latest_human=c.get("latest_human", ""),
            receipts={k: Receipt(**r) for k, r in c.get("receipts", {}).items()},
            waivers={k: (str(w[0]), str(w[1])) for k, w in c.get("waivers", {}).items()},
        )
        return cls(
            context=ctx, workspace=str(d["workspace"]), id=str(d["id"]),
            created=d.get("created", _now()), updated=d.get("updated", _now()),
            assignment=d.get("assignment", ""), instructions_path=d.get("instructions_path", ""),
            decisions=[Decision(**x) for x in d.get("decisions", [])],
            questions=list(d.get("questions", [])), next_action=d.get("next_action", ""),
            revision_budget=int(d.get("revision_budget", 1)),
            revisions_used=int(d.get("revisions_used", 0)),
            delegation_budget=int(d.get("delegation_budget", 2)),
            human_turns=[str(t) for t in d.get("human_turns", [])],
            artifact_revisions=dict(d.get("artifact_revisions", {})),
            closed=bool(d.get("closed", False)),
        )


class TaskState:
    """Loop-facing holder: forwards to the active record, inert without one."""

    def __init__(self, path: Path | None = None, workspace: str | None = None) -> None:
        self.current: TaskRecord | None = None
        self.path = path
        self.workspace = workspace
        self.recent_turns: list[str] = []
        self.notes: list[str] = []  # runtime-only restart notes

    @classmethod
    def load(cls, path: Path, workspace: str) -> tuple[TaskState, str | None]:
        state = cls(path, workspace)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return state, None
        try:
            record = TaskRecord.from_dict(json.loads(raw))
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            aside = path.with_name(path.name + ".corrupt")
            log.warning("the task record at %s could not be read (%s); moved to %s", path, exc, aside)
            os.replace(path, aside)
            return state, CORRUPT_NOTICE
        if record.workspace != workspace:
            log.info("the task record at %s belongs to %s, not %s; left in place",
                     path, record.workspace, workspace)
            return state, None
        state.current = record
        state.recent_turns = list(record.human_turns)
        state.reconcile()
        return state, None

    def reconcile(self) -> None:
        if self.current is None:
            return
        now = self.current.context.revisions()
        for key, saved in self.current.artifact_revisions.items():
            if key in now and now[key] is None:
                self.notes.append(f"{key} missing")
            elif key in now and now[key] != saved:
                self.notes.append(f"{key} changed since last session")

    @property
    def active(self) -> bool:
        return self.current is not None

    @property
    def status(self) -> str:
        return self.current.context.status if self.current else "none"

    @status.setter
    def status(self, value: str) -> None:
        if self.current is not None:
            self.current.context.status = value  # type: ignore[assignment]

    def substantiated(self, quote: str | None) -> bool:
        q = normalize_quote(quote or "")
        return len(q) >= 8 and any(q in normalize_quote(t) for t in self.recent_turns)

    def observe_human(self, text: str) -> None:
        turn = text[:MAX_TURN_CHARS]
        self.recent_turns = [*self.recent_turns, turn][-MAX_HUMAN_TURNS:]
        if self.current is not None:
            self.current.context.observe_human(text)
            self.current.human_turns = [*self.current.human_turns, turn][-MAX_HUMAN_TURNS:]
            self.save()

    def revisions(self) -> dict[str, str | None]:
        return self.current.context.revisions() if self.current else {}

    def record_result(self, tool: str, arguments: dict[str, Any], call_id: str, *,
                      success: bool, metadata: dict[str, Any], before: dict[str, str | None]) -> None:
        if self.current is not None:
            self.current.context.record_result(tool, arguments, call_id, success=success,
                                               metadata=metadata, before=before)
            self.save()

    def packet(self) -> str:
        rec = self.current
        if rec is None or rec.closed or rec.context.status == "complete":
            return ""
        lines = [f"Task {rec.id} (working record; update it with the task tool)"]
        if rec.assignment:
            lines.append(f"Current assignment: {rec.assignment}")
        if rec.decisions:
            lines.append("Accepted decisions: " + "; ".join(
                f"[{'human' if d.origin == 'human' else 'assumption'}] {d.text}" for d in rec.decisions))
        if rec.questions:
            lines.append("Open questions: " + "; ".join(rec.questions))
        if rec.next_action:
            lines.append(f"Next action: {rec.next_action}")
        if self.notes:
            lines.append("Since last session: " + "; ".join(self.notes))
        return "\n".join([*lines, rec.context.packet()])

    def finalize(self, candidate: str) -> str:
        if self.current is None:
            return candidate
        result = self.current.context.finalize(candidate)
        self.save()
        return result

    def begin(self, record: TaskRecord) -> TaskRecord | None:
        replaced = self.current if self.current and not self.current.closed \
            and self.current.context.status != "complete" else None
        record.human_turns = list(self.recent_turns[-MAX_HUMAN_TURNS:])
        if self.recent_turns and not record.context.latest_human:
            record.context.latest_human = self.recent_turns[-1]  # the turn that asked for the work
        self.current, self.notes = record, []
        return replaced

    def save(self) -> None:
        if self.path is None or self.current is None:
            return
        rec = self.current
        rec.updated = _now()
        rec.artifact_revisions = rec.context.revisions()
        data = json.dumps(rec.to_dict(), ensure_ascii=True).encode()
        while len(data) > MAX_RECORD_BYTES and rec.human_turns:
            rec.human_turns = rec.human_turns[1:]
            data = json.dumps(rec.to_dict(), ensure_ascii=True).encode()
        if len(data) > MAX_RECORD_BYTES:
            raise TaskRecordTooLarge("Task record exceeds 64 KiB; narrow the record")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        write_private_bytes(tmp, data)
        os.replace(tmp, self.path)

    def clear(self) -> None:
        self.current, self.notes = None, []
        if self.path is not None:
            self.path.unlink(missing_ok=True)

    def show(self) -> str:
        rec = self.current
        if rec is None:
            return "No task record."
        ctx = rec.context
        if rec.closed:
            status = ("closed (no machine checks declared)" if not ctx.requirements
                      else f"closed ({ctx.status})")
        else:
            status = ctx.status
        revs, outcomes = ctx.revisions(), ctx.outcomes()
        lines = [f"Task {rec.id}", f"status: {status}", f"requested status: {ctx.requested_status}",
                 f"objective: {ctx.objective}", f"assignment: {rec.assignment or '-'}",
                 f"stop boundary: {ctx.stop_boundary or '-'}", "decisions:"]
        lines += [f"  {'human' if d.origin == 'human' else 'assumption'}: {d.text}" for d in rec.decisions]
        lines.append("artifacts:")
        lines += [f"  {k}: {p} ({(revs.get(k) or 'missing')[:12]})" for k, p in ctx.artifacts.items()]
        lines.append("checks:")
        lines += [f"  {k}: {outcomes[k]} ({r.origin}) — {r.description}" for k, r in ctx.requirements.items()]
        lines.append("waivers:")
        lines += [f"  {k}: {w[1]}" for k, w in ctx.waivers.items()]
        lines.append("questions:")
        lines += [f"  {q}" for q in rec.questions]
        lines += [f"next action: {rec.next_action or '-'}",
                  f"budgets: revision {rec.revisions_used}/{rec.revision_budget}, "
                  f"delegation {rec.delegation_budget}",
                  f"file: {self.path}"]
        if self.notes:
            lines.append("since last session: " + "; ".join(self.notes))
        return "\n".join(lines)
