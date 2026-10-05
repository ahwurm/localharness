"""One persisted working record per agent, composed over the TaskContext evidence core.

The record keeps what a coworker keeps on substantive work: the objective, the current bounded
assignment, accepted decisions, artifacts, file-backed references, editorial judgments, and
machine-checkable checks. The runtime owns
receipts, artifact hashes, human turns, and timestamps; the model changes the record only
through the bounded `task` tool, and a human decision counts only when its quoted words are
found in a turn the runtime itself recorded (`substantiated`). `TaskState` is the holder the
loop and the tool share: with no record it is inert (no packet, no file).

References are re-snapshotted from disk before every request of an active task
(`refresh_references`) and re-declared as active references, so their current bodies survive
eviction, file changes and restarts; there is no file watcher. Judgments are model opinion,
never evidence. The revision budget is counted by the runtime: an edit to an artifact that a
receipt verified spends one revision.

Delegations are owned by the record (0.16.5 D8): the loop calls `begin_delegation` before an
`agent` call (refusing past `delegation_budget`) and `record_delegation` after it, so every
delegation ends in a runtime status; one still "running" when a human turn arrives or the record
is reloaded was cancelled and becomes "interrupted". Only the coordinator's `integrate` resolves
one. Unintegrated delegations also keep a record whose checks all passed from finishing: its
status becomes unknown. Delegations are awaited one at a time, so one writer holds an artifact at
a time.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from localharness.agent.context import MAX_ACTIVE_REFERENCES, ActiveReferenceError
from localharness.agent.task_context import Receipt, Requirement, TaskContext
from localharness.core.private_files import write_private_bytes

if TYPE_CHECKING:
    from localharness.tools.base import ToolResult

log = logging.getLogger(__name__)

FORMAT = 1
MAX_TEXT = 400
MAX_OBJECTIVE = 600
MAX_DECISIONS = 8
MAX_QUESTIONS = 8
MAX_HUMAN_TURNS = 24
MAX_TURN_CHARS = 2000
MAX_RECORD_BYTES = 64 * 1024
MAX_REFERENCES = MAX_ACTIVE_REFERENCES
MAX_JUDGMENTS = 8
MAX_REFERENCE_BYTES = 200 * 1024
MAX_DELEGATIONS = 8
OPEN_JUDGMENTS = "Open editorial judgments (not assessed): "
BUDGET_EXHAUSTED = ("Revision budget exhausted: stop with the usable artifact and list the remaining gaps; "
                    "continue only on new user direction.")
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
class Reference:
    source: str
    path: str | None = None  # workspace-relative
    handle: str | None = None
    sha256: str | None = None
    status: str = "pending"


@dataclass
class Judgment:
    key: str
    criterion: str
    assessment: str = ""
    passages: str = ""
    fix: str = ""
    status: Literal["open", "assessed", "waived"] = "open"


@dataclass
class Delegation:
    id: str
    agent: str
    purpose: str = ""
    status: str = "running"
    artifacts: list[str] = field(default_factory=list)
    findings: str = ""
    uncertainties: str = ""
    remaining: str = ""
    integrated: str = ""
    call_id: str = ""


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
    references: list[Reference] = field(default_factory=list)
    judgments: list[Judgment] = field(default_factory=list)
    verified_revision: dict[str, str | None] = field(default_factory=dict)
    delegations: list[Delegation] = field(default_factory=list)

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
            "references": [vars(r) for r in self.references],
            "judgments": [vars(j) for j in self.judgments],
            "verified_revision": self.verified_revision,
            "delegations": [vars(x) for x in self.delegations],
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
            references=[Reference(**x) for x in d.get("references", [])],
            judgments=[Judgment(**x) for x in d.get("judgments", [])],
            verified_revision=dict(d.get("verified_revision", {})),
            delegations=[Delegation(**x) for x in d.get("delegations", [])],
        )


class TaskState:
    """Loop-facing holder: forwards to the active record, inert without one."""

    def __init__(self, path: Path | None = None, workspace: str | None = None) -> None:
        self.current: TaskRecord | None = None
        self.path = path
        self.workspace = workspace
        self.recent_turns: list[str] = []
        self.notes: list[str] = []  # runtime-only restart notes
        self.turn_dispatches = 0  # tool dispatches since the last human turn (runtime-only)

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
        if self._interrupt_running():
            self._save_quietly()
        now = self.current.context.revisions()
        for key, saved in self.current.artifact_revisions.items():
            if key in now and now[key] is None:
                self.notes.append(f"{key} missing")
            elif key in now and now[key] != saved:
                self.notes.append(f"{key} changed since last session")
        root = Path(self.current.workspace).resolve()
        for ref in self.current.references:
            ref.handle = None  # the content store is per process; re-snapshot on the next request
            if ref.path is None:
                ref.status = "unavailable; read it again"
                continue
            try:
                digest = hashlib.sha256((root / ref.path).read_bytes()).hexdigest()
            except OSError:
                self.notes.append(f"reference {ref.source} missing")
                continue
            if ref.sha256 is not None and digest != ref.sha256:
                self.notes.append(f"reference {ref.source} changed since last session")

    def _interrupt_running(self) -> bool:
        """A delegation still "running" at a step boundary or reload was cancelled mid-flight."""
        changed = False
        for d in self.current.delegations if self.current else []:
            if d.status == "running":
                d.status, changed = "interrupted", True
                self.notes.append(f"delegation {d.id} {d.agent} interrupted")
        return changed

    def _save_quietly(self) -> None:
        try:
            self.save()
        except TaskRecordTooLarge:
            log.warning("task record too large to save")

    def _live(self) -> TaskRecord | None:
        rec = self.current
        return None if rec is None or rec.closed or rec.context.status == "complete" else rec

    def refresh_references(self, store: Any) -> None:
        """Re-snapshot declared references into `store` and declare them for this request.

        Called before every request of an active task. Every available snapshot is re-declared
        each time because the per-turn reset clears protection; a file is re-put only when its
        handle is gone or its hash moved. Never raises: problems become the reference status."""
        rec = self._live()
        if store is None or rec is None or not rec.references:
            return
        root = Path(rec.workspace).resolve()
        step = store.active_step or f"task {rec.id}"
        before = [(r.handle, r.sha256, r.status) for r in rec.references]
        for ref in rec.references:
            if ref.path is not None:
                target = (root / ref.path).resolve()
                if not target.is_relative_to(root):
                    ref.handle, ref.status = None, "unprotected: the path leaves the workspace"
                    continue
                try:
                    too_large = target.stat().st_size > MAX_REFERENCE_BYTES
                    data = b"" if too_large else target.read_bytes()
                except OSError:
                    ref.handle, ref.status = None, "missing; the file could not be read"
                    continue
                if too_large:
                    ref.handle, ref.status = None, "too large; narrow the reference"
                    continue
                digest = hashlib.sha256(data).hexdigest()
                changed = ref.sha256 is not None and digest != ref.sha256
                if changed or ref.handle is None or store.get(ref.handle) is None:
                    ref.handle, ref.sha256 = store.put(data.decode("utf-8", "replace")), digest
                status = "refreshed (changed)" if changed else "current"
            elif ref.handle is None or store.get(ref.handle) is None:
                ref.status = "unavailable; read it again"
                continue
            else:
                status = "current"
            try:
                store.declare_active_reference(step, ref.handle, ref.source)
                ref.status = status
            except ActiveReferenceError as exc:
                ref.status = f"unprotected: {exc}"
        if before != [(r.handle, r.sha256, r.status) for r in rec.references]:
            try:
                self.save()
            except TaskRecordTooLarge:
                log.warning("task record too large to save after a reference refresh")

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

    def observe_human(self, text: str, new_turn: bool = True) -> None:
        """`new_turn=False` for a mid-turn steering nudge: it does not reset the turn's dispatch
        count, so a nudge after tool activity cannot exempt the final reply from the check."""
        if new_turn:
            self.turn_dispatches = 0
        turn = text[:MAX_TURN_CHARS]
        self.recent_turns = [*self.recent_turns, turn][-MAX_HUMAN_TURNS:]
        if self.current is not None:
            self.current.context.observe_human(text)
            self.current.human_turns = [*self.current.human_turns, turn][-MAX_HUMAN_TURNS:]
            self._interrupt_running()
            self.save()

    def revisions(self) -> dict[str, str | None]:
        return self.current.context.revisions() if self.current else {}

    def record_result(self, tool: str, arguments: dict[str, Any], call_id: str, *,
                      success: bool, metadata: dict[str, Any], before: dict[str, str | None]) -> None:
        self.turn_dispatches += 1
        rec = self.current
        if rec is None:
            return
        ctx = rec.context
        after = ctx.revisions()
        for key, verified in list(rec.verified_revision.items()):
            if verified is not None and after.get(key) != verified:
                rec.revisions_used += 1  # an edit to a verified artifact is one focused revision
                del rec.verified_revision[key]
        ctx.record_result(tool, arguments, call_id, success=success, metadata=metadata, before=before)
        for key, req in ctx.requirements.items():
            receipt = ctx.receipts.get(key)
            if receipt is not None and receipt.call_id == call_id:
                for dep in req.dependencies:
                    rec.verified_revision[dep] = after.get(dep)
        self.save()

    def begin_delegation(self, call_id: str, agent: str, purpose: str) -> str | None:
        """Open a "running" entry before an `agent` call, or return why it is refused. Inert
        (None, no entry) without a live record, so users without a task are unaffected."""
        rec = self._live()
        if rec is None:
            return None
        if len(rec.delegations) >= rec.delegation_budget:
            return (f"Delegation budget ({rec.delegation_budget}) used for this task; integrate the "
                    "results you have or ask the user to raise it")
        if len(rec.delegations) >= MAX_DELEGATIONS:
            return f"delegations is limited to {MAX_DELEGATIONS} per task"
        rec.delegations.append(Delegation(f"d{len(rec.delegations) + 1}", agent[:200],
                                          purpose[:MAX_TEXT], call_id=call_id))
        self.save()
        return None

    def record_delegation(self, call_id: str, result: ToolResult | None) -> None:
        """Settle the entry for `call_id` from the agent tool's result (None: dispatch raised).
        Status and artifacts are runtime facts; findings/uncertainties/remaining are the child's
        HANDOFF text when present, else the tail of its output (its own final words)."""
        rec = self.current
        d = next((x for x in rec.delegations if x.call_id == call_id), None) if rec else None
        if d is None:
            return
        if result is None or not result.success:
            kind = result.error_type if result is not None else None
            d.status = "timeout" if kind == "timeout_error" else f"failed: {kind or 'execution_error'}"
            d.findings = ((result.error or "") if result is not None else "")[:MAX_TEXT]
        else:
            meta = result.metadata
            handoff = meta.get("handoff") or {}
            d.status = str(meta.get("status") or "completed")[:MAX_TEXT]
            d.artifacts = [str(p)[:MAX_TEXT] for p in (meta.get("artifacts") or [])][:16]
            d.findings = str(handoff.get("findings") or result.output[-MAX_TEXT:])[:MAX_TEXT]
            d.uncertainties = str(handoff.get("uncertainties") or "")[:MAX_TEXT]
            d.remaining = str(handoff.get("remaining") or "")[:MAX_TEXT]
        self._save_quietly()

    def unresolved_delegations(self) -> list[Delegation]:
        return [d for d in self.current.delegations if not d.integrated] if self.current else []

    def packet(self) -> str:
        rec = self._live()
        if rec is None:
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
        if rec.references:
            lines.append("References: " + "; ".join(f"{r.source}: {r.status}" for r in rec.references))
        if rec.judgments:
            lines.append("Editorial judgments (model opinion, not evidence): " + "; ".join(
                f"{j.key}: {j.status} — {j.criterion}" for j in rec.judgments))
        if rec.delegations:
            lines.append("Delegations: " + "; ".join(
                f"{d.id} {d.agent}: {d.status}" + (", integrated" if d.integrated else "")
                for d in rec.delegations))
        lines.append(f"Revision budget: used {rec.revisions_used} of {rec.revision_budget}")
        if rec.revisions_used > rec.revision_budget:
            lines.append(BUDGET_EXHAUSTED)
        if self.notes:
            lines.append("Since last session: " + "; ".join(self.notes))
        return "\n".join([*lines, rec.context.packet()])

    def finalize(self, candidate: str) -> str:
        # A record already finalized as complete never rewrites later, unrelated replies.
        if self.current is None or self.current.context.status == "complete":
            return candidate
        if self.turn_dispatches == 0:
            return candidate  # a tool-less aside changed nothing the record could check
        result = self.current.context.finalize(candidate)
        unchanged = result == candidate
        open_ = [j.key for j in self.current.judgments if j.status == "open"]
        if unchanged and open_:
            result = f"{candidate}\n\n{OPEN_JUDGMENTS}{', '.join(open_)}."
        pending = self.unresolved_delegations()
        downgraded = bool(pending) and self.current.context.status == "complete"
        if downgraded:
            self.current.context.status = "unknown"  # unintegrated delegated work keeps the record live
        if pending and (unchanged or downgraded):
            result += "\n\nDelegated work unresolved: " + "; ".join(
                f"{d.id} {d.agent}: {d.status}" for d in pending) + "."
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
        lines.append("references:")
        lines += [f"  {r.source}: {r.path or 'handle ' + str(r.handle)} ({r.status})" for r in rec.references]
        lines.append("judgments:")
        lines += [f"  {j.key}: {j.status} — {j.criterion}" + "".join(
            f" | {name}: {getattr(j, name)}" for name in ("assessment", "passages", "fix") if getattr(j, name))
            for j in rec.judgments]
        lines.append("delegations:")
        lines += [f"  {d.id} {d.agent}: {d.status} — {d.purpose}"
                  + (f" | artifacts: {', '.join(d.artifacts)}" if d.artifacts else "")
                  + (f" | integrated: {d.integrated}" if d.integrated else "")
                  for d in rec.delegations]
        lines.append("questions:")
        lines += [f"  {q}" for q in rec.questions]
        lines += [f"next action: {rec.next_action or '-'}",
                  f"budgets: revision {rec.revisions_used}/{rec.revision_budget}, "
                  f"delegation {rec.delegation_budget}",
                  f"file: {self.path}"]
        if self.notes:
            lines.append("since last session: " + "; ".join(self.notes))
        return "\n".join(lines)
