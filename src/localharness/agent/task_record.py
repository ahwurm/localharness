"""One persisted working record per agent, composed over the TaskContext evidence core.

The runtime is substrate: it records receipts, artifact hashes, human turns (verbatim, numbered in
arrival order), delegation outcomes and timestamps, bounds them, and renders them. The model keeps
the rest (assignment, decisions, artifacts, references, judgments, checks) through the bounded
`task` tool. A citation names a stored human turn by number; the runtime checks only that the turn
is stored and copies its text. There are no budgets, no quote matching and no refusals on the
model's behalf; policy lives in the prompt and workflow instructions. The record retires only when
it was closed as complete with every declared check passed or waived and every delegation
integrated. Replies are never replaced: at most one `Task evidence:` line is appended. References
are re-snapshotted before every request of a live task (`refresh_references`); judgments are
opinion, never evidence. Without a record `TaskState` is inert (no packet, no file).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
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
MAX_ARTIFACT_HISTORY = 16
MAX_QUOTE_DISPLAY = 200
CORRUPT_NOTICE = "Task state could not be read; starting without it"
UNFIT = "unprotected: cannot fit with reply reserve; drop or narrow references"


class TaskRecordTooLarge(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clip(text: str, cap: int = MAX_QUOTE_DISPLAY) -> str:
    """Display capacity only; the stored turn stays verbatim."""
    return text if len(text) <= cap else text[:cap] + "…"


def _turn_label(n: int, text: str) -> str:
    return f"turn {n}: {json.dumps(text, ensure_ascii=False)}"


@dataclass(frozen=True)
class Decision:
    text: str
    origin: Literal["human", "model"]
    human_turn: int | None = None
    human_text: str = ""  # the cited turn's verbatim text, copied by the runtime


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
    waived_by: str = ""  # 'turn N: "..."' of the human turn that waived it


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
    human_turns: list[dict[str, Any]] = field(default_factory=list)  # [{"n": int, "text": str}]
    artifact_revisions: dict[str, str | None] = field(default_factory=dict)  # hashes at last save
    artifact_history: dict[str, list[str]] = field(default_factory=dict)  # distinct hashes seen
    artifact_history_dropped: dict[str, int] = field(default_factory=dict)  # keeps rev numbers stable
    checkpoint_turn: int | None = None
    checkpoint_text: str = ""
    closed: bool = False  # the model closed it as complete
    retired: bool = False  # latched at finalize once closed and settled
    references: list[Reference] = field(default_factory=list)
    judgments: list[Judgment] = field(default_factory=list)
    delegations: list[Delegation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        ctx = self.context
        return {
            "format": self.format, "id": self.id, "created": self.created, "updated": self.updated,
            "workspace": self.workspace, "assignment": self.assignment,
            "instructions_path": self.instructions_path,
            "decisions": [vars(d) for d in self.decisions], "questions": self.questions,
            "next_action": self.next_action,
            "human_turns": self.human_turns, "artifact_revisions": self.artifact_revisions,
            "artifact_history": self.artifact_history,
            "artifact_history_dropped": self.artifact_history_dropped,
            "checkpoint_turn": self.checkpoint_turn, "checkpoint_text": self.checkpoint_text,
            "closed": self.closed, "retired": self.retired,
            "references": [vars(r) for r in self.references],
            "judgments": [vars(j) for j in self.judgments],
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
            human_turns=[{"n": int(t["n"]), "text": str(t["text"])} for t in d.get("human_turns", [])],
            artifact_revisions=dict(d.get("artifact_revisions", {})),
            artifact_history={k: [str(h) for h in v] for k, v in d.get("artifact_history", {}).items()},
            artifact_history_dropped={k: int(v) for k, v in d.get("artifact_history_dropped", {}).items()},
            checkpoint_turn=None if d.get("checkpoint_turn") is None else int(d["checkpoint_turn"]),
            checkpoint_text=str(d.get("checkpoint_text", "")),
            closed=bool(d.get("closed", False)), retired=bool(d.get("retired", False)),
            references=[Reference(**x) for x in d.get("references", [])],
            judgments=[Judgment(**x) for x in d.get("judgments", [])],
            delegations=[Delegation(**x) for x in d.get("delegations", [])],
        )


class TaskState:
    """Loop-facing holder: forwards to the active record, inert without one."""

    def __init__(self, path: Path | None = None, workspace: str | None = None) -> None:
        self.current: TaskRecord | None = None
        self.path = path
        self.workspace = workspace
        self.recent_turns: list[dict[str, Any]] = []  # [{"n": int, "text": str}]
        self.notes: list[str] = []  # runtime-only restart notes
        self._store: Any = None  # the content store of the last refresh (for withdraw/drop)

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
        state.recent_turns = [dict(t) for t in record.human_turns]
        state.reconcile()
        return state, None

    def reconcile(self) -> None:
        """Restart notes are a plain saved-vs-current hash diff: changed, removed, or nothing."""
        if self.current is None:
            return
        if self._interrupt_running():
            self._save_quietly()
        now = self.current.context.revisions()
        for key, saved in self.current.artifact_revisions.items():
            if key in now and saved:
                self._diff_note(key, saved, now[key])
        root = Path(self.current.workspace).resolve()
        for ref in self.current.references:
            ref.handle = None  # the content store is per process; re-snapshot on the next request
            if ref.path is None:
                ref.status = "unavailable; read it again"
                continue
            try:
                digest: str | None = hashlib.sha256((root / ref.path).read_bytes()).hexdigest()
            except OSError:
                digest = None
            if ref.sha256:
                self._diff_note(f"reference {ref.source}", ref.sha256, digest)

    def _diff_note(self, label: str, saved: str, now: str | None) -> None:
        if now is None:
            self.notes.append(f"{label} removed since last session")
        elif now != saved:
            self.notes.append(f"{label} changed since last session")

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
        return None if rec is None or rec.retired or (rec.closed and self._settled(rec)) else rec

    def _settled(self, rec: TaskRecord) -> bool:
        return (all(o in {"passed", "waived"} for o in rec.context.outcomes().values())
                and all(d.integrated for d in rec.delegations))

    def refresh_references(self, store: Any) -> None:
        """Re-snapshot declared references into `store` and declare them for this request.

        Called before every request of an active task. Every available snapshot is re-declared
        each time because the per-turn reset clears protection; a file is re-put only when its
        handle is gone or its hash moved. Never raises: problems become the reference status.

        A file snapshot is the read tool's numbered rendering (`render_numbered`), so a whole-file
        `read` still in the request is byte-identical and `ensure_active_references` adds no second
        copy; after eviction it comes back in the same form. A file beyond the read tool's default
        view (2000 lines / MAX_RETURNED_CHARS) was never fully shown, so a restore is correct then.
        An editor-attached read (file_read_hook) reads the buffer while the snapshot reads the
        disk, so the two differ while the buffer is unsaved."""
        from localharness.tools.builtin.read_tool import render_numbered  # local: import cycle

        if store is not None:
            self._store = store
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
                    ref.handle, ref.sha256 = store.put(render_numbered(data.decode("utf-8", "replace"))), digest
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

    def withdraw_references(self) -> bool:
        """The packed request cannot fit with its references: mark each one UNFIT and clear the
        store's active set for this request (the whole set, including a model's own
        tool_result_get declarations merged into the task step). False when there is nothing to
        withdraw, so the caller lets the overflow stand."""
        rec = self._live()
        if rec is None or not rec.references or self._store is None:
            return False
        for ref in rec.references:
            ref.status = UNFIT
        self._store.clear_active_references()
        self._save_quietly()
        return True

    def drop_reference(self, source: str) -> bool:
        """Remove the reference named `source`; the next refresh declares only the rest."""
        rec = self.current
        if rec is None or not any(r.source == source for r in rec.references):
            return False
        rec.references = [r for r in rec.references if r.source != source]
        if self._store is not None:
            self._store.clear_active_references()
        return True

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

    def cite(self, n: Any) -> dict[str, Any] | None:
        """The stored human turn numbered `n`, or None. The only check: the turn exists."""
        turns = self.current.human_turns if self.current else self.recent_turns
        return next((t for t in turns if t["n"] == n), None) if isinstance(n, int) else None

    def turn_range(self) -> str:
        turns = self.current.human_turns if self.current else self.recent_turns
        return f"{turns[0]['n']}-{turns[-1]['n']}" if turns else ""

    def observe_human(self, text: str) -> None:
        n = self.recent_turns[-1]["n"] + 1 if self.recent_turns else 1
        turn = {"n": n, "text": text[:MAX_TURN_CHARS]}
        self.recent_turns = [*self.recent_turns, turn][-MAX_HUMAN_TURNS:]
        if self.current is not None:
            self.current.context.observe_human(text)
            self.current.human_turns = [*self.current.human_turns, dict(turn)][-MAX_HUMAN_TURNS:]
            self._interrupt_running()
            self.save()

    def revisions(self) -> dict[str, str | None]:
        return self.current.context.revisions() if self.current else {}

    def record_result(self, tool: str, arguments: dict[str, Any], call_id: str, *,
                      success: bool, metadata: dict[str, Any], before: dict[str, str | None]) -> None:
        if self.current is None:
            return
        self.current.context.record_result(tool, arguments, call_id, success=success,
                                           metadata=metadata, before=before)
        self.save()

    def begin_delegation(self, call_id: str, agent: str, purpose: str) -> None:
        """Open a "running" entry before an `agent` call; never refuses. Inert without a live
        record. At capacity the oldest integrated entry is dropped; with every entry still
        unintegrated this call is not recorded (it still runs)."""
        rec = self._live()
        if rec is None:
            return
        n = 1 + max((int(d.id[1:]) for d in rec.delegations if d.id[1:].isdigit()), default=0)
        if len(rec.delegations) >= MAX_DELEGATIONS:
            old = next((d for d in rec.delegations if d.integrated), None)
            if old is None:
                log.warning("task record holds %d unintegrated delegations; this one is not recorded",
                            MAX_DELEGATIONS)
                return
            rec.delegations.remove(old)
        rec.delegations.append(Delegation(f"d{n}", agent[:200], purpose[:MAX_TEXT], call_id=call_id))
        self.save()

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

    def _observe_artifacts(self) -> None:
        """Append each newly seen artifact hash to its history (distinct contents, last 16 kept;
        dropped entries are counted so revision numbers never change)."""
        rec = self.current
        if rec is None:
            return
        for key, digest in rec.context.revisions().items():
            history = rec.artifact_history.setdefault(key, [])
            if digest is None or digest in history:
                continue
            history.append(digest)
            while len(history) > MAX_ARTIFACT_HISTORY:
                history.pop(0)
                rec.artifact_history_dropped[key] = rec.artifact_history_dropped.get(key, 0) + 1

    def _rev(self, key: str, digest: str | None) -> str:
        rec = self.current
        if digest is None:
            return "missing"
        history = rec.artifact_history.get(key, []) if rec else []
        if digest not in history:
            return "an unrecorded revision"
        return f"rev {rec.artifact_history_dropped.get(key, 0) + history.index(digest) + 1}"  # type: ignore[union-attr]

    def _check_facts(self) -> list[str]:
        """One fact per declared check, in declaration order; nothing here counts or judges."""
        rec = self.current
        if rec is None:
            return []
        ctx = rec.context
        outcomes, now = ctx.outcomes(), ctx.revisions()
        facts = []
        for key, req in ctx.requirements.items():
            outcome, receipt = outcomes[key], ctx.receipts.get(key)
            if outcome == "waived":
                facts.append(f"{key}: waived ({ctx.waivers[key][1]})")
            elif outcome == "stale" and receipt is not None \
                    and receipt.requirement_revision == req.fingerprint():
                facts.append(f"{key}: {receipt.outcome} (" + "; ".join(
                    f"ran at {dep} {self._rev(dep, h)}, {dep} now {self._rev(dep, now.get(dep))}"
                    for dep, h in receipt.dependencies.items() if h != now.get(dep)) + ")")
            elif outcome == "stale":
                facts.append(f"{key}: stale (declared again after its run)")
            else:
                facts.append(f"{key}: {outcome}")
        return facts

    def evidence_line(self) -> str:
        """`Task evidence: ...` when some declared check is not passed or some delegation is not
        integrated; "" otherwise (including when nothing is declared)."""
        rec = self.current
        if rec is None:
            return ""
        pending = self.unresolved_delegations()
        if all(o == "passed" for o in rec.context.outcomes().values()) and not pending:
            return ""
        items = self._check_facts() + [f"{d.agent} ({d.id}): {d.status}, not integrated" for d in pending]
        return "Task evidence: " + "; ".join(items) + "."

    def unresolved_delegations(self) -> list[Delegation]:
        return [d for d in self.current.delegations if not d.integrated] if self.current else []

    def packet(self) -> str:
        rec = self._live()
        if rec is None:
            return ""
        self._observe_artifacts()
        ctx = rec.context
        lines = [f"Task {rec.id} (working record; update it with the task tool)"]
        if rec.assignment:
            lines.append(f"Current assignment: {rec.assignment}")
        if rec.decisions:
            lines.append("Accepted decisions: " + "; ".join(
                f"[human {_turn_label(d.human_turn, _clip(d.human_text))}] {d.text}"
                if d.human_turn is not None else f"[assumption] {d.text}" for d in rec.decisions))
        if rec.human_turns:
            lines.append(f"Human turns on record: {self.turn_range()} (cite one with human_turn)")
        if rec.checkpoint_turn is not None:
            lines.append(f"Checkpoint requested at {_turn_label(rec.checkpoint_turn, _clip(rec.checkpoint_text))}")
        if rec.questions:
            lines.append("Open questions: " + "; ".join(rec.questions))
        if rec.next_action:
            lines.append(f"Next action: {rec.next_action}")
        if ctx.artifacts:
            now = ctx.revisions()
            lines.append("Artifacts: " + "; ".join(f"{k}: {self._rev(k, now.get(k))}" for k in ctx.artifacts))
        if rec.references:
            lines.append("References: " + "; ".join(f"{r.source}: {r.status}" for r in rec.references))
        if rec.judgments:
            lines.append("Editorial judgments (model opinion, not evidence): " + "; ".join(
                f"{j.key}: {j.status} — {j.criterion}" for j in rec.judgments))
        if rec.delegations:
            lines.append(f"Delegations ({len(rec.delegations)}): " + "; ".join(
                f"{d.id} {d.agent}: {d.status}" + (", integrated" if d.integrated else "")
                for d in rec.delegations))
        if self.notes:
            lines.append("Since last session: " + "; ".join(self.notes))
        evidence = [f"{fact} — {req.description}"
                    for fact, req in zip(self._check_facts(), ctx.requirements.values())]
        return "\n".join([*lines, ctx.packet(evidence)])

    def finalize(self, candidate: str) -> str:
        """Return the model's words unchanged, plus at most one appended evidence line. A record
        closed as complete whose declared evidence is settled retires here (latched)."""
        rec = self.current
        if rec is None or rec.retired:
            return candidate
        self._observe_artifacts()
        rec.context.assign_status()
        if rec.context.status == "complete" and self.unresolved_delegations():
            rec.context.status = "unknown"  # unintegrated delegated work is not finished work
        if rec.closed and self._settled(rec):
            rec.retired = True
            self.save()
            return candidate
        line = self.evidence_line()
        self.save()
        return f"{candidate}\n\n{line}" if line else candidate

    def begin(self, record: TaskRecord) -> TaskRecord | None:
        replaced = self.current if self._live() is not None else None
        record.human_turns = [dict(t) for t in self.recent_turns[-MAX_HUMAN_TURNS:]]
        if self.recent_turns and not record.context.latest_human:
            record.context.latest_human = self.recent_turns[-1]["text"]  # the turn that asked for the work
        self.current, self.notes = record, []
        return replaced

    def save(self) -> None:
        if self.path is None or self.current is None:
            return
        rec = self.current
        rec.updated = _now()
        rec.artifact_revisions = rec.context.revisions()
        self._observe_artifacts()
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
        self._observe_artifacts()
        ctx = rec.context
        if rec.retired:
            status = ("closed (no machine checks declared)" if not ctx.requirements
                      else f"closed ({ctx.status})")
        elif rec.closed:
            status = f"{ctx.status} (closed as complete; live until its evidence is settled)"
        else:
            status = ctx.status
        revs = ctx.revisions()
        lines = [f"Task {rec.id}", f"status: {status}", f"requested status: {ctx.requested_status}",
                 f"objective: {ctx.objective}", f"assignment: {rec.assignment or '-'}",
                 f"stop boundary: {ctx.stop_boundary or '-'}"]
        if rec.checkpoint_turn is not None:
            lines.append(f"checkpoint requested at {_turn_label(rec.checkpoint_turn, rec.checkpoint_text)}")
        lines.append("decisions:")
        lines += [f"  human {_turn_label(d.human_turn, d.human_text)} -> {d.text}" if d.human_turn is not None
                  else f"  assumption: {d.text}" for d in rec.decisions]
        lines.append("artifacts:")
        lines += [f"  {k}: {p} ({(revs.get(k) or 'missing')[:12]}, {self._rev(k, revs.get(k))})"
                  for k, p in ctx.artifacts.items()]
        lines.append("checks:")
        lines += [f"  {fact} ({r.origin}"
                  + (f", {_turn_label(r.human_turn, r.human_text)}" if r.human_turn is not None else "")
                  + f") — {r.description}"
                  for fact, r in zip(self._check_facts(), ctx.requirements.values())]
        lines.append("waivers:")
        lines += [f"  {k}: {w[1]}" for k, w in ctx.waivers.items()]
        lines.append("references:")
        lines += [f"  {r.source}: {r.path or 'handle ' + str(r.handle)} ({r.status})" for r in rec.references]
        lines.append("judgments:")
        lines += [f"  {j.key}: {j.status} — {j.criterion}" + "".join(
            f" | {name.replace('_', ' ')}: {getattr(j, name)}" for name in ("assessment", "passages", "fix")
            if getattr(j, name)) + (f" | waived by {j.waived_by}" if j.waived_by else "")
            for j in rec.judgments]
        lines.append("delegations:")
        lines += [f"  {d.id} {d.agent}: {d.status} — {d.purpose}"
                  + (f" | artifacts: {', '.join(d.artifacts)}" if d.artifacts else "")
                  + (f" | integrated: {d.integrated}" if d.integrated else "")
                  for d in rec.delegations]
        lines.append("questions:")
        lines += [f"  {q}" for q in rec.questions]
        lines.append("human turns:")
        lines += [f"  {t['n']}: {_clip(t['text'])}" for t in rec.human_turns]
        lines += [f"next action: {rec.next_action or '-'}", f"file: {self.path}"]
        if self.notes:
            lines.append("since last session: " + "; ".join(self.notes))
        return "\n".join(lines)
