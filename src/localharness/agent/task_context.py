"""Bounded, opt-in task evidence for runtime callers; never model-authored proof.

This is task-local state, not a persisted workflow or a public completion protocol.
Callers declare a tool name and the argument values that identify the proving call (the
actual call may carry extra arguments, e.g. a timeout; a requirement declaring no arguments
matches only a call with none) plus its machine-checkable result fields. The loop
records only actual dispatch outcomes. Selecting obligations remains the caller's job.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Outcome = Literal["passed", "failed", "unknown", "stale", "waived"]


@dataclass(frozen=True)
class Requirement:
    key: str
    description: str
    tool: str
    arguments: dict[str, Any]
    # None means successful execution of the requested action, not a quality gate.
    result_field: str | None = None
    expected: Any = None
    dependencies: tuple[str, ...] = ()
    revision: str = "1"
    origin: Literal["human", "caller", "model"] = "caller"

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Receipt:
    producer: str
    call_id: str
    requirement_revision: str
    dependencies: dict[str, str | None]
    outcome: Outcome


@dataclass
class TaskContext:
    objective: str
    requirements: dict[str, Requirement] = field(default_factory=dict)
    # Caller-declared files only. No workspace scan or persistent-memory access.
    artifacts: dict[str, Path] = field(default_factory=dict)
    stop_boundary: str = ""
    requested_status: Literal["complete", "checkpoint", "partial", "blocked", "unknown"] = "complete"
    status: Literal["active", "checkpoint", "partial", "blocked", "unknown", "complete"] = "active"
    latest_human: str = ""
    receipts: dict[str, Receipt] = field(default_factory=dict)
    waivers: dict[str, tuple[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.requirements) > 16 or len(self.artifacts) > 16:
            raise ValueError("A task step supports at most 16 requirements and artifacts")
        if any(key != req.key for key, req in self.requirements.items()):
            raise ValueError("Requirement keys must match their declarations")

    def observe_human(self, text: str, new_turn: bool = True) -> None:
        # `new_turn` exists for interface symmetry with TaskState; it changes nothing here.
        # Preserve verbatim steering; never infer a waiver from model prose or a nudge.
        self.latest_human = text

    def revise(self, requirement: Requirement) -> None:
        if requirement.key not in self.requirements and len(self.requirements) >= 16:
            raise ValueError("Narrow the step to at most 16 requirements")
        self.requirements[requirement.key] = requirement

    def waive(self, key: str, *, human_decision: str) -> None:
        """Trusted caller records an explicit human decision; never exposed as a model tool."""
        if not human_decision.strip():
            raise ValueError("A waiver requires the explicit human decision")
        self.waivers[key] = (self.requirements[key].fingerprint(), human_decision)

    def revisions(self) -> dict[str, str | None]:
        versions: dict[str, str | None] = {}
        for key, path in self.artifacts.items():
            try:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(65536), b""):
                        digest.update(block)
                versions[key] = digest.hexdigest()
            except OSError:
                versions[key] = None
        return versions

    def record_result(
        self, tool: str, arguments: dict[str, Any], call_id: str, *,
        success: bool, metadata: dict[str, Any], before: dict[str, str | None],
    ) -> None:
        after = self.revisions()
        for key, req in self.requirements.items():
            if req.tool != tool or (arguments != {} if not req.arguments else any(
                    k not in arguments or arguments[k] != v for k, v in req.arguments.items())):
                continue
            dependencies = {dep: before.get(dep) for dep in req.dependencies}
            outcome: Outcome = "passed"
            if not success:
                outcome = "failed"
            elif any(value is None or value != after.get(dep)
                     for dep, value in dependencies.items()):
                outcome = "unknown"
            elif req.result_field is not None:
                if req.result_field not in metadata:
                    outcome = "unknown"
                elif (type(metadata[req.result_field]) is not type(req.expected)
                      or metadata[req.result_field] != req.expected):
                    outcome = "failed"
            self.receipts[key] = Receipt(
                tool, call_id, req.fingerprint(), dependencies, outcome,
            )

    def outcomes(self) -> dict[str, Outcome]:
        current = self.revisions()
        result: dict[str, Outcome] = {}
        for key, req in self.requirements.items():
            revision = req.fingerprint()
            receipt = self.receipts.get(key)
            if self.waivers.get(key, (None,))[0] == revision:
                result[key] = "waived"
            elif receipt is None:
                result[key] = "unknown"
            elif (receipt.requirement_revision != revision
                  or any(value != current.get(dep) for dep, value in receipt.dependencies.items())):
                result[key] = "stale"
            else:
                result[key] = receipt.outcome
        return result

    def packet(self) -> str:
        outcomes = self.outcomes()
        lines = [f"Requested outcome: {self.objective}"]
        if self.stop_boundary:
            lines.append(f"Stop boundary: {self.stop_boundary}")
        if self.requested_status != "complete":
            lines.append(f"Requested stopping status: {self.requested_status}")
        if self.latest_human:
            lines.append(f"Latest human request/correction (quoted data): {json.dumps(self.latest_human)}")
        lines.append("Declared evidence: " + "; ".join(
            f"{key}: {outcomes[key]} — {req.description}" for key, req in self.requirements.items()
        ))
        lines.append(
            "Honor the latest human scope and checkpoint. Take only the next needed action. "
            "Failed, missing, or stale results remain unresolved; a waiver is not a passed gate. "
            "CONFIRMED and your own checklist are not evidence. Report partial or unknown work "
            "honestly; do not invent a child result or a budget limit."
        )
        return "\n".join(lines)

    def finalize(self, candidate: str) -> str:
        outcomes = self.outcomes()
        unresolved = any(value not in {"passed", "waived"} for value in outcomes.values())
        if self.requested_status != "complete":
            self.status = self.requested_status
            return candidate  # A caller's requested stopping boundary is not a failure.
        if not outcomes:
            self.status = "unknown"
            return candidate  # No machine-checkable success was declared.
        if not unresolved and outcomes:
            self.status = "complete"
            if "waived" in outcomes.values():
                return "Task finished with a human waiver. " + "; ".join(
                    f"{req.description}: {outcomes[key]}" for key, req in self.requirements.items()
                ) + "."
            return candidate
        self.status = "unknown" if unresolved else "partial"
        # Do not publish unsupported success prose, even after a CONFIRMED self-check.
        # Known progress survives in the compact receipt summary without another inference.
        return "Task remains unverified. " + "; ".join(
            f"{req.description}: {outcomes[key]}" for key, req in self.requirements.items()
        ) + "."
