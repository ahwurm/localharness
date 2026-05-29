"""AUTO-01: the autonomous self-improvement loop driver (pure orchestration; NO Typer).

Integrates the three Wave-2 leaf modules + the Phase 16/17 primitives into ONE strictly
sequential driver:

    sample (18-02 ParentSampler)
      -> propose (16 proposer)
        -> write in_flight archive row (15)
          -> run_experiment under a per-proposal timeout (17 gate; exit code = verdict)
            -> interpret exit code + the post-run row -> adopt / hold / reject / skip
              -> adopt (18-04) commits a clean win to HEAD -> next experiment composes on it
                -> journal the loop-level "why" (per-run JSONL)
    ... repeat until the PRE-FLIGHT budget gate (18-03 BudgetController) trips, the
        consecutive-failure circuit breaker halts, or a graceful interrupt is requested.

Design north star (CONTEXT): "I really don't want to be involved." Fully non-blocking —
per-iteration detail goes to the journal / stdout / EventBus; Discord stays quiet during a
run (only the run-complete summary + the Phase 19 report/sentinel reach Discord).

CRITICAL invariants:
  - STRICTLY SEQUENTIAL — write in_flight -> run_experiment -> adopt -> next. NEVER
    batch-propose-then-batch-run: the gate always runs at HEAD, which already holds every
    prior adoption, so the single-component overlay composes for free (the ``before`` is
    never stale; no overlay replay).
  - The TOTAL-cap gate is ``BudgetController.can_start_iteration()`` at the TOP of the loop
    ONLY. It NEVER cancels a running experiment — a total breach halts BEFORE the next
    iteration. The ONLY mid-experiment kill is ``asyncio.wait_for(.., proposal_timeout)``.
  - The circuit breaker increments on proposer error / timeout / structural-skip (>=4) and
    RESETS to 0 on ANY completed gate verdict (exit 0-3).
  - An ``AdoptionRefused`` is a guard, not an outage — count it skipped, reset the breaker.
  - The run halts cleanly: the gate exit codes NEVER propagate as the run's own exit code.

Pure module: inject ``propose_fn`` / ``experiment_fn`` / ``adopt_fn`` / ``clock`` / ``meter``
/ ``rng`` / ``interrupt`` for hermetic, LLM-free / bench-free tests.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# Exit constants — SINGLE source of truth (the gate writes them; the loop reads them).
from localharness.autoresearch.experiment import (
    EXIT_INCONCLUSIVE,
    EXIT_PROMOTE,
    EXIT_REJECT_HOLDOUT,
    EXIT_REJECT_TRAIN,
)

# Loop-level decisions (distinct from the gate's exit codes).
ADOPT, HOLD, REJECT, SKIP = "adopt", "hold", "reject", "skip"


def interpret(exit_code: int, entry, min_lift: float | None) -> tuple[str, str]:
    """Map a gate exit code + the post-run archive row to a (decision, reason).

    'lift' is the effect size: v1 uses the ``train_score`` the gate already wrote (the gate's
    statistical significance is the FLOOR; ``--min-lift`` adds an effect-size floor ON TOP).
    Per 18-RESEARCH Open Q #1 we ship ``--min-lift`` with NO hard-coded "calibrated" number —
    ``min_lift=None`` adopts ANY clean win, and the observed lift is emitted into the journal so
    the user calibrates the floor from real data rather than a guessed constant.

      exit 0 (promote): adopt if min_lift is None or lift >= min_lift; else hold (thin lift).
      exit 3 (inconclusive): hold.
      exit 1/2 (reject train/holdout): reject (no adopt; the gate already wrote the verdict).
      exit >=4 (structural refusal): skip + log.
    """
    if exit_code == EXIT_PROMOTE:
        lift = entry.train_score if entry is not None else None
        if min_lift is None or (lift is not None and lift >= min_lift):
            return ADOPT, f"clean win, lift={lift}"
        return HOLD, f"below --min-lift ({lift} < {min_lift})"
    if exit_code == EXIT_INCONCLUSIVE:
        return HOLD, "inconclusive (gate exit 3)"
    if exit_code in (EXIT_REJECT_TRAIN, EXIT_REJECT_HOLDOUT):
        return REJECT, f"gate reject (exit {exit_code})"
    return SKIP, f"structural refusal (exit {exit_code})"


@dataclass
class RunSummary:
    """The terminal artifact of a run: counts + consumption + the journal path.

    ``halt_reason`` is one of "budget" | "circuit_breaker" | "interrupt" | "complete".
    ``top_wins`` is a list of (archive_id, component, train_score) for the run-complete report.
    """

    run_id: str
    iterations: int = 0
    adopted: int = 0
    held: int = 0
    rejected: int = 0
    skipped: int = 0
    failures: int = 0
    seconds_elapsed: float = 0.0
    tokens_spent: int = 0
    top_wins: list = field(default_factory=list)
    journal_path: str = ""
    halt_reason: str = ""


class RunJournal:
    """Per-run JSONL audit at ``.localharness/autoresearch/runs/<run_id>.jsonl``.

    Atomic O_APPEND lines (the EventBus idiom, core/bus.py:106): a JSON object per line,
    appended in mode "a" (O_APPEND is atomic under PIPE_BUF on POSIX). The archive stays the
    source of truth for mutation DETAIL; each journal line points at ``archive_id`` and records
    the loop-level "why" (branch + ε roll, lineage, gate verdict, decision + reason, budget).
    """

    def __init__(self, run_id: str, base_dir):
        self._run_id = run_id
        self._path = Path(base_dir) / "autoresearch" / "runs" / f"{run_id}.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def write(self, record: dict) -> None:
        line = json.dumps({"ts": time.time(), "run_id": self._run_id, **record}) + "\n"
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line)
