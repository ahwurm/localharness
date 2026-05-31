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
    iteration. The ONLY mid-experiment kill is the per-proposal timeout (the single wait-for
    around ``experiment_fn`` below).
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

from localharness.autoresearch.adoption import AdoptionRefused
from localharness.autoresearch.adoption import adopt as _default_adopt
from localharness.autoresearch.archive import ArchiveEntry, ArchiveQuery

# Exit constants — SINGLE source of truth (the gate writes them; the loop reads them).
from localharness.autoresearch.experiment import (
    EXIT_INCONCLUSIVE,
    EXIT_PROMOTE,
    EXIT_REJECT_HOLDOUT,
    EXIT_REJECT_TRAIN,
    _provenance_agent_cfg,
)
from localharness.config.overlay import _resolve_user_overlay_path, load_overlay
from localharness.autoresearch.budget import BudgetController, WindowMeter
from localharness.autoresearch.sampler import BASELINE_ROOT, ParentSampler
from localharness.registry import build_catalogue

# Loop-level decisions (distinct from the gate's exit codes).
ADOPT, HOLD, REJECT, SKIP = "adopt", "hold", "reject", "skip"

# Consecutive proposer-error / timeout / structural-skip failures before the loop halts.
# Research says 3-5; 4 balances tolerating a transient blip against bleaking budget on a
# systemic outage (e.g. the proposer model is down). Resets to 0 on any real gate verdict.
CIRCUIT_BREAKER_N = 4

# The high metering-bug backstop: even with no wallclock/window cap, the loop can never run
# unbounded. The CLI (18-06) always passes a high value; this is the library default.
DEFAULT_MAX_ITERATIONS = 1000


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

    @property
    def wallclock_elapsed(self) -> float:
        """Alias for ``seconds_elapsed`` (the CLI/report + tests read wallclock_elapsed)."""
        return self.seconds_elapsed


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


# ---------------------------------------------------------------------------
# Phase 19 — the CHEAP, write-only, NON-BLOCKING inline sentinel hook (REP-03/04)
#
# A module-level seam (tests monkeypatch ``run_inline_sentinel`` to assert a sentinel
# bug NEVER crashes the fire-and-forget loop — Pitfall 4). Per locked decision 5 +
# 19-RESEARCH § Pattern 3, the inline hook runs ONLY the cheap signals over a small
# recent window (the just-written row's overfit gap if it reached holdout + the near-
# duplicate check over the recent same-component proposals). The full saturation/
# rotation pass stays on-demand in ``autoresearch report`` (19-04) — never per-iteration.
# This function WRITES to the journal (and, if a bus is given, the EventBus) only; it
# NEVER writes to the archive's train/holdout columns and NEVER halts the loop.
# ---------------------------------------------------------------------------


def _sentinel_cfg(cfg):
    """The SentinelConfig to read thresholds from — defaults when cfg/cfg.sentinel is absent.

    The loop tolerates ``cfg=None`` (the hermetic loop tests pass it), so the inline hook falls
    back to a default-constructed SentinelConfig rather than dereferencing None.
    """
    sentinel = getattr(cfg, "sentinel", None)
    if sentinel is None:
        from localharness.config.models import SentinelConfig
        sentinel = SentinelConfig()
    return sentinel


async def run_inline_sentinel(store, cfg, journal, bus, component):
    """CHEAP, NON-BLOCKING sentinel check beside the per-iteration journal write.

    Fetches a bounded recent same-component window (NOT the whole archive — that is the on-demand
    pass's job), runs the cheap overfit-gap + near-duplicate signals, journals any SentinelAlert
    as a write-only ``phase="sentinel"`` line, and (if a bus is attached) publishes it. Returns
    the SentinelReport. The CALLER wraps this in try/except so ANY failure here is swallowed —
    a sentinel bug can NEVER halt the loop (AUTO-04 amended, Pitfall 4).
    """
    from localharness.autoresearch.sentinel import (
        RotationSuggestion,
        SentinelReport,
        alerts_from_report,
        near_duplicate_runs,
        overfit_gaps,
    )

    sentinel = _sentinel_cfg(cfg)
    # Bounded recent window: the just-written row + enough same-component history for the dup check.
    recent_rows = await store.query(
        ArchiveQuery(component=component, limit=sentinel.duplicate_consecutive_k + 1)
    )
    gaps = overfit_gaps(recent_rows, sentinel.overfit_gap_threshold)
    dups = near_duplicate_runs(
        sorted(recent_rows, key=lambda r: r.ts),
        sentinel.duplicate_similarity,
        sentinel.duplicate_consecutive_k,
    )
    # Saturation/rotation stays on-demand (empty here): the inline check is intentionally cheap.
    report = SentinelReport(
        gaps=gaps, duplicates=dups, rotation=RotationSuggestion([], "", {})
    )

    alerts = alerts_from_report(report)
    for alert in alerts:
        journal.write({
            "phase": "sentinel", "kind": alert.kind, "detail": alert.detail,
            "mutation_id": alert.mutation_id, "metric_value": alert.metric_value,
        })
        if bus is not None:
            await bus.publish(alert)
    return report


# ---------------------------------------------------------------------------
# Target derivation + real-fn factories (the loop body calls a uniform 2-arg
# propose_fn(component, run_ids) and 1-arg experiment_fn(pid); tests inject these).
# ---------------------------------------------------------------------------


def _diff_blob(proposal, cfg) -> str:
    """Single-encoded archive diff blob incl. rationale + kind (GAP-1)."""
    try:
        _user_ov = load_overlay(_resolve_user_overlay_path())
        type_name = build_catalogue(
            cfg,
            agent_cfg=_provenance_agent_cfg(),
            overlays={"user": _user_ov},
        )[proposal.component].type_name
    except Exception:
        type_name = ""
    kind = "hyperparameter" if type_name in ("int", "float") else "prompt"
    return json.dumps({
        "before": proposal.before,
        "after": proposal.after,
        "rationale": proposal.rationale,
        "kind": kind,
    })


async def _derive_target(parent, store, cfg) -> tuple[str, list, str]:
    """Pick (component, run_ids, branch) for this iteration.

    branch is "cold_start" when ``parent is BASELINE_ROOT`` else "sampled" (the sampler
    internally chose explore/exploit; the loop records cold-vs-sampled, keeping it simple).
    Component policy is "go broad, not deep": bias toward the parent's component when a parent
    exists; on a cold start pick a component nothing has touched yet (so coverage spreads),
    falling back to the first catalogue entry. ``run_ids`` are the live failed-TRAIN run_ids the
    proposer reads (derived from the latest failed train traces in production); in tests
    ``propose_fn`` is injected so run_ids is a passthrough. The proposer's PROP-03 seal validates
    run_ids, so an empty/garbage list is refused there (not here).
    """
    if parent is not BASELINE_ROOT and parent is not None:
        return parent.component, [], "sampled"
    _user_ov = load_overlay(_resolve_user_overlay_path())
    catalogue = build_catalogue(
        cfg,
        agent_cfg=_provenance_agent_cfg(),
        overlays={"user": _user_ov},
    )
    touched = {e.component for e in await store.query(ArchiveQuery(limit=10_000))}
    untouched = [p for p in catalogue if p not in touched]
    component = untouched[0] if untouched else next(iter(catalogue))
    return component, [], "cold_start"


def _real_propose(cfg, corpus_path, results_path):
    """Bind the real proposer into the loop's uniform 2-arg propose_fn(component, run_ids) -> pid.

    The real ``propose()`` returns a Proposal; the loop archives it to an in_flight row and runs
    the gate on that row. Tests inject ``propose_fn`` directly (returning a seeded pid), so this
    closure is only used in production.
    """

    async def _fn(component, run_ids):
        from localharness.autoresearch.proposer import propose

        return await propose(
            component, run_ids, cfg=cfg, corpus_path=corpus_path, results_path=results_path
        )

    return _fn


def _real_experiment(cfg, repo_root, store, bus):
    """Bind the real gate into the loop's uniform 1-arg experiment_fn(pid) -> exit_code."""

    async def _fn(pid):
        from localharness.autoresearch.experiment import run_experiment

        return await run_experiment(pid, store=store, repo_root=repo_root, cfg=cfg, bus=bus)

    return _fn


def _stop_requested(should_stop, interrupt) -> bool:
    """True if a graceful stop was requested via the should_stop callable OR the interrupt event.

    The binding test passes a ``threading.Event`` as ``interrupt``; the plan also documents a
    ``should_stop`` callable. Both are honored (checked at the TOP of the loop, so the current
    experiment always finishes before the next starts).
    """
    if should_stop is not None and should_stop():
        return True
    if interrupt is not None and interrupt.is_set():
        return True
    return False


async def run_loop(
    *,
    store,
    cfg,
    repo_root,
    budget=None,
    max_iterations=DEFAULT_MAX_ITERATIONS,
    max_cost=None,
    epsilon=0.2,
    min_lift=None,
    proposal_timeout=1800.0,
    window_tokens=None,
    corpus_path=None,
    results_path=None,
    journal=None,
    rng=None,
    clock=None,
    wall_clock=None,
    propose_fn=None,
    experiment_fn=None,
    adopt_fn=None,
    meter=None,
    should_stop=None,
    interrupt=None,
    bus=None,
) -> RunSummary:
    """Drive the sequential self-improvement loop until a budget gate / breaker / interrupt halts.

    sample -> propose -> write in_flight -> run_experiment (timeout-bounded) -> interpret ->
    adopt/hold/reject/skip -> journal -> repeat. Returns a RunSummary; ALWAYS exits cleanly
    (the gate's exit codes never become the run's exit code). See the module docstring for the
    invariants this function is contractually bound to.
    """
    run_id = uuid.uuid4().hex[:12]
    base_dir = Path(repo_root) / ".localharness"
    journal = journal or RunJournal(run_id, base_dir)
    summary = RunSummary(run_id=run_id, journal_path=str(journal.path))

    clock = clock or time.monotonic
    meter = meter or WindowMeter(
        window_budget_tokens=window_tokens,
        state_path=base_dir / "autoresearch" / "window.json",
        clock=(wall_clock or time.time),
    )
    sampler = ParentSampler(store, epsilon=epsilon, rng=rng)
    budget_ctl = BudgetController(
        budget_seconds=budget,
        max_iterations=max_iterations,
        max_cost=max_cost,
        meter=meter,
        clock=clock,
    )
    propose_fn = propose_fn or _real_propose(cfg, corpus_path, results_path)
    experiment_fn = experiment_fn or _real_experiment(cfg, repo_root, store, bus)
    adopt_fn = adopt_fn or _default_adopt

    consecutive_failures = 0
    run_start = clock()

    # PRE-FLIGHT gate at the TOP of the loop — the ONLY total-cap check (never mid-experiment).
    while budget_ctl.can_start_iteration():
        # Graceful interrupt: checked at the top so the current experiment always finishes
        # before the next starts (the in-flight iteration below runs to completion uninterrupted).
        if _stop_requested(should_stop, interrupt):
            summary.halt_reason = "interrupt"
            break

        summary.iterations += 1

        # 1. SAMPLE — a parent (lineage + "go broad" bias) or the cold-start baseline root.
        epsilon_roll = sampler._rng.random() if hasattr(sampler, "_rng") else None
        parent = await sampler.sample()
        is_cold = parent is BASELINE_ROOT
        parent_id = None if is_cold else parent.id
        component, run_ids, branch = await _derive_target(parent, store, cfg)

        # 2. PROPOSE — a proposer failure is a circuit-breaker increment, not a crash.
        try:
            proposal = await propose_fn(component, run_ids)
        except Exception as exc:  # ProposerError or any transient model/IO failure
            consecutive_failures += 1
            summary.failures += 1
            journal.write({
                "iteration": summary.iterations, "phase": "propose", "branch": branch,
                "epsilon_roll": epsilon_roll, "parent_id": parent_id, "component": component,
                "archive_id": None, "exit_code": None, "decision": SKIP,
                "reason": f"proposer error: {exc}", "budget": budget_ctl.snapshot(),
            })
            if consecutive_failures >= CIRCUIT_BREAKER_N:
                summary.halt_reason = "circuit_breaker"
                break
            continue

        # The real propose() returns a Proposal; the injected fake returns a seeded pid directly.
        # Self-meter ONLY the proposer's reported tokens (local bench inference is never metered).
        tokens_used = getattr(proposal, "tokens_used", None)
        if tokens_used is not None:
            meter.record_tokens(tokens_used or 0)
        summary.tokens_spent = meter.snapshot().get("tokens_spent", summary.tokens_spent)

        # 3. RESOLVE / WRITE the in_flight row. A Proposal is archived now; a bare pid (test seam)
        #    points at an already-seeded row. STRICTLY SEQUENTIAL: one row, run it, act, then next.
        if isinstance(proposal, str):
            pid = proposal
        else:
            pid = (await store.write(ArchiveEntry(
                id=uuid.uuid4().hex, parent_id=parent_id, component=proposal.component,
                diff=_diff_blob(proposal, cfg), train_score=None,
                train_scores_per_fixture=None, holdout_score=None, p_value=None, cost=None,
                ts=int(time.time()), approved_by=None, status="in_flight",
            ))).id

        # 4. RUN_EXPERIMENT under the per-proposal timeout. This wait_for is the ONLY mid-experiment
        #    kill — the total-cap gate above NEVER cancels a running experiment.
        try:
            exit_code = await asyncio.wait_for(experiment_fn(pid), timeout=proposal_timeout)
        except asyncio.TimeoutError:
            await store.update_verdict(pid, status="train_rejected")  # killed run = negative signal
            consecutive_failures += 1
            summary.failures += 1
            journal.write({
                "iteration": summary.iterations, "phase": "experiment", "branch": branch,
                "epsilon_roll": epsilon_roll, "parent_id": parent_id, "component": component,
                "archive_id": pid, "exit_code": None, "decision": SKIP,
                "reason": f"proposal-timeout after {proposal_timeout}s",
                "budget": budget_ctl.snapshot(),
            })
            if consecutive_failures >= CIRCUIT_BREAKER_N:
                summary.halt_reason = "circuit_breaker"
                break
            continue

        # 5. INTERPRET + ACT on the gate's verdict.
        row = await store.get(pid)
        decision, reason = interpret(exit_code, row, min_lift)

        if decision == ADOPT:
            try:
                await adopt_fn(pid, store=store, cfg=cfg, repo_root=repo_root, bus=bus)
                await store.update_verdict(pid, status="adopted")
                summary.adopted += 1
                summary.top_wins.append(
                    (pid, getattr(row, "component", component),
                     row.train_score if row else None)
                )
                consecutive_failures = 0  # a real verdict resets the breaker
            except AdoptionRefused:
                # A seal guard is NOT an outage — count it skipped, reset the breaker, continue.
                summary.skipped += 1
                consecutive_failures = 0
        elif decision == HOLD:
            await store.update_verdict(pid, status="held")
            summary.held += 1
            consecutive_failures = 0
        elif decision == REJECT:
            # The real run_experiment already wrote train_rejected / holdout_rejected; persist it
            # idempotently here too so the row reflects the gate's verdict even under an injected
            # experiment_fn that returns the exit code only (and the status never silently stays
            # in_flight on a reject).
            reject_status = (
                "train_rejected" if exit_code == EXIT_REJECT_TRAIN else "holdout_rejected"
            )
            await store.update_verdict(pid, status=reject_status)
            summary.rejected += 1
            consecutive_failures = 0
        else:  # SKIP (>=4 structural refusal)
            summary.skipped += 1
            consecutive_failures += 1
            if consecutive_failures >= CIRCUIT_BREAKER_N:
                summary.halt_reason = "circuit_breaker"

        journal.write({
            "iteration": summary.iterations, "phase": "decision", "branch": branch,
            "epsilon_roll": epsilon_roll, "parent_id": parent_id, "component": component,
            "archive_id": pid, "exit_code": exit_code, "decision": decision,
            "reason": reason, "budget": budget_ctl.snapshot(),
        })

        # CHEAP, write-only, NON-BLOCKING inline sentinel check (Phase 19, REP-03/04).
        # Resolved through the module global so tests can monkeypatch the seam; the bare
        # ``except Exception: pass`` is REQUIRED — a sentinel bug NEVER halts the fire-and-forget
        # loop (AUTO-04 amended, Pitfall 4). It journals + (if bus) emits alerts; never writes back.
        try:
            await run_inline_sentinel(store, cfg, journal, bus, component)
        except Exception:
            pass

        if summary.halt_reason == "circuit_breaker":
            break

    if not summary.halt_reason:
        summary.halt_reason = "budget"  # the clean cap-trip default
    summary.seconds_elapsed = clock() - run_start
    summary.tokens_spent = meter.snapshot().get("tokens_spent", summary.tokens_spent)
    journal.write({
        "iteration": summary.iterations, "phase": "complete", "decision": "complete",
        "reason": summary.halt_reason, "adopted": summary.adopted, "held": summary.held,
        "rejected": summary.rejected, "skipped": summary.skipped,
        "halt_reason": summary.halt_reason, "budget": budget_ctl.snapshot(),
    })
    return summary
