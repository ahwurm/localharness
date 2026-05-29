"""EXP-01..05: the promotion-gate orchestration core (pure; NO Typer).

Composes the Phase 13/14/15/16 + 17-02 primitives into the two-stage promotion gate:
load an in_flight proposal, refuse structurally-invalid proposals BEFORE any bench run
(multi-component / unknown-id / off-registry-grader / malformed → exit >=4, OUTSIDE the
0-3 gate band), run the proposal inside a throwaway ``git worktree``, gate on Welch
improvement (TRAIN) then Bonferroni-corrected non-regression (HOLDOUT), and write the
verdict back to the archive with the EXP-05 exit-code semantics.

The CLI wrapper is 17-04. Tests drive this hermetically by injecting ``run_slice``
(a fake per-fixture success map) + a temp ``ArchiveStore``. Composed primitives:
  - 17-02:  welch_improvement / welch_regression  (bench.aggregator)
            ArchiveStore.update_verdict           (autoresearch.archive)
  - 14:     build_catalogue / coerce_value / set_value_in_dict  (registry)
            atomic_write_overlay                  (config.overlay)
            ComponentMutated / EventBus           (core)
  - 13:     accumulate_runs / metrics_summary / scenario discovery  (bench)
  - prefix: _resolve                              (cli.autoresearch_cmd)
"""
from __future__ import annotations

import contextlib
import os
import re
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from localharness.bench.aggregator import welch_improvement, welch_regression
from localharness.config.overlay import atomic_write_overlay
from localharness.registry import build_catalogue, coerce_value, set_value_in_dict

# ---------------------------------------------------------------------------
# Exit-code scheme (EXP-05). The gate verdicts occupy 0-3; structural refusals
# occupy >=4 so a refusal can NEVER be confused with a statistical verdict.
# ---------------------------------------------------------------------------

EXIT_PROMOTE = 0
EXIT_REJECT_TRAIN = 1
EXIT_REJECT_HOLDOUT = 2
EXIT_INCONCLUSIVE = 3
EXIT_REFUSE_MULTI_COMPONENT = 4
EXIT_REFUSE_UNKNOWN_ID = 5
EXIT_REFUSE_OFFREGISTRY = 6
EXIT_REFUSE_MALFORMED = 7

# Belt-and-suspenders prefixes for the anti-reward-hacking seal (Pitfall 3). The
# registry already excludes these by omission; this is defense-in-depth.
_OFFREGISTRY_PREFIXES = ("bench.", "scenario", "grader", "success_criteria", "holdout")

# A component path encoding >1 dot-path (mirrors components_cmd._MULTI_PATH_PATTERN).
_MULTI_PATH_PATTERN = re.compile(r"[,\s;]")


class ExperimentRefusal(Exception):
    """Structural refusal raised BEFORE any bench run. Carries an exit code (>=4)."""

    def __init__(self, exit_code: int, message: str = "") -> None:
        super().__init__(message or f"experiment refused (exit {exit_code})")
        self.exit_code = exit_code


# ---------------------------------------------------------------------------
# repo-root resolution
# ---------------------------------------------------------------------------


def _git_root(start: Path | None = None) -> Path:
    """Resolve the enclosing git toplevel. Defaults to this module's repo."""
    base = Path(start) if start is not None else Path(__file__).resolve()
    out = subprocess.run(
        ["git", "-C", str(base if base.is_dir() else base.parent),
         "rev-parse", "--show-toplevel"],
        check=True, capture_output=True, text=True,
    )
    return Path(out.stdout.strip())


# ---------------------------------------------------------------------------
# Pattern 2: throwaway git worktree at the current committed tree (SC1)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def experiment_worktree(repo_root: Path, *, keep: bool = False):
    """Yield a throwaway worktree at HEAD; teardown on exit (unless keep=True).

    ``--detach`` gives a throwaway HEAD with NO stray branch and NO branch move; the
    linked worktree shares the object DB + refs but has a per-worktree HEAD/index/tree,
    so writes inside never touch the main checkout. ``remove --force`` is mandatory: the
    tree is always unclean (we materialize the experiment overlay + bench results into it).
    A new worktree carries ONLY the committed tree — the corpus (tracked) is present; the
    generated overlay (untracked) is NOT, so it must be written in (write_experiment_overlay).
    """
    wt = Path(tempfile.mkdtemp(prefix="lh-exp-"))
    subprocess.run(
        ["git", "-C", str(repo_root), "worktree", "add", "--detach", str(wt), "HEAD"],
        check=True, capture_output=True,
    )
    try:
        yield wt
    finally:
        if not keep:
            subprocess.run(
                ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(wt)],
                check=False, capture_output=True,
            )


# ---------------------------------------------------------------------------
# Pattern 3 (Option B): materialize the mutation INSIDE the worktree so it
# physically cannot leak into the user's real ~/.localharness.
# ---------------------------------------------------------------------------


def write_experiment_overlay(
    worktree: Path, component: str, after: Any, annotation: Any = None
) -> Path:
    """Write the experiment overlay (the single-component mutation) into the worktree.

    Builds a nested overlay dict via set_value_in_dict({}, component, after) — coercing
    ``after`` through the SAME coerce_value the proposer/components-set use when an
    annotation is supplied, so types round-trip identically — then atomic_write_overlay
    to <worktree>/.localharness/experiment-overlay.yaml. Returns the written path.
    """
    value = coerce_value(str(after), annotation) if annotation is not None else after
    overlay_dict = set_value_in_dict({}, component, value)
    overlay_path = worktree / ".localharness" / "experiment-overlay.yaml"
    atomic_write_overlay(overlay_path, overlay_dict)
    return overlay_path


# ---------------------------------------------------------------------------
# Structural-refusal gate (EXP-02 + anti-reward-hacking seal) — BEFORE any bench
# ---------------------------------------------------------------------------


def _is_multi_component(component: str, after: Any) -> bool:
    """True iff the proposal resolves to >1 component path.

    entry.component is the canonical SINGLE path; a multi-path diff is detected either by
    a delimiter in the component string OR an after-value that is a path→value MAP whose
    keys look like >1 dot-path (mirrors the proposer's single-component contract).
    """
    if _MULTI_PATH_PATTERN.search(component):
        return True
    if isinstance(after, dict) and len(after) > 1:
        # A diff whose `after` is a {dot.path: value} map for multiple components.
        if all(isinstance(k, str) and "." in k for k in after):
            return True
    return False


async def _load_and_validate(store, proposal_id: str, cfg) -> tuple[Any, str, Any]:
    """Resolve + structurally validate a proposal. Returns (entry, component, after_value).

    Raises ExperimentRefusal(exit_code>=4) for every structural problem, ALWAYS before any
    worktree creation or bench run (the seal must refuse without giving the proposal a
    chance to move the goalposts).
    """
    from localharness.cli.autoresearch_cmd import _resolve

    # 1. id resolution (full UUID or 8-char prefix) — reuse the archive resolver.
    entry, matches = await _resolve(store, proposal_id)
    if entry is None:  # not-found ([]) or ambiguous ([>1]) — both are structural.
        raise ExperimentRefusal(EXIT_REFUSE_UNKNOWN_ID,
                                f"proposal id not uniquely resolvable: {proposal_id!r}")

    # 2. malformed diff: must decode to a dict with before/after keys.
    try:
        decoded = entry.diff_decoded
    except (ValueError, TypeError):
        raise ExperimentRefusal(EXIT_REFUSE_MALFORMED, "diff is not valid JSON")
    if not isinstance(decoded, dict) or "before" not in decoded or "after" not in decoded:
        raise ExperimentRefusal(EXIT_REFUSE_MALFORMED, "diff missing before/after keys")
    after = decoded["after"]
    component = entry.component

    # 3. EXP-02 multi-component refusal.
    if _is_multi_component(component, after):
        raise ExperimentRefusal(EXIT_REFUSE_MULTI_COMPONENT,
                                f"proposal resolves to >1 component: {component!r}")

    # 4. Anti-reward-hacking seal: off-registry / grader / holdout / bench surface.
    #    Belt: prefix check first (cheap, explicit), then the registry membership check.
    if any(component.startswith(p) for p in _OFFREGISTRY_PREFIXES):
        raise ExperimentRefusal(EXIT_REFUSE_OFFREGISTRY,
                                f"component targets a sealed surface: {component!r}")
    catalogue = build_catalogue(cfg)
    if catalogue.get(component) is None:
        raise ExperimentRefusal(EXIT_REFUSE_OFFREGISTRY,
                                f"component not in the mutable registry: {component!r}")

    return entry, component, after


# ---------------------------------------------------------------------------
# Per-fixture success-rate vector extraction (Task 2 implements the body)
# ---------------------------------------------------------------------------


async def slice_success_by_fixture(
    scenarios, model, results_root, factory, *, min_reps: int = 5
) -> dict[str, float]:
    raise NotImplementedError  # Task 2


def _pair_vectors(base_map, head_map):
    raise NotImplementedError  # Task 2


# ---------------------------------------------------------------------------
# run_experiment (Task 3 implements the gate body)
# ---------------------------------------------------------------------------


async def run_experiment(
    proposal_id,
    *,
    trials: int = 1,
    keep: bool = False,
    store=None,
    run_slice=None,
    repo_root=None,
    cfg=None,
    bus=None,
) -> int:
    """Run the two-stage promotion gate. ALWAYS returns an int exit code.

    Structural refusals are caught here and returned as their exit code (>=4) so the
    caller's contract is simply "an int". Task 3 wires the gate body.
    """
    if cfg is None:
        from localharness.cli.components_cmd import _build_loader
        cfg = _build_loader().load_harness()
    try:
        await _load_and_validate(store, proposal_id, cfg)
    except ExperimentRefusal as exc:
        return exc.exit_code
    raise NotImplementedError  # Task 3
