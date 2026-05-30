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
from localharness.config.models import AgentConfig
from localharness.config.overlay import atomic_write_overlay, deep_merge, load_overlay
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
_OFFREGISTRY_PREFIXES = ("bench.", "scenario", "grader", "success_criteria", "holdout", "sentinel")

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
    catalogue = build_catalogue(cfg, agent_cfg=_provenance_agent_cfg())
    if catalogue.get(component) is None:
        raise ExperimentRefusal(EXIT_REFUSE_OFFREGISTRY,
                                f"component not in the mutable registry: {component!r}")

    return entry, component, after


# ---------------------------------------------------------------------------
# Worktree config-cascade → AgentConfig resolver (BLOCKER-1 fix)
# ---------------------------------------------------------------------------


def _resolve_worktree_agent_cfg(root, scenario, *, include_experiment_overlay):
    """Build the per-scenario AgentConfig from the worktree config cascade.

    Cascade: base ({}) -> <root>/.localharness/overrides.yaml (adopted mutations, BOTH arms)
             -> <root>/.localharness/experiment-overlay.yaml (the candidate, PROPOSAL arm ONLY).
    Mirrors adoption.py:83-103 (the agent.* -> AgentConfig validate precedent) but RETURNS the
    built runtime config. The `agent` subtree is the registry addressing namespace; scenario
    identity (name) is synthesized, role/everything-else comes from the overlay when present.
    """
    root = Path(root)
    merged = deep_merge({}, load_overlay(root / ".localharness" / "overrides.yaml"))
    if include_experiment_overlay:
        merged = deep_merge(merged, load_overlay(root / ".localharness" / "experiment-overlay.yaml"))
    agent_overlay = merged.get("agent", {})
    identity = {
        "name": f"bench-{scenario.name}",
        "role": f"Bench harness execution for scenario {scenario.name}",
        # Thread scenario budget so the cascade AgentConfig enforces the scenario cap.
        # The overlay wins (deep_merge below), so an explicit agent.permissions overlay
        # can still override the scenario default.
        "permissions": {"budget": {"max_actions": scenario.budget.max_actions}},
    }
    # Scenario identity is the base; the overlay's agent subtree wins (so an agent.role
    # mutation is observable). name is always synthesized last to satisfy the validator.
    built = deep_merge(identity, agent_overlay)
    built["name"] = f"bench-{scenario.name}"
    return AgentConfig.model_validate(built)


def _provenance_agent_cfg():
    """The live ADOPTED agent.* config (overrides.yaml, no experiment overlay) for catalogue
    `before` provenance. Returns None when nothing is adopted — behavior-identical to the old
    agent_cfg=None (build_catalogue's own model_construct default); only becomes a truthful
    live config once an agent.* mutation IS adopted (WARNING-2).
    """
    from localharness.config.overlay import _resolve_user_overlay_path
    agent_overlay = load_overlay(_resolve_user_overlay_path()).get("agent", {})
    if not agent_overlay:
        return None
    return AgentConfig.model_validate(deep_merge({"name": "provenance", "role": "provenance"}, agent_overlay))


# ---------------------------------------------------------------------------
# Per-fixture success-rate vector extraction (Task 2 implements the body)
# ---------------------------------------------------------------------------


async def slice_success_by_fixture(
    scenarios, model, results_root, factory, *, min_reps: int = 5, agent_config=None
) -> dict[str, float]:
    """Run each scenario (>=min_reps reps via accumulate_runs) → {scenario_name: success_rate}.

    The Welch gate's n is the FIXTURE count (len of this map), NOT the rep count (Pitfall 2).
    Per scenario, the success_rate is metrics_summary(samples)["success_rate"]["rate"] — a
    proportion over that scenario's reps. CONTEXT locks >=5 reps/fixture (ADAS precedent).
    """
    from localharness.bench.aggregator import metrics_summary
    from localharness.bench.runner import accumulate_runs

    out: dict[str, float] = {}
    for scen in scenarios:
        samples, _stop = await accumulate_runs(
            scen, model, results_root, factory,
            min_runs_override=max(min_reps, scen.min_runs),
            max_runs_override=scen.max_runs,
            agent_config=agent_config,
        )
        if not samples:
            continue
        out[scen.name] = metrics_summary(samples)["success_rate"]["rate"]
    return out


def _pair_vectors(base_map, head_map):
    """Pair two arms fixture-for-fixture by scenario name (intersection, sorted).

    The Welch vectors MUST be aligned by fixture; only names present in BOTH arms enter
    the comparison. Returns (names, base_vec, head_vec) with matching order.
    """
    names = sorted(set(base_map) & set(head_map))
    return names, [base_map[n] for n in names], [head_map[n] for n in names]


# ---------------------------------------------------------------------------
# run_experiment helpers
# ---------------------------------------------------------------------------


async def _maybe_await(value):
    """Support both an async real run_slice and a sync injected fake (tests use a plain dict)."""
    if hasattr(value, "__await__"):
        return await value
    return value


def _estimate_cost(*maps: dict[str, float]) -> float:
    """Cost proxy (Open Q #5): recorded, NOT gated (efficiency-as-objective deferred).

    Summed fixture-runs across both arms of both stages — a simple, monotone, defensible
    number filled into the archive. The injected fakes don't surface tokens, so this is a
    count, not a token sum; the tests assert cost is a number, not its exact value.
    """
    return float(sum(len(m) for m in maps if m))


def _build_default_run_slice(model, factory, *, cfg=None, annotation=None, component=None, after=None):
    """The real bench-backed slice runner used when the caller injects no run_slice.

    Closure signature matches the injected fake: ``run_slice(worktree, *, slice, with_overlay)``.
    The proposal arm (with_overlay=True) runs with the experiment overlay materialized in the
    worktree; the baseline arm (with_overlay=False) WITHOUT it. CONTEXT: the baseline is a FRESH
    re-run, never cached — each arm re-discovers + re-runs the worktree corpus from scratch.

    cfg: HarnessConfig threaded from run_experiment so the gate's bench client resolves
    provider/model/base_url from cfg.provider + a detect_capabilities probe (FIDEL-01).
    """
    from localharness.bench.orchestrator import (
        _build_bench_client,
        _discover_scenarios,
        _filter_scenarios_by_slice,
        _load_scenarios_from_paths,
        _synthesize_default_entry,
        build_llm_client_factory,
    )

    async def _run_slice(worktree, *, slice, with_overlay):
        corpus = Path(worktree) / "bench" / "scenarios"
        results_root = Path(worktree) / "bench" / "results"
        # _discover_scenarios returns list[Path]; load them into ScenarioSpec objects
        # (mirrors bench.orchestrator) BEFORE filtering — the filter reads scen.slice.
        scenarios = _filter_scenarios_by_slice(
            _load_scenarios_from_paths(_discover_scenarios(corpus)), slice
        )
        if factory is not None:
            client_factory = factory
        elif cfg is not None:
            # FIDEL-01: resolve from cfg.provider + probe, mirroring the matrix path
            # (_run_one_model: orchestrator.py:143-148). Model-agnostic — no Qwen/Ollama branch.
            from localharness.bench.config import MatrixEntry
            entry = MatrixEntry(
                name=cfg.provider.default_model,
                provider=cfg.provider.provider_type,
                model_id=cfg.provider.default_model,
                base_url=cfg.provider.base_url,
            )
            probed_client = _build_bench_client(entry)
            await probed_client.detect_capabilities()
            client_factory = lambda _scen: probed_client
        else:
            client_factory = build_llm_client_factory(_synthesize_default_entry())
        # The bench resolves each arm's AgentConfig from the worktree cascade. with_overlay
        # (NOT the filesystem) decides whether the candidate experiment-overlay layer is included
        # — both arms share the same worktree where the overlay is materialized (experiment.py:355).
        out: dict[str, float] = {}
        from localharness.bench.aggregator import metrics_summary
        from localharness.bench.runner import accumulate_runs
        for scen in scenarios:
            agent_cfg = _resolve_worktree_agent_cfg(worktree, scen, include_experiment_overlay=with_overlay)
            samples, _stop = await accumulate_runs(
                scen, model, results_root, client_factory,
                min_runs_override=max(5, scen.min_runs),
                max_runs_override=scen.max_runs,
                agent_config=agent_cfg,
            )
            if not samples:
                continue
            out[scen.name] = metrics_summary(samples)["success_rate"]["rate"]
        return out

    return _run_slice


def _resolve_store(store, cfg):
    """Open a default ArchiveStore at the LOCALHARNESS_HOME archive.db if none injected."""
    if store is not None:
        return store, False
    from localharness.autoresearch.archive import ArchiveStore
    from localharness.cli.autoresearch_cmd import _archive_db_path

    return ArchiveStore(_archive_db_path()), True


# ---------------------------------------------------------------------------
# run_experiment — the two-stage promotion gate (EXP-01..05)
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

    Structural refusals are caught here and returned as their exit code (>=4) so the caller's
    contract is simply "an int". Flow: load+validate → throwaway worktree → materialize the
    experiment overlay + emit the audit event → TRAIN Welch improvement (p<0.05) → conditional
    HOLDOUT Bonferroni-corrected non-regression (alpha=0.05/trials) → verdict write-back.
    Promote is archive-ONLY (no live-config write; baseline adoption is Phase 18).
    """
    if cfg is None:
        from localharness.cli.components_cmd import _build_loader
        cfg = _build_loader().load_harness()

    store, _opened = _resolve_store(store, cfg)
    if _opened:
        await store.open()
    try:
        # 1. Structural validation (refusals return >=4 BEFORE any worktree/bench).
        try:
            entry, component, after = await _load_and_validate(store, proposal_id, cfg)
        except ExperimentRefusal as exc:
            return exc.exit_code

        # Resolve the typed annotation for overlay coercion (off-registry already refused).
        cat_entry = build_catalogue(cfg, agent_cfg=_provenance_agent_cfg()).get(component)
        annotation = cat_entry.annotation if cat_entry is not None else None

        # 2. Default real run_slice if the caller injected none.
        if run_slice is None:
            model = cfg.provider.default_model
            run_slice = _build_default_run_slice(
                model, None, cfg=cfg, annotation=annotation, component=component, after=after
            )

        # 3. One throwaway worktree for the whole run.
        root = Path(repo_root) if repo_root is not None else _git_root()
        with experiment_worktree(root, keep=keep) as wt:
            # 3a. Materialize the mutation INTO the worktree (cannot leak to real ~/.localharness).
            write_experiment_overlay(wt, component, after, annotation=annotation)

            # 3b. Audit event for the proposal arm.
            await _emit_audit(bus, cfg, component, entry, after, proposal_id)

            # 3c. TRAIN stage — fresh baseline + proposal (CONTEXT: baseline is NOT cached).
            base_train = await _maybe_await(run_slice(wt, slice="train", with_overlay=False))
            head_train = await _maybe_await(run_slice(wt, slice="train", with_overlay=True))
            names, base_vec, head_vec = _pair_vectors(base_train, head_train)
            head_map = head_train  # TRAIN per-fixture map (TRAIN keys ONLY — sealed-slice).

            # Inconclusive guard (Pitfall 5): too few paired fixtures to support a Welch call.
            if len(names) < 2:
                await store.update_verdict(proposal_id, status="in_flight")
                return EXIT_INCONCLUSIVE

            _t, p, improved = welch_improvement(base_vec, head_vec, alpha=0.05)
            train_cost = _estimate_cost(base_train, head_train)

            if not improved:
                await store.update_verdict(
                    proposal_id, status="train_rejected",
                    train_score=statistics.mean(head_vec),
                    train_scores_per_fixture=head_map, p_value=p, cost=train_cost,
                )
                return EXIT_REJECT_TRAIN

            # 3d. HOLDOUT stage — reached ONLY on a train pass (conditional; never on reject).
            base_hold = await _maybe_await(run_slice(wt, slice="holdout", with_overlay=False))
            head_hold = await _maybe_await(run_slice(wt, slice="holdout", with_overlay=True))
            hnames, bh_vec, hh_vec = _pair_vectors(base_hold, head_hold)
            alpha_corr = 0.05 / trials  # Bonferroni multi-TRIAL (NOT multi-metric).
            regressed = welch_regression(bh_vec, hh_vec, alpha=alpha_corr)
            holdout_score = statistics.mean(hh_vec) if hh_vec else None
            total_cost = _estimate_cost(base_train, head_train, base_hold, head_hold)

            if regressed:
                await store.update_verdict(
                    proposal_id, status="holdout_rejected",
                    train_score=statistics.mean(head_vec),
                    train_scores_per_fixture=head_map,  # TRAIN-only; holdout never enters the blob
                    holdout_score=holdout_score, p_value=p, cost=total_cost,
                )
                return EXIT_REJECT_HOLDOUT

            # 3e. PROMOTE — archive-ONLY. NO atomic_write_overlay on the user overlay here.
            await store.update_verdict(
                proposal_id, status="promoted",
                train_score=statistics.mean(head_vec),
                train_scores_per_fixture=head_map,  # TRAIN scenario names only (sealed-slice)
                holdout_score=holdout_score, p_value=p, cost=total_cost,
            )
            return EXIT_PROMOTE
    finally:
        if _opened:
            await store.close()


async def _emit_audit(bus, cfg, component, entry, after, proposal_id) -> None:
    """Publish ComponentMutated(layer='experiment', actor='experiment', actor_detail=pid).

    Uses the injected bus if provided (tests subscribe to it); otherwise builds one pointed at
    cfg.org.audit_log_path (mirrors components_cmd's audit path).
    """
    from localharness.core.events import ComponentMutated

    target_bus = bus
    if target_bus is None:
        from localharness.core.bus import EventBus
        audit_path = getattr(getattr(cfg, "org", None), "audit_log_path", None)
        target_bus = EventBus(persist_path=Path(audit_path).expanduser() if audit_path else None)

    before = entry.diff_decoded.get("before")
    await target_bus.publish(
        ComponentMutated(
            path=component,
            before_value=before,
            after_value=after,
            layer="experiment",
            actor="experiment",
            actor_detail=proposal_id,
        )
    )
