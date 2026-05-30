"""Phase 21 AUDIT-04/06 — config → runtime application gaps at the promotion gate.

This file characterizes three intertwined findings (tests-only; zero ``src/`` change):

* **AUDIT-04a (identical arms):** ``_resolve_worktree_agent_cfg`` (experiment.py:205-227)
  extracts ONLY ``merged.get("agent", {})`` (:218). A ``provider.*`` / ``org.*`` /
  ``compaction.*`` mutation lives OUTSIDE the ``agent`` subtree, so it is dropped and BOTH
  gate arms resolve the IDENTICAL ``AgentConfig`` — the Welch/Bonferroni gate then compares a
  config against itself (model-vs-itself noise). Contrast with the agent.* mutation in
  test_experiment_overlay_e2e.py::test_overlay_diverges_arms (arms DO diverge ``{5, 9}``):
  the asymmetry IS the finding.

* **AUDIT-04b (provenance drift):** ``build_catalogue`` carries two provenance axes —
  ``agent_cfg=`` (current_value for agent.* paths) and ``overlays=`` (winning_layer). Phase 20
  ALREADY fixed the gate/proposer current_value axis (``agent_cfg=_provenance_agent_cfg()`` at
  experiment.py:192/402 + proposer.py:210) — we GREEN-PIN those so a refactor cannot silently
  regress them (asserting they use model_construct defaults would be the inverse green-check
  trap). The STILL-bare sites (adoption.py:138, loop.py:231/257, propose_cmd.py:137) call
  ``build_catalogue(cfg)`` with no ``agent_cfg`` and fall back to the model_construct default —
  RED-characterized here. And NO non-CLI site passes ``overlays=``, so ``winning_layer`` stays
  ``"default"`` everywhere off the CLI (the attribution half of WARNING-2 that still holds).

* **AUDIT-06 (real-exec divergence):** the existing overlay-divergence proof
  (test_experiment_overlay_e2e.py:117-118) is a CONFIG-CAPTURE stub — it monkeypatches
  ``_build_agent_loop`` to a ``_StubLoop`` (no ``run_turn``) and ``_run_loop`` to a no-op, so a
  real loop is NEVER constructed. We re-verify by WRAPPING (not replacing) the real builder and
  running one real turn per arm via the faithful-fake: the two arms construct DIFFERENT REAL
  ``AgentLoop._config`` (``{5, 9}``) — strictly stronger than a captured config object.

SC9: this file never drives a holdout slice (TRAIN path only).
"""
from __future__ import annotations

import inspect
import subprocess

import yaml  # noqa: F401 — parity with the experiment overlay (de)serializer surface

import pytest  # noqa: F401 — asyncio_mode=auto; imported for explicit test-runner discovery

from localharness.autoresearch import experiment as exp
from localharness.bench import runner as bench_runner
from localharness.config.models import AgentConfig
from localharness.config.overlay import atomic_write_overlay, load_overlay
from localharness.registry import build_catalogue
from localharness.registry.paths import get_value


# --------------------------------------------------------------------------- #
# Shared corpus builder (mirrors test_experiment_overlay_e2e._make_corpus_repo).
# A hyphenated scenario name (train-01): the bench synthesizes
# AgentConfig(name=f"bench-{name}") and AgentConfig.name forbids underscores.
# --------------------------------------------------------------------------- #


def _scenario_body(name: str, slice_: str) -> str:
    """A complete, VALID ScenarioSpec YAML (mirrors conftest._scenario_yaml).

    The minimal name:/slice: stubs in test_experiment_e2e.py are NOT valid and get dropped
    by _load_scenarios_from_paths — the corpus MUST write the full body or the slice resolves
    empty and _build_agent_loop is never reached.
    """
    return (
        f"name: {name}\n"
        'prompt: "What is 2 + 2? Give just the number."\n'
        'expected_outcome: "Returns 4."\n'
        "success_criteria:\n"
        '  golden_output: "4"\n'
        "budget:\n"
        "  max_actions: 5\n"
        "  max_duration_minutes: 1.0\n"
        "  max_context_tokens: 32000\n"
        "limits:\n"
        "  max_latency_s: 30.0\n"
        "  max_tool_calls: 0\n"
        "tools_allowed: []\n"
        f"slice: {slice_}\n"
        "category: tool_basics\n"
    )


def _make_corpus_repo(root):
    """git init a worktree-style repo with ONE valid committed train fixture."""
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    (root / "bench").mkdir()
    (root / "bench" / "bench.yaml").write_text(
        "corpus_path: bench/scenarios\nresults_path: bench/results\n", encoding="utf-8"
    )
    train = root / "bench" / "scenarios" / "train"
    train.mkdir(parents=True)
    (train / "train-01.yaml").write_text(_scenario_body("train-01", "train"), encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init corpus"], cwd=root, check=True)
    return root


# --------------------------------------------------------------------------- #
# Task 1 — AUDIT-04a: a NON-agent.* mutation yields IDENTICAL arms.
# --------------------------------------------------------------------------- #


async def test_non_agent_mutation_yields_identical_arms(tmp_path, monkeypatch):
    """The two gate arms resolve the IDENTICAL AgentConfig for a non-agent.* mutation.

    Drives the REAL _build_default_run_slice / _run_slice (NO injected fake slice runner,
    NO monkeypatched run_experiment). The mutation targets org.context.compaction_threshold_pct
    — a VALID registry path OUTSIDE the agent subtree — so _resolve_worktree_agent_cfg (which
    reads only merged.get("agent", {}), experiment.py:218) drops it and both arms build the same
    config. The gate therefore runs model-vs-itself for this whole class of proposal.
    """
    wt = _make_corpus_repo(tmp_path / "wt")

    # Materialize a NON-agent.* mutation into the worktree (the gate's experiment.py:355 call).
    # org.context.compaction_threshold_pct (ge=50.0) lives OUTSIDE the agent.* subtree — the
    # point: a path the agent-only resolver cannot see.
    exp.write_experiment_overlay(
        wt, "org.context.compaction_threshold_pct", 80.0, annotation=float
    )

    # Capture the resolved AgentConfig that reaches the loop, per arm, WITHOUT running a turn.
    # This is a config-capture (the AgentConfig IS the artifact under audit here — NOT the
    # AUDIT-01 spine). Mirror the live builder's fallback so the captured value is the real
    # per-arm config in both the threaded (post-20-02) and the no-thread era.
    captured: dict = {}

    def _capture(bus, llm_client, scenario, session_id="", agent_config=None, base_registry=None):
        resolved = agent_config if agent_config is not None else AgentConfig(
            name=f"bench-{scenario.name}",
            role=f"Bench harness execution for scenario {scenario.name}",
        )
        captured.setdefault("cfgs", []).append(resolved)

        class _StubLoop:  # no run_turn — _run_loop is patched to a no-op below
            pass

        return _StubLoop()

    async def _noop_run_loop(loop, prompt, on_token):
        return None

    monkeypatch.setattr(bench_runner, "_build_agent_loop", _capture)
    monkeypatch.setattr(bench_runner, "_run_loop", _noop_run_loop)

    # NOTE: NO _filter_scenarios_by_slice monkeypatch here. Plan 20-02 wired
    # _load_scenarios_from_paths into _run_slice (experiment.py:327-328) BEFORE the slice filter,
    # so the 20-01 load-tolerant-filter monkeypatch is OBSOLETE — _run_slice loads Paths itself.

    # Stay offline: a canned client factory (the captured cfg is what matters; _run_loop is a no-op).
    factory = lambda _scen: object()  # noqa: E731

    # Drive the REAL slice runner for BOTH arms.
    run_slice = exp._build_default_run_slice(
        "test-model",
        factory,
        annotation=float,
        component="org.context.compaction_threshold_pct",
        after=80.0,
    )
    await run_slice(wt, slice="train", with_overlay=False)  # baseline arm
    await run_slice(wt, slice="train", with_overlay=True)  # proposal arm

    cfgs = [c for c in captured.get("cfgs", []) if c is not None]
    assert len(cfgs) >= 2, (
        "fewer than two AgentConfigs captured — _build_agent_loop was not reached for both arms "
        "(corpus invalid?)"
    )
    base_cfg, head_cfg = cfgs[0], cfgs[1]

    # THE characterization (pins the AUDIT-04 correctness hole): the non-agent.* mutation is
    # dropped by _resolve_worktree_agent_cfg (it reads only the agent subtree), so BOTH arms
    # resolve the IDENTICAL AgentConfig -> the gate runs model-vs-itself for this proposal.
    # This PASSES today; a future fix that widens the resolver to merge the FULL config (so a
    # provider/org/compaction mutation actually changes the runtime) would FLIP this assertion.
    # CONTRAST: test_experiment_overlay_e2e::test_overlay_diverges_arms uses an agent.* mutation
    # (agent.stuck_detector.window_size) -> arms DO diverge {5, 9}. The asymmetry IS the finding.
    assert base_cfg.model_dump() == head_cfg.model_dump(), (
        "arms differ — a future fix that widens _resolve_worktree_agent_cfg to the full config "
        "would flip this characterization (a non-agent.* mutation would then change the runtime)"
    )


# --------------------------------------------------------------------------- #
# Task 2 — AUDIT-04b provenance: GREEN-pin the fixed sites, RED-characterize the
# bare sites, characterize winning_layer="default".
#
# _provenance_agent_cfg() reads the USER overlay (_resolve_user_overlay_path() ->
# LOCALHARNESS_HOME/overrides.yaml, set by the components_home fixture), NOT the worktree.
# So a distinct adopted agent.role is staged by writing that user overlay.
# --------------------------------------------------------------------------- #


_ADOPTED_ROLE = "ADOPTED ROLE — distinct from the AgentConfig default"


def _cfg():
    """A minimal valid HarnessConfig (mirrors test_proposer._cfg)."""
    from localharness.config.models import HarnessConfig

    return HarnessConfig.model_validate(
        {
            "version": "1",
            "provider": {
                "provider_type": "ollama",
                "base_url": "http://localhost:11434/v1",
                "default_model": "test-model",
            },
        }
    )


def _stage_live_agent_role(home, role: str = _ADOPTED_ROLE) -> None:
    """Write a LIVE adopted agent.role into the user overlay (LOCALHARNESS_HOME/overrides.yaml).

    This is the layer _provenance_agent_cfg() reads (config.overlay._resolve_user_overlay_path).
    """
    overlay_path = home / "overrides.yaml"
    existing = load_overlay(overlay_path)
    existing.setdefault("agent", {})["role"] = role
    atomic_write_overlay(overlay_path, existing)


def test_provenance_fixed_sites_reflect_live_overlay(components_home):
    """GREEN-PIN the Phase-20-FIXED sites (experiment.py:192/402, proposer.py:210).

    With a live agent.role adopted in the user overlay, build_catalogue(cfg,
    agent_cfg=_provenance_agent_cfg()) — exactly the call those three sites make — reflects the
    ADOPTED role, NOT the AgentConfig.model_construct default. This PINS the Phase-20 fix so a
    refactor cannot silently regress it.

    DO NOT assert experiment.py:192/402 or proposer.py:210 use model_construct defaults — per the
    RESEARCH "DRIFTED" section those are ALREADY FIXED, so such a RED would FAIL WRONGLY (the
    inverse green-check trap). The Phase-20 test_before_is_current_component_value covers the
    proposer.before via the same catalogue; this is the explicit AUDIT-04 anchor on the gate sites.
    """
    _stage_live_agent_role(components_home)
    cfg = _cfg()

    from localharness.autoresearch.experiment import _provenance_agent_cfg

    prov = _provenance_agent_cfg()
    assert prov is not None, "user overlay seeded agent.role but _provenance_agent_cfg returned None"

    fixed = build_catalogue(cfg, agent_cfg=prov).get("agent.role")
    assert fixed is not None
    # The fixed sites reflect the LIVE adopted role (Phase-20 fix), not the model_construct default.
    assert fixed.current_value == _ADOPTED_ROLE, (
        "fixed-site provenance must reflect the live adopted agent.role — experiment.py:192/402 + "
        "proposer.py:210 pass agent_cfg=_provenance_agent_cfg() (Phase 20); a regression here means "
        "one of those dropped agent_cfg= back to the bare call"
    )
    assert fixed.current_value != "<default>", (
        "fixed-site current_value collapsed to the AgentConfig.model_construct default — the "
        "Phase-20 agent_cfg= wiring regressed"
    )


def test_provenance_bare_sites_use_model_construct_default(components_home):
    """RED-characterize the STILL-bare sites (adoption.py:138, loop.py:231/257, propose_cmd.py:137).

    Those four sites call build_catalogue(cfg) with NO agent_cfg, so agent.* provenance falls back
    to AgentConfig.model_construct(name="<default>", role="<default>") (catalogue.py:149-153) — the
    live adopted role is NOT reflected (the residual WARNING-2 gap at these sites). This PASSES
    today (the bare call DOES return the default), documenting the gap; a fix that adds agent_cfg=
    at those sites would flip it.
    """
    _stage_live_agent_role(components_home)
    cfg = _cfg()

    # EXACTLY as adoption.py:138 / loop.py:231,257 / propose_cmd.py:137 call it (no agent_cfg).
    bare = build_catalogue(cfg).get("agent.role")
    assert bare is not None
    # The bare call returns the model_construct default, NOT the live adopted role.
    assert bare.current_value == "<default>", (
        "bare build_catalogue(cfg) unexpectedly reflected the live overlay — if a fix added "
        "agent_cfg= at the bare sites, retarget this characterization"
    )
    assert bare.current_value != _ADOPTED_ROLE, (
        "bare build_catalogue must NOT reflect the live overlay (documents the WARNING-2 gap at "
        "adoption.py:138 / loop.py:231,257 / propose_cmd.py:137)"
    )

    # Source-level regression guard (14-03 precedent): a flat `build_catalogue(cfg)` call (no
    # agent_cfg) is still present at the bare sites. Flips the day a fix threads agent_cfg= in.
    from localharness.autoresearch import adoption
    from localharness.autoresearch import loop as aloop
    from localharness.cli import propose_cmd

    def _has_bare_call(mod) -> bool:
        src = inspect.getsource(mod)
        return "build_catalogue(cfg)" in src

    assert _has_bare_call(adoption), "adoption.py no longer has a bare build_catalogue(cfg) — retarget"
    assert _has_bare_call(aloop), "loop.py no longer has a bare build_catalogue(cfg) — retarget"
    assert _has_bare_call(propose_cmd), "propose_cmd.py no longer has a bare build_catalogue(cfg) — retarget"


def test_winning_layer_default_at_gate_sites(components_home):
    """Characterize winning_layer == 'default' at every non-CLI build_catalogue call.

    winning_layer is driven by overlays= (catalogue._detect_layer). Only components_cmd.py
    (138/192/260, the CLI list/get/set) passes overlays=; the gate / proposer / adoption sites
    never do. So even with a live agent.role overlay active, build_catalogue WITHOUT overlays=
    reports winning_layer == 'default' (the attribution half of WARNING-2 that still holds).
    """
    _stage_live_agent_role(components_home)
    cfg = _cfg()

    # No overlays= (as every non-CLI site calls it) -> winning_layer is 'default' regardless of
    # an active overlay, because layer attribution is never fed the overlay dicts.
    entry = build_catalogue(cfg).get("agent.role")
    assert entry is not None
    assert entry.winning_layer == "default", (
        "winning_layer is non-'default' without overlays= — a non-CLI site started passing "
        "overlays= (only components_cmd.py:138/192/260 should)"
    )


# --------------------------------------------------------------------------- #
# Task 3 — AUDIT-06: real-execution overlay divergence (the arms build DIFFERENT
# REAL AgentLoop._config when the loop ACTUALLY constructs + runs).
# --------------------------------------------------------------------------- #


async def test_overlay_divergence_real_loop(tmp_path, monkeypatch, faithful_fake_llm):
    """The two arms construct DIFFERENT REAL AgentLoop._config — proven by real construction.

    Re-verifies the overlay-divergence claim the config-capture stub (test_experiment_overlay_e2e.py
    :117-118) leaves open: that stub returns a bare loop placeholder with NO run_turn (a real loop
    NEVER exists), so it only proves a captured AgentConfig differs. Here we WRAP (not replace) the
    real _build_agent_loop — calling through to the genuine constructor and recording loop._config —
    and
    run ONE real turn per arm via the faithful-fake (empty tool_plan -> the fake immediately produces
    a final answer; the loop natural-completes offline, no real model). Strictly stronger than the
    capture-stub.
    """
    wt = _make_corpus_repo(tmp_path / "wt")
    # An agent.* mutation that changes observable construction (the proven 5->9 window_size).
    exp.write_experiment_overlay(wt, "agent.stuck_detector.window_size", 9, annotation=int)

    # WRAP the real builder: construct the genuine AgentLoop, then record it. No loop placeholder,
    # no run-loop no-op — the loop genuinely runs.
    real_build = bench_runner._build_agent_loop
    built = []

    def _spy_build(*a, **kw):
        loop = real_build(*a, **kw)  # the REAL AgentLoop is constructed
        built.append(loop)
        return loop

    monkeypatch.setattr(bench_runner, "_build_agent_loop", _spy_build)

    # The faithful-fake with an empty plan: the fake immediately emits a final answer, so one
    # real turn per arm runs to natural completion without a real model.
    factory = lambda _scen: faithful_fake_llm(tool_plan=[])  # noqa: E731

    run_slice = exp._build_default_run_slice(
        "test-model",
        factory,
        annotation=int,
        component="agent.stuck_detector.window_size",
        after=9,
    )
    await run_slice(wt, slice="train", with_overlay=False)  # baseline arm
    await run_slice(wt, slice="train", with_overlay=True)  # proposal arm

    # THE AUDIT-06 closure: the REAL loops' configs diverge (baseline 5 vs proposal 9), and they
    # are genuine AgentLoops (have run_turn) — NOT the capture-stub's bare loop placeholder.
    assert len(built) >= 2, "fewer than two real AgentLoops constructed — both arms must build one"
    cfgs = [l._config for l in built]
    windows = {get_value(c, "stuck_detector.window_size") for c in cfgs}
    # Proven by real construction, NOT a captured config object (the Phase-20 stub never built a loop).
    assert windows == {5, 9}, (
        f"real arms did not diverge — got window sizes {windows}; the proposal arm "
        f"(with_overlay=True) must construct a loop with window_size=9, the baseline (False) with 5"
    )
    assert all(hasattr(l, "run_turn") for l in built), (
        "constructed loops lack run_turn — these must be REAL AgentLoops, not the capture-stub's "
        "bare loop placeholder (test_experiment_overlay_e2e.py:117-118)"
    )
