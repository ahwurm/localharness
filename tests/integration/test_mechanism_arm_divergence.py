"""Phase 25 (MECH-02) — real per-arm divergence on the MECHANISM axis.

Mirrors test_gate_arm_divergence.py::test_overlay_divergence_real_loop (the Phase-20 {5,9}
window_size proof) for the boolean mechanism leaf agent.self_check.enabled, opened first-class in
25-01. The point: the two gate arms must construct DIFFERENT REAL ``AgentLoop._config`` — they
genuinely diverge ON THE MECHANISM ({False, True}), NOT model-vs-itself.

Two independent signals, both asserted:
  (a) CONFIG-LEVEL (mandatory, the Phase-20 {5,9} analogue): the set of resolved
      ``AgentLoop._config.self_check.enabled`` across the two arms is exactly {False, True}. Driven
      through the REAL ``_build_default_run_slice`` / ``_run_slice`` per-arm seam (with_overlay
      False vs True), wrapping (not replacing) the real async ``_build_agent_loop``.
  (b) BEHAVIORAL (the 'not a no-op' proof): the self_check=True arm runs exactly one EXTRA loop
      iteration (the bounded review round-trip) versus the OFF arm for the same scenario. The
      AgentLoop does not expose its per-run Session (run_turn builds it locally, experiment.py:331
      analogue at loop.py:331), so — per the plan's sanctioned fallback — this is asserted at
      integration scale by driving ``_execute_loop`` with an externally-held Session and reading
      ``session.iteration`` (exactly how 25-01's test_agent_loop_selfcheck reads the count).

Offline: FaithfulFakeLLM(tool_plan=[]) emits a final answer immediately, so every arm hits the
``if not tool_calls:`` natural-completion seam where the self-check block lives. No live model.
SC: train slice only (mirrors test_gate_arm_divergence SC9) — never drives a holdout slice.
"""
from __future__ import annotations

import pytest  # noqa: F401 — asyncio_mode=auto; imported for explicit test-runner discovery

from localharness.agent.context import ContextManager
from localharness.agent.loop import AgentLoop, Session
from localharness.agent.permissions import PermissionEvaluator
from localharness.autoresearch import experiment as exp
from localharness.bench import runner as bench_runner
from localharness.config.models import AgentConfig
from localharness.registry.paths import get_value

# Reuse the canonical corpus builder verbatim — a worktree-style git repo with ONE valid committed
# train fixture (hyphenated train-01; toolless; golden_output "4"). Self-contained, importable
# because the repo root is on sys.path (testpaths=["tests"]).
from tests.integration.test_gate_arm_divergence import _make_corpus_repo


async def test_mechanism_arm_divergence_real_loop(tmp_path, monkeypatch, faithful_fake_llm):
    """MECH-02 (a): the two arms construct DIFFERENT REAL AgentLoop._config.self_check.enabled
    ({False, True}) — they genuinely diverge ON THE MECHANISM, not model-vs-itself. Mirrors
    Phase-20's {5,9} window_size proof for the new boolean mechanism leaf. SC: train slice only."""
    wt = _make_corpus_repo(tmp_path / "wt")
    # The mechanism mutation: self_check OFF (baseline) vs ON (proposal). annotation=bool.
    exp.write_experiment_overlay(wt, "agent.self_check.enabled", True, annotation=bool)

    # WRAP (not replace) the real builder: construct the genuine AgentLoop, then record it.
    real_build = bench_runner._build_agent_loop
    built = []

    async def _spy_build(*a, **kw):
        loop = await real_build(*a, **kw)  # the REAL AgentLoop is constructed (builder is async — Phase 24)
        built.append(loop)
        return loop

    monkeypatch.setattr(bench_runner, "_build_agent_loop", _spy_build)

    # tool_plan=[] -> the fake emits a final answer immediately; the loop natural-completes offline.
    factory = lambda _scen: faithful_fake_llm(tool_plan=[])  # noqa: E731
    run_slice = exp._build_default_run_slice(
        "test-model",
        factory,
        annotation=bool,
        component="agent.self_check.enabled",
        after=True,
    )
    await run_slice(wt, slice="train", with_overlay=False)  # baseline arm: self_check OFF
    await run_slice(wt, slice="train", with_overlay=True)  # proposal arm: self_check ON

    # (a) CONFIG-LEVEL divergence (mandatory, the Phase-20 {5,9} analogue):
    assert len(built) >= 2, "both arms must construct a real AgentLoop"
    cfgs = [l._config for l in built]
    flags = {get_value(c, "self_check.enabled") for c in cfgs}
    assert flags == {False, True}, (
        f"arms did not diverge on the mechanism — got self_check.enabled {flags}; the proposal arm "
        f"(with_overlay=True) must resolve True, the baseline (with_overlay=False) must resolve False"
    )
    assert all(hasattr(l, "run_turn") for l in built), (
        "constructed loops lack run_turn — these must be REAL AgentLoops, not a capture-stub placeholder"
    )


def _make_loop(llm, bus, *, enabled: bool) -> AgentLoop:
    """A real AgentLoop with offline deps (mirrors test_agent_loop_selfcheck._make_loop)."""
    cfg = AgentConfig.model_validate(
        {"name": "mech-arm", "role": "Test agent.", "self_check": {"enabled": enabled, "max_passes": 1}}
    )
    return AgentLoop(
        config=cfg,
        llm=llm,
        bus=bus,
        context_manager=ContextManager(),
        tool_registry=None,
        permission_evaluator=PermissionEvaluator(),
    )


async def test_mechanism_arm_divergence_behavioral_signal(faithful_fake_llm, bus):
    """MECH-02 (b): the mechanism changes MEASURED behavior — the self_check=True arm runs exactly
    one EXTRA loop iteration (the review round-trip) versus the OFF arm for the same scenario. This
    is the 'not model-vs-itself' proof: the same offline LLM produces a different iteration count
    SOLELY because the mechanism axis flips. The AgentLoop does not expose its per-run Session, so
    (per the plan's sanctioned fallback) this drives _execute_loop with an externally-held Session
    and reads session.iteration — the same accessor 25-01's loop unit test uses."""
    off_loop = _make_loop(faithful_fake_llm(tool_plan=[]), bus, enabled=False)
    on_loop = _make_loop(faithful_fake_llm(tool_plan=[]), bus, enabled=True)

    off_session = Session(agent_id="mech-arm", session_id="arm-off", messages=[])
    on_session = Session(agent_id="mech-arm", session_id="arm-on", messages=[])

    await off_loop._execute_loop(off_session, "do the task", None)
    await on_loop._execute_loop(on_session, "do the task", None)

    off_iter = off_session.iteration
    on_iter = on_session.iteration
    assert on_iter == off_iter + 1, (
        f"the mechanism did not change measured behavior — OFF={off_iter}, ON={on_iter}; the "
        f"self_check=True arm must run exactly one extra loop iteration (the review pass)"
    )
    # Both arms still finalize cleanly (the ON arm is bounded, never a hang/budget termination).
    assert off_session.terminated_reason == "complete"
    assert on_session.terminated_reason == "complete"
