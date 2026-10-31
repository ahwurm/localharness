"""Real per-arm divergence on the MECHANISM axis.

Mirrors test_gate_arm_divergence.py::test_overlay_divergence_real_loop (the {5,9} window_size
proof) for the boolean mechanism leaf agent.baton_gate.enabled. The point: the two gate arms must
construct DIFFERENT REAL ``AgentLoop._config`` — they genuinely diverge ON THE MECHANISM
({True, False}), NOT model-vs-itself.

Two independent signals, both asserted:
  (a) CONFIG-LEVEL: the set of resolved ``AgentLoop._config.baton_gate.enabled`` across the two
      arms is exactly {True, False}. Driven through the REAL ``_build_default_run_slice`` /
      ``_run_slice`` per-arm seam (with_overlay False vs True), wrapping (not replacing) the real
      async ``_build_agent_loop``.
  (b) BEHAVIORAL (the 'not a no-op' proof): for a reply whose closing move announces further work,
      the gate-ON arm runs exactly one EXTRA loop iteration (the baton nudge round-trip) versus the
      OFF arm. The AgentLoop does not expose its per-run Session (run_turn builds it locally), so
      this drives ``_execute_loop`` with an externally-held Session and reads ``session.iteration``.

Offline, no live model. Train slice only — never drives a holdout slice.
"""
from __future__ import annotations

from types import SimpleNamespace

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
    """The two arms construct DIFFERENT REAL AgentLoop._config.baton_gate.enabled ({True, False})
    — they genuinely diverge ON THE MECHANISM, not model-vs-itself. Train slice only."""
    wt = _make_corpus_repo(tmp_path / "wt")
    # The mechanism mutation: gate ON (baseline default) vs OFF (proposal). annotation=bool.
    exp.write_experiment_overlay(wt, "agent.baton_gate.enabled", False, annotation=bool)

    # WRAP (not replace) the real builder: construct the genuine AgentLoop, then record it.
    real_build = bench_runner._build_agent_loop
    built = []

    async def _spy_build(*a, **kw):
        loop = await real_build(*a, **kw)  # the REAL AgentLoop is constructed
        built.append(loop)
        return loop

    monkeypatch.setattr(bench_runner, "_build_agent_loop", _spy_build)

    # tool_plan=[] -> the fake emits a final answer immediately; the loop natural-completes offline.
    factory = lambda _scen: faithful_fake_llm(tool_plan=[])  # noqa: E731
    run_slice = exp._build_default_run_slice(
        "test-model",
        factory,
        annotation=bool,
        component="agent.baton_gate.enabled",
        after=False,
    )
    await run_slice(wt, slice="train", with_overlay=False)  # baseline arm: gate ON
    await run_slice(wt, slice="train", with_overlay=True)  # proposal arm: gate OFF

    # (a) CONFIG-LEVEL divergence:
    assert len(built) >= 2, "both arms must construct a real AgentLoop"
    cfgs = [l._config for l in built]
    flags = {get_value(c, "baton_gate.enabled") for c in cfgs}
    assert flags == {True, False}, (
        f"arms did not diverge on the mechanism — got baton_gate.enabled {flags}; the proposal arm "
        f"(with_overlay=True) must resolve False, the baseline (with_overlay=False) must resolve True"
    )
    assert all(hasattr(l, "run_turn") for l in built), (
        "constructed loops lack run_turn — these must be REAL AgentLoops, not a capture-stub placeholder"
    )


class _AnnouncingLLM:
    """Tool-less replies in order: the first closes by announcing further work, the rest are final."""

    def __init__(self):
        self.calls = 0
        self.config = SimpleNamespace(tool_call_mode="native", context_window=128_000)

    async def stream_complete(self, messages=None, tools=None, on_token=None):
        self.calls += 1
        text = "Now let me read the notes." if self.calls == 1 else "The notes say 4."
        return SimpleNamespace(content=text, tool_calls=None), None


def _make_loop(llm, bus, *, enabled: bool) -> AgentLoop:
    """A real AgentLoop with offline deps."""
    cfg = AgentConfig.model_validate(
        {"name": "mech-arm", "role": "Test agent.", "baton_gate": {"enabled": enabled}}
    )
    return AgentLoop(
        config=cfg,
        llm=llm,
        bus=bus,
        context_manager=ContextManager(),
        tool_registry=None,
        permission_evaluator=PermissionEvaluator(),
    )


async def test_mechanism_arm_divergence_behavioral_signal(bus):
    """The mechanism changes MEASURED behavior — the gate-ON arm runs exactly one EXTRA loop
    iteration (the baton nudge round-trip) versus the OFF arm for the same scripted replies. The
    same offline LLM produces a different iteration count SOLELY because the mechanism axis flips."""
    off_loop = _make_loop(_AnnouncingLLM(), bus, enabled=False)
    on_loop = _make_loop(_AnnouncingLLM(), bus, enabled=True)

    off_session = Session(agent_id="mech-arm", session_id="arm-off", messages=[])
    on_session = Session(agent_id="mech-arm", session_id="arm-on", messages=[])

    await off_loop._execute_loop(off_session, "do the task", None)
    await on_loop._execute_loop(on_session, "do the task", None)

    off_iter = off_session.iteration
    on_iter = on_session.iteration
    assert on_iter == off_iter + 1, (
        f"the mechanism did not change measured behavior — OFF={off_iter}, ON={on_iter}; the "
        f"gate-ON arm must run exactly one extra loop iteration (the baton nudge)"
    )
    # Both arms still finalize cleanly (the ON arm is bounded, never a hang/budget termination).
    assert off_session.terminated_reason == "complete"
    assert on_session.terminated_reason == "complete"
