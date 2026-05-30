"""AUDIT-02: scenario budgets/limits are loaded but never applied to the real agent loop.

`bench/runner.py::_build_agent_loop` builds a default `AgentConfig` when the bench threads
no `agent_config` (runner.py:273-279). That config carries the DEFAULT `BudgetConfig`
(`max_actions=100`, `max_duration_minutes=30.0`, models.py:132-144). The loop then reads its
budget from `self._config.permissions.budget.*` (loop.py:435-437). Only
`scenario.budget.max_context_tokens` is threaded into the runtime, via
`ContextManager(max_context_tokens=...)` (runner.py:281-282). `scenario.budget.max_actions`,
`scenario.budget.max_duration_minutes`, and `limits.max_tool_calls` NEVER reach the loop:

  * `scenario.budget.max_actions`  -> the loop's `BudgetTracker.max_actions` is the default 100
  * `scenario.budget.max_duration_minutes` -> the inner `BudgetTracker.max_duration_minutes` is 30.0
  * `limits.max_tool_calls`        -> ZERO readers anywhere in src (grep-confirmed)

Even post-Phase-20 the cascade resolver `_resolve_worktree_agent_cfg` merges only the `agent`
overlay subtree, never `scenario.budget`, so scenario budgets still never reach the loop.

These three characterizations drive the REAL `execute_one_run` (via `accumulate_runs`) — no
stubbed spine for the count tests — and stay CI-green: each RED test is
`xfail(strict=True, reason="AUDIT-02: ...")` so it counts as a documented gap (a green check ==
fails for EXACTLY the documented reason), and the two structural characterizations PIN today's
behavior so they FLIP RED the moment a future fix threads the budget through.
"""
from __future__ import annotations

import subprocess

import pytest

from localharness.bench.runner import accumulate_runs
from localharness.bench.schema import load_scenario


# ---------------------------------------------------------------------------
# Tunable-budget scenario builder.
#
# Mirrors conftest._tool_scenario_yaml (a complete, VALID ScenarioSpec with non-empty
# tools_allowed so the faithful-fake can emit REAL `read` dispatches), but the budget/limits
# knobs under audit are templated so each test can set them to the value being characterized.
# Round-tripped through load_scenario so a malformed scenario fails LOUDLY here, not silently
# downstream. SC9: slice is hard-pinned "train" — the holdout slice is never named or executed.
# ---------------------------------------------------------------------------
def _budget_scenario(
    tmp_path,
    *,
    name: str,
    max_actions: int = 100,
    max_duration_minutes: float = 30.0,
    max_tool_calls: int = 200,
):
    read_target = "/tmp/bench_fixtures/single_read_target.txt"  # staged by bench_fixtures_staged (content: apricot)
    body = (
        f"name: {name}\n"
        f'prompt: "Read the file at {read_target} and report its contents."\n'
        'expected_outcome: "Reports the real tool result."\n'
        "success_criteria:\n"
        '  rubric: ["contains:apricot"]\n'
        "budget:\n"
        f"  max_actions: {max_actions}\n"
        f"  max_duration_minutes: {max_duration_minutes}\n"
        "  max_context_tokens: 32000\n"
        "limits:\n"
        "  max_latency_s: 30.0\n"
        f"  max_tool_calls: {max_tool_calls}\n"
        "tools_allowed: [read]\n"
        "slice: train\n"
        "category: tool_basics\n"
    )
    path = tmp_path / f"{name}.yaml"
    path.write_text(body, encoding="utf-8")
    return load_scenario(path)


def _never_completes_read_plan(n: int):
    """A faithful-fake tool_plan that emits `n` benign `read` calls of the staged target with
    VARIED args (cycling `offset`) so each tool-call SIGNATURE is distinct — the loop's
    StuckDetector (window_size=5, escalation_threshold=3, loop.py:64-92) escalates only on a
    REPEATED signature, so varied offsets never collapse the run early. Bounded at `n` so a
    NON-enforcing loop terminates (it runs the whole plan then echoes) instead of hanging.
    """
    target = "/tmp/bench_fixtures/single_read_target.txt"
    return [("read", {"path": target, "offset": i + 1}) for i in range(n)]


# ---------------------------------------------------------------------------
# AUDIT-02a: scenario.budget.max_actions is never threaded into the loop.
# ---------------------------------------------------------------------------
async def test_scenario_max_actions_not_enforced(tool_scenario_corpus, faithful_fake_llm, tmp_path):
    # RED today (loop built with default max_actions=100). xfail strict flips to a real pass when
    # a future phase threads scenario.budget.max_actions onto the AgentConfig the bench builds.
    scen = _budget_scenario(tmp_path, name="cap_actions", max_actions=1)
    assert scen.budget.max_actions == 1

    # Plan emits 10 read calls with varied offsets — an ENFORCING loop (cap=1) dispatches exactly
    # 1 before the next iteration's budget check (actions_taken >= 1) terminates it; a
    # NON-enforcing loop (today's default cap=100) dispatches all 10.
    fake = faithful_fake_llm(tool_plan=_never_completes_read_plan(10))
    samples, _reason = await accumulate_runs(
        scen,
        "test-model",
        tmp_path / "results",
        llm_client_factory=lambda _s: fake,
        min_runs_override=1,
        max_runs_override=1,
    )
    completed = samples[0]

    # THE FIX target: the loop must honor the scenario's max_actions=1 ceiling. tool_call_count is
    # one Action per dispatched tool (runner.py:64-66); with the cap enforced it cannot exceed 1.
    # Today the loop uses max_actions=100, so all 10 reads dispatch (tool_call_count == 10) and
    # this fails -> xfail(strict=True) PASSES the suite (the documented gap).
    assert completed.tool_call_count <= scen.budget.max_actions


# ---------------------------------------------------------------------------
# AUDIT-02b: limits.max_tool_calls has ZERO readers; the run is not capped.
# ---------------------------------------------------------------------------
def test_max_tool_calls_has_zero_readers():
    # GREEN source-level characterization (14-03 regression-guard precedent): PINS the zero-readers
    # gap. Flips RED the moment a future fix adds a reader, alerting the fix phase to update it.
    hits = subprocess.run(
        ["grep", "-rn", "max_tool_calls", "src/localharness/"],
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    # Exactly ONE line — the schema.py:96 DEFINITION. Zero readers anywhere in src.
    assert len(hits) == 1, f"expected only the schema definition, got readers: {hits}"
    assert "schema.py" in hits[0] and "max_tool_calls" in hits[0]


@pytest.mark.xfail(
    strict=True,
    reason="AUDIT-02: limits.max_tool_calls is enforced nowhere (zero readers, schema.py:96 is "
    "the only line); the run is not capped — execute_one_run reads scen.limits.max_latency_s "
    "but never scen.limits.max_tool_calls",
)
async def test_scenario_max_tool_calls_not_enforced(tool_scenario_corpus, faithful_fake_llm, tmp_path):
    scen = _budget_scenario(tmp_path, name="cap_tool_calls", max_tool_calls=1)
    assert scen.limits.max_tool_calls == 1

    # Fake emits >=2 read calls (varied args) so >=2 tools dispatch.
    fake = faithful_fake_llm(tool_plan=_never_completes_read_plan(3))
    samples, _reason = await accumulate_runs(
        scen,
        "test-model",
        tmp_path / "results",
        llm_client_factory=lambda _s: fake,
        min_runs_override=1,
        max_runs_override=1,
    )
    completed = samples[0]

    # tool_call_count as a CEILING is the one valid count-based assertion (RESEARCH AUDIT-02b): the
    # test is about the CEILING on the count, not whether a tool ran. RED today because
    # max_tool_calls has zero readers -> tool_call_count >= 2 > 1 -> fails -> xfail strict.
    assert completed.tool_call_count <= scen.limits.max_tool_calls


# ---------------------------------------------------------------------------
# AUDIT-02c: scenario.budget.max_duration_minutes never reaches the inner BudgetTracker.
# ---------------------------------------------------------------------------
async def test_scenario_max_duration_not_threaded(tool_scenario_corpus, faithful_fake_llm, tmp_path, monkeypatch):
    # Characterization, NOT wall-clock. Documents the unreconciled gap: the outer
    # asyncio.wait_for(timeout=scen.limits.max_latency_s) (runner.py:175-178) and the inner
    # BudgetTracker(max_duration_minutes=self._config.permissions.budget.max_duration_minutes,
    # default 30.0 — loop.py:435-437, models.py:139-144) are INDEPENDENT;
    # scenario.budget.max_duration_minutes never reaches the inner tracker. The real wall-clock
    # variant is a live_vllm/slow opt-in (not written here — a ~30s termination is slow/flaky).
    #
    # Form A (the test_experiment_overlay_e2e.py:102-118 precedent): capture the AgentConfig that
    # ACTUALLY reaches the loop at the real _build_agent_loop seam, return a stub loop, and
    # no-op _run_loop so no turn runs (deterministic, fast). The captured AgentConfig IS the
    # artifact under audit here (this is NOT stubbing the spine-under-test for AUDIT-01 — that is
    # 21-02; here the construction-only capture of the config object is the whole point).
    from localharness.bench import runner as bench_runner

    scen = _budget_scenario(tmp_path, name="cap_duration", max_duration_minutes=0.5)
    assert scen.budget.max_duration_minutes == 0.5

    captured: dict = {}
    real_build = bench_runner._build_agent_loop

    def _capture(bus, llm_client, scenario, session_id="", agent_config=None, base_registry=None):
        # Build the loop exactly as production does (so the captured config is the genuine one the
        # live loop would read its budget from), then record the AgentConfig it was built with.
        loop = real_build(
            bus=bus,
            llm_client=llm_client,
            scenario=scenario,
            session_id=session_id,
            agent_config=agent_config,
            base_registry=base_registry,
        )
        captured["cfg"] = loop._config
        return loop

    async def _noop_run_loop(loop, prompt, on_token):
        return None

    monkeypatch.setattr(bench_runner, "_build_agent_loop", _capture)
    monkeypatch.setattr(bench_runner, "_run_loop", _noop_run_loop)

    fake = faithful_fake_llm(tool_plan=_never_completes_read_plan(1))

    samples, _reason = await accumulate_runs(
        scen,
        "test-model",
        tmp_path / "results",
        llm_client_factory=lambda _s: fake,
        min_runs_override=1,
        max_runs_override=1,
    )
    assert samples  # the run completed (no turn ran — _run_loop is a no-op)

    cfg = captured["cfg"]
    # The DEFAULT BudgetConfig.max_duration_minutes (30.0) reaches the loop, NOT the scenario's 0.5.
    # Written so it FLIPS when a future phase threads scenario.budget.max_duration_minutes onto the
    # AgentConfig the bench builds (then the captured value becomes 0.5 and this assertion fails,
    # alerting the fix phase).
    assert cfg.permissions.budget.max_duration_minutes == 30.0
    assert cfg.permissions.budget.max_duration_minutes != scen.budget.max_duration_minutes
