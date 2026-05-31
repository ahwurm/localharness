"""Phase 20 Wave-0 RED Nyquist gate — the WARNING-3 coverage closure.

This is the ONLY test in the suite that references `_build_default_run_slice`. The v1.1
milestone audit found ZERO tests exercise the real bench-backed slice runner, which is
exactly how the load-bearing blocker (the bench never applies the experiment overlay, so
the promotion gate compares a config against itself) hid behind 742 green tests.

It drives the REAL `_build_default_run_slice` / `run_slice` (NO injected fake, NO
monkeypatched `run_experiment`) and asserts the proposal arm (`with_overlay=True`) and the
baseline arm (`with_overlay=False`) resolve a DIFFERENT live `AgentConfig.stuck_detector.
window_size`.

RED before the Sites-1/2 wiring (today the bench ignores the overlay: experiment.py:280-283
is a bare `pass`, and bench/runner.py:252 builds `AgentConfig(name, role)` from scratch, so
both arms resolve the identical default window_size=5 → `{5}` != `{5, 9}` → the assertion
fails). GREEN after Plan 20-02 wires the worktree cascade into the bench.
"""
from __future__ import annotations

import subprocess

import yaml  # noqa: F401 — kept for parity with the experiment overlay (de)serializer surface

import pytest  # noqa: F401 — asyncio_mode=auto; imported for explicit test-runner discovery

from localharness.autoresearch import experiment as exp
from localharness.bench import orchestrator as bench_orch
from localharness.bench import runner as bench_runner
from localharness.bench.schema import ScenarioSpec, load_scenario
from localharness.registry.paths import get_value


def _scenario_body(name: str, slice_: str) -> str:
    """A complete, VALID ScenarioSpec YAML (mirrors conftest._scenario_yaml).

    The minimal `name:/slice:` stubs in test_experiment_e2e.py are NOT valid and get
    dropped by _load_scenarios_from_paths — the corpus MUST write the full body or the
    slice resolves empty and _build_agent_loop is never reached.
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
    # Hyphen (not underscore) in the name: the bench synthesizes AgentConfig(name=f"bench-{name}")
    # and AgentConfig.name must be lowercase-alnum-hyphen (models.py:474) — "bench-train_01" would
    # fail validation, "bench-train-01" passes.
    (train / "train-01.yaml").write_text(_scenario_body("train-01", "train"), encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init corpus"], cwd=root, check=True)
    return root


async def test_overlay_diverges_arms(tmp_path, monkeypatch):
    wt = _make_corpus_repo(tmp_path / "wt")
    # Materialize the candidate mutation INTO the worktree, exactly as run_experiment does
    # (experiment.py:355): agent.stuck_detector.window_size 5 (default) -> 9.
    exp.write_experiment_overlay(wt, "agent.stuck_detector.window_size", 9, annotation=int)

    # Capture the resolved AgentConfig that ACTUALLY reaches the loop, per arm — WITHOUT
    # completing a turn. Accept the FUTURE signature (agent_config kwarg) so this stays valid
    # after Plan 20-02 threads it in as the last kwarg to _build_agent_loop.
    #
    # Today the bench passes NO agent_config (execute_one_run calls _build_agent_loop with only
    # bus/llm_client/scenario/session_id), so the config the live loop is built from is the
    # default `AgentConfig(name=..., role=...)` constructed INSIDE _build_agent_loop at
    # runner.py:252. To capture the config the loop is genuinely built from in BOTH eras, mirror
    # that fallback here: record the threaded agent_config when present (post-20-02), else
    # synthesize today's default exactly as the live builder does. This makes the captured value
    # the real per-arm config — window_size=5 for both arms today (overlay ignored → RED), and
    # 5 vs 9 once Plan 20-02 wires the worktree cascade (→ GREEN).
    from localharness.config.models import AgentConfig

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

    # The live `_run_slice` passes the raw `_discover_scenarios(...)` -> list[Path] straight into
    # `_filter_scenarios_by_slice` (experiment.py:279), which reads `s.slice` — a latent production
    # bug never exercised before this test (the WARNING-3 hole: the canonical caller at
    # orchestrator.py:283 loads via `_load_scenarios_from_paths` first). Patch the filter at the
    # `_run_slice` import seam (bench.orchestrator) to load any Path entries (and pass ScenarioSpec
    # entries through, so it stays correct once Plan 20-02 wires the load step into `_run_slice`).
    # NOTE: this injects NO fake slice runner and patches NO top-level gate entrypoint — the REAL
    # `_build_default_run_slice` / `run_slice` / `slice_success_by_fixture` / `_build_agent_loop`
    # chain still executes; only the broken discovery-filter seam is compensated.
    def _load_tolerant_filter(scenarios, slice_):
        specs = [load_scenario(s) if not isinstance(s, ScenarioSpec) else s for s in scenarios]
        if slice_ == "all":
            return specs
        return [s for s in specs if s.slice == slice_]

    monkeypatch.setattr(bench_orch, "_filter_scenarios_by_slice", _load_tolerant_filter)
    # Stay offline: inject a canned client factory so no Ollama is reached (the captured cfg
    # is what matters — _run_loop never touches the stub loop or the client).
    factory = lambda _scen: object()  # noqa: E731

    # Drive the REAL slice runner — this is the WARNING-3 closure; it references
    # _build_default_run_slice by name and runs the genuine config-resolution path.
    run_slice = exp._build_default_run_slice(
        "test-model",
        factory,
        annotation=int,
        component="agent.stuck_detector.window_size",
        after=9,
    )
    base = await run_slice(wt, slice="train", with_overlay=False)  # baseline arm  # noqa: F841
    head = await run_slice(wt, slice="train", with_overlay=True)  # proposal arm   # noqa: F841

    # The captured cfgs are interleaved baseline-then-proposal across the two run_slice calls.
    cfgs = [c for c in captured.get("cfgs", []) if c is not None]
    assert cfgs, "no AgentConfig captured — _build_agent_loop was never reached (corpus invalid?)"
    windows = {get_value(c, "stuck_detector.window_size") for c in cfgs}
    # CAPABILITY assertion: the two arms must resolve DIFFERENT window_size (baseline 5 vs
    # proposal 9). NOT a score lift, NOT a Welch verdict (CONTEXT/RESEARCH LOCKED).
    assert windows == {5, 9}, (
        f"arms did not diverge — got window sizes {windows}; "
        f"with_overlay=True must apply the experiment overlay (window_size=9), "
        f"with_overlay=False must NOT (default 5). Today the bench ignores the overlay "
        f"(experiment.py:280-283 is a no-op)."
    )
