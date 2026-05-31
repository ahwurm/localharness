"""AUDIT-01 + AUDIT-06: the un-stubbed bench-spine end-to-end characterization.

GREEN against the current working tree (the four point-fixes ARE present); it is the
un-stubbed proof that AUDIT-01's spine works AND the AUDIT-06 fix#1/#4 re-verification.
It goes RED the instant `_build_agent_loop` stops receiving a populated `base_registry`
or the agent-name sanitizer is removed.

Why this test exists (the AUDIT-01 meta-finding): EVERY pre-existing bench driver injects a
fake `run_slice`/`run_experiment`, stubs `_build_agent_loop`/`_run_loop`, or feeds a scripted
`MockLLMClient` that emits tool_calls REGARDLESS of whether the tools actually exist. That is
precisely how a zero-tool bench passed CI. This test drives the GENUINE chain —
`accumulate_runs -> execute_one_run -> _get_base_registry -> _build_agent_loop -> real AgentLoop
-> real tool dispatch -> real file I/O` — with NO monkeypatch of any spine seam. The injection
point is a fake MODEL (the 21-01 `faithful_fake_llm`, whose final answer ECHOES the last real
tool result), NOT faked wiring. Its success is causally dependent on the real `read`/`write`
tools running.

THE GREEN-CHECK TRAP (the reason these assertions check real side-effects, not tool_call_count):
`MetricAccumulator.on_action` (runner.py:64-66) increments `tool_call_count` on the
`Action(action_type="tool_call", ...)` event, which the loop publishes BEFORE dispatch
(loop.py:716-723). Dispatch failure is swallowed into a tool-result string (loop.py:759-761).
So `tool_call_count > 0` increments EVEN when the tool does not exist and dispatch fails —
it passes against a zero-tool bench and is therefore NOT a valid AUDIT-01 assertion. The valid
assertions are REAL side-effects: single_read -> `completed.success is True` (the apricot rubric
only matches if the real `read` returned the staged file content); write_execute -> the on-disk
file exists with HELLO_BENCH_OK.

SC9 (sealed holdout): every scenario here is slice='train'. The holdout slice is NEVER executed.
"""
from __future__ import annotations

import pathlib

import pytest

from localharness.bench.runner import accumulate_runs
from localharness.bench.schema import load_scenario


async def test_spine_single_read_real_dispatch(tool_scenario_corpus, faithful_fake_llm, tmp_path):
    """AUDIT-01 core + AUDIT-06 fix#1: drive the real spine through a single `read` and prove the
    success is causally dependent on the real tool returning the staged file's content."""
    # 1. Load the single_read ScenarioSpec (rubric: contains:apricot; tools_allowed: [read]).
    scen = load_scenario(tool_scenario_corpus["single_read"])

    # 2. The faithful-fake emits ONE `read` against the staged target, then echoes the result.
    #    read_target == /tmp/bench_fixtures/single_read_target.txt (staged content: "apricot").
    fake = faithful_fake_llm(tool_plan=[("read", {"path": tool_scenario_corpus["read_target"]})])

    # 3. Drive the REAL spine via accumulate_runs — the legitimate LLM-client-factory seam
    #    (a fake MODEL, real wiring). NO monkeypatch of _build_agent_loop / _run_loop / run_slice /
    #    run_experiment: execute_one_run calls _get_base_registry() (the real builtin registry) then
    #    _build_agent_loop(base_registry=...) -> real AgentLoop -> fake. min/max override 1 = one rep.
    results_root = tmp_path / "results"
    samples, _stop = await accumulate_runs(
        scen,
        "faithful-fake",
        results_root,
        llm_client_factory=lambda _scen: fake,
        min_runs_override=1,
        max_runs_override=1,
    )
    completed = samples[0]

    # 4. The green-check-trap-aware assertion set.
    # tool_call_count > 0 is NECESSARY but NOT SUFFICIENT: on_action increments BEFORE dispatch
    # (runner.py:64-66 fires on the Action event the loop publishes at loop.py:716-723, pre-dispatch;
    #  dispatch failure is swallowed at loop.py:759-761). So this alone passes against a zero-tool bench.
    assert completed.tool_call_count >= 1
    # THE REAL PROOF: success is True ONLY because the real `read` tool returned the staged file's
    # content ("apricot") and the rubric matched. A broken/empty registry -> read dispatch fails ->
    # the fake echoes the error string -> rubric fails -> success is False. THIS is what makes it RED
    # when the seam is broken.
    assert completed.success is True, (
        "spine did not actually read the file — the rubric (contains 'apricot') failed, "
        "which means the real `read` tool did not return the staged content"
    )

    # 5. fix#1 (agent-name sanitizer) re-verification by REAL execution.
    # An unsanitized name would crash AgentConfig.name (models.py:474, [a-z0-9-] only) at
    # construction; a completed underscore-named run proves the bench-single-read sanitization held
    # (runner.py:277 turns `single_read` into the valid agent name `bench-single-read`).
    assert completed.scenario_name == "single_read"


async def test_spine_write_execute_real_file(tool_scenario_corpus, faithful_fake_llm, tmp_path):
    """AUDIT-01 strongest proof: an on-disk side-effect, foolproof vs a model merely saying the
    token in prose. A model that SAYS HELLO_BENCH_OK with zero tools is the false-positive the
    brief calls out; the on-disk file is not foolable."""
    # 1. Pre-clean the real target so a stale file can't mask a non-dispatching spine.
    target = pathlib.Path(tool_scenario_corpus["write_target"])  # /tmp/bench_fixtures/hello_bench.py
    target.unlink(missing_ok=True)

    try:
        # 2. Load write_execute (rubric: contains:HELLO_BENCH_OK; tools_allowed: [write, bash_exec]).
        scen = load_scenario(tool_scenario_corpus["write_execute"])

        # Two-step plan: write the file, then run it. args match the tool schemas verbatim
        # (write requires path+content, bash_exec requires command) or dispatch validation fails.
        fake = faithful_fake_llm(
            tool_plan=[
                ("write", {"path": str(target), "content": "print('HELLO_BENCH_OK')"}),
                ("bash_exec", {"command": f"python3 {target}"}),
            ]
        )

        # 3. Drive the REAL spine exactly as the single_read test (no spine monkeypatch).
        results_root = tmp_path / "results"
        samples, _stop = await accumulate_runs(
            scen,
            "faithful-fake",
            results_root,
            llm_client_factory=lambda _s: fake,
            min_runs_override=1,
            max_runs_override=1,
        )
        completed = samples[0]

        # 4. THE proof — the real file exists on disk with the written content.
        # This is RED if base_registry is broken (write never dispatches -> no file). A model
        # merely echoing HELLO_BENCH_OK in prose cannot make this pass.
        assert target.exists(), (
            "write tool did not create the real file — the spine did not dispatch `write`"
        )
        assert "HELLO_BENCH_OK" in target.read_text()
        # The rubric success (echoed bash_exec output contains HELLO_BENCH_OK) corroborates dispatch.
        assert completed.success is True
    finally:
        # 5. Teardown — the on-disk artifact is volatile test state.
        target.unlink(missing_ok=True)


def test_from_allowed_none_base_is_empty():
    """REG-01 (AUDIT-06 fix#4): from_allowed(base_registry=None) must RAISE, not silently return
    a zero-tool registry.

    The prior characterization documented the foot-gun (from_allowed returned an empty registry
    for None base, a silent failure mode). The fix (phase-22-plan-05) makes None an explicit
    ValueError so a caller that forgets _get_base_registry() gets an immediate, actionable error
    rather than a zero-tool agent loop.

    POST-FIX: from_allowed(["read"], base_registry=None) raises ValueError.
    """
    import pytest as _pytest

    from localharness.tools.registry import ToolRegistry

    with _pytest.raises(ValueError, match="base_registry"):
        ToolRegistry.from_allowed(["read"], base_registry=None)


@pytest.mark.live_vllm
async def test_spine_single_read_live_vllm(tool_scenario_corpus, tmp_path):
    """AUDIT-01 SC1 honest end-to-end: the SAME accumulate_runs spine, but against a REAL probed
    LLMClient instead of the faithful-fake. Skipped by default — the 21-01 autouse `_skip_live_vllm`
    guard skips it unless LOCALHARNESS_LIVE_VLLM=1 (no real endpoint is hit in CI).

    Model-agnostic (LOCKED guardrail): the provider/model/base_url are resolved from the loaded cfg
    — NEVER a baked model id. Mirrors orchestrator._build_bench_client + the _run_one_model probe.
    Guardrail: runs ONLY the train-slice single_read scenario — the holdout slice is SEALED and
    is never executed, read, or passed anywhere in this file.
    """
    # 1. Resolve the real provider/model from cfg (never bake an id) and build a probed client.
    from localharness.bench.config import MatrixEntry
    from localharness.bench.orchestrator import _build_bench_client
    from localharness.cli.components_cmd import _build_loader

    cfg = _build_loader().load_harness()
    entry = MatrixEntry(
        name=cfg.provider.default_model,
        provider=cfg.provider.provider_type,
        model_id=cfg.provider.default_model,
        base_url=cfg.provider.base_url,
    )
    client = _build_bench_client(entry)
    # The PROBE — sets the real tool_call_mode (native vs xml). The gate path skips this (AUDIT-03a);
    # the honest end-to-end variant resolves the mode the same way production _run_one_model does.
    await client.detect_capabilities()

    # 2. Drive the same single_read spine with the real client.
    scen = load_scenario(tool_scenario_corpus["single_read"])
    results_root = tmp_path / "results"
    samples, _stop = await accumulate_runs(
        scen,
        cfg.provider.default_model,
        results_root,
        llm_client_factory=lambda _s: client,
        min_runs_override=1,
        max_runs_override=1,
    )
    completed = samples[0]

    # 3. A real model, with real tools + the apricot rubric, reads the staged file.
    assert completed.success is True


# ---------------------------------------------------------------------------
# LIVE-01: the full-spine + autoresearch-loop dimension.
#
# This invokes the REAL two-stage gate (run_experiment) ONCE against live vLLM, but is
# constructed so the TRAIN arm CANNOT improve — the gate returns a train-reject verdict
# (EXIT_REJECT_TRAIN, experiment.py:479) or, with too few paired fixtures, EXIT_INCONCLUSIVE —
# BEFORE the sealed HOLDOUT stage (experiment.py:482-483) is ever reached. The holdout seal is
# proven structurally: a recording wrapper over the run_slice closure captures every requested
# slice and we assert "holdout" was NEVER among them.
#
# Holdout-unreached construction (RESEARCH Pitfall 3, option (b)): the proposal is a NO-OP
# mutation — the experiment overlay sets agent.role to the SAME value already adopted in the
# worktree overrides.yaml — so both arms resolve the identical AgentConfig and the proposal
# CANNOT outperform the baseline. welch_improvement -> False -> train-reject before holdout.
# Model-agnostic (LOCKED): provider/model/base_url come from cfg.provider; no baked model id.
# ---------------------------------------------------------------------------


def _make_train_corpus_repo(path):
    """git init a repo with a committed TRAIN-ONLY bench corpus (two scenarios, never holdout).

    The corpus is intentionally train-only: even a logic slip in the gate cannot load a holdout
    scenario because none exists on disk. Two fixtures so the Welch pair-count path is reachable.
    """
    import subprocess

    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True, capture_output=True)
    scen_dir = path / "bench" / "scenarios" / "train"
    scen_dir.mkdir(parents=True)
    for n in ("noop_01", "noop_02"):
        # A COMPLETE valid ScenarioSpec (minimal name:/slice: stubs are dropped by
        # _load_scenarios_from_paths). A zero-tool, trivially-satisfiable scenario keeps the
        # live generation cheap while still driving the real spine.
        (scen_dir / f"{n}.yaml").write_text(
            "\n".join(
                [
                    f"name: {n}",
                    "slice: train",
                    "category: tool_selection",
                    "prompt: 'Reply however you like, then end your turn.'",
                    "success_criteria:",
                    "  - kind: contains",
                    "    value: ''",
                    "tools_allowed: []",
                    "budget:",
                    "  max_actions: 1",
                    "  max_duration_minutes: 1",
                    "limits:",
                    "  max_tool_calls: 1",
                    "  max_latency_s: 120",
                    "min_runs: 1",
                    "max_runs: 1",
                    "tags: []",
                    "context_files: []",
                    "expected_outcome: 'noop'",
                    "ts: 0",
                ]
            ),
            encoding="utf-8",
        )
    subprocess.run(["git", "add", "-A"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "init train corpus"], cwd=path, check=True, capture_output=True)
    return path


@pytest.mark.live_vllm
async def test_live_full_loop_holdout_unreached(live_endpoint, tmp_path):
    """LIVE-01: the REAL run_experiment gate runs e2e against live vLLM and train-rejects a no-op
    mutation, so the sealed HOLDOUT stage is provably never reached.

    Skipped by default (autouse _skip_live_vllm); the live_endpoint preflight hard-fails if the
    user opted in but the endpoint is down. EXIT_* are imported from experiment.py (single source
    of truth) — no integer exit-code literal is asserted for the verdict.
    """
    import uuid

    from localharness.autoresearch.archive import ArchiveEntry, ArchiveStore
    from localharness.autoresearch.experiment import (
        EXIT_INCONCLUSIVE,
        EXIT_REJECT_TRAIN,
        _build_default_run_slice,
        run_experiment,
    )
    from localharness.cli.components_cmd import _build_loader
    from localharness.config.overlay import atomic_write_overlay

    cfg = _build_loader().load_harness()  # provider/model/base_url resolved from config

    # 1. A real train-only worktree-source repo (holdout fixtures do not exist on disk).
    repo = _make_train_corpus_repo(tmp_path / "repo")

    # 2. The NO-OP mutation: adopt agent.role=<X> as baseline (BOTH arms) AND make the proposal's
    #    after=<X> too. Both arms then resolve the identical AgentConfig -> the proposal cannot
    #    outperform the baseline -> welch_improvement is False -> train-reject before holdout.
    noop_role = "Bench harness execution baseline role"
    atomic_write_overlay(repo / ".localharness" / "overrides.yaml", {"agent": {"role": noop_role}})

    # 3. Seed a real in_flight proposal in a temp archive (agent.role IS in the mutable catalogue).
    store = ArchiveStore(tmp_path / "archive.db")
    await store.open()
    proposal_id = uuid.uuid4().hex
    await store.write(
        ArchiveEntry(
            id=proposal_id,
            parent_id=None,
            component="agent.role",
            diff=__import__("json").dumps({"before": noop_role, "after": noop_role}),  # no-op
            train_score=None,
            train_scores_per_fixture=None,
            holdout_score=None,
            p_value=None,
            cost=None,
            ts=0,
            approved_by=None,
            status="in_flight",
        )
    )

    # 4. Wrap the REAL bench-backed run_slice closure in a recorder so every requested slice is
    #    captured — the structural proof that the holdout slice is NEVER constructed. The inner
    #    closure resolves the live client from cfg.provider + detect_capabilities (FIDEL-01).
    real_slice = _build_default_run_slice(cfg.provider.default_model, None, cfg=cfg)
    slices_requested: list[str] = []

    async def recording_run_slice(worktree, *, slice, with_overlay):
        slices_requested.append(slice)
        return await real_slice(worktree, slice=slice, with_overlay=with_overlay)

    try:
        # 5. Drive the REAL two-stage gate ONCE against live vLLM via the recording closure.
        exit_code = await run_experiment(
            proposal_id,
            trials=1,
            store=store,
            run_slice=recording_run_slice,
            repo_root=repo,
            cfg=cfg,
            bus=None,
        )
    finally:
        await store.close()

    # 6. STRUCTURAL assertions only (model output is non-deterministic).
    # 6a. Verdict is in the train-reject band — NOT EXIT_PROMOTE(0)/EXIT_REJECT_HOLDOUT(2),
    #     which would imply the holdout stage ran. Named constants, no integer literal.
    assert exit_code in (EXIT_REJECT_TRAIN, EXIT_INCONCLUSIVE), (
        f"expected a train-reject-band verdict (no holdout), got exit {exit_code}"
    )
    # 6b. THE SEAL: the holdout slice was never requested by the gate (STATE.md 17-01 pattern).
    assert "holdout" not in slices_requested, (
        f"sealed holdout slice was constructed: {slices_requested}"
    )
    # 6c. The gate DID run the real train spine (it is a full-loop test, not a no-op refusal).
    assert "train" in slices_requested
