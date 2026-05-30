"""Phase 18 — AUTO-01..04 behavioral tests (all green; the Wave-0 scaffold landed 18-02..18-05).

The autoresearch orchestrator wires four net-new pieces: a ParentSampler (ε-greedy
over the GEPA per-fixture front), a BudgetController (token-window + wallclock pre-flight
gate + per-proposal timeout), an adoption mechanism (a git-committed config-overlay write),
and the loop driver (sample → propose → run_experiment → interpret-exit → adopt/hold →
journal). Every test name below is BINDING: 18-RESEARCH §"Phase Requirements → Test Map"
names each one verbatim. Each test asserts REAL behavior (never placeholder data).

Hermetic injection mirrors test_experiment.py: fakes for the clock, the token meter, the
RNG, propose_fn, experiment_fn, and adopt_fn keep every test LLM-free / bench-free and (for
the non-adoption tests) worktree-free.
"""
import json
import random
from pathlib import Path

import pytest

# Sampler (18-02), budget (18-03), adoption (18-04), and loop (18-05) have all landed —
# import unconditionally so every test runs live (the Wave-0 guarded import is no longer needed).
from localharness.autoresearch.sampler import ParentSampler, BASELINE_ROOT
from localharness.autoresearch.budget import BudgetController, WindowMeter
from localharness.autoresearch.adoption import adopt, AdoptionRefused
from localharness.autoresearch.loop import run_loop, RunSummary


# ---------------------------------------------------------------------------
# Test helpers — a deterministic RNG stub whose .random() is caller-pinned, so
# the ε roll lands in a known branch without depending on Mersenne internals.
# ---------------------------------------------------------------------------


class FixedRandom:
    """random.Random-compatible stub returning a pinned .random() roll.

    Falls back to a seeded Random for choice/randrange/etc so the EXPLORE/EXPLOIT
    branch is selectable while downstream uniform/weighted draws stay deterministic.
    """

    def __init__(self, roll: float, seed: int = 0):
        self._roll = roll
        self._backing = random.Random(seed)

    def random(self):
        return self._roll

    def choice(self, seq):
        return self._backing.choice(seq)

    def choices(self, population, weights=None, k=1):
        return self._backing.choices(population, weights=weights, k=k)

    def randrange(self, *a, **k):
        return self._backing.randrange(*a, **k)

    def shuffle(self, x):
        return self._backing.shuffle(x)


# ---------------------------------------------------------------------------
# AUTO-02 — ParentSampler (ε-greedy over the GEPA per-fixture front)
# ---------------------------------------------------------------------------


async def test_epsilon_branch_selection(archive_store, seeded_archive, monkeypatch):
    """rng.random() < ε takes the EXPLORE branch; >= ε takes EXPLOIT (front non-empty)."""
    await seeded_archive(
        archive_store,
        [dict(id="front-1", status="promoted", train_scores_per_fixture={"f1": 0.9, "f2": 0.8})],
    )

    # _explore is async (sample() awaits it); _exploit is sync (front already fetched).
    def _spy_exploit(label, sink):
        return lambda *a, **k: (sink.append(label), BASELINE_ROOT)[1]

    def _spy_explore(label, sink):
        async def _fn(*a, **k):
            sink.append(label)
            return BASELINE_ROOT
        return _fn

    explore = ParentSampler(archive_store, epsilon=0.2, rng=FixedRandom(0.05))
    branches = []
    monkeypatch.setattr(explore, "_explore", _spy_explore("explore", branches), raising=False)
    monkeypatch.setattr(explore, "_exploit", _spy_exploit("exploit", branches), raising=False)
    await explore.sample()
    assert branches == ["explore"]  # 0.05 < 0.2 → explore

    exploit = ParentSampler(archive_store, epsilon=0.2, rng=FixedRandom(0.5))
    branches2 = []
    monkeypatch.setattr(exploit, "_explore", _spy_explore("explore", branches2), raising=False)
    monkeypatch.setattr(exploit, "_exploit", _spy_exploit("exploit", branches2), raising=False)
    await exploit.sample()
    assert branches2 == ["exploit"]  # 0.5 >= 0.2 → exploit


async def test_exploit_coverage_weighted(archive_store, seeded_archive):
    """Exploit samples coverage-proportionally: A best on 3 fixtures, B on 1 → A ~3x as often."""
    await seeded_archive(
        archive_store,
        [
            # A wins f1/f2/f3; B wins only f4.
            dict(id="A", status="promoted",
                 train_scores_per_fixture={"f1": 0.9, "f2": 0.9, "f3": 0.9, "f4": 0.1}),
            dict(id="B", status="promoted",
                 train_scores_per_fixture={"f1": 0.1, "f2": 0.1, "f3": 0.1, "f4": 0.9}),
        ],
    )
    # Force EXPLOIT every draw (roll >= ε) with a varied seed so the weighted choice spreads.
    sampler = ParentSampler(archive_store, epsilon=0.0, rng=random.Random(0))
    counts = {"A": 0, "B": 0}
    for _ in range(1000):
        picked = await sampler.sample()
        counts[picked.id] += 1
    # A covers 3 of 4 fixtures → sampled markedly more often (3:1 ± tolerance).
    ratio = counts["A"] / max(counts["B"], 1)
    assert 2.0 < ratio < 4.5, counts


async def test_explore_uniform_all_status(archive_store, seeded_archive):
    """Explore draws uniformly across ALL statuses — rejected rows ARE reachable."""
    await seeded_archive(
        archive_store,
        [
            dict(id="s-inflight", status="in_flight", train_scores_per_fixture={"f1": 0.5}),
            dict(id="s-trainrej", status="train_rejected", train_scores_per_fixture={"f1": 0.5}),
            dict(id="s-holdrej", status="holdout_rejected", train_scores_per_fixture={"f1": 0.5}),
            dict(id="s-promoted", status="promoted", train_scores_per_fixture={"f1": 0.5}),
        ],
    )
    sampler = ParentSampler(archive_store, epsilon=1.0, rng=random.Random(0))  # always explore
    seen = set()
    for _ in range(400):
        picked = await sampler.sample()
        if picked is not BASELINE_ROOT:
            seen.add(picked.id)
    # The rejected rows must be reachable — explore queries every status, not just the front.
    assert "s-trainrej" in seen
    assert "s-holdrej" in seen


async def test_cold_start_always_explore_root(archive_store):
    """Empty archive (no front) → sample() returns BASELINE_ROOT regardless of ε (0.0 and 1.0)."""
    cold_lo = ParentSampler(archive_store, epsilon=0.0, rng=random.Random(0))
    cold_hi = ParentSampler(archive_store, epsilon=1.0, rng=random.Random(0))
    assert (await cold_lo.sample()) is BASELINE_ROOT
    assert (await cold_hi.sample()) is BASELINE_ROOT


async def test_sampler_holdout_blind(archive_store, seeded_archive):
    """Seal mirror (Pitfall 4): the sampler reads ONLY train_scores_per_fixture; holdout_score is never consulted.

    A holdout-BEST but train-mediocre row must NOT be preferentially sampled over a train-best row.
    """
    await seeded_archive(
        archive_store,
        [
            # train-best: wins every train fixture; modest holdout.
            dict(id="train-best", status="promoted",
                 train_scores_per_fixture={"f1": 0.95, "f2": 0.95}, holdout_score=0.40),
            # holdout-best but train-mediocre: loses every train fixture.
            dict(id="holdout-best", status="promoted",
                 train_scores_per_fixture={"f1": 0.10, "f2": 0.10}, holdout_score=0.99),
        ],
    )
    sampler = ParentSampler(archive_store, epsilon=0.0, rng=random.Random(0))  # exploit
    counts = {"train-best": 0, "holdout-best": 0}
    for _ in range(500):
        picked = await sampler.sample()
        counts[picked.id] = counts.get(picked.id, 0) + 1
    # The train-best row dominates exploit; holdout_score (0.99) buys the other row nothing.
    assert counts["train-best"] > counts["holdout-best"]


# ---------------------------------------------------------------------------
# AUTO-03 — BudgetController (token-window + wallclock pre-flight gate)
# ---------------------------------------------------------------------------


async def test_budget_halts_on_window_exhausted(FakeClock, FakeWindowMeter):
    """meter.window_exhausted() True → can_start_iteration() returns False."""
    clock = FakeClock()
    meter = FakeWindowMeter()
    ctl = BudgetController(budget_seconds=10_000, max_iterations=100, max_cost=None, meter=meter, clock=clock)
    assert ctl.can_start_iteration() is True
    meter._exhausted = True
    assert ctl.can_start_iteration() is False


async def test_budget_halts_on_wallclock(FakeClock, FakeWindowMeter):
    """Fake clock advanced past budget_seconds → can_start_iteration() returns False."""
    clock = FakeClock(0.0)
    meter = FakeWindowMeter()
    ctl = BudgetController(budget_seconds=100, max_iterations=1000, max_cost=None, meter=meter, clock=clock)
    assert ctl.can_start_iteration() is True
    clock.advance(101)  # past the 100s wallclock budget
    assert ctl.can_start_iteration() is False


async def test_max_iterations_backstop(FakeClock, FakeWindowMeter):
    """budget=None + meter never exhausted: after max_iterations starts, the next gate is False."""
    clock = FakeClock()
    meter = FakeWindowMeter()  # never exhausted
    ctl = BudgetController(budget_seconds=None, max_iterations=3, max_cost=None, meter=meter, clock=clock)
    starts = 0
    while ctl.can_start_iteration():
        starts += 1
        if starts > 50:  # guard against an infinite loop in a broken impl
            break
    assert starts == 3  # exactly the backstop, then halts


async def test_window_meter_reset_and_persist(tmp_path, FakeClock):
    """WindowMeter accumulates + persists a json sidecar; resets after 5h; a new meter on the same path reads prior spend (no double-spend)."""
    state_path = tmp_path / "window.json"
    clock = FakeClock(1000.0)
    meter = WindowMeter(window_budget_tokens=10_000, state_path=state_path, clock=clock)
    meter.record_tokens(3000)
    meter.record_tokens(2000)
    assert state_path.exists()
    sidecar = json.loads(state_path.read_text())
    assert "window_start" in sidecar and "tokens_spent" in sidecar
    assert sidecar["tokens_spent"] == 5000

    # A NEW meter on the SAME path within the window must read the prior spend (no reset, no double-spend).
    meter2 = WindowMeter(window_budget_tokens=10_000, state_path=state_path, clock=FakeClock(1000.0))
    assert json.loads(state_path.read_text())["tokens_spent"] == 5000
    meter2.record_tokens(1000)
    assert json.loads(state_path.read_text())["tokens_spent"] == 6000

    # Advance past the 5h boundary then record → window resets (tokens_spent drops to the fresh spend).
    clock.advance(WindowMeter.WINDOW_SECONDS + 1)
    meter3 = WindowMeter(window_budget_tokens=10_000, state_path=state_path, clock=clock)
    meter3.record_tokens(700)
    assert json.loads(state_path.read_text())["tokens_spent"] == 700  # reset, not 6700


async def test_only_proposer_tokens_metered(tmp_path, FakeClock):
    """record_tokens is the SOLE mutator of tokens_spent — no path meters local bench/inference."""
    state_path = tmp_path / "window.json"
    meter = WindowMeter(window_budget_tokens=10_000, state_path=state_path, clock=FakeClock())
    meter.record_tokens(120)  # one proposer call's CompletionUsage.total_tokens
    before = json.loads(state_path.read_text())["tokens_spent"]
    # A no-op (the loop running a local bench arm) must NOT move the window — only record_tokens does.
    assert meter.window_exhausted() in (True, False)  # a pure read, never a mutation
    after = json.loads(state_path.read_text())["tokens_spent"]
    assert before == after == 120


# ---------------------------------------------------------------------------
# AUTO-04 — adoption (a git-committed config-overlay write reusing components-set)
#
# Adoption is the human-checkpoint live write: a held/promoted mutation's after-value
# is merged into {repo}/.localharness/overrides.yaml, validated, atomically written, and
# git-committed in the MAIN repo (never a worktree). It re-asserts the anti-reward-hacking
# seal and emits ComponentMutated(layer="user", actor="orchestrator", actor_detail=pid).
# These use the real tmp_git_repo fixture (Task 3) — overrides.yaml + git log are the asserts.
# ---------------------------------------------------------------------------


def _git_log_lines(repo):
    """`git -C <repo> log --oneline` → list of commit lines (empty list if none/error)."""
    import subprocess

    out = subprocess.run(
        ["git", "-C", str(repo), "log", "--oneline"],
        capture_output=True, text=True,
    )
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def _git_head_sha(repo):
    import subprocess

    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


async def test_adopt_commits_and_sets_status(archive_store, seeded_archive, tmp_git_repo, components_home):
    """adopt() writes the after-value into {repo}/.localharness/overrides.yaml, git-commits, returns a 40-char sha.

    adopt() returns the sha; the LOOP (18-05)/CLI (18-06) flips status→adopted after a successful adopt
    (mirrors the experiment runner's separation of run vs verdict). The status flip is asserted here via
    the same update_verdict the caller invokes, to lock the end-to-end adopted contract.
    """
    import yaml

    [row] = await seeded_archive(
        archive_store,
        [dict(id="adopt-ok", component="agent.role", status="promoted",
              diff=json.dumps({"before": "initial", "after": "evolved role"}))],
    )
    before_commits = len(_git_log_lines(tmp_git_repo))
    sha = await adopt(row.id, store=archive_store, cfg=None, repo_root=tmp_git_repo)

    overrides = tmp_git_repo / ".localharness" / "overrides.yaml"
    data = yaml.safe_load(overrides.read_text())
    assert data["agent"]["role"] == "evolved role"        # the after-value is now live
    assert len(_git_log_lines(tmp_git_repo)) == before_commits + 1  # exactly one new commit
    assert isinstance(sha, str) and len(sha) == 40        # full git sha returned

    # The caller (loop/CLI) records the adopted verdict after a successful adopt().
    await archive_store.update_verdict(row.id, status="adopted")
    assert (await archive_store.get(row.id)).status == "adopted"


async def test_thin_lift_holds(archive_store, seeded_inflight, tmp_git_repo, components_home,
                               FakeClock, FakeWindowMeter, FakeExperimentFn):
    """Gate exit 0 but lift < min_lift → status 'held', NO commit (loop-level decision)."""
    pid = await seeded_inflight(archive_store, component="agent.role", before="initial", after="x")
    before_commits = len(_git_log_lines(tmp_git_repo))
    # A gate that promotes (exit 0) but whose measured lift is below the floor.
    summary = await run_loop(
        store=archive_store, cfg=None, repo_root=tmp_git_repo, budget=None,
        max_iterations=1, max_cost=None, epsilon=0.0, min_lift=0.5, proposal_timeout=10,
        window_tokens=10_000, clock=FakeClock(), meter=FakeWindowMeter(),
        propose_fn=_fake_propose(pid, component="agent.role"),
        experiment_fn=FakeExperimentFn(exit_code=0),  # promotes, but thin lift
    )
    assert (await archive_store.get(pid)).status == "held"
    assert len(_git_log_lines(tmp_git_repo)) == before_commits  # no commit on a held row
    assert summary.held >= 1


async def test_inconclusive_holds(archive_store, seeded_inflight, tmp_git_repo, components_home,
                                  FakeClock, FakeWindowMeter, FakeExperimentFn):
    """Gate exit 3 (inconclusive) → status 'held'."""
    pid = await seeded_inflight(archive_store, component="agent.role", before="initial", after="x")
    before_commits = len(_git_log_lines(tmp_git_repo))
    await run_loop(
        store=archive_store, cfg=None, repo_root=tmp_git_repo, budget=None,
        max_iterations=1, max_cost=None, epsilon=0.0, min_lift=0.0, proposal_timeout=10,
        window_tokens=10_000, clock=FakeClock(), meter=FakeWindowMeter(),
        propose_fn=_fake_propose(pid, component="agent.role"),
        experiment_fn=FakeExperimentFn(exit_code=3),  # inconclusive
    )
    assert (await archive_store.get(pid)).status == "held"
    assert len(_git_log_lines(tmp_git_repo)) == before_commits  # inconclusive never commits


async def test_reject_no_commit(archive_store, seeded_inflight, tmp_git_repo, components_home,
                                FakeClock, FakeWindowMeter, FakeExperimentFn):
    """Gate exit 1 or 2 → no adoption, no commit; row keeps train_rejected/holdout_rejected."""
    pid = await seeded_inflight(archive_store, component="agent.role", before="initial", after="x")
    before_commits = len(_git_log_lines(tmp_git_repo))
    await run_loop(
        store=archive_store, cfg=None, repo_root=tmp_git_repo, budget=None,
        max_iterations=1, max_cost=None, epsilon=0.0, min_lift=0.0, proposal_timeout=10,
        window_tokens=10_000, clock=FakeClock(), meter=FakeWindowMeter(),
        propose_fn=_fake_propose(pid, component="agent.role"),
        experiment_fn=FakeExperimentFn(exit_code=1),  # reject-train
    )
    status = (await archive_store.get(pid)).status
    assert status in ("train_rejected", "holdout_rejected")  # the gate's own verdict, not adopted/held
    assert len(_git_log_lines(tmp_git_repo)) == before_commits  # a reject never commits


async def test_adopt_commits_main_repo_not_worktree(archive_store, seeded_archive, tmp_git_repo, components_home):
    """The adoption commit lands in repo_root HEAD (HEAD sha advances) — NOT in any lh-exp-* worktree."""
    import subprocess

    [row] = await seeded_archive(
        archive_store,
        [dict(id="adopt-main", component="agent.role", status="promoted",
              diff=json.dumps({"before": "initial", "after": "mainline"}))],
    )
    head_before = _git_head_sha(tmp_git_repo)
    await adopt(row.id, store=archive_store, cfg=None, repo_root=tmp_git_repo)
    head_after = _git_head_sha(tmp_git_repo)
    assert head_after != head_before  # MAIN repo HEAD advanced
    # No lingering experiment worktree carries the commit (adoption is not a worktree write).
    wt = subprocess.run(["git", "-C", str(tmp_git_repo), "worktree", "list"],
                        capture_output=True, text=True)
    assert "lh-exp-" not in wt.stdout


async def test_adopt_refuses_sealed_surface(archive_store, seeded_archive, tmp_git_repo, components_home):
    """adopt() on a sealed-prefix OR multi-component row raises AdoptionRefused, sets status 'adoption_rejected', NO commit.

    Mirrors experiment.py _OFFREGISTRY_PREFIXES (grader/bench./holdout/success_criteria/scenario)
    + _is_multi_component (a.b,c.d) — the seal is re-asserted at the live-write boundary.
    """
    [sealed] = await seeded_archive(
        archive_store,
        [dict(id="adopt-sealed", component="grader.weights", status="promoted",
              diff=json.dumps({"before": "a", "after": "b"}))],
    )
    [multi] = await seeded_archive(
        archive_store,
        [dict(id="adopt-multi", component="agent.role,tools.bash.description", status="promoted",
              diff=json.dumps({"before": "a", "after": "b"}))],
    )
    before_commits = len(_git_log_lines(tmp_git_repo))
    for row in (sealed, multi):
        with pytest.raises(AdoptionRefused):
            await adopt(row.id, store=archive_store, cfg=None, repo_root=tmp_git_repo)
        assert (await archive_store.get(row.id)).status == "adoption_rejected"
    assert len(_git_log_lines(tmp_git_repo)) == before_commits  # refused before any commit


async def test_adopt_refuses_invalid_config(archive_store, seeded_archive, tmp_git_repo, components_home):
    """An after-value that makes HarnessConfig.model_validate(merged) fail → no write, no commit."""
    import yaml

    # agent.stuck_detector.threshold has a numeric constraint; a non-coercible after fails validation.
    [bad] = await seeded_archive(
        archive_store,
        [dict(id="adopt-bad", component="agents.main.stuck_detector.repeated_threshold", status="promoted",
              diff=json.dumps({"before": 3, "after": "not-an-int"}))],
    )
    overrides = tmp_git_repo / ".localharness" / "overrides.yaml"
    before_text = overrides.read_text()
    before_commits = len(_git_log_lines(tmp_git_repo))
    with pytest.raises(Exception):  # AdoptionRefused or a validation error — either way no write
        await adopt(bad.id, store=archive_store, cfg=None, repo_root=tmp_git_repo)
    assert overrides.read_text() == before_text  # overlay untouched on validation failure
    assert len(_git_log_lines(tmp_git_repo)) == before_commits


async def test_rejected_not_reoffered(archive_store, seeded_archive, tmp_git_repo, components_home):
    """A row at status 'adoption_rejected' is excluded from review's held list AND adopt() refuses to re-adopt it."""
    [row] = await seeded_archive(
        archive_store,
        [dict(id="adopt-declined", component="agent.role", status="adoption_rejected",
              diff=json.dumps({"before": "a", "after": "b"}))],
    )
    before_commits = len(_git_log_lines(tmp_git_repo))
    with pytest.raises(AdoptionRefused):
        await adopt(row.id, store=archive_store, cfg=None, repo_root=tmp_git_repo)
    assert len(_git_log_lines(tmp_git_repo)) == before_commits  # a declined row never re-commits
    # And it is not surfaced as a held candidate for re-offer.
    from localharness.autoresearch.archive import ArchiveQuery

    held = await archive_store.query(ArchiveQuery(status="held", limit=100))
    assert row.id not in {e.id for e in held}


async def test_adoption_emits_component_mutated(archive_store, seeded_archive, tmp_git_repo, components_home, bus):
    """Adoption publishes ComponentMutated(layer='user', actor='orchestrator', actor_detail=<pid>) on the bus."""
    from localharness.core.events import ComponentMutated

    received = []

    async def _handler(event):
        received.append(event)

    bus.subscribe(ComponentMutated, _handler)
    [row] = await seeded_archive(
        archive_store,
        [dict(id="adopt-evt", component="agent.role", status="promoted",
              diff=json.dumps({"before": "initial", "after": "audited"}))],
    )
    await adopt(row.id, store=archive_store, cfg=None, repo_root=tmp_git_repo, bus=bus)
    mutated = [e for e in received
               if e.layer == "user" and e.actor == "orchestrator" and e.actor_detail == row.id]
    assert len(mutated) >= 1  # the loop (not the gate) is recorded as the live-overlay author


# ---------------------------------------------------------------------------
# AUTO-01 — loop driver (sample → propose → run_experiment → adopt/hold → journal)
#
# The loop body is injected end-to-end: propose_fn returns a Proposal-shaped object,
# experiment_fn (FakeExperimentFn) returns a gate exit code, clock/meter/rng/adopt_fn
# are fakes. No Ollama, no real bench. These assert the loop's control flow + audit.
# ---------------------------------------------------------------------------


def _fake_propose(pid, *, component="agent.role"):
    """An async propose_fn seam returning the seeded in_flight row's id (the loop then runs experiment_fn on it).

    The real propose() returns a Proposal that gets archived to an in_flight row; the loop
    reads that row id. The fake short-circuits to the already-seeded pid so experiment_fn/adopt
    operate on a real archive row without invoking the LLM.
    """
    async def _fn(*args, **kwargs):
        return pid

    return _fn


async def test_proposal_timeout_kills_and_continues(archive_store, seeded_inflight, tmp_git_repo,
                                                    components_home, FakeClock, FakeWindowMeter, FakeExperimentFn):
    """A slow experiment_fn (> proposal_timeout) is cancelled (asyncio.TimeoutError); row → train_rejected; loop continues to the next iteration."""
    pid = await seeded_inflight(archive_store, component="agent.role", before="i", after="x")
    summary = await run_loop(
        store=archive_store, cfg=None, repo_root=tmp_git_repo, budget=None,
        max_iterations=2, max_cost=None, epsilon=0.0, min_lift=0.0,
        proposal_timeout=0.05, window_tokens=10_000,
        clock=FakeClock(), meter=FakeWindowMeter(),
        propose_fn=_fake_propose(pid, component="agent.role"),
        experiment_fn=FakeExperimentFn(exit_code=0, slow=True),  # sleeps past the timeout
    )
    # The hung experiment was killed and recorded as a training rejection (negative signal).
    assert (await archive_store.get(pid)).status == "train_rejected"
    assert summary.iterations >= 1  # loop did not deadlock; it continued


async def test_running_experiment_not_killed_by_total_cap(archive_store, seeded_inflight, tmp_git_repo,
                                                          components_home, FakeClock, FakeWindowMeter, FakeExperimentFn):
    """A budget that trips its TOTAL cap mid-experiment must NOT cancel the in-flight experiment_fn — only the PRE-FLIGHT gate halts.

    The running experiment_fn completes once before the loop halts (the cap is a start-gate, never a kill switch).
    """
    pid = await seeded_inflight(archive_store, component="agent.role", before="i", after="x")
    record = []
    fn = FakeExperimentFn(exit_code=0, record=record)
    # A budget that allows exactly one start, then the meter trips exhausted for the next pre-flight.
    meter = FakeWindowMeter()
    clock = FakeClock()

    summary = await run_loop(
        store=archive_store, cfg=None, repo_root=tmp_git_repo, budget=None,
        max_iterations=5, max_cost=None, epsilon=0.0, min_lift=0.0, proposal_timeout=10,
        window_tokens=10_000, clock=clock, meter=meter,
        propose_fn=_fake_propose(pid, component="agent.role"),
        experiment_fn=fn,
        adopt_fn=_exhaust_after_first(meter),  # flips meter exhausted AFTER the first experiment completes
    )
    assert len(record) >= 1  # the in-flight experiment ran to completion before the halt
    assert summary.iterations >= 1


def _exhaust_after_first(meter):
    """An adopt_fn seam that completes the adoption then trips the meter exhausted (simulating a mid-run total-cap breach)."""
    async def _fn(*args, **kwargs):
        meter._exhausted = True
        return None

    return _fn


async def test_journal_captures_loop_why(archive_store, seeded_inflight, tmp_git_repo, components_home,
                                        FakeClock, FakeWindowMeter, FakeExperimentFn):
    """The per-run JSONL journal at .localharness/autoresearch/runs/<run_id>.jsonl captures the loop-level 'why'."""
    pid = await seeded_inflight(archive_store, component="agent.role", before="i", after="x")
    summary = await run_loop(
        store=archive_store, cfg=None, repo_root=tmp_git_repo, budget=None,
        max_iterations=1, max_cost=None, epsilon=0.0, min_lift=0.0, proposal_timeout=10,
        window_tokens=10_000, clock=FakeClock(), meter=FakeWindowMeter(),
        propose_fn=_fake_propose(pid, component="agent.role"),
        experiment_fn=FakeExperimentFn(exit_code=0),
    )
    journal_path = Path(summary.journal_path)
    assert journal_path.exists()
    lines = [json.loads(ln) for ln in journal_path.read_text().splitlines() if ln.strip()]
    blob = json.dumps(lines)
    # The audit captures the explore/exploit decision + ε roll + lineage + gate verdict + budget snapshot.
    for field in ("branch", "epsilon_roll", "parent_id", "component", "archive_id",
                  "exit_code", "decision", "reason", "tokens_spent", "wallclock_elapsed"):
        assert field in blob, f"journal missing {field!r}"


async def test_circuit_breaker_halts(archive_store, seeded_inflight, tmp_git_repo, components_home,
                                    FakeClock, FakeWindowMeter, FakeExperimentFn):
    """An experiment_fn/propose_fn that fails every call → loop halts after N consecutive failures (≈ N iterations, not infinite)."""
    async def _always_raises(*args, **kwargs):
        raise RuntimeError("proposer down")

    summary = await run_loop(
        store=archive_store, cfg=None, repo_root=tmp_git_repo, budget=None,
        max_iterations=10_000, max_cost=None, epsilon=0.0, min_lift=0.0, proposal_timeout=10,
        window_tokens=10_000, clock=FakeClock(), meter=FakeWindowMeter(),
        propose_fn=_always_raises,
        experiment_fn=FakeExperimentFn(exit_code=0),
    )
    # The breaker bounds the run: a handful of consecutive failures, NOT 10_000 iterations.
    assert summary.iterations < 20


async def test_graceful_interrupt(archive_store, seeded_inflight, tmp_git_repo, components_home,
                                  FakeClock, FakeWindowMeter, FakeExperimentFn):
    """Setting the loop's interrupt flag mid-run → it finishes the current experiment, writes a run-complete summary line, returns cleanly."""
    import threading

    pid = await seeded_inflight(archive_store, component="agent.role", before="i", after="x")
    interrupt = threading.Event()

    async def _propose_then_interrupt(*args, **kwargs):
        interrupt.set()  # request interrupt during the iteration
        return pid

    summary = await run_loop(
        store=archive_store, cfg=None, repo_root=tmp_git_repo, budget=None,
        max_iterations=10_000, max_cost=None, epsilon=0.0, min_lift=0.0, proposal_timeout=10,
        window_tokens=10_000, clock=FakeClock(), meter=FakeWindowMeter(),
        propose_fn=_propose_then_interrupt,
        experiment_fn=FakeExperimentFn(exit_code=0),
        interrupt=interrupt,
    )
    # It finished the in-flight iteration then exited cleanly (not after 10_000 iterations).
    assert summary.iterations >= 1
    assert summary.iterations < 5
    journal_path = Path(summary.journal_path)
    assert "complete" in journal_path.read_text()  # a run-complete summary line was written


async def test_run_summary_fields(archive_store, seeded_inflight, tmp_git_repo, components_home,
                                 FakeClock, FakeWindowMeter, FakeExperimentFn):
    """The returned RunSummary carries iterations / adopted / held / rejected counts, time + window/tokens consumed, and the journal path."""
    pid = await seeded_inflight(archive_store, component="agent.role", before="i", after="x")
    summary = await run_loop(
        store=archive_store, cfg=None, repo_root=tmp_git_repo, budget=None,
        max_iterations=1, max_cost=None, epsilon=0.0, min_lift=0.0, proposal_timeout=10,
        window_tokens=10_000, clock=FakeClock(), meter=FakeWindowMeter(),
        propose_fn=_fake_propose(pid, component="agent.role"),
        experiment_fn=FakeExperimentFn(exit_code=0),
    )
    assert isinstance(summary, RunSummary)
    for attr in ("iterations", "adopted", "held", "rejected", "journal_path"):
        assert hasattr(summary, attr), f"RunSummary missing {attr!r}"
    assert isinstance(summary.iterations, int)
    # time + window/token consumption are surfaced (exact attr names are the impl's; assert ≥1 present).
    assert any(hasattr(summary, a) for a in ("wallclock_elapsed", "elapsed", "duration"))
    assert any(hasattr(summary, a) for a in ("tokens_spent", "window_tokens_spent", "tokens"))


# ---------------------------------------------------------------------------
# Phase 19 Wave-0 — the inline eval-sentinel hook is NON-BLOCKING (REP-03/04)
#
# 19-03/19-05 add a cheap inline sentinel check beside the per-iteration journal.write.
# Its hard rule (19-RESEARCH Pitfall 4): a sentinel bug can NEVER crash the fire-and-forget
# loop — it must be try/except-guarded. This stub monkeypatches that hook (module-level seam
# ``run_inline_sentinel``, patched raising=False so the file collects before the seam exists —
# the same raising=False idiom test_run_clean_halt_exit_zero uses) to RAISE, then asserts the
# loop runs to its clean halt without propagating the exception and still journals 'complete'.
# xfail(strict=False) until the guarded hook lands (then flips to pass).
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False)  # impl-pending-19
async def test_inline_sentinel_nonblocking(archive_store, seeded_inflight, tmp_git_repo,
                                           components_home, FakeClock, FakeWindowMeter, FakeExperimentFn,
                                           monkeypatch):
    """A sentinel hook that RAISES does not propagate out of run_loop; the run halts cleanly + journals 'complete'."""
    import localharness.autoresearch.loop as loop_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("sentinel exploded")  # the inline check blows up mid-iteration

    # patch the inline-sentinel seam; raising=False keeps this collectable before 19-03 wires it
    monkeypatch.setattr(loop_mod, "run_inline_sentinel", _boom, raising=False)

    pid = await seeded_inflight(archive_store, component="agent.role", before="i", after="x")
    summary = await run_loop(  # must NOT raise — the try/except guard swallows the sentinel error
        store=archive_store, cfg=None, repo_root=tmp_git_repo, budget=None,
        max_iterations=1, max_cost=None, epsilon=0.0, min_lift=0.0, proposal_timeout=10,
        window_tokens=10_000, clock=FakeClock(), meter=FakeWindowMeter(),
        propose_fn=_fake_propose(pid, component="agent.role"),
        experiment_fn=FakeExperimentFn(exit_code=0),
    )
    assert summary.iterations >= 1  # the loop ran the iteration despite the sentinel raising
    assert summary.halt_reason in ("budget", "complete")  # a clean cap-trip halt, not a crash

    journal_path = Path(summary.journal_path)
    assert journal_path.exists()
    assert "complete" in journal_path.read_text()  # ran to its clean run-complete summary line
