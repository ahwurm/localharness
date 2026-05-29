"""Phase 18 Wave-0 scaffold — AUTO-01..04 behaviors as xfail(strict=False) stubs; flipped green as 18-02..18-05 land.

The autoresearch orchestrator wires four net-new pieces: a ParentSampler (ε-greedy
over the GEPA per-fixture front), a BudgetController (token-window + wallclock pre-flight
gate + per-proposal timeout), an adoption mechanism (a git-committed config-overlay write),
and the loop driver (sample → propose → run_experiment → interpret-exit → adopt/hold →
journal). Every test name below is BINDING: 18-RESEARCH §"Phase Requirements → Test Map"
names each one verbatim, and plans 18-02..18-06 resolve their <automated> verify command
to a test that already exists here. Each stub asserts REAL behavior (never placeholder
data), so it is meaningful the moment the impl lands and flips it RED→GREEN.

Hermetic injection mirrors test_experiment.py: fakes for the clock, the token meter, the
RNG, propose_fn, experiment_fn, and adopt_fn keep every test LLM-free / bench-free and (for
the non-adoption tests) worktree-free. The guarded module-top import lets this file COLLECT
before sampler/budget/adoption/loop exist (17-01 precedent: pytest.skip allow_module_level).
"""
import json
import random

import pytest

try:
    from localharness.autoresearch.sampler import ParentSampler, BASELINE_ROOT
    from localharness.autoresearch.budget import BudgetController, WindowMeter
except ImportError:
    pytest.skip(
        "autoresearch.sampler/budget not yet implemented (18-02/18-03)",
        allow_module_level=True,
    )


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


@pytest.mark.xfail(strict=False)  # impl pending 18-02/18-03
async def test_epsilon_branch_selection(archive_store, seeded_archive, monkeypatch):
    """rng.random() < ε takes the EXPLORE branch; >= ε takes EXPLOIT (front non-empty)."""
    await seeded_archive(
        archive_store,
        [dict(id="front-1", status="promoted", train_scores_per_fixture={"f1": 0.9, "f2": 0.8})],
    )

    explore = ParentSampler(archive_store, epsilon=0.2, rng=FixedRandom(0.05))
    branches = []
    monkeypatch.setattr(explore, "_explore", lambda *a, **k: branches.append("explore") or BASELINE_ROOT, raising=False)
    monkeypatch.setattr(explore, "_exploit", lambda *a, **k: branches.append("exploit") or BASELINE_ROOT, raising=False)
    await explore.sample()
    assert branches == ["explore"]  # 0.05 < 0.2 → explore

    exploit = ParentSampler(archive_store, epsilon=0.2, rng=FixedRandom(0.5))
    branches2 = []
    monkeypatch.setattr(exploit, "_explore", lambda *a, **k: branches2.append("explore") or BASELINE_ROOT, raising=False)
    monkeypatch.setattr(exploit, "_exploit", lambda *a, **k: branches2.append("exploit") or BASELINE_ROOT, raising=False)
    await exploit.sample()
    assert branches2 == ["exploit"]  # 0.5 >= 0.2 → exploit


@pytest.mark.xfail(strict=False)  # impl pending 18-02/18-03
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


@pytest.mark.xfail(strict=False)  # impl pending 18-02/18-03
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


@pytest.mark.xfail(strict=False)  # impl pending 18-02/18-03
async def test_cold_start_always_explore_root(archive_store):
    """Empty archive (no front) → sample() returns BASELINE_ROOT regardless of ε (0.0 and 1.0)."""
    cold_lo = ParentSampler(archive_store, epsilon=0.0, rng=random.Random(0))
    cold_hi = ParentSampler(archive_store, epsilon=1.0, rng=random.Random(0))
    assert (await cold_lo.sample()) is BASELINE_ROOT
    assert (await cold_hi.sample()) is BASELINE_ROOT


@pytest.mark.xfail(strict=False)  # impl pending 18-02/18-03
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


@pytest.mark.xfail(strict=False)  # impl pending 18-02/18-03
async def test_budget_halts_on_window_exhausted(FakeClock, FakeWindowMeter):
    """meter.window_exhausted() True → can_start_iteration() returns False."""
    clock = FakeClock()
    meter = FakeWindowMeter()
    ctl = BudgetController(budget_seconds=10_000, max_iterations=100, max_cost=None, meter=meter, clock=clock)
    assert ctl.can_start_iteration() is True
    meter._exhausted = True
    assert ctl.can_start_iteration() is False


@pytest.mark.xfail(strict=False)  # impl pending 18-02/18-03
async def test_budget_halts_on_wallclock(FakeClock, FakeWindowMeter):
    """Fake clock advanced past budget_seconds → can_start_iteration() returns False."""
    clock = FakeClock(0.0)
    meter = FakeWindowMeter()
    ctl = BudgetController(budget_seconds=100, max_iterations=1000, max_cost=None, meter=meter, clock=clock)
    assert ctl.can_start_iteration() is True
    clock.advance(101)  # past the 100s wallclock budget
    assert ctl.can_start_iteration() is False


@pytest.mark.xfail(strict=False)  # impl pending 18-02/18-03
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


@pytest.mark.xfail(strict=False)  # impl pending 18-02/18-03
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


@pytest.mark.xfail(strict=False)  # impl pending 18-02/18-03
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
