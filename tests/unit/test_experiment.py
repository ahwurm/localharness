"""Phase 17 Wave 0 RED stubs — experiment gate logic (EXP-01..05 + promotion/seal/audit).

xfail(strict=False) until experiment.py lands in 17-03. The module-top import is guarded
so collection never breaks before the module exists; once it ships the xfail markers govern.

These stubs are the binding RED→GREEN contract: 17-03's `experiment.py` must satisfy every
assertion here. The bench is injected via a fake `run_slice` callable
(``run_slice(worktree, *, slice, with_overlay) -> dict[str, float]``) so the gate logic is
exercised LLM-free and worktree-free where possible.
"""
import json
import subprocess

import pytest

try:
    from localharness.autoresearch import experiment as exp  # noqa: F401
except ImportError:
    pytest.skip("experiment.py not yet implemented (17-03)", allow_module_level=True)


# ---------------------------------------------------------------------------
# Test helpers (fakes + a tmp git repo)
# ---------------------------------------------------------------------------


class FakeRunSlice:
    """Records every run_slice call + returns caller-configured success maps.

    Contract the 17-03 runner must honor:
      run_slice(worktree, *, slice: str, with_overlay: bool) -> dict[str, float]
    ``with_overlay=False`` is the baseline arm; ``True`` is the proposal arm.
    """

    def __init__(self, *, train_base=None, train_head=None, holdout_base=None, holdout_head=None):
        self._maps = {
            ("train", False): train_base or {},
            ("train", True): train_head or {},
            ("holdout", False): holdout_base or {},
            ("holdout", True): holdout_head or {},
        }
        self.calls = []  # list of (slice, with_overlay)

    def __call__(self, worktree, *, slice, with_overlay):
        self.calls.append((slice, with_overlay))
        return dict(self._maps[(slice, with_overlay)])

    @property
    def slices_requested(self):
        return {c[0] for c in self.calls}


def _make_git_repo(path):
    """git init a tmp dir with one trivial commit. Returns the repo Path."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "seed.txt").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)
    return path


@pytest.fixture
def tmp_git_repo(tmp_path):
    return _make_git_repo(tmp_path / "repo")


# Equal arms (no improvement): train gate must REJECT.
_EQUAL_TRAIN = {"01_a": 0.5, "02_b": 0.5, "03_c": 0.5}
# Clear improvement: baseline ~0.2, proposal ~0.9 per train fixture.
_IMPROVED_TRAIN_BASE = {"01_a": 0.2, "02_b": 0.2, "03_c": 0.2}
_IMPROVED_TRAIN_HEAD = {"01_a": 0.9, "02_b": 0.9, "03_c": 0.9}
# Non-regressing holdout: proposal == baseline (not significantly worse) → promote.
_HOLDOUT_BASE = {"h1": 0.6, "h2": 0.6, "h3": 0.6}
_HOLDOUT_HEAD_OK = {"h1": 0.6, "h2": 0.6, "h3": 0.6}


# ---------------------------------------------------------------------------
# EXP-01 — worktree lifecycle + overlay materialization
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="experiment.py lands in 17-03")
def test_worktree_cleanup(tmp_git_repo):
    """experiment_worktree yields a real path that exists inside the with and is gone after (keep=False)."""
    with exp.experiment_worktree(repo_root=tmp_git_repo) as wt:
        assert wt.exists()
        captured = wt
    assert not captured.exists()


@pytest.mark.xfail(strict=False, reason="experiment.py lands in 17-03")
def test_overlay_written_into_worktree(tmp_git_repo):
    """write_experiment_overlay materializes the after-value into the worktree at the dot-path."""
    import yaml

    with exp.experiment_worktree(repo_root=tmp_git_repo) as wt:
        overlay_path = exp.write_experiment_overlay(wt, "agent.role", "NEW ROLE")
        assert overlay_path.exists()
        data = yaml.safe_load(overlay_path.read_text())
        # round-trips the after value at the dot-path agent.role
        assert data["agent"]["role"] == "NEW ROLE"


# ---------------------------------------------------------------------------
# EXP-02 — structural refusals (anti-reward-hacking seal, exit >=4)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="experiment.py lands in 17-03")
async def test_refuses_multi_component_diff(archive_store, seeded_inflight):
    """A diff resolving to >1 component path → EXIT_REFUSE_MULTI_COMPONENT (>=4); bench never runs."""
    multi = json.dumps({
        "before": {"agent.role": "a", "tools.bash.description": "b"},
        "after": {"agent.role": "x", "tools.bash.description": "y"},
    })
    pid = await seeded_inflight(archive_store, component="agent.role", diff=multi)
    fake = FakeRunSlice(train_base=_EQUAL_TRAIN, train_head=_EQUAL_TRAIN)
    code = await exp.run_experiment(pid, store=archive_store, run_slice=fake)
    assert code == exp.EXIT_REFUSE_MULTI_COMPONENT
    assert code >= 4
    assert fake.calls == []  # refused before any bench arm


@pytest.mark.xfail(strict=False, reason="experiment.py lands in 17-03")
async def test_refuses_offregistry_component(archive_store, seeded_inflight):
    """Off-registry / grader-targeting component → EXIT_REFUSE_OFFREGISTRY (>=4); bench never runs."""
    pid = await seeded_inflight(
        archive_store, component="bench.scenarios.holdout.01", before="a", after="b"
    )
    fake = FakeRunSlice(train_base=_EQUAL_TRAIN, train_head=_EQUAL_TRAIN)
    code = await exp.run_experiment(pid, store=archive_store, run_slice=fake)
    assert code == exp.EXIT_REFUSE_OFFREGISTRY
    assert code >= 4
    assert len(fake.calls) == 0  # seal refuses BEFORE running bench (no goalpost-moving)


# ---------------------------------------------------------------------------
# EXP-03 — two-stage train→holdout flow + Welch direction + per-fixture shape
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="experiment.py lands in 17-03")
async def test_holdout_skipped_on_train_reject(archive_store, seeded_inflight):
    """No train improvement → EXIT_REJECT_TRAIN (1) and holdout bench NEVER invoked."""
    pid = await seeded_inflight(archive_store, component="agent.role")
    fake = FakeRunSlice(train_base=_EQUAL_TRAIN, train_head=_EQUAL_TRAIN,
                        holdout_base=_HOLDOUT_BASE, holdout_head=_HOLDOUT_HEAD_OK)
    code = await exp.run_experiment(pid, store=archive_store, run_slice=fake)
    assert code == exp.EXIT_REJECT_TRAIN
    assert "train" in fake.slices_requested
    assert "holdout" not in fake.slices_requested  # conditional holdout never reached


@pytest.mark.xfail(strict=False, reason="experiment.py lands in 17-03")
async def test_welch_improvement_direction(archive_store, seeded_inflight):
    """Clear train improvement + non-regressing holdout → does NOT reject-train (promotes)."""
    pid = await seeded_inflight(archive_store, component="agent.role")
    fake = FakeRunSlice(train_base=_IMPROVED_TRAIN_BASE, train_head=_IMPROVED_TRAIN_HEAD,
                        holdout_base=_HOLDOUT_BASE, holdout_head=_HOLDOUT_HEAD_OK)
    code = await exp.run_experiment(pid, store=archive_store, run_slice=fake)
    assert code != exp.EXIT_REJECT_TRAIN  # direction correct: improvement is not a regression
    assert code == exp.EXIT_PROMOTE


@pytest.mark.xfail(strict=False, reason="experiment.py lands in 17-03")
async def test_per_fixture_vector_shape(archive_store, seeded_inflight):
    """train_scores_per_fixture has one key per TRAIN fixture (not rep count) — train names only."""
    pid = await seeded_inflight(archive_store, component="agent.role")
    fake = FakeRunSlice(train_base=_IMPROVED_TRAIN_BASE, train_head=_IMPROVED_TRAIN_HEAD,
                        holdout_base=_HOLDOUT_BASE, holdout_head=_HOLDOUT_HEAD_OK)
    await exp.run_experiment(pid, store=archive_store, run_slice=fake)
    row = await archive_store.get(pid)
    assert len(row.train_scores_per_fixture) == len(_IMPROVED_TRAIN_HEAD)  # 3 fixtures → 3 keys
    assert set(row.train_scores_per_fixture) == set(_IMPROVED_TRAIN_HEAD)  # TRAIN scenario names


# ---------------------------------------------------------------------------
# EXP-04 — Bonferroni multi-trial α scaling + holdout non-regression
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="experiment.py lands in 17-03")
async def test_bonferroni_alpha_scaling(archive_store, seeded_inflight):
    """A borderline-worse holdout: rejected at α=0.05 (trials=1) but passes at α=0.0125 (trials=4).

    Calibration: holdout proposal arm is mildly lower than baseline so the one-sided
    non-regression p sits BETWEEN 0.0125 and 0.05 — i.e. significant at the looser α only.
    Shrinking α via --trials must flip reject-holdout → promote.
    """
    # baseline mean ~0.68, proposal mean ~0.56 per fixture: a small drop with realistic
    # per-fixture spread so the one-sided non-regression p lands ~0.032 (between 0.0125
    # and 0.05). Near-constant vectors yield catastrophic-cancellation t-stats (p≈1e-11),
    # which cannot satisfy this test's own "p between 0.0125 and 0.05" premise.
    hb = {f"h{i}": v for i, v in enumerate([0.75, 0.60, 0.80, 0.55, 0.70, 0.65, 0.85, 0.50])}
    hh = {f"h{i}": v for i, v in enumerate([0.58, 0.52, 0.62, 0.48, 0.66, 0.44, 0.72, 0.48])}
    common = dict(train_base=_IMPROVED_TRAIN_BASE, train_head=_IMPROVED_TRAIN_HEAD,
                  holdout_base=hb, holdout_head=hh)

    pid1 = await seeded_inflight(archive_store, component="agent.role")
    code1 = await exp.run_experiment(pid1, trials=1, store=archive_store, run_slice=FakeRunSlice(**common))
    assert code1 == exp.EXIT_REJECT_HOLDOUT  # significant at α=0.05

    pid4 = await seeded_inflight(archive_store, component="agent.role")
    code4 = await exp.run_experiment(pid4, trials=4, store=archive_store, run_slice=FakeRunSlice(**common))
    assert code4 == exp.EXIT_PROMOTE  # NOT significant at α=0.05/4=0.0125 → non-regression passes


@pytest.mark.xfail(strict=False, reason="experiment.py lands in 17-03")
async def test_holdout_nonregression_promotes(archive_store, seeded_inflight):
    """Train improves + holdout not significantly worse → EXIT_PROMOTE (0)."""
    pid = await seeded_inflight(archive_store, component="agent.role")
    fake = FakeRunSlice(train_base=_IMPROVED_TRAIN_BASE, train_head=_IMPROVED_TRAIN_HEAD,
                        holdout_base=_HOLDOUT_BASE, holdout_head=_HOLDOUT_HEAD_OK)
    code = await exp.run_experiment(pid, store=archive_store, run_slice=fake)
    assert code == exp.EXIT_PROMOTE


# ---------------------------------------------------------------------------
# Promotion semantics — archive write-back, NOT live config
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="experiment.py lands in 17-03")
async def test_promote_writes_archive_not_live_config(archive_store, seeded_inflight, components_home):
    """Promote flips status + fills scores in the archive; does NOT write the user overlay."""
    pid = await seeded_inflight(archive_store, component="agent.role")
    fake = FakeRunSlice(train_base=_IMPROVED_TRAIN_BASE, train_head=_IMPROVED_TRAIN_HEAD,
                        holdout_base=_HOLDOUT_BASE, holdout_head=_HOLDOUT_HEAD_OK)
    await exp.run_experiment(pid, store=archive_store, run_slice=fake)
    row = await archive_store.get(pid)
    assert row.status == "promoted"
    assert row.train_score is not None
    assert row.holdout_score is not None
    assert row.p_value is not None
    assert row.cost is not None
    # Promotion is archive-only; live adoption is Phase 18. No user overlay written.
    assert not (components_home / "overrides.yaml").exists()


@pytest.mark.xfail(strict=False, reason="experiment.py lands in 17-03")
async def test_reject_status_writeback(archive_store, seeded_inflight):
    """Train-reject writes status 'train_rejected'; holdout-reject writes 'holdout_rejected'."""
    pid_tr = await seeded_inflight(archive_store, component="agent.role")
    await exp.run_experiment(
        pid_tr, store=archive_store,
        run_slice=FakeRunSlice(train_base=_EQUAL_TRAIN, train_head=_EQUAL_TRAIN),
    )
    assert (await archive_store.get(pid_tr)).status == "train_rejected"

    # holdout reject: train improves but holdout clearly regresses
    hb = {f"h{i}": v for i, v in enumerate([0.8, 0.9, 0.85, 0.8, 0.9])}
    hh = {f"h{i}": v for i, v in enumerate([0.2, 0.3, 0.2, 0.25, 0.3])}
    pid_hr = await seeded_inflight(archive_store, component="agent.role")
    await exp.run_experiment(
        pid_hr, store=archive_store,
        run_slice=FakeRunSlice(train_base=_IMPROVED_TRAIN_BASE, train_head=_IMPROVED_TRAIN_HEAD,
                               holdout_base=hb, holdout_head=hh),
    )
    assert (await archive_store.get(pid_hr)).status == "holdout_rejected"


# ---------------------------------------------------------------------------
# Audit — ComponentMutated(layer="experiment", actor="experiment")
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="experiment.py lands in 17-03")
async def test_emits_component_mutated_experiment(tmp_path, bus, seeded_inflight):
    """Running the proposal arm publishes ComponentMutated(layer/actor='experiment', actor_detail=pid)."""
    from localharness.autoresearch.archive import ArchiveStore
    from localharness.core.events import ComponentMutated

    received = []

    async def _handler(event):
        received.append(event)

    bus.subscribe(ComponentMutated, _handler)
    store = ArchiveStore(tmp_path / ".localharness" / "archive.db", bus=bus)
    await store.open()
    pid = await seeded_inflight(store, component="agent.role")
    fake = FakeRunSlice(train_base=_IMPROVED_TRAIN_BASE, train_head=_IMPROVED_TRAIN_HEAD,
                        holdout_base=_HOLDOUT_BASE, holdout_head=_HOLDOUT_HEAD_OK)
    await exp.run_experiment(pid, store=store, run_slice=fake, bus=bus)
    mutated = [e for e in received
               if e.layer == "experiment" and e.actor == "experiment" and e.actor_detail == pid]
    assert len(mutated) >= 1
    await store.close()


# ---------------------------------------------------------------------------
# Seal — holdout per-fixture map NEVER leaks into train_scores_per_fixture
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="experiment.py lands in 17-03")
async def test_holdout_not_in_train_blob(archive_store, seeded_inflight):
    """After a promote, train_scores_per_fixture holds ONLY train names — no holdout names (pareto-blind)."""
    pid = await seeded_inflight(archive_store, component="agent.role")
    fake = FakeRunSlice(train_base=_IMPROVED_TRAIN_BASE, train_head=_IMPROVED_TRAIN_HEAD,
                        holdout_base=_HOLDOUT_BASE, holdout_head=_HOLDOUT_HEAD_OK)
    await exp.run_experiment(pid, store=archive_store, run_slice=fake)
    blob = (await archive_store.get(pid)).train_scores_per_fixture
    assert set(blob) == set(_IMPROVED_TRAIN_HEAD)        # train scenario names only
    assert set(blob).isdisjoint(set(_HOLDOUT_HEAD_OK))   # zero holdout names in the train blob
