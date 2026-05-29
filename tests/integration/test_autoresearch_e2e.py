"""Phase 18 integration — ONE real git repo asserting compound-baseline advance after an adoption; fake experiment_fn (no Ollama/bench).

This is the single end-to-end seal for AUTO-04's compounding property: when the loop adopts
a promoted mutation, the live overlay is git-committed in the MAIN repo and becomes the
NEXT iteration's baseline. We drive run_loop against a real `git init` repo (tmp_git_repo)
+ a real ArchiveStore, injecting a FakeExperimentFn that returns a clean PROMOTE (exit 0) and
a propose_fn returning a clean proposal — so there is zero Ollama / bench / worktree-bench
dependency. The asserts are physical: `git log` gained a commit and overrides.yaml reflects
the adopted value. Guarded module-top import → skip until run_loop lands (18-05/18-06).
"""
import json

import pytest

try:
    from localharness.autoresearch.loop import run_loop  # noqa: F401
except ImportError:
    pytest.skip(
        "autoresearch.loop not yet implemented (18-05/18-06)",
        allow_module_level=True,
    )


def _git_log_lines(repo):
    import subprocess

    out = subprocess.run(
        ["git", "-C", str(repo), "log", "--oneline"], capture_output=True, text=True
    )
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


@pytest.mark.xfail(strict=False)  # impl pending 18-06
async def test_compound_baseline_advances(archive_store, seeded_inflight, tmp_git_repo,
                                          components_home, FakeClock, FakeWindowMeter, FakeExperimentFn):
    """After ONE adoption, the MAIN repo gained a commit and overrides.yaml reflects the adopted value (the next baseline)."""
    import subprocess

    import yaml

    pid = await seeded_inflight(
        archive_store, component="agent.role", before="initial", after="evolved"
    )

    async def _propose(*args, **kwargs):
        return pid  # the loop reads this in_flight row, runs the gate, then adopts on exit 0

    before_commits = len(_git_log_lines(tmp_git_repo))
    await run_loop(
        store=archive_store, cfg=None, repo_root=tmp_git_repo, budget=None,
        max_iterations=1, max_cost=None, epsilon=0.0, min_lift=0.0, proposal_timeout=10,
        window_tokens=10_000, clock=FakeClock(), meter=FakeWindowMeter(),
        propose_fn=_propose,
        experiment_fn=FakeExperimentFn(exit_code=0),  # clean PROMOTE → adopt
    )

    # The adoption commit landed in the MAIN repo (the next iteration's baseline advances).
    assert len(_git_log_lines(tmp_git_repo)) == before_commits + 1
    overrides = tmp_git_repo / ".localharness" / "overrides.yaml"
    data = yaml.safe_load(overrides.read_text())
    assert data["agent"]["role"] == "evolved"  # the committed change is the new baseline

    # And it is in repo_root HEAD, NOT a throwaway lh-exp-* worktree.
    wt = subprocess.run(
        ["git", "-C", str(tmp_git_repo), "worktree", "list"], capture_output=True, text=True
    )
    assert "lh-exp-" not in wt.stdout
