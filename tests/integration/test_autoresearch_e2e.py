"""Phase 18 integration — ONE real git repo asserting compound-baseline advance after an adoption; fake experiment_fn (no Ollama/bench).

This is the single end-to-end seal for AUTO-04's compounding property: when the loop adopts
a promoted mutation, the live overlay is git-committed in the MAIN repo and becomes the
NEXT iteration's baseline. We drive run_loop against a real `git init` repo (tmp_git_repo)
+ a real ArchiveStore, injecting an experiment_fn that writes the gate's verdict (the
realistic run_experiment shape: update_verdict(status="promoted", train_score=...) THEN return
exit 0) and a propose_fn returning the seeded in_flight row — so there is zero Ollama / bench /
worktree-bench dependency. The REAL adopt() runs (NOT injected) so the commit actually lands.
The asserts are physical: `git log` gained a commit and overrides.yaml reflects the adopted value.

The loop (18-05) is shipped, so `run_loop` imports unconditionally (the Wave-0 guard is gone).
"""
import json

from localharness.autoresearch.loop import run_loop


def _git_log_lines(repo):
    import subprocess

    out = subprocess.run(
        ["git", "-C", str(repo), "log", "--oneline"], capture_output=True, text=True
    )
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


async def test_compound_baseline_advances(archive_store, seeded_inflight, tmp_git_repo,
                                          components_home, FakeClock, FakeWindowMeter):
    """After ONE adoption, the MAIN repo gained a commit and overrides.yaml reflects the adopted value (the next baseline).

    Uses a REAL registry path (``agent.role``) so adopt()'s seal + ``AgentConfig`` re-validation
    pass, and the REAL adopt() (no ``adopt_fn`` injection) so the commit physically lands. The
    injected experiment_fn writes the verdict THEN returns the exit code, mirroring the real
    run_experiment contract (the gate writes train_score/status, then surfaces the gate code).
    """
    import subprocess

    import yaml

    pid = await seeded_inflight(
        archive_store, component="agent.role", before="initial", after="evolved"
    )

    async def _propose(*args, **kwargs):
        return pid  # the loop reads this in_flight row, runs the gate, then adopts on exit 0

    async def _experiment(proposal_id, **kwargs):
        # Realistic gate shape: write the verdict (promoted + a train_score the loop reads as
        # lift) THEN return the exit code. With min_lift=None any clean win auto-adopts.
        await archive_store.update_verdict(
            proposal_id, status="promoted", train_score=0.9,
            train_scores_per_fixture={"f1": 0.9}, holdout_score=0.8, p_value=0.01,
        )
        return 0  # clean PROMOTE -> adopt

    before_commits = len(_git_log_lines(tmp_git_repo))
    summary = await run_loop(
        store=archive_store, cfg=None, repo_root=tmp_git_repo, budget=None,
        max_iterations=1, max_cost=None, epsilon=0.0, min_lift=None, proposal_timeout=10,
        window_tokens=10_000, clock=FakeClock(), meter=FakeWindowMeter(),
        propose_fn=_propose,
        experiment_fn=_experiment,  # writes verdict then returns 0 (the real run_experiment shape)
    )

    # The adoption commit landed in the MAIN repo (the next iteration's baseline advances).
    assert len(_git_log_lines(tmp_git_repo)) == before_commits + 1
    log = subprocess.run(
        ["git", "-C", str(tmp_git_repo), "log", "-1", "--pretty=%s"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "autoresearch: adopt agent.role" in log  # the adoption commit message

    overrides = tmp_git_repo / ".localharness" / "overrides.yaml"
    data = yaml.safe_load(overrides.read_text())
    assert data["agent"]["role"] == "evolved"  # the committed change is the new baseline

    # The adopted archive row reflects the adopted verdict.
    assert (await archive_store.get(pid)).status == "adopted"
    assert summary.adopted == 1

    # And it is in repo_root HEAD, NOT a throwaway lh-exp-* worktree.
    wt = subprocess.run(
        ["git", "-C", str(tmp_git_repo), "worktree", "list"], capture_output=True, text=True
    )
    assert "lh-exp-" not in wt.stdout

    # The per-run JSONL journal exists with a 'complete' line.
    journal_path = tmp_git_repo / ".localharness" / "autoresearch" / "runs" / f"{summary.run_id}.jsonl"
    assert journal_path.exists()
    assert "complete" in journal_path.read_text()
