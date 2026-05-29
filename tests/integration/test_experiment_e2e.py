"""Phase 17 — ONE real git-worktree lifecycle test. No Ollama (no bench LLM here; this
exercises the worktree context manager directly). xfail until experiment.py lands in 17-03.

SC1: `experiment run` must isolate all work in a throwaway git worktree so the main
checkout's working tree, HEAD, and branch are untouched, and no stale `lh-exp-*` worktree
is left behind. Guarded module-top import skips cleanly until 17-03.
"""
from __future__ import annotations

import subprocess

import pytest

try:
    from localharness.autoresearch import experiment as exp  # noqa: F401
except ImportError:
    pytest.skip("experiment.py not yet implemented (17-03)", allow_module_level=True)


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


def _make_repo(path):
    """git init a repo with a committed bench corpus (train + holdout fixtures)."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "bench").mkdir()
    (path / "bench" / "bench.yaml").write_text(
        "corpus_path: bench/scenarios\nresults_path: bench/results\n", encoding="utf-8"
    )
    for sl in ("train", "holdout"):
        d = path / "bench" / "scenarios" / sl
        d.mkdir(parents=True)
        (d / f"{sl}_01.yaml").write_text(f"name: {sl}_01\nslice: {sl}\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init corpus"], cwd=path, check=True)
    return path


@pytest.fixture
def tmp_git_repo(tmp_path):
    return _make_repo(tmp_path / "repo")


@pytest.mark.xfail(strict=False, reason="experiment.py lands in 17-03")
def test_worktree_no_main_contamination(tmp_git_repo):
    """A real experiment_worktree run leaves the main checkout's tree/HEAD/branch unchanged + no stale worktree."""
    repo = tmp_git_repo
    head_before = _git(repo, "rev-parse", "HEAD").strip()
    status_before = _git(repo, "status", "--porcelain")
    branch_before = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()

    with exp.experiment_worktree(repo_root=repo) as wt:
        # Write the overlay + a fake results dir INSIDE the worktree (committed corpus is present).
        exp.write_experiment_overlay(wt, "agent.role", "EXPERIMENTAL ROLE")
        (wt / "bench" / "results").mkdir(parents=True, exist_ok=True)
        (wt / "bench" / "results" / "run.json").write_text("{}", encoding="utf-8")
        assert (wt / "bench" / "scenarios" / "train" / "train_01.yaml").exists()  # corpus carried

    # Main checkout is pristine after the worktree is torn down.
    # Invariants (in `git` CLI terms): `git status --porcelain` unchanged, HEAD + branch
    # unchanged, and `git worktree list` shows no stale lh-exp-* entry.
    assert _git(repo, "rev-parse", "HEAD").strip() == head_before
    assert _git(repo, "status", "--porcelain") == status_before  # git status --porcelain
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip() == branch_before
    assert "lh-exp-" not in _git(repo, "worktree", "list")  # git worktree list — no stale worktree


@pytest.mark.xfail(strict=False, reason="experiment.py lands in 17-03")
def test_keep_retains_worktree(tmp_git_repo):
    """`keep=True` retains the worktree after exit; it appears in `git worktree list`."""
    repo = tmp_git_repo
    with exp.experiment_worktree(repo_root=repo, keep=True) as wt:
        kept = wt
    assert kept.exists()
    assert "lh-exp-" in _git(repo, "worktree", "list")
    # Leave no residue for other tests.
    subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(kept)], check=False)
