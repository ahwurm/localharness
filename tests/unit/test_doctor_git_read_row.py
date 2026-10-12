"""Doctor's `git_read` row (GITREAD-06, D-13, D-30, D-31, D-59, D-66, D-71).

`start` prints nothing when `git_read` is not registered, so the reason lives in `doctor`: the third
independent row beside the posture and window rows. It computes the boundary `start` computes (the
same two helpers, narrowed by the root agent's `permissions.workspace_root`), then:

- no boundary, or no `.git` entry from the boundary upward (a lexical walk, no git run) -> no row;
- otherwise the registration's own `git_read_pin_failure(boundary)`: a failure is an info row in the
  fixed wording of D-31, success is `git_read: available`. The row is informational: it is never a
  doctor failure and never carries git's stderr.

Every expected sentence below is a literal written out here, never rebuilt from the module's
templates. Every layout is a real repository built by `tests/unit/git_support.py` under a scratch
home; nothing touches `~/.localharness` or the project's own `.git`. The configured `base_url` is
port 9 (discard), refused instantly on loopback, so doctor exits 1 on the endpoint probe in every
test here and the exit code is only ever compared BETWEEN runs.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.tools.builtin import git_read_tool
from localharness.tools.builtin.git_read_tool import GitReadPinFailure
from tests.unit.git_support import git, make_repo, requires_git

runner = CliRunner()

_UNREACHABLE_BASE_URL = "http://127.0.0.1:9/v1"
_CONFIG = {
    "version": "1",
    "provider": {
        "provider_type": "vllm",
        "base_url": _UNREACHABLE_BASE_URL,
        "default_model": "test-model",
        "available_models": ["test-model"],
    },
}


def _machine(tmp_path: Path, monkeypatch, fake_home) -> SimpleNamespace:
    """A fake `$HOME` holding a valid machine config (the global layer); cwd is left to the test."""
    home = tmp_path / "home"
    global_dir = home / ".localharness"
    global_dir.mkdir(parents=True)
    (global_dir / "config.yaml").write_text(yaml.dump(_CONFIG, sort_keys=False), encoding="utf-8")
    fake_home(home)
    monkeypatch.setenv("COLUMNS", "400")
    return SimpleNamespace(home=home, global_dir=global_dir)


def _run():
    result = runner.invoke(app, ["doctor"])
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"doctor raised {result.exception!r}\n{result.stdout}"
    )
    return result


def _git_read_lines(out: str) -> list[str]:
    return [ln.strip() for ln in out.splitlines() if "git_read" in ln]


def _row(out: str) -> str:
    lines = _git_read_lines(out)
    assert len(lines) == 1, out
    return lines[0]


# ------------------------------------------------------------------------------- Task 1


@requires_git
def test_a_healthy_repository_prints_the_available_row_once(tmp_path, monkeypatch, fake_home):
    m = _machine(tmp_path, monkeypatch, fake_home)
    repo = make_repo(m.home / "proj")
    monkeypatch.chdir(repo)

    out = _run().stdout

    assert out.count("git_read: available") == 1, out
    assert _row(out).endswith("git_read: available")


def test_no_repository_prints_no_row_and_runs_no_git(tmp_path, monkeypatch, fake_home):
    m = _machine(tmp_path, monkeypatch, fake_home)
    plain = m.home / "plain" / "deep"
    plain.mkdir(parents=True)
    monkeypatch.chdir(plain)
    git_calls: list[object] = []
    real_run = subprocess.run

    def _spy(argv, *a, **kw):
        if isinstance(argv, (list, tuple)) and argv and os.path.basename(str(argv[0])) == "git":
            git_calls.append(argv)
        return real_run(argv, *a, **kw)

    monkeypatch.setattr(subprocess, "run", _spy)

    out = _run().stdout

    assert _git_read_lines(out) == [], out
    assert git_calls == []


@requires_git
def test_a_boundary_of_none_prints_no_row(tmp_path, monkeypatch, fake_home):
    # The cwd IS the fake $HOME: the derived root is $HOME, so there is no boundary (critic 1) and
    # no row, even though $HOME itself holds a repository.
    m = _machine(tmp_path, monkeypatch, fake_home)
    make_repo(m.home, {"README": "x\n"})
    monkeypatch.chdir(m.home)

    out = _run().stdout

    assert _git_read_lines(out) == [], out


@requires_git
def test_doctor_calls_the_registrations_own_pin_function(tmp_path, monkeypatch, fake_home):
    m = _machine(tmp_path, monkeypatch, fake_home)
    repo = make_repo(m.home / "proj")
    monkeypatch.chdir(repo)
    monkeypatch.setattr(git_read_tool, "git_read_pin_failure", lambda root: GitReadPinFailure("toplevel", "/x"))

    out = _run().stdout

    assert "git_read not registered: the repository toplevel is not the workspace root ('/x')" in out, out
    assert "git_read: available" not in out


@requires_git
def test_a_pin_failure_does_not_change_the_exit_code(tmp_path, monkeypatch, fake_home):
    m = _machine(tmp_path, monkeypatch, fake_home)
    repo = make_repo(m.home / "proj")
    monkeypatch.chdir(repo)

    passing = _run()
    assert "git_read: available" in passing.stdout, passing.stdout
    monkeypatch.setattr(git_read_tool, "git_read_pin_failure", lambda root: GitReadPinFailure("git", ""))
    failing = _run()

    assert "git_read not registered: git was not found on PATH" in failing.stdout, failing.stdout
    assert failing.exit_code == passing.exit_code
    assert "issue(s) found" in failing.stdout  # the endpoint failure; the git_read row adds none
    assert passing.stdout.count("issue(s) found") == failing.stdout.count("issue(s) found")
    n = lambda text: next(ln for ln in text.splitlines() if "issue(s) found" in ln)  # noqa: E731
    assert n(passing.stdout) == n(failing.stdout)
