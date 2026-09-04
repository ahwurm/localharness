"""E cluster (d) — ten `init --workspace` at once, and nobody sees a traceback.

Scaffolding used to be check-then-create: `if target.exists(): refuse`, then `mkdir(parents=True)`.
There is a window between those two calls, and a CI matrix, a `make -j`, or two terminals is all it
takes to land in it — nine processes crashing with FileExistsError over a workspace that got
created perfectly well.

The fix is to stop asking and start claiming: a bare `mkdir()` is atomic in the kernel, so exactly
one process wins and the rest get EEXIST, which is the same answer the pre-check gives and gets the
same words.

Real processes, not threads: `os.mkdir` releases the GIL, so threads would mostly serialize and the
window this grades would rarely open. `fork` also gives each child its own CWD and its own exit
code, which is what the assertions are actually about.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

import pytest
import yaml

from localharness.config.paths import WORKSPACE_DIR_NAME

_RACERS = 10


def _run_one(project: str, home: str, queue, barrier=None) -> None:  # pragma: no cover - child
    """One `init --workspace`, in its own process, reporting (exit code, output)."""
    from typer.testing import CliRunner

    from localharness.cli.app import app

    for var in ("LOCALHARNESS_DIR", "LOCALHARNESS_HOME", "LOCALHARNESS_ENDPOINT",
                "LOCALHARNESS_MODEL"):
        os.environ.pop(var, None)
    os.environ["HOME"] = home
    os.environ["COLUMNS"] = "400"
    os.chdir(project)
    if barrier is not None:
        barrier.wait(timeout=30)  # every racer hits mkdir in the same instant
    result = CliRunner().invoke(app, ["init", "--workspace"])
    queue.put((result.exit_code, result.output, repr(result.exception)))


@pytest.fixture
def project(tmp_path) -> tuple[Path, Path]:
    home = tmp_path / "home"
    (home / WORKSPACE_DIR_NAME).mkdir(parents=True)
    proj = tmp_path / "proj"
    proj.mkdir()
    return proj, home


def test_ten_parallel_scaffolds_all_exit_politely(project):
    proj, home = project
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    barrier = ctx.Barrier(_RACERS)
    procs = [
        ctx.Process(target=_run_one, args=(str(proj), str(home), queue, barrier))
        for _ in range(_RACERS)
    ]
    for p in procs:
        p.start()
    results = [queue.get(timeout=60) for _ in range(_RACERS)]
    for p in procs:
        p.join(timeout=60)

    # A typer.Exit surfaces through CliRunner as SystemExit — that is the polite path. Anything
    # else is the traceback this test exists to rule out.
    crashed = [
        (code, exc) for code, _out, exc in results
        if exc != "None" and not exc.startswith("SystemExit(")
    ]
    assert not crashed, f"a racer raised instead of reporting: {crashed}"

    winners = [r for r in results if r[0] == 0]
    losers = [r for r in results if r[0] == 1]
    assert len(winners) == 1, f"expected exactly one winner, got {[r[0] for r in results]}"
    assert len(losers) == _RACERS - 1
    for _code, out, _exc in losers:
        assert "already exists" in out


def test_the_workspace_the_race_leaves_behind_is_complete(project):
    """A winner that got interrupted by the losers' cleanup would leave a config-less directory —
    which the next run refuses to touch, so the project would be permanently un-scaffoldable."""
    proj, home = project
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    barrier = ctx.Barrier(_RACERS)
    procs = [
        ctx.Process(target=_run_one, args=(str(proj), str(home), queue, barrier))
        for _ in range(_RACERS)
    ]
    for p in procs:
        p.start()
    for _ in range(_RACERS):
        queue.get(timeout=60)
    for p in procs:
        p.join(timeout=60)

    workspace = proj / WORKSPACE_DIR_NAME
    assert (workspace / "agents").is_dir()
    config = workspace / "config.yaml"
    assert config.exists()
    # Parses, and carries no keys — the scaffold template is all comments.
    assert yaml.safe_load(config.read_text(encoding="utf-8")) is None


def test_losing_the_race_is_the_same_answer_as_finding_it_there(project, monkeypatch):
    """The race, made deterministic — the half a real ten-way run cannot be relied on to hit.

    The pre-check is forced to answer "nothing here" for a workspace that IS here, which is
    exactly the state a process is in when it loses: it looked, saw nothing, and by the time it
    created, somebody else had. The kernel's EEXIST is then the only thing standing between the
    user and a traceback — and, just as importantly, the loser must not clean up: the tree it
    found belongs to the winner, who is still filling it in.
    """
    from typer.testing import CliRunner

    from localharness.cli.app import app

    proj, home = project
    for var in ("LOCALHARNESS_DIR", "LOCALHARNESS_HOME", "LOCALHARNESS_ENDPOINT",
                "LOCALHARNESS_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COLUMNS", "400")
    monkeypatch.chdir(proj)

    winner = proj / WORKSPACE_DIR_NAME
    winner.mkdir()
    (winner / "config.yaml").write_text("# the winner's file\n", encoding="utf-8")

    real_exists = Path.exists
    monkeypatch.setattr(
        Path, "exists",
        lambda self, *a, **k: False if self.name == WORKSPACE_DIR_NAME else real_exists(self),
    )

    result = CliRunner().invoke(app, ["init", "--workspace"])

    assert result.exit_code == 1
    assert "already exists" in result.output
    # The loser must not have cleaned up the winner's tree on its way out.
    monkeypatch.undo()
    assert (winner / "config.yaml").read_text(encoding="utf-8") == "# the winner's file\n"
