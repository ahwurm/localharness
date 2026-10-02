"""The autoresearch surface a caller depends on, pinned through the ROOT `app` before the move
(Phase 50 Wave 1): `experiment run` turns each verdict into the process exit code, and the
autoresearch-proposer skill's seeding snippet composes with `experiment run`.

`run_experiment` is always stubbed: no model, no worktree, no holdout. After the commands move
behind the plugin (50-05) these must pass unedited through the lazy mount.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

import pytest
from typer.testing import CliRunner

import localharness.cli.experiment_cmd as experiment_cmd
from localharness.cli.app import app

pytestmark = pytest.mark.plugin("autoresearch")
runner = CliRunner()


def _stub(monkeypatch, result, seen: list | None = None) -> None:
    async def fake_run_experiment(proposal_id, *, trials=1, keep=False):
        if seen is not None:
            seen.append(proposal_id)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(experiment_cmd, "run_experiment", fake_run_experiment)


@pytest.mark.parametrize("code", [0, 1, 2, 3, 4])
def test_experiment_verdict_is_the_exit_code(code, monkeypatch):
    _stub(monkeypatch, code)
    result = runner.invoke(app, ["experiment", "run", "abcd1234"])
    assert result.exit_code == code, result.output


def test_a_raising_experiment_is_exit_4(monkeypatch):
    _stub(monkeypatch, RuntimeError("boom"))
    result = runner.invoke(app, ["experiment", "run", "abcd1234"])
    assert result.exit_code == 4, result.output
    assert "experiment failed: boom" in result.output


def test_skill_snippet_shape(monkeypatch):
    """This is the autoresearch-proposer skill's contract; a rename here breaks the owner's skill.

    Step 4 (verbatim imports and calls): `ArchiveEntry`/`ArchiveStore` from
    `localharness.autoresearch.archive`, `registry.catalogue.build_catalogue`,
    `cli.components_cmd._build_loader`, the db at `$LOCALHARNESS_HOME/archive.db`; step 5:
    `experiment run <id>` reads that same db and its exit code is the verdict."""
    from localharness.autoresearch.archive import ArchiveEntry, ArchiveStore
    from localharness.cli.components_cmd import _build_loader
    from localharness.config.paths import resolve_archive_db_path
    from localharness.registry.catalogue import build_catalogue

    cfg = _build_loader().load_harness()
    assert build_catalogue(cfg)["agent.role"].type_name == "str"
    db = Path(os.environ["LOCALHARNESS_HOME"]) / "archive.db"
    pid = str(uuid.uuid4())

    async def seed():
        store = ArchiveStore(db)
        await store.open()
        try:
            await store.write(ArchiveEntry(
                id=pid, parent_id=None, component="agent.role",
                diff=json.dumps({"before": "a", "after": "b", "rationale": "r", "kind": "prompt"}),
                train_score=None, train_scores_per_fixture=None, holdout_score=None,
                p_value=None, cost=None, ts=int(time.time()), approved_by=None,
                status="in_flight"))
        finally:
            await store.close()

    asyncio.run(seed())
    assert resolve_archive_db_path() == db  # seeding and `experiment run` agree on the db

    seen: list[str] = []
    _stub(monkeypatch, 3, seen)
    result = runner.invoke(app, ["experiment", "run", pid[:8]])
    assert result.exit_code == 3, result.output
    assert seen == [pid[:8]]
