"""Phase 19 Wave-0 — `autoresearch report` CLI (REP-01 trajectory, REP-02 top-mutations + lineage + drill-down).

RED stubs in the established Wave-0 cadence. Both files drive the CLI through ``app`` + ``CliRunner``;
``app`` already exists, so NO guarded module import is needed. The ``report`` subcommand simply isn't
registered yet (it lands in 19-04), so every ``runner.invoke(app, ["autoresearch", "report", ...])``
returns a non-zero / usage error until then — that is exactly what keeps these ``xfail(strict=False)``
stubs RED without a collection error. Rows are seeded through an ArchiveStore opened on the SAME
``LOCALHARNESS_HOME``-resolved ``archive.db`` the CLI reads (set by the components_home fixture),
mirroring test_autoresearch_cmd.py.
"""
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.autoresearch.archive import ArchiveStore, ArchiveEntry

runner = CliRunner()


def _archive_db_path() -> Path:
    """The archive.db the CLI resolves under LOCALHARNESS_HOME (components_home fixture)."""
    return Path(os.environ["LOCALHARNESS_HOME"]) / "archive.db"


async def _seed_rows(specs: list[dict]) -> None:
    """Open a store on the CLI's resolved path and write the given rows (mirrors test_autoresearch_cmd)."""
    import time
    import uuid

    store = ArchiveStore(_archive_db_path())
    await store.open()
    for spec in specs:
        await store.write(
            ArchiveEntry(
                id=spec.get("id", str(uuid.uuid4())),
                parent_id=spec.get("parent_id"),
                component=spec.get("component", "agents.main.system_prompt"),
                diff=spec.get("diff", json.dumps({"before": "a", "after": "b"})),
                train_score=spec.get("train_score"),
                train_scores_per_fixture=spec.get("train_scores_per_fixture"),
                holdout_score=spec.get("holdout_score"),
                p_value=spec.get("p_value"),
                cost=spec.get("cost"),
                ts=spec.get("ts", int(time.time())),
                approved_by=spec.get("approved_by"),
                status=spec.get("status", "in_flight"),
            )
        )
    await store.close()


# ---------------------------------------------------------------------------
# REP-01 — improvement-trajectory sparkline + baseline curve
# ---------------------------------------------------------------------------


async def test_report_shows_trajectory(components_home):
    """A promoted lineage over rising ts/train_score ⇒ report exits 0 and emits a sparkline + 'train'."""
    await _seed_rows(
        [
            dict(id="traj-1", component="agent.role", status="promoted", ts=100, train_score=0.40),
            dict(id="traj-2", component="agent.role", status="promoted", ts=200, train_score=0.55),
            dict(id="traj-3", component="agent.role", status="promoted", ts=300, train_score=0.70),
        ]
    )
    result = runner.invoke(app, ["autoresearch", "report"])
    assert result.exit_code == 0, result.output
    assert any(block in result.output for block in "▁▂▃▄▅▆▇█"), "trajectory sparkline glyph missing"
    assert "train" in result.output.lower()  # the train trajectory is labelled


# ---------------------------------------------------------------------------
# REP-02 — top mutations + lineage child→root via --show
# ---------------------------------------------------------------------------


async def test_top_mutations_and_lineage(components_home):
    """Top-mutations table lists a Pareto child; `--show <child>` prints lineage child→root."""
    parent_id = "parent00deadbeef"
    child_id = "child00cafef00d1"
    await _seed_rows(
        [
            dict(id=parent_id, parent_id=None, component="agent.role", status="promoted",
                 ts=100, train_score=0.60, p_value=0.04, cost=1.0,
                 train_scores_per_fixture={"fx_a": 0.6}),
            dict(id=child_id, parent_id=parent_id, component="agent.role", status="promoted",
                 ts=200, train_score=0.85, p_value=0.01, cost=1.0,
                 train_scores_per_fixture={"fx_a": 0.9}),
        ]
    )
    overview = runner.invoke(app, ["autoresearch", "report"])
    assert overview.exit_code == 0, overview.output
    assert child_id[:8] in overview.output  # the front member shows in the top-mutations table
    assert "agent.role" in overview.output

    drill = runner.invoke(app, ["autoresearch", "report", "--show", child_id])
    assert drill.exit_code == 0, drill.output
    # lineage child→root: BOTH the child and its parent appear
    assert child_id[:8] in drill.output
    assert parent_id[:8] in drill.output


# ---------------------------------------------------------------------------
# REP-02 — drill-down: change + hypothesis + proof; hyperparameter labelled, no invented mechanism
# (depends on GAP-1 fix from 19-02: rationale persisted into the diff blob)
# ---------------------------------------------------------------------------


async def test_drilldown_hypothesis_and_proof(components_home):
    """Prompt-edit `--show` prints rationale + p + a per-fixture mover; hyperparameter says 'no mechanism'."""
    prompt_id = "prompt00edit0001"
    await _seed_rows(
        [
            dict(
                id=prompt_id, component="agent.role", status="promoted",
                diff=json.dumps({
                    "before": "old prompt", "after": "new prompt",
                    "rationale": "the agent ignored the deny list", "kind": "prompt",
                }),
                train_score=0.82, holdout_score=0.80, p_value=0.01,
                train_scores_per_fixture={"fx_a": 0.9, "fx_b": 0.7},
            )
        ]
    )
    prompt_show = runner.invoke(app, ["autoresearch", "report", "--show", prompt_id])
    assert prompt_show.exit_code == 0, prompt_show.output
    assert "the agent ignored the deny list" in prompt_show.output  # the hypothesis (rationale)
    assert "0.01" in prompt_show.output  # the proof p-value is surfaced
    assert ("fx_a" in prompt_show.output or "fx_b" in prompt_show.output)  # a per-fixture mover

    hp_id = "hyperparam000001"
    await _seed_rows(
        [
            dict(
                id=hp_id, component="org.context.compaction_threshold_pct", status="promoted",
                diff=json.dumps({"before": 70, "after": 75, "rationale": "", "kind": "hyperparameter"}),
                train_score=0.80, holdout_score=0.78, p_value=0.02,
            )
        ]
    )
    hp_show = runner.invoke(app, ["autoresearch", "report", "--show", hp_id])
    assert hp_show.exit_code == 0, hp_show.output
    # hyperparameter mutations get NO manufactured hypothesis — the exact label, verbatim
    assert "hyperparameter (numeric tuning), no mechanism" in hp_show.output
