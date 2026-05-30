"""Phase 19 integration Wave-0 — seeded archive → `autoresearch report` → markdown snapshot (REP-01..04 e2e).

ONE end-to-end RED stub: a realistic archive (adopted / held / holdout_rejected / a promoted gap-row)
drives ``autoresearch report``; the report must exit 0 and write a durable markdown snapshot under
``<LOCALHARNESS_HOME>/.localharness/autoresearch/reports/`` whose inbox sections read Adopted/Held/Rejected.
``app`` exists; the ``report`` subcommand lands in 19-04, so this ``xfail(strict=False)`` stays RED
(no collection error) until then. Mirrors test_autoresearch_e2e.py's seeded-archive → CLI → assert shape.
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
    return Path(os.environ["LOCALHARNESS_HOME"]) / "archive.db"


async def _seed_rows(specs: list[dict]) -> None:
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


@pytest.mark.xfail(strict=False)  # impl-pending-19
async def test_report_end_to_end(components_home, tmp_git_repo, seeded_archive):
    """Seeded inbox (adopted/held/holdout_rejected + a promoted gap-row) ⇒ report exits 0 and writes a markdown snapshot with the inbox sections."""
    await _seed_rows(
        [
            dict(id="e2e-adopted", component="agent.role", status="adopted", ts=100,
                 train_score=0.80, holdout_score=0.78, p_value=0.01),
            dict(id="e2e-held", component="agent.role", status="held", ts=200,
                 train_score=0.83, holdout_score=0.80, p_value=0.02),
            dict(id="e2e-hold-rej", component="agent.role", status="holdout_rejected", ts=300,
                 train_score=0.90, holdout_score=0.50, p_value=0.20),
            dict(id="e2e-gap", component="agent.role", status="promoted", ts=400,
                 train_score=0.90, holdout_score=0.70, p_value=0.01),  # gap 0.20 → overfit alert
        ]
    )

    result = runner.invoke(app, ["autoresearch", "report"])
    assert result.exit_code == 0, result.output

    reports_dir = Path(os.environ["LOCALHARNESS_HOME"]) / "autoresearch" / "reports"
    snapshots = list(reports_dir.glob("*.md"))
    assert len(snapshots) >= 1, "report must write a durable markdown snapshot"

    snapshot_text = snapshots[0].read_text(encoding="utf-8")
    # the inbox sorts loop activity into the three review buckets
    assert "Adopted" in snapshot_text
    assert "Held" in snapshot_text
    assert "Rejected" in snapshot_text
