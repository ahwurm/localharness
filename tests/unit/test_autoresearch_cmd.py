"""ARCH-03 — `localharness autoresearch archive` CLI surface.

Each test exercises one command path via CliRunner. Rows are seeded through an
ArchiveStore opened on the SAME .localharness/archive.db path the CLI resolves from
LOCALHARNESS_HOME (set by the components_home fixture). The `autoresearch archive`
sub-app landed in 15-04; the Phase 15 Wave 0 xfail(strict=False) stubs are retired
to plain assertions.
"""
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

try:
    from localharness.cli.app import app
    from localharness.autoresearch.archive import ArchiveStore, ArchiveEntry  # noqa: F401
except ImportError:
    pytest.skip("autoresearch CLI not yet wired (15-04)", allow_module_level=True)

runner = CliRunner()


def _archive_db_path() -> Path:
    """The archive.db the CLI resolves under LOCALHARNESS_HOME (components_home fixture)."""
    return Path(os.environ["LOCALHARNESS_HOME"]) / "archive.db"


async def _seed_rows(specs: list[dict]) -> ArchiveStore:
    """Open a store on the CLI's resolved path and write rows; returns the open store."""
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
    return store


def test_subapp_registered():
    """`autoresearch archive --help` exits 0 and advertises list/show/approve."""
    result = runner.invoke(app, ["autoresearch", "archive", "--help"])
    assert result.exit_code == 0, result.output
    assert "list" in result.output
    assert "show" in result.output
    assert "approve" in result.output


async def test_list_table(components_home):
    """`archive list` renders a table containing each row's 8-char id + component."""
    ids = ["aaaaaaaa1111", "bbbbbbbb2222", "cccccccc3333"]
    await _seed_rows([dict(id=i, component=f"agents.main.c{n}") for n, i in enumerate(ids)])
    result = runner.invoke(app, ["autoresearch", "archive", "list"])
    assert result.exit_code == 0, result.output
    for i in ids:
        assert i[:8] in result.output
    assert "agents.main.c0" in result.output


async def test_list_json(components_home):
    """`archive list --json` emits a list of dicts with the documented keys."""
    await _seed_rows([dict(id="json-row-1", train_score=0.5, p_value=0.01, cost=1.0)])
    result = runner.invoke(app, ["autoresearch", "archive", "list", "--json"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    assert isinstance(rows, list) and rows
    keys = set(rows[0].keys())
    assert {
        "id", "component", "train_score", "holdout_score",
        "p_value", "cost", "status", "approved_by", "ts",
    }.issubset(keys)


async def test_list_filters(components_home):
    """`--component`, `--status`, and `--limit` each narrow the result set."""
    await _seed_rows(
        [
            dict(id="f1", component="agents.main.system_prompt", status="promoted", ts=100),
            dict(id="f2", component="tools.bash.description", status="in_flight", ts=200),
            dict(id="f3", component="agents.main.system_prompt", status="in_flight", ts=300),
        ]
    )
    by_comp = runner.invoke(
        app, ["autoresearch", "archive", "list", "--component", "agents.main.system_prompt", "--json"]
    )
    assert by_comp.exit_code == 0, by_comp.output
    assert {r["id"] for r in json.loads(by_comp.stdout)} == {"f1", "f3"}

    by_status = runner.invoke(app, ["autoresearch", "archive", "list", "--status", "promoted", "--json"])
    assert by_status.exit_code == 0, by_status.output
    assert {r["id"] for r in json.loads(by_status.stdout)} == {"f1"}

    limited = runner.invoke(app, ["autoresearch", "archive", "list", "--limit", "1", "--json"])
    assert limited.exit_code == 0, limited.output
    assert len(json.loads(limited.stdout)) == 1


async def test_show_prefix_resolution(components_home):
    """`show <id[:8]>` resolves the unique row and prints fields + diff + lineage."""
    full_id = "abc12345deadbeef"
    await _seed_rows([dict(id=full_id, diff=json.dumps({"before": "X", "after": "Y"}))])
    result = runner.invoke(app, ["autoresearch", "archive", "show", full_id[:8]])
    assert result.exit_code == 0, result.output
    assert "agents.main.system_prompt" in result.output
    assert "Y" in result.output  # diff content rendered
    assert "lineage" in result.output.lower()


async def test_show_ambiguous_prefix(components_home):
    """`show <prefix>` matching >1 row exits 2 and lists the colliding ids."""
    await _seed_rows(
        [
            dict(id="dupprefix0001"),
            dict(id="dupprefix0002"),
        ]
    )
    result = runner.invoke(app, ["autoresearch", "archive", "show", "dupprefi"])
    assert result.exit_code == 2, result.output
    assert "dupprefix0001" in result.output
    assert "dupprefix0002" in result.output


async def test_show_lineage_chain(components_home):
    """`show <c>` for an a->b->c chain lists c, b, a in lineage order."""
    await _seed_rows(
        [
            dict(id="chain-a", parent_id=None),
            dict(id="chain-b", parent_id="chain-a"),
            dict(id="chain-c", parent_id="chain-b"),
        ]
    )
    result = runner.invoke(app, ["autoresearch", "archive", "show", "chain-c"])
    assert result.exit_code == 0, result.output
    pos_c = result.output.find("chain-c")
    pos_b = result.output.find("chain-b")
    pos_a = result.output.find("chain-a")
    assert -1 < pos_c < pos_b < pos_a  # rendered root-ward from the target


async def test_show_json(components_home):
    """`show <id> --json` includes a lineage array whose length equals the chain depth."""
    await _seed_rows(
        [
            dict(id="jc-a", parent_id=None),
            dict(id="jc-b", parent_id="jc-a"),
            dict(id="jc-c", parent_id="jc-b"),
        ]
    )
    result = runner.invoke(app, ["autoresearch", "archive", "show", "jc-c", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert len(payload["lineage"]) == 3


# ---------------------------------------------------------------------------
# Phase 18 Wave 0 — `autoresearch run` / `review` / `adopt` CLI (impl lands 18-06)
#
# The run command drives loop.run_loop autonomously; its process exit code is the
# RUN code (clean halt = 0), DISTINCT from the gate verdict code (a reject inside the
# loop is not a non-zero run). review/adopt are the human-checkpoint surface over held rows.
# Tests monkeypatch the module-level run_loop seam (raising=False until 18-05 wires it),
# mirroring how test_propose_cmd patches the proposer's module-level LLMClient.
# ---------------------------------------------------------------------------


def test_run_clean_halt_exit_zero(components_home, monkeypatch):
    """`autoresearch run` with injected fakes exits 0 even when the last gate verdict was a reject (run-code distinct from gate-code)."""
    import localharness.cli.autoresearch_cmd as cmd_mod

    class _Summary:
        iterations = 3
        adopted = 0
        held = 0
        rejected = 3  # every iteration rejected — still a CLEAN run halt
        journal_path = str(_archive_db_path().parent / "run.jsonl")

    async def _fake_run_loop(*args, **kwargs):
        return _Summary()

    monkeypatch.setattr(cmd_mod, "run_loop", _fake_run_loop, raising=False)
    result = runner.invoke(app, ["autoresearch", "run", "--budget", "1", "--max-iterations", "3"])
    assert result.exit_code == 0, result.output  # a loop full of rejects is still exit 0


async def test_review_and_adopt_held(components_home, tmp_git_repo, monkeypatch):
    """`review` lists a held row (component, diff, lift, p-value, id); `adopt <8-char-prefix>` flips it to 'adopted'.

    The seeded row addresses a REAL registry path (``agent.role``) so the adoption seal +
    ``HarnessConfig`` re-validation pass. ``adopt`` git-commits in the cwd's git toplevel, so
    the test chdirs into the throwaway ``tmp_git_repo`` — NEVER the real project repo.
    """
    monkeypatch.chdir(tmp_git_repo)
    held_id = "held1234abcd5678"
    await _seed_rows(
        [
            dict(
                id=held_id,
                component="agent.role",
                status="held",
                diff=json.dumps({"before": "old role", "after": "new role"}),
                train_score=0.82,
                p_value=0.01,
            )
        ]
    )
    review = runner.invoke(app, ["autoresearch", "review"])
    assert review.exit_code == 0, review.output
    assert held_id[:8] in review.output
    assert "agent.role" in review.output

    adopt_res = runner.invoke(app, ["autoresearch", "adopt", held_id[:8]])
    assert adopt_res.exit_code == 0, adopt_res.output

    store = ArchiveStore(_archive_db_path())
    await store.open()
    got = await store.get(held_id)
    await store.close()
    assert got.status == "adopted"
