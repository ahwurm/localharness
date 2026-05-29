"""PROP-01 / SC1 — `localharness propose` CLI surface (Phase 16 Wave 0 RED stub).

CliRunner coverage of the `propose` command. The command registers in 16-03; until
then the invocation exits nonzero and this stub goes RED→GREEN as the CLI lands. The
app import is guarded so collection never breaks before the command is wired.
"""
import pytest
from typer.testing import CliRunner

try:
    from localharness.cli.app import app
    _APP_READY = True
except Exception:
    _APP_READY = False

runner = CliRunner()


@pytest.mark.xfail(strict=False)
def test_propose_returns_diff_and_rationale(proposer_corpus, proposer_results, monkeypatch):
    """SC1: `propose --component agent.role --traces <train_run>` exits 0 and prints a diff + rationale."""
    if not _APP_READY:
        pytest.xfail("propose command not yet implemented")
    result = runner.invoke(
        app,
        [
            "propose",
            "--component",
            "agent.role",
            "--traces",
            proposer_results["train_run_id"],
        ],
    )
    assert result.exit_code == 0, result.output
    out = result.output.lower()
    assert "after" in out  # diff (before/after) present
    assert "rationale" in out
