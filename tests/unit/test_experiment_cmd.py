"""EXP-05 — `localharness experiment run` CLI exit-code matrix (Phase 17 Wave 0 RED stubs).

CliRunner coverage of the `experiment run` command (lands 17-04). The gate verdict is the
process exit code (0=promote, 1=reject-train, 2=reject-holdout, 3=inconclusive; >=4 structural).
The command wraps ``run_experiment``; tests monkeypatch that module-level seam with a fake
returning a chosen exit code — the command under test is the wrapper, mirroring how
test_propose_cmd patches the proposer's module-level LLMClient.

Guarded module-top import skips cleanly until experiment_cmd ships in 17-04.
"""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

try:
    from localharness.cli.experiment_cmd import experiment_app
except ImportError:
    pytest.skip("experiment_cmd not yet implemented (17-04)", allow_module_level=True)

runner = CliRunner()


def _patch_runner(monkeypatch, *, exit_code=0, capture=None):
    """Patch experiment_cmd.run_experiment with a fake returning exit_code; record kwargs."""
    import localharness.cli.experiment_cmd as cmd_mod

    async def _fake(proposal_id, **kwargs):
        if capture is not None:
            capture["proposal_id"] = proposal_id
            capture.update(kwargs)
        return exit_code

    monkeypatch.setattr(cmd_mod, "run_experiment", _fake, raising=False)


@pytest.mark.xfail(strict=False, reason="experiment_cmd lands in 17-04")
@pytest.mark.parametrize(
    "fake_code,expected",
    [(0, 0), (1, 1), (2, 2), (3, 3)],
)
def test_exit_codes(monkeypatch, fake_code, expected):
    """Gate verdicts map 1:1 to process exit codes: 0/1/2/3."""
    _patch_runner(monkeypatch, exit_code=fake_code)
    result = runner.invoke(experiment_app, ["run", "abcd1234"])
    assert result.exit_code == expected


@pytest.mark.xfail(strict=False, reason="experiment_cmd lands in 17-04")
def test_structural_refusal_exit(monkeypatch):
    """A structural refusal (fake returns >=4) exits in the >=4 band, distinct from 0-3 gate codes."""
    _patch_runner(monkeypatch, exit_code=4)
    result = runner.invoke(experiment_app, ["run", "abcd1234"])
    assert result.exit_code >= 4
    assert result.exit_code not in (1, 2, 3)


@pytest.mark.xfail(strict=False, reason="experiment_cmd lands in 17-04")
def test_trials_flag_passed_through(monkeypatch):
    """`--trials 4` reaches run_experiment as trials=4."""
    capture = {}
    _patch_runner(monkeypatch, exit_code=0, capture=capture)
    runner.invoke(experiment_app, ["run", "abcd1234", "--trials", "4"])
    assert capture.get("trials") == 4


@pytest.mark.xfail(strict=False, reason="experiment_cmd lands in 17-04")
def test_unknown_proposal_id_structural(components_home, monkeypatch):
    """Unknown id against a real (empty) archive is a structural refusal (exit >=4), NOT a gate verdict.

    Does NOT patch run_experiment — exercises the real resolver against an empty archive.
    """
    result = runner.invoke(experiment_app, ["run", "deadbeef"])
    assert result.exit_code >= 4


@pytest.mark.xfail(strict=False, reason="experiment_cmd lands in 17-04")
def test_json_output_shape(monkeypatch):
    """`--json` on a promote emits JSON with {proposal_id, verdict, exit_code}."""
    _patch_runner(monkeypatch, exit_code=0)
    result = runner.invoke(experiment_app, ["run", "abcd1234", "--json"])
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert {"proposal_id", "verdict", "exit_code"} <= set(payload.keys())
