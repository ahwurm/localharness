"""PROP-01 / SC1 — `localharness propose` CLI surface.

CliRunner coverage of the `propose` command (registered in 16-03). The pipeline's
model call is made hermetic by monkeypatching the module-level ``LLMClient`` in
``autoresearch.proposer`` with a fake returning valid JSON — the command under test
is the CLI wrapper, NOT the pipeline, so patching the client (not ``propose``) keeps
the wrapper fully exercised. bench paths are resolved from a tmp ``bench/bench.yaml``
(via ``monkeypatch.chdir``); config + archive isolate via ``components_home``'s
``LOCALHARNESS_HOME``.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from localharness.cli.app import app

runner = CliRunner()

# Repo-root categories file (absolute, captured before any monkeypatch.chdir). The
# proposer's seal calls load_scenario, which validates each fixture's `category`
# against bench/categories.yaml — resolved relative to cwd unless this env points
# elsewhere. Tests chdir into a tmp dir without that file, so we pin the override.
_REPO_CATEGORIES = Path(__file__).resolve().parents[2] / "bench" / "categories.yaml"

_GOOD_AFTER = "You are a careful, terse assistant."
_GOOD_RATIONALE = "because the train trace failed on verbosity"


class _FakeProposerClient:
    """Module-level LLMClient stand-in: returns a valid proposer JSON reply."""

    def __init__(self, llm_cfg):
        self.config = llm_cfg

    async def detect_capabilities(self):
        class _Cap:
            tool_call_mode = "xml"
        return _Cap()

    async def complete(self, messages, tools=None, stream=False):
        class _Msg:
            content = json.dumps({"after": _GOOD_AFTER, "rationale": _GOOD_RATIONALE})

        return _Msg(), None

    async def stream_complete(self, messages, tools=None, on_token=None):
        return await self.complete(messages)  # #18: proposer uses the streaming path


def _setup(home, corpus, results, monkeypatch, tmp_path):
    """Wire a hermetic env: proposer-blocked config.yaml + tmp bench.yaml + faked client."""
    # 1. Project config WITH a distinct proposer block (so cfg.proposer is non-None).
    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "version": "1",
                "provider": {
                    "provider_type": "ollama",
                    "base_url": "http://localhost:11434/v1",
                    "default_model": "test-model",
                },
                "proposer": {
                    "base_url": "http://localhost:11434/v1",
                    "model": "frontier-strong:latest",
                },
            }
        ),
        encoding="utf-8",
    )
    # 2. tmp bench.yaml pointing corpus/results at the proposer fixtures, run from cwd.
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "bench.yaml").write_text(
        yaml.safe_dump(
            {"corpus_path": str(corpus), "results_path": str(results)}
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    # 3. The seal's load_scenario validates `category` against bench/categories.yaml,
    #    which it resolves relative to cwd; point the override at the repo file so the
    #    tmp cwd (no categories.yaml) doesn't make every scenario look "unknown".
    monkeypatch.setenv("LOCALHARNESS_CATEGORIES_PATH", str(_REPO_CATEGORIES))
    # 4. Make the model call hermetic — patch the proposer's module-level client.
    import localharness.autoresearch.proposer as prop_mod

    monkeypatch.setattr(prop_mod, "LLMClient", _FakeProposerClient, raising=False)


def test_propose_returns_diff_and_rationale(
    proposer_corpus, proposer_results, components_home, monkeypatch, tmp_path
):
    """SC1: `propose --component agent.role --traces <train_run>` exits 0 and prints a diff + rationale."""
    _setup(components_home, proposer_corpus, proposer_results["results"], monkeypatch, tmp_path)
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
    assert "after" in out or _GOOD_AFTER.lower() in out  # diff (after side) present
    assert "rationale" in out
    assert "train trace failed" in out  # the model's rationale text surfaced


def test_propose_holdout_exits_2(
    proposer_corpus, proposer_results, components_home, monkeypatch, tmp_path
):
    """SC1/PROP-03: a HOLDOUT run_id surfaces the seal refusal as exit code 2."""
    _setup(components_home, proposer_corpus, proposer_results["results"], monkeypatch, tmp_path)
    result = runner.invoke(
        app,
        [
            "propose",
            "--component",
            "agent.role",
            "--traces",
            proposer_results["holdout_run_id"],
        ],
    )
    assert result.exit_code == 2, result.output


def test_propose_json_shape(
    proposer_corpus, proposer_results, components_home, monkeypatch, tmp_path
):
    """SC1: `--json` emits a dict with keys {component, diff{before,after}, rationale}."""
    _setup(components_home, proposer_corpus, proposer_results["results"], monkeypatch, tmp_path)
    result = runner.invoke(
        app,
        [
            "propose",
            "--component",
            "agent.role",
            "--traces",
            proposer_results["train_run_id"],
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert set(payload.keys()) == {"component", "diff", "rationale"}
    assert payload["component"] == "agent.role"
    assert set(payload["diff"].keys()) == {"before", "after"}
    assert payload["diff"]["after"] == _GOOD_AFTER
    assert payload["rationale"] == _GOOD_RATIONALE
