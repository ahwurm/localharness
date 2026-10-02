"""AUTO-01/02/03 end to end through the ROOT app, after the cut (Phase 50).

- ON by default: `plugins info autoresearch` prints `autoresearch.enabled`, the `proposer.*` and
  `sentinel.*` rows and the disclosure line; doctor prints the autoresearch row; `--help` lists
  `autoresearch`, `experiment`, `propose`.
- `plugins disable autoresearch` (the real command, tmp home) writes the switch to the global
  overrides; then the three commands are gone from `--help`, a named call exits 4 with the enable
  hint on stderr (the command module never runs), `components list` keeps only
  `autoresearch.enabled`, `components set proposer.model` is refused as unknown, a `proposer:`
  section still loads, and nothing is deleted (the archive db, the tracked `autoresearch/` files).
  `plugins enable autoresearch` brings the three commands back.
- AUTO-03: with autoresearch off, an offline bench run (mock LLM, fixture scenario in a tmp corpus,
  tmp results dir) completes and writes its summary; `bench --help` exits 0; no holdout path loads.

Not marked `plugin("autoresearch")`: every test sets the plugin's state itself.
REAL: the Typer root app, the overlay writer, the loader, the resolver, the off-command stub, the
bench orchestrator. STUBBED: `run_experiment` (fails the test if reached), the bench's LLM client.
NOT proven: a real model or a real experiment.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import yaml
from typer.testing import CliRunner

from localharness.cli.app import app

REPO = Path(__file__).resolve().parents[2]
THREE = ("autoresearch", "experiment", "propose")
HINT = ("command '{}' is provided by the autoresearch plugin, which is off — "
        "run `localharness plugins enable autoresearch`")
PROPOSER = "proposer:\n  base_url: http://127.0.0.1:9/v1\n  model: fake-proposer\n"
runner = CliRunner()


def _run(*args: str):
    return runner.invoke(app, list(args), env={"COLUMNS": "400"})


def _listed(help_text: str) -> set[str]:
    return {n for n in THREE if re.search(rf"│ {n}\s", help_text)}


def _rows() -> set[str]:
    res = _run("components", "list", "--json")
    assert res.exit_code == 0, res.output
    return {r["path"] for r in json.loads(res.stdout)}


def _tracked() -> list[str]:
    return subprocess.run(["git", "ls-files", "src/localharness/autoresearch"], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.split()


def _with_proposer(home: Path) -> None:
    with (home / "config.yaml").open("a", encoding="utf-8") as f:
        f.write(PROPOSER)


def _on_by_default(home: Path) -> None:
    """The fixture home under LOCALHARNESS_TEST_PLUGINS_OFF seeds `autoresearch: enabled: false`;
    this module states the plugin's state itself, so it starts from the shipped default."""
    cfg = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    cfg.pop("autoresearch", None)
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


def test_on_by_default_info_doctor_and_help(components_home):
    _on_by_default(components_home)
    _with_proposer(components_home)
    info = _run("plugins", "info", "autoresearch")
    assert info.exit_code == 0, info.output
    assert "autoresearch.enabled" in info.output
    assert re.search(r"^\s+proposer\.model\s.*'fake-proposer'", info.output, re.M), info.output
    assert re.search(r"^\s+sentinel\.\w+\s", info.output, re.M), info.output
    assert "proposer: and sentinel: keep their pre-plugin names" in info.output
    doctor = _run("doctor").output  # exit 1 here = the unreachable test provider
    assert "✓ autoresearch: proposer: fake-proposer at http://127.0.0.1:9/v1" in doctor, doctor
    assert _listed(_run("--help").output) == set(THREE)


def test_disable_removes_refuses_and_deletes_nothing_then_enable_restores(components_home, monkeypatch):
    _on_by_default(components_home)
    _with_proposer(components_home)
    import localharness.cli.experiment_cmd as experiment_cmd

    async def never(*a, **k):
        raise AssertionError("run_experiment ran with autoresearch off")
    monkeypatch.setattr(experiment_cmd, "run_experiment", never)
    archive = components_home / "archive.db"
    archive.write_bytes(b"seeded archive")  # stands in for a user's archive; must survive
    tracked = _tracked()
    assert {"proposer.model", "sentinel.saturation_k", "autoresearch.enabled"} <= _rows()

    off = _run("plugins", "disable", "autoresearch")
    assert off.exit_code == 0, off.output
    assert yaml.safe_load((components_home / "overrides.yaml").read_text()) == {"autoresearch": {"enabled": False}}

    assert _listed(_run("--help").output) == set()
    for args, cmd in ((["experiment", "run", "abcd1234"], "experiment"), (["propose", "--help"], "propose"),
                      (["autoresearch", "archive", "list"], "autoresearch")):
        ran = _run(*args)
        assert (ran.exit_code, ran.stderr.strip(), ran.stdout) == (4, HINT.format(cmd), ""), ran.output
    rows = _rows()
    assert not {r for r in rows if r.startswith(("proposer.", "sentinel."))}, sorted(rows)
    assert "autoresearch.enabled" in rows
    refused = _run("components", "set", "proposer.model", "x")
    assert refused.exit_code != 0 and "Unknown path: 'proposer.model'" in refused.output, refused.output
    from localharness.cli.components_cmd import _build_loader
    assert _build_loader().load_harness().proposer.model == "fake-proposer"  # the section still validates
    assert archive.read_bytes() == b"seeded archive" and _tracked() == tracked

    on = _run("plugins", "enable", "autoresearch")
    assert on.exit_code == 0, on.output
    assert _listed(_run("--help").output) == set(THREE)
    assert "proposer.model" in _rows()


async def test_the_bench_runs_with_autoresearch_off(tmp_path, components_home, monkeypatch,
                                                    mock_llm_client, fixture_scenario_path):
    _on_by_default(components_home)
    with (components_home / "config.yaml").open("a", encoding="utf-8") as f:
        f.write("autoresearch:\n  enabled: false\n")
    monkeypatch.setenv("LOCALHARNESS_CATEGORIES_PATH", str(REPO / "bench" / "categories.yaml"))
    monkeypatch.chdir(tmp_path)
    assert _listed(_run("--help").output) == set()  # really off
    assert _run("bench", "--help").exit_code == 0

    import localharness.bench.orchestrator as orch
    loaded: list[Path] = []
    real_load = orch.load_scenario
    monkeypatch.setattr(orch, "load_scenario", lambda p: loaded.append(Path(p)) or real_load(p))
    corpus = tmp_path / "scenarios"
    corpus.mkdir()
    (corpus / "minimal_golden.yaml").write_text(fixture_scenario_path.read_text())
    client = mock_llm_client([mock_llm_client.Response(content="4")])
    rc = await orch.run_bench(scenario="minimal_golden", matrix=False, models=[], threshold_overrides=[],
                              corpus_path=corpus, results_path=tmp_path / "results", json_output=True,
                              llm_client_factory=lambda _: client, min_runs_override=1, max_runs_override=1)
    assert rc == 0
    model_dirs = list((tmp_path / "results").iterdir())
    assert model_dirs and all((d / "summary.json").exists() and (d / "summary.md").exists() for d in model_dirs)
    assert loaded and not [p for p in loaded if "holdout" in p.parts], loaded
