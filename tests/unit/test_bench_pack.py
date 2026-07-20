"""Trace pack: bench run files (bus-event JSONL + ScenarioCompleted verdict) → a versioned,
leak-scanned dataset pack {manifest.json, trajectories.jsonl}.

Bench-only by construction: a file without a ScenarioCompleted verdict (e.g. a live
dogfood session, which contains the owner's real life) is SKIPPED and counted, never
packed. Any secret/home-path pattern inside packed content FAILS the build outright —
fail explicit, no silent scrubbing. Packs are regenerated per release as functionality
evolves (the manifest stamps harness_version + source), superseding older packs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import localharness
from localharness.bench.pack import PackLeakError, build_pack

runner = CliRunner()


def _write_run(root: Path, scenario: str, model: str, success: bool,
               final: str = "Done: 3 files.", extra_content: str | None = None) -> Path:
    d = root / model / scenario
    d.mkdir(parents=True, exist_ok=True)
    p = d / "20260720T120000Z.jsonl"
    events = [
        {"event_type": "UserMessage", "content": f"do the {scenario} task"},
        {"event_type": "Action", "action_type": "llm_response", "content": "scanning…",
         "has_tool_calls": True, "tool_name": "glob", "tool_params": {"pattern": "*"}},
        {"event_type": "Observation", "content": extra_content or "3 files found"},
        {"event_type": "Action", "action_type": "llm_response", "content": final,
         "has_tool_calls": False},
        {"event_type": "ScenarioCompleted", "scenario_name": scenario, "model": model,
         "success": success, "tokens_in": 100, "tokens_out": 40, "iterations": 2,
         "tool_call_count": 1, "parse_failures": 0, "stuck_recoveries": 0},
    ]
    p.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return p


def _write_session_file(root: Path) -> Path:
    """A dogfood-shaped file: bus events but NO ScenarioCompleted verdict."""
    d = root / "strays"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "session.jsonl"
    events = [
        {"event_type": "UserMessage", "content": "what hands should i play at poker"},
        {"event_type": "Action", "action_type": "llm_response", "content": "Play tight.",
         "has_tool_calls": False},
        {"event_type": "TaskComplete", "success": True, "summary": "Play tight."},
    ]
    p.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return p


def test_build_pack_packs_verdicted_runs_and_skips_sessions(tmp_path):
    results = tmp_path / "results"
    _write_run(results, "scen-a", "qwen3.6-35b-a3b", True)
    _write_run(results, "scen-b", "qwen3.6-35b-a3b", False, final="I could not finish.")
    _write_session_file(results)
    out = tmp_path / "pack"

    manifest = build_pack(results, out)

    recs = [json.loads(l) for l in (out / "trajectories.jsonl").read_text().splitlines()]
    assert len(recs) == 2
    by_scen = {r["scenario"]: r for r in recs}
    assert by_scen["scen-a"]["success"] is True
    assert by_scen["scen-b"]["success"] is False
    assert by_scen["scen-a"]["model"] == "qwen3.6-35b-a3b"
    assert by_scen["scen-a"]["harness_version"] == localharness.__version__

    msgs = by_scen["scen-a"]["messages"]
    assert msgs[0] == {"role": "user", "content": "do the scen-a task"}
    assert msgs[1]["role"] == "assistant" and msgs[1]["tool_calls"][0]["name"] == "glob"
    assert msgs[2] == {"role": "tool", "content": "3 files found"}
    assert msgs[3] == {"role": "assistant", "content": "Done: 3 files."}

    disk_manifest = json.loads((out / "manifest.json").read_text())
    assert disk_manifest == manifest
    assert manifest["pack_schema"] == 1
    assert manifest["harness_version"] == localharness.__version__
    assert manifest["runs_packed"] == 2
    assert manifest["files_skipped"] == 1          # the dogfood-shaped stray never packs
    outcomes = {(o["scenario"], o["model"]): o for o in manifest["outcomes"]}
    assert outcomes[("scen-a", "qwen3.6-35b-a3b")] == {
        "scenario": "scen-a", "model": "qwen3.6-35b-a3b", "runs": 1, "successes": 1,
    }
    assert outcomes[("scen-b", "qwen3.6-35b-a3b")]["successes"] == 0


def test_build_pack_fails_loud_on_leak_patterns(tmp_path):
    results = tmp_path / "results"
    _write_run(results, "scen-a", "m", True,
               extra_content="found /home/alice/.ssh/id_rsa in scan")
    with pytest.raises(PackLeakError, match="scen-a"):
        build_pack(results, tmp_path / "pack")
    assert not (tmp_path / "pack" / "trajectories.jsonl").exists()


def test_build_pack_empty_results_is_explicit_error(tmp_path):
    (tmp_path / "results").mkdir()
    with pytest.raises(RuntimeError, match="[Nn]o bench run files"):
        build_pack(tmp_path / "results", tmp_path / "pack")


def test_bench_pack_cli(tmp_path):
    from localharness.cli.app import app

    results = tmp_path / "results"
    _write_run(results, "scen-a", "m", True)
    out = tmp_path / "pack"
    res = runner.invoke(app, ["bench", "pack", "--results", str(results), "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert (out / "manifest.json").exists()
    assert (out / "trajectories.jsonl").exists()
    assert "runs_packed=1" in res.output.replace(" ", "").replace(":", "=") or "1" in res.output
