"""Hermetic behavior tests for scripts/coll_distribution_report.py (COLL-05).

All tests run on a synthetic JSONL fixture in tmp_path — no real-trace dependency, no
network, no vLLM. The script is invoked as a subprocess (exercises argparse + exit codes)
using the same interpreter as pytest, so localharness is importable.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "coll_distribution_report.py"
BASE = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)


def _ev(**kw) -> str:
    return json.dumps(kw)


def _write_synthetic_trace(path: Path) -> None:
    """25 Action/Observation pairs for tool 't1' (20 successes, then 1 error at pair 20,
    then 4 more successes) + 3 UserMessages (one correction, two neutral) + 1 TurnCompleted.
    Constant duration + output so latency/size variance is 0 (z=0): the lone error — a
    reliable tool suddenly failing — must own the max composite score."""
    lines: list[str] = []
    for i in range(25):
        t_act = (BASE + timedelta(seconds=i)).isoformat()
        t_obs = (BASE + timedelta(seconds=i, milliseconds=100)).isoformat()
        lines.append(_ev(
            id=f"act-{i}", seq=2 * i, timestamp=t_act, agent_id="a", session_id="s",
            event_type="Action", action_type="tool_call", tool_call_id=f"tc-{i}", tool_name="t1",
        ))
        is_err = (i == 20)
        lines.append(_ev(
            id=f"obs-{i}", seq=2 * i + 1, timestamp=t_obs, agent_id="a", session_id="s",
            event_type="Observation", observation_type="tool_result", tool_call_id=f"tc-{i}",
            tool_name="t1", output=("" if is_err else "ok-output-"),
            error=("boom" if is_err else None),
        ))
    # user-signal channel: one correction, one TurnCompleted, two neutral messages.
    lines.append(_ev(id="um-1", seq=100, timestamp=(BASE + timedelta(seconds=30)).isoformat(),
                     agent_id="a", session_id="s", event_type="UserMessage",
                     content="no, that's wrong", channel="terminal"))
    lines.append(_ev(id="tcpl", seq=101, timestamp=(BASE + timedelta(seconds=31)).isoformat(),
                     agent_id="a", session_id="s", event_type="TurnCompleted", iterations=1,
                     duration_seconds=1.0, elapsed_tokens=10, summary="did a thing"))
    lines.append(_ev(id="um-2", seq=102, timestamp=(BASE + timedelta(seconds=32)).isoformat(),
                     agent_id="a", session_id="s", event_type="UserMessage",
                     content="please summarize the document", channel="terminal"))
    lines.append(_ev(id="um-3", seq=103, timestamp=(BASE + timedelta(seconds=33)).isoformat(),
                     agent_id="a", session_id="s", event_type="UserMessage",
                     content="run the analysis now", channel="terminal"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(trace: Path, results: Path, hand_labels: Path | None = None):
    cmd = [sys.executable, str(SCRIPT), "--trace", str(trace), "--results", str(results)]
    if hand_labels is not None:
        cmd += ["--hand-labels", str(hand_labels)]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))


def test_smoke_synthetic_trace(tmp_path):
    trace = tmp_path / "bus-events.jsonl"
    _write_synthetic_trace(trace)
    out = tmp_path / "out"
    proc = _run(trace, out)
    assert proc.returncode == 0, proc.stderr

    # All four artifacts produced.
    for name in ("report.md", "scores.jsonl", "signals.jsonl", "verdict.json"):
        assert (out / name).exists(), f"missing {name}"

    # The error-after-20-successes observation owns the max score.
    scores = [json.loads(l) for l in (out / "scores.jsonl").read_text().splitlines() if l.strip()]
    assert len(scores) == 25
    assert sum(r["is_error"] for r in scores) == 1
    top = max(scores, key=lambda r: r["score"])
    assert top["is_error"] == 1, top

    # Exactly one detected correction in the census.
    sigs = [json.loads(l) for l in (out / "signals.jsonl").read_text().splitlines() if l.strip()]
    assert len(sigs) == 3
    corrections = [s for s in sigs if s["detected"] and s["detected"]["signal_type"] == "correction"]
    assert len(corrections) == 1, sigs

    verdict = json.loads((out / "verdict.json").read_text())
    assert verdict["statistical_channel"] in ("separates", "kill")


def test_refuses_bench_results(tmp_path):
    trace = tmp_path / "bus-events.jsonl"
    _write_synthetic_trace(trace)
    bad = tmp_path / "bench" / "results" / "out"  # resolves to a path containing bench/results
    proc = _run(trace, bad)
    assert proc.returncode != 0
    # Guard fires before any output is written.
    assert not (bad / "report.md").exists()
    assert not (bad / "verdict.json").exists()


def test_results_flag_required(tmp_path):
    trace = tmp_path / "bus-events.jsonl"
    _write_synthetic_trace(trace)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--trace", str(trace)],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0  # argparse required=True


def test_no_network_surface():
    src = SCRIPT.read_text(encoding="utf-8")
    for banned in ("httpx", "openai", "requests", "LLMClient"):
        assert banned not in src, f"network/LLM surface leaked: {banned}"
