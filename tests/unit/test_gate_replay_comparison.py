"""Hermetic tests for scripts/gate_replay_comparison.py (PGATE-04).

Mirrors tests/unit/test_coll_report.py: a synthetic JSONL fixture in tmp_path, subprocess
invocation (exercises argparse + exit codes), and a direct import of the pure kill predicate.
No real-trace dependency, no network, no vLLM.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "gate_replay_comparison.py"
BASE = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)

# The pure kill predicate is imported directly (mirrors how test_coll_report probes helpers).
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from gate_replay_comparison import (  # noqa: E402
    _gate_metrics,
    _sensitivity_verdict_line,
    _verdict_line,
)


def _ev(**kw) -> str:
    return json.dumps(kw)


def _write_synthetic_trace(path: Path) -> None:
    """tool t1: 6 reliable successes (prior warms past min_prior_n=5), then an error that a
    later success RESOLVES (motif-captured, a resolved_error fire), then a final error with no
    following success (motif MISS). Two surprising_failure events: the motif floor captures
    one (recall 1/2), the stat channel covers both (recall 2/2) — the gap the stat channel
    closes. Plus one negation correction ('no, that's wrong') + a TurnCompleted + one neutral
    message for the census."""
    lines: list[str] = []
    pattern = [False] * 6 + [True, False, True]  # succ*6, err, succ(resolves), err(miss)
    for i, is_err in enumerate(pattern):
        t_act = (BASE + timedelta(seconds=i)).isoformat()
        t_obs = (BASE + timedelta(seconds=i, milliseconds=100)).isoformat()
        lines.append(_ev(
            id=f"act-{i}", seq=2 * i, timestamp=t_act, agent_id="a", session_id="s",
            event_type="Action", action_type="tool_call", tool_call_id=f"tc-{i}", tool_name="t1",
        ))
        lines.append(_ev(
            id=f"obs-{i}", seq=2 * i + 1, timestamp=t_obs, agent_id="a", session_id="s",
            event_type="Observation", observation_type="tool_result", tool_call_id=f"tc-{i}",
            tool_name="t1", output=("" if is_err else "ok-output-"),
            error=("boom" if is_err else None),
        ))
    lines.append(_ev(id="um-1", seq=100, timestamp=(BASE + timedelta(seconds=30)).isoformat(),
                     agent_id="a", session_id="s", event_type="UserMessage",
                     content="no, that's wrong", channel="terminal"))
    lines.append(_ev(id="tcpl", seq=101, timestamp=(BASE + timedelta(seconds=31)).isoformat(),
                     agent_id="a", session_id="s", event_type="TurnCompleted", iterations=1,
                     duration_seconds=1.0, elapsed_tokens=10, summary="did a thing"))
    lines.append(_ev(id="um-2", seq=102, timestamp=(BASE + timedelta(seconds=32)).isoformat(),
                     agent_id="a", session_id="s", event_type="UserMessage",
                     content="please summarize the document", channel="terminal"))
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

    # Both artifacts produced.
    for name in ("report.md", "verdict.json"):
        assert (out / name).exists(), f"missing {name}"

    verdict = json.loads((out / "verdict.json").read_text())
    # An honest KILL or a HOLDS are BOTH successful outcomes.
    assert verdict["gate_comparison"] in ("holds", "kill")
    # Two surprising_failure events; the stat channel writes on them.
    assert verdict["surprising_failure_total"] == 2
    assert verdict["correction_writes"] >= 1  # the negation correction fired in-scope
    # THE gap: new-gate capture recall of the surprising_failure population >= the motif's.
    assert verdict["capture_recall_new"] is not None
    assert verdict["capture_recall_motif"] is not None
    assert verdict["capture_recall_new"] >= verdict["capture_recall_motif"]
    # motif captured only one of the two (the resolved one); the stat channel covers both.
    assert verdict["motif_captured"] == 1
    assert verdict["capture_recall_new"] == 1.0


def test_refuses_bench_results(tmp_path):
    trace = tmp_path / "bus-events.jsonl"
    _write_synthetic_trace(trace)
    bad = tmp_path / "bench" / "results" / "out"  # path containing bench/results -> refusal
    proc = _run(trace, bad)
    assert proc.returncode != 0
    assert not (bad / "report.md").exists()  # guard fires before any output is written
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


def test_kill_logic_mechanical():
    """The kill is MECHANICAL, not massaged: correction false-positives that drag the new
    gate's precision below the motif's flip gate_comparison to 'kill'; stat writes that dilute
    the motif's novelty junk keep it 'holds'. Probe the pure predicate directly (real-trace-
    shaped fire counts: motif 67 = 53 resolved + 14 novelty)."""
    # 100 correction writes, 0 true-positives -> new precision collapses below motif's -> KILL.
    killed = _gate_metrics(
        resolved=53, novelty=14, stuck=0, stat_writes=0,
        correction_writes=100, correction_tp=0,
        sf_total=71, sf_covered=71, motif_captured=66,
    )
    assert killed["gate_comparison"] == "kill"
    assert killed["new_gate_precision"] < killed["motif_precision"]
    line = _verdict_line(killed)
    assert "KILL" in line and "revert to motifs" in line

    # Same motif baseline, but stat writes (precise by quadrant) + mostly-true corrections
    # dilute the novelty junk -> new precision >= motif -> HOLDS (proves it is not always-kill).
    held = _gate_metrics(
        resolved=53, novelty=14, stuck=0, stat_writes=50,
        correction_writes=8, correction_tp=3,
        sf_total=71, sf_covered=71, motif_captured=66,
    )
    assert held["gate_comparison"] == "holds"
    assert held["new_gate_precision"] >= held["motif_precision"]
    assert "HOLDS" in _verdict_line(held)


def test_sensitivity_both_gradings_and_absolute_junk():
    """MAJOR 1: the capture-precision KILL is methodology-sensitive. On the real-trace-
    shaped fire counts (motif 67 = 53 resolved + 14 novelty; 22 stat writes; 8 scoped
    corrections, 3 census TPs) the shipped quadrant proxy HOLDS by a thread, but re-grading
    the 22 stat writes at the motif's OWN novelty base rate (not assumed-1.0) flips it to
    KILL — and absolute junk writes RISE 14 -> 19 even as the rate 'improves' (denominator
    dilution). Both gradings + absolute counts must be computed and reported verbatim."""
    m = _gate_metrics(
        resolved=53, novelty=14, stuck=0, stat_writes=22,
        correction_writes=8, correction_tp=3,
        sf_total=71, sf_covered=71, motif_captured=66,
    )
    # (1) shipped quadrant proxy: HOLDS by a thread (0.804 >= 0.791).
    assert m["gate_comparison"] == "holds"
    assert abs(m["new_gate_precision"] - 0.804) < 0.005
    assert abs(m["motif_precision"] - 0.791) < 0.005
    # (2) motif-base-rate re-grade of the stat writes: flips to KILL (0.757 < 0.791).
    assert m["gate_comparison_regrade"] == "kill"
    assert abs(m["new_gate_precision_regrade"] - 0.757) < 0.005
    # absolute junk writes ROSE 14 -> 19 (delta +5) though the rate fell by dilution.
    assert m["junk_write_count_motif"] == 14
    assert m["junk_write_count_new"] == 19
    assert m["junk_write_count_delta"] == 5
    # capture recall is proxy-INDEPENDENT (the defensible win: 71/71 vs 66/71).
    assert m["capture_recall_new"] == 1.0
    assert m["capture_recall_new"] > m["capture_recall_motif"]
    # the sensitivity line reports BOTH verdicts verbatim (no massaging toward either).
    line = _sensitivity_verdict_line(m)
    assert "HOLDS under the quadrant proxy" in line
    assert "KILL under the motif-base-rate re-grade" in line
    assert "proxy-INDEPENDENT" in line
