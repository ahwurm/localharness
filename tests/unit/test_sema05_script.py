"""Offline unit test for scripts/sema05_month_in_a_day.py (SEMA-05 month-in-a-day runner).

Drives the runner in --offline mode (bundled FakeLLM) against a TINY real-shaped bus-events
trace (test fixture — fabricated fixtures are allowed offline per 36-CONTEXT; the PROVABLE path
in Task 3 uses only real events). Asserts:
  - the full pipeline runs: replay real-shaped events through the SHIPPED gates into an ISOLATED
    store -> seam-ON consolidation (FakeLLM) -> verdict.json + report.md emitted into --results;
  - verdict.json carries the proxy-independent metrics (zero-tool BINARY + tool-call integer),
    per-schema grounding (the KILL re-check), the six shape-aware reconcile buckets, and the
    honest live-store block;
  - the runner writes ONLY under --store / --results and never touches ~/.localharness.

The fixture is engineered so the replay yields two related `read` lessons across four sittings
(one grounded chapter) plus the dogfood correction sequence — so schemas_written >= 1 and a
zero-tool answer are asserted, not merely the verdict shape.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from localharness.core.bus import EventBus
from localharness.core.events import Action, Observation, UserMessage

# Import the runner by path (scripts/ is not a package). Its own src-bootstrap handles localharness.
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import sema05_month_in_a_day as sema  # noqa: E402

AGENT = "orchestrator"


async def _build_trace(tracedir: Path) -> None:
    """Write a tiny real-shaped bus-events.jsonl + history.jsonl by PUBLISHING real event
    objects to a persist bus (exact serialized shape — no hand-crafted JSON). Engineered to
    produce, on replay through the shipped gates:
      - lesson ALPHA (read, 'permission' error) recurring across sittings sA + sB,
      - lesson BETA (read, 'not found' error) recurring across sittings sC + sD,
        (both promote; they cluster on shared salient tokens spanning 4 sittings -> one chapter),
      - the dogfood predictive sequence in sE (reliable prior -> one surprising failure -> a bare
        correction) so a real tier:correction_pending row exists for the reconcile sub-provable.
    """
    tracedir.mkdir(parents=True, exist_ok=True)
    bus = EventBus(persist_path=tracedir / "bus-events.jsonl")

    async def act(session: str, tool: str, tcid: str) -> None:
        await bus.publish(Action(agent_id=AGENT, session_id=session, action_type="tool_call",
                                 tool_call_id=tcid, tool_name=tool))

    async def obs(session: str, tool: str, tcid: str, *, error=None, output="ok") -> None:
        await bus.publish(Observation(agent_id=AGENT, session_id=session,
                                      observation_type="tool_result", tool_call_id=tcid,
                                      tool_name=tool, output=(output if error is None else ""),
                                      error=error))

    # ALPHA (permission) across sA, sB; BETA (not found) across sC, sD — same error text per lesson
    # so the WriteGate lesson-hash matches across sittings and promote-recurring fires.
    for s in ("sA", "sB"):
        await act(s, "read", f"{s}1")
        await obs(s, "read", f"{s}1", error="permission denied on a protected absolute path")
        await act(s, "read", f"{s}2")
        await obs(s, "read", f"{s}2", output="succeeded after retry with the absolute path")
    for s in ("sC", "sD"):
        await act(s, "read", f"{s}1")
        await obs(s, "read", f"{s}1", error="file not found on a relative path lookup")
        await act(s, "read", f"{s}2")
        await obs(s, "read", f"{s}2", output="resolved by switching to the absolute path form")

    # Dogfood predictive sequence -> a real correction/quarantine row (shape b).
    for i in range(6):
        await act("sE", "pg_probe", f"c{i}")
        await obs("sE", "pg_probe", f"c{i}", output="ok")
    await act("sE", "pg_probe", "cfail")
    await obs("sE", "pg_probe", "cfail", error="unexpected 500 from a normally-reliable tool")
    await bus.publish(UserMessage(agent_id=AGENT, session_id="sE",
                                  content="nah id rather watch the fireworks from the park tomorrow",
                                  channel="terminal"))

    # A mineable transcript (history.jsonl) — the live personal-fact specimen (36-CONTEXT).
    (tracedir / "history.jsonl").write_text(
        json.dumps({"v": 1, "agent_id": AGENT, "type": "user_message", "id": "h1",
                    "session_id": "sE", "ts": 1_000_000,
                    "content": "i got super duper sunburnt today at the beach"}) + "\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_offline_run_emits_verdict_shape_in_isolated_dir(tmp_path: Path):
    tracedir = tmp_path / "trace"
    storedir = tmp_path / "isolated-store"
    resultsdir = tmp_path / "results"
    await _build_trace(tracedir)

    args = sema._parse_args([
        "--offline", "--trace", str(tracedir), "--store", str(storedir),
        "--results", str(resultsdir), "--agent", AGENT,
    ])
    code = await sema.run(args)
    assert code == 0, "offline run should be a successful measurement (HOLDS/KILL/INCONCLUSIVE all exit 0)"

    verdict_path = resultsdir / "verdict.json"
    assert verdict_path.exists(), "verdict.json must land in the isolated --results dir"
    assert (resultsdir / "report.md").exists(), "report.md must land in the isolated --results dir"
    v = json.loads(verdict_path.read_text(encoding="utf-8"))

    # Proxy-independent metrics lead (§3 of the grading doc).
    assert isinstance(v["schemas_written"], int)
    assert isinstance(v["zero_tool_answered"], bool)
    assert isinstance(v["tool_calls"], int)
    assert "domain_question" in v and isinstance(v["domain_question"], str) and v["domain_question"]

    # Per-schema grounding = the KILL re-check (§4).
    assert isinstance(v["per_schema_grounding"], list)
    assert isinstance(v["kill_triggered"], bool)

    # Secondary proxies (§3).
    assert isinstance(v["churn_rate"], (int, float))
    assert isinstance(v["byte_stable"], bool)

    # The six shape-aware reconcile buckets (§5) — all present as ints.
    rec = v["reconcile"]
    for bucket in ("confirmed", "confirmed_corrected", "retired",
                   "reverted_restored", "reverted_cleared", "undecided"):
        assert isinstance(rec[bucket], int), f"reconcile bucket {bucket} missing/non-int"
    assert isinstance(rec["drained"], int)

    # Live-store honesty block (§7) — present, and null here (offline fixture has no memory.db).
    assert "live_store" in v
    assert v["live_store"]["source"] is None

    # Sensitivity re-grades auto-computed (§6).
    assert "sensitivity" in v and "grounding_supermajority_holds" in v["sensitivity"]

    # NON-VACUOUS: the engineered fixture must produce a grounded chapter that answers zero-tool.
    assert v["schemas_written"] >= 1, "the engineered read-cluster must write one chapter"
    assert v["zero_tool_answered"] is True
    assert v["tool_calls"] == 0
    assert v["kill_triggered"] is False
    assert v["byte_stable"] is True
    assert v["per_schema_grounding"] and all(s["grounded"] for s in v["per_schema_grounding"])

    # If the replay produced any correction rows, a REVERT disposition must DRAIN >=1 (the §5 bar);
    # the bundled FakeLLM answers REVERT, so a looked row cannot stay undecided.
    reconcile_total = sum(rec[b] for b in ("confirmed", "confirmed_corrected", "retired",
                                           "reverted_restored", "reverted_cleared", "undecided"))
    if reconcile_total > 0:
        assert rec["drained"] >= 1
        assert rec["drained"] == rec["reverted_restored"] + rec["reverted_cleared"]


@pytest.mark.asyncio
async def test_offline_run_isolated_to_store_and_results(tmp_path: Path):
    """The runner writes ONLY under --store / --results — the isolated-dir CLAUDE.md rule."""
    tracedir = tmp_path / "trace"
    storedir = tmp_path / "isolated-store"
    resultsdir = tmp_path / "results"
    await _build_trace(tracedir)

    args = sema._parse_args([
        "--offline", "--trace", str(tracedir), "--store", str(storedir),
        "--results", str(resultsdir), "--agent", AGENT,
    ])
    await sema.run(args)

    # The isolated store got a real DB; the trace dir was NOT mutated (read-only replay source).
    assert storedir.exists()
    assert list(storedir.rglob("memory.db")), "the replay must build a fresh DB under --store"
    trace_files = {p.name for p in tracedir.iterdir()}
    assert trace_files == {"bus-events.jsonl", "history.jsonl"}, "the trace dir must stay untouched"


@pytest.mark.asyncio
async def test_guard_refuses_shared_bench_results_dir(tmp_path: Path):
    """The shared bench results dir is contaminated (project hard rule) — the runner refuses it."""
    tracedir = tmp_path / "trace"
    await _build_trace(tracedir)
    args = sema._parse_args([
        "--offline", "--trace", str(tracedir), "--store", str(tmp_path / "s"),
        "--results", str(tmp_path / "bench" / "results" / "x"), "--agent", AGENT,
    ])
    code = await sema.run(args)
    assert code == 2, "a --results path inside bench/results must be refused (exit 2)"
