"""Offline unit test for scripts/sema05_month_in_a_day.py (SEMA-05 month-in-a-day runner).

Covers BOTH accumulation modes offline (fabricated fixtures are allowed in tests per
36-CONTEXT; the PROVABLE in Task 3 uses only real data):

1. LIVE-SESSION ACCUMULATION (--history; THE provable method, owner steer 2026-07-05 17:53):
   a tiny synthetic history.jsonl of "owner month queries" is re-run through the REAL
   AgentLoop (real bus, real gates, real builtin tools, real memory store; fake LLM at the
   client seam only) — one fresh sitting per original day — then the seam-ON pass + the
   LOCKED grading. Asserts day->sitting grouping, real-loop/real-tool invocation (sessions +
   tool_observations rows), the --max-turns-per-day cap, and the verdict shape + the
   method-amendment disclosure.

2. TRACE REPLAY (--trace; legacy cross-check): a real-shaped bus-events trace replayed
   through the shipped gates. Asserts the verdict shape (proxy-independent metrics,
   per-schema KILL re-check, six shape-aware reconcile buckets incl. the shape-(b) DRAIN,
   live-store honesty) and that the runner writes ONLY under --store / --results.
"""
from __future__ import annotations

import json
import sqlite3
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

    # Snapshot the trace dir (bytes + listing) so we can prove the replay was READ-ONLY.
    before = {p.name: p.read_bytes() for p in tracedir.rglob("*") if p.is_file()}

    args = sema._parse_args([
        "--offline", "--trace", str(tracedir), "--store", str(storedir),
        "--results", str(resultsdir), "--agent", AGENT,
    ])
    await sema.run(args)

    # The isolated store got a real DB; the outputs landed only under --results.
    assert storedir.exists()
    assert list(storedir.rglob("memory.db")), "the replay must build a fresh DB under --store"
    assert {p.name for p in resultsdir.iterdir()} == {"verdict.json", "report.md"}

    # The trace dir is byte-for-byte unchanged — the real trace is only ever read, never mutated.
    after = {p.name: p.read_bytes() for p in tracedir.rglob("*") if p.is_file()}
    assert after == before, "the trace dir must stay byte-identical (read-only replay source)"


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


# ---------------------------------------------------------------------------
# Live-session accumulation mode (--history) — the provable's method (owner steer)
# ---------------------------------------------------------------------------

def _write_month_history(path: Path, *, days: int = 2) -> None:
    """A tiny synthetic 'owner month' history.jsonl (fixture — tests may fabricate): the SAME
    two queries repeated on `days` distinct calendar days (3 days apart), plus one assistant
    row and one foreign-agent row that extraction must ignore. The queries make the bundled
    offline loop-LLM read a missing path (real ReadTool error) then recover — so the shipped
    WriteGate captures a real resolved_error lesson per query per sitting, and recurrence
    across sittings promotes them (the honest ≥2-sittings stability warrant)."""
    base = 1_750_000_000
    rows = []
    for d in range(days):
        for i, tag in enumerate(("alpha", "beta")):
            rows.append({"v": 1, "agent_id": AGENT, "type": "user_message", "id": f"m{d}{i}",
                         "session_id": f"orig-{d}", "ts": base + d * 3 * 86400 + i * 60,
                         "content": f"please read the {tag} file"})
    rows.append({"v": 1, "agent_id": AGENT, "type": "assistant_message", "id": "a0",
                 "session_id": "orig-0", "ts": base + 5, "content": "sure thing"})
    rows.append({"v": 1, "agent_id": "someone-else", "type": "user_message", "id": "f0",
                 "session_id": "orig-0", "ts": base + 6, "content": "not our agent's query"})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_live_mode_groups_days_into_sittings_and_drives_real_loop(tmp_path: Path):
    """--history mode: each original day becomes one fresh sitting driven through the REAL
    AgentLoop; real tools execute; the seam-ON pass then grades with the LOCKED bars."""
    hist = tmp_path / "scratch" / "history.jsonl"
    _write_month_history(hist)
    storedir, resultsdir = tmp_path / "store", tmp_path / "results"

    args = sema._parse_args([
        "--offline", "--history", str(hist), "--store", str(storedir),
        "--results", str(resultsdir), "--agent", AGENT,
    ])
    code = await sema.run(args)
    assert code == 0

    v = json.loads((resultsdir / "verdict.json").read_text(encoding="utf-8"))
    # Day -> sitting grouping: 2 original days -> 2 fresh sittings, distinct session ids,
    # 2 turns each (the assistant + foreign-agent rows were ignored).
    assert v["method"] == "live_session_accumulation"
    assert len(v["sittings"]) == 2
    assert len({s["session_id"] for s in v["sittings"]}) == 2
    assert [s["turns"] for s in v["sittings"]] == [2, 2]

    # REAL-loop invocation: real sessions rows (create/end_session per sitting) and real
    # tool_observations (the loop dispatched REAL read tools: 2 days x 2 queries x 2 reads).
    db = next(storedir.rglob("memory.db"))
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    n_sessions = con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    n_obs = con.execute("SELECT COUNT(*) FROM tool_observations").fetchone()[0]
    con.close()
    assert n_sessions == 2
    assert n_obs >= 8

    # Verdict shape intact (same LOCKED bars as trace mode) + non-vacuous: the repeated real
    # tool-error lessons cluster across the two sittings into one grounded zero-tool chapter.
    assert v["schemas_written"] >= 1
    assert v["zero_tool_answered"] is True and v["tool_calls"] == 0
    assert v["kill_triggered"] is False and v["byte_stable"] is True
    for bucket in ("confirmed", "confirmed_corrected", "retired",
                   "reverted_restored", "reverted_cleared", "undecided"):
        assert isinstance(v["reconcile"][bucket], int)
    assert "sensitivity" in v and "live_store" in v

    # The method amendment is disclosed in the report, quoting the LOCKED grading timestamp.
    rep = (resultsdir / "report.md").read_text(encoding="utf-8")
    assert "Method amendment" in rep
    assert "2026-07-05T17:43:47Z" in rep


@pytest.mark.asyncio
async def test_live_mode_max_turns_per_day_cap_and_mode_exclusivity(tmp_path: Path):
    """--max-turns-per-day bounds each sitting; --history/--trace are mutually exclusive."""
    hist = tmp_path / "scratch" / "history.jsonl"
    _write_month_history(hist)

    args = sema._parse_args([
        "--offline", "--history", str(hist), "--store", str(tmp_path / "store"),
        "--results", str(tmp_path / "results"), "--agent", AGENT,
        "--max-turns-per-day", "1",
    ])
    code = await sema.run(args)
    assert code == 0  # INCONCLUSIVE (1 lesson -> no 2-member cluster) is still a measurement
    v = json.loads((tmp_path / "results" / "verdict.json").read_text(encoding="utf-8"))
    assert [s["turns"] for s in v["sittings"]] == [1, 1]
    assert v["max_turns_per_day"] == 1

    # Exactly one accumulation mode: both --history and --trace -> guard refusal (exit 2).
    both = sema._parse_args([
        "--offline", "--history", str(hist), "--trace", str(tmp_path / "scratch"),
        "--store", str(tmp_path / "s2"), "--results", str(tmp_path / "r2"), "--agent", AGENT,
    ])
    assert await sema.run(both) == 2
    # Neither mode -> same refusal.
    neither = sema._parse_args([
        "--offline", "--store", str(tmp_path / "s3"),
        "--results", str(tmp_path / "r3"), "--agent", AGENT,
    ])
    assert await sema.run(neither) == 2
