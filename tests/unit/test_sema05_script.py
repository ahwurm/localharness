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
import subprocess
import sys
import textwrap
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

    # A mineable transcript (history.jsonl): two DAY-DISTINCT subagent lines in two sessions so
    # MOVE-2 mining writes two sem/ atoms that cluster across ≥2 sittings into one chapter, plus
    # the live personal-fact specimen (a lone atom — no cluster, exercises single-atom mining).
    mine_rows = [
        {"v": 1, "agent_id": AGENT, "type": "user_message", "id": "h1", "session_id": "s-mon",
         "ts": 1_000_000, "content": "i am building a summarizer subagent for the harness"},
        {"v": 1, "agent_id": AGENT, "type": "user_message", "id": "h2", "session_id": "s-tue",
         "ts": 1_000_100, "content": "i am building a citation subagent for the harness"},
        {"v": 1, "agent_id": AGENT, "type": "user_message", "id": "h3", "session_id": "s-wed",
         "ts": 1_000_200, "content": "i got super duper sunburnt today at the beach"},
    ]
    (tracedir / "history.jsonl").write_text(
        "\n".join(json.dumps(r) for r in mine_rows) + "\n", encoding="utf-8",
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

_SUBAGENTS = ["summarizer", "citation", "screenshot", "linter", "formatter"]


def _write_month_history(path: Path, *, days: int = 2) -> None:
    """A tiny synthetic 'owner month' history.jsonl (fixture — tests may fabricate): two queries
    per day across `days` distinct calendar days (3 days apart), plus one assistant row and one
    foreign-agent row that extraction must ignore. Day 0 carries the LEGACY root alias
    agent_id='default' (the real month's shape); day 1 carries 'orchestrator' — extraction must
    unify them (B1). q0 is topic-coherent (SUBAGENTS) but DAY-DISTINCT so post-re-center mining
    writes one sem/ atom per day that clusters across ≥2 sittings into a chapter; q1 is a read
    task that makes the offline loop read a missing path then recover (real tool observations)."""
    base = 1_750_000_000
    rows = []
    for d in range(days):
        alias = "default" if d % 2 == 0 else AGENT  # mixed root aliases across days (B1)
        rows.append({"v": 1, "agent_id": alias, "type": "user_message", "id": f"m{d}0",
                     "session_id": f"orig-{d}", "ts": base + d * 3 * 86400,
                     "content": f"i am building a {_SUBAGENTS[d % len(_SUBAGENTS)]} subagent for the harness"})
        rows.append({"v": 1, "agent_id": alias, "type": "user_message", "id": f"m{d}1",
                     "session_id": f"orig-{d}", "ts": base + d * 3 * 86400 + 60,
                     "content": "please read the project notes file"})
    rows.append({"v": 1, "agent_id": AGENT, "type": "assistant_message", "id": "a0",
                 "session_id": "orig-0", "ts": base + 5, "content": "sure thing"})
    rows.append({"v": 1, "agent_id": "someone-else", "type": "user_message", "id": "f0",
                 "session_id": "orig-0", "ts": base + 6, "content": "not our agent's query"})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_extract_day_queries_unifies_root_aliases(tmp_path: Path):
    """BLOCKER 1: the real month is agent_id='default' (135 msgs/14 days — the 33.1 rename
    migrated the DB, not old history lines). Requesting either root alias must capture
    {None, 'default', 'orchestrator'} as ONE identity; a non-root agent keeps the narrow filter."""
    hist = tmp_path / "history.jsonl"
    rows = [
        {"type": "user_message", "agent_id": "default", "ts": 1_750_000_000, "content": "q-default"},
        {"type": "user_message", "agent_id": "orchestrator", "ts": 1_750_000_060, "content": "q-orch"},
        {"type": "user_message", "ts": 1_750_000_120, "content": "q-none"},  # no agent_id field
        {"type": "user_message", "agent_id": "someone-else", "ts": 1_750_000_180, "content": "q-foreign"},
    ]
    hist.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    for root in ("orchestrator", "default"):
        got = sema._extract_day_queries(hist, root)
        queries = [q for _, day_q in got for q in day_q]
        assert queries == ["q-default", "q-orch", "q-none"], f"alias unification failed for {root}"
    got = sema._extract_day_queries(hist, "someone-else")
    assert [q for _, day_q in got for q in day_q] == ["q-none", "q-foreign"]


def test_live_store_paths_denied_by_permissions():
    """BLOCKER 3: the composed agent config must deny write/edit/bash_exec on the live agent
    stores (~/.localharness/agents/*) — belt and braces over the isolated-store guarantee."""
    from localharness.agent.permissions import PermissionEvaluator
    from localharness.core.types import ToolCall

    cfg = sema._root_agent_config("orchestrator")
    home = str(Path.home() / ".localharness" / "agents")
    ev = PermissionEvaluator()
    assert ev.evaluate(ToolCall(name="bash_exec",
                                arguments={"command": f"rm -rf {home}/orchestrator"}, id="t"),
                       cfg.permissions).denied
    assert ev.evaluate(ToolCall(name="write",
                                arguments={"path": f"{home}/orchestrator/memory.db", "content": "x"},
                                id="t"), cfg.permissions).denied
    assert ev.evaluate(ToolCall(name="edit",
                                arguments={"path": f"{home}/x/history.jsonl", "old": "a", "new": "b"},
                                id="t"), cfg.permissions).denied
    # SAFETY B1 residual: ~-prefixed forms must ALSO be caught — fnmatch runs on the RAW
    # pre-expansion argument strings (write/edit expanduser AFTER the permission check).
    assert ev.evaluate(ToolCall(name="bash_exec",
                                arguments={"command": "cat ~/.localharness/agents/orchestrator/memory.db"},
                                id="t"), cfg.permissions).denied
    assert ev.evaluate(ToolCall(name="write",
                                arguments={"path": "~/.localharness/agents/orchestrator/MEMORY.md",
                                           "content": "x"}, id="t"), cfg.permissions).denied
    assert ev.evaluate(ToolCall(name="edit",
                                arguments={"path": "~/.localharness/agents/x/history.jsonl",
                                           "old": "a", "new": "b"}, id="t"), cfg.permissions).denied
    # The P-A capability floor is applied too (web_* denied for the bash-holding root).
    assert "web_fetch" in cfg.tools.deny
    # Innocuous calls stay allowed.
    assert not ev.evaluate(ToolCall(name="bash_exec", arguments={"command": "echo hi"}, id="t"),
                           cfg.permissions).denied


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

    # REAL-loop invocation: real sessions rows (2 sittings + the probe turn's session) and
    # real tool_observations (the loop dispatched REAL read tools: 2 days x 2 queries x 2 reads).
    db = next(storedir.rglob("memory.db"))
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    n_sessions = con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    n_obs = con.execute("SELECT COUNT(*) FROM tool_observations").fetchone()[0]
    con.close()
    assert n_sessions == 3  # 2 sittings + sema05-probe
    assert n_obs >= 8

    # BLOCKER 2: the zero-tool metric comes from a REAL probe turn (not a substring match):
    # the domain question was actually asked through the loop against the consolidated store,
    # and tool_calls is the count of real Action(tool_call) events that turn emitted.
    assert v["probe"]["session_id"] == "sema05-probe"
    assert v["probe"]["question"] == v["domain_question"]
    assert isinstance(v["probe"]["answer"], str) and v["probe"]["answer"]
    assert v["probe"]["tool_calls"] == 0
    assert v["tool_calls"] == v["probe"]["tool_calls"]
    assert v["probe"]["turn_failed"] is False  # a TurnFailed probe can never mint HOLDS
    assert isinstance(v["probe"]["chapter_content_in_answer"], bool)
    assert v["probe"]["chapter_content_in_answer"] is True  # the fake echoes the injected chapter line

    # Sensitivity disclosure: per-cluster sorted session lists make plus1 legitimacy
    # hostile-read-verifiable from the artifact alone.
    assert isinstance(v["sensitivity"]["stricter_cluster_sessions"], list)

    # Verdict shape intact (same LOCKED bars as trace mode) + non-vacuous: the repeated real
    # tool-error lessons cluster across the two sittings into one grounded zero-tool chapter.
    assert v["schemas_written"] >= 1
    assert v["zero_tool_answered"] is True and v["tool_calls"] == 0
    assert v["kill_triggered"] is False and v["byte_stable"] is True
    for bucket in ("confirmed", "confirmed_corrected", "retired",
                   "reverted_restored", "reverted_cleared", "undecided"):
        assert isinstance(v["reconcile"][bucket], int)
    assert "live_store" in v

    # MAJOR 5 boundary: a 2-session cluster with its schema already written must NOT pass the
    # stricter cluster_min_sessions+1 (=3) re-grade — the schema's own "cluster:..." provenance
    # previously counted as a fake third session (sensitivity pollution).
    assert v["sensitivity"]["cluster_min_sessions_plus1_holds"] is False

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


@pytest.mark.asyncio
async def test_kill_file_stops_sittings_before_any_turn(tmp_path: Path):
    """MAJOR 6b: a KILL file present at the top of the day loop stops accumulation — killed
    runs must not keep iterating and inflating turns_driven."""
    hist = tmp_path / "scratch" / "history.jsonl"
    _write_month_history(hist)
    storedir = tmp_path / "store"
    storedir.mkdir()
    (storedir / "KILL").write_text("")

    args = sema._parse_args([
        "--offline", "--history", str(hist), "--store", str(storedir),
        "--results", str(tmp_path / "results"), "--agent", AGENT,
    ])
    code = await sema.run(args)
    assert code == 1  # zero turns driven -> honest processing failure, no fabricated verdict
    assert not (tmp_path / "results" / "verdict.json").exists()


@pytest.mark.asyncio
async def test_kill_file_aborts_grading_with_honest_verdict(tmp_path: Path):
    """MAJOR 6a: a KILL file present before the grading-phase LLM work (consolidation +
    reconcile looks) aborts with an honest ABORTED verdict + exit 1 — never a silent skip."""
    tracedir = tmp_path / "trace"
    await _build_trace(tracedir)
    storedir = tmp_path / "store"
    storedir.mkdir()
    (storedir / "KILL").write_text("")

    args = sema._parse_args([
        "--offline", "--trace", str(tracedir), "--store", str(storedir),
        "--results", str(tmp_path / "results"), "--agent", AGENT,
    ])
    code = await sema.run(args)
    assert code == 1
    v = json.loads((tmp_path / "results" / "verdict.json").read_text(encoding="utf-8"))
    assert v["verdict"] == "ABORTED"


@pytest.mark.asyncio
async def test_failed_probe_turn_cannot_mint_holds(tmp_path: Path, monkeypatch):
    """Probe-gate BLOCKER: run_turn NEVER returns an empty string (error/kill summaries are
    non-empty prose), so bool(answer) was dead code — a failed probe turn minted HOLDS. The
    gate must consume the turn's TurnFailed event: a probe whose LLM dies -> INCONCLUSIVE."""
    orig = sema._OfflineLoopLLM.stream_complete

    async def failing(self, messages=None, tools=None, on_token=None):
        last_user = next((m for m in reversed(messages or []) if m.get("role") == "user"), {})
        task = str(last_user.get("content", ""))
        if task.startswith(("What has the harness learned", "What durable domain knowledge")):
            raise ConnectionError("vLLM died mid-probe")
        return await orig(self, messages, tools, on_token)

    monkeypatch.setattr(sema._OfflineLoopLLM, "stream_complete", failing)
    hist = tmp_path / "scratch" / "history.jsonl"
    _write_month_history(hist)
    args = sema._parse_args([
        "--offline", "--history", str(hist), "--store", str(tmp_path / "store"),
        "--results", str(tmp_path / "results"), "--agent", AGENT,
    ])
    assert await sema.run(args) == 0  # still a measurement — an honest INCONCLUSIVE
    v = json.loads((tmp_path / "results" / "verdict.json").read_text(encoding="utf-8"))
    assert v["schemas_written"] >= 1          # the chapter exists...
    assert v["probe"]["turn_failed"] is True  # ...but the probe turn FAILED
    assert v["zero_tool_answered"] is False
    assert v["verdict"] == "INCONCLUSIVE"     # never HOLDS on a failed probe


@pytest.mark.asyncio
async def test_off_chapter_answer_demotes_to_inconclusive(tmp_path: Path, monkeypatch):
    """Attribution joins the HOLDS conjunction (orchestrator ruling): the locked bar is
    'answered BY the Knowledge line' — a zero-tool answer that ignores the chapter is an
    honest not-proven (false-negative is the only permitted error direction)."""
    orig = sema._OfflineLoopLLM.stream_complete

    async def off_chapter(self, messages=None, tools=None, on_token=None):
        last_user = next((m for m in reversed(messages or []) if m.get("role") == "user"), {})
        task = str(last_user.get("content", ""))
        if task.startswith(("What has the harness learned", "What durable domain knowledge")):
            return sema._Msg(content="The weather is lovely."), None  # zero-tool, OFF-chapter
        return await orig(self, messages, tools, on_token)

    monkeypatch.setattr(sema._OfflineLoopLLM, "stream_complete", off_chapter)
    hist = tmp_path / "scratch" / "history.jsonl"
    _write_month_history(hist)
    args = sema._parse_args([
        "--offline", "--history", str(hist), "--store", str(tmp_path / "store"),
        "--results", str(tmp_path / "results"), "--agent", AGENT,
    ])
    assert await sema.run(args) == 0
    v = json.loads((tmp_path / "results" / "verdict.json").read_text(encoding="utf-8"))
    assert v["schemas_written"] >= 1
    assert v["probe"]["tool_calls"] == 0
    assert v["probe"]["turn_failed"] is False
    assert v["probe"]["chapter_content_in_answer"] is False
    assert v["zero_tool_answered"] is False
    assert v["verdict"] == "INCONCLUSIVE"


@pytest.mark.asyncio
async def test_sitting_turnfailed_rate_gate_stamps_invalid(tmp_path: Path, monkeypatch):
    """MOVE 0b: a sitting whose TurnFailed rate exceeds 20% aborts the run loudly and stamps
    verdict INVALID (measurement failure != INCONCLUSIVE). The SEMA-05 P0: 13/15 days failed
    EVERY turn on the second-system-message HTTP-400, yet the verdict graded a 2-day store as
    15 days with no failure signal. The probe already has a TurnFailed gate; sittings mirror it."""
    orig = sema._OfflineLoopLLM.stream_complete

    async def dying_accumulation(self, messages=None, tools=None, on_token=None):
        last_user = next((m for m in reversed(messages or []) if m.get("role") == "user"), {})
        task = str(last_user.get("content", ""))
        if task.startswith(("What has the harness learned", "What durable domain knowledge")):
            return await orig(self, messages, tools, on_token)  # the probe (never reached) still works
        raise ConnectionError("400 — System message must be at the beginning")  # every driven turn dies

    monkeypatch.setattr(sema._OfflineLoopLLM, "stream_complete", dying_accumulation)
    hist = tmp_path / "scratch" / "history.jsonl"
    _write_month_history(hist)
    args = sema._parse_args([
        "--offline", "--history", str(hist), "--store", str(tmp_path / "store"),
        "--results", str(tmp_path / "results"), "--agent", AGENT,
    ])
    code = await sema.run(args)
    assert code == 1, "a measurement failure is NOT a successful measurement (exit 1, like ABORTED)"
    v = json.loads((tmp_path / "results" / "verdict.json").read_text(encoding="utf-8"))
    assert v["verdict"] == "INVALID"
    assert "TurnFailed" in v["reason"]
    # The failing sitting is NAMED (per-stage/site attribution is the point).
    assert v["invalid_sitting"]["session_id"].startswith("sema05-")
    assert v["invalid_sitting"]["failed"] >= 1 and v["invalid_sitting"]["turns"] >= 1
    # A dead store was NEVER graded as if it held knowledge (no proxy fields fabricated).
    assert "schemas_written" not in v and "zero_tool_answered" not in v


@pytest.mark.asyncio
async def test_watchdog_aborts_before_any_sitting(tmp_path: Path, monkeypatch):
    """Machine-safety: MemAvailable below threshold -> honest abort (exit 1) BEFORE any
    sitting fires a live prefill. (The KILL mid-sitting per-turn break is inspection-covered:
    the same predicate runs at the top of every turn iteration in _run_sittings.)"""
    monkeypatch.setattr(sema, "_mem_available_gib", lambda: 5.0)  # far below the 30 GiB bar
    hist = tmp_path / "scratch" / "history.jsonl"
    _write_month_history(hist)
    # NOT --offline: the watchdog only guards the live path. The abort fires before any
    # client construction, so the fake endpoint is never contacted.
    args = sema._parse_args([
        "--history", str(hist), "--store", str(tmp_path / "store"),
        "--results", str(tmp_path / "results"), "--agent", AGENT,
        "--model", "fake-model", "--base-url", "http://localhost:1/v1",
    ])
    assert await sema.run(args) == 1
    assert not (tmp_path / "results" / "verdict.json").exists()
    assert not (tmp_path / "store").exists()  # no sitting ever composed


@pytest.mark.asyncio
async def test_guards_refuse_live_store_shaped_paths(tmp_path: Path):
    """minor 9: --results refuses /.localharness/agents/ paths (like --store); --history and
    --trace refuse pointing AT a live-store path directly — a copy is required."""
    hist = tmp_path / "scratch" / "history.jsonl"
    _write_month_history(hist)
    fake_live = tmp_path / ".localharness" / "agents" / "orchestrator"

    # --results under a live agents dir -> refused.
    args = sema._parse_args([
        "--offline", "--history", str(hist), "--store", str(tmp_path / "s"),
        "--results", str(fake_live / "reports"), "--agent", AGENT,
    ])
    assert await sema.run(args) == 2
    # --history pointing at a live-store path -> refused (copy required).
    args = sema._parse_args([
        "--offline", "--history", str(fake_live / "history.jsonl"),
        "--store", str(tmp_path / "s"), "--results", str(tmp_path / "r"), "--agent", AGENT,
    ])
    assert await sema.run(args) == 2
    # --trace pointing at a live-store path -> refused (copy required).
    args = sema._parse_args([
        "--offline", "--trace", str(fake_live / "bus-events.jsonl"),
        "--store", str(tmp_path / "s"), "--results", str(tmp_path / "r"), "--agent", AGENT,
    ])
    assert await sema.run(args) == 2


# ---------------------------------------------------------------------------
# Day-granularity checkpoint/resume — survive a hard freeze, resume without redoing
# completed days AND without changing the graded outcome (the load-bearing invariant).
# ---------------------------------------------------------------------------

# The proxy-independent / structural fields a deterministic resume MUST reproduce exactly.
# Time-based fields (generated_at, duration_s, sittings[].turns provenance) are excluded —
# the offline fake is content-deterministic, so everything below is a wall-clock invariant.
_STRUCTURAL = (
    "zero_tool_answered", "tool_calls", "schemas_written", "kill_triggered",
    "byte_stable", "n_lessons", "reconcile", "drain_ok", "shape_a", "shape_b",
)


def _structural(v: dict) -> dict:
    out = {k: v[k] for k in _STRUCTURAL}
    # per-schema KILL re-check: compare the grounding DISPOSITION, order-independent. The schema
    # KEY itself is deliberately dropped — it is _h8(sorted member lesson keys), and each member
    # key embeds _h8(prior_error) (gate.py:133), whose synthetic offline error text contains the
    # absolute missing path `{store_dir}/absent-*.md`. Baseline and a resumed run run in DIFFERENT
    # store dirs by construction, so those hashes can never coincide across the two — a pure test-
    # setup artifact, not a resume divergence. What must match is the grade: grounded, majority,
    # no unverified numbers, and the same schema COUNT.
    out["schema_count"] = len(v["per_schema_grounding"])
    out["grounding"] = sorted(
        (p["grounded"], p["grounded_majority"], tuple(p["unverified_numbers"]))
        for p in v["per_schema_grounding"]
    )
    # session count is THE forked-id tripwire: a suffixed redo shows up as an extra session.
    out["n_sittings"] = len(v["sittings"])
    out["distinct_sessions"] = len({s["session_id"] for s in v["sittings"]})
    out["sensitivity_holds"] = (
        v["sensitivity"]["grounding_supermajority_holds"],
        v["sensitivity"]["cluster_min_sessions_plus1_holds"],
    )
    return out


def _sessions_count(storedir: Path) -> int:
    db = next(storedir.rglob("memory.db"))
    con = sqlite3.connect(str(db))  # plain connect: the frozen store's WAL may be uncheckpointed
    try:
        return con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    finally:
        con.close()


@pytest.mark.asyncio
async def test_freeze_resume_matches_baseline_without_redoing_done_days(tmp_path: Path):
    """A hard freeze mid-accumulation (box bricks — os._exit, no finally, no end_session) must
    resume on the SAME command: skip durable days, redo the dangling day UNDER THE SAME id, run
    never-started days, then grade to a verdict that equals a clean baseline on every structural
    field. A forked-id regression surfaces as a differing session count (asserted below)."""
    hist = tmp_path / "scratch" / "history.jsonl"
    _write_month_history(hist, days=5)

    # 1) BASELINE — a clean full offline run.
    base_store, base_results = tmp_path / "store_b", tmp_path / "results_b"
    assert await sema.run(sema._parse_args([
        "--offline", "--history", str(hist), "--store", str(base_store),
        "--results", str(base_results), "--agent", AGENT,
    ])) == 0
    baseline = json.loads((base_results / "verdict.json").read_text(encoding="utf-8"))

    # 2) HARD FREEZE a fresh run: a driver hard-exits (os._exit — no finally, no end_session) at
    # the 4th day's close-out, leaving days 1-3 durable + day 4 dangling (day 5 never created).
    # A faithful brick, NOT a graceful KILL (which would write an ABORTED verdict instead).
    kr_store, kr_results = tmp_path / "store_kr", tmp_path / "results_kr"
    driver = tmp_path / "freeze_driver.py"
    driver.write_text(textwrap.dedent(f"""
        import asyncio, os, sys
        sys.path.insert(0, {str(_SCRIPTS)!r})
        import sema05_month_in_a_day as sema
        from localharness.memory.sqlite import MemoryStore
        _n = {{"d": 0}}
        _oc, _oe = MemoryStore.create_session, MemoryStore.end_session
        async def _c(self, session_id, *a, **k):
            await _oc(self, session_id, *a, **k)
            if session_id.startswith("sema05-") and session_id != "sema05-probe":
                _n["d"] += 1
        async def _e(self, session_id, *a, **k):
            if session_id.startswith("sema05-") and session_id != "sema05-probe" and _n["d"] >= 4:
                sys.stdout.flush(); os._exit(137)  # brick at day 4 close-out -> day 4 dangling
            await _oe(self, session_id, *a, **k)
        MemoryStore.create_session, MemoryStore.end_session = _c, _e
        sys.exit(asyncio.run(sema.run(sema._parse_args(sys.argv[1:]))))
    """), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(driver), "--offline", "--history", str(hist),
         "--store", str(kr_store), "--results", str(kr_results), "--agent", AGENT],
        capture_output=True, text=True,
    )
    assert proc.returncode == 137, f"expected a hard-freeze exit 137, got {proc.returncode}: {proc.stderr[-800:]}"
    assert not (kr_results / "verdict.json").exists(), "the frozen run must not have reached grading"

    # The freeze left exactly the dangling shape: 3 durable days, 1 dangling, no probe.
    kr_db = next(kr_store.rglob("memory.db"))
    con = sqlite3.connect(str(kr_db))
    rows = con.execute("SELECT id, ended_at FROM sessions WHERE id LIKE 'sema05-%'").fetchall()
    con.close()
    done_ids = sorted(r[0] for r in rows if r[1] is not None)
    dangling_ids = [r[0] for r in rows if r[1] is None]
    assert len(done_ids) == 3 and len(dangling_ids) == 1, f"unexpected freeze shape: {rows}"
    assert "sema05-probe" not in [r[0] for r in rows]

    # 3) RESUME — re-invoke the SAME command. Skips the 3 durable days, purges + redoes the
    # dangling day UNDER THE SAME id, runs the never-started day 5, then grades to a verdict.
    assert await sema.run(sema._parse_args([
        "--offline", "--history", str(hist), "--store", str(kr_store),
        "--results", str(kr_results), "--agent", AGENT,
    ])) == 0
    resumed = json.loads((kr_results / "verdict.json").read_text(encoding="utf-8"))

    # The dangling day was redone under the SAME id, NOT forked: one row per day + one probe,
    # identical session count to the baseline (a suffixed-id bug would make this 7, not 6).
    assert _sessions_count(kr_store) == _sessions_count(base_store) == 6
    con = sqlite3.connect(str(next(kr_store.rglob("memory.db"))))
    n_dangle = con.execute("SELECT COUNT(*) FROM sessions WHERE id = ?", (dangling_ids[0],)).fetchone()[0]
    con.close()
    assert n_dangle == 1, "the redone day must reuse its id, never mint a second session"

    # 4) The resumed verdict equals the baseline on every structural / proxy-independent field.
    diffs = {k: (bv, resumed_v) for k, bv in _structural(baseline).items()
             if (resumed_v := _structural(resumed)[k]) != bv}
    assert not diffs, f"resume diverged from baseline on: {diffs}"

    # Non-vacuous: the run actually produced a graded, zero-tool-answered chapter over 5 sittings
    # (else the equality above would be a trivial match of two empty verdicts).
    assert resumed["schemas_written"] >= 1 and resumed["zero_tool_answered"] is True
    resumed_sids = {s["session_id"] for s in resumed["sittings"]}
    assert len(resumed["sittings"]) == 5 and len(resumed_sids) == 5
    # Every day the freeze had already made durable reappears in the resumed plan (skipped, not
    # re-driven) — resume neither drops a done day nor forks a new id for the redone one.
    assert set(done_ids).issubset(resumed_sids)
    assert dangling_ids[0] in resumed_sids
