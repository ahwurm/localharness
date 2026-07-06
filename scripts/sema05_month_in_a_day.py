#!/usr/bin/env python3
"""SEMA-05 month-in-a-day provable — replay REAL accumulated events into an ISOLATED store,
run the seam-ON idle pass, and grade against the pre-committed method (36-SEMA05-GRADING.md).

This is the phase gate for the chapter-writer: "100 lessons -> one honest chapter the index
routes through." An honest KILL is a successful outcome. The grading method is committed BEFORE
this ever runs so results cannot move the goalposts (the M1 methodology lesson).

METHOD AMENDMENT (pre-run, owner-directed — Discord 2026-07-05 17:53): the provable does NOT
replay old stored trace/DB shapes. It re-runs the REAL queries the owner ran over the month
LIVE through the harness — the same run_turn hot path the CLI uses — in fresh sittings (one
per original calendar day), so the ≥2-sittings stability gate is satisfied by real temporal
structure and the proof exercises the live end-to-end pipeline. The GRADING DOC IS UNTOUCHED
(committed 2026-07-05T17:43:47Z, before any run); only the accumulation method changed.

TWO ACCUMULATION MODES (exactly one):
  --history <history.jsonl>   LIVE-SESSION ACCUMULATION (the provable's method). Extract the
      owner's real user messages READ-ONLY, group by original day (local box time — the days
      the owner actually lived), and drive each query through the REAL AgentLoop: real bus,
      real WriteGate/PredictiveGate/UserSignalDetector/PredictiveWriteGate, real builtin
      tools, real memory store — against the ISOLATED agent home. One fresh sitting (distinct
      session id; fresh loop/gates/store instances, emulating one process per sitting) per
      original day, with real create_session/end_session close-outs. Root aliases are ONE
      identity ({None, 'default', 'orchestrator'} — the 33.1 rename migrated the DB, not old
      history lines): the real month measures 148 user messages across 15 original days
      (135 legacy-'default' + 13 'orchestrator'; offline dry-run 2026-07-05).
  --trace <dir|bus-events>    LEGACY TRACE REPLAY (retained as an offline cross-check: it
      rehearses the gate shapes — incl. both correction_pending shapes — at zero loop cost).
      Feeds the copied real bus-events through the shipped gates into the isolated store.

THEN (both modes — the LOCKED grading, unchanged):
  1. CONSOLIDATE: seam-ON ConsolidationPass (chapter-writer + mining) with the subject model
     (live) or bundled fakes (--offline), then reconcile_corrections directly so the verdict
     records the six shape-aware disposition buckets.
  2. PROBE (proxy-independent FIRST): is the domain question answered by the "### Knowledge"
     schema line with ZERO tool calls (BINARY)? tool-call count (integer)? Re-apply the KILL
     per schema (majority-token grounding + numeric net) mechanically.
  3. PROXIES + SENSITIVITY: churn_rate + byte-stability (double-render ==); re-grade under a
     stricter grounding bar and a stricter cluster-stability bar.
  4. EMIT verdict.json + report.md into the ISOLATED --results dir (with a Method-amendment
     section + per-sitting turn counts); live-store organic counts read-only as a WATCH ITEM.

MACHINE-SAFETY (binding — this box hard-hung twice in 24h under vLLM prefill): the live path is
attended-only, context-bounded (32k loop window; idle passes char-capped), and gated by a
MemAvailable watchdog checked before EVERY live turn (aborts below ~30 GiB). `touch <store>/KILL`
stops accumulation at the next turn/day boundary and aborts grading-phase LLM work with an
honest ABORTED verdict. Offline never touches vLLM. Tools run FOR REAL during live sittings
(bash/write/edit included; web_* structurally denied) — attended-only is the containment.
RUNBOOK (attended): before the run, `chmod -R a-w ~/.localharness/agents` (the PHYSICAL
write-protection layer for the live stores — the deny-pattern globs are path-form-limited);
restore with `chmod -R u+w ~/.localharness/agents` after. The orchestrator applies/restores
this at run time. RECOMMENDED: --max-turns-per-day 5 (owner token/GPU discipline — ~75 turns
instead of the full 148; the uncapped full month remains available).

usage (live, ATTENDED only):
  .venv/bin/python scripts/sema05_month_in_a_day.py \
      --history <scratch>/history.jsonl \
      --store <scratch>/isolated-store \
      --results ~/.localharness/sema05-reports/phase36-$(date +%Y%m%dT%H%M%SZ) \
      --model <subject> --base-url <url> --max-turns-per-day 5

Exit codes: 0 = a measurement was produced (HOLDS / KILL / INCONCLUSIVE all succeed);
            1 = processing failure / watchdog abort; 2 = guard refusal (bench/results, live
            store, or not-exactly-one accumulation mode).
"""
from __future__ import annotations

import argparse
import re
import asyncio
import hashlib
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

# Robust src bootstrap: run from any CWD (computed from __file__, not the working dir).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from localharness.config.models import (  # noqa: E402
    MemoryConsolidationConfig,
    PredictiveGateConfig,
)
from localharness.core.bus import EventBus  # noqa: E402
from localharness.core.events import (  # noqa: E402
    Action,
    Observation,
    StuckRecovered,
    TurnFailed,
    UserMessage,
)
from localharness.memory.consolidation import ConsolidationPass  # noqa: E402
from localharness.memory.gate import WriteGate  # noqa: E402
from localharness.memory.idle_llm import LLMTextAdapter, ground_numbers, grounded  # noqa: E402
from localharness.memory.predictive_gate import PredictiveGate  # noqa: E402
from localharness.memory.predictive_write_gate import PredictiveWriteGate  # noqa: E402
from localharness.memory.reconciliation import reconcile_corrections  # noqa: E402
from localharness.memory.sqlite import (  # noqa: E402
    _LEGACY_ROOT_AGENT_ID,
    _ROOT_AGENT_ID,
    MemoryStore,
    _row_to_fact,
)

_MIN_MEM_GIB = 30.0  # RANK-06 practice: kill/abort the live LLM path below this MemAvailable.
_LOOP_CONTEXT_TOKENS = 32768  # machine-safety context bound: far below the 96k hard-hang class
_TURN_FAILED_RATE_MAX = 0.20  # MOVE 0b: a sitting above this TurnFailed rate -> verdict INVALID.
_GRADING_DOC = ".planning/phases/36-chapter-writer/36-SEMA05-GRADING.md"
_GRADING_DOC_36_1 = ".planning/phases/36-chapter-writer/36.1-DESIGNED-MONTH-GRADING.md"  # MOVE 4
_GRADING_COMMITTED = "2026-07-05T17:43:47Z"  # the LOCKED pre-commitment timestamp (quoted in report)
_ROLE = (
    "Personal assistant on the owner's box. Answer directly and use tools only when the "
    "request actually needs them."
)


class _WatchdogAbort(RuntimeError):
    """Raised when MemAvailable drops below the safety threshold mid-run (live mode)."""


class _MeasurementInvalid(RuntimeError):
    """MOVE 0b: a sitting's TurnFailed rate exceeded _TURN_FAILED_RATE_MAX — the instrument, not
    the subject, failed. run() catches this and stamps verdict INVALID (measurement failure !=
    INCONCLUSIVE), naming the failing sitting. Mirrors the probe's own TurnFailed gate."""

    def __init__(self, *, day: str, session_id: str, failed: int, turns: int) -> None:
        self.day, self.session_id, self.failed, self.turns = day, session_id, failed, turns
        rate = failed / turns if turns else 1.0
        super().__init__(
            f"sitting {day} ({session_id}): TurnFailed rate {failed}/{turns} = {rate:.0%} "
            f"> {_TURN_FAILED_RATE_MAX:.0%} — measurement failure, verdict INVALID"
        )


# ---------------------------------------------------------------------------
# Machine-safety watchdog (poll /proc/meminfo MemAvailable; abort the live path below threshold)
# ---------------------------------------------------------------------------
def _mem_available_gib() -> float | None:
    """MemAvailable in GiB from /proc/meminfo, or None if unmeasurable (non-Linux)."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        return None
    return None


def _watchdog_ok(min_gib: float = _MIN_MEM_GIB) -> bool:
    """True if it is safe to launch a live-LLM prefill. Unmeasurable (None) -> True (attended-only
    runs still respect this, but we do not hard-block where MemAvailable cannot be read)."""
    g = _mem_available_gib()
    return g is None or g >= min_gib


# ---------------------------------------------------------------------------
# The bundled offline double (used ONLY with --offline; never the provable's lesson source)
# ---------------------------------------------------------------------------
class _OfflineFakeLLM:
    """Deterministic CI double dispatching on prompt content exactly like the 36-07 composed
    proof's _DispatchLLM: a grounded verbatim corpus echo for the chapter-writer, REVERT for a
    reconcile look (reverts are the 0.250-precision common case), a grounded personal fact for
    the miner, and '' (inert) for the replay seam. The lessons it summarizes are replay-derived;
    only the CHAPTER PROSE is generated (the honest generator/lesson split, offline)."""

    async def complete(self, prompt: str) -> str:
        if "Write ONE" in prompt:  # chapter-writer — echo a verbatim (self-grounded) corpus slice
            corpus = prompt.split("\n\n", 1)[1] if "\n\n" in prompt else prompt
            lines = [ln for ln in corpus.splitlines() if ln.strip()]
            return " ".join(lines[0].split()[:12]) if lines else "chapter"
        if "quarantined pending review" in prompt or "disputed by a user correction" in prompt:
            return "REVERT"  # tri-outcome REVERT -> DRAIN (restore shape a / clear shape b)
        if "USER'S WORLD" in prompt:  # MOVE 2 transcript mining — typed topic|claim|evidence atoms
            # CORPUS-AWARE (like a real extractor): emit only atoms whose subject is actually in
            # this chunk, so a per-day (between-sitting) pass mines only that day's atom and the
            # two attribute to two distinct sittings (no cross-grounding, no provenance collapse).
            out = []
            if "summarizer" in prompt:
                out.append("subagents | building a summarizer subagent for the harness | summarizer subagent for the harness")
            if "citation" in prompt:
                out.append("subagents | building a citation subagent for the harness | citation subagent for the harness")
            return "\n".join(out)
        return ""  # anything else: inert


class _Msg:
    """Loop-contract response message (mirrors conftest FakeLLMResponse: .content/.tool_calls)."""

    def __init__(self, content: str | None = None, tool_calls: list | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class _ToolCall:
    """Loop-contract tool call (mirrors conftest FakeToolCall): .id/.name/.arguments dict plus
    the openai-shape .function view (name + JSON arguments) the native extractor consumes."""

    def __init__(self, id: str, name: str, arguments: dict) -> None:  # noqa: A002
        self.id = id
        self.name = name
        self.arguments = arguments

    @property
    def function(self):
        class _Fn:
            pass

        fn = _Fn()
        fn.name = self.name
        fn.arguments = json.dumps(self.arguments)
        return fn


class _OfflineLoopLLM:
    """Loop-client double for --offline: drives the REAL AgentLoop + REAL builtin tools
    deterministically. For an ordinary task it reads a missing per-task path (a REAL ReadTool
    error), then a real existing path (recovery), then answers — so the shipped WriteGate
    captures a real resolved_error lesson from real tool events; the SAME task text repeated
    on another day recurs naturally (the tag hashes the task). For the PROBE question it
    answers zero-tool, echoing the injected '### Knowledge' chapter line if present — so the
    probe turn exercises real tool-call counting. Neither fake authors lesson text; offline
    chapter prose comes from the seam fake's verbatim corpus echo."""

    def __init__(self, existing_path: str, missing_dir: str) -> None:
        self._existing = existing_path
        self._missing_dir = missing_dir
        self._n = 0

        class _Cfg:
            tool_call_mode = "native"
            context_window = _LOOP_CONTEXT_TOKENS

        self.config = _Cfg()

    async def complete(self, messages=None, tools=None):
        """The compaction summarize seam (make_compaction_summarize_fn unpacks a tuple)."""
        return _Msg(content="compact summary of the earlier turns"), None

    @staticmethod
    def _knowledge_line(msgs: list) -> str | None:
        for m in msgs:
            if m.get("role") == "system" and "### Knowledge" in str(m.get("content", "")):
                section = str(m["content"]).split("### Knowledge", 1)[1]
                for ln in section.splitlines():
                    if ln.strip().startswith("- "):
                        return ln.strip()[2:]
        return None

    async def stream_complete(self, messages=None, tools=None, on_token=None):
        self._n += 1
        msgs = messages or []
        last_user = next((m for m in reversed(msgs) if m.get("role") == "user"), {})
        task = str(last_user.get("content", ""))
        if task.startswith("Review your answer"):  # MECH-01 self-check pass (if enabled)
            return _Msg(content="CONFIRMED"), None
        if task.startswith("You ended your reply"):
            # The loop's act-guard nudge after a tool-less completion: the genuine-no-tool
            # contract is the bare CONFIRMED sentinel (the prior reply is delivered unchanged).
            return _Msg(content="CONFIRMED"), None
        if task.startswith("What has the harness learned") or task.startswith(
            "What durable domain knowledge"
        ):
            # The PROBE turn: answer zero-tool from the injected index (BLOCKER 2 rehearsal).
            chapter = self._knowledge_line(msgs)
            if chapter:
                return _Msg(content=f"From memory: {chapter}"), None
            return _Msg(content="I have no durable knowledge chapter yet."), None
        n_tool = 0
        for m in reversed(msgs):  # tool results since the last user turn = the stage
            if m.get("role") == "user":
                break
            if m.get("role") == "tool":
                n_tool += 1
        # Per-task stable tag: the SAME query text re-run on another day produces the SAME
        # missing path -> the same real error -> honest cross-sitting recurrence.
        tag = hashlib.sha1(" ".join(task.lower().split()).encode("utf-8")).hexdigest()[:6]
        if n_tool == 0:  # stage 1: read a missing path -> a REAL deterministic tool error
            return _Msg(tool_calls=[_ToolCall(
                id=f"tc{self._n}", name="read",
                arguments={"path": f"{self._missing_dir}/absent-{tag}.md"},
            )]), None
        if n_tool == 1:  # stage 2: recover by reading a real existing path
            return _Msg(tool_calls=[_ToolCall(
                id=f"tc{self._n}", name="read", arguments={"path": self._existing},
            )]), None
        return _Msg(content="Handled the request after recovering from the missing path."), None


# ---------------------------------------------------------------------------
# REPLAY — real events through the SHIPPED gates into the isolated store (real events only)
# ---------------------------------------------------------------------------
async def _replay(store: MemoryStore, bus_events: Path, agent_id: str) -> int:
    """Feed the primary events of a real bus-events trace through the shipped live gates into the
    isolated store. Only PRIMARY events are re-published (Action/Observation/UserMessage/
    StuckRecovered); the gates generate their own derived events (SurpriseScored/MemoryGateFired)
    live on the bus, so re-publishing those would double-fire. Never writes near the trace."""
    live = EventBus()
    wg = WriteGate(store, live, agent_id)
    await wg.open()
    pg_cfg = PredictiveGateConfig()
    pg_cfg.write_live = True  # exercise the LIVE write path (stat + correction writes) end-to-end
    pg = PredictiveGate(store, live, agent_id, pg_cfg)
    await pg.open()
    pw = PredictiveWriteGate(store, live, agent_id, pg_cfg)
    await pw.open()

    n = 0
    reader = EventBus(persist_path=bus_events)  # READ-ONLY: only replay() is called
    async for ev in reader.replay(
        event_types=[Action, Observation, UserMessage, StuckRecovered]
    ):
        if getattr(ev, "agent_id", None) != agent_id:
            continue
        # A replayed event already carries a seq; publish() refuses a sequenced event, so reset it.
        await live.publish(ev.model_copy(update={"seq": None}))
        n += 1

    await pw.close()
    await pg.close()
    await wg.close()
    return n


async def _load_transcript(store: MemoryStore, history_path: Path) -> int:
    """Load the REAL history.jsonl transcript into the isolated store (for mining + the payload
    dereference). Best-effort; skips corrupt lines. Real records only — nothing invented."""
    if not history_path.exists():
        return 0
    n = 0
    for raw in history_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        await store.append_history(rec)
        n += 1
    return n


# ---------------------------------------------------------------------------
# LIVE-SESSION ACCUMULATION — the provable's method (owner steer 2026-07-05 17:53):
# re-run the owner's REAL month queries through the REAL agent loop, one fresh
# sitting per original day, against the ISOLATED agent home.
# ---------------------------------------------------------------------------
def _accepted_agent_ids(agent_id: str) -> set[str | None]:
    """BLOCKER 1 (mirrors the ORCH-01/03 root-alias handling): the Phase-33.1 default->
    orchestrator rename migrated the DB but NOT old history.jsonl lines, so the real month's
    user messages carry agent_id='default'. Requesting EITHER root alias treats
    {None, 'default', 'orchestrator'} as one root identity; non-root agents keep the narrow
    filter (their own id + agent-less lines)."""
    if agent_id in (_ROOT_AGENT_ID, _LEGACY_ROOT_AGENT_ID):
        return {None, _ROOT_AGENT_ID, _LEGACY_ROOT_AGENT_ID}
    return {None, agent_id}


def _extract_day_queries(history_path: Path, agent_id: str) -> list[tuple[str, list[str]]]:
    """READ-ONLY extraction of the owner's real user messages from a copied history.jsonl,
    grouped by ORIGINAL calendar day (local box time — the days the owner actually lived,
    which is what the ≥2-sittings stability gate should honestly measure). Within a day,
    original ts order. Non-user records and other agents' records are ignored; root aliases
    are unified (the real month ≈ 135 user messages / 14 days under agent_id='default')."""
    accepted = _accepted_agent_ids(agent_id)
    by_day: dict[str, list[tuple[int, str]]] = {}
    for raw in history_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if rec.get("type") != "user_message":
            continue
        if rec.get("agent_id") not in accepted:
            continue
        content = str(rec.get("content", "")).strip()
        if not content:
            continue
        ts = int(rec.get("ts", 0) or 0)
        day = time.strftime("%Y%m%d", time.localtime(ts))
        by_day.setdefault(day, []).append((ts, content))
    return [(day, [c for _, c in sorted(rows)]) for day, rows in sorted(by_day.items())]


def _root_agent_config(agent: str):
    """The composed AgentConfig, EXACTLY as start_cmd shapes the root agent's plus the
    live-store belt-and-braces (BLOCKER 3):
    - 32k context bound (machine-safety: far below the 96k hard-hang class);
    - P-A capability floor: web_* (untrusted-ingest) DENIED for a bash/write/edit-holding
      agent (prompt-injection->host hole; the real root delegates ingestion to a subagent,
      which this bounded run omits);
    - deny_patterns for write/edit/bash_exec touching the live agent stores.

    HONEST COVERAGE NOTE (safety-critic residual): these are fnmatch globs over the RAW,
    PRE-EXPANSION argument strings (write/edit expanduser AFTER the permission check), so
    coverage is path-form-limited — absolute and ~-prefixed forms are caught; shell-split /
    cd-relative bash forms are NOT (no shell parsing is attempted here). The PHYSICAL layer
    for attended runs is `chmod -R a-w ~/.localharness/agents` applied/restored by the
    orchestrator at run time; these patterns are belt-and-braces above that and above the
    isolated-store guarantee."""
    from localharness.config.models import AgentConfig
    from localharness.tools.capabilities import apply_root_capability_floor

    a_cfg = AgentConfig(name=agent, role=_ROLE)
    a_cfg.context.max_context_tokens = _LOOP_CONTEXT_TOKENS
    apply_root_capability_floor(a_cfg.tools)
    home_agents = str(Path.home() / ".localharness" / "agents")
    for form in (home_agents, "~/.localharness/agents"):
        a_cfg.permissions.deny_patterns += [
            f"write({form}/*)",
            f"edit({form}/*)",
            f"bash_exec(*{form}*)",
        ]
    return a_cfg


async def _build_loop(args, store_dir: Path, store: MemoryStore, bus: EventBus,
                      sid: str, loop_llm: object, token_counter: object | None,
                      compact_md: Path | None = None):
    """The REAL loop composition shared by accumulation sittings and the probe turn (mirrors
    start_cmd): builtin tools + memory tools + eviction ContentStore bound, the root config
    (_root_agent_config), and — MAJOR 8 — the production CompactionPipeline wired with the
    same LLM, so context pressure compacts exactly as the live CLI does instead of warning.
    `compact_md` overrides the compact.md path — the probe turn passes a FRESH one so a
    carried accumulation summary can never be a second answer source faking attribution."""
    from localharness.agent.context import (
        CompactionPipeline,
        ContentStore,
        ContextManager,
        TokenCounter,
        make_compaction_summarize_fn,
    )
    from localharness.agent.loop import AgentLoop
    from localharness.agent.permissions import PermissionEvaluator
    from localharness.tools.builtin import bind_agent_store_tools, register_builtin_tools
    from localharness.tools.registry import ToolRegistry

    registry = ToolRegistry()
    content_store = ContentStore()
    await register_builtin_tools(registry, memory_store=store, eviction_store=content_store)
    bind_agent_store_tools(registry, content_store)

    a_cfg = _root_agent_config(args.agent)
    if compact_md is None:
        compact_md = store_dir / "agents" / args.agent / "compact.md"  # isolated (never ~/. path)
    tc = token_counter or TokenCounter()  # offline: the same estimator ContextManager defaults to
    pipeline = CompactionPipeline(
        token_counter=tc,
        tool_result_cap=a_cfg.context.max_tool_output_chars,
        preserve_first_n=a_cfg.context.preserve_first_n_messages,
        preserve_last_n=a_cfg.context.preserve_last_n_messages,
        llm_summarize_fn=make_compaction_summarize_fn(loop_llm),
        compact_md_path=compact_md,
    )
    ctx = ContextManager(
        max_context_tokens=_LOOP_CONTEXT_TOKENS,
        preserve_first_n=a_cfg.context.preserve_first_n_messages,
        preserve_last_n=a_cfg.context.preserve_last_n_messages,
        pipeline=pipeline,
        eviction_store=content_store,
        tool_evict_threshold_chars=a_cfg.context.tool_result_evict_threshold_chars,
        tool_evict_enabled=a_cfg.context.tool_result_eviction,
        token_counter=tc,
    )
    return AgentLoop(
        config=a_cfg, llm=loop_llm, bus=bus, context_manager=ctx,
        tool_registry=registry, permission_evaluator=PermissionEvaluator(),
        memory_loader=store,
        compact_md_path=compact_md,
        kill_file_path=store_dir / "KILL",  # attended stop: `touch <store>/KILL`
        session_id=sid,
    )


async def _purge_dangling_session(store: MemoryStore, agent_id: str, sid: str) -> None:
    """Day-checkpoint resume: drop ALL partial state a hard freeze left under `sid` so the day
    can be redone from turn 0 UNDER THE IDENTICAL id — never a forked/suffixed session, which
    would inflate the >=2-distinct-sessions clustering bar (clustering.py:266) and flip the
    grade. Deletes in FK-safe order (PRAGMA foreign_keys=ON): session-scoped facts first
    (provenance == sid for the standard lesson/stat rows — gate.py:206, pwg.py:191 — plus the
    rarer correction rows whose provenance is a user_signals.event_id for this session,
    pwg.py:235/248), then the v4 collect tables (children before parents), then the row itself.
    (Accumulation sittings create no edges — those are grading-phase only — so none are orphaned.)"""
    db = store._db
    assert db is not None
    await db.execute(
        "DELETE FROM facts WHERE agent_id = ? AND (provenance = ? OR provenance IN "
        "(SELECT event_id FROM user_signals WHERE agent_id = ? AND session_id = ? "
        "AND event_id IS NOT NULL))",
        (agent_id, sid, agent_id, sid),
    )
    await db.execute(
        "DELETE FROM staged_snapshots WHERE user_signal_id IN "
        "(SELECT id FROM user_signals WHERE agent_id = ? AND session_id = ?)",
        (agent_id, sid),
    )
    await db.execute(
        "DELETE FROM surprise_scores WHERE agent_id = ? AND session_id = ?", (agent_id, sid)
    )
    await db.execute(
        "DELETE FROM user_signals WHERE agent_id = ? AND session_id = ?", (agent_id, sid)
    )
    await db.execute(
        "DELETE FROM tool_observations WHERE agent_id = ? AND session_id = ?", (agent_id, sid)
    )
    await db.execute("DELETE FROM sessions WHERE agent_id = ? AND id = ?", (agent_id, sid))
    await db.commit()


async def _run_sittings(
    args: argparse.Namespace,
    store_dir: Path,
    day_queries: list[tuple[str, list[str]]],
    loop_llm: object,
    token_counter: object | None,
    model_label: str,
) -> list[dict]:
    """Drive each original day's real queries through the REAL AgentLoop as ONE fresh sitting:
    fresh store/bus/gates/loop instances per day (emulating one process per sitting), real
    create_session/end_session close-outs, the bus persisted into the isolated agent home.
    This is the same hot path the CLI runs (start_cmd composition, repl.py drive): the loop
    publishes real Action/Observation events, executes REAL tools, and the shipped gates
    capture real lessons/corrections. Watchdog-checked before every live turn; a KILL file
    stops accumulation at the next boundary (MAJOR 6b) so killed runs never keep iterating.
    Returns [{day, session_id, turns}] in run order."""
    from localharness.cli.session_accumulator import SessionAccumulator, derive_session_summary
    from localharness.memory.user_signals import UserSignalDetector

    events_path = store_dir / "agents" / args.agent / "bus-events.jsonl"
    sittings: list[dict] = []

    # DAY-GRANULARITY CHECKPOINT/RESUME (survive a hard freeze): a completed day is durable iff
    # its sessions row has ended_at set (end_session, run in the finally below). Read the isolated
    # store ONCE before the loop so a resumed run skips durable days, redoes a dangling
    # (freeze-interrupted) day under the SAME id, and runs never-started days. No new checkpoint
    # write — ended_at already IS the durability signal. Directions never cross: only a committed
    # end_session sets ended_at, so a truly-done day can never read back as dangling, and a
    # dangling day can never read back as done (worst case under synchronous=NORMAL a completed
    # day whose WAL was not yet fsynced looks dangling and is redone — safe, only ever redundant).
    done: set[str] = set()
    dangling: set[str] = set()
    db_path = store_dir / "agents" / args.agent / "memory.db"
    if db_path.exists():
        con = sqlite3.connect(str(db_path))
        try:
            for row_sid, ended in con.execute(
                "SELECT id, ended_at FROM sessions WHERE id LIKE 'sema05-%'"
            ):
                if row_sid == "sema05-probe":
                    continue  # the grading-phase probe is not an accumulation day
                (done if ended is not None else dangling).add(row_sid)
        finally:
            con.close()
    if done or dangling:
        print(f"resume: {len(done)} day(s) durable (skip), {len(dangling)} dangling (redo same id)")

    for day, queries in day_queries:
        sid = f"sema05-{day}"
        capped = queries[: args.max_turns_per_day] if args.max_turns_per_day else queries
        if sid in done:  # already durable — SKIP entirely (no create_session, no turns re-run)
            sittings.append({"day": day, "session_id": sid, "turns": len(capped)})
            print(f"sitting {day}: SKIPPED — already durable (session {sid})")
            continue
        if (store_dir / "KILL").exists():  # MAJOR 6b: stop BEFORE starting another sitting
            print(f"KILL file present — stopping accumulation before sitting {day}.")
            break
        if not args.offline and not _watchdog_ok():
            raise _WatchdogAbort(f"MemAvailable below {_MIN_MEM_GIB} GiB before sitting {day}")

        # Fresh per-sitting composition (mirrors start_cmd; one continuous bus-events log).
        bus = EventBus(persist_path=events_path)
        store = MemoryStore(agent_id=args.agent, division_id="", org_id="",
                            base_dir=str(store_dir), bus=bus)
        await store.open()
        if sid in dangling:  # freeze-interrupted: drop the partial state, redo from turn 0 (SAME id).
            await _purge_dangling_session(store, args.agent, sid)
            print(f"sitting {day}: purged dangling partial state — redoing under {sid}")
        await store.create_session(sid, budget={}, model=model_label,
                                   context_tokens_available=_LOOP_CONTEXT_TOKENS)
        wg = WriteGate(store, bus, args.agent)
        await wg.open()
        pg_cfg = PredictiveGateConfig()
        pg_cfg.write_live = True
        pg = PredictiveGate(store, bus, args.agent, pg_cfg)
        await pg.open()
        usig = UserSignalDetector(store, bus, args.agent, pg_cfg)
        await usig.open()
        pw = PredictiveWriteGate(store, bus, args.agent, pg_cfg)
        await pw.open()
        acc = SessionAccumulator(bus, args.agent)
        await acc.open()
        loop = await _build_loop(args, store_dir, store, bus, sid, loop_llm, token_counter)

        # MOVE 0b: sitting-level TurnFailed gate. run_turn NEVER raises (errors become a TurnFailed
        # event + a prose summary), so `turns` counts attempts, not successes — the failure signal
        # is the event, exactly like the probe's gate. Count failures per sitting; a rate over the
        # threshold aborts the run as INVALID (measurement failure), naming this sitting.
        failed = 0

        def _on_turn_failed(_ev: object) -> None:
            nonlocal failed
            failed += 1

        bus.subscribe(TurnFailed, _on_turn_failed, agent_id=args.agent)

        turns = 0
        exit_reason = "complete"
        try:
            for q in capped:
                if (store_dir / "KILL").exists():  # MAJOR 6b: stop at the next turn boundary
                    exit_reason = "kill_file"
                    print(f"KILL file present — stopping sitting {day} after {turns} turns.")
                    break
                if not args.offline and not _watchdog_ok():
                    exit_reason = "watchdog_abort"
                    raise _WatchdogAbort(
                        f"MemAvailable below {_MIN_MEM_GIB} GiB mid-sitting {day} (turn {turns + 1})"
                    )
                # Exactly the repl.py drive: publish the UserMessage for the memory pipeline,
                # then run the turn through the real loop.
                await bus.publish(UserMessage(agent_id=args.agent, session_id=sid,
                                              content=q, channel="terminal"))
                await loop.run_turn(task=q)
                turns += 1
        finally:
            # Mirror start_cmd's ordered shutdown: accumulator stops counting, end_session
            # writes the real close-out, gates close, store closes.
            await acc.close()
            try:
                await store.end_session(
                    sid, exit_reason=exit_reason, summary=derive_session_summary(acc),
                    turn_count=acc.turn_count, action_count=acc.action_count,
                    tokens_in=acc.tokens_in, tokens_out=acc.tokens_out,
                )
            except Exception:  # noqa: BLE001 — a close-out fault must not mask the turn error
                pass
            for g in (pw, usig, pg, wg):
                try:
                    await g.close()
                except Exception:  # noqa: BLE001
                    pass
            await store.close()
        sittings.append({"day": day, "session_id": sid, "turns": turns, "turn_failed": failed})
        print(f"sitting {day}: {turns}/{len(queries)} real queries re-run "
              f"({failed} failed) (session {sid})")
        # MOVE 0b: abort LOUDLY at the failing sitting — a dead sitting must never be graded as
        # if it held knowledge (the SEMA-05 verdict graded a 2-day store as 15 days silently).
        if turns and failed / turns > _TURN_FAILED_RATE_MAX:
            raise _MeasurementInvalid(day=day, session_id=sid, failed=failed, turns=turns)
    return sittings


async def _probe_turn(args, store_dir: Path, question: str, loop_llm: object,
                      token_counter: object | None, schema_values: list[str],
                      model_label: str) -> dict:
    """BLOCKER 2: the zero-tool metric comes from a REAL turn, not a substring heuristic.
    After consolidation, ASK the domain question through the same AgentLoop machinery the
    sittings used — a fresh probe session against the consolidated isolated store, the
    chapter-bearing index injected via memory_loader, tools REGISTERED — and derive:
      tool_calls   = the count of real Action(tool_call) events the turn emitted;
      turn_failed  = whether the turn published TurnFailed (run_turn NEVER returns an empty
                     string — error/kill summaries are non-empty prose — so the failure
                     signal must come from the event, not the answer text);
      chapter_content_in_answer = any written chapter's tokens majority-appear in the answer
                     (substring/paraphrase-tolerant via the majority-token net).
    The caller gates zero_tool on ALL THREE (locked bar: 'answered BY the Knowledge line');
    a failed or off-chapter probe demotes to INCONCLUSIVE — false-negative is the only
    permitted error direction. The probe gets a FRESH compact-probe.md so a carried
    accumulation summary can never be a second answer source faking attribution. The write
    gates are deliberately NOT subscribed — a measurement turn must not write new candidates
    into the store it is grading."""
    bus = EventBus(persist_path=store_dir / "agents" / args.agent / "bus-events.jsonl")
    tool_calls: list = []
    failures: list = []

    async def _count(e) -> None:
        if getattr(e, "action_type", None) == "tool_call":
            tool_calls.append(e)

    async def _failed(e) -> None:
        failures.append(e)

    bus.subscribe(Action, _count, agent_id=args.agent)
    bus.subscribe(TurnFailed, _failed, agent_id=args.agent)
    store = MemoryStore(agent_id=args.agent, division_id="", org_id="",
                        base_dir=str(store_dir), bus=bus)
    await store.open()
    sid = "sema05-probe"
    try:
        # Grading-phase resume: a prior run that froze during/after the probe left a dangling
        # sema05-probe row; the probe is redone wholesale, so clear it first (never PK-collide).
        await _purge_dangling_session(store, args.agent, sid)
        await store.create_session(sid, budget={}, model=model_label,
                                   context_tokens_available=_LOOP_CONTEXT_TOKENS)
        loop = await _build_loop(args, store_dir, store, bus, sid, loop_llm, token_counter,
                                 compact_md=store_dir / "compact-probe.md")
        await bus.publish(UserMessage(agent_id=args.agent, session_id=sid,
                                      content=question, channel="terminal"))
        answer = (await loop.run_turn(task=question) or "").strip()
        try:
            await store.end_session(sid, exit_reason="complete", summary=None,
                                    turn_count=1, action_count=len(tool_calls),
                                    tokens_in=0, tokens_out=0)
        except Exception:  # noqa: BLE001 — close-out is best-effort for the probe
            pass
    finally:
        await store.close()
    return {
        "session_id": sid,
        "question": question,
        "answer": answer,
        "tool_calls": len(tool_calls),
        "turn_failed": bool(failures),
        "chapter_content_in_answer": bool(
            answer and any(grounded(sv, answer) for sv in schema_values)
        ),
    }


# ---------------------------------------------------------------------------
# Live-store honesty (read the COPIED memory.db READ-ONLY — never the live store itself)
# ---------------------------------------------------------------------------
def _live_store_counts(db_path: Path, agent_id: str) -> dict:
    """Report the real organic state of the (copied) live store as a WATCH ITEM: active schema
    chapters, active correction_pending rows, promoted lessons. READ-ONLY (mode=ro). Nulls when
    no db was copied (offline)."""
    empty = {"schemas": None, "correction_pending": None, "learned": None, "source": None}
    if not db_path.exists():
        return empty
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = con.cursor()

        def _one(where: str) -> int:
            cur.execute(
                f"SELECT COUNT(*) FROM facts WHERE agent_id=? AND status='active' AND {where}",
                (agent_id,),
            )
            return int(cur.fetchone()[0])

        out = {
            "schemas": _one("node_kind='schema'"),
            "correction_pending": _one("tags LIKE '%\"tier:correction_pending\"%'"),
            "learned": _one("key LIKE 'learned/%'"),
            "source": str(db_path),
        }
        con.close()
        return out
    except Exception as exc:  # noqa: BLE001 — a read failure is disclosed, never fatal
        return {**empty, "source": f"unreadable: {type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Probe helpers (proxy-independent metrics + the mechanical KILL re-check)
# ---------------------------------------------------------------------------
def _member_tool(key: str) -> str | None:
    parts = key.split("/")
    return parts[1] if len(parts) >= 2 and parts[0] == "learned" else None


def _supermajority_grounded(claim: str, corpus: str, *, min_token_len: int = 6) -> bool:
    """The §6 sensitivity bar: >= 2/3 of >=6-char tokens verbatim in the corpus (vs the shipped
    >= 1/2 majority). An empty-token claim is vacuously grounded."""
    toks = [t for t in claim.split() if len(t) >= min_token_len]
    if not toks:
        return True
    matched = sum(1 for t in toks if t in corpus)
    return matched * 3 >= len(toks) * 2


async def _active(store: MemoryStore, where: str) -> list:
    assert store._db is not None
    async with store._db.execute(
        f"SELECT {store._FACT_COLS} FROM facts "
        f"WHERE agent_id=? AND status='active' AND {where}",
        (store._agent_id,),
    ) as cur:
        return [_row_to_fact(r) for r in await cur.fetchall()]


async def _schema_members(store: MemoryStore, schema_id: int) -> list:
    nb = await store.neighborhood(schema_id, depth=1)
    ids = [nid for nid, _d in nb if nid != schema_id]
    return await store.get_facts_by_ids(ids)


async def _correction_shape_split(store: MemoryStore) -> tuple[int, int]:
    """(shape_a, shape_b) among active tier:correction_pending rows — snapshot BEFORE reconcile
    drains them. Shape (b) = the quarantine-only majority (fresh correction/quarantine/ key)."""
    rows = await _active(store, "tags LIKE '%\"tier:correction_pending\"%'")
    shape_b = sum(1 for r in rows if r.key.startswith("correction/quarantine/"))
    return len(rows) - shape_b, shape_b


async def _schema_corpus(store: MemoryStore, schema) -> tuple[list[str], str]:
    """MAJOR 4: the grading corpus must equal the WRITER'S corpus (chapter_writer._write_one):
    member+aux bodies PLUS the event-log dereference of payload-thin stat rows (CONTEXT
    ruling 1), char-capped identically — otherwise a validly-grounded chapter can spuriously
    KILL because the checker sees less than the writer did. Returns (bodies, capped_corpus);
    the numeric net grades against bodies only, exactly as the writer does."""
    from localharness.memory.chapter_writer import _dereference

    members = await _schema_members(store, schema.id)
    bodies = [m.value for m in members]
    derefs = await _dereference(store, members)
    return bodies, "\n".join(bodies + derefs)[:6000]  # == chapter_writer corpus_char_cap default


async def _static_checks(store: MemoryStore) -> dict:
    """The store-side halves of the grade: the mechanical KILL re-check (majority-token
    grounding + numeric net, per schema, against the WRITER'S corpus), byte-stability
    (double-render ==), N, and the domain question. The zero-tool metric itself comes from
    the REAL probe turn (_probe_turn) — never a substring heuristic (BLOCKER 2)."""
    index = (await store.load_context(index_mode=True)).agent_memory_md
    index2 = (await store.load_context(index_mode=True)).agent_memory_md
    byte_stable = index == index2  # SEMA-04 double-render ==

    schemas = await _active(store, "node_kind='schema'")
    per_schema: list[dict] = []
    member_tools: list[str] = []
    for s in schemas:
        bodies, corpus = await _schema_corpus(store, s)
        g_majority = grounded(s.value, corpus)            # the shipped >=50% majority-token net
        unverified = ground_numbers(s.value, bodies)      # the numeric net (empty == clean)
        per_schema.append({
            "key": s.key,
            "value": s.value[:300],  # the actual chapter text, for the human read
            "grounded": bool(g_majority and not unverified),  # KILL trip: either failing -> kill
            "grounded_majority": bool(g_majority),
            "unverified_numbers": unverified,
        })
        members = await _schema_members(store, s.id)
        member_tools += [t for t in (_member_tool(m.key) for m in members) if t]

    kill_triggered = any(not p["grounded"] for p in per_schema)
    domain_token = Counter(member_tools).most_common(1)[0][0] if member_tools else None
    domain_question = (
        f"What has the harness learned about the `{domain_token}` tool?"
        if domain_token else "What durable domain knowledge has the harness formed?"
    )
    n_lessons = len(await _active(store, "key LIKE 'learned/%'"))
    return {
        "schemas_written": len(schemas),
        "schema_values": [s.value for s in schemas],
        "per_schema_grounding": per_schema,
        "kill_triggered": kill_triggered,
        "domain_token": domain_token,
        "domain_question": domain_question,
        "byte_stable": byte_stable,
        "n_lessons": n_lessons,
    }


async def _stricter_clusters(store: MemoryStore, min_sessions: int) -> tuple[int, list[list[str]]]:
    """MAJOR 5: re-derive stable clusters under the stricter bar over a pool that EXCLUDES
    schema nodes — a just-written chapter otherwise joins _load_pool and its own
    'cluster:sessA|sessB' provenance counts as a FAKE extra session, so the stricter bar
    falsely holds. Composes clustering.py's own primitives (same defaults as
    find_stable_clusters) rather than forking its logic; 'cluster:' provenances are filtered
    out of session sets as a second belt. Returns (n_meeting_bar, per-candidate-cluster
    sorted session-id lists) — the lists go into the verdict so plus1 legitimacy is
    hostile-read-verifiable from the artifact alone."""
    from localharness.memory import clustering as _clustering

    pool = [f for f in await _clustering._load_pool(store) if f.node_kind != "schema"]
    if len(pool) < 2:
        return 0, []
    adj = await _clustering._relatedness_edges(store, pool, fts_top_k=5, graph_depth=2)
    n = 0
    per_cluster: list[list[str]] = []
    for members in _clustering._connected_components(pool, adj):
        if len(members) < 2:
            continue
        sessions = await _clustering._component_sessions(store, members)
        sessions = {s for s in sessions if not s.startswith("cluster:")}
        per_cluster.append(sorted(sessions))
        if len(sessions) >= min_sessions:
            n += 1
    return n, per_cluster


async def _sensitivity(store: MemoryStore, cfg: MemoryConsolidationConfig) -> dict:
    """§6 mandatory re-grade under >=1 alternate assumption: (1) a stricter super-majority
    grounding bar (against the WRITER'S corpus, MAJOR 4); (2) a stricter cluster-stability
    bar (cluster_min_sessions + 1) over a schema-free pool (MAJOR 5)."""
    schemas = await _active(store, "node_kind='schema'")
    holds_super = True
    for s in schemas:
        _bodies, corpus = await _schema_corpus(store, s)
        if not _supermajority_grounded(s.value, corpus):
            holds_super = False
    n_strict, cluster_sessions = await _stricter_clusters(store, cfg.cluster_min_sessions + 1)
    return {
        "grounding_supermajority_holds": bool(holds_super),
        "cluster_min_sessions_plus1_holds": n_strict >= 1,
        "stable_clusters_at_min_sessions_plus1": n_strict,
        # Per-candidate-cluster sorted session ids (schema-free pool, 'cluster:' filtered) —
        # a hostile reader can verify the plus1 verdict from the artifact alone.
        "stricter_cluster_sessions": cluster_sessions,
        "notes": (
            "Auto re-grade, post-hoc over the promoted population (members folded but still "
            "confidence>=0.7; schema nodes excluded so a written chapter cannot vouch for its "
            "own stability). If the headline zero-tool verdict flips under either assumption "
            "it is disclosed here, never smoothed (the M1 methodology-sensitivity failure mode)."
        ),
    }


# ===========================================================================
# MOVE 4 — the designed-month manifest mode. Drive from a ground-truth manifest
# (query -> expected topic) and grade GROUPING QUALITY DIRECTLY (Stages A/B/C per
# .planning/phases/36-chapter-writer/36.1-DESIGNED-MONTH-GRADING.md), not the tool-avoidance
# proxy. Session ids are `designed-{day}`; consolidation runs BETWEEN days (as today's pass).
# ===========================================================================

def _load_manifest(path: Path) -> dict:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _c2(n: int) -> int:
    return n * (n - 1) // 2


def _ari(labels_true: list, labels_pred: list) -> float:
    """Adjusted Rand Index over two partitions of the same items (inline, no new deps). 1.0 =
    identical partitions; ~0 = random; the overall Stage-B grouping-quality number."""
    n = len(labels_true)
    if n < 2:
        return 1.0
    cont: dict[tuple, int] = {}
    for a, b in zip(labels_true, labels_pred):
        cont[(a, b)] = cont.get((a, b), 0) + 1
    a_sums = Counter(labels_true)
    b_sums = Counter(labels_pred)
    index = sum(_c2(v) for v in cont.values())
    sum_a = sum(_c2(v) for v in a_sums.values())
    sum_b = sum(_c2(v) for v in b_sums.values())
    expected = (sum_a * sum_b) / _c2(n)
    maximum = 0.5 * (sum_a + sum_b)
    if maximum == expected:
        return 1.0  # both partitions trivial (all-singletons or all-one) -> perfect agreement
    return (index - expected) / (maximum - expected)


def _tok5(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 5}


def _day_of(prov: str) -> str:
    """A sem atom's provenance is the SOURCE sitting session `designed-{day}`; strip to the day."""
    return prov[len("designed-"):] if prov.startswith("designed-") else prov


def _attribute_topic(atom_value: str, day_queries: list[dict]) -> str | None:
    """The atom's GROUND-TRUTH topic (Stage-B join): the manifest topic of the best token-overlap
    driven query within the atom's provenance day. None when nothing overlaps."""
    probe = _tok5(atom_value)
    best, best_score = None, 0
    for q in day_queries:
        score = len(probe & _tok5(q["text"]))
        if score > best_score:
            best, best_score = q, score
    return best["topic"] if best else None


async def _grade_designed_month(
    store: MemoryStore, manifest: dict, turn_query_map: dict,
    *, min_sessions: int = 2, min_cluster_size: int = 2,
) -> dict:
    """Grade the isolated store against the manifest — Stages A (extraction) + B (grouping, the
    HEADLINE) + the verdict. PURE store read (no LLM). Stage C (behavior) is driven separately in
    run() and is never part of the verdict. The failing stage is NAMED (per-stage attribution)."""
    expected = {t for t, m in manifest["topics"].items() if m.get("expected_chapter")}
    driven_by_day = {
        d: [{"id": qid, "topic": tp, "text": txt} for (qid, tp, txt) in turn_query_map.get(d, [])]
        for d in manifest["days"]
    }
    # Day transcript corpora (A2 grounds each atom against the text mining actually saw).
    day_corpus: dict[str, list[str]] = {}
    for r in await store.get_history(limit=1_000_000):
        day_corpus.setdefault(_day_of(r.get("session_id") or ""), []).append(str(r.get("content", "")))
    day_corpus = {d: "\n".join(v) for d, v in day_corpus.items()}

    sem_atoms = await _active(store, "key LIKE 'sem/%'")
    gt: dict[int, str | None] = {}          # atom.id -> ground-truth topic (or 'noise' or None)
    a2_kill = False
    a3_ok = True
    for a in sem_atoms:
        day = _day_of(a.provenance)
        if not a.provenance or a.provenance.startswith("mined-from:"):
            a3_ok = False                   # A3: batch-level provenance (the SEMA-05 defect)
        gt[a.id] = _attribute_topic(a.value, driven_by_day.get(day, []))
        if not grounded(a.value, day_corpus.get(day, "")):
            a2_kill = True                  # A2: ungrounded atom -> KILL (zero tolerance)

    # --- Stage A1 recall over topic-sittings ((topic, day) with >= 1 EXTRACTION-EXPECTING
    # manifest query). Ruling 4a: queries flagged expects_extraction: false (decisions/
    # bookkeeping) are not extraction targets — counting them inflated the denominator and
    # starved A1 (run 2: INCONCLUSIVE @A1). ---
    topic_sittings = sorted({
        (q["topic"], q["day"]) for q in manifest["queries"]
        if q["topic"] in expected and q.get("expects_extraction", True)
    })
    covered = sum(
        1 for (t, d) in topic_sittings
        if any(gt.get(a.id) == t and _day_of(a.provenance) == d for a in sem_atoms)
    )
    a1_recall = covered / len(topic_sittings) if topic_sittings else 0.0

    # --- Stage B: the DISCOVERED grouping (chapters + their sem members) ---
    schemas = await _active(store, "node_kind='schema'")
    chapter_members = {
        s.id: [m for m in await _schema_members(store, s.id) if m.key.startswith("sem/")]
        for s in schemas
    }
    chapter_label = {
        sid: (Counter(l for m in members if (l := gt.get(m.id))).most_common(1) or [(None, 0)])[0][0]
        for sid, members in chapter_members.items()
    }
    atoms_by_topic = Counter(gt.get(a.id) for a in sem_atoms)

    # B1: >= 5/6 expected topics form a chapter — at most one may fail to form and only if it is a
    # correct-null (< min_cluster_size atoms). Non-vacuous: >= 1 chapter MUST actually form (a
    # degenerate run where every topic nulls is NOT a HOLD). Generalized for any manifest size.
    formed = {lbl for lbl in chapter_label.values() if lbl in expected}
    nullable = {t for t in expected if atoms_by_topic.get(t, 0) < min_cluster_size}
    non_formed = expected - formed
    b1_count = len(formed | (nullable & non_formed))
    b1_ok = len(formed) >= max(1, len(expected) - 1) and all(t in nullable for t in non_formed)

    # B2: per formed chapter member recall >= 0.6, precision >= 0.7; overall ARI.
    per_chapter: list[dict] = []
    b2_ok = True
    for sid, members in chapter_members.items():
        lbl = chapter_label[sid]
        if lbl not in expected:
            continue
        tp = sum(1 for m in members if gt.get(m.id) == lbl)
        precision = tp / len(members) if members else 0.0
        total_lbl = sum(1 for a in sem_atoms if gt.get(a.id) == lbl)
        recall = tp / total_lbl if total_lbl else 0.0
        per_chapter.append({"label": lbl, "members": len(members),
                            "precision": round(precision, 3), "recall": round(recall, 3)})
        if recall < 0.6 or precision < 0.7:
            b2_ok = False
    order = [a.id for a in sem_atoms]
    atom_chapter = {m.id: f"ch{sid}" for sid, members in chapter_members.items() for m in members}
    true_labels = [gt.get(aid) or f"none-{aid}" for aid in order]
    pred_labels = [atom_chapter.get(aid, f"solo-{aid}") for aid in order]
    ari = round(_ari(true_labels, pred_labels), 3)

    # B3: 0 noise-attributed atoms inside any chapter.
    noise_in_chapter = sum(
        1 for members in chapter_members.values() for m in members if gt.get(m.id) == "noise"
    )
    b3_ok = noise_in_chapter == 0

    # B4: correction arc — corrected value reachable from the arc topic's chapter/members; the
    # stale value NOT asserted current in any active fact.
    arc = manifest.get("correction_arc", {})
    corrected_nums = re.findall(r"\d+", arc.get("corrected", ""))
    reachable = False
    for sid, members in chapter_members.items():
        if chapter_label[sid] != arc.get("topic"):
            continue
        s = next(s for s in schemas if s.id == sid)
        blob = s.value + " " + " ".join(m.value for m in members)
        if corrected_nums and all(nm in blob for nm in corrected_nums):
            reachable = True
    if not reachable and corrected_nums:  # or reachable directly from an arc-topic member atom
        reachable = any(
            gt.get(a.id) == arc.get("topic") and all(nm in a.value for nm in corrected_nums)
            for a in sem_atoms
        )
    all_active = await _active(store, "1=1")
    stale = arc.get("stale", "")
    stale_active = bool(stale) and any(stale in f.value for f in all_active)
    b4_ok = (not arc) or (reachable and not stale_active)  # no arc in the manifest -> vacuously ok

    # B5 + byte-stability + the operational-rows-in-no-chapter invariant (reuse the store checks).
    static = await _static_checks(store)
    b5_kill = static["kill_triggered"]
    byte_stable = static["byte_stable"]
    op_in_chapter = sum(
        1 for members in chapter_members.values() for m in members
        if m.key.startswith(("gate/", "predgate/", "learned/"))
    )

    stage_a = {"a1_recall": round(a1_recall, 3), "a1_covered": covered,
               "a1_topic_sittings": len(topic_sittings), "a1_ok": a1_recall >= 0.8,
               "a2_kill": a2_kill, "a3_provenance_ok": a3_ok, "sem_atoms": len(sem_atoms)}
    stage_b = {"b1_ok": b1_ok, "b1_chapters_or_null": b1_count, "formed_topics": sorted(formed),
               "nullable_topics": sorted(nullable), "b2_ok": b2_ok, "per_chapter": per_chapter,
               "ari": ari, "b3_ok": b3_ok, "noise_in_chapter": noise_in_chapter,
               "b4_ok": b4_ok, "correction_reachable": reachable, "stale_active": stale_active,
               "b5_kill": b5_kill, "operational_rows_in_chapter": op_in_chapter}

    if not a3_ok:
        verdict, failing = "INVALID", "A3 (batch provenance)"
    elif a2_kill or b5_kill:
        verdict, failing = "KILL", ("A2 (ungrounded atom)" if a2_kill else "B5 (chapter kill)")
    elif b1_ok and b2_ok and b3_ok and b4_ok and byte_stable:
        verdict, failing = "HOLDS", None
    else:
        failing = next(
            name for name, ok in [
                ("A1", stage_a["a1_ok"]), ("B1", b1_ok), ("B2", b2_ok),
                ("B3", b3_ok), ("B4", b4_ok), ("byte_stable", byte_stable),
            ] if not ok
        )
        verdict = "INCONCLUSIVE"
    return {"verdict": verdict, "failing_stage": failing, "byte_stable": byte_stable,
            "stage_a": stage_a, "stage_b": stage_b,
            "per_schema_grounding": static["per_schema_grounding"]}


async def _run_manifest_sittings(
    args: argparse.Namespace, store_dir: Path, manifest: dict, loop_llm: object,
    token_counter: object, model_label: str,
) -> tuple[list[dict], dict]:
    """Drive the manifest's days/queries as `designed-{day}` sittings through the REAL loop,
    recording query_id per driven turn. The seam-ON ConsolidationPass runs ONCE at grading over
    the accumulated store (as the --history runner does today) — equivalent to per-day idle
    consolidation for grading, and robust to the offline fast-run's same-second timestamps (a
    per-day pass would advance the mining watermark to that shared ts and skip later days).
    Returns (sittings, turn_query_map: {day: [(qid, topic, text)]}). Mirrors _run_sittings'
    composition + the MOVE-0b sitting TurnFailed gate."""
    from localharness.cli.session_accumulator import SessionAccumulator, derive_session_summary
    from localharness.memory.user_signals import UserSignalDetector

    events_path = store_dir / "agents" / args.agent / "bus-events.jsonl"
    sittings: list[dict] = []
    turn_query_map: dict[str, list] = {}
    cap = args.max_turns_per_day
    for day in manifest["days"]:
        sid = f"designed-{day}"
        day_qs = [q for q in manifest["queries"] if q["day"] == day]
        if cap:
            day_qs = day_qs[:cap]
        if (store_dir / "KILL").exists():
            print(f"KILL file present — stopping accumulation before sitting {day}.")
            break
        if not args.offline and not _watchdog_ok():
            raise _WatchdogAbort(f"MemAvailable below {_MIN_MEM_GIB} GiB before sitting {day}")

        bus = EventBus(persist_path=events_path)
        store = MemoryStore(agent_id=args.agent, division_id="", org_id="",
                            base_dir=str(store_dir), bus=bus)
        await store.open()
        await store.create_session(sid, budget={}, model=model_label,
                                   context_tokens_available=_LOOP_CONTEXT_TOKENS)
        wg = WriteGate(store, bus, args.agent); await wg.open()
        pg_cfg = PredictiveGateConfig(); pg_cfg.write_live = True
        pg = PredictiveGate(store, bus, args.agent, pg_cfg); await pg.open()
        usig = UserSignalDetector(store, bus, args.agent, pg_cfg); await usig.open()
        pw = PredictiveWriteGate(store, bus, args.agent, pg_cfg); await pw.open()
        acc = SessionAccumulator(bus, args.agent); await acc.open()
        loop = await _build_loop(args, store_dir, store, bus, sid, loop_llm, token_counter)

        failed = 0

        def _on_turn_failed(_ev: object) -> None:
            nonlocal failed
            failed += 1

        bus.subscribe(TurnFailed, _on_turn_failed, agent_id=args.agent)
        driven: list = []
        exit_reason = "complete"
        try:
            for q in day_qs:
                if (store_dir / "KILL").exists():
                    exit_reason = "kill_file"
                    break
                if not args.offline and not _watchdog_ok():
                    exit_reason = "watchdog_abort"
                    raise _WatchdogAbort(f"MemAvailable below {_MIN_MEM_GIB} GiB mid-sitting {day}")
                await bus.publish(UserMessage(agent_id=args.agent, session_id=sid,
                                              content=q["text"], channel="terminal"))
                await loop.run_turn(task=q["text"])
                driven.append((q["id"], q["topic"], q["text"]))
        finally:
            await acc.close()
            try:
                await store.end_session(sid, exit_reason=exit_reason,
                                        summary=derive_session_summary(acc),
                                        turn_count=acc.turn_count, action_count=acc.action_count,
                                        tokens_in=acc.tokens_in, tokens_out=acc.tokens_out)
            except Exception:  # noqa: BLE001
                pass
            for g in (pw, usig, pg, wg):
                try:
                    await g.close()
                except Exception:  # noqa: BLE001
                    pass
            await store.close()

        turn_query_map[day] = driven
        sittings.append({"day": day, "session_id": sid, "turns": len(driven), "turn_failed": failed})
        print(f"designed sitting {day}: {len(driven)}/{len(day_qs)} queries ({failed} failed)")
        if driven and failed / len(driven) > _TURN_FAILED_RATE_MAX:
            raise _MeasurementInvalid(day=day, session_id=sid, failed=failed, turns=len(driven))
    return sittings, turn_query_map


# ---------------------------------------------------------------------------
# Report (owner-facing, plain language first, hostile-read-proof)
# ---------------------------------------------------------------------------
def _report(v: dict) -> str:
    live_mode = v["method"] == "live_session_accumulation"
    L: list[str] = []
    add = L.append
    add("# SEMA-05 month-in-a-day — verdict\n")
    if live_mode:
        add(
            "The owner's REAL month of queries was re-run LIVE through the harness — the same "
            "run_turn hot path the CLI uses — one fresh sitting per original day, against an "
            "ISOLATED agent home. The seam-ON idle pass then wrote chapter(s), graded against the "
            "pre-committed method (36-SEMA05-GRADING.md). An honest KILL is a successful outcome.\n"
        )
    else:
        add(
            "Real accumulated events were replayed through the shipped gates into an ISOLATED store, "
            "the seam-ON idle pass wrote chapter(s), and the result is graded against the "
            "pre-committed method (36-SEMA05-GRADING.md). An honest KILL is a successful outcome.\n"
        )
    add(f"**VERDICT: {v['verdict']}**  ({'offline rehearsal' if v['offline'] else 'live subject model'})\n")

    if live_mode:
        add("## Method amendment (pre-run, owner-directed)\n")
        add(
            "- The accumulation method was changed BEFORE the run at owner direction (Discord "
            "2026-07-05 17:53): the owner's real month queries are re-run live through the harness "
            "in fresh per-day sittings, instead of replaying old stored trace/DB shapes — so the "
            "proof exercises the live end-to-end pipeline and the ≥2-sittings stability gate is "
            "satisfied by real temporal structure."
        )
        add(
            f"- The grading doc is UNTOUCHED: committed {v['grading_committed']} "
            f"(`{v['grading_doc']}`), before any run. Bars applied verbatim below."
        )
        cap = v["max_turns_per_day"]
        add(
            f"- Sittings driven: **{v['days']}** (one per original day) | turns driven: "
            f"**{v['turns_driven']}** of {v['queries_total']} extracted queries"
            + (f" (capped at {cap}/day)" if cap else " (uncapped)") + "."
        )
        add("- Per-sitting turn counts (disclosure):")
        add("")
        add("| day | session id | turns |")
        add("|-----|------------|-------|")
        for s in v["sittings"] or []:
            add(f"| {s['day']} | {s['session_id']} | {s['turns']} |")
        add("")

    add("## Proxy-independent metrics (lead)\n")
    add(f"- Zero-tool answer (BINARY): **{v['zero_tool_answered']}** — tool calls: **{v['tool_calls']}**")
    p = v["probe"]
    add(
        f"- Measured by a REAL probe turn (session `{p['session_id']}`, same loop machinery, "
        f"tools registered, chapter-bearing index injected): the question was actually asked; "
        f"tool_calls = the turn's emitted Action(tool_call) events."
    )
    add(f"- Domain question: {p['question']}")
    add(f"- Probe answer: \"{p['answer'][:300]}\"")
    add(f"- Chapter content substantively in the answer (report-only, not a gate): **{p['chapter_content_in_answer']}**")
    add(f"- N real accumulated lessons (isolated store): **{v['n_lessons']}**  |  chapters written: **{v['schemas_written']}**")
    if live_mode:
        add(f"- Turns driven: {v['turns_driven']}  |  isolated-home transcript records: {v['transcript_records']}\n")
    else:
        add(f"- Events replayed: {v['events_replayed']}  |  transcript records loaded: {v['transcript_records']}\n")

    add("## KILL re-check (mechanical — any schema token not derivable from its members)\n")
    add(f"- kill_triggered: **{v['kill_triggered']}**")
    for p in v["per_schema_grounding"]:
        add(
            f"  - `{p['key']}`: grounded={p['grounded']} "
            f"(majority={p['grounded_majority']}, unverified_numbers={p['unverified_numbers']})"
        )
    add("")

    add("## Secondary proxies (within Phase-31 bounds = byte-stability + reported churn)\n")
    add(f"- Byte-stability (double-render ==): **{v['byte_stable']}**")
    add(f"- churn_rate: **{v['churn_rate']:.3f}** (qualitative bound — no upstream numeric ceiling exists)\n")

    add("## Reconciliation sub-provable (grade the DRAIN, both shapes)\n")
    r = v["reconcile"]
    add(f"- Shape split of replayed correction_pending rows: shape_a(restorable)={v['shape_a']}, shape_b(quarantine-only)={v['shape_b']}")
    add(
        f"- Dispositions: confirmed={r['confirmed']} confirmed_corrected={r['confirmed_corrected']} "
        f"retired={r['retired']} reverted_restored={r['reverted_restored']} "
        f"reverted_cleared={r['reverted_cleared']} undecided={r['undecided']}"
    )
    add(f"- **Rows DRAINED by a REVERT: {r['drained']}** (restore for shape a, else clear for shape b)")
    add(f"- Drain bar met (>=1 revert drained a row, or N/A when no correction rows): **{v['drain_ok']}**\n")

    add("## Sensitivity (mandatory)\n")
    s = v["sensitivity"]
    add(f"- Stricter grounding (super-majority) still holds: **{s['grounding_supermajority_holds']}**")
    add(
        f"- Stricter stability (cluster_min_sessions+1) still forms a chapter: "
        f"**{s['cluster_min_sessions_plus1_holds']}** "
        f"({s['stable_clusters_at_min_sessions_plus1']} stable clusters)"
    )
    add(f"- {s['notes']}\n")

    add("## Live-store organic state (WATCH ITEM — read-only, NOT a gate)\n")
    ls = v["live_store"]
    add(
        f"- source: {ls['source']}  |  organic chapters: {ls['schemas']}  |  "
        f"correction_pending: {ls['correction_pending']}  |  promoted lessons: {ls['learned']}"
    )
    add(
        "- The live store is NEVER mutated (only read-only copies are consumed). Its first organic "
        "chapter is a post-phase watch item, not this phase's gate.\n"
    )

    add("## Sensitivity re-grade — human note (fill at run time)\n")
    add(
        "> The runner auto-computed the two re-grades above. Human: confirm the headline zero-tool "
        "verdict holds (or record which assumption flips it and why). Read the actual chapter text "
        "in the verdict's per-schema block before signing off.\n"
    )
    return "\n".join(L)


def _manifest_report(v: dict) -> str:
    """Owner-facing report for the designed-month grade — plain verdict first, then the per-stage
    attribution (grouping quality is the headline), hostile-read-proof."""
    L: list[str] = []
    a, b = v["stage_a"], v["stage_b"]
    fail = f"  (failing stage: {v['failing_stage']})" if v["failing_stage"] else ""
    L.append("# Designed-month grade — verdict\n")
    L.append(f"**VERDICT: {v['verdict']}**{fail}  ({'offline rehearsal' if v['offline'] else 'live subject model'})\n")
    L.append(f"Graded against `{v['grading_doc']}` (pre-committed). Manifest: `{v['manifest']}`.\n")
    L.append("## Stage A — extraction\n")
    L.append(f"- A1 recall: **{a['a1_recall']}** ({a['a1_covered']}/{a['a1_topic_sittings']} topic-sittings "
             f"with a grounded atom) — pass(>=0.80): **{a['a1_ok']}**")
    L.append(f"- A2 grounding kill: **{a['a2_kill']}**  ·  A3 per-atom provenance ok: **{a['a3_provenance_ok']}**  "
             f"·  sem atoms mined: **{a['sem_atoms']}**\n")
    L.append("## Stage B — grouping (THE HEADLINE)\n")
    L.append(f"- B1 chapter recall: **{b['b1_ok']}** (formed+null {b['b1_chapters_or_null']}; "
             f"formed={b['formed_topics']}, correct-null={b['nullable_topics']})")
    L.append(f"- B2 membership: **{b['b2_ok']}**  ·  overall **ARI = {b['ari']}**")
    for pc in b["per_chapter"]:
        L.append(f"  - `{pc['label']}`: {pc['members']} members, precision {pc['precision']}, recall {pc['recall']}")
    L.append(f"- B3 distractor exclusion: **{b['b3_ok']}** ({b['noise_in_chapter']} noise atoms in chapters)")
    L.append(f"- B4 correction arc: **{b['b4_ok']}** (corrected reachable={b['correction_reachable']}, "
             f"stale still active={b['stale_active']})")
    L.append(f"- B5 chapter kill: **{b['b5_kill']}**  ·  operational rows in any chapter (ruling c): "
             f"**{b['operational_rows_in_chapter']}**  ·  byte-stable: **{v['byte_stable']}**\n")
    L.append("## Stage C — behavior (secondary, never the headline)\n")
    for c in v["stage_c"]:
        L.append(f"- `{c['topic']}`: {c['keyword_hits']} keyword hits (>=2: {c['answer_has_min2_hits']}), "
                 f"zero-tool bonus: {c['zero_tool_bonus']}")
    L.append("")
    L.append(f"Sensitivity: supermajority-grounding holds={v['sensitivity']['grounding_supermajority_holds']}, "
             f"min_sessions+1 holds={v['sensitivity']['cluster_min_sessions_plus1_holds']}.\n")
    return "\n".join(L)


async def _run_designed_month(args: argparse.Namespace, results: Path, store_dir: Path) -> int:
    """MOVE 4 orchestration: load the manifest, drive `designed-{day}` sittings (consolidation
    between days), grade Stages A/B + the verdict (pure), then Stage C probes (secondary)."""
    manifest = _load_manifest(Path(args.manifest))
    if args.offline:
        # The offline loop's recovery-read file must be NEUTRAL: reading the manifest (or any file
        # with cross-day query text) would leak every day's content into every day's transcript,
        # collapsing per-atom source provenance. A generic note keeps each day's corpus day-local.
        neutral = store_dir / "_recovery_note.txt"
        neutral.parent.mkdir(parents=True, exist_ok=True)
        neutral.write_text("recovery note: the requested item was located and handled.\n")
        m_llm: object = _OfflineFakeLLM()
        m_loop_llm: object = _OfflineLoopLLM(str(neutral), str(store_dir))
        m_tc: object = None
    else:
        if not args.model or not args.base_url:
            print("REFUSED: live manifest mode requires --model and --base-url.", file=sys.stderr)
            return 2
        if not _watchdog_ok():
            print("ABORT (machine-safety): MemAvailable below threshold — run attended.", file=sys.stderr)
            return 1
        from localharness.agent.context import TokenCounter
        from localharness.provider.client import LLMClient, LLMConfig
        client = LLMClient(LLMConfig(base_url=args.base_url, model=args.model))
        await client.detect_capabilities()
        m_loop_llm, m_llm = client, LLMTextAdapter(client)
        m_tc = TokenCounter(base_url=args.base_url, model=args.model)

    results.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    try:
        sittings, tqm = await _run_manifest_sittings(
            args, store_dir, manifest, m_loop_llm, m_tc,
            model_label=args.model or "offline-fake",
        )
    except _WatchdogAbort as exc:
        print(f"ABORT (machine-safety): {exc}.", file=sys.stderr)
        return 1
    except _MeasurementInvalid as exc:  # MOVE 0b validity gate
        invalid = {
            "verdict": "INVALID", "mode": "designed_month_manifest",
            "reason": f"sitting TurnFailed rate > {_TURN_FAILED_RATE_MAX:.0%} — {exc}",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "grading_doc": _GRADING_DOC_36_1,
            "invalid_sitting": {"day": exc.day, "session_id": exc.session_id,
                                "failed": exc.failed, "turns": exc.turns},
        }
        (results / "verdict.json").write_text(json.dumps(invalid, indent=2) + "\n", encoding="utf-8")
        print(f"INVALID (measurement failure): {exc}", file=sys.stderr)
        return 1
    if sum(s["turns"] for s in sittings) == 0:
        print("PROCESSING FAILURE: zero turns driven from the manifest.", file=sys.stderr)
        return 1

    # GRADING-phase seam-ON consolidation (as today's --history runner): one pass over the
    # accumulated month mines every sitting's atoms, clusters them, and settles corrections.
    if not args.offline and not _watchdog_ok():
        print("ABORT (machine-safety): MemAvailable dropped below threshold pre-consolidation.", file=sys.stderr)
        return 1
    store = MemoryStore(agent_id=args.agent, division_id="", org_id="", base_dir=str(store_dir))
    await store.open()
    try:
        pass_report = await ConsolidationPass(
            store, MemoryConsolidationConfig(reconcile_enabled=True), llm=m_llm
        ).run()
        grade = await _grade_designed_month(store, manifest, tqm)
        schema_values = [s.value for s in await _active(store, "node_kind='schema'")]
        sens = await _sensitivity(store, MemoryConsolidationConfig())
    finally:
        await store.close()
    # Ruling 4b: per_schema_grounding carries EVERY chapter-writer attempt — the static KILL
    # re-check entries for written schemas, PLUS the rejected attempts (reason + grounding) that
    # were previously invisible ('no chapter written' left an empty list and no forensic trail).
    per_schema = [dict(p, written=True) for p in grade["per_schema_grounding"]]
    per_schema += [a for a in pass_report.schema_attempts if not a.get("written")]

    # Stage C (secondary): one domain probe per formed chapter; keyword-hit count only.
    stage_c: list[dict] = []
    for pc in grade["stage_b"]["per_chapter"]:
        topic = pc["label"]
        kws = [k.lower() for k in manifest["topics"].get(topic, {}).get("keywords", [])]
        if not args.offline and not _watchdog_ok():
            break
        probe = await _probe_turn(
            args, store_dir, f"what do you know about {topic.replace('_', ' ')}?",
            m_loop_llm, m_tc, schema_values=schema_values, model_label=args.model or "offline-fake",
        )
        ans = (probe.get("answer") or "").lower()
        hits = sum(1 for k in kws if k in ans)
        stage_c.append({"topic": topic, "keyword_hits": hits, "answer_has_min2_hits": hits >= 2,
                        "zero_tool_bonus": probe["tool_calls"] == 0 and not probe["turn_failed"]})

    v = {
        "verdict": grade["verdict"], "failing_stage": grade["failing_stage"],
        "mode": "designed_month_manifest", "offline": bool(args.offline),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "grading_doc": _GRADING_DOC_36_1, "manifest": str(Path(args.manifest).resolve()),
        "sittings": sittings, "days": len(sittings), "max_turns_per_day": args.max_turns_per_day,
        "stage_a": grade["stage_a"], "stage_b": grade["stage_b"], "stage_c": stage_c,
        "byte_stable": grade["byte_stable"], "per_schema_grounding": per_schema,
        "sensitivity": sens, "duration_s": round(time.monotonic() - t0, 2),
    }
    (results / "verdict.json").write_text(json.dumps(v, indent=2) + "\n", encoding="utf-8")
    (results / "report.md").write_text(_manifest_report(v), encoding="utf-8")
    a, b = grade["stage_a"], grade["stage_b"]
    print(
        f"DESIGNED-MONTH {v['verdict']}" + (f" (failing: {v['failing_stage']})" if v["failing_stage"] else "")
        + f" | atoms={a['sem_atoms']} A1={a['a1_recall']} ARI={b['ari']} "
        + f"B1={b['b1_ok']} B2={b['b2_ok']} B3={b['b3_ok']} B4={b['b4_ok']} byte_stable={v['byte_stable']}"
    )
    print(f"report: {results / 'report.md'}")
    return 1 if v["verdict"] == "INVALID" else 0


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
async def run(args: argparse.Namespace) -> int:
    results = Path(args.results).expanduser().resolve()
    # GUARD (before any output): never contaminate the shared bench results dir.
    if "bench/results" in str(results):
        print(
            "REFUSED: --results points inside bench/results (contaminated shared dir). Use an "
            "isolated dir (e.g. ~/.localharness/sema05-reports/...).",
            file=sys.stderr,
        )
        return 2
    # GUARD (minor 9): outputs must never land inside a live agent store either.
    if "/.localharness/agents/" in str(results):
        print(
            "REFUSED: --results points inside ~/.localharness/agents/ (a live agent store). "
            "Use an isolated dir (e.g. ~/.localharness/sema05-reports/...).",
            file=sys.stderr,
        )
        return 2
    store_dir = Path(args.store).expanduser().resolve()
    # GUARD: the provable MUST use an isolated store — never the live agent store.
    if "/.localharness/agents/" in str(store_dir):
        print(
            "REFUSED: --store points at a live agent store. The provable runs against an ISOLATED "
            "fresh store; the real trace is copied read-only.",
            file=sys.stderr,
        )
        return 2

    # MOVE 4 — the designed-month manifest mode: a self-contained flow (drive from a ground-truth
    # manifest, grade grouping directly, Stages A/B/C). Bypasses the history/trace modes below.
    if args.manifest:
        return await _run_designed_month(args, results, store_dir)

    # GUARD: exactly ONE accumulation mode (the owner-directed live mode, or the legacy trace).
    if bool(args.history) == bool(args.trace):
        print(
            "REFUSED: pass exactly one of --history (live-session accumulation — the provable's "
            "method) or --trace (legacy trace replay cross-check).",
            file=sys.stderr,
        )
        return 2
    live_mode = bool(args.history)

    if live_mode:
        history_path = Path(args.history).expanduser().resolve()
        # GUARD (minor 9): never point the runner AT the live store's own files — a COPY is
        # required (read-only intent is not enough protection against future edits).
        if "/.localharness/agents/" in str(history_path):
            print(
                "REFUSED: --history points at a live agent store file. Copy it to a scratch dir "
                "first (cp ~/.localharness/agents/<agent>/history.jsonl <scratch>/).",
                file=sys.stderr,
            )
            return 2
        bus_events = None
        live_db = history_path.parent / "memory.db"  # sibling of the copied history (read-only)
        if not history_path.exists():
            print(f"PROCESSING FAILURE: history file not found: {history_path}", file=sys.stderr)
            return 1
    else:
        tp = Path(args.trace).expanduser().resolve()
        if "/.localharness/agents/" in str(tp):  # minor 9: same copy-required rule for traces
            print(
                "REFUSED: --trace points at a live agent store path. Copy the trace to a scratch "
                "dir first.",
                file=sys.stderr,
            )
            return 2
        if tp.is_dir():
            bus_events, history_path, live_db = (
                tp / "bus-events.jsonl", tp / "history.jsonl", tp / "memory.db",
            )
        else:  # a bus-events file was passed directly — resolve siblings from its parent
            bus_events, history_path, live_db = (
                tp, tp.parent / "history.jsonl", tp.parent / "memory.db",
            )
        if not bus_events.exists():
            print(f"PROCESSING FAILURE: bus-events trace not found: {bus_events}", file=sys.stderr)
            return 1

    # LLM selection + machine-safety. ONE client serves both seams in live mode: the loop uses
    # it directly (stream_complete) and the idle passes go through LLMTextAdapter — one serial
    # inference gate for everything.
    token_counter = None
    if args.offline:
        llm: object = _OfflineFakeLLM()  # the idle-pass (seam-contract) fake
        loop_llm: object = _OfflineLoopLLM(str(history_path), str(store_dir))
    else:
        if not args.model or not args.base_url:
            print("REFUSED: live mode requires --model and --base-url.", file=sys.stderr)
            return 2
        if not _watchdog_ok():
            g = _mem_available_gib()
            print(
                f"ABORT (machine-safety): MemAvailable {g:.1f} GiB < {_MIN_MEM_GIB} GiB — refusing to "
                "launch a live vLLM prefill (this box hard-hung twice in 24h). Run attended when free.",
                file=sys.stderr,
            )
            return 1
        from localharness.agent.context import TokenCounter
        from localharness.provider.client import LLMClient, LLMConfig
        client = LLMClient(LLMConfig(base_url=args.base_url, model=args.model))
        # FIDEL-04: probe the real tool_call_mode + served window (never raises) — the same
        # capability detection start_cmd runs before its loop.
        await client.detect_capabilities()
        loop_llm = client
        llm = LLMTextAdapter(client)
        token_counter = TokenCounter(base_url=args.base_url, model=args.model)

    results.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    # ACCUMULATE (live mode): re-run the owner's real month queries, one sitting per day.
    sittings: list[dict] | None = None
    queries_total = 0
    if live_mode:
        day_queries = _extract_day_queries(history_path, args.agent)
        queries_total = sum(len(q) for _, q in day_queries)
        if not day_queries:
            print("PROCESSING FAILURE: no user messages extracted from --history for this agent.",
                  file=sys.stderr)
            return 1
        print(f"extracted {queries_total} real queries across {len(day_queries)} original days")
        try:
            sittings = await _run_sittings(
                args, store_dir, day_queries, loop_llm, token_counter,
                model_label=args.model or "offline-fake",
            )
        except _WatchdogAbort as exc:
            print(f"ABORT (machine-safety): {exc}. Partial isolated store left for inspection.",
                  file=sys.stderr)
            return 1
        except _MeasurementInvalid as exc:
            # MOVE 0b: the instrument failed, not the subject. Stamp verdict INVALID with the
            # failing sitting named — never grade a dead store (the SEMA-05 P0 silent 2-as-15).
            invalid = {
                "verdict": "INVALID",
                "reason": f"sitting TurnFailed rate > {_TURN_FAILED_RATE_MAX:.0%} — {exc}",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "method": "live_session_accumulation",
                "grading_doc": _GRADING_DOC,
                "grading_committed": _GRADING_COMMITTED,
                "invalid_sitting": {"day": exc.day, "session_id": exc.session_id,
                                    "failed": exc.failed, "turns": exc.turns},
                "sittings": sittings,
            }
            (results / "verdict.json").write_text(json.dumps(invalid, indent=2) + "\n",
                                                  encoding="utf-8")
            print(f"INVALID (measurement failure): {exc}", file=sys.stderr)
            return 1
        if sum(s["turns"] for s in sittings) == 0:
            print("PROCESSING FAILURE: zero turns driven.", file=sys.stderr)
            return 1

    # GRADE (both modes — the LOCKED bars, unchanged). Store handle #1 for accumulation-side
    # loading + the idle LLM passes + static checks; it CLOSES before the probe turn opens
    # its own fresh composition (one clean handle per phase, like one process per sitting).
    method = "live_session_accumulation" if live_mode else "trace_replay"
    store = MemoryStore(agent_id=args.agent, division_id="", org_id="", base_dir=str(store_dir))
    await store.open()
    try:
        if live_mode:
            events_replayed = None  # not a trace replay — turns_driven carries the volume
            transcript_records = len(await store.get_history(limit=1_000_000))
        else:
            events_replayed = await _replay(store, bus_events, args.agent)
            transcript_records = await _load_transcript(store, history_path)
            if events_replayed == 0:
                print("PROCESSING FAILURE: zero primary events replayed (empty/wrong trace or agent).",
                      file=sys.stderr)
                return 1

        # Snapshot the correction-queue shape split BEFORE reconcile drains it.
        shape_a, shape_b = await _correction_shape_split(store)

        # MAJOR 6a: a KILL file must also stop the GRADING-phase LLM work (consolidation
        # generations + reconcile looks) — honest ABORTED verdict, never a silent skip.
        if (store_dir / "KILL").exists():
            aborted = {
                "verdict": "ABORTED",
                "reason": "KILL file present before grading-phase LLM work",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "method": method,
                "grading_doc": _GRADING_DOC,
                "grading_committed": _GRADING_COMMITTED,
                "sittings": sittings,
            }
            (results / "verdict.json").write_text(json.dumps(aborted, indent=2) + "\n",
                                                  encoding="utf-8")
            print("ABORTED: KILL file present — no grading-phase LLM work performed.",
                  file=sys.stderr)
            return 1

        # Seam-ON pass (chapter-writer + mining). reconcile_enabled=False so reconcile runs as an
        # owned direct call below that returns the six shape-aware buckets the grading doc requires.
        cfg = MemoryConsolidationConfig(reconcile_enabled=False)
        if not args.offline and not _watchdog_ok():  # re-check right before the heavy generation
            print("ABORT (machine-safety): MemAvailable dropped below threshold pre-consolidation.",
                  file=sys.stderr)
            return 1
        report = await ConsolidationPass(store, cfg, llm=llm).run()
        cancel = asyncio.Event()
        rec = await reconcile_corrections(store, llm, cancel, ttl_looks=cfg.reconcile_ttl_looks)

        static = await _static_checks(store)
        sens = await _sensitivity(store, cfg)
        live = _live_store_counts(live_db, args.agent)
    finally:
        await store.close()

    # BLOCKER 2: the REAL probe turn — ASK the domain question through the same loop machinery
    # against the consolidated store (chapter-bearing index injected, tools registered) and
    # count the turn's real tool-call events. Never a substring heuristic.
    if not args.offline and not _watchdog_ok():
        print("ABORT (machine-safety): MemAvailable dropped below threshold pre-probe.",
              file=sys.stderr)
        return 1
    probe = await _probe_turn(
        args, store_dir, static["domain_question"], loop_llm, token_counter,
        schema_values=static["schema_values"], model_label=args.model or "offline-fake",
    )
    # The zero-tool gate (locked bar: 'answered BY the Knowledge line'): the turn must have
    # SUCCEEDED (run_turn never returns empty — error/kill summaries are prose, so the
    # failure signal is the TurnFailed event), used ZERO tools, and the answer must carry
    # the chapter's content (attribution; substring/paraphrase-tolerant majority-token net).
    # A failed or off-chapter probe demotes to INCONCLUSIVE — false-negative is the only
    # permitted error direction.
    zero_tool_answered = (
        probe["tool_calls"] == 0
        and not probe["turn_failed"]
        and probe["chapter_content_in_answer"]
    )

    drained = rec.reverted_restored + rec.reverted_cleared
    reconcile_total = (rec.confirmed + rec.confirmed_corrected + rec.retired
                       + drained + rec.undecided)
    drain_ok = (reconcile_total == 0) or (drained >= 1)

    # Mechanical verdict (36-SEMA05-GRADING.md): KILL leads, then HOLDS, else INCONCLUSIVE.
    if static["kill_triggered"]:
        verdict = "KILL"
    elif (static["schemas_written"] >= 1 and zero_tool_answered
          and static["byte_stable"] and drain_ok):
        verdict = "HOLDS"
    else:
        verdict = "INCONCLUSIVE"

    v = {
        "verdict": verdict,
        "offline": bool(args.offline),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "grading_doc": _GRADING_DOC,
        "grading_committed": _GRADING_COMMITTED,
        "method": method,
        "sittings": sittings,
        "days": len(sittings) if sittings else None,
        "queries_total": queries_total if live_mode else None,
        "turns_driven": sum(s["turns"] for s in sittings) if sittings else None,
        "max_turns_per_day": args.max_turns_per_day,
        "n_lessons": static["n_lessons"],
        "domain_token": static["domain_token"],
        "domain_question": static["domain_question"],
        "zero_tool_answered": zero_tool_answered,
        "tool_calls": probe["tool_calls"],
        "probe": probe,
        "schemas_written": static["schemas_written"],
        "pass_schemas_written": report.schemas_written,
        # Ruling 4b: written schemas (the static KILL re-check) + the pass's rejected
        # chapter-writer attempts — a failed write is observable, never an empty list.
        "per_schema_grounding": (
            [dict(p, written=True) for p in static["per_schema_grounding"]]
            + [a for a in report.schema_attempts if not a.get("written")]
        ),
        "kill_triggered": static["kill_triggered"],
        "churn_rate": report.churn_rate,
        "byte_stable": static["byte_stable"],
        "mined": report.mined,
        "reconcile": {
            "confirmed": rec.confirmed,
            "confirmed_corrected": rec.confirmed_corrected,
            "retired": rec.retired,
            "reverted_restored": rec.reverted_restored,
            "reverted_cleared": rec.reverted_cleared,
            "undecided": rec.undecided,
            "drained": drained,
        },
        "shape_a": shape_a,
        "shape_b": shape_b,
        "drain_ok": drain_ok,
        "sensitivity": sens,
        "live_store": live,
        "events_replayed": events_replayed,
        "transcript_records": transcript_records,
        "duration_s": round(time.monotonic() - t0, 2),
    }
    (results / "verdict.json").write_text(json.dumps(v, indent=2) + "\n", encoding="utf-8")
    (results / "report.md").write_text(_report(v), encoding="utf-8")

    print(
        f"SEMA-05 {verdict} [{method}] | N={static['n_lessons']} "
        f"chapters={static['schemas_written']} "
        f"zero_tool={zero_tool_answered} tool_calls={probe['tool_calls']} "
        f"kill={static['kill_triggered']} byte_stable={static['byte_stable']} "
        f"churn={report.churn_rate:.3f} | reconcile drained={drained} "
        f"(shape_a={shape_a} shape_b={shape_b}) | live_store_chapters={live['schemas']}"
    )
    print(f"report: {results / 'report.md'}")
    return 0  # HOLDS / KILL / INCONCLUSIVE are ALL a successful measurement.


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SEMA-05 month-in-a-day: live re-run of the owner's real month queries "
                    "(or legacy trace replay) into an isolated store + seam-ON pass + verdict"
    )
    p.add_argument("--history", default=None,
                   help="COPIED real history.jsonl (READ-ONLY; a copy is REQUIRED — live-store "
                        "paths are refused). LIVE-SESSION ACCUMULATION — the provable's method: "
                        "each original day's real queries re-run through the REAL agent loop as "
                        "one fresh sitting (root aliases unified; the real month = 148 queries / "
                        "15 days). Sibling memory.db (if present) is read for the live-store "
                        "watch item.")
    p.add_argument("--trace", default=None,
                   help="LEGACY cross-check: dir (or bus-events file) with the COPIED real trace "
                        "to replay through the shipped gates. READ-ONLY. Mutually exclusive with "
                        "--history.")
    p.add_argument("--manifest", default=None,
                   help="MOVE 4 designed-month mode: a ground-truth manifest JSON (query -> "
                        "expected topic). Drives `designed-{day}` sittings with consolidation "
                        "between days, then grades GROUPING directly (Stages A/B/C per "
                        "36.1-DESIGNED-MONTH-GRADING.md). Bypasses --history/--trace.")
    p.add_argument("--store", required=True,
                   help="ISOLATED fresh store dir (base_dir). NEVER a live agent store.")
    p.add_argument("--results", required=True,
                   help="ISOLATED output dir (verdict.json + report.md). Refuses bench/results.")
    p.add_argument("--agent", default="orchestrator",
                   help="Agent id to run under (must match the history/trace records; default orchestrator).")
    p.add_argument("--model", default=None, help="Subject model (live path only).")
    p.add_argument("--base-url", default=None, help="vLLM base URL (live path only).")
    p.add_argument("--max-turns-per-day", type=int, default=None,
                   help="Optional cap on re-run queries per sitting (default: uncapped).")
    p.add_argument("--offline", action="store_true",
                   help="Use the bundled fakes + skip the live vLLM/watchdog (CI).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
