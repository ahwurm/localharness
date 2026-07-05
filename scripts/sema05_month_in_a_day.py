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
      original day, with real create_session/end_session close-outs.
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
stops the current turn via the loop's own KillWatcher. Offline never touches vLLM. Tools run
FOR REAL during live sittings (bash/web/write included) — attended-only is the containment.

usage (live, ATTENDED only):
  .venv/bin/python scripts/sema05_month_in_a_day.py \
      --history <scratch>/history.jsonl \
      --store <scratch>/isolated-store \
      --results ~/.localharness/sema05-reports/phase36-$(date +%Y%m%dT%H%M%SZ) \
      --model <subject> --base-url <url> [--max-turns-per-day N]

Exit codes: 0 = a measurement was produced (HOLDS / KILL / INCONCLUSIVE all succeed);
            1 = processing failure / watchdog abort; 2 = guard refusal (bench/results, live
            store, or not-exactly-one accumulation mode).
"""
from __future__ import annotations

import argparse
import asyncio
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
    UserMessage,
)
from localharness.memory.clustering import find_stable_clusters  # noqa: E402
from localharness.memory.consolidation import ConsolidationPass  # noqa: E402
from localharness.memory.gate import WriteGate  # noqa: E402
from localharness.memory.idle_llm import LLMTextAdapter, ground_numbers, grounded  # noqa: E402
from localharness.memory.predictive_gate import PredictiveGate  # noqa: E402
from localharness.memory.predictive_write_gate import PredictiveWriteGate  # noqa: E402
from localharness.memory.reconciliation import reconcile_corrections  # noqa: E402
from localharness.memory.sqlite import MemoryStore, _row_to_fact  # noqa: E402

_MIN_MEM_GIB = 30.0  # RANK-06 practice: kill/abort the live LLM path below this MemAvailable.
_LOOP_CONTEXT_TOKENS = 32768  # machine-safety context bound: far below the 96k hard-hang class
_GRADING_DOC = ".planning/phases/36-chapter-writer/36-SEMA05-GRADING.md"
_GRADING_COMMITTED = "2026-07-05T17:43:47Z"  # the LOCKED pre-commitment timestamp (quoted in report)
_ROLE = (
    "Personal assistant on the owner's box. Answer directly and use tools only when the "
    "request actually needs them."
)


class _WatchdogAbort(RuntimeError):
    """Raised when MemAvailable drops below the safety threshold mid-run (live mode)."""


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
        if "colleague would remember" in prompt:  # transcript mining
            return "user got sunburnt today"       # grounded on 'sunburnt' in the span
        return ""  # replay seam + anything else: inert


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
    """Loop-client double for --offline --history: drives the REAL AgentLoop + REAL builtin
    tools deterministically. For a 'please read the {tag} file' task it reads a missing {tag}
    path (a REAL ReadTool error), then a real existing path (recovery), then answers — so the
    shipped WriteGate captures a real resolved_error lesson from real tool events, recurring
    across sittings. Neither fake authors lesson text; the offline chapter prose comes from
    the seam fake's verbatim corpus echo (the honest generator/lesson split)."""

    def __init__(self, existing_path: str, missing_dir: str) -> None:
        self._existing = existing_path
        self._missing_dir = missing_dir
        self._n = 0

        class _Cfg:
            tool_call_mode = "native"
            context_window = _LOOP_CONTEXT_TOKENS

        self.config = _Cfg()

    async def stream_complete(self, messages=None, tools=None, on_token=None):
        self._n += 1
        msgs = messages or []
        last_user = next((m for m in reversed(msgs) if m.get("role") == "user"), {})
        task = str(last_user.get("content", ""))
        if task.startswith("Review your answer"):  # MECH-01 self-check pass (if enabled)
            return _Msg(content="CONFIRMED"), None
        n_tool = 0
        for m in reversed(msgs):  # tool results since the last user turn = the stage
            if m.get("role") == "user":
                break
            if m.get("role") == "tool":
                n_tool += 1
        tag = next((w for w in ("alpha", "beta", "gamma") if w in task), "alpha")
        if n_tool == 0:  # stage 1: read a missing path -> a REAL deterministic tool error
            return _Msg(tool_calls=[_ToolCall(
                id=f"tc{self._n}", name="read",
                arguments={"path": f"{self._missing_dir}/absent-{tag}.md"},
            )]), None
        if n_tool == 1:  # stage 2: recover by reading a real existing path
            return _Msg(tool_calls=[_ToolCall(
                id=f"tc{self._n}", name="read", arguments={"path": self._existing},
            )]), None
        return _Msg(content=f"Read the {tag} file after recovering from the missing path."), None


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
def _extract_day_queries(history_path: Path, agent_id: str) -> list[tuple[str, list[str]]]:
    """READ-ONLY extraction of the owner's real user messages from a copied history.jsonl,
    grouped by ORIGINAL calendar day (local box time — the days the owner actually lived,
    which is what the ≥2-sittings stability gate should honestly measure). Within a day,
    original ts order. Non-user records and other agents' records are ignored."""
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
        if rec.get("agent_id") not in (None, agent_id):
            continue
        content = str(rec.get("content", "")).strip()
        if not content:
            continue
        ts = int(rec.get("ts", 0) or 0)
        day = time.strftime("%Y%m%d", time.localtime(ts))
        by_day.setdefault(day, []).append((ts, content))
    return [(day, [c for _, c in sorted(rows)]) for day, rows in sorted(by_day.items())]


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
    capture real lessons/corrections. Watchdog-checked before every live turn; raises
    _WatchdogAbort on a trip. Returns [{day, session_id, turns}] in run order."""
    # Real-composition imports (lazy: only the --history mode pays them).
    from localharness.agent.context import ContentStore, ContextManager
    from localharness.agent.loop import AgentLoop
    from localharness.agent.permissions import PermissionEvaluator
    from localharness.cli.session_accumulator import SessionAccumulator, derive_session_summary
    from localharness.config.models import AgentConfig
    from localharness.memory.user_signals import UserSignalDetector
    from localharness.tools.builtin import bind_agent_store_tools, register_builtin_tools
    from localharness.tools.registry import ToolRegistry

    events_path = store_dir / "agents" / args.agent / "bus-events.jsonl"
    sittings: list[dict] = []
    for day, queries in day_queries:
        if not args.offline and not _watchdog_ok():
            raise _WatchdogAbort(f"MemAvailable below {_MIN_MEM_GIB} GiB before sitting {day}")
        capped = queries[: args.max_turns_per_day] if args.max_turns_per_day else queries

        # Fresh per-sitting composition (mirrors start_cmd; one continuous bus-events log).
        bus = EventBus(persist_path=events_path)
        store = MemoryStore(agent_id=args.agent, division_id="", org_id="",
                            base_dir=str(store_dir), bus=bus)
        await store.open()
        sid = f"sema05-{day}"
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

        registry = ToolRegistry()
        content_store = ContentStore()
        await register_builtin_tools(registry, memory_store=store, eviction_store=content_store)
        bind_agent_store_tools(registry, content_store)

        a_cfg = AgentConfig(name=args.agent, role=_ROLE)
        a_cfg.context.max_context_tokens = _LOOP_CONTEXT_TOKENS  # machine-safety context bound
        # P-A capability floor, EXACTLY as start_cmd applies it to the root agent: web_*
        # (untrusted-ingest) is DENIED for a bash/write/edit-holding agent (prompt-injection->
        # host hole). Queries needing web ingestion run without web tools — production behavior
        # (the real root delegates ingestion to a subagent, which this bounded run omits).
        from localharness.tools.capabilities import apply_root_capability_floor
        apply_root_capability_floor(a_cfg.tools)
        ctx = ContextManager(
            max_context_tokens=_LOOP_CONTEXT_TOKENS,
            eviction_store=content_store,
            token_counter=token_counter,
        )
        loop = AgentLoop(
            config=a_cfg, llm=loop_llm, bus=bus, context_manager=ctx,
            tool_registry=registry, permission_evaluator=PermissionEvaluator(),
            memory_loader=store,
            compact_md_path=store_dir / "agents" / args.agent / "compact.md",  # isolated
            kill_file_path=store_dir / "KILL",  # attended stop: `touch <store>/KILL`
            session_id=sid,
        )

        turns = 0
        exit_reason = "complete"
        try:
            for q in capped:
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
        sittings.append({"day": day, "session_id": sid, "turns": turns})
        print(f"sitting {day}: {turns}/{len(queries)} real queries re-run (session {sid})")
    return sittings


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


async def _probe(store: MemoryStore) -> dict:
    """Proxy-independent FIRST: zero-tool BINARY + tool-call integer, then the mechanical KILL
    re-check (majority-token grounding + numeric net) per schema, then byte-stability."""
    index = (await store.load_context(index_mode=True)).agent_memory_md
    index2 = (await store.load_context(index_mode=True)).agent_memory_md
    byte_stable = index == index2  # SEMA-04 double-render ==
    knowledge_present = "### Knowledge" in index
    knowledge_section = (
        index.split("### Knowledge", 1)[1].split("\n\n", 1)[0].lower()
        if knowledge_present else ""
    )

    schemas = await _active(store, "node_kind='schema'")
    per_schema: list[dict] = []
    member_tools: list[str] = []
    for s in schemas:
        members = await _schema_members(store, s.id)
        bodies = [m.value for m in members]
        corpus = "\n".join(bodies)
        g_majority = grounded(s.value, corpus)            # the shipped >=50% majority-token net
        unverified = ground_numbers(s.value, bodies)      # the numeric net (empty == clean)
        per_schema.append({
            "key": s.key,
            "grounded": bool(g_majority and not unverified),  # KILL trip: either failing -> kill
            "grounded_majority": bool(g_majority),
            "unverified_numbers": unverified,
        })
        member_tools += [t for t in (_member_tool(m.key) for m in members) if t]

    kill_triggered = any(not p["grounded"] for p in per_schema)
    domain_token = Counter(member_tools).most_common(1)[0][0] if member_tools else None
    zero_tool = bool(knowledge_present and domain_token and domain_token in knowledge_section)
    domain_question = (
        f"What has the harness learned about the `{domain_token}` tool?"
        if domain_token else "What durable domain knowledge has the harness formed?"
    )
    n_lessons = len(await _active(store, "key LIKE 'learned/%'"))
    return {
        "schemas_written": len(schemas),
        "per_schema_grounding": per_schema,
        "kill_triggered": kill_triggered,
        "domain_token": domain_token,
        "domain_question": domain_question,
        "zero_tool_answered": zero_tool,
        "tool_calls": 0 if zero_tool else 1,
        "byte_stable": byte_stable,
        "n_lessons": n_lessons,
        "knowledge_present": knowledge_present,
    }


async def _sensitivity(store: MemoryStore, cfg: MemoryConsolidationConfig) -> dict:
    """§6 mandatory re-grade under >=1 alternate assumption: (1) a stricter super-majority
    grounding bar; (2) a stricter cluster-stability bar (cluster_min_sessions + 1)."""
    schemas = await _active(store, "node_kind='schema'")
    holds_super = True
    for s in schemas:
        members = await _schema_members(store, s.id)
        corpus = "\n".join(m.value for m in members)
        if not _supermajority_grounded(s.value, corpus):
            holds_super = False
    stricter = await find_stable_clusters(store, min_sessions=cfg.cluster_min_sessions + 1)
    return {
        "grounding_supermajority_holds": bool(holds_super),
        "cluster_min_sessions_plus1_holds": len(stricter) >= 1,
        "stable_clusters_at_min_sessions_plus1": len(stricter),
        "notes": (
            "Auto re-grade, post-hoc over the promoted population (members folded but still "
            "confidence>=0.7). If the headline zero-tool verdict flips under either assumption it "
            "is disclosed here, never smoothed (the M1 methodology-sensitivity failure mode)."
        ),
    }


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
    add(f"- Domain question: {v['domain_question']}")
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
    store_dir = Path(args.store).expanduser().resolve()
    # GUARD: the provable MUST use an isolated store — never the live agent store.
    if "/.localharness/agents/" in str(store_dir):
        print(
            "REFUSED: --store points at a live agent store. The provable runs against an ISOLATED "
            "fresh store; the real trace is copied read-only.",
            file=sys.stderr,
        )
        return 2

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
        history_path = Path(args.history).expanduser()
        bus_events = None
        live_db = history_path.parent / "memory.db"  # sibling of the copied history (read-only)
        if not history_path.exists():
            print(f"PROCESSING FAILURE: history file not found: {history_path}", file=sys.stderr)
            return 1
    else:
        tp = Path(args.trace).expanduser()
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
        if sum(s["turns"] for s in sittings) == 0:
            print("PROCESSING FAILURE: zero turns driven.", file=sys.stderr)
            return 1

    # GRADE (both modes — the LOCKED bars, unchanged). Fresh store handle over the isolated home.
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

        probe = await _probe(store)
        sens = await _sensitivity(store, cfg)
        live = _live_store_counts(live_db, args.agent)

        drained = rec.reverted_restored + rec.reverted_cleared
        reconcile_total = (rec.confirmed + rec.confirmed_corrected + rec.retired
                           + drained + rec.undecided)
        drain_ok = (reconcile_total == 0) or (drained >= 1)

        # Mechanical verdict (36-SEMA05-GRADING.md): KILL leads, then HOLDS, else INCONCLUSIVE.
        if probe["kill_triggered"]:
            verdict = "KILL"
        elif (probe["schemas_written"] >= 1 and probe["zero_tool_answered"]
              and probe["byte_stable"] and drain_ok):
            verdict = "HOLDS"
        else:
            verdict = "INCONCLUSIVE"

        v = {
            "verdict": verdict,
            "offline": bool(args.offline),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "grading_doc": _GRADING_DOC,
            "grading_committed": _GRADING_COMMITTED,
            "method": "live_session_accumulation" if live_mode else "trace_replay",
            "sittings": sittings,
            "days": len(sittings) if sittings else None,
            "queries_total": queries_total if live_mode else None,
            "turns_driven": sum(s["turns"] for s in sittings) if sittings else None,
            "max_turns_per_day": args.max_turns_per_day,
            "n_lessons": probe["n_lessons"],
            "domain_token": probe["domain_token"],
            "domain_question": probe["domain_question"],
            "zero_tool_answered": probe["zero_tool_answered"],
            "tool_calls": probe["tool_calls"],
            "schemas_written": probe["schemas_written"],
            "pass_schemas_written": report.schemas_written,
            "per_schema_grounding": probe["per_schema_grounding"],
            "kill_triggered": probe["kill_triggered"],
            "churn_rate": report.churn_rate,
            "byte_stable": probe["byte_stable"],
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
            f"SEMA-05 {verdict} [{v['method']}] | N={probe['n_lessons']} "
            f"chapters={probe['schemas_written']} "
            f"zero_tool={probe['zero_tool_answered']} tool_calls={probe['tool_calls']} "
            f"kill={probe['kill_triggered']} byte_stable={probe['byte_stable']} "
            f"churn={report.churn_rate:.3f} | reconcile drained={drained} "
            f"(shape_a={shape_a} shape_b={shape_b}) | live_store_chapters={live['schemas']}"
        )
        print(f"report: {results / 'report.md'}")
        return 0  # HOLDS / KILL / INCONCLUSIVE are ALL a successful measurement.
    finally:
        await store.close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SEMA-05 month-in-a-day: live re-run of the owner's real month queries "
                    "(or legacy trace replay) into an isolated store + seam-ON pass + verdict"
    )
    p.add_argument("--history", default=None,
                   help="COPIED real history.jsonl (READ-ONLY). LIVE-SESSION ACCUMULATION — the "
                        "provable's method: each original day's real queries re-run through the "
                        "REAL agent loop as one fresh sitting. Sibling memory.db (if present) is "
                        "read for the live-store watch item.")
    p.add_argument("--trace", default=None,
                   help="LEGACY cross-check: dir (or bus-events file) with the COPIED real trace "
                        "to replay through the shipped gates. READ-ONLY. Mutually exclusive with "
                        "--history.")
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
