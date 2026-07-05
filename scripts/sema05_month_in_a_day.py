#!/usr/bin/env python3
"""SEMA-05 month-in-a-day provable — replay REAL accumulated events into an ISOLATED store,
run the seam-ON idle pass, and grade against the pre-committed method (36-SEMA05-GRADING.md).

This is the phase gate for the chapter-writer: "100 lessons -> one honest chapter the index
routes through." An honest KILL is a successful outcome. The grading method is committed BEFORE
this ever runs so results cannot move the goalposts (the M1 methodology lesson).

WHAT IT DOES (mirrors scripts/gate_replay_comparison.py's isolated-store + verdict shape):
  1. REPLAY (real events only, nothing invented): read the copied real bus-events trace READ-ONLY and
     feed its primary events through the SHIPPED gates (WriteGate + PredictiveGate +
     PredictiveWriteGate) into a FRESH isolated store — producing real learned/* lessons + real
     tier:correction_pending rows. The live transcript (history.jsonl) is loaded for mining/deref.
  2. CONSOLIDATE: run the seam-ON ConsolidationPass (chapter-writer + mining) with the subject
     model (live) or a bundled FakeLLM (--offline), then reconcile_corrections directly so the
     verdict can record the six shape-aware disposition buckets (the pass's summed reconcile step
     is regression-proven wired in 36-07; the direct call is only to surface the breakdown).
  3. PROBE (proxy-independent FIRST): render the injected index; is the domain question answered
     by the "### Knowledge" schema line with ZERO tool calls (BINARY)? tool-call count (integer)?
     Re-apply the KILL to every schema (majority-token grounding + numeric net) mechanically.
  4. PROXIES + SENSITIVITY: churn_rate + byte-stability (double-render ==); re-grade under a
     stricter grounding bar and a stricter cluster-stability bar.
  5. EMIT verdict.json + report.md into the ISOLATED --results dir; report the live store's real
     organic counts (read-only) as a WATCH ITEM.

MACHINE-SAFETY (binding — this box hard-hung twice in 24h under vLLM prefill): the live path is
attended-only, char-bounded (idle_llm), and gated by a MemAvailable watchdog that aborts below
~30 GiB. Offline uses a fake LLM and never touches vLLM.

usage (live, ATTENDED only):
  .venv/bin/python scripts/sema05_month_in_a_day.py \
      --trace <scratch-with-copied-bus-events+history+memory.db> \
      --store <scratch>/isolated-store \
      --results ~/.localharness/sema05-reports/phase36-$(date +%Y%m%dT%H%M%SZ) \
      --model <subject> --base-url <url>

Exit codes: 0 = a measurement was produced (HOLDS / KILL / INCONCLUSIVE all succeed);
            1 = processing failure / watchdog abort; 2 = guard refusal (bench/results or live store).
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
    L: list[str] = []
    add = L.append
    add("# SEMA-05 month-in-a-day — verdict\n")
    add(
        "Real accumulated events were replayed through the shipped gates into an ISOLATED store, "
        "the seam-ON idle pass wrote chapter(s), and the result is graded against the "
        "pre-committed method (36-SEMA05-GRADING.md). An honest KILL is a successful outcome.\n"
    )
    add(f"**VERDICT: {v['verdict']}**  ({'offline rehearsal' if v['offline'] else 'live subject model'})\n")

    add("## Proxy-independent metrics (lead)\n")
    add(f"- Zero-tool answer (BINARY): **{v['zero_tool_answered']}** — tool calls: **{v['tool_calls']}**")
    add(f"- Domain question: {v['domain_question']}")
    add(f"- N real accumulated lessons (isolated store): **{v['n_lessons']}**  |  chapters written: **{v['schemas_written']}**")
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
        "- The live store is NEVER mutated (only a read-only copy was replayed). Its first organic "
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

    # LLM selection + machine-safety.
    if args.offline:
        llm: object = _OfflineFakeLLM()
    else:
        if not _watchdog_ok():
            g = _mem_available_gib()
            print(
                f"ABORT (machine-safety): MemAvailable {g:.1f} GiB < {_MIN_MEM_GIB} GiB — refusing to "
                "launch a live vLLM prefill (this box hard-hung twice in 24h). Run attended when free.",
                file=sys.stderr,
            )
            return 1
        from localharness.provider.client import LLMClient, LLMConfig
        llm = LLMTextAdapter(
            LLMClient(LLMConfig(base_url=args.base_url, model=args.model,
                                max_tokens=512, temperature=0.2))
        )

    results.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(agent_id=args.agent, division_id="", org_id="", base_dir=str(store_dir))
    await store.open()
    t0 = time.monotonic()
    try:
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
            "grading_doc": ".planning/phases/36-chapter-writer/36-SEMA05-GRADING.md",
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
            f"SEMA-05 {verdict} | N={probe['n_lessons']} chapters={probe['schemas_written']} "
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
        description="SEMA-05 month-in-a-day: isolated-store replay + seam-ON pass + verdict"
    )
    p.add_argument("--trace", required=True,
                   help="Dir (or bus-events file) with the COPIED real trace: bus-events.jsonl "
                        "[+ history.jsonl + memory.db]. READ-ONLY.")
    p.add_argument("--store", required=True,
                   help="ISOLATED fresh store dir (base_dir). NEVER a live agent store.")
    p.add_argument("--results", required=True,
                   help="ISOLATED output dir (verdict.json + report.md). Refuses bench/results.")
    p.add_argument("--agent", default="orchestrator",
                   help="Agent id to replay under (must match the trace's events; default orchestrator).")
    p.add_argument("--model", default=None, help="Subject model (live path only).")
    p.add_argument("--base-url", default=None, help="vLLM base URL (live path only).")
    p.add_argument("--offline", action="store_true",
                   help="Use the bundled FakeLLM + skip the live vLLM/watchdog (CI).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
