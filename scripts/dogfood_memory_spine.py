#!/usr/bin/env python3
"""Dogfood the v2.0 memory spine end-to-end with REAL components — no fakes, no LLM.

PREDICTION (written before first run — owner ask 2026-07-02): an agent hits
`uv: command not found` on bash_exec in session A and recovers; the same mistake
recurs in session B. A one-off web_fetch timeout also happens once. After the idle
consolidation pass:
  1. BEFORE consolidation the injected memory block shows NOTHING (candidates sit
     below the 0.7 injection threshold — junk never pollutes the prompt).
  2. AFTER, the recurring uv lesson IS in the injected block — payload FIRST, the
     recurrence bookkeeping as suffix ("[recurring: 2 episodes, …]") — so the next
     session's system prompt knows it with zero user action, zero LLM calls.
  3. The one-off web_fetch error is NOT promoted (stays a [pending] candidate).
  4. A stuck-recovery (salient) promotes from ONE episode.
  5. memory_search("uv") finds the lesson; the superseded/history door still works.
  6. A fresh sitting's injected block answers "what did we do last sitting?" — the
     session-history shelf self-restores with the uv payload, zero tool calls.

Everything below drives the REAL EventBus → WriteGate subscription → MemoryStore →
ConsolidationPass → _render_memory_index → MemorySearchTool. The deterministic spine
needs no model by design — that IS the design claim being dogfooded.
"""
import asyncio
import sys
import tempfile

sys.path.insert(0, "src")

from localharness.cli.session_accumulator import SessionAccumulator, derive_session_summary
from localharness.config.models import MemoryConsolidationConfig
from localharness.core.bus import EventBus
from localharness.core.events import Observation, StuckRecovered
from localharness.memory.consolidation import ConsolidationPass, ConsolidationScheduler
from localharness.memory.gate import WriteGate
from localharness.memory.sqlite import MemoryStore
from localharness.tools.builtin.memory_tools import MemorySearchTool

# Phase 35 (check 7) — the LIVE predictive-write path. Added as new import lines (never
# editing the 1-6 imports above) so the additive-only diff stays a pure insertion.
from localharness.config.models import PredictiveGateConfig
from localharness.core.events import Action, UserMessage
from localharness.memory.predictive_gate import PredictiveGate
from localharness.memory.predictive_write_gate import PredictiveWriteGate
from localharness.memory.sqlite import FactQuery

AGENT = "dogfood-agent"


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


async def tool_event(bus, session, tool, *, error=None, output="ok"):
    await bus.publish(Observation(
        agent_id=AGENT, session_id=session, observation_type="tool_result",
        tool_call_id="t1", tool_name=tool,
        output=output if error is None else "",
        error=error,
    ))


async def main() -> None:
    mem_dir = tempfile.mkdtemp(prefix="dogfood-mem-")
    store = MemoryStore(agent_id=AGENT, division_id="", org_id="", base_dir=mem_dir)
    await store.open()
    bus = EventBus()
    gate = WriteGate(store, bus, AGENT)
    await gate.open()  # the REAL subscription path, agent-id filtered
    acc = SessionAccumulator(bus, AGENT)
    await acc.open()  # accumulates the SAME events sections A/B publish (no synthetic extras)

    gate_events = []
    from localharness.core.events import MemoryGateFired

    async def _collect(e):
        gate_events.append(e)

    bus.subscribe(MemoryGateFired, _collect)

    section("SESSION A — agent fumbles uv, recovers; a one-off web timeout")
    await tool_event(bus, "session-A", "bash_exec", error="uv: command not found")
    await tool_event(bus, "session-A", "bash_exec", output="ok via .venv/bin/python")
    await tool_event(bus, "session-A", "web_fetch", error="timeout after 30s (one-off)")
    await tool_event(bus, "session-A", "web_fetch", output="fetched fine on retry")
    print(f"gate fired: {[(e.tier, e.tool_name) for e in gate_events]}")

    section("SESSION B — the SAME uv mistake recurs; agent also gets stuck+recovers")
    await tool_event(bus, "session-B", "bash_exec", error="uv: command not found")
    await tool_event(bus, "session-B", "bash_exec", output="ok via .venv/bin/python")
    await bus.publish(StuckRecovered(agent_id=AGENT, session_id="session-B",
                                     iteration=7, stuck_signature="read:{'path':'missing.md'}"))
    print(f"gate fired (cumulative): {[(e.tier, e.tool_name) for e in gate_events]}")

    section("INJECTED MEMORY BLOCK — BEFORE consolidation (prediction: no candidates)")
    index_before = await store._render_memory_index(5)
    print(index_before)

    section("IDLE ARRIVES — scheduler check + REAL consolidation pass")
    sched = ConsolidationScheduler(store, bus, AGENT, MemoryConsolidationConfig())
    print(f"scheduler.should_run() = {await sched.should_run()}  (stale watermark + pending work)")
    report = await ConsolidationPass(store, MemoryConsolidationConfig()).run()
    print(f"report: promoted={report.promoted} folded={report.folded} "
          f"decayed={report.decayed} demoted={report.demoted} churn={report.churn_rate:.2f}")
    print(f"promoted keys: {report.promoted_keys}")

    section("INJECTED MEMORY BLOCK — AFTER (prediction: uv lesson + stuck lesson, no one-off)")
    index = await store._render_memory_index(5)
    print(index)

    section("memory_search('uv command') — the tool path")
    result = await MemorySearchTool(store)._execute(query="uv command")
    print(result.output)

    section("SITTING CLOSES — end_session flushes a payload-first history line")
    # The dogfood's sittings are simulated ids (not process lifetimes), so mint the row here
    # for end_session's UPDATE to land. The summary is DERIVED from the real accumulator —
    # the string is NOT hand-written (driving the derive path is the point of this check).
    await store.create_session("session-B", budget={}, model="dogfood",
                               context_tokens_available=8192)
    summary = derive_session_summary(acc)
    print(f"derived summary: {summary!r}")
    await acc.close()
    await store.end_session("session-B", exit_reason="complete", summary=summary,
                            turn_count=2, action_count=acc.action_count,
                            tokens_in=0, tokens_out=0)

    section("FRESH SITTING — a NEW store instance renders the injected block")
    store2 = MemoryStore(agent_id=AGENT, division_id="", org_id="", base_dir=mem_dir)
    await store2.open()  # new process lifetime over the same base_dir = the next sitting
    fresh_index = (await store2.load_context(index_mode=True)).agent_memory_md
    print(fresh_index)
    await store2.close()

    section("CHECK 7 — the LIVE predictive write gate composes on the real bus (Phase 35)")
    # store2 is CLOSED above (MemoryStore.close() sets _db=None), so open a FRESH third store and
    # drive PredictiveGate + PredictiveWriteGate on the SAME real bus. ISOLATED mem_dir3 (not the
    # shared mem_dir): the earlier memory_search('uv command') left those facts staged
    # (access_count_staged>0) in the shared DB; an isolated store keeps 7b a clean test of the
    # no-suspect quarantine branch (and post-BLOCKER-1 a bare `nah` negation is quarantine-only
    # regardless of staged suspects). write_live is set True
    # EXPLICITLY: this proves the live path's CAPABILITY end-to-end independent of the shipped
    # default (the PGATE-04 KILL lever) — a KILL flipping the default to False must NOT regress
    # this proof. Observations are published inline (not via the 1-6 tool_event helper, left
    # byte-identical) because PredictiveGate needs a matched Action->Observation tool_call_id per
    # pair, which tool_event does not thread.
    mem_dir3 = tempfile.mkdtemp(prefix="dogfood-pg-")
    store3 = MemoryStore(agent_id=AGENT, division_id="", org_id="", base_dir=mem_dir3)
    await store3.open()
    pg_cfg = PredictiveGateConfig()
    pg_cfg.write_live = True
    pgate = PredictiveGate(store3, bus, AGENT, pg_cfg)
    await pgate.open()
    pwrite = PredictiveWriteGate(store3, bus, AGENT, pg_cfg)
    await pwrite.open()
    TOOL = "pg_probe"
    for i in range(6):  # build a reliable prior (>= min_prior_n) via real Action+Observation pairs
        await bus.publish(Action(agent_id=AGENT, session_id="session-C", action_type="tool_call",
                                 tool_call_id=f"c{i}", tool_name=TOOL))
        await bus.publish(Observation(agent_id=AGENT, session_id="session-C",
                                      observation_type="tool_result", tool_call_id=f"c{i}",
                                      tool_name=TOOL, output="ok", error=None))
    await bus.publish(Action(agent_id=AGENT, session_id="session-C", action_type="tool_call",
                             tool_call_id="c-fail", tool_name=TOOL))
    await bus.publish(Observation(agent_id=AGENT, session_id="session-C",  # -> surprising_failure
                                  observation_type="tool_result", tool_call_id="c-fail",
                                  tool_name=TOOL, output="",
                                  error="unexpected 500 from a normally-reliable tool"))
    fireworks = UserMessage(agent_id=AGENT, session_id="session-C",
                            content="nah id rather watch the fireworks from the park with friends tomorrow",
                            channel="terminal")
    await bus.publish(fireworks)
    stat_facts = await store3.query_facts(FactQuery(tags=["tier:surprising_failure"], min_confidence=0.0, limit=50))
    quar_facts = await store3.query_facts(FactQuery(tags=["tier:correction_pending"], min_confidence=0.0, limit=50))
    pg_index = (await store3.load_context(index_mode=True)).agent_memory_md
    await pgate.close()
    await pwrite.close()
    await store3.close()
    print(f"stat_facts={len(stat_facts)} quar_facts={len(quar_facts)}")

    section("VERDICT vs prediction")
    checks = {
        "1. candidates invisible pre-consolidation": "gate/" not in index_before,
        "2. recurring uv lesson IN injected block (payload-first)":
            "uv: command not found" in index and "[recurring: 2 episodes" in index
            and index.find("uv: command not found") < index.find("[recurring: 2 episodes"),
        "3. one-off web_fetch NOT promoted": not any("/web_fetch/" in k for k in report.promoted_keys),
        "4. salient stuck-recovery promoted from 1 episode": any("stuck_recovered" in k for k in report.promoted_keys),
        "5. search finds the lesson": result.success and "uv" in result.output.lower(),
        # SESS-04 end-to-end proxy + SESS-05 KILL tooth: `summary is not None` is a hard
        # AND-term — a vacuous derivation fails RED here (no silent pass on a suppressed
        # shelf); "uv" IN the history block is the discriminating-content bar a generic
        # "worked on stuff" line could never clear.
        "6. fresh sitting answers 'what did we do last sitting' zero-tool":
            summary is not None
            and "Recent Session History" in fresh_index
            and "uv" in fresh_index.split("Recent Session History")[1],
        # Phase 35 — the LIVE predictive write path composed on the real bus (check 7).
        "7a. live surprising_failure -> sub-0.7 stat write, not injected":
            any(f.confidence < 0.7 and f.importance > 0 for f in stat_facts)
            and "predgate/surprising_failure" not in pg_index,
        "7b. fireworks correction -> standalone quarantine, not injected":
            any(f.key.startswith("correction/quarantine/") and f.provenance == fireworks.id for f in quar_facts)
            and "correction/quarantine" not in pg_index,
    }
    ok = True
    for name, passed in checks.items():
        print(f"  {'✅' if passed else '❌'} {name}")
        ok &= passed
    await gate.close()
    await store.close()
    print(f"\n{'ALL PREDICTIONS HELD' if ok else 'PREDICTION FAILED — investigate'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
