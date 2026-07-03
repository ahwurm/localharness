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

Everything below drives the REAL EventBus → WriteGate subscription → MemoryStore →
ConsolidationPass → _render_memory_index → MemorySearchTool. The deterministic spine
needs no model by design — that IS the design claim being dogfooded.
"""
import asyncio
import sys
import tempfile

sys.path.insert(0, "src")

from localharness.config.models import MemoryConsolidationConfig
from localharness.core.bus import EventBus
from localharness.core.events import Observation, StuckRecovered
from localharness.memory.consolidation import ConsolidationPass, ConsolidationScheduler
from localharness.memory.gate import WriteGate
from localharness.memory.sqlite import MemoryStore
from localharness.tools.builtin.memory_tools import MemorySearchTool

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
    store = MemoryStore(agent_id=AGENT, division_id="", org_id="",
                        base_dir=tempfile.mkdtemp(prefix="dogfood-mem-"))
    await store.open()
    bus = EventBus()
    gate = WriteGate(store, bus, AGENT)
    await gate.open()  # the REAL subscription path, agent-id filtered

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

    section("VERDICT vs prediction")
    checks = {
        "1. candidates invisible pre-consolidation": "gate/" not in index_before,
        "2. recurring uv lesson IN injected block (payload-first)":
            "uv: command not found" in index and "[recurring: 2 episodes" in index
            and index.find("uv: command not found") < index.find("[recurring: 2 episodes"),
        "3. one-off web_fetch NOT promoted": not any("/web_fetch/" in k for k in report.promoted_keys),
        "4. salient stuck-recovery promoted from 1 episode": any("stuck_recovered" in k for k in report.promoted_keys),
        "5. search finds the lesson": result.success and "uv" in result.output.lower(),
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
