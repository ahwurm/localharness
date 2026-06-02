"""Phase 28 (SUBAGENT-05): bench-layer real-vs-say-not-do proof for Explore-subagent delegation.

The bench `agent` seam now delegates to the REAL read-only Explore subagent (Phase 27) instead of
the canned STUB_SUBAGENT_OK stub. This drives the FULL bench pipeline (`execute_one_run` ->
`_build_agent_loop` -> real `AgentLoop` -> the registered `agent` tool -> `dispatch_explore_subagent`
-> child `AgentLoop`) against a converted scenario, using a SHARED MockLLMClient queue so the parent
and child consume scripted responses strictly in order (single `_index`), and asserts:

  (a) real delegation  => ScenarioCompleted.tool_call_count >= 2 (1 parent agent-call + >=1 child
      read), child Actions carry parent_id == run session_id, success is True;
  (b) say-not-do       => the parent emits the right "[explore findings]" text WITHOUT calling
      `agent`, so tool_call_count <= 1 and success is False (the event_counts floor rejects it
      even though the rubric text matches) — "no say-not-do pass" (ROADMAP SC3).

No live model, no placeholder data: the child reads a REAL staged fixture file (its content is the
data; the bench_fixtures_staged session fixture in conftest stages it under /tmp/bench_fixtures/).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from localharness.bench.runner import execute_one_run
from localharness.bench.schema import load_scenario
from localharness.core.events import Action

CORPUS_DIR = Path(__file__).resolve().parents[2] / "bench" / "scenarios"
# data/values.txt lives under the staged exploration_root fixture (conftest bench_fixtures_staged).
FIXTURE_VALUES = "/tmp/bench_fixtures/exploration_root/data/values.txt"


def _agent_creation_scenario():
    """The real, converted 06_agent_creation scenario (rubric '[explore findings]', floor min: 2)."""
    return load_scenario(CORPUS_DIR / "train" / "06_agent_creation.yaml")


@pytest.mark.asyncio
async def test_real_delegation_records_subagent_tool_calls_and_passes(
    mock_llm_client, tmp_path, bench_fixtures_staged
):
    """Real path: parent calls `agent`, child reads a real file => tool_call_count >= 2, success True."""
    scen = _agent_creation_scenario()
    assert Path(FIXTURE_VALUES).is_file(), f"fixture not staged: {FIXTURE_VALUES}"

    Response = mock_llm_client.Response
    ToolCall = mock_llm_client.ToolCall

    # Shared queue consumed strictly in order across parent + child (one MockLLMClient _index):
    #   1) parent  -> `agent` tool call (delegate the explore task)
    #   2) child   -> `read` on the REAL staged values.txt
    #   3) child   -> summary content (no tool calls) => child completes
    #   4) parent  -> final message echoing the subagent's findings header (rubric anchor)
    llm = mock_llm_client([
        Response(content=None, tool_calls=[
            ToolCall(id="p-1", name="agent",
                     arguments={"agent_id": "explorer", "task": f"Read {FIXTURE_VALUES} and report it"}),
        ]),
        Response(content=None, tool_calls=[
            ToolCall(id="c-1", name="read", arguments={"path": FIXTURE_VALUES}),
        ]),
        Response(content="The file holds MAGIC_VALUE_777."),
        Response(content="Subagent returned: [explore findings] task: explore | the value is MAGIC_VALUE_777."),
    ])

    run_path = tmp_path / "real.jsonl"
    completed = await execute_one_run(scen, "mock-model", run_path, llm)

    # Genuine delegation: 1 parent `agent` Action + >=1 child `read` Action on the SAME bus.
    assert completed.tool_call_count >= 2, (
        f"expected >=2 tool calls (parent agent + child read), got {completed.tool_call_count}"
    )
    assert completed.success is True, "real delegation matching rubric + count floor must PASS"

    # The child's tool-call Actions are attributed to the run session via parent_id (Phase 27).
    # Read the persisted JSONL trace directly (deterministic — no async replay needed).
    from localharness.core.events import deserialize_event

    events = [deserialize_event(ln) for ln in run_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    child_reads = [e for e in events if isinstance(e, Action) and e.tool_name == "read"]
    assert child_reads, "expected at least one child `read` Action in the persisted trace"
    run_session_id = f"{scen.name}:{run_path.stem}"
    assert all(e.parent_id == run_session_id for e in child_reads), (
        "child read Actions must carry parent_id == run session_id (Phase 27 _ParentIdBus stamping)"
    )


@pytest.mark.asyncio
async def test_say_not_do_delegation_fails_the_count_floor(
    mock_llm_client, tmp_path, bench_fixtures_staged
):
    """Say-not-do path: parent emits the right text but NEVER calls `agent` => count <=1, success False."""
    scen = _agent_creation_scenario()

    Response = mock_llm_client.Response

    # Parent emits a final message that SATISFIES the rubric ("[explore findings]") but performs
    # NO tool call at all — the canonical say-not-do. The event_counts floor (min: 2) must reject it.
    llm = mock_llm_client([
        Response(content="Subagent returned: [explore findings] task: explore | the value is MAGIC_VALUE_777."),
    ])

    run_path = tmp_path / "saynotdo.jsonl"
    completed = await execute_one_run(scen, "mock-model", run_path, llm)

    assert completed.tool_call_count <= 1, (
        f"say-not-do must not reach the delegation floor, got {completed.tool_call_count}"
    )
    assert completed.success is False, (
        "say-not-do (right text, zero real delegation) must FAIL the tool_call_count floor"
    )
