"""Actual receipts, revision changes, and the natural completion seam."""
from dataclasses import replace

import pytest

from localharness.agent.context import COMPACT_DISABLED, ContextManager
from localharness.agent.loop import AgentLoop, Session
from localharness.agent.permissions import PermissionEvaluator
from localharness.agent.task_context import Requirement, TaskContext
from localharness.config.models import AgentConfig
from localharness.core.events import TaskComplete
from localharness.tools.base import ToolResult


def contract(tmp_path):
    draft = tmp_path / "draft.txt"
    draft.write_text("Original draft")
    req = Requirement("lint", "Requested lint gate", "check", {}, "exit_code", 0, ("draft",))
    return TaskContext("Finish the draft", {"lint": req}, {"draft": draft})


def record(ctx, *, success=True, metadata=None):
    ctx.record_result("check", {}, "call-1", success=success,
                      metadata=metadata or {}, before=ctx.revisions())


@pytest.mark.parametrize("success,metadata,outcome", [
    (True, {"exit_code": 0}, "passed"),
    (True, {"exit_code": 1}, "failed"),
    (True, {"exit_code": False}, "failed"),
    (True, {}, "unknown"),
    (False, {"exit_code": 0}, "failed"),
])
def test_actual_structured_gate_result_not_prose(tmp_path, success, metadata, outcome):
    ctx = contract(tmp_path)
    record(ctx, success=success, metadata=metadata)
    assert ctx.outcomes() == {"lint": outcome}
    answer = ctx.finalize("All gates green. CONFIRMED")
    assert ("All gates green" in answer) == (outcome == "passed")


def test_only_changed_dependency_invalidates_receipt(tmp_path):
    ctx = contract(tmp_path)
    unrelated = tmp_path / "other.txt"
    unrelated.write_text("Unrelated")
    ctx.artifacts["other"] = unrelated
    record(ctx, metadata={"exit_code": 0})
    ctx.observe_human("Use a shorter final reply.")
    unrelated.write_text("Changed")
    assert ctx.outcomes()["lint"] == "passed"
    ctx.artifacts["draft"].write_text("Original draft")
    assert ctx.outcomes()["lint"] == "passed"
    ctx.artifacts["draft"].write_text("Changed draft")
    assert ctx.outcomes()["lint"] == "stale"
    assert "All gates green" not in ctx.finalize("All gates green")
    assert "All gates green" not in ctx.finalize("All gates green")


def test_changed_requirement_and_explicit_waiver_are_not_passes(tmp_path):
    ctx = contract(tmp_path)
    record(ctx, metadata={"exit_code": 0})
    ctx.revise(replace(ctx.requirements["lint"], expected=2, revision="2"))
    assert ctx.outcomes()["lint"] == "stale"
    ctx.waive("lint", human_decision="Skip this gate for the outline checkpoint.")
    assert ctx.outcomes()["lint"] == "waived"
    assert "All gates green" not in ctx.finalize("All gates green")
    assert "human waiver" in ctx.finalize("Done")
    ctx.revise(replace(ctx.requirements["lint"], expected=3, revision="3"))
    assert ctx.outcomes()["lint"] == "stale"


def test_dependency_changed_during_check_is_unverified(tmp_path):
    ctx = contract(tmp_path)
    before = ctx.revisions()
    ctx.artifacts["draft"].write_text("New draft")
    ctx.record_result("check", {}, "call", success=True, metadata={"exit_code": 0}, before=before)
    assert ctx.outcomes()["lint"] in {"unknown", "stale"}


def test_checkpoint_retains_unfinished_asks_then_correction(tmp_path):
    ctx = contract(tmp_path)
    ctx.requested_status = "checkpoint"
    assert ctx.finalize("Outline checkpoint reached; draft remains.") == "Outline checkpoint reached; draft remains."
    assert ctx.status == "checkpoint"
    ctx.observe_human("Continue, and use the stricter gate.")
    ctx.requested_status = "complete"
    ctx.revise(replace(ctx.requirements["lint"], revision="2"))
    assert "unverified" in ctx.finalize("Done")
    assert "stricter gate" in ctx.packet()


def make_loop(llm, bus, tmp_path, task=None, registry=None, window=32768):
    cfg = AgentConfig.model_validate({
        "name": "evidence", "role": "Complete the requested task.",
        "permissions": {"mode": "unattended", "deny_patterns": []},
    })
    return AgentLoop(cfg, llm, bus, ContextManager(max_context_tokens=window), registry,
                     PermissionEvaluator(), compact_md_path=COMPACT_DISABLED,
                     config_dir=tmp_path, task_context=task)


@pytest.mark.asyncio
@pytest.mark.parametrize("success,metadata,expected", [
    (True, {"exit_code": 0}, "passed"),
    (True, {"exit_code": 9}, "failed"),
    (False, {}, "failed"),
    (True, {}, "unknown"),
])
async def test_loop_records_dispatch_and_preserves_public_event(
    tmp_path, bus, mock_llm_client, success, metadata, expected,
):
    ctx = contract(tmp_path)

    class Registry:
        def get_tools_for_agent(self, *args):
            return {}

        async def dispatch(self, *args):
            return ToolResult(output="CONFIRMED; all gates green", success=success,
                              error="requested critic failed", metadata=metadata)

    llm = mock_llm_client([
        mock_llm_client.Response(content=None, tool_calls=[
            mock_llm_client.ToolCall(id="c1", name="check", arguments={}),
        ]), mock_llm_client.Response(content="All gates green."),
    ])
    loop = make_loop(llm, bus, tmp_path, ctx, Registry())
    answer = await loop.run_turn("Run the gate, then report the draft status.")
    assert ctx.outcomes()["lint"] == expected
    assert ("All gates green" in answer) == (expected == "passed")
    events = bus.history(event_types=[TaskComplete])
    assert events[-1].success is True  # Turn execution semantics stay compatible.
    assert events[-1].summary == answer
    if expected != "passed":
        assert not any(m.get("content") == "All gates green." for m in loop._conversation)


@pytest.mark.asyncio
async def test_missing_child_and_sentinel_cannot_complete(tmp_path, bus, mock_llm_client):
    ctx = contract(tmp_path)
    ctx.revise(Requirement("critic", "Requested critic", "agent", {"task": "review"}))
    llm = mock_llm_client([mock_llm_client.Response(content="CONFIRMED")])
    loop = make_loop(llm, bus, tmp_path, ctx)
    answer = await loop.run_turn("Finish with the critic result.")
    assert "Requested critic: unknown" in answer
    assert ctx.status == "unknown"


@pytest.mark.asyncio
async def test_task_packet_survives_lossy_packing_and_human_steering(tmp_path, bus, mock_llm_client):
    ctx = contract(tmp_path)
    loop = make_loop(mock_llm_client([]), bus, tmp_path, ctx)
    ctx.observe_human("Answer only; stop after the outline.")
    original_build = loop._ctx.build_messages

    async def lossy(messages, tools):
        return await original_build([{"role": "user", "content": "old request"}], tools)

    loop._ctx.build_messages = lossy
    packed, _ = await loop._build_request([], None)
    assert packed[-1]["_lh"] == {"origin": "harness", "subtype": "active_task"}
    assert "Answer only; stop after the outline." in packed[-1]["content"]
    assert "lint: unknown" in packed[-1]["content"]


@pytest.mark.asyncio
async def test_oversized_task_packet_blocks_without_model_call(tmp_path, bus, mock_llm_client):
    ctx = contract(tmp_path)
    ctx.objective = "required detail " * 10000
    llm = mock_llm_client([mock_llm_client.Response(content="Never called")])
    loop = make_loop(llm, bus, tmp_path, ctx, window=2048)
    session = Session("evidence", "test", [])
    answer = await loop._execute_loop(session, "Finish", None)
    assert answer.startswith("Active step blocked:")
    assert session.input_tokens == 0
    assert ctx.status == "blocked"


@pytest.mark.asyncio
async def test_ordinary_query_one_call_and_no_task_packet(tmp_path, bus, mock_llm_client):
    llm = mock_llm_client([mock_llm_client.Response(content="42")])
    loop = make_loop(llm, bus, tmp_path)
    session = Session("evidence", "ordinary", [])
    assert await loop._execute_loop(session, "Answer only: 17 + 25?", None) == "42"
    assert session.iteration == 1
    assert not any(m.get("_lh", {}).get("subtype") == "active_task" for m in session.messages)


@pytest.mark.asyncio
async def test_xml_injection_is_counted_at_final_boundary(tmp_path, bus):
    from localharness.agent.context import ActiveReferenceError, TokenCounter
    from localharness.provider.client import LLMClient, LLMConfig
    from localharness.tools.base import ToolSchema

    class Counter(TokenCounter):
        def count_messages(self, messages, tools=None):
            # Canonical packet fits; XML tool instruction expansion does not.
            return 9000 if any("giant tool instruction" in (m.get("content") or "")
                               for m in messages) else 100

    llm = LLMClient(LLMConfig(base_url="http://127.0.0.1:9/v1", model="test",
                              is_local=False, tool_call_mode="xml"))
    loop = make_loop(llm, bus, tmp_path, contract(tmp_path), window=8192)
    loop._ctx._token_counter = Counter()
    tools = [ToolSchema(name="check", description="giant tool instruction", parameters={})]
    with pytest.raises(ActiveReferenceError, match="XML prompt"):
        await loop._build_request([{"role": "user", "content": "Check"}], tools)
    await llm._client.close()
