"""Tests for ContextManager.repair_tool_pairing and build_messages."""
import pytest
from localharness.agent.context import ContextManager


def _make_assistant_with_tool_call(tool_call_id: str, tool_name: str = "bash") -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": tool_call_id, "function": {"name": tool_name, "arguments": "{}"}}],
    }


def _make_tool_result(tool_call_id: str, content: str = "ok") -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


# ---------------------------------------------------------------------------
# repair_tool_pairing
# ---------------------------------------------------------------------------

def test_repair_removes_orphaned_tool_result():
    """Tool result with no matching assistant tool_calls entry is removed."""
    cm = ContextManager()
    messages = [
        {"role": "user", "content": "do something"},
        _make_tool_result("orphan-id"),  # no preceding assistant with tool_calls
    ]
    repaired = cm.repair_tool_pairing(messages)
    assert all(m.get("role") != "tool" for m in repaired)


def test_repair_keeps_valid_pairs():
    cm = ContextManager()
    messages = [
        {"role": "user", "content": "go"},
        _make_assistant_with_tool_call("tc-1"),
        _make_tool_result("tc-1"),
    ]
    repaired = cm.repair_tool_pairing(messages)
    tool_msgs = [m for m in repaired if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "tc-1"


def test_repair_handles_empty_message_list():
    cm = ContextManager()
    assert cm.repair_tool_pairing([]) == []


@pytest.mark.asyncio
async def test_build_messages_returns_copy_not_original():
    cm = ContextManager()
    original = [{"role": "user", "content": "hello"}]
    result, budget = await cm.build_messages(original)
    assert result is not original
    assert result == original


@pytest.mark.asyncio
async def test_build_messages_calls_repair_internally():
    """build_messages should strip orphaned tool results via repair_tool_pairing."""
    cm = ContextManager()
    messages = [
        {"role": "user", "content": "go"},
        _make_tool_result("orphan"),
    ]
    result, budget = await cm.build_messages(messages)
    assert all(m.get("role") != "tool" for m in result)


# --- Phase 4: TokenCounter + TokenBudget ---

def test_token_counter_tiktoken():
    """TokenCounter produces non-zero count for non-empty text."""
    from localharness.agent.context import TokenCounter
    tc = TokenCounter()
    count = tc.count("Hello, world! This is a test sentence.")
    assert count > 0
    assert count < 100  # sanity


def test_token_counter_messages():
    from localharness.agent.context import TokenCounter
    tc = TokenCounter()
    msgs = [
        {"role": "system", "content": "You are a helper."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    count = tc.count_messages(msgs)
    assert count > 10  # at least the text content
    assert count < 200  # sanity


def test_token_budget_usage_fraction():
    from localharness.agent.context import TokenBudget
    budget = TokenBudget(total_limit=100_000, current_usage=70_000, tool_schema_tokens=10_000)
    assert budget.usage_fraction == pytest.approx(0.80, abs=0.01)
    assert budget.needs_summary_compact is True
    assert budget.needs_full_compact is False


def test_token_budget_below_threshold():
    from localharness.agent.context import TokenBudget
    budget = TokenBudget(total_limit=100_000, current_usage=50_000, tool_schema_tokens=5_000)
    assert budget.usage_fraction == pytest.approx(0.55, abs=0.01)
    assert budget.needs_summary_compact is False


# --- Phase 4: CompactionPipeline ---

def test_tool_result_cap_truncates():
    from localharness.agent.context import ToolResultCapStage, TokenBudget, TokenCounter
    stage = ToolResultCapStage(max_chars=100)
    messages = [{"role": "tool", "tool_call_id": "tc-1", "content": "x" * 200}]
    budget = TokenBudget(total_limit=128000, current_usage=1000, tool_schema_tokens=0)
    result, modified = stage.apply(messages, budget, TokenCounter())
    assert modified is True
    assert len(result[0]["content"]) <= 150  # 100 + truncation suffix


def test_tool_result_cap_no_op_when_short():
    from localharness.agent.context import ToolResultCapStage, TokenBudget, TokenCounter
    stage = ToolResultCapStage(max_chars=100)
    messages = [{"role": "tool", "tool_call_id": "tc-1", "content": "short"}]
    budget = TokenBudget(total_limit=128000, current_usage=1000, tool_schema_tokens=0)
    result, modified = stage.apply(messages, budget, TokenCounter())
    assert modified is False


@pytest.mark.asyncio
async def test_summary_compaction_fires_at_80_pct():
    from localharness.agent.context import SummaryCompactionStage, TokenBudget, TokenCounter
    async def mock_summarize(msgs):
        return "Summary of middle messages"
    stage = SummaryCompactionStage(preserve_first_n=2, preserve_last_n=2, llm_summarize_fn=mock_summarize)
    # Build messages: 2 preserved first + 6 middle + 2 preserved last = 10
    messages = [{"role": "system", "content": "sys"}]
    messages.append({"role": "user", "content": "task"})
    for i in range(6):
        messages.append({"role": "assistant", "content": f"response {i}"})
    messages.append({"role": "user", "content": "recent"})
    messages.append({"role": "assistant", "content": "latest"})
    budget = TokenBudget(total_limit=100_000, current_usage=82_000, tool_schema_tokens=0)
    result, modified = await stage.apply(messages, budget, TokenCounter())
    assert modified is True
    assert len(result) < len(messages)
    # Summary message should be present
    assert any("[Context Summary]" in (m.get("content") or "") for m in result)


@pytest.mark.asyncio
async def test_summary_compaction_skips_below_80_pct():
    from localharness.agent.context import SummaryCompactionStage, TokenBudget, TokenCounter
    async def mock_summarize(msgs):
        return "Should not be called"
    stage = SummaryCompactionStage(preserve_first_n=2, preserve_last_n=2, llm_summarize_fn=mock_summarize)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    budget = TokenBudget(total_limit=100_000, current_usage=50_000, tool_schema_tokens=0)
    result, modified = await stage.apply(messages, budget, TokenCounter())
    assert modified is False
    assert result == messages


@pytest.mark.asyncio
async def test_compaction_pipeline_preserves_tool_pairs():
    from localharness.agent.context import CompactionPipeline, TokenBudget, TokenCounter
    async def mock_summarize(msgs):
        return "Summarized"
    tc = TokenCounter()
    pipeline = CompactionPipeline(token_counter=tc, preserve_first_n=2, preserve_last_n=2, llm_summarize_fn=mock_summarize)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "tc-1", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "tc-1", "content": "result"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "recent"},
        {"role": "assistant", "content": "latest"},
    ]
    budget = TokenBudget(total_limit=100_000, current_usage=82_000, tool_schema_tokens=0)
    result, modified = await pipeline.run(messages, budget)
    # No orphaned tool messages in result
    tool_msgs = [m for m in result if m.get("role") == "tool"]
    for tm in tool_msgs:
        tc_id = tm.get("tool_call_id")
        assert any(
            tc_id in [tc.get("id") for tc in (m.get("tool_calls") or [])]
            for m in result if m.get("role") == "assistant"
        )


# --- Phase 4: compact.md load path ---

def test_load_compact_md_returns_message_when_file_exists(tmp_path):
    """load_compact_md returns a system message when compact.md exists with content."""
    from localharness.agent.context import load_compact_md
    compact_file = tmp_path / "compact.md"
    compact_file.write_text("Previous session summary: user was building a research agent.")
    msg = load_compact_md(compact_file)
    assert msg is not None
    assert msg["role"] == "system"
    assert "[Prior Session Context]" in msg["content"]
    assert "research agent" in msg["content"]


def test_load_compact_md_returns_none_when_missing(tmp_path):
    """load_compact_md returns None when compact.md does not exist."""
    from localharness.agent.context import load_compact_md
    compact_file = tmp_path / "compact.md"
    msg = load_compact_md(compact_file)
    assert msg is None


def test_load_compact_md_returns_none_when_empty(tmp_path):
    """load_compact_md returns None when compact.md is empty."""
    from localharness.agent.context import load_compact_md
    compact_file = tmp_path / "compact.md"
    compact_file.write_text("")
    msg = load_compact_md(compact_file)
    assert msg is None


# ---------------------------------------------------------------------------
# SCEN-04 plumbing: CompactionTriggered publication (Plan 12-01 Task 2)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compaction_publishes_event():
    """When pipeline modifies messages, CompactionTriggered is published once."""
    from localharness.agent.context import ContextManager
    from localharness.core.events import CompactionTriggered

    published: list = []

    class FakeBus:
        async def publish(self, event):
            published.append(event)

    # Stub pipeline that always reports modified=True
    class StubPipeline:
        async def run(self, messages, budget):
            return messages[:1], True   # truncate to first message; any_modified=True

    # Use a tiny budget so needs_summary_compact triggers
    cm = ContextManager(
        max_context_tokens=100,
        pipeline=StubPipeline(),
        bus=FakeBus(),
        agent_id="agent-1",
        session_id="session-1",
    )
    cm.set_iteration(7)

    # Build messages large enough to exceed needs_summary_compact threshold
    big = [{"role": "user", "content": "x" * 5000}] * 5
    await cm.build_messages(big, tool_schemas=None)

    assert len(published) == 1
    ev = published[0]
    assert isinstance(ev, CompactionTriggered)
    assert ev.agent_id == "agent-1"
    assert ev.session_id == "session-1"
    assert ev.iteration == 7
    assert ev.pre_usage_fraction >= 0.0
    assert ev.post_usage_fraction >= 0.0


@pytest.mark.asyncio
async def test_compaction_no_publish_when_unchanged():
    """When pipeline reports any_modified=False, no event is published."""
    from localharness.agent.context import ContextManager

    published: list = []

    class FakeBus:
        async def publish(self, event):
            published.append(event)

    class NoOpPipeline:
        async def run(self, messages, budget):
            return messages, False

    cm = ContextManager(
        max_context_tokens=100,
        pipeline=NoOpPipeline(),
        bus=FakeBus(),
        agent_id="a",
        session_id="s",
    )
    big = [{"role": "user", "content": "x" * 5000}] * 5
    await cm.build_messages(big, tool_schemas=None)
    assert published == []


@pytest.mark.asyncio
async def test_compaction_no_bus_no_publish():
    """Back-compat — ContextManager without bus must not raise."""
    from localharness.agent.context import ContextManager

    class StubPipeline:
        async def run(self, messages, budget):
            return messages[:1], True

    cm = ContextManager(max_context_tokens=100, pipeline=StubPipeline())
    big = [{"role": "user", "content": "x" * 5000}] * 5
    out, _budget = await cm.build_messages(big, tool_schemas=None)
    # Should not raise. No bus, no publication.
    assert out == big[:1]


def test_default_deny_patterns_use_bash_exec():
    """Default deny_patterns reference bash_exec (the actual tool name), not legacy bash."""
    from localharness.config.models import PermissionConfig
    patterns = PermissionConfig().deny_patterns
    assert "bash_exec(sudo:*)" in patterns
    assert "bash_exec(rm -rf *)" in patterns
    assert "bash_exec(chmod 777 *)" in patterns
    assert "bash(sudo:*)" not in patterns
    assert "bash(rm -rf *)" not in patterns


# ---------------------------------------------------------------------------
# Stale web-result eviction (_evict_stale_web_results + build_messages gate)
# ---------------------------------------------------------------------------

import json

from localharness.agent.context import (
    WEB_EVICT_KEEP_LAST,
    _evict_stale_web_results,
)


def _web_exchange(i: int, tool: str = "web_fetch", body_chars: int = 3000):
    """One assistant tool-call + tool-result pair for a web tool."""
    hint = {"url": f"https://example.test/p{i}"} if tool == "web_fetch" else {"query": f"q{i}"}
    return [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"wc-{i}", "type": "function",
            "function": {"name": tool, "arguments": json.dumps(hint)},
        }]},
        {"role": "tool", "tool_call_id": f"wc-{i}", "content": "x" * body_chars},
    ]


def test_evict_stubs_all_but_newest_keeping_hint():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    for i in range(4):
        msgs += _web_exchange(i)
    out, evicted = _evict_stale_web_results(msgs, keep_last=2)
    assert evicted == 2
    stubbed = [m for m in out if m.get("role") == "tool" and "omitted" in m["content"]]
    assert len(stubbed) == 2
    # oldest two stubbed, URL hint preserved, newest two intact
    assert "https://example.test/p0" in stubbed[0]["content"]
    assert out[-1]["content"] == "x" * 3000
    # original list untouched (no mutation)
    assert msgs[3]["content"] == "x" * 3000


def test_evict_skips_small_and_non_web_results():
    msgs = [
        *_web_exchange(0, body_chars=100),               # small web result — skip
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "rc-1", "type": "function",
            "function": {"name": "read", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "rc-1", "content": "y" * 5000},  # non-web — skip
        *_web_exchange(1),
        *_web_exchange(2),
        *_web_exchange(3),
    ]
    out, evicted = _evict_stale_web_results(msgs, keep_last=2)
    assert evicted == 1  # only exchange 1 (oldest big web beyond keep-last)
    assert out[1]["content"] == "x" * 100          # small survives
    assert any(m.get("content") == "y" * 5000 for m in out)  # read result survives


def test_evict_idempotent_on_stubs():
    msgs = []
    for i in range(4):
        msgs += _web_exchange(i)
    once, n1 = _evict_stale_web_results(msgs, keep_last=1)
    twice, n2 = _evict_stale_web_results(once, keep_last=1)
    assert n1 == 3 and n2 == 0
    assert once == twice


@pytest.mark.asyncio
async def test_build_messages_evicts_only_over_threshold():
    from localharness.agent.context import ContextManager

    msgs = [{"role": "system", "content": "sys"}]
    for i in range(4):
        msgs += _web_exchange(i, body_chars=4000)
    # tiny window -> usage fraction far over 0.50 -> eviction fires
    cm_small = ContextManager(max_context_tokens=2000)
    built, _ = await cm_small.build_messages(list(msgs), None)
    assert sum("omitted" in (m.get("content") or "") for m in built) == 4 - WEB_EVICT_KEEP_LAST
    # huge window -> under threshold -> untouched
    cm_big = ContextManager(max_context_tokens=1_000_000)
    built2, _ = await cm_big.build_messages(list(msgs), None)
    assert all("omitted" not in (m.get("content") or "") for m in built2)
