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
    result = await cm.build_messages(original)
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
    result = await cm.build_messages(messages)
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
