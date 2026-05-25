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


def test_build_messages_returns_copy_not_original():
    cm = ContextManager()
    original = [{"role": "user", "content": "hello"}]
    result = cm.build_messages(original)
    assert result is not original
    assert result == original


def test_build_messages_calls_repair_internally():
    """build_messages should strip orphaned tool results via repair_tool_pairing."""
    cm = ContextManager()
    messages = [
        {"role": "user", "content": "go"},
        _make_tool_result("orphan"),
    ]
    result = cm.build_messages(messages)
    assert all(m.get("role") != "tool" for m in result)
