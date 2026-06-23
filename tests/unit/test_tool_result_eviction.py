"""Feature 2: conversation history as a queryable handle (tool-result eviction).

- A large tool result is evicted to a restorable stub.
- tool_result_get restores the EXACT body.
- ids are DETERMINISTIC (same body -> same id) for prefix-cache stability.
- tool_use/tool_result pairing is preserved after eviction.
"""
import pytest

from localharness.agent.context import (
    ContentStore,
    _content_handle,
    _evict_large_tool_results,
)
from localharness.tools.builtin.tool_result_get_tool import ToolResultGetTool


def _msgs(big_body: str, n_results: int = 4):
    """Build a conversation: one assistant tool_call + one tool result per call."""
    out = []
    for i in range(n_results):
        cid = f"call_{i}"
        out.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": cid, "type": "function",
                            "function": {"name": "bash_exec", "arguments": "{}"}}],
        })
        out.append({"role": "tool", "tool_call_id": cid, "content": big_body})
    return out


def test_large_result_evicted_to_stub():
    store = ContentStore()
    big = "X" * 20_000
    msgs = _msgs(big, n_results=4)
    out, n = _evict_large_tool_results(msgs, store, threshold_chars=8_000, keep_last=1)
    # 4 bulky results, keep_last=1 => 3 evicted.
    assert n == 3
    stubs = [m for m in out if m["role"] == "tool" and m["content"].startswith("[tool result evicted")]
    assert len(stubs) == 3
    # The kept (newest) one is still the full body.
    assert out[-1]["content"] == big
    # Stub names the restore call.
    assert "tool_result_get(" in stubs[0]["content"]


@pytest.mark.asyncio
async def test_tool_result_get_restores_exact_body():
    store = ContentStore()
    big = "lorem ipsum dolor\n" * 1_000
    msgs = _msgs(big, n_results=2)
    out, n = _evict_large_tool_results(msgs, store, threshold_chars=8_000, keep_last=0)
    assert n == 2
    # Extract an id from a stub and restore via the tool.
    stub = next(m["content"] for m in out
                if m["role"] == "tool" and (m["content"] or "").startswith("[tool result evicted"))
    rid = stub.split("tool_result_get('")[1].split("')")[0]
    tool = ToolResultGetTool(store)
    res = await tool.run(id=rid)
    assert res.success
    assert res.output == big


@pytest.mark.asyncio
async def test_tool_result_get_unknown_id():
    tool = ToolResultGetTool(ContentStore())
    res = await tool.run(id="deadbeef")
    assert not res.success
    assert res.error_type == "not_found"


def test_ids_are_deterministic():
    # Same body -> same id, independent of store instance/time (no randomness).
    body = "deterministic body content"
    assert _content_handle(body) == _content_handle(body)
    s1, s2 = ContentStore(), ContentStore()
    assert s1.put(body) == s2.put(body)
    # Different body -> different id.
    assert _content_handle(body) != _content_handle(body + "!")


def test_eviction_preserves_tool_pairing():
    """Every evicted tool message keeps its tool_call_id, and every tool message still has a
    matching preceding assistant tool_call — no orphaned pairs introduced by eviction."""
    store = ContentStore()
    big = "Y" * 12_000
    msgs = _msgs(big, n_results=3)
    out, n = _evict_large_tool_results(msgs, store, threshold_chars=8_000, keep_last=0)
    assert n == 3
    valid_ids = {
        tc["id"]
        for m in out if m["role"] == "assistant"
        for tc in (m.get("tool_calls") or [])
    }
    tool_msgs = [m for m in out if m["role"] == "tool"]
    assert len(tool_msgs) == 3
    for m in tool_msgs:
        assert "tool_call_id" in m
        assert m["tool_call_id"] in valid_ids  # never orphaned


def test_web_results_skipped():
    """Web tool results are handled by the web-eviction path; the generic path skips them."""
    store = ContentStore()
    big = "Z" * 20_000
    msgs = [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "w1", "type": "function",
                         "function": {"name": "web_fetch", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "w1", "content": big},
    ]
    out, n = _evict_large_tool_results(msgs, store, threshold_chars=8_000, keep_last=0)
    assert n == 0
    assert out[-1]["content"] == big
