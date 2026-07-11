"""De-false-green the two gaps flagged in the audit:
1. b1b4974's (message, usage) tuple-unpack in compaction summarize had NO regression test and a
   revert stayed green because SummaryCompactionStage.apply swallows the exception. Test the shared
   summarize fn DIRECTLY (outside the swallow) so a regression fails loudly.
2. Over-window eviction + summary-compaction were only ever asserted at CONSTRUCTION (pipeline is
   not None) or via a StubPipeline. Drive the REAL build_messages over-window and assert both
   actually FIRE mid-run."""
from __future__ import annotations

import pytest

from localharness.agent.context import (
    CompactionPipeline,
    ContentStore,
    ContextManager,
    TokenCounter,
    _content_handle,
    make_compaction_summarize_fn,
)


class _Msg:
    content = "DENSE SUMMARY"


@pytest.mark.asyncio
async def test_compaction_summarize_unpacks_tuple_and_returns_content():
    """complete() returns (message, usage). The shared summarize fn must unpack it. A regress to a
    bare `result.content` on the tuple raises/returns '' HERE (direct call, not behind the stage's
    try/except that originally hid this for months)."""
    class _TupleLLM:
        async def complete(self, prompt, tools=None, disable_thinking=False):
            return (_Msg(), {"prompt_tokens": 1})  # production shape

    out = await make_compaction_summarize_fn(_TupleLLM())([{"role": "user", "content": "hi"}])
    assert out == "DENSE SUMMARY"

    class _BareLLM:  # robustness: a bare message (non-tuple) still works
        async def complete(self, prompt, tools=None, disable_thinking=False):
            return _Msg()

    assert await make_compaction_summarize_fn(_BareLLM())([{"role": "user", "content": "hi"}]) == "DENSE SUMMARY"


@pytest.mark.asyncio
async def test_over_window_eviction_and_compaction_fire_in_build_messages(bus):
    """Over-window mid-run: build_messages must (a) evict a bulky tool result to the store
    LOSSLESSLY and (b) fire summary-compaction — not merely be 'wired'. Drives the real pipeline."""
    class _SummLLM:
        async def complete(self, prompt, tools=None, disable_thinking=False):
            return (_Msg(), None)

    tc = TokenCounter()  # tiktoken (no endpoint) — fine for the test
    store = ContentStore()
    pipeline = CompactionPipeline(
        token_counter=tc, llm_summarize_fn=make_compaction_summarize_fn(_SummLLM()),
        preserve_first_n=1, preserve_last_n=1,
    )
    cm = ContextManager(
        max_context_tokens=2_000, pipeline=pipeline, eviction_store=store, content_store=store,
        token_counter=tc, bus=bus, agent_id="a", session_id="s",
    )

    big = "X" * 40_000  # ~10k tokens each — way over the 2k window
    messages = [{"role": "user", "content": "start"}]
    for i in range(1, 5):  # 4 big tool results (>KEEP_LAST=3 so the oldest evicts)
        messages.append({"role": "assistant", "content": None,
                         "tool_calls": [{"id": f"c{i}", "function": {"name": "read", "arguments": "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": big})
    messages.append({"role": "user", "content": "end"})

    out, _budget = await cm.build_messages(messages)

    # (a) eviction fired: the bulky body is retained LOSSLESSLY in the store (restorable), even though
    #     the stub itself may then be folded into the summary.
    assert store.get(_content_handle(big)) == big, "evicted body must be retained losslessly mid-run"
    # (b) summary-compaction fired: a [Context Summary] message is now present.
    assert any("[Context Summary]" in (m.get("content") or "") for m in out), \
        "summary compaction must FIRE mid-run, not just be wired"
