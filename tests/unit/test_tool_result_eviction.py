"""Feature 2: conversation history as a queryable handle (tool-result eviction).

- A large tool result is evicted to a restorable stub.
- tool_result_get restores the EXACT body.
- ids are DETERMINISTIC (same body -> same id) for prefix-cache stability.
- tool_use/tool_result pairing is preserved after eviction.
"""
import json

import pytest

from localharness.agent.context import (
    ContentStore,
    ContextManager,
    TokenCounter,
    _TOOL_STUB_PREFIX,
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


# ---------------------------------------------------------------------------
# #134: the restore/evict LOOP. Eviction arms at 0.50 usage; the model restores a
# body with tool_result_get; the restored body re-inflates usage, re-arms the pass,
# and the just-restored body is evicted AGAIN — the same handle, forever. The fix is
# a TURN-SCOPED pin: a body restored this turn is not re-evicted by the 0.50 pass.
# ---------------------------------------------------------------------------

_PROSE ="the harness restored the evicted body and the model kept reasoning over it; "


def _big_body(i: int) -> str:
    """A distinct, prose-shaped body well over the 8k eviction threshold (~4.5k tokens)."""
    return f"BODY-{i}\n" + f"{_PROSE}{i} " * 300


def _exchange(call_id: str, body: str, tool: str = "read_file"):
    """One assistant tool-call + its (bulky) tool result."""
    return [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": tool, "arguments": json.dumps({"path": f"/f/{call_id}"})},
        }]},
        {"role": "tool", "tool_call_id": call_id, "content": body},
    ]


def _restore_exchange(handle: str, body: str, call_id: str = "get-1"):
    """The model restoring an evicted body: tool_result_get('<handle>') + the returned body."""
    return [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "tool_result_get", "arguments": json.dumps({"id": handle})},
        }]},
        {"role": "tool", "tool_call_id": call_id, "content": body},
    ]


def _stub_handle(messages) -> str:
    """The restore id carried by the first eviction stub in `messages`."""
    stub = next(m["content"] for m in messages
                if (m.get("content") or "").startswith(_TOOL_STUB_PREFIX))
    return stub.split("tool_result_get('")[1].split("')")[0]


def _n_stubs(messages) -> int:
    return sum(1 for m in messages if (m.get("content") or "").startswith(_TOOL_STUB_PREFIX))


def _evicting_cm(store, msgs, frac: float = 0.55, **kw) -> ContextManager:
    """A ContextManager whose window puts `msgs` at ~`frac` usage — above the 0.50 eviction
    gate, below the 0.80 compaction trigger and the emergency floor."""
    tc = TokenCounter()
    return ContextManager(
        max_context_tokens=int(tc.estimate_messages(msgs) / frac),
        eviction_store=store, content_store=store, token_counter=tc, **kw,
    )


async def _looping_turn(cm, store, msgs):
    """Drive one turn to the point the live loop hit: evict -> restore -> keep working, so the
    restored body has drifted out of the keep-last window and is eligible again."""
    built, _ = await cm.build_messages(msgs)
    assert _n_stubs(built) > 0, "eviction gate never armed — test setup is wrong"
    handle = _stub_handle(built)
    restored = store.get(handle)
    assert restored is not None
    convo = list(built) + _restore_exchange(handle, restored)
    for i in range(5, 8):  # three more bulky results push the restore past keep-last
        convo += _exchange(f"c{i}", _big_body(i))
    return convo, restored


def _base_convo():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "research six files"}]
    for i in range(5):
        msgs += _exchange(f"c{i}", _big_body(i))
    return msgs


@pytest.mark.asyncio
async def test_restored_body_is_not_re_evicted_in_the_same_turn():
    """#134 (a): the loop breaker. A body pulled back with tool_result_get survives the NEXT
    usage-fraction eviction pass in the same turn — while other eligible results still evict."""
    store = ContentStore()
    msgs = _base_convo()
    cm = _evicting_cm(store, msgs)
    convo, restored = await _looping_turn(cm, store, msgs)

    built2, budget = await cm.build_messages(convo)

    restored_msg = next(m for m in built2 if m.get("tool_call_id") == "get-1")
    assert restored_msg["content"] == restored, (
        "restored body was evicted AGAIN in the same turn — the #134 loop"
    )
    # The pass still did its job: other eligible bodies were stubbed this pass.
    assert _n_stubs(built2) > _n_stubs(convo)
    # No emergency-floor distortion in this window (guards the sizing of the fixture).
    assert not any("chars elided" in (m.get("content") or "") for m in built2)
    assert budget.usage_fraction < 0.80


@pytest.mark.asyncio
async def test_pin_expires_at_turn_rollover():
    """#134 (b): pins are TURN-scoped. After the loop's per-turn reset, the same restored body
    is ordinary evictable content again — a pin never becomes permanent context bloat."""
    store = ContentStore()
    msgs = _base_convo()
    cm = _evicting_cm(store, msgs)
    convo, restored = await _looping_turn(cm, store, msgs)
    await cm.build_messages(convo)  # same turn: pinned

    cm.reset_compaction_guard()  # the agent loop's per-turn reset = a new user turn
    built3, _ = await cm.build_messages(convo)

    restored_msg = next(m for m in built3 if m.get("tool_call_id") == "get-1")
    assert restored_msg["content"].startswith(_TOOL_STUB_PREFIX), (
        "pin outlived its turn — restored bodies would accumulate forever"
    )


@pytest.mark.asyncio
async def test_hard_overflow_stages_still_act_on_pinned_content():
    """#134 (c): the pin shields ONLY against the 0.50 usage-fraction pass. A turn that genuinely
    cannot fit still degrades through compaction/the emergency floor — pinned content included."""
    from localharness.agent.context import CompactionPipeline

    async def tiny(middle):
        return "s"

    store = ContentStore()
    tc = TokenCounter()
    restored = _big_body(0)
    pipeline = CompactionPipeline(
        tc, llm_summarize_fn=tiny, preserve_first_n=1, preserve_last_n=1,
    )
    cm = ContextManager(
        max_context_tokens=2_000, pipeline=pipeline, eviction_store=store, content_store=store,
        token_counter=tc,
    )
    # Baseline the turn FIRST, so the restore below is a THIS-TURN restore and really is pinned
    # (a restore already in history at the turn's first build is a prior turn's — not pinned).
    convo = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    await cm.build_messages(list(convo))
    convo += _restore_exchange("deadbeef", restored)
    assert cm._restore_pins(convo) == frozenset({"get-1"})  # pin is ACTIVE for this pass
    for i in range(1, 4):
        convo += _exchange(f"c{i}", _big_body(i))

    built, budget = await cm.build_messages(convo)

    assert any("[Context Summary]" in (m.get("content") or "") for m in built)  # 0.80 stage fired
    assert not any((m.get("content") or "") == restored for m in built), (
        "pinned content escaped the hard-overflow stages"
    )
    assert budget.usage_fraction <= 1.0, "request overflowed the window despite the floor"


@pytest.mark.asyncio
async def test_no_restores_means_unchanged_eviction():
    """#134 (d) regression: with no tool_result_get in the conversation, the pass behaves exactly
    as before — all eligible results beyond the keep-last window are stubbed."""
    from localharness.agent.context import TOOL_EVICT_KEEP_LAST

    store = ContentStore()
    msgs = _base_convo()
    cm = _evicting_cm(store, msgs)
    built, _ = await cm.build_messages(msgs)
    assert _n_stubs(built) == 5 - TOOL_EVICT_KEEP_LAST


def test_pin_costs_exactly_one_eviction_not_the_keep_last_window():
    """#134 ordering: a pinned result is SKIPPED by the pass, it does not push another result
    into the protected keep-last window — non-pinned eligible results are still evicted first."""
    store = ContentStore()
    msgs = []
    for i in range(5):
        msgs += _exchange(f"c{i}", _big_body(i))
    baseline, n_base = _evict_large_tool_results(msgs, store, threshold_chars=8_000, keep_last=3)
    assert n_base == 2  # c0, c1

    pinned, n_pinned = _evict_large_tool_results(
        msgs, store, threshold_chars=8_000, keep_last=3, pinned_call_ids=frozenset({"c0"}),
    )
    assert n_pinned == 1, "pinning the oldest must cost exactly one eviction"
    assert pinned[1]["content"] == _big_body(0)                       # pinned survives
    assert pinned[3]["content"].startswith(_TOOL_STUB_PREFIX)         # c1 still evicted
    assert pinned[5]["content"] == _big_body(2)                       # c2 stays in keep-last
