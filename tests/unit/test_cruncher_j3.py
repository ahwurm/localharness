"""J3 cruncher — reachable end-to-end map-reduce over a GRANTED over-window body + store-aware
delivery. The leaf path runs for REAL (a reactive fake LLM emits an actual tool_result_get call →
real read-through → extract); only the model's *reasoning* is faked. Deterministic, no live model.
The faithful-on-the-real-27B claim is the live dogfood; this pins the plumbing."""
from __future__ import annotations

import re

import pytest

import localharness.agent.subagent as subagent
from localharness.agent.context import ContentStore, ContextManager
from localharness.agent.permissions import PermissionEvaluator
from localharness.tools.builtin import register_builtin_tools
from localharness.tools.builtin.load_document_tool import LoadDocumentTool
from localharness.tools.registry import ToolRegistry

SECRET = "ZephyrFalcon7X"  # buried mid-document; appears in exactly one (non-first) section


def _reactive_cruncher_llm(Response, ToolCall):
    """A fake LLM that DRIVES the real leaf + combine loops:
    - leaf turn 1 (tool_result_get available, no tool result yet): emit a real tool_result_get call
      for the hex handle named in the task (read-through happens for real);
    - leaf turn 2 (the chunk body is now in context): reply with the secret if the section has it,
      else NONE;
    - combine (no tool, sees the EXTRACTS): answer with the secret iff it actually reached the prompt.
    """
    class _LLM:
        def __init__(self):
            class _C:
                tool_call_mode = "native"
                context_window = 128_000
            self.config = _C()

        async def stream_complete(self, messages=None, tools=None, on_token=None):
            msgs = messages or []
            last_tool = next((m for m in reversed(msgs) if m.get("role") == "tool"), None)
            user_msgs = [m for m in msgs if m.get("role") == "user"]
            task_text = (user_msgs[-1].get("content") if user_msgs else "") or ""
            handle_m = re.search(r"tool_result_get\('([0-9a-f]{6,})'\)", task_text)

            if last_tool is None and handle_m:  # leaf turn 1: read the granted chunk for real
                r = Response(content=None, tool_calls=[ToolCall(id="t1", name="tool_result_get",
                                                                arguments={"id": handle_m.group(1)})])
                return r, r.usage
            if last_tool is not None:  # leaf turn 2: extract from the chunk now in context
                body = str(last_tool.get("content") or "")
                r = Response(content=(SECRET if SECRET in body else "NONE"))
                return r, r.usage
            joined = "\n".join(str(m.get("content") or "") for m in msgs)  # combine turn (no handle)
            r = Response(content=(f"FINAL ANSWER: {SECRET}" if SECRET in joined else "FINAL: not found"))
            return r, r.usage

    return _LLM()


@pytest.mark.asyncio
async def test_cruncher_reduces_granted_overwindow_body_faithfully(mock_llm_client, bus):
    """Root grants a handle to an over-window body; the cruncher splits it, a fresh leaf reads each
    section (real tool_result_get read-through), and the combine surfaces the buried secret — which
    lives in a NON-FIRST section, so a one-window read could not have found it."""
    Response, ToolCall = mock_llm_client.Response, mock_llm_client.ToolCall

    head = "Filler sentence about federalism and checks. " * 110   # ~5k chars, no secret
    tail = "More filler about ratification and republics. " * 110
    body = head + f"\nThe hidden passphrase is {SECRET}.\n" + tail   # secret buried in the middle
    assert SECRET not in head  # buried strictly after section 1

    parent = ContentStore()
    granted_h = parent.put(body, origin="trusted")

    base = ToolRegistry()
    await register_builtin_tools(base, eviction_store=ContentStore())  # registers tool_result_get

    # Granted read-through store + a SMALL window so the body is genuinely multi-section.
    ctx = ContextManager(
        content_store=ContentStore(parent=parent, granted=frozenset({granted_h})),
        max_context_tokens=8_000,
    )
    result = await subagent.dispatch_cruncher_subagent(
        "What is the hidden passphrase?",
        grant_handles=[granted_h], llm=_reactive_cruncher_llm(Response, ToolCall), bus=bus,
        base_registry=base, parent_session_id="run", permission_evaluator=PermissionEvaluator(),
        context_manager=ctx, depth=0, max_subagent_depth=2,
    )

    assert SECRET in result, f"cruncher must surface the buried fact; got: {result!r}"
    m = re.search(r"sections: (\d+)", result)
    assert m and int(m.group(1)) >= 2, "must have split the over-window body into multiple sections"


@pytest.mark.asyncio
async def test_cruncher_reports_not_found_when_fact_absent(mock_llm_client, bus):
    """If no section is relevant, the cruncher says so plainly — never fabricates the buried fact."""
    Response, ToolCall = mock_llm_client.Response, mock_llm_client.ToolCall
    body = "Only filler here about unrelated topics. " * 300  # no secret anywhere
    parent = ContentStore()
    h = parent.put(body, origin="trusted")
    base = ToolRegistry()
    await register_builtin_tools(base, eviction_store=ContentStore())
    ctx = ContextManager(content_store=ContentStore(parent=parent, granted=frozenset({h})),
                         max_context_tokens=8_000)
    result = await subagent.dispatch_cruncher_subagent(
        "What is the hidden passphrase?", grant_handles=[h],
        llm=_reactive_cruncher_llm(Response, ToolCall), bus=bus, base_registry=base,
        parent_session_id="run", permission_evaluator=PermissionEvaluator(), context_manager=ctx,
        depth=0, max_subagent_depth=2,
    )
    assert SECRET not in result and "not found" in result.lower()


@pytest.mark.asyncio
async def test_cruncher_routed_and_granted_via_run_agent(monkeypatch):
    """_run_agent routes agent_id='cruncher' to dispatch_cruncher_subagent with the grant threaded
    and the child ctx carrying the granted read-through store (the live delegation seam)."""
    captured: dict = {}

    async def _spy_cruncher(task, **kwargs):
        captured["task"] = task
        captured.update(kwargs)
        return "[cruncher] sections: 3 | tool calls: 0\n\nok"

    monkeypatch.setattr(subagent, "dispatch_cruncher_subagent", _spy_cruncher)

    parent = ContentStore()
    h = parent.put("the granted over-window doc")
    runner = subagent.make_explore_agent_runner(
        llm=object(), bus=object(), base_registry=object(), permission_evaluator=object(),
        get_parent_session_id=lambda: "sid", parent_store=parent,
    )
    out = await runner("cruncher", "distill it", grant_handles=[h])
    assert out.startswith("[cruncher]")
    assert captured["grant_handles"] == [h]
    assert captured["context_manager"]._content_store.get(h) == "the granted over-window doc"  # read-through


# --- store-aware delivery (Decision B): retain full body, return a grantable handle, no bytes inline


@pytest.mark.asyncio
async def test_load_document_retains_full_body_and_returns_handle(tmp_path):
    doc = tmp_path / "big.txt"
    text = "REAL document content. " * 5000  # ~115k chars
    doc.write_text(text, encoding="utf-8")

    store = ContentStore()
    res = await LoadDocumentTool(store).run(path=str(doc))
    assert res.success
    handle = res.metadata["doc_handle"]
    assert res.metadata["chars"] == len(text)
    assert text not in res.output                  # the body is NOT inlined — only a stub + handle
    assert store.get(handle) == text               # the FULL body is retained losslessly
    assert store.origin(handle) == "trusted"
    assert "grant_handles" in res.output           # guides the orchestrator to the cruncher


@pytest.mark.asyncio
async def test_load_document_not_found(tmp_path):
    res = await LoadDocumentTool(ContentStore()).run(path=str(tmp_path / "missing.txt"))
    assert not res.success and res.error_type == "not_found"


# --- Decision C: cruncher_exec wired, exec_enabled-gated, F3 origin-gated (no green-on-dead-code) ---


@pytest.mark.asyncio
async def test_cruncher_exec_offered_for_clean_and_withheld_for_untrusted(monkeypatch, mock_llm_client, bus):
    """exec_enabled + a CLEAN-origin granted handle => cruncher_exec is offered (constructed); an
    UNTRUSTED granted handle => bind_clean_origin_bodies refuses and exec is WITHHELD. This makes
    agent.cruncher.exec_enabled real and F3 a LIVE check, not a unit test on an unwired tool."""
    import localharness.tools.builtin.cruncher_exec as ce
    from localharness.config.models import CruncherConfig

    constructed: list = []
    real = ce.CruncherExecTool
    monkeypatch.setattr(ce, "CruncherExecTool",
                        lambda seed, **kw: (constructed.append(seed) or real(seed, **kw)))

    Response, ToolCall = mock_llm_client.Response, mock_llm_client.ToolCall
    base = ToolRegistry()
    await register_builtin_tools(base, eviction_store=ContentStore())
    cfg = CruncherConfig(exec_enabled=True)

    async def _run(origin: str):
        parent = ContentStore()
        h = parent.put("a small body to aggregate over", origin=origin)
        ctx = ContextManager(content_store=ContentStore(parent=parent, granted=frozenset({h})),
                             max_context_tokens=8_000)
        await subagent.dispatch_cruncher_subagent(
            "summarize", grant_handles=[h], llm=_reactive_cruncher_llm(Response, ToolCall), bus=bus,
            base_registry=base, parent_session_id="run", permission_evaluator=PermissionEvaluator(),
            context_manager=ctx, cruncher_config=cfg, max_subagent_depth=2,
        )

    await _run("trusted")
    assert len(constructed) == 1, "clean-origin grant + exec_enabled => cruncher_exec offered"
    constructed.clear()
    await _run("untrusted")
    assert constructed == [], "untrusted grant => cruncher_exec WITHHELD (F3 refuses to bind it)"


@pytest.mark.asyncio
async def test_cruncher_exec_not_offered_when_disabled(monkeypatch, mock_llm_client, bus):
    """Default (exec_enabled=False): cruncher_exec is never constructed — verbs-only."""
    import localharness.tools.builtin.cruncher_exec as ce
    from localharness.config.models import CruncherConfig
    constructed: list = []
    real = ce.CruncherExecTool
    monkeypatch.setattr(ce, "CruncherExecTool",
                        lambda seed, **kw: (constructed.append(seed) or real(seed, **kw)))
    Response, ToolCall = mock_llm_client.Response, mock_llm_client.ToolCall
    base = ToolRegistry()
    await register_builtin_tools(base, eviction_store=ContentStore())
    parent = ContentStore()
    h = parent.put("clean body", origin="trusted")
    ctx = ContextManager(content_store=ContentStore(parent=parent, granted=frozenset({h})),
                         max_context_tokens=8_000)
    await subagent.dispatch_cruncher_subagent(
        "q", grant_handles=[h], llm=_reactive_cruncher_llm(Response, ToolCall), bus=bus,
        base_registry=base, parent_session_id="run", permission_evaluator=PermissionEvaluator(),
        context_manager=ctx, cruncher_config=CruncherConfig(exec_enabled=False), max_subagent_depth=2,
    )
    assert constructed == []
