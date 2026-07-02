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


def _extract_all_llm(Response, ToolCall, marker):
    """A reactive fake whose leaves extract EVERY chunk (forcing many extracts → a broad query),
    and whose combine echoes whether the marker is present — so we can assert the needle survives
    the hierarchical (batched) reduce."""
    class _LLM:
        def __init__(self):
            class _C:
                tool_call_mode = "native"
                context_window = 128_000
            self.config = _C()

        async def stream_complete(self, messages=None, tools=None, on_token=None):
            msgs = messages or []
            last_tool = next((m for m in reversed(msgs) if m.get("role") == "tool"), None)
            users = [m for m in msgs if m.get("role") == "user"]
            task = (users[-1].get("content") if users else "") or ""
            hm = re.search(r"tool_result_get\('([0-9a-f]{6,})'\)", task)
            if last_tool is None and hm:
                r = Response(content=None, tool_calls=[ToolCall(id="t", name="tool_result_get",
                                                                arguments={"id": hm.group(1)})])
                return r, r.usage
            if last_tool is not None:
                r = Response(content=str(last_tool.get("content") or "")[:1500])  # every chunk -> an extract
                return r, r.usage
            joined = "\n".join(str(m.get("content") or "") for m in msgs)
            r = Response(content=f"COMBINED[{'HAS:' + marker if marker in joined else 'none'}]")
            return r, r.usage
    return _LLM()


@pytest.mark.asyncio
async def test_cruncher_broad_query_hierarchical_reduce_preserves_needle(mock_llm_client, bus, caplog):
    """A broad query makes MANY sections relevant → the extracts exceed one window → the reduce must
    BATCH (hierarchical) and still preserve a needle buried in one section. Guards the single-pass
    combine overflow seen in the live injection dogfood."""
    import logging
    Response, ToolCall = mock_llm_client.Response, mock_llm_client.ToolCall
    marker = "NEEDLE-7Q"
    filler = "alpha beta gamma delta epsilon. " * 1400  # ~43k chars, no newlines -> hard chunks
    body = filler[:21000] + f" {marker} " + filler[21000:]
    parent = ContentStore()
    h = parent.put(body, origin="trusted")
    # Small window => small chunks => many sections => extracts exceed the combine budget => batches.
    ctx = ContextManager(content_store=ContentStore(parent=parent, granted=frozenset({h})),
                         max_context_tokens=4_000)
    base = ToolRegistry()
    await register_builtin_tools(base, eviction_store=ContentStore())

    with caplog.at_level(logging.INFO, logger="localharness.agent.subagent"):
        res = await subagent.dispatch_cruncher_subagent(
            "Summarize every point in the document.", grant_handles=[h],
            llm=_extract_all_llm(Response, ToolCall, marker), bus=bus, base_registry=base,
            parent_session_id="run", permission_evaluator=PermissionEvaluator(), context_manager=ctx,
            max_subagent_depth=2,
        )

    assert marker in res, f"needle must survive the hierarchical reduce; got: {res[:300]!r}"
    assert any("reduce L1" in r.message for r in caplog.records), "batched (hierarchical) reduce must trigger on a broad query"


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


@pytest.mark.asyncio
async def test_chunk_summarizer_prepends_contextual_header(monkeypatch):
    """#1 contextual tagging: the leaf task carries a 'section i of N of a larger document' header so a
    leaf keeps cross-reference orientation and doesn't over-claim about the whole document from one
    slice. The header is added to the PROMPT only — the stored chunk bytes stay byte-identical
    (lossless). This is the AUTHORITATIVE check: live session traces don't persist the leaf prompt, so
    the header is invisible to trace-grep even though it IS sent to the leaf."""
    captured: dict = {}

    class _FakeLeafLoop:
        def __init__(self, **kw):
            pass

        async def run_turn(self, task, *a, **k):
            captured["task"] = task
            return "EXTRACT-OK"

    monkeypatch.setattr("localharness.agent.loop.AgentLoop", _FakeLeafLoop)

    store = ContentStore()
    h = store.put("the section body text", origin="trusted")
    out = await subagent._run_chunk_summarizer(
        h, "what governs Venice?", store, llm=None, bus=None, base_registry=ToolRegistry(),
        parent_session_id=None, permission_evaluator=None, token_counter=None,
        max_context_tokens=8000, depth=1, max_subagent_depth=3,
        section_label="section 4 of 12 of a larger document; you are reading ONE section",
    )
    assert out == "EXTRACT-OK"
    assert "section 4 of 12 of a larger document" in captured["task"]   # header reached the leaf prompt
    assert store.get(h) == "the section body text"                      # stored chunk NOT mutated


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


# --- R2: number-provenance net (a figure in the final answer must trace to a leaf, not a lossy node) ---


def test_cruncher_unverified_numbers_flags_only_ungrounded():
    """R2 pure-unit (tests the CODE, no LLM): a figure absent from every leaf extract is flagged; a
    grounded figure — including a reformatted one — is not; same number two ways de-dups to one flag."""
    extracts = [
        "[section 1]\nRevenue was $5,140 million for the quarter.",
        "[section 2]\nThe effective tax rate was 15% in Q3 2026.",
    ]
    # grounded incl. reformatting ($5,140M==$5,140 million, 15.0%==15%) -> no flag
    assert subagent._cruncher_unverified_numbers("Revenue hit $5,140M at a 15.0% tax rate.", extracts) == []
    # a figure in NO leaf extract -> flagged (returns the surface token)
    flagged = subagent._cruncher_unverified_numbers("The consolidated total was $9,999M.", extracts)
    assert len(flagged) == 1 and "9,999" in flagged[0]
    # de-dup by normalized form: the same number written two ways -> a single flag
    assert len(subagent._cruncher_unverified_numbers("$9,999M, i.e. 9,999 million", extracts)) == 1


def _fabricating_combine_llm(Response, ToolCall, answer_text):
    """Leaves echo their chunk body (number-free filler -> number-free extracts); every combine turn
    (partial AND final) returns `answer_text` -> the final answer's figure originates from a NODE, not
    a leaf — exactly reproducing the R2 lossy leak the number-provenance net must catch."""
    class _LLM:
        def __init__(self):
            class _C:
                tool_call_mode = "native"
                context_window = 128_000
            self.config = _C()

        async def stream_complete(self, messages=None, tools=None, on_token=None):
            msgs = messages or []
            last_tool = next((m for m in reversed(msgs) if m.get("role") == "tool"), None)
            users = [m for m in msgs if m.get("role") == "user"]
            task = (users[-1].get("content") if users else "") or ""
            hm = re.search(r"tool_result_get\('([0-9a-f]{6,})'\)", task)
            if last_tool is None and hm:  # leaf turn 1: read the granted chunk for real
                r = Response(content=None, tool_calls=[ToolCall(id="t", name="tool_result_get",
                                                                arguments={"id": hm.group(1)})])
                return r, r.usage
            if last_tool is not None:  # leaf turn 2: echo the chunk body -> an extract
                r = Response(content=str(last_tool.get("content") or "")[:1500])
                return r, r.usage
            r = Response(content=answer_text)  # combine turn (partial + final): inject the figure
            return r, r.usage
    return _LLM()


@pytest.mark.asyncio
async def test_cruncher_flags_unverified_figure_on_broad_query(mock_llm_client, bus, caplog):
    """R2 WIRING: on a broad query (level>=1, lossy partial nodes inserted) a fabricated figure in the
    final combine — absent from every leaf extract — is flagged in the RETURNED result + a log.warning.
    Proves the check is reachable end-to-end, not a green unit test on code nothing calls."""
    import logging
    Response, ToolCall = mock_llm_client.Response, mock_llm_client.ToolCall
    filler = "alpha beta gamma delta epsilon. " * 1400  # ~43k chars, NO digits anywhere
    parent = ContentStore()
    h = parent.put(filler, origin="trusted")
    # Tiny window => many sections => extracts exceed the combine budget => level>=1 (lossy partials).
    ctx = ContextManager(content_store=ContentStore(parent=parent, granted=frozenset({h})),
                         max_context_tokens=4_000)
    base = ToolRegistry()
    await register_builtin_tools(base, eviction_store=ContentStore())

    with caplog.at_level(logging.WARNING, logger="localharness.agent.subagent"):
        res = await subagent.dispatch_cruncher_subagent(
            "Summarize every figure in the document.", grant_handles=[h],
            llm=_fabricating_combine_llm(Response, ToolCall, "Final: the consolidated total is $9,999."),
            bus=bus, base_registry=base, parent_session_id="run",
            permission_evaluator=PermissionEvaluator(), context_manager=ctx, max_subagent_depth=2,
        )
    assert "unverified figures" in res and "9,999" in res, f"R2 flag must reach the result; got {res[:300]!r}"
    assert any("not found in any leaf extract" in r.message for r in caplog.records), "log.warning must fire"


@pytest.mark.asyncio
async def test_cruncher_skips_number_check_on_targeted_query(mock_llm_client, bus):
    """R2 GATE: a TARGETED query does ONE final combine over the raw extracts (level==0, no lossy
    node), so the check must NOT fire — even on an ungrounded figure — to avoid importing the
    normalization false-positive rate where there is no bug to catch."""
    Response, ToolCall = mock_llm_client.Response, mock_llm_client.ToolCall
    body = "A short note about widgets and gadgets. " * 8  # ~320 chars -> ONE section -> level 0
    parent = ContentStore()
    h = parent.put(body, origin="trusted")
    ctx = ContextManager(content_store=ContentStore(parent=parent, granted=frozenset({h})),
                         max_context_tokens=8_000)
    base = ToolRegistry()
    await register_builtin_tools(base, eviction_store=ContentStore())
    res = await subagent.dispatch_cruncher_subagent(
        "What is the widget count?", grant_handles=[h],
        llm=_fabricating_combine_llm(Response, ToolCall, "There are $9,999 widgets."),
        bus=bus, base_registry=base, parent_session_id="run",
        permission_evaluator=PermissionEvaluator(), context_manager=ctx, max_subagent_depth=2,
    )
    assert "9,999" in res, f"combine must have run + emitted the ungrounded figure; got {res[:200]!r}"  # non-vacuous
    assert "unverified figures" not in res, f"targeted query (level 0) must skip the check; got {res[:200]!r}"


def test_cruncher_chunk_chars_knee():
    """v1.7 speed<->quality: the chunk-size cap is 32k — the validated knee (real-27B sweep: 100%
    needle recall at 24-32k, 89% at the old 16k cap on a dense doc, cliff past 32k). Half the window
    for mid sizes; floored for tiny windows; never above the 32k knee (the cliff above is sharp)."""
    f = subagent._cruncher_chunk_chars
    assert f(126_976) == 32_000, "prod window -> capped at the 32k knee (was 16k)"
    assert f(40_000) == 20_000, "mid window -> 0.5x, under the cap"
    assert f(2_000) == 2_000, "tiny window -> floor (tiny-window safety)"
    assert f(None) == subagent._CRUNCHER_DEFAULT_CHUNK_CHARS, "no window -> default"
    assert f(1_000_000) <= 32_000, "never exceed the 32k knee — the recall cliff above is sharp"


@pytest.mark.asyncio
async def test_cruncher_persists_gist_tree_via_real_dispatch(mock_llm_client, bus, tmp_path):
    """Whole-milestone critic M2: exercise the ACTUAL wired path — dispatch_cruncher_subagent
    (memory_store=...) → persist_gist_tree — not a hand-called persist. A green test on the
    composed path, per the project's own 'prove it's wired' rule."""
    from localharness.memory.sqlite import FactQuery, MemoryStore

    Response, ToolCall = mock_llm_client.Response, mock_llm_client.ToolCall
    head = "Filler sentence about federalism and checks. " * 110
    tail = "More filler about ratification and republics. " * 110
    body = head + f"\nThe hidden passphrase is {SECRET}.\n" + tail

    parent = ContentStore()
    granted_h = parent.put(body, origin="trusted")
    base = ToolRegistry()
    await register_builtin_tools(base, eviction_store=ContentStore())
    ctx = ContextManager(
        content_store=ContentStore(parent=parent, granted=frozenset({granted_h})),
        max_context_tokens=8_000,
    )
    store = MemoryStore(agent_id="cruncher-mem", division_id="", org_id="", base_dir=str(tmp_path))
    await store.open()
    try:
        result = await subagent.dispatch_cruncher_subagent(
            "What is the hidden passphrase?",
            grant_handles=[granted_h], llm=_reactive_cruncher_llm(Response, ToolCall), bus=bus,
            base_registry=base, parent_session_id="run", permission_evaluator=PermissionEvaluator(),
            context_manager=ctx, depth=0, max_subagent_depth=2,
            memory_store=store,
        )
        assert SECRET in result  # the answer path is untouched by persistence
        rows = await store.query_facts(FactQuery(min_confidence=0.0, limit=100))
        kinds = {f.node_kind for f in rows}
        assert "schema" in kinds and "gist" in kinds
        finals = [f for f in rows if f.key.endswith("/final")]
        assert finals and "session:run" in finals[0].provenance  # verbatim pointer intact
    finally:
        await store.close()
