"""P3/P6 — blind search-verifier: parsers, keep-flag ledger, dispatch wiring, and the
DETERMINISTIC regression gate (a fixture whose entity-mismatch sentence sits PAST the inline
cap → verifier returns WRONG_ENTITY). Non-stochastic: the LLM is a scripted mock.
"""
from __future__ import annotations

import json

import pytest

from localharness.agent.context import ContentStore, ContextManager
from localharness.agent.permissions import PermissionEvaluator
from localharness.agent import subagent
from localharness.agent.subagent import (
    SEARCH_VERIFIER_MAX_ACTIONS,
    SEARCH_VERIFIER_MAX_TOOL_CALLS,
    SEARCH_VERIFIER_TOOLS,
    build_search_verifier_config,
    dispatch_search_verifier_subagent,
    format_verifier_flag,
    write_verification_ledger,
    _parse_verifier_task,
    _parse_verifier_verdict,
)
from localharness.tools.builtin import register_builtin_tools, web_tool
from localharness.tools.builtin.web_tool import WebPageQueryTool
from localharness.tools.registry import ToolRegistry


def _fake_httpx(monkeypatch, *, text, content_type="text/html"):
    class _Resp:
        def __init__(self):
            self.text = text
            self.headers = {"content-type": content_type}
            self.url = "https://news.test/article"
            self.encoding = "utf-8"
        def raise_for_status(self): pass
        def json(self): return None
        async def aiter_bytes(self):
            yield text.encode("utf-8")

    class _Stream:
        async def __aenter__(self): return _Resp()
        async def __aexit__(self, *a): return False

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, **k): return _Resp()
        def stream(self, method, url, **k): return _Stream()

    monkeypatch.setattr(web_tool.httpx, "AsyncClient", _Client)


# --- pure helpers (fully deterministic) ---------------------------------------

def test_parse_verifier_task_extracts_fields():
    task = "claim: QNT added to S&P 500\nentity: QNT\nsource_url: https://news.test/x"
    claim, entity, url = _parse_verifier_task(task)
    assert claim == "QNT added to S&P 500"
    assert entity == "QNT"
    assert url == "https://news.test/x"


def test_parse_verifier_verdict_valid_wrapped_and_malformed():
    good = _parse_verifier_verdict('prose... {"verdict":"WRONG_ENTITY","evidence":"about SPCX"} trailing')
    assert good["verdict"] == "WRONG_ENTITY" and good["evidence"] == "about SPCX"
    # malformed / no JSON → UNVERIFIABLE fallback (still produces a kept, flagged row)
    assert _parse_verifier_verdict("no json here")["verdict"] == "UNVERIFIABLE"
    assert _parse_verifier_verdict("{not valid}")["verdict"] == "UNVERIFIABLE"


def test_write_verification_ledger_keeps_and_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALHARNESS_VERIFICATION_LEDGER_DIR", str(tmp_path))
    row = write_verification_ledger(
        run_id="run-1", claim="QNT added to S&P 500", entity="QNT",
        source_url="https://news.test/x",
        verdict={"verdict": "WRONG_ENTITY", "evidence": "source is about SPCX"},
    )
    assert row["kept_in_report"] is True            # bad claims are KEPT, never dropped
    assert row["flags"] == ["WRONG_ENTITY"]
    assert row["ticker"] == "QNT" == row["entity"]
    written = (tmp_path / "verification-ledger.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(written)["verdict"] == "WRONG_ENTITY"

    # SUPPORTED carries no flags.
    sup = write_verification_ledger(run_id="r", claim="c", entity="e", source_url="u",
                                    verdict={"verdict": "SUPPORTED"})
    assert sup["flags"] == []


def test_format_verifier_flag_is_compact():
    flag = format_verifier_flag("QNT added to S&P 500", "QNT",
                                {"verdict": "WRONG_ENTITY", "evidence": "about SPCX"}, 4)
    assert flag.startswith("[search-verifier] verdict=WRONG_ENTITY")
    assert "entity=QNT" in flag and "\n" not in flag  # one compact line, not a transcript


# --- config / registry / wiring -----------------------------------------------

def test_search_verifier_config_is_leaf_budget():
    cfg = build_search_verifier_config()
    assert cfg.name == "search-verifier"
    assert cfg.permissions.budget.max_actions == SEARCH_VERIFIER_MAX_ACTIONS == 12
    assert SEARCH_VERIFIER_MAX_TOOL_CALLS == SEARCH_VERIFIER_MAX_ACTIONS + 1 == 13
    assert "STRICT JSON" in cfg.role and "BLIND" in cfg.role


@pytest.mark.asyncio
async def test_search_verifier_registry_is_web_only_no_agent():
    base = ToolRegistry()
    await register_builtin_tools(base)
    child = ToolRegistry.from_allowed(SEARCH_VERIFIER_TOOLS, base_registry=base)
    assert set(child._tools["global"].keys()) == {"web_search", "web_fetch", "web_page_query"}
    assert child.has("agent") is False  # the verifier is a leaf — it can never delegate


@pytest.mark.asyncio
async def test_runner_routes_search_verifier(monkeypatch):
    seen = {}
    async def _spy(task, **kw):
        seen["task"] = task
        return "[search-verifier] verdict=SUPPORTED"
    monkeypatch.setattr(subagent, "dispatch_search_verifier_subagent", _spy)

    runner = subagent.make_explore_agent_runner(
        llm=object(), bus=object(), base_registry=object(),
        permission_evaluator=object(), get_parent_session_id=lambda: "sid",
    )
    out = await runner("search-verifier", "claim: x\nentity: y\nsource_url: z")
    assert out.startswith("[search-verifier]") and seen["task"].startswith("claim:")


@pytest.mark.asyncio
async def test_web_researcher_is_non_leaf_advertising_only_the_verifier(monkeypatch):
    """With the real NON_LEAF_AGENTS, a depth-0 runner hands web-researcher an `agent` tool that
    advertises ONLY search-verifier (not explore/itself)."""
    captured = {}
    async def _spy_web(task, **kw):
        captured["tool"] = kw["child_agent_tool"]
        return "[web research] ok"
    monkeypatch.setattr(subagent, "dispatch_web_subagent", _spy_web)

    runner = subagent.make_explore_agent_runner(
        llm=object(), bus=object(), base_registry=object(),
        permission_evaluator=object(), get_parent_session_id=lambda: "sid",
        depth=0, max_subagent_depth=2, available_agents=["web-researcher", "search-verifier"],
    )
    await runner("web-researcher", "research QNT")
    tool = captured["tool"]
    assert tool is not None and tool._available_agents == ["search-verifier"]


# --- DETERMINISTIC GATE: past-cap entity-mismatch → WRONG_ENTITY --------------

@pytest.mark.asyncio
async def test_deterministic_gate_wrong_entity_past_inline_cap(mock_llm_client, bus, tmp_path, monkeypatch):
    """The regression gate for the QNT/S&P fabrication class. A fixture page mentions the target
    ticker early, then states the S&P addition for a DIFFERENT ticker PAST the 5000-char inline cap.
    The blind verifier re-fetches, grounds in the FULL page via web_page_query (lossless), and the
    scripted verdict is WRONG_ENTITY. Deterministic — no live model."""
    web_tool._reset_page_store()
    monkeypatch.setenv("LOCALHARNESS_VERIFICATION_LEDGER_DIR", str(tmp_path))

    early = "QNT (Quant) is a small-cap names investors are watching this quarter. " * 100  # ~6700 chars
    needle = "SPCX was added to the S&P 500 index effective 2026-01-15, replacing an outgoing member."
    page = f"<html><body>{early}\n{needle}\n</body></html>"
    _fake_httpx(monkeypatch, text=page, content_type="text/html")

    Response = mock_llm_client.Response
    ToolCall = mock_llm_client.ToolCall
    # After _reset_page_store, the verifier's own first web_fetch deterministically yields pg-1.
    llm = mock_llm_client([
        Response(content=None, tool_calls=[ToolCall(id="f1", name="web_fetch",
                 arguments={"url": "https://news.test/article"})]),
        Response(content=None, tool_calls=[ToolCall(id="q1", name="web_page_query",
                 arguments={"fetch_id": "pg-1", "pattern": "S&P 500"})]),
        Response(content=json.dumps({
            "verdict": "WRONG_ENTITY", "entity_in_source": False, "source_date": "2026-01-15",
            "evidence": "SPCX was added to the S&P 500 index", "fresh_search_note": "no QNT addition found",
        })),
    ])

    base = ToolRegistry()
    await register_builtin_tools(base)
    task = "claim: QNT added to S&P 500\nentity: QNT\nsource_url: https://news.test/article"
    # Per-agent store cutover: hand the verifier a known ContentStore so the test can assert the
    # lossless past-cap retrieval against the SAME store the verifier's own web_fetch wrote to.
    store = ContentStore()
    flag = await dispatch_search_verifier_subagent(
        task, llm=llm, bus=bus, base_registry=base, parent_session_id="run-qnt",
        permission_evaluator=PermissionEvaluator(),
        context_manager=ContextManager(content_store=store),
    )

    # 1) verdict surfaced to the parent as a compact flag
    assert "verdict=WRONG_ENTITY" in flag and "entity=QNT" in flag

    # 2) lossless: the verifier's own fetch (pg-1) retained the FULL page, and web_page_query
    #    surfaces the PAST-cap disconfirming sentence (the inline preview clips it at 5000 chars).
    assert len(early) > 5000, "fixture must push the needle past the inline cap to be a real gate"
    q = await WebPageQueryTool(store).run(fetch_id="pg-1", pattern="S&P 500")
    assert needle in q.output, "web_page_query must surface the disconfirming sentence past the cap"

    # 3) keep-flag ledger row written (kept + flagged, not dropped)
    row = json.loads((tmp_path / "verification-ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()[0])
    assert row["verdict"] == "WRONG_ENTITY"
    assert row["kept_in_report"] is True and row["flags"] == ["WRONG_ENTITY"]
    assert row["entity"] == "QNT" and row["run_id"] == "run-qnt"


@pytest.mark.asyncio
async def test_deterministic_gate_stale_verdict(mock_llm_client, bus, tmp_path, monkeypatch):
    """A STALE fixture: the scripted verdict carries an old source_date and STALE verdict; the
    ledger keeps + flags it."""
    web_tool._reset_page_store()
    monkeypatch.setenv("LOCALHARNESS_VERIFICATION_LEDGER_DIR", str(tmp_path))
    _fake_httpx(monkeypatch, text="<html><body>" + ("old news. " * 80) + "</body></html>")

    Response = mock_llm_client.Response
    ToolCall = mock_llm_client.ToolCall
    llm = mock_llm_client([
        Response(content=None, tool_calls=[ToolCall(id="f1", name="web_fetch",
                 arguments={"url": "https://news.test/article"})]),
        Response(content=json.dumps({
            "verdict": "STALE", "entity_in_source": True, "source_date": "2019-03-01",
            "evidence": "figure from 2019", "fresh_search_note": "superseded since",
        })),
    ])
    base = ToolRegistry()
    await register_builtin_tools(base)
    flag = await dispatch_search_verifier_subagent(
        "claim: ACME revenue is $1B\nentity: ACME\nsource_url: https://news.test/article",
        llm=llm, bus=bus, base_registry=base, parent_session_id="run-stale",
        permission_evaluator=PermissionEvaluator(),
    )
    assert "verdict=STALE" in flag
    row = json.loads((tmp_path / "verification-ledger.jsonl").read_text(encoding="utf-8").strip())
    assert row["verdict"] == "STALE" and row["flags"] == ["STALE"] and row["kept_in_report"] is True
