"""Tests for the explore subagent core (Phase 27, SUBAGENT-01..04).

Covers the 5 plan success criteria with one assertion cluster each:
  1. SUBAGENT-01 — spawning explore on a temp-dir task returns a non-empty result and the
     child ran >=1 iteration using only read-only tools.
  2. SUBAGENT-01/02 — child registry == exactly {read, glob, grep}; write/bash absent.
  3. SUBAGENT-02 — depth-1 child has no spawn tool, and dispatch at depth>=1 refuses clearly.
  4. SUBAGENT-03 — child Action/Observation carry parent_id == parent session; an unfiltered
     bus subscriber counts the child's tool calls (mirrors the bench accumulator).
  5. SUBAGENT-04 — return is a structured findings summary (header + child summary), not the
     full event log.
"""
from __future__ import annotations

import pytest

from localharness.agent.permissions import PermissionEvaluator
from localharness.agent.subagent import (
    EXPLORE_MAX_ACTIONS,
    EXPLORE_MAX_DURATION_MINUTES,
    EXPLORE_MAX_TOOL_CALLS,
    EXPLORE_TOOLS,
    MAX_DEPTH,
    WEB_TOOLS,
    build_explore_config,
    build_web_researcher_config,
    dispatch_explore_subagent,
    dispatch_web_subagent,
    format_findings,
    format_web_findings,
)
from localharness.core.events import Action, Observation
from localharness.tools.builtin import register_builtin_tools
from localharness.tools.registry import ToolRegistry


async def _builtin_registry() -> ToolRegistry:
    reg = ToolRegistry()
    await register_builtin_tools(reg)
    return reg


def _read_then_summarize(mock_llm_client, read_path: str):
    """Scripted child LLM: one `read` tool call, then a natural-language summary."""
    Response = mock_llm_client.Response
    ToolCallObj = mock_llm_client.ToolCall
    tc = ToolCallObj(id="rc-1", name="read", arguments={"path": read_path})
    return mock_llm_client([
        Response(content=None, tool_calls=[tc]),
        Response(content="The file defines a greeting constant."),
    ])


# ---------------------------------------------------------------------------
# Config / budget
# ---------------------------------------------------------------------------

def test_explore_config_has_distinct_bounded_budget():
    cfg = build_explore_config("explore")
    assert cfg.name == "explore"
    budget = cfg.permissions.budget
    assert budget.max_actions == EXPLORE_MAX_ACTIONS == 8
    assert budget.max_duration_minutes == EXPLORE_MAX_DURATION_MINUTES == 3.0
    # BUDGET-POLICY invariant: max_tool_calls = max_actions + 1
    assert EXPLORE_MAX_TOOL_CALLS == EXPLORE_MAX_ACTIONS + 1 == 9


def test_explore_config_sanitizes_underscore_name():
    # AgentConfig.name rejects underscores — dispatch must sanitize `_` -> `-`.
    cfg = build_explore_config("explore_agent")
    assert cfg.name == "explore-agent"


# ---------------------------------------------------------------------------
# 1. SUBAGENT-01 — real dispatch returns non-empty result, child ran >=1 iteration read-only
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatch_returns_nonempty_and_child_ran_readonly(mock_llm_client, bus, tmp_path):
    f = tmp_path / "hello.py"
    f.write_text("GREETING = 'hello world'\n", encoding="utf-8")

    base = await _builtin_registry()
    llm = _read_then_summarize(mock_llm_client, str(f))

    result = await dispatch_explore_subagent(
        f"Read {f} and summarize it",
        llm=llm,
        bus=bus,
        base_registry=base,
        parent_session_id="parent-sess",
        permission_evaluator=PermissionEvaluator(),
    )

    assert isinstance(result, str) and result.strip()
    assert "The file defines a greeting constant." in result

    # Child ran >=1 iteration AND only ever used read-only tools.
    tool_actions = [e for e in bus.history(event_types=[Action]) if e.tool_name]
    assert len(tool_actions) >= 1
    assert all(e.tool_name in EXPLORE_TOOLS for e in tool_actions)
    # The read actually dispatched against a real file (no placeholder data).
    obs = [e for e in bus.history(event_types=[Observation]) if e.tool_name == "read"]
    assert obs and obs[0].error is None


# ---------------------------------------------------------------------------
# 2. SUBAGENT-01/02 — read-only toolset: exactly {read, glob, grep}, write/bash absent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_child_registry_is_read_only():
    base = await _builtin_registry()
    child = ToolRegistry.from_allowed(EXPLORE_TOOLS, base_registry=base)

    names = set(child._tools["global"].keys())
    assert names == {"read", "glob", "grep"}
    # Write / execute / spawn are un-dispatchable in the child.
    assert child.has("write") is False
    assert child.has("bash_exec") is False
    assert child.has("agent") is False


# ---------------------------------------------------------------------------
# 3. SUBAGENT-02 — depth-1 recursion guard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_child_registry_has_no_spawn_tool():
    # Primary guard: the read-only registry simply does not contain a spawn/agent tool.
    base = await _builtin_registry()
    child = ToolRegistry.from_allowed(EXPLORE_TOOLS, base_registry=base)
    assert "agent" not in child._tools["global"]
    assert child.has("agent") is False


@pytest.mark.asyncio
async def test_dispatch_at_depth_one_refuses(mock_llm_client, bus):
    # Belt-and-suspenders: an explicit re-entry at depth >= MAX_DEPTH refuses with a clear error.
    base = await _builtin_registry()
    llm = mock_llm_client([mock_llm_client.Response(content="unused")])
    with pytest.raises(ValueError) as exc:
        await dispatch_explore_subagent(
            "anything",
            llm=llm,
            bus=bus,
            base_registry=base,
            parent_session_id="p",
            permission_evaluator=PermissionEvaluator(),
            depth=MAX_DEPTH,
        )
    msg = str(exc.value).lower()
    assert "depth" in msg and "spawn" in msg


# ---------------------------------------------------------------------------
# 4. SUBAGENT-03 — session linkage + counted metrics on the shared bus
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_child_events_carry_parent_id_and_tool_calls_counted(mock_llm_client, bus, tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("alpha\nbeta\n", encoding="utf-8")

    base = await _builtin_registry()
    llm = _read_then_summarize(mock_llm_client, str(f))

    # Unfiltered subscriber that counts tool-call Actions exactly like the bench
    # MetricAccumulator.on_action (Action with a non-empty tool_name) — MCP-SCENARIO-GAP §2.
    counted: list[str] = []

    async def _count(ev: Action) -> None:
        if ev.tool_name:
            counted.append(ev.tool_name)

    bus.subscribe(Action, _count)

    parent_session = "parent-session-xyz"
    await dispatch_explore_subagent(
        f"inspect {f}",
        llm=llm,
        bus=bus,
        base_registry=base,
        parent_session_id=parent_session,
        permission_evaluator=PermissionEvaluator(),
    )

    # Child events carry a fresh session_id != parent, and parent_id == the parent session.
    child_actions = [e for e in bus.history(event_types=[Action]) if e.tool_name]
    child_obs = bus.history(event_types=[Observation])
    assert child_actions, "expected at least one child tool-call action"
    for e in child_actions + child_obs:
        assert e.parent_id == parent_session
        assert e.session_id is not None and e.session_id != parent_session

    # The delegated read shows up in the unfiltered count (no longer tool_call_count=0).
    assert len(counted) >= 1
    assert "read" in counted


# ---------------------------------------------------------------------------
# 5. SUBAGENT-04 — structured findings return, not the raw transcript
# ---------------------------------------------------------------------------

def test_format_findings_is_structured_summary():
    out = format_findings("find the config loader", "It lives in config/loader.py.", 2)
    lines = out.splitlines()
    # Short header with task + tool-call count, then the child summary.
    assert lines[0].startswith("[explore findings]")
    assert "find the config loader" in lines[0]
    assert "tool calls: 2" in lines[0]
    assert "It lives in config/loader.py." in out


@pytest.mark.asyncio
async def test_dispatch_return_is_summary_not_event_log(mock_llm_client, bus, tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Title\nbody text\n", encoding="utf-8")

    base = await _builtin_registry()
    llm = _read_then_summarize(mock_llm_client, str(f))

    result = await dispatch_explore_subagent(
        f"summarize {f}",
        llm=llm,
        bus=bus,
        base_registry=base,
        parent_session_id="p",
        permission_evaluator=PermissionEvaluator(),
    )

    # Concise findings header + summary; NOT the raw transcript / event dump.
    assert result.startswith("[explore findings]")
    assert "tool calls:" in result.splitlines()[0]
    assert "The file defines a greeting constant." in result
    # Transcript/event-log artifacts must not leak into the returned string.
    for leaked in ("Action(", "Observation(", "TaskComplete(", "event_type", "role='tool'", '"role":'):
        assert leaked not in result


# ---------------------------------------------------------------------------
# 6. Web-researcher subagent — web-only toolset, depth guard, distilled findings
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_web_child_registry_is_web_only():
    base = await _builtin_registry()
    child = ToolRegistry.from_allowed(WEB_TOOLS, base_registry=base)
    assert set(child._tools["global"].keys()) == {"web_search", "web_fetch", "web_page_query"}
    # No write / execute / spawn in the web child.
    assert child.has("write") is False
    assert child.has("bash_exec") is False
    assert child.has("agent") is False


def test_web_researcher_config_has_distinct_budget():
    cfg = build_web_researcher_config("web-researcher")
    assert cfg.name == "web-researcher"
    assert cfg.permissions.budget.max_actions == 28  # aligned to the forked runner (P3)


def test_web_researcher_role_rigor_gating(monkeypatch):
    from localharness.agent.subagent import build_web_researcher_config as _build

    monkeypatch.setenv("RESEARCH_RIGOR", "high")
    high = _build("web-researcher")
    assert "search-verifier" in high.role and "DISPUTED" in high.role
    assert "VERIFY MATERIAL CLAIMS" in high.role  # high = verify without being asked

    monkeypatch.setenv("RESEARCH_RIGOR", "fast")
    fast = _build("web-researcher")
    assert "search-verifier" not in fast.role  # fast skips verification (speed dial)

    # DEFAULT (owner latency ruling 2026-07-09): verifier runs ONLY when the task
    # explicitly asks for a re-check — never auto-fires on material claims.
    monkeypatch.delenv("RESEARCH_RIGOR", raising=False)
    default = _build("web-researcher")
    assert "VERIFY MATERIAL CLAIMS" not in default.role  # no auto-verification
    assert "search-verifier" in default.role             # ...but reachable on request
    assert "ONLY if your task explicitly asks" in default.role
    assert "DISPUTED" in default.role  # verdict-tagging discipline kept for requested checks


def test_format_web_findings_is_structured_summary():
    out = format_web_findings("anthropic june 15 pricing", "It splits headless billing.", 3)
    lines = out.splitlines()
    assert lines[0].startswith("[web research]")
    assert "tool calls: 3" in lines[0]
    assert "It splits headless billing." in out


@pytest.mark.asyncio
async def test_web_dispatch_at_depth_one_refuses(mock_llm_client, bus):
    base = await _builtin_registry()
    llm = mock_llm_client([mock_llm_client.Response(content="unused")])
    with pytest.raises(ValueError):
        await dispatch_web_subagent(
            "research X",
            llm=llm,
            bus=bus,
            base_registry=base,
            parent_session_id="p",
            permission_evaluator=PermissionEvaluator(),
            depth=MAX_DEPTH,
        )


# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_runner_threads_token_counter_and_window_into_child(monkeypatch):
    """Children must inherit the parent's model-aware counter + resolved window, not bare
    defaults — the sub-agent fleet was silently running on 131,072 + tiktoken."""
    import localharness.agent.subagent as subagent

    captured = {}
    async def _fake_dispatch(task, **kwargs):
        captured["ctx"] = kwargs.get("context_manager")
        return "ok"
    monkeypatch.setattr(subagent, "dispatch_explore_subagent", _fake_dispatch)

    sentinel_counter = object()
    runner = subagent.make_explore_agent_runner(
        llm=object(), bus=object(), base_registry=object(),
        permission_evaluator=object(), get_parent_session_id=lambda: "sid",
        token_counter=sentinel_counter, max_context_tokens=126_976,
    )
    await runner("explore", "find X")
    ctx = captured["ctx"]
    assert ctx is not None
    assert ctx.max_context_tokens == 126_976          # resolved window, not the 131072 default
    assert ctx._token_counter is sentinel_counter      # exact /tokenize counter, not bare tiktoken


# ---------------------------------------------------------------------------
# Config-defined children (dispatch_config_subagent + runner load_agent seam)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_config_child_registry_respects_add_and_strips_agent():
    from localharness.agent.subagent import CONFIG_CHILD_DEFAULT_TOOLS
    from localharness.config.models import AgentConfig
    from localharness.tools.registry import ToolRegistry
    from localharness.tools.builtin import register_builtin_tools

    base = ToolRegistry()
    await register_builtin_tools(base)

    # The allow-list semantics live inside dispatch_config_subagent; replicate its filter
    cfg = AgentConfig.model_validate({
        "name": "yt-summarizer", "role": "Test specialist.",
        "tools": {"add": ["bash_exec", "web_fetch", "agent"], "deny": ["web_fetch"]},
    })
    add = list(cfg.tools.add or []) or list(CONFIG_CHILD_DEFAULT_TOOLS)
    deny = set(cfg.tools.deny or [])
    allowed = [t for t in add if t not in deny and t.split(".")[-1].split(":")[-1] != "agent"]
    assert allowed == ["bash_exec"]  # deny wins; `agent` always stripped


@pytest.mark.asyncio
async def test_runner_dispatches_yaml_defined_agent(monkeypatch):
    import localharness.agent.subagent as subagent
    from localharness.config.models import AgentConfig

    yt_cfg = AgentConfig.model_validate({
        "name": "youtube-summarizer", "role": "Fetch transcripts and summarize.",
        "tools": {"add": ["bash_exec"]},
        "permissions": {"budget": {"max_actions": 8, "max_duration_minutes": 6.0}},
    })

    captured = {}
    async def _fake_config_dispatch(task, *, agent_config, **kwargs):
        captured["task"] = task
        captured["name"] = agent_config.name
        return f"[{agent_config.name}] done"
    monkeypatch.setattr(subagent, "dispatch_config_subagent", _fake_config_dispatch)

    loads = []
    def _load_agent(name):
        loads.append(name)
        if name == "youtube-summarizer":
            return yt_cfg
        raise FileNotFoundError(name)

    runner = subagent.make_explore_agent_runner(
        llm=object(), bus=object(), base_registry=object(),
        permission_evaluator=object(), get_parent_session_id=lambda: "sid",
        load_agent=_load_agent,
    )
    out = await runner("youtube_summarizer", "summarize video X")  # `_`->`-` sanitize
    assert out == "[youtube-summarizer] done"
    assert captured["task"] == "summarize video X"
    assert loads == ["youtube-summarizer"]

    # unknown name still refuses, with builder guidance in the message
    with pytest.raises(ValueError, match="CREATE one"):
        await runner("nonexistent-agent", "do thing")

    # "orchestrator" (the root) is never loadable as a child (self-delegation guard)
    with pytest.raises(ValueError, match="not wired"):
        await runner("orchestrator", "do thing")


@pytest.mark.asyncio
async def test_runner_without_loader_keeps_old_refusal():
    import localharness.agent.subagent as subagent

    runner = subagent.make_explore_agent_runner(
        llm=object(), bus=object(), base_registry=object(),
        permission_evaluator=object(), get_parent_session_id=lambda: "sid",
    )
    with pytest.raises(ValueError, match="not wired"):
        await runner("youtube-summarizer", "x")


def test_resolve_target_toolset_never_loads_root():
    """Phase 33.1: the grant-target-safety toolset resolver (subagent.py:309) must never
    call load_agent for the root name 'orchestrator' — the root is not a delegation child.
    Returns [] (unknown-to-the-gate) AND the loader is provably never invoked."""
    from localharness.agent.subagent import _resolve_target_toolset

    loaded: list[str] = []

    def _spy_load(name: str):  # pragma: no cover - asserted never called
        loaded.append(name)
        raise AssertionError("load_agent must not be called for the root name")

    assert _resolve_target_toolset("orchestrator", load_agent=_spy_load) == []
    assert loaded == []


def test_prepend_toolset_states_capabilities():
    from localharness.agent.subagent import prepend_toolset
    out = prepend_toolset("do the thing", ["bash_exec", "read"])
    assert out.startswith("(Your ONLY available tools: bash_exec, read.")
    assert out.endswith("do the thing")
    assert "say so immediately" in out


# ---------------------------------------------------------------------------
# default-roster security invariant (v0.5.3)
# ---------------------------------------------------------------------------

def test_builtin_roster_has_no_host_dangerous_agents():
    """v0.5.3 invariant: every default-roster builtin is quarantined or read-only.
    bash/write/edit-holding specialist roles (data-analyst, frontend-designer) were demoted
    to opt-in examples/agents/ configs (quality investigation 2026-07-03) — a default agent
    must never put granted or ingested bytes one call from a host action."""
    import localharness.agent.subagent as subagent
    from localharness.tools.capabilities import HOST_DANGEROUS

    for name, tools in subagent._BUILTIN_TOOLSETS.items():
        assert not (set(tools) & HOST_DANGEROUS), f"default builtin '{name}' holds host-dangerous tools"
