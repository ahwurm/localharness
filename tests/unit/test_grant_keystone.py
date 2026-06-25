"""J3 grant keystone — REACHABLE-path tests (no green-on-dead-code).

The store-level read-through primitive is covered by test_per_agent_store.py
(test_tool_result_get_reads_granted_parent_handle). These tests prove the keystone is reachable
through the LIVE delegation seam — AgentTool's grant_handles param validates + forwards, the
make_explore_agent_runner/_run_agent path builds a child whose ContentStore reads through ONLY the
granted parent handles, and the structural grant-target-safety invariant fails closed for a
host-dangerous target. Fully deterministic (no live model)."""
from __future__ import annotations

import pytest

import localharness.agent.subagent as subagent
from localharness.agent.context import ContentStore
from localharness.config.models import AgentConfig, ToolConfig
from localharness.tools.builtin import register_builtin_tools
from localharness.tools.builtin.agent_tool import AgentTool
from localharness.tools.capabilities import GrantTargetError
from localharness.tools.registry import ToolRegistry

_CLEAN = ToolConfig(deny=["web_search", "web_fetch", "web_page_query"])  # host-acting agent under the P-A floor


@pytest.mark.asyncio
async def test_agent_tool_validates_and_forwards_grant_handles():
    """grant_handles is in the schema, survives argument validation, and reaches the runner — the
    full AgentTool→runner channel. Omitting it forwards None (back-compat)."""
    reg = ToolRegistry()
    await register_builtin_tools(reg)
    seen: dict = {}

    async def _spy_runner(agent_id: str, task: str, grant_handles=None) -> str:
        seen["call"] = (agent_id, task, grant_handles)
        return "ok"

    await reg.register(AgentTool(agent_runner=_spy_runner, available_agents=["cruncher"]), scope="global")

    res = await reg.dispatch(
        "agent",
        {"agent_id": "cruncher", "task": "distill", "grant_handles": ["H123", "pg-1"]},
        agent_id="default", division_id="default", tool_config=_CLEAN,
    )
    assert res.success, res.error
    assert seen["call"] == ("cruncher", "distill", ["H123", "pg-1"])

    await reg.dispatch(
        "agent", {"agent_id": "explore", "task": "look"},
        agent_id="default", division_id="default", tool_config=_CLEAN,
    )
    assert seen["call"][2] is None  # optional → None, not a crash


@pytest.mark.asyncio
async def test_runner_builds_granted_readthrough_store(monkeypatch):
    """The live seam: delegating with grant_handles builds the child's ContextManager over a
    ContentStore(parent, granted={H}) that resolves the GRANTED parent handle by read-through and
    keeps an UNGRANTED handle invisible. Spies the final dispatch to inspect the constructed store
    (same style as test_runner_routes_*), so it's reachable-path, not a hand-built store."""
    parent = ContentStore()
    granted_h = parent.put("THE GRANTED OVER-WINDOW BODY")
    secret_h = parent.put("UNGRANTED SECRET BODY")

    captured: dict = {}

    async def _spy_config(task, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(subagent, "dispatch_config_subagent", _spy_config)

    runner = subagent.make_explore_agent_runner(
        llm=object(), bus=object(), base_registry=object(),
        permission_evaluator=object(), get_parent_session_id=lambda: "sid",
        load_agent=lambda n: AgentConfig(
            name="doc-reader", role="reads granted handles", tools=ToolConfig(add=["tool_result_get"]),
        ),
        parent_store=parent,
    )

    await runner("doc-reader", "read the granted handle", grant_handles=[granted_h])

    store = captured["context_manager"]._content_store
    assert store.get(granted_h) == "THE GRANTED OVER-WINDOW BODY"  # read-through capability
    assert store.origin(granted_h) == "trusted"
    assert store.get(secret_h) is None  # ungranted handle stays invisible — capability, not ambient


@pytest.mark.asyncio
async def test_no_grant_means_no_parent_store(monkeypatch):
    """Without grant_handles the child gets a FRESH isolated store (parent=None) — no ambient
    cross-agent read. Guards against accidentally making every child read the parent."""
    parent = ContentStore()
    h = parent.put("parent body the child was NOT granted")
    captured: dict = {}

    async def _spy_config(task, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(subagent, "dispatch_config_subagent", _spy_config)
    runner = subagent.make_explore_agent_runner(
        llm=object(), bus=object(), base_registry=object(),
        permission_evaluator=object(), get_parent_session_id=lambda: "sid",
        load_agent=lambda n: AgentConfig(name="leaf", role="r", tools=ToolConfig(add=["tool_result_get"])),
        parent_store=parent,
    )
    await runner("leaf", "no grant")  # grant_handles defaults None
    store = captured["context_manager"]._content_store
    assert store.get(h) is None


@pytest.mark.asyncio
async def test_grant_to_host_dangerous_target_is_refused(monkeypatch):
    """The structural invariant: a granted handle is readable via tool_result_get/chunk (NOT
    untrusted-ingest), so granting to a host-dangerous target would put attacker-controllable bytes
    one call from a host action. Refuse it — FAIL CLOSED before dispatch. data-analyst holds
    bash_exec; the grant must raise GrantTargetError and the dispatch must NEVER run."""
    async def _must_not_run(task, **kwargs):  # pragma: no cover - asserted never reached
        raise AssertionError("dispatch must not run when the grant target is host-dangerous")

    monkeypatch.setattr(subagent, "dispatch_data_subagent", _must_not_run)
    runner = subagent.make_explore_agent_runner(
        llm=object(), bus=object(), base_registry=object(),
        permission_evaluator=object(), get_parent_session_id=lambda: "sid",
        parent_store=ContentStore(),
    )
    with pytest.raises(GrantTargetError):
        await runner("data-analyst", "here is a big doc", grant_handles=["H"])


@pytest.mark.asyncio
async def test_grant_to_no_danger_target_passes_the_gate(monkeypatch):
    """A grant to a no-host-dangerous target (explore: read/glob/grep) passes the safety gate and
    proceeds to dispatch — the invariant blocks only host-dangerous grantees, not all grants."""
    captured: dict = {}

    async def _spy_explore(task, **kwargs):
        captured.update(kwargs)
        return "explored"

    monkeypatch.setattr(subagent, "dispatch_explore_subagent", _spy_explore)
    runner = subagent.make_explore_agent_runner(
        llm=object(), bus=object(), base_registry=object(),
        permission_evaluator=object(), get_parent_session_id=lambda: "sid",
        parent_store=ContentStore(),
    )
    out = await runner("explore", "read this", grant_handles=["H"])
    assert out == "explored" and captured  # gate allowed it through


def test_resolve_target_toolset_flags_builtin_danger():
    """The grant-safety resolver sees host-dangerous builtins (data-analyst/frontend hold bash/
    write/edit) and clean ones (explore), so the gate can decide. Config children resolve via
    their yaml allowlist."""
    from localharness.tools.capabilities import HOST_DANGEROUS

    danger = set(subagent._resolve_target_toolset("data-analyst", None))
    assert danger & HOST_DANGEROUS, "data-analyst must surface its host-dangerous tools to the gate"
    assert not (set(subagent._resolve_target_toolset("explore", None)) & HOST_DANGEROUS)

    cfg = AgentConfig(name="danger-cfg", role="r", tools=ToolConfig(add=["bash_exec", "read"]))
    assert "bash_exec" in subagent._resolve_target_toolset("danger-cfg", lambda n: cfg)
