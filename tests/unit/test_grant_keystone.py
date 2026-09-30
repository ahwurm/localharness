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
from localharness.tools.capabilities import GrantTargetError, is_host_dangerous
from localharness.tools.registry import ToolRegistry
from tests.unit.test_capabilities import _Bare, _builtin_registry, _production_registry

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
        llm=object(), bus=object(), base_registry=await _production_registry(),
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
    one call from a host action. Refuse it — FAIL CLOSED before dispatch. The fixture is a config
    child holding bash_exec (since v0.5.3 no default builtin is host-dangerous); the grant must
    raise GrantTargetError and the dispatch must NEVER run."""
    async def _must_not_run(task, **kwargs):  # pragma: no cover - asserted never reached
        raise AssertionError("dispatch must not run when the grant target is host-dangerous")

    monkeypatch.setattr(subagent, "dispatch_config_subagent", _must_not_run)
    bash_child = AgentConfig(name="bash-child", role="r", tools=ToolConfig(add=["bash_exec", "read"]))
    runner = subagent.make_explore_agent_runner(
        llm=object(), bus=object(), base_registry=await _production_registry(),
        permission_evaluator=object(), get_parent_session_id=lambda: "sid",
        load_agent=lambda n: bash_child,
        parent_store=ContentStore(),
    )
    with pytest.raises(GrantTargetError):
        await runner("bash-child", "here is a big doc", grant_handles=["H"])


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
        llm=object(), bus=object(), base_registry=await _production_registry(),
        permission_evaluator=object(), get_parent_session_id=lambda: "sid",
        parent_store=ContentStore(),
    )
    out = await runner("explore", "read this", grant_handles=["H"])
    assert out == "explored" and captured  # gate allowed it through


async def test_resolve_target_toolset_flags_builtin_danger():
    """The grant-safety resolver sees clean builtins (explore) and host-dangerous config
    children (yaml allowlist), so the gate can decide — by each name's DECLARATION."""
    reg = await _builtin_registry()
    explore = [reg.schema_of(n) for n in subagent._resolve_target_toolset("explore", None)]
    assert explore and None not in explore
    assert not any(is_host_dangerous(s) for s in explore)

    cfg = AgentConfig(name="danger-cfg", role="r", tools=ToolConfig(add=["bash_exec", "read"]))
    assert "bash_exec" in subagent._resolve_target_toolset("danger-cfg", lambda n: cfg)
    assert is_host_dangerous(reg.schema_of("bash_exec"))


@pytest.mark.parametrize("entry", ["plugin:research_tools.exa_search", "exa_search"])
async def test_a_grant_to_an_undeclared_plugin_tool_holder_is_refused(monkeypatch, entry):
    """A config child may name a tool in the `plugin:PLUGIN.TOOL` form, which from_allowed resolves
    to the bare TOOL. The grant check resolves it the SAME way and reads the declaration: a tool
    that declares nothing is host-dangerous by default, so the grant is refused — the prefix is not
    a way around the check, and neither is a name the old name set never listed."""
    async def _must_not_run(task, **kwargs):  # pragma: no cover - asserted never reached
        raise AssertionError("dispatch must not run when the grant target is host-dangerous")

    monkeypatch.setattr(subagent, "dispatch_config_subagent", _must_not_run)
    reg = await _production_registry()
    await reg.register(_Bare("exa_search"), scope="global")
    child = AgentConfig(name="plug-child", role="r", tools=ToolConfig(add=[entry]))
    runner = subagent.make_explore_agent_runner(
        llm=object(), bus=object(), base_registry=reg,
        permission_evaluator=object(), get_parent_session_id=lambda: "sid",
        load_agent=lambda n: child,
        parent_store=ContentStore(),
    )
    with pytest.raises(GrantTargetError, match="exa_search"):
        await runner("plug-child", "here is a big doc", grant_handles=["H"])
