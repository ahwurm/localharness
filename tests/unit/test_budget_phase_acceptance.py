"""ROADMAP Phase 8 success criteria, driven across module boundaries.

Every earlier plan of the phase proved its own piece alone. These tests wire the pieces together
the way a running install does: real config layers on disk (a global dir and a workspace
`.localharness/`) read by a real `ConfigLoader`, the loaded `AgentConfig` handed to the real
dispatch seam (`subagent._child_budget`), whose three outputs build a real `AgentLoop` on a real
`PermissionGate`, with a fake model making distinct tool calls. Nothing here is mocked between the
file on disk and the limit the loop enforces, on either gate surface (an attended terminal and a
remote unattended channel).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from localharness.agent import subagent as sub
from localharness.agent.context import ContextManager
from localharness.agent.loop import AgentLoop
from localharness.agent.permissions import PermissionEvaluator
from localharness.config.loader import ConfigLoader
from localharness.config.models import AgentConfig, BudgetConfig, PermissionConfig
from localharness.core.events import TurnFailed

_PROVIDER = {"provider_type": "vllm", "base_url": "http://localhost:8000/v1",
             "default_model": "global-model"}

ONLY_LOWER = "a project may only lower it"
KNOB_KEY = "channels.unattended_max_actions"
RUNAWAY = "Stopping to prevent runaway execution."


# --------------------------------------------------------------------------- the layers on disk


def _put(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data), encoding="utf-8")


def _agent(name: str, **budget) -> dict:
    data = {"name": name, "role": "Do the work."}
    if budget:
        data["permissions"] = {"budget": dict(budget)}
    return data


def _world(tmp_path: Path, *, g_cfg=None, ws_cfg=None, g_agents=None, ws_agents=None):
    """A global dir and a workspace `.localharness/` dir, each with the files given.

    `g_agents` / `ws_agents` map a file stem to the agent's yaml data."""
    g, ws = tmp_path / "global", tmp_path / "proj" / ".localharness"
    ws.mkdir(parents=True)
    _put(g / "config.yaml", {"version": "1", "provider": _PROVIDER, **(g_cfg or {})})
    if ws_cfg is not None:
        _put(ws / "config.yaml", ws_cfg)
    for base, agents in ((g, g_agents), (ws, ws_agents)):
        for stem, data in (agents or {}).items():
            _put(base / "agents" / f"{stem}.yaml", data)
    return ConfigLoader(config_dir=g, local_config_dir=ws), ws


def _only_lower_note(path: Path, field: str) -> str:
    return f"ignoring permissions.budget.{field} in {str(path)!r}: {ONLY_LOWER}"


# --------------------------------------------------------------------------- gate, loop, model


def _channel(*, local: bool) -> SimpleNamespace:
    """The few attributes a gate reads off a channel (it asks nothing: `can_ask` is False)."""
    return SimpleNamespace(channel_id="terminal" if local else "phone", local_operator=local,
                           can_ask=False)


class _Registry:
    """Runs every call; the output differs per call so nothing reads as stuck."""

    def __init__(self):
        self.calls: list[dict] = []

    def get_tools_for_agent(self, agent_id, division_id, tool_config):
        return {}

    async def dispatch(self, name, arguments, agent_id, division_id, tool_config, **kw):
        from localharness.tools.base import ToolResult

        self.calls.append(dict(arguments))
        return ToolResult(output=f"ran {arguments}", success=True)


def _distinct_calls(mock_llm_client, n: int) -> list:
    """`n` iterations of one distinct tool call each (different arguments), then a plain answer."""
    Response, ToolCall = mock_llm_client.Response, mock_llm_client.ToolCall
    out = [Response(content=None, tool_calls=[
        ToolCall(id=f"tc-{i}", name="bash", arguments={"cmd": f"echo {i}"})]) for i in range(n)]
    out.append(Response(content="all done"))
    return out


def _gate_loop(mock_llm_client, bus, cfg: AgentConfig, *, local: bool, responses=None,
               knob: int | None = None, **kw) -> tuple[AgentLoop, _Registry]:
    """A root loop on a real gate in `unattended` mode with a terminal or a remote channel attached."""
    cfg = cfg.model_copy(deep=True)
    cfg.permissions.mode = "unattended"
    cfg.permissions.deny_patterns = []
    registry = _Registry()
    loop = AgentLoop(
        config=cfg, llm=mock_llm_client(responses or []), bus=bus, context_manager=ContextManager(),
        tool_registry=registry, permission_evaluator=PermissionEvaluator(),
        unattended_max_actions=knob, **kw)
    loop.gate.attach_channel(_channel(local=local))
    return loop, registry


def _parent_gate(mock_llm_client, bus, *, local: bool):
    """The parent's real PermissionGate in `unattended` mode, with a terminal or a remote channel."""
    parent, _ = _gate_loop(mock_llm_client, bus,
                           AgentConfig(name="root", role="root",
                                       permissions=PermissionConfig(budget=BudgetConfig())),
                           local=local)
    return parent.gate


def _seam_child(mock_llm_client, bus, cfg: AgentConfig, name: str, *, local: bool, knob: int,
                children_budget=None, parent=(0, 0.0), responses=None) -> tuple[AgentLoop, _Registry]:
    """A child built the way `dispatch_config_subagent` builds it: the seam's three outputs, the
    parent's gate, the machine knob."""
    out, fallback, labels = sub._child_budget(
        cfg, name, children_budget=children_budget, shape=None, parent_remaining=lambda: parent)
    registry = _Registry()
    loop = AgentLoop(
        config=out, llm=mock_llm_client(responses or []), bus=bus, context_manager=ContextManager(),
        tool_registry=registry, permission_evaluator=PermissionEvaluator(),
        gate=_parent_gate(mock_llm_client, bus, local=local),
        budget_fallback=fallback, budget_labels=labels, unattended_max_actions=knob)
    return loop, registry


def _marks(cfg: AgentConfig) -> frozenset:
    return cfg.budget_kept_from_workspace


# =========================================================================== criteria 2 and 4


def test_a_workspace_only_agent_runs_its_numbers_attended_and_the_org_rung_remotely(
        tmp_path, mock_llm_client, bus):
    """Criterion 4: a workspace-only agent at 16 / 12 under a silent global loads with no warning
    and runs 16 / 12.0 under an attended terminal gate, min(16, 8) / min(12, 3) = 8 / 3.0 under a
    remote unattended gate (the seam's org rung caps what the workspace kept)."""
    loader, _ = _world(tmp_path, ws_agents={"analyst": _agent(
        "analyst", max_actions=16, max_duration_minutes=12)})
    harness = loader.load_harness()
    cfg = loader.load_agent("analyst")

    assert loader.agent_warnings == []
    assert _marks(cfg) == frozenset({"max_actions", "max_duration_minutes"})
    knob = harness.channels.unattended_max_actions
    assert knob == 100

    attended, _ = _seam_child(mock_llm_client, bus, cfg, "analyst", local=True, knob=knob,
                              children_budget=harness.org.children_budget)
    assert (attended.action_limit(), attended.time_limit()) == (16, 12.0)

    remote, _ = _seam_child(mock_llm_client, bus, cfg, "analyst", local=False, knob=knob,
                            children_budget=harness.org.children_budget)
    assert (remote.action_limit(), remote.time_limit()) == (8, 3.0)


def test_a_global_agent_file_ceiling_beats_the_workspace_and_warns(tmp_path, mock_llm_client, bus):
    """Criterion 4: the same agent under a global agents/analyst.yaml max_actions 10 loads 10 with
    the warning, and the loop enforces 10; the untouched minutes keep the workspace's 12."""
    loader, ws = _world(
        tmp_path,
        g_agents={"analyst": _agent("analyst", max_actions=10)},
        ws_agents={"analyst": _agent("analyst", max_actions=16, max_duration_minutes=12)})
    harness = loader.load_harness()
    cfg = loader.load_agent("analyst")

    assert cfg.permissions.budget.max_actions == 10
    assert loader.agent_warnings == [_only_lower_note(ws / "agents" / "analyst.yaml", "max_actions")]
    assert _marks(cfg) == frozenset({"max_duration_minutes"})

    attended, _ = _seam_child(mock_llm_client, bus, cfg, "analyst", local=True,
                              knob=harness.channels.unattended_max_actions,
                              children_budget=harness.org.children_budget)
    assert (attended.action_limit(), attended.time_limit()) == (10, 12.0)


def test_a_workspace_root_runs_150_attended_and_the_knob_remotely(tmp_path, mock_llm_client, bus):
    """Criterion 2 / 4: a workspace root agents/orchestrator.yaml at 150 under a silent global runs
    150 attended and 100 (the machine knob) on a remote unattended gate."""
    loader, _ = _world(tmp_path, ws_agents={"orchestrator": _agent("orchestrator", max_actions=150)})
    knob = loader.load_harness().channels.unattended_max_actions
    cfg = loader.load_agent("orchestrator")

    assert cfg.permissions.budget.max_actions == 150
    assert loader.agent_warnings == []

    attended, _ = _gate_loop(mock_llm_client, bus, cfg, local=True, knob=knob)
    assert attended.action_limit() == 150
    remote, _ = _gate_loop(mock_llm_client, bus, cfg, local=False, knob=knob)
    assert remote.action_limit() == 100


async def test_the_remote_root_stops_at_the_global_knob_and_names_it(tmp_path, mock_llm_client, bus):
    """Criterion 2: with the global knob at 2, the workspace root's 150 never applies remotely; the
    third distinct tool call is never run, and the stop text (summary and TurnFailed.detail) names
    channels.unattended_max_actions."""
    loader, _ = _world(
        tmp_path,
        g_cfg={"channels": {"unattended_max_actions": 2}},
        ws_agents={"orchestrator": _agent("orchestrator", max_actions=150)})
    knob = loader.load_harness().channels.unattended_max_actions
    assert knob == 2
    cfg = loader.load_agent("orchestrator")

    loop, registry = _gate_loop(mock_llm_client, bus, cfg, local=False, knob=knob,
                                responses=_distinct_calls(mock_llm_client, 3))
    summary = await loop.run_turn("go")

    assert len(registry.calls) == 2
    failed = [e for e in bus.history(event_types=[TurnFailed]) if e.reason == "budget_exceeded"]
    assert len(failed) == 1
    expected = f"{RUNAWAY} ({KNOB_KEY})"
    assert expected in summary
    assert expected in failed[0].detail


def test_a_workspace_root_cannot_climb_over_a_global_fifty(tmp_path, mock_llm_client, bus):
    """Criterion 4: a workspace orchestrator max_actions 100 under a global agent file at 50 loads
    50 with the warning, and an attended loop enforces 50."""
    loader, ws = _world(
        tmp_path,
        g_agents={"orchestrator": _agent("orchestrator", max_actions=50)},
        ws_agents={"orchestrator": _agent("orchestrator", max_actions=100)})
    cfg = loader.load_agent("orchestrator")

    assert cfg.permissions.budget.max_actions == 50
    assert loader.agent_warnings == [
        _only_lower_note(ws / "agents" / "orchestrator.yaml", "max_actions")]
    attended, _ = _gate_loop(mock_llm_client, bus, cfg, local=True)
    assert attended.action_limit() == 50


# =========================================================================== criteria 3 and 2


async def test_an_unconfigured_child_stops_at_the_org_rung_remotely_and_names_it(
        tmp_path, mock_llm_client, bus):
    """Criterion 3 / 2: an unconfigured config child dispatched under an infinite parent on a remote
    unattended gate stops at 8 actions with a budget-limit summary naming
    org.children_budget.default; the same child on an attended terminal has no limit."""
    loader, _ = _world(tmp_path, ws_agents={"coder": _agent("coder")})
    harness = loader.load_harness()
    cfg = loader.load_agent("coder")
    knob = harness.channels.unattended_max_actions
    assert knob == 100
    assert _marks(cfg) == frozenset()

    attended, _ = _seam_child(mock_llm_client, bus, cfg, "coder", local=True, knob=knob,
                              children_budget=harness.org.children_budget)
    assert (attended.action_limit(), attended.time_limit()) == (0, 0.0)

    loop, registry = _seam_child(
        mock_llm_client, bus, cfg, "coder", local=False, knob=knob,
        children_budget=harness.org.children_budget, parent=(0, 0.0),
        responses=_distinct_calls(mock_llm_client, 9))
    assert loop.action_limit() == 8
    summary = await loop.run_turn("go")

    assert len(registry.calls) == 8
    failed = [e for e in bus.history(event_types=[TurnFailed]) if e.reason == "budget_exceeded"]
    assert len(failed) == 1
    expected = f"{RUNAWAY} (org.children_budget.default)"
    assert expected in summary
    assert expected in failed[0].detail


# =========================================================================== criterion 4: built-ins


def _explore_overlay(loader: ConfigLoader):
    harness = loader.load_harness()
    base = sub.build_explore_config(budget=harness.org.children_budget.for_name("explore"))
    return loader.overlay_builtin_config("explore", base).permissions.budget


def test_a_workspace_overlay_of_a_built_in_is_capped_by_the_operators_file(tmp_path):
    """Criterion 4: a global agents/explore.yaml 20 with a workspace agents/explore.yaml 25 yields
    20 for the built-in; the workspace at 15 yields 15."""
    loader, _ = _world(
        tmp_path,
        g_agents={"explore": {"permissions": {"budget": {"max_actions": 20}}}},
        ws_agents={"explore": {"permissions": {"budget": {"max_actions": 25}}}})
    assert _explore_overlay(loader).max_actions == 20

    loader, _ = _world(
        tmp_path / "lower",
        g_agents={"explore": {"permissions": {"budget": {"max_actions": 20}}}},
        ws_agents={"explore": {"permissions": {"budget": {"max_actions": 15}}}})
    assert _explore_overlay(loader).max_actions == 15


def test_a_workspace_config_cannot_move_the_unattended_knob(tmp_path):
    """Criterion 4: a workspace channels.unattended_max_actions is dropped with the warning."""
    loader, ws = _world(tmp_path, ws_cfg={"channels": {"unattended_max_actions": 5000}})
    assert loader.load_harness().channels.unattended_max_actions == 100
    assert loader.harness_warnings == [
        f"ignoring {KNOB_KEY} in {ws / 'config.yaml'}: only the global config may set it"]


def test_one_key_of_a_children_budget_entry_reaches_the_built_in_config(tmp_path):
    """D-69 / criterion 4: a workspace config.yaml writing one key of an entry (explore.actions 4)
    under a silent global loads with no warning as explore 4 / 3.0, and the built-in explore config
    built from that block carries 4 / 3.0."""
    loader, _ = _world(tmp_path, ws_cfg={"org": {"children_budget": {"explore": {"actions": 4}}}})
    harness = loader.load_harness()
    entry = harness.org.children_budget.explore

    assert loader.harness_warnings == []
    assert (entry.actions, entry.minutes) == (4, 3.0)
    budget = sub.build_explore_config(
        budget=harness.org.children_budget.for_name("explore")).permissions.budget
    assert (budget.max_actions, budget.max_duration_minutes) == (4, 3.0)


def test_a_higher_children_budget_value_is_put_back_with_the_warning(tmp_path):
    """D-69 / criterion 4: the same file with explore.actions 99 loads 8 / 3.0 and says why."""
    loader, ws = _world(tmp_path, ws_cfg={"org": {"children_budget": {"explore": {"actions": 99}}}})
    harness = loader.load_harness()
    entry = harness.org.children_budget.explore

    assert (entry.actions, entry.minutes) == (8, 3.0)
    assert loader.harness_warnings == [
        f"ignoring org.children_budget.explore.actions in {ws / 'config.yaml'}: {ONLY_LOWER}"]
    budget = sub.build_explore_config(
        budget=harness.org.children_budget.for_name("explore")).permissions.budget
    assert (budget.max_actions, budget.max_duration_minutes) == (8, 3.0)
