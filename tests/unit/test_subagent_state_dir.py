"""A subagent's compact.md follows the work; its kill switch does NOT (phase 41, MEMS-01).

Two artifacts, two layers, one helper — and the whole point of this file is that the two layers
stay textually and behaviorally independent:

- The KILL file is a machine-global CONTROL artifact. It stops the one daemon's sessions, so it
  resolves against `config_dir` and can never be relocated into a project folder — a kill switch a
  workspace could move is not a kill switch.
- compact.md is a per-session WORK artifact. It follows `state_dir` — the discovered workspace
  layer when one applies, otherwise the same value as `config_dir`.

The bug this file exists to prevent: before the split, both paths were derived from ONE `base`
inside `_child_runtime_paths`, so widening that single parameter to
"the workspace when present" — the obvious way to make compact.md follow the work — would have
silently moved every subagent's kill switch into the user's repository. A test that only asserted
"subagent state is workspace-local" would have shipped that regression green. Test 2 and test 5
are the ones that cannot.

The tests here were written AFTER the code, so they are graded by mutation rather than by a
RED-first commit that never happened: each was checked to fail against a deliberately broken
`_child_runtime_paths` before it was trusted.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from localharness.agent.permissions import PermissionEvaluator
from localharness.agent.subagent import (
    _child_runtime_paths,
    dispatch_explore_subagent,
    make_explore_agent_runner,
)
from localharness.config.models import AgentConfig, BudgetConfig, PermissionConfig
from localharness.tools.builtin import register_builtin_tools
from localharness.tools.registry import ToolRegistry


# Copied (not imported) from tests/unit/test_subagent_config_dir.py, following 40-03's `_hermetic`
# precedent: this file should read as the whole of what it asserts.
def _cfg(name: str = "explore", kill_file: str | None = None) -> AgentConfig:
    return AgentConfig(
        name=name,
        role="test child",
        permissions=PermissionConfig(budget=BudgetConfig(kill_file=kill_file)),
    )


async def _builtin_registry() -> ToolRegistry:
    reg = ToolRegistry()
    await register_builtin_tools(reg)
    return reg


@pytest.fixture
def global_and_workspace(tmp_path) -> tuple[Path, Path]:
    """The two layers, as separate trees — a path under one can never be under the other.

    `G` stands in for ~/.localharness (the machine-global layer, where the daemon's KILL lives);
    `W` for a discovered `<project>/.localharness` workspace.
    """
    g = tmp_path / "global"
    w = tmp_path / "project" / ".localharness"
    g.mkdir(parents=True)
    w.mkdir(parents=True)
    return g, w


# ---------------------------------------------------------------------------
# The helper — the split itself
# ---------------------------------------------------------------------------

def test_compact_md_follows_the_workspace(global_and_workspace):
    """compact.md is a per-session WORK artifact: given a workspace state dir, it lands there."""
    g, w = global_and_workspace
    _kill, compact = _child_runtime_paths(_cfg("explore"), g, state_dir=w)
    assert compact == w / "agents" / "explore" / "compact.md"
    assert not compact.is_relative_to(g), "compact.md stayed on the global layer — state_dir was ignored"


def test_kill_file_stays_machine_global_when_compact_moves(global_and_workspace):
    """THE load-bearing one (41-RESEARCH Pitfall 1).

    The SAME call that moves compact.md into the workspace must leave the kill file on the global
    layer. A workspace-local KILL would mean a project folder could hold — or fail to hold — the
    switch that stops the machine's sessions.
    """
    g, w = global_and_workspace
    kill, compact = _child_runtime_paths(_cfg("explore"), g, state_dir=w)
    assert kill == g / "KILL", (
        "the kill file is a machine-global control artifact and must resolve against config_dir, "
        "never follow a workspace"
    )
    assert str(w) not in str(kill), (
        f"the workspace path {w} appears in the kill path {kill} — the machine-global kill switch "
        "was relocated into a project folder (Pitfall 1)"
    )
    # ...and the move that was asked for did happen, so this is not passing because nothing moved.
    assert compact == w / "agents" / "explore" / "compact.md"


def test_omitting_state_dir_is_a_no_op(global_and_workspace):
    """Every caller that has not been re-pointed yet (all of them until 41-05) must be unchanged."""
    g, _w = global_and_workspace
    two_arg = _child_runtime_paths(_cfg("explore"), g)
    explicit_none = _child_runtime_paths(_cfg("explore"), g, state_dir=None)
    assert two_arg == explicit_none
    kill, compact = two_arg
    assert kill == g / "KILL"
    assert compact == g / "agents" / "explore" / "compact.md"


def test_absolute_kill_file_passes_through_under_both_forms(global_and_workspace):
    """resolve_runtime_path's standing contract survives the split: an absolute value is never re-rooted.

    Checked with AND without a workspace, because the split introduced a second `resolve_config_dir`
    call and an absolute kill value must ignore both of them.
    """
    g, w = global_and_workspace
    cfg = _cfg("explore", kill_file="/var/run/CUSTOMKILL")

    kill_plain, compact_plain = _child_runtime_paths(cfg, g)
    kill_ws, compact_ws = _child_runtime_paths(cfg, g, state_dir=w)

    assert kill_plain == Path("/var/run/CUSTOMKILL")
    assert kill_ws == Path("/var/run/CUSTOMKILL")
    assert compact_plain == g / "agents" / "explore" / "compact.md"
    assert compact_ws == w / "agents" / "explore" / "compact.md"


# ---------------------------------------------------------------------------
# The threading — a real dispatch, one captured AgentLoop
# ---------------------------------------------------------------------------

@pytest.fixture
def recorded_loops(monkeypatch):
    """Record the kwargs every AgentLoop construction receives; no-op the turn.

    The dispatch functions import AgentLoop INSIDE their bodies, so patching the attribute on the
    `localharness.agent.loop` MODULE is what bites — the drive test asserts a recording actually
    happened, which is the proof of that.
    """
    calls: list[dict] = []

    class _RecordingLoop:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.current_session_id = None

        async def run_turn(self, task, *a, **k):
            return ""

    monkeypatch.setattr("localharness.agent.loop.AgentLoop", _RecordingLoop)
    return calls


async def test_dispatched_child_splits_its_two_paths(recorded_loops, bus, global_and_workspace):
    """The split is reachable through a real dispatch, not just by calling the helper by hand.

    Both assertions live in ONE test on ONE captured construction on purpose: checking only
    compact.md is exactly the blind spot Pitfall 1 names.
    """
    g, w = global_and_workspace
    base = await _builtin_registry()

    await dispatch_explore_subagent(
        "find X", llm=object(), bus=bus, base_registry=base,
        parent_session_id="parent-sess", permission_evaluator=PermissionEvaluator(),
        config_dir=g, state_dir=w,
    )

    assert len(recorded_loops) == 1, "AgentLoop patch did not take effect at the call site"
    assert recorded_loops[0]["compact_md_path"] == w / "agents" / "explore" / "compact.md"
    assert recorded_loops[0]["kill_file_path"] == g / "KILL", (
        "a dispatched child's kill file must stay on the machine-global layer even when its "
        "compact.md follows the workspace"
    )


async def test_runner_threads_state_dir_into_the_child(recorded_loops, bus, global_and_workspace):
    """The RUNNER carries it too — this is the seam 41-05 wires, not a hand-built dispatch.

    `start_cmd` builds children through `make_explore_agent_runner(...)` and plan 41-05 will pass
    `config_dir=cfg_path, state_dir=state_dir` there. If the runner's pass-through into the dispatch
    functions dropped `state_dir`, that wiring would be dead weight nothing reaches and the phase's
    headline criterion would silently fail — with every direct-helper test in this file still green.
    (Not in the plan's five; added because mutation (d) showed the pass-throughs were ungraded.)
    """
    g, w = global_and_workspace
    base = await _builtin_registry()

    runner = make_explore_agent_runner(
        llm=object(), bus=bus, base_registry=base, permission_evaluator=PermissionEvaluator(),
        get_parent_session_id=lambda: "parent-sess",
        config_dir=g,     # start_cmd: config_dir=cfg_path  (global — the kill file)
        state_dir=w,      # 41-05:     state_dir=state_dir  (workspace — compact.md)
    )
    await runner("explore", "find X")

    assert len(recorded_loops) == 1, "AgentLoop patch did not take effect at the call site"
    assert recorded_loops[0]["compact_md_path"] == w / "agents" / "explore" / "compact.md"
    assert recorded_loops[0]["kill_file_path"] == g / "KILL", (
        "the kill file stays machine-global through the runner seam too"
    )
