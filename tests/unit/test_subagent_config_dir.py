"""Subagent runtime paths resolve against the SESSION's config dir (LAYR-02, v013 Risk #6).

Before phase 38, a `--config-dir /foo` session rooted its ROOT agent at `/foo` while every one
of its subagents silently fell through to `AgentLoop`'s own fallbacks: compact.md under a
hardcoded `~/.localharness` (loop.py:781) and a bare `"KILL"` resolved against the PROCESS CWD
(loop.py:725). These tests pin the fix at two levels:

- Layer 1: `_child_runtime_paths` itself — including the `config_dir=None` case, which must stay
  BYTE-IDENTICAL to today's `~/.localharness` fallback (the zero-behavior-change invariant).
- Layer 2 (task 2): every one of the 6 `AgentLoop(...)` construction sites in subagent.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from localharness.agent.permissions import PermissionEvaluator
from localharness.agent.subagent import (
    _child_runtime_paths,
    _cruncher_combine_turn,
    _run_chunk_summarizer,
    build_explore_config,
    dispatch_config_subagent,
    dispatch_explore_subagent,
    dispatch_search_verifier_subagent,
    dispatch_web_subagent,
    make_explore_agent_runner,
)
from localharness.config.models import AgentConfig, BudgetConfig, PermissionConfig
from localharness.tools.builtin import register_builtin_tools
from localharness.tools.registry import ToolRegistry


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
def recorded_loops(monkeypatch):
    """Record the kwargs every AgentLoop construction receives; no-op the turn.

    The 6 sites import AgentLoop INSIDE their functions
    (``from localharness.agent.loop import AgentLoop``), so patching the attribute on the
    ``localharness.agent.loop`` MODULE is what takes effect — the first Layer-2 test asserts
    a recording actually happened, which is the proof the patch bites.
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


# ---------------------------------------------------------------------------
# Layer 1 — the helper
# ---------------------------------------------------------------------------

def test_explicit_config_dir_roots_both_paths(tmp_path):
    """`--config-dir <dir>` puts BOTH the kill file and compact.md under that dir."""
    kill, compact = _child_runtime_paths(_cfg("explore"), tmp_path)
    assert kill == tmp_path / "KILL"
    assert compact == tmp_path / "agents" / "explore" / "compact.md"


def test_default_reproduces_todays_home_fallback(monkeypatch):
    """config_dir=None with no env override == loop.py:781's hardcoded fallback, byte-identical.

    The autouse `_isolate_localharness_home` conftest fixture sets LOCALHARNESS_HOME for EVERY
    test, so this one must unset it (and LOCALHARNESS_DIR) to see the real default chain.
    """
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    cfg = build_explore_config("explore")

    kill, compact = _child_runtime_paths(cfg, None)

    assert compact == Path.home() / ".localharness" / "agents" / cfg.name / "compact.md"
    assert kill == Path.home() / ".localharness" / "KILL"


def test_env_override_roots_paths_under_localharness_home(tmp_path, monkeypatch):
    """LOCALHARNESS_HOME (legacy alias) and LOCALHARNESS_DIR (canonical) both move the children."""
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.setenv("LOCALHARNESS_HOME", str(tmp_path / "envhome"))
    kill, compact = _child_runtime_paths(_cfg("web-researcher"), None)
    assert kill == tmp_path / "envhome" / "KILL"
    assert compact == tmp_path / "envhome" / "agents" / "web-researcher" / "compact.md"

    monkeypatch.setenv("LOCALHARNESS_DIR", str(tmp_path / "canonical"))
    kill2, compact2 = _child_runtime_paths(_cfg("web-researcher"), None)
    assert kill2 == tmp_path / "canonical" / "KILL"
    assert compact2 == tmp_path / "canonical" / "agents" / "web-researcher" / "compact.md"


def test_absolute_kill_file_is_honored_not_rerooted(tmp_path):
    """resolve_runtime_path's standing contract: an absolute value is never re-rooted."""
    kill, compact = _child_runtime_paths(_cfg("explore", kill_file="/var/run/CUSTOMKILL"), tmp_path)
    assert kill == Path("/var/run/CUSTOMKILL")
    # ...while compact.md still follows the config dir.
    assert compact == tmp_path / "agents" / "explore" / "compact.md"


def test_no_path_resolves_against_the_process_cwd(tmp_path):
    """The CWD-relative bare "KILL" (loop.py:725) is exactly what this plan retires for children."""
    kill, compact = _child_runtime_paths(_cfg("cruncher"), tmp_path)
    cwd = str(Path.cwd())
    assert not str(kill).startswith(cwd)
    assert not str(compact).startswith(cwd)
    assert kill.is_absolute() and compact.is_absolute()


# ---------------------------------------------------------------------------
# Layer 2 — every one of the 6 AgentLoop construction sites
# ---------------------------------------------------------------------------

# Each site asserts its own concrete pair inline rather than through a shared helper: a reader
# should see exactly what a given construction site is expected to produce without a hop.

@pytest.mark.asyncio
async def test_config_child_loop_is_rooted_at_config_dir(recorded_loops, bus, tmp_path):
    """Site 1 (dispatch_config_subagent) — also the proof the AgentLoop patch takes effect."""
    base = await _builtin_registry()
    await dispatch_config_subagent(
        "look at the repo",
        agent_config=_cfg("yaml-child"),
        llm=object(),
        bus=bus,
        base_registry=base,
        parent_session_id="parent-sess",
        permission_evaluator=PermissionEvaluator(),
        config_dir=tmp_path,
    )
    assert len(recorded_loops) == 1, "AgentLoop patch did not take effect at the call site"
    assert recorded_loops[0]["compact_md_path"] == tmp_path / "agents" / "yaml-child" / "compact.md"
    assert recorded_loops[0]["kill_file_path"] == tmp_path / "KILL"


@pytest.mark.asyncio
async def test_explore_child_loop_is_rooted_at_config_dir(recorded_loops, bus, tmp_path):
    """Site 2 (dispatch_explore_subagent)."""
    base = await _builtin_registry()
    await dispatch_explore_subagent(
        "find X", llm=object(), bus=bus, base_registry=base,
        parent_session_id="parent-sess", permission_evaluator=PermissionEvaluator(),
        config_dir=tmp_path,
    )
    assert recorded_loops[0]["compact_md_path"] == tmp_path / "agents" / "explore" / "compact.md"
    assert recorded_loops[0]["kill_file_path"] == tmp_path / "KILL"


@pytest.mark.asyncio
async def test_web_child_loop_is_rooted_at_config_dir(recorded_loops, bus, tmp_path):
    """Site 3 (dispatch_web_subagent)."""
    base = await _builtin_registry()
    await dispatch_web_subagent(
        "research X", llm=object(), bus=bus, base_registry=base,
        parent_session_id="parent-sess", permission_evaluator=PermissionEvaluator(),
        config_dir=tmp_path,
    )
    assert recorded_loops[0]["compact_md_path"] == tmp_path / "agents" / "web-researcher" / "compact.md"
    assert recorded_loops[0]["kill_file_path"] == tmp_path / "KILL"


@pytest.mark.asyncio
async def test_search_verifier_child_loop_is_rooted_at_config_dir(
    recorded_loops, bus, tmp_path, monkeypatch
):
    """Site 4 (dispatch_search_verifier_subagent). The ledger write is redirected off the repo."""
    monkeypatch.setenv("LOCALHARNESS_VERIFICATION_LEDGER_DIR", str(tmp_path / "ledger"))
    base = await _builtin_registry()
    await dispatch_search_verifier_subagent(
        "claim: X\nentity: X\nsource_url: https://news.test/x",
        llm=object(), bus=bus, base_registry=base,
        parent_session_id="parent-sess", permission_evaluator=PermissionEvaluator(),
        config_dir=tmp_path,
    )
    assert recorded_loops[0]["compact_md_path"] == tmp_path / "agents" / "search-verifier" / "compact.md"
    assert recorded_loops[0]["kill_file_path"] == tmp_path / "KILL"


@pytest.mark.asyncio
async def test_chunk_summarizer_leaf_is_rooted_at_config_dir(recorded_loops, bus, tmp_path):
    """Site 5 (_run_chunk_summarizer) — the cruncher's map leaf, tested directly."""
    from localharness.agent.context import ContentStore

    base = await _builtin_registry()
    store = ContentStore()
    handle = store.put("a section of the granted document")
    await _run_chunk_summarizer(
        handle, "what does it say?", store, llm=object(), bus=bus, base_registry=base,
        parent_session_id="parent-sess", permission_evaluator=PermissionEvaluator(),
        token_counter=None, max_context_tokens=None, depth=1, max_subagent_depth=2,
        config_dir=tmp_path,
    )
    assert recorded_loops[0]["compact_md_path"] == tmp_path / "agents" / "chunk-summarizer" / "compact.md"
    assert recorded_loops[0]["kill_file_path"] == tmp_path / "KILL"


@pytest.mark.asyncio
async def test_cruncher_combine_turn_is_rooted_at_config_dir(recorded_loops, bus, tmp_path):
    """Site 6 (_cruncher_combine_turn) — the cruncher's reduce turn, tested directly."""
    from localharness.agent.context import ContextManager

    base = await _builtin_registry()
    await _cruncher_combine_turn(
        "what does it say?", ["[section 1]\nsome extract"], partial=False, llm=object(),
        child_bus=bus, base_registry=base, permission_evaluator=PermissionEvaluator(),
        ctx=ContextManager(), config_dir=tmp_path,
    )
    assert recorded_loops[0]["compact_md_path"] == tmp_path / "agents" / "cruncher" / "compact.md"
    assert recorded_loops[0]["kill_file_path"] == tmp_path / "KILL"


@pytest.mark.asyncio
async def test_runner_threads_config_dir_into_the_child(recorded_loops, bus, tmp_path):
    """The RUNNER threads it — not just the dispatch functions called by hand.

    This is the seam start_cmd/bench wire in plan 38-05; without it the config_dir parameters
    would be dead weight nothing reaches.
    """
    base = await _builtin_registry()
    runner = make_explore_agent_runner(
        llm=object(), bus=bus, base_registry=base, permission_evaluator=PermissionEvaluator(),
        get_parent_session_id=lambda: "parent-sess", config_dir=tmp_path,
    )
    await runner("explore", "find X")
    assert recorded_loops[0]["compact_md_path"] == tmp_path / "agents" / "explore" / "compact.md"
    assert recorded_loops[0]["kill_file_path"] == tmp_path / "KILL"


@pytest.mark.asyncio
async def test_runner_default_leaves_children_on_todays_paths(
    recorded_loops, bus, tmp_path, monkeypatch
):
    """Zero-behavior-change backstop: no config_dir anywhere == today's ~/.localharness compact.md.

    Every EXISTING caller is in exactly this state until plan 38-05 wires them, so this is the
    regression that would fire if the threading silently moved anyone's files.
    """
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    base = await _builtin_registry()
    runner = make_explore_agent_runner(
        llm=object(), bus=bus, base_registry=base, permission_evaluator=PermissionEvaluator(),
        get_parent_session_id=lambda: "parent-sess",
    )
    await runner("explore", "find X")
    assert recorded_loops[0]["compact_md_path"] == Path.home() / ".localharness" / "agents" / "explore" / "compact.md"
    assert recorded_loops[0]["kill_file_path"] == Path.home() / ".localharness" / "KILL"


# ---------------------------------------------------------------------------
# Plan 38-05 — the criterion-2 acceptance test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_criterion_2_dispatched_child_lands_under_the_session_config_dir(
    recorded_loops, bus, tmp_path, monkeypatch
):
    """ROADMAP Phase 38 criterion 2, end to end: a `--config-dir <D>` session's dispatched child
    resolves compact.md and its kill-file under <D>.

    Plan 38-02 made this *resolvable*; plan 38-05 made it *reachable* by passing
    `config_dir=cfg_path` at start_cmd.py:1122. The runner below is built the way start_cmd now
    builds it (same kwarg, same value shape), so this asserts a user-visible property of
    `localharness start --config-dir <D>` rather than a hand-built runner's behavior. The env
    overrides are unset on purpose: with them set, a leak would land in the hermetic fixture dir
    and pass; with them unset it would land in the real ~/.localharness and fail loudly.
    """
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    session_dir = tmp_path / "session"
    base = await _builtin_registry()

    runner = make_explore_agent_runner(
        llm=object(), bus=bus, base_registry=base, permission_evaluator=PermissionEvaluator(),
        get_parent_session_id=lambda: "parent-sess",
        depth=0,
        available_agents=["explore", "web-researcher", "cruncher", "search-verifier"],
        config_dir=session_dir,  # start_cmd.py: config_dir=cfg_path
    )
    await runner("explore", "find X")

    assert recorded_loops[0]["compact_md_path"] == session_dir / "agents" / "explore" / "compact.md"
    assert recorded_loops[0]["kill_file_path"] == session_dir / "KILL"
    # nothing escaped the session dir — not the home fallback (loop.py:781), not the CWD-relative
    # bare "KILL" (loop.py:725)
    assert recorded_loops[0]["compact_md_path"].is_relative_to(session_dir)
    assert recorded_loops[0]["kill_file_path"].is_relative_to(session_dir)
