"""`compact_md_path=None` meant two different things, one of them wrong (F7/A-M6).

Bench passed None to say "this scenario has NO prior-session context"; AgentLoop read None as
"nothing was configured, fall back to a hardcoded ~/.localharness/agents/<name>/compact.md". So a
bench run inherited whatever the operator's own agent had last compacted — invisible cross-talk
into a measurement, and the one place a hardcoded home path survived phase 38's config-dir sweep.

COMPACT_DISABLED says the first thing explicitly; the fallback derives from the config dir.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from localharness.agent.context import COMPACT_DISABLED
from localharness.agent.loop import AgentLoop


def _loop(tmp_path, **kwargs) -> AgentLoop:
    """An AgentLoop with only what the compact.md resolution reads."""
    config = SimpleNamespace(name="a1", permissions=SimpleNamespace(budget=SimpleNamespace(kill_file=None)))
    return AgentLoop(
        config=config,
        llm=SimpleNamespace(),
        bus=SimpleNamespace(),
        context_manager=SimpleNamespace(),
        tool_registry=SimpleNamespace(),
        permission_evaluator=SimpleNamespace(),
        **kwargs,
    )


def _plant_home_compact(tmp_path, monkeypatch) -> Path:
    """A compact.md exactly where the old hardcoded default pointed."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    planted = tmp_path / "home" / ".localharness" / "agents" / "a1" / "compact.md"
    planted.parent.mkdir(parents=True)
    planted.write_text("# leaked prior session\n", encoding="utf-8")
    # The premise every test here rests on: this IS the path the old hardcoded default produced,
    # so "not this file" is a real answer and not a vacuous one.
    assert planted == Path.home() / ".localharness" / "agents" / "a1" / "compact.md"
    return planted


def test_disabled_reads_no_compact_md_at_all(tmp_path, monkeypatch):
    """The bench contract: a planted home compact.md is NOT inherited."""
    _plant_home_compact(tmp_path, monkeypatch)

    loop = _loop(tmp_path, compact_md_path=COMPACT_DISABLED)

    assert loop._resolve_compact_md_path() is None


def test_default_derives_from_the_config_dir_not_home(tmp_path, monkeypatch):
    """An isolated config dir keeps a session out of the operator's home compact.md."""
    _plant_home_compact(tmp_path, monkeypatch)
    isolated = tmp_path / "isolated"

    loop = _loop(tmp_path, config_dir=isolated)

    assert loop._resolve_compact_md_path() == isolated / "agents" / "a1" / "compact.md"


def test_default_honors_the_config_dir_env_chain(tmp_path, monkeypatch):
    """No explicit arg: the ONE resolver answers, so LOCALHARNESS_DIR is honored (#35)."""
    _plant_home_compact(tmp_path, monkeypatch)
    monkeypatch.setenv("LOCALHARNESS_DIR", str(tmp_path / "env-dir"))

    loop = _loop(tmp_path)

    assert loop._resolve_compact_md_path() == tmp_path / "env-dir" / "agents" / "a1" / "compact.md"


def test_an_explicit_path_still_wins(tmp_path):
    """Direct construction with a real path is unchanged."""
    explicit = tmp_path / "ws" / "agents" / "a1" / "compact.md"

    loop = _loop(tmp_path, compact_md_path=explicit)

    assert loop._resolve_compact_md_path() == explicit


def test_the_write_side_skips_the_sentinel(tmp_path):
    """The pipeline must never try to WRITE to the sentinel either."""
    from localharness.agent.context import SummaryCompactionStage, _writes_compact_md

    stage = SummaryCompactionStage(
        preserve_first_n=1, preserve_last_n=1, compact_md_path=COMPACT_DISABLED
    )

    assert stage.compact_md_path is COMPACT_DISABLED
    assert not _writes_compact_md(stage.compact_md_path)
    assert not _writes_compact_md(None)
    assert _writes_compact_md(Path("/tmp/x/compact.md"))


@pytest.mark.asyncio
async def test_bench_construction_reads_no_home_compact_md(tmp_path, monkeypatch):
    """The measurement this protects: a bench loop must not inherit the operator's compact.md."""
    from localharness.bench.runner import _build_agent_loop
    from localharness.bench.schema import BudgetSpec, LimitsSpec, ScenarioSpec, SuccessCriteria
    from localharness.config.models import AgentConfig
    from localharness.core.bus import EventBus
    from localharness.tools.registry import ToolRegistry

    _plant_home_compact(tmp_path, monkeypatch)  # plants under HOME for agent "a1"
    scenario = ScenarioSpec(
        name="compact-isolation",
        prompt="say ok",
        success_criteria=SuccessCriteria(golden_output="ok"),
        budget=BudgetSpec(),
        limits=LimitsSpec(),
        tools_allowed=[],
        slice="train",
        category="tool_basics",
    )
    loop = await _build_agent_loop(
        EventBus(),
        SimpleNamespace(),
        scenario,
        session_id="s1",
        agent_config=AgentConfig(name="a1", role="r"),
        base_registry=ToolRegistry(),
    )

    assert loop._resolve_compact_md_path() is None
