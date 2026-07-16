"""Built-in subagents are TUNABLE via an optional agents/<name>.yaml overlay (BUILD 1).

The web-researcher (and the other single-loop builtins) ship a code-defined base config —
role + budget + toolset. Before this, that budget was HARDCODED and a user web-researcher.yaml
was silently never read (the dispatcher string-matched the builtin name before the load_agent
path ever ran). This makes the budget a REAL config knob: agents/<name>.yaml deep-merges ON TOP
of the built-in base (absent file = pure defaults; malformed = explicit error, never a silent
fallback). The structural toolset stays fixed by the dispatcher — only AgentConfig fields overlay.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import localharness.agent.subagent as subagent
from localharness.agent.permissions import PermissionEvaluator
from localharness.agent.subagent import build_web_researcher_config
from localharness.config.loader import (
    ConfigLoader,
    ConfigParseError,
    ConfigValidationError,
)


def _loader(tmp_path: Path) -> ConfigLoader:
    # config_dir = tmp_path (agents/<name>.yaml resolves under it); local_dir points at an
    # absent dir so the cwd's real .localharness never leaks into the test.
    return ConfigLoader(config_dir=tmp_path, local_config_dir=tmp_path / "no-local")


def _write_agent_yaml(tmp_path: Path, name: str, body: str) -> None:
    d = tmp_path / "agents"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.yaml").write_text(body, encoding="utf-8")


async def _builtin_registry():
    from localharness.tools.builtin import register_builtin_tools
    from localharness.tools.registry import ToolRegistry
    reg = ToolRegistry()
    await register_builtin_tools(reg)
    return reg


# --- loader-level overlay behaviour --------------------------------------------------------

def test_overlay_absent_yaml_returns_base_unchanged(tmp_path):
    base = build_web_researcher_config("web-researcher")
    out = _loader(tmp_path).overlay_builtin_config("web-researcher", base)
    # No behavior change when no yaml exists — the SAME base, pure built-in defaults.
    assert out is base
    assert out.permissions.budget.max_actions == 56


def test_overlay_yaml_overrides_budget_only(tmp_path):
    _write_agent_yaml(
        tmp_path, "web-researcher",
        "permissions:\n  budget:\n    max_actions: 40\n    max_duration_minutes: 12.0\n",
    )
    base = build_web_researcher_config("web-researcher")
    out = _loader(tmp_path).overlay_builtin_config("web-researcher", base)
    assert out is not base
    assert out.permissions.budget.max_actions == 40            # yaml wins
    assert out.permissions.budget.max_duration_minutes == 12.0
    # Fields the yaml did NOT set keep the built-in base — the overlay is per-field, not a replace.
    # A budget-only yaml (no name/role) still validates *because the base supplies them* — proof
    # this is overlay-onto-base, not a standalone AgentConfig validate (which would need name+role).
    assert out.name == "web-researcher"
    assert out.role == base.role                               # the RESEARCH_RIGOR-assembled role kept


def test_overlay_bad_value_raises_not_silent_fallback(tmp_path):
    # max_actions: 0 violates BudgetConfig ge=1 — must raise, never silently return the base.
    _write_agent_yaml(tmp_path, "web-researcher", "permissions:\n  budget:\n    max_actions: 0\n")
    base = build_web_researcher_config("web-researcher")
    with pytest.raises(ConfigValidationError):
        _loader(tmp_path).overlay_builtin_config("web-researcher", base)


def test_overlay_unknown_key_raises(tmp_path):
    # extra="forbid": a typo'd key is a hard error, not a silent no-op that drops the override.
    _write_agent_yaml(tmp_path, "web-researcher", "budgett: 40\n")
    base = build_web_researcher_config("web-researcher")
    with pytest.raises(ConfigValidationError):
        _loader(tmp_path).overlay_builtin_config("web-researcher", base)


def test_overlay_malformed_yaml_raises(tmp_path):
    _write_agent_yaml(tmp_path, "web-researcher", "permissions: [unclosed\n")
    base = build_web_researcher_config("web-researcher")
    with pytest.raises(ConfigParseError):
        _loader(tmp_path).overlay_builtin_config("web-researcher", base)


# --- runner wiring: the override reaches the dispatch; customs are untouched ----------------

@pytest.mark.asyncio
async def test_runner_passes_override_config_to_web_dispatch(monkeypatch):
    """The runner resolves the builtin override and hands it to the dispatch as config_override."""
    captured = {}
    async def _fake_web(task, **kwargs):
        captured["config_override"] = kwargs.get("config_override")
        return "ok"
    monkeypatch.setattr(subagent, "dispatch_web_subagent", _fake_web)

    overridden = build_web_researcher_config("web-researcher")
    overridden.permissions.budget.max_actions = 40

    runner = subagent.make_explore_agent_runner(
        llm=object(), bus=object(), base_registry=object(),
        permission_evaluator=object(), get_parent_session_id=lambda: "sid",
        load_builtin_override=lambda name, base: overridden if name == "web-researcher" else base,
    )
    await runner("web-researcher", "research X")
    assert captured["config_override"] is overridden
    assert captured["config_override"].permissions.budget.max_actions == 40


@pytest.mark.asyncio
async def test_runner_without_override_hook_passes_none(monkeypatch):
    """No load_builtin_override wired (bench / old callers) -> config_override is None so the
    dispatch builds its own default config: strictly no behavior change."""
    captured = {}
    async def _fake_web(task, **kwargs):
        captured["config_override"] = kwargs.get("config_override", "MISSING")
        return "ok"
    monkeypatch.setattr(subagent, "dispatch_web_subagent", _fake_web)
    runner = subagent.make_explore_agent_runner(
        llm=object(), bus=object(), base_registry=object(),
        permission_evaluator=object(), get_parent_session_id=lambda: "sid",
    )
    await runner("web-researcher", "research X")
    assert captured["config_override"] is None


@pytest.mark.asyncio
async def test_runner_custom_agent_unaffected_by_override_wiring(monkeypatch):
    """A non-builtin name still routes through load_agent -> dispatch_config_subagent unchanged —
    the override hook must never intercept a custom agent's own yaml path."""
    from localharness.config.models import AgentConfig
    seen = {}
    async def _fake_cfg_dispatch(task, **kwargs):
        seen["agent_config"] = kwargs.get("agent_config")
        return "ok"
    monkeypatch.setattr(subagent, "dispatch_config_subagent", _fake_cfg_dispatch)

    custom = AgentConfig(name="data-analyst", role="Analyze data.")
    runner = subagent.make_explore_agent_runner(
        llm=object(), bus=object(), base_registry=object(),
        permission_evaluator=object(), get_parent_session_id=lambda: "sid",
        load_agent=lambda n: custom,
        load_builtin_override=lambda n, base: base,  # present, but must not touch customs
    )
    await runner("data-analyst", "crunch the numbers")
    assert seen["agent_config"] is custom


@pytest.mark.asyncio
async def test_web_dispatch_uses_config_override_not_builder(monkeypatch, bus):
    """dispatch_web_subagent uses a passed config_override verbatim and does NOT rebuild the
    default config — proof the override actually drives the child loop, not just the wire."""
    base = await _builtin_registry()
    override = build_web_researcher_config("web-researcher")
    override.permissions.budget.max_actions = 41

    def _boom(*a, **k):
        raise AssertionError("build_web_researcher_config must not run when config_override is set")
    monkeypatch.setattr(subagent, "build_web_researcher_config", _boom)

    captured = {}
    class _FakeLoop:
        def __init__(self, **kwargs):
            captured["config"] = kwargs.get("config")
            self.current_session_id = "child-sid"
        async def run_turn(self, task):
            return "child summary"
    monkeypatch.setattr("localharness.agent.loop.AgentLoop", _FakeLoop)
    monkeypatch.setattr(subagent, "_count_session_tool_calls", lambda bus, sid: 0)

    out = await subagent.dispatch_web_subagent(
        "research X", llm=object(), bus=bus, base_registry=base,
        parent_session_id="p", permission_evaluator=PermissionEvaluator(),
        config_override=override,
    )
    assert captured["config"] is override
    assert captured["config"].permissions.budget.max_actions == 41
    assert "child summary" in out
