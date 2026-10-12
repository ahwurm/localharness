"""A project may only lower the window ratios, at every door.

`org.context.tool_output_ratio`, `evict_ratio` and `result_cap_ratio` are the share of the live
context window one tool result may take, the size above which a result is evicted, and what one
result may put into the conversation. A cloned repository's `.localharness/` may make them smaller
and never larger: a larger, zero or non-numeric value is dropped with a warning naming the key and
the file, a `null` is dropped quietly. The same rule holds on the harness view (the raw config
files), on an agent's own `context` (its file, its division, the workspace `org.context`) and on a
workspace override of a built-in agent.
"""
from __future__ import annotations

import math

import pytest

from localharness.config.models import ContextConfig
from tests.unit.test_floor_machine_level import _layers, _warnings

NAMES = ("tool_output_ratio", "evict_ratio", "result_cap_ratio")
DEFAULTS = {n: ContextConfig.model_fields[n].default for n in NAMES}
TOR = "tool_output_ratio"


def _ctx(**kw) -> dict:
    return {"org": {"context": kw}}


def _got(loader, name: str = TOR) -> float:
    return getattr(loader.load_harness().org.context, name)


def _note(name: str, file) -> str:
    return f"ignoring org.context.{name} in {file}: a project may only lower it"


# ---------------------------------------------------------------------------
# the harness view
# ---------------------------------------------------------------------------


def test_the_harness_lower_only_set_names_the_three_ratios():
    from localharness.config.loader import HARNESS_LOWER_ONLY_FIELDS
    assert {f"org.context.{n}" for n in NAMES} <= HARNESS_LOWER_ONLY_FIELDS


def test_a_workspace_may_lower_a_ratio_under_a_silent_global(tmp_path):
    loader, _ = _layers(tmp_path, ws_cfg=_ctx(tool_output_ratio=0.03))
    assert _got(loader) == 0.03
    assert _warnings(loader) == []


@pytest.mark.parametrize("bad", [0.2, 0, -0.1, math.nan, "lots", True, 1.5, [0.03]])
def test_a_higher_or_junk_workspace_ratio_gives_the_default_with_a_note(tmp_path, bad):
    loader, ws = _layers(tmp_path, ws_cfg=_ctx(tool_output_ratio=bad))
    assert _got(loader) == DEFAULTS[TOR]
    assert _warnings(loader) == [_note(TOR, ws / "config.yaml")]


def test_a_workspace_null_ratio_is_restored_silently(tmp_path):
    loader, _ = _layers(tmp_path, ws_cfg=_ctx(tool_output_ratio=None))
    assert _got(loader) == DEFAULTS[TOR]
    assert _warnings(loader) == []


def test_a_workspace_may_equal_the_default(tmp_path):
    loader, _ = _layers(tmp_path, ws_cfg=_ctx(tool_output_ratio=DEFAULTS[TOR]))
    assert _got(loader) == DEFAULTS[TOR]
    assert _warnings(loader) == []


def test_a_global_ratio_is_the_ceiling_for_the_workspace(tmp_path):
    loader, ws = _layers(tmp_path, g_cfg=_ctx(evict_ratio=0.01), ws_cfg=_ctx(evict_ratio=0.02))
    assert _got(loader, "evict_ratio") == 0.01
    assert _warnings(loader) == [_note("evict_ratio", ws / "config.yaml")]


def test_a_workspace_below_the_global_ratio_is_taken(tmp_path):
    loader, _ = _layers(tmp_path, g_cfg=_ctx(evict_ratio=0.01), ws_cfg=_ctx(evict_ratio=0.005))
    assert _got(loader, "evict_ratio") == 0.005
    assert _warnings(loader) == []


def test_a_workspace_overrides_file_is_judged_too(tmp_path):
    loader, ws = _layers(tmp_path, ws_over=_ctx(result_cap_ratio=0.5))
    assert _got(loader, "result_cap_ratio") == DEFAULTS["result_cap_ratio"]
    assert _warnings(loader) == [_note("result_cap_ratio", ws / "overrides.yaml")]


def test_a_refused_overrides_value_does_not_discard_a_valid_config_lowering(tmp_path):
    loader, ws = _layers(tmp_path, ws_cfg=_ctx(tool_output_ratio=0.03),
                         ws_over=_ctx(tool_output_ratio=0.9))
    assert _got(loader) == 0.03
    assert _warnings(loader) == [_note(TOR, ws / "overrides.yaml")]


def test_with_no_workspace_a_global_ratio_loads_as_written(tmp_path):
    loader, _ = _layers(tmp_path, g_cfg=_ctx(tool_output_ratio=0.5))
    assert _got(loader) == 0.5
    assert _warnings(loader) == []


def test_a_global_ratio_above_the_default_is_the_ceiling(tmp_path):
    loader, _ = _layers(tmp_path, g_cfg=_ctx(tool_output_ratio=0.5), ws_cfg=_ctx(tool_output_ratio=0.4))
    assert _got(loader) == 0.4


def test_a_zero_ratio_is_still_refused_by_the_schema():
    with pytest.raises(ValueError):
        ContextConfig(tool_output_ratio=0)


# ---------------------------------------------------------------------------
# an agent's own context: its file, its division, the workspace org.context
# ---------------------------------------------------------------------------

import yaml  # noqa: E402

from tests.unit.test_floor_machine_level import PROVIDER  # noqa: E402

DEF = DEFAULTS[TOR]


def _put(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data), encoding="utf-8")


def _agent(name="worker", **ctx) -> dict:
    data = {"name": name, "role": "Work"}
    if ctx:
        data["context"] = ctx
    return data


def _world(tmp_path, *, g_agents=None, g_divs=None, ws_agents=None, ws_divs=None, g_cfg=None,
           g_org=None, g_over=None, ws_cfg=None, ws_over=None, workspace=True):
    """A global dir and (unless `workspace` is False) a project dir, with the files given."""
    from localharness.config.loader import ConfigLoader
    g, ws = tmp_path / "global", tmp_path / "proj" / ".localharness"
    _put(g / "config.yaml", {"version": "1", "provider": PROVIDER, **(g_cfg or {})})
    for path, data in ((g / "org.yaml", g_org), (g / "overrides.yaml", g_over)):
        if data is not None:
            _put(path, data)
    ws.mkdir(parents=True)
    for path, data in ((ws / "config.yaml", ws_cfg), (ws / "overrides.yaml", ws_over)):
        if data is not None:
            _put(path, data)
    for base, agents, divs in ((g, g_agents, g_divs), (ws, ws_agents, ws_divs)):
        for stem, data in (agents or {}).items():
            _put(base / "agents" / f"{stem}.yaml", data)
        for stem, data in (divs or {}).items():
            _put(base / "divisions" / f"{stem}.yaml", data)
    return ConfigLoader(config_dir=g, local_config_dir=ws if workspace else None), ws


def _notes(loader) -> list[str]:
    return [n for n in loader.agent_warnings if "a project may only lower it" in n]


def _only(name: str, path) -> str:
    return f"ignoring context.{name} in {str(path)!r}: a project may only lower it"


def _ratio(loader, agent="worker", name=TOR) -> float:
    return getattr(loader.load_agent(agent).context, name)


def test_the_agent_lower_only_set_gains_the_three_ratios():
    from localharness.config.loader import AGENT_CORE_LOWER_ONLY_KEYS
    assert {f"context.{n}" for n in NAMES} <= AGENT_CORE_LOWER_ONLY_KEYS
    assert len(AGENT_CORE_LOWER_ONLY_KEYS) == 5


def test_a_workspace_only_agent_may_lower_a_ratio(tmp_path):
    loader, _ = _world(tmp_path, ws_agents={"analyst": _agent("analyst", tool_output_ratio=0.03)})
    assert _ratio(loader, "analyst") == 0.03
    assert _notes(loader) == []


@pytest.mark.parametrize("bad", [0.2, 0, -1, "lots", True, 2])
def test_a_workspace_only_agent_cannot_raise_or_zero_a_ratio(tmp_path, bad):
    loader, ws = _world(tmp_path, ws_agents={"analyst": _agent("analyst", tool_output_ratio=bad)})
    assert _ratio(loader, "analyst") == DEF
    assert _notes(loader) == [_only(TOR, ws / "agents" / "analyst.yaml")]


def test_a_workspace_null_ratio_is_the_baseline_silently(tmp_path):
    loader, _ = _world(tmp_path, ws_agents={"analyst": _agent("analyst", tool_output_ratio=None)})
    assert _ratio(loader, "analyst") == DEF
    assert _notes(loader) == []


def test_a_shadowing_file_cannot_raise_a_global_agent_ratio(tmp_path):
    loader, ws = _world(tmp_path,
                        g_agents={"worker": _agent(tool_output_ratio=0.02)},
                        ws_agents={"worker": _agent(tool_output_ratio=0.03)})
    assert _ratio(loader) == 0.02
    assert _notes(loader) == [_only(TOR, ws / "agents" / "worker.yaml")]


def test_a_shadowing_file_may_lower_a_global_agent_ratio(tmp_path):
    loader, _ = _world(tmp_path,
                       g_agents={"worker": _agent(tool_output_ratio=0.02)},
                       ws_agents={"worker": _agent(tool_output_ratio=0.01)})
    assert _ratio(loader) == 0.01
    assert _notes(loader) == []


def test_a_shadowing_file_that_omits_context_keeps_the_global_ratio(tmp_path):
    loader, _ = _world(tmp_path,
                       g_agents={"worker": _agent(tool_output_ratio=0.02)},
                       ws_agents={"worker": _agent()})
    assert _ratio(loader) == 0.02
    assert _notes(loader) == []


def test_a_workspace_division_cannot_raise_a_global_division_ratio(tmp_path):
    div = lambda **ctx: {"name": "research", "description": "d", "context": ctx}  # noqa: E731
    loader, ws = _world(tmp_path,
                        g_agents={"worker": {"name": "worker", "role": "Work", "division": "research"}},
                        g_divs={"research": div(evict_ratio=0.01)},
                        ws_divs={"research": div(evict_ratio=0.05)})
    assert _ratio(loader, name="evict_ratio") == 0.01
    assert _notes(loader) == [_only("evict_ratio", ws / "divisions" / "research.yaml")]


def test_a_workspace_division_may_lower_a_global_division_ratio(tmp_path):
    div = lambda **ctx: {"name": "research", "description": "d", "context": ctx}  # noqa: E731
    loader, _ = _world(tmp_path,
                       g_agents={"worker": {"name": "worker", "role": "Work", "division": "research"}},
                       g_divs={"research": div(evict_ratio=0.01)},
                       ws_divs={"research": div(evict_ratio=0.005)})
    assert _ratio(loader, name="evict_ratio") == 0.005
    assert _notes(loader) == []


@pytest.mark.parametrize("door", ["config.yaml", "overrides.yaml"])
def test_a_workspace_org_context_cannot_raise_a_global_agents_ratio(tmp_path, door):
    data = _ctx(tool_output_ratio=0.2)
    loader, ws = _world(tmp_path, g_agents={"worker": _agent()},
                        **{"ws_cfg" if door == "config.yaml" else "ws_over": data})
    assert _ratio(loader) == DEF
    assert _notes(loader) == []  # the harness walker's one line is the only note
    harness = loader.load_harness()
    assert harness.org.context.tool_output_ratio == DEF
    assert [w for w in _warnings(loader) if TOR in w] == [_note(TOR, ws / door)]


def test_a_workspace_org_context_may_lower_a_global_agents_ratio(tmp_path):
    loader, _ = _world(tmp_path, g_agents={"worker": _agent()}, ws_cfg=_ctx(tool_output_ratio=0.03))
    assert _ratio(loader) == 0.03
    assert _notes(loader) == []


def test_a_workspace_org_context_is_judged_against_the_global_org_context(tmp_path):
    loader, _ = _world(tmp_path, g_agents={"worker": _agent()},
                       g_cfg=_ctx(tool_output_ratio=0.02), ws_cfg=_ctx(tool_output_ratio=0.05))
    assert _ratio(loader) == 0.02


def test_a_workspace_org_context_cannot_hide_the_legacy_global_org_yaml(tmp_path):
    loader, _ = _world(tmp_path, g_agents={"worker": _agent()},
                       g_org={"context": {"result_cap_ratio": 0.05}},
                       ws_cfg=_ctx(max_context_tokens=4096))
    assert _ratio(loader, name="result_cap_ratio") == 0.05
    assert loader.load_agent("worker").context.max_context_tokens == 4096


def test_a_workspace_null_org_context_cannot_hide_a_lower_global_ratio(tmp_path):
    loader, _ = _world(tmp_path, g_agents={"worker": _agent()},
                       g_cfg=_ctx(evict_ratio=0.01), ws_cfg={"org": {"context": None}})
    assert _ratio(loader, name="evict_ratio") == 0.01
    assert _notes(loader) == []


def test_the_global_overrides_agent_section_is_a_baseline_rung(tmp_path):
    loader, ws = _world(tmp_path, g_over={"agent": {"context": {"tool_output_ratio": 0.04}}},
                        ws_agents={"analyst": _agent("analyst", tool_output_ratio=0.05)})
    assert _ratio(loader, "analyst") == 0.04
    assert _notes(loader) == [_only(TOR, ws / "agents" / "analyst.yaml")]


def test_a_junk_global_value_is_not_taken_as_the_baseline(tmp_path):
    loader, _ = _world(tmp_path, g_agents={"worker": _agent(tool_output_ratio="lots")},
                       ws_agents={"worker": _agent(tool_output_ratio=0.03)})
    assert _ratio(loader) == 0.03


def test_with_no_workspace_layer_a_global_ratio_loads_as_written(tmp_path):
    loader, _ = _world(tmp_path, g_agents={"worker": _agent(tool_output_ratio=0.5)}, workspace=False)
    assert _ratio(loader) == 0.5
    assert _notes(loader) == []


def test_a_ratio_never_marks_the_workspace_budget(tmp_path):
    loader, _ = _world(tmp_path, ws_agents={"analyst": _agent("analyst", tool_output_ratio=0.03)})
    assert loader.load_agent("analyst").budget_kept_from_workspace == frozenset()


def test_the_same_note_is_recorded_once(tmp_path):
    loader, ws = _world(tmp_path, ws_agents={"analyst": _agent("analyst", tool_output_ratio=0.3)})
    loader.load_agent("analyst")
    loader.load_agent("analyst", bypass_cache=True)
    assert _notes(loader) == [_only(TOR, ws / "agents" / "analyst.yaml")]


def test_global_layer_budget_is_a_call_of_the_path_reader(tmp_path):
    loader, _ = _world(tmp_path, g_agents={"worker": {"name": "worker", "role": "Work",
                                                       "permissions": {"budget": {"max_actions": 7}}}})
    assert loader._global_layer_budget("worker", None, "max_actions") == 7
    assert loader._global_layer_value("worker", None, "permissions.budget.max_actions") == 7


# ---------------------------------------------------------------------------
# a project's override of a built-in agent
# ---------------------------------------------------------------------------

import logging  # noqa: E402


def _builtin(tmp_path, *, g=None, ws=None, name="explore"):
    """The overlaid built-in explore config, through the real loader and the built-in base."""
    from localharness.agent.subagent import build_explore_config
    from localharness.config.loader import ConfigLoader
    gd, wd = tmp_path / "global", tmp_path / "proj" / ".localharness"
    _put(gd / "config.yaml", {"version": "1", "provider": PROVIDER})
    wd.mkdir(parents=True)
    if g is not None:
        _put(gd / "agents" / f"{name}.yaml", g)
    if ws is not None:
        _put(wd / "agents" / f"{name}.yaml", ws)
    return ConfigLoader(config_dir=gd, local_config_dir=wd).overlay_builtin_config(
        name, build_explore_config(name))


def _ratio_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "a project may only lower it" in r.getMessage()
            and "context." in r.getMessage()]


def test_a_workspace_overlay_may_lower_a_builtins_ratio(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        cfg = _builtin(tmp_path, ws={"context": {TOR: 0.03}})
    assert cfg.context.tool_output_ratio == 0.03
    assert _ratio_warnings(caplog) == []


@pytest.mark.parametrize("bad", [0.2, 0, "big", True, -1])
def test_a_workspace_overlay_cannot_raise_a_builtins_ratio(tmp_path, caplog, bad):
    with caplog.at_level(logging.WARNING):
        cfg = _builtin(tmp_path, ws={"context": {TOR: bad}})
    assert cfg.context.tool_output_ratio == DEF
    [warning] = _ratio_warnings(caplog)
    assert "context.tool_output_ratio" in warning and "'explore'" in warning


def test_a_workspace_null_overlay_ratio_is_the_baseline_silently(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        cfg = _builtin(tmp_path, ws={"context": {TOR: None}})
    assert cfg.context.tool_output_ratio == DEF
    assert _ratio_warnings(caplog) == []


def test_the_operators_own_builtin_file_is_the_ratio_ceiling(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        cfg = _builtin(tmp_path, g={"context": {"evict_ratio": 0.01}},
                       ws={"context": {"evict_ratio": 0.02}})
    assert cfg.context.evict_ratio == 0.01
    assert len(_ratio_warnings(caplog)) == 1


def test_a_workspace_overlay_below_the_operators_ratio_is_taken(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        cfg = _builtin(tmp_path, g={"context": {"evict_ratio": 0.01}},
                       ws={"context": {"evict_ratio": 0.005}})
    assert cfg.context.evict_ratio == 0.005
    assert _ratio_warnings(caplog) == []


def test_a_workspace_overlay_that_omits_context_keeps_the_operators_ratio(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        cfg = _builtin(tmp_path, g={"context": {"evict_ratio": 0.01}}, ws={"role": "Look around"})
    assert cfg.context.evict_ratio == 0.01
    assert _ratio_warnings(caplog) == []
