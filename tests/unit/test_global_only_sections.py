"""`task_flow` is a machine-only SECTION: a workspace agent file and a workspace overlay of a
built-in get it restored WHOLE from the global layer.

Whole, not key by key: a workspace file that merely OMITS `executor` must not let the schema default
lift a stricter global value. Every scenario goes through the real ConfigLoader with files authored
the way an install has them.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from localharness.config.loader import AGENT_CORE_GLOBAL_ONLY_SECTIONS, ConfigLoader
from localharness.config.models import ChildBudget, TaskFlowConfig
from localharness.orchestrator.task_flow.record import limited

DEFAULT = TaskFlowConfig()
NOTE = "only the global config may set it"

_MINIMAL = {
    "version": "1",
    "provider": {
        "provider_type": "vllm",
        "base_url": "http://localhost:8000/v1",
        "default_model": "global-model",
    },
}


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data), encoding="utf-8")


@pytest.fixture
def layers(tmp_path: Path) -> tuple[Path, Path]:
    """A global config dir (config.yaml only) and a workspace `.localharness/` under a project."""
    global_dir = tmp_path / "global"
    workspace_dir = tmp_path / "proj" / ".localharness"
    workspace_dir.mkdir(parents=True)
    _write_yaml(global_dir / "config.yaml", _MINIMAL)
    return global_dir, workspace_dir


def _agent(name: str = "deployer", **extra) -> dict:
    return {"name": name, "role": "Deploy agent", **extra}


def _load(global_dir: Path, ws: Path | None, name: str = "deployer"):
    loader = ConfigLoader(config_dir=global_dir, local_config_dir=ws)
    cfg = loader.load_agent(name)
    return cfg, loader


def _notes(loader: ConfigLoader) -> list[str]:
    return [n for n in loader.agent_warnings if NOTE in n]


def test_the_declaration_names_the_global_only_sections() -> None:
    assert AGENT_CORE_GLOBAL_ONLY_SECTIONS == frozenset({"task_flow"})


_EXECUTOR = {"executor": {"actions": 12, "minutes": 10}}
_CHILD = ChildBudget(actions=12, minutes=10)


def _cap(cfg) -> float | None:
    return cfg.task_flow.flow_cap_minutes


# ---------------------------------------------------------------------------
# load_agent_file: a workspace agent file that shadows a global one
# ---------------------------------------------------------------------------

def test_a_shadowing_file_gets_the_global_task_flow_whole(layers) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml", _agent(task_flow=_EXECUTOR))
    ws_file = ws / "agents" / "deployer.yaml"
    _write_yaml(ws_file, _agent(task_flow={"flow_cap_minutes": 360}))

    cfg, loader = _load(global_dir, ws)

    assert cfg.task_flow.executor == _CHILD
    assert _cap(cfg) is None
    notes = _notes(loader)
    assert len(notes) == 1 and "task_flow" in notes[0] and repr(str(ws_file)) in notes[0]


def test_an_omitted_key_cannot_lift_a_stricter_global_value(layers) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml", _agent(task_flow=_EXECUTOR))
    _write_yaml(ws / "agents" / "deployer.yaml", _agent(task_flow={"panels": {"plan": ["security"]}}))

    cfg, _ = _load(global_dir, ws)

    assert cfg.task_flow.executor == _CHILD
    assert cfg.task_flow.panels == {}


def test_a_file_that_omits_the_section_is_restored_silently(layers) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml", _agent(task_flow=_EXECUTOR))
    _write_yaml(ws / "agents" / "deployer.yaml", _agent())

    cfg, loader = _load(global_dir, ws)

    assert cfg.task_flow.executor == _CHILD
    assert _notes(loader) == []


def test_a_silent_global_gives_the_schema_defaults_and_one_note(layers) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml", _agent())
    ws_file = ws / "agents" / "deployer.yaml"
    _write_yaml(ws_file, _agent(task_flow={"flow_cap_minutes": 360, "executor": {"actions": 40, "minutes": 30}}))

    cfg, loader = _load(global_dir, ws)

    assert cfg.task_flow == DEFAULT
    notes = _notes(loader)
    assert len(notes) == 1 and repr(str(ws_file)) in notes[0]


def test_a_workspace_only_agent_cannot_set_the_section(layers) -> None:
    global_dir, ws = layers
    _write_yaml(ws / "agents" / "solo.yaml", _agent("solo", task_flow={"flow_cap_minutes": 360}))

    cfg, loader = _load(global_dir, ws, "solo")

    assert cfg.task_flow == DEFAULT
    assert len(_notes(loader)) == 1


def test_the_global_overrides_agent_section_is_the_next_rung(layers) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "overrides.yaml", {"agent": {"task_flow": _EXECUTOR}})
    _write_yaml(ws / "agents" / "deployer.yaml", _agent(task_flow={"flow_cap_minutes": 360}))

    cfg, loader = _load(global_dir, ws)

    assert cfg.task_flow.executor == _CHILD
    assert _cap(cfg) is None
    assert len(_notes(loader)) == 1


def test_the_restore_is_the_global_cascade_not_the_first_section_found(layers) -> None:
    """The agent file's section and the overrides' `agent:` section are merged, as a global-only load
    merges them, so a shadowing file that declares no section cannot drop the operator's overrides key."""
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml", _agent(task_flow=_EXECUTOR))
    _write_yaml(global_dir / "overrides.yaml", {"agent": {"task_flow": {"flow_cap_minutes": 120}}})
    _write_yaml(ws / "agents" / "deployer.yaml", _agent())

    alone, _ = _load(global_dir, None)
    cfg, loader = _load(global_dir, ws)

    assert (alone.task_flow.executor, _cap(alone)) == (_CHILD, 120)
    assert cfg.task_flow == alone.task_flow
    assert _notes(loader) == []


def test_the_agent_files_key_wins_over_the_overrides_key_in_the_restore(layers) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml", _agent(task_flow={"flow_cap_minutes": 100}))
    _write_yaml(global_dir / "overrides.yaml",
                {"agent": {"task_flow": {"flow_cap_minutes": 200, "executor": {"actions": 12, "minutes": 10}}}})
    _write_yaml(ws / "agents" / "deployer.yaml", _agent(task_flow={"flow_cap_minutes": 300}))

    alone, _ = _load(global_dir, None)
    cfg, loader = _load(global_dir, ws)

    assert (_cap(cfg), cfg.task_flow.executor) == (100, _CHILD)
    assert cfg.task_flow == alone.task_flow
    assert len(_notes(loader)) == 1


def test_a_workspace_section_equal_to_the_global_one_earns_no_note(layers) -> None:
    global_dir, ws = layers
    section = {"flow_cap_minutes": 120, "executor": {"actions": 12, "minutes": 10}}
    _write_yaml(global_dir / "agents" / "deployer.yaml", _agent(task_flow=section))
    _write_yaml(ws / "agents" / "deployer.yaml", _agent(task_flow=dict(section)))

    cfg, loader = _load(global_dir, ws)

    assert (_cap(cfg), cfg.task_flow.executor) == (120, _CHILD)
    assert _notes(loader) == []


def test_a_global_file_and_a_workspace_less_session_load_their_own_section(layers) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml", _agent(task_flow={"flow_cap_minutes": 120}))
    _write_yaml(ws / "agents" / "other.yaml", _agent("other"))

    in_ws, loader = _load(global_dir, ws)
    alone, loader2 = _load(global_dir, None)

    assert (_cap(in_ws), _cap(alone)) == (120, 120)
    assert _notes(loader) == [] and _notes(loader2) == []


def test_loading_the_same_workspace_file_twice_notes_once(layers) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml", _agent())
    _write_yaml(ws / "agents" / "deployer.yaml", _agent(task_flow={"flow_cap_minutes": 360}))
    loader = ConfigLoader(config_dir=global_dir, local_config_dir=ws)

    loader.load_agent("deployer")
    loader.load_agent_file(ws / "agents" / "deployer.yaml")

    assert len(_notes(loader)) == 1


def test_a_crafted_stem_is_spelled_by_repr(layers) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "x.yaml", _agent("x"))
    ws_file = ws / "agents" / "evil\x1b[31m.yaml"
    _write_yaml(ws_file, _agent("evil", task_flow={"flow_cap_minutes": 360}))
    loader = ConfigLoader(config_dir=global_dir, local_config_dir=ws)

    loader.load_agent_file(ws_file)

    notes = _notes(loader)
    assert len(notes) == 1
    assert "\x1b" not in notes[0] and repr(str(ws_file)) in notes[0]


# ---------------------------------------------------------------------------
# overlay_builtin_config: a workspace agents/explore.yaml over a built-in
# ---------------------------------------------------------------------------

def _builtin_base(name: str = "explore"):
    from localharness.agent.subagent import build_explore_config

    return build_explore_config(name)


_OVERLAY_NOTICES: list = []
"""What the built-in overlay handed to its notice reporter during the current test."""


@pytest.fixture(autouse=True)
def _fresh_overlay_notices():
    _OVERLAY_NOTICES.clear()
    yield
    _OVERLAY_NOTICES.clear()


def _builtin(global_dir: Path, ws: Path | None, name: str = "explore"):
    loader = ConfigLoader(config_dir=global_dir, local_config_dir=ws)
    return loader.overlay_builtin_config(name, _builtin_base(name), notices=_OVERLAY_NOTICES.append)


def _flow_notices(caplog=None) -> list[str]:
    """The overlay's task_flow notice texts; the overlay never logs a rejected value."""
    if caplog is not None:
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    return [n.text for n in _OVERLAY_NOTICES if NOTE in n.text and n.subject == "task_flow"]


def test_a_workspace_overlay_of_a_builtin_gets_the_base_section(layers, caplog) -> None:
    global_dir, ws = layers
    ws_file = ws / "agents" / "explore.yaml"
    _write_yaml(ws_file, {"task_flow": {"flow_cap_minutes": 360}})

    with caplog.at_level(logging.WARNING):
        cfg = _builtin(global_dir, ws)

    assert cfg.task_flow == _builtin_base().task_flow
    notices = _flow_notices(caplog)
    assert len(notices) == 1 and repr(str(ws_file)) in notices[0]


def test_a_global_explore_file_is_the_rung_above_the_base(layers, caplog) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "explore.yaml", {"task_flow": _EXECUTOR})
    _write_yaml(ws / "agents" / "explore.yaml", {"task_flow": {"flow_cap_minutes": 360}})

    with caplog.at_level(logging.WARNING):
        cfg = _builtin(global_dir, ws)

    assert cfg.task_flow.executor == _CHILD
    assert _cap(cfg) is None
    assert len(_flow_notices(caplog)) == 1


def test_an_overlay_that_leaves_the_section_alone_keeps_the_base_silently(layers, caplog) -> None:
    global_dir, ws = layers
    _write_yaml(ws / "agents" / "explore.yaml", {"permissions": {"mode": "read-only"}})

    with caplog.at_level(logging.WARNING):
        cfg = _builtin(global_dir, ws)

    assert cfg.task_flow == _builtin_base().task_flow
    assert _flow_notices(caplog) == []


def test_a_workspace_section_the_base_already_holds_earns_no_note(layers, caplog) -> None:
    global_dir, ws = layers
    base = _builtin_base().task_flow
    _write_yaml(ws / "agents" / "explore.yaml", {"task_flow": {"panels": {}}})

    with caplog.at_level(logging.WARNING):
        cfg = _builtin(global_dir, ws)

    assert cfg.task_flow == base
    assert _flow_notices(caplog) == []


def test_a_global_only_explore_file_and_a_workspace_less_overlay_are_untouched(layers, caplog) -> None:
    global_dir, _ = layers
    _write_yaml(global_dir / "agents" / "explore.yaml", {"task_flow": _EXECUTOR})

    with caplog.at_level(logging.WARNING):
        cfg = _builtin(global_dir, None)

    assert cfg.task_flow.executor == _CHILD
    assert _flow_notices(caplog) == []


# ---------------------------------------------------------------------------
# Both doors note by one predicate, and a no-op declaration earns no note
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("declared", [{"panels": {}}, {}])
def test_a_no_op_section_under_a_silent_global_earns_no_note(layers, declared) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml", _agent())
    _write_yaml(ws / "agents" / "deployer.yaml", _agent(task_flow=declared))

    cfg, loader = _load(global_dir, ws)

    assert cfg.task_flow == DEFAULT
    assert _notes(loader) == []


def test_an_empty_section_under_a_global_section_earns_no_note(layers) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml", _agent(task_flow=_EXECUTOR))
    _write_yaml(ws / "agents" / "deployer.yaml", _agent(task_flow={}))

    cfg, loader = _load(global_dir, ws)

    assert cfg.task_flow.executor == _CHILD
    assert _notes(loader) == []


def test_a_builtin_key_equal_to_what_a_global_file_restores_earns_no_note(layers, caplog) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "explore.yaml", {"task_flow": _EXECUTOR})
    _write_yaml(ws / "agents" / "explore.yaml", {"task_flow": {"panels": {}}})

    with caplog.at_level(logging.WARNING):
        cfg = _builtin(global_dir, ws)

    assert cfg.task_flow.executor == _CHILD
    assert _flow_notices(caplog) == []


@pytest.mark.parametrize("declared", [{"flow_cap_minutes": 360}, {"no_such_key": 1}, "loose"])
def test_both_doors_note_a_section_that_would_change_something(layers, caplog, declared) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml", _agent())
    _write_yaml(ws / "agents" / "deployer.yaml", _agent(task_flow=declared))
    _write_yaml(ws / "agents" / "explore.yaml", {"task_flow": declared})

    cfg, loader = _load(global_dir, ws)
    with caplog.at_level(logging.WARNING):
        builtin = _builtin(global_dir, ws)

    assert cfg.task_flow == DEFAULT
    assert builtin.task_flow == _builtin_base().task_flow
    assert len(_notes(loader)) == 1 and len(_flow_notices(caplog)) == 1


# ---------------------------------------------------------------------------
# task_flow: the code limits on what the global layer sets
# ---------------------------------------------------------------------------

def test_task_flow_a_global_value_outside_the_table_loads_and_is_held_by_limited(layers) -> None:
    global_dir, _ = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml", _agent(task_flow={
        "executor": {"actions": 49, "minutes": 41},
        "critics": {"answer": {"actions": 2, "minutes": 0.5}},
        "explorer": {"actions": 12, "minutes": 6.5},
        "flow_cap_minutes": 1000}))

    cfg, _ = _load(global_dir, None)
    held, notes = limited(cfg.task_flow)

    assert (held.executor.actions, held.executor.minutes) == (48, 40)
    assert (held.critics["answer"].actions, held.critics["answer"].minutes) == (3, 1)
    assert (held.explorer.actions, held.explorer.minutes) == (12, 6.5)
    assert held.flow_cap_minutes == 360
    assert len(notes) == 3 and not any("explorer" in n for n in notes)


def test_task_flow_a_value_exactly_at_a_floor_or_a_ceiling_earns_no_note() -> None:
    held, notes = limited(TaskFlowConfig(
        executor={"actions": 48, "minutes": 40}, critics={"answer": {"actions": 3, "minutes": 1}},
        flow_cap_minutes=360))

    assert notes == ()
    assert (held.executor.actions, held.executor.minutes, held.flow_cap_minutes) == (48, 40, 360)


@pytest.mark.parametrize("given", [
    {"revisions": {"plan": -1}},
    {"revisions": {"plan": 1.5}},
    {"budgets": {}},
    {"panels": {"plan": ["Security!"]}},
    {"executor": {"actions": 1.5, "minutes": 5}},
    {"explorer": {"actions": 5, "minutes": "soon"}},
    {"flow_cap_minutes": "long"},
])
def test_task_flow_a_structurally_invalid_value_is_refused(given) -> None:
    with pytest.raises(ValidationError):
        TaskFlowConfig(**given)
