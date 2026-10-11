"""`stuck_detector` is a machine-only SECTION: a workspace agent file and a workspace overlay of a
built-in get it restored WHOLE from the global layer.

Whole, not key by key: a workspace file that merely OMITS `escalation_threshold` must not let the
schema default (3) lift a stricter global value (2). Every scenario goes through the real
ConfigLoader with files authored the way an install has them.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from localharness.config.loader import AGENT_CORE_GLOBAL_ONLY_SECTIONS, ConfigLoader
from localharness.config.models import StuckDetectorConfig

DEFAULT = StuckDetectorConfig()
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


def test_the_declaration_names_the_stuck_section() -> None:
    assert AGENT_CORE_GLOBAL_ONLY_SECTIONS == frozenset({"stuck_detector"})


# ---------------------------------------------------------------------------
# load_agent_file: a workspace agent file that shadows a global one
# ---------------------------------------------------------------------------

def test_a_shadowing_file_gets_the_global_section_whole(layers) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml",
                _agent(stuck_detector={"escalation_threshold": 2}))
    ws_file = ws / "agents" / "deployer.yaml"
    _write_yaml(ws_file, _agent(stuck_detector={"window_size": 3}))

    cfg, loader = _load(global_dir, ws)

    assert cfg.stuck_detector.escalation_threshold == 2
    assert cfg.stuck_detector.window_size == DEFAULT.window_size == 5
    notes = _notes(loader)
    assert len(notes) == 1
    assert repr(str(ws_file)) in notes[0]


def test_an_omitted_key_cannot_lift_a_stricter_global_value(layers) -> None:
    """STUCK-03: the workspace file's section omits escalation_threshold; the global says 2."""
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml",
                _agent(stuck_detector={"escalation_threshold": 2}))
    _write_yaml(ws / "agents" / "deployer.yaml",
                _agent(stuck_detector={"recovery_threshold": 1, "max_nudges_per_turn": 9}))

    cfg, _ = _load(global_dir, ws)

    assert cfg.stuck_detector.escalation_threshold == 2
    assert cfg.stuck_detector.recovery_threshold == DEFAULT.recovery_threshold
    assert cfg.stuck_detector.max_nudges_per_turn == DEFAULT.max_nudges_per_turn


def test_a_file_that_omits_the_section_is_restored_silently(layers) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml",
                _agent(stuck_detector={"escalation_threshold": 2}))
    _write_yaml(ws / "agents" / "deployer.yaml", _agent())

    cfg, loader = _load(global_dir, ws)

    assert cfg.stuck_detector.escalation_threshold == 2
    assert _notes(loader) == []


def test_a_silent_global_gives_the_schema_defaults_and_one_note(layers) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml", _agent())
    ws_file = ws / "agents" / "deployer.yaml"
    _write_yaml(ws_file, _agent(stuck_detector={"escalation_threshold": 50, "window_size": 2}))

    cfg, loader = _load(global_dir, ws)

    assert cfg.stuck_detector.escalation_threshold == DEFAULT.escalation_threshold == 3
    assert cfg.stuck_detector.window_size == DEFAULT.window_size == 5
    notes = _notes(loader)
    assert len(notes) == 1 and repr(str(ws_file)) in notes[0]


def test_a_workspace_only_agent_cannot_set_the_section(layers) -> None:
    """No global agent file at all: the schema default stands."""
    global_dir, ws = layers
    _write_yaml(ws / "agents" / "solo.yaml", _agent("solo", stuck_detector={"window_size": 1}))

    cfg, loader = _load(global_dir, ws, "solo")

    assert cfg.stuck_detector == DEFAULT
    assert len(_notes(loader)) == 1


def test_the_global_overrides_agent_section_is_the_next_rung(layers) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "overrides.yaml", {"agent": {"stuck_detector": {"escalation_threshold": 2}}})
    _write_yaml(ws / "agents" / "deployer.yaml", _agent(stuck_detector={"escalation_threshold": 9}))

    cfg, loader = _load(global_dir, ws)

    assert cfg.stuck_detector.escalation_threshold == 2
    assert len(_notes(loader)) == 1


def test_a_workspace_section_equal_to_the_global_one_earns_no_note(layers) -> None:
    global_dir, ws = layers
    section = {"escalation_threshold": 2, "window_size": 4}
    _write_yaml(global_dir / "agents" / "deployer.yaml", _agent(stuck_detector=section))
    _write_yaml(ws / "agents" / "deployer.yaml", _agent(stuck_detector=dict(section)))

    cfg, loader = _load(global_dir, ws)

    assert (cfg.stuck_detector.escalation_threshold, cfg.stuck_detector.window_size) == (2, 4)
    assert _notes(loader) == []


def test_a_global_file_and_a_workspace_less_session_load_their_own_section(layers) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml",
                _agent(stuck_detector={"escalation_threshold": 2, "window_size": 7}))
    _write_yaml(ws / "agents" / "other.yaml", _agent("other"))

    in_ws, loader = _load(global_dir, ws)  # the workspace does not shadow deployer
    alone, loader2 = _load(global_dir, None)

    for cfg in (in_ws, alone):
        assert (cfg.stuck_detector.escalation_threshold, cfg.stuck_detector.window_size) == (2, 7)
    assert _notes(loader) == [] and _notes(loader2) == []


def test_loading_the_same_workspace_file_twice_notes_once(layers) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml", _agent())
    _write_yaml(ws / "agents" / "deployer.yaml", _agent(stuck_detector={"window_size": 2}))
    loader = ConfigLoader(config_dir=global_dir, local_config_dir=ws)

    loader.load_agent("deployer")
    loader.load_agent_file(ws / "agents" / "deployer.yaml")

    assert len(_notes(loader)) == 1


def test_a_crafted_stem_is_spelled_by_repr(layers) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "x.yaml", _agent("x"))
    ws_file = ws / "agents" / "evil\x1b[31m.yaml"
    _write_yaml(ws_file, _agent("evil", stuck_detector={"window_size": 2}))
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


def _builtin(global_dir: Path, ws: Path | None, name: str = "explore"):
    loader = ConfigLoader(config_dir=global_dir, local_config_dir=ws)
    return loader.overlay_builtin_config(name, _builtin_base(name))


def _stuck_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and NOTE in r.getMessage()
            and "stuck_detector" in r.getMessage()]


def test_a_workspace_overlay_cannot_loosen_a_builtins_ladder(layers, caplog) -> None:
    global_dir, ws = layers
    ws_file = ws / "agents" / "explore.yaml"
    _write_yaml(ws_file, {"stuck_detector": {"escalation_threshold": 50}})

    with caplog.at_level(logging.WARNING):
        cfg = _builtin(global_dir, ws)

    assert cfg.stuck_detector == _builtin_base().stuck_detector
    warnings = _stuck_warnings(caplog)
    assert len(warnings) == 1 and repr(str(ws_file)) in warnings[0]


def test_a_global_explore_file_is_the_rung_above_the_base(layers, caplog) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "explore.yaml", {"stuck_detector": {"escalation_threshold": 2}})
    _write_yaml(ws / "agents" / "explore.yaml", {"stuck_detector": {"window_size": 3}})

    with caplog.at_level(logging.WARNING):
        cfg = _builtin(global_dir, ws)

    assert cfg.stuck_detector.escalation_threshold == 2
    assert cfg.stuck_detector.window_size == 5
    assert len(_stuck_warnings(caplog)) == 1


def test_an_overlay_that_leaves_the_section_alone_keeps_the_base_silently(layers, caplog) -> None:
    global_dir, ws = layers
    _write_yaml(ws / "agents" / "explore.yaml", {"permissions": {"mode": "read-only"}})

    with caplog.at_level(logging.WARNING):
        cfg = _builtin(global_dir, ws)

    assert cfg.stuck_detector == _builtin_base().stuck_detector
    assert _stuck_warnings(caplog) == []


def test_a_workspace_section_the_base_already_holds_earns_no_note(layers, caplog) -> None:
    global_dir, ws = layers
    base = _builtin_base().stuck_detector
    _write_yaml(ws / "agents" / "explore.yaml",
                {"stuck_detector": {"window_size": base.window_size}})

    with caplog.at_level(logging.WARNING):
        cfg = _builtin(global_dir, ws)

    assert cfg.stuck_detector == base
    assert _stuck_warnings(caplog) == []


def test_a_global_only_explore_file_and_a_workspace_less_overlay_are_untouched(layers, caplog) -> None:
    global_dir, _ = layers
    _write_yaml(global_dir / "agents" / "explore.yaml", {"stuck_detector": {"escalation_threshold": 2}})

    with caplog.at_level(logging.WARNING):
        cfg = _builtin(global_dir, None)

    assert cfg.stuck_detector.escalation_threshold == 2
    assert _stuck_warnings(caplog) == []


# ---------------------------------------------------------------------------
# call_only_multiplier: the ceiling a workspace must never raise (Nyquist)
# ---------------------------------------------------------------------------

def test_a_workspace_agent_file_cannot_raise_the_call_only_multiplier(layers) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml", _agent(stuck_detector={"call_only_multiplier": 2}))
    _write_yaml(ws / "agents" / "deployer.yaml", _agent(stuck_detector={"call_only_multiplier": 10}))

    cfg, loader = _load(global_dir, ws)

    assert cfg.stuck_detector.call_only_multiplier == 2
    assert len(_notes(loader)) == 1


def test_a_workspace_only_agent_cannot_raise_the_call_only_multiplier(layers) -> None:
    global_dir, ws = layers
    _write_yaml(ws / "agents" / "solo.yaml", _agent("solo", stuck_detector={"call_only_multiplier": 10}))

    cfg, _ = _load(global_dir, ws, "solo")

    assert cfg.stuck_detector.call_only_multiplier == DEFAULT.call_only_multiplier


def test_a_workspace_overrides_agent_section_cannot_raise_the_call_only_multiplier(layers) -> None:
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml", _agent())
    _write_yaml(ws / "overrides.yaml", {"agent": {"stuck_detector": {"call_only_multiplier": 10,
                                                                     "escalation_threshold": 50}}})

    cfg, _ = _load(global_dir, ws)

    assert cfg.stuck_detector.call_only_multiplier == DEFAULT.call_only_multiplier
    assert cfg.stuck_detector.escalation_threshold == DEFAULT.escalation_threshold


def test_a_workspace_overlay_cannot_raise_a_builtins_call_only_multiplier(layers, caplog) -> None:
    global_dir, ws = layers
    _write_yaml(ws / "agents" / "explore.yaml", {"stuck_detector": {"call_only_multiplier": 10}})

    with caplog.at_level(logging.WARNING):
        cfg = _builtin(global_dir, ws)

    assert cfg.stuck_detector.call_only_multiplier == _builtin_base().stuck_detector.call_only_multiplier
    assert len(_stuck_warnings(caplog)) == 1


def test_the_loaded_multiplier_is_the_ceiling_the_running_detector_uses(layers) -> None:
    """End to end from the files: the workspace's 10 never reaches the detector's ceiling."""
    from localharness.agent.loop import StuckDetector

    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml",
                _agent(stuck_detector={"call_only_multiplier": 1, "escalation_threshold": 3}))
    _write_yaml(ws / "agents" / "deployer.yaml", _agent(stuck_detector={"call_only_multiplier": 10}))
    cfg, _ = _load(global_dir, ws)
    sd_cfg = cfg.stuck_detector
    sd = StuckDetector(window_size=sd_cfg.window_size, recovery_threshold=sd_cfg.recovery_threshold,
                       escalation_threshold=sd_cfg.escalation_threshold,
                       call_only_multiplier=sd_cfg.call_only_multiplier,
                       result_origin=lambda _n: "trusted")
    for i in range(3):
        sd.record("bash_exec", {"cmd": "gh run list"}, (True, f"tick {i}"))
        sd.classify()
    assert sd.ceiling_hit == sd.compute_signature("bash_exec", {"cmd": "gh run list"})
