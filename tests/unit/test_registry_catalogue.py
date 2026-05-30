"""Phase 14-03 tests for localharness.registry.catalogue.

Covers REG-04 (six-surface presence) + layer attribution.
"""
from __future__ import annotations

import re

import pytest


def _make_minimal_harness_cfg():
    """Build a HarnessConfig with the minimum required fields for catalogue walks."""
    from localharness.config.models import HarnessConfig, ProviderConfig
    return HarnessConfig(
        provider=ProviderConfig(
            provider_type="ollama",
            base_url="http://x",
            default_model="m",
        ),
    )


def test_six_distinct_surface_types(components_home):
    """REG-04: catalogue exposes >=6 distinct top-level surface families.

    Surfaces enumerated:
      1. agent.role / agent.context.system_prompt_file
      2. tools.*.description
      3. org.context.compaction_threshold_pct
      4. agent.stuck_detector.window_size
      5. agent.recovery_injection.message
      6. hooks.*.config / org.hooks
    """
    from localharness.registry.catalogue import build_catalogue, SURFACE_FAMILIES

    cfg = _make_minimal_harness_cfg()

    # Synthetic ToolRegistry with one schema so tools.*.description surfaces
    class _Schema:
        description = "demo tool"

    class _ToolRegistry:
        _schemas = {"demo": _Schema()}

    # Synthetic HookSystem with one plugin so hooks.*.config surfaces
    class _HookSystem:
        loaded_plugin_names = ["demo_hook"]

    entries = build_catalogue(
        cfg,
        overlays={},
        tool_registry=_ToolRegistry(),
        hook_system=_HookSystem(),
    )

    # Each family must have at least one match
    matched_families = set()
    for family, patterns in SURFACE_FAMILIES.items():
        for path in entries:
            if any(re.search(p, path) for p in patterns):
                matched_families.add(family)
                break
    assert len(matched_families) >= 6, (
        f"Expected >=6 surface families covered, got {matched_families}"
    )


def test_required_surfaces_present(components_home):
    """REG-04: each named required surface appears in the catalogue."""
    from localharness.registry.catalogue import build_catalogue

    cfg = _make_minimal_harness_cfg()
    entries = build_catalogue(cfg, overlays={})

    required = {
        "org.context.compaction_threshold_pct",
        "agent.stuck_detector.window_size",
        "agent.recovery_injection.message",
    }
    missing = required - set(entries.keys())
    assert not missing, f"Missing required surfaces: {missing}"


def test_layer_attribution_default_when_no_overlay(components_home):
    """With no overlay, every entry's winning_layer is 'default' or 'project'."""
    from localharness.registry.catalogue import build_catalogue

    cfg = _make_minimal_harness_cfg()
    entries = build_catalogue(cfg, overlays={})
    bad = [(p, e.winning_layer) for p, e in entries.items()
           if e.winning_layer not in {"default", "project"}]
    assert not bad, f"Non-default/project layers without overlay: {bad[:5]}"


def test_layer_attribution_user_when_overlay_present(components_home):
    """Overlay sets a path → that path's entry has winning_layer == 'user'."""
    from localharness.registry.catalogue import build_catalogue

    cfg = _make_minimal_harness_cfg()
    overlays = {"user": {"org": {"context": {"compaction_threshold_pct": 0.85}}}}
    entries = build_catalogue(cfg, overlays=overlays)
    target = entries["org.context.compaction_threshold_pct"]
    assert target.winning_layer == "user"


def test_catalogue_returns_componententry_dataclasses(components_home):
    """build_catalogue returns dict[str, ComponentEntry] with the documented fields."""
    from localharness.registry.catalogue import build_catalogue, ComponentEntry

    cfg = _make_minimal_harness_cfg()
    entries = build_catalogue(cfg, overlays={})
    assert isinstance(entries, dict)
    sample = next(iter(entries.values()))
    assert isinstance(sample, ComponentEntry)
    # Documented fields per 14-RESEARCH.md Example B
    for attr in ("path", "annotation", "type_name", "current_value",
                 "default_value", "winning_layer"):
        assert hasattr(sample, attr), f"ComponentEntry missing field {attr!r}"


def test_catalogue_includes_audit_log_path(components_home):
    """org.audit_log_path is a top-level OrgConfig leaf, must appear with default value."""
    from localharness.registry.catalogue import build_catalogue

    cfg = _make_minimal_harness_cfg()
    entries = build_catalogue(cfg, overlays={})
    assert "org.audit_log_path" in entries
    e = entries["org.audit_log_path"]
    # Default per OrgConfig.audit_log_path
    assert e.current_value == "~/.localharness/audit.jsonl"


def test_catalogue_tool_registry_descriptions(components_home):
    """tools.<name>.description path appears for every registered tool."""
    from localharness.registry.catalogue import build_catalogue

    cfg = _make_minimal_harness_cfg()

    class _Schema:
        description = "exec command"

    class _ToolRegistry:
        _schemas = {"bash_exec": _Schema(), "read_file": _Schema()}

    entries = build_catalogue(cfg, overlays={}, tool_registry=_ToolRegistry())
    assert "tools.bash_exec.description" in entries
    assert "tools.read_file.description" in entries
    assert entries["tools.bash_exec.description"].current_value == "exec command"


def test_catalogue_hook_configs(components_home):
    """hooks.<name>.config path appears for every loaded hook plugin."""
    from localharness.registry.catalogue import build_catalogue

    cfg = _make_minimal_harness_cfg()

    class _HookSystem:
        loaded_plugin_names = ["audit_logger", "pev_check"]

    entries = build_catalogue(cfg, overlays={}, hook_system=_HookSystem())
    assert "hooks.audit_logger.config" in entries
    assert "hooks.pev_check.config" in entries


def test_agent_cfg_drives_agent_star_current_value(components_home):
    """WARNING-2: agent.* current_value must reflect the LIVE AgentConfig, not field defaults.

    With agent_cfg=None the catalogue reports the StuckDetectorConfig default (window_size=5).
    With a live AgentConfig carrying window_size=9, the catalogue MUST report 9 — this is the
    provenance the experiment/proposer call sites get wrong today (build_catalogue(cfg) with no
    agent_cfg=), making the recorded `before` value detached from the live overlay-resolved config.
    """
    from localharness.config.models import AgentConfig
    from localharness.registry import build_catalogue

    cfg = _make_minimal_harness_cfg()

    # Baseline: no agent_cfg -> field default (5).
    default_entries = build_catalogue(cfg, overlays={})
    assert default_entries["agent.stuck_detector.window_size"].current_value == 5

    # Live: a resolved AgentConfig with the overlay value (9).
    live = AgentConfig.model_validate(
        {"name": "bench-x", "role": "r", "stuck_detector": {"window_size": 9}}
    )
    live_entries = build_catalogue(cfg, overlays={}, agent_cfg=live)
    assert live_entries["agent.stuck_detector.window_size"].current_value == 9, (
        "build_catalogue must thread agent_cfg into agent.* current_value so the recorded "
        "`before` value is the live overlay-resolved config (WARNING-2), not AgentConfig defaults."
    )
