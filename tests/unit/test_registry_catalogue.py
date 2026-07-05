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


# ---------------------------------------------------------------------------
# MECH-01 — SelfCheckConfig auto-enumerates as a new mechanism-class mutable axis.
# Adding `self_check: SelfCheckConfig` to AgentConfig makes walk_model_fields recurse
# the nested BaseModel into agent.self_check.{enabled,max_passes} with ZERO catalogue edit
# (mirrors how agent.stuck_detector.* and agent.memory.inject_into_context enumerate).
# ---------------------------------------------------------------------------


def test_self_check_leaves_enumerate(components_home):
    """MECH-01 Test A: agent.self_check.{enabled,max_passes} both appear; catalogue is 82 (was 80)."""
    from localharness.config.models import AgentConfig
    from localharness.registry.catalogue import build_catalogue

    cfg = _make_minimal_harness_cfg()
    entries = build_catalogue(cfg, overlays={}, agent_cfg=AgentConfig(name="x", role="y"))

    assert "agent.self_check.enabled" in entries
    assert "agent.self_check.max_passes" in entries
    # New context-efficiency leaves: memory.{index_mode,max_session_history_entries} +
    # context.{tool_result_eviction,tool_result_evict_threshold_chars}. The two context.*
    # leaves enumerate under BOTH agent.context.* and org.context.* (shared ContextConfig),
    # so 90 -> 96 (+2 memory agent leaves, +2 context leaves x2 scopes).
    assert "agent.memory.index_mode" in entries
    assert "agent.memory.max_session_history_entries" in entries
    assert "agent.context.tool_result_eviction" in entries
    assert "agent.context.tool_result_evict_threshold_chars" in entries
    assert "agent.max_subagent_depth" in entries  # P2: delegation-depth cap is addressable
    assert "agent.cruncher.exec_enabled" in entries  # P-CRUNCH B: cruncher exec is addressable
    assert "agent.memory.predictive_gate.write_live" in entries  # Phase 35 PGATE: the KILL-revert lever is addressable
    # Phase 36 (the chapter-writer): the eight agent.memory.consolidation.* idle-LLM axes.
    assert "agent.memory.consolidation.schema_writer_enabled" in entries
    assert "agent.memory.consolidation.reconcile_enabled" in entries
    assert "agent.memory.consolidation.mining_enabled" in entries
    assert "agent.memory.consolidation.cluster_min_sessions" in entries
    assert "agent.memory.consolidation.schema_write_budget" in entries
    assert "agent.memory.consolidation.schema_depth_cap" in entries
    assert "agent.memory.consolidation.reconcile_ttl_looks" in entries
    assert "agent.memory.consolidation.mining_write_budget" in entries
    assert len(entries) == 125, (
        f"catalogue should be 125 entries (93 after RLM removal + agent.cruncher.* x3 + "
        f"org.enforce_capability_floor x1 + agent.memory.write_gate_enabled x1 [v2.0 WRITE-03] + agent.memory.consolidation.* x14 [v2.0 CONS-01 x6 + Phase 36 chapter-writer x8] + agent.memory.predictive_gate.* x13 [Phase 34 COLL + Phase 35 write_live]), got {len(entries)}"
    )


def test_self_check_leaf_annotations(components_home):
    """MECH-01 Test B: enabled is a bool leaf, max_passes is an int leaf.

    Mirrors agent.memory.inject_into_context (bool) + agent.stuck_detector.window_size (int).
    """
    from localharness.config.models import AgentConfig
    from localharness.registry.catalogue import build_catalogue

    cfg = _make_minimal_harness_cfg()
    entries = build_catalogue(cfg, overlays={}, agent_cfg=AgentConfig(name="x", role="y"))

    assert entries["agent.self_check.enabled"].annotation is bool
    assert entries["agent.self_check.max_passes"].annotation is int


def test_self_check_defaults_and_bounds():
    """MECH-01 Test C: defaults (enabled=False, max_passes=1) + max_passes bounds (ge=1, le=3)."""
    import pydantic

    from localharness.config.models import AgentConfig

    a = AgentConfig(name="x", role="y")
    assert a.self_check.enabled is False
    assert a.self_check.max_passes == 1

    # Out-of-bound max_passes must raise (le=3 and ge=1, so the review step provably terminates).
    for bad in (0, 4):
        with pytest.raises(pydantic.ValidationError):
            AgentConfig.model_validate(
                {"name": "x", "role": "y", "self_check": {"max_passes": bad}}
            )


# ---------------------------------------------------------------------------
# MODP-01 — RoleSectionsConfig auto-enumerates as four orthogonal mutable axes.
# Adding `role_sections: RoleSectionsConfig` (four str fields) to AgentConfig makes
# walk_model_fields recurse the nested BaseModel into
# agent.role_sections.{identity,tool_use,stopping,output} with ZERO catalogue edit
# (mirrors agent.self_check.* and agent.stuck_detector.*). Catalogue 82 -> 86.
# ---------------------------------------------------------------------------


def test_role_sections_leaves_enumerate(components_home):
    """MODP-01 Test A/B/C: all four agent.role_sections.* str leaves appear; catalogue is 93 (agent.rlm.* removed)."""
    from localharness.config.models import AgentConfig
    from localharness.registry.catalogue import build_catalogue

    cfg = _make_minimal_harness_cfg()
    entries = build_catalogue(cfg, overlays={}, agent_cfg=AgentConfig(name="x", role="y"))

    leaves = [f"agent.role_sections.{s}" for s in ("identity", "tool_use", "stopping", "output")]
    missing = [leaf for leaf in leaves if leaf not in entries]
    assert not missing, f"Missing role_sections leaves: {missing}"

    # Test C: each section is a str leaf (mirrors agent.role itself being a str leaf).
    for leaf in leaves:
        assert entries[leaf].annotation is str, (
            f"{leaf} should be a str leaf, got {entries[leaf].annotation}"
        )

    assert len(entries) == 125, (
        f"catalogue should be 125 entries (93 after RLM removal + agent.cruncher.* x3 + "
        f"org.enforce_capability_floor x1 + agent.memory.write_gate_enabled x1 [v2.0 WRITE-03] + agent.memory.consolidation.* x14 [v2.0 CONS-01 x6 + Phase 36 chapter-writer x8] + agent.memory.predictive_gate.* x13 [Phase 34 COLL + Phase 35 write_live]), got {len(entries)}"
    )


def test_role_sections_defaults_empty():
    """MODP-01 Test D: all four sections default to '' — the structural basis of byte-identity."""
    from localharness.config.models import AgentConfig

    a = AgentConfig(name="x", role="y")
    assert a.role_sections.identity == ""
    assert a.role_sections.tool_use == ""
    assert a.role_sections.stopping == ""
    assert a.role_sections.output == ""
