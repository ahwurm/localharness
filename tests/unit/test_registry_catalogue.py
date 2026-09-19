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
    """With no overlay, every entry's winning_layer is 'default' or the global config band.

    Band names corrected in 43-03: the old `project` meant the GLOBAL config.yaml, not a project
    folder. Same assertion, same strength — only the spelling is honest now.
    """
    from localharness.registry.catalogue import (
        LAYER_DEFAULT,
        LAYER_GLOBAL_CONFIG,
        build_catalogue,
    )

    cfg = _make_minimal_harness_cfg()
    entries = build_catalogue(cfg, overlays={})
    bad = [(p, e.winning_layer) for p, e in entries.items()
           if e.winning_layer not in {LAYER_DEFAULT, LAYER_GLOBAL_CONFIG}]
    assert not bad, f"Non-default/global-config layers without overlay: {bad[:5]}"


def test_layer_attribution_user_when_overlay_present(components_home):
    """Overlay sets a path → that path's entry names the global OVERRIDES band.

    Band name corrected in 43-03 (`user` → `global-overrides`): the dict this test passes is the
    machine-global overrides.yaml, and `user` never said which of the two global files it meant.
    """
    from localharness.registry.catalogue import LAYER_GLOBAL_OVERRIDES, build_catalogue

    cfg = _make_minimal_harness_cfg()
    overlays = {LAYER_GLOBAL_OVERRIDES: {"org": {"context": {"compaction_threshold_pct": 0.85}}}}
    entries = build_catalogue(cfg, overlays=overlays)
    target = entries["org.context.compaction_threshold_pct"]
    assert target.winning_layer == LAYER_GLOBAL_OVERRIDES


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
    # Default per OrgConfig.audit_log_path — now a bare relative name that resolves UNDER the
    # config dir at use (#35); the default single-instance setup still lands at ~/.localharness.
    assert e.current_value == "audit.jsonl"


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
    # #84: the deterministic baton-gate kill-switch auto-enumerates as a bool leaf too.
    assert "agent.baton_gate.enabled" in entries
    assert entries["agent.baton_gate.enabled"].annotation is bool
    # FIX 4: the nudge bound is a configurable int leaf (agent-scope only — BatonGateConfig is
    # not mirrored onto OrgConfig, unlike PermissionConfig/ContextConfig, so this is +1 not +2).
    assert "agent.baton_gate.max_nudges" in entries
    assert entries["agent.baton_gate.max_nudges"].annotation is int
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
    # Resonance rebuild (2026-09-19): the embedding-model axis is addressable; the
    # gate/chapter/mining knobs are GONE with their mechanisms.
    assert "agent.memory.embedding_model" in entries
    assert "agent.memory.consolidation.iteration_cap" in entries
    assert "agent.memory.archival.enabled" in entries

    # Tag-graph (Amendment 4): two more agent.memory.consolidation.* axes.
    # Phase 36.2 (RULING-D): tags-as-grouping-truth KILL-revert lever is registry-addressable.
    # FIX 3: mining chunk size + known-atoms window are now config knobs (two more axes).
    # FIX 4: mining's operative conversational surface (echo-collapse guard) is a config knob.
    # Residue ledger (core repair loop): enabled + K + per-pass record budget + intake filter.
    # Novelty gate (mining precision): paraphrase-fold threshold.
    # Embedding edge signal (clustering tier-1): cosine threshold for the 2-factor link.
    # Chapter refresh (run-14 fix): member-overlap threshold for identity adoption.
    # Chapter containment guard (validation-20260712 fix): the set-containment kill lever.
    # Absorption guard (d1v2 gpu_ops-into-subagents weld): the kill lever + its two overlap knobs.
    # Chapter staleness re-check (B5 / ANALYSIS §7 fix): the kill lever + its per-pass work cap.
    # issue #15: the opt-in confinement lever is registry-addressable.
    assert "agent.permissions.workspace_root" in entries
    # #62: the inference queue-wait ceiling is a provider-level tuning knob.
    assert "provider.inference_queue_wait_seconds" in entries
    # type-anytime input box: harness-level terminal switches are registry-addressable.
    assert "terminal.inputbox_enabled" in entries
    assert "terminal.input_router_tier2_enabled" in entries
    # #84 baton gate: the kill-switch is registry-addressable.
    assert "agent.baton_gate.enabled" in entries
    # FIX 3: the tool-dispatch cap (decoupled from max_actions) is registry-addressable,
    # org+agent scopes like permissions.workspace_root above.
    assert "agent.permissions.budget.max_tool_calls" in entries
    assert "org.permissions.budget.max_tool_calls" in entries
    assert "agent.baton_gate.max_nudges" in entries
    # #132: per-model context pins are registry-addressable, org+agent scopes like the levers above.
    assert "org.context.model_context_overrides" in entries
    assert "agent.context.model_context_overrides" in entries
    # #152: the degenerate-repetition guard's kill-switch and both thresholds are addressable.
    assert "agent.repetition_guard.enabled" in entries
    assert "agent.repetition_guard.min_lines" in entries
    assert "agent.repetition_guard.max_unique_ratio" in entries
    # Memory rung 1: the dormancy-archival rollout gate is registry-addressable.
    assert "agent.memory.archival.enabled" in entries
    assert len(entries) == 178, (
        "catalogue should be 178 entries: the 223-entry v0.14 ledger minus the resonance "
        "rebuild's removals (agent.memory.write_gate_enabled x1, "
        "agent.memory.predictive_gate.* x13, and 32 of the 36 agent.memory.consolidation.* "
        "knobs — chapters/mining/tags/micro-pass/decay/cap machinery deleted with their "
        "mechanisms) plus agent.memory.embedding_model x1 "
        "(the subject-family resonance space is owner-addressable). "
        f"got {len(entries)}"
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

    # #62: the inference queue-wait ceiling is a provider-level tuning knob.
    assert "provider.inference_queue_wait_seconds" in entries
    # type-anytime input box: harness-level terminal switches are registry-addressable.
    assert "terminal.inputbox_enabled" in entries
    assert "terminal.input_router_tier2_enabled" in entries
    # #84 baton gate: the kill-switch is registry-addressable.
    assert "agent.baton_gate.enabled" in entries
    assert "agent.baton_gate.max_nudges" in entries
    # #132: per-model context pins are registry-addressable, org+agent scopes like the levers above.
    assert "org.context.model_context_overrides" in entries
    assert "agent.context.model_context_overrides" in entries
    # #152: the degenerate-repetition guard's kill-switch and both thresholds are addressable.
    assert "agent.repetition_guard.enabled" in entries
    assert "agent.repetition_guard.min_lines" in entries
    assert "agent.repetition_guard.max_unique_ratio" in entries
    # Memory rung 1: the dormancy-archival rollout gate is registry-addressable.
    assert "agent.memory.archival.enabled" in entries
    assert len(entries) == 178, (
        "catalogue should be 178 entries: the 223-entry v0.14 ledger minus the resonance "
        "rebuild's removals (agent.memory.write_gate_enabled x1, "
        "agent.memory.predictive_gate.* x13, and 32 of the 36 agent.memory.consolidation.* "
        "knobs — chapters/mining/tags/micro-pass/decay/cap machinery deleted with their "
        "mechanisms) plus agent.memory.embedding_model x1 "
        "(the subject-family resonance space is owner-addressable). "
        f"got {len(entries)}"
    )


def test_role_sections_defaults_empty():
    """MODP-01 Test D: all four sections default to '' — the structural basis of byte-identity."""
    from localharness.config.models import AgentConfig

    a = AgentConfig(name="x", role="y")
    assert a.role_sections.identity == ""
    assert a.role_sections.tool_use == ""
    assert a.role_sections.stopping == ""
    assert a.role_sections.output == ""





# ------------------------------------------------------------------ #
# MERG-03 / v0.13: the ruled layer order (phase 40-03)
# ------------------------------------------------------------------ #


def test_layer_priority_records_the_ruled_order():
    """The tuple IS the owner's ruling, pinned as a value so a reorder is a deliberate act.

    Owner ruling 2026-09-03 (Option A): the workspace layer outranks the global overrides file
    (the SPECIFIC beats the GENERAL); `experiment` stays on top. 43-03 split each side into its
    two real FILES and renamed both global bands, so the tuple is five long.

    Pinned as LITERAL strings on purpose: these are the words `components list` prints, so an
    assertion written in terms of the constants alone would stay green while the user-visible
    vocabulary changed underneath it. The second assertion ties the constants to those literals,
    so a caller importing the constants and a user reading the output cannot drift apart.
    """
    from localharness.registry import catalogue

    assert catalogue._LAYER_PRIORITY == (
        "experiment",
        "workspace-overrides",
        "workspace-config",
        "global-overrides",
        "global-config",
    ), (
        "Owner ruling 2026-09-03 (Option A): workspace outranks the global layer, experiment "
        "stays on top, and within each layer the overrides file outranks the config file. "
        "Changing this tuple changes what `components list` tells a user owns their setting — "
        "reorder it only with a new ruling."
    )
    assert catalogue._LAYER_PRIORITY == (
        catalogue.LAYER_EXPERIMENT,
        catalogue.LAYER_WORKSPACE_OVERRIDES,
        catalogue.LAYER_WORKSPACE_CONFIG,
        catalogue.LAYER_GLOBAL_OVERRIDES,
        catalogue.LAYER_GLOBAL_CONFIG,
    ), "the exported constants must BE the tuple — one anchor, no second spelling"


def test_workspace_outranks_user_when_both_declare_a_path():
    """The ruled order proven behaviorally, not only as a tuple literal.

    Band names corrected in 43-03. The old keys (`workspace`, `user`) match no band after the
    rename, so `_detect_layer` would fall through to `default` and this test would grade nothing.
    """
    from localharness.registry import catalogue

    both = catalogue._detect_layer(
        "org.log_level",
        {
            catalogue.LAYER_WORKSPACE_CONFIG: {"org": {"log_level": "debug"}},
            catalogue.LAYER_GLOBAL_OVERRIDES: {"org": {"log_level": "info"}},
        },
    )
    assert both == catalogue.LAYER_WORKSPACE_CONFIG, (
        "the workspace config.yaml must win over the global overrides.yaml (Option A)"
    )

    top = catalogue._detect_layer(
        "org.log_level",
        {
            catalogue.LAYER_EXPERIMENT: {"org": {"log_level": "warning"}},
            catalogue.LAYER_WORKSPACE_CONFIG: {"org": {"log_level": "debug"}},
        },
    )
    assert top == catalogue.LAYER_EXPERIMENT, "experiment stays on top of workspace"


def test_absent_workspace_overlay_changes_nothing():
    """A band NAME in the priority tuple is inert until something populates that key.

    `_detect_layer` reads `overlays.get(layer, {})`, so an unpopulated band can never win. With
    no workspace up-tree `build_layer_overlays` returns the two GLOBAL keys only (43-03), which
    is what keeps a no-workspace session's attribution identical to pre-v0.13 (LAYR-03) — this
    asserts that property at the detector rather than leaving it as a comment.
    """
    from localharness.registry import catalogue

    assert catalogue._detect_layer(
        "org.log_level", {catalogue.LAYER_GLOBAL_CONFIG: {"org": {"log_level": "info"}}}
    ) == catalogue.LAYER_GLOBAL_CONFIG
    assert catalogue._detect_layer("org.log_level", {}) == catalogue.LAYER_DEFAULT
