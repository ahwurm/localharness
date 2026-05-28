"""Phase 14 Wave 0 scaffolding for localharness.registry.catalogue.

Covers REG-04 (six-surface presence) + layer attribution.
Every test is xfail-marked until Phase 14-03 wires the catalogue.
"""
from __future__ import annotations

import pytest


@pytest.mark.xfail(reason="Phase 14-03 registry/catalogue.py not yet implemented", strict=False)
def test_six_distinct_surface_types(components_home):
    """REG-04: catalogue exposes ≥6 distinct top-level surface families.

    Surfaces enumerated:
      1. agent.role / agent.context.system_prompt_file
      2. tools.*.description
      3. org.context.compaction_threshold_pct
      4. agent.stuck_detector.window_size
      5. agent.recovery_injection.message
      6. hooks.*.config
    """
    try:
        from localharness.registry.catalogue import build_catalogue
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-03")
    catalogue = build_catalogue(cfg=None, overlays={})
    # Implementer asserts ≥6 distinct top-level surface families
    raise NotImplementedError("Stub for 14-03")


@pytest.mark.xfail(reason="Phase 14-03 registry/catalogue.py not yet implemented", strict=False)
def test_required_surfaces_present(components_home):
    """REG-04: each named required surface appears in the catalogue."""
    try:
        from localharness.registry.catalogue import build_catalogue
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-03")
    catalogue = build_catalogue(cfg=None, overlays={})
    required = {
        "org.context.compaction_threshold_pct",
        "agent.stuck_detector.window_size",
        "agent.recovery_injection.message",
    }
    # Implementer asserts each required path exists in catalogue keys
    raise NotImplementedError("Stub for 14-03")


@pytest.mark.xfail(reason="Phase 14-03 registry/catalogue.py not yet implemented", strict=False)
def test_layer_attribution_default_when_no_overlay(components_home):
    """With no overlay, every entry's winning_layer is 'default' or 'project'."""
    try:
        from localharness.registry.catalogue import build_catalogue
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-03")
    catalogue = build_catalogue(cfg=None, overlays={})
    # Implementer asserts each entry.winning_layer in {"default", "project"}
    raise NotImplementedError("Stub for 14-03")


@pytest.mark.xfail(reason="Phase 14-03 registry/catalogue.py not yet implemented", strict=False)
def test_layer_attribution_user_when_overlay_present(components_home):
    """Overlay sets a path → that path's entry has winning_layer == 'user'."""
    try:
        from localharness.registry.catalogue import build_catalogue
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-03")
    overlays = {"user": {"org": {"context": {"compaction_threshold_pct": 0.85}}}}
    catalogue = build_catalogue(cfg=None, overlays=overlays)
    # Implementer asserts the targeted path's winning_layer == "user"
    raise NotImplementedError("Stub for 14-03")
