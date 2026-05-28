"""Phase 14 Wave 0 scaffolding for localharness.config.overlay.

Covers requirement REG-03 (atomic write + deep_merge + load round-trip).
Every test is xfail-marked until Phase 14-02 lands overlay.py.
"""
from __future__ import annotations

import pytest


@pytest.mark.xfail(reason="Phase 14-02 overlay.py not yet implemented", strict=False)
def test_atomic_write_no_torn_file(components_home):
    """REG-03: 100 sequential writes with concurrent reads never produce torn YAML."""
    try:
        from localharness.config.overlay import atomic_write_overlay, load_overlay
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-02")
    # Implementer must run a tight write/read loop and assert load_overlay always parses
    raise NotImplementedError("Stub for 14-02")


@pytest.mark.xfail(reason="Phase 14-02 overlay.py not yet implemented", strict=False)
def test_deep_merge_scalars_replace_recurse_dicts():
    """deep_merge replaces scalars and recurses into dicts."""
    try:
        from localharness.config.overlay import deep_merge
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-02")
    result = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 99}})
    assert result == {"a": {"b": 99, "c": 2}}
    raise NotImplementedError("Stub for 14-02")


@pytest.mark.xfail(reason="Phase 14-02 overlay.py not yet implemented", strict=False)
def test_load_overlay_missing_file_returns_empty_dict(components_home):
    """load_overlay returns {} when overlay file does not exist."""
    try:
        from localharness.config.overlay import load_overlay
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-02")
    overlay_path = components_home / "overrides.yaml"
    assert not overlay_path.exists()
    assert load_overlay(overlay_path) == {}
    raise NotImplementedError("Stub for 14-02")


@pytest.mark.xfail(reason="Phase 14-02 overlay.py not yet implemented", strict=False)
def test_load_overlay_round_trip(components_home):
    """atomic_write_overlay then load_overlay returns identical dict."""
    try:
        from localharness.config.overlay import atomic_write_overlay, load_overlay
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-02")
    overlay_path = components_home / "overrides.yaml"
    data = {"agent": {"stuck_detector": {"window_size": 7}}}
    atomic_write_overlay(overlay_path, data)
    restored = load_overlay(overlay_path)
    assert restored == data
    raise NotImplementedError("Stub for 14-02")


@pytest.mark.xfail(reason="Phase 14-02 overlay.py not yet implemented", strict=False)
def test_atomic_write_uses_same_dir_tempfile(components_home, monkeypatch):
    """NamedTemporaryFile must receive dir=parent so os.replace stays atomic."""
    try:
        from localharness.config.overlay import atomic_write_overlay
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-02")
    import tempfile

    captured_dirs = []
    original = tempfile.NamedTemporaryFile

    def _spy(*args, **kwargs):
        captured_dirs.append(kwargs.get("dir"))
        return original(*args, **kwargs)

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", _spy)
    overlay_path = components_home / "overrides.yaml"
    atomic_write_overlay(overlay_path, {"a": 1})
    assert captured_dirs and captured_dirs[-1] == str(overlay_path.parent)
    raise NotImplementedError("Stub for 14-02")
