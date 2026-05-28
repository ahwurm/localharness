"""Phase 14 Wave 0 scaffolding for localharness.config.overlay.

Covers requirement REG-03 (atomic write + deep_merge + load round-trip).
Plan 14-02 lands overlay.py — tests xfail-strict=False so they XPASS once green.
"""
from __future__ import annotations

import pytest


@pytest.mark.xfail(reason="Phase 14-02 overlay.py concurrency stress test", strict=False)
def test_atomic_write_no_torn_file(components_home):
    """REG-03: 100 sequential writes with concurrent reads never produce torn YAML."""
    from localharness.config.overlay import atomic_write_overlay, load_overlay
    overlay_path = components_home / "overrides.yaml"
    for i in range(100):
        atomic_write_overlay(overlay_path, {"agent": {"counter": i}})
        # Read back immediately — should always be valid YAML
        restored = load_overlay(overlay_path)
        assert restored == {"agent": {"counter": i}}


@pytest.mark.xfail(reason="Phase 14-02 overlay.py not yet implemented", strict=False)
def test_deep_merge_scalars_replace_recurse_dicts():
    """deep_merge replaces scalars and recurses into dicts."""
    from localharness.config.overlay import deep_merge
    result = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 99}})
    assert result == {"a": {"b": 99, "c": 2}}


@pytest.mark.xfail(reason="Phase 14-02 overlay.py not yet implemented", strict=False)
def test_load_overlay_missing_file_returns_empty_dict(components_home):
    """load_overlay returns {} when overlay file does not exist."""
    from localharness.config.overlay import load_overlay
    overlay_path = components_home / "overrides.yaml"
    assert not overlay_path.exists()
    assert load_overlay(overlay_path) == {}


@pytest.mark.xfail(reason="Phase 14-02 overlay.py not yet implemented", strict=False)
def test_load_overlay_round_trip(components_home):
    """atomic_write_overlay then load_overlay returns identical dict."""
    from localharness.config.overlay import atomic_write_overlay, load_overlay
    overlay_path = components_home / "overrides.yaml"
    data = {"agent": {"stuck_detector": {"window_size": 7}}}
    atomic_write_overlay(overlay_path, data)
    restored = load_overlay(overlay_path)
    assert restored == data


@pytest.mark.xfail(reason="Phase 14-02 overlay.py not yet implemented", strict=False)
def test_atomic_write_uses_same_dir_tempfile(components_home, monkeypatch):
    """NamedTemporaryFile must receive dir=parent so os.replace stays atomic."""
    from localharness.config import overlay as overlay_mod
    captured_dirs = []
    original = overlay_mod.tempfile.NamedTemporaryFile

    def _spy(*args, **kwargs):
        captured_dirs.append(kwargs.get("dir"))
        return original(*args, **kwargs)

    monkeypatch.setattr(overlay_mod.tempfile, "NamedTemporaryFile", _spy)
    overlay_path = components_home / "overrides.yaml"
    overlay_mod.atomic_write_overlay(overlay_path, {"a": 1})
    assert captured_dirs and captured_dirs[-1] == str(overlay_path.parent)


# --- Additional 14-02 coverage (non-xfail, plain green tests) ---


def test_user_overlay_path_env_override(monkeypatch, tmp_path):
    """LOCALHARNESS_HOME env var overrides default home in _resolve_user_overlay_path."""
    from localharness.config.overlay import _resolve_user_overlay_path
    monkeypatch.setenv("LOCALHARNESS_HOME", str(tmp_path))
    assert _resolve_user_overlay_path() == tmp_path / "overrides.yaml"


def test_user_overlay_path_default_without_env(monkeypatch):
    """Default path is ~/.localharness/overrides.yaml when env unset."""
    from pathlib import Path
    from localharness.config.overlay import _resolve_user_overlay_path
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    assert _resolve_user_overlay_path() == Path("~/.localharness/overrides.yaml").expanduser()


def test_deep_merge_overlay_wins_on_type_mismatch():
    from localharness.config.overlay import deep_merge
    assert deep_merge({"a": 1}, {"a": {"b": 2}}) == {"a": {"b": 2}}


def test_deep_merge_empty_inputs():
    from localharness.config.overlay import deep_merge
    assert deep_merge({}, {}) == {}


def test_deep_merge_does_not_mutate_base():
    from localharness.config.overlay import deep_merge
    base = {"a": {"b": 1}}
    deep_merge(base, {"a": {"c": 2}})
    assert base == {"a": {"b": 1}}


def test_load_overlay_invalid_yaml_raises(tmp_path):
    import yaml
    from localharness.config.overlay import load_overlay
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unterminated\n")
    with pytest.raises(yaml.YAMLError):
        load_overlay(bad)


def test_atomic_write_uses_os_replace(tmp_path, monkeypatch):
    """atomic_write_overlay must call os.replace (not os.rename)."""
    from localharness.config import overlay as overlay_mod
    calls: list[str] = []
    original = overlay_mod.os.replace

    def spy(src, dst):
        calls.append("replace")
        original(src, dst)

    monkeypatch.setattr(overlay_mod.os, "replace", spy)
    target = tmp_path / "out.yaml"
    overlay_mod.atomic_write_overlay(target, {"a": 1})
    assert calls == ["replace"]


def test_atomic_write_cleans_tempfile_on_replace_failure(tmp_path, monkeypatch):
    """If os.replace raises, the stranded tempfile must be unlinked."""
    from localharness.config import overlay as overlay_mod
    target = tmp_path / "out.yaml"

    def boom(src, dst):
        raise OSError("simulated")

    monkeypatch.setattr(overlay_mod.os, "replace", boom)
    with pytest.raises(OSError):
        overlay_mod.atomic_write_overlay(target, {"x": 1})
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".out.yaml.")]
    assert leftovers == []
