"""Phase 14 Wave 0 scaffolding for localharness.registry.paths + coerce.

Covers walk_model_fields / get_value / set_value_in_dict / coerce_value
(supporting REG-01..04). Every test is xfail-marked until Phase 14-02 lands.
"""
from __future__ import annotations

import pytest


@pytest.mark.xfail(reason="Phase 14-02 registry/paths.py not yet implemented", strict=False)
def test_walk_model_fields_enumerates_harness_config_leaves():
    """walk_model_fields(HarnessConfig) yields all leaf dot-paths."""
    try:
        from localharness.config.models import HarnessConfig
        from localharness.registry.paths import walk_model_fields
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-02")
    paths = {p for p, _ann in walk_model_fields(HarnessConfig)}
    assert "provider.provider_type" in paths
    assert "org.context.compaction_threshold_pct" in paths
    assert "org.audit_log_path" in paths
    raise NotImplementedError("Stub for 14-02")


@pytest.mark.xfail(reason="Phase 14-02 registry/paths.py not yet implemented", strict=False)
def test_walk_model_fields_descends_nested_models():
    """walk_model_fields recurses into nested BaseModel fields."""
    try:
        from localharness.config.models import HarnessConfig
        from localharness.registry.paths import walk_model_fields
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-02")
    paths = {p for p, _ann in walk_model_fields(HarnessConfig)}
    # At least one dot-nested path must exist (nested-model descent)
    assert any("." in p for p in paths)
    raise NotImplementedError("Stub for 14-02")


@pytest.mark.xfail(reason="Phase 14-02 registry/paths.py not yet implemented", strict=False)
def test_walk_model_fields_treats_list_of_str_as_leaf():
    """list[str] fields are leaves; the walker MUST NOT descend into them."""
    try:
        from localharness.config.models import HarnessConfig
        from localharness.registry.paths import walk_model_fields
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-02")
    paths = {p for p, _ann in walk_model_fields(HarnessConfig)}
    # Any path that touches deny_patterns must not be indexed past the leaf
    assert not any(p.startswith("permissions.deny_patterns.") for p in paths)
    raise NotImplementedError("Stub for 14-02")


@pytest.mark.xfail(reason="Phase 14-02 registry/paths.py not yet implemented", strict=False)
def test_get_value_resolves_dot_path():
    """get_value walks getattr-style dot path against an object."""
    try:
        from localharness.config.models import HarnessConfig
        from localharness.registry.paths import get_value
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-02")
    cfg = HarnessConfig.model_validate({
        "version": "1",
        "provider": {"provider_type": "ollama", "base_url": "http://x", "default_model": "m"},
    })
    val = get_value(cfg, "org.context.compaction_threshold_pct")
    assert isinstance(val, float)
    raise NotImplementedError("Stub for 14-02")


@pytest.mark.xfail(reason="Phase 14-02 registry/paths.py not yet implemented", strict=False)
def test_get_value_unknown_path_raises_attribute_error():
    """get_value on a path that doesn't exist raises AttributeError."""
    try:
        from localharness.config.models import HarnessConfig
        from localharness.registry.paths import get_value
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-02")
    cfg = HarnessConfig.model_validate({
        "version": "1",
        "provider": {"provider_type": "ollama", "base_url": "http://x", "default_model": "m"},
    })
    with pytest.raises(AttributeError):
        get_value(cfg, "totally.bogus.path")
    raise NotImplementedError("Stub for 14-02")


@pytest.mark.xfail(reason="Phase 14-02 registry/paths.py not yet implemented", strict=False)
def test_set_value_in_dict_creates_nested_keys():
    """set_value_in_dict creates missing intermediate dict layers."""
    try:
        from localharness.registry.paths import set_value_in_dict
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-02")
    result = set_value_in_dict({}, "a.b.c", 5)
    assert result == {"a": {"b": {"c": 5}}}
    raise NotImplementedError("Stub for 14-02")


@pytest.mark.xfail(reason="Phase 14-02 registry/paths.py not yet implemented", strict=False)
def test_set_value_in_dict_preserves_siblings():
    """Setting a nested key never drops sibling entries."""
    try:
        from localharness.registry.paths import set_value_in_dict
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-02")
    base = {"a": {"b": 1, "c": 2}}
    set_value_in_dict(base, "a.b", 99)
    assert base == {"a": {"b": 99, "c": 2}}
    raise NotImplementedError("Stub for 14-02")


@pytest.mark.xfail(reason="Phase 14-02 registry/coerce.py not yet implemented", strict=False)
def test_coerce_value_int():
    """coerce_value('5', int) returns 5."""
    try:
        from localharness.registry.coerce import coerce_value
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-02")
    assert coerce_value("5", int) == 5
    raise NotImplementedError("Stub for 14-02")


@pytest.mark.xfail(reason="Phase 14-02 registry/coerce.py not yet implemented", strict=False)
def test_coerce_value_float():
    """coerce_value('0.75', float) returns 0.75."""
    try:
        from localharness.registry.coerce import coerce_value
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-02")
    assert coerce_value("0.75", float) == 0.75
    raise NotImplementedError("Stub for 14-02")


@pytest.mark.xfail(reason="Phase 14-02 registry/coerce.py not yet implemented", strict=False)
def test_coerce_value_bool():
    """coerce_value handles canonical truthy/falsey strings against bool target."""
    try:
        from localharness.registry.coerce import coerce_value
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-02")
    assert coerce_value("true", bool) is True
    assert coerce_value("false", bool) is False
    raise NotImplementedError("Stub for 14-02")


@pytest.mark.xfail(reason="Phase 14-02 registry/coerce.py not yet implemented", strict=False)
def test_coerce_value_rejects_bool_for_float_target():
    """Pitfall 4: 'False' must NOT silently coerce to 0.0 when target is float."""
    try:
        from localharness.registry.coerce import coerce_value
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-02")
    with pytest.raises(ValueError):
        coerce_value("False", float)
    raise NotImplementedError("Stub for 14-02")


@pytest.mark.xfail(reason="Phase 14-02 registry/coerce.py not yet implemented", strict=False)
def test_coerce_value_str_passthrough():
    """coerce_value('hello', str) returns 'hello' unchanged."""
    try:
        from localharness.registry.coerce import coerce_value
    except ImportError:
        pytest.xfail("scaffolded; awaiting plan 14-02")
    assert coerce_value("hello", str) == "hello"
    raise NotImplementedError("Stub for 14-02")
