"""Phase 14 Wave 0 scaffolding for localharness.registry.paths + coerce.

Covers walk_model_fields / get_value / set_value_in_dict / coerce_value
(supporting REG-01..04). Tests xfail-strict=False so they XPASS once green.
"""
from __future__ import annotations

import pytest


@pytest.mark.xfail(reason="Phase 14-02 registry/paths.py not yet implemented", strict=False)
def test_walk_model_fields_enumerates_harness_config_leaves():
    """walk_model_fields(HarnessConfig) yields all leaf dot-paths."""
    from localharness.config.models import HarnessConfig
    from localharness.registry.paths import walk_model_fields
    paths = {p for p, _ann in walk_model_fields(HarnessConfig)}
    assert "provider.provider_type" in paths
    assert "org.context.compaction_threshold_pct" in paths
    assert "org.audit_log_path" in paths


@pytest.mark.xfail(reason="Phase 14-02 registry/paths.py not yet implemented", strict=False)
def test_walk_model_fields_descends_nested_models():
    """walk_model_fields recurses into nested BaseModel fields."""
    from localharness.config.models import HarnessConfig
    from localharness.registry.paths import walk_model_fields
    paths = {p for p, _ann in walk_model_fields(HarnessConfig)}
    assert any("." in p for p in paths)


@pytest.mark.xfail(reason="Phase 14-02 registry/paths.py not yet implemented", strict=False)
def test_walk_model_fields_treats_list_of_str_as_leaf():
    """list[str] fields are leaves; the walker MUST NOT descend into them."""
    from localharness.config.models import HarnessConfig
    from localharness.registry.paths import walk_model_fields
    paths = {p for p, _ann in walk_model_fields(HarnessConfig)}
    assert not any(p.startswith("permissions.deny_patterns.") for p in paths)


@pytest.mark.xfail(reason="Phase 14-02 registry/paths.py not yet implemented", strict=False)
def test_get_value_resolves_dot_path():
    """get_value walks getattr-style dot path against an object."""
    from localharness.config.models import HarnessConfig
    from localharness.registry.paths import get_value
    cfg = HarnessConfig.model_validate({
        "version": "1",
        "provider": {"provider_type": "ollama", "base_url": "http://x", "default_model": "m"},
    })
    val = get_value(cfg, "org.context.compaction_threshold_pct")
    assert isinstance(val, float)


@pytest.mark.xfail(reason="Phase 14-02 registry/paths.py not yet implemented", strict=False)
def test_get_value_unknown_path_raises_attribute_error():
    """get_value on a path that doesn't exist raises AttributeError."""
    from localharness.config.models import HarnessConfig
    from localharness.registry.paths import get_value
    cfg = HarnessConfig.model_validate({
        "version": "1",
        "provider": {"provider_type": "ollama", "base_url": "http://x", "default_model": "m"},
    })
    with pytest.raises(AttributeError):
        get_value(cfg, "totally.bogus.path")


@pytest.mark.xfail(reason="Phase 14-02 registry/paths.py not yet implemented", strict=False)
def test_set_value_in_dict_creates_nested_keys():
    """set_value_in_dict creates missing intermediate dict layers."""
    from localharness.registry.paths import set_value_in_dict
    result = set_value_in_dict({}, "a.b.c", 5)
    assert result == {"a": {"b": {"c": 5}}}


@pytest.mark.xfail(reason="Phase 14-02 registry/paths.py not yet implemented", strict=False)
def test_set_value_in_dict_preserves_siblings():
    """Setting a nested key never drops sibling entries."""
    from localharness.registry.paths import set_value_in_dict
    base = {"a": {"b": 1, "c": 2}}
    set_value_in_dict(base, "a.b", 99)
    assert base == {"a": {"b": 99, "c": 2}}


@pytest.mark.xfail(reason="Phase 14-02 registry/coerce.py not yet implemented", strict=False)
def test_coerce_value_int():
    """coerce_value('5', int) returns 5."""
    from localharness.registry.coerce import coerce_value
    assert coerce_value("5", int) == 5


@pytest.mark.xfail(reason="Phase 14-02 registry/coerce.py not yet implemented", strict=False)
def test_coerce_value_float():
    """coerce_value('0.75', float) returns 0.75."""
    from localharness.registry.coerce import coerce_value
    assert coerce_value("0.75", float) == 0.75


@pytest.mark.xfail(reason="Phase 14-02 registry/coerce.py not yet implemented", strict=False)
def test_coerce_value_bool():
    """coerce_value handles canonical truthy/falsey strings against bool target."""
    from localharness.registry.coerce import coerce_value
    assert coerce_value("true", bool) is True
    assert coerce_value("false", bool) is False


@pytest.mark.xfail(reason="Phase 14-02 registry/coerce.py not yet implemented", strict=False)
def test_coerce_value_rejects_bool_for_float_target():
    """Pitfall 4: 'False' must NOT silently coerce to 0.0 when target is float."""
    from localharness.registry.coerce import coerce_value
    with pytest.raises(ValueError):
        coerce_value("False", float)


@pytest.mark.xfail(reason="Phase 14-02 registry/coerce.py not yet implemented", strict=False)
def test_coerce_value_str_passthrough():
    """coerce_value('hello', str) returns 'hello' unchanged."""
    from localharness.registry.coerce import coerce_value
    assert coerce_value("hello", str) == "hello"


# --- Additional 14-02 coverage (non-xfail, plain green tests) ---


def test_set_value_in_dict_overwrites_scalar_with_nested():
    """Overwrite scalar leaf with nested dict when path forces it."""
    from localharness.registry.paths import set_value_in_dict
    assert set_value_in_dict({"a": 1}, "a.b", 2) == {"a": {"b": 2}}


def test_walk_model_fields_uses_class_level_model_fields():
    """Walker uses model_cls.model_fields (class-level, not instance — Pitfall 1)."""
    from localharness.config.models import HarnessConfig
    from localharness.registry.paths import walk_model_fields
    # Must accept the class directly, not require an instance
    result = walk_model_fields(HarnessConfig)
    assert isinstance(result, list)
    assert len(result) > 0


def test_walk_model_fields_treats_dict_as_leaf():
    """dict[K, V] fields are leaves; walker MUST NOT descend into them."""
    from localharness.config.models import HarnessConfig
    from localharness.registry.paths import walk_model_fields
    paths = {p for p, _ann in walk_model_fields(HarnessConfig)}
    # org.hooks is dict[str, dict[str, Any]] — leaf, not descended
    assert "org.hooks" in paths
    assert not any(p.startswith("org.hooks.") for p in paths)


def test_walk_model_fields_unwraps_optional():
    """Optional[X] should descend per the unwrapped X when X is a BaseModel."""
    from localharness.config.models import HarnessConfig
    from localharness.registry.paths import walk_model_fields
    paths = {p for p, _ann in walk_model_fields(HarnessConfig)}
    # provider.supports_function_calling is Optional[bool] — should be a leaf
    assert "provider.supports_function_calling" in paths


def test_coerce_value_bool_1_and_0():
    from localharness.registry.coerce import coerce_value
    assert coerce_value("1", bool) is True
    assert coerce_value("0", bool) is False


def test_coerce_value_optional_unwraps():
    from typing import Optional
    from localharness.registry.coerce import coerce_value
    assert coerce_value("5", Optional[int]) == 5


def test_coerce_value_optional_null_sentinels():
    from typing import Optional
    from localharness.registry.coerce import coerce_value
    assert coerce_value("null", Optional[int]) is None
    assert coerce_value("none", Optional[int]) is None
    assert coerce_value("~", Optional[int]) is None


def test_coerce_value_bad_int_raises():
    from localharness.registry.coerce import coerce_value
    with pytest.raises(ValueError):
        coerce_value("not_a_number", int)


def test_coerce_value_any_passthrough():
    from typing import Any
    from localharness.registry.coerce import coerce_value
    assert coerce_value("any", Any) == "any"
