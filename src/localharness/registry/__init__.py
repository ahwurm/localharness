"""Component registry: enumerate, resolve, mutate harness components by dot path.

Phase 14 ships the primitives (walk_model_fields, get_value, set_value_in_dict,
coerce_value) and the catalogue builder (plan 14-03). CLI surface in plan 14-04.
"""
from localharness.registry.catalogue import (
    ComponentEntry,
    SURFACE_FAMILIES,
    build_catalogue,
)
from localharness.registry.coerce import coerce_value
from localharness.registry.paths import (
    get_value,
    set_value_in_dict,
    walk_model_fields,
)

__all__ = [
    "walk_model_fields",
    "get_value",
    "set_value_in_dict",
    "coerce_value",
    "ComponentEntry",
    "build_catalogue",
    "SURFACE_FAMILIES",
]
