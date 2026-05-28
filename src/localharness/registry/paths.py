"""Pydantic-tree walker + dot-path get/set primitives.

walk_model_fields(cls) — enumerate every leaf (dot_path, annotation) by recursing
    cls.model_fields. Stops at non-BaseModel types (list/dict/scalar = leaf).
get_value(root, path) — functools.reduce(getattr, parts, root); raises AttributeError.
set_value_in_dict(d, path, value) — create-or-update nested key; preserves siblings.

See 14-RESEARCH.md Patterns 1, 2, 3 + Pitfall 1 (always use class-level model_fields).
"""
from __future__ import annotations

from functools import reduce
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel


def _unwrap_optional(ann: Any) -> Any:
    """Optional[X] == Union[X, None] → X. Leaves non-Optional unchanged."""
    if get_origin(ann) is Union:
        args = [a for a in get_args(ann) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return ann


def walk_model_fields(
    model_cls: type[BaseModel],
    prefix: str = "",
) -> list[tuple[str, Any]]:
    """Recursively enumerate (dot_path, annotation) for every leaf field.

    Stops descent at:
      - non-BaseModel leaf types (str, int, float, bool, etc.)
      - list[X] (treated as leaf — per-index addressing deferred)
      - dict[K, V] (treated as leaf — dynamic keys, not pre-enumerable)
      - Union[A, B] non-Optional (skipped with no path emitted; logged warning)

    IMPORTANT: uses class-level `model_cls.model_fields`. Instance-level access
    deprecated in pydantic 2.11, removed in v3 (see 14-RESEARCH.md Pitfall 1).
    """
    out: list[tuple[str, Any]] = []
    for name, info in model_cls.model_fields.items():
        ann = info.annotation
        path = f"{prefix}.{name}" if prefix else name

        inner = _unwrap_optional(ann)

        # Skip ambiguous non-Optional unions silently for now (no such fields today)
        if get_origin(inner) is Union:
            # Multi-member union (not Optional) — can't disambiguate. Emit as leaf.
            out.append((path, ann))
            continue

        # Descend into nested BaseModel; stop at dict/list/scalar leaves
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            out.extend(walk_model_fields(inner, prefix=path))
        else:
            out.append((path, ann))
    return out


def get_value(root: Any, path: str) -> Any:
    """Walk dot-path against any object using getattr.

    Raises AttributeError if any segment is missing — caller maps this to a
    CLI exit code (see plan 14-04, components_cmd.components_get).
    """
    return reduce(getattr, path.split("."), root)


def set_value_in_dict(d: dict, path: str, value: Any) -> dict:
    """Create-or-update nested key in dict. Returns d (mutated in place).

    Overwrites scalar leaves with nested dicts when path forces it (e.g.
    set_value_in_dict({"a": 1}, "a.b", 2) → {"a": {"b": 2}}).
    Sibling keys at every level are preserved.
    """
    parts = path.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value
    return d
