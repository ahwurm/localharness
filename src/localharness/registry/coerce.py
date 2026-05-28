"""CLI-string → typed-value coercion for `components set`.

Dispatch on the TARGET annotation (Pitfall 4 in 14-RESEARCH.md). Refuses
cross-type coercion (e.g. `"False"` → float is rejected; Python's bool-is-int
inheritance would otherwise silently produce 0.0).
"""
from __future__ import annotations

from typing import Any, Union, get_args, get_origin


_TRUE_LITERALS = frozenset({"true", "1", "yes", "on"})
_FALSE_LITERALS = frozenset({"false", "0", "no", "off"})
_NULL_LITERALS = frozenset({"null", "none", "~"})
_BOOL_LITERALS = _TRUE_LITERALS | _FALSE_LITERALS


def _unwrap_optional(ann: Any) -> tuple[Any, bool]:
    """Return (inner_type, is_optional). Optional[X] → (X, True)."""
    if get_origin(ann) is Union:
        args = [a for a in get_args(ann) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return ann, False


def coerce_value(raw: str, annotation: Any) -> Any:
    """Coerce a CLI string to a typed value per the target annotation.

    Refuses bool-looking input for non-bool numeric targets (Pitfall 4).
    Optional[X] → returns None for raw in {"null", "none", "~"}, else coerces to X.
    Any → passthrough (returns raw unchanged).
    """
    inner, is_optional = _unwrap_optional(annotation)

    # Optional → null sentinels
    if is_optional and raw.strip().lower() in _NULL_LITERALS:
        return None

    # Any → passthrough
    if inner is Any:
        return raw

    # Refuse bool literal coerced to numeric
    if inner in (int, float) and raw.strip().lower() in _BOOL_LITERALS:
        raise ValueError(
            f"Refusing to coerce bool-like {raw!r} to {inner.__name__} "
            f"(target type is numeric — use a numeric value)"
        )

    if inner is bool:
        normalized = raw.strip().lower()
        if normalized in _TRUE_LITERALS:
            return True
        if normalized in _FALSE_LITERALS:
            return False
        raise ValueError(
            f"Cannot coerce {raw!r} to bool — expected one of "
            f"{sorted(_BOOL_LITERALS)}"
        )

    if inner is int:
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"Cannot coerce {raw!r} to int: {exc}") from exc

    if inner is float:
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"Cannot coerce {raw!r} to float: {exc}") from exc

    if inner is str:
        return raw

    # Fallback for complex types (Literal, dict, list...): best-effort passthrough
    return raw
