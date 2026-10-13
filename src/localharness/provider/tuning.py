"""What a launch derives from the model's own header and the machine's knobs (spec 20 §6.0-6.2).

Pure: this module reads nothing but its arguments (a later plan adds the one bounded config.json
reader for vLLM parsers), runs no process, and executes nothing from any file. `derive` decides the
K/V cache type and MTP drafting of a llama.cpp launch; it never writes a record and never raises on
a hostile header. Every derived field carries one reason (`Tuned.reasons`, keyed by field name) so
the launch can say why.

Precedence, for every derived flag: the record field > `extra_args` / `env:` in any spelling >
the derivation. A derivation is the fallback for a person who said nothing.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from localharness.provider.autosense import (
    flash_attn_off,
    has_cache_type_flag,
    has_spec_flag,
)

# llama-server's own default; deriving it would change the 0.18 argv, so it derives nothing (D-08).
_LLAMA_DEFAULT_CACHE_TYPE = "f16"


def _empty_reasons() -> Mapping[str, str]:
    return MappingProxyType({})


@dataclass(frozen=True)
class Tuned:
    """What `derive` decided for one launch. Every field is None when nothing was derived for it;
    `reasons` holds one phrase per derived field, keyed by the field's name."""

    cache_type: str | None = None
    spec_type: str | None = None
    draft_p_min: float | None = None
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None
    mtp_layers: int | None = None
    reasons: Mapping[str, str] = field(default_factory=_empty_reasons)

    def __post_init__(self) -> None:
        # A frozen record holds a read-only copy, whatever mapping the caller handed in.
        object.__setattr__(self, "reasons", MappingProxyType(dict(self.reasons)))


NOTHING_DERIVED = Tuned()


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def mtp_layers(header: Mapping[str, Any]) -> int | None:
    """The model's MTP layer count: `nextn_predict_layers`, only when `block_count` is an int above
    zero and the count is an int with 1 <= n <= block_count. A bool, a string, a negative, zero,
    a count above the block count or a missing key gives None; a hostile header never raises."""
    try:
        blocks, nextn = _int(header.get("block_count")), _int(header.get("nextn_predict_layers"))
    except AttributeError:
        return None
    if blocks is None or blocks <= 0 or nextn is None or not 1 <= nextn <= blocks:
        return None
    return nextn


def draft_share(header: Mapping[str, Any]) -> float:
    """The multiple of the main context's KV a drafting launch holds: the MTP draft context keeps
    its own KV for nextn_predict_layers layers, so `1 + nextn / block_count`; 1.0 when the header
    declares no valid MTP layers."""
    n = mtp_layers(header)
    return 1.0 if n is None else 1 + n / header["block_count"]


def derive(spec: Any, header: Mapping[str, Any], autosense: Any, *,
           model_config: Mapping[str, Any] | None = None) -> Tuned:
    """The flags a launch of `spec` derives (llama.cpp side; `model_config` is the vLLM side's input).

    `cache_type` is `autosense.kv_cache_type` unless the record sets a cache type, `extra_args` /
    `env` pin one in any spelling, flash attention is switched off, or the knob is `f16`.
    `spec_type` is `draft-mtp` (with `autosense.draft_p_min`) only when `autosense.draft_mtp` is on,
    the record's `speculative` is unset, no speculative flag is set by hand, and the header declares
    MTP layers. Any other runtime derives nothing here."""
    if getattr(spec, "runtime", None) != "llamacpp":
        return NOTHING_DERIVED
    extra = list(getattr(spec, "extra_args", None) or [])
    env = dict(getattr(spec, "env", None) or {})
    fields: dict[str, Any] = {}
    reasons: dict[str, str] = {}

    kv = autosense.kv_cache_type
    if (kv != _LLAMA_DEFAULT_CACHE_TYPE
            and spec.cache_type_k is None and spec.cache_type_v is None
            and not has_cache_type_flag(extra, env) and not flash_attn_off(extra, env)):
        fields["cache_type"] = kv
        reasons["cache_type"] = "autosense.kv_cache_type"

    n = mtp_layers(header)
    if (autosense.draft_mtp and spec.speculative is None and not has_spec_flag(extra, env)
            and n is not None):
        fields.update(spec_type="draft-mtp", draft_p_min=autosense.draft_p_min, mtp_layers=n)
        reasons["spec_type"] = f"nextn_predict_layers={n}"

    return Tuned(reasons=reasons, **fields) if fields else NOTHING_DERIVED
