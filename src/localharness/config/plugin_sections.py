"""Plugin-owned settings, routed out BEFORE core validation (ENAB-01, PRD decision 8).

Every core model is extra="forbid", so a plugin's `<name>:` (harness level) and `agent.<name>` (agent
level) keys are split off the merged config first; the plugin validates its own subtree with its own
ConfigModel / AgentConfigModel once it is loaded. A key that is neither a core field nor a known
plugin still fails validation — the typo guard stays. Global-only fields (network endpoints,
credentials, access lists — ENAB-02) are narrowed here, the AskConfig.mcp_trusted_servers rule
applied generically: a workspace-layer value is dropped with a warning and the global value stands.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from types import UnionType
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from localharness.config.models import AgentConfig, HarnessConfig
from localharness.config.overlay import deep_merge
from localharness.plugins.api import GLOBAL_ONLY_KEY

CORE_HARNESS_KEYS: frozenset[str] = frozenset(HarnessConfig.model_fields) | {"agent"}
"""HarnessConfig's top-level keys plus `agent:`, the overrides file's agent-default section."""
CORE_AGENT_KEYS: frozenset[str] = frozenset(AgentConfig.model_fields)

_WORKSPACE_LAYERS = (2, 3)  # of the four ruled sources: workspace config.yaml, workspace overrides
_MISSING = object()


class PluginSectionError(ValueError):
    """A plugin's settings section, in some layer, is not a mapping."""


def split_plugin_keys(data: Mapping[str, Any], names: Iterable[str],
                      core_keys: frozenset[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(data without plugin sections, {name: section}) — never mutates `data`; a name in
    `core_keys` is never split (the plan refuses such a plugin)."""
    owned = frozenset(names) - core_keys
    return ({k: v for k, v in data.items() if k not in owned},
            {k: v for k, v in data.items() if k in owned})


def global_only_paths(model: type[BaseModel] | None, prefix: str = "",
                      _chain: tuple[type[BaseModel], ...] = ()) -> frozenset[str]:
    """Dot-paths of the fields a plugin marked GLOBAL_ONLY (by name and by alias), recursing into
    nested models. A marked field no dot-path can reach — inside a list or dict element, one arm of
    a union, behind a recursive reference — makes the field holding it global-only as a whole."""
    if model is None:
        return frozenset()
    chain = (*_chain, model)
    out: set[str] = set()
    for name, field in model.model_fields.items():
        keys = {name} | {a for a in (field.alias, field.validation_alias) if isinstance(a, str)}
        nested = _direct_model(field.annotation)
        if _marked(field):
            out |= {prefix + k for k in keys}
        elif nested is not None and nested not in chain:
            for k in keys:
                out |= global_only_paths(nested, f"{prefix}{k}.", chain)
        elif any(_holds_marked(m) for m in _models_in(field.annotation)):
            out |= {prefix + k for k in keys}
    return frozenset(out)


def merge_plugin_layers(name: str, layers: Sequence[Any], *, global_only: Iterable[str],
                        layer_files: Sequence[str]) -> tuple[dict[str, Any], list[str]]:
    """Merge a plugin's four raw subtrees (global config, global overrides, workspace config,
    workspace overrides — the ruled order); drop workspace values at global-only paths with a
    warning; raise PluginSectionError if a layer's section is not a mapping. A workspace value
    equal to the global one is the operator's own and stays. Never mutates `layers`."""
    sections = [_section(name, s, f) for s, f in zip(layers, layer_files, strict=True)]
    global_view = deep_merge(sections[0], sections[1])
    warnings: list[str] = []
    for i in _WORKSPACE_LAYERS:
        for path in sorted(global_only):
            parts = path.split(".")
            depth = _reach(sections[i], parts)
            if not depth or _get(sections[i], parts[:depth]) == _get(global_view, parts[:depth]):
                continue
            sections[i] = _drop(sections[i], parts[:depth])
            replaced = ".".join(parts[:depth])
            warnings.append(
                f"ignoring {name}.{path} in {layer_files[i]}: only the global config may set it"
                if depth == len(parts) else
                f"ignoring {name}.{replaced} in {layer_files[i]}: it would replace {name}.{path}, "
                f"which only the global config may set")
    merged: dict[str, Any] = {}
    for section in sections:
        merged = deep_merge(merged, section)
    return merged, warnings


def _section(name: str, section: Any, file: str) -> dict[str, Any]:
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise PluginSectionError(
            f"`{name}:` in {file} must be a mapping of settings, not {type(section).__name__}")
    return dict(section)


def _marked(field: FieldInfo) -> bool:
    extra = field.json_schema_extra
    return isinstance(extra, dict) and bool(extra.get(GLOBAL_ONLY_KEY))


def _direct_model(annotation: Any) -> type[BaseModel] | None:
    """The model an annotation IS — `Model` or `Optional[Model]` — else None."""
    if get_origin(annotation) in (Union, UnionType):
        arms = [a for a in get_args(annotation) if a is not type(None)]
        annotation = arms[0] if len(arms) == 1 else None
    return annotation if isinstance(annotation, type) and issubclass(annotation, BaseModel) else None


def _models_in(annotation: Any) -> list[type[BaseModel]]:
    """Every model an annotation mentions anywhere (list/dict elements, union arms)."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    return [m for arg in get_args(annotation) for m in _models_in(arg)]


def _holds_marked(model: type[BaseModel], seen: frozenset[type[BaseModel]] = frozenset()) -> bool:
    """Whether `model`, or a model inside it, has a GLOBAL_ONLY field (cycle-safe)."""
    seen = seen | {model}
    return any(_marked(f) or any(_holds_marked(m, seen) for m in _models_in(f.annotation)
                                 if m not in seen)
               for f in model.model_fields.values())


def _reach(d: Any, parts: list[str]) -> int:
    """How far a layer reaches along `parts`: len(parts) if it sets the leaf, k if it sets the
    k-th key to something that is not a mapping (replacing the subtree below), 0 if it is silent."""
    for k, key in enumerate(parts, 1):
        if not isinstance(d, Mapping) or key not in d:
            return 0
        d = d[key]
        if k < len(parts) and not isinstance(d, Mapping):
            return k
    return len(parts)


def _get(d: Any, parts: list[str]) -> Any:
    for key in parts:
        if not isinstance(d, Mapping) or key not in d:
            return _MISSING
        d = d[key]
    return d


def _drop(d: Mapping[str, Any], parts: list[str]) -> dict[str, Any]:
    """A copy of `d` without the key at `parts`, copying only along the path."""
    head, *rest = parts
    if not rest:
        return {k: v for k, v in d.items() if k != head}
    return {**d, head: _drop(d[head], rest)}
