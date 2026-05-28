"""ComponentEntry dataclass + build_catalogue.

Enumerates every mutable component for `localharness components list`. Merges:
  - Static HarnessConfig leaves via walk_model_fields(HarnessConfig)
  - Static AgentConfig leaves via walk_model_fields(AgentConfig) under "agent." prefix
  - Dynamic tools.<name>.description from ToolRegistry._schemas
  - Dynamic hooks.<name>.config from HookSystem.loaded_plugin_names

Layer attribution: pass overlays={"project": {...}, "user": {...}, "experiment": {...}};
catalogue records the highest-priority layer that owns each path.

See 14-RESEARCH.md Example B for the reference implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union, get_args, get_origin

from localharness.config.models import AgentConfig, HarnessConfig
from localharness.registry.paths import get_value, walk_model_fields


@dataclass(frozen=True)
class ComponentEntry:
    path: str                  # dot-path: "org.context.compaction_threshold_pct"
    annotation: Any            # python type annotation (e.g. float, str, Literal[...])
    type_name: str             # human-readable: "int", "float", "str", "Literal['debug',...]"
    current_value: Any         # resolved-cascade value (what `get` returns)
    default_value: Any         # the Pydantic-baked default
    winning_layer: str         # "default" | "project" | "user" | "experiment"


_LAYER_PRIORITY = ("experiment", "user", "project")  # highest-priority first


def _path_exists_in_dict(d: dict, path: str) -> bool:
    """True iff `d` has the given dot-path as a nested key."""
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def _detect_layer(path: str, overlays: dict[str, dict]) -> str:
    """Scan overlays top-down ('experiment' -> 'user' -> 'project'); first hit wins.
    Returns 'default' if no overlay contains the path.
    """
    for layer in _LAYER_PRIORITY:
        layer_dict = overlays.get(layer, {})
        if _path_exists_in_dict(layer_dict, path):
            return layer
    return "default"


def _type_name(ann: Any) -> str:
    """Render annotation as a short human-readable string for CLI display."""
    if ann is None:
        return "None"
    origin = get_origin(ann)
    if origin is Union:
        inner = [a for a in get_args(ann) if a is not type(None)]
        if len(inner) == 1:
            return f"Optional[{_type_name(inner[0])}]"
        return " | ".join(_type_name(a) for a in get_args(ann))
    if origin is list:
        args = get_args(ann)
        return f"list[{_type_name(args[0])}]" if args else "list"
    if origin is dict:
        args = get_args(ann)
        return f"dict[{_type_name(args[0])}, {_type_name(args[1])}]" if args else "dict"
    if isinstance(ann, type):
        return ann.__name__
    return str(ann)


def _get_default(model_cls: type, path: str) -> Any:
    """Return the Pydantic-baked default value for a dot-path within model_cls.
    Walks model_cls.model_fields recursively. Returns None if no default declared.
    """
    parts = path.split(".")
    cur_cls: Any = model_cls
    for i, part in enumerate(parts):
        if not hasattr(cur_cls, "model_fields"):
            return None
        field_info = cur_cls.model_fields.get(part)
        if field_info is None:
            return None
        if i == len(parts) - 1:
            # Leaf -- return default
            if field_info.default_factory is not None:
                try:
                    return field_info.default_factory()
                except Exception:
                    return None
            return field_info.default
        # Descend into nested annotation (unwrap Optional)
        ann = field_info.annotation
        if get_origin(ann) is Union:
            inner = [a for a in get_args(ann) if a is not type(None)]
            if len(inner) == 1:
                ann = inner[0]
        cur_cls = ann
    return None


def build_catalogue(
    cfg: HarnessConfig,
    *,
    overlays: dict[str, dict] | None = None,
    agent_cfg: Optional[AgentConfig] = None,
    tool_registry: Any = None,
    hook_system: Any = None,
) -> dict[str, ComponentEntry]:
    """Build the complete component catalogue.

    Args:
        cfg: resolved HarnessConfig (post-cascade).
        overlays: layer dicts keyed "project" | "user" | "experiment" for layer attribution.
                  Pass {} for default-only attribution.
        agent_cfg: optional AgentConfig to enumerate `agent.*` paths against the live agent;
                   if None, uses AgentConfig.model_construct() defaults (REG-04 always exposes
                   agent surfaces in list).
        tool_registry: ToolRegistry instance (provides `._schemas` for tools.<name>.description).
        hook_system: HookSystem instance (provides `.loaded_plugin_names` for hooks.<name>.config).
    """
    overlays = overlays or {}
    entries: dict[str, ComponentEntry] = {}

    # 1. Static harness-level paths (provider.*, org.*, version)
    if cfg is not None:
        for path, ann in walk_model_fields(HarnessConfig):
            try:
                current = get_value(cfg, path)
            except AttributeError:
                current = None
            entries[path] = ComponentEntry(
                path=path,
                annotation=ann,
                type_name=_type_name(ann),
                current_value=current,
                default_value=_get_default(HarnessConfig, path),
                winning_layer=_detect_layer(path, overlays),
            )

    # 2. Static agent-level paths (agent.role, agent.stuck_detector.*, agent.recovery_injection.*, etc.)
    #    Use provided agent_cfg or fall back to defaults so REG-04 surfaces always render.
    if agent_cfg is None:
        try:
            agent_cfg = AgentConfig.model_construct(name="<default>", role="<default>")
        except Exception:
            agent_cfg = None

    for path, ann in walk_model_fields(AgentConfig):
        agent_path = f"agent.{path}"
        current = None
        if agent_cfg is not None:
            try:
                current = get_value(agent_cfg, path)
            except AttributeError:
                current = None
        entries[agent_path] = ComponentEntry(
            path=agent_path,
            annotation=ann,
            type_name=_type_name(ann),
            current_value=current,
            default_value=_get_default(AgentConfig, path),
            winning_layer=_detect_layer(agent_path, overlays),
        )

    # 3. Dynamic: tools.<name>.description
    if tool_registry is not None and hasattr(tool_registry, "_schemas"):
        for tool_name, schema in tool_registry._schemas.items():
            # Strip scope prefix (e.g. "agent:foo:exec" -> "exec") for the path; if you want
            # full scoping use the raw key. Phase 14 keeps it simple: use the raw key so
            # operators see every registered tool variant.
            path = f"tools.{tool_name}.description"
            description = getattr(schema, "description", "")
            entries[path] = ComponentEntry(
                path=path,
                annotation=str,
                type_name="str",
                current_value=description,
                default_value=description,
                winning_layer=_detect_layer(path, overlays),
            )

    # 4. Dynamic: hooks.<name>.config
    if hook_system is not None and hasattr(hook_system, "loaded_plugin_names"):
        org_hooks = {}
        if cfg is not None:
            try:
                org_hooks = cfg.org.hooks
            except AttributeError:
                org_hooks = {}
        for hook_name in hook_system.loaded_plugin_names:
            path = f"hooks.{hook_name}.config"
            entries[path] = ComponentEntry(
                path=path,
                annotation=dict,
                type_name="dict[str, Any]",
                current_value=org_hooks.get(hook_name, {}),
                default_value={},
                winning_layer=_detect_layer(path, overlays),
            )

    return entries


# ------------------------------------------------------------------ #
# Surface family enumeration (used by REG-04 test_six_distinct_surface_types)
# ------------------------------------------------------------------ #

SURFACE_FAMILIES = {
    "system_prompt": (r"^agent\.role$", r"^agent\.context\.system_prompt_file$"),
    "tool_description": (r"^tools\..+\.description$",),
    "compaction_threshold": (r"^org\.context\.compaction_threshold_pct$",
                             r"^agent\.context\.compaction_threshold_pct$"),
    "stuck_detector": (r"^agent\.stuck_detector\.",),
    "recovery_injection": (r"^agent\.recovery_injection\.",),
    "hook_config": (r"^hooks\..+\.config$", r"^org\.hooks$"),
}
