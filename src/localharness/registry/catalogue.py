"""ComponentEntry dataclass + build_catalogue.

Enumerates every mutable component for `localharness components list`. Merges:
  - Static HarnessConfig leaves via walk_model_fields(HarnessConfig)
  - Static AgentConfig leaves via walk_model_fields(AgentConfig) under "agent." prefix
  - Dynamic tools.<name>.description from ToolRegistry._schemas
  - Dynamic plugins' settings (ENAB-04): each plugin's `<name>.enabled` and, once loaded, the
    leaves of its ConfigModel (`<name>.*`) and AgentConfigModel (`agent.<name>.*`) — PluginRows

Layer attribution: pass overlays={"global-config": {...}, "global-overrides": {...},
"workspace-config": {...}, "workspace-overrides": {...}, "experiment": {...}}; catalogue records
the highest-priority band that owns each path. `registry/provenance.build_layer_overlays` builds
that dict for the CLI; every caller should use it rather than assembling one by hand.

See 14-RESEARCH.md Example B for the reference implementation.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union, get_args, get_origin

from pydantic import BaseModel

from localharness.config.loader import is_harness_global_only
from localharness.config.models import AgentConfig, HarnessConfig
from localharness.registry.paths import get_value, walk_model_fields

if TYPE_CHECKING:
    from localharness.plugins.resolve import Resolution


@dataclass(frozen=True)
class ComponentEntry:
    path: str                  # dot-path: "org.context.compaction_threshold_pct"
    annotation: Any            # python type annotation (e.g. float, str, Literal[...])
    type_name: str             # human-readable: "int", "float", "str", "Literal['debug',...]"
    current_value: Any         # resolved-cascade value (what `get` returns)
    default_value: Any         # the Pydantic-baked default
    winning_layer: str         # one of the LAYER_* constants below, or LAYER_DEFAULT
    plugin: str | None = None  # the plugin that owns this setting (ENAB-04); None for core


@dataclass(frozen=True)
class PluginRows:
    """What the catalogue lists for one plugin (ENAB-04): its enable switch, and — once loaded — every
    leaf of its ConfigModel (`<name>.*`) and AgentConfigModel (`agent.<name>.*`). `sections`: the
    top-level core settings a bundled plugin owns (its manifest's `sections`; empty otherwise)."""
    name: str
    enabled: bool
    enabled_default: bool
    config_model: type[BaseModel] | None = None
    config: BaseModel | None = None
    agent_config_model: type[BaseModel] | None = None
    agent_config: BaseModel | None = None
    global_only: frozenset[str] = frozenset()   # paths relative to `<name>.` (incl. "enabled" for a plugin you installed)
    sections: frozenset[str] = frozenset()


def plugin_catalogue_rows(resolution: Resolution) -> tuple[PluginRows, ...]:
    """One PluginRows per plugin whose settings validated: every bundled plugin, on or off (its rows
    show before it is switched on), and every loaded plugin you installed. One that is only
    available, failed, or holds invalid settings has none, and nothing is imported to list it.
    `global_only` is the resolver's own machine-level rule, so the attribution below drops exactly
    what the resolver dropped."""
    from localharness.plugins.resolve import _machine_only

    out = []
    for name, settings in resolution.settings.items():
        # Every name in settings has a plan entry; a missing one would get the stricter rule.
        entry, cls = resolution.plan.entry(name), resolution.classes[name]
        bundled = entry is not None and entry.bundled
        out.append(PluginRows(
            name, resolution.enabled[name], bundled and cls.manifest.enabled_by_default,
            cls.ConfigModel, settings.config, cls.AgentConfigModel, settings.agent_config,
            _machine_only(cls.ConfigModel, bundled),
            frozenset(cls.manifest.sections) if bundled else frozenset()))
    return tuple(out)


# The four config FILES the owner's 2026-09-03 ruling merges, named for what they are. Until v0.13
# these were `project` (which meant the global config.yaml, not a project folder) and `user` (which
# meant the global overrides.yaml) — two names that became actively misleading the moment a real
# workspace layer existed. Every band now says WHICH LAYER and WHICH FILE.
LAYER_GLOBAL_CONFIG = "global-config"
LAYER_GLOBAL_OVERRIDES = "global-overrides"
LAYER_WORKSPACE_CONFIG = "workspace-config"
LAYER_WORKSPACE_OVERRIDES = "workspace-overrides"
LAYER_EXPERIMENT = "experiment"
LAYER_DEFAULT = "default"

# Highest-priority FIRST. The order IS the owner's ruled merge order read backwards:
# global config < global overrides < workspace config < workspace overrides, with `experiment`
# (the gate's throwaway worktree overlay) on top of all of it. `_detect_layer`'s
# `overlays.get(layer, {})` makes an unpopulated band inert, which is what keeps a no-workspace
# session's attribution identical to pre-v0.13 (LAYR-03): `build_layer_overlays` returns the two
# global keys only when no workspace applies.
_LAYER_PRIORITY = (
    LAYER_EXPERIMENT,
    LAYER_WORKSPACE_OVERRIDES,
    LAYER_WORKSPACE_CONFIG,
    LAYER_GLOBAL_OVERRIDES,
    LAYER_GLOBAL_CONFIG,
)


def _global_bands(overlays: dict[str, dict]) -> dict[str, dict]:
    """`overlays` narrowed to the two global bands — the only files a GLOBAL_ONLY plugin path can
    take its value from: a workspace value there is dropped at load (ENAB-02), so crediting a
    workspace band would name a file the value never came from."""
    return {k: v for k, v in overlays.items() if k in (LAYER_GLOBAL_OVERRIDES, LAYER_GLOBAL_CONFIG)}


def _path_exists_in_dict(d: dict, path: str) -> bool:
    """True iff `d` has the given dot-path as a nested key."""
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def _detect_layer(path: str, overlays: dict[str, dict]) -> str:
    """Scan overlays top-down in _LAYER_PRIORITY order; first hit wins.
    Returns LAYER_DEFAULT if no overlay contains the path.
    """
    for layer in _LAYER_PRIORITY:
        layer_dict = overlays.get(layer, {})
        if _path_exists_in_dict(layer_dict, path):
            return layer
    return LAYER_DEFAULT


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
    plugins: Sequence[PluginRows] = (),
) -> dict[str, ComponentEntry]:
    """Build the complete component catalogue.

    Args:
        cfg: resolved HarnessConfig (post-cascade).
        overlays: layer dicts keyed by the LAYER_* band names for layer attribution
                  (registry.provenance.build_layer_overlays builds this). Pass {} for
                  default-only attribution.
        agent_cfg: optional AgentConfig to enumerate `agent.*` paths against the live agent;
                   if None, uses AgentConfig.model_construct() defaults (REG-04 always exposes
                   agent surfaces in list).
        tool_registry: ToolRegistry instance (provides `._schemas` for tools.<name>.description).
        plugins: one PluginRows per plugin (`plugin_catalogue_rows(resolve(loader))`); none, and
                 the catalogue is core rows only (the autoresearch callers).
    """
    overlays = overlays or {}
    entries: dict[str, ComponentEntry] = {}

    # 1. Static harness-level paths (provider.*, org.*, version). A top-level section a bundled
    #    plugin claims (PluginRows.sections) is tagged as that plugin's while it is on, and left
    #    out while it is off; the config loader validates it the same either way. A core key only
    #    the global layers may set (HARNESS_GLOBAL_ONLY_FIELDS, a whole section's leaves included)
    #    is attributed from them alone: a workspace value there was dropped at load, so it never
    #    supplied what is shown.
    owned = {s: p.name for p in plugins if p.enabled for s in p.sections}
    machine = _global_bands(overlays)
    hidden = {s for p in plugins if not p.enabled for s in p.sections} - owned.keys()
    if cfg is not None:
        for path, ann in walk_model_fields(HarnessConfig):
            if path.split(".")[0] in hidden:
                continue
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
                winning_layer=_detect_layer(path, machine if is_harness_global_only(path) else overlays),
                plugin=owned.get(path.split(".")[0]),
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

    # 4. Dynamic: each plugin's settings (ENAB-04). A GLOBAL_ONLY path is attributed from the two
    #    global bands alone (`machine`, above); agent-level GLOBAL_ONLY is refused by the resolver.
    for p in plugins:
        path = f"{p.name}.enabled"
        entries[path] = ComponentEntry(
            path, bool, "bool", p.enabled, p.enabled_default,
            _detect_layer(path, machine if "enabled" in p.global_only else overlays), plugin=p.name)
        for prefix, model, instance, marked in (
            (p.name, p.config_model, p.config, p.global_only),
            (f"agent.{p.name}", p.agent_config_model, p.agent_config, frozenset()),
        ):
            if model is None:
                continue
            for rel, ann in walk_model_fields(model):
                path = f"{prefix}.{rel}"
                try:
                    current = get_value(instance, rel)
                except AttributeError:
                    current = None
                entries[path] = ComponentEntry(
                    path, ann, _type_name(ann), current, _get_default(model, rel),
                    _detect_layer(path, machine if rel in marked else overlays), plugin=p.name)

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
    "hook_config": (r"^org\.hooks$",),
}
