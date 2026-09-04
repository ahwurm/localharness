"""Which file set this key — the one answer three commands ask.

`components get`, `doctor`'s overridden-key report and `config show` are three views of a single
question. This module owns the answer so they cannot drift: one overlay builder, one catalogue
builder, one definition of "overridden".

Deliberately NOT built on ConfigLoader._raw_config_sources(). That method strips the `agent:`
section from BOTH overlays because `agent:` is not a HarnessConfig field and must never reach
harness validation. The registry needs the opposite for one of them: `components set agent.*`
writes into the GLOBAL overrides' `agent:` section and `get` must read it back, so that section
has to survive. The two needs are genuinely different, which is why this is a second, small
reader rather than a shared one.

The asymmetry below is the shipped truth, not an oversight:
  * global overrides `agent:` — KEPT. `load_agent` reads it; it is the per-agent default layer.
  * workspace overrides `agent:` — DROPPED. Nothing reads it in v0.13 (loader.py strips it,
    06-config.md documents it). Attributing a key to a file the harness never consults would be
    the same lie F4 exists to fix, just relocated.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, get_origin

from localharness.config.loader import ConfigLoader, _load_yaml_file
from localharness.config.overlay import load_overlay
from localharness.registry.catalogue import (
    LAYER_GLOBAL_CONFIG,
    LAYER_GLOBAL_OVERRIDES,
    LAYER_WORKSPACE_CONFIG,
    LAYER_WORKSPACE_OVERRIDES,
    _LAYER_PRIORITY,
    ComponentEntry,
    build_catalogue,
)
from localharness.registry.paths import _unwrap_optional

_AGENT_KEY = "agent"
_AGENT_PREFIX = _AGENT_KEY + "."

# Fields whose resolved value is a UNION across layers rather than one layer's value. The deny
# list is the only one: `ConfigLoader._layered_org_deny` accumulates it so a workspace can never
# subtract a global denial (MERG-02). Every other list is replaced wholesale by `deep_merge`.
_UNION_PATHS = frozenset({"org.permissions.deny_patterns"})


def build_layer_overlays(
    loader: ConfigLoader, workspace: Optional[Path]
) -> dict[str, dict]:
    """The raw config sources keyed by their honest band names, for layer attribution.

    Two keys with no workspace, four with one. `workspace` is the value
    `cli.workspace.resolve_workspace_layer()` returned for THIS invocation — never re-discovered
    here, so an explicit --config-dir stays a full replacement (LAYR-02).

    `_load_yaml_file` is package-private in config/loader, and is used anyway rather than a bare
    `yaml.safe_load`: it is what raises ConfigParseError carrying the WORKSPACE file's own path
    and line, which is exactly the attribution 43-01 just built. A public re-export is a wider
    change than this module needs.
    """
    overlays: dict[str, dict] = {
        LAYER_GLOBAL_CONFIG: loader.raw_harness_dict(),
        LAYER_GLOBAL_OVERRIDES: load_overlay(loader.user_overlay_path),
    }
    if workspace is None:
        return overlays
    ws_cfg_path = workspace / "config.yaml"
    overlays[LAYER_WORKSPACE_CONFIG] = (
        _load_yaml_file(ws_cfg_path) if ws_cfg_path.exists() else {}
    )
    ws_overlay = load_overlay(workspace / "overrides.yaml")
    overlays[LAYER_WORKSPACE_OVERRIDES] = {
        k: v for k, v in ws_overlay.items() if k != _AGENT_KEY
    }
    return overlays


def _dig_dict(d: dict, dotpath: str) -> tuple[Any, bool]:
    """Walk a dot-path through a nested dict. Returns (value, True) if present, else (None, False)."""
    cur: Any = d
    for part in dotpath.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None, False
        cur = cur[part]
    return cur, True


def apply_agent_overlay_values(
    catalogue: dict[str, ComponentEntry], overlays: dict[str, dict]
) -> dict[str, ComponentEntry]:
    """Reflect the GLOBAL overrides' `agent:` leaves in current_value so `set agent.*` round-trips.

    Moved from components_cmd._apply_agent_overlay_values (behavior unchanged), now reading the
    band by name so there is one implementation for all three callers. Only paths EXPLICITLY
    present are patched — every other agent.* axis keeps its compiled-in default, so the
    name-derived memory paths a model-validated placeholder would introduce never leak in.

    Reads the GLOBAL band only, deliberately: `set` writes there, and the workspace overrides'
    `agent:` section is not in `overlays` at all (see the module docstring).
    """
    user_overlay = overlays.get(LAYER_GLOBAL_OVERRIDES, {})
    agent_overlay = user_overlay.get(_AGENT_KEY) if isinstance(user_overlay, dict) else None
    if not isinstance(agent_overlay, dict):
        return catalogue
    for cat_path, entry in list(catalogue.items()):
        if not cat_path.startswith(_AGENT_PREFIX):
            continue
        value, found = _dig_dict(agent_overlay, cat_path[len(_AGENT_PREFIX):])
        if found:
            catalogue[cat_path] = replace(entry, current_value=value)
    return catalogue


def _contributing_layers(path: str, overlays: dict[str, dict]) -> list[str]:
    """Every band that sets this exact dot-path, in MERGE order (lowest priority first)."""
    return [
        layer
        for layer in reversed(_LAYER_PRIORITY)
        if _dig_dict(overlays.get(layer, {}), path)[1]
    ]


def _attributing_layer(path: str, overlays: dict[str, dict]) -> Optional[str]:
    """The band whose value the merge actually took for this path, including BLOCK CLOBBERS.

    `_detect_layer` asks only "does this band carry the full dot-path". A workspace that writes
    `server: null` carries no `server.runtime` key, so every leaf under the nulled block was
    credited to the global layer that no longer supplies it — the display said `global-config`
    beside a value of None the global file does not contain. Walking the PREFIXES catches it: a
    band that replaces an ancestor with a non-dict has replaced the whole subtree.

    Scanned highest-priority first, so a full-path hit above a clobber still wins — the same order
    the merge itself resolves in.
    """
    parts = path.split(".")
    for layer in _LAYER_PRIORITY:
        cur: Any = overlays.get(layer, {})
        for i, part in enumerate(parts):
            if not isinstance(cur, dict) or part not in cur:
                break
            cur = cur[part]
            if i == len(parts) - 1 or not isinstance(cur, dict):
                return layer  # the leaf itself, or an ancestor this band replaced wholesale
    return None


def _accumulated(layers: list[str]) -> str:
    """The honest winning_layer for a value no single band produced."""
    return f"accumulated ({' + '.join(layers)})"


def _is_dict_leaf(entry: ComponentEntry) -> bool:
    """A `dict[K, V]` leaf, which `deep_merge` MERGES key-wise rather than replacing."""
    return get_origin(_unwrap_optional(entry.annotation)) is dict


def honest_attribution(
    catalogue: dict[str, ComponentEntry], overlays: dict[str, dict]
) -> dict[str, ComponentEntry]:
    """Repair the two attributions `_detect_layer` cannot express on its own.

    ACCUMULATED values. `org.permissions.deny_patterns` is unioned across layers (MERG-02) and a
    `dict[K, V]` leaf is deep-merged key-wise, so when two bands contribute, the resolved value
    came from BOTH and naming one of them is a fabrication — the display said `workspace-config`
    beside a list holding the global layer's patterns. Such an entry is labelled
    `accumulated (global-config + workspace-config)` instead.

    CLOBBERED blocks: see `_attributing_layer`.

    KNOWN LIMIT, stated rather than hidden: a dict leaf's attribution is per-FIELD, never per-KEY.
    With one band setting `model_context_overrides.modelA` and another `modelB`, this says both
    contributed; it cannot say which band owns which key, because `walk_model_fields` stops at
    dict leaves and the catalogue has no entry below them. `config show` prints what this returns.
    """
    for path, entry in list(catalogue.items()):
        contributors = _contributing_layers(path, overlays)
        if len(contributors) > 1 and (path in _UNION_PATHS or _is_dict_leaf(entry)):
            catalogue[path] = replace(entry, winning_layer=_accumulated(contributors))
            continue
        layer = _attributing_layer(path, overlays)
        if layer is not None and layer != entry.winning_layer:
            catalogue[path] = replace(entry, winning_layer=layer)
    return catalogue


def layered_catalogue(
    config_dir: Path,
    workspace: Optional[Path],
    *,
    tool_registry: Any = None,
    loader: Optional[ConfigLoader] = None,
) -> tuple[dict[str, ComponentEntry], dict[str, dict]]:
    """(catalogue, overlays) for exactly one layering. Callers that need BOTH the workspace-on and
    the workspace-off view (doctor's diff) call this twice with different `workspace` values.

    `loader` lets a caller that ALREADY built the loader for this layering hand it over instead of
    paying for a second parse of the same files (doctor). It must be a loader for exactly this
    `config_dir`/`workspace` pair — the catalogue would otherwise describe a layering the caller
    never asked for.
    """
    if loader is None:
        loader = ConfigLoader(config_dir=config_dir, local_config_dir=workspace)
    cfg = loader.load_harness()
    overlays = build_layer_overlays(loader, workspace)
    cat = build_catalogue(cfg, overlays=overlays, tool_registry=tool_registry)
    return honest_attribution(apply_agent_overlay_values(cat, overlays), overlays), overlays


def overridden_paths(
    effective: dict[str, ComponentEntry],
    global_only: dict[str, ComponentEntry],
) -> list[tuple[str, ComponentEntry, Any]]:
    """Paths the workspace layer actually CHANGES, as (path, effective_entry, global_value).

    A VALUE comparison, not a presence check. A workspace config.yaml that restates a key with the
    value the global layer already had has not overridden anything, and reporting it as an override
    would make doctor's section noise on the exact configs people copy between projects.

    Sorted by path so two runs of `doctor` on one machine print the same order.
    """
    out: list[tuple[str, ComponentEntry, Any]] = []
    for path, entry in effective.items():
        other = global_only.get(path)
        before = other.current_value if other is not None else None
        if other is None or entry.current_value != before:
            out.append((path, entry, before))
    return sorted(out, key=lambda row: row[0])
